"""Streamlit UI for the simplified mold flow simulator.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import dataclasses
import io
import json
import math
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from core import (
    DirectGateConfig,
    FilmGate2Config,
    GateProfileSpec,
    HeleShawSolver,
    MaterialDB,
    MultilayerHeleShawSolver,
    ProfilePlateConfig,
    build_direct_gate_geometry,
    build_fill_player_html,
    build_film_gate2_geometry,
    build_profile_gate_geometry,
    export_frames,
    fill_frame_fractions,
    fill_frame_times,
    fill_player_height_px,
    render_3d_fill_time,
    render_3d_pressure,
    render_3d_thickness_map,
    render_core_layer_map,
    render_fill_animation,
    render_pressure_map,
    render_skin_layer_map,
    render_weldlines,
    wrap_standalone_html,
)
from core.geometry import Geometry
from core.profile_gate import IslandSpec, LandSpec, MainRampSpec, ValveSpec, WellSpec
from core.settings_record import config_settings, file_fingerprint, settings_json
from core.spec_source import (
    SPEC_LINK_NAME,
    SpecMode,
    SpecOrigin,
    choose_spec_origin,
    list_spec_files,
    spec_link_exists,
    spec_root,
)
from core.two_phase import solve_two_phase_short_shot
from core.version import build_label
from core.visualizer import (
    ISOCHRONE_LEVELS,
    THICKNESS_CMAP,
    render_layer_grid,
    render_short_shot_map,
    render_two_phase_animation,
    render_two_phase_map,
)

APP_DIR = Path(__file__).parent
DEMO_PROFILE_JSON = APP_DIR / "data" / "gate_profiles" / "demo_profile_gate.json"
# Film gate 1: the well's sloped wall angle is fixed (a cutter-geometry
# constant on the drawings this input reproduces, not a design variable).
_WELL_WALL_ANGLE_DEG = 60.0

# Radio labels live here, next to the rest of the UI text; the logic in
# ``core.spec_source`` switches on :class:`SpecMode` so that rewording one of
# these cannot change which source a run reads from.
SPEC_MODE_LABELS = {
    SpecMode.DEMO: "デモプリセット（架空寸法）",
    SpecMode.LOCAL: "ローカルから読込",
    SpecMode.PASTE: "JSON貼り付け",
}
SPEC_MODE_BY_LABEL = {v: k for k, v in SPEC_MODE_LABELS.items()}
#: Sentinel occupying index 0 of the spec dropdown. See
#: ``choose_spec_origin`` for why the list does not default to a real file.
SPEC_UNSELECTED = "— 未選択 —"
#: Widget key for the spec uploader. Needed so the dropdown above it can
#: read the dropped file from session state in the same run it lands.
SPEC_UPLOAD_KEY = "spec_upload_pg"

st.set_page_config(page_title="極薄プレート 簡易流動解析", layout="wide")
st.title("極薄プレート 簡易流動解析")
st.caption(
    "薄板部品の射出成形における樹脂流動を簡易解析するツール。"
    "ゲート位置・肉厚・ランナー形状の方向性検討を、実機評価前の"
    "初期検討段階で迅速に行うことを目的とする。商用 CAE"
    "（Moldflow 等）の代替を意図したものではない。"
)

with st.expander("📐 使用している方程式と適用範囲"):
    st.markdown("### 1. 全体モデル：Hele-Shaw 近似（薄板潤滑流れ）")
    st.markdown(
        "金型キャビティが**薄板（厚み $h \\ll$ 平面サイズ）**であることを前提に、"
        "面内 2D + 厚み方向の解析積分という簡略モデル。"
        "射出成形の薄肉製品では商用 CAE（Moldflow / Moldex3D）の中間面ソルバーも"
        "本質的に同じ Hele-Shaw 系を使う。"
    )
    st.latex(
        r"\nabla \cdot \left( S \, \nabla p \right) = 0,"
        r"\quad S = \frac{h^3}{12\,\eta_{\text{eff}}}"
    )
    st.markdown(
        "- $h(x,y)$: 局所キャビティ厚み [m]\n"
        "- $\\eta_{\\text{eff}}$: コンダクタンス計算用の有効粘度 [Pa·s]\n"
        "- $S$: コンダクタンス（流れやすさ）。$h^3$ で効くので**厚み変化が支配的**\n"
        "- 圧力 $p$ はゲートで一定、流動先端で 0 を境界条件にして解く"
    )

    st.markdown("### 2. 充填時間場：Pseudo-Conduction 法")
    st.markdown(
        "Hele-Shaw の圧力場を時間ステップで進めず、**楕円型（熱伝導型）方程式に置き換えて 1 発で解く**"
        "高速化テクニック。$\\tau$ は擬似的な「ゲートからの到達時間場」。"
    )
    st.latex(r"-\,\nabla \cdot \left( S\, \nabla \tau \right) = 1 \quad \text{in cavity}")
    st.markdown(
        "- ゲートで $\\tau = 0$（ディリクレ境界）、キャビティ壁で no-flux（ノイマン境界）\n"
        "- 解いた $\\tau$ を最大値で正規化し、絶対時間に換算: "
        r"$t_{\text{fill}}(x,y) = \dfrac{\tau(x,y)}{\tau_{\max}} \cdot T_{\text{fill}}$"
        "\n- $T_{\\text{fill}} = V_{\\text{cavity}} / Q$（射出率一定）\n"
        "- 流動先端の進行は $\\tau$ の等値線として可視化"
    )

    st.markdown("### 3. 粘度モデル：Cross-WLF")
    st.markdown(
        "剪断速度依存の擬塑性 + 温度依存（WLF型）を組み合わせた業界標準モデル。"
        "`data/materials.json` に PP / PP_T10 / PP_T20 / PP_T30 / ABS / PC / PA66 / PMMA の代表値を保持。"
    )
    st.latex(
        r"\eta(\dot\gamma, T) = \frac{\eta_0(T)}"
        r"{1 + \left( \dfrac{\eta_0(T)\,\dot\gamma}{\tau^{*}} \right)^{1-n}}"
    )
    st.latex(r"\eta_0(T) = D_1 \exp\!\left[\,-\,\frac{A_1 (T - T^{*})}{A_2 + (T - T^{*})}\,\right]")
    st.markdown(
        "**温度・剪断速度の評価は壁面冷却モデルで変わる**:\n"
        "- **なし**: バルク温度 $T_{\\text{bulk}} = 0.7\\,T_{\\text{melt}} + 0.3\\,T_{\\text{mold}}$ と"
        " 代表剪断速度 $\\dot\\gamma = 6V/h$ で 1 回だけ評価\n"
        "- **スキン層 / 層別**: 厚み方向に **層別** $T_k$・$\\dot\\gamma_k$ を割り振り、各層で別個に $\\eta_k$"
        " を評価（後述 §4・§5）"
    )

    st.markdown("### 4. 壁面冷却モデル A：スキン層（Stefan / Neumann 1 層近似）")
    st.markdown(
        "金型壁で樹脂が固化して育つ「スキン層」を熱拡散の Stefan 解で表現し、"
        "流動はコア層 $h_{\\text{core}} = h - 2s$ のみを通る。**コア温度は melt のまま固定**。"
    )
    st.latex(r"s(t) = c_{\text{skin}} \sqrt{\alpha\,t}")
    st.markdown(
        "- $\\alpha$: 樹脂の熱拡散率（材料 DB から取得）\n"
        "- $c_{\\text{skin}}$: 成長定数（$\\sim 1.0$ が物理的代表値、UI で調整）\n"
        "- $\\tau$ と $s$ が相互依存するので fixed-point 反復で釣り合わせる\n"
        "- $h_{\\text{core}}$ が下限を切ったセル＝**ショートショット候補**として赤マーク"
    )

    st.markdown("### 5. 壁面冷却モデル B：層別 N 層離散化（推奨・極薄向け既定）")
    st.markdown(
        "厚み方向を $N$ 層に離散化し、**各層に固有の温度・粘度・剪断速度** を持たせる。"
        "スキン層モデルが「壁面凍結フロント」しか扱わないのに対し、こちらは**コア内部の温度・粘度プロファイル**"
        "まで解像する。極薄プレート（$t < 0.5$ mm）で必須。"
    )

    st.markdown("**5-1. 厚み離散化**")
    st.latex(r"\zeta_k \in [0, 1],\quad h_k(x,y) = (\zeta_k - \zeta_{k-1}) \cdot h(x,y)")
    st.markdown(
        "- **wall_refined**（既定）: Chebyshev-Lobatto 点 "
        r"$\zeta_k = \tfrac{1}{2}(1 - \cos(\pi k / N))$ で壁近傍を細かく"
        "\n- **uniform**: 等間隔"
    )

    st.markdown("**5-2. 層別温度（Neumann 1D 重ね合わせ）**")
    st.latex(
        r"T_k(x,y) = T_{\text{mold}} + (T_{\text{melt}} - T_{\text{mold}}) \cdot "
        r"\left[\,\operatorname{erf}\!\left(\tfrac{z_k}{2\sqrt{\alpha t_{\text{arr}}}}\right)"
        r"+ \operatorname{erf}\!\left(\tfrac{h - z_k}{2\sqrt{\alpha t_{\text{arr}}}}\right) - 1\,\right]"
    )
    st.markdown(
        "両壁から育つ熱境界層の重ね合わせ。長時間極限の数値発散を避けるため "
        r"$T_k \ge T_{\text{mold}}$ で clamp。$t_{\text{arr}}(x,y) = (\tau/\tau_{\max}) \cdot T_{\text{fill}}$"
        " はセル到達時間。"
    )

    st.markdown("**5-3. 層別剪断速度（Poiseuille 解析微分）**")
    st.latex(r"\dot\gamma_k(x,y) = \frac{6 V}{h(x,y)} \cdot |2\zeta_k - 1|")
    st.markdown(
        "壁で最大、中央でゼロ。中央層は Cross-WLF の $D_1$ 発散を避けるため "
        "$\\dot\\gamma_{\\text{floor}} = 0.01 \\cdot 6V/h$ でクリップ。"
    )

    st.markdown("**5-4. 並列流路統合（Poiseuille モーメント積分）**")
    st.latex(r"S_{\text{total}}(x,y) = \frac{h(x,y)^3}{2} \sum_{k=1}^{N} \frac{m_k}{\eta_k(x,y)}")
    st.latex(
        r"m_k = \left[\frac{\zeta^2}{2} - \frac{\zeta^3}{3}\right]_{\zeta_{k-1}}^{\zeta_k},"
        r"\quad \sum_k m_k = \frac{1}{6}"
    )
    st.markdown(
        r"$\sum m_k = 1/6$ が保存するので $N=1$ では従来 $S = h^3/(12\eta)$ と厳密一致（後方互換）。"
    )

    st.markdown("**5-5. 固定点反復で $\\tau$ と層フィールドを結合**")
    st.markdown(
        r"$\tau \to t_{\text{arr}} \to T_k \to \eta_k \to S_{\text{total}} \to \tau_{\text{new}}$"
        " を $\\|\\Delta\\tau\\|_2 / \\|\\tau\\|_2 < $ tol まで反復。"
        "発散時のみ $\\omega = 0.7$ で適応的 damping。**ショートショット判定**は最終 iteration の中央層温度ベース:"
        r" $T_{\text{solid}} = T_{\text{mold}} + f_{\text{solid}} (T_{\text{melt}} - T_{\text{mold}})$。"
    )

    st.markdown("### 6. 剪断発熱（viscous dissipation, 段階1）")
    st.markdown(
        "粘性散逸による発熱を**層別モード内で**取り込む補正。"
        "極薄プレート + 高速射出で Brinkman 数 $Br \\gg 1$ になりがちな領域で必須。"
    )
    st.latex(
        r"\Delta T_{\text{shear},k}(x,y) = \frac{\eta_k \,\dot\gamma_k^{\,2}}{\rho \, c_p}"
        r"\cdot \min\!\left(t_{\text{arr}},\; \tau_{\text{thermal}}\right)"
    )
    st.latex(r"\tau_{\text{thermal}} = \frac{h^2}{\pi^2 \, \alpha}")
    st.markdown(
        "$\\tau_{\\text{thermal}}$ は厚み方向 1D 拡散の最低モード時定数で頭打ち。"
        "実態は **粘性散逸 vs 1D 壁面冷却** の準定常バランス近似。"
        " $T_k \\leftarrow T_{k,\\text{Neumann}} + \\Delta T_{\\text{shear},k}$ で Cross-WLF を再評価、"
        "粘度低下→流動加速→発熱低下の負のフィードバックは fixed-point 反復で自然収束。"
    )

    st.markdown("**Brinkman 数（剪断発熱の必要性診断、補正 OFF でも常時計算）**")
    st.latex(
        r"Br = \frac{\eta \,\dot\gamma^{\,2}\, h^2}{k \, (T_{\text{melt}} - T_{\text{mold}})},"
        r"\quad k = \alpha \cdot \rho \cdot c_p"
    )
    st.markdown(
        "- 🟢 $Br < 0.5$: 熱伝導支配、剪断発熱無視可\n"
        "- 🟡 $0.5 \\le Br < 2$: 同程度、補正 ON 推奨\n"
        "- 🔴 $Br \\ge 2$: 剪断発熱支配、本来は **段階2（1D FDM 陰解法）** が必要"
    )

    st.markdown("### 7. 射出圧縮成形（ICM、オプション）：等価厚み膨張モデル")
    st.markdown(
        "圧縮位相を時間ステッピングで解かず、**製品本体の厚みを膨張**させた等価モデルとして扱う。"
        "流路抵抗 $S \\propto h^3$ が一気に下がる効果を擬似再現。"
        "膨張対象は「製品本体セルだけ」、ランナー・スプルー・ゲートは射出時の肉厚のまま不変。"
        "UI は **stroke モード**（金型シム量の物理に整合）で統一。"
        "倍率指定の **factor モード**は CLI / solver 引数で後方互換のためだけに残す。"
    )

    st.markdown("**stroke モード（絶対加算、段差保存、UI 既定）**")
    st.markdown(
        "全 target セルに同じ絶対量（ストローク $s$）を**加算**する。"
        "**金型シム量**が設計指標のとき（＝実機の射出圧縮成形そのもの）に使う。"
        "段差プレート（例: 薄肉部 $t_0=0.35$ mm ／ 厚肉部 $t_0=0.50$ mm）に "
        "$s=0.70$ mm を加算すると薄肉部 $1.05$ mm ／ 厚肉部 $1.20$ mm となり、"
        "**段差 $0.15$ mm が圧縮位相中も保存される**（factor モードだと段差が $0.45$ mm に膨らんで非物理）。"
    )
    st.latex(r"h_{\text{eff}}(x,y) = h(x,y) + s \quad \text{on compression cells}")
    st.latex(
        r"T_{\text{fill}}^{\text{ICM}} = T_{\text{fill}}^{\text{base}} \cdot "
        r"\left[\,\frac{f_{\text{cmp}}}{1 + s \cdot A_{\text{cm}} / V_{\text{total}}}\,"
        r"+\,(1 - f_{\text{cmp}})\,\right]"
    )
    st.markdown(
        "- $s$: `compression_stroke_mm`（圧縮ストローク [mm]、絶対加算量）\n"
        "- $A_{\\text{cm}}$: 圧縮対象セルの**面積** [mm²]（`compression_area_mm2()`）\n"
        "- $V_{\\text{total}}$: 全キャビティ体積 [mm³]\n"
        "- $f_{\\text{cmp}}$: 充填占有率（圧縮開始時にキャビティ何%まで充填されているか）"
    )

    st.markdown("---")
    st.markdown("### ✅ モデル化している現象")
    st.markdown(
        "- 薄板キャビティ内の 2D 流動（局所剪断と局所抵抗の効果）\n"
        "- 樹脂物性（密度・比熱・熱拡散率 → 熱伝導率派生・Cross-WLF 粘度パラメータ）\n"
        "- ゲート位置・ゲート径・ランナー形状・プレート分割（2 層肉厚）・バランサー（最大5段）\n"
        "- 流動先端の到達順、ウェルドライン、エアトラップ\n"
        "- 圧力分布の相対値（ゲート＝1、最終充填点＝0 の正規化）\n"
        "- 充填時間（射出率 $Q$ から逆算した絶対時間）\n"
        "- **壁面冷却**: スキン層 1 層モデル または **層別 $N$ 層モデル**（厚み方向温度・粘度プロファイル）\n"
        "- **剪断発熱（段階1, 層別モード内）**: 粘性散逸による局所温度上昇、Brinkman 数診断\n"
        "- **ショートショット予測**: スキン層モデルでは $h_{\\text{core}} \\le h_{\\min}$ 判定、"
        "層別モデルでは中央層温度ベース判定\n"
        "- 射出圧縮成形による等価流路拡大（stroke モード、CLI に factor 後方互換あり、オプション）"
    )

    st.markdown("### ❌ モデル化していない現象（重要）")
    st.markdown(
        "- **面内 3D 流れ・ジェッティング・噴流・コーナー渦**: あくまで 2D Hele-Shaw（厚み方向は層別化済みだが、"
        "**面内**の 3D 性は完全 3D FVM / FEM ソルバーでないと出ない。Hele-Shaw 系の根本限界）\n"
        "- **剪断発熱の自己整合（段階2）**: 段階1 は閉形式の局所近似のみ。"
        r"$\rho c_p \partial_t T = k \partial_z^2 T + \eta \dot\gamma^2$ を厚み方向 1D FDM で陰解法積分する"
        "段階2 は別ロードマップ。$Br \\ge 2$ 領域では段階1 がズレる\n"
        "- **保圧（パッキング段階）**: 充填までしかモデル化しない\n"
        "- **収縮・反り・残留応力**: 熱固化収縮も結晶化も入っていない\n"
        "- **層内対流項**: 1D Neumann は純粋拡散のみ（薄板では妥当な近似だが、極厚 $h > 4$ mm では破綻）\n"
        "- **ベント・脱気挙動**: エアトラップ位置は予測するが圧抜けは考慮しない\n"
        "- **STL/STEP 直接読み込み**: パラメトリック形状（Film gate / Direct gate）"
        "または JSON スペック（Profile gate）のみ\n"
        "- **非構造格子・中立面メッシュ**: 構造格子（正方形セル）固定\n"
        "- **絶対圧力場の出力**: 圧力は正規化値（ゲート=1 / フロント=0）のみ。"
        "実機の必要型締力評価には未対応"
    )

    st.markdown("### 用途と適用範囲")
    st.markdown(
        "本ツールは**初期スクリーニング・概念検証**用途。商用 CAE "
        "（Moldflow / Moldex3D 等）の置き換えではない。\n\n"
        "- ◯ 向く：ゲート位置候補の比較、ランナー形状の方向性決定、"
        "**極薄プレート（$t < 0.5$ mm）の壁面冷却・剪断発熱の効きの可視化**、"
        "プレート薄肉化時のショートショット予兆、フローバランサー（▽肉盗み）の段数効果\n"
        "- × 向かない：寸法精度予測、保圧設計、収縮反り、面内コーナー流動詳細、最終肉厚分布の精密計算"
    )


# NOTE: Material DB schema version is embedded as a cache key so that
# Streamlit Cloud invalidates the @st.cache_resource entry whenever the
# Material dataclass shape changes (e.g. new field added). Without this,
# old Material instances pickled in the deploy's persistent cache lack
# the new attribute and the solver raises AttributeError.
# Bump this string when you add/remove fields on `core.materials.Material`.
_MATERIAL_DB_SCHEMA_VERSION = "v2_shear_heating"


@st.cache_resource
def _load_db(_schema_version: str = _MATERIAL_DB_SCHEMA_VERSION) -> MaterialDB:
    return MaterialDB()


db = _load_db()
material_keys = list(db.keys())


# ----------------------- sidebar: inputs -----------------------
with st.sidebar:
    _hdr_col, _run_col = st.columns([1.4, 1], vertical_alignment="bottom")
    with _hdr_col:
        st.header("成形品設計")
    with _run_col:
        do_run = st.button("解析実行", type="primary", use_container_width=True)
    geom_source = st.radio(
        "入力",
        [
            "Film gate 1 (肉厚調整ゲート)",
            "Film gate 2 (ゲート位置可変)",
            "Direct gate (parametric)",
            "Profile gate (JSONスペック)",
        ],
        index=1,
        key="geom_source",
    )

    if geom_source.startswith("Direct gate"):
        with st.expander("製品・ゲート形状", expanded=False):
            # Match Film gate defaults: plate 300×50, with the optional 2-zone
            # split (gate-side 0.35 / far-side 0.50, switching at 20 mm from
            # the gate-side edge).
            plate_w = st.slider("製品幅 Wp [mm]", 40.0, 300.0, 300.0, step=5.0)
            plate_h = st.slider("製品高さ Hp [mm]", 30.0, 200.0, 50.0, step=5.0)

            st.markdown("**製品肉厚（ゲート側／反ゲート側で2層化可）**")
            plate_split_dg = st.slider(
                "段差位置 [mm]",
                0.0,
                float(plate_h),
                min(20.0, float(plate_h)),
                step=1.0,
                help="ゲート側辺（下辺）からの距離。0 で均一肉厚。",
            )
            if plate_split_dg > 0:
                plate_lower_thk_dg = st.slider(
                    "ゲート側肉厚 [mm]",
                    0.2,
                    2.0,
                    0.35,
                    step=0.05,
                )
                plate_upper_thk_dg = st.slider(
                    "反ゲート側肉厚 [mm]",
                    0.2,
                    2.0,
                    0.50,
                    step=0.05,
                )
                plate_thk = float(plate_lower_thk_dg)
            else:
                plate_thk = st.slider("製品肉厚 [mm]", 0.2, 4.0, 0.4, step=0.1)
                plate_lower_thk_dg = float(plate_thk)
                plate_upper_thk_dg = float(plate_thk)

            st.markdown("**ダイレクトゲート**")
            gate_diameter = st.slider(
                "ゲート径 Φ [mm]",
                1.0,
                10.0,
                3.0,
                step=0.5,
                help="製品内部に配置されるバルブゲート円の直径。",
            )
            # ゲートはプレート内部にあるので、offset の上限はプレート高さに
            # 制約される（ゲート円が反対端を突き抜けない）。下限はゲート半径
            # （ゲート円がゲート側辺を突き抜けない）。
            _g_off_min = float(gate_diameter / 2.0)
            _g_off_max = float(plate_h - gate_diameter / 2.0)
            _g_off_default = max(_g_off_min, min(20.0, _g_off_max))
            gate_offset = st.slider(
                "ゲート位置（ゲート側辺から内側へ）[mm]",
                _g_off_min,
                _g_off_max,
                _g_off_default,
                step=1.0,
                help=(
                    "プレートのゲート側辺（下辺）から内側へ何 mm の位置に"
                    "ゲート中心を置くか。ゲートはプレート内部に直接ある"
                    "（ランナーもスプルーもない、垂直に注入する）。"
                ),
            )
            cell_size = st.slider("メッシュ粗さ [mm/cell]", 0.2, 3.0, 0.5, step=0.1)
            st.caption(
                "細かいほど解析精度が上がり 3D 表示も滑らかになるが、解析が重くなる"
                "（0.2mm・層別で1回数十秒）。普段は0.5で速く回し、精密な結果や"
                "滑らかな3Dが要るときだけ下げる。"
            )
    elif geom_source.startswith("Film gate 2"):
        with st.expander("製品形状", expanded=False):
            plate_w_f2 = st.slider("製品幅 Wp [mm]", 40.0, 300.0, 300.0, step=5.0)
            plate_h_f2 = st.slider("製品高さ Hp [mm]", 30.0, 200.0, 50.0, step=5.0)

            plate_2layer_f2 = st.checkbox(
                "製品肉厚を2層化する（ゲート側／反ゲート側）",
                value=True,
                help="ONで段差位置を境にゲート側・反ゲート側を別肉厚に。OFFで均一肉厚。",
            )
            if plate_2layer_f2:
                plate_split_f2 = st.slider(
                    "段差位置（製品長辺から）[mm]",
                    1.0,
                    float(plate_h_f2),
                    min(20.0, float(plate_h_f2)),
                    step=1.0,
                    help="製品長辺からの距離。ここを境に肉厚が切り替わる。",
                )
                plate_lower_f2 = st.slider("ゲート側肉厚 [mm]", 0.2, 2.0, 0.35, step=0.05)
                plate_upper_f2 = st.slider("反ゲート側肉厚 [mm]", 0.2, 2.0, 0.50, step=0.05)
                plate_thk_f2 = float(plate_lower_f2)
            else:
                plate_thk_f2 = st.slider("製品肉厚 [mm]", 0.2, 2.0, 0.4, step=0.1)
                plate_split_f2 = 0.0
                plate_lower_f2 = float(plate_thk_f2)
                plate_upper_f2 = float(plate_thk_f2)

        with st.expander("ゲート形状", expanded=False):
            st.markdown("**ゲート台形（直角台形）**")
            gate_depth_f2 = st.slider(
                "ゲート高さ D [mm]（注入点側）",
                10.0,
                60.0,
                20.0,
                step=1.0,
                help="製品長辺から注入点までの距離（台形の高さ）。",
            )
            gate_position_f2 = st.slider(
                "ゲート位置 [mm]（0=右端 / Wp÷2=中央=二等辺）",
                0.0,
                float(plate_w_f2),
                0.0,
                step=5.0,
                help="注入バルブゲートの右端からの距離。0で直角台形、Wp÷2で二等辺。",
            )
            left_edge_f2 = st.slider(
                "左端の高さ [mm]",
                0.0,
                min(15.0, float(gate_depth_f2)),
                min(10.0, float(gate_depth_f2)),
                step=1.0,
                help="製品長辺端での台形高さ（台形化＝左端の尖り防止）。",
            )
            valve_f2 = st.slider("バルブゲート径 Φ [mm]", 1.0, 10.0, 3.0, step=0.5)
            _gate_left_offset_max_f2 = float(max(plate_w_f2 - gate_position_f2 - 5.0, 0.0))

            st.markdown("**ゲートランド（製品長辺接続部）**")
            land_width_f2 = st.slider("ランド幅 [mm]", 1.0, 5.0, 1.0, step=0.5)
            land_depth_f2 = st.slider("ランド深さ [mm]", 0.2, 2.0, 0.35, step=0.05)

            st.markdown("**テーパ面形状**")
            mid_a_f2 = st.slider(
                "1段目の深さ [mm]（厚・深ランナー寄り）",
                0.2,
                5.0,
                2.0,
                step=0.1,
                help="テーパ面の基本段（深ランナー寄り・厚め）の肉厚。",
            )
            taper1_len_f2 = st.slider(
                "上段テーパ長 L1 [mm]",
                1.0,
                30.0,
                8.0,
                step=0.5,
                help="ランドから1段目深さに達するまでの底辺距離。",
            )
            taper_2stage_f2 = st.checkbox(
                "テーパ面を左右で段化する（右側に薄い2段目を追加）",
                value=True,
                help=(
                    "ONで製品幅方向に左右分割。左=1段テーパのみ、"
                    "右=薄い2段目を足した2段テーパ。OFFで全幅が単一テーパ。"
                ),
            )
            if taper_2stage_f2:
                gate_left_offset_f2 = st.slider(
                    "2段境界の位置（製品左端から）[mm]",
                    0.0,
                    _gate_left_offset_max_f2,
                    min(150.0, _gate_left_offset_max_f2),
                    step=5.0,
                    help="この位置より右に薄い2段目が入る。左は1段テーパのみ。",
                )
                mid_b_f2 = st.slider(
                    "2段目の深さ [mm]（薄・製品長辺側）",
                    0.2,
                    5.0,
                    1.8,
                    step=0.1,
                    help="ランド直後（製品長辺側）の薄い段の肉厚。1段目との境界はスロープで連続。",
                )
                taper2_left_f2 = st.slider(
                    "2段目テーパ 端側の最遠点 [mm]",
                    1.0,
                    20.0,
                    5.0,
                    step=0.5,
                    help="2段目の製品長辺から一番離れた点。端（楔先端）側。",
                )
                taper2_right_f2 = st.slider(
                    "2段目テーパ 注入点側の最遠点 [mm]",
                    1.0,
                    20.0,
                    10.0,
                    step=0.5,
                    help="2段目の製品長辺から一番離れた点。注入点側（台形の大きい辺）。",
                )
            else:
                # 全幅で完全に同一の単一テーパにする（gate_position に依存しない）。
                # gate_left_offset=0 で has_2nd を全幅 True にし、2段目の遠点を
                # ランド端 (land_width) に潰す（far2=w_land 一定）と in2 は空になり、
                # in1 が mid_b→mid_a を [w_land, w_land+L1] で描く。mid_b=land_depth に
                # すると in1 は land_depth→mid_a の単一ランプ（長さ L1）になり、
                # ランプ開始が land と連続。far2 一定なので左右で差が出ず、
                # far_max=land_width なので validate の extent 膨張もない。
                gate_left_offset_f2 = 0.0
                mid_b_f2 = float(land_depth_f2)
                taper2_left_f2 = float(land_width_f2)
                taper2_right_f2 = float(land_width_f2)

            st.markdown("**深ランナー（左斜辺沿い・台形断面）**")
            runner_depth_f2 = st.slider("深さ [mm]", 2.0, 5.0, 3.0, step=0.5)
            runner_top_f2 = st.slider("上底（開口幅）[mm]", 2.0, 5.0, 4.0, step=0.5)
            runner_bottom_f2 = st.slider("下底（底幅・抜き勾配）[mm]", 1.0, 3.0, 2.0, step=0.5)

            land_step_f2 = st.checkbox(
                "ランド↔テーパ境界に段差をつける",
                value=True,
                help="ランド直後のテーパ始端深さを独立指定して段差を作る。OFFで連続（段差なし）。",
            )
            if land_step_f2:
                taper2_near_f2 = st.slider(
                    "ランド↔2段目テーパ 境界深さ [mm]（2段目がある領域）",
                    0.2,
                    float(runner_depth_f2),
                    min(float(land_depth_f2), float(runner_depth_f2)),
                    step=0.05,
                    help="2段目テーパのランド側始端の深さ。ランド深さと変えると段差。上限は深ランナー深さ。",
                )
                taper1_near_f2 = st.slider(
                    "ランド↔1段目テーパ 境界深さ [mm]（2段目が無い領域）",
                    0.2,
                    float(runner_depth_f2),
                    min(1.50, float(runner_depth_f2)),
                    step=0.05,
                    help="2段目が無い左側で、1段目テーパのランド側始端の深さ。上限は深ランナー深さ。",
                )
            else:
                taper2_near_f2 = None
                taper1_near_f2 = None

        with st.expander("メッシュ", expanded=False):
            cell_size_f2 = st.slider("メッシュ粗さ [mm/cell]", 0.2, 3.0, 0.5, step=0.1)
            st.caption(
                "細かいほど解析精度が上がり 3D 表示も滑らかになるが、解析が重くなる"
                "（0.2mm・層別で1回数十秒）。普段は0.5で速く回し、精密な結果や"
                "滑らかな3Dが要るときだけ下げる。"
            )
    elif geom_source.startswith("Film gate 1"):
        # Parametric "肉厚調整ゲート" (ranner-block depth field) -- the same
        # family as the Profile gate JSON (land / main ramp / island / well /
        # outer wall), but driven by sliders and assembled into a
        # ``GateProfileSpec`` here. The defaults reproduce
        # hamoko_gate_furiwake_20260703 so the sliders start on a real
        # balanced-gate design; the derived quantities below (boundary-line
        # t-endpoints, outer-wall start half-width, well floor) are tied to
        # the major dimensions the way that drawing ties them.
        with st.expander("製品形状", expanded=False):
            plate_w = st.slider("製品幅 Wp [mm]", 40.0, 400.0, 300.0, step=5.0)
            plate_h = st.slider("製品高さ Hp [mm]", 30.0, 200.0, 50.0, step=5.0)

            plate_2layer_f1 = st.checkbox(
                "製品肉厚を2層化する（ゲート側／反ゲート側）",
                value=True,
                help="ONで段差位置を境にゲート側・反ゲート側を別肉厚に。OFFで均一肉厚。",
                key="f1_plate_2layer",
            )
            if plate_2layer_f1:
                plate_split = st.slider(
                    "段差位置（製品長辺から）[mm]",
                    1.0,
                    float(plate_h),
                    min(20.0, float(plate_h)),
                    step=1.0,
                    help="製品長辺からの距離。ここを境に肉厚が切り替わる。",
                )
                plate_lower_thk = st.slider("ゲート側肉厚 [mm]", 0.2, 2.0, 0.35, step=0.05)
                plate_upper_thk = st.slider("反ゲート側肉厚 [mm]", 0.2, 2.0, 0.50, step=0.05)
                plate_thk = float(plate_lower_thk)
            else:
                plate_thk = st.slider("製品肉厚 [mm]", 0.2, 2.0, 0.4, step=0.1)
                plate_split = 0.0
                plate_lower_thk = float(plate_thk)
                plate_upper_thk = float(plate_thk)

        with st.expander("ゲートブロック（肉厚調整ゲート）", expanded=False):
            st.caption(
                "t = ゲート出口（製品長辺）からの距離、w = バルブ軸からの半幅。"
                "左右対称。深さ = 流路肉厚。"
            )
            gate_exit_width = st.slider(
                "ゲート出口幅 [mm] (≤ 製品幅)",
                min_value=10.0,
                max_value=float(plate_w),
                value=float(min(298.0, plate_w)),
                step=1.0,
            )

            st.markdown("**ランド（出口）**")
            land_depth = st.slider("ランド深さ [mm]", 0.1, 2.0, 0.35, step=0.05)
            land_length = st.slider("ランド長さ [mm]", 0.5, 5.0, 1.0, step=0.1)

            st.markdown("**メインランプ**")
            ramp_angle = st.number_input(
                "ランプ角 [deg]",
                min_value=1.0,
                max_value=45.0,
                value=10.95,
                step=0.05,
                format="%.2f",
                help="ランド終端から深さが tan(角)·(t − ランド長) で増える。",
            )
            ramp_cap = st.slider(
                "ランプ上限深さ [mm] (≥ ランド深さ)",
                min_value=float(land_depth),
                max_value=10.0,
                value=float(max(2.5, land_depth)),
                step=0.1,
            )

            st.markdown("**アイランド（中央の浅い帯＝振り分け）**")
            island_on = st.checkbox(
                "アイランドを有効化",
                value=True,
                key="f1_island_on",
                help=(
                    "中央帯だけランプ角を緩くして流路を絞り、樹脂を両端へ振り分ける。"
                    "境界線はランド終端 (t=ランド長) からアイランド終端 (t=end) まで"
                    "の直線で、半幅を出口側・終端側の2点で指定する。"
                ),
            )
            if island_on:
                island_angle = st.number_input(
                    "アイランド角 [deg] (≤ ランプ角)",
                    min_value=0.0,
                    max_value=float(ramp_angle),
                    value=float(min(2.5, ramp_angle)),
                    step=0.05,
                    format="%.2f",
                )
                island_end = st.slider(
                    "アイランド終端 t_end [mm] (> ランド長)",
                    min_value=float(land_length + 0.5),
                    max_value=60.0,
                    value=float(max(17.0, land_length + 0.5)),
                    step=0.1,
                )
                island_w_near = st.slider(
                    "境界半幅（出口側、t=ランド長）[mm]",
                    min_value=1.0,
                    max_value=float(gate_exit_width / 2.0),
                    value=float(min(52.7, gate_exit_width / 2.0)),
                    step=0.1,
                )
                island_w_far = st.slider(
                    "境界半幅（終端側、t=t_end）[mm]",
                    min_value=0.5,
                    max_value=float(gate_exit_width / 2.0),
                    value=float(min(10.0, gate_exit_width / 2.0)),
                    step=0.1,
                )
            else:
                island_angle = 0.0
                island_end = 0.0
                island_w_near = 0.0
                island_w_far = 0.0

            st.markdown("**外壁線（ポケット外形）**")
            st.caption("出口側は t=外壁開始 までゲート出口の全幅、そこから終端へ直線で狭まる。")
            wall_t1 = st.slider("外壁開始 t [mm]", 0.0, 30.0, 3.0, step=0.1)
            wall_t2 = st.slider(
                "外壁終端 t [mm] (> 開始)",
                min_value=float(wall_t1 + 0.5),
                max_value=80.0,
                value=float(max(23.3, wall_t1 + 0.5)),
                step=0.1,
            )
            wall_w2 = st.slider(
                "外壁終端の半幅 [mm]",
                min_value=0.5,
                max_value=float(gate_exit_width / 2.0),
                value=float(min(4.5, gate_exit_width / 2.0)),
                step=0.1,
            )

            st.markdown("**井戸（バルブ周りの長穴ポケット）**")
            well_on = st.checkbox("井戸を有効化", value=True, key="f1_well_on")
            if well_on:
                well_t1 = st.slider("井戸開始 t [mm]", 0.0, 60.0, 15.5, step=0.1)
                well_t2 = st.slider(
                    "井戸終端 t [mm] (> 開始)",
                    min_value=float(well_t1 + 0.5),
                    max_value=80.0,
                    value=float(max(27.5, well_t1 + 0.5)),
                    step=0.1,
                )
                well_half_w = st.slider("井戸半幅 [mm]", 0.5, 20.0, 4.5, step=0.1)
                # The 60° wall climbs from the rim, so the deepest point the
                # pocket can reach is half_width·tan(60°) at the centreline.
                # A deeper request would be accepted by the spec, recorded in
                # settings.json, and silently built shallower (Codex P1).
                _well_depth_max = float(
                    min(15.0, well_half_w * math.tan(math.radians(_WELL_WALL_ANGLE_DEG)))
                )
                _well_depth_max = max(0.5, math.floor(_well_depth_max * 10.0) / 10.0)
                well_depth = st.slider(
                    "井戸深さ [mm] (≤ 半幅·tan60°)",
                    0.5,
                    _well_depth_max,
                    float(min(4.5, _well_depth_max)),
                    step=0.1,
                    help="壁角 60° で到達できる最大深さ = 半幅 × tan(60°)。",
                )
                _well_t_mid = 0.5 * (well_t1 + well_t2)
                _pocket_t_end = max(float(wall_t2), float(well_t2))
            else:
                well_t1 = well_t2 = well_half_w = well_depth = 0.0
                _well_t_mid = 0.5 * (wall_t1 + wall_t2)
                _pocket_t_end = float(wall_t2)

            st.markdown("**バルブゲート**")
            valve_d = st.slider("バルブオリフィス径 [mm]", 1.0, 10.0, 3.0, step=0.5)
            # Keep the orifice inside the pocket along t. Outside it the
            # builder snaps the gate to the nearest masked cell and the solver
            # injects somewhere other than the recorded position (Codex P1).
            _valve_t_min = float(valve_d / 2.0)
            _valve_t_max = float(max(_valve_t_min + 0.1, _pocket_t_end - valve_d / 2.0))
            valve_t = st.slider(
                "バルブ位置 t [mm] (ポケット内)",
                _valve_t_min,
                _valve_t_max,
                float(min(max(round(_well_t_mid, 1), _valve_t_min), _valve_t_max)),
                step=0.1,
                help=(
                    "既定は井戸の中央（井戸 OFF なら外壁線の中点）。"
                    "上限はポケット終端（外壁終端／井戸終端の遠い方）− 半径。"
                ),
            )

            cell_size = st.slider("メッシュ粗さ [mm/cell]", 0.2, 3.0, 1.0, step=0.1)
            st.caption(
                "この形状の想定解像度は 1.0mm（ランド長 1mm が 1 セル）。"
                "細かいほど深さ場の再現精度が上がるが、解析が重くなる。"
            )
    elif geom_source.startswith("Profile gate"):
        with st.expander("ゲートプロファイル (JSON)", expanded=False):
            spec_mode_pg = SPEC_MODE_BY_LABEL[
                st.radio(
                    "スペック入力",
                    list(SPEC_MODE_LABELS.values()),
                    horizontal=True,
                    key="spec_mode_pg",
                    help=(
                        "図面から抽出したゲートブロック深さ場の JSON スペックを読み込む。"
                        "実図面由来のスペックはリポジトリに含めず、ここでローカル読込する運用。"
                    ),
                )
            ]
            upload_pg = None
            json_text_pg = ""
            local_spec_pg = None

            if spec_mode_pg is SpecMode.LOCAL:
                spec_root_pg = spec_root(APP_DIR)
                # Read the uploader out of session state *before* drawing it, so
                # the dropdown can be disabled in the same run the file lands
                # rather than one rerun later. Streamlit writes widget state
                # before rerunning the script, so this is the current value, not
                # a stale one.
                dropped_pg = st.session_state.get(SPEC_UPLOAD_KEY)
                if spec_root_pg is not None:
                    try:
                        found_pg = list_spec_files(spec_root_pg)
                    except OSError:
                        # Never render the exception: its message carries the
                        # absolute path, which names the customer and the job.
                        found_pg = []
                        st.error("スペックフォルダを読めない（権限またはマウント切れ）。")
                    picked_pg = st.selectbox(
                        "スペック",
                        [SPEC_UNSELECTED] + [f.name for f in found_pg],
                        index=0,
                        disabled=dropped_pg is not None,
                        key="spec_pick_pg",
                        help="リポジトリ外のローカルフォルダにあるスペック。",
                    )
                    if picked_pg != SPEC_UNSELECTED:
                        local_spec_pg = spec_root_pg / picked_pg
                elif spec_link_exists(APP_DIR):
                    # Something is at the link path but does not resolve to a
                    # directory. Staying silent here would look identical to the
                    # feature simply not existing, which is what the person who
                    # set it up would least expect.
                    st.caption(f"{SPEC_LINK_NAME} がディレクトリとして解決できない（リンク切れ）。")
                upload_pg = st.file_uploader(
                    "またはスペック JSON をドロップ", type=["json"], key=SPEC_UPLOAD_KEY
                )
            elif spec_mode_pg is SpecMode.PASTE:
                json_text_pg = st.text_area(
                    "スペック JSON を貼り付け", height=240, placeholder='{\n  "name": ...\n}'
                )
            else:
                st.caption(f"同梱デモ: {DEMO_PROFILE_JSON.name}（架空寸法）")

            spec_origin_pg = choose_spec_origin(
                spec_mode_pg,
                has_upload=upload_pg is not None,
                has_local=local_spec_pg is not None,
                has_paste=bool(json_text_pg.strip()),
            )
            # Say which source won, here in the sidebar where the controls are.
            # The geometry is built in the main column, so a notice raised there
            # would sit far from the two widgets that disagree. Only file names
            # are ever shown -- the directory above them names the customer.
            if spec_origin_pg is SpecOrigin.UPLOAD:
                if local_spec_pg is not None:
                    st.info(
                        f"読込元: アップロード **{upload_pg.name}**"
                        "（一覧の選択は使わない。解除はアップロード欄の ✕）"
                    )
                else:
                    st.caption(f"読込元: アップロード {upload_pg.name}")
            elif spec_origin_pg is SpecOrigin.LOCAL:
                st.caption(f"読込元: {local_spec_pg.name}")

        with st.expander("製品形状", expanded=False):
            plate_w_pg = st.slider("製品幅 Wp [mm]", 40.0, 400.0, 300.0, step=5.0)
            plate_h_pg = st.slider("製品高さ Hp [mm]", 30.0, 200.0, 50.0, step=5.0)

            plate_2layer_pg = st.checkbox(
                "製品肉厚を2層化する（ゲート側／反ゲート側）",
                value=True,
                help="ONで段差位置を境にゲート側・反ゲート側を別肉厚に。OFFで均一肉厚。",
            )
            if plate_2layer_pg:
                plate_split_pg = st.slider(
                    "段差位置（製品長辺から）[mm]",
                    1.0,
                    float(plate_h_pg),
                    min(20.0, float(plate_h_pg)),
                    step=1.0,
                    help="製品長辺からの距離。ここを境に肉厚が切り替わる。",
                )
                plate_lower_pg = st.slider("ゲート側肉厚 [mm]", 0.2, 2.0, 0.35, step=0.05)
                plate_upper_pg = st.slider("反ゲート側肉厚 [mm]", 0.2, 2.0, 0.50, step=0.05)
                plate_thk_pg = float(plate_lower_pg)
            else:
                plate_thk_pg = st.slider("製品肉厚 [mm]", 0.2, 2.0, 0.4, step=0.1)
                plate_split_pg = 0.0
                plate_lower_pg = float(plate_thk_pg)
                plate_upper_pg = float(plate_thk_pg)

        with st.expander("メッシュ", expanded=False):
            cell_size_pg = st.slider("メッシュ粗さ [mm/cell]", 0.2, 3.0, 1.0, step=0.1)
            st.caption(
                "スペックの想定解像度は 1.0mm。細かいほど深さ場の再現精度が上がるが、"
                "解析が重くなる。"
            )

    with st.expander("材料", expanded=False):
        material_key = st.selectbox("樹脂", material_keys, index=material_keys.index("PP_T20"))
        mat = db[material_key]
        st.caption(f"{mat.name}")
        st.caption(
            f"推奨 melt: {mat.T_melt_recommended[0] - 273.15:.0f}–{mat.T_melt_recommended[1] - 273.15:.0f} ℃, "
            f"mold: {mat.T_mold_recommended[0] - 273.15:.0f}–{mat.T_mold_recommended[1] - 273.15:.0f} ℃"
        )

    with st.expander("射出条件", expanded=False):
        _melt_min = int(mat.T_melt_recommended[0] - 273.15) - 20
        _melt_max = int(mat.T_melt_recommended[1] - 273.15) + 20
        melt_C = st.slider(
            "樹脂温度 [℃]",
            _melt_min,
            _melt_max,
            max(_melt_min, min(260, _melt_max)),
        )
        _mold_min = int(mat.T_mold_recommended[0] - 273.15) - 10
        _mold_max = int(mat.T_mold_recommended[1] - 273.15) + 30
        mold_C = st.slider(
            "金型温度 [℃]",
            _mold_min,
            _mold_max,
            max(_mold_min, min(50, _mold_max)),
        )
        inj_v = st.slider("射出速度 [mm/s] (代表)", 5.0, 400.0, 200.0, step=5.0)
        inj_Q = st.slider(
            "射出率 [cm³/s]",
            1.0,
            800.0,
            589.0,
            step=1.0,
            help="ソディック等の成形機取説の射出率に対応。",
        )

    with st.expander("壁面冷却モデル", expanded=False):
        wall_model = st.radio(
            "壁面冷却の表現",
            options=("none", "skin", "multilayer"),
            index=2,
            key="wall_model",
            format_func=lambda m: {
                "none": "なし（等温・代表粘度のみ）",
                "skin": "スキン層 (1層 + Stefan/Neumann)",
                "multilayer": "層別 (N 層離散化 + Cross-WLF 結合)",
            }[m],
            help=(
                "なし: 既存 HeleShawSolver 相当、温度結合なし。\n"
                "スキン層: 壁面で固化するスキン層を s(t)=c_skin·√(αt) で取り込み、"
                "コア層 h_core=h-2s だけが流れる。ショートショットも検出。\n"
                "層別: 厚み方向を N 層に分割、Neumann 1D 温度プロファイルから "
                "層別粘度を Cross-WLF で評価。fixed-point で τ ↔ T_k ↔ η_k を結合。\n"
                "極薄プレート (t<0.5mm) では層別を推奨。"
            ),
        )

        # default container (so downstream `solver = HeleShawSolver(...)` /
        # `MultilayerHeleShawSolver(...)` always has the kwargs it expects).
        # 極薄プレート (t0.35〜0.50 想定) 向けに既定値を調整:
        #   モード: 層別 (index=2)
        #   層数 N: 7 (壁勾配が急なので N=5 から増量)
        #   反復上限: 12 (収束が遅くなりがちなので上限緩め)
        skin_on = wall_model == "skin"
        c_skin = 0.0
        skin_max_iter = 5
        skin_tol = 1e-3
        multilayer_on = wall_model == "multilayer"
        num_layers = 7
        layer_distribution = "wall_refined"
        multilayer_max_iter = 12
        multilayer_tol = 1e-3
        solid_fraction = 0.3

        if wall_model == "skin":
            c_skin = st.slider(
                "スキン層成長定数 c_skin",
                0.0,
                2.0,
                1.0,
                step=0.05,
                help="0で OFF と同等。1.0 付近が物理的代表値。薄肉ほど効果大。",
            )
            skin_max_iter = st.slider(
                "fixed-point 反復上限",
                1,
                10,
                5,
                help="τ ↔ h_core 結合の反復回数。3〜5で十分なケースが多い。",
            )
            skin_tol_log10 = st.slider(
                "収束判定 log10(tol)",
                -5,
                -1,
                -3,
                help="τ場の相対L2変化が 10^tol を下回ったら収束。",
            )
            skin_tol = 10.0 ** float(skin_tol_log10)
        elif wall_model == "multilayer":
            num_layers = st.slider(
                "層数 N",
                3,
                9,
                7,
                help=(
                    "厚み方向の離散化数。奇数で中央層がショートショット判定の代表セルに。"
                    "極薄プレートでは壁勾配が急なので N=7 を推奨。"
                ),
            )
            layer_distribution = st.radio(
                "層分布",
                options=("wall_refined", "uniform"),
                index=0,
                format_func=lambda m: {
                    "wall_refined": "壁近傍密 (Chebyshev-Lobatto)",
                    "uniform": "等間隔",
                }[m],
                help=(
                    "wall_refined: ζ_k = 0.5·(1 - cos(πk/N))。Neumann 勾配の急な"
                    "壁面で解像度を稼ぐ。layer 数が同じなら推奨。\n"
                    "uniform: 等間隔。デバッグ・解析比較用。"
                ),
            )
            multilayer_max_iter = st.slider(
                "fixed-point 反復上限",
                1,
                20,
                12,
                help=(
                    "τ ↔ T_k ↔ η_k 結合の反復回数。極薄プレートでは収束が遅くなりがちなので 12 を推奨。"
                ),
            )
            multilayer_tol_log10 = st.slider(
                "収束判定 log10(tol)",
                -5,
                -1,
                -3,
                help="τ場の相対L2変化が 10^tol を下回ったら収束。",
            )
            multilayer_tol = 10.0 ** float(multilayer_tol_log10)
            solid_fraction = st.slider(
                "固化判定 fraction",
                0.0,
                0.9,
                0.3,
                step=0.05,
                help=(
                    "中央層温度が T_mold + fraction·(T_melt - T_mold) を下回るセルを"
                    "ショートショットにマーク。PP は 0.3 が目安。"
                ),
            )
            shear_heating_enabled = st.checkbox(
                "剪断発熱補正 (viscous dissipation, 段階1)",
                value=True,
                help=(
                    "ON で Neumann 温度に剪断発熱補正項 ΔT_k = (η_k·γ̇_k²)·min(t_arr, τ_thermal)/(ρ·cp) を加算。"
                    "τ_thermal = h²/(π²·α) で頭打ち。"
                    "極薄プレート (t<0.5mm) では Brinkman 数 Br ≫ 1 になりがちなので推奨。"
                    "OFF でも Br 数は結果ペインに表示されるので、必要性を事前判定できる。"
                ),
            )
        else:
            shear_heating_enabled = False

    with st.expander("射出圧縮成形 (ICM)", expanded=False):
        icm = st.checkbox("圧縮成形ON", value=False, key="icm_on")
        if icm:
            # ストローク (絶対加算) モードに統一。圧縮 mask 内の全セルに同じ絶対量を加算
            # するので段差プレートでも段差が保存される (金型シム量の物理に整合)。
            # 旧倍率モードは solver / CLI には後方互換で残しているが UI には出さない。
            comp_stroke = st.slider(
                "圧縮ストローク [mm]",
                0.0,
                2.0,
                0.70,
                step=0.05,
                help=(
                    "金型シム量。圧縮 mask セル全てに加算される絶対量。"
                    "段差プレートでも段差が圧縮位相中も保存される (実機の挙動と整合)。"
                ),
            )
            comp_factor = 1.0  # unused, kept for solver kwargs symmetry
            comp_frac = st.slider(
                "圧縮位相の充填占有率",
                0.1,
                1.0,
                0.60,
                step=0.05,
                help="充填全体に対し、圧縮位相 (型開き状態) で占める時間比率。",
            )
        else:
            comp_factor = 1.0
            comp_stroke = None
            comp_frac = 0.0

    with st.expander("ショートショット（計量制限）", expanded=False):
        # 二相モデル: (1) 射出相 = 型開きギャップで計量体積ぶん充填、
        # (2) 圧縮相 = 型閉じで溶融プールを等圧ソースとして前進（体積保存）。
        # 線形求解2回・時間積分なし。実機の計量値をそのまま入れて
        # 段階ショートショットの現物形状と直接比較する用途。
        two_phase_on = st.checkbox(
            "二相ショートショット解析ON",
            value=False,
            key="two_phase_on",
            help=(
                "計量を意図的に絞ったショートショットの最終形状を予測する。"
                "射出相（型開きギャップで計量体積まで充填）→ 圧縮相（型閉じで"
                "溶融プールを前進、体積保存）の二相。壁面冷却モデル『なし』専用"
                "（体積律速のショートショットは凍結の物理を含まない）。"
            ),
        )
        if two_phase_on and wall_model != "none":
            # 実行時の一過性警告だけだと rerun で消えて「ON にしたのに何も
            # 出ない」に見える。設定と同じ場所に常時出す。
            st.warning(
                "壁面冷却モデルが『なし』のときだけ実行される。"
                "現在の設定では二相解析はスキップされる。"
            )
        if two_phase_on:
            shot_volume_cm3 = st.number_input(
                "計量体積 V_shot [cm³]",
                min_value=0.01,
                value=5.0,
                step=0.1,
                key="two_phase_shot_volume",
                help=(
                    "実機の計量値（ショット体積）。キャビティ体積との比較は"
                    "実行後の結果ペインに出る。"
                ),
            )
            _hint_geom = st.session_state.get("mfs_geom")
            if _hint_geom is not None:
                _v_fin = _hint_geom.volume_cm3()
                _hint = f"参考（前回実行の形状）: 最終キャビティ体積 {_v_fin:.2f} cm³"
                if icm and comp_stroke is not None:
                    _v_open = _v_fin + comp_stroke * _hint_geom.compression_area_mm2() / 1000.0
                    _hint += f" / 開きギャップ体積 ≈ {_v_open:.2f} cm³"
                st.caption(_hint + "。計量が最終キャビティ体積以上だと完全充填になる。")
        else:
            shot_volume_cm3 = None

    with st.expander("出力", expanded=False):
        num_frames = st.slider("アニメーションフレーム数", 12, 60, 60)
        # 既定の turbo は商用 CAE と同じ虹配色。色相コントラストで等時線が
        # 読めるのが狙いで、赤=最後に充填=リスク箇所という意味とも一致する。
        # 赤緑色覚に配慮するときは cividis / viridis を選ぶ。
        fill_cmap = st.selectbox(
            "充填アニメの配色",
            options=["turbo", "jet", "viridis", "cividis"],
            index=0,
            format_func=lambda c: {
                "turbo": "turbo（既定・虹／偽の縞が出ない）",
                "jet": "jet（従来の商用 CAE と同じ虹）",
                "viridis": "viridis（知覚均等・色覚配慮）",
                "cividis": "cividis（色覚配慮を最優先）",
            }[c],
            help="虹系は等時線の形が読みやすく、viridis / cividis は量の大小比較と色覚配慮に向く。",
        )
        iso_levels = st.slider(
            "等時線の本数",
            0,
            24,
            ISOCHRONE_LEVELS,
            help="同時に充填される位置を結んだ線。線が詰まる=流れが遅い、"
            "ぶつかる=ウェルド、途切れた先=最後に充填。0 で非表示。",
        )

    # Version / build label.
    # Rendered here (end of the sidebar) rather than at the end of the script
    # because the main flow has several ``st.stop()`` calls for parameter
    # validation — a footer placed after them would vanish exactly when a user
    # screenshots an error and asks which build produced it.
    st.divider()
    st.caption(build_label())


# ----------------------- main panel -----------------------
def build_geometry() -> tuple[Geometry, dict]:
    """Return the geometry and a record of the inputs that produced it.

    The settings travel with the results ZIP so a downloaded run can be
    reproduced without measuring the images and solving for the volume.
    """
    if geom_source.startswith("Direct gate"):
        try:
            cfg_dg = DirectGateConfig(
                plate_w_mm=plate_w,
                plate_h_mm=plate_h,
                plate_thk_mm=plate_thk,
                gate_diameter_mm=gate_diameter,
                gate_offset_mm=gate_offset,
                cell_size_mm=cell_size,
                plate_split_height_mm=plate_split_dg if plate_split_dg > 0 else 0.0,
                plate_lower_thk_mm=plate_lower_thk_dg if plate_split_dg > 0 else None,
                plate_upper_thk_mm=plate_upper_thk_dg if plate_split_dg > 0 else None,
            )
            return build_direct_gate_geometry(cfg_dg), config_settings(geom_source, cfg_dg)
        except ValueError as exc:
            st.error(f"パラメータ不整合: {exc}")
            st.stop()
    if geom_source.startswith("Film gate 2"):
        try:
            cfg_f2 = FilmGate2Config(
                plate_w_mm=plate_w_f2,
                plate_h_mm=plate_h_f2,
                plate_thk_mm=plate_thk_f2,
                gate_depth_mm=gate_depth_f2,
                gate_position_mm=gate_position_f2,
                gate_left_offset_mm=gate_left_offset_f2,
                left_edge_mm=left_edge_f2,
                land_width_mm=land_width_f2,
                land_depth_mm=land_depth_f2,
                taper1_len_mm=taper1_len_f2,
                mid_depth_a_mm=mid_a_f2,
                mid_depth_b_mm=mid_b_f2,
                taper2_near_depth_mm=taper2_near_f2,
                taper1_near_depth_mm=taper1_near_f2,
                taper2_left_mm=taper2_left_f2,
                taper2_right_mm=taper2_right_f2,
                runner_depth_mm=runner_depth_f2,
                runner_top_mm=runner_top_f2,
                runner_bottom_mm=runner_bottom_f2,
                valve_gate_diameter_mm=valve_f2,
                cell_size_mm=cell_size_f2,
                plate_split_height_mm=plate_split_f2 if plate_split_f2 > 0 else 0.0,
                plate_lower_thk_mm=plate_lower_f2 if plate_split_f2 > 0 else None,
                plate_upper_thk_mm=plate_upper_f2 if plate_split_f2 > 0 else None,
            )
            return build_film_gate2_geometry(cfg_f2), config_settings(geom_source, cfg_f2)
        except ValueError as exc:
            st.error(f"パラメータ不整合: {exc}")
            st.stop()
    if geom_source.startswith("Film gate 1"):
        try:
            _t_land = float(land_length)
            _well_f1 = None
            if well_on:
                # Floor extent follows from the sloped wall: the 60° wall eats
                # depth/tan(60°) of t at each end. When the pocket is too short
                # for a flat floor the floor range is simply not reported.
                _wall_eat = float(well_depth) / math.tan(math.radians(_WELL_WALL_ANGLE_DEG))
                _floor = (float(well_t1) + _wall_eat, float(well_t2) - _wall_eat)
                _well_f1 = WellSpec(
                    shape="obround",
                    t_range=(float(well_t1), float(well_t2)),
                    half_width=float(well_half_w),
                    depth=float(well_depth),
                    floor_t_range=_floor if _floor[1] > _floor[0] + 1e-6 else None,
                    wall_angle_deg=_WELL_WALL_ANGLE_DEG,
                )
            _island_f1 = None
            if island_on:
                _island_f1 = IslandSpec(
                    angle_deg=float(island_angle),
                    boundary_line=(
                        (_t_land, float(island_w_near)),
                        (float(island_end), float(island_w_far)),
                    ),
                    end_dist=float(island_end),
                )
            spec_f1 = GateProfileSpec(
                name="film_gate_1_parametric",
                units="mm",
                symmetric=True,
                gate_exit_width=float(gate_exit_width),
                land=LandSpec(depth=float(land_depth), length=_t_land),
                main_ramp=MainRampSpec(angle_deg=float(ramp_angle), cap_depth=float(ramp_cap)),
                outer_wall_line=(
                    (float(wall_t1), float(gate_exit_width) / 2.0),
                    (float(wall_t2), float(wall_w2)),
                ),
                valve=ValveSpec(t=float(valve_t), w=0.0, orifice_diameter=float(valve_d)),
                island=_island_f1,
                well=_well_f1,
            )
            plate_f1 = ProfilePlateConfig(
                plate_w_mm=plate_w,
                plate_h_mm=plate_h,
                plate_thk_mm=plate_thk,
                plate_split_height_mm=plate_split if plate_split > 0 else 0.0,
                plate_lower_thk_mm=plate_lower_thk if plate_split > 0 else None,
                plate_upper_thk_mm=plate_upper_thk if plate_split > 0 else None,
            )
            geom_f1 = build_profile_gate_geometry(spec_f1, plate_f1, cell_size_mm=cell_size)
            # The builder snaps a gate whose orifice misses the pocket to the
            # nearest masked cell. The slider bounds above keep the orifice
            # inside the pocket along t; this catches the width-wise miss
            # (orifice wider than the pocket tip) the bounds cannot express.
            _iy_v = int((plate_f1.pad_mm + spec_f1.t_max() - float(valve_t)) / cell_size)
            _ix_v = int((plate_f1.pad_mm + float(plate_w) / 2.0) / cell_size)
            if not (
                0 <= _iy_v < geom_f1.mask.shape[0]
                and 0 <= _ix_v < geom_f1.mask.shape[1]
                and geom_f1.mask[_iy_v, _ix_v]
            ):
                raise ValueError(
                    f"バルブ位置 t={valve_t:g} mm がポケットの外にある。"
                    "外壁終端／井戸の範囲内に移動するか、外壁終端の半幅を広げる。"
                )
            return geom_f1, config_settings(
                geom_source,
                plate_f1,
                cell_size_mm=cell_size,
                # Slider values, not a drawing: record the assembled spec in
                # full so the ZIP reproduces the geometry.
                gate_profile=dataclasses.asdict(spec_f1),
            )
        except ValueError as exc:
            st.error(f"パラメータ不整合: {exc}")
            st.stop()
    if geom_source.startswith("Profile gate"):
        try:
            if spec_origin_pg is SpecOrigin.UPLOAD:
                _spec_text = upload_pg.read().decode("utf-8")
                _spec_fp = file_fingerprint(upload_pg.name, _spec_text)
            elif spec_origin_pg is SpecOrigin.LOCAL:
                _spec_text = local_spec_pg.read_text(encoding="utf-8")
                # File name only. The directory holding it names the customer
                # and the job, and this record travels inside the results ZIP.
                _spec_fp = file_fingerprint(local_spec_pg.name, _spec_text)
            elif spec_origin_pg is SpecOrigin.PASTE:
                _spec_text = json_text_pg
                _spec_fp = file_fingerprint("(貼り付け)", _spec_text)
            elif spec_origin_pg is SpecOrigin.DEMO:
                _spec_text = DEMO_PROFILE_JSON.read_text(encoding="utf-8")
                _spec_fp = file_fingerprint(DEMO_PROFILE_JSON.name, _spec_text)
            else:
                st.warning("スペック JSON を一覧から選ぶか、アップロード／貼り付けしてください。")
                st.stop()
            spec_pg = GateProfileSpec.from_json(_spec_text)
            plate_pg = ProfilePlateConfig(
                plate_w_mm=plate_w_pg,
                plate_h_mm=plate_h_pg,
                plate_thk_mm=plate_thk_pg,
                plate_split_height_mm=plate_split_pg if plate_split_pg > 0 else 0.0,
                plate_lower_thk_mm=plate_lower_pg if plate_split_pg > 0 else None,
                plate_upper_thk_mm=plate_upper_pg if plate_split_pg > 0 else None,
            )
            return build_profile_gate_geometry(
                spec_pg, plate_pg, cell_size_mm=cell_size_pg
            ), config_settings(
                geom_source,
                plate_pg,
                cell_size_mm=cell_size_pg,
                # The spec usually comes off a real drawing and this ZIP is
                # made to be forwarded, so record which file it was -- not
                # what was in it. That rules out the spec's own ``name``
                # field too: it is content, and it is exactly the field that
                # carries a part or customer identifier.
                spec=_spec_fp,
            )
        except json.JSONDecodeError as exc:
            st.error(f"JSON構文エラー: {exc}")
            st.stop()
        except UnicodeDecodeError:
            # Before ValueError, which it subclasses.
            st.error("スペックファイルを UTF-8 として読めない。")
            st.stop()
        except OSError:
            # Deliberately drops the exception. Its message carries the absolute
            # path, and ``client.showErrorDetails`` defaults to "full", so an
            # uncaught OSError would print that path plus a traceback into the
            # page -- the same directory name the fingerprint is careful to omit.
            st.error("スペックファイルを読めない（権限またはマウント切れ）。")
            st.stop()
        except ValueError as exc:
            st.error(f"パラメータ不整合: {exc}")
            st.stop()
    # Every branch above returns, so this is reachable only if a new entry is
    # added to the input radio without a matching branch here.
    raise AssertionError(f"no geometry builder for input: {geom_source!r}")


col_left, col_right = st.columns([1, 1.3])

with col_left:
    st.subheader("成形品設計図")
    geom, geom_settings = build_geometry()
    fig_data = np.where(geom.mask, geom.thickness_mm, np.nan)
    st.write(
        f"格子: {geom.nx} × {geom.ny}, セル {geom.cell_size_mm} mm, 体積 {geom.volume_cm3():.2f} cm³"
    )
    fig_buf = io.BytesIO()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4), dpi=110)
    # Recenter on the gate (≒ product center). Tick "0" lines up with the
    # gate centroid; the half-circle on the gate-near side stays at y < 0
    # without the bottom spine running through it, since the spine keeps
    # its default position at the plot edge. Same convention is shared by
    # every result-time map in core/visualizer.py.
    x0_mm, y0_mm = geom.gate_origin_mm()
    extent = [
        -x0_mm,
        geom.nx * geom.cell_size_mm - x0_mm,
        -y0_mm,
        geom.ny * geom.cell_size_mm - y0_mm,
    ]
    im = ax.imshow(fig_data, origin="lower", extent=extent, cmap=THICKNESS_CMAP)
    for iy, ix in geom.gates:
        ax.plot(
            (ix + 0.5) * geom.cell_size_mm - x0_mm,
            (iy + 0.5) * geom.cell_size_mm - y0_mm,
            "ro",
            markersize=8,
            markeredgecolor="white",
        )
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_aspect("equal")
    ax.set_title("thickness map [mm]")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="h [mm]")
    fig.tight_layout()
    fig.savefig(fig_buf, format="png")
    plt.close(fig)
    st.image(fig_buf.getvalue())
    st.caption(
        "原点 (x, y) = (0, 0) はゲート中央 = 製品中央。ゲートブロック／ランナーは y < 0 側。赤丸はバルブゲート位置。"
    )


if do_run:
    # --- セル数ガード（メッシュ下限 0.2mm 開放に伴う安全弁）---
    # 最大寸法のプレート × 0.2mm では cavity が 100 万セルを超え得る。
    # 行列組立は Python ループ＋直接 spsolve、さらに層別モードは (N, ny, nx) の
    # 場を複数確保するため、Streamlit Cloud のメモリ枠を食い潰してプロセスごと
    # 落ちる恐れがある（Codex P1）。ソルバー起動前に cavity セル数を見て、
    # 安全上限を超えるなら明示して止める（クラッシュさせず綺麗に停止）。
    n_cavity = int(geom.mask.sum())
    if multilayer_on:
        # 層別はメモリが層数 N に比例するため上限を N で割る。
        cell_limit = max(120_000, 1_500_000 // num_layers)
    else:
        cell_limit = 500_000
    if n_cavity > cell_limit:
        _hint = (
            " 層別モードは厚み方向 N 層分のメモリを使うため上限が低めです。"
            "「なし」/「スキン層」に切り替えるか N を下げると緩和されます。"
            if multilayer_on
            else ""
        )
        st.error(
            f"この設定は格子が大きすぎます（cavity {n_cavity:,} セル ＞ 上限 "
            f"{cell_limit:,} セル）。メッシュ粗さを上げるか、製品・ランナー寸法を"
            f"小さくしてください。{_hint}"
            "（メモリ枯渇によるアプリのクラッシュを防ぐためのガードです。）"
        )
        st.stop()
    with st.spinner("Hele-Shaw方程式を解いている…"):
        if multilayer_on:
            solver = MultilayerHeleShawSolver(
                geometry=geom,
                material=mat,
                melt_temperature_K=melt_C + 273.15,
                mold_temperature_K=mold_C + 273.15,
                injection_velocity_mms=inj_v,
                injection_volume_flow_cm3s=inj_Q,
                compression_molding=icm,
                compression_factor=comp_factor,
                compression_stroke_mm=comp_stroke,
                compression_fraction=comp_frac,
                num_layers=num_layers,
                layer_distribution=layer_distribution,
                thermal_coupling=True,
                max_iterations=multilayer_max_iter,
                convergence_tol=multilayer_tol,
                solidification_temperature_fraction=solid_fraction,
                shear_heating_enabled=shear_heating_enabled,
            )
        else:
            solver = HeleShawSolver(
                geometry=geom,
                material=mat,
                melt_temperature_K=melt_C + 273.15,
                mold_temperature_K=mold_C + 273.15,
                injection_velocity_mms=inj_v,
                injection_volume_flow_cm3s=inj_Q,
                compression_molding=icm,
                compression_factor=comp_factor,
                compression_stroke_mm=comp_stroke,
                compression_fraction=comp_frac,
                skin_layer_enabled=skin_on,
                skin_growth_constant=c_skin,
                skin_max_iterations=skin_max_iter,
                skin_convergence_tol=skin_tol,
            )
        result = solver.solve(num_frames=num_frames)

        # 二相ショートショット。プレーンな HeleShawSolver 専用 —
        # 体積律速の短絡は凍結を含まないので、壁面冷却モデルとは組まない。
        two_phase_result = None
        two_phase_skip_reason: str | None = None
        if two_phase_on:
            if skin_on or multilayer_on:
                two_phase_skip_reason = "壁面冷却モデルが『なし』以外に設定されている（併用不可）"
                st.warning(
                    "二相ショートショット解析は壁面冷却モデル『なし』専用です。"
                    "今回はスキップしました。"
                )
            else:
                try:
                    two_phase_result = solve_two_phase_short_shot(solver, shot_volume_cm3)
                except ValueError as e:
                    # 例: 計量がゲート群の開ギャップ体積を下回る。メッセージは
                    # モデル側の固定文言 + 体積数値のみで、パス等の秘匿情報は
                    # 含まない。
                    two_phase_result = None
                    two_phase_skip_reason = str(e)
                    st.warning(f"二相ショートショット解析をスキップしました: {e}")

        # 入力の記録。metadata.json は解いた結果しか持たないので、これが無いと
        # ダウンロードした ZIP から設定を復元できない (画像から寸法を測って
        # 体積と tau_max で逆算する羽目になる)。
        run_settings = {
            "app_version": build_label(),
            "geometry": geom_settings,
            "material": material_key,
            "injection": {
                "melt_temperature_C": melt_C,
                "mold_temperature_C": mold_C,
                "injection_velocity_mms": inj_v,
                "injection_volume_flow_cm3s": inj_Q,
            },
            "wall_cooling": (
                {
                    "model": "skin",
                    "skin_growth_constant": c_skin,
                    "skin_max_iterations": skin_max_iter,
                    "skin_convergence_tol": skin_tol,
                }
                if skin_on
                else {
                    "model": "multilayer",
                    "num_layers": num_layers,
                    "layer_distribution": layer_distribution,
                    "max_iterations": multilayer_max_iter,
                    "convergence_tol": multilayer_tol,
                    "solidification_temperature_fraction": solid_fraction,
                    "shear_heating_enabled": shear_heating_enabled,
                }
                if multilayer_on
                else {"model": "none"}
            ),
            "compression_molding": (
                {
                    "enabled": True,
                    "mode": "stroke",
                    "stroke_mm": comp_stroke,
                    "fraction": comp_frac,
                }
                if icm
                else {"enabled": False}
            ),
            "two_phase_short_shot": (
                {"enabled": True, "shot_volume_cm3": shot_volume_cm3}
                if two_phase_result is not None
                else {"enabled": False}
            ),
            "output": {
                "num_frames": num_frames,
                "fill_cmap": fill_cmap,
                "isochrone_levels": iso_levels,
            },
        }

        # 重い PNG/GIF レンダリングはここで一回だけやる。後段の widget 操作で
        # rerun が走っても再生成しないよう、すべて session_state に置く。
        _tmp_dir = Path(tempfile.mkdtemp())
        _gif_path = render_fill_animation(
            result,
            _tmp_dir / "fill.gif",
            num_frames=num_frames,
            fps=8,
            cmap=fill_cmap,
            isochrone_levels=iso_levels,
        )
        # 各フレームの PNG 連番も書き出す。GIF と同じ ZIP に frames/ で同梱して、
        # ユーザーが GIF とフレーム画像を 1 ダウンロードで両取りできるようにする。
        _frame_paths = export_frames(
            result,
            _tmp_dir / "frames",
            num_frames=num_frames,
            cmap=fill_cmap,
            isochrone_levels=iso_levels,
        )
        # スクラバ用プレイヤーの HTML はここで一度だけ組む。フレーム PNG を
        # data URI で埋め込むので、後段の再生・シーク操作はサーバに戻らない。
        _player_html = build_fill_player_html(
            _frame_paths,
            fill_frame_times(result, num_frames),
            fill_frame_fractions(result, num_frames),
            fps=8,
        )
        _press_path = render_pressure_map(result, _tmp_dir / "pressure.png")
        _weld_path = render_weldlines(result, _tmp_dir / "weld.png")
        _skin_path: Path | None = None
        _core_path: Path | None = None
        _layer_T_grid_path: Path | None = None
        _layer_eta_grid_path: Path | None = None
        _layer_short_shot_path: Path | None = None
        if skin_on and result.skin_thickness_mm is not None:
            _skin_path = render_skin_layer_map(result, _tmp_dir / "skin.png")
            _core_path = render_core_layer_map(result, _tmp_dir / "core.png")
        if multilayer_on and getattr(result, "layer_temperature_K", None) is not None:
            _layer_T_grid_path = render_layer_grid(
                result, _tmp_dir / "layer_temperature_grid.png", field="temperature"
            )
            _layer_eta_grid_path = render_layer_grid(
                result, _tmp_dir / "layer_viscosity_grid.png", field="viscosity"
            )
            _layer_short_shot_path = render_short_shot_map(
                result, _tmp_dir / "multilayer_short_shot.png"
            )

        _two_phase_path: Path | None = None
        _two_phase_gif_path: Path | None = None
        if two_phase_result is not None:
            _two_phase_path = render_two_phase_map(
                two_phase_result, _tmp_dir / "two_phase_short_shot.png"
            )
            _two_phase_gif_path = render_two_phase_animation(
                two_phase_result,
                _tmp_dir / "two_phase.gif",
                num_frames=num_frames,
                fps=8,
            )

        _zip_buf_run = io.BytesIO()
        with zipfile.ZipFile(_zip_buf_run, "w", zipfile.ZIP_DEFLATED) as _zf_run:
            for _p in (
                _gif_path,
                _press_path,
                _weld_path,
                _skin_path,
                _core_path,
                _layer_T_grid_path,
                _layer_eta_grid_path,
                _layer_short_shot_path,
                _two_phase_path,
                _two_phase_gif_path,
            ):
                if _p is not None and _p.exists():
                    _zf_run.write(_p, _p.name)
            # 各フレーム PNG を frames/ 配下に同梱
            for _fp in _frame_paths:
                if _fp.exists():
                    _zf_run.write(_fp, f"frames/{_fp.name}")
            # ZIP を渡された相手が、Streamlit も追加ソフトも無しに
            # 画面と同じコマ送りを使えるようプレイヤーを単体 HTML で同梱する。
            # フレームは data URI で埋まっているのでオフラインで完結し、
            # HTML が受信側のフィルタで弾かれても frames/ の連番 PNG が残る。
            _zf_run.writestr(
                "player.html",
                wrap_standalone_html(
                    _player_html,
                    title="充填アニメーション",
                    note=build_label(),
                ),
            )
            _zf_run.writestr("settings.json", settings_json(run_settings))
            _zf_run.writestr(
                "metadata.json",
                json.dumps(result.metadata, indent=2, ensure_ascii=False, default=str),
            )
            if two_phase_result is not None:
                _zf_run.writestr(
                    "two_phase_metadata.json",
                    json.dumps(
                        two_phase_result.metadata, indent=2, ensure_ascii=False, default=str
                    ),
                )

        # 解析結果の一式を session_state に格納。次回 rerun（3D スライダー操作等）
        # でも下のブロックがこれを拾って表示する。
        st.session_state["mfs_result"] = result
        st.session_state["mfs_settings"] = run_settings
        st.session_state["mfs_geom"] = geom
        st.session_state["mfs_skin_on"] = skin_on
        st.session_state["mfs_multilayer_on"] = multilayer_on
        st.session_state["mfs_num_frames"] = num_frames
        st.session_state["mfs_tmp_dir"] = _tmp_dir
        st.session_state["mfs_gif_path"] = _gif_path
        st.session_state["mfs_player_html"] = _player_html
        st.session_state["mfs_player_height"] = fill_player_height_px(_frame_paths)
        st.session_state["mfs_press_path"] = _press_path
        st.session_state["mfs_weld_path"] = _weld_path
        st.session_state["mfs_skin_path"] = _skin_path
        st.session_state["mfs_core_path"] = _core_path
        st.session_state["mfs_layer_T_grid_path"] = _layer_T_grid_path
        st.session_state["mfs_layer_eta_grid_path"] = _layer_eta_grid_path
        st.session_state["mfs_layer_short_shot_path"] = _layer_short_shot_path
        st.session_state["mfs_two_phase_path"] = _two_phase_path
        st.session_state["mfs_two_phase_gif_path"] = _two_phase_gif_path
        st.session_state["mfs_two_phase_result"] = two_phase_result
        st.session_state["mfs_two_phase_skip"] = two_phase_skip_reason
        st.session_state["mfs_zip_bytes"] = _zip_buf_run.getvalue()

# 結果が session_state にある間は、do_run=False のときも（3D 倍率スライダー
# などのウィジェット操作で rerun が走った場合も）表示を維持する。
if "mfs_result" in st.session_state:
    result = st.session_state["mfs_result"]
    geom = st.session_state["mfs_geom"]
    skin_on = st.session_state["mfs_skin_on"]
    multilayer_on = st.session_state.get("mfs_multilayer_on", False)
    num_frames = st.session_state["mfs_num_frames"]
    gif_path = st.session_state["mfs_gif_path"]
    player_html = st.session_state.get("mfs_player_html")
    player_height = st.session_state.get("mfs_player_height")
    press_path = st.session_state["mfs_press_path"]
    weld_path = st.session_state["mfs_weld_path"]
    skin_path = st.session_state["mfs_skin_path"]
    core_path = st.session_state["mfs_core_path"]
    layer_T_grid_path = st.session_state.get("mfs_layer_T_grid_path")
    layer_eta_grid_path = st.session_state.get("mfs_layer_eta_grid_path")
    layer_short_shot_path = st.session_state.get("mfs_layer_short_shot_path")
    two_phase_path = st.session_state.get("mfs_two_phase_path")
    two_phase_gif_path = st.session_state.get("mfs_two_phase_gif_path")
    two_phase_result = st.session_state.get("mfs_two_phase_result")
    two_phase_skip = st.session_state.get("mfs_two_phase_skip")
    _zip_bytes = st.session_state["mfs_zip_bytes"]

    with col_right:
        st.subheader("結果")
        c1, c2, c3 = st.columns(3)
        c1.metric("総充填時間 T_fill", f"{result.total_fill_time_s:.3f} s")
        c2.metric("代表粘度 η_eff", f"{result.viscosity_Pa_s:.1f} Pa·s")
        c3.metric("キャビティ体積", f"{geom.volume_cm3():.2f} cm³")

        def _download(label: str, path: Path, mime: str, key: str) -> None:
            with open(path, "rb") as _f:
                st.download_button(
                    label,
                    data=_f.read(),
                    file_name=path.name,
                    mime=mime,
                    key=key,
                )

        st.markdown("**充填先端アニメーション**")
        if player_html:
            # 高さはフレーム PNG の実寸から導出する。プレイヤー側が画像を
            # ネイティブ幅で頭打ちにするので、列がどれだけ広くても操作列が
            # この高さから溢れない（scrolling=False で切れると操作不能になる）。
            components.html(player_html, height=player_height, scrolling=False)
        else:
            st.image(str(gif_path))
        st.download_button(
            "⬇ GIF + フレーム画像をダウンロード",
            data=_zip_bytes,
            file_name="mold_flow_results.zip",
            mime="application/zip",
            key="dl_zip_all",
            help=(
                "GIF（fill.gif）・各フレーム PNG（frames/frame_NNN.png）・"
                "各マップ PNG・metadata.json（解析結果）・settings.json（入力設定）を"
                "1つの ZIP にまとめてダウンロード"
            ),
        )

        with st.expander("圧力マップ"):
            st.image(str(press_path))
            st.caption("0=ゲート遠端、1=ゲート。実圧力スケールではなく相対分布。")
            _download("⬇ PNGをダウンロード", press_path, "image/png", "dl_press_png")

        with st.expander("等値線・ウェルドライン候補・エアトラップ"):
            st.image(str(weld_path))
            st.caption("赤=合流（ウェルド）候補、黄×=最終充填位置（エアトラップ候補）")
            _download("⬇ PNGをダウンロード", weld_path, "image/png", "dl_weld_png")

        if two_phase_skip is not None:
            # 実行時の警告は rerun で流れるので、結果ペイン側にも理由を残す
            st.info(f"二相ショートショット解析はスキップされました: {two_phase_skip}")

        if two_phase_path is not None and two_phase_result is not None:
            with st.expander("二相ショートショット（計量制限 + 圧縮前進）", expanded=True):
                md2 = two_phase_result.metadata
                st.image(str(two_phase_path))
                st.caption(
                    "青=射出相で充填（白線=射出等時線）、橙=圧縮相で前進、"
                    "灰=未充填。実機の計量値・ストロークをそのまま入れて"
                    "段階ショートショットの現物形状と比較する。"
                )
                tc1, tc2, tc3 = st.columns(3)
                tc1.metric("計量体積 V_shot", f"{md2['shot_volume_cm3']:.2f} cm³")
                tc2.metric(
                    "射出終了時 充填率",
                    f"{md2['injection_fill_fraction'] * 100:.1f} %",
                )
                tc3.metric(
                    "圧縮後 充填率",
                    f"{md2['final_fill_fraction'] * 100:.1f} %",
                )
                if md2["final_complete"]:
                    st.info("この計量では圧縮後に完全充填する（ショートショットにならない）。")
                if two_phase_gif_path is not None:
                    st.markdown("**二相アニメーション**")
                    st.image(str(two_phase_gif_path))
                    st.caption(
                        "射出相は実時間で進む。圧縮相は前進の順序のみ"
                        "（モデルは圧縮の時間スケールを持たない）。"
                    )
                _download(
                    "⬇ PNGをダウンロード",
                    two_phase_path,
                    "image/png",
                    "dl_two_phase_png",
                )
                if two_phase_gif_path is not None:
                    _download(
                        "⬇ GIFをダウンロード",
                        two_phase_gif_path,
                        "image/gif",
                        "dl_two_phase_gif",
                    )

        if skin_path is not None and core_path is not None:
            with st.expander("スキン層 / コア層 / ショートショット"):
                st.image(str(skin_path))
                st.caption("スキン層厚さ s(x,y) [mm]。流動が遅いほど・薄肉ほど s が大きい。")
                _download("⬇ スキン層 PNGをダウンロード", skin_path, "image/png", "dl_skin_png")
                st.image(str(core_path))
                st.caption(
                    "コア層 h_core = h - 2s。赤マーク = スキン同士が会合したショートショット候補。"
                )
                _download("⬇ コア層 PNGをダウンロード", core_path, "image/png", "dl_core_png")

        if multilayer_on and layer_T_grid_path is not None:
            with st.expander("層別プロファイル (Multi-layer N=...)"):
                md = result.metadata
                st.caption(
                    f"層数 N={md.get('num_layers')}, 分布={md.get('layer_distribution')}, "
                    f"反復={md.get('multilayer_iterations')}, "
                    f"収束={md.get('multilayer_converged')}, "
                    f"T_fill_inflation={md.get('T_fill_inflation', 1.0):.3f}, "
                    f"ショートショット率={md.get('short_shot_fraction', 0.0):.3f}"
                )
                _br_max = md.get("brinkman_number_max", 0.0)
                _br_mean = md.get("brinkman_number_mean", 0.0)
                _sh_max = md.get("shear_heating_max_K", 0.0)
                _sh_mean = md.get("shear_heating_mean_K", 0.0)
                _sh_enabled = md.get("shear_heating_enabled", False)
                # Brinkman number sanity-band: < 0.5 negligible, < 2 moderate, >= 2 strong.
                if _br_max < 0.5:
                    _br_emoji = "🟢"
                elif _br_max < 2.0:
                    _br_emoji = "🟡"
                else:
                    _br_emoji = "🔴"
                _badge = "✅ ON" if _sh_enabled else "OFF"
                st.caption(
                    f"剪断発熱 {_badge}: ΔT_max={_sh_max:.1f}K / mean={_sh_mean:.2f}K　"
                    f"{_br_emoji} Brinkman数 Br_max={_br_max:.2f} / mean={_br_mean:.3f}"
                    "（Br>1 で剪断発熱が支配的）"
                )
                st.markdown(
                    "**各層の温度マップ T_k(x,y)** — 壁層は T_mold へ、中央層は T_melt 寄り"
                )
                st.image(str(layer_T_grid_path))
                _download(
                    "⬇ 温度グリッド PNG",
                    layer_T_grid_path,
                    "image/png",
                    "dl_layer_T_png",
                )
                if layer_eta_grid_path is not None:
                    st.markdown(
                        "**各層の粘度マップ η_k(x,y)** — 対数スケール、壁層は剪断高でも低温で η 大"
                    )
                    st.image(str(layer_eta_grid_path))
                    _download(
                        "⬇ 粘度グリッド PNG",
                        layer_eta_grid_path,
                        "image/png",
                        "dl_layer_eta_png",
                    )
                if layer_short_shot_path is not None:
                    st.markdown(
                        "**ショートショット予測** — 中央層温度が T_solid を切ったセルを赤マーク"
                    )
                    st.image(str(layer_short_shot_path))
                    _download(
                        "⬇ ショートショット PNG",
                        layer_short_shot_path,
                        "image/png",
                        "dl_layer_short_png",
                    )

        with st.expander("3D表示（plotly）"):
            st.caption(
                "PL（パーティングライン）= Z=0 を底面とし、各セルを厚み h(x,y) 分だけ"
                "立ち上げたソリッド表示。x / y / z すべて同じ mm スケール（実物等倍）"
                "で描画。**天面と側壁の両方が物理量で着色**され、1つのカラーバーで"
                "読める（PLの薄グレー床は形状参照用）。ドラッグで回転、スクロール"
                "でズーム。物理は 2D Hele-Shaw のまま（表現上の3D化のみ）。"
            )
            t3d_h, t3d_fill, t3d_press = st.tabs(["厚み h(x,y)", "充填時間", "圧力"])
            with t3d_h:
                st.plotly_chart(
                    render_3d_thickness_map(result),
                    use_container_width=True,
                    config={"displaylogo": False},
                )
            with t3d_fill:
                st.plotly_chart(
                    render_3d_fill_time(result),
                    use_container_width=True,
                    config={"displaylogo": False},
                )
            with t3d_press:
                st.plotly_chart(
                    render_3d_pressure(result),
                    use_container_width=True,
                    config={"displaylogo": False},
                )

        with st.expander("この結果を出した設定"):
            st.caption(
                "ZIP の settings.json と同じ内容。metadata.json は解いた結果しか"
                "持たないので、設定を辿るならこちら。"
                "アップロードしたスペック JSON は名前と SHA-256 だけを記録する"
                "（ZIP は人に渡す前提なので、図面由来の寸法は載せない）。"
            )
            st.json(st.session_state.get("mfs_settings", {}))

        with st.expander("生データ"):
            st.json(result.metadata)
else:
    with col_right:
        st.info("左側でパラメータを設定し、「解析実行」を押してください。")
