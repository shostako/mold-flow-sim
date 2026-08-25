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
from collections.abc import Callable
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from core import (
    DirectGateConfig,
    GateProfileSpec,
    HeleShawSolver,
    MaterialDB,
    MultilayerHeleShawSolver,
    ProfilePlateConfig,
    build_direct_gate_geometry,
    build_fill_player_html,
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
from core.profile_gate import (
    EdgeChannelSpec,
    IslandSpec,
    LandSpec,
    MainRampSpec,
    RunnerSpec,
    SubGateSpec,
    SubIslandSpec,
    ValveSpec,
    WeldSpec,
    WellSpec,
)
from core.settings_record import config_settings, file_fingerprint, settings_json
from core.solver import WELD_MIN_ANGLE_DEG
from core.spec_source import (
    SPEC_LINK_NAME,
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
    export_two_phase_frames,
    render_layer_grid,
    render_short_shot_map,
    render_two_phase_animation,
    render_two_phase_map,
    two_phase_frame_labels,
)

APP_DIR = Path(__file__).parent

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
        "- 時計は**露光時計**: 壁は先端が通過した瞬間から老化し続け、求解にはセルの役務期間の時間平均スキン $\\tfrac{2}{3}s$ を当てる\n"
        "- スキンが出会う年齢 $t_c$ に役務が届いたセル＝**封止**（充填後に閉じた、赤マーク）。閉じた後に届くセルは**未充填**（充填時間なし）"
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
        "- **ショートショット予測**: スキン層モデルでは露光時計の封止（$t_{\\text{close}} = t_{\\text{arr}} + t_c$）で切られたセル、"
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


# ----------------------- Film gate 1–4: parametric gate-block inputs -----------------------
# All Film gates are the same family as the Profile gate JSON (land /
# main ramp / island (肉盗み) / well / outer wall), driven by sliders and
# assembled into a ``GateProfileSpec``:
#   Film gate 1 (扇状/肉盗み1)   symmetric, hamoko_gate_furiwake_20260703
#   Film gate 2 (扇状/肉盗み2)   symmetric, hamoko_gate_furiwake_weld_20260818
#                                (the 肉盗み has a flat dam: ``island.weld``)
#   Film gate 3 (片側/二倍流動長) one-sided (valve at the w=0 edge),
#                                hamoko_gate_2bai_20260703
#   Film gate 4 (振り分け/ミニ扇×2) symmetric, two mirrored mini fans
#                                (``sub_gates``) fed by a runner from the
#                                valve well; the steel between the fans is a
#                                full cut-out (hamoko_gate_furiwake_twin_mini_20260824)
#   Film gate 5 (振り分け/L字ランナー) same twin fans, but the runner runs
#                                sideways from the valve and enters each fan
#                                tip from below, perpendicular and centred
#                                (hamoko_gate_furiwake_twin_mini_L_20260824)
# The derived quantities (島 boundary t-endpoints, outer-wall start width,
# well floor) are tied to the major dimensions the way those drawings tie
# them.


@dataclasses.dataclass(frozen=True)
class _ProfileGateDefaults:
    gate_exit_width: float
    island_w_near: float  # 肉盗み境界の幅, at t = land length
    island_w_far: float  # 肉盗み境界の幅, at t = island end
    wall_t1: float
    wall_t2: float
    wall_w2: float
    valve_t: float | None  # None = centre of the well
    # The well's sloped wall angle is a cutter-geometry constant on each
    # drawing, not a design variable -- so it lives here, not on a slider.
    well_wall_angle_deg: float
    # Flat dam inside the 肉盗み: (start t, residual depth from PL). None = no dam.
    weld: tuple[float, float] | None = None


_FILM_GATE1_DEFAULTS = _ProfileGateDefaults(
    gate_exit_width=298.0,
    island_w_near=52.7,
    island_w_far=10.0,
    wall_t1=3.0,
    wall_t2=23.3,
    wall_w2=4.5,
    valve_t=None,
    well_wall_angle_deg=60.0,
)
# The drawing's 肉盗み boundary runs from (t=0, w=50.0) to (t=17, w=9.9); the
# sidebar pins the near end to t = land length (1.0), where that line reads
# 47.64 -- same line, expressed at the point the UI exposes.
_FILM_GATE2_DEFAULTS = _ProfileGateDefaults(
    gate_exit_width=298.0,
    island_w_near=47.64,
    island_w_far=9.9,
    wall_t1=5.0,
    wall_t2=23.28,
    wall_w2=4.48,
    valve_t=None,
    well_wall_angle_deg=71.6,
    weld=(7.0, 0.1),
)
_FILM_GATE3_DEFAULTS = _ProfileGateDefaults(
    gate_exit_width=299.0,
    island_w_near=95.3,
    island_w_far=20.0,
    wall_t1=3.0,
    wall_t2=23.6,
    wall_w2=4.45,
    valve_t=20.0,
    well_wall_angle_deg=60.0,
)


# Narrowest fan tip the Film gate 4 sliders offer. It also sets how close the
# tip axis may come to the valve axis or the exit edge: the room for the tip
# width is 2·min(axis, w_full − axis), and a slider whose min equals its max
# is a StreamlitAPIException, not a disabled control.
_MIN_TIP_WIDTH = 1.0


@dataclasses.dataclass(frozen=True)
class _TwinFanDefaults:
    """Film gate 4: two mirrored mini fans + runner, steel rhombus between."""

    gate_exit_width: float
    tip_t: float  # fan tip (where the runner enters)
    tip_axis_w: float  # half-width of the fan axis at the tip
    tip_width: float  # fan width at the tip (= runner width by default)
    island_end: float
    island_near: tuple[float, float]  # (inner, outer) half-widths at t = land length
    island_far: tuple[float, float]  # (inner, outer) half-widths at t = island end
    runner_depth: float
    well_wall_angle_deg: float
    # "straight": valve -> fan tip in one segment (Film gate 4).
    # "L": valve -> sideways at the valve's t -> up into the fan tip from
    # below (Film gate 5). The perpendicular, centred entry keeps the flow
    # path from the tip to both base corners of the fan equal, which is the
    # whole point of the variant: the straight runner grazes the fan's inner
    # wall and biases the fill toward the centre.
    runner_style: str = "straight"


_FILM_GATE4_DEFAULTS = _TwinFanDefaults(
    gate_exit_width=298.0,
    tip_t=14.0,
    tip_axis_w=74.5,
    tip_width=8.0,
    island_end=10.0,
    island_near=(48.2, 100.8),
    island_far=(69.5, 79.5),
    runner_depth=2.5,
    well_wall_angle_deg=60.0,
)

# Film gate 5 = the same design study with the runner L-shaped
# (hamoko_gate_furiwake_twin_mini_L_20260824): every dimension is identical,
# only the runner path differs.
_FILM_GATE5_DEFAULTS = dataclasses.replace(_FILM_GATE4_DEFAULTS, runner_style="L")


def _tagged_widgets(tag: str):
    """``st.slider`` / ``st.number_input`` with ``tag``-prefixed keys.

    Every widget gets a tag-prefixed key. Without one Streamlit identifies a
    widget by its label + parameters, so a Film gate 3 slider's value would
    survive a switch to Film gate 1 (same label, same bounds) and override
    that input's default (Codex P2).
    """

    def slider(label: str, *args, **kwargs):
        return st.slider(label, *args, key=f"{tag}_{label}", **kwargs)

    def number_input(label: str, *args, **kwargs):
        return st.number_input(label, *args, key=f"{tag}_{label}", **kwargs)

    return slider, number_input


def _plate_shape_inputs(tag: str, v: dict) -> None:
    """The 製品形状 expander shared by every Film gate (fills ``v`` in place)."""
    slider, _ = _tagged_widgets(tag)
    with st.expander("製品形状", expanded=False):
        v["plate_w"] = slider("製品幅 Wp [mm]", 40.0, 400.0, 300.0, step=5.0)
        v["plate_h"] = slider("製品高さ Hp [mm]", 30.0, 200.0, 50.0, step=5.0)
        two_layer = st.checkbox(
            "製品肉厚を2層化する（ゲート側／反ゲート側）",
            value=True,
            help="ONで段差位置を境にゲート側・反ゲート側を別肉厚に。OFFで均一肉厚。",
            key=f"{tag}_plate_2layer",
        )
        if two_layer:
            v["plate_split"] = slider(
                "段差位置（製品長辺から）[mm]",
                1.0,
                float(v["plate_h"]),
                min(20.0, float(v["plate_h"])),
                step=1.0,
                help="製品長辺からの距離。ここを境に肉厚が切り替わる。",
            )
            v["plate_lower_thk"] = slider("ゲート側肉厚 [mm]", 0.2, 2.0, 0.35, step=0.05)
            v["plate_upper_thk"] = slider("反ゲート側肉厚 [mm]", 0.2, 2.0, 0.50, step=0.05)
            v["plate_thk"] = float(v["plate_lower_thk"])
        else:
            v["plate_thk"] = slider("製品肉厚 [mm]", 0.2, 2.0, 0.4, step=0.1)
            v["plate_split"] = 0.0
            v["plate_lower_thk"] = v["plate_upper_thk"] = float(v["plate_thk"])


def _plate_from_inputs(v: dict) -> ProfilePlateConfig:
    return ProfilePlateConfig(
        plate_w_mm=v["plate_w"],
        plate_h_mm=v["plate_h"],
        plate_thk_mm=v["plate_thk"],
        plate_split_height_mm=v["plate_split"] if v["plate_split"] > 0 else 0.0,
        plate_lower_thk_mm=v["plate_lower_thk"] if v["plate_split"] > 0 else None,
        plate_upper_thk_mm=v["plate_upper_thk"] if v["plate_split"] > 0 else None,
    )


def _well_inputs(tag: str, v: dict, *, symmetric: bool, wall_angle_deg: float) -> None:
    """The 井戸 block shared by every Film gate (fills ``v`` in place).

    Sets ``v["well_on"]`` and, when on, ``well_t1 / well_t2 / well_half_w /
    well_depth / well_wall_angle``.
    """
    slider, _ = _tagged_widgets(tag)
    st.markdown("**井戸（バルブ周りの長穴ポケット）**")
    v["well_on"] = st.checkbox("井戸を有効化", value=True, key=f"{tag}_well_on")
    if not v["well_on"]:
        return
    v["well_t1"] = slider("井戸開始 t [mm]", 0.0, 60.0, 15.5, step=0.1)
    v["well_t2"] = slider(
        "井戸終端 t [mm] (> 開始)",
        min_value=float(v["well_t1"] + 0.5),
        max_value=80.0,
        value=float(max(27.5, v["well_t1"] + 0.5)),
        step=0.1,
    )
    # One-sided: the well is centred on the w=0 edge, so half of it
    # overhangs past the pocket into the plate margin. Cap the
    # half-width at the room available there or the builder rejects
    # the grid overhang (a ValueError the slider can prevent).
    hw_max = 20.0
    if not symmetric:
        edge_room = ProfilePlateConfig().pad_mm + (v["plate_w"] - v["gate_exit_width"]) / 2.0
        hw_max = max(0.5, min(20.0, math.floor(edge_room * 10.0) / 10.0))
    v["well_half_w"] = slider(
        "井戸半幅 [mm]" + ("" if symmetric else " (≤ 端の余白)"),
        0.5,
        float(hw_max),
        float(min(4.5, hw_max)),
        step=0.1,
    )
    # The sloped wall climbs from the rim, so the deepest point the
    # pocket can reach is half_width·tan(angle) at the centreline. A
    # deeper request is rejected by validate() and would otherwise be
    # recorded as asked and built shallower (Codex P1).
    tan_wall = math.tan(math.radians(wall_angle_deg))
    depth_max = float(min(15.0, v["well_half_w"] * tan_wall))
    depth_max = max(0.5, math.floor(depth_max * 10.0) / 10.0)
    v["well_depth"] = slider(
        f"井戸深さ [mm] (≤ 半幅·tan{wall_angle_deg:g}°)",
        0.5,
        depth_max,
        float(min(4.5, depth_max)),
        step=0.1,
        help=(f"壁角 {wall_angle_deg:g}° で到達できる最大深さ = 半幅 × tan({wall_angle_deg:g}°)。"),
    )
    v["well_wall_angle"] = float(wall_angle_deg)


def _well_from_inputs(v: dict) -> WellSpec | None:
    if not v["well_on"]:
        return None
    # Floor extent follows from the sloped wall: it eats depth/tan(angle)
    # of t at each end. When the pocket is too short for a flat floor the
    # floor range is simply not reported.
    eat = float(v["well_depth"]) / math.tan(math.radians(v["well_wall_angle"]))
    floor = (float(v["well_t1"]) + eat, float(v["well_t2"]) - eat)
    return WellSpec(
        shape="obround",
        t_range=(float(v["well_t1"]), float(v["well_t2"])),
        half_width=float(v["well_half_w"]),
        depth=float(v["well_depth"]),
        floor_t_range=floor if floor[1] > floor[0] + 1e-6 else None,
        wall_angle_deg=float(v["well_wall_angle"]),
    )


def _edge_channel_inputs(
    tag: str,
    v: dict,
    *,
    sided: bool,
    t_max_mm: float,
    t_default: tuple[float, float],
    ramp_cap: float,
) -> None:
    """The 縁部深彫り block shared by every Film gate (fills ``v`` in place).

    Sets ``v["ec_on"]`` and, when on, ``ec_width / ec_depth / ec_t_range``
    (+ ``ec_side`` when ``sided``). The t-range is a two-handle slider so
    its bounds can never collide (min 0 < max = the wall's t-extent); a
    range collapsed to zero width is rejected by ``validate()`` like any
    other inconsistency.
    """
    slider, _ = _tagged_widgets(tag)
    st.markdown("**縁部深彫り（エッジチャネル）**")
    v["ec_on"] = st.checkbox(
        "縁部深彫りを有効化",
        value=False,
        key=f"{tag}_ec_on",
        help=(
            "壁沿いに一定幅・一定深さの帯を彫り、縁を低抵抗の先行流路にする（S ∝ h³）。"
            "帯はポケット内側のセルを深くするだけで、外形（シルエット）は変えない。"
        ),
    )
    if not v["ec_on"]:
        return
    if sided:
        v["ec_side"] = st.radio(
            "対象の辺",
            ["外側", "内側", "両側"],
            horizontal=True,
            key=f"{tag}_ec_side",
            help="扇の外壁沿い／内壁沿い／その両方。",
        )
    # Same session-state trick as the runner width: the mesh slider is drawn
    # after this block, so read its current value and fall back to the
    # default on the first run. A band narrower than a cell can miss every
    # cell centre and the builder rejects it (zero-cell false green).
    dx_now = float(st.session_state.get(f"{tag}_メッシュ粗さ [mm/cell]", 1.0))
    w_min = max(0.5, math.ceil(dx_now * 2.0) / 2.0)
    v["ec_width"] = slider(
        "帯幅 [mm]（壁からの垂直距離）",
        float(w_min),
        20.0,
        float(min(max(3.0, w_min), 20.0)),
        step=0.5,
        help="下限はメッシュで解像できる幅（おおよそ1セル）。",
    )
    v["ec_depth"] = slider(
        "帯深さ [mm]（絶対深さ＝流路肉厚）",
        0.1,
        15.0,
        float(min(max(ramp_cap + 1.0, 0.1), 15.0)),
        step=0.1,
        help="帯の中は d = max(既存深さ, この値)。既存より浅い指定はその場所では効かない。",
    )
    lo, hi = t_default
    v["ec_t_range"] = slider(
        "範囲 t [mm]（壁に沿った区間）",
        0.0,
        float(t_max_mm),
        (float(max(0.0, min(lo, t_max_mm))), float(min(hi, t_max_mm))),
        step=0.1,
    )


def _edge_channels_from_inputs(v: dict, sides: tuple[str, ...]) -> tuple[EdgeChannelSpec, ...]:
    if not v.get("ec_on"):
        return ()
    lo, hi = v["ec_t_range"]
    return tuple(
        EdgeChannelSpec(
            width=float(v["ec_width"]),
            depth=float(v["ec_depth"]),
            t_range=(float(lo), float(hi)),
            side=side,
        )
        for side in sides
    )


_EC_SIDES = {"外側": ("outer",), "内側": ("inner",), "両側": ("outer", "inner")}


def _profile_gate_sidebar(tag: str, symmetric: bool, d: _ProfileGateDefaults) -> dict:
    """Draw the Film gate sidebar and return the raw slider values.

    ``tag`` prefixes widget keys so the Film gates never share state.
    Width-type values are half-widths from the valve axis when ``symmetric``,
    else widths from the valve-side (w=0) edge -- the labels say which.
    """
    w_word = "半幅" if symmetric else "幅"
    w_origin = "バルブ軸からの半幅" if symmetric else "バルブ側端（w=0）からの幅"
    v: dict = {"symmetric": symmetric}
    slider, number_input = _tagged_widgets(tag)

    _plate_shape_inputs(tag, v)

    with st.expander("ゲート形状", expanded=False):
        st.caption(
            f"t = ゲート出口（製品長辺）からの距離、w = {w_origin}。"
            + ("左右対称。" if symmetric else "片側のみ（バルブは端）。")
            + "深さ = 流路肉厚。"
        )
        gew = slider(
            "ゲート出口幅 [mm] (≤ 製品幅)",
            min_value=10.0,
            max_value=float(v["plate_w"]),
            value=float(min(d.gate_exit_width, v["plate_w"])),
            step=1.0,
        )
        v["gate_exit_width"] = gew
        # the pocket's full width coordinate: half of it when symmetric
        w_full = gew / 2.0 if symmetric else gew

        st.markdown("**ランド（出口）**")
        v["land_depth"] = slider("ランド深さ [mm]", 0.1, 2.0, 0.35, step=0.05)
        v["land_length"] = slider("ランド長さ [mm]", 0.5, 5.0, 1.0, step=0.1)

        st.markdown("**メインランプ**")
        v["ramp_angle"] = number_input(
            "ランプ角 [deg]",
            min_value=1.0,
            max_value=45.0,
            value=10.95,
            step=0.05,
            format="%.2f",
            help="ランド終端から深さが tan(角)·(t − ランド長) で増える。",
        )
        v["ramp_cap"] = slider(
            "ランプ上限深さ [mm] (≥ ランド深さ)",
            min_value=float(v["land_depth"]),
            max_value=10.0,
            value=float(max(2.5, v["land_depth"])),
            step=0.1,
        )

        st.markdown("**肉盗み（浅い帯＝振り分け）**")
        v["island_on"] = st.checkbox(
            "肉盗みを有効化",
            value=True,
            key=f"{tag}_island_on",
            help=(
                "バルブ側の帯だけランプ角を緩くして流路を絞り、樹脂を遠方へ振り分ける。"
                "境界線はランド終端 (t=ランド長) から肉盗み終端 (t=end) までの直線で、"
                f"{w_word}を出口側・終端側の2点で指定する。"
            ),
        )
        if v["island_on"]:
            v["island_angle"] = number_input(
                "肉盗み角 [deg] (≤ ランプ角)",
                min_value=0.0,
                max_value=float(v["ramp_angle"]),
                value=float(min(2.5, v["ramp_angle"])),
                step=0.05,
                format="%.2f",
            )
            v["island_end"] = slider(
                "肉盗み終端 t_end [mm] (> ランド長)",
                min_value=float(v["land_length"] + 0.5),
                max_value=60.0,
                value=float(max(17.0, v["land_length"] + 0.5)),
                step=0.1,
            )
            v["island_w_near"] = slider(
                f"境界{w_word}（出口側、t=ランド長）[mm]",
                min_value=1.0,
                max_value=float(w_full),
                value=float(min(d.island_w_near, w_full)),
                step=0.1,
            )
            v["island_w_far"] = slider(
                f"境界{w_word}（終端側、t=t_end）[mm]",
                min_value=0.5,
                max_value=float(w_full),
                value=float(min(d.island_w_far, w_full)),
                step=0.1,
            )
            v["weld_on"] = st.checkbox(
                "水平部（溶接ダム）を有効化",
                value=d.weld is not None,
                key=f"{tag}_weld_on",
                help=(
                    "肉盗みの下流側を溶接で肉盛りして天面を PL と平行にした区間。"
                    "終端は肉盗み終端と同じ。PL からの距離（残り流路厚）が 0 なら "
                    "鋼材が PL に接して樹脂が入らない＝完全な肉抜き空洞（穴）になる。"
                ),
            )
            if v["weld_on"]:
                weld_t0, weld_h = d.weld if d.weld is not None else (7.0, 0.1)
                v["weld_t1"] = slider(
                    "水平部開始 t [mm] (≥ ランド長、< 肉盗み終端)",
                    min_value=float(v["land_length"]),
                    max_value=float(v["island_end"] - 0.5),
                    value=float(min(max(weld_t0, v["land_length"]), v["island_end"] - 0.5)),
                    step=0.1,
                )
                v["weld_depth"] = slider(
                    "水平部の PL からの距離（残り流路厚）[mm] (≤ ランド深さ、0 = 空洞)",
                    min_value=0.0,
                    max_value=float(v["land_depth"]),
                    value=float(min(weld_h, v["land_depth"])),
                    step=0.05,
                )

        st.markdown("**外壁線（ポケット外形）**")
        st.caption("出口側は t=外壁開始 までゲート出口の全幅、そこから終端へ直線で狭まる。")
        v["wall_t1"] = slider("外壁開始 t [mm]", 0.0, 30.0, float(d.wall_t1), step=0.1)
        v["wall_t2"] = slider(
            "外壁終端 t [mm] (> 開始)",
            min_value=float(v["wall_t1"] + 0.5),
            max_value=80.0,
            value=float(max(d.wall_t2, v["wall_t1"] + 0.5)),
            step=0.1,
        )
        v["wall_w2"] = slider(
            f"外壁終端の{w_word} [mm]",
            min_value=0.5,
            max_value=float(w_full),
            value=float(min(d.wall_w2, w_full)),
            step=0.05,
        )
        v["wall_w1"] = float(w_full)

        _edge_channel_inputs(
            tag,
            v,
            sided=False,
            t_max_mm=float(v["wall_t2"]),
            t_default=(float(v["wall_t1"]), float(v["wall_t2"])),
            ramp_cap=float(v["ramp_cap"]),
        )

        _well_inputs(tag, v, symmetric=symmetric, wall_angle_deg=d.well_wall_angle_deg)
        if v["well_on"]:
            well_t_mid = 0.5 * (v["well_t1"] + v["well_t2"])
            pocket_t_end = max(float(v["wall_t2"]), float(v["well_t2"]))
        else:
            well_t_mid = 0.5 * (v["wall_t1"] + v["wall_t2"])
            pocket_t_end = float(v["wall_t2"])

        st.markdown("**バルブゲート**")
        v["valve_d"] = slider("バルブゲート径 [mm]", 1.0, 10.0, 3.0, step=0.5)
        # Keep the gate circle inside the pocket along t. Outside it the builder
        # snaps the gate to the nearest masked cell and the solver injects
        # somewhere other than the recorded position (Codex P1).
        t_min = float(v["valve_d"] / 2.0)
        t_max = float(max(t_min + 0.1, pocket_t_end - v["valve_d"] / 2.0))
        t_default = well_t_mid if d.valve_t is None else d.valve_t
        v["valve_t"] = slider(
            "バルブ位置 t [mm] (ポケット内)",
            t_min,
            t_max,
            float(min(max(round(t_default, 1), t_min), t_max)),
            step=0.1,
            help=(
                ("既定は井戸の中央（井戸 OFF なら外壁線の中点）。" if d.valve_t is None else "")
                + "上限はポケット終端（外壁終端／井戸終端の遠い方）− 半径。"
            ),
        )

        v["cell_size"] = slider("メッシュ粗さ [mm/cell]", 0.2, 3.0, 1.0, step=0.1)
        st.caption(
            "この形状の想定解像度は 1.0mm（ランド長 1mm が 1 セル）。"
            "細かいほど深さ場の再現精度が上がるが、解析が重くなる。"
        )
    return v


def _profile_gate_from_inputs(
    name: str, v: dict
) -> tuple[GateProfileSpec, ProfilePlateConfig, float]:
    """Assemble the spec + plate from ``_profile_gate_sidebar`` values."""
    t_land = float(v["land_length"])
    well = _well_from_inputs(v)
    island = None
    if v["island_on"]:
        weld = None
        if v.get("weld_on"):
            weld = WeldSpec(
                t_range=(float(v["weld_t1"]), float(v["island_end"])),
                depth=float(v["weld_depth"]),
            )
        island = IslandSpec(
            angle_deg=float(v["island_angle"]),
            boundary_line=(
                (t_land, float(v["island_w_near"])),
                (float(v["island_end"]), float(v["island_w_far"])),
            ),
            end_dist=float(v["island_end"]),
            weld=weld,
        )
    spec = GateProfileSpec(
        name=name,
        units="mm",
        symmetric=bool(v["symmetric"]),
        gate_exit_width=float(v["gate_exit_width"]),
        land=LandSpec(depth=float(v["land_depth"]), length=t_land),
        main_ramp=MainRampSpec(angle_deg=float(v["ramp_angle"]), cap_depth=float(v["ramp_cap"])),
        outer_wall_line=(
            (float(v["wall_t1"]), float(v["wall_w1"])),
            (float(v["wall_t2"]), float(v["wall_w2"])),
        ),
        valve=ValveSpec(t=float(v["valve_t"]), w=0.0, orifice_diameter=float(v["valve_d"])),
        island=island,
        well=well,
        edge_channels=_edge_channels_from_inputs(v, ("outer",)),
    )
    return spec, _plate_from_inputs(v), float(v["cell_size"])


def _twin_fan_sidebar(tag: str, d: _TwinFanDefaults) -> dict:
    """Film gate 4 / 5: two mirrored mini fans fed by a runner from the well.

    The centre of the block is steel at the PL (a full cut-out shaped like a
    deformed rhombus): each fan's inner wall starts on the land at w = 0 and
    runs out to the fan tip, the outer wall starts at the gate-exit edge and
    runs in to the tip. The runner leaves the valve and enters each fan at
    the tip's centre, so its path is derived from the valve position and the
    tip, not dimensioned separately. ``d.runner_style`` picks the route:
    "straight" (Film gate 4) or "L" (Film gate 5, sideways then up into the
    tip from below).
    """
    v: dict = {"symmetric": True, "runner_style": d.runner_style}
    slider, number_input = _tagged_widgets(tag)

    _plate_shape_inputs(tag, v)

    with st.expander("ゲート形状", expanded=False):
        st.caption(
            "t = ゲート出口（製品長辺）からの距離、w = バルブ軸からの半幅。左右対称。"
            "深さ = 流路肉厚。中央（扇の内壁より内側）は鋼材が PL に接した完全な肉盗み。"
        )
        gew = slider(
            "ゲート出口幅 [mm] (≤ 製品幅)",
            min_value=10.0,
            max_value=float(v["plate_w"]),
            value=float(min(d.gate_exit_width, v["plate_w"])),
            step=1.0,
        )
        v["gate_exit_width"] = gew
        w_full = gew / 2.0

        st.markdown("**ランド（出口）**")
        v["land_depth"] = slider("ランド深さ [mm]", 0.1, 2.0, 0.35, step=0.05)
        v["land_length"] = slider("ランド長さ [mm]", 0.5, 5.0, 1.0, step=0.1)
        t_land = float(v["land_length"])

        st.markdown("**メインランプ（扇の中）**")
        v["ramp_angle"] = number_input(
            "ランプ角 [deg]",
            min_value=1.0,
            max_value=45.0,
            value=10.95,
            step=0.05,
            format="%.2f",
            help="ランド終端から深さが tan(角)·(t − ランド長) で増える。",
        )
        v["ramp_cap"] = slider(
            "ランプ上限深さ [mm] (≥ ランド深さ)",
            min_value=float(v["land_depth"]),
            max_value=10.0,
            value=float(max(2.5, v["land_depth"])),
            step=0.1,
        )

        st.markdown("**ミニ扇（左右対称に2つ）**")
        st.caption(
            "内壁はランド終端の w=0 から扇先端へ、外壁はゲート出口端から扇先端へ走る直線。"
            "2つの扇の内壁に挟まれた菱形が完全肉盗み（0 mm）。"
        )
        v["tip_t"] = slider(
            "扇先端 t [mm] (> ランド長)",
            min_value=float(t_land + 1.0),
            max_value=40.0,
            value=float(max(d.tip_t, t_land + 1.0)),
            step=0.1,
            help="扇が終わりランナーが入る位置。",
        )
        # The tip width available is 2·min(axis, w_full − axis): the axis must
        # stay far enough from both the valve axis and the exit edge to leave
        # a fan wider than the width slider's own minimum. At axis = w_full
        # that room is zero, and a slider whose min equals its max raises a
        # StreamlitAPIException that kills the rest of the sidebar.
        # A full _MIN_TIP_WIDTH of margin, not half: at half the room comes to
        # exactly _MIN_TIP_WIDTH and min == max is the very case that raises.
        axis_margin = _MIN_TIP_WIDTH
        v["tip_axis_w"] = slider(
            "扇先端の中心半幅 [mm]",
            min_value=float(axis_margin),
            max_value=float(w_full - axis_margin),
            value=float(min(d.tip_axis_w, w_full - axis_margin)),
            step=0.5,
            help=(
                "扇先端（＝ランナー接続点）のバルブ軸からの半幅。"
                "上下限は扇先端幅を最小値ぶん残せる範囲。"
            ),
        )
        tip_w_max = 2.0 * float(min(v["tip_axis_w"], w_full - v["tip_axis_w"]))
        tip_w_max = max(_MIN_TIP_WIDTH, math.floor(tip_w_max * 10.0) / 10.0)
        v["tip_width"] = slider(
            "扇先端幅 [mm]",
            min_value=float(_MIN_TIP_WIDTH),
            max_value=float(tip_w_max),
            value=float(min(d.tip_width, tip_w_max)),
            step=0.5,
            help="扇先端での内壁〜外壁の幅。上限は中心半幅で決まる（内壁が w=0 を越えない）。",
        )

        v["island_on"] = st.checkbox(
            "肉盗み（扇の中の浅い帯）を有効化",
            value=True,
            key=f"{tag}_island_on",
            help=(
                "各扇の中央帯だけランプ角を緩くして流路を絞る。境界は内側線・外側線の2本で、"
                "出口側 (t=ランド長) と終端側 (t=t_end) の半幅で指定する。"
            ),
        )
        if v["island_on"]:
            v["island_angle"] = number_input(
                "肉盗み角 [deg] (≤ ランプ角)",
                min_value=0.0,
                max_value=float(v["ramp_angle"]),
                value=float(min(2.5, v["ramp_angle"])),
                step=0.05,
                format="%.2f",
            )
            v["island_end"] = slider(
                "肉盗み終端 t_end [mm] (> ランド長、≤ 扇先端)",
                min_value=float(t_land + 0.5),
                max_value=float(v["tip_t"]),
                value=float(min(max(d.island_end, t_land + 0.5), v["tip_t"])),
                step=0.1,
            )
            near_in, near_out = d.island_near
            far_in, far_out = d.island_far
            # The outer slider's min is inner + 0.5 and its max is w_full +
            # 0.5, so an inner at the full half-width collapses it to
            # min == max -- a StreamlitAPIException, not a clamp. Leave the
            # gap here rather than padding the outer bound, which would offer
            # island lines outside the fan.
            island_in_max = max(0.5, w_full - 0.5)
            v["island_near_in"] = slider(
                "内側線 半幅（出口側、t=ランド長）[mm]",
                0.0,
                float(island_in_max),
                float(min(near_in, island_in_max)),
                step=0.1,
            )
            v["island_near_out"] = slider(
                "外側線 半幅（出口側、t=ランド長）[mm] (> 内側)",
                min_value=float(v["island_near_in"] + 0.5),
                max_value=float(w_full + 0.5),
                value=float(min(max(near_out, v["island_near_in"] + 0.5), w_full + 0.5)),
                step=0.1,
            )
            v["island_far_in"] = slider(
                "内側線 半幅（終端側、t=t_end）[mm]",
                0.0,
                float(island_in_max),
                float(min(far_in, island_in_max)),
                step=0.1,
            )
            v["island_far_out"] = slider(
                "外側線 半幅（終端側、t=t_end）[mm] (> 内側)",
                min_value=float(v["island_far_in"] + 0.5),
                max_value=float(w_full + 0.5),
                value=float(min(max(far_out, v["island_far_in"] + 0.5), w_full + 0.5)),
                step=0.1,
            )

        _edge_channel_inputs(
            tag,
            v,
            sided=True,
            t_max_mm=float(v["tip_t"]),
            t_default=(t_land, float(v["tip_t"])),
            ramp_cap=float(v["ramp_cap"]),
        )

        st.markdown("**ランナー（井戸 → 扇先端）**")
        if d.runner_style == "L":
            st.caption(
                "経路は L字: バルブ (t_valve, 0) → 真横 (t_valve, 中心半幅) → "
                "垂直に扇先端 (t_tip, 中心半幅)。扇先端に下から中央接続する。"
            )
        else:
            st.caption(
                "経路はバルブ位置 (t_valve, 0) から扇先端の中心 (t_tip, 中心半幅) への直線。"
            )
        # The band reaches w = 中心半幅 + 幅/2, and the builder rejects a pocket
        # that overhangs the raster (x_valve + reach > pad + Wp/2). Every other
        # Film gate keeps its sliders inside what the builder accepts, so this
        # one must too -- otherwise the tip-axis slider at its far end turns a
        # legal-looking runner width into a パラメータ不整合 error.
        reach_room = (
            ProfilePlateConfig().pad_mm + float(v["plate_w"]) / 2.0 - float(v["tip_axis_w"])
        )
        # The band has to survive rasterisation: below roughly one cell
        # diagonal it passes between cell centres and breaks into islands,
        # which the builder rejects. The mesh slider is drawn after this one,
        # so read its current value out of session state (the same trick the
        # Profile gate uses for its uploader) and fall back to the default on
        # the first run.
        dx_now = float(st.session_state.get(f"{tag}_メッシュ粗さ [mm/cell]", 1.0))
        runner_w_min = max(1.0, math.ceil(dx_now * math.sqrt(2.0) * 2.0) / 2.0)
        runner_w_max = max(runner_w_min + 0.5, min(30.0, math.floor(2.0 * reach_room * 2.0) / 2.0))
        # Keep the label static: it is part of the widget key, so a label that
        # changes with the bound would re-key the slider and drop the value
        # the user set.
        v["runner_width"] = slider(
            "ランナー幅 [mm]",
            float(runner_w_min),
            float(runner_w_max),
            float(min(max(d.tip_width, runner_w_min), runner_w_max)),
            step=0.5,
            help=(
                "初期値は扇先端幅と同じ（先端でランナーが扇に接続する想定）。"
                "先端幅を変えても自動では追従しない。"
                "上限は帯の外側 (中心半幅 + 幅/2) がプレート外へ出ない範囲、"
                "下限はメッシュで解像できる幅（細いと帯が点線状に千切れる）。"
            ),
        )
        v["runner_depth"] = slider(
            "ランナー深さ [mm]",
            0.5,
            10.0,
            float(d.runner_depth),
            step=0.1,
            help="ランナー帯の中では深さ = max(扇の深さ, この値)。既定はランプ上限深さ。",
        )

        _well_inputs(tag, v, symmetric=True, wall_angle_deg=d.well_wall_angle_deg)
        well_t_mid = 0.5 * (v["well_t1"] + v["well_t2"]) if v["well_on"] else v["tip_t"] + 7.5

        st.markdown("**バルブゲート**")
        v["valve_d"] = slider("バルブゲート径 [mm]", 1.0, 10.0, 3.0, step=0.5)
        # The runner starts at the valve, so the orifice always sits in the
        # pocket; only the block extent bounds the position. For the L route
        # the valve must not sit closer to the product than the fan tip: below
        # it the sideways trunk would cut a deep stripe across the fan
        # interiors and feed them mid-fan -- a materially different experiment
        # from the advertised design (Codex P2). Equality stays allowed; the
        # L's corner then collapses onto its end.
        t_min = float(v["valve_d"] / 2.0)
        if d.runner_style == "L":
            t_min = max(t_min, float(v["tip_t"]))
        t_max = float(max(t_min + 0.1, 60.0 - v["valve_d"] / 2.0))
        v["valve_t"] = slider(
            "バルブ位置 t [mm]",
            t_min,
            t_max,
            float(min(max(round(well_t_mid, 1), t_min), t_max)),
            step=0.1,
            help=(
                "既定は井戸の中央。ランナーはここから各扇先端へ走る。"
                + ("下限は扇先端 t（トランクは扇より奥を通る）。" if d.runner_style == "L" else "")
            ),
        )

        v["cell_size"] = slider("メッシュ粗さ [mm/cell]", 0.2, 3.0, 1.0, step=0.1)
        st.caption(
            "この形状の想定解像度は 1.0mm（ランド長 1mm が 1 セル）。"
            "細かいほど深さ場の再現精度が上がるが、解析が重くなる。"
        )
    return v


def _twin_fan_from_inputs(name: str, v: dict) -> tuple[GateProfileSpec, ProfilePlateConfig, float]:
    """Assemble the spec + plate from ``_twin_fan_sidebar`` values."""
    t_land = float(v["land_length"])
    w_full = float(v["gate_exit_width"]) / 2.0
    tip_t = float(v["tip_t"])
    axis = float(v["tip_axis_w"])
    half_tip = float(v["tip_width"]) / 2.0
    island = None
    if v["island_on"]:
        island = SubIslandSpec(
            angle_deg=float(v["island_angle"]),
            inner_line=(
                (t_land, float(v["island_near_in"])),
                (float(v["island_end"]), float(v["island_far_in"])),
            ),
            outer_line=(
                (t_land, float(v["island_near_out"])),
                (float(v["island_end"]), float(v["island_far_out"])),
            ),
            end_dist=float(v["island_end"]),
        )
    fan = SubGateSpec(
        inner_wall_line=((t_land, 0.0), (tip_t, axis - half_tip)),
        outer_wall_line=((t_land, w_full), (tip_t, axis + half_tip)),
        tip_t=tip_t,
        island=island,
        edge_channels=_edge_channels_from_inputs(v, _EC_SIDES[v.get("ec_side", "外側")]),
    )
    spec = GateProfileSpec(
        name=name,
        units="mm",
        symmetric=True,
        gate_exit_width=float(v["gate_exit_width"]),
        land=LandSpec(depth=float(v["land_depth"]), length=t_land),
        main_ramp=MainRampSpec(angle_deg=float(v["ramp_angle"]), cap_depth=float(v["ramp_cap"])),
        outer_wall_line=None,
        valve=ValveSpec(t=float(v["valve_t"]), w=0.0, orifice_diameter=float(v["valve_d"])),
        well=_well_from_inputs(v),
        sub_gates=(fan,),
        runner=RunnerSpec(
            width=float(v["runner_width"]),
            depth=float(v["runner_depth"]),
            path=_twin_fan_runner_path(
                str(v.get("runner_style", "straight")), float(v["valve_t"]), tip_t, axis
            ),
        ),
    )
    return spec, _plate_from_inputs(v), float(v["cell_size"])


def _twin_fan_runner_path(
    style: str, valve_t: float, tip_t: float, axis: float
) -> tuple[tuple[float, float], ...]:
    """The runner route from the valve to the fan tip.

    "straight" is one segment. "L" goes sideways at the valve's t and then
    perpendicular into the tip from below -- and when the valve sits exactly
    on the tip line the middle corner coincides with the end, which would be
    a zero-length segment ``validate()`` rejects, so consecutive duplicate
    points are dropped (the collapsed 2-point path is the same route).
    """
    if style != "L":
        return ((valve_t, 0.0), (tip_t, axis))
    points = [(valve_t, 0.0), (valve_t, axis), (tip_t, axis)]
    path = [points[0]]
    for p in points[1:]:
        if p != path[-1]:
            path.append(p)
    return tuple(path)


@dataclasses.dataclass(frozen=True)
class _FilmGate:
    """One entry of the Film gate radio: how to draw its sidebar and build it."""

    tag: str  # widget key prefix
    record_name: str  # geometry name recorded in settings.json
    sidebar: Callable[[], dict]
    assemble: Callable[[str, dict], tuple[GateProfileSpec, ProfilePlateConfig, float]]


# radio label -> entry. Single source for the radio options, the sidebar
# branch and the geometry branch.
_FILM_GATES: dict[str, _FilmGate] = {
    "Film gate 1 (扇状/肉盗み1)": _FilmGate(
        "f1",
        "film_gate_1_parametric",
        lambda: _profile_gate_sidebar("f1", True, _FILM_GATE1_DEFAULTS),
        _profile_gate_from_inputs,
    ),
    "Film gate 2 (扇状/肉盗み2)": _FilmGate(
        "f2",
        "film_gate_2_parametric",
        lambda: _profile_gate_sidebar("f2", True, _FILM_GATE2_DEFAULTS),
        _profile_gate_from_inputs,
    ),
    "Film gate 3 (片側/二倍流動長)": _FilmGate(
        "f3",
        "film_gate_3_parametric",
        lambda: _profile_gate_sidebar("f3", False, _FILM_GATE3_DEFAULTS),
        _profile_gate_from_inputs,
    ),
    "Film gate 4 (振り分け/ミニ扇×2)": _FilmGate(
        "f4",
        "film_gate_4_parametric",
        lambda: _twin_fan_sidebar("f4", _FILM_GATE4_DEFAULTS),
        _twin_fan_from_inputs,
    ),
    "Film gate 5 (振り分け/L字ランナー)": _FilmGate(
        "f5",
        "film_gate_5_parametric",
        lambda: _twin_fan_sidebar("f5", _FILM_GATE5_DEFAULTS),
        _twin_fan_from_inputs,
    ),
}


def _valve_orifice_hits_pocket(
    geom: Geometry, spec: GateProfileSpec, plate: ProfilePlateConfig, dx: float
) -> bool:
    """Whether the valve orifice intersects the rasterised pocket.

    The same test the builder applies before falling back to "snap to the
    nearest masked cell"; mirroring it exactly means we reject precisely the
    cases where that snap would have moved the gate. A centre-cell test is
    too strict for the one-sided block, whose valve centre sits *on* the w=0
    boundary and can floor() into the cell just outside (Codex P2).
    """
    ny, nx = geom.mask.shape
    iy, ix = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    xx = (ix + 0.5) * dx
    yy = (iy + 0.5) * dx
    # Same x_valve as the builder: plate centre (symmetric) or the valve-side
    # edge cx - gew/2 (one-sided), plus the spec's w offset. The UI always
    # passes w = 0; the offset is kept here so the two stay in step if that
    # ever changes.
    cx = plate.pad_mm + plate.plate_w_mm / 2.0
    x_valve = (cx if spec.symmetric else cx - spec.gate_exit_width / 2.0) + spec.valve.w
    y_valve = plate.pad_mm + spec.t_max() - spec.valve.t
    r = spec.valve.orifice_diameter / 2.0
    in_valve = (xx - x_valve) ** 2 + (yy - y_valve) ** 2 <= r**2
    return bool(np.any(in_valve & geom.mask))


def _build_film_gate(entry: _FilmGate, v: dict, source: str) -> tuple[Geometry, dict]:
    spec, plate, dx = entry.assemble(entry.record_name, v)
    geom = build_profile_gate_geometry(spec, plate, cell_size_mm=dx)
    # The builder snaps a gate whose orifice misses the pocket to the nearest
    # masked cell. The slider bounds keep the orifice inside the pocket along
    # t; this catches the miss the bounds cannot express.
    if not _valve_orifice_hits_pocket(geom, spec, plate, dx):
        raise ValueError(
            f"バルブ位置 t={spec.valve.t:g} mm がポケットの外にある。"
            "外壁終端／井戸の範囲内に移動するか、外壁終端の幅を広げる。"
        )
    return geom, config_settings(
        source,
        plate,
        cell_size_mm=dx,
        # Slider values, not a drawing: record the assembled spec in full so
        # the ZIP reproduces the geometry.
        gate_profile=dataclasses.asdict(spec),
    )


# ----------------------- sidebar: inputs -----------------------
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
    if geom_source in _FILM_GATES:
        try:
            return _build_film_gate(_FILM_GATES[geom_source], pg_inputs, geom_source)
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
            else:
                st.warning("スペック JSON を一覧から選ぶか、アップロードしてください。")
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


with st.sidebar:
    _hdr_col, _run_col = st.columns([1.4, 1], vertical_alignment="bottom")
    with _hdr_col:
        st.header("成形品設計")
    with _run_col:
        do_run = st.button("解析実行", type="primary", use_container_width=True)
    geom_source = st.radio(
        "入力",
        [
            *_FILM_GATES,
            "Direct gate (parametric)",
            "Profile gate (JSONスペック)",
        ],
        index=0,
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
    elif geom_source in _FILM_GATES:
        pg_inputs = _FILM_GATES[geom_source].sidebar()
    elif geom_source.startswith("Profile gate"):
        with st.expander("ゲートプロファイル (JSON)", expanded=False):
            st.caption(
                "図面から抽出したゲートブロック深さ場の JSON スペックを読み込む。"
                "実図面由来のスペックはリポジトリに含めず、ここでローカル読込する運用。"
            )
            upload_pg = None
            local_spec_pg = None

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

            spec_origin_pg = choose_spec_origin(
                has_upload=upload_pg is not None,
                has_local=local_spec_pg is not None,
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
            index=0,
            key="wall_model",
            format_func=lambda m: {
                "none": "なし（等温・代表粘度のみ）",
                "skin": "スキン層 (1層 + Stefan/Neumann)",
                "multilayer": "層別 (N 層離散化 + Cross-WLF 結合)",
            }[m],
            help=(
                "なし: 既存 HeleShawSolver 相当、温度結合なし。\n"
                "スキン層: 壁面で固化するスキン層を s(t)=c_skin·√(αt) で取り込み、"
                "コア層 h_core=h-2s だけが流れる（露光時計、役務平均）。封止と未充填も検出。\n"
                "層別: 厚み方向を N 層に分割、Neumann 1D 温度プロファイルから "
                "層別粘度を Cross-WLF で評価。fixed-point で τ ↔ T_k ↔ η_k を結合。\n"
                "極薄プレート (t<0.5mm) では層別を推奨。"
            ),
        )

        # default container (so downstream `solver = HeleShawSolver(...)` /
        # `MultilayerHeleShawSolver(...)` always has the kwargs it expects).
        # 既定モードは『なし』(index=0) — 二相ショートショット（計量律速、凍結なし）
        # を既定 ON にしているため。層別を選んだときの既定値 (極薄 t0.35〜0.50 向け):
        #   層数 N: 7 (壁勾配が急なので N=5 から増量)
        #   反復上限: 12 (収束が遅くなりがちなので上限緩め)
        skin_on = wall_model == "skin"
        c_skin = 0.0
        skin_max_iter = 5
        skin_tol = 1e-3
        skin_clock_mode = "constant_pressure"
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
            skin_clock_mode = st.radio(
                "スキン層の時計",
                options=("constant_pressure", "constant_rate"),
                index=0,
                key="skin_clock",
                format_func=lambda m: {
                    "constant_pressure": "圧力一定（従来）: 抵抗増で流量が細り T_fill が伸びる",
                    "constant_rate": "速度制御: 射出時間 V/Q 固定、圧力が上がる",
                }[m],
                help=(
                    "スキンで流路が痩せたとき機械がどう応えるか。速度制御で射出する"
                    "実機（充填時間が設定どおりに出る）なら「速度制御」。従来の圧力一定は"
                    "T_fill を体積重み付き τ 比で膨らませる近似で、既存結果の再現用に残す。"
                    "二相ショートショットの射出相は計量 V/Q の定義上つねに速度制御。"
                ),
            )
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
        icm = st.checkbox("圧縮成形ON", value=True, key="icm_on")
        if icm:
            # ストローク (絶対加算) モードに統一。圧縮 mask 内の全セルに同じ絶対量を加算
            # するので段差プレートでも段差が保存される (金型シム量の物理に整合)。
            # 旧倍率モードは solver / CLI には後方互換で残しているが UI には出さない。
            comp_stroke = st.slider(
                "圧縮ストローク [mm]",
                0.0,
                2.0,
                0.50,
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

    # ショートショット欄はこの位置に出すが、中身は版表示の後で埋める:
    # build_geometry() は不整合で st.stop() するので、サイドバー内で先に呼ぶと
    # 「出力」欄と版表示がエラーのたびに消える（版表示をサイドバーに置いた理由
    # そのもの）。container は生成位置に描画されるので見た目の順序は変わらない。
    _short_shot_slot = st.container()

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
        # ウェルドは「2つの流れが出会う角度」で描く。45° 以上（商用 CAE の
        # 合流角 135° 境界）は濃い赤＝ウェルド、そこから下限までは薄い赤＝
        # メルド（ほぼ平行に合流する痕、強度欠陥より外観）。下限を下げるほど
        # 長く描かれ、数値ノイズも拾う。
        weld_min_angle = st.slider(
            "メルド表示の下限角 [deg]",
            0,
            40,
            int(WELD_MIN_ANGLE_DEG),
            step=5,
            key="weld_min_angle",
            help="2 つの流れが出会うときの開き角。45° 以上は濃い赤（ウェルド）、"
            "この角度から 45° までは薄い赤（メルド）、未満は描かない。"
            "穴の後ろに残る遅れ帯の痕を追いたいときは下げる。",
        )

    # Version / build label.
    # Rendered here (end of the sidebar) rather than at the end of the script
    # because the main flow has several ``st.stop()`` calls for parameter
    # validation — a footer placed after them would vanish exactly when a user
    # screenshots an error and asks which build produced it.
    st.divider()
    st.caption(build_label())

    with _short_shot_slot:
        geom, geom_settings = build_geometry()

        with st.expander("ショートショット（計量制限）", expanded=False):
            # 二相モデル: (1) 射出相 = 型開きギャップで計量体積ぶん充填、
            # (2) 圧縮相 = 型閉じで溶融プールを等圧ソースとして前進（体積保存）。
            # 線形求解2回・時間積分なし。実機の計量値をそのまま入れて
            # 段階ショートショットの現物形状と直接比較する用途。
            two_phase_on = st.checkbox(
                "二相ショートショット解析ON",
                value=True,
                key="two_phase_on",
                help=(
                    "計量を意図的に絞ったショートショットの最終形状を予測する。"
                    "射出相（型開きギャップで計量体積まで充填）→ 圧縮相（型閉じで"
                    "溶融プールを前進、体積保存）の二相。壁面冷却モデルは『なし』か"
                    "『スキン層』で実行（スキン層は射出相に乗る: 開いた薄板が射出中に"
                    "痩せてゲート部が先に埋まる順番を出す）。『層別』とは併用不可。"
                ),
            )
            if two_phase_on and wall_model == "multilayer":
                # 実行時の一過性警告だけだと rerun で消えて「ON にしたのに何も
                # 出ない」に見える。設定と同じ場所に常時出す。
                st.warning(
                    "壁面冷却モデルが『なし』または『スキン層』のときだけ実行される。"
                    "現在の設定（層別）では二相解析はスキップされる。"
                )
            if two_phase_on:
                # 既定値は現在の形状の最終キャビティ体積。形状を変えると追従するが、
                # ユーザーが値を触った後は（前回の自動値から動いているので）触らない。
                # 丸めない: solver は素の体積と比較するので、下に丸めた既定は
                # 「完全充填ちょうど」でなく極小のショートショットになる（Codex P2）。
                _v_cav = float(geom.volume_cm3())
                _prev_auto = st.session_state.get("mfs_shot_volume_auto")
                _current = st.session_state.get("two_phase_shot_volume")
                if _prev_auto is None or _current is None or _current == _prev_auto:
                    st.session_state["two_phase_shot_volume"] = _v_cav
                st.session_state["mfs_shot_volume_auto"] = _v_cav
                shot_volume_cm3 = st.number_input(
                    "計量体積 V_shot [cm³]",
                    min_value=0.01,
                    step=0.1,
                    key="two_phase_shot_volume",
                    help=(
                        "実機の計量値（ショット体積）。既定は現在の形状の最終キャビティ"
                        "体積（完全充填ちょうど）。減らすとショートショットになる。"
                    ),
                )
                _hint = f"最終キャビティ体積 {_v_cav:.2f} cm³"
                if icm and comp_stroke is not None:
                    _v_open = _v_cav + comp_stroke * geom.compression_area_mm2() / 1000.0
                    _hint += f" / 開きギャップ体積 ≈ {_v_open:.2f} cm³"
                st.caption(_hint + "。計量が最終キャビティ体積以上だと完全充填になる。")
            else:
                shot_volume_cm3 = None


# ----------------------- main panel -----------------------


col_left, col_right = st.columns([1, 1.3])

with col_left:
    st.subheader("成形品設計図")
    fig_data = np.where(geom.mask, geom.thickness_mm, np.nan)
    st.write(
        f"格子: {geom.nx} × {geom.ny}, セル {geom.cell_size_mm} mm, 体積 {geom.volume_cm3():.2f} cm³"
    )
    fig_buf = io.BytesIO()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4), dpi=110)
    # Product-referenced frame: x = 0 on the valve axis (≒ product center),
    # y = 0 on the product's gate-side bottom edge. A film gate's gate
    # block / runner reads y < 0, a direct gate's gate lands at y > 0
    # inside the product. Same convention is shared by every result-time
    # map in core/visualizer.py and the 3D views.
    x0_mm, y0_mm = geom.display_origin_mm()
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
        "y = 0 は製品下端ライン、x = 0 はバルブゲート軸（赤丸＝ゲート）。"
        "フィルムゲートのゲートブロック／ランナーは y < 0 側、"
        "ダイレクトゲートのゲートは製品内（y > 0）。"
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
                skin_clock_mode=skin_clock_mode,
            )
        try:
            result = solver.solve(num_frames=num_frames)
        except ValueError as exc:
            # The solver rejects a cavity whose cells cannot all be reached
            # from a gate (Issue #58). The builders catch the shapes they can
            # name, but a combination they do not model still has to reach the
            # user as a message -- an uncaught exception here renders as a raw
            # traceback in the app.
            st.error(f"解析できない形状: {exc}")
            st.stop()

        # 二相ショートショット。HeleShawSolver 専用（等温、またはスキン層を
        # 射出相に乗せる）— 層別ソルバーには射出相の時計が無い。
        two_phase_result = None
        two_phase_skip_reason: str | None = None
        if two_phase_on:
            if multilayer_on:
                two_phase_skip_reason = "壁面冷却モデルが『層別』に設定されている（併用不可）"
                st.warning(
                    "二相ショートショット解析は壁面冷却モデル『なし』または『スキン層』専用です。"
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
                    "skin_clock_mode": skin_clock_mode,
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
                {
                    "enabled": True,
                    "shot_volume_cm3": shot_volume_cm3,
                    "skin_layer": bool(skin_on),
                }
                if two_phase_result is not None
                else {"enabled": False}
            ),
            "output": {
                "num_frames": num_frames,
                "fill_cmap": fill_cmap,
                "isochrone_levels": iso_levels,
                "weld_min_angle_deg": float(weld_min_angle),
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
        _weld_path = render_weldlines(
            result, _tmp_dir / "weld.png", weld_min_angle_deg=float(weld_min_angle)
        )
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
        _two_phase_player_html: str | None = None
        _two_phase_player_height: int | None = None
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
            # 充填先端と同じスクラバ。フレーム系列は frame_states が単一ソース
            # なので GIF のコマ k とプレイヤーのコマ k は同じ状態を指す。
            _two_phase_frame_paths = export_two_phase_frames(
                two_phase_result, _tmp_dir / "two_phase_frames", num_frames=num_frames
            )
            _n_tp = len(_two_phase_frame_paths)
            _two_phase_player_html = build_fill_player_html(
                _two_phase_frame_paths,
                [0.0] * _n_tp,
                [0.0] * _n_tp,
                fps=8,
                labels=two_phase_frame_labels(two_phase_result, num_frames),
            )
            _two_phase_player_height = fill_player_height_px(_two_phase_frame_paths)

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
                if _two_phase_player_html is not None:
                    _zf_run.writestr(
                        "two_phase_player.html",
                        wrap_standalone_html(
                            _two_phase_player_html,
                            title="二相ショートショット アニメーション",
                            note=build_label(),
                        ),
                    )
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
        st.session_state["mfs_weld_min_angle"] = float(weld_min_angle)
        st.session_state["mfs_skin_path"] = _skin_path
        st.session_state["mfs_core_path"] = _core_path
        st.session_state["mfs_layer_T_grid_path"] = _layer_T_grid_path
        st.session_state["mfs_layer_eta_grid_path"] = _layer_eta_grid_path
        st.session_state["mfs_layer_short_shot_path"] = _layer_short_shot_path
        st.session_state["mfs_two_phase_path"] = _two_phase_path
        st.session_state["mfs_two_phase_gif_path"] = _two_phase_gif_path
        st.session_state["mfs_two_phase_player_html"] = _two_phase_player_html
        st.session_state["mfs_two_phase_player_height"] = _two_phase_player_height
        st.session_state["mfs_two_phase_result"] = two_phase_result
        st.session_state["mfs_two_phase_skip"] = two_phase_skip_reason
        st.session_state["mfs_zip_bytes"] = _zip_buf_run.getvalue()


# 結果が session_state にある間は、do_run=False のときも（3D 倍率スライダー
# などのウィジェット操作で rerun が走った場合も）表示を維持する。
def _refresh_weld_assets(min_angle: float) -> None:
    """Re-threshold the cached weld map when the slider moves after a run.

    The solver keeps ``weld_angle_deg`` precisely so this does not need a
    re-solve: redraw weld.png from the cached result, update the recorded
    setting, and swap both entries inside the cached ZIP so a download taken
    after moving the slider matches what the screen shows (Codex P2).
    """
    cached = st.session_state["mfs_result"]
    tmp_dir = st.session_state["mfs_tmp_dir"]
    new_path = render_weldlines(cached, tmp_dir / "weld.png", weld_min_angle_deg=min_angle)
    st.session_state["mfs_weld_path"] = new_path
    st.session_state["mfs_weld_min_angle"] = min_angle
    settings = st.session_state["mfs_settings"]
    settings["output"]["weld_min_angle_deg"] = min_angle
    old_zip = zipfile.ZipFile(io.BytesIO(st.session_state["mfs_zip_bytes"]))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for info in old_zip.infolist():
            if info.filename in (new_path.name, "settings.json"):
                continue
            zf.writestr(info, old_zip.read(info.filename))
        zf.write(new_path, new_path.name)
        zf.writestr("settings.json", settings_json(settings))
    st.session_state["mfs_zip_bytes"] = buf.getvalue()


if "mfs_result" in st.session_state:
    if st.session_state.get("mfs_weld_min_angle") != float(weld_min_angle):
        _refresh_weld_assets(float(weld_min_angle))
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
    two_phase_player_html = st.session_state.get("mfs_two_phase_player_html")
    two_phase_player_height = st.session_state.get("mfs_two_phase_player_height")
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
            st.caption(
                "濃い赤=ウェルド（開き角 45° 以上）、薄い赤=メルド（下限角〜45°）、"
                "黄×=最終充填位置（エアトラップ候補）。下限角はサイドバー「出力」で変えられる"
            )
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
                if md2.get("skin_layer_enabled"):
                    st.caption(
                        "スキン層を射出相に乗せた結果（時計は計量 V/Q 固定）: "
                        f"射出終了時のスキン最大 {md2.get('injection_skin_max_mm', 0.0):.3f} mm、"
                        f"封止 {md2.get('injection_sealed_cells', 0)} セル、"
                        f"封止で届かず {md2.get('injection_unfillable_cells', 0)} セル。"
                        "圧縮相は等温（プールは等圧ソースなので内部のスキンは前進に効かない）。"
                    )
                    if md2.get("injection_sealed_cells", 0) > 0:
                        _short = md2["shot_volume_cm3"] - md2["achieved_volume_final_cm3"]
                        st.warning(
                            "射出中に封止したセルがある（濃赤）。射出時間が長すぎるか、"
                            "モデルがゲート部の剪断発熱を持っていないためランドが早く閉じている。"
                            "実機のランドが開いたままなら、この封止は模型の限界と読む。"
                            "封止は圧縮相でも閉じたまま（封止の奥へは前進しない、"
                            f"届かないセル {md2.get('compression_unreachable_cells', 0)}）。"
                            + (
                                f" 計量のうち {_short:.2f} cm³ はキャビティに入らない。"
                                if _short > 1e-9
                                else ""
                            )
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
                    if two_phase_player_html:
                        components.html(
                            two_phase_player_html,
                            height=two_phase_player_height,
                            scrolling=False,
                        )
                    else:
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
