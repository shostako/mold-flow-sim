"""Streamlit UI for the simplified mold flow simulator.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import io
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st

from core import (
    DirectGateConfig,
    FilmGateConfig,
    HeleShawSolver,
    MaterialDB,
    MultilayerHeleShawSolver,
    build_direct_gate_geometry,
    build_film_gate_geometry,
    geometry_from_image,
    render_3d_fill_time,
    render_3d_pressure,
    render_3d_thickness_map,
    render_core_layer_map,
    render_fill_animation,
    render_pressure_map,
    render_skin_layer_map,
    render_weldlines,
)
from core.geometry import Geometry
from core.visualizer import render_layer_grid, render_short_shot_map

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
        "金型キャビティが**薄板（厚み h ≪ 平面サイズ）**であることを前提に、"
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
        "- $\\eta_{\\text{eff}}$: 代表剪断速度・代表温度で評価した粘度 [Pa·s]\n"
        "- $S$: コンダクタンス（流れやすさ）。$h^3$ で効くので**厚み変化が支配的**\n"
        "- 圧力 $p$ はゲートで一定、流動先端で 0 を境界条件にして解く"
    )

    st.markdown("### 2. 粘度モデル：Cross-WLF")
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
        "- 代表剪断速度は Newtonian plate 近似 "
        r"$\dot\gamma = 6V/h$ で 1 回だけ評価（ローカル反復なし）"
        "\n- バルク温度は射出温度と金型温度の重み付き平均 "
        r"$T_{\text{bulk}} = 0.7\,T_{\text{melt}} + 0.3\,T_{\text{mold}}$"
    )

    st.markdown("### 3. 充填時間場：Pseudo-Conduction 法")
    st.markdown(
        "Hele-Shaw の圧力場を時間ステップで進めず、**楕円型（熱伝導型）方程式に置き換えて 1 発で解く**"
        "高速化テクニック。$\\tau$ は擬似的な「ゲートからの到達時間場」。"
    )
    st.latex(
        r"-\,\nabla \cdot \left( S\, \nabla \tau \right) = 1 "
        r"\quad \text{in cavity}"
    )
    st.markdown(
        "- ゲートで $\\tau = 0$（ディリクレ境界）、キャビティ壁で no-flux（ノイマン境界）\n"
        "- 解いた $\\tau$ を最大値で正規化し、絶対時間に換算: "
        r"$t_{\text{fill}}(x,y) = \dfrac{\tau(x,y)}{\tau_{\max}} \cdot T_{\text{fill}}$"
        "\n- $T_{\\text{fill}} = V_{\\text{cavity}} / Q$（射出率一定）\n"
        "- 流動先端の進行は $\\tau$ の等値線として可視化"
    )

    st.markdown("### 4. スキン層（オプション）：Stefan / Neumann 近似")
    st.markdown(
        "金型壁で樹脂が固化して育つ「スキン層」を熱拡散の Stefan 解で表現し、"
        "流動はコア層 $h_{\\text{core}} = h - 2s$ のみを通る。"
    )
    st.latex(r"s(t) = c_{\text{skin}} \sqrt{\alpha\,t}")
    st.markdown(
        "- $\\alpha$: 樹脂の熱拡散率（材料 DB から取得）\n"
        "- $c_{\\text{skin}}$: 成長定数（$\\sim 1.0$ が物理的代表値、UI で調整）\n"
        "- $\\tau$ と $s$ が相互依存するので fixed-point 反復で釣り合わせる\n"
        "- $h_{\\text{core}}$ が下限を切ったセル＝**ショートショット候補**として赤マーク"
    )

    st.markdown("### 5. 射出圧縮成形（ICM、オプション）：等価厚み膨張モデル")
    st.markdown(
        "圧縮位相を時間ステッピングで解かず、**製品本体の厚みを膨張**させた等価モデルとして扱う。"
        "流路抵抗 $S \\propto h^3$ が一気に下がる効果を擬似再現。"
        "膨張対象は「製品本体セルだけ」、ランナー・スプルー・ゲートは射出時の肉厚のまま不変。"
        "UI は **stroke モード**（金型シム量の物理に整合）で統一。"
        "倍率指定の **factor モード**は CLI / solver 引数で後方互換のためだけに残す。"
    )

    st.markdown("#### 5-1. stroke モード（絶対加算、段差保存、UI 既定）")
    st.markdown(
        "全 target セルに同じ絶対量（ストローク $s$）を**加算**する。"
        "**金型シム量**が設計指標のとき（＝実機の射出圧縮成形そのもの）に使う。"
        "段差プレート（例: 薄肉部 $t_0=0.35$ mm ／ 厚肉部 $t_0=0.50$ mm）に "
        "$s=0.70$ mm を加算すると薄肉部 $1.05$ mm ／ 厚肉部 $1.20$ mm となり、"
        "**段差 $0.15$ mm が圧縮位相中も保存される**（factor モードだと "
        "$0.35 \\to 1.05$ / $0.50 \\to 1.50$ で段差が $0.45$ mm に膨らんで非物理）。"
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
        "- uniform プレートで $CF = (h + s)/h$ に揃えると factor モードと厳密に等価"
    )

    st.markdown("---")
    st.markdown("### ✅ モデル化している現象")
    st.markdown(
        "- 薄板キャビティ内の 2D 流動（局所剪断と局所抵抗の効果）\n"
        "- 樹脂物性（密度・熱拡散率・Cross-WLF 粘度パラメータ）\n"
        "- ゲート位置・ゲート径・ランナー形状・プレート分割（2 層肉厚）\n"
        "- 流動先端の到達順、ウェルドライン、エアトラップ\n"
        "- 圧力分布の相対値（ゲート＝1、最終充填点＝0 の正規化）\n"
        "- 充填時間（射出率 $Q$ から逆算した絶対時間）\n"
        "- スキン層形成によるコア閉塞・ショートショット予測（オプション）\n"
        "- 射出圧縮成形による等価流路拡大（stroke モード、CLI に factor 後方互換あり、オプション）"
    )

    st.markdown("### ❌ モデル化していない現象（重要）")
    st.markdown(
        "- **コアバルク温度の動的低下と粘度の局所更新**：粘度は単一代表値で固定。"
        "実際は流動中に温度が落ちて粘度が上がるが、本ツールは捉えない\n"
        "- **真の 3D 流れ・ジェッティング・噴流・コーナー効果**：あくまで 2D Hele-Shaw\n"
        "- **保圧（パッキング段階）**：充填までしかモデル化しない\n"
        "- **収縮・反り・残留応力**：熱固化収縮も結晶化も入っていない\n"
        "- **局所剪断速度の反復**：粘度評価は代表点 1 回のみ\n"
        "- **ベント・脱気挙動**：エアトラップ位置は予測するが圧抜けは考慮しない\n"
        "- **STL/STEP 直接読み込み**：パラメトリック形状（Film gate / Direct gate）"
        "または PNG/JPG 二値画像のみ\n"
        "- **非構造格子・中立面メッシュ**：構造格子（正方形セル）固定"
    )

    st.markdown("### 用途と適用範囲")
    st.markdown(
        "本ツールは**初期スクリーニング・概念検証**用途。商用 CAE "
        "（Moldflow / Moldex3D 等）の置き換えではない。\n\n"
        "- ◯ 向く：ゲート位置候補の比較、ランナー形状の方向性決定、"
        "プレート薄肉化時のショートショット予兆、フローバランサー（▽肉盗み）の効きの感覚的把握\n"
        "- × 向かない：寸法精度予測、保圧設計、収縮反り、最終肉厚分布の精密計算"
    )


@st.cache_resource
def _load_db() -> MaterialDB:
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
            "Film gate (parametric)",
            "Direct gate (parametric)",
            "画像から生成 (PNG/JPG)",
        ],
        index=0,
    )

    if geom_source.startswith("Direct gate"):
        # Match Film gate defaults: plate 300×50, with the optional 2-zone
        # split (gate-side 0.35 / far-side 0.50, switching at 20 mm from
        # the gate-side edge).
        plate_w = st.slider("製品幅 Wp [mm]", 40.0, 300.0, 300.0, step=5.0)
        plate_h = st.slider("製品高 Hp [mm]", 30.0, 200.0, 50.0, step=5.0)

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
        cell_size = st.slider("メッシュ粗さ [mm/cell]", 0.5, 3.0, 0.5, step=0.1)
        upload = None
    elif geom_source.startswith("Film gate"):
        plate_w = st.slider("製品幅 Wp [mm]", 40.0, 300.0, 300.0, step=5.0)
        plate_h = st.slider("製品高 Hp [mm]", 30.0, 160.0, 50.0, step=5.0)

        st.markdown("**製品肉厚（ゲート側／反ゲート側で2層化可）**")
        plate_split = st.slider(
            "段差位置 [mm]",
            0.0,
            float(plate_h),
            min(20.0, float(plate_h)),
            step=1.0,
            help="ゲート側長辺からの距離。0 で均一肉厚（旧挙動）",
        )
        if plate_split > 0:
            plate_lower_thk = st.slider(
                "ゲート側肉厚 [mm]",
                0.2,
                2.0,
                0.35,
                step=0.05,
            )
            plate_upper_thk = st.slider(
                "反ゲート側肉厚 [mm]",
                0.2,
                2.0,
                0.50,
                step=0.05,
            )
            plate_thk = float(plate_lower_thk)
        else:
            plate_thk = st.slider("製品肉厚 [mm]", 0.2, 2.0, 0.4, step=0.1)
            plate_lower_thk = float(plate_thk)
            plate_upper_thk = float(plate_thk)

        st.markdown("**ランナー上面投影**")
        runner_long = st.slider(
            "長辺 L_long [mm] (≤ 製品幅)",
            min_value=10.0,
            max_value=float(plate_w),
            value=float(min(250.0, plate_w)),
            step=1.0,
        )
        runner_short_d = st.slider(
            "短辺直径 d [mm] (≤ 長辺)",
            min_value=4.0,
            max_value=float(runner_long),
            value=float(min(10.0, runner_long)),
            step=0.5,
        )
        runner_depth = st.slider(
            "ランナー高さ D [mm] (長辺〜短辺直径線距離)",
            5.0,
            60.0,
            20.0,
            step=1.0,
        )

        st.markdown("**ランナー肉厚**")
        runner_thk_film = st.slider("厚肉部 h_runner [mm]", 1.0, 10.0, 2.5, step=0.1)
        flat_ratio = st.slider(
            "厚肉部の比率 D_flat / D",
            0.0,
            1.0,
            0.35,
            step=0.05,
            help="0で全スロープ（製品まで連続変化）、1で全フラット（製品との段差大）",
        )

        st.markdown("**バルブゲート**")
        valve_d = st.slider(
            "バルブゲート径 [mm] (≤ d)",
            min_value=1.0,
            max_value=float(runner_short_d),
            value=float(min(3.0, runner_short_d)),
            step=0.5,
        )

        # 製品の長辺とランナー長辺は直接接続（くびれ＝ゲート土手なし）
        gate_w = runner_long

        st.markdown("**フローバランサー（中央肉盗み）**")
        balancer_on = st.checkbox(
            "肉盗み（▽）を有効化",
            value=True,
            help=(
                "ランナー中央軸に逆三角形の薄領域を作り、中央への流れを"
                "意図的に阻害して長辺全体から均一に充填させるLGP系の手法。"
            ),
        )
        if balancer_on:
            bal_offset_ratio = st.slider(
                "底辺位置 / D（ゲートからの距離 ÷ ランナー深さ）",
                0.5,
                1.0,
                1.0,
                step=0.05,
                help="1.0で底辺が長辺と一致（製品まで肉盗みが届く）",
            )
            bal_height_ratio = st.slider(
                "▽の高さ H_bal / D",
                0.1,
                0.95,
                0.70,
                step=0.05,
            )

            bal_stage_count = st.slider(
                "肉盗み段数",
                1,
                5,
                2,
                step=1,
                help=(
                    "ネスト数。1=単一▽（旧挙動）、2以上で中央＋外側の階段状肉盗み。"
                    "番号 1 が中央（最薄・最大抵抗）、番号 N が外側。"
                ),
            )
            # default presets: width as a ratio of L_long, thickness as
            # absolute mm values (clamped to ≤ plate_lower_thk).
            _w_defaults = {
                1: [0.60],
                2: [0.30, 0.60],
                3: [0.20, 0.45, 0.70],
                4: [0.15, 0.35, 0.55, 0.80],
                5: [0.10, 0.30, 0.50, 0.70, 0.95],
            }
            _h_defaults_abs = {
                1: [0.30],
                2: [0.25, 0.30],
                3: [0.20, 0.25, 0.30],
                4: [0.15, 0.20, 0.25, 0.30],
                5: [0.10, 0.15, 0.20, 0.25, 0.30],
            }
            bal_widths_mm: list[float] = []
            bal_thks: list[float] = []
            for _k in range(1, bal_stage_count + 1):
                _label = "中央" if _k == 1 else ("外側" if _k == bal_stage_count else "")
                _label_suffix = f"（{_label}）" if _label else ""
                _w_default = _w_defaults[bal_stage_count][_k - 1]
                _h_default = max(0.05, _h_defaults_abs[bal_stage_count][_k - 1])
                _w_ratio = st.slider(
                    f"底辺幅{_k} / L_long{_label_suffix}",
                    0.05,
                    1.0,
                    _w_default,
                    step=0.05,
                    key=f"bal_w_{_k}",
                )
                _h_val = st.slider(
                    f"残り肉厚{_k} [mm]{_label_suffix}",
                    0.05,
                    1.0,
                    float(_h_default),
                    step=0.05,
                    key=f"bal_h_{_k}",
                    help=(
                        "肉盗み(▽)内の残り肉厚。プレート側肉厚 "
                        "(plate_lower_thk) より大きく取ると、その段は"
                        "「肉盛り凸部」になる（流路を逆に広げる用途）。"
                    ),
                )
                bal_widths_mm.append(_w_ratio * runner_long)
                bal_thks.append(float(_h_val))
        else:
            bal_offset_ratio = 1.0
            bal_height_ratio = 0.7
            bal_widths_mm = []
            bal_thks = []

        cell_size = st.slider("メッシュ粗さ [mm/cell]", 0.5, 3.0, 0.5, step=0.1)
        upload = None
    else:
        upload = st.file_uploader(
            "キャビティ画像（暗部=キャビティ、白=外）", type=["png", "jpg", "jpeg"]
        )
        plate_thk = st.slider("均一肉厚 [mm]", 0.2, 2.0, 2.0, step=0.1)
        cell_size = st.slider("ピクセル->mm 換算 [mm/cell]", 0.2, 3.0, 1.0, step=0.1)
        invert = st.checkbox("白を内部として扱う（反転）", value=False)
        threshold = st.slider("二値化しきい値", 16, 240, 128)

    st.header("材料")
    material_key = st.selectbox("樹脂", material_keys, index=material_keys.index("PP_T20"))
    mat = db[material_key]
    st.caption(f"{mat.name}")
    st.caption(
        f"推奨 melt: {mat.T_melt_recommended[0] - 273.15:.0f}–{mat.T_melt_recommended[1] - 273.15:.0f} ℃, "
        f"mold: {mat.T_mold_recommended[0] - 273.15:.0f}–{mat.T_mold_recommended[1] - 273.15:.0f} ℃"
    )

    st.header("射出条件")
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

    st.header("ウォール冷却モデル")
    wall_model = st.radio(
        "壁面冷却の表現",
        options=("none", "skin", "multilayer"),
        index=1,
        format_func=lambda m: {
            "none": "なし（等温・代表粘度のみ）",
            "skin": "スキン層 (1層 + Stefan/Neumann)",
            "multilayer": "層別 (N 層離散化 + Cross-WLF 結合)",
        }[m],
        help=(
            "なし: 既存 HeleShawSolver 相当、温度結合なし。\n"
            "スキン層: 壁面で固化するスキン層を s(t)=c_skin·√(αt) で取り込み、"
            "コア層 h_core=h-2s だけが流れる。短ショットも検出。\n"
            "層別: 厚み方向を N 層に分割、Neumann 1D 温度プロファイルから "
            "層別粘度を Cross-WLF で評価。fixed-point で τ ↔ T_k ↔ η_k を結合。"
        ),
    )

    # default container (so downstream `solver = HeleShawSolver(...)` /
    # `MultilayerHeleShawSolver(...)` always has the kwargs it expects).
    skin_on = wall_model == "skin"
    c_skin = 0.0
    skin_max_iter = 5
    skin_tol = 1e-3
    multilayer_on = wall_model == "multilayer"
    num_layers = 5
    layer_distribution = "wall_refined"
    multilayer_max_iter = 8
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
            7,
            5,
            help="厚み方向の離散化数。奇数で中央層が短ショット判定の代表セルに。",
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
            15,
            8,
            help="τ ↔ T_k ↔ η_k 結合の反復回数。",
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
                " short shot にマーク。PP は 0.3 が目安。"
            ),
        )

    st.header("射出圧縮成形 (ICM)")
    icm = st.checkbox("圧縮成形ON", value=True)
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

    st.header("出力")
    num_frames = st.slider("アニメーションフレーム数", 12, 60, 30)


# ----------------------- main panel -----------------------
def build_geometry() -> Geometry:
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
            return build_direct_gate_geometry(cfg_dg)
        except ValueError as exc:
            st.error(f"パラメータ不整合: {exc}")
            st.stop()
    if geom_source.startswith("Film gate"):
        try:
            cfg = FilmGateConfig(
                plate_w_mm=plate_w,
                plate_h_mm=plate_h,
                plate_thk_mm=plate_thk,
                runner_long_mm=runner_long,
                runner_short_diameter_mm=runner_short_d,
                runner_depth_mm=runner_depth,
                runner_thk_mm=runner_thk_film,
                runner_flat_depth_mm=runner_depth * flat_ratio,
                runner_slope_depth_mm=runner_depth * (1.0 - flat_ratio),
                valve_gate_diameter_mm=valve_d,
                gate_width_mm=gate_w,
                cell_size_mm=cell_size,
                balancer_enabled=balancer_on,
                balancer_height_mm=runner_depth * bal_height_ratio,
                balancer_base_distance_from_gate_mm=runner_depth * bal_offset_ratio,
                balancer_base_widths_mm=tuple(bal_widths_mm),
                balancer_thicknesses_mm=tuple(bal_thks),
                plate_split_height_mm=plate_split if plate_split > 0 else 0.0,
                plate_lower_thk_mm=plate_lower_thk if plate_split > 0 else None,
                plate_upper_thk_mm=plate_upper_thk if plate_split > 0 else None,
            )
            return build_film_gate_geometry(cfg)
        except ValueError as exc:
            st.error(f"パラメータ不整合: {exc}")
            st.stop()
    if upload is None:
        st.warning("画像をアップロードしてください。")
        st.stop()
    img_bytes = upload.read()
    tmp_path = Path(tempfile.mkdtemp()) / upload.name
    tmp_path.write_bytes(img_bytes)
    g = geometry_from_image(
        tmp_path,
        cell_size_mm=cell_size,
        plate_thk_mm=plate_thk,
        invert=invert,
        threshold=threshold,
    )
    if not g.gates:
        # default gate: leftmost cavity column, vertical center
        ys, xs = np.where(g.mask)
        if ys.size == 0:
            st.error("キャビティ領域が検出できませんでした。しきい値か反転設定を見直してください。")
            st.stop()
        ix = int(xs.min())
        col_ys = ys[xs == xs.min()]
        iy = int(np.median(col_ys))
        g.add_gate(iy, ix)
    return g


col_left, col_right = st.columns([1, 1.3])

with col_left:
    st.subheader("成形品設計図")
    geom = build_geometry()
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
    im = ax.imshow(fig_data, origin="lower", extent=extent, cmap="cividis")
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
        "原点 (x, y) = (0, 0) はゲート中央 = 製品中央。半円部は y < 0 側。赤丸はバルブゲート位置。"
    )


if do_run:
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

        # 重い PNG/GIF レンダリングはここで一回だけやる。後段の widget 操作で
        # rerun が走っても再生成しないよう、すべて session_state に置く。
        _tmp_dir = Path(tempfile.mkdtemp())
        _gif_path = render_fill_animation(
            result, _tmp_dir / "fill.gif", num_frames=num_frames, fps=8
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
            ):
                if _p is not None and _p.exists():
                    _zf_run.write(_p, _p.name)
            _zf_run.writestr(
                "metadata.json",
                json.dumps(result.metadata, indent=2, ensure_ascii=False, default=str),
            )

        # 解析結果の一式を session_state に格納。次回 rerun（3D スライダー操作等）
        # でも下のブロックがこれを拾って表示する。
        st.session_state["mfs_result"] = result
        st.session_state["mfs_geom"] = geom
        st.session_state["mfs_skin_on"] = skin_on
        st.session_state["mfs_multilayer_on"] = multilayer_on
        st.session_state["mfs_num_frames"] = num_frames
        st.session_state["mfs_tmp_dir"] = _tmp_dir
        st.session_state["mfs_gif_path"] = _gif_path
        st.session_state["mfs_press_path"] = _press_path
        st.session_state["mfs_weld_path"] = _weld_path
        st.session_state["mfs_skin_path"] = _skin_path
        st.session_state["mfs_core_path"] = _core_path
        st.session_state["mfs_layer_T_grid_path"] = _layer_T_grid_path
        st.session_state["mfs_layer_eta_grid_path"] = _layer_eta_grid_path
        st.session_state["mfs_layer_short_shot_path"] = _layer_short_shot_path
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
    press_path = st.session_state["mfs_press_path"]
    weld_path = st.session_state["mfs_weld_path"]
    skin_path = st.session_state["mfs_skin_path"]
    core_path = st.session_state["mfs_core_path"]
    layer_T_grid_path = st.session_state.get("mfs_layer_T_grid_path")
    layer_eta_grid_path = st.session_state.get("mfs_layer_eta_grid_path")
    layer_short_shot_path = st.session_state.get("mfs_layer_short_shot_path")
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
        st.image(str(gif_path))
        st.download_button(
            "⬇ GIFをダウンロード",
            data=_zip_bytes,
            file_name="mold_flow_results.zip",
            mime="application/zip",
            key="dl_zip_all",
            help="GIF・各マップ PNG・metadata.json を1つの ZIP にまとめてダウンロード",
        )

        with st.expander("圧力マップ"):
            st.image(str(press_path))
            st.caption("0=ゲート遠端、1=ゲート。実圧力スケールではなく相対分布。")
            _download("⬇ PNGをダウンロード", press_path, "image/png", "dl_press_png")

        with st.expander("等値線・ウェルドライン候補・エアトラップ"):
            st.image(str(weld_path))
            st.caption("赤=合流（ウェルド）候補、黄×=最終充填位置（エアトラップ候補）")
            _download("⬇ PNGをダウンロード", weld_path, "image/png", "dl_weld_png")

        if skin_path is not None and core_path is not None:
            with st.expander("スキン層 / コア層 / short shot"):
                st.image(str(skin_path))
                st.caption("スキン層厚さ s(x,y) [mm]。流動が遅いほど・薄肉ほど s が大きい。")
                _download("⬇ スキン層 PNGをダウンロード", skin_path, "image/png", "dl_skin_png")
                st.image(str(core_path))
                st.caption(
                    "コア層 h_core = h - 2s。赤マーク = スキン同士が会合した short shot 候補。"
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
                    f"短ショット率={md.get('short_shot_fraction', 0.0):.3f}"
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
                    st.markdown("**短ショット予測** — 中央層温度が T_solid を切ったセルを赤マーク")
                    st.image(str(layer_short_shot_path))
                    _download(
                        "⬇ 短ショット PNG",
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

        with st.expander("生データ"):
            st.json(result.metadata)
else:
    with col_right:
        st.info("左側でパラメータを設定し、「解析実行」を押してください。")
