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
    FilmGateConfig,
    HeleShawSolver,
    MaterialDB,
    build_demo_geometry,
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

st.set_page_config(page_title="Mold Flow Sim (simplified)", layout="wide")
st.title("射出成形 簡易流動解析")
st.caption(
    "Hele-Shaw近似 + Cross-WLF粘度 + Pseudo-Conduction Fill Time モデルによる、"
    "射出成形流動の初期スクリーニング・概念検証ツール。"
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
            "Demo plate (synthetic)",
            "Film gate (parametric)",
            "画像から生成 (PNG/JPG)",
        ],
        index=1,
    )

    if geom_source.startswith("Demo"):
        plate_w = st.slider("製品幅 [mm]", 40.0, 300.0, 120.0, step=5.0)
        plate_h = st.slider("製品高 [mm]", 30.0, 160.0, 80.0, step=5.0)
        plate_thk = st.slider("製品肉厚 [mm]", 0.2, 2.0, 2.0, step=0.1)
        runner_thk = st.slider("ランナー肉厚 [mm]", 1.0, 8.0, 4.0, step=0.1)
        sprue_thk = st.slider("スプルー肉厚 [mm]", 2.0, 10.0, 6.0, step=0.1)
        cell_size = st.slider("メッシュ粗さ [mm/cell]", 0.5, 3.0, 1.0, step=0.1)
        gate_count = st.slider("ゲート数", 1, 4, 1)
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
                _h_default = max(
                    0.05,
                    min(_h_defaults_abs[bal_stage_count][_k - 1], plate_lower_thk),
                )
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
                    float(plate_lower_thk),
                    float(_h_default),
                    step=0.05,
                    key=f"bal_h_{_k}",
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
    material_key = st.selectbox("樹脂", material_keys, index=material_keys.index("PP"))
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
        400.0,
        step=5.0,
        help="ソディック等の成形機取説の射出率に対応。",
    )

    st.header("スキン層モデル")
    skin_on = st.checkbox(
        "スキン層形成を考慮",
        value=True,
        help=(
            "金型壁面で樹脂が固化してスキン層が育つ現象を Stefan/Neumann 形 "
            "s(t) = c_skin · √(α·t) で取り込む。流路はコア層 h_core = h - 2·s "
            "のみを通る。コアが閉塞したセルは short shot 候補。バルクのコア "
            "温度低下や粘度の動的追跡は引き続き無視。"
        ),
    )
    if skin_on:
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
    else:
        c_skin = 0.0
        skin_max_iter = 5
        skin_tol = 1e-3

    st.header("射出圧縮成形 (ICM)")
    icm = st.checkbox("圧縮成形ON", value=True)
    if icm:
        comp_factor = st.slider("初期隙間倍率 h_init/h_final", 1.05, 3.0, 2.2, step=0.05)
        comp_frac = st.slider("圧縮位相の充填占有率", 0.1, 1.0, 0.95, step=0.05)
    else:
        comp_factor = 1.0
        comp_frac = 0.0

    st.header("出力")
    num_frames = st.slider("アニメーションフレーム数", 12, 60, 30)


# ----------------------- main panel -----------------------
def build_geometry() -> Geometry:
    if geom_source.startswith("Demo"):
        return build_demo_geometry(
            plate_w_mm=plate_w,
            plate_h_mm=plate_h,
            plate_thk_mm=plate_thk,
            runner_thk_mm=runner_thk,
            sprue_thk_mm=sprue_thk,
            cell_size_mm=cell_size,
            gate_count=gate_count,
        )
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
        solver = HeleShawSolver(
            geometry=geom,
            material=mat,
            melt_temperature_K=melt_C + 273.15,
            mold_temperature_K=mold_C + 273.15,
            injection_velocity_mms=inj_v,
            injection_volume_flow_cm3s=inj_Q,
            compression_molding=icm,
            compression_factor=comp_factor,
            compression_fraction=comp_frac,
            skin_layer_enabled=skin_on,
            skin_growth_constant=c_skin,
            skin_max_iterations=skin_max_iter,
            skin_convergence_tol=skin_tol,
        )
        result = solver.solve(num_frames=num_frames)

    with col_right:
        st.subheader("結果")
        c1, c2, c3 = st.columns(3)
        c1.metric("総充填時間 T_fill", f"{result.total_fill_time_s:.3f} s")
        c2.metric("代表粘度 η_eff", f"{result.viscosity_Pa_s:.1f} Pa·s")
        c3.metric("キャビティ体積", f"{geom.volume_cm3():.2f} cm³")

        if skin_on:
            inflation = result.metadata.get("T_fill_inflation", 1.0)
            short_count = (
                int(result.short_shot_mask.sum()) if result.short_shot_mask is not None else 0
            )
            cells_total = int(geom.mask.sum())
            short_pct = 100.0 * short_count / max(cells_total, 1)
            iters = result.metadata.get("skin_iterations", 0)
            converged = result.metadata.get("skin_converged", False)
            s1, s2, s3 = st.columns(3)
            s1.metric(
                "T_fill 増分（スキン層）",
                f"×{inflation:.2f}",
                help="スキン層なしの T_fill_baseline に対する倍率（圧力一定近似）",
            )
            s2.metric(
                "short shot セル",
                f"{short_count} / {cells_total}",
                f"{short_pct:.1f} %",
                delta_color="inverse",
            )
            s3.metric(
                "fixed-point 反復",
                f"{iters} 回",
                "収束" if converged else "上限到達",
                delta_color="off" if converged else "inverse",
            )
            if result.skin_thickness_mm is not None:
                s_max_mm = float(np.nanmax(result.skin_thickness_mm[geom.mask]))
                h_core_min = float(np.nanmin(result.core_thickness_mm[geom.mask]))
                st.caption(
                    f"スキン最大 {s_max_mm * 1e3:.1f} μm,  コア最小 h_core = {h_core_min:.3f} mm"
                )

        tmp_dir = Path(tempfile.mkdtemp())
        gif_path = render_fill_animation(result, tmp_dir / "fill.gif", num_frames=num_frames, fps=8)
        press_path = render_pressure_map(result, tmp_dir / "pressure.png")
        weld_path = render_weldlines(result, tmp_dir / "weld.png")
        skin_path: Path | None = None
        core_path: Path | None = None
        if skin_on and result.skin_thickness_mm is not None:
            skin_path = render_skin_layer_map(result, tmp_dir / "skin.png")
            core_path = render_core_layer_map(result, tmp_dir / "core.png")

        def _download(label: str, path: Path, mime: str, key: str) -> None:
            with open(path, "rb") as _f:
                st.download_button(
                    label,
                    data=_f.read(),
                    file_name=path.name,
                    mime=mime,
                    key=key,
                )

        # Build a ZIP of every artifact so the user can grab the whole
        # animation set (GIF + maps + metadata) in a single click.
        _zip_buf = io.BytesIO()
        with zipfile.ZipFile(_zip_buf, "w", zipfile.ZIP_DEFLATED) as _zf:
            for _p in (gif_path, press_path, weld_path, skin_path, core_path):
                if _p is not None and _p.exists():
                    _zf.write(_p, _p.name)
            _zf.writestr(
                "metadata.json",
                json.dumps(result.metadata, indent=2, ensure_ascii=False, default=str),
            )
        _zip_bytes = _zip_buf.getvalue()

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

        with st.expander("3D表示（plotly・実験的）"):
            st.caption(
                "PL（パーティングライン）= Z=0 を底面とし、各セルを厚み h(x,y) 分だけ"
                "立ち上げたソリッド表示。**天面と側壁の両方が物理量で着色**され、"
                "1つのカラーバーで読める（PLの薄グレー床は形状参照用）。"
                "ドラッグで回転、スクロールでズーム。物理は 2D Hele-Shaw のまま"
                "（表現上の3D化のみ）、Z軸スケールは視認性のため誇張あり。"
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
        st.markdown(
            """
            **物理モデル**
            - Hele-Shaw近似（薄肉樹脂流動）
            - Cross-WLF粘度モデル（温度・せん断速度依存）
            - Pseudo-Conduction法（楕円型方程式 ∇·(S∇τ)=1 の一発解）
            - 流動先端 = τの等値面、絶対時間 = τ正規化×(V/Q)
            - スキン層モデル（オプション）：Stefan/Neumann 形 s(t)=c_skin·√(αt)
              で壁面固化を取り込み、コア層 h_core = h - 2s のみが流路として効く。
              τ ↔ h_core を fixed-point 反復で釣り合わせる。

            **モデル化の対象外（簡略化のため割り切っている範囲）**
            - コアのバルク温度低下（粘度の動的更新は無し、Neumann近似で熱結合を切離）
            - 真の3D流れ（コーナー効果、ジェッティング）
            - 結晶化・収縮・反り
            - パッキング段階の保圧

            このツールは流動の傾向把握・パラメータ感度の検討を目的にした
            初期スクリーニング向けです。最終的な型設計の検証には商用CAEの併用を推奨します。
            """
        )
