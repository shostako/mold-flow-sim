"""CLI demo: run a few simulations with parameter sweeps and write outputs.

Usage::

    python run_demo.py                           # run all cases
    python run_demo.py --cases PP_baseline       # run a single case
    python run_demo.py --cases FilmGate_PP_default

Cases are split into three families:

- *demo cases* use :func:`build_demo_geometry` (plate + runner + sprue).
- *film-gate cases* use :func:`build_film_gate_geometry` driven by a
  :class:`FilmGateConfig`.
- *direct-gate cases* use :func:`build_direct_gate_geometry` driven by a
  :class:`DirectGateConfig` (Φ-pin gate + thin sprue strip + plate).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from core import (
    DirectGateConfig,
    FilmGate2Config,
    FilmGateConfig,
    Geometry,
    HeleShawSolver,
    MaterialDB,
    MultilayerHeleShawSolver,
    build_demo_geometry,
    build_direct_gate_geometry,
    build_film_gate2_geometry,
    build_film_gate_geometry,
    export_frames,
    render_core_layer_map,
    render_fill_animation,
    render_pressure_map,
    render_skin_layer_map,
    render_weldlines,
)
from core.visualizer import render_layer_grid, render_short_shot_map


def _solve_and_export(
    label: str,
    out_root: Path,
    geom: Geometry,
    *,
    material_key: str,
    melt_K: float,
    mold_K: float,
    inj_velocity_mms: float,
    inj_Q_cm3s: float,
    compression: bool = False,
    compression_factor: float = 1.5,
    compression_stroke_mm: float | None = None,
    compression_fraction: float = 0.6,
    skin_layer: bool = False,
    skin_growth_constant: float = 0.5,
    skin_max_iterations: int = 5,
    skin_convergence_tol: float = 1e-3,
    multilayer: bool = False,
    num_layers: int = 5,
    layer_distribution: str = "wall_refined",
    multilayer_max_iterations: int = 8,
    multilayer_convergence_tol: float = 1e-3,
    solidification_temperature_fraction: float = 0.3,
    shear_heating_enabled: bool = False,
    num_frames: int = 30,
) -> None:
    if skin_layer and multilayer:
        raise ValueError(
            "skin_layer and multilayer are mutually exclusive — choose one wall-cooling model"
        )
    db = MaterialDB()
    if multilayer:
        solver = MultilayerHeleShawSolver(
            geometry=geom,
            material=db[material_key],
            melt_temperature_K=melt_K,
            mold_temperature_K=mold_K,
            injection_velocity_mms=inj_velocity_mms,
            injection_volume_flow_cm3s=inj_Q_cm3s,
            compression_molding=compression,
            compression_factor=compression_factor,
            compression_stroke_mm=compression_stroke_mm,
            compression_fraction=compression_fraction,
            num_layers=num_layers,
            layer_distribution=layer_distribution,
            thermal_coupling=True,
            max_iterations=multilayer_max_iterations,
            convergence_tol=multilayer_convergence_tol,
            solidification_temperature_fraction=solidification_temperature_fraction,
            shear_heating_enabled=shear_heating_enabled,
        )
    else:
        solver = HeleShawSolver(
            geometry=geom,
            material=db[material_key],
            melt_temperature_K=melt_K,
            mold_temperature_K=mold_K,
            injection_velocity_mms=inj_velocity_mms,
            injection_volume_flow_cm3s=inj_Q_cm3s,
            compression_molding=compression,
            compression_factor=compression_factor,
            compression_stroke_mm=compression_stroke_mm,
            compression_fraction=compression_fraction,
            skin_layer_enabled=skin_layer,
            skin_growth_constant=skin_growth_constant,
            skin_max_iterations=skin_max_iterations,
            skin_convergence_tol=skin_convergence_tol,
        )
    print(f"[{label}] solving... cells={int(geom.mask.sum())} V={geom.volume_cm3():.2f} cm^3")
    result = solver.solve(num_frames=num_frames)
    extra = ""
    if skin_layer:
        short = int(result.short_shot_mask.sum()) if result.short_shot_mask is not None else 0
        inflation = result.metadata.get("T_fill_inflation", 1.0)
        extra = f"  skin: x{inflation:.2f} T_fill, short_shot {short}/{int(geom.mask.sum())}"
    elif multilayer:
        short_frac = result.metadata.get("short_shot_fraction", 0.0)
        iters = result.metadata.get("multilayer_iterations", 0)
        conv = result.metadata.get("multilayer_converged", False)
        inflation = result.metadata.get("T_fill_inflation", 1.0)
        extra = (
            f"  multilayer N={num_layers}/{layer_distribution}: "
            f"iters={iters} conv={conv} x{inflation:.2f} T_fill, "
            f"short_shot {short_frac * 100:.1f}%"
        )
    print(
        f"[{label}] T_fill={result.total_fill_time_s:.3f} s  "
        f"eta_eff={result.viscosity_Pa_s:.1f} Pa.s{extra}"
    )

    out_dir = out_root / label
    out_dir.mkdir(parents=True, exist_ok=True)

    render_fill_animation(result, out_dir / "fill.gif", num_frames=num_frames, fps=8)
    render_pressure_map(result, out_dir / "pressure.png")
    render_weldlines(result, out_dir / "weld_airtraps.png")
    if skin_layer and result.skin_thickness_mm is not None:
        render_skin_layer_map(result, out_dir / "skin.png")
        render_core_layer_map(result, out_dir / "core.png")
    if multilayer and getattr(result, "layer_temperature_K", None) is not None:
        render_layer_grid(result, out_dir / "layer_temperature_grid.png", field="temperature")
        render_layer_grid(result, out_dir / "layer_viscosity_grid.png", field="viscosity")
        render_short_shot_map(result, out_dir / "multilayer_short_shot.png")
    export_frames(result, out_dir / "frames", num_frames=8)


def run_demo_case(
    label: str,
    out_root: Path,
    *,
    cell_size_mm: float = 1.0,
    plate_thk_mm: float = 2.0,
    gate_count: int = 1,
    **solver_kwargs,
) -> None:
    geom = build_demo_geometry(
        cell_size_mm=cell_size_mm,
        plate_thk_mm=plate_thk_mm,
        gate_count=gate_count,
    )
    _solve_and_export(label, out_root, geom, **solver_kwargs)


def run_film_gate_case(
    label: str,
    out_root: Path,
    *,
    cfg: FilmGateConfig,
    **solver_kwargs,
) -> None:
    geom = build_film_gate_geometry(cfg)
    _solve_and_export(label, out_root, geom, **solver_kwargs)


def run_direct_gate_case(
    label: str,
    out_root: Path,
    *,
    cfg: DirectGateConfig,
    **solver_kwargs,
) -> None:
    geom = build_direct_gate_geometry(cfg)
    _solve_and_export(label, out_root, geom, **solver_kwargs)


# ---------------------- case definitions ----------------------

DEMO_CASES: dict[str, dict] = {
    "PP_baseline": dict(
        material_key="PP",
        melt_K=503.15,
        mold_K=313.15,
        inj_velocity_mms=100.0,
        inj_Q_cm3s=20.0,
    ),
    "PP_hot_melt": dict(
        material_key="PP",
        melt_K=533.15,
        mold_K=323.15,
        inj_velocity_mms=100.0,
        inj_Q_cm3s=20.0,
    ),
    "PP_slow_inj": dict(
        material_key="PP",
        melt_K=503.15,
        mold_K=313.15,
        inj_velocity_mms=30.0,
        inj_Q_cm3s=6.0,
    ),
    "PC_baseline": dict(
        material_key="PC",
        melt_K=583.15,
        mold_K=373.15,
        inj_velocity_mms=80.0,
        inj_Q_cm3s=15.0,
    ),
    "PA66_baseline": dict(
        material_key="PA66",
        melt_K=563.15,
        mold_K=343.15,
        inj_velocity_mms=120.0,
        inj_Q_cm3s=25.0,
    ),
    "PP_compression": dict(
        material_key="PP",
        melt_K=503.15,
        mold_K=313.15,
        inj_velocity_mms=80.0,
        inj_Q_cm3s=15.0,
        compression=True,
        compression_factor=1.6,
        compression_fraction=0.65,
    ),
    "PP_dual_gate": dict(
        material_key="PP",
        melt_K=503.15,
        mold_K=313.15,
        inj_velocity_mms=100.0,
        inj_Q_cm3s=20.0,
        gate_count=2,
    ),
    "PP_skin_layer": dict(
        material_key="PP",
        melt_K=503.15,
        mold_K=313.15,
        inj_velocity_mms=100.0,
        inj_Q_cm3s=20.0,
        plate_thk_mm=1.5,  # thinner plate so the skin layer is visible
        skin_layer=True,
        skin_growth_constant=0.5,
    ),
}


def _film_gate_cfg_default() -> FilmGateConfig:
    return FilmGateConfig(
        plate_w_mm=120.0,
        plate_h_mm=80.0,
        plate_thk_mm=2.0,
        runner_long_mm=80.0,
        runner_short_diameter_mm=12.0,
        runner_depth_mm=20.0,
        runner_thk_mm=4.0,
        runner_flat_depth_mm=8.0,
        runner_slope_depth_mm=12.0,
        valve_gate_diameter_mm=4.0,
        gate_width_mm=60.0,
        cell_size_mm=1.0,
    )


def _film_gate_cfg_narrow_aperture() -> FilmGateConfig:
    return FilmGateConfig(
        plate_w_mm=120.0,
        plate_h_mm=80.0,
        plate_thk_mm=2.0,
        runner_long_mm=80.0,
        runner_short_diameter_mm=12.0,
        runner_depth_mm=20.0,
        runner_thk_mm=4.0,
        runner_flat_depth_mm=8.0,
        runner_slope_depth_mm=12.0,
        valve_gate_diameter_mm=4.0,
        gate_width_mm=20.0,  # narrow aperture → expect localized flow
        cell_size_mm=1.0,
    )


def _film_gate_cfg_full_aperture_thin_runner() -> FilmGateConfig:
    return FilmGateConfig(
        plate_w_mm=120.0,
        plate_h_mm=80.0,
        plate_thk_mm=2.0,
        runner_long_mm=110.0,  # close to plate width
        runner_short_diameter_mm=10.0,
        runner_depth_mm=15.0,
        runner_thk_mm=2.5,  # close to plate thickness
        runner_flat_depth_mm=3.0,  # mostly slope → smooth thickness transition
        runner_slope_depth_mm=12.0,
        valve_gate_diameter_mm=3.0,
        gate_width_mm=110.0,  # full aperture
        cell_size_mm=1.0,
    )


def _film_gate_cfg_stepped_plate() -> FilmGateConfig:
    """Stepped plate (t0.35 gate-side / t0.50 far-side) — mimics the
    real ultra-thin product. Used as the baseline for stroke-mode
    compression demos where the mold shim adds a fixed 0.7 mm stroke."""
    return FilmGateConfig(
        plate_w_mm=120.0,
        plate_h_mm=80.0,
        plate_thk_mm=0.50,  # fallback (ignored, split + lower/upper override)
        runner_long_mm=80.0,
        runner_short_diameter_mm=12.0,
        runner_depth_mm=20.0,
        runner_thk_mm=4.0,
        runner_flat_depth_mm=8.0,
        runner_slope_depth_mm=12.0,
        valve_gate_diameter_mm=4.0,
        gate_width_mm=60.0,
        plate_split_height_mm=20.0,
        plate_lower_thk_mm=0.35,
        plate_upper_thk_mm=0.50,
        cell_size_mm=1.0,
    )


def _film_gate_cfg_with_balancer() -> FilmGateConfig:
    """LGP-style flow balancer: same outer geometry as the default case
    but with a ▽-shaped local thinning carved in the runner centerline."""
    return FilmGateConfig(
        plate_w_mm=120.0,
        plate_h_mm=80.0,
        plate_thk_mm=2.0,
        runner_long_mm=80.0,
        runner_short_diameter_mm=12.0,
        runner_depth_mm=20.0,
        runner_thk_mm=4.0,
        runner_flat_depth_mm=8.0,
        runner_slope_depth_mm=12.0,
        valve_gate_diameter_mm=4.0,
        gate_width_mm=60.0,
        cell_size_mm=1.0,
        balancer_enabled=True,
        balancer_base_width_mm=36.0,  # 0.6 × W_gate
        balancer_height_mm=14.0,  # 0.7 × D
        balancer_base_distance_from_gate_mm=20.0,  # base at long edge (= D)
        balancer_target_thickness_mm=2.0,  # = plate_thk (parallel to plate)
    )


def _direct_gate_cfg_default() -> DirectGateConfig:
    return DirectGateConfig(
        plate_w_mm=120.0,
        plate_h_mm=80.0,
        plate_thk_mm=2.0,
        gate_diameter_mm=3.0,
        gate_offset_mm=20.0,
        cell_size_mm=1.0,
    )


def _direct_gate_cfg_compression() -> DirectGateConfig:
    return DirectGateConfig(
        plate_w_mm=120.0,
        plate_h_mm=80.0,
        plate_thk_mm=1.5,
        gate_diameter_mm=3.0,
        gate_offset_mm=20.0,
        cell_size_mm=1.0,
    )


DIRECT_GATE_CASES: dict[str, dict] = {
    "DirectGate_PP_default": dict(
        cfg=_direct_gate_cfg_default(),
        material_key="PP",
        melt_K=503.15,
        mold_K=313.15,
        inj_velocity_mms=100.0,
        inj_Q_cm3s=20.0,
    ),
    "DirectGate_PP_compression": dict(
        cfg=_direct_gate_cfg_compression(),
        material_key="PP",
        melt_K=503.15,
        mold_K=313.15,
        inj_velocity_mms=80.0,
        inj_Q_cm3s=15.0,
        compression=True,
        compression_factor=1.6,
        compression_fraction=0.65,
    ),
}


FILM_GATE_CASES: dict[str, dict] = {
    "FilmGate_PP_default": dict(
        cfg=_film_gate_cfg_default(),
        material_key="PP",
        melt_K=503.15,
        mold_K=313.15,
        inj_velocity_mms=100.0,
        inj_Q_cm3s=20.0,
    ),
    "FilmGate_PP_narrow": dict(
        cfg=_film_gate_cfg_narrow_aperture(),
        material_key="PP",
        melt_K=503.15,
        mold_K=313.15,
        inj_velocity_mms=100.0,
        inj_Q_cm3s=20.0,
    ),
    "FilmGate_PP_full_thin": dict(
        cfg=_film_gate_cfg_full_aperture_thin_runner(),
        material_key="PP",
        melt_K=503.15,
        mold_K=313.15,
        inj_velocity_mms=100.0,
        inj_Q_cm3s=20.0,
    ),
    "FilmGate_PP_balancer": dict(
        cfg=_film_gate_cfg_with_balancer(),
        material_key="PP",
        melt_K=503.15,
        mold_K=313.15,
        inj_velocity_mms=100.0,
        inj_Q_cm3s=20.0,
    ),
    # Stepped plate (t0.35 / t0.50) + stroke compression. Mold shim adds
    # 0.7 mm to every compression cell so the 0.15 mm step is preserved
    # (factor mode would distort it). Direct counterpart to the gokuusu
    # STEP4 design discussion.
    "FilmGate_PP_stepped_stroke": dict(
        cfg=_film_gate_cfg_stepped_plate(),
        material_key="PP",
        melt_K=503.15,
        mold_K=313.15,
        inj_velocity_mms=100.0,
        inj_Q_cm3s=20.0,
        compression=True,
        compression_stroke_mm=0.70,
        compression_fraction=0.95,
    ),
    # Same stepped-plate baseline as above, but driven by the multilayer
    # solver with N=5 wall-refined layers. Per-layer Neumann temperature
    # + Cross-WLF viscosity coupling exposes wall freezing and centre-
    # core temperature drop, and the short-shot mask is populated when
    # the centre layer cools past T_solid. Use this case to compare the
    # baseline single-layer τ-only run against the layered prediction
    # for the gokuusu STEP4 design conversations.
    "FilmGate_PP_multilayer_5L": dict(
        cfg=_film_gate_cfg_stepped_plate(),
        material_key="PP",
        melt_K=503.15,
        mold_K=313.15,
        inj_velocity_mms=100.0,
        inj_Q_cm3s=20.0,
        compression=True,
        compression_stroke_mm=0.70,
        compression_fraction=0.95,
        multilayer=True,
        num_layers=5,
        layer_distribution="wall_refined",
        multilayer_max_iterations=8,
        multilayer_convergence_tol=1e-3,
        solidification_temperature_fraction=0.3,
    ),
    # ---- Stage-1 shear-heating reference case ----
    # Same stepped-plate geometry as FilmGate_PP_multilayer_5L but with the
    # viscous-dissipation correction enabled. Comparing the metadata
    # (brinkman_number_max / shear_heating_max_K / tau_max) between the
    # two cases shows what fraction of the temperature rise comes from
    # shear heating, and how much the local viscosity drop accelerates
    # the τ field. Targets gokuusu STEP4 ultra-thin (t < 0.5 mm) regime
    # where Br ≫ 1 is the norm.
    "FilmGate_PP_multilayer_5L_shear": dict(
        cfg=_film_gate_cfg_stepped_plate(),
        material_key="PP",
        melt_K=503.15,
        mold_K=313.15,
        inj_velocity_mms=300.0,  # high V → γ̇ large
        inj_Q_cm3s=60.0,
        compression=True,
        compression_stroke_mm=0.70,
        compression_fraction=0.60,
        multilayer=True,
        num_layers=7,
        layer_distribution="wall_refined",
        multilayer_max_iterations=12,
        multilayer_convergence_tol=1e-3,
        solidification_temperature_fraction=0.3,
        shear_heating_enabled=True,
    ),
}


# ---------------------- entrypoint ----------------------


def _film_gate2_cfg_rightangle() -> FilmGate2Config:
    """Right trapezoid (gate_position=0, valve at the right end)."""
    return FilmGate2Config(
        plate_w_mm=300.0,
        plate_h_mm=50.0,
        plate_thk_mm=0.35,
        gate_depth_mm=30.0,
        gate_position_mm=0.0,
        left_edge_mm=10.0,
        land_width_mm=1.0,
        land_depth_mm=0.35,
        taper1_len_mm=8.0,
        mid_depth_a_mm=1.5,
        mid_depth_b_mm=1.5,
        taper2_left_mm=5.0,
        taper2_right_mm=10.0,
        runner_depth_mm=3.0,
        runner_top_mm=4.0,
        runner_bottom_mm=2.0,
        valve_gate_diameter_mm=3.0,
        cell_size_mm=1.0,
        plate_split_height_mm=20.0,
        plate_lower_thk_mm=0.35,
        plate_upper_thk_mm=0.50,
    )


def _film_gate2_cfg_isosceles() -> FilmGate2Config:
    """Isosceles trapezoid (gate_position=Wp/2, valve at the center)."""
    return FilmGate2Config(
        plate_w_mm=300.0,
        plate_h_mm=50.0,
        plate_thk_mm=0.35,
        gate_depth_mm=30.0,
        gate_position_mm=150.0,
        left_edge_mm=10.0,
        land_width_mm=1.0,
        land_depth_mm=0.35,
        taper1_len_mm=8.0,
        mid_depth_a_mm=1.5,
        mid_depth_b_mm=1.5,
        taper2_left_mm=5.0,
        taper2_right_mm=10.0,
        runner_depth_mm=3.0,
        runner_top_mm=4.0,
        runner_bottom_mm=2.0,
        valve_gate_diameter_mm=3.0,
        cell_size_mm=1.0,
        plate_split_height_mm=20.0,
        plate_lower_thk_mm=0.35,
        plate_upper_thk_mm=0.50,
    )


def _film_gate2_cfg_stepped() -> FilmGate2Config:
    """Right trapezoid with a depth step between the two taper stages
    (mid_depth_a != mid_depth_b)."""
    return FilmGate2Config(
        plate_w_mm=300.0,
        plate_h_mm=50.0,
        plate_thk_mm=0.35,
        gate_depth_mm=30.0,
        gate_position_mm=0.0,
        left_edge_mm=10.0,
        land_width_mm=1.0,
        land_depth_mm=0.35,
        taper1_len_mm=8.0,
        mid_depth_a_mm=1.0,
        mid_depth_b_mm=2.4,
        taper2_left_mm=5.0,
        taper2_right_mm=10.0,
        runner_depth_mm=3.0,
        runner_top_mm=4.0,
        runner_bottom_mm=2.0,
        valve_gate_diameter_mm=3.0,
        cell_size_mm=1.0,
        plate_split_height_mm=20.0,
        plate_lower_thk_mm=0.35,
        plate_upper_thk_mm=0.50,
    )


def run_film_gate2_case(
    label: str,
    out_root: Path,
    *,
    cfg: FilmGate2Config,
    **solver_kwargs,
) -> None:
    geom = build_film_gate2_geometry(cfg)
    _solve_and_export(label, out_root, geom, **solver_kwargs)


FILM_GATE2_CASES: dict[str, dict] = {
    "FilmGate2_rightangle": dict(
        cfg=_film_gate2_cfg_rightangle(),
        material_key="PP_T20",
        melt_K=523.15,
        mold_K=323.15,
        inj_velocity_mms=400.0,
        inj_Q_cm3s=589.0,
    ),
    "FilmGate2_isosceles": dict(
        cfg=_film_gate2_cfg_isosceles(),
        material_key="PP_T20",
        melt_K=523.15,
        mold_K=323.15,
        inj_velocity_mms=400.0,
        inj_Q_cm3s=589.0,
    ),
    "FilmGate2_stepped": dict(
        cfg=_film_gate2_cfg_stepped(),
        material_key="PP_T20",
        melt_K=523.15,
        mold_K=323.15,
        inj_velocity_mms=400.0,
        inj_Q_cm3s=589.0,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs", help="output root directory")
    parser.add_argument(
        "--cases",
        nargs="*",
        default=None,
        help="case keys to run (default: all demo + film-gate cases)",
    )
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    all_keys = (
        list(DEMO_CASES.keys())
        + list(FILM_GATE_CASES.keys())
        + list(FILM_GATE2_CASES.keys())
        + list(DIRECT_GATE_CASES.keys())
    )
    keys = args.cases or all_keys
    for k in keys:
        if k in DEMO_CASES:
            run_demo_case(k, out_root, **DEMO_CASES[k])
        elif k in FILM_GATE_CASES:
            run_film_gate_case(k, out_root, **FILM_GATE_CASES[k])
        elif k in FILM_GATE2_CASES:
            run_film_gate2_case(k, out_root, **FILM_GATE2_CASES[k])
        elif k in DIRECT_GATE_CASES:
            run_direct_gate_case(k, out_root, **DIRECT_GATE_CASES[k])
        else:
            print(f"unknown case: {k}")
            continue

    print(f"\nDone. See {out_root.resolve()}")


if __name__ == "__main__":
    main()
