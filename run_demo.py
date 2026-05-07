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
    FilmGateConfig,
    Geometry,
    HeleShawSolver,
    MaterialDB,
    build_demo_geometry,
    build_direct_gate_geometry,
    build_film_gate_geometry,
    export_frames,
    render_core_layer_map,
    render_fill_animation,
    render_pressure_map,
    render_skin_layer_map,
    render_weldlines,
)


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
    compression_fraction: float = 0.6,
    skin_layer: bool = False,
    skin_growth_constant: float = 0.5,
    skin_max_iterations: int = 5,
    skin_convergence_tol: float = 1e-3,
    num_frames: int = 30,
) -> None:
    db = MaterialDB()
    solver = HeleShawSolver(
        geometry=geom,
        material=db[material_key],
        melt_temperature_K=melt_K,
        mold_temperature_K=mold_K,
        injection_velocity_mms=inj_velocity_mms,
        injection_volume_flow_cm3s=inj_Q_cm3s,
        compression_molding=compression,
        compression_factor=compression_factor,
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
}


# ---------------------- entrypoint ----------------------


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
        list(DEMO_CASES.keys()) + list(FILM_GATE_CASES.keys()) + list(DIRECT_GATE_CASES.keys())
    )
    keys = args.cases or all_keys
    for k in keys:
        if k in DEMO_CASES:
            run_demo_case(k, out_root, **DEMO_CASES[k])
        elif k in FILM_GATE_CASES:
            run_film_gate_case(k, out_root, **FILM_GATE_CASES[k])
        elif k in DIRECT_GATE_CASES:
            run_direct_gate_case(k, out_root, **DIRECT_GATE_CASES[k])
        else:
            print(f"unknown case: {k}")
            continue

    print(f"\nDone. See {out_root.resolve()}")


if __name__ == "__main__":
    main()
