"""CLI demo: run a few simulations with parameter sweeps and write outputs.

Usage:
    python run_demo.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from core import (
    HeleShawSolver,
    MaterialDB,
    build_demo_geometry,
    export_frames,
    render_fill_animation,
    render_pressure_map,
    render_weldlines,
)


def run_case(
    label: str,
    out_root: Path,
    material_key: str,
    melt_K: float,
    mold_K: float,
    inj_velocity_mms: float,
    inj_Q_cm3s: float,
    compression: bool = False,
    compression_factor: float = 1.5,
    compression_fraction: float = 0.6,
    cell_size_mm: float = 1.0,
    plate_thk_mm: float = 2.0,
    gate_count: int = 1,
    num_frames: int = 30,
) -> None:
    db = MaterialDB()
    geom = build_demo_geometry(
        cell_size_mm=cell_size_mm,
        plate_thk_mm=plate_thk_mm,
        gate_count=gate_count,
    )
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
    )
    print(f"[{label}] solving... cells={int(geom.mask.sum())} V={geom.volume_cm3():.2f} cm^3")
    result = solver.solve(num_frames=num_frames)
    print(
        f"[{label}] T_fill={result.total_fill_time_s:.3f} s  eta_eff={result.viscosity_Pa_s:.1f} Pa.s"
    )

    out_dir = out_root / label
    out_dir.mkdir(parents=True, exist_ok=True)

    render_fill_animation(result, out_dir / "fill.gif", num_frames=num_frames, fps=8)
    render_pressure_map(result, out_dir / "pressure.png")
    render_weldlines(result, out_dir / "weld_airtraps.png")
    export_frames(result, out_dir / "frames", num_frames=8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs", help="output root directory")
    parser.add_argument("--cases", nargs="*", default=None, help="case keys to run (default: all)")
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    cases = {
        "PP_baseline": dict(
            material_key="PP", melt_K=503.15, mold_K=313.15, inj_velocity_mms=100.0, inj_Q_cm3s=20.0
        ),
        "PP_hot_melt": dict(
            material_key="PP", melt_K=533.15, mold_K=323.15, inj_velocity_mms=100.0, inj_Q_cm3s=20.0
        ),
        "PP_slow_inj": dict(
            material_key="PP", melt_K=503.15, mold_K=313.15, inj_velocity_mms=30.0, inj_Q_cm3s=6.0
        ),
        "PC_baseline": dict(
            material_key="PC", melt_K=583.15, mold_K=373.15, inj_velocity_mms=80.0, inj_Q_cm3s=15.0
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
    }

    keys = args.cases or list(cases.keys())
    for k in keys:
        if k not in cases:
            print(f"unknown case: {k}")
            continue
        run_case(k, out_root, **cases[k])

    print(f"\nDone. See {out_root.resolve()}")


if __name__ == "__main__":
    main()
