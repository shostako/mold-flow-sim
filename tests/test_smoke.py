"""Smoke tests: import-level sanity checks for the core package.

These tests do not solve the Hele-Shaw system; they only verify that
the package imports cleanly and the bundled material data is loadable.
"""

from __future__ import annotations


def test_core_imports() -> None:
    from core import (  # noqa: F401
        FlowResult,
        Geometry,
        HeleShawSolver,
        MaterialDB,
        build_demo_geometry,
        cross_wlf_viscosity,
        export_frames,
        geometry_from_image,
        render_fill_animation,
        render_pressure_map,
        render_weldlines,
    )


def test_material_db_loads_bundled_json() -> None:
    from core import MaterialDB

    db = MaterialDB()
    assert "PP" in db
    pp = db["PP"]
    assert pp.name == "Polypropylene (generic)"
    assert 0 < pp.n < 1
    assert pp.D1 > 0
    assert pp.tau_star > 0


def test_demo_geometry_is_well_formed() -> None:
    from core import build_demo_geometry

    g = build_demo_geometry()
    assert g.mask.any(), "demo geometry must contain cavity cells"
    assert g.gates, "demo geometry must define at least one gate"
    assert g.volume_cm3() > 0
    iy, ix = g.gates[0]
    assert g.mask[iy, ix], "gate must lie inside the cavity mask"


def test_cross_wlf_viscosity_monotone_in_temperature() -> None:
    """At fixed shear rate, η decreases as T increases (basic sanity)."""
    from core import MaterialDB, cross_wlf_viscosity

    pp = MaterialDB()["PP"]
    eta_low = float(cross_wlf_viscosity(pp, temperature_K=453.15, shear_rate=100.0))
    eta_high = float(cross_wlf_viscosity(pp, temperature_K=533.15, shear_rate=100.0))
    assert eta_low > eta_high > 0
