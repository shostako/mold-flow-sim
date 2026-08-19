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


def test_pp_talc_grades_are_loaded() -> None:
    """PP_T10 / PP_T20 / PP_T30 must be present in the bundled DB."""
    from core import MaterialDB

    db = MaterialDB()
    for key in ("PP_T10", "PP_T20", "PP_T30"):
        assert key in db, f"{key} missing from MaterialDB"
        m = db[key]
        assert "Talc" in m.name
        assert m.D1 > 0
        assert 0 < m.n < 1
        assert m.thermal_diffusivity_m2_s > 0
        assert m.density_melt_kgm3 > 0


def test_pp_talc_viscosity_monotone_in_filler_loading() -> None:
    """At identical T / shear, viscosity should rise monotonically with
    talc loading (PP < PP_T10 < PP_T20 < PP_T30)."""
    from core import MaterialDB, cross_wlf_viscosity

    db = MaterialDB()
    keys = ["PP", "PP_T10", "PP_T20", "PP_T30"]
    etas = [float(cross_wlf_viscosity(db[k], temperature_K=503.15, shear_rate=100.0)) for k in keys]
    for a, b in zip(etas[:-1], etas[1:], strict=True):
        assert b > a, f"viscosity should increase with talc loading: {etas}"


def test_pp_talc_thermal_diffusivity_monotone() -> None:
    """Thermal diffusivity α should rise monotonically with talc loading
    (talc has ~10× higher conductivity than PP)."""
    from core import MaterialDB

    db = MaterialDB()
    alphas = [db[k].thermal_diffusivity_m2_s for k in ["PP", "PP_T10", "PP_T20", "PP_T30"]]
    for a, b in zip(alphas[:-1], alphas[1:], strict=True):
        assert b > a, f"alpha should increase with talc loading: {alphas}"


def test_pp_talc_melt_density_monotone() -> None:
    """Melt density should rise monotonically with talc loading
    (talc 2.7 g/cc vs PP melt 0.738 g/cc)."""
    from core import MaterialDB

    db = MaterialDB()
    rhos = [db[k].density_melt_kgm3 for k in ["PP", "PP_T10", "PP_T20", "PP_T30"]]
    for a, b in zip(rhos[:-1], rhos[1:], strict=True):
        assert b > a, f"density should increase with talc loading: {rhos}"
