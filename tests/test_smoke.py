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


def _direct_cross_wlf(m, T, g):
    """The Cross-WLF expression evaluated as written (the pre-fix form)."""
    import numpy as np

    dT = T - m.D2
    denom = np.where(m.A2_tilde + dT <= 1e-6, 1e-6, m.A2_tilde + dT)
    eta0 = m.D1 * np.exp(-m.A1 * dT / denom)
    ratio = eta0 * np.where(g <= 1e-12, 1e-12, g) / m.tau_star
    return eta0 / (1.0 + ratio ** (1.0 - m.n))


def test_cross_wlf_is_frozen_not_nan_or_zero_below_its_range() -> None:
    """Codex P1 on PR #111: near and below D2 − A2 (PA66: 164 °C, above its
    mold temperatures) the direct expression overflows to NaN (inf/inf) or
    0 (finite/inf). The viscosity must instead stay positive and never fall
    as T drops -- a large finite value, or inf where the melt is frozen --
    and match the direct expression wherever that does not overflow."""
    import warnings

    import numpy as np

    from core import MaterialDB, cross_wlf_viscosity

    db = MaterialDB()
    pa = db["PA66"]
    T = np.linspace(pa.D2 - pa.A2_tilde - 30.0, max(pa.T_melt_recommended) + 30.0, 2001)
    for g in (1e-3, 1.0, 1e2, 1e4, 1e6):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            eta = cross_wlf_viscosity(pa, T, np.full_like(T, g))
        assert not np.isnan(eta).any()
        assert np.all(eta > 0.0)
        assert np.isinf(eta[0])  # well below D2 − A2: frozen
        assert np.all(eta[1:] <= eta[:-1])  # never falls as T drops
        with np.errstate(all="ignore"):
            direct = _direct_cross_wlf(pa, T, np.full_like(T, g))
        assert np.isnan(direct).any()  # the case reaches the broken range
        plain = np.isfinite(direct) & (direct > 0) & (direct < 1e300)
        np.testing.assert_allclose(eta[plain], direct[plain], rtol=1e-12)
    # the ordinary range is the direct expression bit for bit
    for key in db.keys():
        m = db[key]
        Tm = np.linspace(min(m.T_melt_recommended) - 40.0, max(m.T_melt_recommended) + 40.0, 401)
        gm = np.geomspace(1e-2, 1e6, 401)
        np.testing.assert_array_equal(cross_wlf_viscosity(m, Tm, gm), _direct_cross_wlf(m, Tm, gm))
    assert isinstance(cross_wlf_viscosity(pa, 300.0, 100.0), np.floating)


def test_cross_wlf_does_not_return_zero_where_only_eta0_times_shear_overflows() -> None:
    """@claude review on PR #111: eta0 itself may still fit while eta0·gamma_dot
    (formed before dividing by tau*) overflows. With tau* ≥ 1.8e4 Pa that
    happens below the ratio threshold, and the direct form then returns
    eta0/inf = 0 -- the frozen-as-fluid defect again. Pick the temperature
    where ln eta0 = 699 and shear at 1e5 1/s, for every resin."""
    import numpy as np

    from core import MaterialDB, cross_wlf_viscosity

    db = MaterialDB()
    zeros_direct = 0
    for key in db.keys():
        m = db[key]
        c = (np.log(m.D1) - 699.0) / m.A1  # dT/(A2 + dT) giving ln eta0 = 699
        T = m.D2 + c * m.A2_tilde / (1.0 - c)
        g = 1e5
        with np.errstate(all="ignore"):
            zeros_direct += int(_direct_cross_wlf(m, np.array([T]), np.array([g]))[0] == 0.0)
        eta = float(cross_wlf_viscosity(m, T, g))
        assert np.isfinite(eta) and eta > 0.0, key
        # never below the melt just above it
        assert eta >= float(cross_wlf_viscosity(m, T + 0.01, g)), key
    assert zeros_direct >= 1  # the gap is real for some resin (ABS, PC, PA66)


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
