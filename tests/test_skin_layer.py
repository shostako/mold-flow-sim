"""Tests for the skin-layer (Stefan/Neumann) extension to ``HeleShawSolver``.

The model adds wall-side freezing as

    s(t) = c_skin * sqrt(alpha * t)

and conducts only through the live core ``h_core = h - 2*s``. The fields
are coupled by fixed-point iteration; ``T_fill`` scales with the resistance
increase (constant-pressure proxy).
"""

from __future__ import annotations

import numpy as np
import pytest

from core import HeleShawSolver, MaterialDB, build_demo_geometry


@pytest.fixture(scope="module")
def pp() -> object:  # MaterialDB returns Material instances
    return MaterialDB()["PP"]


def _solve(geom, mat, **kwargs):
    solver = HeleShawSolver(
        geometry=geom,
        material=mat,
        injection_volume_flow_cm3s=20.0,
        **kwargs,
    )
    return solver.solve(num_frames=4)


def test_skin_off_matches_legacy_metadata(pp) -> None:
    """When the skin model is disabled, the result carries no skin fields and
    the metadata flags the model as off."""
    geom = build_demo_geometry(plate_thk_mm=2.0, cell_size_mm=1.5)
    res = _solve(geom, pp)
    assert res.skin_thickness_mm is None
    assert res.core_thickness_mm is None
    assert res.short_shot_mask is None
    assert res.metadata["skin_layer_enabled"] is False
    assert "skin_iterations" not in res.metadata


def test_skin_on_emits_skin_and_core_fields(pp) -> None:
    geom = build_demo_geometry(plate_thk_mm=2.0, cell_size_mm=1.5)
    res = _solve(geom, pp, skin_layer_enabled=True, skin_growth_constant=0.3)
    assert res.skin_thickness_mm is not None
    assert res.skin_thickness_mm.shape == geom.shape
    assert res.core_thickness_mm is not None
    assert res.short_shot_mask is not None
    # Inside the cavity, h_core + 2*s == h_open  (within a tiny floor offset)
    h_open = geom.thickness_mm[geom.mask]
    h_core = res.core_thickness_mm[geom.mask]
    s = res.skin_thickness_mm[geom.mask]
    np.testing.assert_allclose(h_core + 2.0 * s, h_open, atol=2e-3)


def test_skin_increases_fill_time(pp) -> None:
    """Skin layer ON should always inflate (or match) the baseline T_fill."""
    geom = build_demo_geometry(plate_thk_mm=2.0, cell_size_mm=1.5)
    base = _solve(geom, pp)
    skin = _solve(geom, pp, skin_layer_enabled=True, skin_growth_constant=0.5)
    assert skin.total_fill_time_s > base.total_fill_time_s
    assert skin.metadata["T_fill_inflation"] > 1.0


def test_skin_zero_constant_recovers_baseline(pp) -> None:
    """c_skin = 0 must recover the no-skin solve (within solver tolerance)."""
    geom = build_demo_geometry(plate_thk_mm=2.0, cell_size_mm=1.5)
    base = _solve(geom, pp)
    skin = _solve(geom, pp, skin_layer_enabled=True, skin_growth_constant=0.0)
    np.testing.assert_allclose(
        skin.total_fill_time_s,
        base.total_fill_time_s,
        rtol=1e-6,
    )
    # Skin field is all-zero inside the cavity
    assert float(np.nanmax(skin.skin_thickness_mm[geom.mask])) <= 1e-9


def test_skin_short_shot_detected_on_thin_plate(pp) -> None:
    """A 0.4 mm plate with a strong c_skin and slow flow seals near the gate
    while the front is still far away, cutting most of the plate off: the
    seal lands in ``short_shot_mask``, the missing melt in ``unfillable_mask``."""
    geom = build_demo_geometry(plate_thk_mm=0.4, cell_size_mm=1.5)
    res = _solve(
        geom,
        pp,
        skin_layer_enabled=True,
        skin_growth_constant=1.5,
        skin_max_iterations=3,
    )
    assert res.short_shot_mask is not None and res.short_shot_mask.any()
    assert res.unfillable_mask is not None
    cavity_cells = int(geom.mask.sum())
    # Most of the plate should fail to fill in this regime.
    assert int(res.unfillable_mask.sum()) / max(cavity_cells, 1) > 0.3
    assert res.metadata["filled_volume_fraction"] < 0.7
    assert not (res.short_shot_mask & res.unfillable_mask).any()


def test_skin_metadata_contains_iteration_info(pp) -> None:
    geom = build_demo_geometry(plate_thk_mm=2.0, cell_size_mm=1.5)
    res = _solve(
        geom,
        pp,
        skin_layer_enabled=True,
        skin_growth_constant=0.3,
        skin_max_iterations=4,
    )
    md = res.metadata
    assert md["skin_layer_enabled"] is True
    assert md["skin_growth_constant"] == pytest.approx(0.3)
    assert md["thermal_diffusivity_m2_s"] == pytest.approx(pp.thermal_diffusivity_m2_s)
    assert 1 <= md["skin_iterations"] <= 4
    assert md["T_fill_baseline_s"] > 0.0
    assert md["T_fill_inflation"] >= 1.0
    assert "short_shot_cells" in md


# ---------------------------------------------------------------------------
# fill clock (v0.37.0): constant-pressure proxy vs rate-controlled machine
# ---------------------------------------------------------------------------


def _thin_plate():
    return build_demo_geometry(plate_thk_mm=0.5, cell_size_mm=1.5)


def test_the_default_clock_is_the_constant_pressure_proxy(pp) -> None:
    geom = _thin_plate()
    default = _solve(geom, pp, skin_layer_enabled=True, skin_growth_constant=1.0)
    explicit = _solve(
        geom,
        pp,
        skin_layer_enabled=True,
        skin_growth_constant=1.0,
        skin_clock_mode="constant_pressure",
    )
    assert default.metadata["skin_clock_mode"] == "constant_pressure"
    assert np.array_equal(default.fill_time_s, explicit.fill_time_s, equal_nan=True)
    assert default.total_fill_time_s == explicit.total_fill_time_s


def test_the_rate_controlled_clock_keeps_the_geometric_fill_time(pp) -> None:
    """A velocity-controlled press fills in V/Q regardless of the skin: the
    reported time is exactly the baseline, the inflation is 1, and the skin
    -- grown over a shorter exposure -- is nowhere thicker than under the
    constant-pressure proxy."""
    geom = _thin_plate()
    kw = dict(skin_layer_enabled=True, skin_growth_constant=1.0, injection_volume_flow_cm3s=3.0)
    proxy = HeleShawSolver(
        geometry=geom, material=pp, skin_clock_mode="constant_pressure", **kw
    ).solve(num_frames=4)
    rate = HeleShawSolver(geometry=geom, material=pp, skin_clock_mode="constant_rate", **kw).solve(
        num_frames=4
    )
    assert proxy.metadata["T_fill_inflation"] > 1.05  # the proxy actually inflates here
    assert rate.metadata["skin_clock_mode"] == "constant_rate"
    assert rate.metadata["T_fill_inflation"] == 1.0
    assert rate.total_fill_time_s == rate.metadata["T_fill_baseline_s"]
    assert rate.total_fill_time_s < proxy.total_fill_time_s
    # The gate group ages for the whole fill under either clock, so its skin
    # reads the clock directly: shorter clock, thinner skin. (Cell by cell
    # the two runs are not ordered -- their fill orders differ.)
    iy, ix = geom.gates[0]
    assert 0.0 < rate.skin_thickness_mm[iy, ix] < proxy.skin_thickness_mm[iy, ix]


def test_an_unknown_clock_mode_is_rejected(pp) -> None:
    geom = _thin_plate()
    with pytest.raises(ValueError, match="skin_clock_mode"):
        _solve(geom, pp, skin_layer_enabled=True, skin_clock_mode="constant_volume")


def test_a_clock_end_stops_the_exposure(pp) -> None:
    """``_solve_domain(clock_end_s=T)``: walls age until T, cells the front
    reaches after T carry no skin, and the gate cell (arrival 0) carries
    the service-mean skin of a T exposure."""
    geom = _thin_plate()
    c = 0.8
    solver = HeleShawSolver(
        geometry=geom,
        material=pp,
        injection_volume_flow_cm3s=3.0,
        skin_layer_enabled=True,
        skin_growth_constant=c,
        skin_max_iterations=40,
        skin_convergence_tol=1e-10,
    )
    eta = solver._effective_viscosity()
    T_total = geom.volume_cm3() / 3.0
    T_end = 0.4 * T_total
    sol = solver._solve_domain(eta, T_fill_baseline_s=T_total, clock_end_s=T_end)
    assert sol.T_fill == T_total  # a clock end never inflates
    skin = sol.skin_thk_mm
    t_arr = sol.t_arr
    assert skin is not None and t_arr is not None
    late = geom.mask & (t_arr >= T_end)
    early = geom.mask & (t_arr < T_end)
    assert late.any() and early.any()
    assert np.all(skin[late] == 0.0)
    assert np.all(skin[early] > 0.0)
    iy, ix = geom.gates[0]
    t_gate = t_arr[iy, ix]  # group-end arrival of the tau = 0 tie group
    assert 0.0 <= t_gate < T_end
    expected = (2.0 / 3.0) * c * np.sqrt(pp.thermal_diffusivity_m2_s * (T_end - t_gate)) * 1e3
    assert skin[iy, ix] == pytest.approx(expected, rel=1e-6)
    # same solver, no clock end: every cell the front passes before the fill
    # ends ages (only the last tie group, arriving at T_total, does not)
    full = solver._solve_domain(eta, T_fill_baseline_s=T_total)
    assert np.all(full.skin_thk_mm[geom.mask & (full.t_arr < T_total)] > 0.0)
    assert (full.skin_thk_mm[geom.mask] > 0).sum() > (skin[geom.mask] > 0).sum()
