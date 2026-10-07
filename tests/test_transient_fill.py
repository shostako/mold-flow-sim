"""Time-marching fill (v0.62.0): ``core.transient_fill.march_fill`` and ``fill_method="march"``.

The marching core is checked against closed forms (a strip fills at the
cumulative volume over the rate, a staged injection is followed exactly, a disc
grows with its area) and for volume conservation. The solver option is checked
for what it was added for -- a deep channel no longer fills before the thin
plate next to the gate -- and for leaving the default (``"tau"``) untouched.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.geometry import Geometry
from core.materials import MaterialDB
from core.multilayer_solver import MultilayerHeleShawSolver
from core.transient_fill import march_fill
from core.two_phase import solve_two_phase_short_shot

MM3 = 1e-9  # m^3


def _const(S):
    return lambda idx, t_arr: np.full(idx.size, S)


def test_a_strip_fills_at_the_cumulative_volume_over_the_rate():
    """One front cell at a time: cell k is full when every cell up to it holds
    its melt, at ``Σ_{j≤k} C_j / Q`` -- exactly, uneven capacities and all."""
    rng = np.random.default_rng(7)
    nx = 30
    mask = np.zeros((3, nx), dtype=bool)
    mask[1] = True
    cap = np.zeros((3, nx))
    cap[1] = rng.uniform(0.5, 2.0, nx) * MM3
    Q = 3.0 * MM3
    t0 = cap[1, 0] / Q
    res = march_fill(
        mask, cap, [(1, 0)], lambda t: Q * t, _const(1e-12), t_start_s=t0, dt_max_s=1.0
    )
    assert res.complete
    np.testing.assert_allclose(res.t_arr_s[1], np.cumsum(cap[1]) / Q, rtol=1e-9)
    assert np.isnan(res.t_arr_s[0]).all() and np.isnan(res.t_arr_s[2]).all()
    assert res.volume_held_m3 == pytest.approx(res.volume_injected_m3, rel=1e-12)


def test_a_staged_injection_is_followed_through_its_volume_map():
    """The march takes in the increments of the delivered volume, so a slow
    first stage stretches the early arrivals by its own factor: in a strip of
    equal cells, cell k is full at the inverse of the volume map at k + 1 cells."""
    nx = 40
    mask = np.zeros((1, nx), dtype=bool)
    mask[0] = True
    cap = np.full((1, nx), 1.0 * MM3)
    q1, q2, v_switch = (
        2.0 * MM3,
        10.0 * MM3,
        15.0 * MM3,
    )  # 2 cells/s up to 15 cells, then 10 cells/s

    def vol(t):
        t_sw = v_switch / q1
        return q1 * t if t <= t_sw else v_switch + q2 * (t - t_sw)

    def t_of(v):
        return v / q1 if v <= v_switch else v_switch / q1 + (v - v_switch) / q2

    res = march_fill(
        mask,
        cap,
        [(0, 0)],
        vol,
        _const(1e-12),
        t_start_s=t_of(1.0 * MM3),
        dt_max_s=0.5,
        breakpoints_s=(v_switch / q1,),
    )
    expect = np.array([t_of((k + 1) * MM3) for k in range(nx)])
    np.testing.assert_allclose(res.t_arr_s[0], expect, rtol=1e-9, atol=1e-12)


def test_a_disc_grows_with_the_injected_area():
    """A point gate in a uniform plate: the area full by time t is about Q t, so
    the arrival at radius r is about π r² / Q, the same along the axes and the
    diagonals (the front stays round on the 5-point stencil)."""
    n = 81
    mask = np.ones((n, n), dtype=bool)
    cap = np.full((n, n), 1.0 * MM3)
    Q = 100.0 * MM3
    res = march_fill(
        mask, cap, [(40, 40)], lambda t: Q * t, _const(1e-12), t_start_s=0.01, dt_max_s=1.0
    )
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(yy - 40, xx - 40)
    sel = (r > 5) & (r < 38)
    ratio = res.t_arr_s[sel] / (np.pi * r[sel] ** 2 / (Q / MM3))
    # a cell counts as full only once the front has passed its far edge, so the
    # stamp runs a few percent behind π r² (the partly filled ring holds the rest)
    assert 1.0 < ratio.mean() < 1.08 and ratio.std() < 0.05
    axis, diag = res.t_arr_s[40, 60], res.t_arr_s[54, 54]  # both at r ≈ 20
    assert abs(axis - diag) / axis < 0.05


def test_the_volume_is_conserved_around_holes_and_uneven_cells():
    rng = np.random.default_rng(3)
    ny, nx = 25, 40
    mask = np.ones((ny, nx), dtype=bool)
    mask[8:14, 10:16] = False  # a hole the front has to go round
    mask[3:6, 25:35] = False
    cap = np.where(mask, rng.uniform(0.3, 3.0, (ny, nx)), 0.0) * MM3
    S = rng.uniform(0.5, 5.0, ny * nx) * 1e-12
    res = march_fill(
        mask,
        cap,
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        lambda t: 50.0 * MM3 * t,
        lambda idx, t_arr: S[idx],
        t_start_s=float(cap[:2, :2].sum() / (50.0 * MM3)),
        dt_max_s=0.5,
    )
    assert res.complete
    assert np.isfinite(res.t_arr_s[mask]).all() and np.isnan(res.t_arr_s[~mask]).all()
    assert res.volume_held_m3 == pytest.approx(res.volume_injected_m3, rel=1e-12)
    # the last delivering step is cut to the room left, and after it the march only moves
    # the surplus on: the cavity ends holding its capacity, not a step's worth more
    assert res.volume_held_m3 == pytest.approx(cap.sum(), rel=1e-9)


def test_the_gate_cells_share_one_pressure():
    """The τ solvers hold every gate cell at τ = 0; the march holds them at one
    pressure, chosen so the melt leaving them is what the machine delivered."""
    n = 31
    mask = np.ones((n, n), dtype=bool)
    cap = np.full((n, n), 1.0 * MM3)
    gates = [(15, 14), (15, 15), (15, 16), (14, 15), (16, 15)]
    res = march_fill(
        mask, cap, gates, lambda t: 40.0 * MM3 * t, _const(1e-12), t_start_s=5 / 40.0, dt_max_s=0.5
    )
    p = np.array([res.pressure_end_Pa[g] for g in gates])
    assert p.max() > 0
    np.testing.assert_allclose(p, p[0], rtol=1e-9)
    assert res.pressure_end_Pa.max() == pytest.approx(p[0])


def test_march_rejects_bad_arguments():
    mask = np.ones((3, 3), dtype=bool)
    cap = np.full((3, 3), MM3)
    with pytest.raises(ValueError):
        march_fill(
            mask,
            cap,
            [(1, 1)],
            lambda t: MM3 * t,
            _const(1e-12),
            t_start_s=1.0,
            dt_max_s=0.1,
            cfl=1.5,
        )
    with pytest.raises(ValueError):
        march_fill(
            mask, cap, [(1, 1)], lambda t: MM3 * t, _const(1e-12), t_start_s=1.0, dt_max_s=0.0
        )
    with pytest.raises(ValueError):
        march_fill(
            mask,
            np.zeros((3, 3)),
            [(1, 1)],
            lambda t: MM3 * t,
            _const(1e-12),
            t_start_s=1.0,
            dt_max_s=0.1,
        )


# ---------------------------------------------------------------------------
# fill_method on MultilayerHeleShawSolver
# ---------------------------------------------------------------------------


def _channel_and_plate() -> Geometry:
    """A deep channel along the gate side (2.0 mm, 4 rows) under a thin plate (0.3 mm),
    gated in the middle of the channel -- a gate block's runner and the product."""
    ny, nx = 24, 61
    mask = np.ones((ny, nx), dtype=bool)
    thk = np.full((ny, nx), 0.3)
    thk[:4] = 2.0
    return Geometry(mask=mask, thickness_mm=thk, cell_size_mm=1.0, gates=[(0, 30)])


def _solver(geom, fill_method="tau", **kw) -> MultilayerHeleShawSolver:
    args = dict(
        geometry=geom,
        material=MaterialDB().get("PP"),
        injection_volume_flow_cm3s=1.0,
        num_layers=1,
        thermal_coupling=False,
    )
    args.update(kw)
    return MultilayerHeleShawSolver(**args, fill_method=fill_method)


def test_the_march_does_not_fill_the_deep_channel_before_the_plate():
    """What the option is for. The τ fill orders the cavity with one solve in
    which every cell is a sink, and the deep channel -- a lot of sink behind
    little resistance -- comes out full before the plate next to the gate even
    starts (the FG9 VP photos show a 64 %-full block with the product centre
    already ahead). The march fills the plate above the gate long before the
    channel's far ends."""
    geom = _channel_and_plate()
    f_tau = _solver(geom, "tau").solve(num_frames=4).fill_time_s
    f_m = _solver(geom, "march").solve(num_frames=4).fill_time_s
    ends_tau, plate_tau = max(f_tau[1, 0], f_tau[1, 60]), f_tau[4, 30]
    ends_m, plate_m = min(f_m[1, 0], f_m[1, 60]), f_m[4, 30]
    assert plate_tau > ends_tau  # τ: the channel first
    assert plate_m < 0.5 * ends_m  # march: the plate long before the channel ends
    # the end-of-fill pressure map: finite, non-negative, highest at the gate
    p = _solver(geom, "march").solve(num_frames=4).pressure_norm
    assert np.isfinite(p).all() and (p >= 0).all() and p[0, 30] == pytest.approx(1.0)


def test_the_march_on_a_strip_matches_the_tau_fill():
    """Where the order is not in question (a strip from one end), the two
    methods agree: the march's own arrival times are the τ fill's volume-CDF
    times on the same clock (the gate cell's tie-group stamp aside)."""
    mask = np.zeros((3, 40), dtype=bool)
    mask[1] = True
    thk = np.where(mask, np.linspace(0.3, 0.6, 40)[None, :], 0.0)
    geom = Geometry(mask=mask, thickness_mm=thk, cell_size_mm=1.0, gates=[(1, 0)])
    f_tau = _solver(geom, "tau").solve(num_frames=4).fill_time_s
    r_m = _solver(geom, "march").solve(num_frames=4)
    np.testing.assert_allclose(r_m.fill_time_s[1], f_tau[1], rtol=1e-9)
    assert r_m.metadata["fill_method"] == "march" and r_m.metadata["march_steps"] > 0
    # the pressure map is the march's own pressure at the end of the fill: highest at the gate,
    # falling along the strip to the front -- the injection's pressure, not the surplus push
    p = r_m.pressure_norm[1]
    assert p[0] == pytest.approx(1.0) and p[-1] == pytest.approx(0.0)
    assert np.all(np.diff(p) <= 1e-12)


def test_the_layered_march_reads_the_layers_on_its_arrival_times():
    """With the thermal coupling on, each cell's layers are read at the time the
    march filled it: the reported temperatures are the Neumann profile (plus the
    shear-heating rise) of the reported fill times, the clock is the machine's
    (no inflation), and the far end -- reached last -- is the coldest at the wall."""
    mask = np.zeros((5, 60), dtype=bool)
    mask[1:4] = True
    thk = np.where(mask, 0.5, 0.0)
    geom = Geometry(mask=mask, thickness_mm=thk, cell_size_mm=1.0, gates=[(2, 0)])
    s = _solver(
        geom,
        "march",
        num_layers=5,
        layer_distribution="wall_refined",
        thermal_coupling=True,
        shear_heating_enabled=True,
        injection_volume_flow_cm3s=0.05,
        material=MaterialDB().get("PP_T20"),
    )
    r = s.solve(num_frames=4)
    assert r.layer_temperature_K is not None and np.isfinite(r.layer_temperature_K[:, mask]).all()
    assert r.metadata["T_fill_inflation"] == pytest.approx(1.0, abs=1e-6)
    f = r.fill_time_s[2]
    assert np.all(np.diff(f) > 0)
    T_wall = r.layer_temperature_K[0, 2]
    assert T_wall[-1] < T_wall[5]
    # the same reading the solver applies to any arrival field
    zc = s.layer_zeta_centers()
    from core.multilayer_thermal import poiseuille_shear_rates

    gam = poiseuille_shear_rates(
        zeta_centers=zc,
        V_mms=s.injection_velocity_mms,
        h_total_mm=s._base._open_thickness_field(),
        floor_factor=s.shear_rate_floor_factor,
    )
    T_re, _ = s._layer_temperatures(
        np.where(np.isnan(r.fill_time_s), 0.0, r.fill_time_s),
        s._base._open_thickness_field(),
        zc,
        s.material.thermal_diffusivity_m2_s,
        gam,
    )
    np.testing.assert_allclose(r.layer_temperature_K[:, mask], T_re[:, mask], rtol=1e-12)


def test_the_default_is_the_tau_fill_and_unknown_methods_are_rejected():
    geom = _channel_and_plate()
    a = _solver(geom).solve(num_frames=4)
    b = MultilayerHeleShawSolver(
        geometry=geom,
        material=MaterialDB().get("PP"),
        injection_volume_flow_cm3s=1.0,
        num_layers=1,
        thermal_coupling=False,
    ).solve(num_frames=4)
    assert a.metadata["fill_method"] == "tau" and "march_steps" not in a.metadata
    np.testing.assert_array_equal(np.nan_to_num(a.fill_time_s), np.nan_to_num(b.fill_time_s))
    with pytest.raises(ValueError):
        _solver(geom, "marching")
    with pytest.raises(ValueError):
        _solver(geom, "march", march_dt_max_fraction=0.0)


def test_the_two_phase_injection_rides_on_one_march(monkeypatch):
    """The two-phase injection phase asks for the same march at every metered
    volume; it runs once. The pool is the volume prefix of the march's order,
    nested in the shot, and the metadata says how it was solved."""
    import core.multilayer_solver as ms

    geom = _channel_and_plate()
    s = _solver(geom, "march")
    calls = {"n": 0}
    real = ms.march_fill

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(ms, "march_fill", counting)
    V = geom.volume_cm3()
    pools = []
    for frac in (0.3, 0.5, 0.8):
        tp = solve_two_phase_short_shot(s, V * frac)
        pools.append(tp.injection_mask)
        assert tp.metadata["fill_method"] == "march"
    assert calls["n"] == 1
    assert (pools[0] <= pools[1]).all() and (pools[1] <= pools[2]).all()
    # at 30 % of the cavity (the channel alone is 57 %) the march is already in the plate
    # above the gate; the τ fill has not left the channel
    tau_pool = solve_two_phase_short_shot(_solver(geom, "tau"), V * 0.3).injection_mask
    assert pools[0][4:6, 30].all() and not tau_pool[4:, :].any()


def test_a_staged_profile_reaches_the_marcher_with_its_switches():
    """With a multi-stage injection profile the solver hands the stage switches,
    on its clock, to the marcher, so no step straddles a rate jump (Codex P2 on
    PR #114). On a strip the march then lands on the τ fill's profile-mapped
    volume-CDF times exactly, slow first stage and all."""
    from core.injection_profile import InjectionProfile, InjectionStage

    mask = np.zeros((3, 50), dtype=bool)
    mask[1] = True
    thk = np.where(mask, 0.5, 0.0)
    geom = Geometry(mask=mask, thickness_mm=thk, cell_size_mm=1.0, gates=[(1, 0)])
    # 10 mm screw: 78.5 mm³ per mm of stroke. 25 mm³ of strip; the switch at 0.1 mm of
    # stroke (7.9 mm³, a third of the strip) from 1 mm/s to 10 mm/s
    prof = InjectionProfile(
        screw_diameter_mm=10.0,
        metering_position_mm=10.0,
        stages=(InjectionStage(9.9, 1.0), InjectionStage(5.0, 10.0)),
    )
    import core.multilayer_solver as ms

    f_tau = _solver(geom, "tau", injection_profile=prof).solve(num_frames=4).fill_time_s[1]
    # the slow stage is visible: a third of the strip takes most of the time
    assert f_tau[15] > 0.7 * f_tau[-1]
    # a cell of the fast stage fills in C / Q_fast; a march is good to about one cell-fill time.
    # A long step cap (a fifth of the fill) lets a step reach across the switch if it may
    t_cell_fast = 0.5 / (prof.area_mm2 * 10.0)
    r_m = _solver(geom, "march", injection_profile=prof, march_dt_max_fraction=0.2).solve(
        num_frames=4
    )
    assert np.max(np.abs(r_m.fill_time_s[1] - f_tau)) < t_cell_fast
    # without the switches the step averages the two rates and the strip comes in several cells late
    real = ms.march_fill

    def no_switches(*a, **k):
        k.pop("breakpoints_s", None)
        return real(*a, **k)

    ms.march_fill = no_switches
    try:
        r_nb = _solver(geom, "march", injection_profile=prof, march_dt_max_fraction=0.2).solve(
            num_frames=4
        )
    finally:
        ms.march_fill = real
    assert np.max(np.abs(r_nb.fill_time_s[1] - f_tau)) > 3 * t_cell_fast
