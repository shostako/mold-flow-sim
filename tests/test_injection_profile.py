"""Tests for the screw-side injection profile and its effect on the time axis.

The point of the feature is that the time axis stops being one number. Two
things therefore have to hold at once: a single-stage profile must reproduce
the constant-rate behaviour it replaces (nothing existing may move), and a
staged profile must actually bend the arrival times -- otherwise the stages
are decoration. The bending is checked against arithmetic that does not go
through the solver: on a uniform strip the volume CDF is exactly the cell
index, so the arrival time of cell k is the profile's own time at that
volume.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from core import HeleShawSolver, MaterialDB
from core.geometry import Geometry
from core.injection_profile import (
    InjectionProfile,
    InjectionStage,
    flow_rate_cm3s,
    screw_area_mm2,
)
from core.multilayer_solver import MultilayerHeleShawSolver
from core.two_phase import solve_two_phase_short_shot


def _strip(n: int = 20, thickness_mm: float = 1.0, cell_mm: float = 1.0) -> Geometry:
    """One-row cavity of equal cells, gate at the left.

    Equal cells make the volume CDF the cell index, so the expected arrival
    times are the profile's map evaluated at ``(k+1) * cell_volume``.
    """
    geom = Geometry(
        mask=np.ones((1, n), dtype=bool),
        thickness_mm=np.full((1, n), float(thickness_mm)),
        cell_size_mm=cell_mm,
    )
    geom.gates = [(0, 0)]
    return geom


def _single(v_mms: float = 100.0) -> InjectionProfile:
    return InjectionProfile(40.0, 40.0, (InjectionStage(5.0, v_mms),))


def _staged(v1: float = 20.0, v2: float = 200.0) -> InjectionProfile:
    """Slow first half of the stroke, fast second half."""
    return InjectionProfile(40.0, 40.0, (InjectionStage(22.5, v1), InjectionStage(5.0, v2)))


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------


def test_screw_area_is_the_circle():
    assert screw_area_mm2(40.0) == pytest.approx(math.pi * 400.0)


def test_flow_rate_is_area_times_speed_in_cm3():
    # 40 mm screw at 100 mm/s: 1256.6 mm^2 * 100 mm/s = 125.66 cm^3/s
    assert flow_rate_cm3s(40.0, 100.0) == pytest.approx(math.pi * 400.0 * 100.0 / 1000.0)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_screw_area_rejects_non_positive_or_non_finite(bad):
    with pytest.raises(ValueError):
        screw_area_mm2(bad)


@pytest.mark.parametrize("bad", [0.0, -5.0, float("nan")])
def test_flow_rate_rejects_bad_velocity(bad):
    with pytest.raises(ValueError):
        flow_rate_cm3s(40.0, bad)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_rejects_no_stages():
    with pytest.raises(ValueError, match="at least one"):
        InjectionProfile(40.0, 40.0, ())


@pytest.mark.parametrize("bad", [0.0, -40.0, float("nan")])
def test_rejects_bad_screw_diameter(bad):
    with pytest.raises(ValueError, match="screw_diameter_mm"):
        InjectionProfile(bad, 40.0, (InjectionStage(5.0, 100.0),))


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf")])
def test_rejects_bad_metering_position(bad):
    with pytest.raises(ValueError, match="metering_position_mm"):
        InjectionProfile(40.0, bad, (InjectionStage(-5.0, 100.0),))


def test_rejects_vp_above_metering_position():
    with pytest.raises(ValueError, match="must be below metering_position_mm"):
        InjectionProfile(40.0, 40.0, (InjectionStage(50.0, 100.0),))


def test_rejects_equal_positions():
    """A zero-length stage injects nothing in no time -- a vertical segment."""
    with pytest.raises(ValueError, match="must be below"):
        InjectionProfile(40.0, 40.0, (InjectionStage(40.0, 100.0),))


def test_rejects_non_monotone_switch_positions():
    with pytest.raises(ValueError, match=r"stages\[1\]"):
        InjectionProfile(40.0, 40.0, (InjectionStage(20.0, 50.0), InjectionStage(25.0, 50.0)))


def test_rejects_negative_end_position():
    with pytest.raises(ValueError, match="finite and >= 0"):
        InjectionProfile(40.0, 40.0, (InjectionStage(-1.0, 100.0),))


@pytest.mark.parametrize("bad", [0.0, -10.0, float("nan")])
def test_rejects_bad_stage_velocity(bad):
    with pytest.raises(ValueError, match="velocity_mms"):
        InjectionProfile(40.0, 40.0, (InjectionStage(5.0, bad),))


# ---------------------------------------------------------------------------
# derived quantities
# ---------------------------------------------------------------------------


def test_single_stage_derived_quantities():
    p = _single(100.0)
    a = math.pi * 400.0
    assert p.num_stages == 1
    assert p.vp_position_mm == 5.0
    assert p.stage_lengths_mm() == (35.0,)
    assert p.stage_rates_cm3s()[0] == pytest.approx(a * 100.0 / 1000.0)
    assert p.stage_times_s()[0] == pytest.approx(0.35)
    assert p.total_volume_mm3 == pytest.approx(a * 35.0)
    assert p.total_time_s == pytest.approx(0.35)
    assert p.mean_rate_cm3s == pytest.approx(p.stage_rates_cm3s()[0])


def test_stage_volume_is_set_by_travel_not_by_speed():
    """The physical fact the whole map rests on: volume breakpoints are fixed.

    Stage lengths are deliberately unequal -- with equal lengths swapping the
    two speeds gives the same total time and the second assertion is vacuous.
    """
    slow = InjectionProfile(40.0, 40.0, (InjectionStage(30.0, 10.0), InjectionStage(5.0, 400.0)))
    fast = InjectionProfile(40.0, 40.0, (InjectionStage(30.0, 400.0), InjectionStage(5.0, 10.0)))
    assert slow.stage_volumes_mm3() == pytest.approx(fast.stage_volumes_mm3())
    assert slow.total_time_s != pytest.approx(fast.total_time_s)


def test_stage_starts_chain_from_metering_position():
    p = InjectionProfile(
        30.0,
        50.0,
        (InjectionStage(30.0, 10.0), InjectionStage(12.0, 20.0), InjectionStage(4.0, 40.0)),
    )
    assert p.stage_starts_mm() == (50.0, 30.0, 12.0)
    assert p.stage_lengths_mm() == (20.0, 18.0, 8.0)
    assert p.total_volume_mm3 == pytest.approx(sum(p.stage_volumes_mm3()))
    assert p.total_time_s == pytest.approx(20 / 10 + 18 / 20 + 8 / 40)


def test_mean_rate_is_between_the_stage_rates():
    p = _staged()
    rates = p.stage_rates_cm3s()
    assert min(rates) < p.mean_rate_cm3s < max(rates)


# ---------------------------------------------------------------------------
# the volume -> time map
# ---------------------------------------------------------------------------


def test_map_endpoints():
    p = _staged()
    assert p.time_at_volume_mm3(0.0) == 0.0
    assert p.time_at_volume_mm3(p.total_volume_mm3) == pytest.approx(p.total_time_s)


def test_single_stage_map_is_v_over_q():
    p = _single(100.0)
    Q_mm3s = p.stage_rates_cm3s()[0] * 1000.0
    for v in (0.0, 1234.5, p.total_volume_mm3 * 0.7):
        assert p.time_at_volume_mm3(v) == pytest.approx(v / Q_mm3s)


def test_map_kinks_at_the_switch_position():
    """Half the stroke at a tenth of the speed takes ten times as long."""
    p = _staged(20.0, 200.0)
    v_switch = p.stage_volumes_mm3()[0]
    t_switch = p.time_at_volume_mm3(v_switch)
    assert t_switch == pytest.approx(p.stage_times_s()[0])
    # First half of the volume, but 10/11 of the time.
    assert t_switch / p.total_time_s == pytest.approx(10.0 / 11.0)


def test_map_extrapolates_past_vp_at_the_last_rate():
    p = _staged()
    over = p.total_volume_mm3 * 1.5
    extra = (over - p.total_volume_mm3) / (p.stage_rates_cm3s()[-1] * 1000.0)
    assert p.time_at_volume_mm3(over) == pytest.approx(p.total_time_s + extra)


def test_map_clamps_negative_volume_to_zero():
    assert _staged().time_at_volume_mm3(-100.0) == 0.0


def test_map_is_array_capable_and_monotone():
    p = _staged()
    v = np.linspace(0.0, p.total_volume_mm3 * 1.2, 50)
    t = p.time_at_volume_mm3(v)
    assert t.shape == v.shape
    assert np.all(np.diff(t) > 0)


def test_scalar_in_scalar_out():
    p = _staged()
    assert isinstance(p.time_at_volume_mm3(100.0), float)
    assert isinstance(p.volume_at_time_s(0.01), float)


def test_volume_at_time_inverts_the_map():
    p = _staged()
    v = np.linspace(0.0, p.total_volume_mm3 * 1.3, 25)
    assert p.volume_at_time_s(p.time_at_volume_mm3(v)) == pytest.approx(v)


def test_as_record_carries_every_stage_and_the_totals():
    p = _staged()
    rec = p.as_record()
    assert rec["screw_diameter_mm"] == 40.0
    assert rec["vp_position_mm"] == 5.0
    assert len(rec["stages"]) == 2
    assert rec["stages"][0]["velocity_mms"] == 20.0
    assert sum(s["volume_cm3"] for s in rec["stages"]) == pytest.approx(rec["total_volume_cm3"])
    assert sum(s["time_s"] for s in rec["stages"]) == pytest.approx(rec["total_time_s"])


# ---------------------------------------------------------------------------
# solver integration
# ---------------------------------------------------------------------------


def _solve(geom: Geometry, **kw):
    args = dict(
        melt_temperature_K=523.15,
        mold_temperature_K=323.15,
        injection_velocity_mms=200.0,
    )
    args.update(kw)
    return HeleShawSolver(geom, MaterialDB()["PP"], **args).solve(num_frames=4)


def test_single_stage_profile_matches_the_equivalent_constant_rate():
    """Nothing existing moves: one stage is the old model with a new input."""
    geom = _strip()
    p = _single(100.0)
    by_profile = _solve(geom, injection_profile=p)
    by_rate = _solve(geom, injection_volume_flow_cm3s=p.mean_rate_cm3s)
    assert by_profile.fill_time_s == pytest.approx(by_rate.fill_time_s, rel=1e-12)
    assert by_profile.total_fill_time_s == pytest.approx(by_rate.total_fill_time_s, rel=1e-12)


#: A strip whose volume the shot actually covers, so the kink lands inside
#: the cavity instead of past the end of it. 20 cells of 10x10x5 mm = 10 cm^3.
_N_BIG = 20
_CELL_BIG = 10.0
_THK_BIG = 5.0


def _big_strip() -> Geometry:
    return _strip(n=_N_BIG, thickness_mm=_THK_BIG, cell_mm=_CELL_BIG)


def _matched_profile(v1: float, v2: float) -> InjectionProfile:
    """Two stages whose stroke displaces exactly the big strip, switching halfway."""
    area = screw_area_mm2(20.0)
    v_total_mm3 = _N_BIG * _CELL_BIG * _CELL_BIG * _THK_BIG
    stroke = v_total_mm3 / area
    vp = 5.0
    return InjectionProfile(
        20.0,
        vp + stroke,
        (InjectionStage(vp + stroke / 2.0, v1), InjectionStage(vp, v2)),
    )


def test_arrival_times_follow_the_profile_map_cell_by_cell():
    """On an equal-cell strip the CDF is the cell index, so this is arithmetic.

    The independent oracle is the profile itself, evaluated outside the
    solver: cell k has swept (k+1) cell volumes, so it arrives when the
    machine has displaced that much. The switch sits at cell 10, so half the
    cells are read off each stage.
    """
    geom = _big_strip()
    p = _matched_profile(10.0, 100.0)
    res = _solve(geom, injection_profile=p)
    cell_vol = _CELL_BIG * _CELL_BIG * _THK_BIG
    expected = p.time_at_volume_mm3(np.arange(1, _N_BIG + 1) * cell_vol)
    assert res.fill_time_s[0] == pytest.approx(expected, rel=1e-9)
    assert res.total_fill_time_s == pytest.approx(p.total_time_s, rel=1e-9)


def test_staged_profile_bends_the_time_axis_against_constant_rate():
    """Same total shot time, different shape: a slow start arrives later."""
    geom = _big_strip()
    p = _matched_profile(10.0, 100.0)
    staged = _solve(geom, injection_profile=p)
    # One stage over the same stroke in the same total time.
    stroke = p.metering_position_mm - p.vp_position_mm
    flat_p = InjectionProfile(
        20.0, p.metering_position_mm, (InjectionStage(p.vp_position_mm, stroke / p.total_time_s),)
    )
    flat = _solve(geom, injection_profile=flat_p)
    assert staged.total_fill_time_s == pytest.approx(flat.total_fill_time_s, rel=1e-9)
    mid = _N_BIG // 2 - 1
    assert staged.fill_time_s[0, mid] > flat.fill_time_s[0, mid] * 1.5
    # The last cell still lands exactly on the reported fill time.
    assert staged.fill_time_s[0, -1] == pytest.approx(staged.total_fill_time_s)


def test_baseline_fill_time_is_the_profile_time_for_the_cavity_volume():
    geom = _strip(n=20)
    p = _staged()
    solver = HeleShawSolver(geom, MaterialDB()["PP"], injection_profile=p)
    V_mm3 = geom.volume_cm3() * 1000.0
    assert solver._baseline_fill_time(geom) == pytest.approx(p.time_at_volume_mm3(V_mm3))


def test_effective_rate_is_the_mean_over_this_cavity():
    geom = _strip(n=20)
    p = _staged()
    solver = HeleShawSolver(geom, MaterialDB()["PP"], injection_profile=p)
    V_cm3 = geom.volume_cm3()
    assert solver._effective_flow_rate_cm3s() == pytest.approx(
        V_cm3 / p.time_at_volume_mm3(V_cm3 * 1000.0)
    )


def test_restricted_solver_keeps_the_profile_and_reads_its_own_volume():
    """A profile is absolute -- the restricted copy is not re-normalized."""
    geom = _strip(n=20)
    p = _staged()
    solver = HeleShawSolver(geom, MaterialDB()["PP"], injection_profile=p)
    live = np.zeros_like(geom.mask)
    live[0, :5] = True
    sub = solver._restricted_to(live)
    assert sub.injection_profile is p
    assert sub.injection_volume_flow_cm3s is None
    assert sub._baseline_fill_time(sub.geometry) == pytest.approx(
        p.time_at_volume_mm3(sub.geometry.volume_cm3() * 1000.0)
    )


def test_metadata_records_the_profile():
    res = _solve(_strip(), injection_profile=_staged())
    rec = res.metadata["injection_profile"]
    assert rec is not None and len(rec["stages"]) == 2
    assert res.metadata["injection_Q_effective_cm3s"] > 0


def test_metadata_profile_is_none_without_one():
    res = _solve(_strip(), injection_volume_flow_cm3s=50.0)
    assert res.metadata["injection_profile"] is None


def test_multilayer_accepts_a_profile():
    geom = _strip(n=12)
    p = _staged()
    res = MultilayerHeleShawSolver(
        geometry=geom,
        material=MaterialDB()["PP"],
        num_layers=3,
        thermal_coupling=False,
        injection_profile=p,
    ).solve(num_frames=3)
    assert res.metadata["injection_profile"]["stages"][0]["velocity_mms"] == 20.0
    assert res.fill_time_s[0, -1] == pytest.approx(res.total_fill_time_s)


def test_two_phase_flags_a_shot_that_runs_past_vp():
    """A metered shot larger than the stroke displaces is an extrapolation.

    The sidebar's own check compares the stroke against the *final* cavity,
    which is neither the metered shot nor the open-gap cavity -- so the
    two-phase result has to carry the flag itself (@claude on PR #89).
    """
    geom = _strip(n=20, thickness_mm=5.0, cell_mm=10.0)  # 10 cm^3
    # A 10 mm screw over 5 mm of stroke displaces 0.39 cm^3: far less than
    # either the cavity or any usable shot.
    p = InjectionProfile(10.0, 20.0, (InjectionStage(15.0, 50.0),))
    solver = HeleShawSolver(
        geometry=geom,
        material=MaterialDB()["PP"],
        injection_profile=p,
        compression_molding=True,
        compression_stroke_mm=0.5,
    )
    res = solve_two_phase_short_shot(solver, 5.0)
    assert res.metadata["injection_extrapolated_past_vp"] is True
    assert res.metadata["injection_profile_volume_cm3"] == pytest.approx(p.total_volume_cm3)
    # Still the profile's own reading, extrapolated at the last stage's rate.
    assert res.injection_time_s == pytest.approx(p.time_at_volume_mm3(5000.0))
    assert res.injection_time_s > p.total_time_s


def test_two_phase_does_not_flag_a_shot_inside_the_stroke():
    geom = _strip(n=20)
    p = _staged()
    solver = HeleShawSolver(
        geometry=geom,
        material=MaterialDB()["PP"],
        injection_profile=p,
        compression_molding=True,
        compression_stroke_mm=0.5,
    )
    res = solve_two_phase_short_shot(solver, geom.volume_cm3() * 0.5)
    assert res.metadata["injection_extrapolated_past_vp"] is False


def test_two_phase_flag_is_false_without_a_profile():
    geom = _strip(n=20)
    solver = HeleShawSolver(
        geometry=geom,
        material=MaterialDB()["PP"],
        injection_volume_flow_cm3s=0.5,
        compression_molding=True,
        compression_stroke_mm=0.5,
    )
    res = solve_two_phase_short_shot(solver, geom.volume_cm3() * 0.5)
    assert res.metadata["injection_extrapolated_past_vp"] is False
    assert res.metadata["injection_profile_volume_cm3"] is None


def test_two_phase_injection_time_comes_from_the_profile():
    """The metered shot ends later when the machine spends it on a slow stage."""
    geom = _strip(n=20)
    p = _staged()
    solver = HeleShawSolver(
        geometry=geom,
        material=MaterialDB()["PP"],
        injection_profile=p,
        compression_molding=True,
        compression_stroke_mm=0.5,
    )
    V_shot = geom.volume_cm3() * 0.5
    res = solve_two_phase_short_shot(solver, V_shot)
    assert res.injection_time_s == pytest.approx(p.time_at_volume_mm3(V_shot * 1000.0))
    # Slower than the constant-rate reading of the same shot.
    assert res.injection_time_s > V_shot / p.mean_rate_cm3s
