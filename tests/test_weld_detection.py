"""Weld-line detector: confluence of flow directions, not a local maximum.

The old heuristic ("6 of 8 neighbours have a smaller tau") was a near-peak
test. A weld that keeps flowing -- the line behind a hole -- always has a
later downstream row, so at most 5 neighbours qualified and nothing was
drawn. The detector now reads the angle between the flow directions of
opposite neighbours that both point *into* the cell (a split around an
obstacle has the same angle but points away), so these tests are about
where the line appears and where it must not.
"""

from __future__ import annotations

import numpy as np
import pytest

from core import HeleShawSolver, MaterialDB
from core.geometry import Geometry
from core.solver import WELD_FULL_ANGLE_DEG, WELD_MIN_ANGLE_DEG


def _plate(ny: int, nx: int, gates: list[tuple[int, int]], hole: tuple | None = None) -> Geometry:
    mask = np.ones((ny, nx), dtype=bool)
    if hole is not None:
        y0, y1, x0, x1 = hole
        mask[y0:y1, x0:x1] = False
    return Geometry(
        mask=mask,
        thickness_mm=np.full((ny, nx), 0.5),
        cell_size_mm=1.0,
        gates=gates,
        label="weld-test",
    )


def _solve(g: Geometry):
    return HeleShawSolver(
        geometry=g,
        material=MaterialDB()["PP"],
        melt_temperature_K=503.15,
        mold_temperature_K=313.15,
        injection_velocity_mms=100.0,
        injection_volume_flow_cm3s=20.0,
    ).solve(num_frames=3)


def test_uniform_flow_has_no_weld():
    """A strip fed along one whole edge: parallel streams, nothing merges."""
    ny, nx = 40, 21
    g = _plate(ny, nx, [(0, ix) for ix in range(nx)])
    r = _solve(g)
    assert not (r.weld_score > 0).any()


def test_two_opposing_gates_weld_head_on_in_the_middle():
    ny, nx = 61, 9
    g = _plate(ny, nx, [(0, ix) for ix in range(nx)] + [(ny - 1, ix) for ix in range(nx)])
    r = _solve(g)
    rows = np.where((r.weld_score > 0).any(axis=1))[0]
    assert rows.size > 0
    # the collision is at mid-height, nowhere else
    assert rows.min() >= ny // 2 - 2 and rows.max() <= ny // 2 + 2
    # a head-on collision is the strongest weld there is
    assert r.weld_score[ny // 2, nx // 2] == pytest.approx(1.0)


def test_weld_forms_behind_a_hole_and_not_in_front_of_it():
    """Flow from the bottom edge splits around a square hole and merges above it."""
    ny, nx = 80, 41
    hole = (20, 30, 15, 26)  # rows 20..29, cols 15..25
    g = _plate(ny, nx, [(0, ix) for ix in range(nx)], hole=hole)
    r = _solve(g)
    w = r.weld_score
    cx = nx // 2
    behind = w[hole[1] : hole[1] + 6, cx - 1 : cx + 2]
    assert (behind > 0).any(), "no weld drawn where the two streams rejoin"
    # the split below the hole opens at the same angle but flows apart
    in_front = w[: hole[0], :]
    assert not (in_front > 0).any()
    # and the line sits on the centreline, not on the hole's flanks
    flank_rows = slice(hole[0], hole[1])
    assert not (w[flank_rows, :] > 0).any()


def test_flagged_cells_stay_off_the_gate_and_out_of_the_wall():
    ny, nx = 61, 9
    g = _plate(ny, nx, [(0, ix) for ix in range(nx)] + [(ny - 1, ix) for ix in range(nx)])
    r = _solve(g)
    assert not (r.weld_score[:3, :] > 0).any()
    assert not (r.weld_score[-3:, :] > 0).any()
    assert not (r.weld_score[~g.mask] > 0).any()


def test_score_is_the_meeting_angle_ramp():
    """Two converging streams at a known angle land on the documented ramp."""
    # synthetic tau: a V-shaped valley of arrival time, tau = |x| * slope + y
    ny, nx = 21, 21
    yy, xx = np.mgrid[:ny, :nx].astype(float)
    for slope, expect in ((10.0, 1.0), (0.0, 0.0)):
        tau = yy - slope * np.abs(xx - nx // 2)
        # the ridge of tau (late arrival) is x = centre; neighbours flow toward it
        s = HeleShawSolver._compute_weld_score(tau + 1000.0)  # keep clear of tau == 0 (gate)
        if expect == 1.0:
            assert s[ny // 2, nx // 2] == pytest.approx(1.0)
        else:
            assert s[ny // 2, nx // 2] == 0.0
    # an angle exactly between the two thresholds scores half
    mid = 0.5 * (WELD_MIN_ANGLE_DEG + WELD_FULL_ANGLE_DEG)
    slope = np.tan(np.radians(mid / 2.0))  # opening angle = 2*atan(slope)
    tau = yy - slope * np.abs(xx - nx // 2) + 100.0
    s = HeleShawSolver._compute_weld_score(tau)
    assert s[ny // 2, nx // 2] == pytest.approx(0.5, abs=0.02)


def test_angle_arguments_are_validated():
    tau = np.ones((5, 5))
    with pytest.raises(ValueError):
        HeleShawSolver._compute_weld_score(tau, min_angle_deg=50.0, full_angle_deg=40.0)
    with pytest.raises(ValueError):
        HeleShawSolver._compute_weld_score(tau, min_angle_deg=-1.0)


def test_angle_field_lets_the_renderer_rethreshold_without_resolving(tmp_path):
    """The hole's meld tail is a matter of threshold, not of solving again."""
    from core.visualizer import render_weldlines, weld_overlay_score

    ny, nx = 80, 41
    g = _plate(ny, nx, [(0, ix) for ix in range(nx)], hole=(20, 30, 15, 26))
    r = _solve(g)
    assert r.weld_angle_deg is not None
    loose = weld_overlay_score(r, min_angle_deg=0.0)
    tight = weld_overlay_score(r, min_angle_deg=30.0)
    assert (loose > 0).sum() > (tight > 0).sum()
    # tightening never invents cells, and the head-on cell survives both
    assert not ((tight > 0) & ~(loose > 0)).any()
    assert tight[31, nx // 2] == pytest.approx(1.0)  # first interior row above the hole
    # the thresholds reach the figure (legend text), and the rendering runs
    p = render_weldlines(r, tmp_path / "w.png", weld_min_angle_deg=30.0)
    assert p.exists()


def test_results_without_an_angle_field_fall_back_to_the_stored_score():
    from dataclasses import replace

    from core.visualizer import weld_overlay_score

    ny, nx = 61, 9
    g = _plate(ny, nx, [(0, ix) for ix in range(nx)] + [(ny - 1, ix) for ix in range(nx)])
    r = replace(_solve(g), weld_angle_deg=None)
    assert np.array_equal(weld_overlay_score(r, min_angle_deg=40.0), r.weld_score)


def test_even_width_grid_with_two_tied_centre_columns_still_finds_the_weld():
    """Symmetric solves tie the two centre columns to machine precision; the
    crest on each of them sees a zero drop toward its twin and must read one
    cell further out instead of calling itself flat."""
    ny, nx = 80, 40
    hole = (20, 30, 14, 26)  # symmetric about the column boundary 19|20
    g = _plate(ny, nx, [(0, ix) for ix in range(nx)], hole=hole)
    r = _solve(g)
    tau = r.tau
    assert np.allclose(tau[31:40, 19], tau[31:40, 20], rtol=1e-12)
    w = r.weld_score
    assert (w[31:34, 19] > 0).all() and (w[31:34, 20] > 0).all()
    assert w[31, 19] == pytest.approx(1.0) and w[31, 20] == pytest.approx(1.0)
    # the line is the twin pair and nothing beside it
    assert not (w[31:40, :19] > 0).any() and not (w[31:40, 21:] > 0).any()
