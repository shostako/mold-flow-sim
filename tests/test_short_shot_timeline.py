"""Tests for what a frozen cell is allowed to do to the timeline.

The skin-layer model floors ``h_core`` at ``min_core_thickness_mm`` for
numerical stability, so a cell whose two skins have met still solves -- with a
tau orders of magnitude above everything else. Letting that tau set the
absolute time scale turned a short shot into "it just takes longer": the
reported total fill time was several times the real one, the color bar spent
most of its range on cells that do not exist, and most animation frames showed
nothing happening. These tests pin the rule that a cell which does not fill
does not get a fill time.
"""

from __future__ import annotations

import numpy as np
import pytest

from core import HeleShawSolver, MaterialDB
from core.geometry import Geometry


def _strip(thickness_mm: np.ndarray, cell_mm: float = 1.0) -> Geometry:
    """One-row cavity with a prescribed thickness profile, gate at the left."""
    mask = np.ones((1, thickness_mm.size), dtype=bool)
    geom = Geometry(
        mask=mask,
        thickness_mm=thickness_mm.reshape(1, -1).astype(float),
        cell_size_mm=cell_mm,
    )
    geom.gates = [(0, 0)]
    return geom


def _solve(geom: Geometry, **kw):
    args = dict(
        melt_temperature_K=523.15,
        mold_temperature_K=323.15,
        injection_velocity_mms=200.0,
        injection_volume_flow_cm3s=50.0,
        skin_layer_enabled=True,
        skin_growth_constant=1.5,
    )
    args.update(kw)
    return HeleShawSolver(geom, MaterialDB()["PP"], **args).solve(num_frames=6)


@pytest.fixture(scope="module")
def frozen_run():
    """A strip whose far half is thin enough for the skins to meet."""
    thk = np.concatenate([np.full(30, 2.0), np.full(30, 0.05)])
    return _solve(_strip(thk))


def test_a_frozen_cell_does_not_set_the_total_fill_time(frozen_run):
    """T_fill must come from the cells that fill, not from the ones that froze."""
    r = frozen_run
    assert r.unfillable_mask is not None and r.unfillable_mask.any()
    live = r.geometry.mask & ~r.unfillable_mask
    assert live.any()
    assert np.nanmax(r.fill_time_s[live]) == pytest.approx(r.total_fill_time_s, rel=1e-9)


def test_cells_that_never_fill_have_no_fill_time(frozen_run):
    """NaN, not a large number: a time says "arrives eventually"."""
    r = frozen_run
    assert np.all(np.isnan(r.fill_time_s[r.unfillable_mask]))
    assert np.all(np.isnan(r.pressure_norm[r.unfillable_mask]))


def test_the_time_scale_comes_from_the_flowing_tau_not_the_frozen_one(frozen_run):
    """The two references differ by orders of magnitude; the scale takes the small one.

    This is the whole defect in one line. With ``tau_max`` in the scale, every
    live cell landed in the bottom fraction of the color bar and most frames
    showed nothing happening -- the size of that wasted band is exactly the
    ratio asserted here.
    """
    md = frozen_run.metadata
    # the cavity-wide solve, which the frozen cells dominate, against the
    # re-solve over the cells that fill
    assert md["tau_max_cavity"] / md["tau_max_flow"] > 10.0
    expected = md["T_fill_baseline_s"] * md["tau_max_flow"] / md["tau_max_baseline"]
    assert frozen_run.total_fill_time_s == pytest.approx(expected, rel=1e-9)


def test_cells_sealed_off_by_frozen_ones_are_unfillable_too():
    """A frozen cell is a wall. What is behind it never sees melt either.

    The thin band here freezes solid across the strip, so the thick zone past
    it is cut off from the gate even though its own core stays wide open.
    """
    thk = np.concatenate([np.full(20, 2.0), np.full(6, 0.04), np.full(20, 2.0)])
    r = _solve(_strip(thk))
    assert r.unfillable_mask is not None
    far_zone = np.zeros_like(r.geometry.mask)
    far_zone[0, 26:] = True
    assert r.unfillable_mask[far_zone].all(), "the sealed-off zone still counts as filling"
    assert r.metadata["sealed_off_cells"] >= int(far_zone.sum())
    assert np.all(np.isnan(r.fill_time_s[far_zone]))


def test_nothing_changes_when_no_cell_freezes():
    """The whole mechanism must be invisible on a part that fills."""
    r = _solve(_strip(np.full(40, 2.0)), skin_growth_constant=0.2)
    assert r.metadata["short_shot_cells"] == 0
    assert r.unfillable_mask is None
    assert np.nanmax(r.fill_time_s[r.geometry.mask]) == pytest.approx(r.total_fill_time_s, rel=1e-9)


def test_skin_off_leaves_no_unfillable_mask():
    """Backward compatibility: the field is None unless the skin model ran."""
    r = _solve(_strip(np.full(40, 2.0)), skin_layer_enabled=False)
    assert r.unfillable_mask is None
    assert r.metadata.get("unfillable_cells") is None


def test_a_frozen_gate_makes_the_whole_cavity_unfillable():
    """If the gate itself freezes there is nothing to be reachable from.

    Driven directly rather than through a solve, because the interesting part
    is the degenerate branch, and coaxing the physics into freezing exactly the
    gate cell would pin the test to whatever the growth law happens to do.
    """
    geom = _strip(np.full(10, 1.0))
    solver = HeleShawSolver(geom, MaterialDB()["PP"])
    frozen = np.zeros_like(geom.mask)
    frozen[0, 0] = True  # the gate
    assert solver._unfillable_cells(frozen)[geom.mask].all()


def test_the_tau_reference_reports_when_nothing_flows():
    """No usable reference must say so, not hand back the global maximum.

    The global maximum in a short shot *is* the dead-cell tau this exists to
    exclude, so a fallback to it would quietly undo the whole change.
    """
    solver = HeleShawSolver(_strip(np.full(4, 1.0)), MaterialDB()["PP"])
    tau = np.array([[0.0, 1.0, 2.0, 3.0]])
    assert solver._tau_reference(tau, np.zeros_like(tau, dtype=bool)) is None
    # a selection holding nothing but the gate's zero is just as unusable
    assert solver._tau_reference(tau, np.array([[True, False, False, False]])) is None
    assert solver._tau_reference(tau, np.ones_like(tau, dtype=bool)) == 3.0


def test_a_part_where_only_the_gate_fills_reports_the_baseline_time():
    """Nothing flows, so there is no resistance to inflate the time with.

    The old fallback took the global tau_max -- the frozen cells' own tau --
    and reported a total fill time tens of times the geometric baseline, while
    the animation independently fell back to a flat 1 s. Two contradictory
    timelines for a part that does not fill at all.
    """
    r = _solve(_sealed_plate())
    md = r.metadata
    assert md["no_flow"] is True
    assert r.total_fill_time_s == pytest.approx(md["T_fill_baseline_s"], rel=1e-9)
    assert md["T_fill_inflation"] == pytest.approx(1.0, rel=1e-9)
    finite = r.fill_time_s[np.isfinite(r.fill_time_s)]
    assert finite.size and np.all(finite == 0.0)


def test_the_color_axis_agrees_with_the_reported_time_when_nothing_flows():
    """One run, one timeline."""
    from core.visualizer import fill_time_max

    r = _solve(_sealed_plate())
    assert fill_time_max(r) == pytest.approx(r.total_fill_time_s, rel=1e-9)


def test_metadata_separates_frozen_cells_from_the_ones_they_sealed_off():
    """Two different failures: a thin wall, and a region stranded behind it."""
    thk = np.concatenate([np.full(20, 2.0), np.full(6, 0.04), np.full(20, 2.0)])
    md = _solve(_strip(thk)).metadata
    assert md["unfillable_cells"] == md["short_shot_cells"] + md["sealed_off_cells"]
    assert md["sealed_off_cells"] > 0


# --- rendering ---------------------------------------------------------------


def test_the_color_field_takes_a_dead_cell_from_its_live_neighbour():
    """A NaN cell must not tint the live cells beside it.

    ``_fill_field_rgb`` interpolates, so whatever value sits at a dead cell is
    smeared half a cell into its neighbours. Built by hand rather than from a
    solve: in a real part the cell next to a frozen one is usually the slowest
    live cell anyway, so the substituted value and the correct value coincide
    and the test would pass either way.
    """
    import matplotlib.pyplot as plt

    from core.visualizer import _fill_field_rgb

    times = np.array([[0.0, 0.1, np.nan, 1.0]])
    result = _fake_result(times, dead=np.array([[False, False, True, False]]))
    rgba = _fill_field_rgb(result, "turbo")
    cmap = plt.get_cmap("turbo")
    # nearest finite neighbour of the dead cell is 0.1 s -> 10 % of the scale
    assert np.allclose(rgba[0, 2, :3], cmap(0.1)[:3], atol=1e-6)
    assert not np.allclose(rgba[0, 2, :3], cmap(1.0)[:3], atol=0.02)
    assert np.isfinite(rgba).all()


def _fake_result(fill_time_s: np.ndarray, dead: np.ndarray):
    """Minimal FlowResult carrying just what the fill renderers read."""
    from core.solver import FlowResult

    mask = np.ones_like(fill_time_s, dtype=bool)
    geom = Geometry(mask=mask, thickness_mm=np.ones_like(fill_time_s), cell_size_mm=1.0)
    geom.gates = [(0, 0)]
    return FlowResult(
        tau=fill_time_s.copy(),
        fill_time_s=fill_time_s,
        pressure_norm=np.zeros_like(fill_time_s),
        weld_score=np.zeros_like(fill_time_s),
        air_traps=np.zeros_like(fill_time_s, dtype=bool),
        total_fill_time_s=float(np.nanmax(fill_time_s)),
        viscosity_Pa_s=100.0,
        geometry=geom,
        metadata={},
        unfillable_mask=dead,
    )


def test_dead_cells_keep_their_own_color_for_the_whole_animation(frozen_run):
    """They must not read as "not filled yet" -- they are never going to fill."""
    from core.visualizer import SHORT_SHOT_RGB, _cavity_backdrop_colors, _unfilled_overlay

    r = frozen_run
    _, cavity_gray = _cavity_backdrop_colors()
    # Ask for the worst case explicitly: even told that every cell has filled,
    # the overlay must keep the dead ones covered.
    everything_filled = np.ones_like(r.geometry.mask)
    overlay = _unfilled_overlay(r, everything_filled)
    dead = r.unfillable_mask
    assert np.allclose(overlay[dead][:, :3], np.asarray(SHORT_SHOT_RGB))
    assert not np.allclose(np.asarray(SHORT_SHOT_RGB), np.asarray(cavity_gray))
    # and they stay painted even on a frame where every live cell has filled
    assert np.all(overlay[dead][:, 3] == 1.0)


def test_the_frame_title_states_how_many_cells_never_fill(frozen_run):
    """ "filled = 100.0 %" rounds a short shot away; the count must be spelled out."""
    from core.visualizer import _fill_title

    n = int(frozen_run.unfillable_mask.sum())
    title = _fill_title(frozen_run, 0.01, 0.9999)
    assert f"{n} cells" in title
    assert "short shot" in title


def test_no_short_shot_leaves_the_title_alone():
    from core.visualizer import _fill_title

    r = _solve(_strip(np.full(40, 2.0)), skin_growth_constant=0.2)
    assert "short shot" not in _fill_title(r, 0.01, 0.5)


# --- downstream of the NaNs --------------------------------------------------


def _sealed_plate(n: int = 24):
    """Plate whose gate is ringed by cells thin enough to freeze shut."""
    thk = np.full((n, n), 2.0)
    thk[2:5, 2:5] = 0.04
    thk[3, 3] = 2.0
    geom = Geometry(mask=np.ones((n, n), dtype=bool), thickness_mm=thk, cell_size_mm=1.0)
    geom.gates = [(3, 3)]
    return geom


def test_the_weld_plot_survives_a_part_that_barely_fills(tmp_path):
    """One distinct fill time left is still an analysis, not an error.

    A ring of frozen cells around the gate leaves the gate as the only cell
    with a time. ``contour`` rejects a flat level list, so the renderer used to
    raise and take a completed run down with it.
    """
    from core.visualizer import render_weldlines

    r = _solve(_sealed_plate())
    finite = r.fill_time_s[np.isfinite(r.fill_time_s)]
    assert np.unique(finite).size < 2, "geometry no longer reproduces the degenerate case"
    out = render_weldlines(r, tmp_path / "weld.png")
    assert out.exists() and out.stat().st_size > 0


def _sealed_pocket(n: int = 30):
    """Plate with a pocket walled off by a frozen ring, gate in a far corner.

    The pocket matters: a dead region that touches the domain border has its
    tau maximum on the border, where the neighbour scan does not look, so the
    false defects never appear and a test built on it proves nothing.
    """
    thk = np.full((n, n), 2.0)
    thk[10:20, 10:20] = 0.04
    thk[12:18, 12:18] = 2.0
    geom = Geometry(mask=np.ones((n, n), dtype=bool), thickness_mm=thk, cell_size_mm=1.0)
    geom.gates = [(2, 2)]
    return geom


def test_defect_diagnostics_ignore_material_that_never_fills():
    """A frozen cell's tau is huge, which is the exact shape of an air trap.

    Weld lines and air traps answer "when did melt arrive"; where none arrives
    there is nothing to answer, and a marker there is a defect the part cannot
    have. The precondition is asserted too: on the raw tau these diagnostics
    *do* fire inside the dead pocket, so this test can tell the two apart.
    """
    geom = _sealed_pocket()
    solver = HeleShawSolver(
        geom,
        MaterialDB()["PP"],
        melt_temperature_K=523.15,
        mold_temperature_K=323.15,
        injection_velocity_mms=200.0,
        injection_volume_flow_cm3s=50.0,
        skin_layer_enabled=True,
        skin_growth_constant=1.5,
    )
    r = solver.solve(num_frames=6)
    dead = r.unfillable_mask
    assert dead is not None and dead.sum() > 10

    # The solved field has no value at all on dead cells, so the diagnostics
    # cannot see them. Proven not vacuous by feeding the same detector a field
    # that *does* carry the frozen cells' tau: it fires inside the pocket.
    assert np.all(np.isnan(r.tau[dead]))
    # a bowl rather than a plateau: the weld heuristic needs neighbours that
    # are strictly smaller, which a flat block of equal values never gives
    yy, xx = np.mgrid[: dead.shape[0], : dead.shape[1]]
    cy, cx = (np.mean(np.where(dead)[0]), np.mean(np.where(dead)[1]))
    # the peak is nudged off the cell centers: a bowl centered exactly between
    # four cells leaves them tied, and the heuristic counts strictly-smaller
    # neighbours
    bowl = (
        1e3
        * float(np.nanmax(r.tau[~dead]))
        / (1.0 + (yy - cy - 0.37) ** 2 + 1.3 * (xx - cx - 0.11) ** 2)
    )
    spiked = np.where(dead, bowl, np.nan_to_num(r.tau))
    assert (solver._compute_air_traps(spiked) & dead).any()
    assert ((solver._compute_weld_score(spiked) > 0.0) & dead).any()

    assert not (r.air_traps & dead).any()
    assert not ((r.weld_score > 0.0) & dead).any()


def test_the_pressure_map_colors_dead_cells_instead_of_leaving_them_black(tmp_path):
    """NaN through a colormap is the "bad" color, and the renderer forces alpha 1.

    The result reads as the bottom of the pressure scale -- the opposite of
    "no melt here". Checked on the rendered pixels, because the defect only
    appears after the alpha is forced.
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    from core.visualizer import SHORT_SHOT_RGB, render_pressure_map

    r = _solve(_sealed_plate())
    path = render_pressure_map(r, tmp_path / "pressure.png")
    px = np.asarray(Image.open(path).convert("RGB")).astype(float)
    short_shot = np.array(SHORT_SHOT_RGB) * 255.0
    bottom = np.array(plt.get_cmap("magma")(0.0)[:3]) * 255.0
    n_short = int((np.abs(px - short_shot).max(axis=2) < 6).sum())
    n_bottom = int((np.abs(px - bottom).max(axis=2) < 6).sum())
    assert n_short > 1000, "dead cells are not marked"
    assert n_short > n_bottom, "dead cells still read as lowest pressure"


def test_every_renderer_reads_the_same_time_axis():
    """One run, one timeline -- checked across all three consumers.

    The color scale, the titles and the frame schedule each used to derive the
    axis themselves, so fixing one of them left the others reporting a
    different total. Asserted together because the defect is the duplication,
    not any single copy of it.
    """
    from core.visualizer import fill_frame_times, fill_time_max

    r = _solve(_sealed_plate())
    assert fill_time_max(r) == pytest.approx(r.total_fill_time_s, rel=1e-9)
    assert fill_frame_times(r, 60)[-1] == pytest.approx(r.total_fill_time_s, rel=1e-9)
    assert fill_frame_times(r, 1)[-1] == pytest.approx(r.total_fill_time_s, rel=1e-9)


@pytest.mark.parametrize("max_iterations", [3, 5, 8])
def test_freezing_everything_mid_iteration_does_not_crash(max_iterations):
    """The fixed-point loop must survive losing its tau reference.

    Once every cell outside the gate has frozen there is no reference left,
    and the next pass divided by it -- ``TypeError: unsupported operand
    type(s) for /: 'float' and 'NoneType'``, on a run that had already
    produced its answer. Parametrized over the iteration cap because whether
    the loop took another pass depended on it.
    """
    n = 12
    thk = np.full((n, n), 0.03)
    thk[3, 3] = 2.0  # the gate stays open, everything else closes
    geom = Geometry(mask=np.ones((n, n), dtype=bool), thickness_mm=thk, cell_size_mm=1.0)
    geom.gates = [(3, 3)]
    r = _solve(geom, skin_growth_constant=3.0, skin_max_iterations=max_iterations)
    assert r.metadata["no_flow"] is True
    assert r.total_fill_time_s == pytest.approx(r.metadata["T_fill_baseline_s"], rel=1e-9)
    assert r.metadata["skin_iterations"] <= max_iterations


def _sealed_strip():
    """Strip cut in two by a band thin enough to freeze shut."""
    return _strip(np.concatenate([np.full(20, 2.0), np.full(6, 0.04), np.full(20, 2.0)]))


def test_the_live_region_is_re_solved_without_the_dead_load():
    """The first solve pushes the dead region's volume through the live cells.

    Every cavity cell contributes a unit source, and a frozen cell still
    conducts through the numerical floor -- so the volume behind the frozen
    band is driven through the cells upstream of it, inflating their tau by
    material that never fills. Measured at 3.3x on this strip.
    """
    r = _solve(_sealed_strip())
    live = r.geometry.mask & ~r.unfillable_mask

    # what the live region looks like when it is the whole problem
    alone = _strip(r.geometry.thickness_mm[0].copy())
    alone.mask = live.copy()
    reference = _solve(alone)

    ours = float(np.nanmax(r.tau[live]))
    theirs = float(np.nanmax(reference.tau[live]))
    # Not equal: the reference runs its own skin iteration on the smaller
    # domain, so its arrival times -- and the skin they grow -- differ a little.
    # The point is the order of magnitude: carrying the dead load put this 3.3x
    # above the reference, which no amount of skin bookkeeping accounts for.
    assert abs(ours / theirs - 1.0) < 0.25, "the live field still carries the dead load"


def test_the_baseline_time_counts_only_the_volume_that_fills():
    """Melt does not spend time on volume it never occupies."""
    r = _solve(_sealed_strip())
    live = r.geometry.mask & ~r.unfillable_mask
    thickness = r.geometry.thickness_mm
    q_cm3s = 50.0
    live_cm3 = float(thickness[live].sum()) * r.geometry.cell_size_mm**2 / 1000.0
    whole_cm3 = r.geometry.volume_cm3()
    assert live_cm3 < whole_cm3 * 0.9  # the case is worth testing
    assert r.metadata["T_fill_baseline_s"] == pytest.approx(live_cm3 / q_cm3s, rel=1e-9)


def test_the_inflation_compares_two_states_of_the_same_region():
    """Freezing slowed *the region that still flows*; both references live there.

    Comparing a skin-carved live domain against a cavity-wide open reference
    would fold the geometry change into a number that is supposed to measure
    the skin.
    """
    r = _solve(_sealed_strip())
    md = r.metadata
    assert md["T_fill_inflation"] == pytest.approx(
        md["tau_max_flow"] / md["tau_max_baseline"], rel=1e-9
    )
    # the cavity-wide solve is orders away; it must not be what fed the ratio
    assert md["tau_max_cavity"] / md["tau_max_flow"] > 100.0
