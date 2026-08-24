"""Tests for what a sealing cell is allowed to do to the timeline.

The skin-layer model runs on the exposure clock (Issue #61): a cell's wall ages
from the moment the front passes it, its core seals at ``t_close = t_arr + t_c``,
and a cell the front reaches only after every path to a gate has sealed does
not fill. Two sets come out of that and must stay apart: ``short_shot_mask``
(cells that sealed -- they filled first, then closed) and ``unfillable_mask``
(cells the melt never reached). Letting a dead cell's tau set the absolute
time scale used to turn a short shot into "it just takes longer": the reported
total fill time was several times the real one, the color bar spent most of
its range on cells that do not exist, and most animation frames showed
nothing happening. These tests pin the rule that a cell which does not fill
does not get a fill time -- and that a cell which sealed does.
"""

from __future__ import annotations

import numpy as np
import pytest

from core import HeleShawSolver, MaterialDB
from core.geometry import Geometry

#: Slow enough that the skins of a 0.04-0.05 mm band reach each other while
#: the strip is still filling (t_c ~ 1-2 ms at c_skin = 1.5, against fills of
#: 0.1 s and up). At the 50 cm3/s of the old fixtures the whole strip filled
#: in a millisecond and nothing could seal.
Q_CM3S = 0.5


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
        injection_volume_flow_cm3s=Q_CM3S,
        skin_layer_enabled=True,
        skin_growth_constant=1.5,
    )
    args.update(kw)
    return HeleShawSolver(geom, MaterialDB()["PP"], **args).solve(num_frames=6)


@pytest.fixture(scope="module")
def frozen_run():
    """A strip whose far half is thin enough for the skins to meet mid-fill.

    The first thin cells seal a couple of milliseconds after the front passes
    them; the thin cells beyond arrive later than that and never fill.
    """
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
    # The inflation ratio is built from volume-weighted representatives over
    # the still-flowing set -- not from single-cell maxima, which is how one
    # pathological cell used to own the whole normalization (Issue #52).
    expected = md["T_fill_baseline_s"] * md["tau_rep_flow"] / md["tau_rep_baseline"]
    assert frozen_run.total_fill_time_s == pytest.approx(expected, rel=1e-9)


def test_cells_sealed_off_by_frozen_ones_are_unfillable_too():
    """A sealed cell is a wall from ``t_close`` on. What arrives later never fills.

    The thin band here seals across the strip a millisecond after the front
    enters it, so the thick zone past it -- whose first cell alone takes four
    milliseconds to fill -- is cut off even though its own core stays open.
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


def test_reachability_is_read_at_each_cells_own_arrival():
    """Unit contract of ``_unfillable_cells``: a seal cuts off what arrives after it closes.

    Driven directly rather than through a solve, because the contract is about
    clocks, and coaxing the physics into sealing exactly one cell would pin the
    test to whatever the growth law happens to do. Cell ``k`` arrives at ``k``
    seconds; cell 3 seals at 5.5 s. Cells 0-5 arrive while it is still open
    (cell 3 itself included -- it filled, then closed), cells 6-9 do not.
    """
    geom = _strip(np.full(10, 1.0))
    solver = HeleShawSolver(geom, MaterialDB()["PP"])
    t_arr = np.arange(10, dtype=float).reshape(1, -1)
    frozen = np.zeros_like(geom.mask)
    frozen[0, 3] = True
    t_close = np.full_like(t_arr, np.inf)
    t_close[0, 3] = 5.5
    dead = solver._unfillable_cells(frozen, t_arr, t_close)
    assert not dead[0, :6].any()
    assert dead[0, 6:].all()

    # a sealing gate: everything that arrives after it closes is cut off, the
    # gate itself still fills (it is open at its own arrival)
    frozen = np.zeros_like(geom.mask)
    frozen[0, 0] = True
    t_close = np.full_like(t_arr, np.inf)
    t_close[0, 0] = 0.5
    dead = solver._unfillable_cells(frozen, t_arr, t_close)
    assert not dead[0, 0]
    assert dead[0, 1:].all()

    # two seals: the later one only matters for cells behind both
    frozen = np.zeros_like(geom.mask)
    frozen[0, [2, 6]] = True
    t_close = np.full_like(t_arr, np.inf)
    t_close[0, 2] = 7.5
    t_close[0, 6] = 8.5
    dead = solver._unfillable_cells(frozen, t_arr, t_close)
    assert not dead[0, :8].any()
    assert dead[0, 8:].all()

    # arriving exactly at the closing time is too late
    t_close[0, 2] = 4.0
    dead = solver._unfillable_cells(frozen, t_arr, t_close)
    assert not dead[0, :4].any()
    assert dead[0, 4:].all()


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
    r = _solve_sealed_plate()
    md = r.metadata
    assert md["no_flow"] is True
    assert r.total_fill_time_s == pytest.approx(md["T_fill_baseline_s"], rel=1e-9)
    assert md["T_fill_inflation"] == pytest.approx(1.0, rel=1e-9)
    # The only cell that fills is the gate, and under the volume map its time
    # is the time to inject its own volume -- which IS the whole fill here.
    finite = r.fill_time_s[np.isfinite(r.fill_time_s)]
    assert finite.size and np.all(finite == pytest.approx(r.total_fill_time_s, rel=1e-9))


def test_the_color_axis_agrees_with_the_reported_time_when_nothing_flows():
    """One run, one timeline."""
    from core.visualizer import fill_time_max

    r = _solve_sealed_plate()
    assert fill_time_max(r) == pytest.approx(r.total_fill_time_s, rel=1e-9)


def test_metadata_separates_the_seal_from_what_it_cut_off():
    """Two different facts: a wall that closed, and a region stranded behind it.

    The seal filled before it closed, so it is not part of the missing melt --
    the two counts must not overlap, and everything unfillable is sealed off.
    """
    thk = np.concatenate([np.full(20, 2.0), np.full(6, 0.04), np.full(20, 2.0)])
    r = _solve(_strip(thk))
    md = r.metadata
    assert md["short_shot_cells"] > 0
    assert md["sealed_off_cells"] > 0
    assert md["sealed_off_cells"] == md["unfillable_cells"]
    assert not (r.short_shot_mask & r.unfillable_mask).any()
    assert md["short_shot_cells"] == int(r.short_shot_mask.sum())
    assert md["filled_volume_fraction"] == pytest.approx(
        float(thk[~r.unfillable_mask[0]].sum()) / float(thk.sum()), rel=1e-9
    )


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
    """Thin plate whose gate seals before its first neighbour arrives.

    Uniformly 0.02 mm: the gate's own skins meet 0.12 ms after it fills, and
    at the slow rate ``_solve_sealed_plate`` uses the next cell takes 0.4 ms.
    So nothing but the gate cell ever fills -- the degenerate branch where the
    tau reference vanishes.
    """
    thk = np.full((n, n), 0.02)
    geom = Geometry(mask=np.ones((n, n), dtype=bool), thickness_mm=thk, cell_size_mm=1.0)
    geom.gates = [(3, 3)]
    return geom


def _solve_sealed_plate():
    return _solve(_sealed_plate(), injection_volume_flow_cm3s=0.05)


def test_the_weld_plot_survives_a_part_that_barely_fills(tmp_path):
    """One distinct fill time left is still an analysis, not an error.

    A gate that seals before its neighbours arrive leaves the gate as the only
    cell with a time. ``contour`` rejects a flat level list, so the renderer
    used to raise and take a completed run down with it.
    """
    from core.visualizer import render_weldlines

    r = _solve_sealed_plate()
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
        injection_volume_flow_cm3s=Q_CM3S,
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

    r = _solve_sealed_plate()
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

    r = _solve_sealed_plate()
    assert fill_time_max(r) == pytest.approx(r.total_fill_time_s, rel=1e-9)
    assert fill_frame_times(r, 60)[-1] == pytest.approx(r.total_fill_time_s, rel=1e-9)
    assert fill_frame_times(r, 1)[-1] == pytest.approx(r.total_fill_time_s, rel=1e-9)


@pytest.mark.parametrize("max_iterations", [3, 5, 8])
def test_freezing_everything_mid_iteration_does_not_crash(max_iterations):
    """The fixed-point loop must survive losing its tau reference.

    Once every cell outside the gate has been cut off there is no reference
    left, and the next pass divided by it -- ``TypeError: unsupported operand
    type(s) for /: 'float' and 'NoneType'``, on a run that had already
    produced its answer. Parametrized over the iteration cap because whether
    the loop took another pass depended on it. The gate here seals before its
    first neighbour arrives, so the gate is all that fills.
    """
    n = 12
    thk = np.full((n, n), 0.03)
    geom = Geometry(mask=np.ones((n, n), dtype=bool), thickness_mm=thk, cell_size_mm=1.0)
    geom.gates = [(3, 3)]
    r = _solve(
        geom,
        skin_growth_constant=3.0,
        skin_max_iterations=max_iterations,
        injection_volume_flow_cm3s=0.05,
    )
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
    q_cm3s = Q_CM3S
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
        md["tau_rep_flow"] / md["tau_rep_baseline"], rel=1e-9
    )
    # the cavity-wide solve is orders away; it must not be what fed the ratio
    assert md["tau_max_cavity"] / md["tau_max_flow"] > 100.0


def test_the_default_rate_scales_with_the_live_volume_too():
    """No explicit rate still means a rate, not a fixed duration.

    ``injection_volume_flow_cm3s=None`` documents a 1.5 s fill of the cavity as
    drawn. Read as a duration it hands the same 1.5 s to a part where a tenth
    of the volume is left -- the live-volume scaling silently stops working on
    the default path, which is the one every other test avoids by passing a
    rate explicitly.
    """
    from core.solver import DEFAULT_FILL_TIME_S

    # Not _sealed_strip(): at the slow 1.5 s default fill a 1.5 growth
    # constant freezes the 2.0 mm cells too (physically fair -- PP skins are
    # ~0.5 mm/side by then and the constant-pressure feedback finishes the
    # job). A gentler constant with a thinner band keeps the near half alive
    # while the band still seals, which is the split this test needs.
    geom = _strip(np.concatenate([np.full(20, 2.0), np.full(6, 0.03), np.full(20, 2.0)]))
    r = HeleShawSolver(
        geom,
        MaterialDB()["PP"],
        melt_temperature_K=523.15,
        mold_temperature_K=323.15,
        injection_velocity_mms=200.0,
        injection_volume_flow_cm3s=None,
        skin_layer_enabled=True,
        skin_growth_constant=0.5,
    ).solve(num_frames=6)
    live = geom.mask & ~r.unfillable_mask
    share = float(geom.thickness_mm[live].sum()) / float(geom.thickness_mm[geom.mask].sum())
    assert share < 0.9  # the case is worth testing
    assert r.metadata["T_fill_baseline_s"] == pytest.approx(DEFAULT_FILL_TIME_S * share, rel=1e-9)


def test_the_default_rate_is_unchanged_on_a_part_that_fills():
    """Backward compatibility: no freezing, no restriction, still 1.5 s."""
    from core.solver import DEFAULT_FILL_TIME_S

    r = _solve(
        _strip(np.full(20, 2.0)),
        injection_volume_flow_cm3s=None,
        skin_growth_constant=0.2,
    )
    assert r.unfillable_mask is None
    # the baseline, not the total: the skin still inflates the reported time,
    # and folding that in would make this test pass for the wrong reason
    assert r.metadata["T_fill_baseline_s"] == pytest.approx(DEFAULT_FILL_TIME_S, rel=1e-9)


# --- the domain and the skin field settle together -----------------------------


def _band_sealed(n_dead: int, live_thk: float = 1.0, n_live: int = 20):
    """Live run of ``n_live`` cells, a band thin enough to close, then dead material.

    The live half is identical whatever ``n_dead`` is, so anything that differs
    between two of these came out of material the melt never reaches.
    """
    return _strip(
        np.concatenate([np.full(n_live, live_thk), np.full(6, 0.04), np.full(n_dead, live_thk)])
    )


def test_the_live_region_does_not_care_how_much_is_sealed_off_behind_it():
    """Dead volume must not reach the answer -- not even through the skin field.

    The skin thickness is driven by arrival times, and the arrival times of a
    cavity that still carries a sealed-off region are set partly by volume that
    never moves. Solving the skin fixed point once on the full cavity and then
    re-solving tau on the live part removes the dead cells from the final
    equation but keeps the core they carved: the same live geometry behind a
    small dead tail and a large one reported fill times 3.9x apart.

    The bisection resolves the boundary to a fraction of the cavity volume,
    so the two runs may disagree by a band cell or two -- within the larger
    cavity's resolution, and never inside the thick run before the band.
    Exactness on the cavity that is reported is the next test's job.
    """
    from core.solver import DOMAIN_VOLUME_RESOLUTION

    small = _solve(_band_sealed(20))
    large = _solve(_band_sealed(100))
    assert small.unfillable_mask is not None and small.unfillable_mask.any()
    assert large.unfillable_mask is not None and large.unfillable_mask.any()

    live_s = ~small.unfillable_mask[0]
    live_l = ~large.unfillable_mask[0]
    assert live_s[:20].all() and live_l[:20].all(), "the thick run before the band was cut"
    assert not live_s[26:].any() and not live_l[26:].any(), "material behind the band filled"

    thk_l = large.geometry.thickness_mm[0]
    tolerance = float(thk_l.sum()) / DOMAIN_VOLUME_RESOLUTION
    v_s = float(small.geometry.thickness_mm[0][live_s].sum())
    v_l = float(thk_l[live_l].sum())
    assert abs(v_s - v_l) <= tolerance, (
        "the live cavity depends on dead material beyond the resolution"
    )


def test_the_reported_run_is_a_fixed_point_of_the_cavity_it_reports():
    """Hand the live cavity back to a fresh solver and nothing may move.

    That is the whole claim of a short-shot result: *this* is the part that
    fills, and these are its times. If solving that part on its own gives a
    different answer, the reported one was still carrying the dead region.
    """
    run = _solve(_band_sealed(100))
    assert run.unfillable_mask is not None and run.unfillable_mask.any()
    live = run.geometry.mask & ~run.unfillable_mask
    assert live.sum() >= 2

    reported = Geometry(
        mask=live.copy(),
        thickness_mm=run.geometry.thickness_mm.copy(),
        cell_size_mm=run.geometry.cell_size_mm,
    )
    reported.gates = [(iy, ix) for iy, ix in run.geometry.gates if live[iy, ix]]
    again = _solve(reported)

    assert again.total_fill_time_s == pytest.approx(run.total_fill_time_s, rel=1e-9)
    assert again.core_thickness_mm[live] == pytest.approx(run.core_thickness_mm[live], rel=1e-9)
    assert again.fill_time_s[live] == pytest.approx(run.fill_time_s[live], rel=1e-9)


def test_the_restricted_solver_keeps_the_rate_it_was_running_at():
    """The restriction changes the cavity, not the machine.

    With no rate given, one is derived from the geometry -- so a restricted
    copy left to derive its own would divide the shrunken volume by a rate read
    off that same shrunken volume, and every short shot would report the
    default fill time no matter how little of it filled.
    """
    geom = _strip(np.full(40, 1.0))
    solver = HeleShawSolver(geom, MaterialDB()["PP"])  # no injection rate given
    assert solver.injection_volume_flow_cm3s is None
    rate = solver._effective_flow_rate_cm3s()

    live = geom.mask.copy()
    live[0, 20:] = False
    sub = solver._restricted_to(live)
    assert sub._effective_flow_rate_cm3s() == pytest.approx(rate, rel=1e-12)
    assert sub._baseline_fill_time(sub.geometry) == pytest.approx(
        solver._baseline_fill_time(sub.geometry), rel=1e-12
    )


# --- the volume map (Issue #52) ----------------------------------------------


def test_arrival_follows_swept_volume_on_a_uniform_strip():
    """Constant rate, uniform strip: the front moves at constant speed.

    Equal cells fill at equal intervals -- cell k at (k+1)/n of the total. The
    old linear map ``tau / tau_max * T_fill`` reported the parabolic tau
    profile as if it were time, putting the mid-strip cell at 0.75 T instead
    of 0.5 T. This is the healthy-case half of Issue #52: the map was wrong
    before any pathological cell entered the picture.
    """
    n = 40
    r = _solve(_strip(np.full(n, 2.0)), skin_layer_enabled=False)
    expected = (np.arange(n) + 1) / n * r.total_fill_time_s
    assert np.allclose(r.fill_time_s[0], expected, rtol=1e-9)


def test_one_slow_cell_does_not_move_anyone_elses_clock():
    """The pathological half of Issue #52, pinned as an exact invariance.

    Append one pathologically thin cell to a healthy strip. Its tau is orders
    of magnitude above everything, but under the volume map a cell's absolute
    time is (volume at or below its tau) / Q -- so the healthy cells' times do
    not change at all when the outlier is added. Under the old map the outlier
    sat in the denominator and rescaled every clock in the cavity.
    """
    healthy = _solve(_strip(np.full(40, 2.0)), skin_layer_enabled=False)
    with_outlier = _solve(
        _strip(np.concatenate([np.full(40, 2.0), np.full(1, 0.05)])),
        skin_layer_enabled=False,
    )
    assert np.allclose(with_outlier.fill_time_s[0, :40], healthy.fill_time_s[0], rtol=1e-9)
    # and the outlier itself is simply the last cell to fill
    assert with_outlier.fill_time_s[0, 40] == pytest.approx(
        with_outlier.total_fill_time_s, rel=1e-9
    )


def test_the_volume_map_helper_handles_ties_nans_and_the_end():
    """Unit contract of ``_arrival_time_field``.

    Ties share the arrival of the last cell in the group (equal tau must not
    order itself by memory layout), excluded cells stay NaN, and the largest
    tau lands exactly on T_fill.
    """
    solver = HeleShawSolver(_strip(np.full(4, 1.0)), MaterialDB()["PP"])
    tau = np.array([[0.0, 2.0, 2.0, 5.0, np.nan]])
    where = np.array([[True, True, True, True, True]])
    vol = np.array([[1.0, 1.0, 1.0, 1.0, 1.0]])
    t = solver._arrival_time_field(tau, where, vol, 8.0)
    assert t[0, 0] == pytest.approx(8.0 * 1 / 4)
    # the tie at tau=2.0 shares one arrival: the group's last cell (3 of 4)
    assert t[0, 1] == t[0, 2] == pytest.approx(8.0 * 3 / 4)
    assert t[0, 3] == pytest.approx(8.0)
    assert np.isnan(t[0, 4])
    # excluded cells stay NaN even with finite tau
    where2 = np.array([[True, False, True, True, True]])
    t2 = solver._arrival_time_field(tau, where2, vol, 8.0)
    assert np.isnan(t2[0, 1])
    # empty selection: all NaN, no division by an empty cumsum
    t3 = solver._arrival_time_field(tau, np.zeros_like(where), vol, 8.0)
    assert np.all(np.isnan(t3))


def test_the_volume_mean_helper_weights_by_volume():
    """Unit contract of ``_tau_volume_mean``."""
    solver = HeleShawSolver(_strip(np.full(4, 1.0)), MaterialDB()["PP"])
    tau = np.array([[1.0, 3.0, np.nan]])
    where = np.array([[True, True, True]])
    vol = np.array([[3.0, 1.0, 5.0]])
    # (1*3 + 3*1) / (3 + 1) = 1.5 -- the NaN cell drops out, volume-weighted
    assert solver._tau_volume_mean(tau, where, vol) == pytest.approx(1.5)
    assert solver._tau_volume_mean(tau, np.zeros_like(where), vol) is None
    assert solver._tau_volume_mean(tau, where, np.zeros_like(vol)) is None
    assert solver._tau_volume_mean(np.zeros_like(tau), where, vol) is None


def test_a_gate_side_choke_seals_and_cuts_the_plate_off():
    """The skin clock is the *exposure* clock (Issue #61).

    A 0.04 mm ring next to the gate is passed by the front almost at t = 0 --
    and then keeps conducting, and ageing, for the rest of the fill. Under the
    arrival clock the conductance snapshot saw almost no skin and the ring
    stayed open (that clock was pinned here until v0.33.0, with its tradeoff
    spelled out). On the exposure clock the ring seals a millisecond after the
    front passes it, long before the 2 mm plate cells behind it -- four
    milliseconds each -- can arrive, so the plate is cut off while the gate
    and the ring itself fill.
    """
    n = 24
    thk = np.full((n, n), 2.0)
    thk[2:5, 2:5] = 0.04
    thk[3, 3] = 2.0
    geom = Geometry(mask=np.ones((n, n), dtype=bool), thickness_mm=thk, cell_size_mm=1.0)
    geom.gates = [(3, 3)]
    r = _solve(geom)
    md = r.metadata
    plate = thk == 2.0
    plate[3, 3] = False
    assert md["no_flow"] is False
    assert r.unfillable_mask[plate].all(), "the plate behind the ring still fills"
    assert not r.unfillable_mask[3, 3]
    assert np.isfinite(r.fill_time_s[3, 3])
    assert r.short_shot_mask.any()
    assert (r.short_shot_mask & (thk == 0.04)).any(), "the seal is not the ring"
    assert not (r.short_shot_mask & r.unfillable_mask).any()
    assert md["filled_volume_fraction"] < 0.01


def test_the_domain_pass_valve_still_buries_its_dead_cells(monkeypatch):
    """Codex P2 on PR #60: tripping the pass cap must not leave dead cells live.

    With the cap forced to zero the loop never re-solves without the cells
    the first solution cut off. They still must end up in ``unfillable_mask``,
    keep NaN fill times, stay apart from the seal that cut them off, and the
    run must say the domain did not converge.
    """
    import core.solver as solver_mod

    monkeypatch.setattr(solver_mod, "MAX_DOMAIN_PASSES", 0)
    r = _solve(_sealed_strip())
    md = r.metadata
    assert md["domain_converged"] is False
    assert md["domain_passes"] == 0
    assert r.short_shot_mask.any()
    assert r.unfillable_mask is not None
    # the seal and what it cut off, exactly like the converged path
    assert not (r.short_shot_mask & r.unfillable_mask).any()
    assert md["short_shot_cells"] + md["unfillable_cells"] <= int(r.geometry.mask.sum())
    assert np.all(np.isnan(r.fill_time_s[r.unfillable_mask]))
    assert np.all(np.isfinite(r.fill_time_s[r.short_shot_mask]))


def test_the_domain_loop_converges_and_says_so():
    """The valve is for pathologies; a normal run reports a settled domain."""
    r = _solve(_sealed_strip())
    assert r.metadata["domain_converged"] is True
    assert r.metadata["domain_passes"] >= 1


# --- the exposure clock (Issue #61) --------------------------------------------


def test_the_skin_is_thickest_at_the_gate_and_thinnest_at_the_front():
    """The wall ages from the moment the front passes it, so the gate ages longest.

    This is the direction the arrival clock had backwards: it evaluated the
    gate-side cells at their (almost zero) arrival time and the last cell at
    the full fill time. On a uniform strip that fills without sealing the
    exposure clock gives a skin that decreases monotonically along the flow
    and vanishes at the last cell.
    """
    r = _solve(_strip(np.full(40, 1.0)), skin_growth_constant=0.5)
    assert r.unfillable_mask is None
    s = r.skin_thickness_mm[0]
    assert np.all(np.diff(s) <= 1e-12)
    assert s[0] > 10 * max(s[-1], 1e-12)
    assert s[-1] == pytest.approx(0.0, abs=1e-9)


def test_the_conductance_carries_the_service_mean_of_the_root_law():
    """One elliptic solve, one conductance per cell: the skin averaged over its service.

    For ``s(t) = c sqrt(alpha t)`` the mean over ``[0, a]`` is ``(2/3) s(a)``
    with ``a = T_fill - t_arr`` the time the cell conducts. Checked against the
    reported fill times so the test reads the same clock the solver did.
    """
    from core.solver import SKIN_SERVICE_MEAN_FACTOR

    c_skin = 0.5
    # The reported skin is the one the last solve conducted through, computed
    # from the clock one iteration back; driving the fixed point to machine
    # precision makes that clock and the reported one the same.
    r = _solve(
        _strip(np.full(40, 1.0)),
        skin_growth_constant=c_skin,
        skin_convergence_tol=1e-12,
        skin_max_iterations=60,
    )
    assert r.metadata["skin_converged"]
    alpha = MaterialDB()["PP"].thermal_diffusivity_m2_s
    service = r.total_fill_time_s - r.fill_time_s[0]
    expected = SKIN_SERVICE_MEAN_FACTOR * c_skin * np.sqrt(alpha * service) * 1e3
    assert SKIN_SERVICE_MEAN_FACTOR == pytest.approx(2.0 / 3.0)
    np.testing.assert_allclose(r.skin_thickness_mm[0], expected, rtol=1e-7, atol=1e-12)


def test_the_core_never_drops_below_a_third_of_the_wall(frozen_run):
    """A sealing cell raises resistance by at most 27x; it never hits the floor.

    The mean skin over a service that ends at ``t_close`` is ``(2/3)`` of the
    half-thickness, so ``h_core >= h/3``. That bound is what keeps the
    inflation finite where the arrival clock used to run away by orders of
    magnitude on a floored cell.
    """
    r = frozen_run
    live = r.geometry.mask & ~r.unfillable_mask
    h = r.geometry.thickness_mm
    assert r.short_shot_mask.any()
    assert np.all(r.core_thickness_mm[live] >= h[live] / 3.0 - 1e-9)
    # and the skin never eats past the floor -- the pair stays additive
    np.testing.assert_allclose(
        r.core_thickness_mm[live] + 2.0 * r.skin_thickness_mm[live], h[live], rtol=1e-12
    )


def test_sealed_cells_filled_before_they_closed(frozen_run):
    """The seal is where freeze-off happened, not where melt is missing."""
    r = frozen_run
    sealed = r.short_shot_mask
    assert sealed.any()
    assert np.all(np.isfinite(r.fill_time_s[sealed]))
    assert np.all(np.isfinite(r.pressure_norm[sealed]))
    assert not (sealed & r.unfillable_mask).any()
    # the seal sits right at the edge of what fills
    from scipy import ndimage as ndi

    assert (ndi.binary_dilation(r.unfillable_mask) & sealed).any()


def test_a_slower_fill_seals_more():
    """Exposure grows with the time the melt spends passing a cell.

    Same strip, ten times the rate: the thin band is swept in a fraction of
    its sealing age and nothing is cut off. At the slow rate the far half is.
    """
    thk = np.concatenate([np.full(30, 2.0), np.full(30, 0.05)])
    fast = _solve(_strip(thk), injection_volume_flow_cm3s=10 * Q_CM3S)
    slow = _solve(_strip(thk), injection_volume_flow_cm3s=Q_CM3S)
    assert fast.unfillable_mask is None
    assert fast.metadata["short_shot_cells"] == 0
    assert slow.unfillable_mask is not None and slow.unfillable_mask.any()
    assert slow.metadata["filled_volume_fraction"] < 1.0


def test_the_first_pass_sealing_set_does_not_leak_into_the_report():
    """Only the seal that cut the dead region off is reported from a pass that carried it.

    A pass that still holds a dead region runs on a clock inflated by that
    region's resistance, and on that clock cells far from the seal can look
    closed too. Those decisions are not the part's: the re-solve without the
    dead load makes them again. So the thick live run must not be reported as
    sealed, whatever sits behind the band.
    """
    r = _solve(_band_sealed(100))
    live_run = np.zeros_like(r.geometry.mask)
    live_run[0, :20] = True
    assert r.short_shot_mask.any()
    assert not (r.short_shot_mask & live_run).any()
    assert r.short_shot_mask[0, 20:26].any()


def test_a_route_through_cells_that_fill_later_does_not_count():
    """Codex P1 on PR #73: reachability follows the arrival order.

    Two rows joined at both ends. The target on the top row sits behind a
    choke that closes before it arrives; the bottom row loops around to it,
    but the cells of that loop arrive *after* the target does. At the target's
    arrival the loop holds no melt, so the target is cut off -- a sweep that
    kept every never-sealing cell present from the start connected it
    through the loop and kept it fillable.
    """
    mask = np.ones((2, 5), dtype=bool)
    geom = Geometry(mask=mask, thickness_mm=np.ones((2, 5)), cell_size_mm=1.0)
    geom.gates = [(0, 0)]
    solver = HeleShawSolver(geom, MaterialDB()["PP"])
    #        gate  choke  target
    t_arr = np.array([[0.0, 1.0, 2.0, 3.0, 4.0], [0.5, 6.0, 7.0, 8.0, 9.0]])
    frozen = np.zeros_like(mask)
    frozen[0, 1] = True
    t_close = np.full_like(t_arr, np.inf)
    t_close[0, 1] = 1.5  # closes before the target (0, 2) arrives at 2.0
    dead = solver._unfillable_cells(frozen, t_arr, t_close)
    assert dead[0, 2], "the target reached the gate through cells that fill later"
    assert dead[0, 3:].all()
    # the bottom row itself is fed from the gate and never sealed
    assert not dead[1].any()
    assert not dead[0, :2].any()


def test_dead_cells_carry_no_skin_and_no_core(frozen_run):
    """Codex P2 on PR #73: the solution of the part that fills says nothing about the rest.

    The restricted solve holds zeros outside its cavity, and copied straight
    into the result those zeros drew the unfillable region as a closed core.
    """
    r = frozen_run
    dead = r.unfillable_mask
    assert dead.any()
    assert np.all(np.isnan(r.skin_thickness_mm[dead]))
    assert np.all(np.isnan(r.core_thickness_mm[dead]))
    live = r.geometry.mask & ~dead
    assert np.all(np.isfinite(r.skin_thickness_mm[live]))
    assert np.all(np.isfinite(r.core_thickness_mm[live]))


def test_a_sliver_below_the_resolution_still_gets_its_own_solve():
    """Codex P1 on PR #73: every bisection candidate can be too large.

    A gate ringed by thin cells fills the gate and the ring -- a few hundredths
    of the resolution. The answer must still be solved on that sliver, not
    fall back to the cavity-wide solution with the plate's resistance in its
    clock: the baseline time is the sliver's volume over Q, and the gate's
    time is finite.
    """
    n = 24
    thk = np.full((n, n), 2.0)
    thk[2:5, 2:5] = 0.04
    thk[3, 3] = 2.0
    geom = Geometry(mask=np.ones((n, n), dtype=bool), thickness_mm=thk, cell_size_mm=1.0)
    geom.gates = [(3, 3)]
    r = _solve(geom)
    live = r.geometry.mask & ~r.unfillable_mask
    live_cm3 = float(thk[live].sum()) / 1000.0
    assert live_cm3 < 0.01 * geom.volume_cm3()
    assert r.metadata["T_fill_baseline_s"] == pytest.approx(live_cm3 / Q_CM3S, rel=1e-9)
    assert r.metadata["domain_converged"] is True
    assert np.isfinite(r.fill_time_s[3, 3])
