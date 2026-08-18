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
    assert md["tau_max"] / md["tau_max_flow"] > 10.0
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


def test_the_tau_reference_falls_back_when_no_cell_is_left():
    """Reducing over an empty selection would be a NaN, and NaN divides badly."""
    solver = HeleShawSolver(_strip(np.full(4, 1.0)), MaterialDB()["PP"])
    tau = np.array([[0.0, 1.0, 2.0, 3.0]])
    assert solver._tau_reference(tau, np.zeros_like(tau, dtype=bool), 7.0) == 7.0
    # a selection that only holds zeros is just as unusable as an empty one
    assert solver._tau_reference(tau, np.array([[True, False, False, False]]), 7.0) == 7.0


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

    raw_traps = solver._compute_air_traps(r.tau)
    raw_weld = solver._compute_weld_score(r.tau)
    assert (raw_traps & dead).any(), "geometry no longer plants a false air trap"
    assert ((raw_weld > 0.0) & dead).any(), "geometry no longer plants a false weld"

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
