"""Tests for the fill-front rendering choices (palette, isochrones, smoothing).

The fill-time field is continuous, so interpolating its *colors* between
cell centers is honest. The cavity outline and the melt front are not — they
are exactly where the mask says they are — so those must stay cell-exact.
These tests pin that split, which is the same principle the 3D renderer
follows when it draws thickness steps as flat-top blocks.
"""

from __future__ import annotations

import numpy as np
import pytest

from core import HeleShawSolver, MaterialDB, build_demo_geometry, export_frames
from core.visualizer import (
    FILL_CMAP,
    _draw_fill_state,
    _draw_gate_markers,
    _draw_isochrones,
    _fill_field_rgb,
    _nearest_extend,
    _unfilled_overlay,
    fill_frame_times,
    fill_time_max,
    render_fill_animation,
)


@pytest.fixture(scope="module")
def result():
    geom = build_demo_geometry(cell_size_mm=3.0)
    return HeleShawSolver(geom, MaterialDB()["PP"]).solve(num_frames=6)


# --- palette ----------------------------------------------------------------


def test_default_palette_is_the_engineered_rainbow():
    """turbo, not jet: same rainbow look without jet's false banding."""
    assert FILL_CMAP == "turbo"


def test_field_colors_span_the_palette_ends(result):
    """The scale is anchored at 0..T_fill, so both ends of the map are used."""
    import matplotlib.pyplot as plt

    rgba = _fill_field_rgb(result, FILL_CMAP)
    cmap = plt.get_cmap(FILL_CMAP)
    g = result.geometry
    ft = result.fill_time_s
    t_max = fill_time_max(result)
    first = np.unravel_index(np.nanargmin(np.where(g.mask, ft, np.inf)), ft.shape)
    last = np.unravel_index(np.nanargmax(np.where(g.mask, ft, -np.inf)), ft.shape)
    assert np.allclose(rgba[first][:3], cmap(float(ft[first]) / t_max)[:3], atol=1e-6)
    assert np.allclose(rgba[last][:3], cmap(float(ft[last]) / t_max)[:3], atol=1e-6)


def test_field_layer_is_fully_opaque_so_it_can_be_interpolated(result):
    """Alpha must not carry the mask: bilinear on alpha would smear the edge."""
    rgba = _fill_field_rgb(result, FILL_CMAP)
    assert np.all(rgba[..., 3] == 1.0)


def test_field_has_no_nan_outside_the_cavity(result):
    """Extended, not left as NaN — otherwise the boundary blends to garbage."""
    rgba = _fill_field_rgb(result, FILL_CMAP)
    assert np.isfinite(rgba).all()


# --- nearest extension ------------------------------------------------------


def test_nearest_extend_keeps_inside_values_untouched():
    values = np.arange(12.0).reshape(3, 4)
    mask = np.zeros((3, 4), dtype=bool)
    mask[1, 1:3] = True
    out = _nearest_extend(values, mask)
    assert out[1, 1] == values[1, 1]
    assert out[1, 2] == values[1, 2]


def test_nearest_extend_fills_outside_from_the_closest_inside_cell():
    values = np.zeros((3, 5))
    values[1, 0] = 7.0
    values[1, 4] = 9.0
    mask = np.zeros((3, 5), dtype=bool)
    mask[1, 0] = mask[1, 4] = True
    out = _nearest_extend(values, mask)
    assert out[1, 1] == 7.0  # nearer the left seed
    assert out[1, 3] == 9.0  # nearer the right seed
    assert set(np.unique(out)) == {7.0, 9.0}


def test_nearest_extend_is_a_noop_when_everything_is_inside():
    values = np.arange(6.0).reshape(2, 3)
    out = _nearest_extend(values, np.ones((2, 3), dtype=bool))
    assert np.array_equal(out, values)


# --- melt front stays cell-exact --------------------------------------------


def test_overlay_reveals_exactly_the_filled_cells(result):
    """The visible region is the mask, cell for cell — no erosion, no bleed."""
    g = result.geometry
    t = fill_frame_times(result, 6)[2]
    filled = result.fill_time_s <= t
    overlay = _unfilled_overlay(result, filled)
    visible = overlay[..., 3] == 0.0
    assert np.array_equal(visible, g.mask & filled)


def test_overlay_hides_everything_before_the_first_frame(result):
    overlay = _unfilled_overlay(result, np.zeros_like(result.geometry.mask))
    assert np.all(overlay[..., 3] == 1.0)


def test_overlay_distinguishes_outside_from_unfilled_cavity(result):
    """Two different grays, so an empty cavity still reads as the part."""
    g = result.geometry
    overlay = _unfilled_overlay(result, np.zeros_like(g.mask))
    outside = overlay[~g.mask][0, :3]
    unfilled = overlay[g.mask][0, :3]
    assert not np.allclose(outside, unfilled)
    assert unfilled.mean() < outside.mean()  # cavity is the darker gray


# --- isochrones -------------------------------------------------------------


def test_isochrone_levels_are_fixed_to_the_total_fill_time(result):
    """Levels come from T_fill, so every frame shows the same contour set.

    Per-frame levels would make the contours crawl as the front advances,
    which reads as flow that is not there.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    cs = _draw_isochrones(ax, result, 8)
    t_max = fill_time_max(result)
    assert np.all(cs.levels > 0.0)
    assert np.all(cs.levels < t_max)
    plt.close(fig)


def test_isochrones_are_drawn_once_and_stacked_under_the_overlay(result):
    """No per-frame contour churn, and the overlay really covers the lines.

    Removing and redrawing the contours each frame needed
    ``ContourSet.remove()``, which only exists from Matplotlib 3.8 (the
    package floor is 3.7) and cost one contour pass per frame. Drawing them
    across the whole cavity *under* the opaque unfilled overlay hides them
    at the melt front for free.

    Asserted on ``zorder``, not on call order: matplotlib defaults images to
    0 and contours to 2, so drawing the overlay last is not enough — the
    first version of this passed a call-order check while the rendered PNG
    showed isochrones bleeding across the unfilled region.
    """
    import matplotlib.pyplot as plt
    from matplotlib.contour import ContourSet
    from matplotlib.image import AxesImage

    fig, ax = plt.subplots()
    _draw_fill_state(
        ax,
        result,
        _fill_field_rgb(result, FILL_CMAP),
        result.fill_time_s <= fill_time_max(result) * 0.5,
        smooth=True,
        isochrone_levels=8,
    )
    images = [c for c in ax.get_children() if isinstance(c, AxesImage)]
    contours = [c for c in ax.get_children() if isinstance(c, ContourSet)]
    assert len(images) == 2, "expected the color field and the unfilled overlay"
    assert len(contours) == 1, "isochrones must be drawn exactly once"
    field_z, overlay_z = sorted(im.get_zorder() for im in images)
    assert field_z < contours[0].get_zorder() < overlay_z
    plt.close(fig)


def test_isochrones_cover_the_whole_cavity_not_just_the_filled_part(result):
    """They are clipped by paint, not by data, so the same set serves all frames."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    early = _draw_isochrones(ax, result, 8)
    late = _draw_isochrones(ax, result, 8)
    assert np.array_equal(early.levels, late.levels)
    plt.close(fig)


def test_isochrones_can_be_switched_off(result):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    assert _draw_isochrones(ax, result, 0) is None
    assert _draw_isochrones(ax, result, -1) is None
    assert _draw_isochrones(ax, result, 1) is not None  # 1 means one line, not none
    plt.close(fig)


# --- renderers still produce files ------------------------------------------


def test_animation_renders_with_isochrones_and_smoothing(result, tmp_path):
    out = render_fill_animation(result, tmp_path / "fill.gif", num_frames=4, fps=4)
    assert out.exists() and out.stat().st_size > 0


def test_animation_renders_with_every_offered_palette(result, tmp_path):
    for cmap in ("turbo", "jet", "viridis", "cividis"):
        out = render_fill_animation(
            result, tmp_path / f"{cmap}.gif", num_frames=2, fps=4, cmap=cmap
        )
        assert out.exists() and out.stat().st_size > 0


def test_frames_carry_their_own_colorbar(result, tmp_path):
    """The player has no chrome around it, so each PNG must be self-explaining.

    Checked by width: adding the colorbar widens the figure's used area, so a
    frame with one is wider than the same figure without.
    """
    from PIL import Image

    paths = export_frames(result, tmp_path / "frames", num_frames=2)
    with Image.open(paths[0]) as im:
        w, h = im.width, im.height
    assert w > h  # 7x5 figure plus the colorbar stays landscape
    assert len(paths) == 2


# --- Codex review follow-ups ------------------------------------------------


@pytest.mark.parametrize("requested", [1, 3, 12, 24])
def test_isochrone_count_matches_the_request(result, requested):
    """Asking for N lines puts N lines on the plot.

    The UI labels this "等時線の本数", so an off-by-one here is a lie to the
    reader — and the earlier guard silently drew nothing for a request of 1.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    cs = _draw_isochrones(ax, result, requested)
    assert len(cs.levels) == requested
    plt.close(fig)


def test_isochrones_skip_a_cavity_too_narrow_to_contour(result):
    """A one-cell-wide cavity solves fine; it must not kill the render.

    ``contour`` needs a 2x2 neighbourhood. A shape small enough, or a mesh
    coarse enough, to leave the grid one cell across still produces a valid
    geometry, and throwing away a finished analysis over a decoration would
    be the wrong trade. The slice keeps a row that actually contains cavity
    cells — an empty row would hit the all-NaN guard instead and never
    exercise this one.
    """
    import dataclasses

    import matplotlib.pyplot as plt

    g = result.geometry
    row = int(np.argmax(g.mask.sum(axis=1)))
    sl = slice(row, row + 1)
    assert g.mask[sl].any(), "the sliced row must contain cavity cells"
    thin = dataclasses.replace(
        result,
        geometry=dataclasses.replace(g, mask=g.mask[sl], thickness_mm=g.thickness_mm[sl]),
        fill_time_s=result.fill_time_s[sl],
    )
    assert np.isfinite(thin.fill_time_s[thin.geometry.mask]).any(), (
        "the row must carry finite fill times, or the all-NaN guard fires first"
    )
    fig, ax = plt.subplots()
    assert _draw_isochrones(ax, thin, 12) is None
    plt.close(fig)


def test_gate_marker_sits_above_the_unfilled_overlay(result, tmp_path):
    """A boundary gate must stay visible in the very first frame.

    The marker is drawn several cells wide, so with matplotlib's default line
    z-order of 2 the opaque overlay (3) eats everything outside the single
    filled gate cell — worst exactly when the animation starts.

    Checked without passing ``zorder``: it is the helper's default, so a
    renderer cannot forget it. An earlier version of this test passed the
    value in itself and therefore proved nothing about the call sites.
    """
    import matplotlib.pyplot as plt
    from matplotlib.image import AxesImage
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots()
    _draw_fill_state(
        ax,
        result,
        _fill_field_rgb(result, FILL_CMAP),
        np.zeros_like(result.geometry.mask),
        smooth=True,
        isochrone_levels=8,
    )
    _draw_gate_markers(ax, result)
    overlay_z = max(im.get_zorder() for im in ax.get_children() if isinstance(im, AxesImage))
    markers = [c for c in ax.get_children() if isinstance(c, Line2D) and c.get_marker() == "o"]
    assert markers, "expected at least one gate marker"
    assert all(m.get_zorder() > overlay_z for m in markers)
    plt.close(fig)


def test_frame_export_contours_once_for_the_whole_sequence(result, tmp_path, monkeypatch):
    """60 PNGs must not mean 60 contour passes over the cavity.

    The GIF renderer builds the figure once and only swaps the overlay array;
    the PNG exporter used to rebuild the whole figure per frame, so the
    one-pass contour optimization was lost exactly where the UI asks for the
    most frames. Counting the contour calls is the only way to see this — the
    output PNGs look identical either way.
    """
    from matplotlib.axes import Axes

    calls = 0
    real_contour = Axes.contour

    def counting_contour(self, *a, **kw):
        nonlocal calls
        calls += 1
        return real_contour(self, *a, **kw)

    monkeypatch.setattr(Axes, "contour", counting_contour)
    paths = export_frames(result, tmp_path / "frames", num_frames=5, isochrone_levels=8)
    assert len(paths) == 5
    assert calls == 1, f"contoured {calls} times for 5 frames"
