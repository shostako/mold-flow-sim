"""Smoke tests for the Plotly-based 3D visualizer.

These tests do not validate visual output (Plotly figure JSON is verbose
and brittle). They confirm:
  - the 3D renderers run end-to-end on a small Hele-Shaw result,
  - they return :class:`plotly.graph_objects.Figure` instances,
  - the figure has the expected three-trace anatomy
      [PL floor (Z=0), side walls, cavity ceiling (Z=h)] — all sparse
      flat-top ``go.Mesh3d`` blocks,
  - every cavity cell is rendered (no boundary erosion),
  - thickness steps render as crisp block steps (not smoothed ramps),
  - the axes are gate-centered.

Phase 1-2 (animation frames) will get its own dedicated test file.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import plotly.graph_objects as go
import pytest

from core import (
    DirectGateConfig,
    HeleShawSolver,
    MaterialDB,
    build_demo_geometry,
    build_direct_gate_geometry,
    render_3d_fill_time,
    render_3d_pressure,
    render_3d_thickness_map,
)
from core.visualizer_3d import _cavity_corner_mesh
from tests.colorimetry import contrast_ratio, css_rgb, relative_luminance, sample_ramp


@pytest.fixture(scope="module")
def small_result():
    """Solve a tiny demo case once and reuse for all tests in this module."""
    geom = build_demo_geometry(cell_size_mm=2.0, plate_thk_mm=2.0, gate_count=1)
    solver = HeleShawSolver(
        geometry=geom,
        material=MaterialDB()["PP"],
        injection_volume_flow_cm3s=20.0,
    )
    return solver.solve(num_frames=4)


def _split_traces(
    fig: go.Figure,
) -> tuple[go.Mesh3d, go.Mesh3d, go.Mesh3d | None]:
    """Identify the three expected traces by name: PL floor (Z=0 mesh),
    cavity ceiling (Z=h flat-top mesh), and side walls (vertical mesh)."""
    floor = next(t for t in fig.data if t.name == "PL (parting line, Z=0)")
    ceiling = next(t for t in fig.data if t.name == "cavity ceiling")
    walls = next((t for t in fig.data if t.name == "cavity walls"), None)
    return floor, ceiling, walls


@pytest.mark.parametrize(
    "renderer",
    [render_3d_thickness_map, render_3d_fill_time, render_3d_pressure],
)
def test_figure_has_block_anatomy(small_result, renderer):
    """PL floor + colored ceiling + side walls, all flat-top Mesh3d blocks.
    The ceiling is one flat quad (2 triangles) per cavity cell, colored
    per-face (``intensitymode='cell'``)."""
    fig = renderer(small_result)
    assert isinstance(fig, go.Figure)
    floor, ceiling, walls = _split_traces(fig)
    assert isinstance(floor, go.Mesh3d)
    assert isinstance(ceiling, go.Mesh3d)
    assert isinstance(walls, go.Mesh3d)
    n_cav = int(small_result.geometry.mask.sum())
    # every cavity cell -> exactly 2 ceiling triangles (flat-top block)
    assert len(ceiling.i) == len(ceiling.j) == len(ceiling.k) == 2 * n_cav
    # color is per face (one value per triangle)
    assert ceiling.intensitymode == "cell"
    assert ceiling.intensity is not None
    assert len(ceiling.intensity) == 2 * n_cav
    # walls have valid (i,j,k) triangle indices
    assert len(walls.i) == len(walls.j) == len(walls.k) > 0


def test_ceiling_covers_every_cavity_cell(small_result):
    """Flat-top blocks cap *every* cavity cell (no boundary erosion):
    2 faces per cell, ceiling heights finite-positive, floor flat at PL."""
    fig = render_3d_thickness_map(small_result)
    floor, ceiling, _walls = _split_traces(fig)
    n_cav = int(small_result.geometry.mask.sum())
    assert len(ceiling.i) == 2 * n_cav
    z_ceil = np.asarray(ceiling.z)
    assert np.all(np.isfinite(z_ceil))
    assert np.all(z_ceil > 0)
    assert np.all(np.asarray(floor.z) == 0.0)


def test_flat_top_caps_diagonal_boundary():
    """A diagonal cavity band — the old cell-center triangulation could not
    cap it (3-cell 2x2 blocks). Flat-top blocks give every cavity cell its
    own quad, so coverage is total: faces == 2 * cavity cells."""
    mask = np.array(
        [
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
        ],
        dtype=bool,
    )
    geom = SimpleNamespace(mask=mask, cell_size_mm=1.0, display_origin_mm=lambda: (0.0, 0.0))
    result = SimpleNamespace(geometry=geom)
    z = np.ones_like(mask, dtype=float)
    _xs, _ys, _zs, (ti, _tj, _tk), _fiy, _fix, _viy, _vix = _cavity_corner_mesh(result, z)
    assert len(ti) == 2 * int(mask.sum())


def test_vertices_use_the_display_origin(small_result):
    """The ceiling vertices are cell corners; the gate cell's corner sits at
    its coordinate in the product-referenced display frame (x on the valve
    axis, y from the product's bottom edge)."""
    fig = render_3d_thickness_map(small_result)
    _floor, ceiling, _walls = _split_traces(fig)
    g = small_result.geometry
    x0, y0 = g.display_origin_mm()
    iy, ix = g.gates[0]
    # the gate cell's top-left corner (cell edge, not center)
    exp_x = ix * g.cell_size_mm - x0
    exp_y = iy * g.cell_size_mm - y0
    xs = np.asarray(ceiling.x)
    ys = np.asarray(ceiling.y)
    dist = np.hypot(xs - exp_x, ys - exp_y)
    assert dist.min() < 1e-6


def test_walls_span_pl_to_ceiling(small_result):
    """Boundary-wall vertices range from Z=0 (PL) up to the local cavity
    height; no wall vertex exceeds the global h_max."""
    fig = render_3d_thickness_map(small_result)
    _floor, _ceiling, walls = _split_traces(fig)
    z_walls = np.asarray(walls.z)
    g = small_result.geometry
    h_max = float(np.nanmax(g.thickness_mm[g.mask]))
    assert z_walls.min() == 0.0  # boundary walls start at PL
    assert z_walls.max() <= h_max + 1e-9
    assert z_walls.max() > 0.0


def test_aspectmode_is_data(small_result):
    """All 3 axes share the same mm scale (aspectmode='data')."""
    fig = render_3d_thickness_map(small_result)
    assert fig.layout.scene.aspectmode == "data"


@pytest.mark.parametrize("renderer", [render_3d_fill_time, render_3d_pressure])
def test_ceiling_hover_exposes_field_value(small_result, renderer):
    """The ceiling hover must still expose the mapped field value (per-vertex
    ``customdata``) so it is readable numerically, not only off the colorbar
    (the flat-top per-face colour dropped the old per-vertex readout)."""
    fig = renderer(small_result)
    _floor, ceiling, _walls = _split_traces(fig)
    assert ceiling.customdata is not None
    cd = np.asarray(ceiling.customdata, dtype=float).ravel()
    assert len(cd) == len(ceiling.x)  # one value per vertex
    assert np.isfinite(cd).mean() > 0.99
    assert "customdata" in (ceiling.hovertemplate or "")


def test_walls_share_ceiling_coloraxis(small_result):
    """Walls and ceiling share ``coloraxis`` so one colorbar covers the
    solid. Ceiling colour is per-face, walls colour is per-vertex."""
    fig = render_3d_pressure(small_result)
    _floor, ceiling, walls = _split_traces(fig)
    assert ceiling.coloraxis == "coloraxis"
    assert walls.coloraxis == "coloraxis"
    # ceiling: per-face intensity (one value per triangle)
    assert ceiling.intensity is not None
    assert len(ceiling.intensity) == len(ceiling.i)
    # walls: per-vertex intensity
    assert walls.intensity is not None
    assert len(walls.intensity) == len(walls.x)
    assert np.isfinite(np.asarray(ceiling.intensity)).mean() > 0.99
    assert np.isfinite(np.asarray(walls.intensity)).mean() > 0.99


def test_thickness_step_renders_as_block_step():
    """A stepped plate (gate-side 0.35 / far-side 0.50) must render the step
    faithfully: the ceiling carries *both* thicknesses (not an averaged
    ramp), and the vertical step face is closed by an internal step wall
    (a wall quad that does not touch the parting line)."""
    cfg = DirectGateConfig(
        plate_w_mm=60.0,
        plate_h_mm=50.0,
        plate_thk_mm=0.5,
        gate_diameter_mm=3.0,
        gate_offset_mm=20.0,
        cell_size_mm=0.5,
        plate_split_height_mm=20.0,
        plate_lower_thk_mm=0.35,
        plate_upper_thk_mm=0.50,
    )
    geom = build_direct_gate_geometry(cfg)
    result = HeleShawSolver(
        geometry=geom,
        material=MaterialDB()["PP"],
        injection_volume_flow_cm3s=20.0,
    ).solve(num_frames=4)
    fig = render_3d_thickness_map(result)
    _floor, ceiling, walls = _split_traces(fig)
    # both step thicknesses present in the ceiling, unaveraged
    zc = np.round(np.asarray(ceiling.z), 3)
    assert 0.35 in zc and 0.5 in zc
    # at least one internal step wall: a 4-vertex quad with min z above PL
    zw = np.asarray(walls.z).reshape(-1, 4)
    assert np.any(zw.min(axis=1) > 1e-6), "no internal step wall emitted"


@pytest.mark.parametrize(
    ("stop", "expected"),
    [
        ("#fee838", (254 / 255, 232 / 255, 56 / 255)),
        ("rgb(254, 232, 56)", (254 / 255, 232 / 255, 56 / 255)),
        ("rgb(0,0,0)", (0.0, 0.0, 0.0)),
        ("rgba(255, 255, 255, 0.5)", (1.0, 1.0, 1.0)),
    ],
)
def test_css_rgb_parses_both_plotly_stop_spellings(stop, expected):
    """Guards the parser the luminance test depends on. Without the functional
    form, swapping in any of the 81 ``rgb(...)``-spelled built-ins turns a
    colormap verdict into an unrelated-looking ValueError."""
    assert css_rgb(stop) == pytest.approx(expected, abs=1e-9)


def test_relative_luminance_linearizes_srgb():
    """Guards the metric the monotonicity tests depend on.

    ``0.2126R + 0.7152G + 0.0722B`` applied to gamma-encoded sRGB is luma, not
    luminance, and the two disagree in *sign* when a palette trades intensity
    between channels. This pair is the concrete case: naive luma calls it a
    darkening step, relative luminance calls it a brightening one. Getting it
    wrong would wave through a replacement ramp that actually gets lighter.
    """
    darker_looking = css_rgb("rgb(205,78,46)")
    brighter = css_rgb("rgb(242,34,36)")

    def luma(rgb):
        return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]

    assert luma(brighter) < luma(darker_looking)  # the wrong answer
    assert relative_luminance(brighter) > relative_luminance(darker_looking)

    # anchors: the transfer function must not disturb black or white
    assert relative_luminance((0.0, 0.0, 0.0)) == pytest.approx(0.0)
    assert relative_luminance((1.0, 1.0, 1.0)) == pytest.approx(1.0)


def test_thickness_colorscale_matches_the_2d_map(small_result):
    """The solid view and the 2D design map must agree on what "thick" looks
    like. Plotly capitalizes its named colorscales, so the two constants are
    spelled differently and nothing but this test keeps them in step."""
    from core.visualizer import THICKNESS_CMAP
    from core.visualizer_3d import THICKNESS_COLORSCALE

    assert THICKNESS_COLORSCALE.lower() == THICKNESS_CMAP

    fig = render_3d_thickness_map(small_result)
    assert fig.layout.coloraxis.colorscale is not None
    # Plotly resolves the name into an (offset, css-color) table. Confirm the
    # *rendered* ramp darkens monotonically, which is the whole point of the
    # reversal. Comparing the name alone would pass even if Plotly's
    # "Cividis_r" were secretly unreversed.
    #
    # Monotone across the whole ramp, not merely light-at-0 and dark-at-1: a
    # two-endpoint check is fooled by 26 of Plotly's 188 built-in scales.
    # ``jet_r`` is the worst — its endpoints differ by only 0.070 luminance
    # while the middle swings back up by 0.719, so it reads as "light to dark"
    # and is in fact a rainbow. A thickness map has to let the reader rank two
    # thicknesses by darkness alone; where luminance reverses, two different
    # thicknesses share a darkness and the ordering stops being recoverable.
    #
    # Sampled between the stops, not only at them: Plotly interpolates in
    # gamma-encoded sRGB while relative luminance linearizes first, so
    # luminance is not linear along a segment and stop-only monotonicity does
    # not imply monotonicity between stops: the transfer function is convex, so
    # luminance can sag below the darker endpoint and climb back inside a
    # segment. ``bluered_r`` exhibits it today (0.0023), and
    # ``test_stop_only_check_misses_a_within_segment_reversal`` pins a
    # constructed witness that does not depend on any palette staying put.
    ramp = sample_ramp(list(fig.layout.coloraxis.colorscale))
    lums = [relative_luminance(rgb) for rgb in ramp]
    drops = [b - a for a, b in zip(lums, lums[1:])]
    assert all(d < 0 for d in drops), (
        "thickness ramp must darken monotonically (thin=light, thick=dark); "
        f"{sum(1 for d in drops if d >= 0)} of {len(drops)} samples do not darken"
    )
    # And it must actually carry information. Monotone alone is satisfied by a
    # near-black-to-black ramp, which orders thicknesses in principle while
    # rendering the map unreadable. WCAG 1.4.11 asks 3:1 of graphical objects
    # that convey meaning; the thin/thick ends here clear that by a wide
    # margin (cividis_r is 12.6:1).
    ends_ratio = contrast_ratio(ramp[0], ramp[-1])
    assert ends_ratio >= 3.0, (
        f"thin and thick ends must be separable: contrast ratio {ends_ratio:.2f}:1 < 3:1"
    )


@pytest.mark.parametrize(
    ("scale", "monotone", "separable"),
    [
        ("Cividis_r", True, True),  # the map in force
        ("Blues", True, True),  # right direction, washes out the thin end
        ("Blues_r", False, True),  # inverted
        # A rainbow: the ends read light->dark, the middle does not — and its
        # ends are barely separable either (1.44:1), so both criteria object.
        ("Jet_r", False, False),
        # Reverses only *between* stops, and spans too little to read (2.15:1).
        ("Bluered_r", False, False),
        ("Greys_r", False, True),  # inverted single-hue
    ],
)
def test_ramp_criteria_discriminate_known_scales(scale, monotone, separable):
    """Pins what the two criteria in the thickness test actually reject.

    Exercising them through ``THICKNESS_COLORSCALE`` cannot work — the name
    equality assertion fires first — so the criteria are pinned here against
    named scales whose verdicts are known. ``Bluered_r`` is the one that only
    a *sampled* ramp catches; ``Jet_r`` is the one that only a *whole-ramp*
    check catches.

    These expectations describe Plotly's palettes, not the criteria. If one of
    them fails because Plotly redefined a scale, **update the expectation** —
    re-measure the scale and record what it now is. Do not loosen a criterion
    to make the row pass: the criteria answer to what a thickness map has to
    do (order thicknesses by darkness, carry enough contrast to read), which
    no upstream palette change can alter.
    """
    fig = go.Figure()
    fig.update_layout(coloraxis=dict(colorscale=scale))
    ramp = sample_ramp(list(fig.layout.coloraxis.colorscale))
    lums = [relative_luminance(rgb) for rgb in ramp]

    assert all(b < a for a, b in zip(lums, lums[1:])) is monotone
    assert (contrast_ratio(ramp[0], ramp[-1]) >= 3.0) is separable


def test_stop_only_check_misses_a_within_segment_reversal():
    """Why the ramp is sampled rather than read at its stops.

    The need is mathematical, not empirical: Plotly interpolates linearly
    between stops in *gamma-encoded* sRGB, while relative luminance applies
    the (convex) sRGB transfer function first. Luminance along a segment is
    therefore a sum of convex functions of a linear argument — convex, not
    linear — so it can dip below the darker endpoint and climb back up while
    the two endpoints still read light-to-dark.

    The witness here is **constructed** from that fact rather than borrowed
    from a built-in palette, so it cannot rot: green -> magenta reads 0.377 ->
    0.285 at the stops and sags to 0.142 in between. ``Bluered_r`` happens to
    do the same thing in the wild today (by 0.0023) and is pinned in the
    discrimination table above, but if Plotly ever redefines it, that only
    retires one example — it does not make sampling unnecessary. Do not delete
    ``sample_ramp`` on the strength of a palette changing.
    """
    scale = [(0.0, "rgb(0,192,0)"), (1.0, "rgb(255,0,255)")]

    at_stops = [relative_luminance(css_rgb(color)) for _offset, color in scale]
    sampled = [relative_luminance(rgb) for rgb in sample_ramp(scale)]

    assert all(b < a for a, b in zip(at_stops, at_stops[1:])), (
        "the endpoints must look monotone, or this witness proves nothing"
    )
    assert min(sampled) < at_stops[-1], "luminance must sag below the darker endpoint"
    assert not all(b < a for a, b in zip(sampled, sampled[1:])), (
        "sampling must reject what the stops accept"
    )
