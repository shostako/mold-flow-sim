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

import matplotlib.colors as mcolors
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
    geom = SimpleNamespace(mask=mask, cell_size_mm=1.0, gate_origin_mm=lambda: (0.0, 0.0))
    result = SimpleNamespace(geometry=geom)
    z = np.ones_like(mask, dtype=float)
    _xs, _ys, _zs, (ti, _tj, _tk), _fiy, _fix, _viy, _vix = _cavity_corner_mesh(result, z)
    assert len(ti) == 2 * int(mask.sum())


def test_vertices_are_gate_centered(small_result):
    """The ceiling vertices are cell corners; the gate cell's corner sits at
    its gate-centered coordinate (the frame is centered on the gate)."""
    fig = render_3d_thickness_map(small_result)
    _floor, ceiling, _walls = _split_traces(fig)
    g = small_result.geometry
    x0, y0 = g.gate_origin_mm()
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


def test_thickness_colorscale_matches_the_2d_map(small_result):
    """The solid view and the 2D design map must agree on what "thick" looks
    like. Plotly capitalizes its named colorscales, so the two constants are
    spelled differently and nothing but this test keeps them in step."""
    from core.visualizer import THICKNESS_CMAP
    from core.visualizer_3d import THICKNESS_COLORSCALE

    assert THICKNESS_COLORSCALE.lower() == THICKNESS_CMAP

    fig = render_3d_thickness_map(small_result)
    assert fig.layout.coloraxis.colorscale is not None
    # Plotly resolves the name into an (offset, css-color) table; take the
    # end points and confirm the *rendered* ramp runs light -> dark, which is
    # the whole point of the reversal. Comparing the name alone would pass
    # even if Plotly's "Cividis_r" were secretly unreversed.
    scale = fig.layout.coloraxis.colorscale
    lo_rgb = mcolors.to_rgb(scale[0][1])
    hi_rgb = mcolors.to_rgb(scale[-1][1])
    lo_lum = 0.2126 * lo_rgb[0] + 0.7152 * lo_rgb[1] + 0.0722 * lo_rgb[2]
    hi_lum = 0.2126 * hi_rgb[0] + 0.7152 * hi_rgb[1] + 0.0722 * hi_rgb[2]
    assert lo_lum > hi_lum, (
        f"thin end must be the lighter one: thin lum={lo_lum:.3f}, thick lum={hi_lum:.3f}"
    )
