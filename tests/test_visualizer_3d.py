"""Smoke tests for the Plotly-based 3D visualizer.

These tests do not validate visual output (Plotly figure JSON is verbose
and brittle). They confirm:
  - the 3D renderers run end-to-end on a small Hele-Shaw result,
  - they return :class:`plotly.graph_objects.Figure` instances, and
  - the figure has the expected three-trace anatomy:
      [PL floor (Z=0), side walls (Mesh3d), cavity ceiling (Z=h)]
  - outside-cavity cells are masked correctly,
  - the axes are gate-centered.

Phase 1-2 (animation frames) will get its own dedicated test file.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest

from core import (
    HeleShawSolver,
    MaterialDB,
    build_demo_geometry,
    render_3d_fill_time,
    render_3d_pressure,
    render_3d_thickness_map,
)


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


def _split_traces(fig: go.Figure) -> tuple[go.Surface, go.Surface, go.Mesh3d | None]:
    """Identify the three expected traces in deterministic order:
    PL floor (Z=0 surface), ceiling (top Surface), and walls (Mesh3d, optional)."""
    surfaces = [t for t in fig.data if isinstance(t, go.Surface)]
    meshes = [t for t in fig.data if isinstance(t, go.Mesh3d)]
    assert len(surfaces) == 2, f"expected 2 Surface traces (PL+ceiling), got {len(surfaces)}"
    # the floor has all-zero Z where mask=True; the ceiling has h where mask=True
    floor = next(s for s in surfaces if np.nanmax(np.asarray(s.z)) <= 0.0)
    ceiling = next(s for s in surfaces if s is not floor)
    walls = meshes[0] if meshes else None
    return floor, ceiling, walls


@pytest.mark.parametrize(
    "renderer",
    [render_3d_thickness_map, render_3d_fill_time, render_3d_pressure],
)
def test_figure_has_pl_extrusion_anatomy(small_result, renderer):
    """The figure must contain a PL floor, a colored ceiling, and side walls."""
    fig = renderer(small_result)
    assert isinstance(fig, go.Figure)
    floor, ceiling, walls = _split_traces(fig)
    # floor and ceiling share the mask shape
    g = small_result.geometry
    assert np.asarray(floor.z).shape == g.mask.shape
    assert np.asarray(ceiling.z).shape == g.mask.shape
    # ceiling carries the physics field as surfacecolor
    assert np.asarray(ceiling.surfacecolor).shape == g.mask.shape
    # walls trace exists and has valid (i,j,k) triangle indices
    assert walls is not None
    assert len(walls.i) == len(walls.j) == len(walls.k)
    assert len(walls.i) > 0


def test_outside_cavity_cells_are_masked(small_result):
    """Both floor and ceiling Z values must be NaN outside the cavity."""
    fig = render_3d_thickness_map(small_result)
    floor, ceiling, _walls = _split_traces(fig)
    outside = ~small_result.geometry.mask
    inside = small_result.geometry.mask
    # ceiling: NaN outside, finite-positive inside
    z_ceiling = np.asarray(ceiling.z)
    assert np.all(np.isnan(z_ceiling[outside]))
    assert np.all(np.isfinite(z_ceiling[inside]))
    assert np.all(z_ceiling[inside] > 0)
    # floor: NaN outside, exactly 0 inside
    z_floor = np.asarray(floor.z)
    assert np.all(np.isnan(z_floor[outside]))
    assert np.all(z_floor[inside] == 0.0)


def test_axes_are_gate_centered(small_result):
    """The x/y coordinate arrays should be centered on the gate origin."""
    fig = render_3d_thickness_map(small_result)
    _floor, ceiling, _walls = _split_traces(fig)
    x = np.asarray(ceiling.x)
    y = np.asarray(ceiling.y)
    g = small_result.geometry
    x0, y0 = g.gate_origin_mm()
    expected_x0 = (0 + 0.5) * g.cell_size_mm - x0
    expected_y0 = (0 + 0.5) * g.cell_size_mm - y0
    assert np.isclose(x[0], expected_x0)
    assert np.isclose(y[0], expected_y0)


def test_walls_span_pl_to_ceiling(small_result):
    """Side-wall vertices should range from Z=0 (PL) up to the local
    cavity height. No wall vertex may sit above the global h_max."""
    fig = render_3d_thickness_map(small_result)
    _floor, _ceiling, walls = _split_traces(fig)
    z_walls = np.asarray(walls.z)
    g = small_result.geometry
    h_max = float(np.nanmax(g.thickness_mm[g.mask]))
    assert z_walls.min() == 0.0  # walls start at PL
    assert z_walls.max() <= h_max + 1e-9  # walls don't exceed global ceiling
    assert z_walls.max() > 0.0  # there is at least one non-degenerate wall


def test_aspectmode_is_data(small_result):
    """All 3 axes must share the same mm scale (aspectmode='data'),
    so the user reads true geometric proportions off the plot."""
    fig = render_3d_thickness_map(small_result)
    assert fig.layout.scene.aspectmode == "data"


def test_walls_share_ceiling_coloraxis(small_result):
    """Walls must use the same coloraxis as the ceiling so that a single
    colorbar covers ceiling+walls and the user reads them as one solid."""
    fig = render_3d_pressure(small_result)
    _floor, ceiling, walls = _split_traces(fig)
    assert ceiling.coloraxis == "coloraxis"
    assert walls.coloraxis == "coloraxis"
    # Wall vertices must carry an intensity array equal in length to xyz
    assert walls.intensity is not None
    assert len(walls.intensity) == len(walls.x)
    # Intensity values should be finite for at least 99% of vertices
    # (there can be NaN cells if the field happens to be undefined for
    # some boundary cell, but that should be rare on a healthy result)
    intensity = np.asarray(walls.intensity)
    finite = np.isfinite(intensity)
    assert finite.mean() > 0.99
