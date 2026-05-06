"""Smoke tests for the Plotly-based 3D visualizer.

These tests do not validate visual output (Plotly figure JSON is verbose
and brittle). They confirm:
  - the 3D renderers run end-to-end on a small Hele-Shaw result,
  - they return :class:`plotly.graph_objects.Figure` instances, and
  - the surface mesh dimensions match the underlying mask shape.

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


@pytest.mark.parametrize(
    "renderer",
    [render_3d_thickness_map, render_3d_fill_time, render_3d_pressure],
)
def test_renderer_returns_figure(small_result, renderer):
    fig = renderer(small_result)
    assert isinstance(fig, go.Figure)
    # exactly one Surface trace
    surfaces = [t for t in fig.data if isinstance(t, go.Surface)]
    assert len(surfaces) == 1
    surf = surfaces[0]
    # Z (height) shape == mask shape
    z_shape = np.asarray(surf.z).shape
    assert z_shape == small_result.geometry.mask.shape
    # surface color array shape matches as well
    assert np.asarray(surf.surfacecolor).shape == z_shape


def test_outside_cavity_cells_are_nan(small_result):
    """Cells with mask=False should be NaN in Z so plotly skips them."""
    fig = render_3d_thickness_map(small_result)
    surf = fig.data[0]
    z = np.asarray(surf.z)
    outside_mask = ~small_result.geometry.mask
    # All outside-cavity Z values must be NaN
    assert np.all(np.isnan(z[outside_mask]))
    # All in-cavity Z values must be finite and positive
    z_in = z[small_result.geometry.mask]
    assert np.all(np.isfinite(z_in))
    assert np.all(z_in > 0)


def test_axes_are_gate_centered(small_result):
    """The x/y coordinate arrays should be centered on the gate origin."""
    fig = render_3d_thickness_map(small_result)
    surf = fig.data[0]
    x = np.asarray(surf.x)
    y = np.asarray(surf.y)
    g = small_result.geometry
    x0, y0 = g.gate_origin_mm()
    # First cell center in gate-centered frame
    expected_x0 = (0 + 0.5) * g.cell_size_mm - x0
    expected_y0 = (0 + 0.5) * g.cell_size_mm - y0
    assert np.isclose(x[0], expected_x0)
    assert np.isclose(y[0], expected_y0)
