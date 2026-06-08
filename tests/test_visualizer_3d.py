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

from types import SimpleNamespace

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
from core.visualizer_3d import _cavity_surface_mesh


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
    cavity ceiling (Z=h mesh), and side walls (vertical mesh, optional).

    All three are now sparse ``go.Mesh3d`` traces (the floor/ceiling were
    full-grid ``go.Surface`` before the analytic-refine perf rework)."""
    floor = next(t for t in fig.data if t.name == "PL (parting line, Z=0)")
    ceiling = next(t for t in fig.data if t.name == "cavity ceiling")
    walls = next((t for t in fig.data if t.name == "cavity walls"), None)
    return floor, ceiling, walls


@pytest.mark.parametrize(
    "renderer",
    [render_3d_thickness_map, render_3d_fill_time, render_3d_pressure],
)
def test_figure_has_pl_extrusion_anatomy(small_result, renderer):
    """The figure must contain a PL floor, a colored ceiling, and side walls
    — all sparse Mesh3d traces (one ceiling/floor vertex per cavity cell)."""
    fig = renderer(small_result)
    assert isinstance(fig, go.Figure)
    floor, ceiling, walls = _split_traces(fig)
    assert isinstance(floor, go.Mesh3d)
    assert isinstance(ceiling, go.Mesh3d)
    assert isinstance(walls, go.Mesh3d)
    g = small_result.geometry
    n_cav = int(g.mask.sum())
    # sparse: exactly one ceiling/floor vertex per cavity cell (no full grid)
    assert len(ceiling.x) == n_cav
    assert len(floor.x) == n_cav
    # ceiling carries a per-vertex physics field via intensity + triangles
    assert ceiling.intensity is not None
    assert len(ceiling.intensity) == n_cav
    assert len(ceiling.i) == len(ceiling.j) == len(ceiling.k) > 0
    # walls have valid (i,j,k) triangle indices
    assert len(walls.i) == len(walls.j) == len(walls.k) > 0


def test_ceiling_floor_span_only_cavity(small_result):
    """The sparse mesh spans exactly the cavity: one vertex per cavity cell,
    ceiling heights finite-positive, floor flat at the parting line."""
    fig = render_3d_thickness_map(small_result)
    floor, ceiling, _walls = _split_traces(fig)
    n_cav = int(small_result.geometry.mask.sum())
    assert len(ceiling.z) == n_cav
    assert len(floor.z) == n_cav
    z_ceil = np.asarray(ceiling.z)
    assert np.all(np.isfinite(z_ceil))
    assert np.all(z_ceil > 0)
    assert np.all(np.asarray(floor.z) == 0.0)


def test_sparse_mesh_caps_diagonal_boundary():
    """A diagonal cavity band — every 2x2 block has only 3 cavity cells, so
    full-quad-only triangulation would emit ZERO faces (ceiling missing,
    walls floating). Three-cell corners must be capped so every cavity
    vertex is referenced by a triangle (Codex P2)."""
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
    _xs, _ys, _cell_idx, (ti, tj, tk) = _cavity_surface_mesh(result)
    referenced = set(ti.tolist()) | set(tj.tolist()) | set(tk.tolist())
    assert referenced == set(range(int(mask.sum())))


def test_vertices_are_gate_centered(small_result):
    """A cavity cell's mesh vertex sits at its gate-centered cell-center
    coordinate (the gate cell maps to the gate origin)."""
    fig = render_3d_thickness_map(small_result)
    _floor, ceiling, _walls = _split_traces(fig)
    g = small_result.geometry
    x0, y0 = g.gate_origin_mm()
    iy, ix = g.gates[0]
    exp_x = (ix + 0.5) * g.cell_size_mm - x0
    exp_y = (iy + 0.5) * g.cell_size_mm - y0
    xs = np.asarray(ceiling.x)
    ys = np.asarray(ceiling.y)
    dist = np.hypot(xs - exp_x, ys - exp_y)
    assert dist.min() < 1e-6


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
    """Walls and ceiling (both Mesh3d) must use the same coloraxis so a
    single colorbar covers the whole solid."""
    fig = render_3d_pressure(small_result)
    _floor, ceiling, walls = _split_traces(fig)
    assert ceiling.coloraxis == "coloraxis"
    assert walls.coloraxis == "coloraxis"
    # both carry per-vertex intensity equal in length to their xyz vertices
    assert ceiling.intensity is not None
    assert len(ceiling.intensity) == len(ceiling.x)
    assert walls.intensity is not None
    assert len(walls.intensity) == len(walls.x)
    # intensities finite for at least 99% of vertices on a healthy result
    assert np.isfinite(np.asarray(ceiling.intensity)).mean() > 0.99
    assert np.isfinite(np.asarray(walls.intensity)).mean() > 0.99
