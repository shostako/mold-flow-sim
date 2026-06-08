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

import dataclasses
from types import SimpleNamespace

import numpy as np
import plotly.graph_objects as go
import pytest

from core import (
    FilmGateConfig,
    HeleShawSolver,
    MaterialDB,
    build_demo_geometry,
    build_film_gate_geometry,
    build_fine_geometry,
    fine_refine_factor,
    refine_for_display,
    render_3d_fill_time,
    render_3d_pressure,
    render_3d_thickness_map,
)
from core.visualizer_3d import (
    FINE_DISPLAY_CELL_CAP,
    _DisplayResult,
    _interp_field_to_fine,
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


# ----------------------------------------------------------------------
# Display-only analytic refinement (finer silhouette without re-solving)
# ----------------------------------------------------------------------


def _film_geoms(cs_coarse: float = 2.0, k: int = 2):
    """Return ``(coarse, fine)`` film-gate geometries — the same analytic
    shape rasterized at ``cs_coarse`` and ``cs_coarse / k``."""
    cfg_c = FilmGateConfig(
        plate_w_mm=120.0,
        plate_h_mm=80.0,
        plate_thk_mm=2.0,
        runner_long_mm=80.0,
        runner_short_diameter_mm=12.0,
        runner_depth_mm=20.0,
        runner_thk_mm=4.0,
        runner_flat_depth_mm=8.0,
        runner_slope_depth_mm=12.0,
        valve_gate_diameter_mm=4.0,
        gate_width_mm=60.0,
        cell_size_mm=cs_coarse,
    )
    cfg_f = dataclasses.replace(cfg_c, cell_size_mm=cs_coarse / k)
    return build_film_gate_geometry(cfg_c), build_film_gate_geometry(cfg_f)


def test_interp_shape_and_fine_mask():
    """Interpolated field has the fine grid shape, is NaN exactly outside
    the *analytic* fine cavity, and finite inside it."""
    coarse, fine = _film_geoms()
    field = np.where(coarse.mask, 1.0, np.nan)
    out = _interp_field_to_fine(field, coarse, fine)
    assert out.shape == fine.mask.shape
    assert np.all(np.isnan(out[~fine.mask]))
    assert np.all(np.isfinite(out[fine.mask]))


def test_interp_preserves_constant():
    """A constant coarse field maps to the same constant on the fine grid
    (normalized sampling does not dilute interior values)."""
    coarse, fine = _film_geoms()
    field = np.where(coarse.mask, 3.5, np.nan)
    out = _interp_field_to_fine(field, coarse, fine)
    assert np.allclose(out[fine.mask], 3.5)


def test_interp_no_nan_bleed_from_outside():
    """Coarse NaNs outside the cavity must not bleed into the fine field —
    every in-fine-cavity cell stays finite (the supersample P1 regression)."""
    coarse, fine = _film_geoms()
    field = np.where(coarse.mask, 0.7, np.nan)
    out = _interp_field_to_fine(field, coarse, fine)
    assert not np.any(np.isnan(out[fine.mask]))


def test_interp_coordinate_mapping_has_no_offset():
    """A coarse field equal to each cell's physical x maps to the fine
    cell's physical x at an interior cell — validates the pad-frame
    coordinate mapping carries no half-cell offset."""
    coarse, fine = _film_geoms()
    cs_c = coarse.cell_size_mm
    cs_f = fine.cell_size_mm
    ix_c = np.arange(coarse.nx)
    phys_x = (ix_c + 0.5) * cs_c  # physical x of each coarse column [mm]
    field = np.where(coarse.mask, phys_x[None, :], np.nan)
    out = _interp_field_to_fine(field, coarse, fine)
    # pick a clearly-interior fine cell (center of mass of the fine cavity)
    iy_f, ix_f = np.argwhere(fine.mask).mean(axis=0).round().astype(int)
    expected_x = (ix_f + 0.5) * cs_f
    assert np.isclose(out[iy_f, ix_f], expected_x, atol=cs_c)


def test_fine_geometry_actually_refines():
    """The analytic fine cavity carries ~k**2 the cells of the coarse one
    (genuine sub-cell refinement, not a value remap)."""
    coarse, fine = _film_geoms(k=2)
    n_c = int(coarse.mask.sum())
    n_f = int(fine.mask.sum())
    assert n_f > n_c
    # ~4x for k=2, allow generous rasterization slack
    assert 3.0 < n_f / n_c < 5.0


def test_refine_for_display_uses_fine_geometry():
    """``refine_for_display`` returns a view on the fine geometry with the
    solved fields interpolated and total fill time preserved."""
    coarse, fine = _film_geoms()
    result = SimpleNamespace(
        geometry=coarse,
        fill_time_s=np.where(coarse.mask, 0.5, np.nan),
        pressure_norm=np.where(coarse.mask, 0.2, np.nan),
        total_fill_time_s=1.23,
    )
    disp = refine_for_display(result, fine)
    assert isinstance(disp, _DisplayResult)
    assert disp.geometry is fine
    assert disp.fill_time_s.shape == fine.mask.shape
    assert disp.pressure_norm.shape == fine.mask.shape
    assert disp.total_fill_time_s == 1.23


@pytest.mark.parametrize(
    "renderer",
    [render_3d_thickness_map, render_3d_fill_time, render_3d_pressure],
)
def test_renderers_run_on_refined_display(renderer):
    """All 3 renderers produce a finer-silhouette figure when fed a refined
    display result: the ceiling Surface matches the fine grid (more cells
    than the coarse solve), and thickness is the analytic fine field."""
    coarse, fine = _film_geoms()
    solver = HeleShawSolver(
        geometry=coarse,
        material=MaterialDB()["PP"],
        injection_volume_flow_cm3s=20.0,
    )
    result = solver.solve(num_frames=4)
    disp = refine_for_display(result, fine)
    fig = renderer(disp)
    assert isinstance(fig, go.Figure)
    _floor, ceiling, _walls = _split_traces(fig)
    assert np.asarray(ceiling.z).shape == fine.mask.shape
    # finer than the coarse solve grid
    assert fine.mask.shape != coarse.mask.shape
    # ceiling color is finite inside the fine cavity, NaN outside
    color = np.asarray(ceiling.surfacecolor)
    assert np.all(np.isnan(color[~fine.mask]))
    assert np.all(np.isfinite(color[fine.mask]))


def test_build_fine_geometry_refines_and_falls_back():
    """``build_fine_geometry`` re-rasterizes a parametric cfg at cs/k, and
    returns None for image input (cfg=None) or k<=1 (native)."""
    coarse, _ = _film_geoms()
    cfg = FilmGateConfig(
        plate_w_mm=120.0,
        plate_h_mm=80.0,
        plate_thk_mm=2.0,
        runner_long_mm=80.0,
        runner_short_diameter_mm=12.0,
        runner_depth_mm=20.0,
        runner_thk_mm=4.0,
        runner_flat_depth_mm=8.0,
        runner_slope_depth_mm=12.0,
        valve_gate_diameter_mm=4.0,
        gate_width_mm=60.0,
        cell_size_mm=2.0,
    )
    fine = build_fine_geometry(cfg, 2)
    assert fine is not None
    assert np.isclose(fine.cell_size_mm, 1.0)
    assert int(fine.mask.sum()) > int(coarse.mask.sum())
    # fallbacks: no analytic shape, or no refinement requested
    assert build_fine_geometry(None, 3) is None
    assert build_fine_geometry(cfg, 1) is None


def test_fine_refine_factor_clamps_to_cell_cap():
    """The refine factor is clamped so the fine cavity stays within the
    display cell cap; small grids keep the requested factor."""
    # small coarse grid: requested factor passes through
    assert fine_refine_factor(10_000, 3) == 3
    # k<=1 or empty: no refinement
    assert fine_refine_factor(10_000, 1) == 1
    assert fine_refine_factor(0, 3) == 1
    # huge coarse grid: clamped below the request
    big = FINE_DISPLAY_CELL_CAP  # k_max = sqrt(cap/big) = 1
    assert fine_refine_factor(big, 3) == 1
    # a grid where k=2 fits but k=3 does not
    mid = FINE_DISPLAY_CELL_CAP // 5  # k_max = floor(sqrt(5)) = 2
    assert fine_refine_factor(mid, 3) == 2
