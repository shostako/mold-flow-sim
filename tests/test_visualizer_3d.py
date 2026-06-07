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


# ----------------------------------------------------------------------
# Display-only supersampling
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "renderer",
    [render_3d_thickness_map, render_3d_fill_time, render_3d_pressure],
)
def test_supersample_default_is_native_resolution(small_result, renderer):
    """supersample=1 must keep the solver-native grid (no refinement)."""
    fig = renderer(small_result, supersample=1)
    _floor, ceiling, _walls = _split_traces(fig)
    assert np.asarray(ceiling.z).shape == small_result.geometry.mask.shape


@pytest.mark.parametrize(
    "renderer",
    [render_3d_thickness_map, render_3d_fill_time, render_3d_pressure],
)
def test_supersample_refines_ceiling_grid(small_result, renderer):
    """supersample=k refines floor/ceiling/surfacecolor to (ny*k, nx*k) and
    keeps a valid wall mesh and the data aspect."""
    ny, nx = small_result.geometry.mask.shape
    fig = renderer(small_result, supersample=2)
    floor, ceiling, walls = _split_traces(fig)
    assert np.asarray(ceiling.z).shape == (ny * 2, nx * 2)
    assert np.asarray(floor.z).shape == (ny * 2, nx * 2)
    assert np.asarray(ceiling.surfacecolor).shape == (ny * 2, nx * 2)
    assert walls is not None and len(walls.i) > 0
    assert fig.layout.scene.aspectmode == "data"


def test_supersample_keeps_silhouette_and_masking(small_result):
    """On the refined grid the cavity silhouette is preserved: some ceiling
    cells finite (>0), some NaN; floor finite cells are exactly 0."""
    fig = render_3d_thickness_map(small_result, supersample=2)
    floor, ceiling, _walls = _split_traces(fig)
    z = np.asarray(ceiling.z)
    finite = np.isfinite(z)
    assert finite.any() and (~finite).any()
    assert np.all(z[finite] > 0)
    zf = np.asarray(floor.z)
    assert np.all(zf[np.isfinite(zf)] == 0.0)


def test_supersample_preserves_gate_origin(small_result):
    """The gate-centered origin must stay exactly on the native gate center
    for any factor, including even k (regression: a single offset fine cell
    shifted the origin by a quarter native cell at the default k=2)."""
    from core.visualizer_3d import _supersample_for_render

    g = small_result.geometry
    x0, y0 = g.gate_origin_mm()
    for k in (2, 3):
        res, _color = _supersample_for_render(small_result, g.thickness_mm.astype(float), k)
        fx0, fy0 = res.geometry.gate_origin_mm()
        assert np.isclose(fx0, x0), f"x origin drift at k={k}: {fx0} vs {x0}"
        assert np.isclose(fy0, y0), f"y origin drift at k={k}: {fy0} vs {y0}"


def test_supersample_grid_mode_value_alignment():
    """grid_mode=True keeps interpolated values aligned to the declared
    fine-cell centers: a linear ramp sampled at native cell centers reproduces
    the ramp at the fine cell centers (interior). Regression for the
    quarter-native-cell value shift the default grid_mode=False introduced."""
    from types import SimpleNamespace

    from core.geometry import Geometry
    from core.visualizer_3d import _supersample_for_render

    cs, nx, k = 2.0, 6, 2
    # 2 identical rows so the x-ramp survives the (scalar-k) zoom on both axes;
    # all-cavity so _fill is identity. f(center)=x along x.
    xramp = (np.arange(nx) + 0.5) * cs
    ramp = np.tile(xramp, (2, 1)).astype(float)
    mask = np.ones((2, nx), dtype=bool)
    # no gates: isolate grid_mode interpolation alignment from the gate-block
    # restamp (which would overwrite a gate cell's value).
    geom = Geometry(mask=mask, thickness_mm=ramp, cell_size_mm=cs, gates=[])
    res, color = _supersample_for_render(SimpleNamespace(geometry=geom), ramp, k)
    g2 = res.geometry
    fine_centers = (np.arange(g2.nx) + 0.5) * g2.cell_size_mm
    interior = (fine_centers >= xramp[0]) & (fine_centers <= xramp[-1])
    # rows are identical; check the x-alignment on row 0
    got = np.asarray(g2.thickness_mm)[0]
    assert np.allclose(got[interior], fine_centers[interior], atol=1e-6)
    assert np.allclose(np.asarray(color)[0][interior], fine_centers[interior], atol=1e-6)


def test_supersample_preserves_gate_field_value():
    """The native gate-cell field value (e.g. pressure_norm==1 'at gate') must
    survive refinement. Bilinear zoom alone would average a single-cell gate
    with neighbors, dropping the extremum for even k. Regression for the
    pressure colorbar/title losing '1 at gate'."""
    from types import SimpleNamespace

    from core.geometry import Geometry
    from core.visualizer_3d import _supersample_for_render

    cs, k, ny, nx = 1.0, 2, 5, 5
    mask = np.ones((ny, nx), dtype=bool)
    thk = np.ones((ny, nx), dtype=float)
    field = np.full((ny, nx), 0.3)  # pressure-like: low everywhere...
    gy, gx = 2, 2
    field[gy, gx] = 1.0  # ...except 1 at the single gate cell
    geom = Geometry(mask=mask, thickness_mm=thk, cell_size_mm=cs, gates=[(gy, gx)])
    _res, color = _supersample_for_render(SimpleNamespace(geometry=geom), field, k)
    color = np.asarray(color)
    block = color[gy * k : (gy + 1) * k, gx * k : (gx + 1) * k]
    assert np.allclose(block, 1.0)  # gate block keeps the native value exactly
    assert np.isclose(np.nanmax(color), 1.0)  # global max still reaches the gate


def test_supersample_no_bleed_between_disconnected_cavities():
    """Two cavities separated by a one-cell gap must not bleed values across the
    gap when refined. Regression: a global nearest-fill + zoom blended a
    neighbour's field into a disconnected region's boundary cells. Each region's
    value is constant, so per-component interpolation keeps it constant."""
    from types import SimpleNamespace

    from core.geometry import Geometry
    from core.visualizer_3d import _supersample_for_render

    cs, k = 1.0, 2
    # one row: [A A . B B] — gap at native col 2. A=1.0, B=5.0.
    mask = np.array([[True, True, False, True, True]])
    thk = np.array([[1.0, 1.0, 0.0, 5.0, 5.0]])
    geom = Geometry(mask=mask, thickness_mm=thk, cell_size_mm=cs, gates=[])
    res, _color = _supersample_for_render(SimpleNamespace(geometry=geom), thk, k)
    z = np.asarray(res.geometry.thickness_mm)[0]
    mf = np.asarray(res.geometry.mask)[0]
    # native cols 0-1 -> fine 0-3 (A); native cols 3-4 -> fine 6-9 (B)
    a_cells = z[:4][mf[:4]]
    b_cells = z[6:][mf[6:]]
    assert a_cells.size and b_cells.size
    assert np.allclose(a_cells, 1.0)  # no upward bleed from B(=5)
    assert np.allclose(b_cells, 5.0)  # no downward bleed from A(=1)


@pytest.mark.parametrize("renderer", [render_3d_fill_time, render_3d_pressure])
def test_supersample_keeps_wall_colors_finite(small_result, renderer):
    """fill-time/pressure are NaN outside the cavity (solver init); the
    mask-weighted upsample must sanitize them (NaN*0 == NaN) so the refined
    side-wall intensities stay finite. Regression: NaN propagated into the wall
    colors at supersample>1 and blanked the physically-colored walls."""
    fig = renderer(small_result, supersample=2)
    _floor, _ceiling, walls = _split_traces(fig)
    intensity = np.asarray(walls.intensity)
    assert np.isfinite(intensity).mean() > 0.99


def test_supersample_no_bleed_across_intracomponent_gap():
    """A single *connected* U-shaped cavity whose two arms are separated locally
    by a one-cell background slot must not blend one arm's value into the other
    across the slot. Normalized-convolution upsampling gives background zero
    weight, so the gap is never crossed regardless of connectivity (per-component
    filling alone could not prevent this)."""
    from types import SimpleNamespace

    from core.geometry import Geometry
    from core.visualizer_3d import _supersample_for_render

    cs, k = 1.0, 2
    # rows: top [A . B], bottom [A A B] -> all True cells are 4-connected (one
    # component) but the two top arms are split by the col-1 slot.
    mask = np.array([[True, False, True], [True, True, True]])
    thk = np.array([[1.0, 0.0, 5.0], [1.0, 1.0, 5.0]])  # left arm=1, right arm=5
    geom = Geometry(mask=mask, thickness_mm=thk, cell_size_mm=cs, gates=[])
    res, _c = _supersample_for_render(SimpleNamespace(geometry=geom), thk, k)
    z = np.asarray(res.geometry.thickness_mm)
    mf = np.asarray(res.geometry.mask)
    # top-left arm fine block (native (0,0)) must never pick up the 5-arm value
    tl = z[:k, :k][mf[:k, :k]]
    assert tl.size and np.all(tl <= 1.0 + 1e-9)


def test_supersample_many_disconnected_components_preserved():
    """Many isolated cavity cells (speckles) each refine to their own value
    block — the bbox-bounded per-component loop keeps them correct (and cheap)."""
    from types import SimpleNamespace

    from core.geometry import Geometry
    from core.visualizer_3d import _supersample_for_render

    cs, k, ny, nx = 1.0, 2, 7, 7
    mask = np.zeros((ny, nx), dtype=bool)
    thk = np.zeros((ny, nx), dtype=float)
    spots = [(1, 1, 1.0), (1, 4, 2.0), (4, 1, 3.0), (4, 4, 4.0)]  # >=1-cell gaps
    for gy, gx, v in spots:
        mask[gy, gx] = True
        thk[gy, gx] = v
    geom = Geometry(mask=mask, thickness_mm=thk, cell_size_mm=cs, gates=[])
    res, _color = _supersample_for_render(SimpleNamespace(geometry=geom), thk, k)
    z = np.asarray(res.geometry.thickness_mm)
    mf = np.asarray(res.geometry.mask)
    for gy, gx, v in spots:
        bm = mf[gy * k : (gy + 1) * k, gx * k : (gx + 1) * k]
        bz = z[gy * k : (gy + 1) * k, gx * k : (gx + 1) * k]
        assert bm.any()  # speckle survived refinement
        assert np.allclose(bz[bm], v)  # with its own value — no bleed across gaps


def test_supersample_preserves_mm_extent_and_grows_walls(small_result):
    """Refinement changes resolution, not physical span; and it yields more
    wall triangles (finer steps) than the native render."""
    cs = small_result.geometry.cell_size_mm
    f1, c1, w1 = _split_traces(render_3d_thickness_map(small_result, supersample=1))
    f2, c2, w2 = _split_traces(render_3d_thickness_map(small_result, supersample=2))
    x1, x2 = np.asarray(c1.x), np.asarray(c2.x)
    y1, y2 = np.asarray(c1.y), np.asarray(c2.y)
    # physical span identical up to a half-cell (finer grid samples centers)
    assert abs((x1.max() - x1.min()) - (x2.max() - x2.min())) <= cs
    assert abs((y1.max() - y1.min()) - (y2.max() - y2.min())) <= cs
    assert len(np.asarray(w2.i)) > len(np.asarray(w1.i))
