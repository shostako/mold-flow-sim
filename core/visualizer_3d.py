"""Interactive 3D visualization for FlowResult (Phase 1: solid extrusion).

This module is a *separate* surface from :mod:`core.visualizer` (which writes
PNG/GIF via matplotlib). 3D output here is a Plotly :class:`~plotly.graph_objects.Figure`
intended to be embedded in the Streamlit UI via ``st.plotly_chart``.

Phase 1 scope (current):
    Each cavity cell is rendered as a solid block extruded **upward** from
    the parting line (PL) at ``Z = 0`` to the cavity ceiling at
    ``Z = h(x, y)``. The figure has three traces:

    1. **Top surface** (Z = h)  — colored by the requested physics field
       (thickness / fill-time / pressure). This is the "active" surface.
    2. **PL floor**    (Z = 0)  — uniform light gray, slightly transparent.
       Represents the parting-line / mold-half boundary.
    3. **Side walls**           — vertical Mesh3d quads on every cavity
       boundary edge. Closes the silhouette so the geometry reads as a
       solid block, not as a floating sheet.

    All three traces are **sparse ``go.Mesh3d``** built over the cavity
    cells only. The ceiling/floor used to be full-grid ``go.Surface``
    traces, but those carry a NaN entry for every out-of-cavity cell and
    grow ~``k**2`` under display refinement; the sparse mesh
    (:func:`_cavity_surface_mesh`) only spans the cavity and is far lighter
    for plotly/WebGL to render and rotate.

Animation of the flow front (frames-based) is deferred to Phase 1-2.

Coordinate convention matches :mod:`core.visualizer`: x/y are in mm with
the valve-gate centroid at the origin; Z = 0 is the parting line; Z > 0
is the cavity height direction. **All three axes use the same mm scale
(`aspectmode="data"`), no exaggeration** — the plate genuinely looks
thin because that is the actual product proportion.

Plotly is imported at module level. The rest of the codebase does not
import this module unless the UI is showing 3D content (the Streamlit
expander is closed by default).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import distance_transform_edt, map_coordinates

from .geometry import (
    DirectGateConfig,
    FilmGate2Config,
    FilmGateConfig,
    Geometry,
    build_direct_gate_geometry,
    build_film_gate2_geometry,
    build_film_gate_geometry,
)
from .solver import FlowResult

# Display-only refinement: the largest fine cavity (cells) the 3D view will
# rasterize. The analytic re-raster runs no solver — only plotly's point
# count grows ~k**2 — so this is a *rendering* budget separate from the
# solver-side cell guard. Keeps the browser from choking on a huge mesh.
FINE_DISPLAY_CELL_CAP = 1_000_000

# ----------------------------------------------------------------------
# Coordinate / mask helpers
# ----------------------------------------------------------------------


def _cavity_surface_mesh(
    result: FlowResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Triangulate the cavity cell-center grid into a *sparse* mesh.

    Returns ``(xs, ys, cell_idx, (i, j, k))``:
      - ``xs`` / ``ys``: per-vertex gate-centered coordinates (mm), one
                          vertex per in-cavity cell center.
      - ``cell_idx``:    flat ``iy*nx + ix`` index of each vertex's cell,
                          for pulling per-cell height / field values.
      - ``(i, j, k)``:   triangle vertex indices. Each 2×2 block of cells is
                          triangulated by how many of its four cells are in
                          the cavity: **4 present → two triangles**, **exactly
                          3 present → the single triangle spanning them** (so
                          diagonal / curved boundaries stay capped instead of
                          leaving the side walls with no ceiling). Blocks with
                          ≤2 cells (a one-cell-wide section) have no area at
                          cell centers and are left uncapped — matching the
                          old ``go.Surface(connectgaps=False)`` behaviour and
                          negligible at the fine display mesh.

    Replaces the full rectangular ``go.Surface`` grid (which carries a NaN
    entry for *every* out-of-cavity cell, exploding the vertex count
    ~``k**2`` under display refinement) with a mesh that only spans the
    cavity — far lighter for plotly / WebGL to render and rotate. Fully
    vectorized (no Python per-cell loop). Winding is CCW (upward normal).
    """
    g = result.geometry
    mask = g.mask
    ny, nx = mask.shape
    cs = g.cell_size_mm
    x0, y0 = g.gate_origin_mm()

    iy_c, ix_c = np.where(mask)
    xs = (ix_c + 0.5) * cs - x0
    ys = (iy_c + 0.5) * cs - y0
    cell_idx = iy_c.astype(np.int64) * nx + ix_c

    if ny < 2 or nx < 2 or iy_c.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return xs, ys, cell_idx, (empty, empty, empty)

    vid = np.full((ny, nx), -1, dtype=np.int64)
    vid[iy_c, ix_c] = np.arange(iy_c.size)

    # 2×2-block triangulation. CCW corner order around a block is
    # 00 → 01 → 11 → 10 (00 = (iy, ix), 01 = (iy, ix+1), 10 = (iy+1, ix),
    # 11 = (iy+1, ix+1)); ix→x, iy→y so this is CCW from +z (upward normal).
    c00, c01 = mask[:-1, :-1], mask[:-1, 1:]
    c10, c11 = mask[1:, :-1], mask[1:, 1:]
    v00, v01 = vid[:-1, :-1], vid[:-1, 1:]
    v10, v11 = vid[1:, :-1], vid[1:, 1:]
    present = c00.astype(np.int8) + c01 + c10 + c11

    tris_i: list[np.ndarray] = []
    tris_j: list[np.ndarray] = []
    tris_k: list[np.ndarray] = []

    def _emit(sel: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> None:
        tris_i.append(a[sel])
        tris_j.append(b[sel])
        tris_k.append(c[sel])

    full = present == 4
    _emit(full, v00, v01, v11)
    _emit(full, v00, v11, v10)
    three = present == 3
    _emit(three & ~c00, v01, v11, v10)  # missing 00
    _emit(three & ~c01, v00, v11, v10)  # missing 01
    _emit(three & ~c10, v00, v01, v11)  # missing 10
    _emit(three & ~c11, v00, v01, v10)  # missing 11

    tri_i = np.concatenate(tris_i)
    tri_j = np.concatenate(tris_j)
    tri_k = np.concatenate(tris_k)
    return xs, ys, cell_idx, (tri_i, tri_j, tri_k)


def _scalar_with_mask(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply NaN outside the cavity so the colormap ignores those cells."""
    out = arr.astype(float).copy()
    out[~mask] = np.nan
    return out


# ----------------------------------------------------------------------
# Side-wall mesh builder
# ----------------------------------------------------------------------


def _build_side_walls(
    result: FlowResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a vertical-wall Mesh3d for every cavity boundary edge.

    For each in-cavity cell, check its 4 neighbors. Whenever a neighbor
    is out-of-bounds or out-of-cavity, emit a vertical quad on the shared
    edge from Z = 0 (PL) to Z = h_local (cavity ceiling).

    Returns five 1-D arrays:
      - xs, ys, zs:    vertex coordinates (mm)
      - tri:           flat triangle index list (3M,) → reshape to (M, 3)
      - cell_idx:      for each vertex, the flat index ``iy*nx + ix`` of
                       the cell that owns the wall. Lets the caller pull
                       per-cell physics field values into the wall mesh
                       (so walls share the ceiling's colormap).
    """
    g = result.geometry
    nx, ny = g.nx, g.ny
    cs = g.cell_size_mm
    x0, y0 = g.gate_origin_mm()
    mask = g.mask
    thk = g.thickness_mm

    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    tri: list[int] = []
    cell_idx: list[int] = []

    def add_quad(
        ax_m: float,
        ay_m: float,
        bx_m: float,
        by_m: float,
        h_top: float,
        owner: int,
    ) -> None:
        """Append a vertical quad (4 vertices, 2 triangles) tagged with the
        owner cell's flat index so plotly can color it like the ceiling."""
        i0 = len(xs)
        for x_m, y_m, z_m in (
            (ax_m, ay_m, 0.0),
            (bx_m, by_m, 0.0),
            (bx_m, by_m, h_top),
            (ax_m, ay_m, h_top),
        ):
            xs.append(x_m)
            ys.append(y_m)
            zs.append(z_m)
            cell_idx.append(owner)
        tri.extend([i0, i0 + 1, i0 + 2, i0, i0 + 2, i0 + 3])

    iy_idx, ix_idx = np.where(mask)
    for iy, ix in zip(iy_idx.tolist(), ix_idx.tolist(), strict=True):
        h_top = float(thk[iy, ix])
        if not np.isfinite(h_top) or h_top <= 0.0:
            continue
        owner = iy * nx + ix
        x_left = ix * cs - x0
        x_right = (ix + 1) * cs - x0
        y_bot = iy * cs - y0
        y_top = (iy + 1) * cs - y0

        if iy + 1 >= ny or not mask[iy + 1, ix]:
            add_quad(x_left, y_top, x_right, y_top, h_top, owner)
        if iy - 1 < 0 or not mask[iy - 1, ix]:
            add_quad(x_left, y_bot, x_right, y_bot, h_top, owner)
        if ix + 1 >= nx or not mask[iy, ix + 1]:
            add_quad(x_right, y_bot, x_right, y_top, h_top, owner)
        if ix - 1 < 0 or not mask[iy, ix - 1]:
            add_quad(x_left, y_bot, x_left, y_top, h_top, owner)

    return (
        np.asarray(xs, dtype=float),
        np.asarray(ys, dtype=float),
        np.asarray(zs, dtype=float),
        np.asarray(tri, dtype=np.int32),
        np.asarray(cell_idx, dtype=np.int64),
    )


# ----------------------------------------------------------------------
# Trace assembly
# ----------------------------------------------------------------------


def _floor_mesh_trace(
    xs: np.ndarray,
    ys: np.ndarray,
    tri: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> go.Mesh3d:
    """PL (parting-line) floor: a flat Z = 0 mesh over the cavity.

    Faint and translucent — anchors the silhouette on the parting plane
    without competing with the colored ceiling/walls. Shares the cavity
    triangulation with the ceiling, so it is sparse (no full-grid NaN
    padding).
    """
    i, j, k = tri
    return go.Mesh3d(
        x=xs,
        y=ys,
        z=np.zeros_like(xs),
        i=i,
        j=j,
        k=k,
        color="rgb(220,220,220)",
        opacity=0.30,
        flatshading=True,
        name="PL (parting line, Z=0)",
        hoverinfo="skip",
        showlegend=False,
    )


def _walls_trace(
    result: FlowResult,
    color_field: np.ndarray,
) -> go.Mesh3d | None:
    """Vertical side walls colored by the same physics field as the ceiling.

    ``color_field`` is the per-cell scalar array (shape == mask.shape).
    Each wall vertex inherits its owner cell's value via ``intensity``,
    sharing the figure's ``coloraxis`` so the colorbar applies to the
    walls too.

    Returns ``None`` if the cavity is empty (degenerate, test only).
    """
    xs, ys, zs, tri, cell_idx = _build_side_walls(result)
    if xs.size == 0:
        return None
    n_tri = tri.size // 3
    ijk = tri.reshape(n_tri, 3)
    field_flat = np.asarray(color_field, dtype=float).ravel()
    intensity = field_flat[cell_idx]
    return go.Mesh3d(
        x=xs,
        y=ys,
        z=zs,
        i=ijk[:, 0],
        j=ijk[:, 1],
        k=ijk[:, 2],
        intensity=intensity,
        intensitymode="vertex",
        coloraxis="coloraxis",
        opacity=1.0,
        flatshading=True,
        name="cavity walls",
        hoverinfo="skip",
        showlegend=False,
    )


def _ceiling_mesh_trace(
    result: FlowResult,
    color_field: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    cell_idx: np.ndarray,
    tri: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> go.Mesh3d:
    """Top surface (Z = h) as a sparse Mesh3d colored by the physics field.

    Per-vertex Z is the cavity thickness at that cell; per-vertex intensity
    is the requested field value, sharing ``coloraxis`` with the walls so a
    single colorbar covers the whole solid. The field value is also carried
    as ``customdata`` for the hover readout.
    """
    g = result.geometry
    i, j, k = tri
    thk = np.asarray(g.thickness_mm, dtype=float).ravel()
    z = thk[cell_idx]
    color = np.asarray(color_field, dtype=float).ravel()[cell_idx]
    return go.Mesh3d(
        x=xs,
        y=ys,
        z=z,
        i=i,
        j=j,
        k=k,
        intensity=color,
        intensitymode="vertex",
        coloraxis="coloraxis",
        customdata=color,
        flatshading=False,
        name="cavity ceiling",
        hovertemplate=(
            "x=%{x:.1f} mm<br>y=%{y:.1f} mm<br>"
            "h=%{z:.2f} mm<br>color=%{customdata:.3g}<extra></extra>"
        ),
    )


def _figure_with_pl_extrusion(
    result: FlowResult,
    color_field: np.ndarray,
    *,
    colorscale: str,
) -> go.Figure:
    """Compose ceiling + PL floor + side walls into one Plotly Figure.

    Ceiling and floor share one *sparse* cavity triangulation
    (:func:`_cavity_surface_mesh`) instead of a full rectangular grid; all
    colored traces share ``coloraxis="coloraxis"`` so a single colorbar
    covers the whole solid.
    """
    xs, ys, cell_idx, tri = _cavity_surface_mesh(result)
    has_surface = xs.size > 0 and tri[0].size > 0
    traces: list = []
    if has_surface:
        traces.append(_floor_mesh_trace(xs, ys, tri))
    walls = _walls_trace(result, color_field)
    if walls is not None:
        traces.append(walls)
    if has_surface:
        traces.append(_ceiling_mesh_trace(result, color_field, xs, ys, cell_idx, tri))
    fig = go.Figure(data=traces)
    fig.update_layout(coloraxis=dict(colorscale=colorscale))
    return fig


def _apply_camera_and_layout(
    fig: go.Figure,
    *,
    title: str,
    cbar_title: str,
) -> go.Figure:
    """Common scene/camera/layout settings shared by all 3D figures.

    Z-axis is rendered at the same scale as x/y (``aspectmode="data"``)
    so distances read directly off the plot in mm. The plate looks like
    a thin sheet — that is the actual product proportion, no
    exaggeration. Rotate with the mouse to perceive runner depth.
    """
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x [mm]",
            yaxis_title="y [mm]",
            zaxis_title="cavity height (PL=0) [mm]",
            aspectmode="data",
            camera=dict(eye=dict(x=1.4, y=-1.4, z=1.1)),
            zaxis=dict(rangemode="tozero"),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        coloraxis_colorbar=dict(title=cbar_title),
    )
    fig.update_layout(modebar=dict(remove=["lasso", "select"]))
    return fig


# ----------------------------------------------------------------------
# Display-only analytic refinement (finer silhouette without re-solving)
# ----------------------------------------------------------------------


@dataclass
class _DisplayResult:
    """A minimal ``FlowResult``-shaped view for the 3D renderers.

    Carries exactly the attributes the three ``render_3d_*`` functions (and
    their trace builders) read: ``geometry``, ``fill_time_s``,
    ``pressure_norm``, ``total_fill_time_s``. Thickness is taken straight
    from ``geometry.thickness_mm`` so no field is stored for it.

    Used by :func:`refine_for_display` to swap in a finer, analytically
    re-rasterized geometry while reusing the unchanged rendering machinery.
    """

    geometry: Geometry
    fill_time_s: np.ndarray
    pressure_norm: np.ndarray
    total_fill_time_s: float


def _interp_field_to_fine(
    field_coarse: np.ndarray,
    geom_coarse: Geometry,
    geom_fine: Geometry,
) -> np.ndarray:
    """Sample a coarse solved field onto a finer geometry's grid.

    The coarse field (``fill_time`` / ``pressure``) is only defined on the
    coarse solve mesh and is NaN outside the coarse cavity. We map every
    *fine* cell center to a fractional *coarse* index by shared physical
    (pad-frame) coordinates — both geometries come from the same parametric
    config with only ``cell_size_mm`` differing, so the physical layout is
    identical — and bilinearly sample.

    To keep the silhouette honest, the result is masked to
    ``geom_fine.mask`` (the *analytic* fine cavity), NOT to an upsampled
    coarse mask. Coarse NaNs are nearest-filled before sampling so no NaN
    bleeds into the colored field; the fill is invisible because anything
    outside the fine cavity is re-NaNed at the end.
    """
    cs_c = float(geom_coarse.cell_size_mm)
    cs_f = float(geom_fine.cell_size_mm)
    fc = np.asarray(field_coarse, dtype=float)
    valid = np.isfinite(fc) & geom_coarse.mask
    if not valid.any():
        return np.full((geom_fine.ny, geom_fine.nx), np.nan)

    # Nearest-fill invalid coarse cells so the bilinear sampler never sees NaN.
    nearest_idx = distance_transform_edt(~valid, return_indices=True)[1]
    filled = fc[tuple(nearest_idx)]

    # Fine cell centers -> coarse fractional index (shared pad/origin in mm).
    iy_f, ix_f = np.indices((geom_fine.ny, geom_fine.nx))
    fy = (iy_f + 0.5) * cs_f / cs_c - 0.5
    fx = (ix_f + 0.5) * cs_f / cs_c - 0.5
    fine = map_coordinates(filled, [fy, fx], order=1, mode="nearest")

    fine[~geom_fine.mask] = np.nan
    return fine


def refine_for_display(result: FlowResult, geom_fine: Geometry) -> _DisplayResult:
    """Return a display-only view of ``result`` on a finer geometry.

    ``geom_fine`` must be the *same parametric shape* as
    ``result.geometry`` rebuilt at a smaller ``cell_size_mm`` (analytic
    re-rasterization — cheap, no re-solve). The cavity silhouette and the
    thickness field come from ``geom_fine`` directly (genuinely fine, no
    interpolation artifacts). The solved fields (fill-time, pressure) are
    smoothly interpolated from the coarse solve mesh onto the fine grid for
    coloring only.

    This sidesteps the fundamental limit of upsampling a rasterized mask
    (bilinear cannot recover sub-cell silhouette) by sourcing the silhouette
    from the analytic geometry instead.
    """
    fine_fill = _interp_field_to_fine(result.fill_time_s, result.geometry, geom_fine)
    fine_pres = _interp_field_to_fine(result.pressure_norm, result.geometry, geom_fine)
    return _DisplayResult(
        geometry=geom_fine,
        fill_time_s=fine_fill,
        pressure_norm=fine_pres,
        total_fill_time_s=float(result.total_fill_time_s),
    )


def build_fine_geometry(cfg: object | None, k: int) -> Geometry | None:
    """Re-rasterize the parametric shape ``cfg`` at ``cell_size_mm / k``.

    Display-only: no solver is run. Returns ``None`` when ``cfg`` is
    ``None`` (image input has no analytic shape) or ``k <= 1`` (native
    resolution, no refinement needed). The same parametric builder is
    reused so the silhouette is genuinely finer (analytic), not an
    upsampled raster mask.
    """
    if cfg is None or k <= 1:
        return None
    fine_cfg = dataclasses.replace(cfg, cell_size_mm=cfg.cell_size_mm / k)
    if isinstance(cfg, DirectGateConfig):
        return build_direct_gate_geometry(fine_cfg)
    if isinstance(cfg, FilmGate2Config):
        return build_film_gate2_geometry(fine_cfg)
    if isinstance(cfg, FilmGateConfig):
        return build_film_gate_geometry(fine_cfg)
    return None


def fine_refine_factor(coarse_cells: int, k_requested: int) -> int:
    """Clamp the requested 3D refine factor to keep the fine cavity under
    :data:`FINE_DISPLAY_CELL_CAP`.

    Fine cells grow ~``k**2`` with the factor; a big plate at high ``k``
    would flood plotly. Returns the largest ``k`` in ``1..k_requested``
    whose estimated fine cell count stays within the cap.
    """
    if k_requested <= 1 or coarse_cells <= 0:
        return 1
    k_max = int((FINE_DISPLAY_CELL_CAP / coarse_cells) ** 0.5)
    return max(1, min(k_requested, k_max))


# ----------------------------------------------------------------------
# Public renderers
# ----------------------------------------------------------------------


def render_3d_thickness_map(result: FlowResult) -> go.Figure:
    """3D solid extrusion (PL→ceiling) with the ceiling colored by thickness."""
    g = result.geometry
    color = _scalar_with_mask(g.thickness_mm, g.mask)
    fig = _figure_with_pl_extrusion(result, color, colorscale="Viridis")
    return _apply_camera_and_layout(
        fig,
        title="Cavity thickness h(x, y) [mm] — solid view from PL",
        cbar_title="h [mm]",
    )


def render_3d_fill_time(result: FlowResult) -> go.Figure:
    """3D solid extrusion with the ceiling colored by fill-time."""
    g = result.geometry
    color = _scalar_with_mask(result.fill_time_s, g.mask)
    fig = _figure_with_pl_extrusion(result, color, colorscale="Plasma")
    return _apply_camera_and_layout(
        fig,
        title=f"Fill time on cavity ceiling — T_fill = {result.total_fill_time_s:.3f} s",
        cbar_title="fill time [s]",
    )


def render_3d_pressure(result: FlowResult) -> go.Figure:
    """3D solid extrusion with the ceiling colored by normalized pressure."""
    g = result.geometry
    color = _scalar_with_mask(result.pressure_norm, g.mask)
    fig = _figure_with_pl_extrusion(result, color, colorscale="Turbo")
    return _apply_camera_and_layout(
        fig,
        title="Normalized pressure on cavity ceiling (1 at gate, 0 at last fill)",
        cbar_title="P_norm",
    )
