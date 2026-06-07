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

import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import distance_transform_edt, label, zoom

from .geometry import Geometry
from .solver import FlowResult

# ----------------------------------------------------------------------
# Coordinate / mask helpers
# ----------------------------------------------------------------------


def _gate_centered_axes(result: FlowResult) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_mm, y_mm) 1-D coordinate arrays in the gate-centered frame.

    Cell centers, not edges; matches the convention used by the matplotlib
    visualizer (``_base_extent`` / ``_gate_xy_mm``).
    """
    g = result.geometry
    x0, y0 = g.gate_origin_mm()
    x = (np.arange(g.nx) + 0.5) * g.cell_size_mm - x0
    y = (np.arange(g.ny) + 0.5) * g.cell_size_mm - y0
    return x, y


def _surface_height(result: FlowResult) -> np.ndarray:
    """Z-height array (mm). Outside-cavity cells are NaN so plotly hides them."""
    g = result.geometry
    z = g.thickness_mm.astype(float).copy()
    z[~g.mask] = np.nan
    return z


def _scalar_with_mask(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply NaN outside the cavity so the colormap ignores those cells."""
    out = arr.astype(float).copy()
    out[~mask] = np.nan
    return out


# ----------------------------------------------------------------------
# Display-only supersampling
# ----------------------------------------------------------------------


class _RenderResult:
    """Lightweight stand-in exposing only ``.geometry``.

    The trace builders (`_floor_trace`, `_walls_trace`, `_ceiling_trace`,
    `_build_side_walls`, `_gate_centered_axes`, `_surface_height`) read
    nothing off the result except ``.geometry``. When a renderer wants to
    draw a display-supersampled copy of the cavity, it wraps the refined
    geometry in this and reuses the existing builders unchanged.
    """

    __slots__ = ("geometry",)

    def __init__(self, geometry: Geometry) -> None:
        self.geometry = geometry


def _supersample_for_render(
    result: FlowResult, color_field: np.ndarray, factor: int
) -> tuple[FlowResult | _RenderResult, np.ndarray]:
    """Refine the cavity ``factor``x for display only and return a
    ``(result_like, color_field)`` pair.

    Display-only: the solver and the stored geometry are untouched. The
    cavity mask, the thickness map (Z heights) and the per-cell ``color_field``
    are bilinearly upsampled by ``factor`` so the ceiling Surface boundary and
    the side-wall Mesh3d read with ``factor``x finer steps — the grid-aligned
    staircase on slanted/curved silhouette edges shrinks by ``1/factor``.

    Out-of-cavity cells are filled with the nearest in-cavity value *before*
    interpolation so the surface does not bleed toward zero across the cavity
    boundary; the refined mask (the ``>= 0.5`` contour of the upsampled binary
    mask) then re-applies the silhouette. This fill is done **per connected
    cavity component** (`scipy.ndimage.label`): a single global nearest-fill
    would let one region's values bleed across a one-cell gap into a separate
    disconnected region (false colors/heights for uploaded geometries with
    narrow gaps/holes), so each component is filled only from its own cells,
    upsampled, and composited by its own refined mask. ``factor <= 1`` is a
    no-op (so the default render path and its tests are bit-for-bit identical),
    and a single connected cavity reduces to the original behaviour exactly.

    Each native gate cell is expanded to its full ``k×k`` fine block (not a
    single offset cell). The averaged gate centroid of a full block lands
    *exactly* on the native gate center for any ``k`` (even ``k`` included),
    so :func:`_gate_centered_axes` keeps the displayed origin pinned to the
    gate — a single offset cell would shift it by up to half a native cell.
    """
    k = max(1, int(factor))
    if k == 1:
        return result, color_field
    g = result.geometry
    mask = g.mask
    thk = np.asarray(g.thickness_mm, dtype=float)
    cf = np.asarray(color_field, dtype=float)
    ny, nx = mask.shape
    fine_shape = (ny * k, nx * k)

    # grid_mode=True treats each value as covering a finite cell extent (not a
    # center-to-center sample span), so an N-cell array upsampled by k yields
    # N*k cells over the *same* physical extent with centers at (j+0.5)*cs/k —
    # exactly the fine Geometry declared below. The default grid_mode=False
    # would align end-cell centers instead, shifting every interpolated value
    # (and the silhouette) by up to a quarter native cell at k=2. mode="nearest"
    # clamps the half-cell margins beyond the outermost native centers.
    zk = dict(order=1, mode="nearest", grid_mode=True)

    # Interpolate each connected cavity component independently and composite,
    # so a narrow gap between two disconnected regions never bleeds one region's
    # field into the other.
    comp_labels, ncomp = label(mask)
    thk_f = np.zeros(fine_shape, dtype=float)
    color_f = np.zeros(fine_shape, dtype=float)
    mask_f = np.zeros(fine_shape, dtype=bool)
    for comp in range(1, ncomp + 1):
        cmask = comp_labels == comp
        if cmask.all():
            thk_fill, color_fill = thk, cf
        else:
            idx = distance_transform_edt(~cmask, return_distances=False, return_indices=True)
            thk_fill = thk[tuple(idx)]
            color_fill = cf[tuple(idx)]
        cmask_f = zoom(cmask.astype(float), k, **zk) >= 0.5
        sel = cmask_f & ~mask_f  # components are disjoint; guard any thin seam
        thk_f[sel] = zoom(thk_fill, k, **zk)[sel]
        color_f[sel] = zoom(color_fill, k, **zk)[sel]
        mask_f |= cmask_f
    # Restamp each native gate cell's value over its k×k refined block. Bilinear
    # zoom would average a single-cell gate (e.g. pressure_norm==1 "at gate")
    # with its lower neighbors, and for even k no fine center lands on the native
    # gate center — so the displayed extremum and its auto-ranged colorbar would
    # miss the stated value. The gate block keeps the native value exactly.
    for iy, ix in g.gates:
        color_f[iy * k : (iy + 1) * k, ix * k : (ix + 1) * k] = cf[iy, ix]
    gates_f = [(iy * k + a, ix * k + b) for iy, ix in g.gates for a in range(k) for b in range(k)]
    fine = Geometry(
        mask=mask_f,
        thickness_mm=thk_f,
        cell_size_mm=g.cell_size_mm / k,
        gates=gates_f,
        label=g.label,
    )
    return _RenderResult(fine), color_f


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


def _floor_trace(result: FlowResult) -> go.Surface:
    """PL (parting-line) floor: a Z = 0 surface masked to the cavity.

    Kept faint and translucent so it does not steal attention from the
    colored ceiling and walls — its job is to anchor the silhouette
    on the parting plane, not to convey data.
    """
    g = result.geometry
    x, y = _gate_centered_axes(result)
    z = np.where(g.mask, 0.0, np.nan)
    return go.Surface(
        x=x,
        y=y,
        z=z,
        showscale=False,
        opacity=0.30,
        colorscale=[[0, "rgb(220,220,220)"], [1, "rgb(220,220,220)"]],
        cmin=0.0,
        cmax=1.0,
        surfacecolor=np.where(g.mask, 0.5, np.nan),
        connectgaps=False,
        name="PL (parting line, Z=0)",
        hovertemplate="PL  x=%{x:.1f}, y=%{y:.1f}<extra></extra>",
        contours=dict(
            x=dict(highlight=False),
            y=dict(highlight=False),
            z=dict(highlight=False),
        ),
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


def _ceiling_trace(
    result: FlowResult,
    color_field: np.ndarray,
    *,
    colorscale: str,
) -> go.Surface:
    """Top surface (Z = h) colored by the requested physics field."""
    x, y = _gate_centered_axes(result)
    z = _surface_height(result)
    return go.Surface(
        x=x,
        y=y,
        z=z,
        surfacecolor=color_field,
        colorscale=colorscale,
        coloraxis="coloraxis",
        showscale=True,
        connectgaps=False,
        name="cavity ceiling",
        hovertemplate=(
            "x=%{x:.1f} mm<br>y=%{y:.1f} mm<br>"
            "h=%{z:.2f} mm<br>color=%{surfacecolor:.3g}<extra></extra>"
        ),
    )


def _figure_with_pl_extrusion(
    result: FlowResult,
    color_field: np.ndarray,
    *,
    colorscale: str,
) -> go.Figure:
    """Compose ceiling + PL floor + side walls into one Plotly Figure.

    All three traces share ``coloraxis="coloraxis"`` so the side walls
    pick up the same colorbar as the ceiling — a single legend covers
    the whole solid.
    """
    traces: list = [_floor_trace(result)]
    walls = _walls_trace(result, color_field)
    if walls is not None:
        traces.append(walls)
    traces.append(_ceiling_trace(result, color_field, colorscale=colorscale))
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
# Public renderers
# ----------------------------------------------------------------------


def render_3d_thickness_map(result: FlowResult, *, supersample: int = 1) -> go.Figure:
    """3D solid extrusion (PL→ceiling) with the ceiling colored by thickness.

    ``supersample`` (>= 1) refines the mesh for display only (see
    :func:`_supersample_for_render`); 1 keeps the native solver resolution.
    """
    res, color = _supersample_for_render(
        result, result.geometry.thickness_mm.astype(float), supersample
    )
    color = _scalar_with_mask(color, res.geometry.mask)
    fig = _figure_with_pl_extrusion(res, color, colorscale="Viridis")
    return _apply_camera_and_layout(
        fig,
        title="Cavity thickness h(x, y) [mm] — solid view from PL",
        cbar_title="h [mm]",
    )


def render_3d_fill_time(result: FlowResult, *, supersample: int = 1) -> go.Figure:
    """3D solid extrusion with the ceiling colored by fill-time.

    ``supersample`` (>= 1) refines the mesh for display only.
    """
    res, color = _supersample_for_render(
        result, np.asarray(result.fill_time_s, dtype=float), supersample
    )
    color = _scalar_with_mask(color, res.geometry.mask)
    fig = _figure_with_pl_extrusion(res, color, colorscale="Plasma")
    return _apply_camera_and_layout(
        fig,
        title=f"Fill time on cavity ceiling — T_fill = {result.total_fill_time_s:.3f} s",
        cbar_title="fill time [s]",
    )


def render_3d_pressure(result: FlowResult, *, supersample: int = 1) -> go.Figure:
    """3D solid extrusion with the ceiling colored by normalized pressure.

    ``supersample`` (>= 1) refines the mesh for display only.
    """
    res, color = _supersample_for_render(
        result, np.asarray(result.pressure_norm, dtype=float), supersample
    )
    color = _scalar_with_mask(color, res.geometry.mask)
    fig = _figure_with_pl_extrusion(res, color, colorscale="Turbo")
    return _apply_camera_and_layout(
        fig,
        title="Normalized pressure on cavity ceiling (1 at gate, 0 at last fill)",
        cbar_title="P_norm",
    )
