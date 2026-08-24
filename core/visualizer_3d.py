"""Interactive 3D visualization for FlowResult (Phase 1: solid extrusion).

This module is a *separate* surface from :mod:`core.visualizer` (which writes
PNG/GIF via matplotlib). 3D output here is a Plotly :class:`~plotly.graph_objects.Figure`
intended to be embedded in the Streamlit UI via ``st.plotly_chart``.

Phase 1 scope (current):
    Each cavity cell is rendered as a **flat-top solid block** extruded from
    the parting line (PL) at ``Z = 0`` to the cell's thickness ``Z = h``.
    The figure has three sparse ``go.Mesh3d`` traces:

    1. **Top faces** (Z = h)  — one flat quad per cell, colored *per face*
       (``intensitymode="cell"``) by the requested physics field
       (thickness / fill-time / pressure). Flat-per-cell, matching the 2D
       maps; thickness steps read as crisp steps, not smoothed ramps.
    2. **PL floor**  (Z = 0)  — uniform light gray, slightly transparent.
       Represents the parting-line / mold-half boundary.
    3. **Side walls**         — vertical quads. *Boundary* walls on every
       cavity edge close the silhouette; *step* walls on internal edges
       where the thickness changes close the vertical face of a step.

    Top faces / floor come from :func:`_cavity_corner_mesh` (cell-edge
    corners, shared between equal-thickness cells so a constant region stays
    ~1 vertex/corner — light). This replaced the original full-grid
    ``go.Surface`` (which carried a NaN entry for every out-of-cavity cell
    and was a heavy trace type) to keep plotly/WebGL rotation responsive.

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

from .solver import FlowResult

# Plotly's built-in name for the same ramp ``core.visualizer.THICKNESS_CMAP``
# gives the 2D maps, so the design view and the solid view agree on what "thick"
# looks like. Kept as a separate constant because Plotly capitalizes its named
# colorscales; ``tests/test_visualizer_3d.py`` is what keeps the two in step.
THICKNESS_COLORSCALE = "Cividis_r"

# ----------------------------------------------------------------------
# Coordinate / mask helpers
# ----------------------------------------------------------------------


def _cavity_corner_mesh(
    result: FlowResult,
    z_per_cell: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[np.ndarray, np.ndarray, np.ndarray],
    np.ndarray,
    np.ndarray,
]:
    """Flat-top block mesh: each cavity cell is a flat quad of its 4 *corners*
    (cell edges, not centers) at ``z = z_per_cell[cell]``.

    Corners are **shared between cells of equal z** (so a constant-thickness
    region stays ~1 vertex/corner — light); cells of different z get separate
    corner vertices, leaving a vertical gap at their shared edge that the
    side-wall builder fills as a *step wall*. This renders true thickness
    steps (plate split, balancer steps) as crisp vertical steps instead of
    smoothing them into a one-cell ramp, and — because every cavity cell gets
    its own quad — there is no boundary erosion at all (the old cell-center
    triangulation could not cap 3-cell corners / 1-cell strips).

    Returns ``(xs, ys, zs, (i, j, k), face_iy, face_ix, vert_iy, vert_ix)``:
    per-vertex coords, the triangle indices, the ``(iy, ix)`` of the cell
    owning each *face* (two faces per cell, in cell order — for **per-face**
    colour, ``intensitymode="cell"``), and the ``(iy, ix)`` of an owner cell
    for each *vertex* (for the hover field readout). Fully vectorized.
    """
    g = result.geometry
    mask = g.mask
    ny, nx = mask.shape
    cs = g.cell_size_mm
    x0, y0 = g.display_origin_mm()

    iy_c, ix_c = np.where(mask)
    n = iy_c.size
    if n == 0:
        empty_f = np.empty(0, dtype=float)
        empty_i = np.empty(0, dtype=np.int64)
        return (
            empty_f,
            empty_f,
            empty_f,
            (empty_i, empty_i, empty_i),
            iy_c,
            ix_c,
            iy_c,
            ix_c,
        )

    zc = np.asarray(z_per_cell, dtype=float)[iy_c, ix_c]  # (n,)

    # Four corners per cell (CCW from +z): TL(0,0) TR(0,1) BL(1,0) BR(1,1)
    # in (corner_iy, corner_ix). ix→x, iy→y.
    coy = np.array([0, 0, 1, 1])
    cox = np.array([0, 1, 0, 1])
    cgid = (iy_c[:, None] + coy[None, :]) * (nx + 1) + (ix_c[:, None] + cox[None, :])
    zq = np.round(zc * 1e6).astype(np.int64)
    zq4 = np.repeat(zq[:, None], 4, axis=1)

    # Dedup vertices by (corner-grid-id, quantized z): same corner + same z
    # collapse to one vertex; different z keeps them apart (= the step gap).
    keys = np.stack([cgid.ravel(), zq4.ravel()], axis=1)  # (4n, 2)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(n, 4)

    ucy = uniq[:, 0] // (nx + 1)
    ucx = uniq[:, 0] % (nx + 1)
    xs = ucx * cs - x0
    ys = ucy * cs - y0
    zs = uniq[:, 1].astype(float) / 1e6

    tl, tr, bl, br = inv[:, 0], inv[:, 1], inv[:, 2], inv[:, 3]
    tri_i = np.concatenate([tl, tl])
    tri_j = np.concatenate([tr, br])
    tri_k = np.concatenate([br, bl])
    # Faces 0..n-1 are each cell's first triangle, n..2n-1 the second; both
    # belong to cell (face % n) → owner index arrays for per-face colour.
    face_iy = np.concatenate([iy_c, iy_c])
    face_ix = np.concatenate([ix_c, ix_c])
    # Per-vertex owner cell (for the hover field readout). A corner shared by
    # equal-z cells keeps an arbitrary one of them (values differ only by the
    # local gradient) — enough to read a number off the hover, not the colorbar.
    num_verts = uniq.shape[0]
    vert_cell = np.empty(num_verts, dtype=np.int64)
    vert_cell[inv.ravel()] = np.repeat(np.arange(n), 4)
    vert_iy = iy_c[vert_cell]
    vert_ix = ix_c[vert_cell]
    return xs, ys, zs, (tri_i, tri_j, tri_k), face_iy, face_ix, vert_iy, vert_ix


def _scalar_with_mask(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply NaN outside the cavity so the colormap ignores those cells."""
    out = arr.astype(float).copy()
    out[~mask] = np.nan
    return out


# ----------------------------------------------------------------------
# Side-wall mesh builder
# ----------------------------------------------------------------------


_STEP_EPS = 1e-6


def _build_side_walls(
    result: FlowResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build vertical-wall Mesh3d quads for the solid block model.

    Two kinds of vertical quad, for each in-cavity cell:

    1. **Boundary walls** — when a 4-neighbor is out-of-bounds / out-of-cavity,
       a quad on the shared edge from Z = 0 (PL) to Z = h_local (ceiling).
    2. **Internal step walls** — when a +x / +y neighbor *is* in-cavity but has
       a different thickness, a quad on the shared edge spanning the two
       thicknesses ``[min, max]``. This closes the vertical face of a thickness
       step so the flat-top blocks read as a crisp step (only +x/+y checked so
       each internal edge is emitted exactly once). Owner = the taller cell.

    **Accepted tradeoff (intentional):** every thickness change is treated as a
    step, so an *intentionally continuous* taper — e.g. ``build_film_gate_geometry``'s
    runner-exit slope, interpolated over many cells — renders as a fine
    staircase rather than a smooth ramp. A designed step (Δ≈0.15 over one cell)
    and a sampled gradient (≈0.15 *per* cell) have **identical per-cell deltas**,
    so they cannot be told apart by magnitude; the only robust signal (local
    curvature / 2nd difference) is fragile at ramp ends and disproportionate for
    a display-only feature. Flat-top is chosen deliberately so product-surface
    steps (plate split, balancer steps) read crisply; the staircased tapers are
    on the flow channel and are the honest rendering of the discretized data.

    Returns five 1-D arrays:
      - xs, ys, zs:    vertex coordinates (mm)
      - tri:           flat triangle index list (3M,) → reshape to (M, 3)
      - cell_idx:      for each vertex, the flat index ``iy*nx + ix`` of the
                       owner cell, so the caller can pull per-cell physics
                       field values into the wall mesh (shared colormap).
    """
    g = result.geometry
    nx, ny = g.nx, g.ny
    cs = g.cell_size_mm
    x0, y0 = g.display_origin_mm()
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
        z_bot: float,
        z_top: float,
        owner: int,
    ) -> None:
        """Append a vertical quad (4 vertices, 2 triangles) from ``z_bot`` to
        ``z_top``, tagged with the owner cell's flat index for coloring."""
        i0 = len(xs)
        for x_m, y_m, z_m in (
            (ax_m, ay_m, z_bot),
            (bx_m, by_m, z_bot),
            (bx_m, by_m, z_top),
            (ax_m, ay_m, z_top),
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

        # +y edge (shared with cell below): boundary, else internal step
        if iy + 1 >= ny or not mask[iy + 1, ix]:
            add_quad(x_left, y_top, x_right, y_top, 0.0, h_top, owner)
        else:
            h_n = float(thk[iy + 1, ix])
            if abs(h_n - h_top) > _STEP_EPS:
                step_owner = owner if h_top >= h_n else (iy + 1) * nx + ix
                add_quad(
                    x_left, y_top, x_right, y_top, min(h_top, h_n), max(h_top, h_n), step_owner
                )
        # -y edge: boundary only (internal step owned by the cell above's +y)
        if iy - 1 < 0 or not mask[iy - 1, ix]:
            add_quad(x_left, y_bot, x_right, y_bot, 0.0, h_top, owner)
        # +x edge (shared with cell to the right): boundary, else internal step
        if ix + 1 >= nx or not mask[iy, ix + 1]:
            add_quad(x_right, y_bot, x_right, y_top, 0.0, h_top, owner)
        else:
            h_n = float(thk[iy, ix + 1])
            if abs(h_n - h_top) > _STEP_EPS:
                step_owner = owner if h_top >= h_n else iy * nx + (ix + 1)
                add_quad(
                    x_right, y_bot, x_right, y_top, min(h_top, h_n), max(h_top, h_n), step_owner
                )
        # -x edge: boundary only (internal step owned by the cell left's +x)
        if ix - 1 < 0 or not mask[iy, ix - 1]:
            add_quad(x_left, y_bot, x_left, y_top, 0.0, h_top, owner)

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


def _floor_block_trace(result: FlowResult) -> go.Mesh3d | None:
    """PL (parting-line) floor: a flat Z = 0 mesh over the cavity (cell-edge
    quads, so it aligns with the block walls/ceiling). Faint and translucent
    — anchors the silhouette on the parting plane.
    """
    g = result.geometry
    zeros = np.zeros_like(np.asarray(g.thickness_mm, dtype=float))
    xs, ys, zs, tri, _fiy, _fix, _viy, _vix = _cavity_corner_mesh(result, zeros)
    if xs.size == 0 or tri[0].size == 0:
        return None
    i, j, k = tri
    return go.Mesh3d(
        x=xs,
        y=ys,
        z=zs,
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


def _ceiling_block_trace(
    result: FlowResult,
    color_field: np.ndarray,
) -> go.Mesh3d | None:
    """Top faces (Z = h) as flat-top per-cell blocks colored by the field.

    Each cavity cell is a flat quad at its thickness; the colour is applied
    **per face** (``intensitymode="cell"``) so each cell reads as one flat
    patch — consistent with the 2D maps and faithful to the discretized
    thickness. Thickness steps therefore show as crisp steps (the vertical
    faces are closed by the step walls in :func:`_build_side_walls`). Shares
    ``coloraxis`` with the walls for a single colorbar. ``flatshading`` keeps
    facet edges crisp.
    """
    g = result.geometry
    thk = np.asarray(g.thickness_mm, dtype=float)
    xs, ys, zs, tri, face_iy, face_ix, vert_iy, vert_ix = _cavity_corner_mesh(result, thk)
    if xs.size == 0 or tri[0].size == 0:
        return None
    i, j, k = tri
    cf = np.asarray(color_field, dtype=float)
    intensity = cf[face_iy, face_ix]  # per-face → drives the colour
    vert_val = cf[vert_iy, vert_ix]  # per-vertex → readable on hover
    return go.Mesh3d(
        x=xs,
        y=ys,
        z=zs,
        i=i,
        j=j,
        k=k,
        intensity=intensity,
        intensitymode="cell",
        coloraxis="coloraxis",
        customdata=vert_val,
        flatshading=True,
        name="cavity ceiling",
        hovertemplate=(
            "x=%{x:.1f} mm<br>y=%{y:.1f} mm<br>"
            "h=%{z:.2f} mm<br>value=%{customdata:.3g}<extra></extra>"
        ),
    )


def _figure_with_pl_extrusion(
    result: FlowResult,
    color_field: np.ndarray,
    *,
    colorscale: str,
) -> go.Figure:
    """Compose ceiling + PL floor + side walls into one Plotly Figure.

    All three are sparse ``go.Mesh3d`` flat-top block traces; ceiling and
    walls share ``coloraxis="coloraxis"`` so a single colorbar covers the
    whole solid.
    """
    traces: list = []
    floor = _floor_block_trace(result)
    if floor is not None:
        traces.append(floor)
    walls = _walls_trace(result, color_field)
    if walls is not None:
        traces.append(walls)
    ceiling = _ceiling_block_trace(result, color_field)
    if ceiling is not None:
        traces.append(ceiling)
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


def render_3d_thickness_map(result: FlowResult) -> go.Figure:
    """3D solid extrusion (PL→ceiling) with the ceiling colored by thickness."""
    g = result.geometry
    color = _scalar_with_mask(g.thickness_mm, g.mask)
    fig = _figure_with_pl_extrusion(result, color, colorscale=THICKNESS_COLORSCALE)
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
