"""Interactive 3D visualization for FlowResult (Phase 1: static surfaces).

This module is a *separate* surface from :mod:`core.visualizer` (which writes
PNG/GIF via matplotlib). 3D output here is a Plotly :class:`~plotly.graph_objects.Figure`
intended to be embedded in the Streamlit UI via ``st.plotly_chart``.

Phase 1 scope:
    - ``render_3d_thickness_map``  — cavity thickness ``h(x,y)`` extruded as
      a Z-axis surface, optionally colored by a scalar field (e.g. ``tau``).
    - ``render_3d_fill_time``      — same surface, colored by fill-time.
    - ``render_3d_pressure``       — same surface, colored by normalized pressure.

Animation of the flow front (frames-based) is deferred to Phase 1-2.

Coordinate convention matches :mod:`core.visualizer`: x/y are in mm with the
valve-gate centroid at the origin; the surface height (Z) is the local
cavity thickness in mm.

Plotly is loaded only when these functions are called (the import lives at
module level but the rest of the codebase does not import this module unless
the UI is showing 3D content).
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .solver import FlowResult

# ----------------------------------------------------------------------
# Internal helpers
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


def _apply_camera_and_layout(
    fig: go.Figure,
    result: FlowResult,
    *,
    title: str,
    cbar_title: str,
) -> go.Figure:
    """Common scene/camera/layout settings shared by all 3D figures."""
    g = result.geometry
    w_mm = float(g.nx * g.cell_size_mm)
    h_mm = float(g.ny * g.cell_size_mm)
    # exaggerate Z so the (typically <5 mm) thickness is visible against
    # the (typically tens-to-hundreds of mm) plate dimensions
    z_max = float(np.nanmax(g.thickness_mm)) if np.isfinite(g.thickness_mm).any() else 1.0
    z_aspect = max(0.05, min(0.5, z_max / max(w_mm, h_mm) * 8.0))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x [mm]",
            yaxis_title="y [mm]",
            zaxis_title="thickness h [mm]",
            aspectmode="manual",
            aspectratio=dict(x=1.0, y=h_mm / w_mm, z=z_aspect),
            camera=dict(eye=dict(x=1.4, y=-1.4, z=1.1)),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        coloraxis_colorbar=dict(title=cbar_title),
    )
    # Hide Plotly's modebar logo to keep the embed clean.
    fig.update_layout(modebar=dict(remove=["lasso", "select"]))
    return fig


def _surface(
    result: FlowResult,
    color_field: np.ndarray,
    *,
    colorscale: str,
    cbar_title: str,
) -> go.Figure:
    """Build a Plotly Surface trace using the cavity thickness as Z and
    ``color_field`` as the surface color."""
    x, y = _gate_centered_axes(result)
    z = _surface_height(result)
    surface = go.Surface(
        x=x,
        y=y,
        z=z,
        surfacecolor=color_field,
        colorscale=colorscale,
        coloraxis="coloraxis",
        showscale=True,
        connectgaps=False,
        hovertemplate=(
            "x=%{x:.1f} mm<br>y=%{y:.1f} mm<br>"
            "h=%{z:.2f} mm<br>color=%{surfacecolor:.3g}<extra></extra>"
        ),
    )
    fig = go.Figure(data=[surface])
    fig.update_layout(coloraxis=dict(colorscale=colorscale))
    return fig


# ----------------------------------------------------------------------
# Public renderers
# ----------------------------------------------------------------------


def render_3d_thickness_map(result: FlowResult) -> go.Figure:
    """3D surface colored by cavity thickness itself (geometry-only view)."""
    g = result.geometry
    color = _scalar_with_mask(g.thickness_mm, g.mask)
    fig = _surface(result, color, colorscale="Viridis", cbar_title="h [mm]")
    return _apply_camera_and_layout(
        fig,
        result,
        title="Cavity thickness h(x, y) [mm] — 3D view",
        cbar_title="h [mm]",
    )


def render_3d_fill_time(result: FlowResult) -> go.Figure:
    """3D surface colored by fill-time (s). Outside cavity → blank."""
    g = result.geometry
    color = _scalar_with_mask(result.fill_time_s, g.mask)
    fig = _surface(result, color, colorscale="Plasma", cbar_title="fill time [s]")
    return _apply_camera_and_layout(
        fig,
        result,
        title=f"Fill time on cavity surface — T_fill = {result.total_fill_time_s:.3f} s",
        cbar_title="fill time [s]",
    )


def render_3d_pressure(result: FlowResult) -> go.Figure:
    """3D surface colored by normalized pressure (1 at gate, 0 at last fill)."""
    g = result.geometry
    color = _scalar_with_mask(result.pressure_norm, g.mask)
    fig = _surface(result, color, colorscale="Turbo", cbar_title="P_norm")
    return _apply_camera_and_layout(
        fig,
        result,
        title="Normalized pressure (1 at gate, 0 at last fill)",
        cbar_title="P_norm",
    )
