"""Visualization helpers: fill animation, pressure map, weld lines, air traps."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Patch
from scipy.ndimage import distance_transform_edt

from .multilayer_solver import MultilayerFlowResult
from .solver import (
    WELD_FULL_ANGLE_DEG,
    WELD_MIN_ANGLE_DEG,
    FlowResult,
    weld_score_from_angle,
)

# Explicit stacking for the fill renderers. matplotlib's defaults would
# put contours (2) above images (0) whatever the call order, so the
# unfilled overlay could never hide the isochrones behind the melt front.
_Z_FIELD = 1
_Z_ISOCHRONE = 2
_Z_OVERLAY = 3
# Above the overlay: the gate marker is bigger than one cell, so at the
# start of the animation — and for gates sitting on the cavity boundary —
# the opaque unfilled paint would eat most of it.
_Z_GATE = 4


def _base_extent(result: FlowResult) -> list[float]:
    """Image extent in mm in the product-referenced frame.

    All result-time maps (fill animation, pressure map, weld lines, skin /
    core layers, frame snapshots) share this extent so the "0" axis ticks
    line up with the valve axis (x) and the product's gate-side bottom
    edge (y) — see ``Geometry.display_origin_mm``. The bottom and left
    spines remain at the plot edges, so a gate block in the y < 0 region
    does not run into the axis frame.
    """
    g = result.geometry
    w_mm = g.nx * g.cell_size_mm
    h_mm = g.ny * g.cell_size_mm
    x0, y0 = g.display_origin_mm()
    return [-x0, w_mm - x0, -y0, h_mm - y0]


def _gate_xy_mm(result: FlowResult, iy: int, ix: int) -> tuple[float, float]:
    """Cell (iy, ix) center in mm, in the product-referenced display frame."""
    g = result.geometry
    x0, y0 = g.display_origin_mm()
    return (
        (ix + 0.5) * g.cell_size_mm - x0,
        (iy + 0.5) * g.cell_size_mm - y0,
    )


def _draw_gate_markers(
    ax,
    result: FlowResult,
    *,
    color: str = "red",
    edgecolor: str = "white",
    size: int = 8,
    zorder: float = _Z_GATE,
) -> None:
    for iy, ix in result.geometry.gates:
        gx_mm, gy_mm = _gate_xy_mm(result, iy, ix)
        ax.plot(
            gx_mm,
            gy_mm,
            marker="o",
            color=color,
            markersize=size,
            markeredgecolor=edgecolor,
            zorder=zorder,
        )


def _draw_geometry(ax, result: FlowResult) -> None:
    g = result.geometry
    extent = _base_extent(result)
    # outside: light gray; cavity: white background
    bg = np.where(g.mask, 0.0, 1.0)
    ax.imshow(
        bg,
        cmap="gray",
        vmin=0,
        vmax=1.4,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        alpha=0.35,
    )


# Fill-front rendering defaults.
#
# ``turbo`` rather than matplotlib's ``viridis``: the fill-time field is read
# for the *shape of its isochrones* (where they bunch up = slow flow, where
# they collide = weld line, where they end = air trap), and a rainbow's hue
# contrast makes those bands legible where a luminance ramp smooths them away.
# It also matches what commercial mold-flow packages plot, so a colleague
# reads the picture without a legend lesson, and it puts red — "look here" —
# on the last-filled region, which is exactly the risky one. ``turbo`` is the
# engineered rainbow: unlike ``jet`` its luminance rises monotonically, so it
# does not paint false banding at cyan and yellow that could be mistaken for
# real isochrones. Pass ``cmap="viridis"`` to get the old look back, or any
# colorblind-safe map if red/green confusion is a concern.
FILL_CMAP = "turbo"

# Thickness is an *input* — the geometry the analyst drew — not a solved field,
# so it gets a single-quantity luminance ramp instead of the rainbow the result
# maps use, and the picture reads as "this is the part" at a glance. The ramp
# runs **light = thin, dark = thick**: ink density reads as material quantity
# (contour shading, layered ink), and for a transparent molded part it is
# literally true — thicker plastic attenuates more light and looks darker. The
# unreversed ``cividis`` had it backwards.
#
# It is ``cividis_r`` and not a single-hue map (``Blues``, ``bone_r``) because
# the thin end has to stay *saturated*. The product plate is both the thinnest
# region and the one region anyone actually looks at; a map whose low end
# approaches white washes the product out, drops the step contrast between
# neighbouring plate zones, and in 3D lets the ceiling blend into the pale-gray
# PL floor and the white background — measured on ``bone_r``, where the 0.35 mm
# band went pure white and the outline disappeared. ``cividis_r`` ends in a
# saturated yellow instead, and cividis is built for color-vision deficiency,
# a property reversal preserves.
THICKNESS_CMAP = "cividis_r"

# Number of isochrone *lines* drawn over the fill front — asking for N puts
# exactly N lines on the plot, which is what the UI label promises and what a
# reader counts. The lines are the quantitative read of the plot; the colors
# only rank them.
ISOCHRONE_LEVELS = 12


# ``_draw_geometry`` paints the cavity and its surroundings as a 35 %-opaque
# gray ramp over the white figure. The fill renderers need those two flat
# colors as *opaque* paint instead, so they can lay a smoothly interpolated
# color field down first and punch the not-yet-filled region back out on top
# with a crisp, cell-exact overlay. Deriving them here keeps the two paths
# from drifting apart if the geometry backdrop is ever retuned.
_GEOM_ALPHA = 0.35
_GEOM_VMAX = 1.4


def _flatten_on_white(rgb: tuple[float, ...], alpha: float) -> tuple[float, float, float]:
    return tuple(alpha * c + (1.0 - alpha) * 1.0 for c in rgb[:3])


def _cavity_backdrop_colors() -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Opaque (outside-cavity, unfilled-cavity) colors matching ``_draw_geometry``."""
    gray = plt.get_cmap("gray")
    outside = _flatten_on_white(gray(1.0 / _GEOM_VMAX), _GEOM_ALPHA)
    cavity = _flatten_on_white(gray(0.0), _GEOM_ALPHA)
    return outside, cavity


def _nearest_extend(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace out-of-cavity values with the nearest in-cavity one.

    Bilinear interpolation reads a half cell past the cavity wall. Without
    this the boundary cells blend toward whatever ``fill_time_s`` happens to
    hold outside the mask (NaN, or a zero that colors as "filled first"),
    drawing a dark rim around the part. Extending the field first makes the
    interpolation see a continuation of the interior instead — the visible
    edge stays exact because the overlay, not this layer, defines it.
    """
    if mask.all():
        return values
    idx = distance_transform_edt(~mask, return_distances=False, return_indices=True)
    return values[tuple(idx)]


def _cell_centers_mm(result: FlowResult) -> tuple[np.ndarray, np.ndarray]:
    """Cell-center coordinates [mm] on the same origin as ``_base_extent``."""
    g = result.geometry
    x0, y0 = g.display_origin_mm()
    ny, nx = g.mask.shape
    xs = (np.arange(nx) + 0.5) * g.cell_size_mm - x0
    ys = (np.arange(ny) + 0.5) * g.cell_size_mm - y0
    return xs, ys


def _fill_field_rgb(result: FlowResult, cmap: str) -> np.ndarray:
    """Opaque RGBA of the whole fill-time field, safe to interpolate.

    The field is extended from the cells that *have* a fill time, not from
    the cavity mask: cells the melt never reaches carry NaN, and letting the
    interpolation read them would bleed a late-fill color into the live cells
    beside them.
    """
    g = result.geometry
    t_max = fill_time_max(result)
    valid = g.mask & np.isfinite(result.fill_time_s)
    if not valid.any():
        return np.zeros((*g.mask.shape, 4))
    field = _nearest_extend(np.nan_to_num(result.fill_time_s, nan=0.0), valid)
    rgba = plt.get_cmap(cmap)(mcolors.Normalize(vmin=0.0, vmax=t_max)(field))
    rgba[..., 3] = 1.0
    return rgba


#: Cells the melt never reaches. Read against the two grays of
#: ``_cavity_backdrop_colors``, this has to say "this does not fill" rather
#: than "not yet" -- the same red the short-shot map uses.
SHORT_SHOT_RGB = (0.75, 0.22, 0.17)


def _unfilled_overlay(result: FlowResult, filled: np.ndarray) -> np.ndarray:
    """Cell-exact paint covering everything that has not filled yet.

    Cells that never fill keep their own color for the whole animation, so a
    short shot does not read as a region that is merely late.
    """
    g = result.geometry
    outside, cavity = _cavity_backdrop_colors()
    overlay = np.zeros((*g.mask.shape, 4))
    overlay[..., :3] = np.asarray(outside)
    overlay[g.mask] = (*cavity, 1.0)
    dead = getattr(result, "unfillable_mask", None)
    revealed = g.mask & filled
    if dead is not None:
        overlay[g.mask & dead] = (*SHORT_SHOT_RGB, 1.0)
        # Never uncover them, whatever ``filled`` says. Today NaN <= t is
        # False so they would stay covered anyway -- but that is a property of
        # NaN comparison, not a decision, and it would quietly stop holding if
        # the fill test ever changed.
        revealed = revealed & ~dead
    overlay[..., 3] = np.where(revealed, 0.0, 1.0)
    return overlay


def fill_time_max(result: FlowResult) -> float:
    """Total fill time [s] used as the shared color/axis scale.

    A part where nothing beyond the gates fills leaves a field of zeros. The
    axis then comes from the reported total fill time, not from a hard-coded
    second: two different numbers for the same run is worse than a coarse one.
    """
    t_max = float(np.nanmax(result.fill_time_s)) if np.isfinite(result.fill_time_s).any() else 0.0
    if not np.isfinite(t_max) or t_max <= 0:
        t_max = float(result.total_fill_time_s)
    if not np.isfinite(t_max) or t_max <= 0:
        t_max = 1.0
    return t_max


def _fill_title(result: FlowResult, t: float, progress: float, *, long: bool = False) -> str:
    """Frame title. Names the cells that never fill, when there are any.

    Without the count the reader sees "filled = 100.0 %" -- the percentage
    rounds up long before the last handful of cells, and a short shot is
    exactly the thing that must not disappear into a rounding.
    """
    t_max = fill_time_max(result)
    if long:
        head = f"t = {t:.3f} s  /  T_fill = {t_max:.3f} s   filled = {progress * 100:5.1f} %"
    else:
        head = f"t={t:.3f}s  filled={progress * 100:.1f}%"
    dead = getattr(result, "unfillable_mask", None)
    n_dead = int(dead.sum()) if dead is not None else 0
    if n_dead:
        head += f"   short shot: {n_dead} cells"
    return head


def _draw_fill_state(
    ax,
    result: FlowResult,
    rgba_full: np.ndarray,
    filled: np.ndarray,
    *,
    smooth: bool,
    isochrone_levels: int = ISOCHRONE_LEVELS,
):
    """Paint one fill state and return the overlay artist that defines it.

    Draw order is the whole point: the color field and the isochrones are
    drawn for the *entire* cavity, then the not-yet-filled region is painted
    over both with an opaque, cell-exact overlay. The contours therefore need
    no per-frame clipping — advancing the front simply uncovers more of a
    picture that was already there. That keeps the levels identical in every
    frame (recomputing them per frame makes the contours crawl, which reads
    as flow that is not happening) and costs one contour pass per render
    instead of one per frame.

    The stacking is set with explicit ``zorder``, not with call order:
    matplotlib gives ``imshow`` a default ``zorder`` of 0 and ``contour`` a
    default of 2, so painting the overlay last still leaves the contours
    drawn on top of it — isochrones bleed across the unfilled region and the
    melt front stops being the boundary of the picture.
    """
    extent = _base_extent(result)
    ax.imshow(
        rgba_full,
        origin="lower",
        extent=extent,
        interpolation="bilinear" if smooth else "nearest",
        zorder=_Z_FIELD,
    )
    _draw_isochrones(ax, result, isochrone_levels)
    overlay_im = ax.imshow(
        _unfilled_overlay(result, filled),
        origin="lower",
        extent=extent,
        interpolation="nearest",
        zorder=_Z_OVERLAY,
    )
    return overlay_im


def _draw_isochrones(ax, result: FlowResult, n_levels: int):
    """Overlay equal-fill-time contours across the whole cavity.

    Not clipped to the filled region on purpose: the caller paints the
    unfilled area over these lines afterwards, which hides them exactly at
    the melt front without recomputing anything.
    """
    if n_levels < 1:
        return None
    g = result.geometry
    if min(g.mask.shape) < 2:
        # contour needs a 2x2 neighbourhood. A cavity one cell across -- a
        # small shape meshed coarsely -- solves fine, so raising here would
        # throw away a finished analysis over a decoration.
        return None
    t_max = fill_time_max(result)
    # n_levels + 2 points, ends dropped: N interior lines for a request of N.
    levels = np.linspace(0.0, t_max, n_levels + 2)[1:-1]
    field = np.where(g.mask, result.fill_time_s, np.nan)
    if not np.isfinite(field).any():
        return None
    xs, ys = _cell_centers_mm(result)
    return ax.contour(
        xs,
        ys,
        field,
        levels=levels,
        colors="black",
        linewidths=0.45,
        alpha=0.35,
        zorder=_Z_ISOCHRONE,
    )


def fill_frame_times(result: FlowResult, num_frames: int) -> np.ndarray:
    """Frame timestamps [s] shared by the GIF, the PNG frames and the player.

    Frame ``k`` shows every cell with ``fill_time_s <= t_k``. The first frame
    is one step in (never empty) and the last frame is the completed fill, so
    the three renderers stay in lockstep: the scrubber's frame ``k`` is the
    same instant as ``frames/frame_kkk.png`` and the GIF's frame ``k``.
    """
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    # Same axis as the color scale and the titles. Computing it here as well
    # is how a part that does not fill ended up with a 1-second animation over
    # a 0.023-second headline.
    t_max = fill_time_max(result)
    return np.linspace(t_max / num_frames, t_max, num_frames)


def fill_frame_fractions(result: FlowResult, num_frames: int) -> np.ndarray:
    """Filled area fraction (0..1) of the cavity at each frame time."""
    g = result.geometry
    cells = max(int(g.mask.sum()), 1)
    times = fill_frame_times(result, num_frames)
    return np.array([float((g.mask & (result.fill_time_s <= t)).sum()) / cells for t in times])


def render_fill_animation(
    result: FlowResult,
    output_path: str | Path,
    num_frames: int = 30,
    fps: int = 8,
    cmap: str = FILL_CMAP,
    show_progress_bar: bool = True,
    isochrone_levels: int = ISOCHRONE_LEVELS,
    smooth: bool = True,
) -> Path:
    """Render filling sequence as animated GIF.

    Each frame shows cells whose fill_time <= t_frame, colored by fill_time
    on a scale fixed over the whole animation. ``isochrone_levels`` draws
    equal-arrival-time contours over the front; ``smooth`` interpolates the
    color field between cell centers, which is honest here because the
    fill-time field is continuous (unlike the thickness field, whose steps
    are real geometry and must stay blocky).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    g = result.geometry
    extent = _base_extent(result)
    t_max = fill_time_max(result)
    frames_t = fill_frame_times(result, num_frames)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=110)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    title_obj = ax.set_title("")

    rgba_full = _fill_field_rgb(result, cmap)
    overlay_im = _draw_fill_state(
        ax,
        result,
        rgba_full,
        np.zeros_like(g.mask),
        smooth=smooth,
        isochrone_levels=isochrone_levels,
    )

    _draw_gate_markers(ax, result, color="red", edgecolor="white", size=8)

    # progress bar
    if show_progress_bar:
        bar_ax = fig.add_axes([0.15, 0.06, 0.7, 0.022])
        bar_ax.set_xlim(0, 1)
        bar_ax.set_ylim(0, 1)
        bar_ax.set_xticks([])
        bar_ax.set_yticks([])
        bar_rect = bar_ax.barh([0.5], [0.0], height=1.0, color="#2ecc71")[0]
        # title above the bar (instead of xlabel below) to keep the figure
        # bottom edge clear and avoid the label getting clipped on small
        # output sizes.
        bar_ax.set_title("fill progress", fontsize=8, pad=2)
    else:
        bar_rect = None

    norm = mcolors.Normalize(vmin=0, vmax=t_max)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("fill time [s]")

    def update(frame_idx):
        t = frames_t[frame_idx]
        filled = result.fill_time_s <= t
        overlay_im.set_array(_unfilled_overlay(result, filled))
        progress = float(filled[g.mask].sum()) / max(int(g.mask.sum()), 1)
        title_obj.set_text(_fill_title(result, t, progress, long=True))
        if bar_rect is not None:
            bar_rect.set_width(progress)
        return [overlay_im, title_obj] + ([bar_rect] if bar_rect else [])

    anim = FuncAnimation(fig, update, frames=num_frames, blit=False)
    writer = PillowWriter(fps=fps)
    anim.save(str(output_path), writer=writer)
    plt.close(fig)
    return output_path


def render_pressure_map(
    result: FlowResult,
    output_path: str | Path,
    cmap: str = "magma",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extent = _base_extent(result)
    g = result.geometry
    x0, y0 = g.display_origin_mm()

    fig, ax = plt.subplots(figsize=(8, 6), dpi=110)
    _draw_geometry(ax, result)

    p = result.pressure_norm.copy()
    rgba = plt.get_cmap(cmap)(np.clip(np.nan_to_num(p, nan=0.0), 0.0, 1.0))
    rgba[..., 3] = np.where(g.mask, 1.0, 0.0)
    # Cells that never fill have no pressure. Left as NaN they take the
    # colormap's "bad" color, which is transparent -- and the alpha is forced
    # to 1 right above, so they would come out solid black and read as the
    # bottom of the pressure scale instead of as empty material.
    dead = getattr(result, "unfillable_mask", None)
    if dead is not None:
        rgba[g.mask & dead] = (*SHORT_SHOT_RGB, 1.0)
    ax.imshow(rgba, origin="lower", extent=extent, interpolation="nearest")

    for iy, ix in g.gates:
        gx_mm = (ix + 0.5) * g.cell_size_mm - x0
        gy_mm = (iy + 0.5) * g.cell_size_mm - y0
        ax.plot(gx_mm, gy_mm, marker="o", color="lime", markersize=8, markeredgecolor="black")

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("Normalized pressure (1=gate, 0=last fill)")

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=mcolors.Normalize(0, 1), cmap=cmap),
        ax=ax,
        fraction=0.04,
        pad=0.02,
    )
    cbar.set_label("relative pressure")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


WELD_ALPHA_FLOOR = 0.35  # meld: faint but always visible once flagged
WELD_ALPHA_FULL = 0.9  # weld: saturated


def weld_overlay_score(
    result: FlowResult,
    *,
    min_angle_deg: float = WELD_MIN_ANGLE_DEG,
    full_angle_deg: float = WELD_FULL_ANGLE_DEG,
) -> np.ndarray:
    """Score [0..1] to draw: re-thresholded from the angle field when present.

    Results solved before the angle field existed (or built by hand in
    tests) fall back to the stored ``weld_score``.
    """
    if result.weld_angle_deg is None:
        return np.clip(result.weld_score, 0.0, 1.0)
    return weld_score_from_angle(
        result.weld_angle_deg, min_angle_deg=min_angle_deg, full_angle_deg=full_angle_deg
    )


def render_weldlines(
    result: FlowResult,
    output_path: str | Path,
    *,
    weld_min_angle_deg: float = WELD_MIN_ANGLE_DEG,
    weld_full_angle_deg: float = WELD_FULL_ANGLE_DEG,
) -> Path:
    """Plot fill-time iso-contours plus weld / meld lines and air traps.

    Confluences are drawn in red with the opening angle of the meeting
    streams as opacity: saturated at ``weld_full_angle_deg`` and above (a
    weld in the usual CAE sense), fading down to ``weld_min_angle_deg``
    (a meld — streams that merge nearly parallel, a visible mark rather
    than a strength defect), nothing below.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extent = _base_extent(result)
    g = result.geometry
    x0, y0 = g.display_origin_mm()

    fig, ax = plt.subplots(figsize=(8, 6), dpi=110)
    _draw_geometry(ax, result)

    masked_t = np.where(g.mask, result.fill_time_s, np.nan)
    # A severe short shot can leave one distinct fill time (the gate) or none
    # at all. contour() rejects a flat level list, and losing the weld/air-trap
    # plot over a decoration would throw away an analysis that ran to the end.
    finite_t = masked_t[np.isfinite(masked_t)]
    if finite_t.size and np.unique(finite_t).size >= 2 and min(g.mask.shape) >= 2:
        t_lo = float(np.min(finite_t[finite_t > 0])) if np.any(finite_t > 0) else 0.0
        levels = np.linspace(t_lo, float(np.max(finite_t)), 12)
        if np.unique(levels).size >= 2:
            cs = ax.contour(
                masked_t,
                levels=levels,
                extent=extent,
                origin="lower",
                colors="#2980b9",
                linewidths=0.7,
            )
            ax.clabel(cs, inline=True, fontsize=7, fmt="%.2fs")

    # weld lines (red overlay). Any flagged cell gets a visible floor: the
    # score is the meeting angle, and a meld (small angle) is still a line
    # worth seeing, just fainter than a head-on weld.
    weld = weld_overlay_score(
        result, min_angle_deg=weld_min_angle_deg, full_angle_deg=weld_full_angle_deg
    )
    weld_rgba = np.zeros((*weld.shape, 4))
    weld_rgba[..., 0] = 1.0  # red
    weld_rgba[..., 3] = np.where(
        weld > 0.0, WELD_ALPHA_FLOOR + (WELD_ALPHA_FULL - WELD_ALPHA_FLOOR) * weld, 0.0
    )
    ax.imshow(weld_rgba, origin="lower", extent=extent, interpolation="nearest")
    legend_extra = [
        Patch(facecolor=(1, 0, 0, WELD_ALPHA_FULL), label=f"weld (≥{weld_full_angle_deg:.0f}°)"),
        Patch(
            facecolor=(1, 0, 0, WELD_ALPHA_FLOOR),
            label=f"meld ({weld_min_angle_deg:.0f}–{weld_full_angle_deg:.0f}°)",
        ),
    ]

    # air traps (yellow X)
    iy_arr, ix_arr = np.where(result.air_traps)
    if iy_arr.size > 0:
        ax.scatter(
            (ix_arr + 0.5) * g.cell_size_mm - x0,
            (iy_arr + 0.5) * g.cell_size_mm - y0,
            marker="x",
            color="#f1c40f",
            s=40,
            linewidths=2,
            label="air trap",
        )

    # gates
    for iy, ix in g.gates:
        gx_mm = (ix + 0.5) * g.cell_size_mm - x0
        gy_mm = (iy + 0.5) * g.cell_size_mm - y0
        ax.plot(
            gx_mm,
            gy_mm,
            marker="o",
            color="lime",
            markersize=8,
            markeredgecolor="black",
            label="gate",
        )

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(
        f"Fill-time iso, weld / meld (red), air traps (yellow x) — "
        f"T_fill = {result.total_fill_time_s:.3f} s, η ≈ {result.viscosity_Pa_s:.1f} Pa·s"
    )

    handles, labels = ax.get_legend_handles_labels()
    seen: dict[str, object] = {}
    for handle, label in zip(handles, labels, strict=False):
        seen[label] = handle
    for patch in legend_extra:
        seen[patch.get_label()] = patch
    ax.legend(seen.values(), seen.keys(), loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def render_skin_layer_map(
    result: FlowResult,
    output_path: str | Path,
    cmap: str = "magma",
) -> Path:
    """Plot the skin-layer thickness s(x,y) [mm] computed by the solver.

    Returns the original output path even when the result has no skin
    field (skin-layer model disabled); in that case no file is written.
    """
    output_path = Path(output_path)
    if result.skin_thickness_mm is None:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extent = _base_extent(result)
    g = result.geometry
    x0, y0 = g.display_origin_mm()

    s_field = np.where(g.mask, result.skin_thickness_mm, np.nan)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=110)
    _draw_geometry(ax, result)
    s_max = float(np.nanmax(s_field)) if np.any(~np.isnan(s_field)) else 0.0
    if s_max <= 0:
        s_max = 1e-6
    norm = mcolors.Normalize(vmin=0.0, vmax=s_max)
    im = ax.imshow(
        s_field,
        origin="lower",
        extent=extent,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )
    for iy, ix in g.gates:
        ax.plot(
            (ix + 0.5) * g.cell_size_mm - x0,
            (iy + 0.5) * g.cell_size_mm - y0,
            marker="o",
            color="lime",
            markersize=8,
            markeredgecolor="black",
        )
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(
        f"Skin layer thickness s(x,y) — max {s_max * 1e3:.1f} μm "
        f"(c_skin={result.metadata.get('skin_growth_constant', '?')})"
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("skin thickness [mm]")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def render_core_layer_map(
    result: FlowResult,
    output_path: str | Path,
    cmap: str = "viridis",
) -> Path:
    """Plot the live core thickness h_core(x,y) = h - 2*s [mm]."""
    output_path = Path(output_path)
    if result.core_thickness_mm is None:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extent = _base_extent(result)
    g = result.geometry
    x0, y0 = g.display_origin_mm()

    h_core = np.where(g.mask, result.core_thickness_mm, np.nan)
    h_open = np.where(g.mask, g.thickness_mm, np.nan)
    h_max = float(np.nanmax(h_open)) if np.any(~np.isnan(h_open)) else 1.0
    if h_max <= 0:
        h_max = 1.0
    norm = mcolors.Normalize(vmin=0.0, vmax=h_max)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=110)
    _draw_geometry(ax, result)
    im = ax.imshow(
        h_core,
        origin="lower",
        extent=extent,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )
    for iy, ix in g.gates:
        ax.plot(
            (ix + 0.5) * g.cell_size_mm - x0,
            (iy + 0.5) * g.cell_size_mm - y0,
            marker="o",
            color="red",
            markersize=8,
            markeredgecolor="white",
        )
    if result.short_shot_mask is not None and result.short_shot_mask.any():
        iy_arr, ix_arr = np.where(result.short_shot_mask)
        ax.scatter(
            (ix_arr + 0.5) * g.cell_size_mm - x0,
            (iy_arr + 0.5) * g.cell_size_mm - y0,
            marker="s",
            color="#e74c3c",
            s=4,
            linewidths=0,
            label="sealed (skins met)",
        )
        ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    short_count = int(result.short_shot_mask.sum()) if result.short_shot_mask is not None else 0
    ax.set_title(f"Core thickness h_core = h - 2s [mm]  (sealed cells: {short_count})")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("h_core [mm]")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


# --------------------------------------------------------------------------
# Multilayer (PR-D)
# --------------------------------------------------------------------------


def _scalar_layer_field(
    result: MultilayerFlowResult, field: str, layer_idx: int
) -> tuple[np.ndarray, str, str]:
    """Helper: fetch the requested per-layer field slice and pick a
    matplotlib colormap / label suitable for that quantity.

    Supported fields: ``"temperature"`` (K), ``"viscosity"`` (Pa·s, log),
    ``"shear_rate"`` (s⁻¹, log), ``"thickness"`` (mm).
    """
    if field == "temperature":
        arr = result.layer_temperature_K
        cmap = "coolwarm"
        label = "T [K]"
    elif field == "viscosity":
        arr = result.layer_viscosity_Pa_s_field
        cmap = "plasma"
        label = "η [Pa·s]"
    elif field == "shear_rate":
        arr = result.layer_shear_rate_s_inv
        cmap = "viridis"
        label = "γ̇ [1/s]"
    elif field == "thickness":
        arr = result.layer_thickness_mm
        cmap = THICKNESS_CMAP
        label = "h_k [mm]"
    else:
        raise ValueError(
            f"field={field!r} not supported "
            "(expected 'temperature' / 'viscosity' / 'shear_rate' / 'thickness')"
        )
    if arr is None:
        raise ValueError(
            f"result has no layer_{field}_* field — was the solver run with "
            "``thermal_coupling=True``?"
        )
    N = arr.shape[0]
    if not 0 <= layer_idx < N:
        raise IndexError(f"layer_idx={layer_idx} out of range [0, {N})")
    return arr[layer_idx], cmap, label


def render_layer_map(
    result: MultilayerFlowResult,
    layer_idx: int,
    output_path: str | Path,
    field: str = "temperature",
    log_scale: bool | None = None,
) -> Path:
    """Plot one per-layer scalar field at the given layer index.

    ``field`` selects which quantity to plot (``temperature`` /
    ``viscosity`` / ``shear_rate`` / ``thickness``). When ``log_scale``
    is left ``None`` the routine auto-enables it for viscosity and shear
    rate (which span orders of magnitude in a typical run).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    layer_field, cmap, label = _scalar_layer_field(result, field, layer_idx)
    g = result.geometry
    masked = np.where(g.mask, layer_field, np.nan)

    if log_scale is None:
        log_scale = field in {"viscosity", "shear_rate"}
    if log_scale:
        valid = masked[np.isfinite(masked) & (masked > 0)]
        if valid.size == 0:
            log_scale = False
            norm = None
        else:
            norm = mcolors.LogNorm(vmin=float(valid.min()), vmax=float(valid.max()))
    if not log_scale:
        valid = masked[np.isfinite(masked)]
        if valid.size == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin = float(valid.min())
            vmax = float(valid.max())
            if vmin == vmax:
                vmax = vmin + 1.0
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    extent = _base_extent(result)
    x0, y0 = g.display_origin_mm()
    fig, ax = plt.subplots(figsize=(8, 6), dpi=110)
    _draw_geometry(ax, result)
    im = ax.imshow(
        masked, origin="lower", extent=extent, cmap=cmap, norm=norm, interpolation="nearest"
    )
    for iy, ix in g.gates:
        ax.plot(
            (ix + 0.5) * g.cell_size_mm - x0,
            (iy + 0.5) * g.cell_size_mm - y0,
            marker="o",
            color="lime",
            markersize=8,
            markeredgecolor="black",
        )
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    zeta = result.metadata.get("layer_zeta")
    if zeta is not None and 0 <= layer_idx < len(zeta) - 1:
        zeta_lo = zeta[layer_idx]
        zeta_hi = zeta[layer_idx + 1]
        title_zeta = f" — layer {layer_idx} (ζ ∈ [{zeta_lo:.3f}, {zeta_hi:.3f}])"
    else:
        title_zeta = f" — layer {layer_idx}"
    ax.set_title(f"{field.replace('_', ' ').title()}{title_zeta}")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(label)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def render_layer_grid(
    result: MultilayerFlowResult,
    output_path: str | Path,
    field: str = "temperature",
    log_scale: bool | None = None,
) -> Path:
    """Render every layer's scalar field as a single tiled PNG.

    The N panels share a colorscale so the temperature drop from wall to
    centre is immediately readable.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Fetch the (N, ny, nx) array and global colorscale anchors.
    layer0, cmap, label = _scalar_layer_field(result, field, 0)
    if field == "temperature":
        arr = result.layer_temperature_K
    elif field == "viscosity":
        arr = result.layer_viscosity_Pa_s_field
    elif field == "shear_rate":
        arr = result.layer_shear_rate_s_inv
    else:  # thickness
        arr = result.layer_thickness_mm
    if arr is None:
        raise ValueError(
            f"result has no layer_{field}_* field — was the solver run with "
            "``thermal_coupling=True``?"
        )
    N = arr.shape[0]
    g = result.geometry
    masked = np.where(g.mask[None, :, :], arr, np.nan)

    if log_scale is None:
        log_scale = field in {"viscosity", "shear_rate"}
    valid = masked[np.isfinite(masked)]
    if log_scale:
        valid = valid[valid > 0]
    if valid.size == 0:
        vmin, vmax = 0.0, 1.0
        log_scale = False
    else:
        vmin = float(valid.min())
        vmax = float(valid.max())
        if vmin == vmax:
            vmax = vmin + 1.0
    norm = (
        mcolors.LogNorm(vmin=vmin, vmax=vmax)
        if log_scale
        else mcolors.Normalize(vmin=vmin, vmax=vmax)
    )

    # Square-ish grid layout.
    ncols = int(np.ceil(np.sqrt(N)))
    nrows = int(np.ceil(N / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.6 * ncols, 3.0 * nrows), dpi=110, squeeze=False
    )
    extent = _base_extent(result)
    zeta = result.metadata.get("layer_zeta")
    last_im = None
    for k in range(nrows * ncols):
        ax = axes[k // ncols][k % ncols]
        if k >= N:
            ax.axis("off")
            continue
        last_im = ax.imshow(
            masked[k],
            origin="lower",
            extent=extent,
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )
        if zeta is not None and k < len(zeta) - 1:
            ax.set_title(f"k={k} (ζ∈[{zeta[k]:.2f},{zeta[k + 1]:.2f}])", fontsize=9)
        else:
            ax.set_title(f"k={k}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    if last_im is not None:
        cbar = fig.colorbar(
            last_im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, shrink=0.85
        )
        cbar.set_label(label)
    fig.suptitle(
        f"Layer-resolved {field} (N={N}, distribution={result.metadata.get('layer_distribution')})"
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_short_shot_map(
    result: MultilayerFlowResult,
    output_path: str | Path,
) -> Path:
    """Plot the short-shot mask predicted by the multilayer solver.

    The cavity outline is drawn underneath; flagged cells are overlaid
    as red squares. When no cells are short-shot the cavity is rendered
    in the standard grey with a 'no short shot' annotation.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    g = result.geometry
    x0, y0 = g.display_origin_mm()
    extent = _base_extent(result)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=110)
    _draw_geometry(ax, result)
    flagged = result.short_shot_mask
    if flagged is not None and flagged.any():
        iy_arr, ix_arr = np.where(flagged)
        ax.scatter(
            (ix_arr + 0.5) * g.cell_size_mm - x0,
            (iy_arr + 0.5) * g.cell_size_mm - y0,
            marker="s",
            color="#e74c3c",
            s=8,
            linewidths=0,
            label=f"short shot ({int(flagged.sum())} cells)",
        )
        ax.legend(loc="upper right", fontsize=9)
    else:
        ax.text(
            0.5,
            0.95,
            "no short shot",
            ha="center",
            va="top",
            transform=ax.transAxes,
            fontsize=11,
            color="#2c7a2c",
        )
    for iy, ix in g.gates:
        ax.plot(
            (ix + 0.5) * g.cell_size_mm - x0,
            (iy + 0.5) * g.cell_size_mm - y0,
            marker="o",
            color="lime",
            markersize=8,
            markeredgecolor="black",
        )
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    frac = result.metadata.get("short_shot_fraction", 0.0)
    T_solid = result.metadata.get("T_solid_K", float("nan"))
    ax.set_title(f"Short-shot prediction (fraction={frac:.3f}, T_solid={T_solid:.1f} K)")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def export_frames(
    result: FlowResult,
    output_dir: str | Path,
    num_frames: int = 12,
    cmap: str = FILL_CMAP,
    isochrone_levels: int = ISOCHRONE_LEVELS,
    smooth: bool = True,
) -> list[Path]:
    """Export individual PNG snapshots of the fill progression.

    These PNGs are what the in-app scrubber and the ZIP's ``player.html``
    show, so they carry their own colorbar: the player has no surrounding
    chrome to explain what the colors mean.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    g = result.geometry
    extent = _base_extent(result)
    t_max = fill_time_max(result)
    frames_t = fill_frame_times(result, num_frames)
    norm = mcolors.Normalize(vmin=0.0, vmax=t_max)
    rgba_full = _fill_field_rgb(result, cmap)

    # One figure for the whole sequence, as the GIF renderer does. Rebuilding
    # it per frame would re-contour the cavity once per PNG -- 60 passes at
    # the UI default, on a cavity the UI allows up to 500k cells.
    fig, ax = plt.subplots(figsize=(7, 5), dpi=100)
    overlay_im = _draw_fill_state(
        ax,
        result,
        rgba_full,
        np.zeros_like(g.mask),
        smooth=smooth,
        isochrone_levels=isochrone_levels,
    )
    _draw_gate_markers(ax, result, color="red", edgecolor="white", size=7)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    title_obj = ax.set_title("")
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("fill time [s]")
    fig.tight_layout()

    out_paths: list[Path] = []
    try:
        for k, t in enumerate(frames_t):
            filled = result.fill_time_s <= t
            overlay_im.set_array(_unfilled_overlay(result, filled))
            title_obj.set_text(
                _fill_title(result, t, filled[g.mask].sum() / max(int(g.mask.sum()), 1))
            )
            path = output_dir / f"frame_{k:03d}.png"
            fig.savefig(path)
            out_paths.append(path)
    finally:
        plt.close(fig)
    return out_paths


# ---------------------------------------------------------------------------
# Two-phase (injection + compression) short-shot map and animation
# ---------------------------------------------------------------------------

# Categorical, not a ramp: the two regions answer "which phase put melt
# here", an unordered fact. Blue = melt pool at the end of injection,
# orange = area gained while the mold closed. Both are opaque — the region
# boundary is the deliverable (it gets compared against a physical short
# shot), so no interpolation, no alpha.
TWO_PHASE_INJECTION_RGB = (0.22, 0.49, 0.72)
TWO_PHASE_COMPRESSION_RGB = (0.95, 0.61, 0.15)
# Pool cells whose skins met before the end of injection (skin model only).
TWO_PHASE_SEALED_RGB = (0.55, 0.05, 0.05)


def _two_phase_legend(fig, result) -> None:
    """Legend BELOW the axes, never inside them.

    The plates this tool draws are wide and shallow, so any in-axes corner
    the legend could pick sits on top of the part (an in-axes upper-right
    legend covered the far corner of the first real render). A figure-level
    legend under the plot cannot collide with the cavity at any aspect
    ratio.
    """
    handles = [
        Patch(facecolor=TWO_PHASE_INJECTION_RGB, label="filled during injection"),
        Patch(facecolor=TWO_PHASE_COMPRESSION_RGB, label="advanced by compression"),
        Patch(facecolor=_cavity_backdrop_colors()[1], label="unfilled"),
    ]
    if _two_phase_sealed(result).any():
        # Only when there is something to explain: an idle legend entry would
        # suggest the model always looks for it.
        handles.append(
            Patch(facecolor=TWO_PHASE_SEALED_RGB, label="sealed during injection (skins met)")
        )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        fontsize=8,
        frameon=False,
    )


def _build_two_phase_figure(result):
    """Figure + axes with the backdrop, extent and out-of-axes legend set up.

    Returns ``(fig, ax)``. Shared by the static map and the animation so a
    layout fix lands in both.
    """
    extent = _base_extent(result)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=110)
    _draw_geometry(ax, result)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    _two_phase_legend(fig, result)
    return fig, ax


def _two_phase_finalize_layout(fig) -> None:
    """Apply the layout AFTER the title exists — tight_layout measures the
    artists present at call time, so calling it from the builder (before the
    renderer sets its title) left the title clipped at the top edge."""
    # leave a strip at the bottom for the figure-level legend
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))


def _two_phase_sealed(result) -> np.ndarray:
    """Pool cells sealed during injection, or an all-False mask without the
    skin model (``TwoPhaseShortShotResult.injection_sealed_mask``)."""
    sealed = getattr(result, "injection_sealed_mask", None)
    if sealed is None:
        return np.zeros(result.geometry.shape, dtype=bool)
    return np.asarray(sealed, dtype=bool)


def _two_phase_rgba(result, injection_filled, compression_filled) -> np.ndarray:
    rgba = np.zeros((*result.geometry.shape, 4))
    rgba[injection_filled] = (*TWO_PHASE_INJECTION_RGB, 1.0)
    rgba[compression_filled] = (*TWO_PHASE_COMPRESSION_RGB, 1.0)
    # A sealed cell filled (it is in the pool) and then closed; painting it
    # plain blue would hide the one fact the skin model adds to this map.
    # Opaque category colour like the others -- the fact has no ordering.
    sealed = _two_phase_sealed(result) & injection_filled
    rgba[sealed] = (*TWO_PHASE_SEALED_RGB, 1.0)
    return rgba


def render_two_phase_map(
    result,
    output_path: str | Path,
) -> Path:
    """Categorical map of a ``TwoPhaseShortShotResult``.

    Injection region (blue), compression advance (orange), unfilled cavity
    (the geometry backdrop gray). Injection isochrones are overlaid inside
    the injection region when the arrival field has enough distinct values.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extent = _base_extent(result)
    g = result.geometry

    fig, ax = _build_two_phase_figure(result)

    omega1 = result.injection_mask
    omega2 = result.final_mask
    ax.imshow(
        _two_phase_rgba(result, omega1, omega2 & ~omega1),
        origin="lower",
        extent=extent,
        interpolation="nearest",
        zorder=_Z_FIELD,
    )

    # Injection-front history: thin isochrones inside the melt pool.
    t = np.where(omega1, result.injection_fill_time_s, np.nan)
    finite_t = t[np.isfinite(t)]
    if finite_t.size and np.unique(finite_t).size >= 2 and min(g.mask.shape) >= 2:
        levels = np.linspace(float(finite_t.min()), float(finite_t.max()), 8)
        if np.unique(levels).size >= 2:
            ax.contour(
                t,
                levels=levels,
                extent=extent,
                origin="lower",
                colors="white",
                linewidths=0.6,
                alpha=0.7,
                zorder=_Z_ISOCHRONE,
            )

    _draw_gate_markers(ax, result)

    md = result.metadata
    title = "Two-phase short shot — shot {v:.1f} cm3, injection {fi:.0%} → after compression {ff:.0%}".format(
        v=result.shot_volume_cm3,
        fi=md.get("injection_fill_fraction", float("nan")),
        ff=md.get("final_fill_fraction", float("nan")),
    )
    if md.get("skin_layer_enabled"):
        title += "\nskin layer c={c:.2f}, T_inj={t:.3f} s".format(
            c=md.get("skin_growth_constant", float("nan")), t=result.injection_time_s
        )
    ax.set_title(title)

    _two_phase_finalize_layout(fig)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def render_two_phase_animation(
    result,
    output_path: str | Path,
    num_frames: int = 24,
    fps: int = 8,
) -> Path:
    """GIF of the two-phase history: the pool grows through injection in
    real arrival time, then the compression advance sweeps forward in
    normalized order (the model has no compression timescale — the frame
    titles say which clock is running)."""
    from .two_phase import frame_states

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extent = _base_extent(result)
    frames = frame_states(result, num_frames=num_frames)

    fig, ax = _build_two_phase_figure(result)
    im = ax.imshow(
        _two_phase_rgba(result, frames[0].injection_filled, frames[0].compression_filled),
        origin="lower",
        extent=extent,
        interpolation="nearest",
        zorder=_Z_FIELD,
    )
    _draw_gate_markers(ax, result)
    T_inj = result.injection_time_s

    ax.set_title(_two_phase_frame_title(frames[0], T_inj))
    _two_phase_finalize_layout(fig)

    def _update(i):
        fr = frames[i]
        im.set_data(_two_phase_rgba(result, fr.injection_filled, fr.compression_filled))
        ax.set_title(_two_phase_frame_title(fr, T_inj))
        return [im]

    anim = FuncAnimation(fig, _update, frames=len(frames), blit=False)
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return output_path


def _two_phase_frame_title(fr, T_inj: float) -> str:
    if fr.phase == "injection":
        return f"Injection  t = {fr.value:.3f} s / {T_inj:.3f} s"
    return f"Compression (order only)  advance = {fr.value:.0%}"


def two_phase_frame_labels(result, num_frames: int) -> list[str]:
    """Per-frame readout for the scrubber, on the same frame series as the
    GIF and the PNG frames (``frame_states`` is the single source).

    Injection frames read a real time; compression frames read an advance
    fraction, because the model has no compression clock — a ``t = …`` there
    would invent one. Both carry the cavity fill fraction.
    """
    from .two_phase import frame_states

    cells = max(int(result.geometry.mask.sum()), 1)
    T_inj = result.injection_time_s
    out: list[str] = []
    for fr in frame_states(result, num_frames=num_frames):
        filled = (fr.injection_filled | fr.compression_filled).sum() / cells
        if fr.phase == "injection":
            head = f"射出  t = {fr.value:.3f} s / {T_inj:.3f} s"
        else:
            head = f"圧縮（順序のみ）  前進 {fr.value * 100:.0f} %"
        out.append(f"{head}   充填 {filled * 100:.1f} %")
    return out


def export_two_phase_frames(
    result,
    output_dir: str | Path,
    num_frames: int = 24,
) -> list[Path]:
    """PNG snapshots of the two-phase history, one per ``frame_states`` frame.

    Same figure, same titles and same frame series as
    :func:`render_two_phase_animation`, so frame ``k`` of the scrubber is the
    GIF's frame ``k``. The figure is built once and only the image data and
    title change between saves.
    """
    from .two_phase import frame_states

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    extent = _base_extent(result)
    frames = frame_states(result, num_frames=num_frames)
    T_inj = result.injection_time_s

    fig, ax = _build_two_phase_figure(result)
    im = ax.imshow(
        _two_phase_rgba(result, frames[0].injection_filled, frames[0].compression_filled),
        origin="lower",
        extent=extent,
        interpolation="nearest",
        zorder=_Z_FIELD,
    )
    _draw_gate_markers(ax, result)
    ax.set_title(_two_phase_frame_title(frames[0], T_inj))
    _two_phase_finalize_layout(fig)

    paths: list[Path] = []
    for i, fr in enumerate(frames):
        im.set_data(_two_phase_rgba(result, fr.injection_filled, fr.compression_filled))
        ax.set_title(_two_phase_frame_title(fr, T_inj))
        path = output_dir / f"frame_{i:03d}.png"
        fig.savefig(path)
        paths.append(path)
    plt.close(fig)
    return paths
