"""Visualization helpers: fill animation, pressure map, weld lines, air traps."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.animation import FuncAnimation, PillowWriter

from .solver import FlowResult


def _base_extent(result: FlowResult) -> list[float]:
    g = result.geometry
    w_mm = g.nx * g.cell_size_mm
    h_mm = g.ny * g.cell_size_mm
    # invert y so origin is bottom-left like a CAD view
    return [0.0, w_mm, 0.0, h_mm]


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


def render_fill_animation(
    result: FlowResult,
    output_path: str | Path,
    num_frames: int = 30,
    fps: int = 8,
    cmap: str = "viridis",
    show_progress_bar: bool = True,
) -> Path:
    """Render filling sequence as animated GIF.

    Each frame shows cells whose fill_time <= t_frame, colored by fill_time.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    g = result.geometry
    extent = _base_extent(result)
    t_max = float(np.nanmax(result.fill_time_s))
    if not np.isfinite(t_max) or t_max <= 0:
        t_max = 1.0
    frames_t = np.linspace(t_max / num_frames, t_max, num_frames)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=110)
    _draw_geometry(ax, result)
    ax.set_xlim(0, extent[1])
    ax.set_ylim(0, extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    title_obj = ax.set_title("")

    norm = mcolors.Normalize(vmin=0, vmax=t_max)
    rgba_full = plt.get_cmap(cmap)(norm(result.fill_time_s))
    rgba_full[..., 3] = np.where(g.mask, 1.0, 0.0)

    image_data = np.zeros_like(rgba_full)
    im = ax.imshow(
        image_data,
        origin="lower",
        extent=extent,
        interpolation="nearest",
    )

    # gate markers
    for iy, ix in g.gates:
        gx_mm = (ix + 0.5) * g.cell_size_mm
        gy_mm = (iy + 0.5) * g.cell_size_mm
        ax.plot(gx_mm, gy_mm, marker="o", color="red", markersize=8, markeredgecolor="white")

    # progress bar
    if show_progress_bar:
        bar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.025])
        bar_ax.set_xlim(0, 1)
        bar_ax.set_ylim(0, 1)
        bar_ax.set_xticks([])
        bar_ax.set_yticks([])
        bar_rect = bar_ax.barh([0.5], [0.0], height=1.0, color="#2ecc71")[0]
        bar_ax.set_xlabel("fill progress")
    else:
        bar_rect = None

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("fill time [s]")

    def update(frame_idx):
        t = frames_t[frame_idx]
        filled = result.fill_time_s <= t
        rgba = rgba_full.copy()
        rgba[..., 3] = np.where(g.mask & filled, 1.0, 0.0)
        im.set_array(rgba)
        progress = float(filled[g.mask].sum()) / max(int(g.mask.sum()), 1)
        title_obj.set_text(
            f"t = {t:.3f} s  /  T_fill = {t_max:.3f} s   filled = {progress * 100:5.1f} %"
        )
        if bar_rect is not None:
            bar_rect.set_width(progress)
        return [im, title_obj] + ([bar_rect] if bar_rect else [])

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

    fig, ax = plt.subplots(figsize=(8, 6), dpi=110)
    _draw_geometry(ax, result)

    p = result.pressure_norm.copy()
    rgba = plt.get_cmap(cmap)(np.clip(p, 0.0, 1.0))
    rgba[..., 3] = np.where(g.mask, 1.0, 0.0)
    ax.imshow(rgba, origin="lower", extent=extent, interpolation="nearest")

    for iy, ix in g.gates:
        gx_mm = (ix + 0.5) * g.cell_size_mm
        gy_mm = (iy + 0.5) * g.cell_size_mm
        ax.plot(gx_mm, gy_mm, marker="o", color="lime", markersize=8, markeredgecolor="black")

    ax.set_xlim(0, extent[1])
    ax.set_ylim(0, extent[3])
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


def render_weldlines(
    result: FlowResult,
    output_path: str | Path,
) -> Path:
    """Plot fill-time iso-contours plus weld score and air traps overlay."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extent = _base_extent(result)
    g = result.geometry

    fig, ax = plt.subplots(figsize=(8, 6), dpi=110)
    _draw_geometry(ax, result)

    masked_t = np.where(g.mask, result.fill_time_s, np.nan)
    levels = np.linspace(
        np.nanmin(masked_t[masked_t > 0]) if np.any(masked_t > 0) else 0, np.nanmax(masked_t), 12
    )
    cs = ax.contour(
        masked_t, levels=levels, extent=extent, origin="lower", colors="#2980b9", linewidths=0.7
    )
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.2fs")

    # weld lines (red overlay)
    weld = result.weld_score
    weld_rgba = np.zeros((*weld.shape, 4))
    weld_rgba[..., 0] = 1.0  # red
    weld_rgba[..., 3] = np.clip(weld, 0.0, 1.0) * 0.9
    ax.imshow(weld_rgba, origin="lower", extent=extent, interpolation="nearest")

    # air traps (yellow X)
    iy_arr, ix_arr = np.where(result.air_traps)
    if iy_arr.size > 0:
        ax.scatter(
            (ix_arr + 0.5) * g.cell_size_mm,
            (iy_arr + 0.5) * g.cell_size_mm,
            marker="x",
            color="#f1c40f",
            s=40,
            linewidths=2,
            label="air trap",
        )

    # gates
    for iy, ix in g.gates:
        gx_mm = (ix + 0.5) * g.cell_size_mm
        gy_mm = (iy + 0.5) * g.cell_size_mm
        ax.plot(
            gx_mm,
            gy_mm,
            marker="o",
            color="lime",
            markersize=8,
            markeredgecolor="black",
            label="gate",
        )

    ax.set_xlim(0, extent[1])
    ax.set_ylim(0, extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(
        f"Fill-time iso, weld lines (red), air traps (yellow x) — "
        f"T_fill = {result.total_fill_time_s:.3f} s, η ≈ {result.viscosity_Pa_s:.1f} Pa·s"
    )

    handles, labels = ax.get_legend_handles_labels()
    if labels:
        seen: dict[str, object] = {}
        for handle, label in zip(handles, labels, strict=False):
            seen[label] = handle
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
            (ix + 0.5) * g.cell_size_mm,
            (iy + 0.5) * g.cell_size_mm,
            marker="o",
            color="lime",
            markersize=8,
            markeredgecolor="black",
        )
    ax.set_xlim(0, extent[1])
    ax.set_ylim(0, extent[3])
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
            (ix + 0.5) * g.cell_size_mm,
            (iy + 0.5) * g.cell_size_mm,
            marker="o",
            color="red",
            markersize=8,
            markeredgecolor="white",
        )
    if result.short_shot_mask is not None and result.short_shot_mask.any():
        iy_arr, ix_arr = np.where(result.short_shot_mask)
        ax.scatter(
            (ix_arr + 0.5) * g.cell_size_mm,
            (iy_arr + 0.5) * g.cell_size_mm,
            marker="s",
            color="#e74c3c",
            s=4,
            linewidths=0,
            label="short shot",
        )
        ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, extent[1])
    ax.set_ylim(0, extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    short_count = int(result.short_shot_mask.sum()) if result.short_shot_mask is not None else 0
    ax.set_title(f"Core thickness h_core = h - 2s [mm]  (short shot cells: {short_count})")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("h_core [mm]")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def export_frames(
    result: FlowResult,
    output_dir: str | Path,
    num_frames: int = 12,
    cmap: str = "viridis",
) -> list[Path]:
    """Export individual PNG snapshots of the fill progression."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    g = result.geometry
    extent = _base_extent(result)
    t_max = float(np.nanmax(result.fill_time_s))
    if not np.isfinite(t_max) or t_max <= 0:
        t_max = 1.0
    frames_t = np.linspace(t_max / num_frames, t_max, num_frames)
    norm = mcolors.Normalize(vmin=0, vmax=t_max)
    rgba_full = plt.get_cmap(cmap)(norm(result.fill_time_s))

    out_paths: list[Path] = []
    for k, t in enumerate(frames_t):
        fig, ax = plt.subplots(figsize=(7, 5), dpi=100)
        _draw_geometry(ax, result)
        filled = result.fill_time_s <= t
        rgba = rgba_full.copy()
        rgba[..., 3] = np.where(g.mask & filled, 1.0, 0.0)
        ax.imshow(rgba, origin="lower", extent=extent, interpolation="nearest")
        for iy, ix in g.gates:
            ax.plot(
                (ix + 0.5) * g.cell_size_mm,
                (iy + 0.5) * g.cell_size_mm,
                marker="o",
                color="red",
                markersize=7,
                markeredgecolor="white",
            )
        ax.set_xlim(0, extent[1])
        ax.set_ylim(0, extent[3])
        ax.set_aspect("equal")
        ax.set_title(f"t={t:.3f}s  filled={filled[g.mask].sum() / max(g.mask.sum(), 1) * 100:.1f}%")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        path = output_dir / f"frame_{k:03d}.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        out_paths.append(path)
    return out_paths
