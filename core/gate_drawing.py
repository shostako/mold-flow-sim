"""A3 drawing of a gate block, drawn from the spec the solver is about to run.

The drawing is built from the same :func:`build_profile_gate_geometry` the
solver uses, on a finer mesh (``FINE_CELL_MM``), so what is drawn is the shape
being solved -- never a stale picture of an earlier slider position. Every
``GateProfileSpec`` works (single pocket, fans, one-sided), because the
picture reads the rasterised depth field rather than re-deriving each feature.

Sheet (A3 landscape, millimetres on paper):

- plan view at 1:1 -- pocket outline, the land (depth = land depth) shaded,
  depth contours every 0.5 mm, the valve orifice and the section cut lines;
- four sections at 5:1 (smaller for a deep block) -- the centre, one third and two thirds of the exit
  half-width, and the last cell row at the edge;
- a table of every value in the spec plus a few derived ones;
- a title block.

Dimension lines are not drawn yet; the table carries the numbers.

Japanese text needs a TrueType-outline font for ``pdf.fonttype = 42``: a CFF
(OpenType) font such as Noto Sans CJK embeds with a broken glyph map and every
character prints as a different one, while text extraction still looks right.
:func:`japanese_font` prefers IPAex / Meiryo / BIZ UD and falls back to a CFF
font with ``pdf.fonttype = 3`` (correct glyphs, no text extraction).
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import io
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import Geometry
from .profile_gate import GateProfileSpec, ProfilePlateConfig, build_profile_gate_geometry

#: Mesh the drawing is built on [mm]. 0.1 mm keeps the A3 1:1 plan at roughly
#: 250 dpi and builds the 300 mm block in about half a second.
FINE_CELL_MM = 0.1
#: Plan scale. Sections are 5:1 when two columns of them fit the sheet.
PLAN_SCALE = 1.0
CONTOUR_STEP_MM = 0.5
A3_MM = (420.0, 297.0)

_JP_TRUETYPE = (
    # (file name, family) -- TrueType outlines, safe with pdf.fonttype 42.
    "ipaexg.ttf",
    "ipag.ttf",
    "meiryo.ttc",
    "BIZ-UDGothicR.ttc",
    "YuGothR.ttc",
    "TakaoGothic.ttf",
)
_JP_EXTRA_DIRS = (
    "/usr/share/fonts",
    "/mnt/c/Windows/Fonts",
    str(Path.home() / ".local/share/fonts"),
    str(Path.home() / ".fonts"),
)


@dataclass(frozen=True)
class JapaneseFont:
    path: str | None
    #: 42 (TrueType embedded) or 3 (glyphs as Type 3 procedures).
    fonttype: int


def _find_font_files(names: tuple[str, ...]) -> Iterator[str]:
    wanted = {n.lower() for n in names}
    for root in _JP_EXTRA_DIRS:
        base = Path(root)
        if not base.is_dir():
            continue
        try:
            for p in base.rglob("*"):
                if p.name.lower() in wanted:
                    yield str(p)
        except OSError:
            continue


def japanese_font() -> JapaneseFont:
    """The font the drawing writes Japanese with, and the PDF font type to use.

    Order: a TrueType Japanese font (fonttype 42), else Noto Sans CJK or any
    other CJK font matplotlib knows (fonttype 3), else none (the text falls
    back to DejaVu and the Japanese prints as boxes -- still a valid PDF).
    """
    found = {Path(p).name.lower(): p for p in _find_font_files(_JP_TRUETYPE)}
    for name in _JP_TRUETYPE:
        if name.lower() in found:
            return JapaneseFont(found[name.lower()], 42)
    from matplotlib import font_manager as fm

    for f in fm.fontManager.ttflist:
        if "CJK JP" in f.name or f.name in ("Noto Sans JP", "IPAexGothic", "IPAGothic"):
            return JapaneseFont(f.fname, 3)
    return JapaneseFont(None, 42)


@dataclass(frozen=True)
class DrawingField:
    """The fine-mesh build, in the display frame of the app's maps.

    ``x`` is 0 on the valve axis, ``y`` is 0 on the product's gate-side edge
    and negative inside the gate block (``t = -y``).
    """

    geometry: Geometry
    x: np.ndarray  # (nx,) cell-centre x [mm]
    y: np.ndarray  # (ny,) cell-centre y [mm]
    depth: np.ndarray  # (ny, nx) pocket depth [mm] in the block, NaN outside the pocket
    t_max: float


def drawing_field(
    spec: GateProfileSpec, plate: ProfilePlateConfig, cell_size_mm: float = FINE_CELL_MM
) -> DrawingField:
    geom = build_profile_gate_geometry(spec, plate, cell_size_mm=cell_size_mm)
    x0, y0 = geom.display_origin_mm()
    dx = geom.cell_size_mm
    x = (np.arange(geom.nx) + 0.5) * dx - x0
    y = (np.arange(geom.ny) + 0.5) * dx - y0
    in_block = (y < 0)[:, None] & geom.mask
    depth = np.where(in_block, geom.thickness_mm, np.nan)
    return DrawingField(geometry=geom, x=x, y=y, depth=depth, t_max=float(spec.t_max()))


def exit_half_width(spec: GateProfileSpec) -> float:
    """The width the sections are spread over: the exit half-width (symmetric)
    or the full exit width measured from the valve-side edge (one-sided)."""
    return spec.gate_exit_width / 2.0 if spec.symmetric else spec.gate_exit_width


def section_positions(
    spec: GateProfileSpec, cell_mm: float = FINE_CELL_MM
) -> list[tuple[str, float]]:
    """Four cuts: centre, about 1/3 and 2/3 of the width (rounded to 5 mm), edge.

    ``w`` is the spec's own width coordinate (distance from the valve axis,
    or from the valve-side edge for a one-sided block).
    """
    wmax = exit_half_width(spec)
    first = 0.0 if spec.symmetric else cell_mm
    mid1 = 5.0 * round(wmax / 15.0)
    mid2 = 5.0 * round(2.0 * wmax / 15.0)
    last = wmax - cell_mm
    return [("A", first), ("B", mid1), ("C", mid2), ("D", last)]


def w_to_x(spec: GateProfileSpec, w: float) -> float:
    """Display x of the spec width ``w`` (the right half when symmetric)."""
    return w if spec.symmetric else w - spec.valve.w


def section_profile(field: DrawingField, x_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """``(t, depth)`` along the column nearest ``x_mm``; steel (outside the
    pocket) reads depth 0, i.e. the parting line."""
    j = int(np.argmin(np.abs(field.x - x_mm)))
    rows = np.where(field.y < 0)[0]
    t = -field.y[rows]
    d = np.nan_to_num(field.depth[rows, j], nan=0.0)
    order = np.argsort(t)
    return t[order], d[order]


# ----------------------------- the spec table -----------------------------
_LABELS = {
    "symmetric": "左右対称",
    "gate_exit_width": "ゲート出口幅",
    "land.depth": "ランド深さ",
    "land.length": "ランド長さ（端）",
    "land.profile.center_length": "中央のランド長",
    "land.profile.power": "ランド長の形の指数",
    "land.closed_line": "ランドの閉鎖線 (t, w)",
    "main_ramp.angle_deg": "斜面角度 [°]",
    "main_ramp.cap_depth": "斜面の上限深さ",
    "outer_wall_line": "外壁線 (t, w)",
    "outer_wall_corner_radius": "外壁の角 R",
    "valve.t": "バルブ位置 t",
    "valve.w": "バルブ位置 w",
    "valve.orifice_diameter": "バルブゲート径",
    "well.shape": "井戸の形",
    "well.t_range": "井戸の範囲 t",
    "well.half_width": "井戸の半幅",
    "well.depth": "井戸の深さ",
    "well.wall_angle_deg": "井戸の壁角 [°]",
    "well.floor_t_range": "井戸の床の範囲 t",
    "island.angle_deg": "肉盗みの角度 [°]",
    "island.boundary_line": "肉盗みの境界線 (t, w)",
    "island.end_dist": "肉盗みの終端 t",
    "island.weld.t_range": "水平部の範囲 t",
    "island.weld.depth": "水平部の PL からの距離",
    "island.weld.w_max": "水平部の幅",
    "ramp_ends.w_from": "斜面角度徐変の開始 w",
    "ramp_ends.t_end": "斜面角度徐変 端の到達 t",
    "ramp_ends.depth_end": "斜面角度徐変 端の上限深さ",
    "ramp_cut.line": "ランプ奥の削り込みの線 (t, w)",
    "ramp_cut.depth": "削り込みの深さ",
    "ramp_cut.slope_angle_deg": "削り込みの斜面角 [°]",
    "land_ends.w_from": "ランド両端の増厚の開始 w",
    "land_ends.depth": "ランド両端の増厚の深さ",
    "runner.width": "ランナー幅",
    "runner.depth": "ランナー深さ",
    "runner.path": "ランナー経路 (t, w)",
}
_LIST_LABELS = {
    "edge_channels": (
        "縁部深彫り",
        {"width": "幅", "depth": "深さ", "t_range": "範囲 t", "side": "側"},
    ),
    "sub_gates": (
        "扇",
        {
            "inner_wall_line": "内壁線 (t, w)",
            "outer_wall_line": "外壁線 (t, w)",
            "tip_t": "先端 t",
            "island.angle_deg": "肉盗みの角度 [°]",
            "island.inner_line": "肉盗みの内側線 (t, w)",
            "island.outer_line": "肉盗みの外側線 (t, w)",
            "island.end_dist": "肉盗みの終端 t",
            "island.floor_depth": "肉盗みの平底深さ",
        },
    ),
}
#: Content that is not geometry, or that can carry a part/customer identifier.
_SKIP = {"name", "units"}


def _fmt_num(v: float) -> str:
    if isinstance(v, bool):
        return "はい" if v else "いいえ"
    if isinstance(v, int):
        return str(v)
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _fmt(v) -> str:
    if isinstance(v, (list, tuple)):
        if v and all(isinstance(p, (list, tuple)) for p in v):
            return " → ".join("(" + ", ".join(_fmt_num(c) for c in p) + ")" for p in v)
        return "〜".join(_fmt_num(c) for c in v) if len(v) == 2 else ", ".join(_fmt(c) for c in v)
    if isinstance(v, (int, float)):
        return _fmt_num(v)
    return str(v)


def _leaves(obj, path: str = "") -> Iterator[tuple[str, object]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _leaves(v, f"{path}.{k}" if path else k)
    else:
        yield path, obj


def spec_rows(spec: GateProfileSpec, plate: ProfilePlateConfig) -> list[tuple[str, str]]:
    """Every geometric value in the spec, plus derived and plate values.

    Driven by ``spec.to_dict()`` so a feature added to the spec cannot be
    left out of the table silently: an unmapped key is listed under its own
    path rather than dropped.
    """
    rows: list[tuple[str, str]] = []
    d = spec.to_dict()
    for key, val in d.items():
        if key in _SKIP or val is None:
            continue
        if key in _LIST_LABELS:
            head, sub = _LIST_LABELS[key]
            for i, item in enumerate(val):
                for p, v in _leaves(item):
                    if v is None:
                        continue
                    rows.append((f"{head} {i + 1}: {sub.get(p, p)}", _fmt(v)))
            continue
        for p, v in _leaves(val, key):
            if v is None:
                continue
            rows.append((_LABELS.get(p, p), _fmt(v)))
    rows.append(("（導出）深さ上限に達する t", _fmt_num(spec.ramp_cap_t())))
    rows.append(("（導出）ブロックの奥行き t_max", _fmt_num(spec.t_max())))
    split, lo, hi = plate.resolved_plate_zones()
    rows.append(("製品 幅 × 高さ", f"{_fmt_num(plate.plate_w_mm)} × {_fmt_num(plate.plate_h_mm)}"))
    if split > 0:
        rows.append(
            (
                "製品 肉厚（ゲート側／反ゲート側）",
                f"{_fmt_num(lo)} ／ {_fmt_num(hi)}（段差 {_fmt_num(split)}）",
            )
        )
    else:
        rows.append(("製品 肉厚", _fmt_num(lo)))
    return rows


# ------------------------------- rendering --------------------------------
@dataclass(frozen=True)
class GateDrawing:
    pdf: bytes
    png: bytes
    font: JapaneseFont
    pocket_volume_mm3: float
    sections: tuple[tuple[str, float], ...]
    section_scale: float


def _pocket_volume_mm3(field: DrawingField) -> float:
    dx = field.geometry.cell_size_mm
    return float(np.nansum(field.depth) * dx * dx)


def render_gate_drawing(
    spec: GateProfileSpec,
    plate: ProfilePlateConfig,
    *,
    title: str,
    solver_cell_mm: float | None = None,
    version: str = "",
    date: _dt.date | None = None,
    preview_dpi: int = 110,
) -> GateDrawing:
    """Draw the A3 sheet and return it as PDF and PNG bytes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    from matplotlib.patches import Circle, Rectangle

    field = drawing_field(spec, plate)
    font = japanese_font()
    family = ["DejaVu Sans"]
    if font.path is not None:
        fm.fontManager.addfont(font.path)
        family = [fm.FontProperties(fname=font.path).get_name(), "DejaVu Sans"]
    date = date or _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=9))).date()
    secs = section_positions(spec)
    vol = _pocket_volume_mm3(field)

    W, H = A3_MM

    def axes_mm(fig, left, top, width, height):
        return fig.add_axes((left / W, 1.0 - (top + height) / H, width / W, height / H))

    rc = {
        "font.family": family,
        "font.size": 6.5,
        "pdf.fonttype": font.fonttype,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
        "xtick.labelsize": 5.5,
        "ytick.labelsize": 5.5,
        "hatch.linewidth": 0.3,
    }
    with matplotlib.rc_context(rc):
        fig = plt.figure(figsize=(W / 25.4, H / 25.4))
        frame = axes_mm(fig, 0, 0, W, H)
        frame.set_xlim(0, W)
        frame.set_ylim(H, 0)
        frame.axis("off")
        frame.add_patch(Rectangle((8, 8), W - 16, H - 16, fill=False, lw=0.8))
        frame.text(14, 16, "ゲートブロック図面（自動作図）", fontsize=11, va="center")
        frame.text(14, 22, title, fontsize=7.5, va="center")

        # ---- plan, 1:1 ----
        ys, xs = np.where(np.isfinite(field.depth))
        x_lo = field.x[xs.min()] - 4.0
        x_hi = field.x[xs.max()] + 4.0
        y_lo = -field.t_max - 3.0
        y_hi = 4.0
        pw = (x_hi - x_lo) * PLAN_SCALE
        ph = (y_hi - y_lo) * PLAN_SCALE
        p_left = max(14.0, (W - 16 - pw) / 2.0) if pw < W - 40 else 14.0
        p_top = 34.0
        ax = axes_mm(fig, p_left, p_top, pw, ph)
        dx = field.geometry.cell_size_mm
        ext = (field.x[0] - dx / 2, field.x[-1] + dx / 2, field.y[0] - dx / 2, field.y[-1] + dx / 2)
        dmax = float(np.nanmax(field.depth))
        ax.imshow(
            field.depth,
            origin="lower",
            extent=ext,
            cmap="Greys",
            vmin=0.0,
            vmax=dmax * 2.2,
            interpolation="nearest",
            zorder=1,
        )
        land = np.where(np.isclose(field.depth, spec.land.depth, atol=1e-9), 1.0, np.nan)
        ax.contourf(
            field.x,
            field.y,
            np.nan_to_num(land, nan=0.0),
            levels=[0.5, 1.5],
            colors="none",
            hatches=["//////"],
            zorder=2,
        )
        levels = np.arange(CONTOUR_STEP_MM, dmax + 1e-9, CONTOUR_STEP_MM)
        if len(levels):
            ax.contour(
                field.x,
                field.y,
                np.nan_to_num(field.depth, nan=0.0),
                levels=levels,
                colors="#555",
                linewidths=0.25,
                zorder=3,
            )
        ax.contour(
            field.x,
            field.y,
            np.isfinite(field.depth).astype(float),
            levels=[0.5],
            colors="k",
            linewidths=0.6,
            zorder=4,
        )
        ax.axhline(0.0, color="k", lw=0.4, zorder=4)
        ax.text(x_lo + 1, 0.8, "製品端（t = 0）", fontsize=5, va="bottom")
        vm = field.geometry.valve_marker_mm
        if vm is not None:
            x0, y0 = field.geometry.display_origin_mm()
            ax.add_patch(
                Circle((vm[0] - x0, vm[1] - y0), vm[2], fill=False, ls="--", lw=0.5, zorder=5)
            )
        for lab, w in secs:
            xc = w_to_x(spec, w)
            ax.plot(
                [xc, xc],
                [y_lo + 0.5, y_hi - 0.5],
                color="#c8551b",
                lw=0.4,
                ls=(0, (6, 2, 1, 2)),
                zorder=6,
            )
            ax.text(
                xc, y_lo + 0.6, lab, color="#c8551b", fontsize=6, ha="center", va="bottom", zorder=6
            )
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect("auto")
        ax.set_xticks(np.arange(math.ceil(x_lo / 50) * 50, x_hi, 50))
        ax.set_yticks(np.arange(math.ceil(y_lo / 5) * 5, y_hi, 5))
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        frame.text(
            p_left, p_top - 2.0, "平面図 1:1（x = 0 はバルブ軸、y = −t）", fontsize=6.5, va="bottom"
        )

        # ---- sections: 5:1 when two columns fit, smaller otherwise ----
        t_hi = field.t_max + 1.0
        d_hi = dmax + 0.6
        scale = next((s_ for s_ in (5.0, 4.0, 3.0, 2.0) if 2 * t_hi * s_ + 22.0 <= 310.0), 1.0)
        sw = t_hi * scale
        sh = d_hi * scale
        s_top0 = p_top + ph + 14.0
        cols = (20.0, 20.0 + sw + 22.0)
        for k, (lab, w) in enumerate(secs):
            left = cols[k % 2]
            top = s_top0 + (k // 2) * (sh + 18.0)
            sa = axes_mm(fig, left, top, sw, sh)
            tt, dd = section_profile(field, w_to_x(spec, w))
            sa.fill_between(tt, 0.0, dd, step="mid", color="#dbe8f5", lw=0, zorder=1)
            sa.fill_between(tt, dd, d_hi, step="mid", color="#e4e4e4", lw=0, zorder=1)
            sa.step(tt, dd, where="mid", color="k", lw=0.6, zorder=3)
            sa.axhline(0.0, color="k", lw=0.8, zorder=3)
            sa.axhline(spec.land.depth, color="#1d5c9e", lw=0.3, ls=":", zorder=2)
            sa.axhline(spec.main_ramp.cap_depth, color="#1d5c9e", lw=0.3, ls=":", zorder=2)
            sa.set_xlim(0.0, t_hi)
            sa.set_ylim(d_hi, -0.0001)
            sa.set_xticks(np.arange(0, t_hi, 5))
            sa.set_xticks(np.arange(0, t_hi, 1), minor=True)
            sa.set_yticks(np.arange(0, d_hi, 0.5 if d_hi < 8 else 1.0))
            sa.tick_params(which="minor", length=1.2, width=0.3)
            sa.set_xlabel("t [mm]（製品端から）", fontsize=5.5, labelpad=1)
            sa.set_ylabel("深さ [mm]", fontsize=5.5, labelpad=1)
            where = "中央" if (spec.symmetric and w == 0) else ("端" if k == 3 else "")
            frame.text(
                left,
                top - 2.0,
                f"{lab}–{lab} 断面 {_fmt_num(scale)}:1　w = {_fmt_num(w)}"
                + (f"（{where}）" if where else ""),
                fontsize=6.5,
                va="bottom",
            )
        s_bottom = s_top0 + 2 * (sh + 18.0)

        # ---- notes, right of the sections ----
        notes = [
            f"平面図 1:1、断面 {_fmt_num(scale)}:1（縦横とも）。単位 mm。",
            "深さはパーティングラインからの深さ。",
            "斜線: ランド（深さ = ランド深さ）。",
            f"細線: 深さ {_fmt_num(CONTOUR_STEP_MM)} mm ごとの等深線。",
            "破線の円: バルブゲート。一点鎖線: 断面の位置。",
            "断面の点線: ランド深さと斜面の上限深さ。",
            f"網目 {_fmt_num(FINE_CELL_MM)} mm で組んだ形状を描いている"
            + (f"（解析は {_fmt_num(solver_cell_mm)} mm）。" if solver_cell_mm else "。"),
            "アプリの入力から自動で作図した。",
            "寸法は寸法表を正とする。",
        ]
        n_left = cols[1] + sw + 14.0
        frame.text(n_left, s_top0 - 2.0, "注記", fontsize=6.5, va="bottom")
        for i, s_ in enumerate(notes):
            frame.text(n_left, s_top0 + 2.5 + i * 3.8, f"・{s_}", fontsize=5.8, va="center")

        # ---- table, full width under the sections ----
        rows = spec_rows(spec, plate)
        rows.append(
            (
                "（導出）ポケット体積",
                f"{vol:,.0f} mm³（網目 {_fmt_num(field.geometry.cell_size_mm)} mm）",
            )
        )
        t_left, t_top = 14.0, s_bottom + 2.0
        t_bottom = H - 14.0 - 26.0 - 6.0
        per_col = max(1, int((t_bottom - t_top - 3.0) // 4.0))
        n_cols = max(3, math.ceil(len(rows) / per_col))
        per_col = math.ceil(len(rows) / n_cols)  # fill the columns evenly
        col_w = (W - 28.0) / n_cols
        frame.text(t_left, t_top - 1.0, "寸法表 [mm]", fontsize=6.5, va="bottom")
        for i, (k_, v_) in enumerate(rows):
            c, r = divmod(i, per_col)
            y_ = t_top + 2.8 + r * 4.0
            x_ = t_left + c * col_w
            frame.text(x_ + 0.5, y_, k_, fontsize=6.0, va="center")
            frame.text(x_ + col_w - 3.0, y_, v_, fontsize=6.0, va="center", ha="right")
            frame.plot([x_, x_ + col_w - 2.0], [y_ + 2.0, y_ + 2.0], color="#ccc", lw=0.25)

        # ---- title block ----
        bx, by, bw, bh = W - 14.0 - 130.0, H - 14.0 - 26.0, 130.0, 26.0
        frame.add_patch(Rectangle((bx, by), bw, bh, fill=False, lw=0.6))
        for yy in (by + 9.0, by + 17.5):
            frame.plot([bx, bx + bw], [yy, yy], color="k", lw=0.4)
        frame.text(bx + 2, by + 4.5, f"名称　{title}", fontsize=6.5, va="center")
        frame.text(
            bx + 2,
            by + 13.2,
            f"尺度　平面 1:1／断面 {_fmt_num(scale)}:1　　単位 mm",
            fontsize=6,
            va="center",
        )
        frame.text(
            bx + 2,
            by + 21.7,
            f"作成　{date.isoformat()}　mold-flow-sim {version}".rstrip(),
            fontsize=6,
            va="center",
        )

        pdf_buf = io.BytesIO()
        fig.savefig(pdf_buf, format="pdf")
        png_buf = io.BytesIO()
        fig.savefig(png_buf, format="png", dpi=preview_dpi)
        plt.close(fig)
    return GateDrawing(
        pdf=pdf_buf.getvalue(),
        png=png_buf.getvalue(),
        font=font,
        pocket_volume_mm3=vol,
        sections=tuple(secs),
        section_scale=scale,
    )


def spec_key(spec: GateProfileSpec, plate: ProfilePlateConfig) -> str:
    """A stable text key for caching a drawing of this exact shape."""
    import json

    return json.dumps({"spec": spec.to_dict(), "plate": dataclasses.asdict(plate)}, sort_keys=True)
