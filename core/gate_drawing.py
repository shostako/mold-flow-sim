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
#: Plan scales tried in order (1:1, 1:2, ...). Sections are 5:1 when two
#: columns of them fit the sheet.
PLAN_SCALES = (1.0, 0.5, 0.25, 0.2, 0.1)
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


#: Cap on the drawing mesh's cell count: the fine mesh doubles until the
#: block fits (a 400 mm exit at 0.1 mm is about 1.6 M cells).
MAX_DRAWING_CELLS = 3_000_000


def block_plate(plate: ProfilePlateConfig) -> ProfilePlateConfig:
    """The plate cut down to a 1 mm strip: the drawing only reads the block.

    The builder rasterises the whole product too; at 0.1 mm a 400 × 200 mm
    plate would be ~10 M cells of which the drawing uses none (Codex P2 on
    PR #104). The block does not depend on the product's height or
    thickness zones, only on its width (the grid-fit check).
    """
    return dataclasses.replace(
        plate,
        plate_h_mm=1.0,
        plate_split_height_mm=0.0,
        plate_lower_thk_mm=None,
        plate_upper_thk_mm=None,
    )


def drawing_cell_mm(spec: GateProfileSpec, plate: ProfilePlateConfig) -> float:
    """``FINE_CELL_MM``, doubled until the block's grid fits ``MAX_DRAWING_CELLS``."""
    dx = FINE_CELL_MM
    w = 2 * plate.pad_mm + plate.plate_w_mm
    h = 2 * plate.pad_mm + spec.t_max() + 1.0
    while (w / dx) * (h / dx) > MAX_DRAWING_CELLS:
        dx *= 2.0
    return dx


def drawing_field(
    spec: GateProfileSpec, plate: ProfilePlateConfig, cell_size_mm: float | None = None
) -> DrawingField:
    """The block's depth field on the drawing mesh (the product is cut to a strip)."""
    if cell_size_mm is None:
        cell_size_mm = drawing_cell_mm(spec, plate)
    geom = build_profile_gate_geometry(spec, block_plate(plate), cell_size_mm=cell_size_mm)
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
    """Four cuts: centre, about 1/3 and 2/3 of the width, edge.

    ``w`` is the spec's own width coordinate (distance from the valve axis,
    or from the valve-side edge for a one-sided block). The middle two are
    rounded to 5 mm when that keeps the four strictly inside and in order;
    a narrow exit gets the plain thirds (a 4 mm half-width would otherwise
    round to 0, 0, 5 -- a repeated cut and one outside the pocket, Codex P2
    on PR #104).
    """
    wmax = exit_half_width(spec)
    first = 0.0 if spec.symmetric else cell_mm
    last = wmax - cell_mm
    mids = [5.0 * round(wmax / 15.0), 5.0 * round(2.0 * wmax / 15.0)]
    if not first < mids[0] < mids[1] < last:
        thirds = [first + (last - first) * k / 3.0 for k in (1, 2)]
        mids = [round(v, 1) for v in thirds]
        if not first < mids[0] < mids[1] < last:  # a sub-millimetre exit
            mids = thirds
    return [("A", first), ("B", mids[0]), ("C", mids[1]), ("D", last)]


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


def column_w(spec: GateProfileSpec, field: DrawingField, x_mm: float) -> float:
    """The spec width ``w`` of the column :func:`section_profile` reads for ``x_mm``."""
    xc = float(field.x[int(np.argmin(np.abs(field.x - x_mm)))])
    return abs(xc) if spec.symmetric else xc + spec.valve.w


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


def _scale_text(s: float) -> str:
    return f"{_fmt_num(s)}:1" if s >= 1 else f"1:{_fmt_num(1.0 / s)}"


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


# ------------------------------- dimensions -------------------------------
@dataclass(frozen=True)
class Dim:
    """One dimension on the sheet.

    ``exact`` is True when the number is the spec's own value (the reading
    off the drawing mesh agreed with it within a cell); False when it was
    read off the mesh, which is what happens wherever a feature the closed
    forms below do not model (a ramp cut, an island, a rounded corner, ...)
    shapes that section. Inexact numbers print with a leading ``≈``.
    """

    view: str  # "平面" or the section label "A".."D"
    name: str
    value: float
    exact: bool

    def text(self) -> str:
        s = f"{self.value:.2f}".rstrip("0").rstrip(".")
        return s if self.exact else f"≈{s}"


def land_end_t(spec: GateProfileSpec, w: float) -> float:
    """``L(w)``: where the land ends at width ``w`` (the land profile, if any)."""
    lp = spec.land.profile
    if lp is None:
        return spec.land.length
    frac = min(max(1.0 - max(w, 0.0) / exit_half_width(spec), 0.0), 1.0)
    return spec.land.length + (lp.center_length - spec.land.length) * frac**lp.power


def cap_reach(spec: GateProfileSpec, w: float) -> tuple[float, float]:
    """``(t, depth)`` where the main ramp tops out at width ``w``: the cap line,
    graded at the ends (``ramp_ends``) and shifted by the land profile."""
    t_c = spec.ramp_cap_t()
    d_c = spec.main_ramp.cap_depth
    re_ = spec.ramp_ends
    W = exit_half_width(spec)
    if re_ is not None and w >= re_.w_from - 1e-9:
        frac = (min(w, W) - re_.w_from) / max(W - re_.w_from, 1e-12)
        t_c = t_c + (re_.t_end - t_c) * frac
        if re_.depth_end is not None:
            d_c = d_c + (re_.depth_end - d_c) * frac
    return t_c + land_end_t(spec, w) - spec.land.length, d_c


def wall_end_t(spec: GateProfileSpec, w: float) -> float | None:
    """t where the outer wall line reaches width ``w`` (single-pocket specs)."""
    if spec.outer_wall_line is None:
        return None
    (t1, w1), (t2, w2) = spec.outer_wall_line
    if abs(w1 - w2) < 1e-12 or w > w1:
        return None
    return t1 + (w1 - w) * (t2 - t1) / (w1 - w2)


def _snap(meas: float, candidates, tol: float) -> tuple[float, bool]:
    best = None
    for c in candidates:
        if (
            c is not None
            and abs(c - meas) <= tol
            and (best is None or abs(c - meas) < abs(best - meas))
        ):
            best = c
    return (best, True) if best is not None else (meas, False)


@dataclass(frozen=True)
class SectionDims:
    land_end: Dim | None
    cap_reach: Dim | None
    pocket_end: Dim | None
    cap_depth: Dim | None
    max_depth: Dim | None
    angle_deg: Dim | None


def section_dims(
    spec: GateProfileSpec,
    label: str,
    w: float,
    t: np.ndarray,
    d: np.ndarray,
    dx: float,
    w_col: float | None = None,
) -> SectionDims:
    """Read the section's key positions off the profile and snap them to the spec.

    The land, the ramp and the cap are read on the first pocket run from the
    product edge; a ramp that only starts behind a steel gap is not
    dimensioned (the dimensions are missing, never wrong). The back of the
    pocket and the deepest point are read over every run.

    Boundaries are read as the midpoint between the last cell on one side and
    the first on the other (error ≤ dx/2), then replaced by the closed-form
    value when that lies within a cell -- so a plain section prints the spec's
    numbers and a section reshaped by a feature prints what is drawn.
    ``w`` is the nominal cut (the closed forms are reported there); ``w_col``
    is the width of the column actually drawn, within half a cell of ``w``,
    which sets the depth the profile is tested against (a graded cap depth
    differs between the two).
    """
    w_col = w if w_col is None else w_col
    none = SectionDims(None, None, None, None, None, None)
    pocket = d > 0
    if not pocket.any():
        return none
    i0 = int(np.argmax(pocket))
    i1 = i0
    while i1 + 1 < len(d) and pocket[i1 + 1]:
        i1 += 1
    st, sd = t[i0 : i1 + 1], d[i0 : i1 + 1]
    land = spec.land.depth
    tc_x, dc_x = cap_reach(spec, w)
    dc_col = cap_reach(spec, w_col)[1]
    l_x = land_end_t(spec, w)

    land_dim = None
    above = np.where(sd > land + 1e-6)[0]
    if i0 == 0 and len(above) and above[0] > 0 and np.allclose(sd[: above[0]], land):
        v, ex = _snap(float(st[above[0]] - dx / 2), [l_x], dx)
        land_dim = Dim(label, "ランド長", v, ex)

    cap_dim = cap_d_dim = None
    reach = np.where(sd >= dc_col - 1e-6)[0]
    if len(reach) and reach[0] > 0:
        v, ex = _snap(float(st[reach[0]] - dx / 2), [tc_x], dx)
        cap_dim = Dim(label, "上限深さに達する t", v, ex)
        cap_d_dim = Dim(label, "上限深さ", dc_x, True)

    # t_max and the well's end are only candidates where the cut can reach
    # them -- across the well (or with no wall line to stop it). Elsewhere a
    # back that merely lands within a cell of t_max would print as exact
    # (review on PR #106).
    cands = [wall_end_t(spec, w)]
    in_well = (
        spec.well is not None
        and abs(w - (spec.valve.w if not spec.symmetric else 0.0)) <= spec.well.half_width
    )
    if in_well or spec.outer_wall_line is None:
        cands.append(spec.t_max())
        if spec.well is not None:
            cands.append(spec.well.t_range[1])
    # The back of the pocket and its deepest point are read over every pocket
    # cell of the section, not just the first run: a fan's centre section has
    # the land, a steel gap, then the well (Codex P1 on PR #106).
    last = int(np.where(pocket)[0][-1])
    v, ex = _snap(float(t[last] + dx / 2), cands, dx)
    end_dim = Dim(label, "ポケットの奥", v, ex)

    dmax = float(d[pocket].max())
    max_dim = None
    if dmax > max(dc_x, dc_col) + 1e-6:
        dc = [spec.well.depth if spec.well is not None else None]
        if spec.ramp_cut is not None:
            dc.append(spec.ramp_cut.depth)
        if spec.runner is not None:
            dc.append(spec.runner.depth)
        v, ex = _snap(dmax, dc, 1e-6)
        max_dim = Dim(label, "最大深さ", v, ex)

    ang = None
    if land_dim is not None and cap_dim is not None and cap_dim.value > land_dim.value:
        a = math.degrees(math.atan((dc_x - land) / (cap_dim.value - land_dim.value)))
        ang = Dim(label, "斜面角度", a, land_dim.exact and cap_dim.exact)
    return SectionDims(land_dim, cap_dim, end_dim, cap_d_dim, max_dim, ang)


def plan_dims(spec: GateProfileSpec) -> list[Dim]:
    """The plan view's dimensions, straight from the spec."""
    out = [Dim("平面", "ゲート出口幅", spec.gate_exit_width, True)]
    if spec.land.closed_line is not None:
        (t1, w1), (t2, w2) = spec.land.closed_line
        w0 = w1 if t1 >= 0 else w1 + (w2 - w1) * (0 - t1) / (t2 - t1)
        out.append(Dim("平面", "閉鎖幅（製品側）", 2 * w0 if spec.symmetric else w0, True))
    if spec.outer_wall_line is not None:
        out.append(Dim("平面", "外壁の始点 t（端）", spec.outer_wall_line[0][0], True))
    out.append(Dim("平面", "ブロックの奥行き", spec.t_max(), True))
    out.append(Dim("平面", "バルブ位置 t", spec.valve.t, True))
    return out


def _hdim(ax, x0, x1, y, text, *, ext=None, fs=5.0, color="#222"):
    """Horizontal dimension line from x0 to x1 at y (data units) with the text above."""
    ax.annotate(
        "",
        xy=(x0, y),
        xytext=(x1, y),
        arrowprops=dict(
            arrowstyle="<->", lw=0.35, color=color, shrinkA=0, shrinkB=0, mutation_scale=4
        ),
        zorder=8,
    )
    if ext is not None:
        for x in (x0, x1):
            ax.plot([x, x], [ext, y], color=color, lw=0.25, zorder=8)
    ax.text(
        (x0 + x1) / 2,
        y,
        text,
        ha="center",
        va="bottom",
        fontsize=fs,
        color=color,
        zorder=9,
        bbox=dict(fc="white", ec="none", pad=0.15, alpha=0.85),
    )


def _vdim(ax, x, y0, y1, text, *, ext=None, fs=5.0, color="#222", side="left"):
    """Vertical dimension line from y0 to y1 at x with the text beside it."""
    ax.annotate(
        "",
        xy=(x, y0),
        xytext=(x, y1),
        arrowprops=dict(
            arrowstyle="<->", lw=0.35, color=color, shrinkA=0, shrinkB=0, mutation_scale=4
        ),
        zorder=8,
    )
    if ext is not None:
        for y in (y0, y1):
            ax.plot([ext, x], [y, y], color=color, lw=0.25, zorder=8)
    ax.text(
        x,
        (y0 + y1) / 2,
        f" {text} ",
        ha="right" if side == "left" else "left",
        va="center",
        fontsize=fs,
        color=color,
        rotation=0,
        zorder=9,
        bbox=dict(fc="white", ec="none", pad=0.1, alpha=0.85),
    )


# ------------------------------- rendering --------------------------------
@dataclass(frozen=True)
class GateDrawing:
    pdf: bytes
    png: bytes
    font: JapaneseFont
    pocket_volume_mm3: float
    sections: tuple[tuple[str, float], ...]
    section_scale: float
    plan_scale: float
    dims: tuple[Dim, ...] = ()


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
    secs = section_positions(spec, field.geometry.cell_size_mm)
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
        x_lo = field.x[xs.min()] - 13.0  # room for the vertical dimensions
        x_hi = field.x[xs.max()] + 4.0
        y_lo = -field.t_max - 3.0
        y_hi = 10.0  # room for the exit width and closure dimensions
        # 1:1 unless the block is wider than the sheet (a 400 mm exit) or so
        # deep it would crowd out the sections: then 1:2, 1:4, ... (Codex P2
        # on PR #104 -- an oversized 1:1 axes ran off the page).
        plan_scale = next(
            (
                s_
                for s_ in PLAN_SCALES
                if (x_hi - x_lo) * s_ <= W - 28.0 and (y_hi - y_lo) * s_ <= 80.0
            ),
            PLAN_SCALES[-1],
        )
        plan_txt = _scale_text(plan_scale)
        pw = (x_hi - x_lo) * plan_scale
        ph = (y_hi - y_lo) * plan_scale
        p_left = (W - pw) / 2.0
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
        # ---- plan dimensions ----
        dims: list[Dim] = []
        pd = plan_dims(spec)
        dims.extend(pd)
        by = {d_.name: d_ for d_ in pd}
        wmax = exit_half_width(spec)
        xa, xb = (
            (w_to_x(spec, -wmax), w_to_x(spec, wmax))
            if spec.symmetric
            else (w_to_x(spec, 0.0), w_to_x(spec, wmax))
        )
        _hdim(ax, xa, xb, 7.0, by["ゲート出口幅"].text(), ext=0.0)
        if "閉鎖幅（製品側）" in by:
            cw = by["閉鎖幅（製品側）"].value
            ca, cb = (-cw / 2, cw / 2) if spec.symmetric else (w_to_x(spec, 0.0), w_to_x(spec, cw))
            _hdim(ax, ca, cb, 3.0, by["閉鎖幅（製品側）"].text(), ext=0.0)
        x_left = field.x[xs.min()]
        if "外壁の始点 t（端）" in by:
            _vdim(
                ax,
                x_left - 3.0,
                0.0,
                -by["外壁の始点 t（端）"].value,
                by["外壁の始点 t（端）"].text(),
                ext=x_left,
            )
        _vdim(
            ax,
            x_left - 8.0,
            0.0,
            -by["ブロックの奥行き"].value,
            by["ブロックの奥行き"].text(),
            ext=x_left,
        )
        if vm is not None:
            vx = vm[0] - x0
            r_off = (spec.well.half_width if spec.well is not None else vm[2]) + 3.0
            _vdim(ax, vx + r_off, 0.0, -spec.valve.t, by["バルブ位置 t"].text(), side="right")
            ax.text(
                vx + vm[2] + 0.3,
                -spec.valve.t - vm[2] - 0.3,
                f"Φ{_fmt_num(spec.valve.orifice_diameter)}",
                fontsize=4.8,
                ha="left",
                va="top",
                zorder=9,
            )
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect("auto")
        ax.set_xticks(np.arange(math.ceil(x_lo / 50) * 50, x_hi, 50))
        ystep = 5.0 if plan_scale >= 1.0 else 10.0
        ax.set_yticks(np.arange(math.ceil(y_lo / ystep) * ystep, y_hi, ystep))
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        frame.text(
            p_left,
            p_top - 2.0,
            f"平面図 {plan_txt}（x = 0 はバルブ軸、y = −t）",
            fontsize=6.5,
            va="bottom",
        )

        # ---- sections: 5:1 when two columns fit, smaller otherwise ----
        t_hi = field.t_max + 1.0
        d_hi = dmax + 0.6
        sec_h = d_hi + 1.6  # with the dimension band above the PL
        scale = next(
            (s_ for s_ in (5.0, 4.0, 3.0, 2.0) if 2 * (t_hi + 2.2) * s_ + 22.0 <= 380.0), 1.0
        )
        sw = (t_hi + 2.2) * scale
        sh = sec_h * scale
        s_top0 = p_top + ph + 16.0
        cols = (20.0, 20.0 + sw + 22.0)
        for k, (lab, w) in enumerate(secs):
            left = cols[k % 2]
            top = s_top0 + (k // 2) * (sh + 20.0)
            sa = axes_mm(fig, left, top, sw, sh)
            tt, dd = section_profile(field, w_to_x(spec, w))
            sa.fill_between(tt, 0.0, dd, step="mid", color="#dbe8f5", lw=0, zorder=1)
            sa.fill_between(tt, dd, d_hi, step="mid", color="#e4e4e4", lw=0, zorder=1)
            sa.step(tt, dd, where="mid", color="k", lw=0.6, zorder=3)
            sa.axhline(0.0, color="k", lw=0.8, zorder=3)
            sa.axhline(spec.land.depth, color="#1d5c9e", lw=0.3, ls=":", zorder=2)
            sa.axhline(spec.main_ramp.cap_depth, color="#1d5c9e", lw=0.3, ls=":", zorder=2)
            # The closed forms are read at the column actually drawn (within
            # half a cell of the nominal w), so a graded cap depth matches.
            sdims = section_dims(
                spec,
                lab,
                w,
                tt,
                dd,
                field.geometry.cell_size_mm,
                w_col=column_w(spec, field, w_to_x(spec, w)),
            )
            dims.extend(d_ for d_ in sdims.__dict__.values() if d_ is not None)
            lv = (-0.35, -0.8, -1.25)
            if sdims.land_end is not None:
                _hdim(sa, 0.0, sdims.land_end.value, lv[0], sdims.land_end.text(), ext=0.0)
            if sdims.cap_reach is not None:
                _hdim(sa, 0.0, sdims.cap_reach.value, lv[1], sdims.cap_reach.text(), ext=0.0)
            if sdims.pocket_end is not None:
                _hdim(sa, 0.0, sdims.pocket_end.value, lv[2], sdims.pocket_end.text(), ext=0.0)
            _vdim(sa, -0.45, 0.0, spec.land.depth, _fmt_num(spec.land.depth))
            if sdims.cap_depth is not None:
                _vdim(sa, -1.3, 0.0, sdims.cap_depth.value, sdims.cap_depth.text())
            if sdims.max_depth is not None:
                _vdim(
                    sa, t_hi - 0.4, 0.0, sdims.max_depth.value, sdims.max_depth.text(), side="left"
                )
            if (
                sdims.angle_deg is not None
                and sdims.land_end is not None
                and sdims.cap_reach is not None
            ):
                tm = (sdims.land_end.value + sdims.cap_reach.value) / 2
                dm = (spec.land.depth + sdims.cap_depth.value) / 2
                sa.text(
                    tm + 0.4,
                    dm - 0.15,
                    f"{sdims.angle_deg.text()}°",
                    fontsize=5,
                    ha="left",
                    va="bottom",
                    zorder=9,
                    bbox=dict(fc="white", ec="none", pad=0.1, alpha=0.85),
                )
            sa.set_xlim(-2.2, t_hi)
            sa.set_ylim(d_hi, -1.6)
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
                f"{lab}–{lab} 断面 {_scale_text(scale)}　w = {_fmt_num(w)}"
                + (f"（{where}）" if where else ""),
                fontsize=6.5,
                va="bottom",
            )
        s_bottom = s_top0 + 2 * (sh + 20.0)

        # ---- notes, bottom left beside the title block ----
        notes = [
            f"平面図 {plan_txt}、断面 {_scale_text(scale)}（縦横とも）。単位 mm。",
            "深さはパーティングラインからの深さ。",
            "斜線: ランド（深さ = ランド深さ）。",
            f"細線: 深さ {_fmt_num(CONTOUR_STEP_MM)} mm ごとの等深線。",
            "破線の円: バルブゲート。一点鎖線: 断面の位置。",
            "断面の点線: ランド深さと斜面の上限深さ。",
            f"網目 {_fmt_num(field.geometry.cell_size_mm)} mm で組んだ形状を描いている"
            + (f"（解析は {_fmt_num(solver_cell_mm)} mm）。" if solver_cell_mm else "。"),
            "アプリの入力から自動で作図した。",
            "寸法の ≈ は網目から読んだ値（spec の式と形が合わない所）。",
            "寸法は寸法表を正とする。",
        ]
        n_top = H - 14.0 - 26.0 + 1.0
        frame.text(14.0, n_top - 1.5, "注記", fontsize=6.5, va="bottom")
        per = math.ceil(len(notes) / 2)
        for i, s_ in enumerate(notes):
            c, r = divmod(i, per)
            frame.text(
                14.0 + c * 128.0, n_top + 2.5 + r * 3.6, f"・{s_}", fontsize=5.6, va="center"
            )

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
            f"尺度　平面 {plan_txt}／断面 {_scale_text(scale)}　　単位 mm",
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
        plan_scale=plan_scale,
        dims=tuple(dims),
    )


def spec_key(spec: GateProfileSpec, plate: ProfilePlateConfig) -> str:
    """A stable text key for caching a drawing of this exact shape."""
    import json

    return json.dumps({"spec": spec.to_dict(), "plate": dataclasses.asdict(plate)}, sort_keys=True)
