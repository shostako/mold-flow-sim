"""Profile-gate geometry: JSON-spec-driven gate-block depth field.

This module rasterizes a machined gate-block "depth field" (as extracted
from a 2D drawing) into a :class:`~core.geometry.Geometry`. The gate block
sits below a rectangular product plate and feeds it through a thin land.

Depth-field model (coordinates: ``t`` = flow-direction distance from the
gate exit / product edge [mm], ``w`` = width-direction position [mm],
``d`` = channel thickness = pocket depth [mm]):

1. **Land** — ``t ∈ [0, land.length]``, full gate width, ``d = land.depth``.
2. **Main ramp** — ``d = land.depth + tan(angle)·(t − land.length)``,
   capped at ``cap_depth`` (flat beyond the cap point).
3. **Island** (optional) — a shallow central band bounded by a straight
   ``boundary_line`` in the (t, w) plane; ``d`` follows the (smaller)
   island angle and the island ends at ``end_dist``. The steep (~60°)
   island edge/end walls are approximated as sharp cuts — at the intended
   cell resolution (~1 mm) the horizontal wall extent is about one cell.
4. **Outer wall** — a straight line in the (t, w) plane beyond which the
   pocket (cavity) ends. For ``t`` before the line's first point the pocket
   spans the full gate width.
5. **Well** (optional) — an obround (capsule) pocket around the valve pin.
   Modeled with a distance field so its sloped wall (``wall_angle_deg``)
   contributes correctly to volume: ``d_well = clip((half_width − dist)·
   tan(wall_angle), 0, depth)`` and ``d = max(d, d_well)``.
6. ``symmetric=True`` mirrors the half-width profile about the valve axis;
   ``symmetric=False`` places the valve at the ``w=0`` end of a one-sided
   band (the well may overhang past the ``w=0`` edge).

Confidentiality note: this repository ships only the format definition,
the builder, and a **fictional-dimension demo spec**
(``data/gate_profiles/demo_profile_gate.json``). Real drawing-derived
specs must stay outside the repo and be loaded locally at runtime.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import Geometry

_EPS = 1e-6


# ---------------------------------------------------------------------------
# nested spec dataclasses (mirror the JSON structure 1:1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LandSpec:
    """Gate land: constant-depth strip attaching the block to the plate."""

    depth: float  # channel thickness [mm]
    length: float  # extent in t [mm]


@dataclass(frozen=True)
class MainRampSpec:
    """Main pocket ramp, capped at ``cap_depth``."""

    angle_deg: float
    cap_depth: float


@dataclass(frozen=True)
class IslandSpec:
    """Shallow central island (flow restrictor)."""

    angle_deg: float
    boundary_line: tuple[tuple[float, float], tuple[float, float]]  # ((t1,w1),(t2,w2))
    end_dist: float


@dataclass(frozen=True)
class WellSpec:
    """Obround valve well (deep pocket around the valve pin)."""

    shape: str  # only "obround" is supported
    t_range: tuple[float, float]
    half_width: float
    depth: float
    floor_t_range: tuple[float, float]
    wall_angle_deg: float = 60.0


@dataclass(frozen=True)
class ValveSpec:
    """Valve-gate pin position and orifice (Dirichlet τ=0 boundary)."""

    t: float
    w: float
    orifice_diameter: float


# ---------------------------------------------------------------------------
# from_dict helpers
# ---------------------------------------------------------------------------


def _req(d: dict, key: str, path: str) -> Any:
    if key not in d:
        raise ValueError(f"gate profile JSON: missing key '{path}{key}'")
    return d[key]


def _num(d: dict, key: str, path: str) -> float:
    val = _req(d, key, path)
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValueError(f"gate profile JSON: '{path}{key}' must be a number, got {val!r}")
    return float(val)


def _line(d: dict, key: str, path: str) -> tuple[tuple[float, float], tuple[float, float]]:
    val = _req(d, key, path)
    try:
        (t1, w1), (t2, w2) = val
        return ((float(t1), float(w1)), (float(t2), float(w2)))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"gate profile JSON: '{path}{key}' must be [[t1, w1], [t2, w2]], got {val!r}"
        ) from exc


def _pair(d: dict, key: str, path: str) -> tuple[float, float]:
    val = _req(d, key, path)
    try:
        a, b = val
        return (float(a), float(b))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"gate profile JSON: '{path}{key}' must be [a, b], got {val!r}") from exc


def _check_unknown(d: dict, known: set[str], path: str) -> None:
    unknown = set(d) - known
    if unknown:
        raise ValueError(
            f"gate profile JSON: unknown key(s) {sorted(unknown)} under '{path or 'root'}'"
        )


# ---------------------------------------------------------------------------
# top-level spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateProfileSpec:
    """Gate-block depth-field specification (JSON-serializable).

    Coordinates: ``t`` = distance from the gate exit (product edge) along
    the flow direction [mm], ``w`` = width position [mm] (half-width from
    the valve axis when ``symmetric``, else offset from the valve-side
    edge). ``units`` must be ``"mm"``.
    """

    name: str
    units: str
    symmetric: bool
    gate_exit_width: float
    land: LandSpec
    main_ramp: MainRampSpec
    outer_wall_line: tuple[tuple[float, float], tuple[float, float]]
    valve: ValveSpec
    island: IslandSpec | None = None
    well: WellSpec | None = None

    # ---- JSON I/O ----

    @classmethod
    def from_dict(cls, d: dict) -> GateProfileSpec:
        if not isinstance(d, dict):
            raise ValueError(f"gate profile JSON: root must be an object, got {type(d).__name__}")
        _check_unknown(
            d,
            {
                "name",
                "units",
                "symmetric",
                "gate_exit_width",
                "land",
                "main_ramp",
                "island",
                "outer_wall_line",
                "well",
                "valve",
            },
            "",
        )

        name = str(_req(d, "name", ""))
        units = str(_req(d, "units", ""))
        symmetric = _req(d, "symmetric", "")
        if not isinstance(symmetric, bool):
            raise ValueError(
                f"gate profile JSON: 'symmetric' must be true/false, got {symmetric!r}"
            )
        gate_exit_width = _num(d, "gate_exit_width", "")

        land_d = _req(d, "land", "")
        _check_unknown(land_d, {"depth", "length"}, "land")
        land = LandSpec(
            depth=_num(land_d, "depth", "land."), length=_num(land_d, "length", "land.")
        )

        ramp_d = _req(d, "main_ramp", "")
        _check_unknown(ramp_d, {"angle_deg", "cap_depth"}, "main_ramp")
        main_ramp = MainRampSpec(
            angle_deg=_num(ramp_d, "angle_deg", "main_ramp."),
            cap_depth=_num(ramp_d, "cap_depth", "main_ramp."),
        )

        island: IslandSpec | None = None
        if d.get("island") is not None:
            isl_d = d["island"]
            _check_unknown(isl_d, {"angle_deg", "boundary_line", "end_dist"}, "island")
            island = IslandSpec(
                angle_deg=_num(isl_d, "angle_deg", "island."),
                boundary_line=_line(isl_d, "boundary_line", "island."),
                end_dist=_num(isl_d, "end_dist", "island."),
            )

        outer_wall_line = _line(d, "outer_wall_line", "")

        well: WellSpec | None = None
        if d.get("well") is not None:
            well_d = d["well"]
            _check_unknown(
                well_d,
                {"shape", "t_range", "half_width", "depth", "floor_t_range", "wall_angle_deg"},
                "well",
            )
            well = WellSpec(
                shape=str(_req(well_d, "shape", "well.")),
                t_range=_pair(well_d, "t_range", "well."),
                half_width=_num(well_d, "half_width", "well."),
                depth=_num(well_d, "depth", "well."),
                floor_t_range=_pair(well_d, "floor_t_range", "well."),
                wall_angle_deg=(
                    _num(well_d, "wall_angle_deg", "well.") if "wall_angle_deg" in well_d else 60.0
                ),
            )

        valve_d = _req(d, "valve", "")
        _check_unknown(valve_d, {"t", "w", "orifice_diameter"}, "valve")
        valve = ValveSpec(
            t=_num(valve_d, "t", "valve."),
            w=_num(valve_d, "w", "valve."),
            orifice_diameter=_num(valve_d, "orifice_diameter", "valve."),
        )

        spec = cls(
            name=name,
            units=units,
            symmetric=symmetric,
            gate_exit_width=gate_exit_width,
            land=land,
            main_ramp=main_ramp,
            outer_wall_line=outer_wall_line,
            valve=valve,
            island=island,
            well=well,
        )
        spec.validate()
        return spec

    @classmethod
    def from_json(cls, text: str) -> GateProfileSpec:
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_json_file(cls, path: Path | str) -> GateProfileSpec:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "name": self.name,
            "units": self.units,
            "symmetric": self.symmetric,
            "gate_exit_width": self.gate_exit_width,
            "land": {"depth": self.land.depth, "length": self.land.length},
            "main_ramp": {
                "angle_deg": self.main_ramp.angle_deg,
                "cap_depth": self.main_ramp.cap_depth,
            },
            "outer_wall_line": [list(p) for p in self.outer_wall_line],
            "valve": {
                "t": self.valve.t,
                "w": self.valve.w,
                "orifice_diameter": self.valve.orifice_diameter,
            },
        }
        if self.island is not None:
            d["island"] = {
                "angle_deg": self.island.angle_deg,
                "boundary_line": [list(p) for p in self.island.boundary_line],
                "end_dist": self.island.end_dist,
            }
        if self.well is not None:
            d["well"] = {
                "shape": self.well.shape,
                "t_range": list(self.well.t_range),
                "half_width": self.well.half_width,
                "depth": self.well.depth,
                "floor_t_range": list(self.well.floor_t_range),
                "wall_angle_deg": self.well.wall_angle_deg,
            }
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    # ---- derived quantities ----

    def ramp_cap_t(self) -> float:
        """t position where the main ramp reaches ``cap_depth``."""
        tan_ramp = math.tan(math.radians(self.main_ramp.angle_deg))
        return self.land.length + (self.main_ramp.cap_depth - self.land.depth) / max(
            tan_ramp, 1e-12
        )

    def t_max(self) -> float:
        """Total t-extent of the gate block."""
        candidates = [
            self.outer_wall_line[1][0],
            self.ramp_cap_t(),
            self.valve.t + self.valve.orifice_diameter / 2.0,
        ]
        if self.well is not None:
            candidates.append(self.well.t_range[1])
        if self.island is not None:
            candidates.append(self.island.end_dist)
        return max(candidates)

    # ---- validation ----

    def validate(self) -> None:
        if self.units != "mm":
            raise ValueError(f"units must be 'mm', got {self.units!r}")
        for label, val in (
            ("gate_exit_width", self.gate_exit_width),
            ("land.depth", self.land.depth),
            ("land.length", self.land.length),
            ("valve.orifice_diameter", self.valve.orifice_diameter),
        ):
            if val <= 0:
                raise ValueError(f"{label} must be positive, got {val}")
        if not (0.0 < self.main_ramp.angle_deg < 89.0):
            raise ValueError(
                f"main_ramp.angle_deg must be in (0, 89), got {self.main_ramp.angle_deg}"
            )
        if self.main_ramp.cap_depth < self.land.depth - _EPS:
            raise ValueError(
                f"main_ramp.cap_depth ({self.main_ramp.cap_depth}) must be ≥ "
                f"land.depth ({self.land.depth})"
            )
        (wt1, ww1), (wt2, ww2) = self.outer_wall_line
        if wt2 <= wt1 + _EPS:
            raise ValueError(f"outer_wall_line t must be increasing, got {wt1} → {wt2}")
        if ww1 <= 0 or ww2 <= 0:
            raise ValueError(f"outer_wall_line w must be positive, got {ww1}, {ww2}")
        if self.valve.t < 0:
            raise ValueError(f"valve.t must be ≥ 0, got {self.valve.t}")

        if self.island is not None:
            isl = self.island
            if isl.angle_deg < 0:
                raise ValueError(f"island.angle_deg must be ≥ 0, got {isl.angle_deg}")
            if isl.angle_deg > self.main_ramp.angle_deg + _EPS:
                raise ValueError(
                    f"island.angle_deg ({isl.angle_deg}) must be ≤ "
                    f"main_ramp.angle_deg ({self.main_ramp.angle_deg}); "
                    f"the island is the shallow side"
                )
            (it1, _iw1), (it2, _iw2) = isl.boundary_line
            if it2 <= it1 + _EPS:
                raise ValueError(f"island.boundary_line t must be increasing, got {it1} → {it2}")
            if isl.end_dist <= self.land.length + _EPS:
                raise ValueError(
                    f"island.end_dist ({isl.end_dist}) must be > land.length ({self.land.length})"
                )

        if self.well is not None:
            w = self.well
            if w.shape != "obround":
                raise ValueError(f"well.shape must be 'obround', got {w.shape!r}")
            if w.t_range[0] >= w.t_range[1] - _EPS:
                raise ValueError(f"well.t_range must be increasing, got {w.t_range}")
            if w.half_width <= 0 or w.depth <= 0:
                raise ValueError(
                    f"well.half_width and well.depth must be positive, "
                    f"got {w.half_width}, {w.depth}"
                )
            if not (0.0 < w.wall_angle_deg <= 90.0):
                raise ValueError(f"well.wall_angle_deg must be in (0, 90], got {w.wall_angle_deg}")
            if (
                w.floor_t_range[0] < w.t_range[0] - _EPS
                or w.floor_t_range[1] > w.t_range[1] + _EPS
                or w.floor_t_range[0] >= w.floor_t_range[1] - _EPS
            ):
                raise ValueError(
                    f"well.floor_t_range ({w.floor_t_range}) must be an increasing range "
                    f"inside well.t_range ({w.t_range})"
                )
            r_floor = w.half_width - (w.floor_t_range[0] - w.t_range[0])
            if r_floor < -_EPS:
                raise ValueError(
                    f"well.floor_t_range inset ({w.floor_t_range[0] - w.t_range[0]}) "
                    f"must be ≤ well.half_width ({w.half_width})"
                )


# ---------------------------------------------------------------------------
# plate config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProfilePlateConfig:
    """Rectangular product plate fed by a profile gate.

    Same vocabulary as the plate section of ``FilmGateConfig``: an optional
    two-band thickness split at ``plate_split_height_mm`` from the gate-side
    edge (``0`` = uniform ``plate_thk_mm``).
    """

    plate_w_mm: float = 300.0
    plate_h_mm: float = 50.0
    plate_thk_mm: float = 0.4
    plate_split_height_mm: float = 0.0
    plate_lower_thk_mm: float | None = None
    plate_upper_thk_mm: float | None = None
    pad_mm: float = 5.0

    def resolved_plate_zones(self) -> tuple[float, float, float]:
        """Return ``(split_height_mm, lower_thk_mm, upper_thk_mm)``."""
        if self.plate_split_height_mm > 0:
            lower = (
                self.plate_lower_thk_mm
                if self.plate_lower_thk_mm is not None
                else self.plate_thk_mm
            )
            upper = (
                self.plate_upper_thk_mm
                if self.plate_upper_thk_mm is not None
                else self.plate_thk_mm
            )
            return float(self.plate_split_height_mm), float(lower), float(upper)
        return 0.0, float(self.plate_thk_mm), float(self.plate_thk_mm)

    def validate(self) -> None:
        for label, val in (
            ("plate_w_mm", self.plate_w_mm),
            ("plate_h_mm", self.plate_h_mm),
            ("plate_thk_mm", self.plate_thk_mm),
            ("pad_mm", self.pad_mm),
        ):
            if val <= 0:
                raise ValueError(f"{label} must be positive, got {val}")
        if self.plate_split_height_mm < 0 or self.plate_split_height_mm > self.plate_h_mm + _EPS:
            raise ValueError(
                f"plate_split_height_mm ({self.plate_split_height_mm}) must be in "
                f"[0, plate_h_mm ({self.plate_h_mm})]"
            )
        for label, val in (
            ("plate_lower_thk_mm", self.plate_lower_thk_mm),
            ("plate_upper_thk_mm", self.plate_upper_thk_mm),
        ):
            if val is not None and val <= 0:
                raise ValueError(f"{label} must be positive, got {val}")


# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------


def _line_eval(
    line: tuple[tuple[float, float], tuple[float, float]],
    t: np.ndarray,
    *,
    before_value: float,
) -> np.ndarray:
    """Evaluate a straight (t, w) boundary line at each t.

    For ``t`` before the first point the result clamps to ``before_value``
    (the pocket spans the full gate width near the exit); beyond the second
    point the line is linearly extrapolated (clipped at 0 by the caller's
    ``wa ≤ w`` test since w goes negative quickly).
    """
    (t1, w1), (t2, w2) = line
    slope = (w2 - w1) / max(t2 - t1, 1e-12)
    w = w1 + slope * (t - t1)
    return np.where(t < t1, before_value, w)


def build_profile_gate_geometry(
    spec: GateProfileSpec,
    plate: ProfilePlateConfig,
    cell_size_mm: float = 1.0,
) -> Geometry:
    """Rasterize a :class:`GateProfileSpec` + plate into a :class:`Geometry`.

    Layout (y up, same convention as the other builders): the gate block
    occupies ``t ∈ [0, spec.t_max()]`` below the plate; the plate sits on
    top. ``t = y_plate_bottom − y`` so the land row attaches directly to
    the plate's bottom row. The valve orifice becomes the Dirichlet gate
    cells; the plate body alone forms the compression mask.
    """
    spec.validate()
    plate.validate()
    if cell_size_mm <= 0:
        raise ValueError(f"cell_size_mm must be positive, got {cell_size_mm}")
    if spec.gate_exit_width > plate.plate_w_mm + _EPS:
        raise ValueError(
            f"gate_exit_width ({spec.gate_exit_width}) must be ≤ plate_w_mm ({plate.plate_w_mm})"
        )

    pad = plate.pad_mm
    Wp = plate.plate_w_mm
    Hp = plate.plate_h_mm
    split_h, h_plate_lower, h_plate_upper = plate.resolved_plate_zones()
    gew = spec.gate_exit_width
    dx = cell_size_mm

    T = spec.t_max()
    y_plate_bottom = pad + T
    y_plate_top = y_plate_bottom + Hp
    total_w = 2 * pad + Wp
    total_h = 2 * pad + T + Hp
    nx = int(round(total_w / dx))
    ny = int(round(total_h / dx))

    # cell-center coordinates [mm]
    iy_idx, ix_idx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    yy = (iy_idx + 0.5) * dx
    xx = (ix_idx + 0.5) * dx

    t = y_plate_bottom - yy  # ≥ 0 inside the gate block, < 0 inside the plate

    # width coordinate wa and valve x position
    cx = pad + Wp / 2.0
    if spec.symmetric:
        x_valve = cx + spec.valve.w
        wa = np.abs(xx - x_valve)
        full_half_width = gew / 2.0
    else:
        # one-sided band: valve-side edge is w=0; gate centered on the plate
        x_edge = cx - gew / 2.0
        x_valve = x_edge + spec.valve.w
        wa = xx - x_edge
        full_half_width = gew

    # --- base depth field (land + capped main ramp) ---
    land_depth = spec.land.depth
    land_len = spec.land.length
    tan_ramp = math.tan(math.radians(spec.main_ramp.angle_deg))
    d_base = np.where(
        t <= land_len,
        land_depth,
        np.minimum(land_depth + tan_ramp * (t - land_len), spec.main_ramp.cap_depth),
    )

    # --- island override (sharp-cut approximation of the steep walls) ---
    if spec.island is not None:
        isl = spec.island
        tan_isl = math.tan(math.radians(isl.angle_deg))
        w_bound = _line_eval(isl.boundary_line, t, before_value=isl.boundary_line[0][1])
        in_island = (t > land_len) & (t <= isl.end_dist) & (wa <= w_bound) & (wa >= 0)
        d_base = np.where(in_island, land_depth + tan_isl * (t - land_len), d_base)

    # --- outer wall (pocket silhouette) ---
    w_wall = _line_eval(spec.outer_wall_line, t, before_value=full_half_width)
    in_gate_base = (t >= 0) & (t <= T) & (wa >= 0) & (wa <= w_wall)

    # --- well (obround capsule with sloped wall, distance field) ---
    in_well = np.zeros_like(in_gate_base)
    d_well = np.zeros_like(d_base)
    if spec.well is not None:
        well = spec.well
        tan_wall = math.tan(math.radians(well.wall_angle_deg))
        # capsule axis: vertical segment at x = x_valve,
        # t from t_range[0]+half_width to t_range[1]-half_width
        t_axis_lo = well.t_range[0] + well.half_width
        t_axis_hi = well.t_range[1] - well.half_width
        if t_axis_hi < t_axis_lo:  # degenerate obround → circle at midpoint
            t_axis_lo = t_axis_hi = 0.5 * (well.t_range[0] + well.t_range[1])
        y_axis_lo = y_plate_bottom - t_axis_hi
        y_axis_hi = y_plate_bottom - t_axis_lo
        dy = yy - np.clip(yy, y_axis_lo, y_axis_hi)
        dist = np.hypot(xx - x_valve, dy)
        d_well = np.clip((well.half_width - dist) * tan_wall, 0.0, well.depth)
        in_well = d_well > 1e-9

    # gate cells: pocket silhouette plus any well overhang
    in_gate = (in_gate_base | in_well) & (t >= 0) & (t <= T)
    d_gate = np.maximum(np.where(in_gate_base, d_base, 0.0), d_well)

    # --- plate ---
    in_plate = (yy >= y_plate_bottom) & (yy <= y_plate_top) & (xx >= pad) & (xx <= pad + Wp)

    mask = in_gate | in_plate

    # --- gate land wall (close plate-bottom row outside the gate exit) ---
    plate_rows = np.where(in_plate.any(axis=1))[0]
    if plate_rows.size > 0:
        iy_plate_bottom = int(plate_rows[0])
        row_wa = wa[iy_plate_bottom, :]
        ix_close = (row_wa < 0) | (row_wa > full_half_width)
        mask[iy_plate_bottom, ix_close] = False

    # --- thickness ---
    thk = np.zeros_like(xx, dtype=float)
    thk[in_gate] = d_gate[in_gate]
    thk[in_plate] = h_plate_lower
    if split_h > 0:
        upper_zone = in_plate & (yy >= y_plate_bottom + split_h)
        thk[upper_zone] = h_plate_upper
    thk[~mask] = 0.0

    # --- valve orifice (circular Dirichlet) ---
    y_valve = y_plate_bottom - spec.valve.t
    r_orifice = spec.valve.orifice_diameter / 2.0
    in_valve = ((xx - x_valve) ** 2 + (yy - y_valve) ** 2) <= r_orifice**2
    valve_iys, valve_ixs = np.where(in_valve & mask)

    compression_mask = in_plate & mask

    geom = Geometry(
        mask=mask,
        thickness_mm=thk,
        cell_size_mm=dx,
        label="profile_gate",
        compression_mask=compression_mask,
    )
    if valve_iys.size == 0:
        # Defensive: snap to the in-mask cell nearest to the valve center
        # (e.g. tiny orifice, or an asymmetric spec whose orifice circle
        # only half-overlaps the pocket).
        masked_iys, masked_ixs = np.where(mask)
        if masked_iys.size > 0:
            d2 = (yy[masked_iys, masked_ixs] - y_valve) ** 2 + (
                xx[masked_iys, masked_ixs] - x_valve
            ) ** 2
            k = int(np.argmin(d2))
            geom.gates.append((int(masked_iys[k]), int(masked_ixs[k])))
    else:
        for iy, ix in zip(valve_iys, valve_ixs, strict=True):
            geom.gates.append((int(iy), int(ix)))

    return geom
