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
   An optional ``weld`` sub-section models a **welded-in dam**: within
   ``t_range`` the island depth is overridden by a constant ``depth``
   (metal deposited on the pocket floor, so ``depth ≤ land.depth``). Its
   entry/exit steps are sharp cuts like the island's own walls. A weld depth
  of 0 means the steel reaches the PL: those cells leave the cavity (a hole
  in the part), except where the well cuts through them.
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
7. **Sub-gates** (optional, replaces the outer wall) — ``sub_gates`` is a
   list of fan-shaped pockets, each bounded by an ``inner_wall_line`` and an
   ``outer_wall_line`` in the (t, w) plane and ending at ``tip_t``. Inside a
   fan the depth is the land + main ramp, optionally overridden by the fan's
   own ``island`` (a shallow band between ``inner_line`` and ``outer_line``,
   ending at ``end_dist``). Everything outside the fans is steel at the PL:
   with two mirrored fans whose inner walls meet on the land at ``w=0`` the
   steel between them is the "deformed rhombus" full cut-out. Exactly one of
   ``outer_wall_line`` / ``sub_gates`` must be given.
8. **Runner** (optional) — a constant-depth band of ``width`` along a
   ``path`` polyline in (t, w); ``d = max(d, depth)`` within ``width/2`` of
   the path, so it can cross steel (where it *is* the pocket) or a fan. With
   sub-gates it is the channel that carries the melt from the valve well to
   each fan tip.
9. **Edge channels** (optional) — deepened bands hugging a pocket wall
   (縁部深彫り): ``d = max(d, depth)`` for pocket cells within ``width``
   (perpendicular distance) of the effective wall polyline, clipped to
   ``t_range``. Top-level ``edge_channels`` follow the single-pocket
   ``outer_wall_line`` (mirrored with the pocket when ``symmetric``); each
   sub-gate fan carries its own list with ``side`` = ``"outer"`` /
   ``"inner"``. Unlike the runner they never extend the silhouette — they
   only deepen existing cavity cells, turning the rim into a
   low-resistance raceway (S ∝ h³).

Confidentiality note: this repository ships only the format definition,
the builder, and a **fictional-dimension demo spec**
(``data/gate_profiles/demo_profile_gate.json``). Real drawing-derived
specs must stay outside the repo and be loaded locally at runtime.
"""

from __future__ import annotations

import dataclasses
import json
import math
import numbers
from collections.abc import Iterator
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
class WeldSpec:
    """Welded-in dam inside the island.

    Weld metal deposited on the island floor over ``t_range``, leaving a
    constant channel thickness ``depth``. Since it fills the pocket it can
    only make the channel shallower (``depth ≤ land.depth``).
    """

    t_range: tuple[float, float]
    depth: float


@dataclass(frozen=True)
class IslandSpec:
    """Shallow central island (flow restrictor)."""

    angle_deg: float
    boundary_line: tuple[tuple[float, float], tuple[float, float]]  # ((t1,w1),(t2,w2))
    end_dist: float
    weld: WeldSpec | None = None


@dataclass(frozen=True)
class WellSpec:
    """Obround valve well (deep pocket around the valve pin).

    The rasterizer derives the well floor from ``depth`` and
    ``wall_angle_deg`` via the distance field; ``floor_t_range`` is
    optional drawing-reference metadata (the floor extent as dimensioned
    on the drawing) kept for round-trip fidelity. It does not affect the
    rasterized geometry.
    """

    shape: str  # only "obround" is supported
    t_range: tuple[float, float]
    half_width: float
    depth: float
    floor_t_range: tuple[float, float] | None = None
    wall_angle_deg: float = 60.0


@dataclass(frozen=True)
class ValveSpec:
    """Valve-gate pin position and orifice (Dirichlet τ=0 boundary)."""

    t: float
    w: float
    orifice_diameter: float


Line = tuple[tuple[float, float], tuple[float, float]]


@dataclass(frozen=True)
class SubIslandSpec:
    """Shallow band inside one sub-gate fan, bounded on both sides.

    Unlike :class:`IslandSpec` (which runs from the valve axis out to a
    single boundary line) a fan's island is a band between two lines,
    ``inner_line`` and ``outer_line``, both ``((t1, w1), (t2, w2))``.
    """

    angle_deg: float
    inner_line: Line
    outer_line: Line
    end_dist: float


@dataclass(frozen=True)
class SubGateSpec:
    """One fan-shaped pocket of a multi-fan gate block.

    The fan is the set ``t ∈ [0, tip_t]``, ``inner_wall(t) ≤ w ≤
    outer_wall(t)``; before a line's first point its width clamps to that
    point's ``w`` (so a wall starting at ``(land.length, 0)`` lets the land
    strip run to the valve axis and the fans meet there in a sharp apex).
    """

    inner_wall_line: Line
    outer_wall_line: Line
    tip_t: float
    island: SubIslandSpec | None = None
    edge_channels: tuple[EdgeChannelSpec, ...] = ()


@dataclass(frozen=True)
class RunnerSpec:
    """Constant-depth runner band along a (t, w) polyline."""

    width: float
    depth: float
    path: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class EdgeChannelSpec:
    """Deepened band hugging a pocket wall (edge channel / 縁部深彫り).

    A strip of constant channel thickness ``depth`` along a wall line,
    ``width`` mm wide measured as the perpendicular distance to the wall
    in the (t, w) plane. It only *deepens* cells already inside the pocket
    (``d = max(d, depth)``, same floor semantics as the runner and the
    well) — the silhouette is untouched, so it can never disconnect or
    extend the cavity. Physically it is a milled groove along the fan
    edge: ``S ∝ h³`` makes the rim a low-resistance raceway that carries
    the melt to the far corners of the fan before the interior fills.

    ``t_range`` limits the band's extent along the wall (``None`` = the
    wall's full extent). ``side`` selects the wall in a sub-gate fan
    (``"outer"`` / ``"inner"``); the single-pocket form has only an outer
    wall, so there ``side`` must stay ``"outer"``.
    """

    width: float
    depth: float
    t_range: tuple[float, float] | None = None
    side: str = "outer"


# ---------------------------------------------------------------------------
# from_dict helpers
# ---------------------------------------------------------------------------


def _req(d: dict, key: str, path: str) -> Any:
    if key not in d:
        raise ValueError(f"gate profile JSON: missing key '{path}{key}'")
    return d[key]


def _scalar(val: Any, label: str) -> float:
    """Accept a JSON number (or any real scalar), reject bool and everything else."""
    if isinstance(val, bool) or not isinstance(val, numbers.Real):
        raise ValueError(f"gate profile JSON: '{label}' must be a number, got {val!r}")
    return float(val)


def _num(d: dict, key: str, path: str) -> float:
    return _scalar(_req(d, key, path), f"{path}{key}")


def _elements(val: Any, count: int, label: str, shape: str) -> list:
    """Require an actual sequence of exactly ``count`` items.

    Unpacking (``a, b = val``) would happily take any 2-item iterable — and a
    two-character string is one. ``"t_range": "68"`` then parses as
    ``(6.0, 8.0)``: a valid-looking band in a completely different place,
    accepted without a word. Malformed JSON must be rejected, not reinterpreted.
    """
    if isinstance(val, (str, bytes)) or not isinstance(val, (list, tuple)) or len(val) != count:
        raise ValueError(f"gate profile JSON: '{label}' must be {shape}, got {val!r}")
    return list(val)


def _line(d: dict, key: str, path: str) -> tuple[tuple[float, float], tuple[float, float]]:
    label = f"{path}{key}"
    shape = "[[t1, w1], [t2, w2]]"
    pts = []
    for pt in _elements(_req(d, key, path), 2, label, shape):
        a, b = _elements(pt, 2, label, shape)
        pts.append((_scalar(a, label), _scalar(b, label)))
    return (pts[0], pts[1])


def _pair(d: dict, key: str, path: str) -> tuple[float, float]:
    label = f"{path}{key}"
    a, b = _elements(_req(d, key, path), 2, label, "[a, b]")
    return (_scalar(a, label), _scalar(b, label))


def _section(d: dict, key: str, *, required: bool) -> dict | None:
    """Fetch a nested JSON object, rejecting non-object values with a path."""
    val = _req(d, key, "") if required else d.get(key)
    if val is None and not required:
        return None
    if not isinstance(val, dict):
        raise ValueError(
            f"gate profile JSON: '{key}' must be an object, got {type(val).__name__} ({val!r})"
        )
    return val


def _edge_channel_dict(ec: EdgeChannelSpec) -> dict:
    ec_d: dict[str, Any] = {"width": ec.width, "depth": ec.depth, "side": ec.side}
    if ec.t_range is not None:
        ec_d["t_range"] = list(ec.t_range)
    return ec_d


def _edge_channels(d: dict, path: str) -> tuple[EdgeChannelSpec, ...]:
    """Parse an optional ``edge_channels`` list under ``path``."""
    val = d.get("edge_channels")
    if val is None:
        return ()
    label = f"{path}edge_channels"
    if isinstance(val, (str, bytes, dict)) or not isinstance(val, (list, tuple)):
        raise ValueError(f"gate profile JSON: '{label}' must be a list of objects, got {val!r}")
    channels = []
    for i, ec_d in enumerate(val):
        p = f"{label}[{i}]"
        if not isinstance(ec_d, dict):
            raise ValueError(
                f"gate profile JSON: '{p}' must be an object, got {type(ec_d).__name__}"
            )
        _check_unknown(ec_d, {"width", "depth", "t_range", "side"}, p)
        channels.append(
            EdgeChannelSpec(
                width=_num(ec_d, "width", f"{p}."),
                depth=_num(ec_d, "depth", f"{p}."),
                t_range=(
                    _pair(ec_d, "t_range", f"{p}.") if ec_d.get("t_range") is not None else None
                ),
                side=str(ec_d.get("side", "outer")),
            )
        )
    return tuple(channels)


def _check_unknown(d: dict, known: set[str], path: str) -> None:
    unknown = set(d) - known
    if unknown:
        raise ValueError(
            f"gate profile JSON: unknown key(s) {sorted(unknown)} under '{path or 'root'}'"
        )


# ---------------------------------------------------------------------------
# top-level spec
# ---------------------------------------------------------------------------


def _iter_numbers(obj: Any, path: str) -> Iterator[tuple[str, float]]:
    """Yield ``(dotted path, value)`` for every number reachable from ``obj``.

    Walks dataclass fields and sequences, so a numeric field added later is
    covered without touching the check that consumes this.

    The scalar test is ``numbers.Real``, not ``(int, float)``: a spec built
    from NumPy-derived values (an optimizer sweep, a value read off an
    array) can hold ``np.float32`` / ``np.int64``, and those are *not*
    subclasses of the builtins — they would be skipped silently, which is
    the one failure mode this walk exists to prevent. ``np.float64`` happens
    to subclass ``float`` and would have been caught either way; relying on
    that is an accident, not a guarantee. ``bool`` is a ``Real`` too, so it
    is filtered out first.
    """
    if obj is None or isinstance(obj, (bool, str)):
        return
    if isinstance(obj, numbers.Real):
        yield path, float(obj)
    elif dataclasses.is_dataclass(obj):
        for f in dataclasses.fields(obj):
            child = getattr(obj, f.name)
            yield from _iter_numbers(child, f"{path}.{f.name}" if path else f.name)
    elif isinstance(obj, (tuple, list)):
        for i, v in enumerate(obj):
            yield from _iter_numbers(v, f"{path}[{i}]")


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
    # Single-pocket form: the silhouette line. ``None`` when ``sub_gates``
    # describe the pocket instead (exactly one of the two must be given).
    outer_wall_line: Line | None
    valve: ValveSpec
    island: IslandSpec | None = None
    well: WellSpec | None = None
    sub_gates: tuple[SubGateSpec, ...] = ()
    runner: RunnerSpec | None = None
    # Single-pocket form only: bands along the outer wall. Fans carry their
    # own ``SubGateSpec.edge_channels``.
    edge_channels: tuple[EdgeChannelSpec, ...] = ()

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
                "sub_gates",
                "runner",
                "edge_channels",
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

        land_d = _section(d, "land", required=True)
        _check_unknown(land_d, {"depth", "length"}, "land")
        land = LandSpec(
            depth=_num(land_d, "depth", "land."), length=_num(land_d, "length", "land.")
        )

        ramp_d = _section(d, "main_ramp", required=True)
        _check_unknown(ramp_d, {"angle_deg", "cap_depth"}, "main_ramp")
        main_ramp = MainRampSpec(
            angle_deg=_num(ramp_d, "angle_deg", "main_ramp."),
            cap_depth=_num(ramp_d, "cap_depth", "main_ramp."),
        )

        island: IslandSpec | None = None
        isl_d = _section(d, "island", required=False)
        if isl_d is not None:
            _check_unknown(isl_d, {"angle_deg", "boundary_line", "end_dist", "weld"}, "island")
            weld: WeldSpec | None = None
            weld_d = _section(isl_d, "weld", required=False)
            if weld_d is not None:
                _check_unknown(weld_d, {"t_range", "depth"}, "island.weld")
                weld = WeldSpec(
                    t_range=_pair(weld_d, "t_range", "island.weld."),
                    depth=_num(weld_d, "depth", "island.weld."),
                )
            island = IslandSpec(
                angle_deg=_num(isl_d, "angle_deg", "island."),
                boundary_line=_line(isl_d, "boundary_line", "island."),
                end_dist=_num(isl_d, "end_dist", "island."),
                weld=weld,
            )

        outer_wall_line = (
            _line(d, "outer_wall_line", "") if d.get("outer_wall_line") is not None else None
        )

        sub_gates: list[SubGateSpec] = []
        sg_val = d.get("sub_gates")
        if sg_val is not None:
            if isinstance(sg_val, (str, bytes, dict)) or not isinstance(sg_val, (list, tuple)):
                raise ValueError(
                    f"gate profile JSON: 'sub_gates' must be a list of objects, got {sg_val!r}"
                )
            for i, sg_d in enumerate(sg_val):
                p = f"sub_gates[{i}]"
                if not isinstance(sg_d, dict):
                    raise ValueError(
                        f"gate profile JSON: '{p}' must be an object, got {type(sg_d).__name__}"
                    )
                _check_unknown(
                    sg_d,
                    {"inner_wall_line", "outer_wall_line", "tip_t", "island", "edge_channels"},
                    p,
                )
                sub_island: SubIslandSpec | None = None
                si_d = _section(sg_d, "island", required=False)
                if si_d is not None:
                    _check_unknown(
                        si_d, {"angle_deg", "inner_line", "outer_line", "end_dist"}, f"{p}.island"
                    )
                    sub_island = SubIslandSpec(
                        angle_deg=_num(si_d, "angle_deg", f"{p}.island."),
                        inner_line=_line(si_d, "inner_line", f"{p}.island."),
                        outer_line=_line(si_d, "outer_line", f"{p}.island."),
                        end_dist=_num(si_d, "end_dist", f"{p}.island."),
                    )
                sub_gates.append(
                    SubGateSpec(
                        inner_wall_line=_line(sg_d, "inner_wall_line", f"{p}."),
                        outer_wall_line=_line(sg_d, "outer_wall_line", f"{p}."),
                        tip_t=_num(sg_d, "tip_t", f"{p}."),
                        island=sub_island,
                        edge_channels=_edge_channels(sg_d, f"{p}."),
                    )
                )

        runner: RunnerSpec | None = None
        rn_d = _section(d, "runner", required=False)
        if rn_d is not None:
            _check_unknown(rn_d, {"width", "depth", "path"}, "runner")
            path_val = _req(rn_d, "path", "runner.")
            if isinstance(path_val, (str, bytes)) or not isinstance(path_val, (list, tuple)):
                raise ValueError(
                    f"gate profile JSON: 'runner.path' must be [[t, w], ...], got {path_val!r}"
                )
            path = []
            for pt in path_val:
                a, b = _elements(pt, 2, "runner.path", "[[t, w], ...]")
                path.append((_scalar(a, "runner.path"), _scalar(b, "runner.path")))
            runner = RunnerSpec(
                width=_num(rn_d, "width", "runner."),
                depth=_num(rn_d, "depth", "runner."),
                path=tuple(path),
            )

        well: WellSpec | None = None
        well_d = _section(d, "well", required=False)
        if well_d is not None:
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
                floor_t_range=(
                    _pair(well_d, "floor_t_range", "well.")
                    if well_d.get("floor_t_range") is not None
                    else None
                ),
                wall_angle_deg=(
                    _num(well_d, "wall_angle_deg", "well.") if "wall_angle_deg" in well_d else 60.0
                ),
            )

        valve_d = _section(d, "valve", required=True)
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
            sub_gates=tuple(sub_gates),
            runner=runner,
            edge_channels=_edge_channels(d, ""),
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
            "valve": {
                "t": self.valve.t,
                "w": self.valve.w,
                "orifice_diameter": self.valve.orifice_diameter,
            },
        }
        if self.outer_wall_line is not None:
            d["outer_wall_line"] = [list(p) for p in self.outer_wall_line]
        if self.sub_gates:
            d["sub_gates"] = []
            for sg in self.sub_gates:
                sg_d: dict[str, Any] = {
                    "inner_wall_line": [list(p) for p in sg.inner_wall_line],
                    "outer_wall_line": [list(p) for p in sg.outer_wall_line],
                    "tip_t": sg.tip_t,
                }
                if sg.island is not None:
                    sg_d["island"] = {
                        "angle_deg": sg.island.angle_deg,
                        "inner_line": [list(p) for p in sg.island.inner_line],
                        "outer_line": [list(p) for p in sg.island.outer_line],
                        "end_dist": sg.island.end_dist,
                    }
                if sg.edge_channels:
                    sg_d["edge_channels"] = [_edge_channel_dict(ec) for ec in sg.edge_channels]
                d["sub_gates"].append(sg_d)
        if self.runner is not None:
            d["runner"] = {
                "width": self.runner.width,
                "depth": self.runner.depth,
                "path": [list(p) for p in self.runner.path],
            }
        if self.edge_channels:
            d["edge_channels"] = [_edge_channel_dict(ec) for ec in self.edge_channels]
        if self.island is not None:
            d["island"] = {
                "angle_deg": self.island.angle_deg,
                "boundary_line": [list(p) for p in self.island.boundary_line],
                "end_dist": self.island.end_dist,
            }
            if self.island.weld is not None:
                d["island"]["weld"] = {
                    "t_range": list(self.island.weld.t_range),
                    "depth": self.island.weld.depth,
                }
        if self.well is not None:
            d["well"] = {
                "shape": self.well.shape,
                "t_range": list(self.well.t_range),
                "half_width": self.well.half_width,
                "depth": self.well.depth,
                "wall_angle_deg": self.well.wall_angle_deg,
            }
            if self.well.floor_t_range is not None:
                d["well"]["floor_t_range"] = list(self.well.floor_t_range)
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
            self.ramp_cap_t(),
            self.valve.t + self.valve.orifice_diameter / 2.0,
        ]
        if self.outer_wall_line is not None:
            candidates.append(self.outer_wall_line[1][0])
        for sg in self.sub_gates:
            candidates.append(sg.tip_t)
            if sg.island is not None:
                candidates.append(sg.island.end_dist)
        if self.runner is not None:
            candidates.append(max(t for t, _w in self.runner.path) + self.runner.width / 2.0)
        if self.well is not None:
            candidates.append(self.well.t_range[1])
        if self.island is not None:
            candidates.append(self.island.end_dist)
        return max(candidates)

    def w_max(self) -> float:
        """Largest width coordinate any pocket feature reaches (grid-fit bound).

        Boundary lines are **evaluated the way the builder evaluates them**
        (``_line_eval``: clamped to the first point's w before it, linearly
        extrapolated after the second), not read off their two stored
        endpoints. A wall whose line ends before the feature does and slopes
        outward keeps widening past its last point; taking the endpoint would
        under-report the reach, the grid-fit check would pass, and the
        rasterized pocket would be silently truncated at the array edge --
        wrong area, volume and conductance with no diagnostic (Codex P1).

        One deliberate inexactness: for the single-pocket ``outer_wall_line``
        the builder clamps the ``t < t1`` stretch to ``full_half_width`` (the
        gate exit), not to the line's own first w, so a line starting
        *narrower* than the exit makes this under-report that stretch. The
        caller keeps ``full_half_width`` as a separate term
        (``max(full_half_width, w_max())``), so the reach is still covered and
        the error can only go the safe way. Reproducing the exit-width clamp
        here would need the plate's width convention and buy nothing.
        """
        candidates = [0.0]
        if self.outer_wall_line is not None:
            candidates.append(_line_reach(self.outer_wall_line, self.t_max()))
        for sg in self.sub_gates:
            candidates.append(_line_reach(sg.outer_wall_line, sg.tip_t))
        if self.runner is not None:
            candidates.append(max(w for _t, w in self.runner.path) + self.runner.width / 2.0)
        return max(candidates)

    def w_min(self) -> float:
        """Most negative width coordinate any pocket feature reaches (≤ 0).

        Only meaningful for ``symmetric=False``, where ``w`` is a signed
        offset from the valve-side edge rather than ``|x − x_valve|``: a
        runner passing near ``w = 0`` sticks out to ``min(path.w) − width/2``
        on the far side of that edge. Walls are validated ``w ≥ 0``, so the
        runner is the only feature that can go negative (Codex P1).
        """
        if self.runner is None:
            return 0.0
        return min(0.0, min(w for _t, w in self.runner.path) - self.runner.width / 2.0)

    # ---- validation ----

    def validate(self) -> None:
        # Non-finite values first: every range check below is a comparison,
        # and every comparison against NaN is False — so a NaN slips through
        # *both* sides of a bounds test. Downstream it either poisons the
        # depth field (NaN volume, NaN solve) or, in a mask test, silently
        # drops the feature it was meant to describe (worse: the geometry is
        # wrong with no diagnostic). Walking the dataclass instead of listing
        # fields keeps this closed when new fields are added, and covers
        # direct construction as well as the JSON path.
        for label, val in _iter_numbers(self, ""):
            if not math.isfinite(val):
                raise ValueError(f"{label} must be a finite number, got {val!r}")
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
        if (self.outer_wall_line is None) == (not self.sub_gates):
            raise ValueError(
                "exactly one of outer_wall_line (single pocket) or sub_gates (fans) "
                "must describe the pocket silhouette"
            )
        if self.outer_wall_line is not None:
            (wt1, ww1), (wt2, ww2) = self.outer_wall_line
            if wt2 <= wt1 + _EPS:
                raise ValueError(f"outer_wall_line t must be increasing, got {wt1} → {wt2}")
            if ww1 <= 0 or ww2 <= 0:
                raise ValueError(f"outer_wall_line w must be positive, got {ww1}, {ww2}")
        if self.valve.t < 0:
            raise ValueError(f"valve.t must be ≥ 0, got {self.valve.t}")

        if self.sub_gates and self.island is not None:
            raise ValueError(
                "a top-level island needs the single-pocket form; with sub_gates each fan "
                "carries its own island"
            )
        if self.edge_channels and self.outer_wall_line is None:
            raise ValueError(
                "top-level edge_channels need the single-pocket form (outer_wall_line); "
                "with sub_gates each fan carries its own edge_channels"
            )
        _validate_edge_channels(
            self.edge_channels,
            "edge_channels",
            allowed_sides=("outer",),
            t_end=self.t_max(),
            t_end_label="t_max()",
        )
        for i, sg in enumerate(self.sub_gates):
            p = f"sub_gates[{i}]"
            if sg.tip_t <= self.land.length + _EPS:
                raise ValueError(
                    f"{p}.tip_t ({sg.tip_t}) must be > land.length ({self.land.length})"
                )
            for label, line in (
                ("inner_wall_line", sg.inner_wall_line),
                ("outer_wall_line", sg.outer_wall_line),
            ):
                (t1, w1), (t2, w2) = line
                if t2 <= t1 + _EPS:
                    raise ValueError(f"{p}.{label} t must be increasing, got {t1} → {t2}")
                if t1 < -_EPS:
                    raise ValueError(f"{p}.{label} t must be ≥ 0, got {t1}")
                if w1 < 0 or w2 < 0:
                    raise ValueError(f"{p}.{label} w must be ≥ 0, got {w1}, {w2}")
            # The fan must have positive width over its whole t-extent, not
            # just at the tip: lines crossed near the land that separate by
            # the tip pass a tip-only check, and the raster then silently
            # drops the crossed part of the fan (with another fan or a runner
            # feeding it, the solve completes and reports that unintended
            # geometry). Both edges are piecewise linear in t with a
            # breakpoint where each line's clamp ends, so the gap's minimum
            # over [0, tip_t] is attained at a breakpoint or an endpoint --
            # checking those is exact, not a sample (Codex P2).
            for t_chk in _fan_breakpoints(sg):
                w_in = _edge_w(sg.inner_wall_line, t_chk)
                w_out = _edge_w(sg.outer_wall_line, t_chk)
                if w_out <= w_in + _EPS:
                    raise ValueError(
                        f"{p}: outer wall ({w_out:.3f}) must be wider than the inner wall "
                        f"({w_in:.3f}) at t = {t_chk:g} (checked over [0, tip_t])"
                    )
            if sg.island is not None:
                si = sg.island
                if si.angle_deg < 0:
                    raise ValueError(f"{p}.island.angle_deg must be ≥ 0, got {si.angle_deg}")
                if si.angle_deg > self.main_ramp.angle_deg + _EPS:
                    raise ValueError(
                        f"{p}.island.angle_deg ({si.angle_deg}) must be ≤ main_ramp.angle_deg "
                        f"({self.main_ramp.angle_deg}); the island is the shallow side"
                    )
                if si.end_dist <= self.land.length + _EPS:
                    raise ValueError(
                        f"{p}.island.end_dist ({si.end_dist}) must be > land.length "
                        f"({self.land.length})"
                    )
                for label, line in (("inner_line", si.inner_line), ("outer_line", si.outer_line)):
                    (t1, _w1), (t2, _w2) = line
                    if t2 <= t1 + _EPS:
                        raise ValueError(
                            f"{p}.island.{label} t must be increasing, got {t1} → {t2}"
                        )
                for t_chk in (self.land.length, si.end_dist):
                    if _line_w(si.outer_line, t_chk) <= _line_w(si.inner_line, t_chk) + _EPS:
                        raise ValueError(
                            f"{p}.island: outer_line must stay outside inner_line over "
                            f"[land.length, end_dist]; they cross by t = {t_chk}"
                        )
            _validate_edge_channels(
                sg.edge_channels,
                f"{p}.edge_channels",
                allowed_sides=("outer", "inner"),
                t_end=sg.tip_t,
                t_end_label="tip_t",
            )

        if self.runner is not None:
            rn = self.runner
            if rn.width <= 0 or rn.depth <= 0:
                raise ValueError(
                    f"runner.width and runner.depth must be positive, got {rn.width}, {rn.depth}"
                )
            if len(rn.path) < 2:
                raise ValueError(f"runner.path needs at least 2 points, got {len(rn.path)}")
            for t, w in rn.path:
                if t < -_EPS or w < -_EPS:
                    raise ValueError(f"runner.path points must have t ≥ 0 and w ≥ 0, got {(t, w)}")
            for (t1, w1), (t2, w2) in zip(rn.path[:-1], rn.path[1:], strict=True):
                if math.hypot(t2 - t1, w2 - w1) <= _EPS:
                    raise ValueError(f"runner.path has a zero-length segment at {(t1, w1)}")

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
            if isl.weld is not None:
                wl, wh = isl.weld.t_range
                if wh <= wl + _EPS:
                    raise ValueError(
                        f"island.weld.t_range must be increasing, got {isl.weld.t_range}"
                    )
                if wl < self.land.length - _EPS or wh > isl.end_dist + _EPS:
                    raise ValueError(
                        f"island.weld.t_range ({isl.weld.t_range}) must lie within "
                        f"[land.length ({self.land.length}), island.end_dist ({isl.end_dist})]"
                    )
                if isl.weld.depth < 0:
                    raise ValueError(
                        f"island.weld.depth must be ≥ 0, got {isl.weld.depth} "
                        "(0 = the steel touches the PL: no flow path, a hole in the part)"
                    )
                if isl.weld.depth > self.land.depth + _EPS:
                    raise ValueError(
                        f"island.weld.depth ({isl.weld.depth}) must be ≤ land.depth "
                        f"({self.land.depth}); weld metal fills the pocket, it cannot deepen it"
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
            # The sloped wall climbs from the rim, so the rasterised depth
            # saturates at half_width·tan(wall_angle) on the centreline. A
            # deeper request would pass every other check, be recorded as
            # asked, and be built shallower with no diagnostic.
            if w.wall_angle_deg < 90.0:
                reach = w.half_width * math.tan(math.radians(w.wall_angle_deg))
                if w.depth > reach + _EPS:
                    raise ValueError(
                        f"well.depth ({w.depth}) exceeds what the {w.wall_angle_deg}° wall "
                        f"can reach at half_width {w.half_width} "
                        f"(max {reach:.3f}); widen the well or steepen the wall"
                    )
            if w.floor_t_range is not None:
                # reference metadata only (see WellSpec docstring) — still
                # reject nonsensical ranges to catch extraction typos
                if (
                    w.floor_t_range[0] < w.t_range[0] - _EPS
                    or w.floor_t_range[1] > w.t_range[1] + _EPS
                    or w.floor_t_range[0] >= w.floor_t_range[1] - _EPS
                ):
                    raise ValueError(
                        f"well.floor_t_range ({w.floor_t_range}) must be an increasing range "
                        f"inside well.t_range ({w.t_range})"
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


def _validate_edge_channels(
    channels: tuple[EdgeChannelSpec, ...],
    path: str,
    *,
    allowed_sides: tuple[str, ...],
    t_end: float,
    t_end_label: str,
) -> None:
    """Shared checks for a list of edge channels (single pocket or one fan)."""
    for i, ec in enumerate(channels):
        p = f"{path}[{i}]"
        if ec.side not in allowed_sides:
            raise ValueError(
                f"{p}.side must be one of {list(allowed_sides)}, got {ec.side!r}"
                + (
                    " (the single-pocket form has no inner wall)"
                    if "inner" not in allowed_sides
                    else ""
                )
            )
        if ec.width <= 0 or ec.depth <= 0:
            raise ValueError(
                f"{p}.width and {p}.depth must be positive, got {ec.width}, {ec.depth}"
            )
        if ec.t_range is not None:
            lo, hi = ec.t_range
            if hi <= lo + _EPS:
                raise ValueError(f"{p}.t_range must be increasing, got {ec.t_range}")
            if lo < -_EPS or hi > t_end + _EPS:
                raise ValueError(
                    f"{p}.t_range ({ec.t_range}) must lie within [0, {t_end_label} ({t_end:g})]"
                )


def _line_w(line: Line, t: float) -> float:
    """Scalar evaluation of a (t, w) line at ``t`` (extrapolated, no clamp)."""
    (t1, w1), (t2, w2) = line
    return w1 + (w2 - w1) / max(t2 - t1, 1e-12) * (t - t1)


def _edge_w(line: Line, t: float) -> float:
    """Scalar twin of :func:`_line_eval`: clamped before the first point."""
    (t1, w1), _ = line
    return w1 if t < t1 else _line_w(line, t)


def _line_reach(line: Line, t_end: float) -> float:
    """Widest w a boundary line reaches over ``t ∈ [0, t_end]``.

    ``_edge_w`` is constant then linear, so its maximum over the interval
    sits at one of the two ends.
    """
    return max(_edge_w(line, 0.0), _edge_w(line, t_end))


def _count_components(mask: np.ndarray) -> int:
    """Number of 4-connected components in a boolean mask.

    4-connectivity, not 8: it is the connectivity the 5-point solver stencil
    uses, so a band that only touches itself at the corners is two channels
    as far as the flow is concerned (see ``tests/test_solver_1d.py`` on the
    same distinction for gate reachability).
    """
    from scipy import ndimage as ndi

    return int(ndi.label(mask)[1])


def _fan_breakpoints(sg: SubGateSpec) -> list[float]:
    """Where the fan's width can turn: both clamp points, plus the ends."""
    pts = {0.0, float(sg.tip_t), sg.inner_wall_line[0][0], sg.outer_wall_line[0][0]}
    return sorted(p for p in pts if -_EPS <= p <= sg.tip_t + _EPS)


def _edge_wall_polyline(
    line: Line, t_lo: float, t_hi: float, before_value: float
) -> tuple[tuple[float, float], ...]:
    """The effective wall over ``t ∈ [t_lo, t_hi]`` as a (t, w) polyline.

    Mirrors ``_line_eval``: constant ``before_value`` before the line's
    first point, linear (extrapolated) after. When ``before_value`` differs
    from the line's first w the clamp is a step — both corner points are
    kept so the polyline carries the vertical jump the raster sees.
    Consecutive duplicates (the common ``before_value == w1`` case) are
    dropped.
    """
    (t1, w1), _ = line

    def w_at(tv: float) -> float:
        return before_value if tv < t1 else _line_w(line, tv)

    pts = [(t_lo, w_at(t_lo))]
    if t_lo < t1 < t_hi:
        pts.append((t1, before_value))
        pts.append((t1, w1))
    pts.append((t_hi, w_at(t_hi)))
    out = [pts[0]]
    for p in pts[1:]:
        if math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) > _EPS:
            out.append(p)
    return tuple(out)


def _apply_edge_channels(
    channels: tuple[EdgeChannelSpec, ...],
    *,
    walls: dict[str, tuple[Line, float]],  # side -> (line, before_value)
    t_end: float,
    in_pocket: np.ndarray,
    d: np.ndarray,
    t: np.ndarray,
    wa: np.ndarray,
    cell_size: float,
    label: str,
) -> np.ndarray:
    """Deepen ``d`` along the requested walls; returns the new depth field.

    The band is the set of pocket cells within ``width`` (perpendicular
    distance) of the effective wall polyline, clipped to ``t_range``. It is
    a floor (``d = max(d, depth)``) restricted to ``in_pocket``, so the
    silhouette never changes. A band that selects zero cells is rejected:
    the spec would be recorded as asked while the built geometry silently
    lacks the feature (same false-green class as the runner thinner than
    the mesh).
    """
    for i, ec in enumerate(channels):
        line, before_value = walls[ec.side]
        t_lo, t_hi = ec.t_range if ec.t_range is not None else (0.0, t_end)
        t_hi = min(t_hi, t_end)
        poly = _edge_wall_polyline(line, t_lo, t_hi, before_value)
        band = in_pocket & (_polyline_distance(poly, t, wa) <= ec.width + 1e-9)
        if not band.any():
            raise ValueError(
                f"{label}[{i}] (width {ec.width} mm, t_range [{t_lo:g}, {t_hi:g}]) "
                f"rasterises to zero cells at cell_size_mm={cell_size}: the band misses "
                "every pocket cell centre. Widen the band, extend t_range, or refine "
                "the mesh."
            )
        d = np.where(band, np.maximum(d, ec.depth), d)
    return d


def _polyline_distance(
    path: tuple[tuple[float, float], ...], t: np.ndarray, w: np.ndarray
) -> np.ndarray:
    """Distance from each (t, w) to the nearest point of a polyline."""
    best = np.full(t.shape, np.inf)
    for (pt, pw), (qt, qw) in zip(path[:-1], path[1:], strict=True):
        vt, vw = qt - pt, qw - pw
        s = np.clip(((t - pt) * vt + (w - pw) * vw) / (vt * vt + vw * vw), 0.0, 1.0)
        best = np.minimum(best, np.hypot(t - (pt + s * vt), w - (pw + s * vw)))
    return best


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

    # --- grid-fit check: the pocket must not overhang the raster grid ---
    # (otherwise it would silently truncate at the array edge and produce
    # wrong volume / conductance with no diagnostic)
    w_wall_max = max(full_half_width, spec.w_max())
    well_hw = spec.well.half_width if spec.well is not None else 0.0
    if spec.symmetric:
        # w = |x − x_valve|, so the pocket is mirrored and the reach is the
        # same on both sides.
        x_lo = x_valve - max(w_wall_max, well_hw)
        x_hi = x_valve + max(w_wall_max, well_hw)
    else:
        # w is a signed offset from the valve-side edge: the well straddles
        # it, and a runner near w = 0 sticks out on the far side too
        # (``w_min``). Missing that lets the builder clip the runner at the
        # array boundary and solve a narrower channel than the spec asks for.
        x_lo = x_edge - max(well_hw, -spec.w_min())
        x_hi = x_edge + max(w_wall_max, well_hw)
    if x_lo < -_EPS or x_hi > total_w + _EPS:
        raise ValueError(
            f"gate pocket x-extent [{x_lo:.1f}, {x_hi:.1f}] mm overhangs the grid "
            f"[0, {total_w:.1f}] mm; widen plate_w_mm/pad_mm or shrink the spec "
            f"(outer_wall_line w, well.half_width, valve.w)"
        )

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
        if isl.weld is not None:
            wt_lo, wt_hi = isl.weld.t_range
            in_weld = in_island & (t >= wt_lo) & (t <= wt_hi)
            d_base = np.where(in_weld, isl.weld.depth, d_base)

    # --- pocket silhouette: one outer wall, or the union of the fans ---
    if spec.outer_wall_line is not None:
        w_wall = _line_eval(spec.outer_wall_line, t, before_value=full_half_width)
        in_gate_base = (t >= 0) & (t <= T) & (wa >= 0) & (wa <= w_wall)
        # A dam welded up to the PL leaves no flow path: those cells are
        # steel, not cavity (a zero-thickness cell in the mask would give
        # S = 0 and a singular system). The well is machined through it, so
        # cells the well still reaches stay cavity via ``in_well`` below.
        if spec.island is not None and spec.island.weld is not None and spec.island.weld.depth <= 0:
            in_gate_base &= ~in_weld
        d_base = _apply_edge_channels(
            spec.edge_channels,
            walls={"outer": (spec.outer_wall_line, full_half_width)},
            t_end=T,
            in_pocket=in_gate_base,
            d=d_base,
            t=t,
            wa=wa,
            cell_size=dx,
            label="edge_channels",
        )
    else:
        # Sub-gate fans: outside every fan is steel at the PL. Each fan
        # carries the land + main ramp, overridden by its own island band.
        in_gate_base = np.zeros(t.shape, dtype=bool)
        d_fans = np.zeros_like(d_base)
        for sg_i, sg in enumerate(spec.sub_gates):
            w_in = _line_eval(sg.inner_wall_line, t, before_value=sg.inner_wall_line[0][1])
            w_out = _line_eval(sg.outer_wall_line, t, before_value=sg.outer_wall_line[0][1])
            in_fan = (t >= 0) & (t <= sg.tip_t) & (wa >= w_in) & (wa <= w_out)
            d_fan = d_base
            if sg.island is not None:
                si = sg.island
                tan_si = math.tan(math.radians(si.angle_deg))
                w_si_in = _line_eval(si.inner_line, t, before_value=si.inner_line[0][1])
                w_si_out = _line_eval(si.outer_line, t, before_value=si.outer_line[0][1])
                in_si = (
                    in_fan
                    & (t > land_len)
                    & (t <= si.end_dist)
                    & (wa >= w_si_in)
                    & (wa <= w_si_out)
                )
                d_fan = np.where(in_si, land_depth + tan_si * (t - land_len), d_base)
            d_fan = _apply_edge_channels(
                sg.edge_channels,
                walls={
                    "outer": (sg.outer_wall_line, sg.outer_wall_line[0][1]),
                    "inner": (sg.inner_wall_line, sg.inner_wall_line[0][1]),
                },
                t_end=sg.tip_t,
                in_pocket=in_fan,
                d=d_fan,
                t=t,
                wa=wa,
                cell_size=dx,
                label=f"sub_gates[{sg_i}].edge_channels",
            )
            # Overlapping fans: the deeper (more open) one wins, as a machined
            # union would.
            d_fans = np.where(in_fan, np.maximum(d_fans, d_fan), d_fans)
            in_gate_base |= in_fan
        d_base = d_fans

    # --- runner band: d = max(d, depth) within width/2 of the path ---
    if spec.runner is not None:
        rn = spec.runner
        in_runner = (
            (t >= 0) & (t <= T) & (_polyline_distance(rn.path, t, wa) <= rn.width / 2.0 + 1e-9)
        )
        # A band thinner than the mesh can pass between cell centres and
        # rasterise into a dotted line of islands instead of a channel. The
        # continuous spec is fine, so nothing downstream looks wrong -- but
        # the fans it feeds are cut off from the valve and the solver reports
        # a disconnected cavity (or, with a coarser look, plausible garbage).
        # Same failure class as an exit width below the mesh spacing, so it
        # gets the same treatment: reject at build time, naming both knobs.
        # The raster is measured, not predicted from a width/spacing rule:
        # how thin is too thin depends on the path's angle to the grid.
        # Count on one half of a symmetric field: it is built in (t, |w|), so
        # a band that never reaches the axis is legitimately two mirror images
        # in x. Folding is not the breakage this guard is looking for, and the
        # half-plane holds each wa exactly once. (``path`` is validated w ≥ 0,
        # so the band is connected in that half whenever it rasterises.)
        band_half = in_runner[:, xx[0, :] >= x_valve] if spec.symmetric else in_runner
        n_band = _count_components(band_half)
        if n_band != 1:
            raise ValueError(
                f"runner.width ({rn.width} mm) rasterises into {n_band} disconnected piece(s) "
                f"at cell_size_mm={dx}: the band passes between cell centres instead of "
                "forming a channel. Widen the runner or refine the mesh."
            )
        d_base = np.where(
            in_runner, np.maximum(np.where(in_gate_base, d_base, 0.0), rn.depth), d_base
        )
        in_gate_base |= in_runner

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
        if not mask[iy_plate_bottom, :].any():
            # The wall just severed the plate from the gate block. Without
            # this, the solver would fill the orphaned plate with garbage that
            # looks like a uniform fill time (Issue #58); its own reachability
            # check now rejects that too, but this message names the knob.
            raise ValueError(
                f"gate_exit_width ({spec.gate_exit_width} mm) rasterises to zero "
                f"open columns at cell_size_mm={dx}: the gate land wall closes the "
                "entire plate-bottom row and severs the plate from the gate. "
                "Widen the exit or refine the mesh."
            )

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
        valve_axis_x_mm=x_valve,
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
