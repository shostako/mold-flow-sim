"""IGES of the gate block's resin, built from the spec the solver rasterises (v0.59.0).

The solid is ``{(t, w, z): (t, w) in the pocket, −d(t, w) ≤ z ≤ 0}`` for the
depth field ``d`` of :func:`core.profile_gate.build_profile_gate_geometry`,
assembled from the same definitions instead of fitted to a raster. Every
step of the builder is a set operation on columns:

    d = max(d1, d2)        →  column(d1) ∪ column(d2)              (floors: runner, well, edge channels …)
    d = min(d, c)          →  column(d) ∩ {z ≥ −c}                 (the ramp's cap, a plateau)
    d = d' on a zone Z     →  (column(d) \\ Z) ∪ (column(d') ∩ Z)   (island, weld, graded ends)
    a cell leaves the mask →  column(d) \\ Z                        (closure, round corner, dam at the PL)

and each surface is exact: planes; the land end ``t = L(w)`` as a B-spline
(a power of 1, 2 or 3 is a polynomial and is reproduced exactly, other
powers are fitted to :data:`CURVE_TOL_MM`); the ramp behind it as that curve
swept along the ramp; the graded ends as the ruled patch between the
land-end curve and the cap line; the well as cones joined by a trapezoidal
prism; the round corner as a cylinder. A symmetric block is built for
``w ≥ 0`` and mirrored about the valve axis; runner and well, which the
builder measures in the plane, are added after the mirror.

CAD frame (that of the received models ``Runner-block_3D_*.igs``):
``x`` = width from the valve axis, ``y = CAD_Y_AT_EXIT − t``, ``z = −depth``
with the PL at ``z = 0``. The file carries the faces only (trimmed
surfaces, IGES type 144), as those models do. The valve orifice and the
product plate are not part of the model.

OCP (``cadquery-ocp-novtk``, the ``cad`` extra) is optional: :func:`available`
says whether this module can run. :func:`raster_depth` reads a solid back
from above (triangulated, rasterised at cell centres) so the caller can
compare it with the solver's own field (:func:`check_against_field`).
"""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from core.profile_gate import GateProfileSpec, Line, _edge_wall_polyline

#: ``y = CAD_Y_AT_EXIT − t``: the received CAD models put the gate exit at y = 20.
CAD_Y_AT_EXIT = 20.0
#: Fit tolerance of a land-end curve that is not a polynomial (power ∉ {1, 2, 3}).
CURVE_TOL_MM = 1e-4
#: Boolean fuzzy tolerance.
FUZZY_MM = 1e-5
#: Linear deflection of the triangulation :func:`raster_depth` reads.
MESH_DEFLECTION_MM = 2e-4
#: Angular deflection of that triangulation on cones and cylinders.
MESH_ANGLE_RAD = 0.01

_Z_TOP = 1.0  # columns start above the PL; the pocket clips them to z ≤ 0


def available() -> bool:
    """True when OCP can be imported (``pip install -e ".[cad]"``)."""
    try:
        import OCP.BRepAlgoAPI  # noqa: F401
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# OCP, imported on first use (the module imports without it)
# ---------------------------------------------------------------------------

_NAMES = {
    "OCP.BRep": ["BRep_Tool"],
    "OCP.BRepAdaptor": ["BRepAdaptor_Surface"],
    "OCP.BRepAlgoAPI": ["BRepAlgoAPI_Common", "BRepAlgoAPI_Cut", "BRepAlgoAPI_Fuse"],
    "OCP.BRepBuilderAPI": [
        "BRepBuilderAPI_MakeFace",
        "BRepBuilderAPI_MakePolygon",
        "BRepBuilderAPI_Transform",
    ],
    "OCP.BRepCheck": ["BRepCheck_Analyzer"],
    "OCP.BRepGProp": ["BRepGProp"],
    "OCP.BRepMesh": ["BRepMesh_IncrementalMesh"],
    "OCP.BRepPrimAPI": [
        "BRepPrimAPI_MakeBox",
        "BRepPrimAPI_MakeCone",
        "BRepPrimAPI_MakeCylinder",
        "BRepPrimAPI_MakePrism",
    ],
    "OCP.Geom": ["Geom_BSplineSurface"],
    "OCP.GeomAbs": ["GeomAbs_Cone", "GeomAbs_Cylinder"],
    "OCP.gp": ["gp_Ax2", "gp_Dir", "gp_Pnt", "gp_Trsf", "gp_Vec"],
    "OCP.GProp": ["GProp_GProps"],
    "OCP.IGESControl": ["IGESControl_Writer"],
    "OCP.Interface": ["Interface_Static"],
    "OCP.ShapeUpgrade": ["ShapeUpgrade_UnifySameDomain"],
    "OCP.TColgp": ["TColgp_Array2OfPnt"],
    "OCP.TColStd": ["TColStd_Array1OfInteger", "TColStd_Array1OfReal"],
    "OCP.TopAbs": ["TopAbs_FACE", "TopAbs_SOLID"],
    "OCP.TopExp": ["TopExp_Explorer"],
    "OCP.TopLoc": ["TopLoc_Location"],
    "OCP.TopoDS": ["TopoDS"],
    "OCP.TopTools": ["TopTools_ListOfShape"],
}
_NS: dict = {}


def _k() -> dict:
    """The OCP names this module uses, imported on first call."""
    if not _NS:
        import importlib

        for mod, names in _NAMES.items():
            m = importlib.import_module(mod)
            _NS.update({n: getattr(m, n) for n in names})
    return _NS


def _p(t: float, w: float, z: float):
    return _k()["gp_Pnt"](float(w), CAD_Y_AT_EXIT - float(t), float(z))


def _op(kind: str, a, tools: Sequence):
    """Boolean ``a <kind> tools`` with ``kind`` in fuse / common / cut."""
    k = _k()
    tools = [s for s in tools if s is not None]
    if not tools:
        return a
    cls = {"fuse": "BRepAlgoAPI_Fuse", "common": "BRepAlgoAPI_Common", "cut": "BRepAlgoAPI_Cut"}
    args, tl = k["TopTools_ListOfShape"](), k["TopTools_ListOfShape"]()
    args.Append(a)
    for s in tools:
        tl.Append(s)
    op = k[cls[kind]]()
    op.SetArguments(args)
    op.SetTools(tl)
    op.SetFuzzyValue(FUZZY_MM)
    op.Build()
    if not op.IsDone():
        raise RuntimeError(f"OCP boolean {kind} failed")
    return op.Shape()


def _union(shapes: Sequence):
    shapes = [s for s in shapes if s is not None]
    return _op("fuse", shapes[0], shapes[1:]) if shapes else None


def _common(a, *tools):
    return _op("common", a, tools)


def _box(t0: float, t1: float, w0: float, w1: float, z0: float, z1: float):
    return _k()["BRepPrimAPI_MakeBox"](_p(t1, w0, z0), _p(t0, w1, z1)).Shape()


def _face(pts):
    k = _k()
    poly = k["BRepBuilderAPI_MakePolygon"]()
    last = None
    for q in pts:
        if last is None or q.Distance(last) > 1e-9:
            poly.Add(q)
            last = q
    poly.Close()
    return k["BRepBuilderAPI_MakeFace"](poly.Wire(), True).Face()


def _prism(shape, dx: float, dy: float, dz: float):
    k = _k()
    return k["BRepPrimAPI_MakePrism"](shape, k["gp_Vec"](dx, dy, dz)).Shape()


def _plan_prism(tw: Sequence[tuple[float, float]], z0: float, z1: float):
    """Polygon in (t, w) extruded from ``z1`` down to ``z0``."""
    return _prism(_face([_p(t, w, z1) for t, w in tw]), 0.0, 0.0, z0 - z1)


def _tz_prism(tz: Sequence[tuple[float, float]], w0: float, w1: float):
    """Polygon in (t, z) swept along w over [w0, w1]: a surface of t only."""
    return _prism(_face([_p(t, w0, z) for t, z in tz]), w1 - w0, 0.0, 0.0)


def _wz_prism(wz: Sequence[tuple[float, float]], t0: float, t1: float):
    """Polygon in (w, z) swept along t over [t0, t1]: a surface of w only."""
    return _prism(_face([_p(t0, w, z) for w, z in wz]), 0.0, -(t1 - t0), 0.0)


def _axis(t: float, w: float, z: float):
    k = _k()
    return k["gp_Ax2"](_p(t, w, z), k["gp_Dir"](0.0, 0.0, 1.0))


def _cylinder(t: float, w: float, r: float, z0: float, z1: float):
    return _k()["BRepPrimAPI_MakeCylinder"](_axis(t, w, z0), r, z1 - z0).Shape()


def _cone(t: float, w: float, r_bot: float, r_top: float, z_bot: float, z_top: float):
    return _k()["BRepPrimAPI_MakeCone"](_axis(t, w, z_bot), r_bot, r_top, z_top - z_bot).Shape()


def _transformed(shape, trsf):
    return _k()["BRepBuilderAPI_Transform"](shape, trsf, True).Shape()


# ---------------------------------------------------------------------------
# boundary lines and (t, w) regions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Edge:
    """A boundary line read like ``_line_eval``: ``before`` for t < t1, then the line."""

    line: Line
    before: float

    def __call__(self, t: float) -> float:
        (t1, w1), (t2, w2) = self.line
        if t < t1:
            return self.before
        return w1 + (w2 - w1) / max(t2 - t1, 1e-12) * (t - t1)

    def chain(self, a: float, b: float) -> list[tuple[float, float]]:
        """(t, w) vertices over [a, b], the step at t1 included when ``before ≠ w1``."""
        (t1, w1), _ = self.line
        pts = [(a, self(a))]
        if a < t1 < b:
            pts.append((t1, self.before))
            pts.append((t1, w1))
        pts.append((b, self(b)))
        return pts


def _band(a: float, b: float, lo, hi) -> list[tuple[float, float]]:
    """Polygon ``{a ≤ t ≤ b, lo(t) ≤ w ≤ hi(t)}``; ``lo`` / ``hi`` are _Edge or float.

    The callers keep ``hi ≥ lo`` (validated fans, or a floor below every
    vertex of ``hi``), so the polygon is simple.
    """

    def chain(f):
        return f.chain(a, b) if isinstance(f, _Edge) else [(a, f), (b, f)]

    return chain(lo) + chain(hi)[::-1]


def _floor_below(*chains: list[tuple[float, float]]) -> float:
    return min([-1.0] + [w - 1.0 for c in chains for _t, w in c])


def _capsules(path: Sequence[tuple[float, float]], r: float, z0: float, z1: float):
    """Union of every point within ``r`` of the (t, w) polyline, as a prism."""
    parts = []
    for (pt, pw), (qt, qw) in zip(path[:-1], path[1:], strict=True):
        ln = math.hypot(qt - pt, qw - pw)
        if ln < 1e-12:
            continue
        nt, nw = -(qw - pw) / ln * r, (qt - pt) / ln * r
        quad = [(pt + nt, pw + nw), (qt + nt, qw + nw), (qt - nt, qw - nw), (pt - nt, pw - nw)]
        parts.append(_plan_prism(quad, z0, z1))
    parts += [_cylinder(t, w, r, z0, z1) for t, w in path]
    return _union(parts)


# ---------------------------------------------------------------------------
# curved surfaces: B-splines over w
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Spline:
    """Clamped B-spline basis over ``v ∈ [0, 1]`` with the width a polynomial ``w(v)``.

    ``fit(f)`` gives the poles of ``f(w(v))``. Least squares in the spline's
    own space is exact for every polynomial of ``v`` up to the degree, so
    ``w(v)`` itself and anything linear in w (the ramp's run, the cap depth)
    come out exact, and their sum with a fitted land end stays a ruled
    surface pole by pole.
    """

    knots: np.ndarray
    deg: int
    w_of: Callable[[np.ndarray], np.ndarray]
    _v: np.ndarray
    _basis: np.ndarray

    def fit(self, f: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        poles, *_ = np.linalg.lstsq(self._basis, f(self.w_of(self._v)), rcond=None)
        return poles

    def eval(self, poles: np.ndarray) -> np.ndarray:
        return self._basis @ poles


_MAX_DEGREE = 25  # OCCT's B-spline limit


def _w_spline(land_end: Callable, w0: float, w1: float, *, m: int, deg: int):
    """The spline a ramp patch over [w0, w1] is built on, and its fit error.

    ``m > 1`` (a non-integer land power on an interval ending at the exit's
    edge, where ``(1 − w/w_edge)^p`` is not smooth -- an infinite slope for
    ``p < 1``, an infinite curvature for ``1 < p < 2``) uses
    ``w = w1 − (w1 − w0)(1 − v)^m``: the land end becomes ``∝ (1 − v)^{mp}``
    with ``mp ≥ 3``, a polynomial when ``mp`` is an integer and a curve a few
    spans follow otherwise. The error is measured normal to the curve.
    """
    from scipy.interpolate import BSpline

    def w_of(v):
        return w0 + (w1 - w0) * v if m == 1 else w1 - (w1 - w0) * (1.0 - v) ** m

    err = math.inf
    for n_inner in (0, 4, 8, 16, 32, 64, 128):
        inner = np.linspace(0.0, 1.0, n_inner + 2)[1:-1]
        knots = np.concatenate([np.zeros(deg + 1), inner, np.ones(deg + 1)])
        v = np.unique(np.concatenate([np.linspace(0.0, 1.0, 1601), inner]))
        sp = _Spline(knots, deg, w_of, v, BSpline.design_matrix(v, knots, deg).toarray())
        t_true, w = land_end(w_of(v)), w_of(v)
        t_fit = sp.eval(sp.fit(land_end))
        # |Δt| projected on the normal of the curve (t(v), w(v)); w is reproduced exactly.
        dt, dw = np.gradient(t_true, v), np.gradient(w, v)
        norm = np.hypot(dt, dw)
        err = float(np.max(np.abs(t_fit - t_true) * np.abs(dw) / np.where(norm > 0, norm, 1.0)))
        if err <= CURVE_TOL_MM:
            return sp, err
    raise ValueError(f"land-end curve not fitted to {CURVE_TOL_MM} mm (best {err:.2e} mm)")


def _ruled_face(sp: _Spline, w_poles: np.ndarray, row0, row1):
    """Face linear between two pole rows ``(t, z)`` sharing the spline's knots and w poles."""
    k = _k()
    n = len(w_poles)
    poles = k["TColgp_Array2OfPnt"](1, 2, 1, n)
    for i, (tp, zp) in enumerate((row0, row1), start=1):
        for j in range(n):
            poles.SetValue(i, j + 1, _p(tp[j], w_poles[j], zp[j]))
    uk, um = k["TColStd_Array1OfReal"](1, 2), k["TColStd_Array1OfInteger"](1, 2)
    for i, v in ((1, 0.0), (2, 1.0)):
        uk.SetValue(i, v)
        um.SetValue(i, 2)
    distinct, mult = np.unique(np.round(sp.knots, 12), return_counts=True)
    vk = k["TColStd_Array1OfReal"](1, len(distinct))
    vm = k["TColStd_Array1OfInteger"](1, len(distinct))
    for i, (kv, mu) in enumerate(zip(distinct, mult, strict=True), start=1):
        vk.SetValue(i, float(kv))
        vm.SetValue(i, int(mu))
    surf = k["Geom_BSplineSurface"](poles, uk, vk, um, vm, 1, sp.deg)
    return k["BRepBuilderAPI_MakeFace"](surf, 1e-7).Face()


def _up(face, z_low: float):
    """Everything above ``face`` up to past ``_Z_TOP``: the face's column."""
    return _prism(face, 0.0, 0.0, _Z_TOP + 1.0 - z_low)


# ---------------------------------------------------------------------------
# the block
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Frame:
    T: float  # pocket end, spec.t_max()
    w_edge: float  # exit half-width (symmetric) / width (one-sided)
    t_lo: float
    t_hi: float
    w_hi: float  # half-block columns run over w ∈ [0, w_hi]
    z_bot: float


def _frame(spec: GateProfileSpec) -> _Frame:
    w_edge = spec.gate_exit_width / 2.0 if spec.symmetric else spec.gate_exit_width
    T = spec.t_max()
    deepest = [spec.main_ramp.cap_depth]
    if spec.ramp_ends is not None and spec.ramp_ends.depth_end is not None:
        deepest.append(spec.ramp_ends.depth_end)
    for x in (spec.well, spec.runner, spec.ramp_cut, spec.land_ends):
        if x is not None:
            deepest.append(x.depth)
    deepest += [ec.depth for ec in spec.edge_channels]
    deepest += [ec.depth for sg in spec.sub_gates for ec in sg.edge_channels]
    if spec.island is not None:
        isl = spec.island
        deepest.append(
            spec.land.depth
            + math.tan(math.radians(isl.angle_deg)) * (isl.end_dist - spec.land.length)
        )
    return _Frame(
        T=T,
        w_edge=w_edge,
        t_lo=-2.0,
        t_hi=T + 2.0,
        w_hi=max(w_edge, spec.w_max()) + 2.0,
        z_bot=-(max(deepest) + 2.0),
    )


def _ramp_start(spec: GateProfileSpec) -> float:
    """How far before the land end the ramp face starts (inside the land: no tangency)."""
    tan = math.tan(math.radians(spec.main_ramp.angle_deg))
    return min(0.5, 0.5 * spec.land.depth / max(tan, 1e-12))


def _ramp_patch(spec: GateProfileSpec, fr: _Frame, w0: float, w1: float):
    """``max(land, min(ramp, cap))`` over w ∈ [w0, w1], all t.

    The ramp is the main one, or the graded one (``ramp_ends``) when the
    interval lies in its zone; with a land profile both start at ``L(w)``.
    Callers split the w axis at ``w_from`` and ``w_edge`` so each interval
    has one formula.
    """
    L0, dL = spec.land.length, spec.land.depth
    cap = spec.main_ramp.cap_depth
    tan = math.tan(math.radians(spec.main_ramp.angle_deg))
    re_ = spec.ramp_ends
    graded = re_ is not None and w0 >= re_.w_from - 1e-12
    profiled = spec.land.profile is not None and w1 <= fr.w_edge + 1e-12
    land = _box(fr.t_lo, fr.t_hi, w0, w1, -dL, _Z_TOP)

    if graded:
        span = max(fr.w_edge - re_.w_from, 1e-12)
        t_c0 = spec.ramp_cap_t()
        d_end = cap if re_.depth_end is None else re_.depth_end

        def frac(w):
            return (np.minimum(w, fr.w_edge) - re_.w_from) / span

        def v_t(w):  # run from the land end to the cap line
            return t_c0 + (re_.t_end - t_c0) * frac(w) - L0

        def d_cap(w):
            return cap + (d_end - cap) * frac(w)

        caps = _wz_prism(
            [(w0, _Z_TOP), (w1, _Z_TOP), (w1, -float(d_cap(w1))), (w0, -float(d_cap(w0)))],
            fr.t_lo,
            fr.t_hi,
        )
    else:

        def v_t(w):
            return np.full_like(np.asarray(w, dtype=float), (cap - dL) / max(tan, 1e-12))

        def d_cap(w):
            return np.full_like(np.asarray(w, dtype=float), cap)

        caps = _box(fr.t_lo, fr.t_hi, w0, w1, -cap, _Z_TOP)

    if not graded and not profiled:  # a plane of t: the plain ramp
        e = _ramp_start(spec)
        ts, te = L0 - e, fr.t_hi
        ramp = _tz_prism(
            [(ts, _Z_TOP), (te, _Z_TOP), (te, -(dL + tan * (te - L0))), (ts, -(dL - tan * e))],
            w0,
            w1,
        )
        return _union([land, _common(ramp, caps)])

    m, deg = 1, 3
    if profiled:
        lp = spec.land.profile
        dl, p = lp.center_length - L0, lp.power
        integer = abs(p - round(p)) < 1e-12
        if integer:  # a polynomial of w: one span of its degree
            deg = max(3, int(round(p)))
        elif p < 3.0 and abs(w1 - fr.w_edge) < 1e-9:
            # (1 − w/w_edge)^p is not C³ at the edge: reparametrise to (1 − v)^{mp}, mp ≥ 3.
            m = math.ceil(3.0 / p - 1e-9)
            mp = m * p
            deg = max(3, m, int(round(mp)) if abs(mp - round(mp)) < 1e-9 else 3)
        if deg > _MAX_DEGREE:
            raise ValueError(f"land.profile.power {p:g} needs a degree-{deg} curve; not modelled")

        def land_end(w):
            return L0 + dl * np.clip(1.0 - np.maximum(w, 0.0) / fr.w_edge, 0.0, 1.0) ** p
    else:

        def land_end(w):
            return np.full_like(np.asarray(w, dtype=float), L0)

    sp, _err = _w_spline(land_end, w0, w1, m=m, deg=deg)
    w_p, t_p = sp.fit(lambda w: w), sp.fit(land_end)
    vt_p = sp.fit(v_t)
    vz_p = sp.fit(lambda w: -(d_cap(w) - dL))  # z change from the land end to the cap line
    # s runs from just inside the land (s0 < 0) to past the block's end (s1).
    vt_s = sp.eval(vt_p)
    s0 = -_ramp_start(spec) / float(np.max(vt_s))
    s1 = float(np.max((fr.t_hi - sp.eval(t_p)) / vt_s)) + 0.5
    row0 = (t_p + s0 * vt_p, -dL + s0 * vz_p)
    row1 = (t_p + s1 * vt_p, -dL + s1 * vz_p)
    face = _ruled_face(sp, w_p, row0, row1)
    ramp = _up(face, float(min(row0[1].min(), row1[1].min())))
    return _union([land, _common(ramp, caps)])


def _base_column(spec: GateProfileSpec, fr: _Frame):
    """The land / ramp / cap field over the half block, before any overlay."""
    cuts = {0.0, fr.w_hi}
    if spec.ramp_ends is not None:
        cuts.add(spec.ramp_ends.w_from)
    if spec.ramp_ends is not None or spec.land.profile is not None:
        cuts.add(fr.w_edge)
    ws = sorted(w for w in cuts if 0.0 <= w <= fr.w_hi)
    return _union([_ramp_patch(spec, fr, a, b) for a, b in zip(ws[:-1], ws[1:]) if b - a > 1e-9])


def _replace(solid, zone, column):
    """``d = column`` inside ``zone``: (solid \\ zone) ∪ (column ∩ zone)."""
    return _union([_op("cut", solid, [zone]), _common(column, zone)])


def _island_zone(spec: GateProfileSpec, fr: _Frame):
    isl = spec.island
    hi = _Edge(isl.boundary_line, isl.boundary_line[0][1])
    a, b = spec.land.length, isl.end_dist
    lo = _floor_below(hi.chain(a, b))
    return _plan_prism(_band(a, b, lo, hi), fr.z_bot - 1.0, _Z_TOP + 1.0), lo


def _slope_of_t(spec: GateProfileSpec, fr: _Frame, angle_deg: float, t_end: float):
    """Column of ``d = land + tan(angle)·(t − land.length)`` (island / band floors)."""
    L0, dL = spec.land.length, spec.land.depth
    tan = math.tan(math.radians(angle_deg))
    te = t_end + 1.0
    return _tz_prism(
        [(fr.t_lo, _Z_TOP), (te, _Z_TOP), (te, -(dL + tan * (te - L0))), (L0, -dL), (fr.t_lo, -dL)],
        -1.0,
        fr.w_hi,
    )


def _pocket_silhouette(spec: GateProfileSpec, fr: _Frame):
    """Plan prism of the single pocket over w ≥ 0, z ∈ [z_bot, 0], before the cuts."""
    wall = _Edge(spec.outer_wall_line, fr.w_edge)
    upper = wall.chain(0.0, fr.T)
    if upper[-1][1] < 0.0:  # the wall reaches the axis before the pocket ends
        (t1, w1), (t2, w2) = spec.outer_wall_line
        t_tip = t1 + (0.0 - w1) * (t2 - t1) / (w2 - w1)
        upper = [q for q in upper if q[0] < t_tip] + [(t_tip, 0.0)]
    poly = [(0.0, 0.0)] + upper + ([(upper[-1][0], 0.0)] if upper[-1][1] > 0.0 else [])
    return _plan_prism(poly, fr.z_bot, 0.0)


def _corner_cut(spec: GateProfileSpec, fr: _Frame):
    """Steel the round at the outer wall's first point leaves: wedge minus disc."""
    from core.profile_gate import _corner_tangent_length

    (t1, w1), (t2, w2) = spec.outer_wall_line
    r = spec.outer_wall_corner_radius
    tan_len, seg = _corner_tangent_length(spec.outer_wall_line, r)
    ut, uw = (t2 - t1) / seg, (w2 - w1) / seg
    ct, cw = t1 - tan_len, w1 - r
    quad = [(ct, cw), (t1 - tan_len, w1), (t1, w1), (t1 + tan_len * ut, w1 + tan_len * uw)]
    z0, z1 = fr.z_bot - 1.0, _Z_TOP + 1.0
    return _op("cut", _plan_prism(quad, z0, z1), [_cylinder(ct, cw, r, z0, z1)])


def _closure_zone(spec: GateProfileSpec, fr: _Frame):
    line = spec.land.closed_line
    hi = _Edge(line, line[0][1])
    a, b = fr.t_lo, spec.land.length
    return _plan_prism(_band(a, b, _floor_below(hi.chain(a, b)), hi), fr.z_bot - 1.0, _Z_TOP + 1.0)


def _edge_channel_floor(ec, wall: _Edge, t_end: float, fr: _Frame, pocket):
    t_lo, t_hi = ec.t_range if ec.t_range is not None else (0.0, t_end)
    poly = _edge_wall_polyline(wall.line, t_lo, min(t_hi, t_end), wall.before)
    band = _capsules(poly, ec.width, fr.z_bot - 1.0, _Z_TOP + 1.0)
    floor = _box(fr.t_lo, fr.t_hi, -1.0, fr.w_hi, -ec.depth, _Z_TOP)
    return _common(_common(floor, band), pocket)


def _ramp_cut_floor(spec: GateProfileSpec, fr: _Frame, pocket):
    rc = spec.ramp_cut
    (t1, w1), (t2, w2) = rc.line
    L0 = spec.land.length
    zone = _box(L0, fr.t_hi, w2, w1, fr.z_bot - 1.0, _Z_TOP + 1.0)
    flat = _box(fr.t_lo, fr.t_hi, w2, w1, -rc.depth, _Z_TOP)
    if rc.slope_angle_deg >= 90.0:
        beyond = _plan_prism([(t1, w1), (fr.t_hi, w1), (fr.t_hi, w2), (t2, w2)], fr.z_bot, _Z_TOP)
        col = _common(flat, beyond)
    else:
        tan = math.tan(math.radians(rc.slope_angle_deg))

        def depth(t, w):  # depth − (t_cut(w) − t)·tan, a plane
            t_cut = t1 + (t2 - t1) * (w1 - w) / (w1 - w2)
            return rc.depth - (t_cut - t) * tan

        corners = [(L0, w2), (fr.t_hi, w2), (fr.t_hi, w1), (L0, w1)]
        face = _face([_p(t, w, -depth(t, w)) for t, w in corners])
        col = _common(_up(face, min(-depth(t, w) for t, w in corners)), flat)
    return _common(_common(col, zone), pocket)


def _single_pocket(spec: GateProfileSpec, fr: _Frame):
    """The half block (w ≥ 0) of the single-pocket form."""
    solid = _base_column(spec, fr)
    in_weld = None
    if spec.island is not None:
        isl = spec.island
        zone, lo = _island_zone(spec, fr)
        solid = _replace(solid, zone, _slope_of_t(spec, fr, isl.angle_deg, isl.end_dist))
        if isl.weld is not None:
            wt0, wt1 = isl.weld.t_range
            w_cap = fr.w_hi if isl.weld.w_max is None else isl.weld.w_max
            in_weld = _common(zone, _box(wt0, wt1, lo - 1.0, w_cap, fr.z_bot - 2.0, _Z_TOP + 2.0))
            if isl.weld.depth > 0:
                dam = _box(fr.t_lo, fr.t_hi, lo - 1.0, fr.w_hi, -isl.weld.depth, _Z_TOP)
                solid = _replace(solid, in_weld, dam)
    pocket = _pocket_silhouette(spec, fr)
    steel = []
    if spec.island is not None and spec.island.weld is not None and spec.island.weld.depth <= 0:
        steel.append(in_weld)
    if spec.outer_wall_corner_radius is not None:
        steel.append(_corner_cut(spec, fr))
    if spec.land.closed_line is not None:
        steel.append(_closure_zone(spec, fr))
    pocket = _op("cut", pocket, steel)
    solid = _common(solid, pocket)
    floors = [
        _edge_channel_floor(ec, _Edge(spec.outer_wall_line, fr.w_edge), fr.T, fr, pocket)
        for ec in spec.edge_channels
    ]
    if spec.ramp_cut is not None:
        floors.append(_ramp_cut_floor(spec, fr, pocket))
    if spec.land_ends is not None:
        le = spec.land_ends
        flat = _box(fr.t_lo, fr.t_hi, le.w_from, fr.w_hi, -le.depth, _Z_TOP)
        floors.append(_common(flat, pocket))
    return _union([solid] + floors)


def _fans(spec: GateProfileSpec, fr: _Frame):
    """The half block (w ≥ 0) of the fan form: the union of the fans."""
    base = _base_column(spec, fr)
    L0 = spec.land.length
    parts = []
    for sg in spec.sub_gates:
        inner = _Edge(sg.inner_wall_line, sg.inner_wall_line[0][1])
        outer = _Edge(sg.outer_wall_line, sg.outer_wall_line[0][1])
        fan = _plan_prism(_band(0.0, sg.tip_t, inner, outer), fr.z_bot, 0.0)
        col = base
        if sg.island is not None:
            si = sg.island
            band = _band(
                L0,
                si.end_dist,
                _Edge(si.inner_line, si.inner_line[0][1]),
                _Edge(si.outer_line, si.outer_line[0][1]),
            )
            zone = _plan_prism(band, fr.z_bot - 1.0, _Z_TOP + 1.0)
            if si.floor_depth is not None:  # a plateau: never deeper than the ramp
                plateau = _box(fr.t_lo, fr.t_hi, -1.0, fr.w_hi, -si.floor_depth, _Z_TOP)
                isl_col = _common(base, plateau)
            else:
                isl_col = _slope_of_t(spec, fr, si.angle_deg, si.end_dist)
            col = _replace(col, zone, isl_col)
        col = _common(col, fan)
        walls = {"outer": outer, "inner": inner}
        floors = [
            _edge_channel_floor(ec, walls[ec.side], sg.tip_t, fr, fan) for ec in sg.edge_channels
        ]
        parts.append(_union([col] + floors))
    return _union(parts)


def _well(spec: GateProfileSpec, fr: _Frame, w_c: float):
    well = spec.well
    R, D = well.half_width, well.depth
    ta, tb = well.t_range[0] + R, well.t_range[1] - R
    if tb < ta:
        ta = tb = 0.5 * (well.t_range[0] + well.t_range[1])
    if well.wall_angle_deg >= 90.0:
        ends = [_cylinder(t, w_c, R, -D, 0.0) for t in (ta, tb)]
        mid = _box(ta, tb, w_c - R, w_c + R, -D, 0.0) if tb > ta else None
    else:
        r = max(R - D / math.tan(math.radians(well.wall_angle_deg)), 0.0)
        ends = [_cone(t, w_c, r, R, -D, 0.0) for t in (ta, tb)]
        trap = [(w_c - r, -D), (w_c + r, -D), (w_c + R, 0.0), (w_c - R, 0.0)]
        mid = _wz_prism(trap, ta, tb) if tb > ta else None
    solid = _union(ends + [mid])
    return _common(solid, _box(0.0, fr.T, w_c - R - 1.0, w_c + R + 1.0, fr.z_bot, 0.0))


def _runner(spec: GateProfileSpec, fr: _Frame):
    rn = spec.runner
    paths = [rn.path]
    if spec.symmetric:
        paths.append(tuple((t, -w) for t, w in rn.path))
    band = _union([_capsules(p, rn.width / 2.0, -rn.depth, 0.0) for p in paths])
    w_reach = max(abs(w) for _t, w in rn.path) + rn.width
    return _common(band, _box(0.0, fr.T, -w_reach, w_reach, fr.z_bot, 0.0))


@dataclass(frozen=True)
class GateSolid:
    """The resin of the gate block, in the CAD frame (see the module docstring)."""

    shape: object  # TopoDS_Shape
    volume_mm3: float
    faces: int


def _count(shape, kind) -> int:
    k = _k()
    n, e = 0, k["TopExp_Explorer"](shape, kind)
    while e.More():
        n += 1
        e.Next()
    return n


def volume_mm3(shape) -> float:
    """Volume by adaptive integration (the fixed Gauss rule misses on high-degree B-splines)."""
    k = _k()
    g = k["GProp_GProps"]()
    k["BRepGProp"].VolumeProperties_s(shape, g, 1e-9, True)
    return float(g.Mass())


def build_gate_solid(spec: GateProfileSpec) -> GateSolid:
    """The gate block's resin as one valid solid (needs OCP)."""
    spec.validate()
    k = _k()
    fr = _frame(spec)
    half = _single_pocket(spec, fr) if spec.outer_wall_line is not None else _fans(spec, fr)
    if spec.symmetric:
        m = k["gp_Trsf"]()
        m.SetMirror(k["gp_Ax2"](k["gp_Pnt"](0.0, 0.0, 0.0), k["gp_Dir"](1.0, 0.0, 0.0)))
        half = _union([half, _transformed(half, m)])
    extra = []
    if spec.runner is not None:
        extra.append(_runner(spec, fr))
    if spec.well is not None:
        extra.append(_well(spec, fr, 0.0 if spec.symmetric else spec.valve.w))
    solid = _union([half] + extra)
    if not spec.symmetric:  # x = 0 on the valve axis, as in the drawing
        sh = k["gp_Trsf"]()
        sh.SetTranslation(k["gp_Vec"](-spec.valve.w, 0.0, 0.0))
        solid = _transformed(solid, sh)
    uni = k["ShapeUpgrade_UnifySameDomain"](solid, True, True, True)
    uni.Build()
    solid = uni.Shape()
    n_solids = _count(solid, k["TopAbs_SOLID"])
    if n_solids != 1 or not k["BRepCheck_Analyzer"](solid).IsValid():
        raise RuntimeError(f"gate solid is not one valid solid ({n_solids} solids)")
    return GateSolid(solid, volume_mm3(solid), _count(solid, k["TopAbs_FACE"]))


def iges_bytes(shape) -> bytes:
    """The faces of ``shape`` as an IGES file (trimmed surfaces, type 144, mm)."""
    k = _k()
    k["Interface_Static"].SetCVal_s("write.iges.unit", "MM")
    k["Interface_Static"].SetIVal_s("write.iges.brep.mode", 0)
    writer = k["IGESControl_Writer"]("MM", 0)
    e = k["TopExp_Explorer"](shape, k["TopAbs_FACE"])
    while e.More():
        writer.AddShape(e.Current())
        e.Next()
    writer.ComputeModel()
    fd, path = tempfile.mkstemp(suffix=".igs")
    os.close(fd)
    try:
        if not writer.Write(path):
            raise RuntimeError("IGES writer failed")
        with open(path, "rb") as f:
            return f.read()
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# reading a solid back from above
# ---------------------------------------------------------------------------


def _triangles(shape) -> np.ndarray:
    """All triangles of ``shape``'s mesh, shape (n, 3, 3), CAD coordinates.

    Cones and cylinders are meshed by angle (a 4.5 mm cone sags 5e-5 mm at
    0.01 rad) and first, so the edges they share with their neighbours are
    cut that finely too; everything else by the linear deflection alone --
    by angle, a land-end curve that turns tightly at the exit's edge would
    be cut into micron segments.
    """
    k = _k()
    faces = []
    e = k["TopExp_Explorer"](shape, k["TopAbs_FACE"])
    while e.More():
        face = k["TopoDS"].Face_s(e.Current())
        kind = k["BRepAdaptor_Surface"](face).GetType()
        faces.append((kind not in (k["GeomAbs_Cone"], k["GeomAbs_Cylinder"]), face))
        e.Next()
    out = []
    for not_rot, face in sorted(faces, key=lambda f: f[0]):
        angle = 0.5 if not_rot else MESH_ANGLE_RAD
        k["BRepMesh_IncrementalMesh"](face, MESH_DEFLECTION_MM, False, angle, False)
        loc = k["TopLoc_Location"]()
        tri = k["BRep_Tool"].Triangulation_s(face, loc)
        if tri is None:
            continue
        trsf = loc.Transformation()
        nodes = np.array(
            [
                [(q := tri.Node(i).Transformed(trsf)).X(), q.Y(), q.Z()]
                for i in range(1, tri.NbNodes() + 1)
            ]
        )
        idx = np.array([tri.Triangle(i).Get() for i in range(1, tri.NbTriangles() + 1)]) - 1
        out.append(nodes[idx])
    return np.concatenate(out) if out else np.zeros((0, 3, 3))


def raster_depth(shape, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Depth (``−z`` of the lowest point) of ``shape`` under each centre, NaN where none.

    ``x`` (nx,) and ``y`` (ny,) are increasing CAD coordinates; the result
    is (ny, nx). The solid is a column ``−d ≤ z ≤ 0``, so the lowest point
    of the non-vertical faces over a centre is the depth there.
    """
    tris = _triangles(shape)
    z_low = np.full((len(y), len(x)), np.inf)
    for (x1, y1, z1), (x2, y2, z2), (x3, y3, z3) in tris:
        den = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        if abs(den) < 1e-12:
            continue  # a vertical face
        i0, i1 = np.searchsorted(x, [min(x1, x2, x3) - 1e-9, max(x1, x2, x3) + 1e-9])
        j0, j1 = np.searchsorted(y, [min(y1, y2, y3) - 1e-9, max(y1, y2, y3) + 1e-9])
        if i0 >= i1 or j0 >= j1:
            continue
        X, Y = np.meshgrid(x[i0:i1], y[j0:j1])
        l1 = ((y2 - y3) * (X - x3) + (x3 - x2) * (Y - y3)) / den
        l2 = ((y3 - y1) * (X - x3) + (x1 - x3) * (Y - y3)) / den
        l3 = 1.0 - l1 - l2
        tol = -1e-9
        inside = (l1 >= tol) & (l2 >= tol) & (l3 >= tol)
        z = l1 * z1 + l2 * z2 + l3 * z3
        sub = z_low[j0:j1, i0:i1]
        np.minimum(sub, np.where(inside, z, np.inf), out=sub)
    return np.where(np.isfinite(z_low), -z_low, np.nan)


@dataclass(frozen=True)
class FieldCheck:
    """The solid read back from above against the solver's depth field."""

    cells: int  # pocket cells of the field
    outline_mismatch: int  # cells in one but not the other
    max_depth_diff_mm: float  # over the cells both have


def check_against_field(shape, x: np.ndarray, y_display: np.ndarray, depth: np.ndarray):
    """Compare with a field in the drawing's frame (x from the valve axis, y = −t).

    ``depth`` is (ny, nx), NaN outside the pocket (``gate_drawing.drawing_field``).
    """
    order = np.argsort(y_display)
    y_cad = CAD_Y_AT_EXIT + y_display[order]
    got = raster_depth(shape, x, y_cad)[np.argsort(order)]
    want_in, got_in = np.isfinite(depth), np.isfinite(got)
    both = want_in & got_in
    diff = float(np.max(np.abs(got[both] - depth[both]))) if both.any() else 0.0
    return FieldCheck(int(want_in.sum()), int((want_in ^ got_in).sum()), diff)


#: Largest depth difference :class:`GateIges` accepts (the cones' mesh sag is ~7e-4 mm).
CHECK_TOL_MM = 2e-3


@dataclass(frozen=True)
class GateIges:
    """An IGES of the block and how it compares with the solver's own field."""

    iges: bytes
    volume_mm3: float
    faces: int
    field_volume_mm3: float  # the same pocket on the check mesh
    cell_mm: float  # the check mesh
    check: FieldCheck

    @property
    def ok(self) -> bool:
        """Same outline cell for cell, and the depths within :data:`CHECK_TOL_MM`."""
        return self.check.outline_mismatch == 0 and self.check.max_depth_diff_mm <= CHECK_TOL_MM


def export_gate_iges(spec: GateProfileSpec, plate, cell_size_mm: float | None = None) -> GateIges:
    """Build the solid, write it, and read it back against the drawing's field.

    ``plate`` only sets the grid the field is built on (the block depends on
    the product's width alone); ``cell_size_mm`` defaults to the drawing's
    mesh (0.1 mm unless the block is too large for it).
    """
    from core.gate_drawing import drawing_field

    solid = build_gate_solid(spec)
    field = drawing_field(spec, plate, cell_size_mm)
    dx = field.geometry.cell_size_mm
    return GateIges(
        iges=iges_bytes(solid.shape),
        volume_mm3=solid.volume_mm3,
        faces=solid.faces,
        field_volume_mm3=float(np.nansum(field.depth) * dx * dx),
        cell_mm=float(dx),
        check=check_against_field(solid.shape, field.x, field.y, field.depth),
    )


def plate_for(spec: GateProfileSpec):
    """The narrowest product the block fits on (for a check without a run's plate).

    The builder centres the valve axis (symmetric) or the exit (one-sided)
    on the plate and rejects a pocket that overhangs the grid.
    """
    from core.profile_gate import ProfilePlateConfig

    gew = spec.gate_exit_width
    well_hw = spec.well.half_width if spec.well is not None else 0.0
    if spec.symmetric:
        reach = max(gew / 2.0, spec.w_max(), well_hw) + abs(spec.valve.w)
        width = max(gew, 2.0 * reach)
    else:
        pos = max(gew, spec.w_max(), well_hw)
        neg = max(well_hw, -spec.w_min())
        width = max(gew, 2.0 * max(neg + gew / 2.0, pos - gew / 2.0))
    return ProfilePlateConfig(plate_w_mm=width, plate_h_mm=1.0, plate_thk_mm=spec.land.depth)


def _spec_from_file(path: str) -> GateProfileSpec:
    """A spec JSON, or the ``settings.json`` of a run (``geometry.gate_profile``)."""
    import json

    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if "gate_profile" in d.get("geometry", {}):
        d = d["geometry"]["gate_profile"]
    return GateProfileSpec.from_dict(d)


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m core.gate_iges <spec.json | settings.json> <out.igs>``."""
    import argparse

    ap = argparse.ArgumentParser(prog="python -m core.gate_iges", description=main.__doc__)
    ap.add_argument("spec", help="gate spec JSON, or a run's settings.json")
    ap.add_argument("out", help="IGES file to write")
    ap.add_argument("--cell", type=float, default=None, help="check mesh [mm] (default 0.1)")
    a = ap.parse_args(argv)
    if not available():
        print("OCP is not installed: pip install -e '.[cad]'")
        return 2
    spec = _spec_from_file(a.spec)
    res = export_gate_iges(spec, plate_for(spec), a.cell)
    with open(a.out, "wb") as f:
        f.write(res.iges)
    c = res.check
    print(
        f"{a.out}: {res.faces} faces, volume {res.volume_mm3:.1f} mm3 "
        f"(field {res.field_volume_mm3:.1f} at {res.cell_mm:g} mm); outline mismatch "
        f"{c.outline_mismatch}/{c.cells} cells, max depth difference {c.max_depth_diff_mm:.4f} mm"
    )
    return 0 if res.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
