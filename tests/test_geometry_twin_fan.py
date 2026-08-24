"""Multi-fan gate blocks: ``sub_gates`` + ``runner`` in ``GateProfileSpec``.

The pocket is the union of fan-shaped sub-gates, a runner band and the well;
everything else is steel at the PL. With two mirrored fans whose inner walls
meet on the land at w = 0 the steel between them is a full cut-out shaped
like a deformed rhombus, and the runner is the only path from the valve to
the fans -- so the runner is load-bearing for solvability, not decoration.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from core import (
    GateProfileSpec,
    HeleShawSolver,
    MaterialDB,
    ProfilePlateConfig,
    build_profile_gate_geometry,
)

DEMO_JSON = Path(__file__).parent.parent / "data" / "gate_profiles" / "demo_twin_fan_gate.json"


def _twin_dict(**overrides) -> dict:
    """Two mirrored straight-walled fans, no island / runner / well.

    Both fan lines start on the land (t = 2), so the land strip is the full
    half-width and the fans meet at w = 0 in a sharp apex. The volume of the
    fan region has a closed form (linear width in t).
    """
    base = {
        "name": "twin",
        "units": "mm",
        "symmetric": True,
        "gate_exit_width": 200.0,
        "land": {"depth": 0.4, "length": 2.0},
        "main_ramp": {"angle_deg": 10.0, "cap_depth": 2.4},
        "sub_gates": [
            {
                "inner_wall_line": [[2.0, 0.0], [12.0, 45.0]],
                "outer_wall_line": [[2.0, 100.0], [12.0, 55.0]],
                "tip_t": 12.0,
            }
        ],
        "valve": {"t": 18.0, "w": 0.0, "orifice_diameter": 3.0},
    }
    base.update(overrides)
    return base


def _twin(**overrides) -> GateProfileSpec:
    return GateProfileSpec.from_dict(_twin_dict(**overrides))


def _demo() -> GateProfileSpec:
    return GateProfileSpec.from_json_file(DEMO_JSON)


def _plate(**overrides) -> ProfilePlateConfig:
    base = dict(plate_w_mm=300.0, plate_h_mm=50.0, plate_thk_mm=0.4, pad_mm=5.0)
    base.update(overrides)
    return ProfilePlateConfig(**base)


def _tw(geom, spec, plate):
    """(t, |w|) at every cell centre, same convention as the builder."""
    iy, ix = np.indices(geom.mask.shape)
    yy = (iy + 0.5) * geom.cell_size_mm
    xx = (ix + 0.5) * geom.cell_size_mm
    t = plate.pad_mm + spec.t_max() - yy
    wa = np.abs(xx - (plate.pad_mm + plate.plate_w_mm / 2.0))
    return t, wa


def _gate_cells(geom):
    return geom.mask & ~geom.compression_mask


# ----------------------- JSON I/O ------------------


def test_demo_twin_fan_roundtrip() -> None:
    spec = _demo()
    assert spec.outer_wall_line is None and len(spec.sub_gates) == 1
    assert spec.runner is not None and spec.sub_gates[0].island is not None
    assert GateProfileSpec.from_json(spec.to_json()) == spec
    # the settings record travels as dataclasses.asdict (tuples → lists,
    # None for the absent outer wall): it must read back too
    assert GateProfileSpec.from_dict(json.loads(json.dumps(dataclasses.asdict(spec)))) == spec


def test_to_dict_omits_the_absent_outer_wall_and_single_pocket_omits_fans() -> None:
    assert "outer_wall_line" not in _twin().to_dict()
    single = GateProfileSpec.from_dict(
        {
            **_twin_dict(),
            "sub_gates": None,
            "outer_wall_line": [[2.0, 100.0], [24.0, 20.0]],
        }
    )
    d = single.to_dict()
    assert "sub_gates" not in d and "runner" not in d


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda d: d.update(outer_wall_line=[[2.0, 100.0], [24.0, 20.0]]), "exactly one"),
        (lambda d: d.pop("sub_gates"), "exactly one"),
        (lambda d: d.update(sub_gates=[]), "exactly one"),
        (
            lambda d: d.update(
                island={
                    "angle_deg": 2.5,
                    "boundary_line": [[2.0, 40.0], [9.0, 10.0]],
                    "end_dist": 9.0,
                }
            ),
            "top-level island",
        ),
        (lambda d: d["sub_gates"][0].update(tip_t=2.0), "tip_t"),
        (lambda d: d["sub_gates"][0].update(inner_wall_line=[[2.0, 0.0], [12.0, 60.0]]), "wider"),
        (
            lambda d: d["sub_gates"][0].update(inner_wall_line=[[12.0, 0.0], [2.0, 45.0]]),
            "increasing",
        ),
        (lambda d: d["sub_gates"][0].update(outer_wall_line=[[2.0, -1.0], [12.0, 55.0]]), "≥ 0"),
        (lambda d: d["sub_gates"][0].update(bogus=1), "unknown key"),
        (
            lambda d: d["sub_gates"][0].update(
                island={
                    "angle_deg": 2.5,
                    "inner_line": [[2.0, 70.0], [9.0, 44.0]],
                    "outer_line": [[2.0, 30.0], [9.0, 56.0]],
                    "end_dist": 9.0,
                }
            ),
            "cross",
        ),
        (
            lambda d: d["sub_gates"][0].update(
                island={
                    "angle_deg": 20.0,
                    "inner_line": [[2.0, 30.0], [9.0, 44.0]],
                    "outer_line": [[2.0, 70.0], [9.0, 56.0]],
                    "end_dist": 9.0,
                }
            ),
            "shallow side",
        ),
        (
            lambda d: d.update(
                runner={"width": 0.0, "depth": 2.0, "path": [[18.0, 0.0], [12.0, 50.0]]}
            ),
            "positive",
        ),
        (
            lambda d: d.update(runner={"width": 6.0, "depth": 2.0, "path": [[18.0, 0.0]]}),
            "at least 2",
        ),
        (
            lambda d: d.update(
                runner={"width": 6.0, "depth": 2.0, "path": [[18.0, 0.0], [18.0, 0.0]]}
            ),
            "zero-length",
        ),
        (
            lambda d: d.update(
                runner={"width": 6.0, "depth": 2.0, "path": [[18.0, -1.0], [12.0, 50.0]]}
            ),
            "≥ 0",
        ),
        (lambda d: d.update(runner={"width": 6.0, "depth": 2.0, "path": "ab"}), "runner.path"),
        (lambda d: d.update(sub_gates={"tip_t": 12.0}), "list of objects"),
        (lambda d: d.update(sub_gates=[3]), "sub_gates\\[0\\]"),
    ],
)
def test_validation_rejects(mutate, match) -> None:
    d = _twin_dict()
    mutate(d)
    with pytest.raises(ValueError, match=match):
        GateProfileSpec.from_dict(d)


def test_unknown_key_under_a_fan_island_reports_its_path() -> None:
    d = _twin_dict()
    d["sub_gates"][0]["island"] = {
        "angle_deg": 2.5,
        "inner_line": [[2.0, 30.0], [9.0, 44.0]],
        "outer_line": [[2.0, 70.0], [9.0, 56.0]],
        "end_dist": 9.0,
        "weld": {"t_range": [3.0, 9.0], "depth": 0.1},
    }
    with pytest.raises(ValueError, match=r"sub_gates\[0\]\.island"):
        GateProfileSpec.from_dict(d)


# ----------------------- extents ------------------


def test_t_max_and_w_max_cover_the_runner_and_the_fan_tip() -> None:
    bare = _twin()
    # nothing reaches past the valve orifice (18 + 1.5) and the ramp cap
    assert bare.t_max() == pytest.approx(max(12.0, bare.ramp_cap_t(), 19.5))
    assert bare.w_max() == pytest.approx(100.0)
    with_runner = _twin(runner={"width": 6.0, "depth": 2.0, "path": [[18.0, 0.0], [30.0, 120.0]]})
    assert with_runner.t_max() == pytest.approx(33.0)  # 30 + width/2
    assert with_runner.w_max() == pytest.approx(123.0)


def test_pocket_overhanging_the_grid_through_the_runner_is_rejected() -> None:
    spec = _twin(runner={"width": 6.0, "depth": 2.0, "path": [[18.0, 0.0], [18.0, 160.0]]})
    with pytest.raises(ValueError, match="overhangs"):
        build_profile_gate_geometry(spec, _plate(), cell_size_mm=1.0)


# ----------------------- silhouette ------------------


def test_land_strip_spans_the_full_exit_width_and_the_centre_is_steel() -> None:
    spec = _twin()
    plate = _plate()
    g = build_profile_gate_geometry(spec, plate, cell_size_mm=0.5)
    t, wa = _tw(g, spec, plate)
    gate = _gate_cells(g)
    land = gate & (t >= 0) & (t < 2.0)
    # every column within the exit half-width is cavity on the land
    assert np.array_equal(land[(t >= 0) & (t < 2.0)], (wa <= 100.0)[(t >= 0) & (t < 2.0)])
    # halfway to the tip the inner wall is at w = 22.5, the outer at 77.5:
    # inside the inner wall (rhombus) and outside the outer wall are steel
    mid = (t > 6.9) & (t < 7.1)
    assert not g.mask[mid & (wa < 22.5 - 0.5)].any()
    assert g.mask[mid & (wa > 22.5 + 0.5) & (wa < 77.5 - 0.5)].all()
    assert not g.mask[mid & (wa > 77.5 + 0.5)].any()
    # past the tip there is no pocket at all
    assert not g.mask[(t > 12.5) & (t <= spec.t_max())].any()


def test_fan_area_matches_the_closed_form() -> None:
    """Fan width is linear in t, so the pocket footprint is a trapezoid plus
    the land strip. Both halves: 2·[land + ∫(w_out − w_in) dt]."""
    spec = _twin()
    plate = _plate()
    dx = 0.25
    g = build_profile_gate_geometry(spec, plate, cell_size_mm=dx)
    area = _gate_cells(g).sum() * dx * dx
    land = 2.0 * 100.0
    fan = (12.0 - 2.0) * ((100.0 - 0.0) + (55.0 - 45.0)) / 2.0
    assert area == pytest.approx(2.0 * (land + fan), rel=0.02)


def test_fan_depth_is_the_land_plus_capped_ramp() -> None:
    spec = _twin()
    plate = _plate()
    g = build_profile_gate_geometry(spec, plate, cell_size_mm=0.5)
    t, wa = _tw(g, spec, plate)
    gate = _gate_cells(g)
    expect = np.where(
        t <= 2.0, 0.4, np.minimum(0.4 + math.tan(math.radians(10.0)) * (t - 2.0), 2.4)
    )
    assert np.allclose(g.thickness_mm[gate], expect[gate])
    # the ramp reaches its cap at t = 2 + 2/tan(10°) ≈ 13.3, past the tip (12):
    # the deepest fan cell is still on the ramp
    assert 2.0 < g.thickness_mm[gate].max() < 2.4


def test_fan_island_is_a_shallower_band_between_its_two_lines() -> None:
    spec = _demo()
    plate = _plate()
    g = build_profile_gate_geometry(spec, plate, cell_size_mm=0.5)
    t, wa = _tw(g, spec, plate)
    row = (t > 5.7) & (t < 6.3)  # inside the island's t-range (2, 9]
    # island lines at t = 6: inner 30 + 14·4/7 = 38, outer 70 − 14·4/7 = 62
    inside = row & (wa > 38.5) & (wa < 61.5)
    outside = row & g.mask & ((wa < 37.5) | (wa > 62.5)) & (wa < 90.0)
    ramp_depth = 0.4 + math.tan(math.radians(10.0)) * (6.0 - 2.0)
    isl_depth = 0.4 + math.tan(math.radians(2.5)) * (6.0 - 2.0)
    assert g.mask[inside].all()
    assert inside.sum() > 40 and outside.sum() > 40
    assert np.allclose(g.thickness_mm[inside], isl_depth, atol=0.06)
    assert np.allclose(g.thickness_mm[outside], ramp_depth, atol=0.06)
    # beyond end_dist the band is back on the main ramp
    after = (t > 10.2) & (t < 10.8) & g.mask & ~g.compression_mask
    assert after.any()
    assert g.thickness_mm[after].min() > isl_depth + 0.5


# ----------------------- runner ------------------


def test_runner_across_steel_adds_a_capsule_of_its_depth() -> None:
    """A runner laid entirely in the steel beyond the fan tips adds exactly
    width·length·depth plus the two half-disc end caps."""
    plate = _plate()
    bare = _twin()
    path = [[20.0, 10.0], [20.0, 40.0]]  # t = 20 > tip 12: steel only
    with_runner = _twin(runner={"width": 4.0, "depth": 1.5, "path": path})
    dx = 0.2
    g0 = build_profile_gate_geometry(bare, plate, cell_size_mm=dx)
    g1 = build_profile_gate_geometry(with_runner, plate, cell_size_mm=dx)
    # the block gets taller (t_max grows) so compare pocket volumes, not arrays
    v0 = g0.thickness_mm[_gate_cells(g0)].sum() * dx * dx
    v1 = g1.thickness_mm[_gate_cells(g1)].sum() * dx * dx
    capsule = 4.0 * 30.0 + math.pi * 2.0**2
    assert v1 - v0 == pytest.approx(2.0 * capsule * 1.5, rel=0.03)  # both halves


def test_runner_depth_is_a_floor_not_an_override() -> None:
    """Inside a fan the runner keeps whichever is deeper: the fan's ramp
    (near the tip, 2.4) or the runner (1.0)."""
    plate = _plate()
    bare = _twin()
    # width 3 keeps t_max at the valve orifice (18 + 1.5) so the grids match
    shallow = _twin(runner={"width": 3.0, "depth": 1.0, "path": [[18.0, 0.0], [12.0, 50.0]]})
    g0 = build_profile_gate_geometry(bare, plate, cell_size_mm=0.5)
    g1 = build_profile_gate_geometry(shallow, plate, cell_size_mm=0.5)
    assert g1.mask.shape == g0.mask.shape
    t, wa = _tw(g1, shallow, plate)
    # distance to the runner segment (18, 0) → (12, 50) in the (t, |w|) plane
    vt, vw = 12.0 - 18.0, 50.0 - 0.0
    s = np.clip(((t - 18.0) * vt + wa * vw) / (vt * vt + vw * vw), 0.0, 1.0)
    band = np.hypot(t - (18.0 + s * vt), wa - s * vw) <= 1.5 - 0.35  # clear of the edge
    # fan cells outside the band are untouched
    off = g0.mask & ~band
    assert np.array_equal(g1.thickness_mm[off], g0.thickness_mm[off])
    # cells that were fan pocket and are also in the runner band keep max(fan, 1.0)
    both = g0.mask & band & ~g1.compression_mask
    assert both.any()
    assert np.all(g1.thickness_mm[both] >= 1.0 - 1e-9)
    deeper_fan = both & (g0.thickness_mm > 1.0)
    assert deeper_fan.any()
    assert np.array_equal(g1.thickness_mm[deeper_fan], g0.thickness_mm[deeper_fan])
    # new cells (runner across steel) sit at exactly the runner depth
    new = g1.mask & ~g0.mask & ~g1.compression_mask
    assert new.any()
    assert np.allclose(g1.thickness_mm[new], 1.0)


def test_runner_is_what_connects_the_valve_to_the_fans() -> None:
    """Without the runner the valve well is an island of cavity: the solver
    rejects the unreachable fans. With it the block solves."""
    plate = _plate()
    db = MaterialDB()

    def solver_for(spec):
        g = build_profile_gate_geometry(spec, plate, cell_size_mm=1.0)
        solver = HeleShawSolver(
            geometry=g,
            material=db["PP"],
            melt_temperature_K=503.15,
            mold_temperature_K=313.15,
            injection_velocity_mms=100.0,
            injection_volume_flow_cm3s=20.0,
        )
        return g, solver

    d = _demo().to_dict()
    d.pop("runner")
    _g, solver = solver_for(GateProfileSpec.from_dict(d))
    with pytest.raises(ValueError):
        solver.solve(num_frames=4)

    g, solver = solver_for(_demo())
    res = solver.solve(num_frames=8)
    assert res.total_fill_time_s > 0
    assert np.isfinite(res.fill_time_s[g.mask]).all()


def test_gate_cells_sit_on_the_valve_inside_the_runner() -> None:
    spec = _demo()
    plate = _plate()
    g = build_profile_gate_geometry(spec, plate, cell_size_mm=1.0)
    t, wa = _tw(g, spec, plate)
    assert len(g.gates) >= 4
    for iy, ix in g.gates:
        assert abs(t[iy, ix] - 18.0) <= 1.5 + 0.5 and wa[iy, ix] <= 1.5 + 0.5
        assert g.mask[iy, ix]


def test_well_is_unioned_with_the_fans_and_runner() -> None:
    plate = _plate()
    spec = _demo()
    g = build_profile_gate_geometry(spec, plate, cell_size_mm=0.5)
    t, wa = _tw(g, spec, plate)
    floor = (t > 15.0) & (t < 21.0) & (wa < 0.6)
    assert g.mask[floor].all()
    assert np.allclose(g.thickness_mm[floor], 4.0, atol=0.05)


def test_single_pocket_specs_are_untouched_by_the_extension() -> None:
    """The legacy demo spec must build bit-identically with the new fields
    at their defaults (sub_gates=(), runner=None)."""
    legacy = GateProfileSpec.from_json_file(DEMO_JSON.with_name("demo_profile_gate.json"))
    assert legacy.sub_gates == () and legacy.runner is None
    g = build_profile_gate_geometry(legacy, _plate(), cell_size_mm=1.0)
    again = build_profile_gate_geometry(copy.deepcopy(legacy), _plate(), cell_size_mm=1.0)
    assert np.array_equal(g.mask, again.mask)
    assert np.array_equal(g.thickness_mm, again.thickness_mm)
