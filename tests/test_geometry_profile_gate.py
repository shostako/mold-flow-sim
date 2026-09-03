"""Tests for the JSON-spec-driven profile-gate builder (`core/profile_gate.py`)."""

from __future__ import annotations

import copy
import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from core import (
    EdgeChannelSpec,
    GateProfileSpec,
    HeleShawSolver,
    LandSpec,
    MaterialDB,
    ProfilePlateConfig,
    WeldSpec,
    build_profile_gate_geometry,
)
from core.profile_gate import _iter_numbers

DEMO_JSON = Path(__file__).parent.parent / "data" / "gate_profiles" / "demo_profile_gate.json"


def _minimal_spec_dict(**overrides) -> dict:
    """Straight-walled pocket (no island, no well) with a closed-form volume."""
    base = {
        "name": "minimal",
        "units": "mm",
        "symmetric": True,
        "gate_exit_width": 200.0,
        "land": {"depth": 0.4, "length": 2.0},
        "main_ramp": {"angle_deg": 10.0, "cap_depth": 2.4},
        "outer_wall_line": [[0.0, 100.0], [24.0, 100.0]],
        "valve": {"t": 20.0, "w": 0.0, "orifice_diameter": 3.0},
    }
    base.update(overrides)
    return base


def _minimal_spec(**overrides) -> GateProfileSpec:
    return GateProfileSpec.from_dict(_minimal_spec_dict(**overrides))


def _demo_spec() -> GateProfileSpec:
    return GateProfileSpec.from_json_file(DEMO_JSON)


def _plate(**overrides) -> ProfilePlateConfig:
    base = dict(plate_w_mm=300.0, plate_h_mm=50.0, plate_thk_mm=0.4, pad_mm=5.0)
    base.update(overrides)
    return ProfilePlateConfig(**base)


def _numeric_slots(obj, path=()):
    """Every numeric leaf in a spec dict, as a path of keys / list indices."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield path
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _numeric_slots(v, (*path, k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _numeric_slots(v, (*path, i))


def _set_at(obj, path, value):
    for step in path[:-1]:
        obj = obj[step]
    obj[path[-1]] = value


def _grid(geom):
    """Cell-center coordinates in mm."""
    iy, ix = np.indices(geom.shape)
    yy = (iy + 0.5) * geom.cell_size_mm
    xx = (ix + 0.5) * geom.cell_size_mm
    return yy, xx


# ----------------------- JSON I/O ------------------


def test_spec_json_roundtrip() -> None:
    spec = _demo_spec()
    again = GateProfileSpec.from_json(spec.to_json())
    assert again == spec


def test_minimal_spec_roundtrip_without_optionals() -> None:
    spec = _minimal_spec()
    assert spec.island is None and spec.well is None
    again = GateProfileSpec.from_json(spec.to_json())
    assert again == spec


def test_from_dict_missing_key_reports_path() -> None:
    d = _minimal_spec_dict()
    del d["land"]["depth"]
    with pytest.raises(ValueError, match="land.depth"):
        GateProfileSpec.from_dict(d)


def test_from_dict_rejects_unknown_key() -> None:
    d = _minimal_spec_dict()
    d["land"]["depht"] = 0.4  # typo
    with pytest.raises(ValueError, match="depht"):
        GateProfileSpec.from_dict(d)


def test_every_numeric_field_rejects_non_finite() -> None:
    """NaN/Inf must die at the parser, not inside validate.

    Every range check in ``validate`` is a comparison, and every comparison
    against NaN is False — so a NaN passes *both* sides of a bounds test.
    Downstream it either poisons the depth field (NaN volume, NaN solve) or,
    in a mask test, silently drops the feature. This sweeps all numeric
    fields rather than naming the two that were reported, so a field added
    later cannot quietly reopen the hole.
    """
    base = _weld_spec().to_dict()
    slots = list(_numeric_slots(base))
    # land 2 + main_ramp 2 + island (1 + 4 line + 1 + weld 3) + wall 4
    # + well (2 t_range + 1 hw + 1 depth + 2 floor + 1 angle) + valve 3 + width 1
    assert len(slots) == 28, slots
    for bad in (float("nan"), float("inf"), float("-inf")):
        for path in slots:
            d = copy.deepcopy(base)
            _set_at(d, path, bad)
            with pytest.raises(ValueError, match="finite"):
                GateProfileSpec.from_dict(d)


def test_validate_walk_sees_every_numeric_leaf() -> None:
    """The non-finite check walks the dataclass, so it must reach everything.

    This is the load-bearing half of the guarantee: ``validate`` runs one
    loop over ``_iter_numbers``, so if the walk reaches every numeric leaf,
    every leaf is checked — including fields added after this was written.
    """
    spec = _weld_spec()
    walked = {label for label, _ in _iter_numbers(spec, "")}
    from_json = {".".join(str(x) for x in path) for path in _numeric_slots(spec.to_dict())}
    assert len(walked) == 28, sorted(walked)
    # same leaf count as the serialized form (paths differ: [i] vs .i)
    assert len(walked) == len(from_json)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda s: dataclasses.replace(s, gate_exit_width=float("nan")), id="top"),
        pytest.param(
            lambda s: dataclasses.replace(s, land=LandSpec(depth=float("nan"), length=1.0)),
            id="nested",
        ),
        pytest.param(
            lambda s: dataclasses.replace(
                s,
                island=dataclasses.replace(
                    s.island, weld=WeldSpec(t_range=(6.0, 14.0), depth=float("nan"))
                ),
            ),
            id="twice-nested",
        ),
        pytest.param(
            lambda s: dataclasses.replace(
                s,
                island=dataclasses.replace(
                    s.island, weld=WeldSpec(t_range=(float("nan"), 14.0), depth=0.1)
                ),
            ),
            id="tuple-element",
        ),
    ],
)
def test_direct_construction_rejects_non_finite(mutate) -> None:
    """The spec dataclasses are exported and can be built without the parser.

    Guarding only ``_num`` / ``_pair`` would leave this path open, and a
    NaN there reaches the rasterizer exactly the same way.
    """
    bad = mutate(_weld_spec())
    with pytest.raises(ValueError, match="finite"):
        bad.validate()


def _sequence_slots(obj, path=()):
    """Paths of every list-valued field in a spec dict (t_range / line pairs)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, list):
                yield (*path, k)
            else:
                yield from _sequence_slots(v, (*path, k))


# Two-item *iterables* that are not two-item arrays of numbers. Unpacking
# accepts several of these silently, which is the whole point.
_MALFORMED_SEQUENCES = [
    "68",  # a 2-char string unpacks into ('6', '8') → (6.0, 8.0)
    [1],
    [1, 2, 3],
    {"a": 1, "b": 2},
    [True, False],
    ["1", "2"],
    5,
    [[1, 2], [3, 4], [5, 6]],
]
# `None` is deliberately absent: JSON ``null`` means "omitted" for the
# optional fields (well.floor_t_range), which is a documented behaviour,
# not a hole.


@pytest.mark.parametrize("bad", _MALFORMED_SEQUENCES, ids=lambda b: repr(b)[:16])
def test_sequence_fields_reject_malformed_values(bad) -> None:
    """Every ``[a, b]`` / ``[[t,w],[t,w]]`` field must require a real array.

    Sweeping all list-valued fields rather than naming the reported one:
    ``t_range`` and the boundary lines share the same two helpers, so a hole
    in either shows up everywhere. Note how weak the accidental protection
    is — most of these were caught only by a later *range* check, which is
    luck, not validation: ``weld.t_range = "68"`` lands inside
    ``[land.length, end_dist]`` and used to sail through as a band at 6–8 mm.
    """
    base = _weld_spec().to_dict()
    slots = list(_sequence_slots(base))
    assert len(slots) == 5, slots  # wall line, island line, weld/well/floor ranges
    for path in slots:
        d = copy.deepcopy(base)
        _set_at(d, path, bad)
        with pytest.raises(ValueError, match="gate profile JSON"):
            GateProfileSpec.from_dict(d)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
def test_numpy_scalar_fields_are_walked(dtype) -> None:
    """A spec can hold NumPy scalars — the walk must not skip them.

    ``np.float32`` / ``np.int64`` are not subclasses of the builtin
    ``float`` / ``int``, so a type gate written as ``(int, float)`` drops
    them without a trace: the leaf count falls and the value reaches the
    rasterizer unchecked. (``np.float64`` does subclass ``float``, so it
    would pass either gate — which is exactly why testing only that one
    would prove nothing.)
    """
    spec = _weld_spec()
    n_plain = len(list(_iter_numbers(spec, "")))
    swapped = dataclasses.replace(
        spec,
        island=dataclasses.replace(spec.island, weld=WeldSpec(t_range=(dtype(6), 14.0), depth=0.1)),
    )
    assert len(list(_iter_numbers(swapped, ""))) == n_plain
    swapped.validate()  # finite NumPy values stay acceptable


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_numpy_nan_is_rejected(dtype) -> None:
    bad = _weld_spec()
    bad = dataclasses.replace(
        bad,
        island=dataclasses.replace(
            bad.island, weld=WeldSpec(t_range=(6.0, 14.0), depth=dtype("nan"))
        ),
    )
    with pytest.raises(ValueError, match="finite"):
        bad.validate()


def test_json_nan_literal_is_rejected() -> None:
    """json.loads accepts the bare NaN literal — the parser must not."""
    text = _weld_spec().to_json().replace('"depth": 0.1', '"depth": NaN')
    assert "NaN" in text
    with pytest.raises(ValueError, match="finite"):
        GateProfileSpec.from_json(text)


def test_from_dict_rejects_non_mm_units() -> None:
    with pytest.raises(ValueError, match="units"):
        GateProfileSpec.from_dict(_minimal_spec_dict(units="inch"))


def test_from_dict_rejects_non_object_section() -> None:
    # a scalar where an object is expected must be a ValueError with the
    # section name (not a TypeError escaping the UI's error handling)
    with pytest.raises(ValueError, match="'land' must be an object"):
        GateProfileSpec.from_dict(_minimal_spec_dict(land=5))
    with pytest.raises(ValueError, match="'well' must be an object"):
        GateProfileSpec.from_dict(_minimal_spec_dict(well="abc"))


def test_well_floor_t_range_is_optional_reference_metadata() -> None:
    # floor_t_range is drawing-reference metadata: omitting it (or changing
    # it) must not change the rasterized geometry
    well = {
        "shape": "obround",
        "t_range": [14.0, 26.0],
        "half_width": 4.0,
        "depth": 4.0,
        "wall_angle_deg": 60,
    }
    spec_none = _minimal_spec(well=dict(well))
    spec_ref = _minimal_spec(well=dict(well, floor_t_range=[16.31, 23.69]))
    assert spec_none.well.floor_t_range is None
    g_none = build_profile_gate_geometry(spec_none, _plate())
    g_ref = build_profile_gate_geometry(spec_ref, _plate())
    np.testing.assert_array_equal(g_none.thickness_mm, g_ref.thickness_mm)
    # round-trip keeps the reference field when present, omits it when absent
    assert GateProfileSpec.from_json(spec_ref.to_json()) == spec_ref
    assert GateProfileSpec.from_json(spec_none.to_json()) == spec_none


# ----------------------- validation ------------------


def test_validation_rejects_cap_below_land_depth() -> None:
    with pytest.raises(ValueError, match="cap_depth"):
        _minimal_spec(main_ramp={"angle_deg": 10.0, "cap_depth": 0.1})


def test_validation_rejects_non_increasing_wall_line() -> None:
    with pytest.raises(ValueError, match="outer_wall_line"):
        _minimal_spec(outer_wall_line=[[24.0, 100.0], [0.0, 100.0]])


def test_validation_rejects_non_obround_well() -> None:
    with pytest.raises(ValueError, match="obround"):
        _minimal_spec(
            well={
                "shape": "circle",
                "t_range": [14.0, 26.0],
                "half_width": 4.0,
                "depth": 4.0,
                "floor_t_range": [16.31, 23.69],
            }
        )


def test_validation_rejects_well_deeper_than_the_wall_can_reach() -> None:
    """The rasteriser saturates the well depth at half_width·tan(wall_angle)
    on the centreline. A deeper request used to pass validation, be recorded
    as asked, and be built shallower with no diagnostic (PR #64, Codex P1)."""
    well = {"shape": "obround", "t_range": [14.0, 26.0], "half_width": 0.5, "depth": 4.0}
    with pytest.raises(ValueError, match="well.depth"):
        _minimal_spec(well=well)
    # exactly at the reach is fine; a vertical wall has no reach limit
    reach = 0.5 * math.tan(math.radians(60.0))
    _minimal_spec(well={**well, "depth": reach})
    _minimal_spec(well={**well, "wall_angle_deg": 90.0})


def test_validation_rejects_island_steeper_than_ramp() -> None:
    with pytest.raises(ValueError, match="island.angle_deg"):
        _minimal_spec(
            island={
                "angle_deg": 45.0,
                "boundary_line": [[2.0, 40.0], [14.0, 10.0]],
                "end_dist": 14.0,
            }
        )


def test_builder_rejects_gate_wider_than_plate() -> None:
    with pytest.raises(ValueError, match="gate_exit_width"):
        build_profile_gate_geometry(_minimal_spec(), _plate(plate_w_mm=100.0))


def test_builder_rejects_pocket_overhanging_grid() -> None:
    # outer wall wider than the raster grid must raise, not silently truncate
    spec = _minimal_spec(outer_wall_line=[[0.0, 400.0], [24.0, 400.0]])
    with pytest.raises(ValueError, match="overhangs the grid"):
        build_profile_gate_geometry(spec, _plate())


# ----------------------- smoke / silhouette ------------------


def test_builds_with_demo_spec() -> None:
    g = build_profile_gate_geometry(_demo_spec(), _plate())
    assert g.mask.any()
    assert g.gates
    assert g.volume_cm3() > 0
    assert g.label == "profile_gate"


def test_gate_cells_inside_mask_and_thickness_zero_outside() -> None:
    g = build_profile_gate_geometry(_demo_spec(), _plate())
    for iy, ix in g.gates:
        assert g.mask[iy, ix]
    assert (g.thickness_mm[~g.mask] == 0.0).all()
    assert (g.thickness_mm[g.mask] > 0.0).all()


def test_land_band_has_uniform_land_depth() -> None:
    spec = _minimal_spec()
    g = build_profile_gate_geometry(spec, _plate(), cell_size_mm=0.5)
    yy, xx = _grid(g)
    y_pb = 5.0 + spec.t_max()
    t = y_pb - yy
    land_band = g.mask & (t > 0) & (t < spec.land.length)
    assert land_band.any()
    np.testing.assert_allclose(g.thickness_mm[land_band], spec.land.depth)


def test_ramp_column_matches_formula_and_caps() -> None:
    spec = _minimal_spec()
    g = build_profile_gate_geometry(spec, _plate(), cell_size_mm=0.5)
    yy, xx = _grid(g)
    y_pb = 5.0 + spec.t_max()
    t = y_pb - yy
    tan_ramp = math.tan(math.radians(spec.main_ramp.angle_deg))
    ramp_zone = g.mask & (t > spec.land.length) & (t < spec.t_max() - 0.5)
    expected = np.minimum(
        spec.land.depth + tan_ramp * (t[ramp_zone] - spec.land.length),
        spec.main_ramp.cap_depth,
    )
    np.testing.assert_allclose(g.thickness_mm[ramp_zone], expected, rtol=1e-9)
    # deep zone actually reaches the cap
    assert np.isclose(g.thickness_mm[g.mask & (t > spec.ramp_cap_t() + 1.0)].max(), 2.4)


def test_island_is_shallower_and_ends_at_end_dist() -> None:
    spec = _demo_spec()
    g = build_profile_gate_geometry(spec, _plate())
    yy, xx = _grid(g)
    y_pb = 5.0 + spec.t_max()
    t = y_pb - yy
    cx = 5.0 + 300.0 / 2.0
    wa = np.abs(xx - cx)
    tan_ramp = math.tan(math.radians(spec.main_ramp.angle_deg))
    tan_isl = math.tan(math.radians(spec.island.angle_deg))
    # deep inside the island (t=8, |w|<5): shallow island formula applies
    inside = g.mask & (np.abs(t - 8.0) < 0.6) & (wa < 5.0)
    assert inside.any()
    d_isl = spec.land.depth + tan_isl * (t[inside] - spec.land.length)
    np.testing.assert_allclose(g.thickness_mm[inside], d_isl, rtol=1e-9)
    assert (
        g.thickness_mm[inside] < spec.land.depth + tan_ramp * (t[inside] - spec.land.length)
    ).all()
    # just past end_dist on the centerline: back to the (capped) main ramp
    past = g.mask & (np.abs(t - 15.0) < 0.6) & (wa < 2.0)
    assert past.any()
    assert (g.thickness_mm[past] > 2.0).all()


def _weld_spec(**weld_overrides) -> GateProfileSpec:
    """Demo spec plus a welded dam over the downstream half of the island."""
    d = _demo_spec().to_dict()
    weld = {"t_range": [6.0, 14.0], "depth": 0.1}
    weld.update(weld_overrides)
    d["island"]["weld"] = weld
    return GateProfileSpec.from_dict(d)


def test_island_weld_overrides_depth_inside_band_only() -> None:
    spec = _weld_spec()
    g = build_profile_gate_geometry(spec, _plate(), cell_size_mm=0.5)
    yy, xx = _grid(g)
    t = 5.0 + spec.t_max() - yy
    wa = np.abs(xx - (5.0 + 150.0))
    tan_isl = math.tan(math.radians(spec.island.angle_deg))
    # inside the weld band: constant weld depth
    banded = g.mask & (np.abs(t - 10.0) < 0.3) & (wa < 5.0)
    assert banded.any()
    np.testing.assert_allclose(g.thickness_mm[banded], 0.1, rtol=1e-9)
    # upstream of the band but still on the island: untouched island ramp
    before = g.mask & (np.abs(t - 4.0) < 0.3) & (wa < 5.0)
    assert before.any()
    d_isl = spec.land.depth + tan_isl * (t[before] - spec.land.length)
    np.testing.assert_allclose(g.thickness_mm[before], d_isl, rtol=1e-9)


def test_island_weld_does_not_touch_the_main_ramp() -> None:
    """At the same t, cells outside the island boundary keep the main ramp."""
    spec = _weld_spec()
    g = build_profile_gate_geometry(spec, _plate(), cell_size_mm=0.5)
    base = build_profile_gate_geometry(_demo_spec(), _plate(), cell_size_mm=0.5)
    yy, xx = _grid(g)
    t = 5.0 + spec.t_max() - yy
    wa = np.abs(xx - (5.0 + 150.0))
    # island boundary at t=10 is w=20 → w=30 is main ramp, both before and after
    outside = g.mask & (np.abs(t - 10.0) < 0.3) & (np.abs(wa - 30.0) < 1.0)
    assert outside.any()
    np.testing.assert_allclose(g.thickness_mm[outside], base.thickness_mm[outside], rtol=1e-12)
    assert (g.thickness_mm[outside] > 0.5).all()


def test_island_weld_volume_drop_matches_quadrature() -> None:
    plate = _plate()
    base = build_profile_gate_geometry(_demo_spec(), plate, cell_size_mm=0.25)
    welded = build_profile_gate_geometry(_weld_spec(), plate, cell_size_mm=0.25)
    removed = (base.volume_cm3() - welded.volume_cm3()) * 1000.0
    # analytic: 2 * integral over the band of (island depth - weld depth) * half width
    tan_isl = math.tan(math.radians(2.5))
    tt = np.linspace(6.0, 14.0, 20001)
    half_w = 40.0 - 2.5 * (tt - 2.0)
    depth = 0.4 + tan_isl * (tt - 2.0)
    expected = np.trapezoid(2.0 * half_w * (depth - 0.1), tt)
    assert expected == pytest.approx(198.5, rel=1e-3)
    assert removed == pytest.approx(expected, rel=0.03)


def test_island_weld_to_the_pl_cuts_a_hole_but_not_through_the_well() -> None:
    """depth 0 = the dam reaches the PL. Those cells are steel: out of the
    mask, never a zero-thickness cavity cell (S = 0 → singular system). The
    well is machined through the dam, so where the well reaches, the cells
    stay cavity at the well's depth."""
    ref = build_profile_gate_geometry(_weld_spec(), _plate(), cell_size_mm=0.5)
    spec = _weld_spec(depth=0.0)
    g = build_profile_gate_geometry(spec, _plate(), cell_size_mm=0.5)
    yy, xx = _grid(g)
    t = 5.0 + spec.t_max() - yy
    wa = np.abs(xx - (5.0 + 150.0))
    banded = ref.mask & (np.abs(t - 10.0) < 0.3) & (wa < 5.0)
    assert banded.any()
    assert not g.mask[banded].any()
    assert (g.thickness_mm[g.mask] > 0).all()
    # outside the band nothing moved
    in_band = (t >= spec.island.weld.t_range[0]) & (t <= spec.island.weld.t_range[1])
    assert np.array_equal(g.mask & ~in_band, ref.mask & ~in_band)
    # the well still reaches its own depth inside the band
    well_t = (t >= spec.well.t_range[0]) & (t <= spec.well.t_range[1])
    on_axis = well_t & in_band & (wa < 0.3)
    if on_axis.any():
        assert g.mask[on_axis].all()
        np.testing.assert_allclose(g.thickness_mm[on_axis], ref.thickness_mm[on_axis])
    # the solver must still reach the plate around the hole
    solver = HeleShawSolver(
        geometry=g,
        material=MaterialDB()["PP"],
        melt_temperature_K=503.15,
        mold_temperature_K=313.15,
        injection_velocity_mms=100.0,
        injection_volume_flow_cm3s=20.0,
    )
    res = solver.solve(num_frames=4)
    assert np.isfinite(res.fill_time_s[g.mask]).all()


def test_island_weld_absent_is_byte_identical_to_legacy() -> None:
    plate = _plate()
    a = build_profile_gate_geometry(_demo_spec(), plate, cell_size_mm=0.5)
    d = _demo_spec().to_dict()
    assert "weld" not in d["island"]  # omitted, not serialized as null
    b = build_profile_gate_geometry(GateProfileSpec.from_dict(d), plate, cell_size_mm=0.5)
    np.testing.assert_array_equal(a.thickness_mm, b.thickness_mm)
    np.testing.assert_array_equal(a.mask, b.mask)


def test_island_weld_json_roundtrip() -> None:
    spec = _weld_spec()
    again = GateProfileSpec.from_json(spec.to_json())
    assert again.island is not None and again.island.weld is not None
    assert again.island.weld.t_range == (6.0, 14.0)
    assert again.island.weld.depth == 0.1


def test_island_weld_validation() -> None:
    # weld metal can only make the channel shallower
    with pytest.raises(ValueError, match="weld.depth"):
        _weld_spec(depth=0.6)
    with pytest.raises(ValueError, match="weld.depth"):
        _weld_spec(depth=-0.1)
    _weld_spec(depth=0.0)  # steel up to the PL: a hole, not an error
    # band must lie inside [land.length, end_dist]
    with pytest.raises(ValueError, match="weld.t_range"):
        _weld_spec(t_range=[6.0, 15.0])
    with pytest.raises(ValueError, match="weld.t_range"):
        _weld_spec(t_range=[1.0, 10.0])
    with pytest.raises(ValueError, match="weld.t_range"):
        _weld_spec(t_range=[10.0, 6.0])
    # typo protection reaches into the nested section
    d = _demo_spec().to_dict()
    d["island"]["weld"] = {"t_range": [6.0, 14.0], "depth": 0.1, "dpeth": 0.2}
    with pytest.raises(ValueError, match="island.weld"):
        GateProfileSpec.from_dict(d)
    d["island"]["weld"] = 0.1
    with pytest.raises(ValueError, match="weld"):
        GateProfileSpec.from_dict(d)


def test_outer_wall_excludes_cells() -> None:
    spec = _demo_spec()
    g = build_profile_gate_geometry(spec, _plate())
    yy, xx = _grid(g)
    y_pb = 5.0 + spec.t_max()
    t = y_pb - yy
    cx = 5.0 + 150.0
    wa = np.abs(xx - cx)
    # wall line [[2,100],[26,20]] → at t=14 the wall is at w=60; w=70 is outside
    # (well overhang is far from there: well is centered at w=0)
    outside = (np.abs(t - 14.0) < 0.6) & (np.abs(wa - 70.0) < 2.0)
    assert outside.any()
    assert not g.mask[outside].any()
    inside = (np.abs(t - 14.0) < 0.6) & (wa < 50.0)
    assert g.mask[inside].all()


def test_symmetric_field_is_mirror() -> None:
    g = build_profile_gate_geometry(_demo_spec(), _plate())
    np.testing.assert_allclose(g.thickness_mm, g.thickness_mm[:, ::-1])
    assert (g.mask == g.mask[:, ::-1]).all()


def test_asymmetric_builds_single_side_with_gates() -> None:
    d = _demo_spec().to_dict()
    d["symmetric"] = False
    spec = GateProfileSpec.from_dict(d)
    g = build_profile_gate_geometry(spec, _plate())
    assert g.gates  # defensive snap must fire even if orifice half-overlaps
    yy, xx = _grid(g)
    y_pb = 5.0 + spec.t_max()
    t = y_pb - yy
    x_edge = 5.0 + 150.0 - spec.gate_exit_width / 2.0
    # far side of the valve edge (beyond well overhang) has no gate cells
    far_left = (t > 0) & (xx < x_edge - spec.well.half_width - 1.0)
    assert not g.mask[far_left].any()
    # band interior is populated (wall line [[2,100],[26,20]] → w ≈ 88 at t=5)
    band = (np.abs(t - 5.0) < 0.6) & (xx > x_edge + 5.0) & (xx < x_edge + 80.0)
    assert g.mask[band].all()


# ----------------------- well ------------------


def test_well_reaches_full_depth_and_max_combination() -> None:
    spec = _demo_spec()
    g = build_profile_gate_geometry(spec, _plate())
    yy, xx = _grid(g)
    y_pb = 5.0 + spec.t_max()
    t = y_pb - yy
    cx = 5.0 + 150.0
    wa = np.abs(xx - cx)
    center = (np.abs(t - 20.0) < 0.6) & (wa < 1.0)
    assert center.any()
    np.testing.assert_allclose(g.thickness_mm[center], spec.well.depth)
    # well depth beats the capped ramp (max combination)
    assert spec.well.depth > spec.main_ramp.cap_depth
    assert g.thickness_mm[g.mask].max() == pytest.approx(spec.well.depth)


def test_well_volume_increment_matches_radial_quadrature() -> None:
    """Volume gained by adding a well to the minimal spec vs. an independent
    radial integration of the capsule depth profile over the flat 2.4 floor."""
    plate = _plate()
    well = {
        "shape": "obround",
        "t_range": [14.0, 26.0],
        "half_width": 4.0,
        "depth": 4.0,
        "floor_t_range": [16.31, 23.69],
        "wall_angle_deg": 60,
    }
    # extend the straight wall to t=26 so t_max is identical with and
    # without the well — the volume delta is then the well alone
    wall = [[0.0, 100.0], [26.0, 100.0]]
    g_no = build_profile_gate_geometry(
        _minimal_spec(outer_wall_line=wall), plate, cell_size_mm=0.25
    )
    g_yes = build_profile_gate_geometry(
        _minimal_spec(outer_wall_line=wall, well=well), plate, cell_size_mm=0.25
    )
    dv_mm3 = (g_yes.volume_cm3() - g_no.volume_cm3()) * 1000.0

    # radial quadrature: capsule area element (2L + 2πr) dr, depth gain
    # max(min((hw − r)·tan60°, depth) − 2.4, 0) over the capped-ramp floor
    hw, depth, tan_wall = 4.0, 4.0, math.tan(math.radians(60.0))
    axis_len = (26.0 - hw) - (14.0 + hw)
    r = np.linspace(0.0, hw, 20001)
    gain = np.maximum(np.minimum((hw - r) * tan_wall, depth) - 2.4, 0.0)
    expected_mm3 = np.trapezoid(gain * (2 * axis_len + 2 * np.pi * r), r)
    assert dv_mm3 == pytest.approx(expected_mm3, rel=0.05)


# ----------------------- volume ------------------


def test_volume_minimal_spec_closed_form() -> None:
    """Land + ramp + capped flat with straight walls has a closed-form volume."""
    spec = _minimal_spec()
    plate = _plate()
    g = build_profile_gate_geometry(spec, plate, cell_size_mm=0.5)

    tan_ramp = math.tan(math.radians(10.0))
    t_cap = 2.0 + (2.4 - 0.4) / tan_ramp
    section = (
        0.4 * 2.0  # land
        + 0.5 * (0.4 + 2.4) * (t_cap - 2.0)  # ramp (trapezoid)
        + 2.4 * (24.0 - t_cap)  # capped flat
    )
    gate_mm3 = section * 200.0
    plate_mm3 = 300.0 * 50.0 * 0.4
    expected_cm3 = (gate_mm3 + plate_mm3) / 1000.0
    assert g.volume_cm3() == pytest.approx(expected_cm3, rel=0.03)


# ----------------------- valve / compression / plate ------------------


def test_valve_orifice_covers_a_small_cell_cluster() -> None:
    spec = _demo_spec()
    g = build_profile_gate_geometry(spec, _plate())
    assert 4 <= len(g.gates) <= 12  # Φ3 on a 1 mm grid ≈ π·1.5² ≈ 7 cells
    yy, xx = _grid(g)
    y_valve = 5.0 + spec.t_max() - spec.valve.t
    cx = 5.0 + 150.0
    for iy, ix in g.gates:
        rr = math.hypot(xx[iy, ix] - cx, yy[iy, ix] - y_valve)
        assert rr <= spec.valve.orifice_diameter / 2.0 + g.cell_size_mm


def test_compression_mask_is_plate_only() -> None:
    spec = _demo_spec()
    g = build_profile_gate_geometry(spec, _plate())
    assert g.compression_mask is not None
    assert g.compression_mask.any()
    yy, _ = _grid(g)
    y_pb = 5.0 + spec.t_max()
    assert not g.compression_mask[yy < y_pb].any()
    # every compression cell is a plate-thickness cell inside the mask
    assert (g.mask[g.compression_mask]).all()
    np.testing.assert_allclose(g.thickness_mm[g.compression_mask], 0.4)


def test_plate_split_two_bands() -> None:
    spec = _demo_spec()
    plate = _plate(plate_split_height_mm=20.0, plate_lower_thk_mm=0.35, plate_upper_thk_mm=0.50)
    g = build_profile_gate_geometry(spec, plate)
    yy, _ = _grid(g)
    y_pb = 5.0 + spec.t_max()
    lower = g.compression_mask & (yy > y_pb + 1.0) & (yy < y_pb + 19.0)
    upper = g.compression_mask & (yy > y_pb + 21.0)
    assert lower.any() and upper.any()
    np.testing.assert_allclose(g.thickness_mm[lower], 0.35)
    np.testing.assert_allclose(g.thickness_mm[upper], 0.50)


# ----------------------- solver integration ------------------


def test_solver_runs_on_profile_gate() -> None:
    g = build_profile_gate_geometry(_demo_spec(), _plate(), cell_size_mm=2.0)
    db = MaterialDB()
    solver = HeleShawSolver(
        geometry=g,
        material=db["PP"],
        melt_temperature_K=503.15,
        mold_temperature_K=313.15,
        injection_velocity_mms=100.0,
        injection_volume_flow_cm3s=20.0,
    )
    res = solver.solve(num_frames=8)
    assert res.total_fill_time_s > 0
    assert np.isfinite(res.fill_time_s[g.mask]).all()


# ----------------------- gate land wall vs mesh (Issue #58) ------------------


def test_exit_width_below_the_mesh_spacing_is_rejected_not_severed() -> None:
    """A gate exit narrower than the mesh must not silently sever the plate.

    At ``gate_exit_width=0.5`` and the default ``cell_size_mm=1.0`` the land
    wall used to close every column of the plate-bottom row, cutting the
    whole plate off from the gate block; the solver then filled the orphaned
    plate with a plausible-looking uniform fill time and no visible cue that
    it was garbage (Issue #58). The builder now rejects the combination and
    names the knob. Asserted on the demo spec so the reproduction stays the
    one actually observed, not a synthetic corner.
    """
    d = _demo_spec().to_dict()
    d["gate_exit_width"] = 0.5
    narrow = GateProfileSpec.from_dict(d)
    with pytest.raises(ValueError, match="gate_exit_width"):
        build_profile_gate_geometry(narrow, _plate(), cell_size_mm=1.0)


def test_exit_width_wider_than_the_mesh_still_builds_connected() -> None:
    """The rejection must not fire for a healthy exit width.

    Also asserts the built mask is one connected component, because that is
    the property the rejection exists to protect -- a builder change that
    kept the error message but started producing severed masks elsewhere
    would still be caught here.
    """
    import scipy.ndimage as ndi

    g = build_profile_gate_geometry(_demo_spec(), _plate(), cell_size_mm=1.0)
    _labels, n = ndi.label(g.mask)
    assert n == 1


# -------------------------- display origin --------------------------


def test_display_origin_x_is_the_nominal_valve_axis_not_the_cell_centroid() -> None:
    """An asymmetric pocket clips the orifice at its w = 0 edge, so the
    surviving gate cells all sit on one side and their centroid drifts off
    the valve axis by a mesh-dependent amount. x = 0 must stay on the
    nominal axis (Codex P2, PR #76)."""
    spec = _minimal_spec(symmetric=False, gate_exit_width=100.0)
    g = build_profile_gate_geometry(spec, _plate())
    x_edge = 5.0 + 150.0 - spec.gate_exit_width / 2.0  # valve axis (w = 0)
    x0, _y0 = g.display_origin_mm()
    assert x0 == pytest.approx(x_edge)
    centroid = float(np.mean([(ix + 0.5) * g.cell_size_mm for _iy, ix in g.gates]))
    assert centroid > x_edge + 0.25 * g.cell_size_mm  # the clipped centroid drifts


def test_gate_marker_is_the_nominal_orifice_even_when_the_mask_clips_it() -> None:
    """Same clipped orifice as above: the surviving Dirichlet cells are a
    half-disc on one side of the valve axis, so a marker rebuilt from them
    would be a shifted, undersized semicircle. The result maps must draw the
    configured Φ at the configured center (Codex P2, PR #80)."""
    from core.visualizer import gate_groups_mm

    spec = _minimal_spec(symmetric=False, gate_exit_width=100.0)
    g = build_profile_gate_geometry(spec, _plate())
    assert g.valve_marker_mm is not None
    groups = gate_groups_mm(g)
    assert len(groups) == 1
    gx, gy, gr = groups[0]
    assert gx == pytest.approx(0.0)  # on the valve axis in the display frame
    assert gr == pytest.approx(spec.valve.orifice_diameter / 2.0)
    _x0, y0 = g.display_origin_mm()
    assert gy + y0 == pytest.approx(g.valve_marker_mm[1])
    # Without the nominal record the raster would have told a different story.
    g.valve_marker_mm = None
    rx, _ry, rr = gate_groups_mm(g)[0]
    assert rx > 0.25 * g.cell_size_mm  # centroid drifted to the surviving side
    assert rr < 0.85 * spec.valve.orifice_diameter / 2.0  # half the cells, smaller radius


def test_display_origin_x_follows_a_symmetric_valve_offset() -> None:
    """symmetric spec with valve.w != 0: x = 0 on the offset valve axis."""
    spec = _minimal_spec()
    d = spec.to_dict()
    d["valve"]["w"] = 10.0
    spec = GateProfileSpec.from_dict(d)
    g = build_profile_gate_geometry(spec, _plate())
    x0, _y0 = g.display_origin_mm()
    assert x0 == pytest.approx(5.0 + 150.0 + 10.0)


# -------------------------- edge channels (縁部深彫り) --------------------------
#
# A band of pocket cells within ``width`` (perpendicular distance) of the
# outer wall gets ``d = max(d, depth)``. The minimal spec's wall is the
# vertical line w = 100 over the whole block, so the band geometry has a
# closed form: a w ∈ [100 − width, 100] rectangle over t_range plus a
# quarter-disc at each end (the wall side of the end circles is steel).


def _ec_dict(**overrides) -> dict:
    ec = {"width": 3.0, "depth": 4.0, "t_range": [17.0, 21.0]}
    ec.update(overrides)
    return ec


def test_edge_channel_deepens_the_band_only_and_mirrors() -> None:
    base = _minimal_spec()
    spec = _minimal_spec(edge_channels=[_ec_dict()])
    g0 = build_profile_gate_geometry(base, _plate())
    g1 = build_profile_gate_geometry(spec, _plate())
    # silhouette untouched: the band only deepens existing cavity cells
    assert np.array_equal(g0.mask, g1.mask)
    diff = g1.thickness_mm != g0.thickness_mm
    assert diff.any()
    yy, xx = _grid(g1)
    wa = np.abs(xx - (5.0 + 150.0))  # symmetric: |x − x_valve|
    # every changed cell hugs the wall (w = 100) within width + a cell slop
    assert wa[diff].min() >= 100.0 - 3.0 - g1.cell_size_mm
    assert wa[diff].max() <= 100.0
    # changed cells get exactly the channel depth (the base there is capped
    # at 2.4 < 4.0, so the floor always wins inside the band)
    assert np.allclose(g1.thickness_mm[diff], 4.0)
    # symmetric: the band stands on both mirrored edges
    nx = diff.shape[1]
    assert diff[:, : nx // 2].sum() == diff[:, (nx + 1) // 2 :].sum() > 0


def test_edge_channel_volume_increment_matches_closed_form() -> None:
    """t_range (17, 21) keeps band + end discs inside the capped-ramp zone
    (t ∈ [14, 24], base depth 2.4 everywhere), so ΔV is exact:
    ΔV = (depth − cap) · (width·len + 2·(π/4)·width²) · 2 sides."""
    base = _minimal_spec()
    spec = _minimal_spec(edge_channels=[_ec_dict()])
    g0 = build_profile_gate_geometry(base, _plate(), cell_size_mm=0.25)
    g1 = build_profile_gate_geometry(spec, _plate(), cell_size_mm=0.25)
    dv_mm3 = (g1.volume_cm3() - g0.volume_cm3()) * 1000.0
    expected = (4.0 - 2.4) * (3.0 * 4.0 + 2.0 * (math.pi / 4.0) * 3.0**2) * 2.0
    assert dv_mm3 == pytest.approx(expected, rel=0.03)


def test_edge_channel_depth_is_a_floor_not_an_override() -> None:
    """A channel shallower than the local depth changes nothing there."""
    base = _minimal_spec()
    spec = _minimal_spec(edge_channels=[_ec_dict(depth=1.0)])  # < cap 2.4
    g0 = build_profile_gate_geometry(base, _plate())
    g1 = build_profile_gate_geometry(spec, _plate())
    assert np.array_equal(g0.thickness_mm, g1.thickness_mm)


def test_edge_channel_default_t_range_spans_the_wall() -> None:
    base = _minimal_spec()
    spec = _minimal_spec(edge_channels=[_ec_dict(t_range=None)])
    g0 = build_profile_gate_geometry(base, _plate())
    g1 = build_profile_gate_geometry(spec, _plate())
    diff = g1.thickness_mm != g0.thickness_mm
    rows = np.where(diff.any(axis=1))[0]
    yy, _xx = _grid(g1)
    t = (5.0 + 24.0) - yy[:, 0]  # y_plate_bottom − y
    ts = t[rows]
    assert ts.min() < 1.0  # reaches the gate exit end of the wall
    assert ts.max() > 23.0  # ... and the far end (t_max = 24)


def test_edge_channel_respects_t_range() -> None:
    base = _minimal_spec()
    spec = _minimal_spec(edge_channels=[_ec_dict()])
    g0 = build_profile_gate_geometry(base, _plate())
    g1 = build_profile_gate_geometry(spec, _plate())
    diff = g1.thickness_mm != g0.thickness_mm
    yy, _xx = _grid(g1)
    t = (5.0 + 24.0) - yy[:, 0]
    ts = t[np.where(diff.any(axis=1))[0]]
    # band + end discs live in t ∈ [14, 24]; nothing changes before that
    assert ts.min() >= 14.0 - g1.cell_size_mm


def test_edge_channel_asymmetric_builds_a_single_band() -> None:
    base = _minimal_spec(symmetric=False, gate_exit_width=100.0)
    spec = _minimal_spec(symmetric=False, gate_exit_width=100.0, edge_channels=[_ec_dict()])
    g0 = build_profile_gate_geometry(base, _plate())
    g1 = build_profile_gate_geometry(spec, _plate())
    diff = g1.thickness_mm != g0.thickness_mm
    assert diff.any()
    # one contiguous column cluster, not a mirrored pair
    cols = np.where(diff.any(axis=0))[0]
    assert cols.max() - cols.min() + 1 == cols.size


def test_edge_channel_json_roundtrip_and_default_omitted() -> None:
    spec = _minimal_spec(edge_channels=[_ec_dict(), _ec_dict(t_range=None, width=1.5)])
    again = GateProfileSpec.from_json(spec.to_json())
    assert again == spec
    # the package facade re-exports the spec type like its siblings (Codex P2)
    assert again.edge_channels[0] == EdgeChannelSpec(width=3.0, depth=4.0, t_range=(17.0, 21.0))
    assert again.edge_channels[0].side == "outer"  # side omitted → default
    assert again.edge_channels[1].t_range is None
    # absent in the JSON → empty tuple, and to_dict leaves the key out
    legacy = _minimal_spec()
    assert legacy.edge_channels == ()
    assert "edge_channels" not in legacy.to_dict()
    assert "edge_channels" not in _demo_spec().to_dict()


def test_edge_channel_validation() -> None:
    with pytest.raises(ValueError, match="side"):
        _minimal_spec(edge_channels=[_ec_dict(side="inner")])  # no inner wall here
    with pytest.raises(ValueError, match="positive"):
        _minimal_spec(edge_channels=[_ec_dict(width=0.0)])
    with pytest.raises(ValueError, match="positive"):
        _minimal_spec(edge_channels=[_ec_dict(depth=-1.0)])
    with pytest.raises(ValueError, match="increasing"):
        _minimal_spec(edge_channels=[_ec_dict(t_range=[21.0, 17.0])])
    with pytest.raises(ValueError, match="t_max"):
        _minimal_spec(edge_channels=[_ec_dict(t_range=[17.0, 999.0])])
    with pytest.raises(ValueError, match="unknown key"):
        _minimal_spec(edge_channels=[{**_ec_dict(), "bogus": 1}])
    with pytest.raises(ValueError, match="must be a list"):
        _minimal_spec(edge_channels="3")
    with pytest.raises(ValueError, match="must be an object"):
        _minimal_spec(edge_channels=[3.0])


def test_edge_channel_zero_cell_raster_is_rejected() -> None:
    """A band that misses every cell centre must fail loudly: the spec would
    be recorded as asked while the built geometry silently lacks the
    feature (same false-green class as the sub-mesh runner)."""
    spec = _minimal_spec(edge_channels=[_ec_dict(width=0.01, t_range=[20.0, 20.05])])
    with pytest.raises(ValueError, match="edge_channels\\[0\\].*zero cells"):
        build_profile_gate_geometry(spec, _plate(), cell_size_mm=1.0)
