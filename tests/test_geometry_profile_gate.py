"""Tests for the JSON-spec-driven profile-gate builder (`core/profile_gate.py`)."""

from __future__ import annotations

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


def test_from_dict_rejects_non_mm_units() -> None:
    with pytest.raises(ValueError, match="units"):
        GateProfileSpec.from_dict(_minimal_spec_dict(units="inch"))


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
