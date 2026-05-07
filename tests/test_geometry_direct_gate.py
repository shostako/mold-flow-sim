"""Tests for the parametric direct-gate geometry builder.

The direct-gate cavity is a single rectangular plate with a circular
Dirichlet τ=0 patch (Φ = ``gate_diameter_mm``) placed **inside** the
plate, on its longitudinal centerline, ``gate_offset_mm`` inward from
the gate-side edge. There is no runner, no sprue strip — molten resin
enters vertically through the patch.
"""

from __future__ import annotations

import numpy as np
import pytest

from core import (
    DirectGateConfig,
    HeleShawSolver,
    MaterialDB,
    build_direct_gate_geometry,
)


def _default_cfg(**overrides) -> DirectGateConfig:
    base = dict(
        plate_w_mm=120.0,
        plate_h_mm=80.0,
        plate_thk_mm=2.0,
        gate_diameter_mm=3.0,
        gate_offset_mm=20.0,
        cell_size_mm=1.0,
        pad_mm=5.0,
    )
    base.update(overrides)
    return DirectGateConfig(**base)


# -------------------------- silhouette --------------------------


def test_direct_gate_builds_with_defaults() -> None:
    g = build_direct_gate_geometry(_default_cfg())
    assert g.mask.any()
    assert g.gates, "direct gate must produce at least one Dirichlet cell"
    assert g.volume_cm3() > 0
    assert g.label == "direct_gate"


def test_silhouette_is_rectangular_plate_only() -> None:
    """No runner, no sprue strip — every live cell must lie inside the
    rectangle [pad, pad+Wp] × [pad, pad+Hp]."""
    cfg = _default_cfg()
    g = build_direct_gate_geometry(cfg)
    iy_idx, ix_idx = np.indices(g.mask.shape)
    yy = (iy_idx + 0.5) * cfg.cell_size_mm
    xx = (ix_idx + 0.5) * cfg.cell_size_mm
    in_plate = (
        (yy >= cfg.pad_mm)
        & (yy <= cfg.pad_mm + cfg.plate_h_mm)
        & (xx >= cfg.pad_mm)
        & (xx <= cfg.pad_mm + cfg.plate_w_mm)
    )
    # Every cavity cell must be inside the plate rectangle
    assert np.all(g.mask <= in_plate)


def test_thickness_is_uniform_plate() -> None:
    """No connector, no sprue → every live cell carries plate_thk_mm."""
    cfg = _default_cfg()
    g = build_direct_gate_geometry(cfg)
    np.testing.assert_allclose(g.thickness_mm[g.mask], cfg.plate_thk_mm, atol=1e-9)
    assert np.all(g.thickness_mm[~g.mask] == 0.0)


def test_all_gate_cells_lie_inside_mask() -> None:
    g = build_direct_gate_geometry(_default_cfg())
    for iy, ix in g.gates:
        assert g.mask[iy, ix], f"gate ({iy},{ix}) is outside mask"


def test_gate_is_inside_plate_at_specified_offset() -> None:
    """Gate-disk center y-coord should sit ``gate_offset_mm`` inside the
    plate, measured from the gate-side edge."""
    cfg = _default_cfg(cell_size_mm=0.5)
    g = build_direct_gate_geometry(cfg)
    # Gate-side edge is the lowest plate row
    plate_w_cells = int(round(cfg.plate_w_mm / cfg.cell_size_mm))
    plate_rows = np.where(g.mask.sum(axis=1) >= plate_w_cells)[0]
    iy_gate_side_edge = int(plate_rows[0])
    gate_iys = np.array([iy for iy, _ in g.gates])
    iy_gate_center = float(gate_iys.mean())
    distance_mm = (iy_gate_center - iy_gate_side_edge) * cfg.cell_size_mm
    # Allow up to one cell pixelation slack
    assert abs(distance_mm - cfg.gate_offset_mm) <= 1.5 * cfg.cell_size_mm


def test_gate_is_horizontally_centered() -> None:
    cfg = _default_cfg(cell_size_mm=0.5)
    g = build_direct_gate_geometry(cfg)
    cx_mm_expected = cfg.pad_mm + cfg.plate_w_mm / 2.0
    gate_ixs = np.array([ix for _, ix in g.gates])
    gate_x_mean = (float(gate_ixs.mean()) + 0.5) * cfg.cell_size_mm
    assert abs(gate_x_mean - cx_mm_expected) <= 1.5 * cfg.cell_size_mm


def test_gate_diameter_matches_request() -> None:
    """The Dirichlet patch should be roughly Φ ``gate_diameter_mm`` wide."""
    cfg = _default_cfg(gate_diameter_mm=3.0, cell_size_mm=0.5)
    g = build_direct_gate_geometry(cfg)
    gate_xs = np.array([ix for _, ix in g.gates])
    gate_ys = np.array([iy for iy, _ in g.gates])
    width_mm = (gate_xs.max() - gate_xs.min() + 1) * cfg.cell_size_mm
    height_mm = (gate_ys.max() - gate_ys.min() + 1) * cfg.cell_size_mm
    slack = 2 * cfg.cell_size_mm
    assert abs(width_mm - cfg.gate_diameter_mm) <= slack
    assert abs(height_mm - cfg.gate_diameter_mm) <= slack


def test_volume_matches_plate_volume() -> None:
    """Cavity volume = plate_w × plate_h × plate_thk (no runners/sprue)."""
    cfg = _default_cfg(cell_size_mm=0.5)
    g = build_direct_gate_geometry(cfg)
    expected_cm3 = cfg.plate_w_mm * cfg.plate_h_mm * cfg.plate_thk_mm / 1000.0
    # Cell-discretization tolerance (the rectangle aligns with the grid so
    # this should be tight)
    assert abs(g.volume_cm3() - expected_cm3) < 0.1


# -------------------------- compression mask --------------------------


def test_compression_mask_covers_entire_plate() -> None:
    """The plate is the entire cavity — all cells should inflate."""
    cfg = _default_cfg()
    g = build_direct_gate_geometry(cfg)
    cm = g.compression_mask
    assert cm is not None
    # Every cavity cell is in the compression mask, and vice versa
    np.testing.assert_array_equal(cm, g.mask)


def test_compression_volume_fraction_is_one() -> None:
    cfg = _default_cfg()
    g = build_direct_gate_geometry(cfg)
    assert abs(g.compression_volume_fraction() - 1.0) < 1e-9


# -------------------------- validation --------------------------


def test_validation_rejects_oversized_gate() -> None:
    with pytest.raises(ValueError, match="gate_diameter_mm"):
        DirectGateConfig(
            plate_w_mm=20.0,
            plate_h_mm=80.0,
            plate_thk_mm=2.0,
            gate_diameter_mm=30.0,
        ).validate()


def test_validation_rejects_negative_offset() -> None:
    with pytest.raises(ValueError, match="gate_offset_mm"):
        DirectGateConfig(
            plate_w_mm=120.0,
            plate_h_mm=80.0,
            plate_thk_mm=2.0,
            gate_offset_mm=-5.0,
        ).validate()


def test_validation_rejects_zero_offset() -> None:
    """Zero offset would put the gate disk on the gate-side edge,
    poking outside the plate. Must be rejected."""
    with pytest.raises(ValueError, match="gate_offset_mm"):
        DirectGateConfig(
            plate_w_mm=120.0,
            plate_h_mm=80.0,
            plate_thk_mm=2.0,
            gate_offset_mm=0.0,
        ).validate()


def test_validation_rejects_offset_too_close_to_edge() -> None:
    """Offset smaller than gate radius → disk pokes past gate-side edge."""
    with pytest.raises(ValueError, match="gate-side edge"):
        DirectGateConfig(
            plate_w_mm=120.0,
            plate_h_mm=80.0,
            plate_thk_mm=2.0,
            gate_diameter_mm=10.0,
            gate_offset_mm=2.0,  # < radius 5.0
        ).validate()


def test_validation_rejects_offset_past_far_edge() -> None:
    """Offset + radius > plate height → disk pokes past far edge."""
    with pytest.raises(ValueError, match="far edge"):
        DirectGateConfig(
            plate_w_mm=120.0,
            plate_h_mm=20.0,
            plate_thk_mm=2.0,
            gate_diameter_mm=4.0,
            gate_offset_mm=19.0,  # 19 + 2 = 21 > 20
        ).validate()


def test_validation_rejects_zero_thickness() -> None:
    with pytest.raises(ValueError, match="plate_thk_mm"):
        DirectGateConfig(
            plate_w_mm=120.0,
            plate_h_mm=80.0,
            plate_thk_mm=0.0,
        ).validate()


# -------------------------- solver integration --------------------------


def test_solver_runs_on_direct_gate_geometry() -> None:
    g = build_direct_gate_geometry(_default_cfg())
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
    msk = ~np.isnan(res.fill_time_s)
    assert msk.any()
    assert float(np.nanmin(res.fill_time_s[msk])) >= 0.0
    assert float(np.nanmax(res.fill_time_s[msk])) > 0.0


def test_compression_shortens_fill_time() -> None:
    """With the entire plate in the compression mask, ICM ON should match
    the legacy whole-cavity inflation behaviour: T_fill shrinks by
    ``compression_fraction / compression_factor + (1 - compression_fraction)``.
    """
    g = build_direct_gate_geometry(_default_cfg())
    db = MaterialDB()
    common = dict(
        material=db["PP"],
        melt_temperature_K=503.15,
        mold_temperature_K=313.15,
        injection_velocity_mms=100.0,
        injection_volume_flow_cm3s=20.0,
    )
    solver_off = HeleShawSolver(geometry=g, **common)
    solver_on = HeleShawSolver(
        geometry=g,
        compression_molding=True,
        compression_factor=1.8,
        compression_fraction=0.7,
        **common,
    )
    res_off = solver_off.solve(num_frames=4)
    res_on = solver_on.solve(num_frames=4)
    assert res_on.total_fill_time_s < res_off.total_fill_time_s
    # f_comp = 1.0 (entire plate in mask) → exact legacy ratio applies
    expected_ratio = 0.7 / 1.8 + 0.3
    actual_ratio = res_on.total_fill_time_s / res_off.total_fill_time_s
    assert abs(actual_ratio - expected_ratio) < 0.05
