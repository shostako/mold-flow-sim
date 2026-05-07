"""Tests for the parametric direct-gate geometry builder.

The direct-gate cavity is a rectangular plate fed from a circular
Dirichlet patch (Φ = ``gate_diameter_mm``) sitting ``gate_offset_mm``
below the plate's gate-side edge, connected by a thin sprue strip of
the same diameter.
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


def test_all_gate_cells_lie_inside_mask() -> None:
    g = build_direct_gate_geometry(_default_cfg())
    for iy, ix in g.gates:
        assert g.mask[iy, ix], f"gate ({iy},{ix}) is outside mask"


def test_thickness_is_zero_outside_mask() -> None:
    g = build_direct_gate_geometry(_default_cfg())
    assert np.all(g.thickness_mm[~g.mask] == 0.0)


def test_gate_diameter_matches_request() -> None:
    """The Dirichlet patch should be roughly Φ ``gate_diameter_mm`` wide."""
    cfg = _default_cfg(gate_diameter_mm=3.0, cell_size_mm=0.5)
    g = build_direct_gate_geometry(cfg)
    gate_xs = np.array([ix for _, ix in g.gates])
    gate_ys = np.array([iy for iy, _ in g.gates])
    # Bounding box of the gate cells in mm
    width_mm = (gate_xs.max() - gate_xs.min() + 1) * cfg.cell_size_mm
    height_mm = (gate_ys.max() - gate_ys.min() + 1) * cfg.cell_size_mm
    # Allow up to 1 cell of pixelation slack on each side
    slack = 2 * cfg.cell_size_mm
    assert abs(width_mm - cfg.gate_diameter_mm) <= slack
    assert abs(height_mm - cfg.gate_diameter_mm) <= slack


def test_gate_is_below_plate_at_specified_offset() -> None:
    """Gate-circle center y-coord should sit ``gate_offset_mm`` below the
    plate bottom row."""
    cfg = _default_cfg(cell_size_mm=0.5)
    g = build_direct_gate_geometry(cfg)
    # Plate rows: any row whose mask spans the full plate width
    # Plate rows = rows where the full plate width is live (connector is
    # only Φ wide, so it never reaches plate_w mask cells).
    plate_w_cells = int(round(cfg.plate_w_mm / cfg.cell_size_mm))
    plate_rows = np.where(g.mask.sum(axis=1) >= plate_w_cells)[0]
    iy_plate_bottom = int(plate_rows[0])
    # Gate row centroid
    gate_iys = np.array([iy for iy, _ in g.gates])
    iy_gate_center = float(gate_iys.mean())
    distance_mm = (iy_plate_bottom - iy_gate_center) * cfg.cell_size_mm
    # Allow up to one cell pixelation slack
    assert abs(distance_mm - cfg.gate_offset_mm) <= 1.5 * cfg.cell_size_mm


def test_connector_strip_is_narrow() -> None:
    """Between the gate and the plate, only a Φ ``gate_diameter_mm`` wide
    strip should be live."""
    cfg = _default_cfg(cell_size_mm=0.5)
    g = build_direct_gate_geometry(cfg)
    # Pick a row in the middle of the connector area: half the offset above
    # the gate center.
    plate_w_cells = int(round(cfg.plate_w_mm / cfg.cell_size_mm))
    plate_rows = np.where(g.mask.sum(axis=1) >= plate_w_cells)[0]
    iy_plate_bottom = int(plate_rows[0])
    gate_iys = np.array([iy for iy, _ in g.gates])
    iy_gate_center = int(round(gate_iys.mean()))
    iy_mid = (iy_plate_bottom + iy_gate_center) // 2
    if iy_mid == iy_plate_bottom or iy_mid == iy_gate_center:
        pytest.skip("offset too small to sample a midline row")
    live_x = np.where(g.mask[iy_mid, :])[0]
    width_mm = (live_x.max() - live_x.min() + 1) * cfg.cell_size_mm
    slack = 2 * cfg.cell_size_mm
    assert abs(width_mm - cfg.gate_diameter_mm) <= slack


# -------------------------- thickness --------------------------


def test_plate_uses_plate_thickness() -> None:
    cfg = _default_cfg()
    g = build_direct_gate_geometry(cfg)
    plate_h = g.thickness_mm[g.thickness_mm > 0]
    assert plate_h.min() >= min(cfg.plate_thk_mm, cfg.resolved_sprue_thk_mm()) - 1e-9
    assert plate_h.max() <= max(cfg.plate_thk_mm, cfg.resolved_sprue_thk_mm()) + 1e-9


def test_sprue_thickness_override() -> None:
    """``sprue_thk_mm`` should change the connector / gate-disk thickness
    without touching the plate."""
    cfg = _default_cfg(sprue_thk_mm=4.0)
    g = build_direct_gate_geometry(cfg)
    # The plate is the largest connected region with thickness == plate_thk_mm
    assert (np.isclose(g.thickness_mm, cfg.plate_thk_mm) & g.mask).sum() > 0
    # Some cells should sit at the override thickness
    assert (np.isclose(g.thickness_mm, 4.0) & g.mask).sum() > 0


# -------------------------- compression mask --------------------------


def test_compression_mask_excludes_sprue_and_gate() -> None:
    """Only the rectangular plate body should inflate during compression."""
    cfg = _default_cfg()
    g = build_direct_gate_geometry(cfg)
    cm = g.compression_mask
    assert cm is not None
    # All compression-mask cells must be inside the plate (thickness =
    # plate_thk_mm). No connector / gate cell should be flagged.
    assert np.all(np.isclose(g.thickness_mm[cm & g.mask], cfg.plate_thk_mm))
    # The connector + gate area should NOT be in the compression mask
    connector_or_gate = g.mask & ~cm
    # When sprue_thk_mm == plate_thk_mm by default, both regions share the
    # same thickness — but the mask should still exclude the lower part.
    assert connector_or_gate.any(), "compression mask should exclude sprue+gate"


def test_compression_volume_fraction_reasonable() -> None:
    """The plate body volume should dominate the cavity volume, but the
    connector + gate disk should still account for a non-trivial share
    when the sprue is thicker than the plate."""
    cfg = _default_cfg(sprue_thk_mm=4.0)
    g = build_direct_gate_geometry(cfg)
    f = g.compression_volume_fraction()
    assert 0.0 < f < 1.0
    # Expected share: V_plate / (V_plate + V_sprue+gate)
    plate_vol = float(np.sum(g.thickness_mm[g.mask & g.compression_mask]))
    total_vol = float(np.sum(g.thickness_mm[g.mask]))
    assert abs(f - plate_vol / total_vol) < 1e-9


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


def test_validation_rejects_zero_thickness() -> None:
    with pytest.raises(ValueError, match="plate_thk_mm"):
        DirectGateConfig(
            plate_w_mm=120.0,
            plate_h_mm=80.0,
            plate_thk_mm=0.0,
        ).validate()


def test_validation_rejects_zero_sprue_thickness() -> None:
    with pytest.raises(ValueError, match="sprue_thk_mm"):
        DirectGateConfig(
            plate_w_mm=120.0,
            plate_h_mm=80.0,
            plate_thk_mm=2.0,
            sprue_thk_mm=0.0,
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
    # All non-gate cells should have positive fill time
    assert float(np.nanmin(res.fill_time_s[msk])) >= 0.0
    assert float(np.nanmax(res.fill_time_s[msk])) > 0.0


def test_compression_only_inflates_plate_body() -> None:
    """T_fill scaling should be diluted by the plate's volume share, not by
    the full ``compression_factor`` as in legacy whole-cavity inflation."""
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
    # Compression should still shorten fill time, but less than the
    # whole-cavity proxy (cf = 1.8 * 0.7 + 0.3 = 0.6×). Plate volume share
    # is around 0.96 here, so the dilution is small but measurable.
    assert res_on.total_fill_time_s < res_off.total_fill_time_s
    # Ratio bounded between (whole-cavity short proxy) and 1.0
    naive_ratio = 0.7 / 1.8 + 0.3
    actual_ratio = res_on.total_fill_time_s / res_off.total_fill_time_s
    assert naive_ratio < actual_ratio <= 1.0
