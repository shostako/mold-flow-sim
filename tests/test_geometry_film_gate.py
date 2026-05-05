"""Tests for the parametric film-gate geometry builder."""

from __future__ import annotations

import numpy as np
import pytest

from core import FilmGateConfig, HeleShawSolver, MaterialDB, build_film_gate_geometry


def _default_cfg(**overrides) -> FilmGateConfig:
    base = dict(
        plate_w_mm=120.0,
        plate_h_mm=80.0,
        plate_thk_mm=2.0,
        runner_long_mm=80.0,
        runner_short_diameter_mm=12.0,
        runner_depth_mm=20.0,
        runner_thk_mm=4.0,
        runner_flat_depth_mm=8.0,
        runner_slope_depth_mm=12.0,
        valve_gate_diameter_mm=4.0,
        gate_width_mm=60.0,
        cell_size_mm=1.0,
        pad_mm=5.0,
    )
    base.update(overrides)
    return FilmGateConfig(**base)


def test_film_gate_builds_with_defaults() -> None:
    g = build_film_gate_geometry(_default_cfg())
    assert g.mask.any()
    assert g.gates, "valve gate must produce at least one Dirichlet cell"
    assert g.volume_cm3() > 0
    assert g.label == "film_gate"


def test_all_gate_cells_lie_inside_mask() -> None:
    g = build_film_gate_geometry(_default_cfg())
    for iy, ix in g.gates:
        assert g.mask[iy, ix], f"gate ({iy},{ix}) is outside mask"


def test_thickness_is_zero_outside_mask() -> None:
    g = build_film_gate_geometry(_default_cfg())
    assert np.all(g.thickness_mm[~g.mask] == 0.0)


def test_thickness_within_expected_range() -> None:
    cfg = _default_cfg()
    g = build_film_gate_geometry(cfg)
    h = g.thickness_mm[g.mask]
    h_min = min(cfg.plate_thk_mm, cfg.runner_thk_mm)
    h_max = max(cfg.plate_thk_mm, cfg.runner_thk_mm)
    assert h.min() >= h_min - 1e-9
    assert h.max() <= h_max + 1e-9


def test_plate_region_has_uniform_plate_thickness() -> None:
    """Cells well inside the plate must carry exactly plate_thk_mm."""
    cfg = _default_cfg()
    g = build_film_gate_geometry(cfg)
    pad = cfg.pad_mm
    y_long = pad + cfg.runner_short_diameter_mm / 2 + cfg.runner_depth_mm
    iy_idx, ix_idx = np.indices(g.mask.shape)
    yy = (iy_idx + 0.5) * cfg.cell_size_mm
    xx = (ix_idx + 0.5) * cfg.cell_size_mm
    # 2 mm above the long edge, well inside the plate
    deep_in_plate = (
        (yy > y_long + 2.0)
        & (yy < y_long + cfg.plate_h_mm - 1.0)
        & (xx > pad + 2.0)
        & (xx < pad + cfg.plate_w_mm - 2.0)
    )
    sample = g.thickness_mm[g.mask & deep_in_plate]
    assert sample.size > 0
    assert np.allclose(sample, cfg.plate_thk_mm)


def test_half_circle_region_has_runner_thickness() -> None:
    cfg = _default_cfg()
    g = build_film_gate_geometry(cfg)
    pad = cfg.pad_mm
    cx = pad + cfg.plate_w_mm / 2
    y_short = pad + cfg.runner_short_diameter_mm / 2
    iy_idx, ix_idx = np.indices(g.mask.shape)
    yy = (iy_idx + 0.5) * cfg.cell_size_mm
    xx = (ix_idx + 0.5) * cfg.cell_size_mm
    in_half = ((xx - cx) ** 2 + (yy - y_short) ** 2 <= (cfg.runner_short_diameter_mm / 2) ** 2) & (
        yy <= y_short
    )
    sample = g.thickness_mm[g.mask & in_half]
    assert sample.size > 0
    assert np.allclose(sample, cfg.runner_thk_mm)


def test_gate_land_closes_outside_aperture() -> None:
    """On the plate-bottom row, only the central W_gate must be open."""
    cfg = _default_cfg(gate_width_mm=40.0, runner_long_mm=80.0)
    g = build_film_gate_geometry(cfg)
    pad = cfg.pad_mm
    cx = pad + cfg.plate_w_mm / 2
    y_long = pad + cfg.runner_short_diameter_mm / 2 + cfg.runner_depth_mm

    iy_plate_bottom = int(np.floor(y_long / cfg.cell_size_mm))
    open_cols = np.where(g.mask[iy_plate_bottom, :])[0]
    assert open_cols.size > 0
    open_x = (open_cols + 0.5) * cfg.cell_size_mm
    # All open cells on this row must lie within ±W_gate/2 of cx (with 1-cell tolerance)
    assert np.all(np.abs(open_x - cx) <= cfg.gate_width_mm / 2 + cfg.cell_size_mm)
    # And the aperture must actually contain cells
    assert open_cols.size >= int(cfg.gate_width_mm / cfg.cell_size_mm) - 2


def test_full_aperture_gate_keeps_long_edge_open() -> None:
    """W_gate == L_long → the gate-land closure should not block any cells
    that the trapezoid silhouette already covers."""
    cfg = _default_cfg(gate_width_mm=80.0, runner_long_mm=80.0)
    g_full = build_film_gate_geometry(cfg)

    cfg_narrow = _default_cfg(gate_width_mm=20.0, runner_long_mm=80.0)
    g_narrow = build_film_gate_geometry(cfg_narrow)

    # Narrow aperture must have strictly fewer cavity cells than full aperture
    assert int(g_narrow.mask.sum()) < int(g_full.mask.sum())


def test_volume_increases_with_plate_size() -> None:
    g_small = build_film_gate_geometry(_default_cfg(plate_w_mm=80.0, plate_h_mm=40.0))
    g_large = build_film_gate_geometry(_default_cfg(plate_w_mm=160.0, plate_h_mm=120.0))
    assert g_large.volume_cm3() > g_small.volume_cm3()


def test_validation_rejects_runner_long_exceeding_plate() -> None:
    with pytest.raises(ValueError, match="must be ≤ plate_w_mm"):
        FilmGateConfig(
            plate_w_mm=80.0,
            plate_h_mm=80.0,
            plate_thk_mm=2.0,
            runner_long_mm=120.0,  # > plate_w_mm
            runner_short_diameter_mm=12.0,
            runner_depth_mm=20.0,
            runner_thk_mm=4.0,
            runner_flat_depth_mm=8.0,
            runner_slope_depth_mm=12.0,
            valve_gate_diameter_mm=4.0,
            gate_width_mm=60.0,
        ).validate()


def test_validation_rejects_inverted_trapezoid() -> None:
    with pytest.raises(ValueError, match="inverted trapezoid"):
        FilmGateConfig(
            plate_w_mm=120.0,
            plate_h_mm=80.0,
            plate_thk_mm=2.0,
            runner_long_mm=10.0,  # < runner_short_diameter_mm
            runner_short_diameter_mm=20.0,
            runner_depth_mm=20.0,
            runner_thk_mm=4.0,
            runner_flat_depth_mm=8.0,
            runner_slope_depth_mm=12.0,
            valve_gate_diameter_mm=4.0,
            gate_width_mm=10.0,
        ).validate()


def test_validation_rejects_gate_wider_than_long_edge() -> None:
    with pytest.raises(ValueError, match="must be ≤ runner_long_mm"):
        FilmGateConfig(
            plate_w_mm=120.0,
            plate_h_mm=80.0,
            plate_thk_mm=2.0,
            runner_long_mm=60.0,
            runner_short_diameter_mm=12.0,
            runner_depth_mm=20.0,
            runner_thk_mm=4.0,
            runner_flat_depth_mm=8.0,
            runner_slope_depth_mm=12.0,
            valve_gate_diameter_mm=4.0,
            gate_width_mm=80.0,  # > runner_long_mm
        ).validate()


def test_validation_rejects_valve_larger_than_short_diameter() -> None:
    with pytest.raises(ValueError, match="valve_gate_diameter_mm"):
        FilmGateConfig(
            plate_w_mm=120.0,
            plate_h_mm=80.0,
            plate_thk_mm=2.0,
            runner_long_mm=80.0,
            runner_short_diameter_mm=8.0,
            runner_depth_mm=20.0,
            runner_thk_mm=4.0,
            runner_flat_depth_mm=8.0,
            runner_slope_depth_mm=12.0,
            valve_gate_diameter_mm=12.0,  # > runner_short_diameter_mm
            gate_width_mm=60.0,
        ).validate()


def test_validation_rejects_inconsistent_flat_slope_sum() -> None:
    with pytest.raises(ValueError, match="must equal runner_depth_mm"):
        FilmGateConfig(
            plate_w_mm=120.0,
            plate_h_mm=80.0,
            plate_thk_mm=2.0,
            runner_long_mm=80.0,
            runner_short_diameter_mm=12.0,
            runner_depth_mm=20.0,
            runner_thk_mm=4.0,
            runner_flat_depth_mm=5.0,
            runner_slope_depth_mm=10.0,  # 5 + 10 ≠ 20
            valve_gate_diameter_mm=4.0,
            gate_width_mm=60.0,
        ).validate()


def test_solver_runs_on_film_gate_geometry() -> None:
    """End-to-end smoke test: solver must accept the new geometry type
    and produce a sensible fill-time field (max τ at the plate corners,
    minimum at the valve gate)."""
    cfg = _default_cfg(cell_size_mm=2.0)  # coarser mesh for speed
    g = build_film_gate_geometry(cfg)
    solver = HeleShawSolver(
        geometry=g,
        material=MaterialDB()["PP"],
        melt_temperature_K=503.15,
        mold_temperature_K=313.15,
        injection_velocity_mms=100.0,
        injection_volume_flow_cm3s=20.0,
    )
    result = solver.solve(num_frames=4)

    # Gate cells must be at τ ≈ 0
    for iy, ix in g.gates:
        assert result.tau[iy, ix] < 1e-9

    # τ_max must lie inside the plate, not in the runner
    iy_max, ix_max = np.unravel_index(int(np.nanargmax(result.tau)), result.tau.shape)
    pad = cfg.pad_mm
    y_long = pad + cfg.runner_short_diameter_mm / 2 + cfg.runner_depth_mm
    y_max = (iy_max + 0.5) * cfg.cell_size_mm
    assert y_max > y_long, "τ should peak inside the plate, not in the runner"


# ============================================================
# Flow balancer (▽-shaped local thinning) tests
# ============================================================


def _balancer_cfg(**overrides) -> FilmGateConfig:
    """Default config with the balancer enabled. Apex sits well above the
    valve-gate disk, base sits at the long edge."""
    base = dict(
        plate_w_mm=120.0,
        plate_h_mm=80.0,
        plate_thk_mm=2.0,
        runner_long_mm=80.0,
        runner_short_diameter_mm=12.0,
        runner_depth_mm=20.0,
        runner_thk_mm=4.0,
        runner_flat_depth_mm=8.0,
        runner_slope_depth_mm=12.0,
        valve_gate_diameter_mm=4.0,
        gate_width_mm=60.0,
        cell_size_mm=1.0,
        pad_mm=5.0,
        balancer_enabled=True,
        balancer_base_width_mm=40.0,
        balancer_height_mm=14.0,
        balancer_base_distance_from_gate_mm=20.0,  # base sits at long edge (= D)
        balancer_target_thickness_mm=2.0,  # = plate_thk
    )
    base.update(overrides)
    return FilmGateConfig(**base)


def test_balancer_off_leaves_geometry_identical_to_pre_balancer_default() -> None:
    """Smoke regression: balancer_enabled=False must reproduce the
    pre-balancer film-gate geometry bit-for-bit."""
    g_off = build_film_gate_geometry(_default_cfg())
    # Construct a config that explicitly sets balancer fields but keeps
    # balancer_enabled=False — must equal the implicit-default version.
    g_explicit = build_film_gate_geometry(
        _default_cfg(
            balancer_enabled=False,
            balancer_base_width_mm=999.0,  # nonsense values must be ignored
            balancer_height_mm=999.0,
            balancer_base_distance_from_gate_mm=999.0,
            balancer_target_thickness_mm=999.0,
        )
    )
    assert np.array_equal(g_off.mask, g_explicit.mask)
    assert np.allclose(g_off.thickness_mm, g_explicit.thickness_mm)


def test_balancer_creates_thinned_region() -> None:
    """With the balancer on, some trapezoid cells must carry h = h_bal."""
    cfg = _balancer_cfg(balancer_target_thickness_mm=2.0)
    g = build_film_gate_geometry(cfg)
    pad = cfg.pad_mm
    cx = pad + cfg.plate_w_mm / 2
    y_short = pad + cfg.runner_short_diameter_mm / 2
    iy_idx, ix_idx = np.indices(g.mask.shape)
    yy = (iy_idx + 0.5) * cfg.cell_size_mm
    xx = (ix_idx + 0.5) * cfg.cell_size_mm

    # Inside the balancer ▽, thickness must equal the target value.
    y_apex = y_short + (cfg.balancer_base_distance_from_gate_mm - cfg.balancer_height_mm)
    y_base = y_short + cfg.balancer_base_distance_from_gate_mm
    in_band = (yy >= y_apex) & (yy <= y_base)
    half_w = (
        0.5 * cfg.balancer_base_width_mm * np.clip((yy - y_apex) / cfg.balancer_height_mm, 0.0, 1.0)
    )
    in_triangle = in_band & (np.abs(xx - cx) <= half_w)
    sample = g.thickness_mm[g.mask & in_triangle]
    assert sample.size > 0, "balancer triangle must contain cavity cells"
    assert np.allclose(sample, cfg.balancer_target_thickness_mm), (
        f"balancer cells expected h = {cfg.balancer_target_thickness_mm}, "
        f"got min={sample.min()}, max={sample.max()}"
    )


def test_balancer_does_not_touch_plate_or_half_circle() -> None:
    """The balancer is a runner-side feature; plate body and half-circle
    region must keep their pre-balancer thicknesses."""
    cfg_off = _default_cfg()
    cfg_on = _balancer_cfg(plate_w_mm=cfg_off.plate_w_mm, plate_h_mm=cfg_off.plate_h_mm)
    g_off = build_film_gate_geometry(cfg_off)
    g_on = build_film_gate_geometry(cfg_on)

    pad = cfg_on.pad_mm
    iy_idx, ix_idx = np.indices(g_on.mask.shape)
    yy = (iy_idx + 0.5) * cfg_on.cell_size_mm
    y_long = pad + cfg_on.runner_short_diameter_mm / 2 + cfg_on.runner_depth_mm
    y_short = pad + cfg_on.runner_short_diameter_mm / 2

    plate_only = (yy > y_long + 0.5) & g_off.mask & g_on.mask
    half_only = (yy < y_short) & g_off.mask & g_on.mask

    assert np.allclose(g_off.thickness_mm[plate_only], g_on.thickness_mm[plate_only])
    assert np.allclose(g_off.thickness_mm[half_only], g_on.thickness_mm[half_only])


def test_balancer_reduces_cavity_volume() -> None:
    """Carving thinner material into the runner must reduce cavity volume
    (when balancer_target_thickness < runner_thk)."""
    cfg_off = _default_cfg()
    cfg_on = _balancer_cfg(balancer_target_thickness_mm=2.0)  # < runner_thk=4
    g_off = build_film_gate_geometry(cfg_off)
    g_on = build_film_gate_geometry(cfg_on)
    assert g_on.volume_cm3() < g_off.volume_cm3()


def test_balancer_changes_flow_pattern() -> None:
    """End-to-end: τ field must differ between balancer-off and balancer-on."""
    cfg_off = _default_cfg(cell_size_mm=2.0)
    cfg_on = _balancer_cfg(cell_size_mm=2.0)
    common_solver_kwargs = dict(
        material=MaterialDB()["PP"],
        melt_temperature_K=503.15,
        mold_temperature_K=313.15,
        injection_velocity_mms=100.0,
        injection_volume_flow_cm3s=20.0,
    )
    r_off = HeleShawSolver(
        geometry=build_film_gate_geometry(cfg_off), **common_solver_kwargs
    ).solve(num_frames=4)
    r_on = HeleShawSolver(geometry=build_film_gate_geometry(cfg_on), **common_solver_kwargs).solve(
        num_frames=4
    )

    common_mask = ~np.isnan(r_off.tau) & ~np.isnan(r_on.tau)
    diff = np.abs(r_off.tau[common_mask] - r_on.tau[common_mask])
    assert float(diff.max()) > 0.0, "balancer must alter the τ field"


def test_balancer_validation_rejects_zero_thickness() -> None:
    with pytest.raises(ValueError, match="balancer_target_thickness_mm"):
        _balancer_cfg(balancer_target_thickness_mm=0.0).validate()


def test_balancer_validation_rejects_base_wider_than_gate() -> None:
    with pytest.raises(ValueError, match="balancer_base_width_mm"):
        _balancer_cfg(balancer_base_width_mm=80.0, gate_width_mm=60.0).validate()


def test_balancer_validation_rejects_base_past_long_edge() -> None:
    with pytest.raises(ValueError, match="balancer_base_distance_from_gate_mm"):
        _balancer_cfg(
            balancer_base_distance_from_gate_mm=25.0,  # > runner_depth_mm=20
        ).validate()


def test_balancer_validation_rejects_apex_into_valve_gate() -> None:
    """Apex y-offset must clear the valve-gate disk radius."""
    with pytest.raises(ValueError, match="balancer apex"):
        _balancer_cfg(
            valve_gate_diameter_mm=8.0,  # radius = 4 mm
            balancer_base_distance_from_gate_mm=10.0,
            balancer_height_mm=8.0,  # apex at y_short + 2 mm < 4 mm
        ).validate()
