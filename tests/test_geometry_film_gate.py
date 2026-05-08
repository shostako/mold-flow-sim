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


# ---------- gate-side / far-side plate split ----------


def _split_cfg(**overrides) -> FilmGateConfig:
    """Helper: 2-zone plate (gate-side / far-side) with safe defaults."""
    base = dict(
        plate_w_mm=120.0,
        plate_h_mm=50.0,
        plate_thk_mm=0.4,  # legacy fallback (irrelevant when split is on)
        runner_long_mm=80.0,
        runner_short_diameter_mm=12.0,
        runner_depth_mm=20.0,
        runner_thk_mm=2.5,
        runner_flat_depth_mm=8.0,
        runner_slope_depth_mm=12.0,
        valve_gate_diameter_mm=4.0,
        gate_width_mm=80.0,
        cell_size_mm=1.0,
        pad_mm=5.0,
        plate_split_height_mm=20.0,
        plate_lower_thk_mm=0.35,
        plate_upper_thk_mm=0.50,
    )
    base.update(overrides)
    return FilmGateConfig(**base)


def test_plate_split_off_uses_uniform_thickness() -> None:
    """``plate_split_height_mm == 0`` must reproduce the uniform-mode geometry."""
    cfg_uniform = _default_cfg(plate_thk_mm=0.4, plate_h_mm=50.0)
    cfg_explicit_off = _default_cfg(
        plate_thk_mm=0.4,
        plate_h_mm=50.0,
        plate_split_height_mm=0.0,
        plate_lower_thk_mm=0.30,  # ignored in uniform mode
        plate_upper_thk_mm=0.55,  # ignored in uniform mode
    )
    g_uniform = build_film_gate_geometry(cfg_uniform)
    g_off = build_film_gate_geometry(cfg_explicit_off)
    assert np.array_equal(g_uniform.mask, g_off.mask)
    assert np.allclose(g_uniform.thickness_mm, g_off.thickness_mm)


def test_plate_split_creates_two_thickness_bands() -> None:
    """Inside the plate body, the gate-side strip must carry plate_lower_thk_mm
    and the far-side strip must carry plate_upper_thk_mm."""
    cfg = _split_cfg()
    g = build_film_gate_geometry(cfg)

    pad = cfg.pad_mm
    iy_idx, ix_idx = np.indices(g.mask.shape)
    yy = (iy_idx + 0.5) * cfg.cell_size_mm
    y_long = pad + cfg.runner_short_diameter_mm / 2 + cfg.runner_depth_mm
    y_split = y_long + cfg.plate_split_height_mm

    in_plate = (yy >= y_long) & (yy <= y_long + cfg.plate_h_mm) & g.mask
    lower_band = in_plate & (yy < y_split)
    upper_band = in_plate & (yy >= y_split)

    assert lower_band.any() and upper_band.any()
    np.testing.assert_allclose(g.thickness_mm[lower_band], cfg.plate_lower_thk_mm, atol=1e-9)
    np.testing.assert_allclose(g.thickness_mm[upper_band], cfg.plate_upper_thk_mm, atol=1e-9)


def test_plate_split_runner_slope_terminates_at_lower_band() -> None:
    """Runner slope zone must aim at the gate-side band (plate_lower_thk_mm),
    not the far-side band — i.e. the runner exit stays continuous with the
    plate it actually feeds. The exact end value depends on cell size, so
    the assertion checks the trajectory rather than the literal floor."""
    cfg = _split_cfg()
    g = build_film_gate_geometry(cfg)

    pad = cfg.pad_mm
    iy_idx, ix_idx = np.indices(g.mask.shape)
    yy = (iy_idx + 0.5) * cfg.cell_size_mm
    xx = (ix_idx + 0.5) * cfg.cell_size_mm
    cx = pad + cfg.plate_w_mm / 2

    y_long = pad + cfg.runner_short_diameter_mm / 2 + cfg.runner_depth_mm
    # last cell-row inside the slope zone, on the centerline
    last_slope = (yy < y_long) & (yy >= y_long - cfg.cell_size_mm * 1.5) & g.mask
    on_axis = last_slope & (np.abs(xx - cx) < cfg.cell_size_mm)
    sample = g.thickness_mm[on_axis]
    assert sample.size > 0

    val = float(sample.min())
    # 1) The slope's terminal value is well below the far-side thickness:
    #    if it had been interpolating toward plate_upper_thk_mm it would
    #    sit above plate_lower_thk_mm and likely above the upper band too.
    assert val < cfg.plate_upper_thk_mm, (
        f"slope-zone end value {val} should be below plate_upper_thk_mm "
        f"{cfg.plate_upper_thk_mm} (the slope must aim at the gate-side band)"
    )
    # 2) The slope has already crossed past the midpoint of
    #    (h_runner, plate_lower_thk_mm), confirming the trajectory.
    midpoint = 0.5 * (cfg.runner_thk_mm + cfg.plate_lower_thk_mm)
    assert val < midpoint, (
        f"slope-zone end value {val} should be past the "
        f"(h_runner, plate_lower_thk_mm) midpoint {midpoint}"
    )


def test_plate_split_validation_rejects_split_above_plate_height() -> None:
    with pytest.raises(ValueError, match="plate_split_height_mm"):
        _split_cfg(plate_split_height_mm=70.0).validate()  # plate_h_mm = 50


def test_plate_split_validation_rejects_negative_lower_thickness() -> None:
    with pytest.raises(ValueError, match="plate_lower_thk_mm"):
        _split_cfg(plate_lower_thk_mm=0.0).validate()


def test_plate_split_validation_rejects_negative_upper_thickness() -> None:
    with pytest.raises(ValueError, match="plate_upper_thk_mm"):
        _split_cfg(plate_upper_thk_mm=-0.1).validate()


# ---------- multi-stage balancer (1..5 nested ▽) ----------


def _multi_stage_cfg(stages: list[tuple[float, float]], **overrides) -> FilmGateConfig:
    """Helper: balancer with N nested stages (center→outer), runner-side
    parameters identical to ``_balancer_cfg`` defaults."""
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
        cell_size_mm=0.5,
        pad_mm=5.0,
        balancer_enabled=True,
        balancer_height_mm=14.0,
        balancer_base_distance_from_gate_mm=20.0,
        balancer_base_widths_mm=tuple(W for W, _ in stages),
        balancer_thicknesses_mm=tuple(h for _, h in stages),
    )
    base.update(overrides)
    return FilmGateConfig(**base)


def test_multi_stage_balancer_resolves_stage_list() -> None:
    cfg = _multi_stage_cfg([(20.0, 0.5), (40.0, 1.0)])
    cfg.validate()
    stages = cfg.resolved_balancer_stages()
    assert stages == [(20.0, 0.5), (40.0, 1.0)]


def test_multi_stage_balancer_two_stages_create_concentric_bands() -> None:
    """Two-stage balancer at the base row: center carries h_1, surrounding
    band carries h_2, both inside the runner trapezoid."""
    cfg = _multi_stage_cfg([(20.0, 0.5), (40.0, 1.0)])
    g = build_film_gate_geometry(cfg)

    pad = cfg.pad_mm
    iy_idx, ix_idx = np.indices(g.mask.shape)
    yy = (iy_idx + 0.5) * cfg.cell_size_mm
    xx = (ix_idx + 0.5) * cfg.cell_size_mm
    cx = pad + cfg.plate_w_mm / 2
    y_short = pad + cfg.runner_short_diameter_mm / 2
    y_base = y_short + cfg.balancer_base_distance_from_gate_mm

    # Use the row immediately below the balancer base (t_y ~ 1).
    near_base = (yy < y_base) & (yy > y_base - cfg.cell_size_mm * 1.2) & g.mask
    inner = near_base & (np.abs(xx - cx) < 8.5)  # well inside W_1/2 = 10
    outer = (
        near_base & (np.abs(xx - cx) > 11.5) & (np.abs(xx - cx) < 18.5)
    )  # between W_1/2 and W_2/2
    sample_inner = g.thickness_mm[inner]
    sample_outer = g.thickness_mm[outer]
    assert sample_inner.size > 0
    assert sample_outer.size > 0
    np.testing.assert_allclose(sample_inner, 0.5, atol=1e-9)
    np.testing.assert_allclose(sample_outer, 1.0, atol=1e-9)


def test_multi_stage_balancer_n5_paints_five_thickness_levels() -> None:
    """Five-stage balancer produces five distinct thickness values inside the ▽."""
    stages = [(8.0, 0.4), (16.0, 0.7), (24.0, 1.0), (32.0, 1.3), (40.0, 1.6)]
    cfg = _multi_stage_cfg(stages)
    g = build_film_gate_geometry(cfg)

    pad = cfg.pad_mm
    iy_idx, _ix_idx = np.indices(g.mask.shape)
    yy = (iy_idx + 0.5) * cfg.cell_size_mm
    y_short = pad + cfg.runner_short_diameter_mm / 2
    y_apex = y_short + (cfg.balancer_base_distance_from_gate_mm - cfg.balancer_height_mm)
    y_base = y_short + cfg.balancer_base_distance_from_gate_mm
    in_band = (yy >= y_apex) & (yy <= y_base)
    near_base = in_band & (yy > y_base - cfg.cell_size_mm * 1.2) & g.mask

    sample = g.thickness_mm[near_base]
    distinct = sorted(set(np.round(sample, 4).tolist()))
    # All 5 stage values must appear, plus optionally the runner-slope value
    # at the very edge of the centerline row. Require at least the 5 stage h.
    for h_k in (0.4, 0.7, 1.0, 1.3, 1.6):
        assert any(abs(v - h_k) < 1e-3 for v in distinct), (
            f"stage thickness {h_k} not found in row distinct values {distinct}"
        )


def test_multi_stage_balancer_inner_stages_overwrite_outer() -> None:
    """The center column inside the balancer must carry h_1 (not h_outer)."""
    cfg = _multi_stage_cfg([(8.0, 0.3), (40.0, 1.5)])
    g = build_film_gate_geometry(cfg)

    pad = cfg.pad_mm
    iy_idx, ix_idx = np.indices(g.mask.shape)
    yy = (iy_idx + 0.5) * cfg.cell_size_mm
    xx = (ix_idx + 0.5) * cfg.cell_size_mm
    cx = pad + cfg.plate_w_mm / 2
    y_short = pad + cfg.runner_short_diameter_mm / 2
    y_base = y_short + cfg.balancer_base_distance_from_gate_mm

    near_base = (yy < y_base) & (yy > y_base - cfg.cell_size_mm * 1.2) & g.mask
    on_axis = near_base & (np.abs(xx - cx) < cfg.cell_size_mm * 0.8)
    sample = g.thickness_mm[on_axis]
    assert sample.size > 0
    np.testing.assert_allclose(sample, 0.3, atol=1e-9)


def test_multi_stage_balancer_validation_rejects_too_many_stages() -> None:
    with pytest.raises(ValueError, match="1..5 stages"):
        _multi_stage_cfg(
            [(5.0, 0.2), (10.0, 0.4), (15.0, 0.6), (20.0, 0.8), (25.0, 1.0), (30.0, 1.2)]
        ).validate()


def test_multi_stage_balancer_validation_rejects_decreasing_widths() -> None:
    with pytest.raises(ValueError, match="non-decreasing"):
        _multi_stage_cfg([(40.0, 0.5), (20.0, 1.0)]).validate()


def test_multi_stage_balancer_validation_rejects_decreasing_thicknesses() -> None:
    with pytest.raises(ValueError, match="non-decreasing"):
        _multi_stage_cfg([(20.0, 1.0), (40.0, 0.5)]).validate()


def test_multi_stage_balancer_validation_rejects_unequal_lengths() -> None:
    cfg = FilmGateConfig(
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
        balancer_height_mm=14.0,
        balancer_base_distance_from_gate_mm=20.0,
        balancer_base_widths_mm=(20.0, 40.0),
        balancer_thicknesses_mm=(0.5,),
    )
    with pytest.raises(ValueError, match="equal length"):
        cfg.validate()


def test_multi_stage_balancer_outer_width_must_fit_gate() -> None:
    with pytest.raises(ValueError, match="outermost"):
        _multi_stage_cfg([(20.0, 0.5), (70.0, 1.0)], gate_width_mm=60.0).validate()


def test_multi_stage_balancer_n1_matches_single_stage_scalar_form() -> None:
    """A single-stage tuple form must produce the same thickness map as the
    equivalent scalar form."""
    scalar_cfg = _balancer_cfg(
        balancer_base_width_mm=36.0,
        balancer_target_thickness_mm=2.0,
    )
    tuple_cfg = _balancer_cfg(
        balancer_base_width_mm=0.0,
        balancer_target_thickness_mm=0.0,
        balancer_base_widths_mm=(36.0,),
        balancer_thicknesses_mm=(2.0,),
    )
    g_scalar = build_film_gate_geometry(scalar_cfg)
    g_tuple = build_film_gate_geometry(tuple_cfg)
    assert np.array_equal(g_scalar.mask, g_tuple.mask)
    assert np.allclose(g_scalar.thickness_mm, g_tuple.thickness_mm)


def test_plate_split_lower_defaults_to_plate_thk() -> None:
    """When ``plate_lower_thk_mm`` is None, the gate-side band falls back
    to ``plate_thk_mm`` while the far-side band uses its own value."""
    cfg = _split_cfg(plate_lower_thk_mm=None, plate_upper_thk_mm=0.6)
    g = build_film_gate_geometry(cfg)
    pad = cfg.pad_mm
    iy_idx, _ = np.indices(g.mask.shape)
    yy = (iy_idx + 0.5) * cfg.cell_size_mm
    y_long = pad + cfg.runner_short_diameter_mm / 2 + cfg.runner_depth_mm
    in_plate = (yy >= y_long) & g.mask
    lower_band = in_plate & (yy < y_long + cfg.plate_split_height_mm)
    assert lower_band.any()
    np.testing.assert_allclose(g.thickness_mm[lower_band], cfg.plate_thk_mm, atol=1e-9)


# -------------------------- compression mask --------------------------


def test_compression_mask_excludes_runner_and_gate() -> None:
    """The plate body should be the only zone flagged for compression
    inflation; runner / half-circle / valve-gate cells stay at their
    cast thickness."""
    cfg = _default_cfg()
    g = build_film_gate_geometry(cfg)
    cm = g.compression_mask
    assert cm is not None
    iy_idx, _ = np.indices(g.mask.shape)
    yy = (iy_idx + 0.5) * cfg.cell_size_mm
    y_long = cfg.pad_mm + cfg.runner_short_diameter_mm / 2 + cfg.runner_depth_mm
    # Every compression cell must sit at or above the long edge (= plate
    # bottom row).
    assert np.all(yy[cm & g.mask] >= y_long - 1e-9)
    # The runner / half-circle area (below long edge) should never be in
    # the mask.
    runner_zone = g.mask & (yy < y_long)
    assert runner_zone.any()
    assert not np.any(cm & runner_zone)


def test_compression_inflates_only_plate_body() -> None:
    """When compression is enabled, the plate cells should grow by
    ``compression_factor`` while runner cells stay put."""
    from core import HeleShawSolver, MaterialDB

    cfg = _default_cfg()
    g = build_film_gate_geometry(cfg)
    db = MaterialDB()

    solver_off = HeleShawSolver(
        geometry=g,
        material=db["PP"],
        melt_temperature_K=503.15,
        mold_temperature_K=313.15,
        injection_velocity_mms=100.0,
        injection_volume_flow_cm3s=20.0,
    )
    solver_on = HeleShawSolver(
        geometry=g,
        material=db["PP"],
        melt_temperature_K=503.15,
        mold_temperature_K=313.15,
        injection_velocity_mms=100.0,
        injection_volume_flow_cm3s=20.0,
        compression_molding=True,
        compression_factor=1.8,
        compression_fraction=0.7,
    )
    h_off = solver_off._open_thickness_field()
    h_on = solver_on._open_thickness_field()
    cm = g.compression_mask
    assert cm is not None
    # Plate body should be inflated by compression_factor
    np.testing.assert_allclose(h_on[cm & g.mask], h_off[cm & g.mask] * 1.8, rtol=1e-9)
    # Runner / half-circle / gate cells should be unchanged
    other = g.mask & ~cm
    np.testing.assert_allclose(h_on[other], h_off[other], rtol=1e-9)
