"""Tests for the right-trapezoid film-gate-2 (肉厚調整ゲート) builder."""

from __future__ import annotations

import numpy as np
import pytest

from core import FilmGate2Config, HeleShawSolver, MaterialDB, build_film_gate2_geometry


def _default_cfg(**overrides) -> FilmGate2Config:
    base = dict(
        plate_w_mm=120.0,
        plate_h_mm=80.0,
        plate_thk_mm=2.0,
        gate_depth_mm=30.0,
        gate_position_mm=0.0,
        left_edge_mm=10.0,
        land_width_mm=1.0,
        land_depth_mm=0.35,
        taper1_len_mm=8.0,
        mid_depth_a_mm=2.0,
        mid_depth_b_mm=1.0,
        taper2_left_mm=5.0,
        taper2_right_mm=10.0,
        runner_depth_mm=3.0,
        runner_top_mm=4.0,
        runner_bottom_mm=2.0,
        valve_gate_diameter_mm=4.0,
        cell_size_mm=1.0,
        pad_mm=5.0,
    )
    base.update(overrides)
    return FilmGate2Config(**base)


def _grid(cfg: FilmGate2Config, shape: tuple[int, int]):
    """Return (yy, xx, y_long, x_g) cell-center coordinates in mm."""
    iy, ix = np.indices(shape)
    yy = (iy + 0.5) * cfg.cell_size_mm
    xx = (ix + 0.5) * cfg.cell_size_mm
    y_long = cfg.pad_mm + cfg.gate_depth_mm
    x_g = cfg.pad_mm + (cfg.plate_w_mm - cfg.gate_position_mm)
    return yy, xx, y_long, x_g


# --------------------------- basics ---------------------------


def test_builds_with_defaults() -> None:
    g = build_film_gate2_geometry(_default_cfg())
    assert g.mask.any()
    assert g.gates, "valve gate must produce at least one Dirichlet cell"
    assert g.volume_cm3() > 0
    assert g.label == "film_gate2"


def test_all_gate_cells_lie_inside_mask() -> None:
    g = build_film_gate2_geometry(_default_cfg())
    for iy, ix in g.gates:
        assert g.mask[iy, ix], f"gate ({iy},{ix}) is outside mask"


def test_thickness_zero_outside_mask() -> None:
    g = build_film_gate2_geometry(_default_cfg())
    assert np.all(g.thickness_mm[~g.mask] == 0.0)


def test_thickness_within_expected_range() -> None:
    cfg = _default_cfg()
    g = build_film_gate2_geometry(cfg)
    h = g.thickness_mm[g.mask]
    lo = min(cfg.land_depth_mm, cfg.plate_thk_mm)
    hi = max(cfg.runner_depth_mm, cfg.plate_thk_mm)
    assert h.min() >= lo - 1e-9
    assert h.max() <= hi + 1e-9


def test_deep_runner_reaches_runner_depth() -> None:
    cfg = _default_cfg()
    g = build_film_gate2_geometry(cfg)
    # Among gate cells, the maximum depth must equal runner_depth (the deep
    # runner / lower-taper deep end).
    gate_max = g.thickness_mm[g.mask & (g.thickness_mm <= cfg.runner_depth_mm + 1e-9)].max()
    assert np.isclose(gate_max, cfg.runner_depth_mm, atol=1e-6)


# ----------------------- silhouette --------------------------


def test_left_edge_keeps_trapezoid_open_at_left() -> None:
    """The left end is a vertical edge of height ~left_edge (right trapezoid),
    so the gate is non-empty there — not a sharp triangle tip."""
    cfg = _default_cfg()
    g = build_film_gate2_geometry(cfg)
    yy, xx, y_long, _ = _grid(cfg, g.mask.shape)
    # near the left long-edge end, just inside the plate width
    left_col = g.mask & (xx > cfg.pad_mm + 0.5) & (xx < cfg.pad_mm + 2.0) & (yy < y_long)
    gate_y = yy[left_col]
    assert gate_y.size > 0
    # gate spans roughly left_edge in y at the left end (allow discretization)
    span = float(y_long - gate_y.min())
    assert span >= cfg.left_edge_mm - 2 * cfg.cell_size_mm


def test_isosceles_silhouette_is_left_right_symmetric() -> None:
    """gate_position = Wp/2 puts the valve at center → symmetric silhouette."""
    cfg = _default_cfg(gate_position_mm=60.0)  # Wp/2
    g = build_film_gate2_geometry(cfg)
    m = g.mask
    flipped = m[:, ::-1]
    # symmetric up to a 1-cell discretization slack
    diff = int(np.sum(m != flipped))
    assert diff <= 2 * m.shape[0], f"silhouette not symmetric (diff={diff})"


def test_gate_left_offset_limits_second_stage() -> None:
    """gate_left_offset は2段目(下段)の左端。左より左は2段目が無く、
    1段目と深ランナーは全幅で残る。"""
    cfg = _default_cfg(
        gate_left_offset_mm=80.0,
        gate_position_mm=0.0,
        taper2_right_mm=18.0,
        taper2_left_mm=12.0,
        gate_depth_mm=40.0,
        land_width_mm=1.0,
        taper1_len_mm=4.0,
    )
    g = build_film_gate2_geometry(cfg)
    yy, xx, y_long, x_g = _grid(cfg, g.mask.shape)
    x_2nd = cfg.pad_mm + cfg.gate_left_offset_mm
    t = y_long - yy
    thk = g.thickness_mm
    t2 = cfg.land_width_mm + cfg.taper1_len_mm
    # 1st stage spans the full width (gate present even left of x_2nd)
    left_1st = g.mask & (xx < x_2nd - 5.0) & (t > cfg.land_width_mm + 0.5) & (t < t2)
    assert left_1st.any()
    # left of x_2nd, beyond t2: NO 2nd stage (mid_b); stays at 1st-stage floor
    # judge just beyond t2 (well clear of the deep runner near the slanted edge)
    left_2nd = (
        g.mask
        & (xx < x_2nd - 5.0)
        & (np.abs(t - (t2 + 2.0)) < 1.0)
        & (np.abs(thk - cfg.mid_depth_b_mm) < 0.1)
    )
    assert not left_2nd.any()
    # right of x_2nd, just beyond t2: 2nd stage (mid_b) present
    right_2nd = (
        g.mask
        & (xx > x_2nd + 5.0)
        & (np.abs(t - (t2 + 2.0)) < 1.0)
        & (np.abs(thk - cfg.mid_depth_b_mm) < 0.1)
    )
    assert right_2nd.any()


def test_gate_left_offset_validation_rejects_past_valve() -> None:
    """左端が注入点を越える設定は reject。"""
    with pytest.raises(ValueError):
        build_film_gate2_geometry(
            _default_cfg(gate_position_mm=0.0, gate_left_offset_mm=130.0, plate_w_mm=120.0)
        )


# ------------------- depth profile (base distance) -----------


def test_taper_depth_is_x_independent_off_runner() -> None:
    """Base-distance method: at a fixed t (off the deep-runner band) the depth
    is independent of x → constant-angle taper (a plane, not a curved surface)."""
    cfg = _default_cfg(gate_position_mm=60.0)  # x_g at center, right side = taper only
    g = build_film_gate2_geometry(cfg)
    yy, xx, y_long, x_g = _grid(cfg, g.mask.shape)
    t = y_long - yy
    # pick one row (= one fixed t) inside the upper-taper band
    upper = (t > cfg.land_width_mm + 1.0) & (t < cfg.land_width_mm + cfg.taper1_len_mm - 1.0)
    rows = np.where((g.mask & upper).any(axis=1))[0]
    assert rows.size > 0
    iy_sel = int(rows[rows.size // 2])
    # right of the valve → deep runner (left edge only) is absent here
    row = g.mask[iy_sel] & (xx[iy_sel] > x_g + 10.0) & (yy[iy_sel] < y_long)
    vals = g.thickness_mm[iy_sel][row]
    assert vals.size >= 2
    assert np.ptp(vals) < 1e-6  # constant depth along x → constant-angle plane


def test_depth_step_between_tapers() -> None:
    """mid_depth_a != mid_depth_b creates a depth step at t = land+L1."""
    cfg = _default_cfg(mid_depth_a_mm=1.0, mid_depth_b_mm=2.2, gate_position_mm=60.0)
    g = build_film_gate2_geometry(cfg)
    yy, xx, y_long, x_g = _grid(cfg, g.mask.shape)
    t = y_long - yy
    tb = cfg.land_width_mm + cfg.taper1_len_mm
    col = g.mask & (xx > x_g + 10.0) & (yy < y_long)
    just_below = col & (t > tb - cfg.cell_size_mm) & (t <= tb)  # upper taper end
    just_above = col & (t > tb) & (t < tb + cfg.cell_size_mm)  # lower taper start
    assert just_below.any() and just_above.any()
    v_below = float(g.thickness_mm[just_below].mean())  # ~ mid_depth_a
    v_above = float(g.thickness_mm[just_above].mean())  # ~ mid_depth_b
    assert v_above - v_below > 0.5


def test_continuous_mid_depth_has_no_step() -> None:
    """mid_depth_a == mid_depth_b → continuous slope (no jump)."""
    cfg = _default_cfg(
        mid_depth_a_mm=1.5,
        mid_depth_b_mm=1.5,
        gate_position_mm=60.0,
        taper2_left_mm=15.0,
        taper2_right_mm=20.0,
        gate_depth_mm=40.0,
    )
    g = build_film_gate2_geometry(cfg)
    yy, xx, y_long, x_g = _grid(cfg, g.mask.shape)
    t = y_long - yy
    tb = cfg.land_width_mm + cfg.taper1_len_mm
    col = g.mask & (xx > x_g + 10.0) & (yy < y_long)
    just_below = col & (t > tb - cfg.cell_size_mm) & (t <= tb)
    just_above = col & (t > tb) & (t < tb + cfg.cell_size_mm)
    if just_below.any() and just_above.any():
        v_below = float(g.thickness_mm[just_below].mean())
        v_above = float(g.thickness_mm[just_above].mean())
        assert abs(v_above - v_below) < 0.4  # continuous across the boundary


def test_lower_taper_far_point_is_trapezoid() -> None:
    """青テーパ下端: 注入点側(taper2_right) > 端側(taper2_left) の台形。"""
    cfg = _default_cfg(
        taper2_left_mm=5.0,
        taper2_right_mm=14.0,
        gate_depth_mm=40.0,
        land_width_mm=1.0,
        taper1_len_mm=3.0,
        gate_position_mm=60.0,
    )
    g = build_film_gate2_geometry(cfg)
    yy, xx, y_long, x_g = _grid(cfg, g.mask.shape)
    t = y_long - yy
    thk = g.thickness_mm

    t2 = cfg.land_width_mm + cfg.taper1_len_mm

    def far_point(x_target: float) -> float:
        # right of the valve; the 2nd stage is the constant mid_b band beyond t2
        col = g.mask & (np.abs(xx - x_target) < 0.6) & (yy < y_long) & (xx > x_g)
        ct, cthk = t[col], thk[col]
        in_2nd = (np.abs(cthk - cfg.mid_depth_b_mm) < 0.1) & (ct > t2)
        return float(ct[in_2nd].max()) if in_2nd.any() else 0.0

    fp_valve = far_point(x_g + 5.0)
    fp_end = far_point(x_g + 50.0)
    assert fp_valve > fp_end + 1.0  # valve-side far point is deeper → trapezoid


def test_lower_taper_present_at_far_end() -> None:
    """taper2_left > land+taper1 → the lower taper survives at the far end
    (no sharp wedge tip)."""
    cfg = _default_cfg(
        taper2_left_mm=8.0,
        taper2_right_mm=12.0,
        land_width_mm=1.0,
        taper1_len_mm=3.0,
        gate_position_mm=60.0,
    )
    g = build_film_gate2_geometry(cfg)
    yy, xx, y_long, x_g = _grid(cfg, g.mask.shape)
    t = y_long - yy
    t2 = cfg.land_width_mm + cfg.taper1_len_mm
    end_col = g.mask & (xx > cfg.pad_mm + cfg.plate_w_mm - 6.0) & (yy < y_long) & (xx > x_g)
    cthk = g.thickness_mm[end_col]
    ct = t[end_col]
    has_2nd = ((np.abs(cthk - cfg.mid_depth_b_mm) < 0.1) & (ct > t2)).any()
    assert has_2nd


# ----------------------- plate split -------------------------


def test_plate_split_creates_two_bands() -> None:
    cfg = _default_cfg(plate_split_height_mm=20.0, plate_lower_thk_mm=0.35, plate_upper_thk_mm=0.5)
    g = build_film_gate2_geometry(cfg)
    yy, xx, y_long, _ = _grid(cfg, g.mask.shape)
    in_plate = g.mask & (yy >= y_long)
    lower = in_plate & (yy > y_long + 2) & (yy < y_long + cfg.plate_split_height_mm - 2)
    upper = in_plate & (yy > y_long + cfg.plate_split_height_mm + 2)
    assert np.allclose(g.thickness_mm[lower], 0.35, atol=1e-6)
    assert np.allclose(g.thickness_mm[upper], 0.5, atol=1e-6)


# ----------------------- compression -------------------------


def test_compression_mask_excludes_gate() -> None:
    cfg = _default_cfg()
    g = build_film_gate2_geometry(cfg)
    assert g.compression_mask is not None
    frac = g.compression_volume_fraction()
    assert 0.0 < frac < 1.0
    # the deep runner (deepest cells) must not be in the compression target
    deep = g.thickness_mm >= cfg.runner_depth_mm - 1e-6
    assert not np.any(g.compression_mask & deep)


# ----------------------- validation --------------------------


def test_validation_rejects_left_edge_exceeding_depth() -> None:
    with pytest.raises(ValueError):
        build_film_gate2_geometry(_default_cfg(left_edge_mm=40.0, gate_depth_mm=30.0))


def test_validation_rejects_runner_bottom_exceeding_top() -> None:
    with pytest.raises(ValueError):
        build_film_gate2_geometry(_default_cfg(runner_bottom_mm=5.0, runner_top_mm=4.0))


def test_validation_rejects_taper_sum_exceeding_depth() -> None:
    with pytest.raises(ValueError):
        build_film_gate2_geometry(_default_cfg(taper2_right_mm=35.0, gate_depth_mm=30.0))


def test_validation_rejects_gate_position_out_of_range() -> None:
    with pytest.raises(ValueError):
        build_film_gate2_geometry(_default_cfg(gate_position_mm=200.0, plate_w_mm=120.0))


def test_validation_rejects_nonpositive_runner_depth() -> None:
    with pytest.raises(ValueError):
        build_film_gate2_geometry(_default_cfg(runner_depth_mm=-1.0))


def test_validation_rejects_split_above_plate_height() -> None:
    with pytest.raises(ValueError):
        build_film_gate2_geometry(_default_cfg(plate_split_height_mm=200.0, plate_h_mm=80.0))


# ----------------------- solver integration ------------------


def test_solver_runs_on_film_gate2() -> None:
    g = build_film_gate2_geometry(_default_cfg())
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
