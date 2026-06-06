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


def test_deep_runner_stays_a_narrow_band_not_flooding_gate_face() -> None:
    # Regression: a past bug defaulted the depth floor to runner_depth, so
    # every gate cell beyond the 2nd-stage far point (t > t_lower) became the
    # deepest value, flooding the whole gate face. The deep runner must stay a
    # NARROW band along the slanted edge; the floor beyond the 2nd stage is the
    # thin mid_b, never runner_depth. (Asserting only that the *max* reaches
    # runner_depth — as test_deep_runner_reaches_runner_depth does — does NOT
    # catch this, because the flooded case has the same max.)
    cfg = _default_cfg()
    g = build_film_gate2_geometry(cfg)
    yy, _xx, y_long, _x_g = _grid(cfg, g.thickness_mm.shape)
    gate = g.mask & (yy < y_long - 1e-9)
    deepest = g.thickness_mm >= cfg.runner_depth_mm - 1e-6
    frac = np.count_nonzero(gate & deepest) / np.count_nonzero(gate)
    assert frac < 0.25, (
        f"deep runner floods {frac:.0%} of the gate face; it must stay a narrow "
        "band along the slanted edge"
    )


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
    """gate_left_offset は2段目テーパの左端。右(x>=x_2nd)はランド直後に薄い
    2段目テーパ、左(x<x_2nd)は2段目が無くランドから1段目テーパ(厚)へ直行する。
    同じ t 帯で左の深さ > 右の深さ になる。"""
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
    # t band near the end of the single 1st taper (left side ~mid_a) while the
    # right side is still in the thin 2nd taper.
    t_band = cfg.land_width_mm + cfg.taper1_len_mm - 0.5
    band = np.abs(t - t_band) < 0.4
    left = g.mask & band & (xx < x_2nd - 5.0) & (thk < cfg.runner_depth_mm - 0.5)
    right = g.mask & band & (xx > x_2nd + 5.0) & (thk < cfg.runner_depth_mm - 0.5)
    assert left.any() and right.any()
    left_mean = float(thk[left].mean())
    right_mean = float(thk[right].mean())
    assert left_mean > right_mean + 0.5  # left=1st taper(thick), right=2nd taper(thin)
    assert left_mean > 1.3  # heading to mid_a (2.0)
    assert right_mean < 1.0  # still thin (toward mid_b)


def test_gate_left_offset_validation_rejects_past_valve() -> None:
    """左端が注入点を越える設定は reject。"""
    with pytest.raises(ValueError):
        build_film_gate2_geometry(
            _default_cfg(gate_position_mm=0.0, gate_left_offset_mm=130.0, plate_w_mm=120.0)
        )


# ------------------- depth profile (base distance) -----------


def test_taper_depth_is_x_independent_off_runner() -> None:
    """Base-distance method: in the single-taper region (no 2nd stage, left of
    x_2nd) at a fixed t the depth is independent of x → constant-angle taper (a
    plane). With a 2nd stage the lower taper is a wedge whose width varies with
    x, so x-independence holds only in the single-taper region."""
    cfg = _default_cfg(gate_position_mm=0.0, gate_left_offset_mm=60.0)
    g = build_film_gate2_geometry(cfg)
    yy, xx, y_long, x_g = _grid(cfg, g.mask.shape)
    x_2nd = cfg.pad_mm + cfg.gate_left_offset_mm
    t = y_long - yy
    # pick one row (= one fixed t) inside the single 1st-taper band
    upper = (t > cfg.land_width_mm + 1.0) & (t < cfg.land_width_mm + cfg.taper1_len_mm - 1.0)
    rows = np.where((g.mask & upper).any(axis=1))[0]
    assert rows.size > 0
    iy_sel = int(rows[rows.size // 2])
    # left of x_2nd (single taper) and off the deep-runner band
    row = (
        g.mask[iy_sel]
        & (xx[iy_sel] > cfg.pad_mm + 8.0)
        & (xx[iy_sel] < x_2nd - 5.0)
        & (g.thickness_mm[iy_sel] < cfg.runner_depth_mm - 0.5)
        & (yy[iy_sel] < y_long)
    )
    vals = g.thickness_mm[iy_sel][row]
    assert vals.size >= 2
    assert np.ptp(vals) < 1e-6  # constant depth along x → constant-angle plane


def test_2nd_to_1st_boundary_continuous_even_with_diff_mid() -> None:
    """The 2nd↔1st taper boundary is a continuous slope (mid_b → mid_a) even
    when mid_depth_a != mid_depth_b: there is NO step between the two tapers.
    Steps are instead made at the land boundary via taper2_near / taper1_near.
    """
    cfg = _default_cfg(
        mid_depth_a_mm=2.2,
        mid_depth_b_mm=1.0,
        gate_position_mm=60.0,
        taper2_left_mm=12.0,
        taper2_right_mm=16.0,
        gate_depth_mm=40.0,
    )
    g = build_film_gate2_geometry(cfg)
    yy, xx, y_long, x_g = _grid(cfg, g.mask.shape)
    t = y_long - yy
    thk = g.thickness_mm
    # fixed x right of the valve (no deep runner there); walk the profile in t
    col = g.mask & (np.abs(xx - (x_g + 15.0)) < cfg.cell_size_mm * 0.6) & (yy < y_long)
    ts, ds = t[col], thk[col]
    order = np.argsort(ts)
    ds_sorted = ds[order]
    keep = ds_sorted < cfg.runner_depth_mm - 0.5  # exclude any deep-runner cell
    ds_k = ds_sorted[keep]
    assert ds_k.size > 5
    diffs = np.abs(np.diff(ds_k))
    assert diffs.max() < 0.4, f"unexpected step between adjacent rows: {diffs.max():.3f}"


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


def test_second_taper_far_point_is_absolute_distance() -> None:
    """taper2_left/right are distances from the product long edge (legacy
    semantics), NOT widths after the land: the 2nd taper reaches mid_b at
    t ≈ taper2_far, so the depth at t = taper2_right is ~mid_b (it would be
    < mid_b if land_width were silently added to the far point)."""
    cfg = _default_cfg(
        gate_position_mm=0.0,
        gate_left_offset_mm=0.0,
        taper2_left_mm=10.0,
        taper2_right_mm=10.0,
        land_width_mm=1.0,
        taper1_len_mm=8.0,
        gate_depth_mm=30.0,
        cell_size_mm=0.5,
    )
    g = build_film_gate2_geometry(cfg)
    yy, xx, y_long, x_g = _grid(cfg, g.mask.shape)
    t = y_long - yy
    thk = g.thickness_mm
    col = (
        g.mask
        & (xx > x_g - 30.0)
        & (xx < x_g - 5.0)
        & (yy < y_long)
        & (thk < cfg.runner_depth_mm - 0.5)
    )
    at_far = col & (np.abs(t - cfg.taper2_right_mm) < cfg.cell_size_mm * 0.6)
    assert at_far.any()
    assert abs(float(thk[at_far].mean()) - cfg.mid_depth_b_mm) < 0.05


# ------------- land-boundary step (taper near depths) --------


def test_resolved_taper_near_depths_fallback() -> None:
    """None falls back to land_depth (both land boundaries continuous); explicit
    values are returned as-is."""
    cfg = _default_cfg(land_depth_mm=0.4, mid_depth_b_mm=1.1)
    d2, d1 = cfg.resolved_taper_near_depths()
    assert d2 == cfg.land_depth_mm
    assert d1 == cfg.land_depth_mm
    cfg2 = _default_cfg(taper2_near_depth_mm=0.7, taper1_near_depth_mm=0.5)
    d2b, d1b = cfg2.resolved_taper_near_depths()
    assert d2b == 0.7
    assert d1b == 0.5


def test_taper_near_none_is_continuous_with_land() -> None:
    """taper2_near = taper1_near = None → the taper starts at land_depth, so
    there is no step right after the land (single-taper region)."""
    cfg = _default_cfg(gate_position_mm=0.0, gate_left_offset_mm=60.0)
    g = build_film_gate2_geometry(cfg)
    yy, xx, y_long, x_g = _grid(cfg, g.mask.shape)
    x_2nd = cfg.pad_mm + cfg.gate_left_offset_mm
    t = y_long - yy
    thk = g.thickness_mm
    band = g.mask & (t > cfg.land_width_mm) & (t < cfg.land_width_mm + 1.0)
    sel = band & (xx > cfg.pad_mm + 8.0) & (xx < x_2nd - 5.0) & (thk < cfg.runner_depth_mm - 0.5)
    assert sel.any()
    assert float(thk[sel].min()) < cfg.land_depth_mm + 0.25  # no big jump from land


def test_taper1_near_makes_land_step_in_single_region() -> None:
    """taper1_near > land_depth steps up right after the land in the
    single-taper (no 2nd stage) region (x < x_2nd)."""
    cfg = _default_cfg(
        gate_position_mm=0.0,
        gate_left_offset_mm=60.0,
        land_depth_mm=0.35,
        taper1_near_depth_mm=1.2,
    )
    g = build_film_gate2_geometry(cfg)
    yy, xx, y_long, x_g = _grid(cfg, g.mask.shape)
    x_2nd = cfg.pad_mm + cfg.gate_left_offset_mm
    t = y_long - yy
    thk = g.thickness_mm
    band = g.mask & (t > cfg.land_width_mm) & (t < cfg.land_width_mm + 1.0)
    sel = band & (xx > cfg.pad_mm + 8.0) & (xx < x_2nd - 5.0) & (thk < cfg.runner_depth_mm - 0.5)
    assert sel.any()
    # the 1st taper now starts near 1.2, a clear step above land_depth 0.35
    assert float(thk[sel].min()) > cfg.land_depth_mm + 0.5


def test_taper2_near_makes_land_step_in_second_stage_region() -> None:
    """taper2_near > land_depth steps up right after the land in the 2nd-stage
    region (x >= x_2nd)."""
    cfg = _default_cfg(
        gate_position_mm=0.0,
        gate_left_offset_mm=40.0,
        land_depth_mm=0.35,
        taper2_near_depth_mm=0.9,
        taper2_right_mm=14.0,
        taper2_left_mm=10.0,
        gate_depth_mm=40.0,
    )
    g = build_film_gate2_geometry(cfg)
    yy, xx, y_long, x_g = _grid(cfg, g.mask.shape)
    x_2nd = cfg.pad_mm + cfg.gate_left_offset_mm
    t = y_long - yy
    thk = g.thickness_mm
    band = g.mask & (t > cfg.land_width_mm) & (t < cfg.land_width_mm + 1.0)
    sel = band & (xx > x_2nd + 8.0) & (xx < x_g - 5.0) & (thk < cfg.runner_depth_mm - 0.5)
    assert sel.any()
    assert float(thk[sel].min()) > cfg.land_depth_mm + 0.4  # step up to ~0.9


def test_validation_rejects_taper_near_exceeding_runner_depth() -> None:
    with pytest.raises(ValueError):
        build_film_gate2_geometry(_default_cfg(taper2_near_depth_mm=5.0, runner_depth_mm=3.0))
    with pytest.raises(ValueError):
        build_film_gate2_geometry(_default_cfg(taper1_near_depth_mm=5.0, runner_depth_mm=3.0))


def test_validation_rejects_nonpositive_taper_near() -> None:
    with pytest.raises(ValueError):
        build_film_gate2_geometry(_default_cfg(taper2_near_depth_mm=-0.1))
    with pytest.raises(ValueError):
        build_film_gate2_geometry(_default_cfg(taper1_near_depth_mm=0.0))


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


def test_validation_rejects_combined_taper_extent_exceeding_depth() -> None:
    """The 1st taper sits after the 2nd taper, so its far endpoint is
    taper2_far + L1. gate_depth=10 with taper2_right=10 and taper1_len=8 gives
    extent 18 > 10 and must be rejected (else the 1st taper is clipped by the
    silhouette and never reaches mid_a)."""
    with pytest.raises(ValueError):
        build_film_gate2_geometry(
            _default_cfg(gate_depth_mm=10.0, taper2_right_mm=10.0, taper1_len_mm=8.0)
        )


def test_validation_rejects_mid_depth_exceeding_runner_depth() -> None:
    """The deep runner must stay the deepest channel: mid_a (or mid_b) above
    runner_depth would let the post-taper floor overwrite the runner."""
    with pytest.raises(ValueError):
        build_film_gate2_geometry(_default_cfg(mid_depth_a_mm=5.0, runner_depth_mm=2.0))
    with pytest.raises(ValueError):
        build_film_gate2_geometry(_default_cfg(mid_depth_b_mm=4.0, runner_depth_mm=2.0))


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
