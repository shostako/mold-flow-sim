"""Tests for the two-phase (injection + compression) short-shot model."""

from __future__ import annotations

import numpy as np
import pytest

from core.geometry import FilmGateConfig, Geometry, build_film_gate_geometry
from core.materials import MaterialDB
from core.solver import HeleShawSolver
from core.two_phase import solve_two_phase_short_shot
from core.visualizer import render_two_phase_map

DB = MaterialDB()
PP = DB.get("PP")
T_MELT = sum(PP.T_melt_recommended) / 2
T_MOLD = sum(PP.T_mold_recommended) / 2


def _strip(n: int = 40, h_mm: float = 1.0, dx: float = 1.0) -> Geometry:
    """1 x n uniform strip with a gate at the left end."""
    mask = np.ones((1, n), dtype=bool)
    thickness = np.full((1, n), h_mm)
    geom = Geometry(mask=mask, thickness_mm=thickness, cell_size_mm=dx)
    geom.add_gate(0, 0)
    return geom


def _solver(geom: Geometry, *, stroke: float | None = 0.5, Q: float = 10.0) -> HeleShawSolver:
    return HeleShawSolver(
        geom,
        PP,
        melt_temperature_K=T_MELT,
        mold_temperature_K=T_MOLD,
        injection_volume_flow_cm3s=Q,
        compression_molding=stroke is not None,
        compression_stroke_mm=stroke,
    )


def _film_gate(stepped: bool = True) -> Geometry:
    cfg = FilmGateConfig(
        plate_w_mm=60.0,
        plate_h_mm=20.0,
        plate_thk_mm=0.5,
        runner_long_mm=50.0,
        runner_short_diameter_mm=6.0,
        runner_depth_mm=8.0,
        runner_thk_mm=2.0,
        runner_flat_depth_mm=4.0,
        runner_slope_depth_mm=4.0,
        valve_gate_diameter_mm=3.0,
        gate_width_mm=44.0,
        cell_size_mm=1.0,
        plate_split_height_mm=8.0 if stepped else 0.0,
        plate_lower_thk_mm=0.35 if stepped else None,
        plate_upper_thk_mm=0.50 if stepped else None,
    )
    return build_film_gate_geometry(cfg)


# ---------------------------------------------------------------------------
# invariants
# ---------------------------------------------------------------------------


def test_the_injection_pool_is_inside_the_final_shape():
    geom = _film_gate()
    solver = _solver(geom)
    V_open = geom.volume_cm3() + 0.5 * geom.compression_area_mm2() / 1000.0
    res = solve_two_phase_short_shot(solver, 0.5 * V_open)
    assert (res.injection_mask <= res.final_mask).all()
    assert res.injection_mask.any()
    assert not res.metadata["final_complete"]


def test_the_final_shape_holds_exactly_the_metered_volume():
    """Volume conservation: the melt squeezed to final thickness covers a
    region whose final-thickness volume matches V_shot to tie-group
    granularity (never exceeding it)."""
    geom = _film_gate()
    solver = _solver(geom)
    dx = geom.cell_size_mm
    vol_fin = dx * dx * geom.thickness_mm
    V_shot = 0.45 * geom.volume_cm3()
    res = solve_two_phase_short_shot(solver, V_shot)
    achieved_mm3 = float(vol_fin[res.final_mask].sum())
    assert achieved_mm3 <= V_shot * 1000.0 * (1 + 1e-9)
    # The shortfall is bounded by the largest tie group at the cut; on this
    # geometry that is a handful of cells, use a generous 20-cell bound.
    max_cell = float(vol_fin[geom.mask].max())
    assert V_shot * 1000.0 - achieved_mm3 < 20 * max_cell


def test_injection_pools_nest_across_shot_volumes():
    """Omega1 is a prefix of one fixed tau1 order, so nesting is exact.
    Omega2 carries NO such guarantee (each shot solves tau2 against its own
    pool boundary — see the module docstring / Codex P2 round 3), so only the
    final-mask behaviour on this particular geometry is pinned as a
    regression, not asserted as an invariant."""
    geom = _film_gate()
    solver = _solver(geom)
    V = geom.volume_cm3()
    small = solve_two_phase_short_shot(solver, 0.3 * V)
    large = solve_two_phase_short_shot(solver, 0.6 * V)
    assert (small.injection_mask <= large.injection_mask).all()
    # Regression pin on this geometry (holds here; not a model guarantee).
    assert (small.final_mask <= large.final_mask).all()


def test_a_full_shot_fills_the_whole_cavity():
    geom = _film_gate()
    solver = _solver(geom)
    res = solve_two_phase_short_shot(solver, 2.0 * geom.volume_cm3())
    assert (res.final_mask == geom.mask).all()
    assert res.metadata["final_complete"]


def test_gates_are_always_inside_the_injection_pool():
    geom = _film_gate()
    solver = _solver(geom)
    # The gate cells are the tau = 0 tie group: any accepted shot covers them.
    gate = np.zeros(geom.shape, dtype=bool)
    for iy, ix in geom.gates:
        gate[iy, ix] = True
    h_open = solver._open_thickness_field()
    v_gate_cm3 = float((geom.cell_size_mm**2 * h_open)[gate].sum()) / 1000.0
    res = solve_two_phase_short_shot(solver, v_gate_cm3 * 1.05)
    for iy, ix in geom.gates:
        assert res.injection_mask[iy, ix]


def test_a_shot_smaller_than_the_gate_region_is_rejected():
    """Forcing the gates in regardless would report an achieved volume larger
    than the metered shot (Codex P2 on PR #62) — so it is an input error."""
    geom = _film_gate()
    solver = _solver(geom)
    with pytest.raises(ValueError, match="gate region"):
        solve_two_phase_short_shot(solver, 1e-4)


def test_arrival_times_are_nan_beyond_the_pool():
    geom = _film_gate()
    solver = _solver(geom)
    res = solve_two_phase_short_shot(solver, 0.4 * geom.volume_cm3())
    inside = res.injection_mask
    assert np.isfinite(res.injection_fill_time_s[inside]).all()
    assert np.isnan(res.injection_fill_time_s[~inside]).all()


# ---------------------------------------------------------------------------
# analytic checks on the uniform strip
# ---------------------------------------------------------------------------


def test_strip_front_positions_are_analytic():
    """On a uniform strip the front position is pure arithmetic: n1 cells at
    the open gap during injection, n2 = n1 * h_open / h_fin after the
    squeeze."""
    n, h, dx, stroke = 40, 1.0, 1.0, 0.5
    geom = _strip(n, h, dx)
    solver = _solver(geom, stroke=stroke)
    h_open = h + stroke
    n1 = 16
    V_shot = n1 * dx * dx * h_open / 1000.0  # cm^3
    res = solve_two_phase_short_shot(solver, V_shot)
    assert int(res.injection_mask.sum()) == n1
    # the pool is the contiguous prefix from the gate
    assert res.injection_mask[0, :n1].all() and not res.injection_mask[0, n1:].any()
    n2 = int(round(n1 * h_open / h))  # 24
    assert int(res.final_mask.sum()) == n2
    assert res.final_mask[0, :n2].all() and not res.final_mask[0, n2:].any()


def test_strip_arrival_clock_is_volume_linear():
    n, h, dx = 30, 1.0, 1.0
    geom = _strip(n, h, dx)
    solver = _solver(geom, stroke=None, Q=5.0)
    V_shot = 20 * dx * dx * h / 1000.0
    res = solve_two_phase_short_shot(solver, V_shot)
    t = res.injection_fill_time_s[0]
    cell_t = dx * dx * h / 1000.0 / 5.0  # seconds per cell at Q=5
    for k in range(20):
        assert t[k] == pytest.approx((k + 1) * cell_t, rel=1e-9)


def test_compression_advance_is_contiguous_ahead_of_the_pool():
    geom = _strip(50, 1.0, 1.0)
    solver = _solver(geom, stroke=1.0)
    V_shot = 10 * 2.0 / 1000.0  # 10 cells at h_open = 2.0
    res = solve_two_phase_short_shot(solver, V_shot)
    advanced = res.final_mask & ~res.injection_mask
    idx = np.where(advanced[0])[0]
    assert idx.size > 0
    assert idx[0] == int(res.injection_mask.sum())  # starts right at the front
    assert np.array_equal(idx, np.arange(idx[0], idx[0] + idx.size))  # contiguous


def test_compression_progress_is_monotone_and_ends_at_one():
    geom = _strip(50, 1.0, 1.0)
    solver = _solver(geom, stroke=1.0)
    res = solve_two_phase_short_shot(solver, 10 * 2.0 / 1000.0)
    advanced = res.final_mask & ~res.injection_mask
    prog = res.compression_progress[advanced]
    assert np.isfinite(prog).all()
    assert (np.diff(prog) > 0).all()  # strip: strictly increasing outward
    assert prog[-1] == pytest.approx(1.0, abs=1e-9)
    assert np.isnan(res.compression_progress[~advanced]).all()


def test_a_final_complete_icm_shot_still_reports_the_compression_order():
    """V_fin <= V_shot < V_open — the everyday ICM full-fill case: injection
    stops short of the open cavity, closure finishes the fill. The shape is
    trivially the whole cavity, but the result contract still promises the
    normalized advance order on the compression-filled cells (Codex P2,
    round 3)."""
    n, h, stroke = 40, 1.0, 1.0
    geom = _strip(n, h)
    solver = _solver(geom, stroke=stroke)
    V_fin = n * h / 1000.0
    V_open = n * (h + stroke) / 1000.0
    V_shot = 0.5 * (V_fin + V_open)  # strictly between
    res = solve_two_phase_short_shot(solver, V_shot)
    assert res.metadata["final_complete"]
    assert not res.metadata["injection_complete"]
    assert (res.final_mask == geom.mask).all()
    assert res.tau2 is not None
    advanced = res.final_mask & ~res.injection_mask
    assert advanced.any()
    prog = res.compression_progress[advanced]
    assert np.isfinite(prog).all()
    assert (np.diff(prog) > 0).all()  # strip: strictly increasing outward
    assert prog[-1] == pytest.approx(1.0, abs=1e-9)
    assert np.isnan(res.compression_progress[res.injection_mask]).all()


# ---------------------------------------------------------------------------
# degradations and rejections
# ---------------------------------------------------------------------------


def test_no_compression_means_no_advance():
    geom = _strip(40)
    solver = _solver(geom, stroke=None)
    res = solve_two_phase_short_shot(solver, 15 * 1.0 / 1000.0)
    assert (res.final_mask == res.injection_mask).all()
    assert res.tau2 is None
    assert res.metadata["compression_mode"] == "off"


def test_zero_stroke_means_no_advance():
    geom = _strip(40)
    solver = _solver(geom, stroke=0.0)
    res = solve_two_phase_short_shot(solver, 15 * 1.0 / 1000.0)
    assert (res.final_mask == res.injection_mask).all()


def test_no_closure_skips_phase_two_even_with_residual_budget():
    """Codex P2 on PR #62: with no gap change, the atomic phase-1 cutoff can
    leave a residual budget, and on a nonuniform cavity the re-solved tau2
    ordering could hand it to some smaller cell. Phase 2 must be gated on the
    physics (the gap actually closing), so tau2 is never even solved."""
    mask = np.ones((1, 30), dtype=bool)
    thickness = np.full((1, 30), 1.0)
    thickness[0, 10:] = 0.25  # thin far region: cells smaller than the residual
    geom = Geometry(mask=mask, thickness_mm=thickness, cell_size_mm=1.0)
    geom.add_gate(0, 0)
    solver = _solver(geom, stroke=None)
    # 5 full cells + a residual of 0.5 mm^3 that a 0.25 mm^3 thin cell would fit
    res = solve_two_phase_short_shot(solver, 5.5 / 1000.0)
    assert int(res.injection_mask.sum()) == 5
    assert (res.final_mask == res.injection_mask).all()
    assert res.tau2 is None


def test_factor_mode_is_accepted():
    geom = _film_gate()
    solver = HeleShawSolver(
        geom,
        PP,
        melt_temperature_K=T_MELT,
        mold_temperature_K=T_MOLD,
        injection_volume_flow_cm3s=10.0,
        compression_molding=True,
        compression_factor=1.5,
    )
    res = solve_two_phase_short_shot(solver, 0.5 * geom.volume_cm3())
    assert res.metadata["compression_mode"] == "factor"
    assert (res.final_mask.sum()) > (res.injection_mask.sum())


def test_the_skin_layer_model_is_rejected():
    geom = _strip(20)
    solver = HeleShawSolver(
        geom,
        PP,
        melt_temperature_K=T_MELT,
        mold_temperature_K=T_MOLD,
        skin_layer_enabled=True,
    )
    with pytest.raises(ValueError, match="skin-layer"):
        solve_two_phase_short_shot(solver, 0.01)


def test_a_nonpositive_shot_volume_is_rejected():
    geom = _strip(20)
    solver = _solver(geom)
    with pytest.raises(ValueError, match="positive"):
        solve_two_phase_short_shot(solver, 0.0)


def test_gap_shrinking_compression_is_rejected():
    """Codex P2 (round 2) on PR #62: factor < 1 or a negative stroke makes
    h_open < h_fin, so a full open-cavity shot holds less than the final
    volume and the achieved volume exceeds the metered shot."""
    geom = _strip(20)
    neg_stroke = _solver(geom, stroke=-0.2)
    with pytest.raises(ValueError, match="open gap"):
        solve_two_phase_short_shot(neg_stroke, 0.01)
    shrink_factor = HeleShawSolver(
        geom,
        PP,
        melt_temperature_K=T_MELT,
        mold_temperature_K=T_MOLD,
        injection_volume_flow_cm3s=10.0,
        compression_molding=True,
        compression_factor=0.8,
    )
    with pytest.raises(ValueError, match="open gap"):
        solve_two_phase_short_shot(shrink_factor, 0.01)


def test_a_gateless_geometry_is_rejected():
    mask = np.ones((1, 10), dtype=bool)
    geom = Geometry(mask=mask, thickness_mm=np.full((1, 10), 1.0), cell_size_mm=1.0)
    solver = _solver(geom)
    with pytest.raises(ValueError, match="gate"):
        solve_two_phase_short_shot(solver, 0.01)


# ---------------------------------------------------------------------------
# metadata contract
# ---------------------------------------------------------------------------


def test_metadata_carries_the_run_conditions():
    geom = _film_gate()
    solver = _solver(geom, stroke=0.5, Q=589.0)
    V_shot = 0.5 * geom.volume_cm3()
    res = solve_two_phase_short_shot(solver, V_shot)
    md = res.metadata
    assert md["model"] == "two_phase_short_shot"
    assert md["shot_volume_cm3"] == pytest.approx(V_shot)
    assert md["flow_rate_cm3s"] == pytest.approx(589.0)
    assert md["compression_mode"] == "stroke"
    assert md["compression_stroke_mm"] == pytest.approx(0.5)
    assert 0.0 < md["injection_fill_fraction"] <= 1.0
    assert 0.0 < md["final_fill_fraction"] <= 1.0
    assert md["injection_fill_fraction"] <= md["final_fill_fraction"] + 1e-12
    assert res.injection_time_s == pytest.approx(V_shot / 589.0)


# ---------------------------------------------------------------------------
# visualization
# ---------------------------------------------------------------------------


def test_the_map_renders_for_a_partial_and_a_complete_shot(tmp_path):
    geom = _film_gate()
    solver = _solver(geom)
    partial = solve_two_phase_short_shot(solver, 0.4 * geom.volume_cm3())
    complete = solve_two_phase_short_shot(solver, 3.0 * geom.volume_cm3())
    p1 = render_two_phase_map(partial, tmp_path / "partial.png")
    p2 = render_two_phase_map(complete, tmp_path / "complete.png")
    assert p1.exists() and p1.stat().st_size > 0
    assert p2.exists() and p2.stat().st_size > 0


def test_the_map_colors_expose_the_two_regions(tmp_path):
    """The drawn image must contain both categorical colors for a stroke
    run with a partial shot — a map that silently drops the compression
    layer would still 'render'."""
    from PIL import Image

    from core.visualizer import TWO_PHASE_COMPRESSION_RGB, TWO_PHASE_INJECTION_RGB

    geom = _film_gate()
    solver = _solver(geom)
    res = solve_two_phase_short_shot(solver, 0.4 * geom.volume_cm3())
    path = render_two_phase_map(res, tmp_path / "map.png")
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=float) / 255.0

    def _count(rgb):
        dist = np.abs(arr - np.array(rgb)).sum(axis=-1)
        return int((dist < 0.08).sum())

    n_blue = _count(TWO_PHASE_INJECTION_RGB)
    n_orange = _count(TWO_PHASE_COMPRESSION_RGB)
    assert n_blue > 0 and n_orange > 0
    # The legend patch alone supplies a few hundred pixels of each color, so
    # "the color exists" is a vacuous check. The drawn *regions* must supply
    # pixels in rough proportion to their cell counts — painting the
    # compression zone with the injection color leaves orange at
    # legend-patch scale and fails this.
    cells_inj = int(res.injection_mask.sum())
    cells_adv = int((res.final_mask & ~res.injection_mask).sum())
    assert cells_adv > 0
    assert n_orange / n_blue > 0.3 * cells_adv / cells_inj


# ---------------------------------------------------------------------------
# tie atomicity and the equipotential-pool contract
# ---------------------------------------------------------------------------


def _twin_strip(n: int = 30, h_mm: float = 1.0) -> Geometry:
    """2 x n strip with a gate in each row: every column is a tau tie pair."""
    mask = np.ones((2, n), dtype=bool)
    geom = Geometry(mask=mask, thickness_mm=np.full((2, n), h_mm), cell_size_mm=1.0)
    geom.add_gate(0, 0)
    geom.add_gate(1, 0)
    return geom


def test_tie_groups_are_atomic_at_both_cuts():
    """A budget of 5.5 columns must stop at 5 whole columns, not 11 half
    cells — a tie group (one column) is either fully molten or fully empty."""
    geom = _twin_strip()
    solver = _solver(geom, stroke=1.0)
    h_open = 2.0
    V_shot = 5.5 * 2 * h_open / 1000.0  # 5.5 columns' worth at the open gap
    res = solve_two_phase_short_shot(solver, V_shot)
    col1 = res.injection_mask.sum(axis=0)
    assert set(np.unique(col1)) <= {0, 2}, "injection cut split a tie column"
    col2 = res.final_mask.sum(axis=0)
    assert set(np.unique(col2)) <= {0, 2}, "compression cut split a tie column"
    assert int(res.injection_mask.sum()) == 10  # 5 whole columns


def test_the_melt_pool_is_an_equipotential_source():
    """Phase 2 pins tau2 = 0 on ALL of Omega1, not just the gates — the pool
    conductance is neglected relative to the unfilled front's."""
    geom = _film_gate()
    solver = _solver(geom)
    res = solve_two_phase_short_shot(solver, 0.4 * geom.volume_cm3())
    assert res.tau2 is not None
    assert np.nanmax(np.abs(res.tau2[res.injection_mask])) == pytest.approx(0.0, abs=1e-12)
    beyond = geom.mask & ~res.injection_mask
    assert (res.tau2[beyond] > 0).all()


# ---------------------------------------------------------------------------
# animation frame states and figure layout
# ---------------------------------------------------------------------------


def _stroked_strip_result(n=40, h=1.0, stroke=1.0, cells=10):
    geom = _strip(n, h)
    solver = _solver(geom, stroke=stroke)
    return solve_two_phase_short_shot(solver, cells * (h + stroke) / 1000.0)


def test_frame_states_grow_monotonically_to_the_final_mask():
    from core.two_phase import frame_states

    res = _stroked_strip_result()
    frames = frame_states(res, num_frames=24)
    assert len(frames) == 24
    prev = np.zeros(res.geometry.shape, dtype=bool)
    for fr in frames:
        filled = fr.injection_filled | fr.compression_filled
        assert (prev <= filled).all(), "a frame lost previously filled cells"
        prev = filled
    assert (prev == res.final_mask).all(), "last frame must cover exactly final_mask"
    first = frames[0].injection_filled | frames[0].compression_filled
    assert first.sum() < res.final_mask.sum() / 4, "first frame should be nearly empty"


def test_frame_states_split_the_phases():
    from core.two_phase import frame_states

    res = _stroked_strip_result()
    frames = frame_states(res, num_frames=24)
    phases = [fr.phase for fr in frames]
    n_inj = phases.count("injection")
    n_comp = phases.count("compression")
    assert n_inj >= 3 and n_comp >= 3
    assert phases == ["injection"] * n_inj + ["compression"] * n_comp
    # injection frames advance the real clock to T_inj; compression to 1.0
    inj_vals = [fr.value for fr in frames if fr.phase == "injection"]
    assert inj_vals[0] == 0.0
    assert inj_vals[-1] == pytest.approx(res.injection_time_s)
    comp_vals = [fr.value for fr in frames if fr.phase == "compression"]
    assert comp_vals[-1] == pytest.approx(1.0)
    # compression frames never touch cells outside the advance zone
    adv = res.final_mask & ~res.injection_mask
    for fr in frames:
        assert not (fr.compression_filled & ~adv).any()


def test_frame_states_without_advance_are_injection_only():
    from core.two_phase import frame_states

    geom = _strip(40)
    solver = _solver(geom, stroke=None)
    res = solve_two_phase_short_shot(solver, 15 * 1.0 / 1000.0)
    frames = frame_states(res, num_frames=12)
    assert all(fr.phase == "injection" for fr in frames)
    assert not any(fr.compression_filled.any() for fr in frames)


def test_frame_states_validate_num_frames_and_survive_small_budgets():
    from core.two_phase import frame_states

    res = _stroked_strip_result()
    with pytest.raises(ValueError, match="num_frames"):
        frame_states(res, num_frames=1)
    frames = frame_states(res, num_frames=4)  # both phases active, tiny budget
    assert len(frames) == 4
    filled_last = frames[-1].injection_filled | frames[-1].compression_filled
    assert (filled_last == res.final_mask).all()


def test_a_single_injection_frame_shows_the_completed_pool():
    """Codex P2 on PR #63: np.linspace(0, T, 1) == [0.0], so a one-frame
    injection phase used to show an EMPTY cavity and then jump straight to
    the final mask — the documented injection endpoint at T_inj never
    appeared. A lone injection frame must show the phase's end state."""
    from core.two_phase import frame_states

    res = _stroked_strip_result()
    hit_single = False
    for nf in (2, 4):  # this fixture allocates exactly one injection frame
        frames = frame_states(res, num_frames=nf)
        inj_frames = [fr for fr in frames if fr.phase == "injection"]
        if len(inj_frames) == 1:
            hit_single = True
            only = inj_frames[0]
            assert only.value == pytest.approx(res.injection_time_s)
            assert (only.injection_filled == res.injection_mask).all()
    assert hit_single, "fixture no longer produces a single-frame injection phase"


def test_the_map_legend_sits_outside_the_axes():
    """The plates are wide and shallow: an in-axes legend lands on the part
    (it covered the far corner of the first real render). The legend must be
    a figure-level artist whose box does not overlap the axes box."""
    import matplotlib.pyplot as plt

    from core.visualizer import _build_two_phase_figure

    res = _stroked_strip_result()
    fig, ax = _build_two_phase_figure(res)
    try:
        assert not ax.get_legend(), "legend must not be attached to the axes"
        assert fig.legends, "figure-level legend missing"
        fig.canvas.draw()
        leg_box = fig.legends[0].get_window_extent()
        ax_box = ax.get_window_extent()
        assert not leg_box.overlaps(ax_box), "legend overlaps the plot area"
    finally:
        plt.close(fig)


def test_the_animation_renders_the_requested_frames(tmp_path):
    from PIL import Image

    from core.visualizer import render_two_phase_animation

    res = _stroked_strip_result()
    path = render_two_phase_animation(res, tmp_path / "two_phase.gif", num_frames=10, fps=8)
    assert path.exists() and path.stat().st_size > 0
    with Image.open(path) as im:
        assert im.n_frames == 10
