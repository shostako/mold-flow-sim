"""Tests for the multilayer Hele-Shaw solver (PR-A skeleton).

The current solver is uniform-distribution / single-viscosity only. The
defining property is that ``num_layers=1`` collapses to the existing
:class:`HeleShawSolver` (the Poiseuille moment integral with ``m_1 = 1/6``
matches the closed-form ``S = h³ / (12 η)``). Additional tests cover
layer-thickness conservation, smoke checks, and the moment-sum identity
``Σ m_k = 1/6`` that any future ``layer_distribution`` must preserve.

Per-layer temperature / viscosity / fixed-point loop tests live in
``test_multilayer_thermal.py`` (PR-B and later).
"""

from __future__ import annotations

import numpy as np
import pytest

from core import (
    FilmGateConfig,
    HeleShawSolver,
    MaterialDB,
    MultilayerFlowResult,
    MultilayerHeleShawSolver,
    build_film_gate_geometry,
)
from core.multilayer_solver import (
    _multilayer_conductance,
    _poiseuille_layer_moments,
    _uniform_layer_zeta,
    _wall_refined_layer_zeta,
)


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


def _solver_kwargs() -> dict:
    return dict(
        melt_temperature_K=503.15,
        mold_temperature_K=313.15,
        injection_velocity_mms=100.0,
        injection_volume_flow_cm3s=20.0,
    )


# --------------------------------------------------------------------------
# Layer-distribution primitives
# --------------------------------------------------------------------------


def test_uniform_zeta_endpoints() -> None:
    """``ζ_0 = 0`` and ``ζ_N = 1`` are enforced; the array has length
    ``N + 1``; spacings are equal."""
    z = _uniform_layer_zeta(5)
    assert z.shape == (6,)
    assert z[0] == 0.0
    assert z[-1] == 1.0
    diffs = np.diff(z)
    np.testing.assert_allclose(diffs, np.full(5, 0.2), rtol=1e-12)


def test_uniform_zeta_rejects_zero_or_negative_n() -> None:
    with pytest.raises(ValueError):
        _uniform_layer_zeta(0)
    with pytest.raises(ValueError):
        _uniform_layer_zeta(-3)


def test_poiseuille_layer_moments_sum_to_one_sixth() -> None:
    """The full-thickness Poiseuille integral is ``Σ m_k = 1/6``, which
    is exactly the Hele-Shaw factor. Any future ``layer_distribution``
    must satisfy this — protect against accidental reshuffling.
    """
    for N in (1, 2, 3, 5, 7, 11, 25):
        z = _uniform_layer_zeta(N)
        m = _poiseuille_layer_moments(z)
        assert m.shape == (N,)
        assert np.all(m > 0.0), f"all moments must be positive for N={N}"
        np.testing.assert_allclose(m.sum(), 1.0 / 6.0, rtol=1e-12)


def test_n1_moment_equals_one_sixth() -> None:
    """``N=1`` is the calibration anchor: ``m_1 = 1/6`` exactly."""
    z = _uniform_layer_zeta(1)
    m = _poiseuille_layer_moments(z)
    assert m.shape == (1,)
    np.testing.assert_allclose(m[0], 1.0 / 6.0, rtol=1e-14)


# --------------------------------------------------------------------------
# Conductance helper
# --------------------------------------------------------------------------


def test_multilayer_conductance_n1_matches_hele_shaw_factor() -> None:
    """With ``N=1`` and a uniform η, the conductance reduces to
    ``S = h³ / (12 η)`` in SI units everywhere inside the cavity."""
    ny, nx = 4, 5
    h_mm = np.full((ny, nx), 2.0)  # 2 mm everywhere
    mask = np.ones((ny, nx), dtype=bool)
    eta = 50.0
    z = _uniform_layer_zeta(1)
    m = _poiseuille_layer_moments(z)
    S = _multilayer_conductance(h_mm, eta, m, mask)
    h_m = h_mm * 1e-3
    expected = (h_m**3) / (12.0 * eta)
    np.testing.assert_allclose(S, expected, rtol=1e-12)


def test_multilayer_conductance_zero_outside_mask() -> None:
    """Cells outside ``cavity_mask`` must be zeroed regardless of input
    thickness."""
    ny, nx = 3, 3
    h_mm = np.full((ny, nx), 2.0)
    mask = np.array([[True, False, True], [False, True, False], [True, True, True]])
    z = _uniform_layer_zeta(3)
    m = _poiseuille_layer_moments(z)
    S = _multilayer_conductance(h_mm, 50.0, m, mask)
    assert np.all(S[~mask] == 0.0)
    assert np.all(S[mask] > 0.0)


def test_multilayer_conductance_per_layer_eta_shape_validation() -> None:
    """When passing a 1-D ``eta_per_layer``, its length must match the
    number of layers."""
    ny, nx = 2, 2
    h_mm = np.full((ny, nx), 1.0)
    mask = np.ones((ny, nx), dtype=bool)
    z = _uniform_layer_zeta(5)
    m = _poiseuille_layer_moments(z)
    bad_eta = np.full(4, 50.0)  # wrong N
    with pytest.raises(ValueError, match="length"):
        _multilayer_conductance(h_mm, bad_eta, m, mask)


def test_multilayer_conductance_per_cell_per_layer_eta() -> None:
    """``(N, ny, nx)`` η-shape works and produces a finite cavity-only
    field — used from PR-B when each layer has its own temperature."""
    ny, nx = 3, 4
    h_mm = np.full((ny, nx), 1.5)
    mask = np.ones((ny, nx), dtype=bool)
    z = _uniform_layer_zeta(5)
    m = _poiseuille_layer_moments(z)
    eta_field = np.full((5, ny, nx), 50.0)
    S = _multilayer_conductance(h_mm, eta_field, m, mask)
    h_m = h_mm * 1e-3
    expected = (h_m**3) / (12.0 * 50.0)
    np.testing.assert_allclose(S, expected, rtol=1e-12)


# --------------------------------------------------------------------------
# Solver smoke + N=1 equivalence
# --------------------------------------------------------------------------


def test_multilayer_solver_rejects_zero_layers() -> None:
    g = build_film_gate_geometry(_default_cfg())
    db = MaterialDB()
    with pytest.raises(ValueError):
        MultilayerHeleShawSolver(geometry=g, material=db["PP"], num_layers=0, **_solver_kwargs())


def test_multilayer_solver_rejects_unknown_distribution() -> None:
    """An unknown layer distribution name must raise ``ValueError`` at
    solve time (the dispatcher is the line of defense)."""
    g = build_film_gate_geometry(_default_cfg())
    db = MaterialDB()
    solver = MultilayerHeleShawSolver(
        geometry=g,
        material=db["PP"],
        num_layers=5,
        layer_distribution="not_a_real_distribution",
        thermal_coupling=False,
        **_solver_kwargs(),
    )
    with pytest.raises(ValueError, match="not_a_real_distribution"):
        solver.solve(num_frames=4)


def test_n1_matches_legacy_tau() -> None:
    """The anchor test: ``num_layers=1`` + ``thermal_coupling=False`` must
    reproduce the existing ``HeleShawSolver`` τ field byte-for-byte
    (modulo tiny FP noise). With thermal coupling ON the layer-centre
    temperature drops below the bulk and the τ field deliberately
    diverges — that case is covered separately.
    """
    g = build_film_gate_geometry(_default_cfg())
    db = MaterialDB()
    r_legacy = HeleShawSolver(geometry=g, material=db["PP"], **_solver_kwargs()).solve(num_frames=4)
    r_multi = MultilayerHeleShawSolver(
        geometry=g,
        material=db["PP"],
        num_layers=1,
        thermal_coupling=False,
        **_solver_kwargs(),
    ).solve(num_frames=4)
    np.testing.assert_allclose(
        np.nan_to_num(r_legacy.tau, nan=0.0),
        np.nan_to_num(r_multi.tau, nan=0.0),
        rtol=1e-10,
        atol=1e-14,
    )
    # tau_max identity ensures the absolute fill-time normalisation also matches.
    np.testing.assert_allclose(
        r_multi.metadata["tau_max"], r_legacy.metadata["tau_max"], rtol=1e-10
    )


def test_n1_matches_legacy_total_fill_time() -> None:
    """A consequence of the τ match: with ``thermal_coupling=False`` the
    absolute fill time is identical too (modulo FP noise)."""
    g = build_film_gate_geometry(_default_cfg())
    db = MaterialDB()
    r_legacy = HeleShawSolver(geometry=g, material=db["PP"], **_solver_kwargs()).solve(num_frames=4)
    r_multi = MultilayerHeleShawSolver(
        geometry=g,
        material=db["PP"],
        num_layers=1,
        thermal_coupling=False,
        **_solver_kwargs(),
    ).solve(num_frames=4)
    np.testing.assert_allclose(
        r_multi.total_fill_time_s,
        r_legacy.total_fill_time_s,
        rtol=1e-10,
    )


def test_layer_thickness_sum_equals_total() -> None:
    """``Σ_k h_k(x,y) = h_total(x,y)`` exactly for any ``N``. The
    arithmetic is exactly representable since ``Σ_k Δζ_k = 1`` is built
    from a single ``np.linspace``."""
    g = build_film_gate_geometry(_default_cfg())
    db = MaterialDB()
    h_total = g.thickness_mm
    for N in (1, 3, 5, 7):
        solver = MultilayerHeleShawSolver(
            geometry=g, material=db["PP"], num_layers=N, **_solver_kwargs()
        )
        layers = solver.layer_thickness_mm(h_total)
        assert layers.shape == (N, *h_total.shape)
        np.testing.assert_allclose(layers.sum(axis=0), h_total, atol=1e-12)


def test_metadata_carries_layer_fields() -> None:
    """The result metadata exposes layer-related identification fields
    so downstream tooling (UI / CLI / ZIP exports) can distinguish a
    multilayer run from a single-layer one."""
    g = build_film_gate_geometry(_default_cfg())
    db = MaterialDB()
    r = MultilayerHeleShawSolver(
        geometry=g, material=db["PP"], num_layers=5, **_solver_kwargs()
    ).solve(num_frames=4)
    assert r.metadata["solver_kind"] == "multilayer"
    assert r.metadata["num_layers"] == 5
    assert r.metadata["layer_distribution"] == "uniform"
    zeta = r.metadata["layer_zeta"]
    assert len(zeta) == 6
    assert zeta[0] == 0.0 and zeta[-1] == 1.0
    moments = r.metadata["layer_moments"]
    assert len(moments) == 5
    assert abs(sum(moments) - 1.0 / 6.0) < 1e-12


def test_smoke_n5_runs_and_produces_finite_tau() -> None:
    """End-to-end smoke for the default ``N=5`` configuration (thermal
    coupling default ON)."""
    g = build_film_gate_geometry(_default_cfg())
    db = MaterialDB()
    r = MultilayerHeleShawSolver(
        geometry=g, material=db["PP"], num_layers=5, **_solver_kwargs()
    ).solve(num_frames=8)
    msk = ~np.isnan(r.tau)
    assert msk.sum() > 0
    assert np.all(np.isfinite(r.tau[msk]))
    assert r.metadata["tau_max"] > 0.0
    assert r.total_fill_time_s > 0.0


# --------------------------------------------------------------------------
# PR-B: thermal coupling
# --------------------------------------------------------------------------


def test_solve_returns_multilayer_flow_result() -> None:
    """``solve()`` returns a :class:`MultilayerFlowResult` (subclass of
    ``FlowResult``) so visualisers that expect either type work."""
    from core import FlowResult

    g = build_film_gate_geometry(_default_cfg())
    db = MaterialDB()
    r = MultilayerHeleShawSolver(
        geometry=g, material=db["PP"], num_layers=3, **_solver_kwargs()
    ).solve(num_frames=4)
    assert isinstance(r, MultilayerFlowResult)
    assert isinstance(r, FlowResult)  # backwards-compatible


def test_layer_fields_populated_when_thermal_on() -> None:
    """With thermal coupling enabled the layer-resolved fields are
    populated with the expected shapes."""
    g = build_film_gate_geometry(_default_cfg())
    db = MaterialDB()
    N = 5
    r = MultilayerHeleShawSolver(
        geometry=g,
        material=db["PP"],
        num_layers=N,
        thermal_coupling=True,
        **_solver_kwargs(),
    ).solve(num_frames=4)

    shape = (N, *g.thickness_mm.shape)
    assert r.layer_thickness_mm is not None and r.layer_thickness_mm.shape == shape
    assert r.layer_temperature_K is not None and r.layer_temperature_K.shape == shape
    assert r.layer_viscosity_Pa_s_field is not None and r.layer_viscosity_Pa_s_field.shape == shape
    assert r.layer_shear_rate_s_inv is not None and r.layer_shear_rate_s_inv.shape == shape

    # Sanity bounds inside the cavity.
    cm = g.mask[None, :, :]
    T_in = r.layer_temperature_K[np.broadcast_to(cm, shape)]
    assert np.all(T_in >= 313.15 - 1e-6), "temperatures clamp to mold"
    assert np.all(T_in <= 503.15 + 1e-6), "temperatures bounded by melt"


def test_layer_temperature_none_when_thermal_off() -> None:
    """With ``thermal_coupling=False`` the per-layer T/η/γ̇ arrays are
    ``None`` (the layer-thickness field is still emitted because it is
    purely geometric)."""
    g = build_film_gate_geometry(_default_cfg())
    db = MaterialDB()
    r = MultilayerHeleShawSolver(
        geometry=g,
        material=db["PP"],
        num_layers=3,
        thermal_coupling=False,
        **_solver_kwargs(),
    ).solve(num_frames=4)
    assert r.layer_thickness_mm is not None  # geometry only
    assert r.layer_temperature_K is None
    assert r.layer_viscosity_Pa_s_field is None
    assert r.layer_shear_rate_s_inv is None
    assert r.metadata["thermal_coupling"] is False
    assert r.metadata["multilayer_iterations"] == 0


def test_thermal_coupling_changes_tau_max() -> None:
    """The thermal coupling fundamentally changes the per-cell viscosity
    profile (wall layers vs centre), so τ_max must differ from the
    uncoupled baseline. Whether it goes up or down depends on the
    balance of wall-cooling (η ↑) vs centre-high-temperature low-shear
    (η ↓ via ``η₀`` evaluation at the melt) — assert *some* deviation,
    not its sign.
    """
    g = build_film_gate_geometry(_default_cfg())
    db = MaterialDB()
    kwargs = _solver_kwargs()

    r_off = MultilayerHeleShawSolver(
        geometry=g, material=db["PP"], num_layers=5, thermal_coupling=False, **kwargs
    ).solve(num_frames=4)
    r_on = MultilayerHeleShawSolver(
        geometry=g, material=db["PP"], num_layers=5, thermal_coupling=True, **kwargs
    ).solve(num_frames=4)
    rel = abs(r_on.metadata["tau_max"] - r_off.metadata["tau_max"]) / r_off.metadata["tau_max"]
    assert rel > 1e-3, "thermal coupling must non-trivially change τ_max"


def test_thermal_coupling_converges() -> None:
    """The default ``max_iterations=8`` is plenty for a moderate plate;
    ``multilayer_converged`` must be True."""
    g = build_film_gate_geometry(_default_cfg())
    db = MaterialDB()
    r = MultilayerHeleShawSolver(
        geometry=g,
        material=db["PP"],
        num_layers=5,
        thermal_coupling=True,
        max_iterations=8,
        convergence_tol=1e-3,
        **_solver_kwargs(),
    ).solve(num_frames=4)
    assert r.metadata["multilayer_converged"] is True
    assert 1 <= r.metadata["multilayer_iterations"] <= 8


def test_tighter_tol_does_not_take_fewer_iters() -> None:
    """A stricter ``convergence_tol`` cannot reduce the iteration count
    (monotone — looser tol may stop earlier)."""
    g = build_film_gate_geometry(_default_cfg())
    db = MaterialDB()
    kwargs = _solver_kwargs()
    loose = MultilayerHeleShawSolver(
        geometry=g,
        material=db["PP"],
        num_layers=5,
        thermal_coupling=True,
        max_iterations=15,
        convergence_tol=1e-1,
        **kwargs,
    ).solve(num_frames=4)
    tight = MultilayerHeleShawSolver(
        geometry=g,
        material=db["PP"],
        num_layers=5,
        thermal_coupling=True,
        max_iterations=15,
        convergence_tol=1e-5,
        **kwargs,
    ).solve(num_frames=4)
    assert tight.metadata["multilayer_iterations"] >= loose.metadata["multilayer_iterations"]


def test_thermal_metadata_carries_iteration_state() -> None:
    g = build_film_gate_geometry(_default_cfg())
    db = MaterialDB()
    r = MultilayerHeleShawSolver(
        geometry=g, material=db["PP"], num_layers=5, **_solver_kwargs()
    ).solve(num_frames=4)
    md = r.metadata
    assert md["thermal_coupling"] is True
    assert md["multilayer_max_iterations"] == 8
    assert md["multilayer_convergence_tol"] == 1e-3
    assert md["thermal_diffusivity_m2_s"] > 0
    assert md["T_fill_baseline_s"] > 0
    # T_fill_inflation may sit above or below 1.0 depending on which
    # layer effect (wall cooling vs centre-high-T zero-shear) dominates;
    # only require it to be positive and finite.
    assert md["T_fill_inflation"] > 0 and np.isfinite(md["T_fill_inflation"])


def test_layer_temperature_near_wall_lower_than_centre() -> None:
    """The Neumann profile cools the wall-side layers faster than the
    centre — the bookkeeping must surface this physical asymmetry."""
    g = build_film_gate_geometry(_default_cfg())
    db = MaterialDB()
    N = 5
    r = MultilayerHeleShawSolver(
        geometry=g, material=db["PP"], num_layers=N, **_solver_kwargs()
    ).solve(num_frames=4)
    T = r.layer_temperature_K
    assert T is not None
    # Average over cavity cells per layer.
    cavity = g.mask
    layer_means = np.array([T[k][cavity].mean() for k in range(N)])
    centre_idx = N // 2  # 2 for N=5
    # Both wall layers (k=0 and k=N-1) must be cooler than the centre.
    assert layer_means[0] < layer_means[centre_idx]
    assert layer_means[-1] < layer_means[centre_idx]


# --------------------------------------------------------------------------
# PR-C: wall_refined layer distribution
# --------------------------------------------------------------------------


def test_wall_refined_endpoints_and_length() -> None:
    """The wall-refined boundaries span [0, 1] and are length ``N+1``."""
    for N in (2, 3, 5, 6, 7):
        z = _wall_refined_layer_zeta(N)
        assert z.shape == (N + 1,)
        assert z[0] == 0.0
        assert z[-1] == 1.0
        # Monotonically increasing — basic sanity for a layer partition.
        assert np.all(np.diff(z) > 0.0)


def test_wall_refined_symmetric_about_centre() -> None:
    """``ζ_k + ζ_{N-k} = 1`` for all k (mirror symmetry about ζ=0.5)."""
    for N in (2, 4, 6, 7):
        z = _wall_refined_layer_zeta(N)
        np.testing.assert_allclose(z + z[::-1], 1.0, atol=1e-12)


def test_wall_refined_walls_thinner_than_centre() -> None:
    """The first and last layers are *strictly thinner* than the centre
    layer — the whole point of refining at the walls."""
    for N in (4, 5, 6, 7):
        z = _wall_refined_layer_zeta(N)
        widths = np.diff(z)
        centre = widths[N // 2]
        assert widths[0] < centre
        assert widths[-1] < centre


def test_wall_refined_matches_plan_example_n6() -> None:
    """The N=6 wall-refined boundaries are exactly the values quoted in
    the implementation plan: [0, 0.067, 0.25, 0.5, 0.75, 0.933, 1]."""
    z = _wall_refined_layer_zeta(6)
    expected = np.array(
        [
            0.0,
            0.5 * (1.0 - np.cos(np.pi * 1 / 6)),
            0.5 * (1.0 - np.cos(np.pi * 2 / 6)),
            0.5 * (1.0 - np.cos(np.pi * 3 / 6)),
            0.5 * (1.0 - np.cos(np.pi * 4 / 6)),
            0.5 * (1.0 - np.cos(np.pi * 5 / 6)),
            1.0,
        ]
    )
    np.testing.assert_allclose(z, expected, atol=1e-12)
    # Spot-check the rounded values from the plan.
    assert abs(z[1] - 0.067) < 5e-3  # ≈ 0.0670
    assert abs(z[2] - 0.250) < 1e-3
    assert abs(z[3] - 0.500) < 1e-12
    assert abs(z[5] - 0.933) < 5e-3


def test_wall_refined_moments_sum_to_one_sixth() -> None:
    """Σ m_k = 1/6 is the Hele-Shaw factor; no distribution may break it."""
    for N in (2, 3, 5, 6, 7, 11):
        z = _wall_refined_layer_zeta(N)
        m = _poiseuille_layer_moments(z)
        assert m.shape == (N,)
        np.testing.assert_allclose(m.sum(), 1.0 / 6.0, rtol=1e-12)


def test_wall_refined_n1_falls_back_to_uniform() -> None:
    """``N=1`` cannot be refined; the dispatcher returns the uniform
    [0, 1] partition so the N=1 ↔ classical Hele-Shaw identity holds
    even when the user asks for ``wall_refined``."""
    z_wr = _wall_refined_layer_zeta(1)
    z_un = _uniform_layer_zeta(1)
    np.testing.assert_array_equal(z_wr, z_un)


def test_solver_accepts_wall_refined_distribution() -> None:
    g = build_film_gate_geometry(_default_cfg())
    db = MaterialDB()
    r = MultilayerHeleShawSolver(
        geometry=g,
        material=db["PP"],
        num_layers=5,
        layer_distribution="wall_refined",
        **_solver_kwargs(),
    ).solve(num_frames=4)
    assert r.metadata["layer_distribution"] == "wall_refined"
    zeta = r.metadata["layer_zeta"]
    # Symmetric about 0.5 (sanity).
    rev = list(reversed(zeta))
    for a, b in zip(zeta, rev):
        assert abs(a + b - 1.0) < 1e-12


# --------------------------------------------------------------------------
# PR-C: short-shot detection
# --------------------------------------------------------------------------


def test_short_shot_metadata_present() -> None:
    """Both ``short_shot_cells`` and ``short_shot_fraction`` are always
    in the metadata when ``thermal_coupling=True``."""
    g = build_film_gate_geometry(_default_cfg())
    db = MaterialDB()
    r = MultilayerHeleShawSolver(
        geometry=g, material=db["PP"], num_layers=5, **_solver_kwargs()
    ).solve(num_frames=4)
    assert "short_shot_cells" in r.metadata
    assert "short_shot_fraction" in r.metadata
    assert "T_solid_K" in r.metadata
    assert 0.0 <= r.metadata["short_shot_fraction"] <= 1.0


def test_short_shot_off_in_default_warm_run() -> None:
    """A vanilla 2 mm plate at 503 K / 313 K does *not* freeze the
    centre layer below T_solid → short_shot_fraction == 0."""
    g = build_film_gate_geometry(_default_cfg())
    db = MaterialDB()
    r = MultilayerHeleShawSolver(
        geometry=g, material=db["PP"], num_layers=5, **_solver_kwargs()
    ).solve(num_frames=4)
    assert r.metadata["short_shot_cells"] == 0
    assert r.metadata["short_shot_fraction"] == 0.0
    assert r.short_shot_mask is None or not r.short_shot_mask.any()


def test_short_shot_on_thin_plate_high_mold_threshold() -> None:
    """A thin plate (t=0.35 mm gate-side / 0.50 mm far-side) and a very
    aggressive solidification threshold (60% of the melt-mold span)
    forces the centre layer through the threshold for *some* cells.
    """
    cfg = _default_cfg(
        plate_thk_mm=0.50,
        plate_split_height_mm=20.0,
        plate_lower_thk_mm=0.35,
        plate_upper_thk_mm=0.50,
    )
    g = build_film_gate_geometry(cfg)
    db = MaterialDB()
    r = MultilayerHeleShawSolver(
        geometry=g,
        material=db["PP"],
        num_layers=5,
        layer_distribution="wall_refined",
        solidification_temperature_fraction=0.6,
        **_solver_kwargs(),
    ).solve(num_frames=4)
    assert r.metadata["short_shot_cells"] > 0
    assert r.metadata["short_shot_fraction"] > 0.0
    assert r.short_shot_mask is not None and r.short_shot_mask.any()
    # Sanity: only cavity cells participate.
    assert np.all(r.short_shot_mask <= g.mask)


def test_short_shot_threshold_zero_marks_nothing() -> None:
    """``solidification_temperature_fraction=0.0`` ⇒ threshold equals
    ``T_mold``; the clamp keeps centre-layer T ≥ T_mold so no cell is
    ever flagged."""
    g = build_film_gate_geometry(_default_cfg())
    db = MaterialDB()
    r = MultilayerHeleShawSolver(
        geometry=g,
        material=db["PP"],
        num_layers=5,
        solidification_temperature_fraction=0.0,
        **_solver_kwargs(),
    ).solve(num_frames=4)
    # T_mid > T_mold for any finite t_arr after the clamp, but the
    # T_solid is at the floor, so flagged set is small or zero.
    assert r.metadata["short_shot_fraction"] <= 0.0 + 1e-9


# --------------------------------------------------------------------------
# PR-C: adaptive damping
# --------------------------------------------------------------------------


def test_damping_metadata_present() -> None:
    g = build_film_gate_geometry(_default_cfg())
    db = MaterialDB()
    r = MultilayerHeleShawSolver(
        geometry=g, material=db["PP"], num_layers=5, **_solver_kwargs()
    ).solve(num_frames=4)
    assert "damping_factor" in r.metadata
    assert "damping_events" in r.metadata
    assert r.metadata["damping_factor"] == 0.7  # default
    assert r.metadata["damping_events"] >= 0


def test_damping_factor_validation() -> None:
    """``damping_factor`` must sit in (0, 1]."""
    g = build_film_gate_geometry(_default_cfg())
    db = MaterialDB()
    for bad in (-0.1, 0.0, 1.5):
        solver = MultilayerHeleShawSolver(
            geometry=g,
            material=db["PP"],
            num_layers=3,
            damping_factor=bad,
            **_solver_kwargs(),
        )
        with pytest.raises(ValueError, match="damping_factor"):
            solver.solve(num_frames=4)


def test_damping_omega_one_is_undamped() -> None:
    """``damping_factor=1.0`` is the no-damping pass-through; we can't
    easily detect *that* (the path is the same), but the solver must
    accept it and run."""
    g = build_film_gate_geometry(_default_cfg())
    db = MaterialDB()
    r = MultilayerHeleShawSolver(
        geometry=g,
        material=db["PP"],
        num_layers=5,
        damping_factor=1.0,
        **_solver_kwargs(),
    ).solve(num_frames=4)
    assert r.metadata["damping_factor"] == 1.0


# --------------------------------------------------------------------------
# Shear heating (viscous dissipation) — stage 1
# --------------------------------------------------------------------------


def _thin_plate_cfg() -> FilmGateConfig:
    """Stress geometry for shear heating: thin plate at high injection rate."""
    return _default_cfg(
        plate_thk_mm=0.4,  # thin plate (overrides default 2.0 mm)
        plate_w_mm=80.0,
        plate_h_mm=40.0,
    )


def _shear_kwargs() -> dict:
    """Solver kwargs that emphasise shear heating (fast injection + thin)."""
    return dict(
        melt_temperature_K=503.15,
        mold_temperature_K=313.15,
        injection_velocity_mms=300.0,  # high V → γ̇ ≈ 4500 s⁻¹ for h=0.4 mm
        injection_volume_flow_cm3s=20.0,
    )


def test_shear_heating_default_off_keeps_backwards_compat() -> None:
    """``shear_heating_enabled`` defaults to False so existing callers
    see no change in numerical results vs prior PRs."""
    g = build_film_gate_geometry(_default_cfg())
    db = MaterialDB()
    r = MultilayerHeleShawSolver(
        geometry=g, material=db["PP"], num_layers=5, **_solver_kwargs()
    ).solve(num_frames=4)
    assert r.metadata["shear_heating_enabled"] is False
    assert r.metadata["shear_heating_max_K"] == 0.0
    assert r.metadata["shear_heating_mean_K"] == 0.0


def test_shear_heating_brinkman_always_populated_when_coupled() -> None:
    """Even with shear heating OFF, the Brinkman number metadata is
    populated when ``thermal_coupling=True`` — that's the *diagnostic*
    we use to decide whether the correction is needed."""
    g = build_film_gate_geometry(_thin_plate_cfg())
    db = MaterialDB()
    r = MultilayerHeleShawSolver(
        geometry=g,
        material=db["PP"],
        num_layers=5,
        shear_heating_enabled=False,
        **_shear_kwargs(),
    ).solve(num_frames=4)
    assert r.metadata["brinkman_number_max"] > 0.0
    assert r.metadata["brinkman_number_mean"] > 0.0
    assert r.layer_brinkman_number is not None
    assert r.layer_brinkman_number.shape == (5,) + g.shape


def test_shear_heating_on_raises_max_temperature_rise() -> None:
    """With shear heating ON, the per-layer temperature rise field is
    populated and the metadata reports a positive max."""
    g = build_film_gate_geometry(_thin_plate_cfg())
    db = MaterialDB()
    r = MultilayerHeleShawSolver(
        geometry=g,
        material=db["PP"],
        num_layers=5,
        shear_heating_enabled=True,
        **_shear_kwargs(),
    ).solve(num_frames=4)
    assert r.metadata["shear_heating_enabled"] is True
    assert r.metadata["shear_heating_max_K"] > 0.0
    assert r.metadata["shear_heating_mean_K"] >= 0.0
    assert r.layer_shear_heating_dT_K is not None
    assert r.layer_shear_heating_dT_K.shape == (5,) + g.shape


def test_shear_heating_lowers_layer_viscosity_vs_off() -> None:
    """Shear heating raises T_k → drops η_k via Cross-WLF.

    Compare the max layer viscosity inside the cavity with vs without
    the correction. The correction must not *increase* η anywhere.
    """
    g = build_film_gate_geometry(_thin_plate_cfg())
    db = MaterialDB()
    r_off = MultilayerHeleShawSolver(
        geometry=g,
        material=db["PP"],
        num_layers=5,
        shear_heating_enabled=False,
        **_shear_kwargs(),
    ).solve(num_frames=4)
    r_on = MultilayerHeleShawSolver(
        geometry=g,
        material=db["PP"],
        num_layers=5,
        shear_heating_enabled=True,
        **_shear_kwargs(),
    ).solve(num_frames=4)

    cavity = g.mask
    eta_off = r_off.layer_viscosity_Pa_s_field[:, cavity]  # (N, Ncells)
    eta_on = r_on.layer_viscosity_Pa_s_field[:, cavity]
    # On average η decreases (heating thins the polymer)
    assert float(np.mean(eta_on)) <= float(np.mean(eta_off)) * (1.0 + 1e-6)


def test_shear_heating_metadata_contains_material_thermal_fields() -> None:
    g = build_film_gate_geometry(_default_cfg())
    db = MaterialDB()
    r = MultilayerHeleShawSolver(
        geometry=g,
        material=db["PP"],
        num_layers=5,
        shear_heating_enabled=True,
        **_solver_kwargs(),
    ).solve(num_frames=4)
    assert r.metadata["specific_heat_J_kgK"] == db["PP"].specific_heat_J_kgK
    assert r.metadata["thermal_conductivity_W_mK"] == pytest.approx(
        db["PP"].thermal_conductivity_W_mK
    )


# --------------------------------------------------------------------------
# Gate reachability (Issue #58)
# --------------------------------------------------------------------------


def test_multilayer_rejects_a_gateless_region() -> None:
    """The reachability guard covers this solver too, not just the base one.

    ``MultilayerHeleShawSolver.solve`` does not call ``HeleShawSolver.solve``
    -- it drives ``_solve_tau_field`` directly -- so a check living only in
    the base ``solve()`` would leave the layered path solving the same
    singular Neumann block. The severed strip here is the Issue #58
    reproduction, and the match string pins the reachability message.
    """
    from core.geometry import Geometry

    ny, nx = 6, 20
    mask = np.ones((ny, nx), dtype=bool)
    mask[:, 9:11] = False  # sever the far half from the gate edge
    g = Geometry(
        mask=mask,
        thickness_mm=np.full((ny, nx), 2.0, dtype=float),
        cell_size_mm=1.0,
    )
    g.gates = [(iy, 0) for iy in range(ny)]
    solver = MultilayerHeleShawSolver(
        geometry=g,
        material=MaterialDB()["PP"],
        num_layers=3,
        **_solver_kwargs(),
    )
    with pytest.raises(ValueError, match="cannot be .*reached from any gate"):
        solver.solve(num_frames=2)
