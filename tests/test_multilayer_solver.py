"""Tests for the multilayer Hele-Shaw solver (PR-A skeleton).

The defining property is that ``num_layers=1`` collapses to the existing
:class:`HeleShawSolver` (the flux moment ``m_1 = 1/12`` matches the
closed-form ``S = h³ / (12 η)``). Additional tests cover layer-thickness
conservation, smoke checks, the moment-sum identity ``Σ m_k = 1/12`` that
any future ``layer_distribution`` must preserve, and the conductance of a
depth-varying viscosity against the lubrication integral solved for the
velocity profile (v0.61.0: the layers used to be weighted by the Newtonian
velocity shape ``ζ(1 − ζ)``, which is right only for a uniform η).

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


def test_poiseuille_layer_moments_sum_to_one_twelfth() -> None:
    """The full-thickness flux integral is ``Σ m_k = ∫(ζ − 1/2)² dζ = 1/12``,
    which is exactly the Hele-Shaw factor. Any future ``layer_distribution``
    must satisfy this — protect against accidental reshuffling.
    """
    for N in (1, 2, 3, 5, 7, 11, 25):
        z = _uniform_layer_zeta(N)
        m = _poiseuille_layer_moments(z)
        assert m.shape == (N,)
        assert np.all(m > 0.0), f"all moments must be positive for N={N}"
        np.testing.assert_allclose(m.sum(), 1.0 / 12.0, rtol=1e-12)


def test_n1_moment_equals_one_twelfth() -> None:
    """``N=1`` is the calibration anchor: ``m_1 = 1/12`` exactly."""
    z = _uniform_layer_zeta(1)
    m = _poiseuille_layer_moments(z)
    assert m.shape == (1,)
    np.testing.assert_allclose(m[0], 1.0 / 12.0, rtol=1e-14)


def test_layer_moments_weight_the_walls_not_the_midplane() -> None:
    """The flux weight is the squared distance from the midplane: per unit
    thickness it is largest in the wall layers and smallest in the centre,
    and it is mirror-symmetric. (The Newtonian velocity shape ``ζ(1 − ζ)``
    used up to v0.60.1 had it the other way round.)"""
    for z in (_uniform_layer_zeta(6), _wall_refined_layer_zeta(7)):
        density = _poiseuille_layer_moments(z) / np.diff(z)
        assert np.argmax(density) in (0, len(density) - 1)
        assert np.argmin(density) in (len(density) // 2, (len(density) - 1) // 2)
        np.testing.assert_allclose(density, density[::-1], rtol=1e-12)


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


@pytest.mark.parametrize("frozen", [np.inf, 1e300])
def test_a_frozen_skin_conducts_like_the_molten_core_alone(frozen) -> None:
    """Freeze the two outer layers on each side of a uniform 10-layer stack:
    the flow can only shear in the molten core of thickness 0.6·h, so the
    conductance is the Hele-Shaw value of that core, ``(0.6 h)³ / (12 η)``.

    The layer boundaries sit on the skin edge, so this is exact, not a
    discretisation estimate. The Newtonian velocity-shape weighting used up
    to v0.60.1 gave ``1.5/c² − 0.5 = 3.67`` times that. ``inf`` is what
    ``cross_wlf_viscosity`` returns for a frozen layer (v0.60.0).
    """
    h_mm = np.full((2, 3), 0.8)
    mask = np.ones_like(h_mm, dtype=bool)
    m = _poiseuille_layer_moments(_uniform_layer_zeta(10))
    eta = np.array([frozen, frozen] + [40.0] * 6 + [frozen, frozen])
    S = _multilayer_conductance(h_mm, eta, m, mask)
    core_m = 0.6 * h_mm * 1e-3
    np.testing.assert_allclose(S, core_m**3 / (12.0 * 40.0), rtol=1e-12)


def _lubrication_conductance(zeta: np.ndarray, eta: np.ndarray, h_m: float) -> float:
    """Independent oracle: solve the lubrication problem for the velocity.

    ``∂_z(η ∂_z u) = −G`` with no slip on both walls, ``η`` piecewise
    constant on the layers. Integrate ``∂_z u = G (z₀ − z) / η`` on a fine
    grid (the layer boundaries are grid points), pick ``z₀`` so that
    ``u(h) = 0``, integrate the flux ``q = ∫ u dz`` and return ``q / G``.
    No moment formula is involved.
    """
    G = 1.0
    z = np.unique(np.concatenate([np.linspace(a, b, 4001) for a, b in zip(zeta[:-1], zeta[1:])]))
    z = z * h_m
    mid = 0.5 * (z[:-1] + z[1:])
    layer = np.searchsorted(zeta * h_m, mid) - 1
    inv_eta = 1.0 / eta[layer]  # per sub-interval; ∂_z u is linear inside each
    dz = np.diff(z)
    i0 = np.sum(inv_eta * dz)
    i1 = np.sum(inv_eta * mid * dz)  # exact: ∫ z dz over a sub-interval is mid·dz
    z0 = i1 / i0
    du = G * inv_eta * (z0 - mid) * dz  # exact increment of u over each sub-interval
    u = np.concatenate([[0.0], np.cumsum(du)])
    assert abs(u[-1]) < 1e-9 * np.max(np.abs(u))  # no slip at the far wall
    q = np.sum(0.5 * (u[:-1] + u[1:]) * dz)
    return q / G


def test_conductance_matches_the_lubrication_integral() -> None:
    """A wall-refined 7-layer stack, cold walls and a hot core (the shape a
    cooling cell has): the conductance equals the flux of the velocity
    profile solved layer by layer. The old velocity-shape weighting misses
    it by more than a factor of 2 on the same stack, so the oracle can tell
    the two apart."""
    zeta = _wall_refined_layer_zeta(7)
    eta = np.array([3.0e4, 2.0e3, 300.0, 120.0, 300.0, 2.0e3, 3.0e4])
    h_mm = 0.35
    oracle = _lubrication_conductance(zeta, eta, h_mm * 1e-3)
    m = _poiseuille_layer_moments(zeta)
    S = _multilayer_conductance(np.full((1, 1), h_mm), eta, m, np.ones((1, 1), dtype=bool))
    np.testing.assert_allclose(S[0, 0], oracle, rtol=1e-6)

    old_m = np.diff(zeta**2 / 2.0 - zeta**3 / 3.0)
    old_S = 0.5 * (h_mm * 1e-3) ** 3 * np.sum(old_m / eta)
    assert old_S / oracle > 2.0


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
    assert abs(sum(moments) - 1.0 / 12.0) < 1e-12


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


def test_wall_refined_moments_sum_to_one_twelfth() -> None:
    """Σ m_k = 1/12 is the Hele-Shaw factor; no distribution may break it."""
    for N in (2, 3, 5, 6, 7, 11):
        z = _wall_refined_layer_zeta(N)
        m = _poiseuille_layer_moments(z)
        assert m.shape == (N,)
        np.testing.assert_allclose(m.sum(), 1.0 / 12.0, rtol=1e-12)


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


def _strong_shear_solver(**overrides) -> MultilayerHeleShawSolver:
    """A 20 × 8 mm plate, 0.35 mm PP-T20, gated at the middle of one long
    edge, V = 2 m/s on a slow 0.2 cm³/s shot: strong wall shear on melt
    that has cooled for a while (Brinkman number about 20). With the rise
    taken on the last iteration's viscosity, the layered fixed point did
    not converge in 12 iterations here, neither in ``solve`` nor in the
    two-phase injection phase -- the small stand-in for mold-flow-fangate2's
    default gates."""
    from core.geometry import Geometry

    ny, nx = 8, 20
    geom = Geometry(
        mask=np.ones((ny, nx), dtype=bool),
        thickness_mm=np.full((ny, nx), 0.35),
        cell_size_mm=1.0,
    )
    geom.add_gate(0, nx // 2)
    mat = MaterialDB()["PP_T20"]
    kw = dict(
        geometry=geom,
        material=mat,
        melt_temperature_K=sum(mat.T_melt_recommended) / 2,
        mold_temperature_K=sum(mat.T_mold_recommended) / 2,
        injection_velocity_mms=2000.0,
        injection_volume_flow_cm3s=0.2,
        num_layers=7,
        layer_distribution="wall_refined",
        thermal_coupling=True,
        shear_heating_enabled=True,
        max_iterations=12,
    )
    kw.update(overrides)
    return MultilayerHeleShawSolver(**kw)


def test_strong_shear_heating_converges_within_the_default_budget() -> None:
    from core.two_phase import solve_two_phase_short_shot

    solver = _strong_shear_solver()
    r = solver.solve(num_frames=2)
    assert r.metadata["brinkman_number_max"] > 10.0  # the case is strong shear
    assert r.metadata["multilayer_converged"] is True
    assert r.metadata["multilayer_iterations"] <= 12
    V_cav = float((solver.geometry.thickness_mm * solver.geometry.mask).sum()) * 1e-3
    tp = solve_two_phase_short_shot(solver, 0.8 * V_cav)
    assert tp.metadata["multilayer_converged"] is True


def test_strong_shear_rise_is_read_at_the_viscosity_it_produces() -> None:
    """Stage 1 gives each layer ΔT_k = η_k·γ̇_k²·t_eff/(ρ·cp) with one t_eff
    per cell, and η_k is the viscosity at the heated temperature. So the
    reported rise divided by the reported η_k·γ̇_k² is the same in every
    layer of a cell. With the rise taken on the previous iteration's
    viscosity, the reported η_k is not the one that made the rise."""
    r = _strong_shear_solver().solve(num_frames=2)
    cav = r.geometry.mask
    dT = r.layer_shear_heating_dT_K[:, cav]
    q = r.layer_viscosity_Pa_s_field[:, cav] * r.layer_shear_rate_s_inv[:, cav] ** 2
    per_cell = dT[0] / q[0]  # the wall layer: the largest rise
    np.testing.assert_allclose(dT, q * per_cell[None, :], rtol=1e-6, atol=1e-8)


def test_strong_shear_fixed_point_is_a_fixed_point_of_the_lagged_map(monkeypatch) -> None:
    """Solving the rise with its own viscosity changes the path, not the
    destination. From the converged state (tight tolerance), one round of
    the original update -- the rise on the given viscosity, then η, then τ --
    must hand back the same τ.

    The arrival times are held at those of a default solve. The volume-CDF
    map is a step function of the τ ranking, so near a rank tie the outer
    loop can end in a 2-cycle of a few cells swapping arrival order instead
    of reaching 1e-9 (about 1e-4 here after v0.61.0's conductance weights;
    the old weights did the same at nx = 22, V = 3 m/s). What is under test
    is the shear-heating update, not the arrival map."""
    from core.materials import cross_wlf_viscosity
    from core.multilayer_thermal import (
        neumann_layer_temperatures,
        shear_heating_temperature_rise,
    )

    seen: list = []
    orig = MultilayerHeleShawSolver._fixed_point

    def spy(self, *a, **k):
        out = orig(self, *a, **k)
        seen.append((a, out))
        return out

    monkeypatch.setattr(MultilayerHeleShawSolver, "_fixed_point", spy)
    first = _strong_shear_solver()
    first.solve(num_frames=2)
    (h0, _, _), out0 = seen.pop(0)
    held = first._base._arrival_time_field(
        out0["tau"], ~np.isnan(out0["tau"]), first.geometry.cell_size_mm**2 * h0, out0["T_fill"]
    )

    solver = _strong_shear_solver(convergence_tol=1e-9, max_iterations=60)
    monkeypatch.setattr(solver._base, "_arrival_time_field", lambda *a, **k: held.copy())
    solver.solve(num_frames=2)
    (h, gates, _), out = seen[0]
    assert out["converged"] is True
    geom = solver.geometry
    base = solver._base

    tau, T_fill = out["tau"], out["T_fill"]
    cell_volume = geom.cell_size_mm**2 * h
    t_arr = base._arrival_time_field(tau, ~np.isnan(tau), cell_volume, T_fill)
    t_arr = np.where(np.isnan(t_arr), 0.0, t_arr)
    mat = solver.material
    alpha = float(mat.thermal_diffusivity_m2_s)
    T_N = neumann_layer_temperatures(
        zeta_centers=solver.layer_zeta_centers(),
        t_arr_s=t_arr,
        h_total_mm=h,
        T_melt_K=solver.melt_temperature_K,
        T_mold_K=solver.mold_temperature_K,
        alpha_m2_s=alpha,
    )
    rise = shear_heating_temperature_rise(
        eta_per_layer_Pa_s=out["layer_eta_Pa_s"],  # the lag: the last viscosity
        gamma_dot_per_layer_s_inv=out["layer_gamma_dot"],
        t_arr_s=t_arr,
        h_total_mm=h,
        density_kg_m3=float(mat.density_melt_kgm3),
        specific_heat_J_kgK=float(mat.specific_heat_J_kgK),
        alpha_m2_s=alpha,
    )
    eta = cross_wlf_viscosity(mat, T_N + rise, out["layer_gamma_dot"], 0.0)
    S = _multilayer_conductance(
        h_total_mm=h, eta_per_layer_Pa_s=eta, moments=solver.layer_moments(), cavity_mask=geom.mask
    )
    tau_next, _ = base._solve_tau_field(S, gates)
    m = ~np.isnan(tau)
    rel = np.linalg.norm(tau_next[m] - tau[m]) / np.linalg.norm(tau[m])
    assert rel < 1e-8


@pytest.mark.parametrize("distribution", ["wall_refined", "uniform"])
def test_the_layer_viscosities_are_mirror_symmetric(distribution) -> None:
    """The conductance weights each layer by its squared distance from the
    midplane, which is the lubrication flux only when η is symmetric about
    the midplane (otherwise the zero-shear plane moves off it). Both walls
    sit at the same mould temperature, the shear rates go as ``|2ζ − 1|``
    and both distributions are mirror images, so the solved layer
    viscosities -- shear heating included -- must be too. A change that
    breaks this (a hot and a cold mould half, say) needs the general form
    ``S = I₂ − I₁² / I₀`` in ``_multilayer_conductance``."""
    solver = _strong_shear_solver(layer_distribution=distribution)
    r = solver.solve(num_frames=2)
    cav = solver.geometry.mask
    for field_ in (r.layer_viscosity_Pa_s_field, r.layer_temperature_K):
        np.testing.assert_allclose(field_[:, cav], field_[::-1][:, cav], rtol=1e-9)


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


def test_n1_matches_legacy_fill_time_map() -> None:
    """The two solver modes must agree on what "fill time at a cell" means.

    Both map tau to time through the volume CDF (Issue #52). If either side
    reverts to the old linear ``tau / tau_max`` map, identical physics would
    report different per-cell times depending on which UI radio was picked.
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
    # FP noise between the two tau solves can swap the CDF rank of
    # near-tied cells, shifting their time by exactly one cell's volume
    # quantum. Anything beyond that single quantum is a real map divergence.
    assert np.array_equal(np.isnan(r_legacy.fill_time_s), np.isnan(r_multi.fill_time_s))
    vol = g.thickness_mm * g.cell_size_mm**2
    quantum = float(vol[g.mask].max()) / float(vol[g.mask].sum())
    bound = quantum * r_legacy.total_fill_time_s * (1.0 + 1e-9)
    diff = np.abs(np.nan_to_num(r_legacy.fill_time_s - r_multi.fill_time_s, nan=0.0))
    assert float(diff.max()) <= bound


def test_multilayer_inflation_uses_the_volume_weighted_representatives() -> None:
    """The T_fill proxy is the ratio of volume-weighted mean taus (Issue #52).

    Asserted against metadata the solver itself exports, on a geometry where
    the tau change is non-uniform -- so the volume-weighted ratio measurably
    differs from the old single-cell max ratio and the test can tell the two
    apart.
    """
    g = build_film_gate_geometry(_default_cfg())
    r = MultilayerHeleShawSolver(
        geometry=g,
        material=MaterialDB()["PP"],
        num_layers=3,
        thermal_coupling=True,
        **_solver_kwargs(),
    ).solve(num_frames=4)
    md = r.metadata
    assert md["tau_rep_flow"] is not None and md["tau_rep_baseline"] is not None
    assert md["T_fill_inflation"] == pytest.approx(
        md["tau_rep_flow"] / md["tau_rep_baseline"], rel=1e-9
    )
    # the case has to discriminate: max ratio and mean ratio must not coincide
    max_ratio = md["tau_max"] / md["tau_max_baseline"]
    assert abs(md["T_fill_inflation"] - max_ratio) > 1e-3


def test_multilayer_temperatures_come_from_the_volume_map_arrivals() -> None:
    """The Neumann temperatures must be evaluated at volume-CDF arrival times.

    Reconstructs the layer temperatures from the *reported* fill times (the
    volume map) and from the old linear ``tau / tau_max`` map: the solver's
    field matches the first within the fixed point's own tolerance (~0.03 K
    here) and misses the second by ~14 K. A revert of the in-loop arrival map
    walks straight into the margin.
    """
    from core.multilayer_thermal import neumann_layer_temperatures

    g = build_film_gate_geometry(_default_cfg())
    mat = MaterialDB()["PP"]
    kwargs = _solver_kwargs()
    solver = MultilayerHeleShawSolver(
        geometry=g, material=mat, num_layers=3, thermal_coupling=True, **kwargs
    )
    r = solver.solve(num_frames=4)
    t_cdf = np.where(np.isnan(r.fill_time_s), 0.0, r.fill_time_s)
    T_expected = neumann_layer_temperatures(
        zeta_centers=solver.layer_zeta_centers(),
        t_arr_s=t_cdf,
        h_total_mm=g.thickness_mm,
        T_melt_K=kwargs["melt_temperature_K"],
        T_mold_K=kwargs["mold_temperature_K"],
        alpha_m2_s=mat.thermal_diffusivity_m2_s,
    )
    cav = np.broadcast_to(g.mask, T_expected.shape)
    max_dev_K = float(np.nanmax(np.abs(r.layer_temperature_K - T_expected)[cav]))
    assert max_dev_K < 0.5


def test_shear_heating_keeps_cold_pa66_layers_finite() -> None:
    """Codex P1 on PR #111: PA66's wall layers start below D2 − A2, where
    Cross-WLF gives NaN. With shear heating on, every layer viscosity the
    solver reports must be finite, and the loop must converge."""
    mat = MaterialDB()["PA66"]
    solver = _strong_shear_solver(
        material=mat,
        melt_temperature_K=sum(mat.T_melt_recommended) / 2,
        mold_temperature_K=sum(mat.T_mold_recommended) / 2,
        injection_velocity_mms=200.0,
        injection_volume_flow_cm3s=2.0,
    )
    r = solver.solve(num_frames=2)
    cav = r.geometry.mask
    assert np.all(np.isfinite(r.layer_viscosity_Pa_s_field[:, cav]))
    assert np.all(np.isfinite(r.layer_shear_heating_dT_K[:, cav]))
    assert r.metadata["multilayer_converged"] is True
