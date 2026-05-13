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
    MultilayerHeleShawSolver,
    build_film_gate_geometry,
)
from core.multilayer_solver import (
    _multilayer_conductance,
    _poiseuille_layer_moments,
    _uniform_layer_zeta,
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


def test_multilayer_solver_rejects_unsupported_distribution() -> None:
    """PR-A only ships ``uniform``; ``wall_refined`` comes in PR-C and
    must be rejected until then (caught when ``solve()`` is called)."""
    g = build_film_gate_geometry(_default_cfg())
    db = MaterialDB()
    solver = MultilayerHeleShawSolver(
        geometry=g,
        material=db["PP"],
        num_layers=5,
        layer_distribution="wall_refined",
        **_solver_kwargs(),
    )
    with pytest.raises(ValueError, match="wall_refined"):
        solver.solve(num_frames=4)


def test_n1_matches_legacy_tau() -> None:
    """The anchor test: ``num_layers=1`` must reproduce the existing
    ``HeleShawSolver`` τ field byte-for-byte (modulo tiny FP noise).
    """
    g = build_film_gate_geometry(_default_cfg())
    db = MaterialDB()
    r_legacy = HeleShawSolver(geometry=g, material=db["PP"], **_solver_kwargs()).solve(num_frames=4)
    r_multi = MultilayerHeleShawSolver(
        geometry=g, material=db["PP"], num_layers=1, **_solver_kwargs()
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
    """A consequence of the τ match: the absolute fill time is identical
    too (modulo FP noise)."""
    g = build_film_gate_geometry(_default_cfg())
    db = MaterialDB()
    r_legacy = HeleShawSolver(geometry=g, material=db["PP"], **_solver_kwargs()).solve(num_frames=4)
    r_multi = MultilayerHeleShawSolver(
        geometry=g, material=db["PP"], num_layers=1, **_solver_kwargs()
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
    """End-to-end smoke for the default ``N=5`` configuration."""
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
