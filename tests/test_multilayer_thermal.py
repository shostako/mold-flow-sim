"""Tests for the pure helper functions in :mod:`core.multilayer_thermal`.

The helpers are deliberately material- and solver-agnostic, so these
tests cover physical limits that any future implementation must
preserve:

* the Neumann temperature equals ``T_melt`` for ``t → 0`` and clamps to
  ``T_mold`` for ``t → ∞``;
* the profile is symmetric around the mid-thickness;
* the wall layers are always at lower temperature than the centre layer
  (under symmetric cooling);
* the Poiseuille shear-rate vanishes at the walls (sic) and peaks at
  the centre — wait, no: in our convention ``γ̇(ζ) = (6V/h) |2ζ − 1|``
  is *zero at the centre* and *maximum at the walls*, with a centreline
  floor that prevents an exact zero;
* shape contracts (N, ny, nx) are satisfied.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.multilayer_thermal import (
    brinkman_number,
    neumann_layer_temperatures,
    poiseuille_shear_rates,
    shear_heating_temperature_rise,
)

# --------------------------------------------------------------------------
# Neumann 1D temperature profile
# --------------------------------------------------------------------------


def test_neumann_t_zero_returns_melt() -> None:
    """For ``t → 0`` every layer keeps the initial melt temperature."""
    zeta = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    h_mm = np.full((2, 3), 1.0)
    t_arr = np.full((2, 3), 1e-12)  # effectively zero
    T = neumann_layer_temperatures(
        zeta_centers=zeta,
        t_arr_s=t_arr,
        h_total_mm=h_mm,
        T_melt_K=503.15,
        T_mold_K=313.15,
        alpha_m2_s=9e-8,
    )
    assert T.shape == (5, 2, 3)
    np.testing.assert_allclose(T, 503.15, rtol=1e-9)


def test_neumann_t_inf_clamps_to_mold() -> None:
    """For ``t → ∞`` the closed-form dips below ``T_mold``; the helper
    must clamp every layer to ``T_mold`` exactly."""
    zeta = np.array([0.1, 0.5, 0.9])
    h_mm = np.full((2, 2), 1.0)
    t_arr = np.full((2, 2), 1e6)  # huge time
    T = neumann_layer_temperatures(
        zeta_centers=zeta,
        t_arr_s=t_arr,
        h_total_mm=h_mm,
        T_melt_K=503.15,
        T_mold_K=313.15,
        alpha_m2_s=9e-8,
    )
    np.testing.assert_allclose(T, 313.15, rtol=1e-9)


def test_neumann_temperature_symmetry() -> None:
    """The closed-form is symmetric about the mid-thickness; layers at
    ``ζ`` and ``1 − ζ`` must produce identical temperatures."""
    zeta = np.array([0.15, 0.85])
    h_mm = np.full((2, 2), 1.5)
    t_arr = np.full((2, 2), 0.1)
    T = neumann_layer_temperatures(
        zeta_centers=zeta,
        t_arr_s=t_arr,
        h_total_mm=h_mm,
        T_melt_K=503.15,
        T_mold_K=313.15,
        alpha_m2_s=9e-8,
    )
    np.testing.assert_allclose(T[0], T[1], rtol=1e-12)


def test_neumann_centre_warmer_than_walls() -> None:
    """Under symmetric mold cooling the centre layer is warmer than the
    near-wall layers at any intermediate time."""
    zeta = np.array([0.05, 0.5, 0.95])
    h_mm = np.full((2, 2), 1.0)
    t_arr = np.full((2, 2), 0.05)
    T = neumann_layer_temperatures(
        zeta_centers=zeta,
        t_arr_s=t_arr,
        h_total_mm=h_mm,
        T_melt_K=503.15,
        T_mold_K=313.15,
        alpha_m2_s=9e-8,
    )
    # Both wall layers must be strictly cooler than the centre layer.
    assert np.all(T[0] < T[1])
    assert np.all(T[2] < T[1])


def test_neumann_temperature_monotone_in_t() -> None:
    """A wall-side layer cools monotonically as ``t`` grows."""
    zeta = np.array([0.1])
    h_mm = np.full((2, 2), 1.0)
    T_short = neumann_layer_temperatures(
        zeta_centers=zeta,
        t_arr_s=np.full((2, 2), 0.01),
        h_total_mm=h_mm,
        T_melt_K=503.15,
        T_mold_K=313.15,
        alpha_m2_s=9e-8,
    )
    T_long = neumann_layer_temperatures(
        zeta_centers=zeta,
        t_arr_s=np.full((2, 2), 1.0),
        h_total_mm=h_mm,
        T_melt_K=503.15,
        T_mold_K=313.15,
        alpha_m2_s=9e-8,
    )
    assert np.all(T_long < T_short)


def test_neumann_input_shape_validation() -> None:
    zeta = np.array([0.5])
    h_mm = np.full((2, 2), 1.0)
    bad_t = np.full((3, 3), 0.1)  # wrong shape
    with pytest.raises(ValueError, match="shapes mismatch"):
        neumann_layer_temperatures(
            zeta_centers=zeta,
            t_arr_s=bad_t,
            h_total_mm=h_mm,
            T_melt_K=503.15,
            T_mold_K=313.15,
            alpha_m2_s=9e-8,
        )
    with pytest.raises(ValueError, match="1-D"):
        neumann_layer_temperatures(
            zeta_centers=zeta[None, :],  # wrong dim
            t_arr_s=np.full((2, 2), 0.1),
            h_total_mm=h_mm,
            T_melt_K=503.15,
            T_mold_K=313.15,
            alpha_m2_s=9e-8,
        )


# --------------------------------------------------------------------------
# Poiseuille shear rate
# --------------------------------------------------------------------------


def test_poiseuille_max_at_walls_floor_at_centre() -> None:
    """The analytic profile ``γ̇(ζ) = (6V/h) |2ζ - 1|`` peaks at ζ=0 and
    ζ=1 (the walls) and would be exactly zero at ζ=0.5 — but the helper
    must enforce a centreline floor of ``floor_factor · 6V/h``.
    """
    zeta = np.array([0.0, 0.5, 1.0])
    h_mm = np.full((2, 3), 2.0)
    V = 100.0
    g = poiseuille_shear_rates(zeta_centers=zeta, V_mms=V, h_total_mm=h_mm, floor_factor=0.01)
    wall_rate = 6.0 * V / 2.0  # = 300 s^-1
    floor = 0.01 * wall_rate

    np.testing.assert_allclose(g[0], wall_rate, rtol=1e-12)  # ζ = 0
    np.testing.assert_allclose(g[2], wall_rate, rtol=1e-12)  # ζ = 1
    np.testing.assert_allclose(g[1], floor, rtol=1e-12)  # ζ = 0.5 → floor


def test_poiseuille_shape() -> None:
    """Output shape is (N, ny, nx)."""
    zeta = np.linspace(0.1, 0.9, 5)
    h_mm = np.full((4, 6), 1.5)
    g = poiseuille_shear_rates(zeta_centers=zeta, V_mms=80.0, h_total_mm=h_mm)
    assert g.shape == (5, 4, 6)
    assert np.all(g > 0.0)


def test_poiseuille_floor_factor_zero_keeps_exact_zero() -> None:
    """A ``floor_factor=0`` reproduces the analytic γ̇=0 at the centre."""
    zeta = np.array([0.5])
    h_mm = np.full((2, 2), 2.0)
    g = poiseuille_shear_rates(zeta_centers=zeta, V_mms=100.0, h_total_mm=h_mm, floor_factor=0.0)
    np.testing.assert_allclose(g, 0.0, atol=1e-15)


def test_poiseuille_rejects_negative_floor_factor() -> None:
    zeta = np.array([0.5])
    h_mm = np.full((2, 2), 2.0)
    with pytest.raises(ValueError, match="floor_factor"):
        poiseuille_shear_rates(zeta_centers=zeta, V_mms=100.0, h_total_mm=h_mm, floor_factor=-1.0)


def test_poiseuille_rejects_non_1d_zeta() -> None:
    zeta = np.array([[0.1, 0.5]])
    h_mm = np.full((2, 2), 2.0)
    with pytest.raises(ValueError, match="1-D"):
        poiseuille_shear_rates(zeta_centers=zeta, V_mms=100.0, h_total_mm=h_mm)


# --------------------------------------------------------------------------
# Shear-heating temperature rise (stage 1)
# --------------------------------------------------------------------------


def _shear_heating_inputs(N: int = 5, ny: int = 2, nx: int = 3, h_mm: float = 1.0):
    eta = np.full((N, ny, nx), 200.0)  # Pa·s
    gamma = np.full((N, ny, nx), 1000.0)  # 1/s
    t_arr = np.full((ny, nx), 0.1)  # s
    h_total = np.full((ny, nx), h_mm)  # mm
    return eta, gamma, t_arr, h_total


def test_shear_heating_shape_and_nonnegative() -> None:
    eta, gamma, t_arr, h_mm = _shear_heating_inputs()
    dT = shear_heating_temperature_rise(
        eta_per_layer_Pa_s=eta,
        gamma_dot_per_layer_s_inv=gamma,
        t_arr_s=t_arr,
        h_total_mm=h_mm,
        density_kg_m3=738.0,
        specific_heat_J_kgK=2400.0,
        alpha_m2_s=9e-8,
    )
    assert dT.shape == (5, 2, 3)
    assert np.all(dT >= 0.0)


def test_shear_heating_zero_when_gamma_zero() -> None:
    """No shear → no dissipation → zero rise."""
    eta, _, t_arr, h_mm = _shear_heating_inputs()
    gamma = np.zeros_like(eta)
    dT = shear_heating_temperature_rise(
        eta_per_layer_Pa_s=eta,
        gamma_dot_per_layer_s_inv=gamma,
        t_arr_s=t_arr,
        h_total_mm=h_mm,
        density_kg_m3=738.0,
        specific_heat_J_kgK=2400.0,
        alpha_m2_s=9e-8,
    )
    np.testing.assert_allclose(dT, 0.0, atol=1e-15)


def test_shear_heating_quadratic_in_gamma() -> None:
    """Doubling γ̇ quadruples ΔT (η·γ̇² scaling) at fixed eta and t."""
    eta, gamma, t_arr, h_mm = _shear_heating_inputs()
    dT1 = shear_heating_temperature_rise(
        eta_per_layer_Pa_s=eta,
        gamma_dot_per_layer_s_inv=gamma,
        t_arr_s=t_arr,
        h_total_mm=h_mm,
        density_kg_m3=738.0,
        specific_heat_J_kgK=2400.0,
        alpha_m2_s=9e-8,
    )
    dT2 = shear_heating_temperature_rise(
        eta_per_layer_Pa_s=eta,
        gamma_dot_per_layer_s_inv=2.0 * gamma,
        t_arr_s=t_arr,
        h_total_mm=h_mm,
        density_kg_m3=738.0,
        specific_heat_J_kgK=2400.0,
        alpha_m2_s=9e-8,
    )
    np.testing.assert_allclose(dT2, 4.0 * dT1, rtol=1e-10)


def test_shear_heating_capped_by_thermal_time() -> None:
    """For t_arr ≫ τ_thermal the rise saturates at the τ_thermal-based value."""
    eta, gamma, _, h_mm = _shear_heating_inputs(h_mm=0.5)
    # τ_thermal = h² / (π²·α). With h=0.5 mm = 5e-4 m, α=9e-8: τ ≈ 0.282 s.
    short = shear_heating_temperature_rise(
        eta_per_layer_Pa_s=eta,
        gamma_dot_per_layer_s_inv=gamma,
        t_arr_s=np.full((2, 3), 10.0),  # 10 s ≫ τ_thermal
        h_total_mm=h_mm,
        density_kg_m3=738.0,
        specific_heat_J_kgK=2400.0,
        alpha_m2_s=9e-8,
    )
    long = shear_heating_temperature_rise(
        eta_per_layer_Pa_s=eta,
        gamma_dot_per_layer_s_inv=gamma,
        t_arr_s=np.full((2, 3), 100.0),  # 100 s ≫ τ_thermal
        h_total_mm=h_mm,
        density_kg_m3=738.0,
        specific_heat_J_kgK=2400.0,
        alpha_m2_s=9e-8,
    )
    np.testing.assert_allclose(short, long, rtol=1e-10)


def test_shear_heating_pp_order_of_magnitude_for_thin_plate() -> None:
    """Sanity check: 0.4 mm PP plate at 4500 s⁻¹ and 200 Pa·s → ΔT ≈ tens of K."""
    eta = np.full((1, 1, 1), 200.0)
    gamma = np.full((1, 1, 1), 4500.0)
    t_arr = np.full((1, 1), 0.5)
    h_mm = np.full((1, 1), 0.4)
    dT = shear_heating_temperature_rise(
        eta_per_layer_Pa_s=eta,
        gamma_dot_per_layer_s_inv=gamma,
        t_arr_s=t_arr,
        h_total_mm=h_mm,
        density_kg_m3=738.0,
        specific_heat_J_kgK=2400.0,
        alpha_m2_s=9e-8,
    )
    # τ_thermal = (4e-4)² / (π²·9e-8) ≈ 0.18 s, capped time = 0.18.
    # ΔT ≈ 200 · 4500² / (738·2400) · 0.18 ≈ 411 K → tens to hundreds K range.
    # We just sanity-check the order of magnitude.
    val = float(dT.ravel()[0])
    assert val > 1.0, f"Shear heating ΔT too small for thin-plate PP: {val} K"
    assert val < 10_000.0, f"Shear heating ΔT clearly diverged: {val} K"


def test_shear_heating_rejects_shape_mismatch() -> None:
    eta = np.zeros((3, 2, 2))
    gamma = np.zeros((3, 2, 3))  # mismatched nx
    with pytest.raises(ValueError, match="shapes mismatch"):
        shear_heating_temperature_rise(
            eta_per_layer_Pa_s=eta,
            gamma_dot_per_layer_s_inv=gamma,
            t_arr_s=np.zeros((2, 2)),
            h_total_mm=np.zeros((2, 2)),
            density_kg_m3=738.0,
            specific_heat_J_kgK=2400.0,
            alpha_m2_s=9e-8,
        )


# --------------------------------------------------------------------------
# Brinkman number
# --------------------------------------------------------------------------


def test_brinkman_shape_and_nonnegative() -> None:
    eta, gamma, _, h_mm = _shear_heating_inputs()
    Br = brinkman_number(
        eta_per_layer_Pa_s=eta,
        gamma_dot_per_layer_s_inv=gamma,
        h_total_mm=h_mm,
        thermal_conductivity_W_mK=0.21,
        delta_T_K=200.0,
    )
    assert Br.shape == (5, 2, 3)
    assert np.all(Br >= 0.0)


def test_brinkman_zero_when_gamma_zero() -> None:
    eta, _, _, h_mm = _shear_heating_inputs()
    gamma = np.zeros_like(eta)
    Br = brinkman_number(
        eta_per_layer_Pa_s=eta,
        gamma_dot_per_layer_s_inv=gamma,
        h_total_mm=h_mm,
        thermal_conductivity_W_mK=0.21,
        delta_T_K=200.0,
    )
    np.testing.assert_allclose(Br, 0.0, atol=1e-15)


def test_brinkman_thin_plate_high_shear_exceeds_unity() -> None:
    """Br ≫ 1 for thin plates at high shear — the textbook example."""
    eta = np.full((1, 1, 1), 200.0)
    gamma = np.full((1, 1, 1), 4500.0)
    h_mm = np.full((1, 1), 0.4)
    Br = brinkman_number(
        eta_per_layer_Pa_s=eta,
        gamma_dot_per_layer_s_inv=gamma,
        h_total_mm=h_mm,
        thermal_conductivity_W_mK=0.16,  # k = α·ρ·cp for PP ≈ 0.16
        delta_T_K=200.0,
    )
    # Br = 200·4500²·(4e-4)² / (0.16·200) ≈ 20
    val = float(Br.ravel()[0])
    assert val > 1.0, f"Br must exceed unity for thin-plate PP at 4500 s⁻¹: got {val}"


def test_brinkman_rejects_nonpositive_k() -> None:
    eta, gamma, _, h_mm = _shear_heating_inputs()
    with pytest.raises(ValueError, match="thermal_conductivity"):
        brinkman_number(
            eta_per_layer_Pa_s=eta,
            gamma_dot_per_layer_s_inv=gamma,
            h_total_mm=h_mm,
            thermal_conductivity_W_mK=0.0,
            delta_T_K=200.0,
        )


def test_brinkman_rejects_nonpositive_delta_T() -> None:
    eta, gamma, _, h_mm = _shear_heating_inputs()
    with pytest.raises(ValueError, match="delta_T"):
        brinkman_number(
            eta_per_layer_Pa_s=eta,
            gamma_dot_per_layer_s_inv=gamma,
            h_total_mm=h_mm,
            thermal_conductivity_W_mK=0.21,
            delta_T_K=0.0,
        )
