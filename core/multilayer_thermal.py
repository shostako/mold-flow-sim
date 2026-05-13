"""Pure helper functions for the multilayer thermal coupling (PR-B).

This module hosts the closed-form 1D thermal profile across the cavity
thickness (Neumann superposition) and the analytic Poiseuille shear-rate
profile. Both are evaluated cell-by-cell, vectorised over ``(N, ny, nx)``.

These functions are intentionally side-effect free and material-agnostic
— the solver hands in the temperature endpoints, the local arrival time
and the thermal diffusivity. Tests can therefore exercise them with
synthetic inputs without spinning up a full ``HeleShawSolver``.

Neumann temperature model
-------------------------

Both walls (z=0 and z=h) are held at ``T_mold``; the cavity is initially
at ``T_melt`` uniformly. Superposing two half-infinite Neumann solutions
gives

    T(z, t) = T_mold + (T_melt - T_mold)
              · [erf(z / (2√(α t))) + erf((h - z) / (2√(α t))) - 1]

At ``t → 0`` the bracket is ``+1`` (interior unchanged → ``T_melt``); at
``t → ∞`` the bracket is ``-1`` so the formula dips below ``T_mold``.
That long-time limit is unphysical for a one-dimensional Stefan/Neumann
problem with finite walls — we clamp ``T ≥ T_mold`` to encode the
physical floor.

Poiseuille shear rate
---------------------

For a Newtonian flow between two parallel walls the speed profile is

    u(ζ) = (3/2) V_avg [1 - (2ζ - 1)²]

so the wall shear rate is ``γ̇(ζ) = 6 V_avg / h · |2ζ - 1|`` (zero at the
centre, ``6V/h`` at each wall). Cross-WLF asks for a *finite* γ̇ even at
the centre — a centreline floor prevents the zero-shear viscosity ``D1``
from dominating the layer-integrated conductance.

Shear-heating temperature rise (stage 1)
----------------------------------------

Viscous dissipation pumps mechanical work into thermal energy at a
volumetric rate ``q̇ = η · γ̇²`` [W/m³]. In an isolated parcel that would
raise the temperature at the rate

    dT/dt = η · γ̇² / (ρ · cp)

Heat conduction simultaneously drains energy to the walls with a
characteristic thickness-direction time constant

    τ_thermal = h² / (π² · α)        (1D slab, lowest eigenmode)

For "stage 1" — a post-Neumann correction that costs nothing in the
fixed-point loop — we cap the integration time at ``τ_thermal``:

    ΔT_shear,k(x,y) = (η_k · γ̇_k²) / (ρ · cp) · min(t_arr, τ_thermal)

This is an upper bound on the steady-state temperature rise from a
constant heat source against linear conduction. It can over-estimate
the rise when ``Br ≫ 1`` (the dissipation outruns conduction), but
gives the correct order of magnitude for the Brinkman regime and
preserves the Neumann baseline at low γ̇.

The Brinkman number ``Br = η·γ̇²·h² / (k·ΔT)`` quantifies the regime:
``Br < 1`` → conduction dominates (rise can be neglected),
``Br ≈ 1`` → balance, ``Br > 1`` → dissipation dominates. Stage 2 will
replace this closed-form with a 1D FDM that resolves the time history
self-consistently.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf

# Numerical floor for ``sqrt(alpha * t)`` so the erf arguments stay finite
# for the initial fill-time (t very small near the gates). Below this we
# treat the bracket as +1 (i.e. all layers at T_melt), which matches the
# physical limit.
_SQRT_ALPHA_T_FLOOR = 1e-9  # in metres


def neumann_layer_temperatures(
    zeta_centers: np.ndarray,
    t_arr_s: np.ndarray,
    h_total_mm: np.ndarray,
    T_melt_K: float,
    T_mold_K: float,
    alpha_m2_s: float,
) -> np.ndarray:
    """Per-layer temperature field via Neumann superposition.

    Parameters
    ----------
    zeta_centers
        ``(N,)`` array of layer-centre coordinates in normalised
        thickness ``ζ ∈ [0, 1]``.
    t_arr_s
        ``(ny, nx)`` array of local arrival times in seconds.
    h_total_mm
        ``(ny, nx)`` array of local cavity thickness in mm.
    T_melt_K, T_mold_K
        Boundary temperatures (interior at ``t=0`` and wall, respectively),
        in K.
    alpha_m2_s
        Thermal diffusivity in m² / s.

    Returns
    -------
    np.ndarray
        ``(N, ny, nx)`` array of layer temperatures in K, clamped to
        ``T_mold_K`` from below.
    """
    if zeta_centers.ndim != 1:
        raise ValueError(f"zeta_centers must be 1-D, got shape {zeta_centers.shape}")
    if t_arr_s.shape != h_total_mm.shape:
        raise ValueError(
            f"t_arr_s and h_total_mm shapes mismatch: {t_arr_s.shape} vs {h_total_mm.shape}"
        )

    h_m = h_total_mm * 1e-3
    # Broadcast (N, 1, 1) * (1, ny, nx)
    z_m = zeta_centers[:, None, None] * h_m[None, :, :]
    h_b = h_m[None, :, :]

    denom = 2.0 * np.sqrt(np.maximum(alpha_m2_s * t_arr_s[None, :, :], _SQRT_ALPHA_T_FLOOR**2))
    erf_z = erf(z_m / denom)
    erf_hz = erf((h_b - z_m) / denom)
    bracket = erf_z + erf_hz - 1.0
    T = T_mold_K + (T_melt_K - T_mold_K) * bracket

    # Long-time clamp: 1D Neumann superposition dips below T_mold in the
    # asymptotic limit (unphysical). Physically, the molten interior can
    # never cool past the mold-wall temperature.
    return np.maximum(T, T_mold_K)


def poiseuille_shear_rates(
    zeta_centers: np.ndarray,
    V_mms: float,
    h_total_mm: np.ndarray,
    floor_factor: float = 0.01,
) -> np.ndarray:
    """Analytic shear-rate profile across the thickness, evaluated at
    layer centres.

    ``γ̇_k(x,y) = (6 V / h) · |2 ζ_k - 1|`` from the Newtonian Poiseuille
    profile, with a centreline floor of ``floor_factor · 6V/h`` so the
    layer that straddles ``ζ = 0.5`` does not produce a vanishing γ̇ and
    a divergent zero-shear viscosity.

    Returns
    -------
    np.ndarray
        ``(N, ny, nx)`` array in s⁻¹.
    """
    if zeta_centers.ndim != 1:
        raise ValueError(f"zeta_centers must be 1-D, got shape {zeta_centers.shape}")
    if floor_factor < 0:
        raise ValueError(f"floor_factor must be >= 0, got {floor_factor}")

    h = np.maximum(h_total_mm, 1e-6)  # avoid div by zero outside mask
    wall_rate = 6.0 * V_mms / h  # (ny, nx)
    profile = np.abs(2.0 * zeta_centers - 1.0)  # (N,)
    raw = profile[:, None, None] * wall_rate[None, :, :]
    floor = floor_factor * wall_rate[None, :, :]
    return np.maximum(raw, floor)


# ---------------------------------------------------------------------------
# Shear heating (viscous dissipation) — stage 1 closed-form correction
# ---------------------------------------------------------------------------


# Minimum thickness used to bound τ_thermal away from zero. 1 µm — well
# below any physically meaningful injection-moulding cavity.
_H_FLOOR_MM = 1e-3


def shear_heating_temperature_rise(
    eta_per_layer_Pa_s: np.ndarray,
    gamma_dot_per_layer_s_inv: np.ndarray,
    t_arr_s: np.ndarray,
    h_total_mm: np.ndarray,
    density_kg_m3: float,
    specific_heat_J_kgK: float,
    alpha_m2_s: float,
) -> np.ndarray:
    """Per-layer temperature rise from viscous dissipation (stage 1).

    Implements the simple bounded form

    ``ΔT_k(x,y) = (η_k · γ̇_k²) / (ρ · cp) · min(t_arr, τ_thermal)``

    where ``τ_thermal = h² / (π² · α)`` is the lowest-mode thermal
    relaxation time of a 1D slab. Capping the integration time at
    ``τ_thermal`` approximates the steady-state balance between the
    constant heat source and 1D wall conduction without solving a PDE.

    Parameters
    ----------
    eta_per_layer_Pa_s
        ``(N, ny, nx)`` array of per-layer viscosity in Pa·s.
    gamma_dot_per_layer_s_inv
        ``(N, ny, nx)`` array of per-layer shear rate in s⁻¹.
    t_arr_s
        ``(ny, nx)`` array of local arrival times in seconds.
    h_total_mm
        ``(ny, nx)`` array of local cavity thickness in mm.
    density_kg_m3, specific_heat_J_kgK
        Volumetric heat capacity factors. Density at the melt state and
        constant-pressure specific heat.
    alpha_m2_s
        Thermal diffusivity in m²/s. Used only to compute τ_thermal.

    Returns
    -------
    np.ndarray
        ``(N, ny, nx)`` array of temperature rises in K. Non-negative
        and bounded above by the per-cell steady-state estimate
        ``η·γ̇²·τ_thermal / (ρ·cp)``.
    """
    rho_cp = max(float(density_kg_m3) * float(specific_heat_J_kgK), 1e-6)
    alpha = max(float(alpha_m2_s), 1e-15)

    h_m = np.maximum(h_total_mm * 1e-3, _H_FLOOR_MM * 1e-3)  # (ny, nx)
    tau_thermal = (h_m**2) / (np.pi**2 * alpha)  # (ny, nx)

    eta = np.asarray(eta_per_layer_Pa_s, dtype=float)
    gamma = np.asarray(gamma_dot_per_layer_s_inv, dtype=float)
    if eta.shape != gamma.shape:
        raise ValueError(f"eta and gamma shapes mismatch: {eta.shape} vs {gamma.shape}")

    # Volumetric heat source [W/m³]
    q_dot = eta * (gamma**2)
    # Cap the effective dwell time at τ_thermal so the rise can't grow
    # without bound for long t_arr.
    t_eff = np.minimum(t_arr_s[None, :, :], tau_thermal[None, :, :])
    t_eff = np.maximum(t_eff, 0.0)
    return q_dot * t_eff / rho_cp


def brinkman_number(
    eta_per_layer_Pa_s: np.ndarray,
    gamma_dot_per_layer_s_inv: np.ndarray,
    h_total_mm: np.ndarray,
    thermal_conductivity_W_mK: float,
    delta_T_K: float,
) -> np.ndarray:
    """Brinkman number ``Br = η·γ̇²·h² / (k·ΔT)``.

    Classical diagnostic for the importance of viscous dissipation
    relative to wall conduction.

    * ``Br < 1`` — conduction dominates, shear heating can be neglected.
    * ``Br ≈ 1`` — both effects are comparable.
    * ``Br > 1`` — shear heating dominates; the stage-1 correction may
      under- or over-estimate the temperature rise and a full 1D FDM
      (stage 2) is needed for accurate predictions.

    Parameters
    ----------
    eta_per_layer_Pa_s, gamma_dot_per_layer_s_inv
        ``(N, ny, nx)`` arrays in Pa·s and s⁻¹.
    h_total_mm
        ``(ny, nx)`` array of local thickness in mm.
    thermal_conductivity_W_mK
        Material thermal conductivity ``k`` in W/(m·K). Derived from
        ``α · ρ · cp`` when not measured separately.
    delta_T_K
        Reference temperature gap used as denominator (typically
        ``T_melt − T_mold``) in K. Must be positive.

    Returns
    -------
    np.ndarray
        ``(N, ny, nx)`` array of dimensionless Brinkman numbers.
    """
    k = float(thermal_conductivity_W_mK)
    dT = float(delta_T_K)
    if k <= 0:
        raise ValueError(f"thermal_conductivity_W_mK must be > 0, got {k}")
    if dT <= 0:
        raise ValueError(f"delta_T_K must be > 0, got {dT}")

    eta = np.asarray(eta_per_layer_Pa_s, dtype=float)
    gamma = np.asarray(gamma_dot_per_layer_s_inv, dtype=float)
    if eta.shape != gamma.shape:
        raise ValueError(f"eta and gamma shapes mismatch: {eta.shape} vs {gamma.shape}")

    h_m = np.maximum(h_total_mm * 1e-3, _H_FLOOR_MM * 1e-3)  # (ny, nx)
    return eta * (gamma**2) * (h_m[None, :, :] ** 2) / (k * dT)
