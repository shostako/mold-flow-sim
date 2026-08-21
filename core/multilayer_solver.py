"""Multilayer Hele-Shaw solver.

This module introduces ``MultilayerHeleShawSolver`` — a Hele-Shaw τ solver
that discretizes the cavity thickness into ``N`` layers. Each layer
contributes to the integrated planar conductance ``S_total`` via the
exact Poiseuille moment integral:

    S_total(x,y) = (h_total³ / 2) · Σ_{k=1..N} m_k / η_k

with the dimensionless moment

    m_k = [ζ² / 2 − ζ³ / 3]_{ζ_{k-1}}^{ζ_k}

where ``ζ_k ∈ [0, 1]`` are the layer boundaries in the normalized
thickness coordinate. For ``N=1`` (``ζ_0=0, ζ_1=1``) the moment evaluates
to ``m_1 = 1/6`` and the formula collapses to the classical Hele-Shaw
relation ``S = h³ / (12 η)`` — so this scheme is **numerically
equivalent** to the existing ``HeleShawSolver`` when ``num_layers == 1``
and the per-layer viscosity equals the single representative value.

Two coupling modes are supported:

* ``thermal_coupling=False`` (PR-A behaviour): every layer carries the
  same representative viscosity. The τ field is solved once and matches
  ``HeleShawSolver`` exactly at ``num_layers == 1``.
* ``thermal_coupling=True`` (PR-B, default): the cavity-wall cooling is
  captured by a 1D Neumann temperature profile (see
  :mod:`core.multilayer_thermal`); each layer gets its own
  ``T_k(x,y)`` and ``η_k(x,y)`` via Cross-WLF, and a fixed-point loop
  couples τ ↔ t_arr ↔ T_k ↔ η_k ↔ S_total until the relative L² change
  in τ drops below ``convergence_tol``.

The existing ``HeleShawSolver`` is **not modified**. We hold a private
instance and reuse its helper methods (``_effective_viscosity``,
``_open_thickness_field``, ``_solve_tau_field``, ``_compute_weld_score``,
``_compute_air_traps``).

Later PRs add: wall-refined layer distribution, adaptive damping,
short-shot detection, dedicated visualisers, and UI/CLI integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import Geometry
from .materials import Material, cross_wlf_viscosity
from .multilayer_thermal import (
    brinkman_number,
    neumann_layer_temperatures,
    poiseuille_shear_rates,
    shear_heating_temperature_rise,
)
from .solver import (
    FlowResult,
    HeleShawSolver,
    check_gate_reachability,
    weld_score_from_angle,
)


@dataclass
class MultilayerFlowResult(FlowResult):
    """Flow result returned by :class:`MultilayerHeleShawSolver`.

    Inherits every field from :class:`FlowResult` (so all existing
    visualisers and exporters that expect ``tau`` / ``fill_time_s`` /
    ``pressure_norm`` continue to work without changes) and adds the
    layer-resolved fields produced when the thermal coupling is active.

    The ``layer_*`` fields are ``None`` when the solver was run with
    ``thermal_coupling=False`` (PR-A behaviour).
    """

    layer_thickness_mm: np.ndarray | None = None  # (N, ny, nx) [mm]
    layer_temperature_K: np.ndarray | None = None  # (N, ny, nx) [K]
    layer_viscosity_Pa_s_field: np.ndarray | None = None  # (N, ny, nx) [Pa·s]
    layer_shear_rate_s_inv: np.ndarray | None = None  # (N, ny, nx) [1/s]
    # Shear-heating (stage 1) diagnostic fields. Always populated when
    # ``thermal_coupling=True`` so visualisers / callers can decide
    # whether the correction is needed even when it was *off* during
    # the solve. ``None`` when the solver was run without thermal
    # coupling at all.
    layer_shear_heating_dT_K: np.ndarray | None = None  # (N, ny, nx) [K]
    layer_brinkman_number: np.ndarray | None = None  # (N, ny, nx) [-]


def _uniform_layer_zeta(num_layers: int) -> np.ndarray:
    """Layer boundaries ζ_k ∈ [0, 1] for the uniform distribution.

    Returns an array of length ``num_layers + 1`` with ``ζ_0 = 0`` and
    ``ζ_N = 1``.
    """
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1 (got {num_layers})")
    return np.linspace(0.0, 1.0, num_layers + 1)


def _wall_refined_layer_zeta(num_layers: int) -> np.ndarray:
    """Layer boundaries ζ_k ∈ [0, 1] clustered near the walls.

    Uses Chebyshev-Lobatto points

        ζ_k = 0.5 · (1 − cos(π k / N))   for k = 0..N

    These are symmetric about ``ζ = 0.5`` (mirror invariance) and put
    extra resolution near the walls — exactly where the Neumann
    temperature gradient is steepest. For ``N = 6`` the boundaries are
    ``[0, 0.067, 0.25, 0.5, 0.75, 0.933, 1]``.

    Notes
    -----
    Requires ``num_layers >= 2`` because Chebyshev-Lobatto with a single
    interval collapses to the endpoints (no refinement effect, and
    moments would equal the uniform case). For ``num_layers == 1`` we
    fall back to ``_uniform_layer_zeta`` so the N=1 ↔ classical
    Hele-Shaw identity is preserved.
    """
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1 (got {num_layers})")
    if num_layers == 1:
        return _uniform_layer_zeta(1)
    k = np.arange(num_layers + 1, dtype=float)
    return 0.5 * (1.0 - np.cos(np.pi * k / num_layers))


def _layer_zeta(num_layers: int, distribution: str) -> np.ndarray:
    """Dispatch to the layer-distribution generator.

    Supported values: ``"uniform"``, ``"wall_refined"``.
    """
    if distribution == "uniform":
        return _uniform_layer_zeta(num_layers)
    if distribution == "wall_refined":
        return _wall_refined_layer_zeta(num_layers)
    raise ValueError(
        f"layer_distribution={distribution!r} not supported (expected 'uniform' or 'wall_refined')"
    )


def _poiseuille_layer_moments(zeta: np.ndarray) -> np.ndarray:
    """Per-layer dimensionless Poiseuille moments ``m_k``.

    Defined as

        m_k = ∫_{ζ_{k-1}}^{ζ_k} ζ (1 − ζ) dζ
            = [ζ² / 2 − ζ³ / 3]_{ζ_{k-1}}^{ζ_k}

    The full-thickness integral is ``Σ m_k = 1/6``, which reproduces the
    classical Hele-Shaw factor ``h³ / 12η = h³ / 2 · 1/6 / η``.

    Returns an array of length ``len(zeta) - 1``.
    """
    primitive = (zeta**2) / 2.0 - (zeta**3) / 3.0
    return np.diff(primitive)


def _multilayer_conductance(
    h_total_mm: np.ndarray,
    eta_per_layer_Pa_s: np.ndarray,
    moments: np.ndarray,
    cavity_mask: np.ndarray,
) -> np.ndarray:
    """Compute the depth-integrated conductance ``S_total`` (mm³ Pa⁻¹ s⁻¹
    converted to SI internally).

    Parameters
    ----------
    h_total_mm
        ``(ny, nx)`` array of local total cavity thickness in mm.
    eta_per_layer_Pa_s
        Either a scalar (one value applied to every layer) or an array of
        shape ``(N,)`` (one value per layer, no spatial variation) or
        ``(N, ny, nx)`` (per-layer per-cell values, used from PR-B
        onward).
    moments
        ``(N,)`` array of dimensionless layer moments ``m_k``.
    cavity_mask
        ``(ny, nx)`` boolean array; cells outside the cavity get
        ``S = 0``.

    Returns
    -------
    np.ndarray
        ``(ny, nx)`` array of ``S_total`` in SI units (m³ / (Pa · s)),
        matching the convention of :meth:`HeleShawSolver._conductance_field`.
    """
    h_m = h_total_mm * 1e-3  # mm -> m
    h_cubed = h_m**3

    eta = np.asarray(eta_per_layer_Pa_s, dtype=float)
    if eta.ndim == 0:
        # Scalar: broadcast m_k / eta over k.
        moment_sum = float(np.sum(moments)) / max(float(eta), 1e-3)
        S = 0.5 * h_cubed * moment_sum
    elif eta.ndim == 1:
        # Per-layer scalar η_k.
        if eta.shape[0] != moments.shape[0]:
            raise ValueError(
                f"eta_per_layer length {eta.shape[0]} does not match "
                f"moments length {moments.shape[0]}"
            )
        moment_sum = float(np.sum(moments / np.maximum(eta, 1e-3)))
        S = 0.5 * h_cubed * moment_sum
    else:
        # Per-layer per-cell η_k(x,y); shape (N, ny, nx).
        if eta.shape[0] != moments.shape[0] or eta.shape[1:] != h_total_mm.shape:
            raise ValueError(
                f"eta_per_layer shape {eta.shape} does not match "
                f"(N={moments.shape[0]}, ny, nx)={(moments.shape[0],) + h_total_mm.shape}"
            )
        m_over_eta = moments[:, None, None] / np.maximum(eta, 1e-3)
        moment_sum = m_over_eta.sum(axis=0)  # (ny, nx)
        S = 0.5 * h_cubed * moment_sum

    S = np.where(cavity_mask, S, 0.0)
    return S


@dataclass
class MultilayerHeleShawSolver:
    """Multilayer Hele-Shaw τ solver with optional thermal coupling.

    Parameters mirror :class:`HeleShawSolver` plus fields for the layer
    discretisation and the fixed-point coupling. The existing solver is
    re-used internally for its helper methods — no duplicated linear
    algebra here.

    With ``num_layers=1`` and ``thermal_coupling=False`` the result is
    byte-equivalent to ``HeleShawSolver``. With ``thermal_coupling=True``
    (the default) the wall-cooling 1D Neumann profile drives a
    per-layer ``T_k`` → ``η_k`` (Cross-WLF) cascade and the τ ↔ S_total
    coupling is iterated until convergence.
    """

    geometry: Geometry
    material: Material

    melt_temperature_K: float = 503.15
    mold_temperature_K: float = 313.15
    injection_velocity_mms: float = 100.0
    injection_volume_flow_cm3s: float | None = None

    compression_molding: bool = False
    compression_factor: float = 1.5
    compression_stroke_mm: float | None = None
    compression_fraction: float = 0.6

    num_layers: int = 5
    layer_distribution: str = "uniform"

    # ----- thermal coupling (PR-B) -----
    thermal_coupling: bool = True
    max_iterations: int = 8
    convergence_tol: float = 1e-3
    shear_rate_floor_factor: float = 0.01

    # ----- PR-C: adaptive damping + short-shot detection -----
    # When the relative L² change in τ grows between two iterations the
    # update is relaxed with ω = ``damping_factor``: τ ← (1-ω)·τ_old + ω·τ_new.
    # The default keeps the loop fully aggressive (ω=1) on the first
    # well-behaved pass and only kicks in on divergence.
    damping_factor: float = 0.7
    # Short-shot is flagged when the centre layer's temperature falls
    # below ``T_mold + solidification_temperature_fraction · (T_melt − T_mold)``.
    # 0.3 is a coarse PP-friendly default (Tg-adjacent for amorphous,
    # safely above no-flow for semi-crystalline). Material-specific values
    # are out of scope until ``data/materials.json`` grows a solidus field.
    solidification_temperature_fraction: float = 0.3

    # ----- shear heating (viscous dissipation), stage 1 -----
    # When ON, the Neumann temperature field is corrected by
    # ΔT_k = (η_k·γ̇_k²)·min(t_arr, τ_thermal) / (ρ·cp). This raises
    # local temperatures, drops the Cross-WLF viscosity, and feeds back
    # into the fixed-point loop. Off by default for backwards
    # compatibility with existing tests / cases; recommended ON for
    # ultra-thin plates (t < 0.5 mm) where Br ≫ 1.
    shear_heating_enabled: bool = False

    # Internal helper, populated in __post_init__.
    _base: HeleShawSolver = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1 (got {self.num_layers})")
        self._base = HeleShawSolver(
            geometry=self.geometry,
            material=self.material,
            melt_temperature_K=self.melt_temperature_K,
            mold_temperature_K=self.mold_temperature_K,
            injection_velocity_mms=self.injection_velocity_mms,
            injection_volume_flow_cm3s=self.injection_volume_flow_cm3s,
            compression_molding=self.compression_molding,
            compression_factor=self.compression_factor,
            compression_stroke_mm=self.compression_stroke_mm,
            compression_fraction=self.compression_fraction,
        )

    # ------------------------------------------------------------------
    # Layer-aware geometry helpers
    # ------------------------------------------------------------------

    def layer_zeta(self) -> np.ndarray:
        """Layer boundaries ζ_k ∈ [0, 1] (length ``num_layers + 1``)."""
        return _layer_zeta(self.num_layers, self.layer_distribution)

    def layer_moments(self) -> np.ndarray:
        """Per-layer Poiseuille moments ``m_k`` (length ``num_layers``)."""
        return _poiseuille_layer_moments(self.layer_zeta())

    def layer_thickness_mm(self, h_total_mm: np.ndarray) -> np.ndarray:
        """Per-layer absolute thickness ``h_k(x,y) = (ζ_k − ζ_{k-1}) · h_total``.

        Returns a ``(N, ny, nx)`` array. The layer-sum is exactly equal
        to ``h_total_mm`` up to floating-point round-off.
        """
        d_zeta = np.diff(self.layer_zeta())  # (N,)
        return d_zeta[:, None, None] * h_total_mm[None, :, :]

    def layer_zeta_centers(self) -> np.ndarray:
        """Layer-centre coordinates ``ζ_k_center = (ζ_{k-1} + ζ_k) / 2``.

        Returns a ``(num_layers,)`` array used as the evaluation point
        for the per-layer temperature and shear rate.
        """
        zeta = self.layer_zeta()
        return 0.5 * (zeta[:-1] + zeta[1:])

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def solve(self, num_frames: int = 24) -> MultilayerFlowResult:
        """Solve the Hele-Shaw τ field using the layer-integrated conductance.

        With ``thermal_coupling=False`` this is a single solve with a
        uniform per-layer viscosity (PR-A behaviour). With
        ``thermal_coupling=True`` (default) a fixed-point loop couples
        τ ↔ t_arr ↔ T_k ↔ η_k ↔ S_total.
        """
        if not self.geometry.gates:
            raise ValueError("Geometry has no gates")
        check_gate_reachability(self.geometry)

        base = self._base
        eta_baseline = base._effective_viscosity()

        dirichlet = np.zeros(self.geometry.shape, dtype=bool)
        for iy, ix in self.geometry.gates:
            dirichlet[iy, ix] = True

        h_open = base._open_thickness_field()  # mm
        cavity_mask = self.geometry.mask

        # absolute time scaling baseline (same logic as HeleShawSolver.solve)
        V_cm3 = self.geometry.volume_cm3()
        if self.injection_volume_flow_cm3s is None:
            T_fill_baseline = 1.5
        else:
            Q = max(float(self.injection_volume_flow_cm3s), 1e-6)
            T_fill_baseline = V_cm3 / Q
        if self.compression_molding:
            V_total_mm3 = self.geometry.volume_cm3() * 1000.0
            if self.compression_stroke_mm is not None:
                stroke = float(self.compression_stroke_mm)
                A_cm_mm2 = self.geometry.compression_area_mm2()
                delta_V = stroke * A_cm_mm2
                effective_factor = 1.0 + (delta_V / max(V_total_mm3, 1e-9))
            else:
                f_comp = float(self.geometry.compression_volume_fraction())
                f_comp = max(min(f_comp, 1.0), 0.0)
                effective_factor = 1.0 + (float(self.compression_factor) - 1.0) * f_comp
            effective_factor = max(effective_factor, 1e-3)
            T_fill_baseline = T_fill_baseline * (
                self.compression_fraction / effective_factor + (1.0 - self.compression_fraction)
            )

        moments = self.layer_moments()
        zeta_centers = self.layer_zeta_centers()
        h_layers = self.layer_thickness_mm(h_open)  # (N, ny, nx)

        # Baseline τ solve with the uniform representative viscosity.
        S_baseline = _multilayer_conductance(
            h_total_mm=h_open,
            eta_per_layer_Pa_s=eta_baseline,
            moments=moments,
            cavity_mask=cavity_mask,
        )
        tau, tau_max = base._solve_tau_field(S_baseline, dirichlet)
        tau_max_baseline = tau_max
        tau_baseline = tau.copy()
        dx = float(self.geometry.cell_size_mm)
        cell_volume = dx * dx * h_open  # mm^3, what each cell adds when swept
        T_fill = T_fill_baseline

        layer_T_K: np.ndarray | None = None
        layer_eta_Pa_s: np.ndarray | None = None
        layer_gamma_dot: np.ndarray | None = None
        layer_shear_dT_K: np.ndarray | None = None
        layer_Brinkman: np.ndarray | None = None
        short_shot_mask: np.ndarray | None = None
        iters_done = 0
        converged = False
        damping_events = 0
        T_fill_inflation = 1.0
        tau_rep_flow: float | None = None
        tau_rep_baseline: float | None = None
        T_solid_K = self.mold_temperature_K + float(self.solidification_temperature_fraction) * (
            self.melt_temperature_K - self.mold_temperature_K
        )

        if self.thermal_coupling:
            alpha = max(float(self.material.thermal_diffusivity_m2_s), 0.0)
            tol = max(float(self.convergence_tol), 1e-12)
            max_iters = max(int(self.max_iterations), 1)
            omega = float(self.damping_factor)
            if not 0.0 < omega <= 1.0:
                raise ValueError(f"damping_factor must be in (0, 1] (got {self.damping_factor})")

            # Pre-compute the layer-resolved shear-rate field — independent
            # of the iteration since V and h_open are fixed.
            layer_gamma_dot = poiseuille_shear_rates(
                zeta_centers=zeta_centers,
                V_mms=float(self.injection_velocity_mms),
                h_total_mm=h_open,
                floor_factor=float(self.shear_rate_floor_factor),
            )

            # Bulk temperature inside the cavity at the start of the loop.
            # Cells outside the cavity get T_melt as a harmless placeholder
            # (their conductance is forced to zero anyway).
            T_bulk = 0.7 * self.melt_temperature_K + 0.3 * self.mold_temperature_K

            prev_rel: float | None = None
            for it in range(1, max_iters + 1):
                msk = ~np.isnan(tau)
                # Arrival time through the volume CDF -- the same map as
                # HeleShawSolver (Issue #52): at constant rate a cell arrives
                # when the volume at or below its tau has been injected. The
                # old linear map reported the tau profile as if it were time,
                # and its single-cell denominator let one pathological cell
                # rescale every Neumann temperature in the cavity.
                t_arr = base._arrival_time_field(tau, msk, cell_volume, T_fill)
                # outside the cavity → tiny ε so neumann_layer_temperatures
                # does not see zero (it has its own floor too).
                t_arr = np.where(np.isnan(t_arr), 0.0, t_arr)

                layer_T_K = neumann_layer_temperatures(
                    zeta_centers=zeta_centers,
                    t_arr_s=t_arr,
                    h_total_mm=h_open,
                    T_melt_K=float(self.melt_temperature_K),
                    T_mold_K=float(self.mold_temperature_K),
                    alpha_m2_s=alpha,
                )

                # Shear-heating correction (stage 1, optional). Uses the
                # *previous* iteration's per-layer viscosity to evaluate
                # the volumetric heat source — this lags by one iteration
                # but converges along with the rest of the fixed-point
                # since η drops as T rises.
                if self.shear_heating_enabled:
                    if layer_eta_Pa_s is None:
                        # First iteration: bootstrap with the bulk
                        # representative viscosity, broadcast to all
                        # layers and cells.
                        eta_prev_field = np.full(
                            (self.num_layers,) + h_open.shape,
                            float(eta_baseline),
                            dtype=float,
                        )
                    else:
                        eta_prev_field = layer_eta_Pa_s
                    layer_shear_dT_K = shear_heating_temperature_rise(
                        eta_per_layer_Pa_s=eta_prev_field,
                        gamma_dot_per_layer_s_inv=layer_gamma_dot,
                        t_arr_s=t_arr,
                        h_total_mm=h_open,
                        density_kg_m3=float(self.material.density_melt_kgm3),
                        specific_heat_J_kgK=float(self.material.specific_heat_J_kgK),
                        alpha_m2_s=alpha,
                    )
                    # Apply only inside the cavity (outside cells have
                    # T_bulk placeholders that should not be perturbed).
                    layer_shear_dT_K = np.where(cavity_mask[None, :, :], layer_shear_dT_K, 0.0)
                    layer_T_K = layer_T_K + layer_shear_dT_K

                # Cells outside the cavity carry no meaningful temperature.
                # Use the bulk so cross_wlf is well-defined (the conductance
                # masks them out anyway).
                no_cavity = ~cavity_mask[None, :, :]
                T_eval = np.where(no_cavity, T_bulk, layer_T_K)

                eta_field = cross_wlf_viscosity(self.material, T_eval, layer_gamma_dot, 0.0)
                layer_eta_Pa_s = np.asarray(eta_field, dtype=float)

                S_new = _multilayer_conductance(
                    h_total_mm=h_open,
                    eta_per_layer_Pa_s=layer_eta_Pa_s,
                    moments=moments,
                    cavity_mask=cavity_mask,
                )
                tau_solved, tau_max_solved = base._solve_tau_field(S_new, dirichlet)

                msk_tau = ~np.isnan(tau)
                num = float(np.linalg.norm(tau_solved[msk_tau] - tau[msk_tau]))
                den = float(np.linalg.norm(tau[msk_tau])) + 1e-12
                rel = num / den

                # Adaptive damping: if the residual grew vs the previous
                # iteration, blend the new field with the old one. This
                # preserves the linear-solve τ_max (the damped τ is a
                # convex combination, so its maximum stays bounded by the
                # two endpoints; we recompute the working τ_max from the
                # damped field).
                if prev_rel is not None and rel > prev_rel:
                    tau_damped = np.where(
                        msk_tau, (1.0 - omega) * tau + omega * tau_solved, tau_solved
                    )
                    tau_new = tau_damped
                    tau_max_new = float(np.nanmax(tau_new)) if msk_tau.any() else tau_max_solved
                    if tau_max_new <= 0:
                        tau_max_new = tau_max_solved
                    damping_events += 1
                else:
                    tau_new = tau_solved
                    tau_max_new = tau_max_solved

                # Constant-pressure proxy: T_fill scales with the growth of
                # the volume-weighted mean τ over the cavity -- a resistance
                # representative no single cell can own (Issue #52). On a
                # uniform inflation (τ scaling by one factor everywhere) it
                # reproduces the old max-ratio exactly.
                rep_new = base._tau_volume_mean(tau_new, cavity_mask, cell_volume)
                rep_base = base._tau_volume_mean(tau_baseline, cavity_mask, cell_volume)
                if rep_new is None or rep_base is None:
                    T_fill_new = T_fill_baseline
                else:
                    T_fill_new = T_fill_baseline * (rep_new / rep_base)
                tau_rep_flow = rep_new
                tau_rep_baseline = rep_base

                tau = tau_new
                tau_max = tau_max_new
                T_fill = T_fill_new
                iters_done = it
                prev_rel = rel
                if rel < tol:
                    converged = True
                    break

            T_fill_inflation = T_fill / T_fill_baseline if T_fill_baseline > 0 else 1.0

            # Short-shot detection: centre-layer temperature has dropped
            # below the solidification threshold. Uses the *final* layer
            # temperature snapshot (post-fixed-point), since that reflects
            # the converged T_fill scaling.
            if layer_T_K is not None and self.num_layers >= 1:
                k_mid = self.num_layers // 2
                T_mid = layer_T_K[k_mid]
                short_shot_mask = cavity_mask & (T_mid <= T_solid_K)

            # Diagnostic Brinkman number from the converged state. Always
            # computed (even when shear_heating_enabled is False) so the
            # user can decide whether the correction is needed for their
            # geometry / injection conditions.
            if layer_eta_Pa_s is not None and layer_gamma_dot is not None:
                delta_T_ref = max(
                    float(self.melt_temperature_K) - float(self.mold_temperature_K), 1.0
                )
                layer_Brinkman = brinkman_number(
                    eta_per_layer_Pa_s=layer_eta_Pa_s,
                    gamma_dot_per_layer_s_inv=layer_gamma_dot,
                    h_total_mm=h_open,
                    thermal_conductivity_W_mK=self.material.thermal_conductivity_W_mK,
                    delta_T_K=delta_T_ref,
                )

        # Standard post-processing (mirrors HeleShawSolver.solve).
        msk = ~np.isnan(tau)
        # Volume-CDF map, same as HeleShawSolver: the two solver modes must
        # not disagree about what "fill time at a cell" means.
        fill_time_s = base._arrival_time_field(tau, msk, cell_volume, T_fill)
        pressure_norm = np.full_like(tau, np.nan)
        pressure_norm[msk] = 1.0 - tau[msk] / tau_max
        weld_angle = base._weld_meeting_angle(tau)
        weld_score = weld_score_from_angle(weld_angle)
        air_traps = base._compute_air_traps(tau)

        metadata = {
            "material": self.material.name,
            "melt_K": self.melt_temperature_K,
            "mold_K": self.mold_temperature_K,
            "injection_velocity_mms": self.injection_velocity_mms,
            "injection_Q_cm3s": self.injection_volume_flow_cm3s,
            "compression": self.compression_molding,
            "compression_factor": self.compression_factor,
            "compression_stroke_mm": self.compression_stroke_mm,
            "compression_mode": ("stroke" if self.compression_stroke_mm is not None else "factor"),
            "compression_fraction": self.compression_fraction,
            "tau_max": tau_max,
            "tau_max_baseline": tau_max_baseline,
            "volume_cm3": V_cm3,
            "num_frames": num_frames,
            "skin_layer_enabled": False,
            # multilayer-specific
            "solver_kind": "multilayer",
            "num_layers": int(self.num_layers),
            "layer_distribution": self.layer_distribution,
            "layer_zeta": self.layer_zeta().tolist(),
            "layer_moments": moments.tolist(),
            "thermal_coupling": self.thermal_coupling,
            "thermal_diffusivity_m2_s": self.material.thermal_diffusivity_m2_s,
            "multilayer_iterations": iters_done,
            "multilayer_converged": converged,
            "multilayer_max_iterations": int(self.max_iterations),
            "multilayer_convergence_tol": float(self.convergence_tol),
            "T_fill_baseline_s": T_fill_baseline,
            "T_fill_inflation": T_fill_inflation,
            "tau_rep_flow": tau_rep_flow,
            "tau_rep_baseline": tau_rep_baseline,
            "damping_factor": float(self.damping_factor),
            "damping_events": int(damping_events),
            "T_solid_K": float(T_solid_K),
            "solidification_temperature_fraction": float(self.solidification_temperature_fraction),
            # shear-heating (stage 1) diagnostics — always populated when
            # thermal_coupling is True so users can judge whether the
            # correction is needed; only meaningful when the layer fields
            # exist (thermal_coupling=False leaves them None).
            "shear_heating_enabled": bool(self.shear_heating_enabled),
            "specific_heat_J_kgK": float(self.material.specific_heat_J_kgK),
            "thermal_conductivity_W_mK": float(self.material.thermal_conductivity_W_mK),
        }
        if layer_shear_dT_K is not None:
            # Only the cavity cells carry meaningful values (outside cells
            # were zeroed during the loop).
            cavity_dT = layer_shear_dT_K[:, cavity_mask] if cavity_mask.any() else layer_shear_dT_K
            metadata.update(
                {
                    "shear_heating_max_K": float(np.max(cavity_dT)) if cavity_dT.size else 0.0,
                    "shear_heating_mean_K": float(np.mean(cavity_dT)) if cavity_dT.size else 0.0,
                }
            )
        else:
            metadata.update(
                {
                    "shear_heating_max_K": 0.0,
                    "shear_heating_mean_K": 0.0,
                }
            )
        if layer_Brinkman is not None:
            cavity_Br = layer_Brinkman[:, cavity_mask] if cavity_mask.any() else layer_Brinkman
            metadata.update(
                {
                    "brinkman_number_max": float(np.max(cavity_Br)) if cavity_Br.size else 0.0,
                    "brinkman_number_mean": float(np.mean(cavity_Br)) if cavity_Br.size else 0.0,
                }
            )
        else:
            metadata.update(
                {
                    "brinkman_number_max": 0.0,
                    "brinkman_number_mean": 0.0,
                }
            )
        if short_shot_mask is not None:
            cells_total = max(int(cavity_mask.sum()), 1)
            short_count = int(short_shot_mask.sum())
            metadata.update(
                {
                    "short_shot_cells": short_count,
                    "short_shot_fraction": short_count / cells_total,
                }
            )
        else:
            metadata.update(
                {
                    "short_shot_cells": 0,
                    "short_shot_fraction": 0.0,
                }
            )

        return MultilayerFlowResult(
            tau=tau,
            fill_time_s=fill_time_s,
            pressure_norm=pressure_norm,
            weld_score=weld_score,
            weld_angle_deg=weld_angle,
            air_traps=air_traps,
            total_fill_time_s=float(T_fill),
            viscosity_Pa_s=eta_baseline,
            geometry=self.geometry,
            metadata=metadata,
            short_shot_mask=short_shot_mask,
            layer_thickness_mm=h_layers,
            layer_temperature_K=layer_T_K,
            layer_viscosity_Pa_s_field=layer_eta_Pa_s,
            layer_shear_rate_s_inv=layer_gamma_dot,
            layer_shear_heating_dT_K=layer_shear_dT_K,
            layer_brinkman_number=layer_Brinkman,
        )
