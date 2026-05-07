"""Hele-Shaw fill-time solver (Pseudo Conduction method).

Reference idea: solve the elliptic problem
    div( S * grad(tau) ) = 1   in cavity Omega
    tau = 0                    at gates (Dirichlet)
    S * grad(tau) . n = 0      at cavity walls (Neumann, no flux)

where S = h^3 / (12 * eta_eff) is the Hele-Shaw flow conductance,
h is local cavity thickness, and eta_eff is an effective viscosity
evaluated at representative shear rate and bulk temperature.

The resulting scalar field tau is a monotonic "arrival time" map
(pseudo conduction time). Normalising by tau_max and scaling by the
total fill time T_fill = V_cavity / Q reproduces the time evolution of
the flow front as level sets of tau.

Caveats:
- Single representative shear rate (no local rate iteration).
- Compression molding modeled as an effective thickness inflation
  during the first compress_fraction of the fill, lowering local
  flow resistance. Only cells flagged in
  ``geometry.compression_mask`` are inflated (parametric builders
  set this to the product-body cells, leaving runners/sprues at
  their cast thickness). When ``compression_mask`` is ``None`` the
  whole cavity inflates (legacy / image input).

Skin-layer model (optional, ``skin_layer_enabled=True``):

The skin layer that forms when melt contacts the cold mold wall is
approximated by a Stefan/Neumann form

    s(t) = c_skin * sqrt(alpha * t)

where ``alpha`` is the material's thermal diffusivity and ``c_skin``
is a non-dimensional growth constant. The flow conducts only through
the live core ``h_core = max(h - 2 * s, h_min)`` and the conductance
becomes ``S = h_core^3 / (12 * eta)``. Because ``s`` depends on the
arrival time and the arrival time depends on ``S``, the fields are
solved by fixed-point iteration. When the skin layers from opposite
walls meet (``h - 2 * s <= h_min``) the cell is flagged as a
short-shot candidate and the absolute fill time ``T_fill`` is scaled
up by the relative growth of ``tau_max`` (constant-pressure proxy:
the inflated runtime mirrors the resistance increase). Bulk-melt
cooling and dynamic viscosity coupling remain disabled — the model
captures the wall-side freezing front in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .geometry import Geometry
from .materials import Material, cross_wlf_viscosity, representative_shear_rate


@dataclass
class FlowResult:
    tau: np.ndarray  # raw pseudo-conduction time field
    fill_time_s: np.ndarray  # actual fill time per cell [s]
    pressure_norm: np.ndarray  # normalized pressure (1 at gate, 0 at last fill)
    weld_score: np.ndarray  # heuristic weld-line indicator [0..1]
    air_traps: np.ndarray  # bool mask of air-trap cells (local tau maxima)
    total_fill_time_s: float  # T_fill from volume / Q (scaled when skin layer ON)
    viscosity_Pa_s: float  # effective representative viscosity used
    geometry: Geometry
    metadata: dict
    # Skin-layer model outputs (None when ``skin_layer_enabled`` is False).
    skin_thickness_mm: np.ndarray | None = None  # s(x,y) [mm]
    core_thickness_mm: np.ndarray | None = None  # h_core(x,y) = h - 2*s [mm]
    short_shot_mask: np.ndarray | None = None  # cells where the two skins met


@dataclass
class HeleShawSolver:
    geometry: Geometry
    material: Material

    melt_temperature_K: float = 503.15
    mold_temperature_K: float = 313.15
    injection_velocity_mms: float = 100.0  # average flow front velocity scale
    injection_volume_flow_cm3s: float | None = None  # if None, derived from V/T_fill_default

    compression_molding: bool = False
    compression_factor: float = 1.5  # h_effective / h_actual during compression phase
    compression_fraction: float = 0.6  # fraction of fill time under compression-open state

    pressure_iters: int = 1  # placeholder for future iteration on viscosity

    # ----- skin-layer (Stefan/Neumann) model -----
    skin_layer_enabled: bool = False
    skin_growth_constant: float = 1.0  # c_skin in s(t) = c_skin * sqrt(alpha * t)
    skin_max_iterations: int = 5  # fixed-point iterations for tau ↔ h_core coupling
    skin_convergence_tol: float = 1e-3  # relative L2 change in tau between iterations
    min_core_thickness_mm: float = 0.01  # h_core floor; cells at this floor are short shots

    def _effective_viscosity(self) -> float:
        # bulk temperature ~ weighted average (melt dominates while flowing)
        T_bulk = 0.7 * self.melt_temperature_K + 0.3 * self.mold_temperature_K
        thickness_mm_avg = float(np.mean(self.geometry.thickness_mm[self.geometry.mask]))
        gamma_dot = representative_shear_rate(
            self.injection_velocity_mms, max(thickness_mm_avg, 0.5)
        )
        eta = float(cross_wlf_viscosity(self.material, T_bulk, gamma_dot, 0.0))
        return eta

    def _open_thickness_field(self) -> np.ndarray:
        """Cavity thickness used as the skin-free reference (mm).

        Equivalent to ``geometry.thickness_mm``, expanded by
        ``compression_factor`` when compression molding is active. Only
        cells flagged in ``geometry.compression_mask`` participate in the
        inflation; runners / sprues / gates keep their original thickness.
        ``compression_mask = None`` falls back to legacy whole-cavity
        inflation. The skin-layer model carves into this reference field.
        """
        h_mm = self.geometry.thickness_mm.copy()
        if self.compression_molding:
            cm = self.geometry.compression_mask
            factor = float(self.compression_factor)
            if cm is None:
                h_mm = h_mm * factor
            else:
                target = cm & self.geometry.mask
                h_mm[target] = h_mm[target] * factor
        return h_mm

    def _conductance_field(
        self,
        eta: float,
        thickness_mm: np.ndarray | None = None,
    ) -> np.ndarray:
        """S = h^3 / (12 * eta) in SI units; h in m, eta in Pa.s, S in m^3/(Pa.s).

        ``thickness_mm`` is the local effective gap (e.g. ``h_core`` when the
        skin-layer model is active). Defaults to the open cavity thickness.
        """
        if thickness_mm is None:
            h_mm = self._open_thickness_field()
        else:
            h_mm = thickness_mm
        h_m = h_mm * 1e-3
        S = (h_m**3) / (12.0 * max(eta, 1e-3))
        S[~self.geometry.mask] = 0.0
        return S

    def _build_linear_system(self, S: np.ndarray, dirichlet: np.ndarray):
        """Assemble Au = b for the Pseudo Conduction equation on the masked grid.
        dirichlet[i,j] = True means tau = 0 enforced at that cell.
        """
        ny, nx = self.geometry.shape
        dx = self.geometry.cell_size_mm * 1e-3  # cell size in meters
        # index map: only masked cells participate
        idx = -np.ones((ny, nx), dtype=np.int64)
        flat_indices = np.where(self.geometry.mask.ravel())[0]
        for k, fi in enumerate(flat_indices):
            iy, ix = divmod(fi, nx)
            idx[iy, ix] = k
        N = len(flat_indices)

        # use lil for build
        A = sp.lil_matrix((N, N), dtype=np.float64)
        b = np.zeros(N, dtype=np.float64)

        for k, fi in enumerate(flat_indices):
            iy, ix = divmod(fi, nx)
            if dirichlet[iy, ix]:
                A[k, k] = 1.0
                b[k] = 0.0
                continue

            S_c = S[iy, ix]
            diag = 0.0
            # neighbors: (di, dj)
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny_, nx_ = iy + di, ix + dj
                if 0 <= ny_ < ny and 0 <= nx_ < nx and self.geometry.mask[ny_, nx_]:
                    S_n = S[ny_, nx_]
                    # harmonic mean for face conductance
                    if S_c + S_n > 0:
                        S_face = 2.0 * S_c * S_n / (S_c + S_n)
                    else:
                        S_face = 0.0
                    coeff = S_face / (dx * dx)
                    diag += coeff
                    A[k, idx[ny_, nx_]] -= coeff
                # else: no-flux wall; do nothing
            A[k, k] = diag if diag > 0 else 1.0  # isolated cell guard
            # RHS: source = 1 (unit forcing).
            # For absolute time-scaling we use V/Q later, so absolute units of A,b cancel.
            b[k] = 1.0

        return A.tocsr(), b, idx

    def _solve_tau_field(self, S: np.ndarray, dirichlet: np.ndarray) -> tuple[np.ndarray, float]:
        """Solve the elliptic system for a given conductance field S.

        Returns ``(tau_grid, tau_max)`` where ``tau_grid`` is NaN outside
        the cavity mask and ``tau_max`` is positive (clamped to 1.0 if the
        solve degenerates).
        """
        A, b, _ = self._build_linear_system(S, dirichlet)
        tau_vec = spla.spsolve(A, b)
        ny, nx = self.geometry.shape
        tau = np.full(self.geometry.shape, np.nan, dtype=float)
        flat_indices = np.where(self.geometry.mask.ravel())[0]
        for k, fi in enumerate(flat_indices):
            iy, ix = divmod(fi, nx)
            tau[iy, ix] = tau_vec[k]
        tau_max = float(np.nanmax(tau)) if np.any(~np.isnan(tau)) else 1.0
        if tau_max <= 0:
            tau_max = 1.0
        return tau, tau_max

    def solve(self, num_frames: int = 24) -> FlowResult:
        if not self.geometry.gates:
            raise ValueError("Geometry has no gates")

        eta = self._effective_viscosity()

        dirichlet = np.zeros(self.geometry.shape, dtype=bool)
        for iy, ix in self.geometry.gates:
            dirichlet[iy, ix] = True

        h_open = self._open_thickness_field()  # mm
        cavity_mask = self.geometry.mask

        # absolute time scaling baseline (skin-layer-free, constant Q)
        V_cm3 = self.geometry.volume_cm3()
        if self.injection_volume_flow_cm3s is None:
            T_fill_baseline = 1.5
        else:
            Q = max(float(self.injection_volume_flow_cm3s), 1e-6)
            T_fill_baseline = V_cm3 / Q
        if self.compression_molding:
            # Effective inflation acting on the whole cavity (Q = const proxy).
            # When only the product body inflates (compression_mask set), the
            # net resistance drop is proportional to that body's volume share,
            # so the compression-phase speed-up is diluted accordingly.
            f_comp = float(self.geometry.compression_volume_fraction())
            f_comp = max(min(f_comp, 1.0), 0.0)
            effective_factor = 1.0 + (float(self.compression_factor) - 1.0) * f_comp
            effective_factor = max(effective_factor, 1e-3)
            T_fill_baseline = T_fill_baseline * (
                self.compression_fraction / effective_factor + (1.0 - self.compression_fraction)
            )

        # baseline solve (no skin) — also serves as the tau_max reference
        S0 = self._conductance_field(eta, h_open)
        tau, tau_max = self._solve_tau_field(S0, dirichlet)
        tau_max_baseline = tau_max
        T_fill = T_fill_baseline

        skin_thk_mm: np.ndarray | None = None
        h_core_mm: np.ndarray | None = None
        short_shot_mask: np.ndarray | None = None
        skin_iters_done = 0
        skin_converged = False

        if self.skin_layer_enabled:
            alpha = max(float(self.material.thermal_diffusivity_m2_s), 0.0)
            c_skin = max(float(self.skin_growth_constant), 0.0)
            min_core = max(float(self.min_core_thickness_mm), 1e-6)
            tol = max(float(self.skin_convergence_tol), 0.0)

            skin_thk_mm = np.zeros_like(h_open)
            h_core_mm = h_open.copy()

            for it in range(int(max(self.skin_max_iterations, 1))):
                msk = cavity_mask & ~np.isnan(tau)
                # arrival time per cell, scaled to current best estimate of T_fill
                t_arr = np.zeros_like(tau)
                t_arr[msk] = (tau[msk] / tau_max) * T_fill

                # skin layer thickness: s(t) = c_skin * sqrt(alpha * t) (m → mm)
                s_m = c_skin * np.sqrt(alpha * np.maximum(t_arr, 0.0))
                s_mm_new = (s_m * 1.0e3).astype(float)
                # cap so that h_core can never go below min_core
                s_mm_max = np.maximum((h_open - min_core) / 2.0, 0.0)
                s_mm_new = np.minimum(s_mm_new, s_mm_max)
                s_mm_new[~cavity_mask] = 0.0

                h_core_new = h_open - 2.0 * s_mm_new
                h_core_new = np.maximum(h_core_new, min_core)
                h_core_new[~cavity_mask] = 0.0

                # re-solve for tau with the carved core
                S_new = self._conductance_field(eta, h_core_new)
                tau_new, tau_max_new = self._solve_tau_field(S_new, dirichlet)

                # constant-pressure proxy: T_fill grows with the resistance
                T_fill_new = T_fill_baseline * (
                    tau_max_new / tau_max_baseline if tau_max_baseline > 0 else 1.0
                )

                # convergence check on tau (relative L2 over masked cells)
                msk_new = cavity_mask & ~np.isnan(tau_new) & ~np.isnan(tau)
                if msk_new.any():
                    diff = float(np.linalg.norm(tau_new[msk_new] - tau[msk_new]))
                    base = float(np.linalg.norm(tau[msk_new])) + 1e-12
                    rel = diff / base
                else:
                    rel = 0.0

                tau = tau_new
                tau_max = tau_max_new
                T_fill = T_fill_new
                skin_thk_mm = s_mm_new
                h_core_mm = h_core_new
                skin_iters_done = it + 1
                if rel < tol:
                    skin_converged = True
                    break

            short_shot_mask = (
                cavity_mask
                & ((h_open - 2.0 * skin_thk_mm) <= min_core + 1e-9)
                & (h_open > min_core + 1e-9)  # ignore cells whose open gap is already tiny
            )

        # absolute time scaling per cell
        msk = ~np.isnan(tau)
        fill_time_s = np.full_like(tau, np.nan)
        fill_time_s[msk] = (tau[msk] / tau_max) * T_fill

        # pressure proxy: P ~ (tau_max - tau) / tau_max -> 1 at gate, 0 at far field
        pressure_norm = np.full_like(tau, np.nan)
        pressure_norm[msk] = 1.0 - tau[msk] / tau_max

        weld_score = self._compute_weld_score(tau)
        air_traps = self._compute_air_traps(tau)

        metadata = {
            "material": self.material.name,
            "melt_K": self.melt_temperature_K,
            "mold_K": self.mold_temperature_K,
            "injection_velocity_mms": self.injection_velocity_mms,
            "injection_Q_cm3s": self.injection_volume_flow_cm3s,
            "compression": self.compression_molding,
            "compression_factor": self.compression_factor,
            "compression_fraction": self.compression_fraction,
            "tau_max": tau_max,
            "tau_max_baseline": tau_max_baseline,
            "volume_cm3": V_cm3,
            "num_frames": num_frames,
            "skin_layer_enabled": self.skin_layer_enabled,
        }
        if self.skin_layer_enabled:
            cells_total = max(int(cavity_mask.sum()), 1)
            short_count = int(short_shot_mask.sum()) if short_shot_mask is not None else 0
            metadata.update(
                {
                    "skin_growth_constant": self.skin_growth_constant,
                    "thermal_diffusivity_m2_s": self.material.thermal_diffusivity_m2_s,
                    "skin_iterations": skin_iters_done,
                    "skin_converged": skin_converged,
                    "min_core_thickness_mm": self.min_core_thickness_mm,
                    "T_fill_baseline_s": T_fill_baseline,
                    "T_fill_inflation": (T_fill / T_fill_baseline if T_fill_baseline > 0 else 1.0),
                    "short_shot_cells": short_count,
                    "short_shot_fraction": short_count / cells_total,
                }
            )

        return FlowResult(
            tau=tau,
            fill_time_s=fill_time_s,
            pressure_norm=pressure_norm,
            weld_score=weld_score,
            air_traps=air_traps,
            total_fill_time_s=float(T_fill),
            viscosity_Pa_s=eta,
            geometry=self.geometry,
            metadata=metadata,
            skin_thickness_mm=skin_thk_mm,
            core_thickness_mm=h_core_mm,
            short_shot_mask=short_shot_mask,
        )

    @staticmethod
    def _compute_weld_score(tau: np.ndarray) -> np.ndarray:
        """Heuristic: a cell where many neighbors have *smaller* tau is a confluence."""
        score = np.zeros_like(tau, dtype=float)
        ny, nx = tau.shape
        # 8-neighborhood
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for iy in range(1, ny - 1):
            for ix in range(1, nx - 1):
                if np.isnan(tau[iy, ix]):
                    continue
                t_c = tau[iy, ix]
                cnt_lower = 0
                cnt_valid = 0
                for di, dj in offsets:
                    t_n = tau[iy + di, ix + dj]
                    if np.isnan(t_n):
                        continue
                    cnt_valid += 1
                    if t_n < t_c:
                        cnt_lower += 1
                if cnt_valid >= 6 and cnt_lower >= 6:
                    # 6+ neighbors flowed in earlier => confluence ridge
                    score[iy, ix] = (cnt_lower - 5) / 3.0
        return np.clip(score, 0.0, 1.0)

    @staticmethod
    def _compute_air_traps(tau: np.ndarray) -> np.ndarray:
        """Local tau maxima (last-to-fill cells) — air-trap candidates."""
        ny, nx = tau.shape
        traps = np.zeros_like(tau, dtype=bool)
        for iy in range(1, ny - 1):
            for ix in range(1, nx - 1):
                t_c = tau[iy, ix]
                if np.isnan(t_c):
                    continue
                is_max = True
                valid = 0
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if di == 0 and dj == 0:
                            continue
                        t_n = tau[iy + di, ix + dj]
                        if np.isnan(t_n):
                            continue
                        valid += 1
                        if t_n > t_c:
                            is_max = False
                            break
                    if not is_max:
                        break
                if is_max and valid >= 5:
                    traps[iy, ix] = True
        return traps
