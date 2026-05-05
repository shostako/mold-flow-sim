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
- Iso-thermal (no thermal coupling).
- Single representative shear rate (no local rate iteration).
- Compression molding modeled as an effective thickness inflation
  during the first compress_fraction of the fill, lowering local
  flow resistance.
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
    total_fill_time_s: float  # T_fill from volume / Q
    viscosity_Pa_s: float  # effective representative viscosity used
    geometry: Geometry
    metadata: dict


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

    def _effective_viscosity(self) -> float:
        # bulk temperature ~ weighted average (melt dominates while flowing)
        T_bulk = 0.7 * self.melt_temperature_K + 0.3 * self.mold_temperature_K
        thickness_mm_avg = float(np.mean(self.geometry.thickness_mm[self.geometry.mask]))
        gamma_dot = representative_shear_rate(
            self.injection_velocity_mms, max(thickness_mm_avg, 0.5)
        )
        eta = float(cross_wlf_viscosity(self.material, T_bulk, gamma_dot, 0.0))
        return eta

    def _conductance_field(self, eta: float) -> np.ndarray:
        """S = h^3 / (12 * eta) in SI units; h in m, eta in Pa.s, S in m^3/(Pa.s)."""
        h_mm = self.geometry.thickness_mm.copy()
        if self.compression_molding:
            # uniform thickness inflation during open state — increases conductance
            h_mm = h_mm * float(self.compression_factor)
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

    def solve(self, num_frames: int = 24) -> FlowResult:
        if not self.geometry.gates:
            raise ValueError("Geometry has no gates")

        eta = self._effective_viscosity()
        S = self._conductance_field(eta)

        dirichlet = np.zeros(self.geometry.shape, dtype=bool)
        for iy, ix in self.geometry.gates:
            dirichlet[iy, ix] = True

        A, b, idx = self._build_linear_system(S, dirichlet)
        tau_vec = spla.spsolve(A, b)

        tau = np.full(self.geometry.shape, np.nan, dtype=float)
        ny, nx = self.geometry.shape
        for iy in range(ny):
            for ix in range(nx):
                k = idx[iy, ix]
                if k >= 0:
                    tau[iy, ix] = tau_vec[k]

        tau_max = float(np.nanmax(tau))
        if tau_max <= 0:
            tau_max = 1.0

        # absolute time scaling
        V_cm3 = self.geometry.volume_cm3()
        if self.injection_volume_flow_cm3s is None:
            # fallback: assume 1.5 s baseline for any volume; user can override
            T_fill = 1.5
        else:
            Q = max(float(self.injection_volume_flow_cm3s), 1e-6)
            T_fill = V_cm3 / Q

        # if compression molding active, fill time partially shortened:
        if self.compression_molding:
            T_fill = T_fill * (
                self.compression_fraction / max(self.compression_factor, 1e-3)
                + (1.0 - self.compression_fraction)
            )

        fill_time_s = (tau / tau_max) * T_fill

        # pressure proxy: P ~ (tau_max - tau) / tau_max -> 1 at gate, 0 at far field
        pressure_norm = np.full_like(tau, np.nan)
        msk = ~np.isnan(tau)
        pressure_norm[msk] = 1.0 - tau[msk] / tau_max

        weld_score = self._compute_weld_score(tau)
        air_traps = self._compute_air_traps(tau)

        return FlowResult(
            tau=tau,
            fill_time_s=fill_time_s,
            pressure_norm=pressure_norm,
            weld_score=weld_score,
            air_traps=air_traps,
            total_fill_time_s=float(T_fill),
            viscosity_Pa_s=eta,
            geometry=self.geometry,
            metadata={
                "material": self.material.name,
                "melt_K": self.melt_temperature_K,
                "mold_K": self.mold_temperature_K,
                "injection_velocity_mms": self.injection_velocity_mms,
                "injection_Q_cm3s": self.injection_volume_flow_cm3s,
                "compression": self.compression_molding,
                "compression_factor": self.compression_factor,
                "compression_fraction": self.compression_fraction,
                "tau_max": tau_max,
                "volume_cm3": V_cm3,
                "num_frames": num_frames,
            },
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
