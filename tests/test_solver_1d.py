"""Analytical-solution verification: 1D strip, full-edge gate.

Setup
-----
Uniform strip with width Ny cells, length Nx cells, uniform thickness h,
uniform effective viscosity. Dirichlet τ=0 imposed on the entire left edge
(x=0), no-flux on the other three edges. Reduces to a strict 1D problem
in x.

The discrete operator in :func:`HeleShawSolver._build_linear_system`
assembles ``-∇·(S∇τ) = 1`` (positive-diagonal SPD form), i.e. the
continuous equation is ``∇·(S∇τ) = -1``. With uniform S this becomes::

    S d²τ/dx² = -1
    τ(0) = 0,   dτ/dx(L) = 0

whose closed form is::

    τ(x)      = x(2L - x) / (2S)
    τ_max     = L² / (2S)              at x = L
    τ̂(x̂)    = τ(x)/τ_max
              = 1 - (1 - x̂)²            (x̂ = x/L)

We verify the **shape** of the normalized profile, not τ_max itself
(absolute scaling depends on η_eff, which is computed internally and
not exposed for cell-by-cell comparison here).
"""

from __future__ import annotations

import numpy as np

from core import HeleShawSolver, MaterialDB
from core.geometry import Geometry


def _build_uniform_strip(
    ny: int,
    nx: int,
    cell_mm: float,
    thk_mm: float,
) -> Geometry:
    mask = np.ones((ny, nx), dtype=bool)
    thk = np.full((ny, nx), thk_mm, dtype=float)
    g = Geometry(mask=mask, thickness_mm=thk, cell_size_mm=cell_mm)
    # Full left-edge gate ⇒ flow is strictly 1D in x.
    g.gates = [(iy, 0) for iy in range(ny)]
    return g


def _solve_strip(
    ny: int = 8,
    nx: int = 200,
    cell_mm: float = 1.0,
    thk_mm: float = 2.0,
):
    g = _build_uniform_strip(ny, nx, cell_mm, thk_mm)
    solver = HeleShawSolver(
        geometry=g,
        material=MaterialDB()["PP"],
        melt_temperature_K=503.15,
        mold_temperature_K=313.15,
        injection_velocity_mms=100.0,
        injection_volume_flow_cm3s=20.0,
    )
    return solver.solve(num_frames=4)


def test_strip_tau_is_monotone_increasing_in_x() -> None:
    """τ must (weakly) increase from the gate toward the far end."""
    result = _solve_strip()
    iy = result.geometry.ny // 2
    tau_row = result.tau[iy, :]
    diffs = np.diff(tau_row)
    assert np.all(diffs > -1e-9), "τ is not monotone increasing along x"


def test_strip_tau_max_at_far_end() -> None:
    """The last cell must be the global τ-maximum on the strip centerline."""
    result = _solve_strip()
    iy = result.geometry.ny // 2
    tau_row = result.tau[iy, :]
    assert int(np.argmax(tau_row)) == len(tau_row) - 1


def test_strip_no_y_dependence() -> None:
    """Full-edge gate: τ must be independent of y (within numerical noise)."""
    result = _solve_strip()
    tau = result.tau
    # std across rows for each column ≈ 0
    col_std = np.nanstd(tau, axis=0)
    col_mean = np.nanmean(tau, axis=0)
    rel_std = col_std / np.where(col_mean > 1e-12, col_mean, 1.0)
    # Skip x=0 where τ=0 (degenerate) and look at the rest.
    assert float(np.nanmax(rel_std[1:])) < 1e-6, (
        f"unexpected y-variation on a 1D strip: max rel std = {float(np.nanmax(rel_std[1:])):.2e}"
    )


def test_strip_normalized_profile_matches_analytical_within_2pct() -> None:
    """Compare τ̂(x̂) = τ/τ_max against analytical 1 − (1 − x̂)².

    Tolerance: max |Δτ̂| < 2 %, mean |Δτ̂| < 0.5 %.
    These bounds reflect the cell-centered FD discretization error and
    the fact that the gate is a Dirichlet point at the leftmost cell
    rather than at x=0 exactly.
    """
    ny, nx = 8, 200
    result = _solve_strip(ny=ny, nx=nx)

    iy = ny // 2
    tau_row = result.tau[iy, :]
    tau_norm = tau_row / np.nanmax(tau_row)

    # Treat the gate (i=0) as x=0 and the last cell (i=nx-1) as x=L.
    i = np.arange(nx)
    x_norm = i / (nx - 1)
    tau_exact_norm = 1.0 - (1.0 - x_norm) ** 2

    err = np.abs(tau_norm - tau_exact_norm)
    max_err = float(np.nanmax(err))
    mean_err = float(np.nanmean(err))

    assert max_err < 0.02, f"max |Δτ̂| = {max_err:.4f} (exceeds 2 %)"
    assert mean_err < 0.005, f"mean |Δτ̂| = {mean_err:.4f} (exceeds 0.5 %)"


def test_strip_mesh_refinement_reduces_error() -> None:
    """Doubling Nx must (roughly) reduce the discretization error.

    Not a strict O(h²) check (boundary handling complicates that), but
    a sanity guard against accidental order-of-accuracy regressions.
    """
    nx_coarse, nx_fine = 50, 200

    err_coarse = _strip_l2_error(nx_coarse)
    err_fine = _strip_l2_error(nx_fine)

    assert err_fine < err_coarse, (
        f"refinement did not improve accuracy: coarse={err_coarse:.4e}, fine={err_fine:.4e}"
    )


def _strip_l2_error(nx: int) -> float:
    result = _solve_strip(ny=4, nx=nx)
    iy = result.geometry.ny // 2
    tau_row = result.tau[iy, :]
    tau_norm = tau_row / np.nanmax(tau_row)
    i = np.arange(nx)
    x_norm = i / (nx - 1)
    tau_exact_norm = 1.0 - (1.0 - x_norm) ** 2
    return float(np.sqrt(np.nanmean((tau_norm - tau_exact_norm) ** 2)))
