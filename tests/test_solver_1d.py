"""Analytical-solution verification: 1D strip, full-edge gate.

Setup
-----
Uniform strip with width Ny cells, length Nx cells, uniform thickness h,
uniform effective viscosity. Dirichlet τ=0 imposed on the entire left edge
(x=0), no-flux on the other three edges. Reduces to a strict 1D problem
in x.

The discrete operator in :func:`HeleShawSolver._build_linear_system`
assembles ``-∇·(S∇τ) = 1`` (positive diagonal, negative off-diagonals),
i.e. the continuous equation is ``∇·(S∇τ) = -1``. With uniform S this becomes::

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
import pytest
import scipy.sparse.linalg as spla

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


# --- Matrix structure: what the docstrings are allowed to claim -------------
#
# ``core/solver.py`` states that the assembled ``A`` is *not* symmetric,
# because Dirichlet is applied to rows only, and that eliminating the gate
# columns would be exact rather than an approximation. Both halves are
# claims about the code, so both are tested here. They also survive the fix
# they describe: if someone symmetrises the assembly, "all asymmetry sits in
# gate columns" becomes vacuously true and the elimination stays a no-op.


def _assembled_system(ny: int = 6, nx: int = 20):
    """Assemble A for a notched strip, and report which matrix rows are gates.

    The mask is punched rather than full on purpose. With every cell inside
    the cavity the compressed matrix index equals the raw grid index, so a
    test that confuses the two still passes -- and that confusion is easy to
    make, since ``Geometry.gates`` is in grid coordinates while ``A`` is not.
    Removing cells ahead of the gate column shifts the numbering so the two
    genuinely disagree.
    """
    g = _build_uniform_strip(ny, nx, cell_mm=1.0, thk_mm=2.0)
    notch = g.mask.copy()
    notch[0, nx // 2 :] = False  # a row that stops short of the far end
    notch[ny - 1, nx - 3 :] = False
    g = Geometry(mask=notch, thickness_mm=g.thickness_mm, cell_size_mm=g.cell_size_mm)
    g.gates = [(iy, 0) for iy in range(ny)]
    solver = HeleShawSolver(
        geometry=g,
        material=MaterialDB()["PP"],
        melt_temperature_K=503.15,
        mold_temperature_K=313.15,
        injection_velocity_mms=100.0,
        injection_volume_flow_cm3s=20.0,
    )
    S = solver._conductance_field(solver._effective_viscosity(), solver._open_thickness_field())
    dirichlet = np.zeros(g.shape, dtype=bool)
    for iy, ix in g.gates:
        dirichlet[iy, ix] = True
    A, b, _ = solver._build_linear_system(S, dirichlet)

    # ``_build_linear_system`` numbers only the masked cells, so a grid index
    # is not a matrix index. Conflating the two silently makes the assertions
    # below inspect unrelated rows.
    flat = np.where(g.mask.ravel())[0]
    gate_rows = {int(np.flatnonzero(flat == iy * g.shape[1] + ix)[0]) for iy, ix in g.gates}
    return A.tocsr(), b, gate_rows


def test_asymmetry_is_confined_to_the_gate_columns() -> None:
    """Every A[i,j] != A[j,i] must involve a gate row.

    The face conductance is a harmonic mean, which is symmetric, so the only
    thing that can break symmetry is the row-only Dirichlet treatment. If an
    asymmetric entry ever shows up away from a gate, the cause is something
    else and the docstring's explanation is wrong.
    """
    A, _b, gate_rows = _assembled_system()
    asym = abs(A - A.T).tocoo()
    offenders = [(int(r), int(c)) for r, c, v in zip(asym.row, asym.col, asym.data) if v > 1e-30]
    stray = [(r, c) for r, c in offenders if r not in gate_rows and c not in gate_rows]
    assert not stray, f"asymmetry away from any gate row: {stray[:5]}"


def test_the_system_without_the_pinned_unknowns_is_spd() -> None:
    """Dropping the gate rows *and* columns leaves a symmetric positive-definite block.

    This is the claim the customer-facing Q&A makes
    (the customer-facing technical Q&A in the private docs repo): existence and uniqueness are
    guaranteed. The guarantee is real, but it belongs to the reduced system, not
    to ``A`` as assembled -- so it is asserted here rather than left as prose. A
    document that promises a mathematical property to a customer should not be
    the only place that property is recorded.
    """
    A, _b, gate_rows = _assembled_system()
    dense = A.toarray()
    keep = np.array([k for k in range(dense.shape[0]) if k not in gate_rows])
    interior = dense[np.ix_(keep, keep)]

    assert np.array_equal(interior, interior.T), "reduced block is not symmetric"
    # eigvalsh needs symmetry, which the line above has just established.
    assert float(np.linalg.eigvalsh(interior).min()) > 0.0, "reduced block is not positive definite"


def _gateless_island_solver() -> HeleShawSolver:
    """A strip whose far half is severed from the gate edge."""
    ny, nx = 6, 20
    mask = np.ones((ny, nx), dtype=bool)
    mask[:, 9:11] = False
    g = Geometry(
        mask=mask,
        thickness_mm=np.full((ny, nx), 2.0, dtype=float),
        cell_size_mm=1.0,
    )
    g.gates = [(iy, 0) for iy in range(ny)]
    return HeleShawSolver(
        geometry=g,
        material=MaterialDB()["PP"],
        melt_temperature_K=503.15,
        mold_temperature_K=313.15,
        injection_velocity_mms=100.0,
        injection_volume_flow_cm3s=20.0,
    )


def test_a_component_with_no_gate_has_no_unique_solution() -> None:
    """The SPD guarantee holds per connected component that reaches a gate.

    A region cut off from every gate has no pinned value, so its block is a
    pure Neumann Laplacian with a zero eigenvalue. The customer-facing Q&A
    states this precondition, so the boundary is asserted, not just described.

    This is a statement about the *mathematics*, which does not change however
    Issue #58 is resolved: a gate-less block is singular whether the solver
    goes on to reject it, exclude it, or (as today) solve it anyway. What the
    solver should *do* about it is the separate test below.
    """
    solver = _gateless_island_solver()
    g = solver.geometry
    S = solver._conductance_field(solver._effective_viscosity(), solver._open_thickness_field())
    dirichlet = np.zeros(g.shape, dtype=bool)
    for iy, ix in g.gates:
        dirichlet[iy, ix] = True
    A, _b, _ = solver._build_linear_system(S, dirichlet)

    dense = A.toarray()
    flat = np.where(g.mask.ravel())[0]
    nx = g.shape[1]
    gate_rows = {int(np.flatnonzero(flat == iy * nx + ix)[0]) for iy, ix in g.gates}
    keep = np.array([k for k in range(dense.shape[0]) if k not in gate_rows])
    interior = dense[np.ix_(keep, keep)]

    ev = np.linalg.eigvalsh((interior + interior.T) / 2)
    scale = float(abs(ev).max())
    assert scale > 0

    # Two assertions, because "has a zero mode" is not the same claim as "is
    # small". A signed ``ev.min() / scale < tol`` passes trivially for anything
    # negative definite, which has no zero mode at all -- it would keep this
    # test green while the thing it names stopped being true.
    assert float(ev.min()) / scale > -1e-12, "block is not positive semi-definite"
    assert float(abs(ev).min()) / scale < 1e-12, (
        "a gate-less component still looks positive definite; "
        "the documented precondition would be unnecessary"
    )


def test_solve_rejects_a_gateless_region() -> None:
    """Issue #58, fixed: a severed component is rejected at the entrance.

    This test previously carried ``xfail(strict=True)`` while the solver
    still ran ``spsolve`` over the singular block and returned garbage; the
    fix made it XPASS, which is what prompted this rewrite.

    Rejection (not ``unfillable_mask``) is asserted deliberately: a
    component no gate can reach is a geometry specification mistake, and
    routing it through the unfillable machinery would relabel "your input
    is wrong" as "the model predicts a short shot". The match string pins
    the reachability message, so the generic "Geometry has no gates" guard
    cannot satisfy this test by accident.
    """
    solver = _gateless_island_solver()
    with pytest.raises(ValueError, match="cannot be .*reached from any gate"):
        solver.solve(num_frames=2)


def test_solve_rejects_gates_that_all_sit_outside_the_mask() -> None:
    """Gates exist but none lands on a cavity cell: reject, do not solve.

    Distinct from the empty-gates guard (which this geometry passes) and
    from the severed-component case (every cavity cell is orphaned here,
    not just an island). Without the check, the whole cavity is one pure
    Neumann block and ``spsolve`` returns garbage for all of it.
    """
    solver = _gateless_island_solver()
    solver.geometry.gates = [(0, 9)]  # inside the punched hole -> mask False
    with pytest.raises(ValueError, match="no gate lies inside"):
        solver.solve(num_frames=2)


def test_eliminating_the_gate_columns_does_not_move_the_solution() -> None:
    """Zeroing the gate columns is exact, not an approximation.

    ``tau`` is pinned to 0 at the gates, so the term those columns contribute
    to a neighbour's equation is identically zero and the right-hand side does
    not need a correction. This is what makes the CG/AMG roadmap item cheap:
    the symmetric system is the same system, not a modified one.
    """
    A, b, gate_rows = _assembled_system()
    tau_raw = spla.spsolve(A.tocsc(), b)

    A_sym = A.tolil()
    for k in gate_rows:
        A_sym[:, k] = 0.0
        A_sym[k, k] = 1.0
    A_sym = A_sym.tocsr()

    asym = abs(A_sym - A_sym.T)
    assert asym.nnz == 0 or float(asym.max()) < 1e-30, "elimination left A asymmetric"

    tau_sym = spla.spsolve(A_sym.tocsc(), b)
    scale = float(np.max(np.abs(tau_raw)))
    assert scale > 0
    assert float(np.max(np.abs(tau_sym - tau_raw))) / scale < 1e-10


def test_a_diagonal_touch_does_not_count_as_reachable() -> None:
    """The reachability check must use 4-connectivity, like the stencil.

    Two regions that touch only at a corner exchange no flux in the 5-point
    discretisation, so the far region is still a singular Neumann block. An
    8-connected labelling would call this geometry connected and wave the
    garbage through -- which is why the connectivity choice is asserted
    here rather than trusted to a comment.
    """
    mask = np.zeros((4, 4), dtype=bool)
    mask[:2, :2] = True  # gate-side block
    mask[2:, 2:] = True  # far block, touching only at the (1,1)/(2,2) corner
    g = Geometry(
        mask=mask,
        thickness_mm=np.where(mask, 2.0, 0.0),
        cell_size_mm=1.0,
    )
    g.gates = [(0, 0)]
    solver = HeleShawSolver(
        geometry=g,
        material=MaterialDB()["PP"],
        melt_temperature_K=503.15,
        mold_temperature_K=313.15,
        injection_velocity_mms=100.0,
        injection_volume_flow_cm3s=20.0,
    )
    with pytest.raises(ValueError, match="cannot be .*reached from any gate"):
        solver.solve(num_frames=2)


def test_vectorised_assembly_matches_a_cell_by_cell_reference():
    """The face-vectorised assembly (v0.33.0) builds the same matrix as the loop it replaced.

    A masked 2D grid with a hole, an isolated cell, two gates and a varying
    conductance exercises every branch: harmonic-mean faces, no-flux walls
    at the mask edge, the Dirichlet identity row with its neighbours keeping
    the gate column, and the unit-diagonal guard on a cell with no faces.
    """
    rng = np.random.default_rng(7)
    ny, nx = 6, 7
    mask = np.ones((ny, nx), dtype=bool)
    mask[2:4, 3] = False  # a hole
    mask[0, 6] = True
    mask[0, 5] = False
    mask[1, 6] = False  # (0, 6) has no masked neighbour: isolated
    thickness = np.ones((ny, nx))
    geom = Geometry(mask=mask, thickness_mm=thickness, cell_size_mm=0.5)
    geom.gates = [(0, 0), (5, 6)]
    solver = HeleShawSolver(geom, MaterialDB()["PP"])
    S = rng.uniform(0.5, 2.0, size=(ny, nx))
    S[4, 1] = 0.0  # a closed cell: its faces conduct nothing
    dirichlet = np.zeros((ny, nx), dtype=bool)
    for iy, ix in geom.gates:
        dirichlet[iy, ix] = True

    A, b, idx = solver._build_linear_system(S, dirichlet)

    # cell-by-cell reference, written the way the original loop was
    dx = geom.cell_size_mm * 1e-3
    cells = [(iy, ix) for iy in range(ny) for ix in range(nx) if mask[iy, ix]]
    ref = np.zeros((len(cells), len(cells)))
    ref_b = np.zeros(len(cells))
    for k, (iy, ix) in enumerate(cells):
        assert idx[iy, ix] == k
        if dirichlet[iy, ix]:
            ref[k, k] = 1.0
            continue
        diag = 0.0
        for dy, dx_ in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            jy, jx = iy + dy, ix + dx_
            if 0 <= jy < ny and 0 <= jx < nx and mask[jy, jx]:
                total = S[iy, ix] + S[jy, jx]
                face = 2.0 * S[iy, ix] * S[jy, jx] / total if total > 0 else 0.0
                coeff = face / (dx * dx)
                diag += coeff
                ref[k, idx[jy, jx]] -= coeff
        ref[k, k] = diag if diag > 0 else 1.0
        ref_b[k] = 1.0

    np.testing.assert_allclose(A.toarray(), ref, rtol=1e-12, atol=0.0)
    np.testing.assert_array_equal(b, ref_b)
    assert idx[~mask].max() == -1
