"""Hele-Shaw fill-time solver (Pseudo Conduction method).

Reference idea: solve the elliptic problem
    -div( S * grad(tau) ) = 1  in cavity Omega
    tau = 0                    at gates (Dirichlet)
    S * grad(tau) . n = 0      at cavity walls (Neumann, no flux)

The leading minus is the form actually assembled: the diagonal carries
+sum(coeff), the off-diagonals -coeff, and the right-hand side +1.
Written without it (continuous form: div(S grad tau) = -1) the signs of
the docstring and the matrix disagree, which they did until v0.24.0.

That sign convention makes the *unconstrained* operator symmetric and
positive semi-definite (face conductances are shared by both neighbours).
The assembled ``A`` is neither, because Dirichlet is applied to rows only:
a gate row is overwritten with the identity while the interior rows next
to it keep their -coeff in the gate column. Do not read "SPD" into this
docstring and reach for a symmetric-only solver -- the CG/AMG item on the
README roadmap needs the gate columns eliminated first. That elimination
is exact rather than an approximation, since tau = 0 at gates means the
term moved to the right-hand side is zero; it is simply not done yet,
because ``spsolve`` does not care.

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
  whole cavity inflates -- ``build_demo_geometry`` is the only
  builder that still leaves it unset.

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

from dataclasses import dataclass, replace

import numpy as np
import scipy.ndimage as ndi
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .geometry import Geometry
from .materials import Material, cross_wlf_viscosity, representative_shear_rate

#: Fill time assumed when no injection rate is given, for the cavity as drawn.
DEFAULT_FILL_TIME_S = 1.5

#: How many times the cavity may be re-solved after shedding cells that never
#: fill. The domain only ever shrinks, so this is a safety stop, not the
#: convergence criterion -- in practice one extra pass settles it.
# Safety valve for the domain loop in ``HeleShawSolver.solve``. The domain
# strictly loses at least one cell per pass, so the loop terminates on its
# own; this cap only guards against a pathological one-cell-per-pass trickle
# turning into thousands of full skin fixed points. When it trips, the
# leftover frozen cells are marked dead without another re-solve and
# ``metadata["domain_converged"]`` is False (Codex P2 on PR #60: exiting with
# them still live handed finite fill times to cells that do not fill, and let
# ``short_shot_cells`` exceed ``unfillable_cells``).
MAX_DOMAIN_PASSES = 64


@dataclass
class _DomainSolution:
    """The skin fixed point solved on one candidate cavity.

    A short shot has two cavities: the one that was drawn and the part of it
    the melt can reach. Which cells belong to the second depends on where the
    skins close, and where they close depends on the arrival times -- which
    are solved on a cavity. The two define each other, so each candidate gets
    its own complete solution and ``solve`` iterates on the domain itself.
    """

    tau: np.ndarray
    tau_max: float
    tau_max_flow: float | None
    T_fill: float
    T_fill_baseline: float
    tau_max_baseline: float
    # The volume-weighted representatives that actually fed the T_fill
    # inflation ratio, both taken over the same final still-flowing set.
    # None until the skin loop runs (or when nothing flows).
    tau_rep_flow: float | None
    tau_rep_baseline: float | None
    skin_thk_mm: np.ndarray | None
    h_core_mm: np.ndarray | None
    frozen_mask: np.ndarray | None
    iterations: int
    converged: bool


def check_gate_reachability(geometry: Geometry) -> None:
    """Reject cavity components that no gate can reach (Issue #58).

    A connected component of the mask with no Dirichlet point in it is a pure
    Neumann Laplacian block: singular, so ``spsolve`` fills it with garbage
    and never warns. Depending on how the disconnection arose, the garbage is
    either astronomically large (visible) or a plausible-looking uniform fill
    time across the severed region (invisible) -- the Profile gate rasteriser
    used to produce the second kind when the gate exit width fell below the
    mesh spacing.

    Rejection, not ``unfillable_mask``: a severed component is a geometry
    specification mistake, not a region the physics says the melt cannot
    reach. Folding it into the unfillable machinery would silently relabel
    "your input is wrong" as "the model predicts a short shot".

    4-connectivity, to match the 5-point stencil in
    ``HeleShawSolver._build_linear_system`` exactly: a diagonal-only "bridge"
    carries no flux in the discretisation, so it must not count as connected
    here either.
    """
    labels, _ = ndi.label(geometry.mask)  # default structure = 4-connectivity
    gate_labels = {int(labels[iy, ix]) for iy, ix in geometry.gates if geometry.mask[iy, ix]}
    gate_labels.discard(0)
    if not gate_labels:
        raise ValueError("no gate lies inside the cavity mask")
    orphaned = geometry.mask & ~np.isin(labels, list(gate_labels))
    if orphaned.any():
        n_cells = int(orphaned.sum())
        n_regions = int(np.unique(labels[orphaned]).size)
        raise ValueError(
            f"{n_cells} cavity cells in {n_regions} connected region(s) cannot be "
            "reached from any gate. The geometry is disconnected -- check the gate "
            "placement, or a feature narrower than the mesh spacing that rasterised "
            "into a closed wall."
        )


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
    # Cells the melt never reaches: the frozen ones plus whatever they cut off
    # from every gate. ``fill_time_s`` is NaN there, so no renderer can show
    # them arriving. None when the skin-layer model is off.
    unfillable_mask: np.ndarray | None = None


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
    # Absolute compression stroke [mm] added to each cell in the compression
    # mask (h_effective = h_actual + stroke). When ``None`` the legacy
    # multiplicative ``compression_factor`` model is used. The stroke model is
    # physically faithful for stepped plates where the mold shim is a fixed
    # absolute distance — both thin and thick zones grow by the same stroke,
    # so the step (e.g. 0.50 - 0.35 = 0.15 mm) is preserved across the
    # compression phase.
    compression_stroke_mm: float | None = None
    compression_fraction: float = 0.6  # fraction of fill time under compression-open state

    pressure_iters: int = 1  # placeholder for future iteration on viscosity

    # ----- skin-layer (Stefan/Neumann) model -----
    skin_layer_enabled: bool = False
    skin_growth_constant: float = 1.0  # c_skin in s(t) = c_skin * sqrt(alpha * t)
    # Fixed-point iterations for the tau <-> h_core coupling. The
    # constant-pressure proxy makes marginal freezing an avalanche by design
    # (skin narrows -> resistance up -> T_fill up -> more skin), and a cap in
    # the middle of the avalanche reports a plausible-looking half-frozen
    # state with only a metadata flag to show for it. 20 rides out every
    # avalanche seen so far; the strips in test_short_shot_timeline needed 9.
    skin_max_iterations: int = 20
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

        When compression molding is active, cells flagged in
        ``geometry.compression_mask`` are inflated; runners / sprues / gates
        keep their original thickness. ``compression_mask = None`` falls
        back to legacy whole-cavity inflation. The skin-layer model carves
        into this reference field.

        Two inflation modes are supported:

        - **Stroke mode** (``compression_stroke_mm`` is not None): each
          target cell grows by the same absolute stroke, ``h_eff = h + s``.
          Physically faithful for stepped plates (mold shim = fixed
          absolute distance, so the step thickness is preserved).
        - **Factor mode** (default, legacy): each target cell is scaled
          by ``compression_factor``, ``h_eff = h * f``. Thin cells inflate
          more than thick cells under the same factor — appropriate when
          the compression ratio (not stroke) is the design quantity.
        """
        h_mm = self.geometry.thickness_mm.copy()
        if self.compression_molding:
            cm = self.geometry.compression_mask
            target = self.geometry.mask if cm is None else (cm & self.geometry.mask)
            if self.compression_stroke_mm is not None:
                stroke = float(self.compression_stroke_mm)
                h_mm[target] = h_mm[target] + stroke
            else:
                factor = float(self.compression_factor)
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

    def _effective_flow_rate_cm3s(self) -> float:
        """Volume rate [cm3/s] the timeline is built on.

        With no rate given, the default is stated as a 1.5 s fill of the cavity
        as drawn; that is a rate, not a duration. Reading it as a duration
        would hand the same 1.5 s to a short shot with a tenth of the volume
        left, which is where the live-volume scaling would quietly stop
        working.
        """
        if self.injection_volume_flow_cm3s is not None:
            return max(float(self.injection_volume_flow_cm3s), 1e-6)
        return max(self.geometry.volume_cm3() / DEFAULT_FILL_TIME_S, 1e-9)

    def _baseline_fill_time(self, geom: Geometry) -> float:
        """Skin-free fill time [s] of ``geom`` at constant Q.

        Takes the geometry as an argument because a short shot has two of
        them: the cavity as drawn, and the part of it the melt can actually
        reach. The time to fill is the second one's -- melt does not spend
        time on volume it never occupies.

        The rate comes from ``self``. A restricted solver carries the rate
        pinned by ``_restricted_to``, so it can ask itself; deriving one from
        its own shrunken volume would cancel against the numerator and hand
        back the default 1.5 s no matter how little is left.
        """
        base = geom.volume_cm3() / self._effective_flow_rate_cm3s()
        if not self.compression_molding:
            return base
        # Effective inflation acting on the whole cavity (Q = const proxy).
        # ``effective_factor`` is the open-state cavity volume divided by
        # the as-cast volume. When only the product body inflates
        # (compression_mask set), the net resistance drop is proportional
        # to that body's contribution, so the compression-phase speed-up
        # is diluted accordingly.
        V_total_mm3 = geom.volume_cm3() * 1000.0
        if self.compression_stroke_mm is not None:
            # Stroke mode: ΔV = stroke * A_target (independent of local h).
            delta_V = float(self.compression_stroke_mm) * geom.compression_area_mm2()
            effective_factor = 1.0 + (delta_V / max(V_total_mm3, 1e-9))
        else:
            # Factor mode (legacy): same expression as before.
            f_comp = float(geom.compression_volume_fraction())
            f_comp = max(min(f_comp, 1.0), 0.0)
            effective_factor = 1.0 + (float(self.compression_factor) - 1.0) * f_comp
        effective_factor = max(effective_factor, 1e-3)
        return base * (
            self.compression_fraction / effective_factor + (1.0 - self.compression_fraction)
        )

    def _restricted_to(self, live: np.ndarray) -> HeleShawSolver:
        """A copy of this solver whose cavity is only the cells that fill.

        The injection rate is pinned to the value ``self`` is running at, so
        the copy is a solver for the same machine on a smaller cavity. Without
        that, an implicit rate would be re-derived from the shrunken volume
        and the restriction would quietly cancel itself out.
        """
        geom = self.geometry
        sub_geom = Geometry(
            mask=live.copy(),
            thickness_mm=geom.thickness_mm.copy(),
            cell_size_mm=geom.cell_size_mm,
            gates=[(iy, ix) for iy, ix in geom.gates if live[iy, ix]],
            label=geom.label,
            compression_mask=(
                None if geom.compression_mask is None else geom.compression_mask & live
            ),
        )
        return replace(
            self,
            geometry=sub_geom,
            injection_volume_flow_cm3s=self._effective_flow_rate_cm3s(),
        )

    def _unfillable_cells(self, frozen: np.ndarray) -> np.ndarray:
        """Cells the melt cannot fill: the frozen ones and everything they seal off.

        A cell whose core has closed is a wall, not a slow path. Anything left
        behind that wall never sees melt either, even though its own core is
        still open -- so connectivity to a gate, not local thickness, decides.
        """
        cavity = self.geometry.mask
        live = cavity & ~frozen
        labels, _ = ndi.label(live)
        gate_labels = {int(labels[iy, ix]) for iy, ix in self.geometry.gates if live[iy, ix]}
        if not gate_labels:
            # Every gate froze. Nothing fills; report that rather than dividing
            # by a tau_max taken over an empty set.
            return cavity.copy()
        reachable = np.isin(labels, sorted(gate_labels)) & live
        return cavity & ~reachable

    @staticmethod
    def _tau_reference(tau: np.ndarray, where: np.ndarray) -> float | None:
        """Largest tau among the cells that actually fill, or None if none do.

        The absolute time scale hangs off this: a single frozen cell carries a
        tau orders of magnitude above the rest, and letting it set the scale
        reports a short shot as "it just takes longer" -- squeezing the real
        fill into the bottom of every color bar and frame schedule.

        Returns None when the selection is empty or holds nothing but zeros
        (the gates, which are pinned at tau = 0). Falling back to the global
        maximum there would restore exactly the dead-cell tau this exists to
        keep out, so the caller has to treat "nothing flows" as its own case.
        """
        sel = where & ~np.isnan(tau)
        if not sel.any():
            return None
        value = float(np.nanmax(tau[sel]))
        return value if value > 0 else None

    @staticmethod
    def _arrival_time_field(
        tau: np.ndarray, where: np.ndarray, cell_volume: np.ndarray, T_fill: float
    ) -> np.ndarray:
        """Map tau to arrival times through the volume CDF (Issue #52).

        At constant volumetric rate the front has swept exactly ``Q * t`` of
        cavity by time ``t``, and cells fill in tau order -- so a cell arrives
        when the volume at or below its tau has been injected. The old linear
        map ``tau / tau_max * T_fill`` got even the healthy 1D strip wrong
        (mid-strip reported at 0.75 T instead of 0.5 T), and its denominator
        was a single cell: one pathologically slow cell rescaled every arrival
        time in the cavity. Here that cell moves only itself -- everyone else
        shifts by no more than its volume fraction.

        Ties share the arrival of the last cell in the group, so equal tau
        never orders itself by memory layout. Cells outside ``where`` (or with
        NaN tau) stay NaN.
        """
        t_arr = np.full_like(tau, np.nan)
        sel = where & ~np.isnan(tau)
        if not sel.any() or T_fill <= 0:
            return t_arr
        tau_v = tau[sel]
        vol_v = cell_volume[sel]
        order = np.argsort(tau_v, kind="stable")
        tau_sorted = tau_v[order]
        cum = np.cumsum(vol_v[order])
        total = float(cum[-1])
        if total <= 0:
            return t_arr
        last = np.searchsorted(tau_sorted, tau_sorted, side="right") - 1
        vals = np.empty_like(tau_v)
        vals[order] = (cum[last] / total) * T_fill
        t_arr[sel] = vals
        return t_arr

    @staticmethod
    def _tau_volume_mean(
        tau: np.ndarray, where: np.ndarray, cell_volume: np.ndarray
    ) -> float | None:
        """Volume-weighted mean tau over ``where``, or None if it is empty.

        The constant-pressure inflation proxy needs a resistance
        representative that one cell cannot own. The maximum was that one
        cell; the volume-weighted mean moves by at most a cell's volume
        fraction, and on a uniform plate (tau scaling by the same factor
        everywhere) it reproduces the max-ratio exactly.
        """
        sel = where & ~np.isnan(tau)
        if not sel.any():
            return None
        w = cell_volume[sel]
        total = float(np.sum(w))
        if total <= 0:
            return None
        value = float(np.sum(tau[sel] * w)) / total
        return value if value > 0 else None

    def _solve_domain(self, eta: float) -> _DomainSolution:
        """Solve tau, the skin fixed point and the fill time on *this* cavity.

        Everything here reads ``self.geometry.mask`` and ``self.geometry.gates``,
        so a restricted copy solves its own cavity rather than inheriting one.
        That matters: the skin thickness is driven by arrival times, and the
        arrival times of a region that has been sealed off are set by volume
        that never moves. Reusing them would leave the reported skin, core and
        fill time depending on material the melt never reaches.
        """
        cavity_mask = self.geometry.mask
        dirichlet = np.zeros(self.geometry.shape, dtype=bool)
        for iy, ix in self.geometry.gates:
            dirichlet[iy, ix] = True

        h_open = self._open_thickness_field()  # mm

        T_fill_baseline = self._baseline_fill_time(self.geometry)

        # baseline solve (no skin) — also serves as the tau_max reference
        S0 = self._conductance_field(eta, h_open)
        tau, tau_max = self._solve_tau_field(S0, dirichlet)
        tau_max_baseline = tau_max
        tau_baseline = tau.copy()
        dx = float(self.geometry.cell_size_mm)
        cell_volume = dx * dx * h_open  # mm^3, the volume each cell adds when swept
        T_fill = T_fill_baseline
        # tau of the slowest cell that still fills. Identical to ``tau_max``
        # until the skin model closes a core somewhere.
        tau_max_flow: float | None = tau_max

        skin_thk_mm: np.ndarray | None = None
        h_core_mm: np.ndarray | None = None
        frozen_mask: np.ndarray | None = None
        tau_rep_flow: float | None = None
        tau_rep_baseline: float | None = None
        skin_iters_done = 0
        skin_converged = False

        if self.skin_layer_enabled:
            alpha = max(float(self.material.thermal_diffusivity_m2_s), 0.0)
            c_skin = max(float(self.skin_growth_constant), 0.0)
            min_core = max(float(self.min_core_thickness_mm), 1e-6)
            tol = max(float(self.skin_convergence_tol), 0.0)

            skin_thk_mm = np.zeros_like(h_open)
            h_core_mm = np.where(cavity_mask, h_open, 0.0)

            for it in range(int(max(self.skin_max_iterations, 1))):
                msk = cavity_mask & ~np.isnan(tau)
                # arrival time per cell: volume CDF against the current best
                # estimate of T_fill. Frozen cells ride at the top of the CDF,
                # so their arrival saturates at T_fill instead of exporting
                # their stand-in tau into everyone else's time scale.
                t_arr = self._arrival_time_field(tau, msk, cell_volume, T_fill)
                t_arr = np.where(np.isnan(t_arr), 0.0, t_arr)

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

                # Cells whose two skins have met. Their core is pinned at the
                # numerical floor, so their tau is a stand-in for "closed" and
                # must not drive the time scale.
                frozen_new = (
                    cavity_mask
                    & ((h_open - 2.0 * s_mm_new) <= min_core + 1e-9)
                    & (h_open > min_core + 1e-9)
                )
                tau_max_flow_new = self._tau_reference(tau_new, cavity_mask & ~frozen_new)

                # constant-pressure proxy: T_fill grows with the resistance.
                # Both the numerator and the denominator are volume-weighted
                # means over the same still-flowing set: when a cell freezes it
                # leaves both sides at once, instead of dropping out of one and
                # avalanching the ratio (Issue #52 -- the old form divided a
                # frozen-free max by an everything max, so one pathological
                # cell freezing moved the reported time by tens of percent).
                # With nothing flowing there is no resistance to speak of, so
                # the baseline stands rather than an inflation off a dead cell.
                flow_new = cavity_mask & ~frozen_new
                tau_rep_new = self._tau_volume_mean(tau_new, flow_new, cell_volume)
                tau_rep_base = self._tau_volume_mean(tau_baseline, flow_new, cell_volume)
                if tau_rep_new is None or tau_rep_base is None:
                    T_fill_new = T_fill_baseline
                else:
                    T_fill_new = T_fill_baseline * (tau_rep_new / tau_rep_base)
                tau_rep_flow = tau_rep_new
                tau_rep_baseline = tau_rep_base

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
                tau_max_flow = tau_max_flow_new
                T_fill = T_fill_new
                skin_thk_mm = s_mm_new
                h_core_mm = h_core_new
                frozen_mask = frozen_new
                skin_iters_done = it + 1
                if tau_max_flow is None:
                    # Nothing outside the gates is still open. Another pass
                    # would divide by this None, and there is nothing left for
                    # it to change: the arrival times it would compute are the
                    # input to a field that no longer flows.
                    break
                if rel < tol:
                    skin_converged = True
                    break

        return _DomainSolution(
            tau=tau,
            tau_max=tau_max,
            tau_max_flow=tau_max_flow,
            T_fill=T_fill,
            T_fill_baseline=T_fill_baseline,
            tau_max_baseline=tau_max_baseline,
            skin_thk_mm=skin_thk_mm,
            h_core_mm=h_core_mm,
            frozen_mask=frozen_mask,
            tau_rep_flow=tau_rep_flow,
            tau_rep_baseline=tau_rep_baseline,
            iterations=skin_iters_done,
            converged=skin_converged,
        )

    def solve(self, num_frames: int = 24) -> FlowResult:
        if not self.geometry.gates:
            raise ValueError("Geometry has no gates")
        check_gate_reachability(self.geometry)

        eta = self._effective_viscosity()
        cavity_mask = self.geometry.mask
        V_cm3 = self.geometry.volume_cm3()

        sol = self._solve_domain(eta)
        # The cavity-wide tau_max, frozen cells and all. Kept as evidence: the
        # gap between this and tau_max_flow is the error that would land in the
        # reported time if the dead cells were left in.
        tau_max_cavity = sol.tau_max

        # Shed the cells that never fill and solve again on what is left.
        # Removing them changes the arrival times of the cells that remain,
        # which changes where the skins close -- so the domain and the skin
        # field have to be settled together, not once each. The domain only
        # ever loses cells, so this terminates; ``MAX_DOMAIN_PASSES`` is a stop
        # for pathological cases, not the convergence test.
        domain_solver = self
        skin_thk_mm = sol.skin_thk_mm
        h_core_mm = sol.h_core_mm
        short_shot_mask = None if sol.frozen_mask is None else sol.frozen_mask.copy()
        no_flow_forced = False
        domain_passes = 0

        for _ in range(MAX_DOMAIN_PASSES):
            frozen = sol.frozen_mask
            if frozen is None or not frozen.any():
                break
            dead = domain_solver._unfillable_cells(frozen)
            live = domain_solver.geometry.mask & ~dead
            sub = domain_solver._restricted_to(live) if live.any() else None
            if sub is None or not sub.geometry.gates:
                # Every gate is behind a closed core. There is no live cavity
                # to solve, so the last solution stands with its time pinned to
                # the geometric baseline rather than inflated off dead cells.
                no_flow_forced = True
                break
            domain_solver = sub
            sol = sub._solve_domain(eta)
            domain_passes += 1
            # The sub-solution only writes the cells it owns; the rest keep the
            # values from the pass that last had them live.
            if sol.skin_thk_mm is not None and skin_thk_mm is not None:
                skin_thk_mm = np.where(live, sol.skin_thk_mm, skin_thk_mm)
            if sol.h_core_mm is not None and h_core_mm is not None:
                h_core_mm = np.where(live, sol.h_core_mm, h_core_mm)
            if sol.frozen_mask is not None and short_shot_mask is not None:
                short_shot_mask = short_shot_mask | sol.frozen_mask

        # Every gate closed: nothing at all fills, so the live cavity is empty
        # rather than "whatever the last pass was still solving".
        live_mask = np.zeros_like(cavity_mask) if no_flow_forced else domain_solver.geometry.mask
        # Safety valve tripped: the final pass froze cells the loop never got
        # to process. They do not fill, so they must not stay live -- mark
        # them (and whatever they seal off) dead without another re-solve.
        # The final solution already excluded them from its own time scale
        # (``tau_max_flow`` and the inflation set both drop frozen cells), so
        # the reported T_fill stands; only the cell bookkeeping needed fixing.
        domain_converged = True
        leftover = sol.frozen_mask
        if not no_flow_forced and leftover is not None and leftover.any():
            domain_converged = False
            live_mask = live_mask & ~domain_solver._unfillable_cells(leftover)
        dead_cells = cavity_mask & ~live_mask
        unfillable_mask = dead_cells if dead_cells.any() else None

        tau = sol.tau
        tau_max = sol.tau_max
        tau_max_flow = None if no_flow_forced else sol.tau_max_flow
        T_fill_baseline = sol.T_fill_baseline
        tau_max_baseline = sol.tau_max_baseline
        # The domain solution already carries its own constant-pressure
        # inflation (volume-weighted, over its own live set). Recomputing it
        # here from single-cell maxima would reintroduce the one-cell
        # normalization this branch exists to remove.
        T_fill = T_fill_baseline if no_flow_forced else sol.T_fill

        # Nothing beyond the gates flows. Every time here is zero, so the
        # divisor only has to be positive -- what matters is that the reported
        # T_fill stays the geometric baseline instead of an inflation read off
        # material the melt never reaches.
        no_flow = tau_max_flow is None
        tau_scale = 1.0 if no_flow else float(tau_max_flow)

        # absolute time scaling per cell. Cells that never fill stay NaN: a
        # time would say they arrive eventually, which is the opposite of what
        # a short shot means.
        fillable = ~np.isnan(tau)
        if unfillable_mask is not None:
            fillable = fillable & ~unfillable_mask
        # Same volume-CDF map as the fixed point: at constant rate the front
        # sweeps volume linearly in time, so a cell's time is the volume at or
        # below its tau. Cells that never fill stay NaN.
        h_open_final = domain_solver._open_thickness_field()
        cell_volume_final = (float(self.geometry.cell_size_mm) ** 2) * h_open_final
        fill_time_s = self._arrival_time_field(tau, fillable, cell_volume_final, T_fill)

        # pressure proxy: P ~ (tau_max - tau) / tau_max -> 1 at gate, 0 at last fill
        pressure_norm = np.full_like(tau, np.nan)
        pressure_norm[fillable] = 1.0 - tau[fillable] / tau_scale

        # Weld lines and air traps read tau as "when did melt get here". A cell
        # that never receives melt has no such time -- and its floored core
        # leaves a tau far above every live cell, which is exactly the shape a
        # local-maximum search calls an air trap. Blank them out first so the
        # diagnostics cannot plant defects in material that stays empty.
        tau_flow = tau
        if unfillable_mask is not None:
            tau_flow = np.where(unfillable_mask, np.nan, tau)
        weld_score = self._compute_weld_score(tau_flow)
        air_traps = self._compute_air_traps(tau_flow)

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
            "skin_layer_enabled": self.skin_layer_enabled,
        }
        if self.skin_layer_enabled:
            cells_total = max(int(cavity_mask.sum()), 1)
            short_count = int(short_shot_mask.sum()) if short_shot_mask is not None else 0
            metadata.update(
                {
                    "skin_growth_constant": self.skin_growth_constant,
                    "thermal_diffusivity_m2_s": self.material.thermal_diffusivity_m2_s,
                    "skin_iterations": sol.iterations,
                    "skin_converged": sol.converged,
                    "min_core_thickness_mm": self.min_core_thickness_mm,
                    "T_fill_baseline_s": T_fill_baseline,
                    "T_fill_inflation": (T_fill / T_fill_baseline if T_fill_baseline > 0 else 1.0),
                    # the volume-weighted representatives that fed the ratio,
                    # both over the same final still-flowing set (Issue #52)
                    "tau_rep_flow": sol.tau_rep_flow,
                    "tau_rep_baseline": sol.tau_rep_baseline,
                    "short_shot_cells": short_count,
                    "short_shot_fraction": short_count / cells_total,
                    "unfillable_cells": (
                        int(unfillable_mask.sum()) if unfillable_mask is not None else 0
                    ),
                    "sealed_off_cells": (
                        int(unfillable_mask.sum()) - short_count
                        if unfillable_mask is not None
                        else 0
                    ),
                    "tau_max_flow": tau_max_flow,
                    # tau of the slowest cell in the cavity-wide solve, frozen
                    # cells included. Kept because the gap between this and
                    # tau_max_flow is the size of the error that would land in
                    # the reported time if the dead cells were left in.
                    "tau_max_cavity": tau_max_cavity,
                    "no_flow": no_flow,
                    # How many times the cavity had to be re-solved after
                    # shedding cells. 0 means nothing froze.
                    "domain_passes": domain_passes,
                    "domain_converged": domain_converged,
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
            unfillable_mask=unfillable_mask,
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
