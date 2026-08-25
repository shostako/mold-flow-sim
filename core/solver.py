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
becomes ``S = h_core^3 / (12 * eta)``.

The clock is the *exposure* clock (Issue #61). A cell's wall starts
ageing when the front passes it at ``t_arr`` and keeps ageing while
melt flows through it, so at time ``t`` its skin is ``s(t - t_arr)``
-- thickest at the gate, zero at the front. Two quantities follow:

- the skins meet at the age ``t_c = ((h - h_min) / (2 c_skin))^2 / alpha``;
  a cell whose service ``T_fill - t_arr`` reaches ``t_c`` *seals* at
  ``t_close = t_arr + t_c`` (it filled, then closed: ``short_shot_mask``);
- the single conductance a one-shot elliptic solve can carry is the
  time-mean skin over the cell's service ``[t_arr, min(T_fill, t_close)]``,
  which for the square-root law is ``(2/3) s(a)`` with ``a`` the service
  duration. That mean never exceeds ``(2/3)(h - h_min)/2``, so
  ``h_core >= h/3`` and a sealing cell raises resistance by at most 27x
  instead of collapsing to the numerical floor.

A cell that the front reaches only after every path to a gate has
sealed does not fill (``unfillable_mask``, NaN fill time). When the
cavity-wide solve cuts cells off, the part that fills is found as the
largest prefix of the fill order that does not cut itself off, by
bisection on filled volume with every candidate solved on its own
(``_largest_consistent_prefix``): a solve that carries the dead region
runs on a clock inflated by material that never fills, and a solve of
only what it left runs on one with too little resistance to have sealed
anything -- neither is the part's.

Because ``s`` depends on the arrival time and the arrival time depends
on ``S``, the fields are solved by fixed-point iteration, and the
absolute fill time ``T_fill`` is scaled up by the relative growth of the
volume-weighted mean tau (constant-pressure proxy: the inflated runtime
mirrors the resistance increase). Bulk-melt cooling and dynamic
viscosity coupling remain disabled — the model captures the wall-side
freezing front in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import scipy.ndimage as ndi
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .geometry import Geometry
from .materials import Material, cross_wlf_viscosity, representative_shear_rate

# Weld-line detector thresholds (opening angle between two converging flow
# directions, degrees). Below MIN nothing is drawn (numerical jitter of nearly
# parallel streams; 0 draws every confluence the crest test admits); at FULL
# the score saturates. 45 deg opening = the
# conventional 135 deg "meeting angle" boundary between a weld (drawn solid)
# and a meld (drawn faint). MIN is the UI's slider default; the angle field
# itself is kept on the result so the threshold can move without re-solving.
WELD_MIN_ANGLE_DEG = 0.0
WELD_FULL_ANGLE_DEG = 45.0
# A confluence cell must be later than both neighbours along the pair axis by
# comparable amounts (smaller drop >= this fraction of the larger). Filters
# the grazing ties of a flow turning around an obstacle corner.
CREST_BALANCE = 0.1
# Relative tolerance under which two neighbouring tau values count as the
# same crest cell (symmetric geometries tie the two centre columns exactly).
CREST_TIE_RTOL = 1e-9


def weld_score_from_angle(
    angle_deg: np.ndarray,
    *,
    min_angle_deg: float = WELD_MIN_ANGLE_DEG,
    full_angle_deg: float = WELD_FULL_ANGLE_DEG,
) -> np.ndarray:
    """Map a meeting-angle field to a [0, 1] score (NaN -> 0).

    Linear from ``min_angle_deg`` (0, nothing drawn) to ``full_angle_deg``
    (1, saturated); angles above ``full_angle_deg`` stay at 1. Kept separate
    from the angle computation so the UI can move the threshold on a solved
    result without solving again.
    """
    if not 0.0 <= min_angle_deg < full_angle_deg <= 180.0:
        raise ValueError(
            "weld angles must satisfy 0 <= min_angle_deg < full_angle_deg <= 180, "
            f"got {min_angle_deg}, {full_angle_deg}"
        )
    score = (angle_deg - min_angle_deg) / (full_angle_deg - min_angle_deg)
    score = np.where(np.isnan(score), 0.0, score)
    return np.clip(score, 0.0, 1.0)


#: Fill time assumed when no injection rate is given, for the cavity as drawn.
DEFAULT_FILL_TIME_S = 1.5

#: How many candidate cavities the bisection in
#: ``HeleShawSolver._largest_consistent_prefix`` may solve when the
#: cavity-wide solve cuts cells off. The interval halves per pass, so
#: ``log2(DOMAIN_VOLUME_RESOLUTION)`` passes settle it -- more only while no
#: consistent candidate has been found yet (a gate ringed by thin cells fills
#: a sliver far below the resolution), down to one thin cell. When the cap
#: trips, the best consistent candidate so far stands and
#: ``metadata["domain_converged"]`` is False; with no candidate at all the
#: cavity-wide solution stands with its cut applied (Codex P2 on PR #60:
#: exiting with cut cells still live handed finite fill times to cells that
#: do not fill).
MAX_DOMAIN_PASSES = 64

#: Resolution of that bisection as a fraction of the cavity volume: the
#: reported live region is within ``V_cavity / DOMAIN_VOLUME_RESOLUTION`` of
#: the largest consistent one (never finer than one thin cell). 256 is eight
#: candidate solves, each a full skin fixed point on a cavity no larger than
#: the whole.
DOMAIN_VOLUME_RESOLUTION = 256.0

#: Time-mean of the square-root growth law over a cell's service: for
#: ``s(t) = c sqrt(alpha t)`` the mean over ``[0, a]`` is ``(2/3) s(a)``. The
#: elliptic solve carries one conductance per cell, so the skin it sees is the
#: average over the time the cell conducts, not a snapshot (Issue #61).
SKIN_SERVICE_MEAN_FACTOR = 2.0 / 3.0

# Fill-clock responses to skin resistance, see ``HeleShawSolver.skin_clock_mode``.
SKIN_CLOCK_MODES = ("constant_pressure", "constant_rate")


@dataclass
class _DomainSolution:
    """The skin fixed point solved on one candidate cavity.

    A short shot has two cavities: the one that was drawn and the part of it
    the melt can reach. Which cells belong to the second depends on when the
    cores seal, and when they seal depends on the arrival times -- which are
    solved on a cavity. The two define each other, so each candidate gets its
    own complete solution and ``solve`` iterates on the domain itself.
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
    # Cells whose skins meet before the fill ends. They filled (the front
    # passed them first) and then closed at ``t_close``.
    frozen_mask: np.ndarray | None
    # Arrival time [s] of every cell in this cavity (NaN outside) and the time
    # its core seals (inf where it never does). Both None with the skin off.
    t_arr: np.ndarray | None
    t_close: np.ndarray | None
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
    weld_score: np.ndarray  # weld-line indicator [0..1] at the default thresholds
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
    # Opening angle [deg] between the two converging flow directions at each
    # cell, NaN where no confluence. ``weld_score`` is a thresholded view of
    # this; renderers re-threshold it when the user moves the slider.
    weld_angle_deg: np.ndarray | None = None


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
    # How the fill clock responds to the resistance the skin adds.
    #
    # ``"constant_pressure"`` (default, the historical proxy): the machine
    # holds pressure, so the flow thins as the core narrows and T_fill
    # inflates by the volume-weighted tau ratio. ``"constant_rate"``: the
    # machine holds velocity, so T_fill stays the geometric V/Q and the
    # pressure rises instead. Velocity-controlled presses are the common
    # case; the proxy stays the default so existing results do not move.
    skin_clock_mode: str = "constant_pressure"

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

        Vectorised over faces: every pair of masked 4-neighbours contributes
        one harmonic-mean face conductance to both rows. A Dirichlet row is
        the identity; the rows next to it keep their -coeff in its column
        (see the module docstring on why ``A`` is not symmetric).
        """
        ny, nx = self.geometry.shape
        mask = self.geometry.mask
        dx = self.geometry.cell_size_mm * 1e-3  # cell size in meters
        # index map: only masked cells participate
        idx = -np.ones((ny, nx), dtype=np.int64)
        flat_indices = np.flatnonzero(mask.ravel())
        idx.ravel()[flat_indices] = np.arange(flat_indices.size)
        N = flat_indices.size

        diag = np.zeros(N, dtype=np.float64)
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        vals: list[np.ndarray] = []
        for dy, dx_ in ((1, 0), (0, 1)):
            a_mask = mask[: ny - dy, : nx - dx_]
            b_mask = mask[dy:, dx_:]
            both = a_mask & b_mask
            if not both.any():
                continue
            S_a = S[: ny - dy, : nx - dx_][both]
            S_b = S[dy:, dx_:][both]
            total = S_a + S_b
            # harmonic mean for face conductance; a closed face conducts nothing
            face = np.where(total > 0, 2.0 * S_a * S_b / np.where(total > 0, total, 1.0), 0.0)
            coeff = face / (dx * dx)
            ia = idx[: ny - dy, : nx - dx_][both]
            ib = idx[dy:, dx_:][both]
            np.add.at(diag, ia, coeff)
            np.add.at(diag, ib, coeff)
            rows += [ia, ib]
            cols += [ib, ia]
            vals += [-coeff, -coeff]

        is_dirichlet = np.zeros(N, dtype=bool)
        is_dirichlet[idx[dirichlet & mask]] = True
        if rows:
            r = np.concatenate(rows)
            c = np.concatenate(cols)
            v = np.concatenate(vals)
            keep = ~is_dirichlet[r]  # Dirichlet rows carry no off-diagonals
            r, c, v = r[keep], c[keep], v[keep]
        else:
            r = c = np.zeros(0, dtype=np.int64)
            v = np.zeros(0, dtype=np.float64)
        # isolated cell guard: a row with no faces gets a unit diagonal
        diag = np.where(is_dirichlet, 1.0, np.where(diag > 0, diag, 1.0))
        own = np.arange(N)
        A = sp.coo_matrix(
            (np.concatenate([v, diag]), (np.concatenate([r, own]), np.concatenate([c, own]))),
            shape=(N, N),
        ).tocsr()
        # RHS: unit source everywhere but the gates (absolute units cancel
        # against V/Q later).
        b = np.where(is_dirichlet, 0.0, 1.0)
        return A, b, idx

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

    def _unfillable_cells(
        self, frozen: np.ndarray, t_arr: np.ndarray, t_close: np.ndarray
    ) -> np.ndarray:
        """Cells the front cannot reach before every path to a gate has sealed.

        A cell is reached at ``t_arr`` if a 4-connected path of cells joins it
        to a gate along which the melt arrived no later than it did and no
        core has closed by then. A cell that never seals is open from its
        arrival on; a sealing cell is open from its arrival until ``t_close``.
        So a choke that closes at ``t_close`` cuts off exactly the cells
        behind it that arrive later -- the ones that arrived earlier filled
        through it.

        Paths run along the arrival order (the volume map is monotone in tau,
        and tau has no interior minimum), so a cell's fate follows from its
        earlier neighbours: carry along the best path the earliest closing
        time on it -- the last instant the cell still has an open route --
        and the cell fills if that instant is after its own arrival. Cells the
        melt has not reached by then are not part of any route (Codex P1 on
        PR #73: a sweep that kept every never-sealing cell present from the
        start connected a target behind a closed choke to the gate through
        cells that fill later, and kept it fillable).
        """
        cavity = self.geometry.mask
        ny, nx = cavity.shape
        n = ny * nx
        t_flat = np.where(cavity, t_arr, np.nan).ravel()
        usable = np.isfinite(t_flat)
        order = np.flatnonzero(usable)
        order = order[np.argsort(t_flat[order], kind="stable")]
        rank = np.full(n, -1, dtype=np.int64)
        rank[order] = np.arange(order.size)
        closes = np.where(cavity & frozen, t_close, np.inf).ravel()
        is_gate = np.zeros(n, dtype=bool)
        for iy, ix in self.geometry.gates:
            if cavity[iy, ix]:
                is_gate[iy * nx + ix] = True

        # latest instant at which the cell still has an open route to a gate
        # *through its own core*; -inf while it has none
        open_until = np.full(n, -np.inf)
        reach = np.zeros(n, dtype=bool)
        for i in order:
            if is_gate[i]:
                best = np.inf
            else:
                best = -np.inf
                iy, ix = divmod(int(i), nx)
                r_i = rank[i]
                for dy, dx_ in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    jy, jx = iy + dy, ix + dx_
                    if 0 <= jy < ny and 0 <= jx < nx:
                        j = jy * nx + jx
                        if 0 <= rank[j] < r_i and open_until[j] > best:
                            best = open_until[j]
            if best > t_flat[i]:
                reach[i] = True
                open_until[i] = min(best, closes[i])
        return cavity & ~reach.reshape(ny, nx)

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

    def _solve_domain(
        self,
        eta: float,
        *,
        T_fill_baseline_s: float | None = None,
        clock_end_s: float | None = None,
    ) -> _DomainSolution:
        """Solve tau, the skin fixed point and the fill time on *this* cavity.

        ``T_fill_baseline_s`` overrides the skin-free clock (the time to
        sweep this cavity; default ``_baseline_fill_time``, which carries the
        ICM equivalent-model speed-up). ``clock_end_s`` stops the exposure
        clock early: walls age from the front's passage until
        ``clock_end_s`` instead of until the fill ends — before the fill
        ends, cells the front reaches later carry no skin at all; past it,
        the walls of the full cavity keep aging (the melt stands still, the
        mold does not). The two-phase model uses it to
        read the skin at the end of a metered injection (``T_inj = V/Q``);
        with a clock end the fill time never inflates (a metered shot is by
        definition rate-controlled). Both None reproduces the plain solve.

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

        if self.skin_clock_mode not in SKIN_CLOCK_MODES:
            raise ValueError(
                f"skin_clock_mode must be one of {SKIN_CLOCK_MODES}, got {self.skin_clock_mode!r}"
            )
        T_fill_baseline = (
            float(T_fill_baseline_s)
            if T_fill_baseline_s is not None
            else self._baseline_fill_time(self.geometry)
        )
        if T_fill_baseline <= 0:
            raise ValueError(f"T_fill_baseline_s must be positive, got {T_fill_baseline}")
        if clock_end_s is not None and clock_end_s < 0:
            raise ValueError(f"clock_end_s must be non-negative, got {clock_end_s}")
        inflate = self.skin_clock_mode == "constant_pressure" and clock_end_s is None

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
        t_arr_out: np.ndarray | None = None
        t_close_out: np.ndarray | None = None
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

            # The skin at which the two walls meet, and the age at which the
            # square-root law gets there: s(t_c) = s_max. Cells already at the
            # floor have nothing to close (t_c = inf), as does a skin that
            # never grows.
            s_mm_max = np.maximum((h_open - min_core) / 2.0, 0.0)
            can_seal = cavity_mask & (h_open > min_core + 1e-9)
            t_c = np.full_like(h_open, np.inf)
            if c_skin > 0.0 and alpha > 0.0:
                t_c[can_seal] = (s_mm_max[can_seal] * 1.0e-3 / c_skin) ** 2 / alpha

            def exposure(t_arrival: np.ndarray, T: float):
                """Service duration, sealing set and time-mean skin at fill time T."""
                # A clock end past the fill keeps the walls aging after the
                # cavity is full: the melt stands still, the mold does not.
                T_end = T if clock_end_s is None else float(clock_end_s)
                a_end = np.maximum(T_end - t_arrival, 0.0)
                sealed = cavity_mask & (a_end >= t_c)
                a_rep = np.minimum(a_end, t_c)
                s_mm = SKIN_SERVICE_MEAN_FACTOR * c_skin * np.sqrt(alpha * a_rep) * 1.0e3
                s_mm = np.minimum(s_mm, s_mm_max)
                s_mm[~cavity_mask] = 0.0
                return sealed, s_mm

            for it in range(int(max(self.skin_max_iterations, 1))):
                msk = cavity_mask & ~np.isnan(tau)
                # arrival time per cell: volume CDF against the current best
                # estimate of T_fill (Issue #52)
                t_arr = self._arrival_time_field(tau, msk, cell_volume, T_fill)
                t_arr = np.where(np.isnan(t_arr), 0.0, t_arr)

                # exposure clock (Issue #61): the wall ages from the moment
                # the front passes until the fill ends or the core seals,
                # and the solve sees the skin averaged over that service
                frozen_new, s_mm_new = exposure(t_arr, T_fill)

                h_core_new = h_open - 2.0 * s_mm_new
                h_core_new = np.maximum(h_core_new, min_core)
                h_core_new[~cavity_mask] = 0.0

                # re-solve for tau with the carved core
                S_new = self._conductance_field(eta, h_core_new)
                tau_new, tau_max_new = self._solve_tau_field(S_new, dirichlet)

                # Every cell of this cavity conducts -- a sealing cell served
                # the melt before it closed, and the cells it cuts off are
                # removed from the cavity by ``solve``, not floored here.
                tau_max_flow_new = self._tau_reference(tau_new, cavity_mask)

                # constant-pressure proxy: T_fill grows with the resistance.
                # Both the numerator and the denominator are volume-weighted
                # means over the same set (Issue #52 -- the old form divided a
                # frozen-free max by an everything max, so one pathological
                # cell freezing moved the reported time by tens of percent).
                # With nothing flowing there is no resistance to speak of, so
                # the baseline stands rather than an inflation off a dead cell.
                tau_rep_new = self._tau_volume_mean(tau_new, cavity_mask, cell_volume)
                tau_rep_base = self._tau_volume_mean(tau_baseline, cavity_mask, cell_volume)
                if not inflate or tau_rep_new is None or tau_rep_base is None:
                    # rate-controlled: the clock is V/Q whatever the resistance
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

            # Read the clock off the converged fields once more, so the sealing
            # set and the closing times reported for this cavity are those of
            # the tau and T_fill it returns (the loop's were one step behind).
            msk = cavity_mask & ~np.isnan(tau)
            t_arr_out = self._arrival_time_field(tau, msk, cell_volume, T_fill)
            frozen_mask, _ = exposure(np.where(np.isnan(t_arr_out), 0.0, t_arr_out), T_fill)
            t_close_out = np.where(cavity_mask, np.nan_to_num(t_arr_out, nan=0.0) + t_c, np.inf)

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
            t_arr=t_arr_out,
            t_close=t_close_out,
            tau_rep_flow=tau_rep_flow,
            tau_rep_baseline=tau_rep_baseline,
            iterations=skin_iters_done,
            converged=skin_converged,
        )

    def _largest_consistent_prefix(
        self, eta: float, sol_full: _DomainSolution, dead_full: np.ndarray
    ) -> tuple[HeleShawSolver, _DomainSolution, np.ndarray, int, bool]:
        """The largest leading part of the fill that does not cut itself off.

        A short shot is a fixed point in two directions at once. Solving the
        whole cavity runs on a clock inflated by material that never fills,
        which seals chokes early and cuts too much; solving only what that
        pass left runs on a clock with too little resistance, under which
        nothing would have sealed -- and reports a fill that is inconsistent
        with the cut it inherited. Neither pass is the part's.

        The consistent answer is a prefix of the fill order: the cells filled
        before the front stalled. Filling more of it raises the fill time
        (more volume, more resistance) while the chokes' closing times barely
        move, so whether a prefix cuts itself off is monotone in its volume
        and bisection finds the largest one that does not. Every candidate is
        solved on its own, so the answer carries no dead load. The order is
        the cavity-wide solve's tau; each candidate is pruned to the cells
        that can still reach a gate (a partial tie group can strand one).

        Returns ``(solver, solution, live_mask, frozen_hi, passes, converged)``.
        ``frozen_hi`` is the sealing set of the smallest candidate that still
        cut itself off. The largest consistent candidate may end just before
        its own seal closes on its own clock -- the fixed point folds when a
        cell's worth of volume tips the constant-pressure feedback -- so the
        seal that cut the rest off is read off the candidate that saw it
        close. When no candidate could be tried (``MAX_DOMAIN_PASSES`` = 0)
        the cavity-wide solution stands with its cut applied and
        ``converged`` is False; its dead cells then still sit in the reported
        time scale (Codex P2 on PR #60), the price of the valve.
        """
        cavity = self.geometry.mask
        h_open = self._open_thickness_field()
        vol = (float(self.geometry.cell_size_mm) ** 2) * h_open
        tau0 = np.where(cavity & ~np.isnan(sol_full.tau), sol_full.tau, np.inf)
        flat = np.flatnonzero(cavity.ravel())
        order = flat[np.argsort(tau0.ravel()[flat], kind="stable")]
        cum = np.cumsum(vol.ravel()[order])
        V_total = float(cum[-1])
        gate_cells = np.zeros_like(cavity)
        for iy, ix in self.geometry.gates:
            gate_cells[iy, ix] = cavity[iy, ix]
        V_gate = float(np.sum(vol[gate_cells]))
        # resolution: a small fraction of the cavity, but never finer than a
        # cell of the thinnest kind (the boundary lands on a cell anyway)
        tol_cell = max(float(np.min(vol[cavity])), 1e-12)
        tol = max(tol_cell, V_total / DOMAIN_VOLUME_RESOLUTION)

        def prefix(volume: float) -> np.ndarray:
            live = np.zeros_like(cavity)
            live.ravel()[order[: int(np.searchsorted(cum, volume, side="right"))]] = True
            live |= gate_cells
            labels, _ = ndi.label(live)
            gate_labels = {int(labels[iy, ix]) for iy, ix in self.geometry.gates if live[iy, ix]}
            return live & np.isin(labels, sorted(gate_labels))

        lo, hi = V_gate, V_total
        live_lo = prefix(lo)
        best: tuple[HeleShawSolver, _DomainSolution, np.ndarray] | None = None
        # the smallest candidate that still cut itself off
        live_hi, frozen_hi, cut_hi = cavity, sol_full.frozen_mask, dead_full
        passes = 0
        while hi - lo > tol and passes < MAX_DOMAIN_PASSES:
            mid = 0.5 * (lo + hi)
            live = prefix(mid)
            sub = self._restricted_to(live)
            # Every candidate starts skin-free, as the process does. The
            # constant-pressure feedback can hold two fixed points on one
            # cavity (open and slow-and-sealing); starting from a neighbour's
            # inflated clock would pick the sealing one even where the fill,
            # run from the start, completes before anything closes.
            sol = sub._solve_domain(eta)
            passes += 1
            cut = None
            if sol.frozen_mask is not None and sol.frozen_mask.any():
                cut = sub._unfillable_cells(sol.frozen_mask, sol.t_arr, sol.t_close)
            if cut is not None and cut.any():
                hi = mid
                live_hi, frozen_hi, cut_hi = live, sol.frozen_mask, cut
            else:
                lo = mid
                live_lo = live
                best = (sub, sol, live)
        converged = hi - lo <= tol

        # The boundary lies in the band between the two candidates, which
        # the bisection resolves to a volume, not to a cell -- a thick cell
        # straddling it is dropped whole. If the over-large candidate's cut
        # lies inside that band, it has located the boundary on a clock that
        # carried at most a band's worth of dead load: take that candidate
        # minus its cut, solved on its own. A cut reaching below the band is
        # the feedback folding over (a sliver more tips the whole fill into
        # sealing); there the largest candidate that fills is the answer.
        band = live_hi & ~live_lo
        refine = cut_hi is not None and cut_hi.any() and not (cut_hi & ~band).any()
        live = live_hi
        cut = cut_hi
        # Shed the cut and solve again, a few times if the cleaner clock cuts
        # once more (it can: the candidate carried up to a band of dead
        # load). Each round loses at least a cell, so this settles; it is
        # also what finds the fillable sliver when every bisection candidate
        # was too large (Codex P1 on PR #73: with nothing consistent solved,
        # the cavity-wide solution stood with the dead region's resistance
        # still in its clock).
        while refine and cut is not None and cut.any() and passes < MAX_DOMAIN_PASSES:
            live = live & ~cut
            if not live.any():
                break
            sub = self._restricted_to(live)
            if not sub.geometry.gates:
                break
            sol = sub._solve_domain(eta)
            passes += 1
            cut = None
            if sol.frozen_mask is not None and sol.frozen_mask.any():
                cut = sub._unfillable_cells(sol.frozen_mask, sol.t_arr, sol.t_close)
            if cut is None or not cut.any():
                best = (sub, sol, live)
        if best is None:
            return self, sol_full, cavity & ~dead_full, frozen_hi, passes, False
        sub, sol, live = best
        return sub, sol, live, frozen_hi, passes, converged

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

        # If the cavity-wide solve cuts cells off, the part that fills is the
        # largest prefix of the fill order that does not cut itself -- found by
        # bisection on filled volume, each candidate solved on its own.
        domain_solver = self
        live_mask = cavity_mask.copy()
        frozen_hi: np.ndarray | None = None
        domain_passes = 0
        domain_converged = True
        if sol.frozen_mask is not None and sol.frozen_mask.any():
            dead0 = self._unfillable_cells(sol.frozen_mask, sol.t_arr, sol.t_close)
            if dead0.any():
                domain_solver, sol, live_mask, frozen_hi, domain_passes, domain_converged = (
                    self._largest_consistent_prefix(eta, sol, dead0)
                )
        dead_cells = cavity_mask & ~live_mask
        unfillable_mask = dead_cells if dead_cells.any() else None
        # The solution of the part that fills says nothing about the cells it
        # does not contain: no skin grew there and no core was carved, but
        # zero would draw them as closed (Codex P2 on PR #73). NaN, like
        # their fill time.
        skin_thk_mm = sol.skin_thk_mm
        h_core_mm = sol.h_core_mm
        if skin_thk_mm is not None and dead_cells.any():
            skin_thk_mm = np.where(dead_cells, np.nan, skin_thk_mm)
        if h_core_mm is not None and dead_cells.any():
            h_core_mm = np.where(dead_cells, np.nan, h_core_mm)

        tau = sol.tau
        tau_max = sol.tau_max
        tau_max_flow = sol.tau_max_flow
        T_fill_baseline = sol.T_fill_baseline
        tau_max_baseline = sol.tau_max_baseline
        # The domain solution already carries its own constant-pressure
        # inflation (volume-weighted, over its own live set). Recomputing it
        # here from single-cell maxima would reintroduce the one-cell
        # normalization this branch exists to remove.
        T_fill = sol.T_fill

        # Where freeze-off happened: the live cells whose core closed during
        # the fill of what fills, plus the ones the smallest over-large
        # candidate saw close -- the seal that ends the shot closes on the
        # clock of a fill one sliver larger than the one that completes. A
        # sealing cell filled before it closed; missing melt is
        # ``unfillable_mask``, and a sealing cell that was itself cut off is
        # dead, not sealed.
        short_shot_mask: np.ndarray | None = None
        if sol.frozen_mask is not None:
            short_shot_mask = sol.frozen_mask & live_mask
            if frozen_hi is not None and dead_cells.any():
                short_shot_mask |= frozen_hi & live_mask

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
        weld_angle = self._weld_meeting_angle(tau_flow)
        weld_score = weld_score_from_angle(weld_angle)
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
                    # cells whose core sealed during the fill (they filled first)
                    "short_shot_cells": short_count,
                    "short_shot_fraction": short_count / cells_total,
                    # cells the front never reached: cut off behind a seal
                    "unfillable_cells": (
                        int(unfillable_mask.sum()) if unfillable_mask is not None else 0
                    ),
                    "sealed_off_cells": (
                        int(unfillable_mask.sum()) if unfillable_mask is not None else 0
                    ),
                    "filled_volume_fraction": (
                        float(np.sum(cell_volume_final[live_mask]))
                        / max(float(np.sum(cell_volume_final[cavity_mask])), 1e-30)
                    ),
                    "skin_clock": "exposure",
                    "skin_clock_mode": self.skin_clock_mode,
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
            weld_angle_deg=weld_angle,
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
    def _flow_direction(tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Unit vector of the front's advance (+grad tau) per cell.

        Differences use only cavity neighbours: central where both sides
        exist, one-sided at walls. Cells with no valid neighbour along an
        axis get zero for that component; cells with no gradient at all
        (isolated, or flat such as a gate plateau) come back NaN.
        """
        valid = ~np.isnan(tau)
        t = np.where(valid, tau, 0.0)
        comps = []
        for axis in (1, 0):  # x then y
            tp = np.roll(t, -1, axis=axis)
            tm = np.roll(t, 1, axis=axis)
            vp = np.roll(valid, -1, axis=axis)
            vm = np.roll(valid, 1, axis=axis)
            # np.roll wraps; the wrapped neighbour is never a real one
            edge_p = np.zeros_like(valid)
            edge_m = np.zeros_like(valid)
            if axis == 1:
                edge_p[:, -1] = True
                edge_m[:, 0] = True
            else:
                edge_p[-1, :] = True
                edge_m[0, :] = True
            vp &= ~edge_p
            vm &= ~edge_m
            g = np.zeros_like(t)
            both = valid & vp & vm
            fwd = valid & vp & ~vm
            bwd = valid & ~vp & vm
            g[both] = 0.5 * (tp[both] - tm[both])
            g[fwd] = tp[fwd] - t[fwd]
            g[bwd] = t[bwd] - tm[bwd]
            comps.append(g)
        gx, gy = comps
        norm = np.hypot(gx, gy)
        ok = valid & (norm > 0)
        safe = np.where(ok, norm, 1.0)
        ux = np.where(ok, gx / safe, np.nan)
        uy = np.where(ok, gy / safe, np.nan)
        return ux, uy

    @staticmethod
    def _compute_weld_score(
        tau: np.ndarray,
        *,
        min_angle_deg: float = WELD_MIN_ANGLE_DEG,
        full_angle_deg: float = WELD_FULL_ANGLE_DEG,
    ) -> np.ndarray:
        """Weld / meld score: the meeting angle, thresholded (see both halves)."""
        return weld_score_from_angle(
            HeleShawSolver._weld_meeting_angle(tau, min_angle_deg=min_angle_deg),
            min_angle_deg=min_angle_deg,
            full_angle_deg=full_angle_deg,
        )

    @staticmethod
    def _weld_meeting_angle(tau: np.ndarray, *, min_angle_deg: float = 0.0) -> np.ndarray:
        """Opening angle [deg] at which fronts meet, NaN where they do not.

        For each cell, look at the four pairs of opposite neighbours (x, y,
        both diagonals). A pair whose flow directions both point *toward*
        the cell is a confluence; the angle between the two directions is
        the opening angle of the notch that closes there (180 deg = head-on
        collision, 0 deg = parallel streams that merely merge). The score is
        that angle mapped linearly from ``min_angle_deg`` (0) to
        ``full_angle_deg`` (1) and clipped -- the usual CAE convention is a
        "weld" below a meeting angle of 135 deg (here: opening angle above
        45 deg) and a fainter "meld" beyond, so the ramp reaches 1 where the
        weld regime begins and fades across the meld regime.

        The converging-flow requirement is what separates a weld from a
        split: the two sides of an obstacle see the same opening angle, but
        their flows point away from the cell, not into it. Cells within two
        of a gate are zeroed -- the rasterised gate rim gives one-sided
        differences whose directions are noise, and a weld cannot sit on
        the gate anyway. ``min_angle_deg`` only sets the approach margin
        below; thresholding into a score is :func:`weld_score_from_angle`.
        For the same reason only directions measured by
        central differences (cells whose 8-neighbourhood is all cavity) take
        part: the pair on either side of a wall cell is still examined, but
        a neighbour *on* the wall contributes nothing.

        The older heuristic ("6 of 8 neighbours filled earlier") was a
        near-local-maximum test and missed every weld that keeps flowing:
        behind an obstacle the downstream row is always later, so at most
        5 neighbours qualify and a straight weld behind a hole drew nothing.
        """
        if not 0.0 <= min_angle_deg <= 180.0:
            raise ValueError(f"min_angle_deg must be within [0, 180], got {min_angle_deg}")
        ux, uy = HeleShawSolver._flow_direction(tau)
        ny, nx = tau.shape
        # A direction is trusted only where it came from central differences
        # on both axes: at a rasterised wall the one-sided difference follows
        # the staircase, and a diagonal edge then sprinkles isolated
        # "confluences" along itself.
        valid = ~np.isnan(tau)
        interior = valid & ndi.binary_erosion(valid, structure=np.ones((3, 3), bool))
        ux = np.where(interior, ux, np.nan)
        uy = np.where(interior, uy, np.nan)

        def shifted(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
            """Value of the neighbour at (iy+dy, ix+dx); NaN past the grid."""
            out = np.full_like(a, np.nan)
            ys = slice(max(dy, 0), ny + min(dy, 0))
            xs = slice(max(dx, 0), nx + min(dx, 0))
            yd = slice(max(-dy, 0), ny + min(-dy, 0))
            xd = slice(max(-dx, 0), nx + min(-dx, 0))
            out[yd, xd] = a[ys, xs]
            return out

        approach = np.sin(np.radians(0.5 * min_angle_deg))
        best = np.full_like(tau, np.nan)
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            ax, ay = shifted(ux, dy, dx), shifted(uy, dy, dx)
            bx, by = shifted(ux, -dy, -dx), shifted(uy, -dy, -dx)
            # offset direction from the centre to neighbour a (b is -that)
            d = np.hypot(dx, dy)
            ex, ey = dx / d, dy / d
            # each stream must actually approach the cell, by at least the
            # transverse component a symmetric confluence of the minimum
            # opening angle would have; flow lines that merely bend around a
            # corner graze the threshold from the wrong side otherwise
            toward_a = ax * ex + ay * ey < -approach
            toward_b = bx * (-ex) + by * (-ey) < -approach
            # and both must have arrived earlier: the cell is the crest of the
            # arrival time along this axis. A flow merely turning in front of
            # an obstacle has one neighbour upstream and one downstream, and
            # passes the direction test at small angles otherwise.
            # "crest" means a crest on both sides: a cell sitting on the
            # contour of one neighbour (drop ~ 0) while the other is far
            # upstream is a flow turning a corner, not two streams closing.
            with np.errstate(invalid="ignore"):
                drop_a = tau - shifted(tau, dy, dx)
                drop_b = tau - shifted(tau, -dy, -dx)
                # A symmetric ridge on an even grid is two *exactly* tied
                # columns wide (machine precision -- symmetric solves tie to
                # ~1e-16). A tied side is the same crest cell, so its drop is
                # read one cell further out. A merely small drop is not a
                # tie: that is the grazing contour of a turning flow.
                tie_tol = CREST_TIE_RTOL * np.abs(tau)
                far_a = tau - shifted(tau, 2 * dy, 2 * dx)
                far_b = tau - shifted(tau, -2 * dy, -2 * dx)
                eff_a = np.where(np.abs(drop_a) <= tie_tol, far_a, drop_a)
                eff_b = np.where(np.abs(drop_b) <= tie_tol, far_b, drop_b)
                crest = (
                    (drop_a >= -tie_tol)
                    & (drop_b >= -tie_tol)
                    & (eff_a >= CREST_BALANCE * eff_b)
                    & (eff_b >= CREST_BALANCE * eff_a)
                )
            with np.errstate(invalid="ignore"):
                dot = np.clip(ax * bx + ay * by, -1.0, 1.0)
                angle = np.degrees(np.arccos(dot))
            best = np.fmax(best, np.where(toward_a & toward_b & crest, angle, np.nan))

        near_gate = ndi.binary_dilation(tau == 0.0, structure=np.ones((5, 5), bool))
        best[near_gate | np.isnan(tau)] = np.nan
        return best

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
