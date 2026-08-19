"""Two-phase (injection + compression) short-shot model.

Purpose: predict the *shape* of a metering-limited short shot molded with
injection-compression (ICM), using the real machine parameters (stroke,
rate, metered shot volume) instead of reverse-fitted knobs.

Model (two linear solves, no time marching):

1. **Injection phase** — solve the pseudo-conduction field ``tau1`` on the
   *open-gap* cavity (``h + stroke`` on the compression mask, per the
   solver's compression settings), then cut the volume CDF at the metered
   shot volume ``V_shot``. The prefix of cells in ``tau1`` order whose
   open-gap volume fits inside ``V_shot`` is the melt region at the end of
   injection, ``Omega1``. At constant rate this is exact for the front
   position: the front sweeps volume linearly in time.
2. **Compression phase** — solve ``tau2`` on the *final-thickness* cavity
   with Dirichlet (``tau = 0``) on **all** of ``Omega1`` (the melt pool acts
   as an equipotential source while the mold closes), then advance cells in
   ``tau2`` order until the final-thickness volume of the filled region
   equals ``V_shot`` (volume conservation: the same melt, squeezed thinner,
   covers more area). The result is ``Omega2 ⊇ Omega1``, the short-shot
   shape after compression.

Deliberate limitations (documented, not bugs):

- **No freezing during compression.** The skin-layer model is rejected at
  the entry; a staged (metering-limited) short stops on volume, not on
  freeze-off, which is exactly the case this model is for.
- **No injection/compression overlap.** The phases are strictly
  sequential; machines that start closing while still injecting are outside
  the model.
- **Melt pool as equipotential source.** Phase 2 pins ``tau = 0`` on all of
  ``Omega1``, i.e. pressure gradients *inside* the pool are neglected
  relative to the resistance of the unfilled front. Reasonable for thin
  plates where the pool conductance dwarfs the front's.
- **Only ``Omega1`` nests across shot volumes.** The injection pool is a
  prefix of one fixed ``tau1`` order, so a larger metered shot strictly
  contains a smaller one. ``Omega2`` carries no such guarantee: each shot's
  phase 2 solves ``tau2`` against its own pool boundary, and on branched or
  nonuniform cavities the reordering can trade a cell near one front for a
  cell near another. Each result is a self-consistent quasi-static history
  of *that* shot; a volume sweep is a family of runs, not one nested film.

With ``compression_molding=False`` (or stroke 0) the open and final gaps
coincide and phase 2 is skipped outright (``Omega2 == Omega1``) — the model
degrades gracefully to a plain volume-limited short shot. A metered shot
smaller than the gate region's open-gap volume is rejected: the gate cells
are the tau = 0 tie group of the volume CDF, so a shot that cannot cover
them has no consistent melt region.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import Geometry
from .solver import HeleShawSolver, check_gate_reachability

# Relative slack for volume comparisons. Tie groups are atomic, so the cut
# never lands mid-cell; the slack only absorbs float noise in the cumsum.
_REL_EPS = 1e-12


@dataclass
class TwoPhaseShortShotResult:
    """Outcome of the two-phase short-shot solve.

    ``geometry`` is carried so the visualizer's extent/gate helpers can
    duck-type this like a ``FlowResult``.
    """

    geometry: Geometry
    # Melt region at the end of injection (open gap), gates always included.
    injection_mask: np.ndarray
    # Melt region after compression (final thickness). Superset of
    # ``injection_mask``.
    final_mask: np.ndarray
    # Arrival time [s] during injection; NaN outside ``injection_mask``.
    injection_fill_time_s: np.ndarray
    # Normalized advance order (0..1] during compression; NaN outside
    # ``final_mask & ~injection_mask``.
    compression_progress: np.ndarray
    # Raw pseudo-conduction fields (NaN outside the cavity). ``tau2`` is
    # None when phase 2 never solved (no gap closure, full fill already at
    # injection, or nothing to advance).
    tau1: np.ndarray
    tau2: np.ndarray | None
    shot_volume_cm3: float
    injection_time_s: float  # V_shot / Q
    viscosity_Pa_s: float
    metadata: dict


def _prefix_by_volume(tau_vals: np.ndarray, volumes: np.ndarray, budget: float) -> np.ndarray:
    """Boolean take-mask: the ``tau``-ordered prefix whose volume fits ``budget``.

    Tie groups are atomic — a group of equal ``tau`` is taken only if the
    *whole* group fits. This matches ``_arrival_time_field`` semantics
    (ties share the group-end arrival), so "arrival <= T_inj" and "group
    cumulative volume <= V_shot" select the same cells.
    """
    take = np.zeros(tau_vals.shape, dtype=bool)
    if budget <= 0 or tau_vals.size == 0:
        return take
    order = np.argsort(tau_vals, kind="stable")
    tau_sorted = tau_vals[order]
    cum = np.cumsum(volumes[order])
    last = np.searchsorted(tau_sorted, tau_sorted, side="right") - 1
    group_cum = cum[last]
    take[order] = group_cum <= budget * (1.0 + _REL_EPS)
    return take


def _group_end_progress(tau_vals: np.ndarray, volumes: np.ndarray, budget: float) -> np.ndarray:
    """Normalized advance order for the advanced cells: group-end cumulative
    volume over ``budget``, so ties share one value and the last group lands
    at (close to) 1."""
    order = np.argsort(tau_vals, kind="stable")
    tau_sorted = tau_vals[order]
    cum = np.cumsum(volumes[order])
    last = np.searchsorted(tau_sorted, tau_sorted, side="right") - 1
    prog = np.empty(tau_vals.shape)
    prog[order] = np.minimum(cum[last] / budget, 1.0)
    return prog


def solve_two_phase_short_shot(
    solver: HeleShawSolver,
    shot_volume_cm3: float,
) -> TwoPhaseShortShotResult:
    """Run the two-phase model on a configured ``HeleShawSolver``.

    The solver supplies the geometry, material, temperatures, rate and the
    compression settings (mode, factor / stroke, mask). ``shot_volume_cm3``
    is the metered shot volume — the real machine number.
    """
    if shot_volume_cm3 <= 0:
        raise ValueError("shot_volume_cm3 must be positive")
    if solver.skin_layer_enabled:
        raise ValueError(
            "two-phase short-shot model does not support the skin-layer model: "
            "a metering-limited short stops on volume, not on freeze-off "
            "(freezing during compression is deliberately out of scope)"
        )
    geom = solver.geometry
    if not geom.gates:
        raise ValueError("Geometry has no gates")
    check_gate_reachability(geom)

    mask = geom.mask
    dx = float(geom.cell_size_mm)
    eta = solver._effective_viscosity()
    Q_cm3s = solver._effective_flow_rate_cm3s()
    V_shot_mm3 = float(shot_volume_cm3) * 1000.0

    gate_dirichlet = np.zeros(geom.shape, dtype=bool)
    for iy, ix in geom.gates:
        gate_dirichlet[iy, ix] = True

    # ---- Phase 1: injection at the open gap -------------------------------
    h_open = solver._open_thickness_field()  # mm; includes compression inflation
    h_fin = geom.thickness_mm
    # The model's arithmetic assumes the mold only ever closes: h_open >=
    # h_fin everywhere. A compression_factor < 1 or a negative stroke would
    # let a shot fill the whole open cavity while holding less than the
    # final cavity volume — phase 2 is then skipped (gap_closes is False)
    # and the achieved volume exceeds the metered shot (Codex P2, round 2).
    if bool(np.any(h_open[mask] < h_fin[mask] * (1.0 - _REL_EPS))):
        raise ValueError(
            "compression settings shrink the open gap below the final "
            "thickness (compression_factor < 1 or a negative stroke) — the "
            "two-phase model requires h_open >= h_final on every cavity cell"
        )
    vol_open = dx * dx * h_open  # mm^3 per cell when swept at the open gap
    S1 = solver._conductance_field(eta, h_open)
    tau1, _ = solver._solve_tau_field(S1, gate_dirichlet)

    V_open_total = float(vol_open[mask].sum())
    T_open_total = V_open_total / 1000.0 / Q_cm3s  # s to fill the whole open cavity
    t_arr1 = solver._arrival_time_field(tau1, mask, vol_open, T_open_total)
    T_inj = V_shot_mm3 / 1000.0 / Q_cm3s

    # The gate cells all sit at tau = 0, so they form the first tie group of
    # the volume CDF. A metered shot that cannot even cover that group has no
    # consistent melt region in this model: forcing the gates in regardless
    # would report an achieved volume larger than the metered shot (Codex P2
    # on PR #62), and leaving them out would strand phase 2 without a
    # Dirichlet source. Reject the input instead.
    V_gate_mm3 = float(vol_open[gate_dirichlet & mask].sum())
    if V_shot_mm3 < V_gate_mm3 * (1.0 - _REL_EPS):
        raise ValueError(
            "shot_volume_cm3 is smaller than the gate region's open-gap volume "
            f"({V_gate_mm3 / 1000.0:.4f} cm^3) — the metered shot cannot cover "
            "the gate cells, so the model has no melt region to grow from"
        )

    injection_complete = V_shot_mm3 >= V_open_total * (1.0 - _REL_EPS)
    omega1 = np.zeros(geom.shape, dtype=bool)
    if injection_complete:
        omega1 |= mask
    else:
        sel = mask & ~np.isnan(tau1)
        take = _prefix_by_volume(tau1[sel], vol_open[sel], V_shot_mm3)
        omega1[sel] = take

    injection_fill_time_s = np.where(omega1, t_arr1, np.nan)

    # ---- Phase 2: compression at the final thickness ----------------------
    vol_fin = dx * dx * h_fin
    V_fin_total = float(vol_fin[mask].sum())

    omega2 = omega1.copy()
    tau2: np.ndarray | None = None
    compression_progress = np.full(geom.shape, np.nan)
    final_complete = V_shot_mm3 >= V_fin_total * (1.0 - _REL_EPS)
    # Phase 2 models the mold closing. When the gap does not actually close
    # anywhere (no ICM, stroke 0, factor 1) there is no squeeze to advance
    # the front — but the atomic phase-1 cutoff can still leave a residual
    # budget, and on a nonuniform-thickness cavity the re-solved tau2
    # ordering could hand that residual to some smaller cell (Codex P2 on
    # PR #62). Gate the phase on the physics, not on the residual being zero.
    gap_closes = bool(np.any(h_open[mask] > h_fin[mask] * (1.0 + _REL_EPS)))

    if final_complete:
        omega2 |= mask
        # Injection incomplete but the shot covers the whole final cavity —
        # the everyday ICM full-fill case (V_fin <= V_shot < V_open). The
        # shape needs no solve, but the result contract promises a normalized
        # advance order on the compression-filled cells, so tau2 is still
        # computed to order them (Codex P2, round 3). With injection already
        # complete there is nothing to order; without gap closure this branch
        # implies injection_complete anyway (V_shot >= V_fin = V_open).
        candidates = mask & ~omega1
        if gap_closes and not injection_complete and candidates.any():
            S2 = solver._conductance_field(eta, h_fin)
            tau2, _ = solver._solve_tau_field(S2, omega1)
            cand_sel = candidates & ~np.isnan(tau2)
            total = float(vol_fin[cand_sel].sum())
            if total > 0:
                compression_progress[cand_sel] = _group_end_progress(
                    tau2[cand_sel], vol_fin[cand_sel], total
                )
    elif gap_closes:
        # h_open >= h_fin everywhere, so V_open_total >= V_fin_total and a
        # complete injection would have implied a complete final fill —
        # reaching this branch means omega1 is a strict subset of the cavity.
        budget = V_shot_mm3 - float(vol_fin[omega1].sum())
        candidates = mask & ~omega1
        if budget > 0 and candidates.any():
            S2 = solver._conductance_field(eta, h_fin)
            tau2, _ = solver._solve_tau_field(S2, omega1)
            cand_sel = candidates & ~np.isnan(tau2)
            take = _prefix_by_volume(tau2[cand_sel], vol_fin[cand_sel], budget)
            advanced = np.zeros(geom.shape, dtype=bool)
            advanced[cand_sel] = take
            omega2 |= advanced
            if advanced.any():
                compression_progress[advanced] = _group_end_progress(
                    tau2[advanced], vol_fin[advanced], budget
                )

    achieved_mm3 = float(vol_fin[omega2].sum())
    n_cavity = int(mask.sum())
    metadata = {
        "model": "two_phase_short_shot",
        "shot_volume_cm3": float(shot_volume_cm3),
        "flow_rate_cm3s": Q_cm3s,
        "injection_time_s": T_inj,
        "cavity_volume_open_cm3": V_open_total / 1000.0,
        "cavity_volume_final_cm3": V_fin_total / 1000.0,
        "injection_complete": bool(injection_complete),
        "final_complete": bool(omega2.sum() == n_cavity),
        "injection_cells": int(omega1.sum()),
        "final_cells": int(omega2.sum()),
        "cavity_cells": n_cavity,
        "injection_fill_fraction": float(vol_open[omega1].sum() / V_open_total)
        if V_open_total > 0
        else 0.0,
        "final_fill_fraction": achieved_mm3 / V_fin_total if V_fin_total > 0 else 0.0,
        "achieved_volume_final_cm3": achieved_mm3 / 1000.0,
        "compression_mode": (
            "off"
            if not solver.compression_molding
            else ("stroke" if solver.compression_stroke_mm is not None else "factor")
        ),
        "compression_stroke_mm": solver.compression_stroke_mm,
        "compression_factor": solver.compression_factor
        if solver.compression_molding and solver.compression_stroke_mm is None
        else None,
    }

    return TwoPhaseShortShotResult(
        geometry=geom,
        injection_mask=omega1,
        final_mask=omega2,
        injection_fill_time_s=injection_fill_time_s,
        compression_progress=compression_progress,
        tau1=tau1,
        tau2=tau2,
        shot_volume_cm3=float(shot_volume_cm3),
        injection_time_s=T_inj,
        viscosity_Pa_s=eta,
        metadata=metadata,
    )


@dataclass
class TwoPhaseFrame:
    """One animation frame of the two-phase history — pure data, no rendering.

    ``phase`` is ``"injection"`` or ``"compression"``. During injection
    ``value`` is the real elapsed time [s]; during compression it is the
    normalized advance fraction (0..1] — the model has no compression
    timescale, only an order.
    """

    phase: str
    value: float
    injection_filled: np.ndarray
    compression_filled: np.ndarray


def frame_states(result: TwoPhaseShortShotResult, num_frames: int = 24) -> list[TwoPhaseFrame]:
    """Frame sequence for the two-phase animation.

    Frames are split between the phases in proportion to the cell count each
    phase fills (minimum 3 per active phase). Injection frames advance in
    real arrival time from 0 to ``injection_time_s``; compression frames
    advance in normalized order. The filled sets grow monotonically and the
    last frame covers exactly ``final_mask``.
    """
    if num_frames < 2:
        raise ValueError("num_frames must be at least 2")
    inj = result.injection_mask
    adv = result.final_mask & ~inj
    n_inj_cells = int(inj.sum())
    n_adv_cells = int(adv.sum())
    total = max(n_inj_cells + n_adv_cells, 1)

    if n_adv_cells == 0:
        n1, n2 = num_frames, 0
    else:
        # both phases active: reserve at least ``lo`` frames for each side
        # (3 when the budget allows, 1 when num_frames is very small)
        lo = 3 if num_frames >= 6 else 1
        n1 = int(round(num_frames * n_inj_cells / total))
        n1 = min(max(n1, lo), num_frames - lo)
        n2 = num_frames - n1

    t_arr = result.injection_fill_time_s
    T_inj = result.injection_time_s
    none_yet = np.zeros(result.geometry.shape, dtype=bool)
    frames: list[TwoPhaseFrame] = []
    for t in np.linspace(0.0, T_inj, n1):
        filled = inj & (np.nan_to_num(t_arr, nan=np.inf) <= t * (1.0 + _REL_EPS))
        frames.append(
            TwoPhaseFrame(
                phase="injection",
                value=float(t),
                injection_filled=filled,
                compression_filled=none_yet,
            )
        )
    if n2:
        prog = np.nan_to_num(result.compression_progress, nan=np.inf)
        # start strictly after 0 — frame n1 already shows the full pool
        for p in np.linspace(1.0 / n2, 1.0, n2):
            filled = adv & (prog <= p * (1.0 + _REL_EPS))
            frames.append(
                TwoPhaseFrame(
                    phase="compression",
                    value=float(p),
                    injection_filled=inj,
                    compression_filled=filled,
                )
            )
    return frames
