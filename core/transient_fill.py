"""Time-marching fill: the melt front advanced in time (v0.62.0).

The τ solvers order the fill with a single elliptic solve, ``-∇·(S∇τ) = 1``.
Every cell is a unit sink in that problem, so the deep parts of a gate block --
where most of the sink volume sits behind a small resistance -- come out
"filled first", and the thin product only starts once the block is nearly full.
The FG9 VP-series photos contradict that: at 5.8 g the block is 64 % full and
the product centre is already ahead (2026-10-07). Advancing the front in time
gets the block's partial fill and the early centre lead right.

This module is the marching core, kept free of the material model: the caller
hands in the conductance of a cell as a function of the time the melt reached
it, and the volume the machine has injected as a function of time.

Scheme (IMPES: pressure implicit, fill explicit), per step of length ``dt``:

* full cells (``V >= C``) carry the unknown pressure; every cell that is not
  full but touches a full cell (or already holds melt) is the front, ``p = 0``;
* the gate cells share one pressure ``p_g`` (the τ solvers hold them all at
  ``τ = 0``), chosen so that the melt leaving them over the step is what the
  machine delivered. The system is linear in ``p_g``, so one factorization with
  two right-hand sides gives it;
* volume balance of a full cell ``i``::

      Σ_j S_ij (p_i - p_j) = (V_i - C_i) / dt + q_i

  where ``S_ij`` is the harmonic mean of the two cell conductances (the same
  face rule as the τ system), ``q_i`` the gate inflow over the step and
  ``(V_i - C_i)/dt`` pushes out what a cell received beyond its capacity in the
  previous step -- the volume is conserved to round-off;
* front cells take the inflow explicitly. A cell that fills during the step is
  stamped with the time its capacity was reached, interpolated linearly in the
  step, and its conductance is evaluated once at that time and kept;
* the next step is ``cfl`` times the time the fastest front cell needs to take
  one cell of melt, capped by ``dt_max_s``.

The conductance of a front cell is evaluated at the current time (the melt
arriving now). Units are SI: volumes m³, conductance m³/(Pa·s), pressure Pa.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

#: A cell counts as full once it holds this fraction of its capacity. The step is
#: sized to fill a cell exactly, and the inflow comes out of a linear solve, so
#: "exactly" lands a few parts in 10^9 either side; a cell left that far short
#: took a second cell of melt in the next step (a strip was stamped 1.5 % late).
FULL_REL = 1.0 - 1e-6


@dataclass
class MarchFillResult:
    """Outcome of :func:`march_fill`.

    ``t_arr_s`` is the time each cell became full (NaN outside the cavity and
    for cells still unfilled when the march stopped). ``pressure_end_Pa`` is
    the pressure solved in the last step -- the field at the end of the fill
    (zero at the front and outside the cavity).
    """

    t_arr_s: np.ndarray
    pressure_end_Pa: np.ndarray
    n_steps: int
    t_end_s: float
    complete: bool
    volume_injected_m3: float
    volume_held_m3: float


def _faces(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """4-neighbour faces inside ``mask`` as flat index pairs ``(a, b)``."""
    ny, nx = mask.shape
    ids = np.arange(ny * nx).reshape(ny, nx)
    fa, fb = [], []
    for dy, dx in ((1, 0), (0, 1)):
        both = mask[: ny - dy, : nx - dx] & mask[dy:, dx:]
        fa.append(ids[: ny - dy, : nx - dx][both])
        fb.append(ids[dy:, dx:][both])
    return np.concatenate(fa), np.concatenate(fb)


def march_fill(
    mask: np.ndarray,
    capacity_m3: np.ndarray,
    gates: list[tuple[int, int]],
    injected_volume_m3: Callable[[float], float],
    conductance: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    t_start_s: float,
    dt_max_s: float,
    cfl: float = 1.0,
    breakpoints_s: tuple[float, ...] = (),
    max_steps: int = 200_000,
) -> MarchFillResult:
    """Advance the melt front from the gates until the cavity is full.

    Parameters
    ----------
    mask
        ``(ny, nx)`` cavity cells.
    capacity_m3
        ``(ny, nx)`` melt volume each cell holds when full.
    gates
        Gate cells ``(iy, ix)``; they start full at ``t_start_s``.
    injected_volume_m3
        Total volume the machine has delivered by time ``t``. It must reach the
        gate cells' capacity at ``t_start_s``; the march takes in the increments
        over each step (a staged profile is followed exactly).
    conductance
        ``conductance(flat_index, t_arr)`` → conductance [m³/(Pa·s)] of those
        cells for melt that reached them at ``t_arr``.
    breakpoints_s
        Times where the delivery rate changes (a staged profile's switches). A
        step never crosses one, and the step after it is sized on the new rate.
    """
    if not 0.0 < cfl <= 1.0:
        raise ValueError(f"cfl must be in (0, 1] (got {cfl})")
    if dt_max_s <= 0:
        raise ValueError(f"dt_max_s must be positive (got {dt_max_s})")
    ny, nx = mask.shape
    n = ny * nx
    m = mask.ravel().astype(bool)
    C = np.where(m, np.asarray(capacity_m3, float).ravel(), 0.0)
    if np.any(C[m] <= 0):
        raise ValueError("every cavity cell needs a positive capacity")
    g = np.array([iy * nx + ix for iy, ix in gates], dtype=np.int64)
    g = g[m[g]]
    if g.size == 0:
        raise ValueError("no gate cell lies in the cavity")
    fa, fb = _faces(mask)
    is_gate = np.zeros(n, dtype=bool)
    is_gate[g] = True

    V = np.zeros(n)
    V[g] = C[g]
    t = float(t_start_s)
    t_arr = np.full(n, np.nan)
    t_arr[g] = t
    S = np.zeros(n)
    S[g] = conductance(g, t_arr[g])
    p = np.zeros(n)
    v_prev_inj = float(injected_volume_m3(t))
    bps = sorted(b for b in breakpoints_s if b > t)
    probe = min(float(dt_max_s), 1e-6 * max(abs(t), 1.0))

    def rate_ahead(tt: float) -> float:
        """Delivery rate just after ``tt`` (one-sided, so a switch at ``tt`` counts)."""
        return max(
            (float(injected_volume_m3(tt + probe)) - float(injected_volume_m3(tt))) / probe, 1e-300
        )

    # first step: the gate's neighbours share the injection; give the smallest of
    # them one cell of melt at most
    rate0 = rate_ahead(t)
    nb = np.zeros(n, dtype=bool)
    nb[fa[is_gate[fb]]] = True
    nb[fb[is_gate[fa]]] = True
    nb &= m & ~is_gate
    dt = float(dt_max_s)
    if nb.any():
        dt = min(dt, cfl * float(np.min(C[nb])) * int(nb.sum()) / rate0)
    steps = 0
    complete = False

    while steps < max_steps:
        full = m & (V >= C * FULL_REL)
        if full[m].all():
            complete = True
            break
        touch = np.zeros(n, dtype=bool)
        touch[fa[full[fb]]] = True
        touch[fb[full[fa]]] = True
        front = m & ~full & (touch | (V > 0))
        fr = np.flatnonzero(front)
        if fr.size == 0:  # nothing left the front can reach (cannot happen on a connected cavity)
            break
        S[fr] = conductance(fr, np.full(fr.size, t))

        # a switch within round-off of now has been reached (t is a running sum)
        while bps and bps[0] <= t + 1e-9 * max(abs(t), dt_max_s):
            bps.pop(0)
        if bps:
            dt = min(dt, bps[0] - t)
        # never deliver more than the cavity still has room for (the last step
        # would otherwise overshoot the full cavity by up to a step's worth)
        # (net of the surplus cells hold and pass on next step)
        room_left = float(np.sum(C[m] - V[m]))
        if room_left > 0:
            dt = max(min(dt, room_left / rate_ahead(t)), 1e-12)
        v_next = float(injected_volume_m3(t + dt))
        q_total = (v_next - v_prev_inj) / dt
        # unknowns: full cells that are not gates; Dirichlet: front (0) and gates (p_g)
        unk = full & ~is_gate
        u = np.flatnonzero(unk)
        uid = -np.ones(n, dtype=np.int64)
        uid[u] = np.arange(u.size)

        Sa, Sb = S[fa], S[fb]
        tot = Sa + Sb
        Sf = np.where(tot > 0, 2.0 * Sa * Sb / np.where(tot > 0, tot, 1.0), 0.0)
        ua, ub = unk[fa], unk[fb]
        uu = ua & ub
        diag = np.zeros(u.size)
        np.add.at(diag, uid[fa[uu]], Sf[uu])
        np.add.at(diag, uid[fb[uu]], Sf[uu])
        # faces from an unknown to a Dirichlet cell (front or gate) add to the diagonal
        ud = ua & (front[fb] | is_gate[fb])
        du = ub & (front[fa] | is_gate[fa])
        np.add.at(diag, uid[fa[ud]], Sf[ud])
        np.add.at(diag, uid[fb[du]], Sf[du])
        rhs0 = (V[u] - C[u]) / dt  # sources: push out the previous step's surplus
        rhs1 = np.zeros(u.size)  # response to a unit gate pressure
        ug = ua & is_gate[fb]
        gu = ub & is_gate[fa]
        np.add.at(rhs1, uid[fa[ug]], Sf[ug])
        np.add.at(rhs1, uid[fb[gu]], Sf[gu])
        if u.size:
            ia, ib = uid[fa[uu]], uid[fb[uu]]
            A = sp.coo_matrix(
                (
                    np.concatenate([-Sf[uu], -Sf[uu], diag]),
                    (
                        np.concatenate([ia, ib, np.arange(u.size)]),
                        np.concatenate([ib, ia, np.arange(u.size)]),
                    ),
                ),
                shape=(u.size, u.size),
            ).tocsc()
            lu = spla.splu(A)
            x0, x1 = lu.solve(rhs0), lu.solve(rhs1)
        else:
            x0 = x1 = np.zeros(0)
        # gate balance: melt leaving the gate cells over the step = what was delivered
        # (the gate cells stay exactly full, so they carry no surplus of their own)
        p0 = np.zeros(n)
        p1 = np.zeros(n)
        p0[u], p1[u] = x0, x1
        p1[is_gate] = 1.0
        gface_a = is_gate[fa] & ~is_gate[fb] & (unk[fb] | front[fb])
        gface_b = is_gate[fb] & ~is_gate[fa] & (unk[fa] | front[fa])
        out1 = float(
            np.sum(Sf[gface_a] * (1.0 - p1[fb[gface_a]]))
            + np.sum(Sf[gface_b] * (1.0 - p1[fa[gface_b]]))
        )
        out0 = float(
            np.sum(Sf[gface_a] * (0.0 - p0[fb[gface_a]]))
            + np.sum(Sf[gface_b] * (0.0 - p0[fa[gface_b]]))
        )
        if out1 <= 0:
            raise RuntimeError("the gate cells have no open face to the melt")
        p_g = (q_total - out0) / out1
        p = p0 + p_g * p1

        flux = Sf * (p[fa] - p[fb])  # a -> b
        inflow = np.zeros(n)
        np.add.at(inflow, fb, flux)
        np.add.at(inflow, fa, -flux)

        V_old = V.copy()
        V[u] = C[u]
        V[g] = C[g]
        V[fr] = V_old[fr] + dt * inflow[fr]
        newly = fr[V[fr] >= C[fr] * FULL_REL]
        if newly.size:
            rate = np.maximum(inflow[newly], 1e-300)
            frac = np.clip((C[newly] - V_old[newly]) / rate, 0.0, dt)
            t_arr[newly] = t + frac
            S[newly] = conductance(newly, t_arr[newly])
        t += dt
        if bps and abs(bps[0] - t) <= 1e-9 * max(abs(t), dt_max_s):
            t = bps[0]  # land on the switch, so the rate read after it is the new stage's
        v_prev_inj = v_next
        steps += 1

        # the next step's inflow scales with the delivery rate (a switch just passed)
        r_next = rate_ahead(t)
        scale = r_next / max(q_total, 1e-300) if q_total > 0 else 1.0
        dt = _next_step(C, V, m, fa, fb, fr, newly, inflow * scale, cfl, dt_max_s, r_next)

    return MarchFillResult(
        t_arr_s=np.where(m, t_arr, np.nan).reshape(ny, nx),
        pressure_end_Pa=np.where(m, p, 0.0).reshape(ny, nx),
        n_steps=steps,
        t_end_s=t,
        complete=complete,
        volume_injected_m3=v_prev_inj,
        volume_held_m3=float(V[m].sum()),
    )


def _next_step(C, V, m, fa, fb, fr, newly, inflow, cfl, dt_max_s, rate) -> float:
    """Length of the next step: no front cell of the *next* step should take more
    than ``cfl`` times its own capacity.

    A cell still at the front keeps its inflow. A cell that just joined the
    front is fed by the cells that just filled next to it; each of those passes
    its inflow on, split over the neighbours that are not yet full. Reading the
    step off the old front instead (the cells that just filled) let a thick cell
    set the step for the thin cell after it -- a ramp feeding a 0.35 mm land took
    several cells of melt in one step, and the surplus rode ahead of the front
    (a strip of uneven cells was stamped up to 7 % late).
    """
    n = C.size
    full = m & (V >= C * FULL_REL)
    est = np.zeros(n)
    keep = fr[~full[fr]]
    est[keep] = np.maximum(inflow[keep], 0.0)
    if newly.size:
        is_new = np.zeros(n, dtype=bool)
        is_new[newly] = True
        open_ = m & ~full
        # neighbours of each newly full cell that are still open
        cnt = np.zeros(n)
        np.add.at(cnt, fa[is_new[fa] & open_[fb]], 1.0)
        np.add.at(cnt, fb[is_new[fb] & open_[fa]], 1.0)
        share = np.where(cnt > 0, np.maximum(inflow, 0.0) / np.where(cnt > 0, cnt, 1.0), 0.0)
        a2b = is_new[fa] & open_[fb]
        b2a = is_new[fb] & open_[fa]
        np.add.at(est, fb[a2b], share[fa[a2b]])
        np.add.at(est, fa[b2a], share[fb[b2a]])
    fed = np.flatnonzero(m & ~full & (est > 0))
    if not fed.size:
        # no inflow to read (a step cut short at a switch): share the delivery rate
        # over the open cells next to the melt, as for the first step
        touch = np.zeros(n, dtype=bool)
        touch[fa[full[fb]]] = True
        touch[fb[full[fa]]] = True
        nxt = np.flatnonzero(m & ~full & touch)
        if not nxt.size:
            return float(dt_max_s)
        return max(min(cfl * float(np.min(C[nxt])) * nxt.size / max(rate, 1e-300), dt_max_s), 1e-12)
    # the whole capacity, not the room left: sizing on the room let nearly full
    # cells force tiny steps (5x the steps on the FG9 block for the same fill
    # pattern); a partly filled cell may overshoot by what it already holds, and
    # the surplus is pushed on in the next step (the volume stays conserved)
    return max(min(cfl * float(np.min(C[fed] / est[fed])), dt_max_s), 1e-12)
