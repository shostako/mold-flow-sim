"""Screw-side injection conditions: positions and speeds instead of a rate.

The solver's time axis is built on a volumetric rate ``Q``: the front has
swept ``Q * t`` of cavity by time ``t``, so a cell arrives when the volume
at or below its ``tau`` has been injected (``HeleShawSolver._arrival_time_field``).
``Q`` used to be typed in directly, and the default was the machine's *maximum*
rate off the spec sheet -- which is the rate at full screw speed, not the rate
of the shot actually being run. A condition set at a third of that speed came
out three times too fast, and the injection-speed slider (which only ever fed
the representative shear rate) did nothing about it.

What the machine is actually set to is a screw diameter and a set of
positions and speeds:

    Q = pi * D^2 / 4 * v        volumetric rate while the screw runs at v
    t = |x_start - x_end| / v   time to cross a stage
    V = pi * D^2 / 4 * L        volume that stage displaces

Note that the stage *volume* does not depend on the speed -- only on how far
the screw travels. So a multi-stage profile has fixed volume breakpoints and
a different rate on each one: the volume-to-time map is piecewise linear with
kinks at the switch positions, and reduces to ``t = V / Q`` for one stage.
That map is the whole point of this module; feeding it to the volume CDF is
what makes a slow first stage show up as a slow start in the animation.

**Screw speed is not the representative flow velocity.** ``v`` here is the
speed of the screw in the barrel. ``HeleShawSolver.injection_velocity_mms``
is the gap-mean velocity in the cavity that ``6V/h`` reads as a shear rate;
for a thin plate it is an order of magnitude larger. They stay separate
inputs.

Positions are machine positions: the screw moves from the metering position
*down* toward the V/P transfer position, so the sequence is strictly
decreasing. ``metering_position_mm`` is where the shot starts and the last
stage's ``end_position_mm`` is the V/P transfer point; everything past that
is packing, which this model does not represent.

This is a first-order theoretical displacement: no acceleration ramps, no
melt compressibility, no check-ring leakage, no machine response.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "InjectionProfile",
    "InjectionStage",
    "flow_rate_cm3s",
    "screw_area_mm2",
]


def screw_area_mm2(diameter_mm: float) -> float:
    """Cross-section the screw displaces [mm^2]."""
    d = float(diameter_mm)
    if not math.isfinite(d) or d <= 0:
        raise ValueError("screw diameter must be a positive finite value")
    return math.pi * d * d / 4.0


def flow_rate_cm3s(diameter_mm: float, velocity_mms: float) -> float:
    """Volumetric rate [cm^3/s] of a screw of ``diameter_mm`` running at ``velocity_mms``."""
    v = float(velocity_mms)
    if not math.isfinite(v) or v <= 0:
        raise ValueError("screw velocity must be a positive finite value")
    return screw_area_mm2(diameter_mm) * v / 1000.0


@dataclass(frozen=True)
class InjectionStage:
    """One velocity step of the injection profile.

    ``end_position_mm`` is the screw position where this stage hands over to
    the next one (the last stage hands over to packing at V/P). The stage's
    start is the previous stage's end, or the metering position for the first.
    """

    end_position_mm: float
    velocity_mms: float


@dataclass(frozen=True)
class InjectionProfile:
    """A screw diameter plus the position/velocity steps of one shot."""

    screw_diameter_mm: float
    metering_position_mm: float
    stages: tuple[InjectionStage, ...]

    def __post_init__(self) -> None:
        self.validate()

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------
    def validate(self) -> None:
        """Reject anything the machine could not run.

        Positions must step strictly down from the metering position: a stage
        that ends where it starts injects nothing in zero time (the map would
        have a vertical segment), and one that ends *above* its start would
        run the screw backwards.
        """
        d = float(self.screw_diameter_mm)
        if not math.isfinite(d) or d <= 0:
            raise ValueError("screw_diameter_mm must be a positive finite value")
        x0 = float(self.metering_position_mm)
        if not math.isfinite(x0) or x0 <= 0:
            raise ValueError("metering_position_mm must be a positive finite value")
        if not self.stages:
            raise ValueError("at least one injection stage is required")
        prev = x0
        for i, stage in enumerate(self.stages):
            x = float(stage.end_position_mm)
            v = float(stage.velocity_mms)
            if not math.isfinite(x) or x < 0:
                raise ValueError(f"stages[{i}].end_position_mm must be finite and >= 0")
            if x >= prev:
                label = "metering_position_mm" if i == 0 else f"stages[{i - 1}].end_position_mm"
                raise ValueError(
                    f"stages[{i}].end_position_mm ({x}) must be below {label} ({prev}) "
                    "-- the screw moves forward, so positions step down"
                )
            if not math.isfinite(v) or v <= 0:
                raise ValueError(f"stages[{i}].velocity_mms must be a positive finite value")
            prev = x

    # ------------------------------------------------------------------
    # derived quantities
    # ------------------------------------------------------------------
    @property
    def num_stages(self) -> int:
        return len(self.stages)

    @property
    def area_mm2(self) -> float:
        return screw_area_mm2(self.screw_diameter_mm)

    @property
    def vp_position_mm(self) -> float:
        """Screw position at V/P transfer (the last stage's end)."""
        return float(self.stages[-1].end_position_mm)

    def stage_starts_mm(self) -> tuple[float, ...]:
        starts = [float(self.metering_position_mm)]
        starts.extend(float(s.end_position_mm) for s in self.stages[:-1])
        return tuple(starts)

    def stage_lengths_mm(self) -> tuple[float, ...]:
        return tuple(
            start - float(s.end_position_mm)
            for start, s in zip(self.stage_starts_mm(), self.stages)
        )

    def stage_rates_cm3s(self) -> tuple[float, ...]:
        a = self.area_mm2
        return tuple(a * float(s.velocity_mms) / 1000.0 for s in self.stages)

    def stage_volumes_mm3(self) -> tuple[float, ...]:
        """Volume each stage displaces. Set by travel alone, not by speed."""
        a = self.area_mm2
        return tuple(a * length for length in self.stage_lengths_mm())

    def stage_times_s(self) -> tuple[float, ...]:
        return tuple(
            length / float(s.velocity_mms)
            for length, s in zip(self.stage_lengths_mm(), self.stages)
        )

    @property
    def total_volume_mm3(self) -> float:
        """Theoretical displacement from metering position to V/P [mm^3]."""
        return self.area_mm2 * (float(self.metering_position_mm) - self.vp_position_mm)

    @property
    def total_volume_cm3(self) -> float:
        return self.total_volume_mm3 / 1000.0

    @property
    def total_time_s(self) -> float:
        """Injection time from metering position to V/P [s]."""
        return float(sum(self.stage_times_s()))

    @property
    def mean_rate_cm3s(self) -> float:
        """Shot volume over shot time -- the single rate this profile averages to."""
        return self.total_volume_cm3 / self.total_time_s

    # ------------------------------------------------------------------
    # the volume -> time map
    # ------------------------------------------------------------------
    def _breakpoints(self) -> tuple[np.ndarray, np.ndarray]:
        vols = np.concatenate(([0.0], np.cumsum(np.asarray(self.stage_volumes_mm3(), float))))
        times = np.concatenate(([0.0], np.cumsum(np.asarray(self.stage_times_s(), float))))
        return vols, times

    def time_at_volume_mm3(self, volume_mm3):
        """Time [s] at which ``volume_mm3`` has been injected.

        Piecewise linear with a kink at every switch position. Volume past the
        V/P point keeps the last stage's rate: the cavity as drawn can hold
        more than the shot displaces, and the fill-time field still has to
        put a number on those cells. That extrapolation is what a short shot
        looks like before the short-shot model is asked about it -- it says
        "if the machine kept going at the last speed", not "this fills".
        """
        vols, times = self._breakpoints()
        v = np.asarray(volume_mm3, dtype=float)
        v_clipped = np.clip(v, 0.0, vols[-1])
        t = np.interp(v_clipped, vols, times)
        over = v > vols[-1]
        if np.any(over):
            last_rate_mm3s = self.stage_rates_cm3s()[-1] * 1000.0
            t = np.where(over, times[-1] + (v - vols[-1]) / last_rate_mm3s, t)
        return t if t.ndim else float(t)

    def volume_at_time_s(self, time_s):
        """Volume [mm^3] injected by ``time_s`` -- the inverse map."""
        vols, times = self._breakpoints()
        t = np.asarray(time_s, dtype=float)
        t_clipped = np.clip(t, 0.0, times[-1])
        v = np.interp(t_clipped, times, vols)
        over = t > times[-1]
        if np.any(over):
            last_rate_mm3s = self.stage_rates_cm3s()[-1] * 1000.0
            v = np.where(over, vols[-1] + (t - times[-1]) * last_rate_mm3s, v)
        return v if v.ndim else float(v)

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------
    def as_record(self) -> dict:
        """JSON-safe summary for ``settings.json`` and solver metadata."""
        return {
            "screw_diameter_mm": float(self.screw_diameter_mm),
            "metering_position_mm": float(self.metering_position_mm),
            "vp_position_mm": self.vp_position_mm,
            "stages": [
                {
                    "end_position_mm": float(s.end_position_mm),
                    "velocity_mms": float(s.velocity_mms),
                    "rate_cm3s": rate,
                    "time_s": t,
                    "volume_cm3": vol / 1000.0,
                }
                for s, rate, t, vol in zip(
                    self.stages,
                    self.stage_rates_cm3s(),
                    self.stage_times_s(),
                    self.stage_volumes_mm3(),
                )
            ],
            "total_volume_cm3": self.total_volume_cm3,
            "total_time_s": self.total_time_s,
            "mean_rate_cm3s": self.mean_rate_cm3s,
        }
