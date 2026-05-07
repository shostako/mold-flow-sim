"""Cavity geometry definition.

The simulation domain is a 2D structured grid where each cell is either
inside the cavity (mask=True) or outside (mask=False). Each in-cavity
cell carries a thickness h [mm] (gap between mold halves). Gates are
point-like Dirichlet boundaries at tau=0.

This module provides:
- Geometry: container of mask, thickness map, gates, and cell size.
- build_demo_geometry: synthetic cavity (rectangular plate + runner + sprue).
- build_film_gate_geometry: parametric rectangular plate fed by a film/side
  gate runner whose top-down silhouette is an isosceles trapezoid with the
  short edge replaced by a half-circle. A circular valve gate (Dirichlet
  τ=0) sits at the half-circle center.
- geometry_from_image: extract cavity mask from an image (PNG/SVG raster).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore


@dataclass
class Geometry:
    mask: np.ndarray  # bool [Ny, Nx]; True=in cavity
    thickness_mm: np.ndarray  # float [Ny, Nx]; mm; valid only where mask
    cell_size_mm: float  # square cell, mm
    gates: list[tuple[int, int]] = field(default_factory=list)  # [(iy, ix), ...]
    label: str = "cavity"
    # Cells that are inflated by ``compression_factor`` while the compression
    # phase is open. ``None`` keeps the legacy behaviour where the whole
    # cavity expands (used by ``build_demo_geometry`` and
    # ``geometry_from_image``). Parametric builders that distinguish a
    # product body from runners/sprues set this to a per-cell bool array
    # (only the product body is True). Cells outside ``mask`` are ignored
    # regardless of the value here.
    compression_mask: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.mask.shape

    @property
    def ny(self) -> int:
        return self.mask.shape[0]

    @property
    def nx(self) -> int:
        return self.mask.shape[1]

    def volume_cm3(self) -> float:
        cell_area_mm2 = self.cell_size_mm**2
        vol_mm3 = float(np.sum(self.thickness_mm[self.mask]) * cell_area_mm2)
        return vol_mm3 / 1000.0

    def compression_volume_fraction(self) -> float:
        """Fraction of the cavity volume that participates in compression.

        Returns ``1.0`` when ``compression_mask`` is ``None`` (legacy mode
        where the whole cavity expands). Otherwise returns the volume of the
        ``compression_mask & mask`` cells divided by the total cavity volume.
        """
        cm = self.compression_mask
        if cm is None:
            return 1.0
        denom = float(np.sum(self.thickness_mm[self.mask]))
        if denom <= 0:
            return 0.0
        return float(np.sum(self.thickness_mm[self.mask & cm]) / denom)

    def add_gate(self, iy: int, ix: int) -> None:
        if not self.mask[iy, ix]:
            raise ValueError(f"gate ({iy},{ix}) is outside the cavity mask")
        self.gates.append((iy, ix))

    def gate_origin_mm(self) -> tuple[float, float]:
        """Return ``(x0, y0)`` in mm — the gate centroid used as the display
        origin in the preview and every result-time map. Falls back to the
        grid center / bottom (0, 0) corner when the geometry has no gates.
        """
        if not self.gates:
            return float(self.nx * self.cell_size_mm) / 2.0, 0.0
        gate_iys = np.fromiter((gy for gy, _ in self.gates), dtype=float)
        gate_ixs = np.fromiter((gx for _, gx in self.gates), dtype=float)
        x0 = float((float(gate_ixs.mean()) + 0.5) * self.cell_size_mm)
        y0 = float((float(gate_iys.mean()) + 0.5) * self.cell_size_mm)
        return x0, y0


def build_demo_geometry(
    plate_w_mm: float = 120.0,
    plate_h_mm: float = 80.0,
    plate_thk_mm: float = 2.0,
    runner_thk_mm: float = 4.0,
    sprue_thk_mm: float = 6.0,
    cell_size_mm: float = 1.0,
    gate_count: int = 1,
) -> Geometry:
    """Build a flat plate + central runner + sprue. The product part is
    the rectangular plate; the runner is a thin horizontal strip below
    feeding into one or more film gates; the sprue is a small square at
    the runner inlet.
    """
    pad = 10.0
    runner_h_mm = 6.0
    sprue_size_mm = 8.0

    total_w = plate_w_mm + 2 * pad
    total_h = plate_h_mm + runner_h_mm + sprue_size_mm + 2 * pad

    nx = int(round(total_w / cell_size_mm))
    ny = int(round(total_h / cell_size_mm))

    mask = np.zeros((ny, nx), dtype=bool)
    thk = np.zeros((ny, nx), dtype=float)

    # plate (product)
    py0 = int(round(pad / cell_size_mm))
    py1 = py0 + int(round(plate_h_mm / cell_size_mm))
    px0 = int(round(pad / cell_size_mm))
    px1 = px0 + int(round(plate_w_mm / cell_size_mm))
    mask[py0:py1, px0:px1] = True
    thk[py0:py1, px0:px1] = plate_thk_mm

    # runner (just below the plate, full plate width)
    ry0 = py1
    ry1 = ry0 + int(round(runner_h_mm / cell_size_mm))
    mask[ry0:ry1, px0:px1] = True
    thk[ry0:ry1, px0:px1] = runner_thk_mm

    # sprue (square, centered on runner)
    sy0 = ry1
    sy1 = sy0 + int(round(sprue_size_mm / cell_size_mm))
    cx_mm = pad + plate_w_mm / 2.0
    sx0 = int(round((cx_mm - sprue_size_mm / 2.0) / cell_size_mm))
    sx1 = sx0 + int(round(sprue_size_mm / cell_size_mm))
    mask[sy0:sy1, sx0:sx1] = True
    thk[sy0:sy1, sx0:sx1] = sprue_thk_mm

    geom = Geometry(
        mask=mask,
        thickness_mm=thk,
        cell_size_mm=cell_size_mm,
        label="demo_plate",
    )

    # gate(s): inject at sprue base (center bottom of sprue)
    sprue_center_iy = sy1 - 1
    sprue_center_ix = (sx0 + sx1) // 2
    if gate_count <= 1:
        geom.add_gate(sprue_center_iy, sprue_center_ix)
    else:
        # multiple gates spread along the runner-plate interface (film gating)
        gate_y = ry0 - 1  # last row of plate adjacent to runner
        # but we need gate inside cavity; ry0-1 is plate, fine.
        positions = np.linspace(px0 + 4, px1 - 5, gate_count, dtype=int)
        for gx in positions:
            geom.add_gate(int(gate_y), int(gx))

    return geom


@dataclass(frozen=True)
class FilmGateConfig:
    """Parameters for :func:`build_film_gate_geometry`.

    Coordinate convention (mm, with y pointing up, x pointing right)::

        y = 0
              y_circle_bottom  = pad_mm
              y_short_edge     = pad_mm + d/2          (= half-circle center;
                                                          short edge of the
                                                          trapezoid; valve
                                                          gate y-coordinate)
              y_flat_top       = y_short_edge + D_flat (= boundary line where
                                                          thickness starts the
                                                          slope toward the
                                                          plate)
              y_long_edge      = y_short_edge + D      (= long edge of the
                                                          trapezoid;
                                                          plate-runner junction)
              y_plate_top      = y_long_edge + Hp

    Constraints (validated at construction time):

    - ``runner_long_mm`` (L_long) must be ``≤ plate_w_mm`` and ``≥ runner_short_diameter_mm``.
    - ``gate_width_mm`` (W_gate) must be ``≤ runner_long_mm``.
    - ``valve_gate_diameter_mm`` (d_valve) must be ``≤ runner_short_diameter_mm``.
    - ``runner_flat_depth_mm + runner_slope_depth_mm`` must equal ``runner_depth_mm``.
    - ``0 ≤ plate_split_height_mm ≤ plate_h_mm`` (0 disables the split and
      the plate is uniform at ``plate_thk_mm``).

    Optional plate split (gate-side / far-side two-zone thickness):

    When ``plate_split_height_mm > 0`` the plate body is split at
    ``y = y_long_edge + plate_split_height_mm`` into

    - a **gate-side band** of thickness ``plate_lower_thk_mm`` (occupying
      the strip ``y_long_edge ≤ y < y_long_edge + plate_split_height_mm``)
    - a **far-side band** of thickness ``plate_upper_thk_mm`` (the rest
      of the plate up to ``y_plate_top``).

    The runner slope zone interpolates from ``runner_thk_mm`` at the
    ``D_flat`` boundary line down to ``plate_lower_thk_mm`` at the long
    edge so the gate-side plate stays continuous with the runner exit.
    ``plate_lower_thk_mm`` / ``plate_upper_thk_mm`` default to
    ``plate_thk_mm`` when ``None``.

    Optional flow balancer (▽-shaped local thinning, used in LGP-style
    film-gate molds to defeat the natural radial flow pattern from the
    valve gate). When ``balancer_enabled`` is ``True``, one or more
    nested inverted isosceles triangles are carved into the runner
    thickness map:

    - Apex (point) sits on the centerline at
      ``y_apex = y_short_edge + (balancer_base_distance_from_gate_mm
      − balancer_height_mm)``.
    - Base (segment) sits on the centerline at
      ``y_base = y_short_edge + balancer_base_distance_from_gate_mm``.
    - All stages share the same apex, base y-coordinate and height;
      only the base-edge width and target thickness vary between stages.

    Two equivalent ways to specify the balancer geometry:

    1. **Scalar form (legacy, single stage)**: set
       ``balancer_base_width_mm`` and ``balancer_target_thickness_mm``
       to positive values. The whole triangle is filled with a single
       thickness — exactly the original 1-stage behaviour.
    2. **Tuple form (1..5 nested stages, center → outer)**: set
       ``balancer_base_widths_mm`` and ``balancer_thicknesses_mm`` to
       tuples of equal length, indexed center → outer. Stage 1 (the
       centermost) carries the smallest width and smallest target
       thickness; each successive stage paints a wider but slightly
       thicker zone, and the inner stages overwrite the outer ones,
       so the resulting cavity profile is a step-down toward the
       centerline.

    When both forms are provided the tuple form wins; an empty tuple
    falls back to the scalar form. ``resolved_balancer_stages()``
    returns the canonical ``[(W_k, h_k), ...]`` list.

    Balancer constraints:

    - 1 ≤ stage count ≤ 5 (tuple form only; the scalar form is always
      a single stage).
    - All ``W_k`` and ``h_k`` strictly positive.
    - ``balancer_base_widths_mm`` non-decreasing in center→outer order.
    - ``balancer_thicknesses_mm`` non-decreasing in center→outer order
      (the centermost stage is the thinnest).
    - The outermost ``W_k`` must be ``≤ gate_width_mm``.
    - ``y_apex`` must clear the valve-gate disk
      (``y_apex ≥ y_short_edge + valve_gate_diameter_mm/2``).
    - ``y_base`` must not exceed the long edge
      (``balancer_base_distance_from_gate_mm ≤ runner_depth_mm``).
    """

    plate_w_mm: float
    plate_h_mm: float
    plate_thk_mm: float
    runner_long_mm: float  # L_long: long-edge length (plate-side)
    runner_short_diameter_mm: float  # d: short-edge half-circle diameter
    runner_depth_mm: float  # D: distance long-edge ↔ short-edge line
    runner_thk_mm: float  # h_runner: uniform flat-zone thickness
    runner_flat_depth_mm: float  # D_flat: flat-zone depth from short-edge line
    runner_slope_depth_mm: float  # D_slope: slope-zone depth (D_flat + D_slope = D)
    valve_gate_diameter_mm: float  # d_valve: valve-gate Dirichlet disk diameter
    gate_width_mm: float  # W_gate: plate-runner aperture width on long edge
    cell_size_mm: float = 1.0
    pad_mm: float = 5.0

    # ----- optional flow balancer (LGP-style local thinning) -----
    balancer_enabled: bool = False
    # Single-stage scalar form (legacy): leave at 0 / 0 when using the
    # multi-stage tuple form below.
    balancer_base_width_mm: float = 0.0  # W_bal: base-edge width on plate side
    balancer_height_mm: float = 0.0  # H_bal: apex ↔ base distance
    balancer_base_distance_from_gate_mm: float = 0.0  # base y-offset from y_short_edge
    balancer_target_thickness_mm: float = 0.0  # h_bal: cavity thickness inside ▽
    # Multi-stage tuple form (1..5 stages, center → outer). Empty tuples
    # fall back to the scalar form above; mixed forms are rejected.
    balancer_base_widths_mm: tuple[float, ...] = ()
    balancer_thicknesses_mm: tuple[float, ...] = ()

    # ----- optional gate-side / far-side plate split -----
    plate_split_height_mm: float = 0.0  # 0 disables the split (uniform plate)
    plate_lower_thk_mm: float | None = None  # gate-side band (y_long .. y_long+split)
    plate_upper_thk_mm: float | None = None  # far-side band (y_long+split .. y_plate_top)

    def resolved_balancer_stages(self) -> list[tuple[float, float]]:
        """Return the canonical balancer stage list ``[(W_k, h_k), ...]`` in
        center → outer order. Empty when the balancer is disabled.

        Tuple form takes precedence; otherwise the scalar form is wrapped
        as a single stage. Returns ``[]`` for incomplete inputs — actual
        validation lives in :meth:`validate`.
        """
        if not self.balancer_enabled:
            return []
        widths_t = self.balancer_base_widths_mm
        thicks_t = self.balancer_thicknesses_mm
        if widths_t and thicks_t:
            n = min(len(widths_t), len(thicks_t))
            return [(float(widths_t[k]), float(thicks_t[k])) for k in range(n)]
        if self.balancer_base_width_mm > 0 and self.balancer_target_thickness_mm > 0:
            return [
                (
                    float(self.balancer_base_width_mm),
                    float(self.balancer_target_thickness_mm),
                )
            ]
        return []

    def resolved_plate_zones(self) -> tuple[float, float, float]:
        """Return ``(split_height_mm, lower_thk_mm, upper_thk_mm)``.

        For uniform mode (``plate_split_height_mm == 0``) the result is
        ``(0.0, plate_thk_mm, plate_thk_mm)``. Otherwise ``None`` fields
        fall back to ``plate_thk_mm``.
        """
        if self.plate_split_height_mm > 0:
            lower = (
                self.plate_lower_thk_mm
                if self.plate_lower_thk_mm is not None
                else self.plate_thk_mm
            )
            upper = (
                self.plate_upper_thk_mm
                if self.plate_upper_thk_mm is not None
                else self.plate_thk_mm
            )
            return float(self.plate_split_height_mm), float(lower), float(upper)
        return 0.0, float(self.plate_thk_mm), float(self.plate_thk_mm)

    def validate(self) -> None:
        eps = 1e-6
        if self.runner_long_mm > self.plate_w_mm + eps:
            raise ValueError(
                f"runner_long_mm ({self.runner_long_mm}) must be ≤ plate_w_mm ({self.plate_w_mm})"
            )
        if self.runner_long_mm < self.runner_short_diameter_mm - eps:
            raise ValueError(
                f"runner_long_mm ({self.runner_long_mm}) must be ≥ "
                f"runner_short_diameter_mm ({self.runner_short_diameter_mm}); "
                f"inverted trapezoid is not supported"
            )
        if self.gate_width_mm > self.runner_long_mm + eps:
            raise ValueError(
                f"gate_width_mm ({self.gate_width_mm}) must be ≤ runner_long_mm "
                f"({self.runner_long_mm})"
            )
        if self.valve_gate_diameter_mm > self.runner_short_diameter_mm + eps:
            raise ValueError(
                f"valve_gate_diameter_mm ({self.valve_gate_diameter_mm}) must be ≤ "
                f"runner_short_diameter_mm ({self.runner_short_diameter_mm})"
            )
        if abs(self.runner_flat_depth_mm + self.runner_slope_depth_mm - self.runner_depth_mm) > eps:
            raise ValueError(
                f"runner_flat_depth_mm + runner_slope_depth_mm "
                f"({self.runner_flat_depth_mm} + {self.runner_slope_depth_mm}) "
                f"must equal runner_depth_mm ({self.runner_depth_mm})"
            )
        for name, val in (
            ("plate_w_mm", self.plate_w_mm),
            ("plate_h_mm", self.plate_h_mm),
            ("plate_thk_mm", self.plate_thk_mm),
            ("runner_long_mm", self.runner_long_mm),
            ("runner_short_diameter_mm", self.runner_short_diameter_mm),
            ("runner_depth_mm", self.runner_depth_mm),
            ("runner_thk_mm", self.runner_thk_mm),
            ("valve_gate_diameter_mm", self.valve_gate_diameter_mm),
            ("gate_width_mm", self.gate_width_mm),
            ("cell_size_mm", self.cell_size_mm),
        ):
            if val <= 0:
                raise ValueError(f"{name} must be positive (got {val})")
        if self.runner_flat_depth_mm < 0 or self.runner_slope_depth_mm < 0:
            raise ValueError("runner_flat_depth_mm and runner_slope_depth_mm must be ≥ 0")

        # plate split validation
        if self.plate_split_height_mm < 0:
            raise ValueError(f"plate_split_height_mm ({self.plate_split_height_mm}) must be ≥ 0")
        if self.plate_split_height_mm > self.plate_h_mm + eps:
            raise ValueError(
                f"plate_split_height_mm ({self.plate_split_height_mm}) must be ≤ "
                f"plate_h_mm ({self.plate_h_mm})"
            )
        if self.plate_lower_thk_mm is not None and self.plate_lower_thk_mm <= 0:
            raise ValueError(f"plate_lower_thk_mm ({self.plate_lower_thk_mm}) must be > 0 when set")
        if self.plate_upper_thk_mm is not None and self.plate_upper_thk_mm <= 0:
            raise ValueError(f"plate_upper_thk_mm ({self.plate_upper_thk_mm}) must be > 0 when set")

        if self.balancer_enabled:
            # shared height / base-position constraints (independent of stage form)
            if self.balancer_height_mm <= 0:
                raise ValueError("balancer_height_mm must be > 0 when balancer_enabled")
            if self.balancer_base_distance_from_gate_mm <= 0:
                raise ValueError(
                    "balancer_base_distance_from_gate_mm must be > 0 when balancer_enabled"
                )
            if self.balancer_base_distance_from_gate_mm > self.runner_depth_mm + eps:
                raise ValueError(
                    f"balancer_base_distance_from_gate_mm "
                    f"({self.balancer_base_distance_from_gate_mm}) must be ≤ "
                    f"runner_depth_mm ({self.runner_depth_mm}); "
                    f"the balancer base would extend past the long edge"
                )
            apex_offset = self.balancer_base_distance_from_gate_mm - self.balancer_height_mm
            valve_radius = self.valve_gate_diameter_mm / 2.0
            if apex_offset < valve_radius - eps:
                raise ValueError(
                    f"balancer apex y-offset ({apex_offset:.3f} mm from short edge) "
                    f"must be ≥ valve_gate_diameter/2 ({valve_radius:.3f} mm); "
                    f"reduce balancer_height_mm or increase "
                    f"balancer_base_distance_from_gate_mm so that the apex clears "
                    f"the valve-gate disk"
                )

            # form-specific stage validation
            widths_t = self.balancer_base_widths_mm
            thicks_t = self.balancer_thicknesses_mm
            tuple_form = bool(widths_t) or bool(thicks_t)

            if tuple_form:
                # at least one of the tuples is non-empty → tuple form takes over
                if len(widths_t) != len(thicks_t):
                    raise ValueError(
                        f"balancer_base_widths_mm and balancer_thicknesses_mm must "
                        f"have equal length (got {len(widths_t)} vs {len(thicks_t)})"
                    )
                if not (1 <= len(widths_t) <= 5):
                    raise ValueError(
                        f"balancer must have 1..5 stages "
                        f"(got {len(widths_t)} from balancer_base_widths_mm)"
                    )
                if any(W <= 0 for W in widths_t) or any(h <= 0 for h in thicks_t):
                    raise ValueError(
                        "every entry in balancer_base_widths_mm and "
                        "balancer_thicknesses_mm must be > 0"
                    )
                widths_list = [float(W) for W in widths_t]
                thicks_list = [float(h) for h in thicks_t]
                if any(
                    widths_list[k] - widths_list[k - 1] < -eps for k in range(1, len(widths_list))
                ):
                    raise ValueError(
                        "balancer_base_widths_mm must be non-decreasing in center→outer order"
                    )
                if any(
                    thicks_list[k] - thicks_list[k - 1] < -eps for k in range(1, len(thicks_list))
                ):
                    raise ValueError(
                        "balancer_thicknesses_mm must be non-decreasing in "
                        "center→outer order (stage 1 is the centermost / thinnest)"
                    )
                if widths_list[-1] > self.gate_width_mm + eps:
                    raise ValueError(
                        f"outermost balancer base width ({widths_list[-1]}) must be "
                        f"≤ gate_width_mm ({self.gate_width_mm})"
                    )
            else:
                # scalar (single-stage) form
                if self.balancer_target_thickness_mm <= 0:
                    raise ValueError(
                        f"balancer_target_thickness_mm must be > 0 when "
                        f"balancer_enabled (got {self.balancer_target_thickness_mm})"
                    )
                if self.balancer_base_width_mm <= 0:
                    raise ValueError("balancer_base_width_mm must be > 0 when balancer_enabled")
                if self.balancer_base_width_mm > self.gate_width_mm + eps:
                    raise ValueError(
                        f"balancer_base_width_mm ({self.balancer_base_width_mm}) "
                        f"must be ≤ gate_width_mm ({self.gate_width_mm})"
                    )


def build_film_gate_geometry(cfg: FilmGateConfig) -> Geometry:
    """Build a rectangular plate fed by a parametric film-gate runner.

    The runner top-down silhouette is an isosceles trapezoid (long edge on
    the plate side, short edge on the gate side) with the short edge
    replaced by a half-circle. A circular valve gate of diameter
    ``cfg.valve_gate_diameter_mm`` sits at the half-circle center and acts
    as the Dirichlet τ=0 boundary.

    Thickness profile (continuous in y):

    - half-circle and trapezoid flat zone (lower ``D_flat``): ``h_runner``
    - trapezoid slope zone (upper ``D_slope``): linear interpolation from
      ``h_runner`` (at the boundary line) to ``plate_lower_thk_mm`` (at the
      long edge, equal to ``plate_thk_mm`` in uniform mode)
    - plate body, gate-side band (height ``plate_split_height_mm``):
      ``plate_lower_thk_mm``
    - plate body, far-side band: ``plate_upper_thk_mm``

    In uniform mode (``plate_split_height_mm == 0``) both bands collapse
    to ``plate_thk_mm``.

    Plate-runner connection:

    - The long edge spans ``L_long`` cells, but only the central
      ``W_gate`` cells couple to the plate; the remainder of the
      junction row is forced ``mask=False`` (gate-land wall). This keeps
      the trapezoid silhouette intact while restricting plate inflow to
      the gate aperture.
    """
    cfg.validate()

    pad = cfg.pad_mm
    Wp = cfg.plate_w_mm
    Hp = cfg.plate_h_mm
    L = cfg.runner_long_mm
    d = cfg.runner_short_diameter_mm
    D = cfg.runner_depth_mm
    D_flat = cfg.runner_flat_depth_mm
    D_slope = cfg.runner_slope_depth_mm
    h_runner = cfg.runner_thk_mm
    split_h, h_plate_lower, h_plate_upper = cfg.resolved_plate_zones()
    W_gate = cfg.gate_width_mm
    d_valve = cfg.valve_gate_diameter_mm
    dx = cfg.cell_size_mm

    cx = pad + Wp / 2.0
    # y_circle_bottom = pad  (half-circle bottom; not directly used in mask logic)
    y_short = pad + d / 2.0  # short-edge line (= half-circle center y)
    y_long = y_short + D  # long-edge line (= plate bottom)
    y_plate_top = y_long + Hp

    total_w = 2 * pad + Wp
    total_h = 2 * pad + d / 2.0 + D + Hp
    nx = int(round(total_w / dx))
    ny = int(round(total_h / dx))

    # cell-center coordinates [mm]
    iy_idx, ix_idx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    yy = (iy_idx + 0.5) * dx
    xx = (ix_idx + 0.5) * dx

    # --- silhouette ---
    in_half_circle = (((xx - cx) ** 2 + (yy - y_short) ** 2) <= (d / 2.0) ** 2) & (yy <= y_short)

    # trapezoid: width(y) = d + (L - d) * (y - y_short) / D
    t_trap = np.clip((yy - y_short) / max(D, 1e-12), 0.0, 1.0)
    width_at_y = d + (L - d) * t_trap
    in_trapezoid = (yy >= y_short) & (yy <= y_long) & (np.abs(xx - cx) <= width_at_y / 2.0)

    in_plate = (yy >= y_long) & (yy <= y_plate_top) & (xx >= pad) & (xx <= pad + Wp)

    mask = in_half_circle | in_trapezoid | in_plate

    # --- gate land (close all but central W_gate on the plate-bottom row) ---
    plate_rows = np.where(in_plate.any(axis=1))[0]
    if plate_rows.size > 0:
        iy_plate_bottom = int(plate_rows[0])
        ix_close = np.abs(xx[iy_plate_bottom, :] - cx) > W_gate / 2.0
        mask[iy_plate_bottom, ix_close] = False

    # --- thickness ---
    thk = np.zeros_like(xx, dtype=float)
    thk[in_half_circle] = h_runner

    # trapezoid flat zone (y_short ≤ y ≤ y_short + D_flat)
    flat_zone = in_trapezoid & (yy <= y_short + D_flat)
    thk[flat_zone] = h_runner

    # trapezoid slope zone (y_short + D_flat < y ≤ y_long)
    # — interpolate from h_runner down to the gate-side plate thickness so
    #   the runner exit is continuous with the plate's gate-side band.
    slope_zone = in_trapezoid & (yy > y_short + D_flat)
    if D_slope > 1e-12:
        t_slope = np.clip((yy - (y_short + D_flat)) / D_slope, 0.0, 1.0)
    else:
        t_slope = np.ones_like(yy)
    thk_slope = h_runner + (h_plate_lower - h_runner) * t_slope
    thk[slope_zone] = thk_slope[slope_zone]

    # plate body — split into gate-side / far-side bands when split_h > 0
    thk[in_plate] = h_plate_lower
    if split_h > 0:
        upper_zone = in_plate & (yy >= y_long + split_h)
        thk[upper_zone] = h_plate_upper

    # --- optional flow balancer (1..5 nested ▽ stages) ---
    if cfg.balancer_enabled:
        H_bal = cfg.balancer_height_mm
        base_offset = cfg.balancer_base_distance_from_gate_mm

        y_apex = y_short + (base_offset - H_bal)
        y_base = y_short + base_offset

        # All stages share the same y-band; only the half-width vs y differs
        # (linear from 0 at apex to W_k/2 at base).
        in_balancer_y = (yy >= y_apex) & (yy <= y_base)
        with np.errstate(invalid="ignore"):
            t_y = np.clip((yy - y_apex) / max(H_bal, 1e-12), 0.0, 1.0)

        stages = cfg.resolved_balancer_stages()  # center → outer
        # Paint outer → inner so inner stages overwrite outer ones; the result
        # is a step-down toward the centerline (h_outer in the outer ring,
        # h_inner inside it, ..., h_1 in the centermost triangle).
        for W_k, h_k in sorted(stages, key=lambda s: -s[0]):
            half_w_k = 0.5 * W_k * t_y
            in_stage = in_balancer_y & (np.abs(xx - cx) <= half_w_k) & in_trapezoid
            thk[in_stage] = h_k

    # cells masked out by gate-land closure stay thk=0; that's fine because
    # mask=False excludes them from the solve.
    thk[~mask] = 0.0

    # --- valve gate (circular Dirichlet at half-circle center) ---
    in_valve = ((xx - cx) ** 2 + (yy - y_short) ** 2) <= (d_valve / 2.0) ** 2
    valve_iys, valve_ixs = np.where(in_valve & mask)

    # Compression mask: only the rectangular plate body inflates during the
    # ICM open phase. Runner / half-circle / valve-gate cells stay at their
    # original thickness. ``in_plate`` already excludes the gate-land row
    # closures because ``mask`` is intersected below.
    compression_mask = in_plate & mask

    geom = Geometry(
        mask=mask,
        thickness_mm=thk,
        cell_size_mm=dx,
        label="film_gate",
        compression_mask=compression_mask,
    )
    if valve_iys.size == 0:
        # Defensive: if d_valve is too small to cover any cell, snap to the
        # single cell nearest to the half-circle center.
        ic_y = int(np.argmin(np.abs(yy[:, 0] - y_short)))
        ic_x = int(np.argmin(np.abs(xx[0, :] - cx)))
        if mask[ic_y, ic_x]:
            geom.gates.append((ic_y, ic_x))
    else:
        for iy, ix in zip(valve_iys, valve_ixs, strict=True):
            geom.gates.append((int(iy), int(ix)))

    return geom


@dataclass(frozen=True)
class DirectGateConfig:
    """Parameters for :func:`build_direct_gate_geometry`.

    The cavity is a single rectangular plate with the gate sitting **inside**
    the plate. The Dirichlet τ=0 patch is a circular disk of diameter
    ``gate_diameter_mm`` placed on the plate's longitudinal centerline,
    ``gate_offset_mm`` away from the gate-side edge (measured inward toward
    the plate interior). No runner, no sprue strip — molten resin enters
    the cavity vertically through this patch.

    Coordinate convention (mm, with y pointing up, x pointing right)::

        y = pad_mm + plate_h_mm             ← far edge (反ゲート側)
        y = pad_mm + gate_offset_mm         ← gate disk center
        y = pad_mm                          ← gate-side edge

    The cavity occupies the rectangle ``[pad_mm, pad_mm + plate_w_mm] ×
    [pad_mm, pad_mm + plate_h_mm]`` with constant thickness ``plate_thk_mm``.

    Parameters:

    - ``plate_w_mm`` (Wp): plate width [mm]
    - ``plate_h_mm`` (Hp): plate height [mm], measured from the gate-side
      edge to the far edge
    - ``plate_thk_mm``: plate body thickness [mm]
    - ``gate_diameter_mm``: diameter of the Dirichlet gate disk [mm]
      (default 3.0)
    - ``gate_offset_mm``: distance from the gate-side edge to the gate disk
      center, measured inward [mm] (default 20.0)
    - ``cell_size_mm``: square mesh size [mm]
    - ``pad_mm``: padding around the cavity silhouette [mm]

    Constraints (validated at construction time):

    - All numeric parameters strictly positive.
    - ``gate_diameter_mm ≤ plate_w_mm`` (gate fits across the plate width).
    - ``gate_diameter_mm/2 ≤ gate_offset_mm`` (gate disk does not poke
      past the gate-side edge).
    - ``gate_offset_mm + gate_diameter_mm/2 ≤ plate_h_mm`` (gate disk does
      not poke past the far edge).
    """

    plate_w_mm: float
    plate_h_mm: float
    plate_thk_mm: float
    gate_diameter_mm: float = 3.0
    gate_offset_mm: float = 20.0
    cell_size_mm: float = 1.0
    pad_mm: float = 5.0

    def validate(self) -> None:
        eps = 1e-6
        for name, val in (
            ("plate_w_mm", self.plate_w_mm),
            ("plate_h_mm", self.plate_h_mm),
            ("plate_thk_mm", self.plate_thk_mm),
            ("gate_diameter_mm", self.gate_diameter_mm),
            ("gate_offset_mm", self.gate_offset_mm),
            ("cell_size_mm", self.cell_size_mm),
        ):
            if val <= 0:
                raise ValueError(f"{name} must be positive (got {val})")
        if self.gate_diameter_mm > self.plate_w_mm + eps:
            raise ValueError(
                f"gate_diameter_mm ({self.gate_diameter_mm}) must be ≤ "
                f"plate_w_mm ({self.plate_w_mm})"
            )
        r = self.gate_diameter_mm / 2.0
        if self.gate_offset_mm + r > self.plate_h_mm + eps:
            raise ValueError(
                f"gate disk would poke past the far edge: "
                f"gate_offset_mm ({self.gate_offset_mm}) + "
                f"gate_diameter_mm/2 ({r}) > plate_h_mm ({self.plate_h_mm})"
            )
        if self.gate_offset_mm < r - eps:
            raise ValueError(
                f"gate disk would poke past the gate-side edge: "
                f"gate_offset_mm ({self.gate_offset_mm}) < "
                f"gate_diameter_mm/2 ({r})"
            )


def build_direct_gate_geometry(cfg: DirectGateConfig) -> Geometry:
    """Build a rectangular plate with a direct gate placed inside the plate.

    The product silhouette is a single rectangle of size
    ``plate_w_mm × plate_h_mm``. A circular Dirichlet τ=0 disk of diameter
    ``gate_diameter_mm`` is placed on the plate centerline,
    ``gate_offset_mm`` inward from the gate-side edge — same coordinate
    convention as :class:`FilmGateConfig` (gate-side edge at the small-y
    side, far edge at the large-y side). The plate has uniform thickness
    ``plate_thk_mm``; the gate cells share the same thickness because the
    valve gate sits on the cavity surface itself, not on a separate sprue.

    Compression molding (when enabled at solver level) inflates the entire
    plate body — including the gate cells, since they belong to the product.
    """
    cfg.validate()

    pad = cfg.pad_mm
    Wp = cfg.plate_w_mm
    Hp = cfg.plate_h_mm
    h_plate = cfg.plate_thk_mm
    d_gate = cfg.gate_diameter_mm
    g_off = cfg.gate_offset_mm
    dx = cfg.cell_size_mm

    cx = pad + Wp / 2.0
    y_plate_bottom = pad  # gate-side edge
    y_plate_top = pad + Hp  # far edge
    y_gate_center = pad + g_off

    total_w = 2 * pad + Wp
    total_h = 2 * pad + Hp
    nx = int(round(total_w / dx))
    ny = int(round(total_h / dx))

    iy_idx, ix_idx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    yy = (iy_idx + 0.5) * dx
    xx = (ix_idx + 0.5) * dx

    # --- silhouette: just the plate, nothing else ---
    in_plate = (yy >= y_plate_bottom) & (yy <= y_plate_top) & (xx >= pad) & (xx <= pad + Wp)

    # Dirichlet gate disk lives inside the plate
    in_gate_disk = ((xx - cx) ** 2 + (yy - y_gate_center) ** 2) <= (d_gate / 2.0) ** 2

    mask = in_plate

    # --- thickness: uniform plate ---
    thk = np.zeros_like(xx, dtype=float)
    thk[in_plate] = h_plate
    thk[~mask] = 0.0

    # --- compression mask: the entire plate body inflates (the gate cells
    # are part of the product and therefore part of the compression zone) ---
    compression_mask = in_plate.copy()

    geom = Geometry(
        mask=mask,
        thickness_mm=thk,
        cell_size_mm=dx,
        label="direct_gate",
        compression_mask=compression_mask,
    )

    # --- Dirichlet τ=0 cells: the gate disk inside the plate ---
    gate_iys, gate_ixs = np.where(in_gate_disk & mask)
    if gate_iys.size == 0:
        # Defensive: snap to the cell nearest the requested gate center
        ic_y = int(np.argmin(np.abs(yy[:, 0] - y_gate_center)))
        ic_x = int(np.argmin(np.abs(xx[0, :] - cx)))
        if mask[ic_y, ic_x]:
            geom.gates.append((ic_y, ic_x))
    else:
        for iy, ix in zip(gate_iys, gate_ixs, strict=True):
            geom.gates.append((int(iy), int(ix)))

    return geom


def geometry_from_image(
    image_path: str | Path,
    cell_size_mm: float,
    plate_thk_mm: float = 2.0,
    threshold: int = 128,
    invert: bool = False,
) -> Geometry:
    """Build a Geometry from an image. Dark pixels are interpreted as cavity
    (set invert=True to swap). The image is downsampled / scaled to match
    cell_size_mm given the image's pixel-to-mm ratio is treated as 1px=1mm
    unless the user resizes externally. Thickness is uniform.
    """
    if Image is None:
        raise RuntimeError("Pillow is required to read images")
    img = Image.open(image_path).convert("L")
    arr = np.asarray(img)
    if invert:
        mask = arr >= threshold
    else:
        mask = arr < threshold
    thk = np.where(mask, plate_thk_mm, 0.0).astype(float)
    geom = Geometry(
        mask=mask,
        thickness_mm=thk,
        cell_size_mm=cell_size_mm,
        label=Path(image_path).stem,
    )
    return geom
