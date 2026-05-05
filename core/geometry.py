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

    def add_gate(self, iy: int, ix: int) -> None:
        if not self.mask[iy, ix]:
            raise ValueError(f"gate ({iy},{ix}) is outside the cavity mask")
        self.gates.append((iy, ix))


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

    Optional flow balancer (▽-shaped local thinning, used in LGP-style
    film-gate molds to defeat the natural radial flow pattern from the
    valve gate). When ``balancer_enabled`` is ``True``, an inverted
    isosceles triangle is carved into the runner thickness map:

    - Apex (point) sits on the centerline at
      ``y_apex = y_short_edge + (balancer_base_distance_from_gate_mm
      − balancer_height_mm)``.
    - Base (segment) sits on the centerline at
      ``y_base = y_short_edge + balancer_base_distance_from_gate_mm``,
      with width ``balancer_base_width_mm``.
    - **Inside the triangle**, the cavity thickness is forced to
      ``balancer_target_thickness_mm`` (a constant; default = plate_thk
      means the cavity ceiling is flush with the plate top, i.e. the
      mold-side coring face is parallel to the plate plane).

    Balancer constraints:

    - ``balancer_base_width_mm`` ``≤ gate_width_mm``.
    - ``y_apex`` must clear the valve-gate disk
      (``y_apex ≥ y_short_edge + valve_gate_diameter_mm/2``).
    - ``y_base`` must not exceed the long edge
      (``balancer_base_distance_from_gate_mm ≤ runner_depth_mm``).
    - ``balancer_target_thickness_mm`` must be ``> 0``.
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
    balancer_base_width_mm: float = 0.0  # W_bal: base-edge width on plate side
    balancer_height_mm: float = 0.0  # H_bal: apex ↔ base distance
    balancer_base_distance_from_gate_mm: float = 0.0  # base y-offset from y_short_edge
    balancer_target_thickness_mm: float = 0.0  # h_bal: cavity thickness inside ▽

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

        if self.balancer_enabled:
            if self.balancer_target_thickness_mm <= 0:
                raise ValueError(
                    f"balancer_target_thickness_mm must be > 0 when balancer_enabled "
                    f"(got {self.balancer_target_thickness_mm})"
                )
            if self.balancer_base_width_mm <= 0 or self.balancer_height_mm <= 0:
                raise ValueError(
                    "balancer_base_width_mm and balancer_height_mm must be > 0 "
                    "when balancer_enabled"
                )
            if self.balancer_base_distance_from_gate_mm <= 0:
                raise ValueError(
                    "balancer_base_distance_from_gate_mm must be > 0 when balancer_enabled"
                )
            if self.balancer_base_width_mm > self.gate_width_mm + eps:
                raise ValueError(
                    f"balancer_base_width_mm ({self.balancer_base_width_mm}) must be ≤ "
                    f"gate_width_mm ({self.gate_width_mm})"
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
      ``h_runner`` (at the boundary line) to ``plate_thk_mm`` (at the long edge)
    - plate body: ``plate_thk_mm``

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
    h_plate = cfg.plate_thk_mm
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
    slope_zone = in_trapezoid & (yy > y_short + D_flat)
    if D_slope > 1e-12:
        t_slope = np.clip((yy - (y_short + D_flat)) / D_slope, 0.0, 1.0)
    else:
        t_slope = np.ones_like(yy)
    thk_slope = h_runner + (h_plate - h_runner) * t_slope
    thk[slope_zone] = thk_slope[slope_zone]

    thk[in_plate] = h_plate

    # --- optional flow balancer (▽ local thinning) ---
    if cfg.balancer_enabled:
        W_bal = cfg.balancer_base_width_mm
        H_bal = cfg.balancer_height_mm
        base_offset = cfg.balancer_base_distance_from_gate_mm
        h_bal = cfg.balancer_target_thickness_mm

        y_apex = y_short + (base_offset - H_bal)
        y_base = y_short + base_offset

        # Inverted triangle (▽): apex is a point at y=y_apex, x=cx.
        # Width at y is W_bal * (y - y_apex) / H_bal: 0 at apex, W_bal at base.
        in_balancer_y = (yy >= y_apex) & (yy <= y_base)
        with np.errstate(invalid="ignore"):
            half_width_at_y = 0.5 * W_bal * np.clip((yy - y_apex) / max(H_bal, 1e-12), 0.0, 1.0)
        in_balancer = in_balancer_y & (np.abs(xx - cx) <= half_width_at_y)

        # Apply only inside the trapezoid (the balancer is a runner-side feature).
        in_balancer = in_balancer & in_trapezoid
        thk[in_balancer] = h_bal

    # cells masked out by gate-land closure stay thk=0; that's fine because
    # mask=False excludes them from the solve.
    thk[~mask] = 0.0

    # --- valve gate (circular Dirichlet at half-circle center) ---
    in_valve = ((xx - cx) ** 2 + (yy - y_short) ** 2) <= (d_valve / 2.0) ** 2
    valve_iys, valve_ixs = np.where(in_valve & mask)

    geom = Geometry(
        mask=mask,
        thickness_mm=thk,
        cell_size_mm=dx,
        label="film_gate",
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
