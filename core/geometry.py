"""Cavity geometry definition.

The simulation domain is a 2D structured grid where each cell is either
inside the cavity (mask=True) or outside (mask=False). Each in-cavity
cell carries a thickness h [mm] (gap between mold halves). Gates are
point-like Dirichlet boundaries at tau=0.

This module provides:
- Geometry: container of mask, thickness map, gates, and cell size.
- build_demo_geometry: synthetic cavity (rectangular plate + runner + sprue).
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
