"""Tests for the absolute-stroke compression model.

The legacy compression model multiplies the cavity thickness by a constant
``compression_factor``. For stepped plates (e.g. lower-zone t=0.35 mm,
upper-zone t=0.50 mm) this is physically inaccurate: the mold shim adds a
fixed absolute distance to every target cell, so both zones should grow by
the same stroke and the step thickness must be preserved.

This module exercises:

* Backward compatibility: ``compression_stroke_mm=None`` is byte-for-byte
  equivalent to the legacy factor model.
* Additivity: in stroke mode every target cell grows by exactly the same
  stroke, preserving the step thickness on a stepped plate.
* ``stroke=0`` collapses to compression OFF.
* The ``effective_factor`` used in T_fill shortening is consistent between
  the two modes on a uniform plate (so a factor matching the stroke gives
  the same T_fill).
* Metadata exposes ``compression_stroke_mm`` / ``compression_mode``.
"""

from __future__ import annotations

import numpy as np

from core import FilmGateConfig, HeleShawSolver, MaterialDB, build_film_gate_geometry


def _stepped_cfg(**overrides) -> FilmGateConfig:
    base = dict(
        plate_w_mm=120.0,
        plate_h_mm=80.0,
        plate_thk_mm=0.50,  # fallback (ignored when split + lower/upper are set)
        runner_long_mm=80.0,
        runner_short_diameter_mm=12.0,
        runner_depth_mm=20.0,
        runner_thk_mm=4.0,
        runner_flat_depth_mm=8.0,
        runner_slope_depth_mm=12.0,
        valve_gate_diameter_mm=4.0,
        gate_width_mm=60.0,
        plate_split_height_mm=20.0,
        plate_lower_thk_mm=0.35,  # gate-side band
        plate_upper_thk_mm=0.50,  # far-side band
        cell_size_mm=1.0,
        pad_mm=5.0,
    )
    base.update(overrides)
    return FilmGateConfig(**base)


def _solver(g, **kwargs) -> HeleShawSolver:
    db = MaterialDB()
    base = dict(
        geometry=g,
        material=db["PP"],
        melt_temperature_K=503.15,
        mold_temperature_K=313.15,
        injection_velocity_mms=100.0,
        injection_volume_flow_cm3s=20.0,
    )
    base.update(kwargs)
    return HeleShawSolver(**base)


# --------------------------------------------------------------------------
# Backward compatibility
# --------------------------------------------------------------------------


def test_stroke_none_matches_legacy_factor_model() -> None:
    """When ``compression_stroke_mm`` is left at the default ``None``,
    ``_open_thickness_field`` must reproduce the exact factor-model output.
    """
    g = build_film_gate_geometry(_stepped_cfg())

    legacy = _solver(g, compression_molding=True, compression_factor=1.8)
    new_default = _solver(
        g,
        compression_molding=True,
        compression_factor=1.8,
        compression_stroke_mm=None,
    )

    np.testing.assert_allclose(
        legacy._open_thickness_field(),
        new_default._open_thickness_field(),
        rtol=1e-12,
    )


def test_stroke_none_metadata_reports_factor_mode() -> None:
    g = build_film_gate_geometry(_stepped_cfg())
    solver = _solver(g, compression_molding=True, compression_factor=1.8)
    result = solver.solve(num_frames=4)
    assert result.metadata["compression_mode"] == "factor"
    assert result.metadata["compression_stroke_mm"] is None


# --------------------------------------------------------------------------
# Stroke additivity / step preservation
# --------------------------------------------------------------------------


def test_stroke_preserves_step_thickness_on_stepped_plate() -> None:
    """On a t=0.35 / t=0.50 stepped plate, applying stroke=0.70 mm must
    grow the thin zone to 1.05 mm and the thick zone to 1.20 mm — the
    0.15 mm step is preserved (factor model would not preserve it).
    """
    g = build_film_gate_geometry(_stepped_cfg())
    solver = _solver(g, compression_molding=True, compression_stroke_mm=0.70)

    h_open = solver._open_thickness_field()
    cm = g.compression_mask
    assert cm is not None

    # Identify lower (t=0.35) and upper (t=0.50) bands by reading
    # geometry.thickness_mm at compression-mask cells.
    h_cast = g.thickness_mm
    lower = cm & g.mask & (np.isclose(h_cast, 0.35))
    upper = cm & g.mask & (np.isclose(h_cast, 0.50))

    assert lower.any(), "stepped cfg must produce some t=0.35 cells"
    assert upper.any(), "stepped cfg must produce some t=0.50 cells"

    np.testing.assert_allclose(h_open[lower], 1.05, rtol=1e-9)
    np.testing.assert_allclose(h_open[upper], 1.20, rtol=1e-9)


def test_stroke_uniform_addition_on_compression_cells() -> None:
    """Every compression-target cell grows by *exactly* the same stroke,
    regardless of its as-cast thickness."""
    g = build_film_gate_geometry(_stepped_cfg())
    solver_off = _solver(g)
    solver_on = _solver(g, compression_molding=True, compression_stroke_mm=0.70)

    h_off = solver_off._open_thickness_field()
    h_on = solver_on._open_thickness_field()

    cm = g.compression_mask
    assert cm is not None
    target = cm & g.mask
    np.testing.assert_allclose(h_on[target] - h_off[target], 0.70, rtol=1e-9)

    # Runner / gate / sprue cells must stay untouched.
    other = g.mask & ~cm
    np.testing.assert_allclose(h_on[other], h_off[other], rtol=1e-9)


def test_stroke_zero_matches_compression_off() -> None:
    g = build_film_gate_geometry(_stepped_cfg())
    solver_off = _solver(g)
    solver_zero = _solver(g, compression_molding=True, compression_stroke_mm=0.0)

    np.testing.assert_allclose(
        solver_off._open_thickness_field(),
        solver_zero._open_thickness_field(),
        rtol=1e-12,
    )


# --------------------------------------------------------------------------
# effective_factor consistency on a uniform plate
# --------------------------------------------------------------------------


def test_uniform_plate_stroke_factor_equivalence_for_T_fill() -> None:
    """On a uniform plate (lower = upper = plate_thk), choosing
    ``factor = (h + stroke) / h`` should yield the same T_fill in both
    modes. The ``effective_factor`` derivations are different but must
    coincide here, which guards against arithmetic drift between the two
    code paths.
    """
    # Uniform plate: split=0, lower=upper=None -> falls back to plate_thk_mm
    cfg = _stepped_cfg(
        plate_thk_mm=0.50,
        plate_split_height_mm=0.0,
        plate_lower_thk_mm=None,
        plate_upper_thk_mm=None,
    )
    g = build_film_gate_geometry(cfg)

    # All compression cells are 0.50 mm. Pick stroke=0.70 -> factor=2.4.
    stroke = 0.70
    h_plate = 0.50
    factor = (h_plate + stroke) / h_plate  # = 2.4

    solver_stroke = _solver(
        g,
        compression_molding=True,
        compression_stroke_mm=stroke,
        compression_fraction=0.7,
    )
    solver_factor = _solver(
        g,
        compression_molding=True,
        compression_factor=factor,
        compression_fraction=0.7,
    )

    r_stroke = solver_stroke.solve(num_frames=4)
    r_factor = solver_factor.solve(num_frames=4)

    # T_fill is the absolute time scaling; both modes must agree.
    np.testing.assert_allclose(
        r_stroke.total_fill_time_s,
        r_factor.total_fill_time_s,
        rtol=1e-6,
    )
    # tau field shape must also coincide (same conductance).
    np.testing.assert_allclose(
        np.nan_to_num(r_stroke.tau, nan=0.0),
        np.nan_to_num(r_factor.tau, nan=0.0),
        rtol=1e-6,
        atol=1e-6,
    )


# --------------------------------------------------------------------------
# Metadata surfacing
# --------------------------------------------------------------------------


def test_stroke_mode_metadata() -> None:
    g = build_film_gate_geometry(_stepped_cfg())
    solver = _solver(g, compression_molding=True, compression_stroke_mm=0.70)
    result = solver.solve(num_frames=4)
    assert result.metadata["compression_mode"] == "stroke"
    assert result.metadata["compression_stroke_mm"] == 0.70


# --------------------------------------------------------------------------
# Geometry helper
# --------------------------------------------------------------------------


def test_compression_area_mm2_matches_plate_body() -> None:
    """``Geometry.compression_area_mm2`` should report the planar area of
    the compression target zone (cell count × cell_area_mm2)."""
    g = build_film_gate_geometry(_stepped_cfg())
    cm = g.compression_mask
    assert cm is not None
    expected = float(np.sum(cm & g.mask)) * g.cell_size_mm**2
    assert g.compression_area_mm2() == expected


def test_compression_area_mm2_full_when_mask_none() -> None:
    """When ``compression_mask`` is ``None`` (legacy whole-cavity mode),
    the helper must return the full cavity area."""
    from core import Geometry

    mask = np.ones((4, 6), dtype=bool)
    thk = np.full(mask.shape, 1.0)
    g = Geometry(mask=mask, thickness_mm=thk, cell_size_mm=2.0)
    # 24 cells × 4 mm² each = 96 mm²
    assert g.compression_area_mm2() == 96.0
