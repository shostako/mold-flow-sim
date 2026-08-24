"""Tests for the skin-layer (Stefan/Neumann) extension to ``HeleShawSolver``.

The model adds wall-side freezing as

    s(t) = c_skin * sqrt(alpha * t)

and conducts only through the live core ``h_core = h - 2*s``. The fields
are coupled by fixed-point iteration; ``T_fill`` scales with the resistance
increase (constant-pressure proxy).
"""

from __future__ import annotations

import numpy as np
import pytest

from core import HeleShawSolver, MaterialDB, build_demo_geometry


@pytest.fixture(scope="module")
def pp() -> object:  # MaterialDB returns Material instances
    return MaterialDB()["PP"]


def _solve(geom, mat, **kwargs):
    solver = HeleShawSolver(
        geometry=geom,
        material=mat,
        injection_volume_flow_cm3s=20.0,
        **kwargs,
    )
    return solver.solve(num_frames=4)


def test_skin_off_matches_legacy_metadata(pp) -> None:
    """When the skin model is disabled, the result carries no skin fields and
    the metadata flags the model as off."""
    geom = build_demo_geometry(plate_thk_mm=2.0, cell_size_mm=1.5)
    res = _solve(geom, pp)
    assert res.skin_thickness_mm is None
    assert res.core_thickness_mm is None
    assert res.short_shot_mask is None
    assert res.metadata["skin_layer_enabled"] is False
    assert "skin_iterations" not in res.metadata


def test_skin_on_emits_skin_and_core_fields(pp) -> None:
    geom = build_demo_geometry(plate_thk_mm=2.0, cell_size_mm=1.5)
    res = _solve(geom, pp, skin_layer_enabled=True, skin_growth_constant=0.3)
    assert res.skin_thickness_mm is not None
    assert res.skin_thickness_mm.shape == geom.shape
    assert res.core_thickness_mm is not None
    assert res.short_shot_mask is not None
    # Inside the cavity, h_core + 2*s == h_open  (within a tiny floor offset)
    h_open = geom.thickness_mm[geom.mask]
    h_core = res.core_thickness_mm[geom.mask]
    s = res.skin_thickness_mm[geom.mask]
    np.testing.assert_allclose(h_core + 2.0 * s, h_open, atol=2e-3)


def test_skin_increases_fill_time(pp) -> None:
    """Skin layer ON should always inflate (or match) the baseline T_fill."""
    geom = build_demo_geometry(plate_thk_mm=2.0, cell_size_mm=1.5)
    base = _solve(geom, pp)
    skin = _solve(geom, pp, skin_layer_enabled=True, skin_growth_constant=0.5)
    assert skin.total_fill_time_s > base.total_fill_time_s
    assert skin.metadata["T_fill_inflation"] > 1.0


def test_skin_zero_constant_recovers_baseline(pp) -> None:
    """c_skin = 0 must recover the no-skin solve (within solver tolerance)."""
    geom = build_demo_geometry(plate_thk_mm=2.0, cell_size_mm=1.5)
    base = _solve(geom, pp)
    skin = _solve(geom, pp, skin_layer_enabled=True, skin_growth_constant=0.0)
    np.testing.assert_allclose(
        skin.total_fill_time_s,
        base.total_fill_time_s,
        rtol=1e-6,
    )
    # Skin field is all-zero inside the cavity
    assert float(np.nanmax(skin.skin_thickness_mm[geom.mask])) <= 1e-9


def test_skin_short_shot_detected_on_thin_plate(pp) -> None:
    """A 0.4 mm plate with a strong c_skin and slow flow seals near the gate
    while the front is still far away, cutting most of the plate off: the
    seal lands in ``short_shot_mask``, the missing melt in ``unfillable_mask``."""
    geom = build_demo_geometry(plate_thk_mm=0.4, cell_size_mm=1.5)
    res = _solve(
        geom,
        pp,
        skin_layer_enabled=True,
        skin_growth_constant=1.5,
        skin_max_iterations=3,
    )
    assert res.short_shot_mask is not None and res.short_shot_mask.any()
    assert res.unfillable_mask is not None
    cavity_cells = int(geom.mask.sum())
    # Most of the plate should fail to fill in this regime.
    assert int(res.unfillable_mask.sum()) / max(cavity_cells, 1) > 0.3
    assert res.metadata["filled_volume_fraction"] < 0.7
    assert not (res.short_shot_mask & res.unfillable_mask).any()


def test_skin_metadata_contains_iteration_info(pp) -> None:
    geom = build_demo_geometry(plate_thk_mm=2.0, cell_size_mm=1.5)
    res = _solve(
        geom,
        pp,
        skin_layer_enabled=True,
        skin_growth_constant=0.3,
        skin_max_iterations=4,
    )
    md = res.metadata
    assert md["skin_layer_enabled"] is True
    assert md["skin_growth_constant"] == pytest.approx(0.3)
    assert md["thermal_diffusivity_m2_s"] == pytest.approx(pp.thermal_diffusivity_m2_s)
    assert 1 <= md["skin_iterations"] <= 4
    assert md["T_fill_baseline_s"] > 0.0
    assert md["T_fill_inflation"] >= 1.0
    assert "short_shot_cells" in md
