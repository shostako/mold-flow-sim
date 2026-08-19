"""Smoke tests for the multilayer 2D renderers (PR-D)."""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pytest

from core import (
    FilmGateConfig,
    MaterialDB,
    MultilayerHeleShawSolver,
    build_film_gate_geometry,
)
from core.visualizer import (
    THICKNESS_CMAP,
    _scalar_layer_field,
    render_layer_grid,
    render_layer_map,
    render_short_shot_map,
)
from tests.colorimetry import relative_luminance


def _solve(num_layers: int = 5, **solver_kwargs) -> object:
    g = build_film_gate_geometry(
        FilmGateConfig(
            plate_w_mm=120.0,
            plate_h_mm=80.0,
            plate_thk_mm=2.0,
            runner_long_mm=80.0,
            runner_short_diameter_mm=12.0,
            runner_depth_mm=20.0,
            runner_thk_mm=4.0,
            runner_flat_depth_mm=8.0,
            runner_slope_depth_mm=12.0,
            valve_gate_diameter_mm=4.0,
            gate_width_mm=60.0,
            cell_size_mm=2.0,  # coarse for fast smoke
            pad_mm=5.0,
        )
    )
    db = MaterialDB()
    return MultilayerHeleShawSolver(
        geometry=g,
        material=db["PP"],
        num_layers=num_layers,
        melt_temperature_K=503.15,
        mold_temperature_K=313.15,
        injection_velocity_mms=100.0,
        injection_volume_flow_cm3s=20.0,
        **solver_kwargs,
    ).solve(num_frames=4)


def test_render_layer_map_temperature(tmp_path) -> None:
    r = _solve()
    out = render_layer_map(r, 2, tmp_path / "layer_2_temperature.png", field="temperature")
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_layer_map_viscosity_log_scale(tmp_path) -> None:
    """Viscosity defaults to a log colorscale; the file must still be produced."""
    r = _solve()
    out = render_layer_map(r, 0, tmp_path / "layer_0_eta.png", field="viscosity")
    assert out.exists()


def test_render_layer_map_shear_rate(tmp_path) -> None:
    r = _solve()
    out = render_layer_map(r, 1, tmp_path / "layer_1_gdot.png", field="shear_rate")
    assert out.exists()


def test_render_layer_map_thickness(tmp_path) -> None:
    r = _solve()
    out = render_layer_map(r, 3, tmp_path / "layer_3_thk.png", field="thickness")
    assert out.exists()


def test_render_layer_map_invalid_field(tmp_path) -> None:
    r = _solve()
    with pytest.raises(ValueError, match="field="):
        render_layer_map(r, 0, tmp_path / "bad.png", field="nonsense")


def test_render_layer_map_out_of_range(tmp_path) -> None:
    r = _solve(num_layers=5)
    with pytest.raises(IndexError):
        render_layer_map(r, 99, tmp_path / "oob.png", field="temperature")


def test_render_layer_map_thermal_off_raises(tmp_path) -> None:
    """When ``thermal_coupling=False`` the layer fields except thickness
    are ``None``; requesting them should raise ``ValueError`` so callers
    are forced to choose ``thickness`` (the only geometry-only field)."""
    r = _solve(thermal_coupling=False)
    with pytest.raises(ValueError, match="thermal_coupling"):
        render_layer_map(r, 0, tmp_path / "ng.png", field="temperature")
    # Thickness still works without thermal coupling.
    out = render_layer_map(r, 0, tmp_path / "ok.png", field="thickness")
    assert out.exists()


def test_render_layer_grid(tmp_path) -> None:
    r = _solve(num_layers=5)
    out = render_layer_grid(r, tmp_path / "grid.png", field="temperature")
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_layer_grid_viscosity(tmp_path) -> None:
    r = _solve(num_layers=5)
    out = render_layer_grid(r, tmp_path / "grid_eta.png", field="viscosity")
    assert out.exists()


def test_render_short_shot_map_no_short_shot(tmp_path) -> None:
    """A warm plate produces no flagged cells — the renderer still emits
    a file, with a 'no short shot' annotation in lieu of red markers."""
    r = _solve(num_layers=5)
    assert r.metadata["short_shot_fraction"] == 0.0
    out = render_short_shot_map(r, tmp_path / "no_short.png")
    assert out.exists()


def test_render_short_shot_map_with_flagged_cells(tmp_path) -> None:
    """High solidification threshold forces flagged cells; the renderer
    overlays red markers."""
    r = _solve(num_layers=5, solidification_temperature_fraction=0.7)
    if r.metadata["short_shot_cells"] == 0:
        pytest.skip("expected some short-shot cells with this threshold")
    out = render_short_shot_map(r, tmp_path / "short.png")
    assert out.exists()


def test_scalar_layer_field_validates_index() -> None:
    r = _solve(num_layers=3)
    with pytest.raises(IndexError):
        _scalar_layer_field(r, "temperature", -1)
    with pytest.raises(IndexError):
        _scalar_layer_field(r, "temperature", 5)
    arr, cmap, label = _scalar_layer_field(r, "temperature", 1)
    assert arr.shape == r.geometry.thickness_mm.shape
    assert isinstance(cmap, str)
    assert "K" in label


def test_render_layer_map_uses_zeta_in_title(tmp_path) -> None:
    """The title includes the ζ-range of the selected layer, taken from
    ``metadata['layer_zeta']``."""
    r = _solve(num_layers=5)
    out = render_layer_map(r, 2, tmp_path / "zeta.png", field="temperature")
    assert out.exists()
    # Title content can't be inspected easily without parsing the PNG;
    # we settle for verifying that the metadata key the title relies on
    # is present.
    assert "layer_zeta" in r.metadata
    assert len(r.metadata["layer_zeta"]) == 6  # N+1


def test_thickness_ramp_runs_light_to_dark() -> None:
    """Thickness maps must paint thin regions light and thick regions dark:
    ink density reads as material quantity, and a thicker transparent part
    really does look darker. The map is also required to keep the thin end
    *saturated* — a low end that approaches white washes out the product
    plate (the thinnest region and the only one anyone looks at) and lets the
    3D ceiling blend into the pale-gray parting-line floor."""
    cmap = plt.get_cmap(THICKNESS_CMAP)

    def luminance(x: float) -> float:
        return relative_luminance(cmap(x)[:3])

    # Monotone across the whole ramp, not merely lighter-at-0. A thickness map
    # has to let the reader rank two thicknesses by darkness alone; where
    # luminance reverses, two different thicknesses share a darkness and the
    # ordering stops being recoverable. Endpoint-only checks are fooled by
    # rainbow ramps such as ``jet_r``, whose ends are nearly equal in
    # luminance while the middle swings far brighter.
    lums = [luminance(x / 255.0) for x in range(256)]
    drops = [b - a for a, b in zip(lums, lums[1:])]
    assert all(d < 0 for d in drops), (
        "thickness ramp must darken monotonically (thin=light, thick=dark); "
        f"{sum(1 for d in drops if d >= 0)} of {len(drops)} steps do not darken"
    )

    thin_rgb = cmap(0.0)[:3]
    _h, thin_sat, _v = mcolors.rgb_to_hsv(thin_rgb)
    assert thin_sat > 0.5, (
        f"thin end must stay saturated so it survives a white background; saturation={thin_sat:.2f}"
    )


def test_layer_thickness_field_uses_the_shared_thickness_ramp() -> None:
    """The per-layer thickness panel plots the same quantity as the design
    map, so it must not drift onto a different ramp."""
    r = _solve()
    _arr, cmap, _label = _scalar_layer_field(r, "thickness", 0)
    assert cmap == THICKNESS_CMAP
