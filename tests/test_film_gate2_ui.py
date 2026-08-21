"""AppTest wiring checks for the parametric Film gate 2 (扇状/肉盗み2).

Film gate 1's sibling with a flat dam in the 肉盗み (``island.weld``): the
drawing is ``hamoko_gate_furiwake_weld_20260818``. What differs from Film
gate 1 and has to be carried by the defaults, not the shared sliders: the
outer wall starts at t=5, the well wall is 71.6° (so the floor is
``depth/tan(71.6°) = 1.5`` inside each end), and the dam runs from t=7 to the
肉盗み end at a residual depth of 0.1 mm from the PL. The dam's residual
depth is the parameter the user asked for — 0 means the steel touches the
PL and the band becomes a hole.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from core import GateProfileSpec, ProfilePlateConfig, build_profile_gate_geometry

APP = Path(__file__).resolve().parent.parent / "app.py"
FILM_GATE2_LABEL = "Film gate 2 (扇状/肉盗み2)"

HAMOKO_WELD_SPEC = {
    "name": "hamoko_gate_furiwake_weld_20260818",
    "units": "mm",
    "symmetric": True,
    "gate_exit_width": 298.0,
    "land": {"depth": 0.35, "length": 1.0},
    "main_ramp": {"angle_deg": 10.95, "cap_depth": 2.5},
    "island": {
        "angle_deg": 2.5,
        "boundary_line": [[0.0, 50.0], [17.0, 9.9]],
        "end_dist": 17.0,
        "weld": {"t_range": [7.0, 17.0], "depth": 0.1},
    },
    "outer_wall_line": [[5.0, 149.0], [23.28, 4.48]],
    "well": {
        "shape": "obround",
        "t_range": [15.5, 27.5],
        "half_width": 4.5,
        "depth": 4.5,
        "floor_t_range": [17.0, 26.0],
        "wall_angle_deg": 71.6,
    },
    "valve": {"t": 21.5, "w": 0.0, "orifice_diameter": 3.0},
}

PLATE = ProfilePlateConfig(
    plate_w_mm=300.0,
    plate_h_mm=50.0,
    plate_thk_mm=0.35,
    plate_split_height_mm=20.0,
    plate_lower_thk_mm=0.35,
    plate_upper_thk_mm=0.50,
)


def _film_gate2_app() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=240.0)
    at.run()
    at.radio(key="geom_source").set_value(FILM_GATE2_LABEL).run()
    at.radio(key="wall_model").set_value("none")
    return at


@pytest.fixture(scope="module")
def film_gate2_run() -> AppTest:
    at = _film_gate2_app()
    at.button[0].click().run()
    assert not at.exception
    return at


def _recorded_spec(at: AppTest) -> dict:
    geom = at.session_state["mfs_settings"]["geometry"]
    assert geom["input"] == FILM_GATE2_LABEL
    return geom["gate_profile"]


def _slider(at: AppTest, label_prefix: str):
    hits = [s for s in at.slider if str(s.label).startswith(label_prefix)]
    assert len(hits) == 1, [str(s.label) for s in at.slider]
    return hits[0]


def test_default_sliders_reproduce_the_weld_spec(film_gate2_run):
    rec = _recorded_spec(film_gate2_run)
    expected = GateProfileSpec.from_dict(HAMOKO_WELD_SPEC)
    got = GateProfileSpec.from_dict({**rec, "name": HAMOKO_WELD_SPEC["name"]})
    assert got.symmetric is True
    assert got.gate_exit_width == expected.gate_exit_width
    assert got.land == expected.land
    assert got.main_ramp == expected.main_ramp
    assert got.island.angle_deg == expected.island.angle_deg
    assert got.island.end_dist == expected.island.end_dist
    # The drawing's boundary line starts at t=0; the UI pins it at t=land
    # length. Same line: compare where both are defined.
    (t1, w1), (t2, w2) = expected.island.boundary_line
    (g1, gw1), (g2, gw2) = got.island.boundary_line
    assert g1 == 1.0 and g2 == t2 and gw2 == w2
    assert gw1 == pytest.approx(w1 + (w2 - w1) * (g1 - t1) / (t2 - t1), abs=0.01)
    assert got.island.weld.t_range == expected.island.weld.t_range
    assert got.island.weld.depth == expected.island.weld.depth
    assert np.asarray(got.outer_wall_line) == pytest.approx(np.asarray(expected.outer_wall_line))
    assert got.well.t_range == expected.well.t_range
    assert got.well.half_width == expected.well.half_width
    assert got.well.depth == expected.well.depth
    assert got.well.wall_angle_deg == expected.well.wall_angle_deg
    assert got.well.floor_t_range == pytest.approx(expected.well.floor_t_range, abs=0.05)
    assert got.valve == expected.valve  # 21.5 = the well centre


def test_default_geometry_matches_the_weld_spec_built_directly(film_gate2_run):
    geom = film_gate2_run.session_state["mfs_geom"]
    ref = build_profile_gate_geometry(GateProfileSpec.from_dict(HAMOKO_WELD_SPEC), PLATE, 1.0)
    assert np.array_equal(geom.mask, ref.mask)
    assert np.array_equal(geom.thickness_mm[geom.mask], ref.thickness_mm[ref.mask])
    assert geom.gates == ref.gates


def test_the_dam_band_is_flat_at_the_residual_depth(film_gate2_run):
    """The 肉盗み between t=7 and t=17 reads 0.1 mm everywhere outside the well."""
    geom = film_gate2_run.session_state["mfs_geom"]
    spec = GateProfileSpec.from_dict(HAMOKO_WELD_SPEC)
    ny, nx = geom.mask.shape
    iy, ix = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    t = PLATE.pad_mm + spec.t_max() - (iy + 0.5)
    wa = np.abs((ix + 0.5) - (PLATE.pad_mm + 150.0))
    band = geom.mask & (t > 7.5) & (t < 15.0) & (wa < 8.0)  # clear of the well (t ≥ 15.5)
    assert band.any()
    np.testing.assert_allclose(geom.thickness_mm[band], 0.1)


def test_dam_depth_zero_cuts_the_band_out_of_the_cavity():
    """PL に接する = 完全な肉抜き空洞: the band leaves the mask and the rest of the
    pocket is untouched; the run still completes (flow goes around)."""
    at = _film_gate2_app()
    at.checkbox(key="two_phase_on").set_value(False)
    _slider(at, "水平部の PL からの距離").set_value(0.0).run()
    at.button[0].click().run()
    assert not at.exception
    geom = at.session_state["mfs_geom"]
    rec = _recorded_spec(at)
    assert rec["island"]["weld"]["depth"] == 0.0
    spec = GateProfileSpec.from_dict({**rec, "name": "x"})
    ref = build_profile_gate_geometry(spec, PLATE, 1.0)
    assert np.array_equal(geom.mask, ref.mask)
    full = build_profile_gate_geometry(GateProfileSpec.from_dict(HAMOKO_WELD_SPEC), PLATE, 1.0)
    lost = full.mask & ~geom.mask
    assert lost.sum() > 50
    assert (geom.thickness_mm[geom.mask] > 0).all()
    assert np.isfinite(at.session_state["mfs_result"].fill_time_s[geom.mask]).all()


def test_dam_off_drops_the_weld_section():
    at = _film_gate2_app()
    at.checkbox(key="two_phase_on").set_value(False)
    at.checkbox(key="f2_weld_on").set_value(False).run()
    at.button[0].click().run()
    assert not at.exception
    assert _recorded_spec(at)["island"].get("weld") is None  # asdict keeps the key as null


def test_dam_bounds_follow_the_land_and_the_island_end():
    at = _film_gate2_app()
    start = _slider(at, "水平部開始")
    assert start.min == pytest.approx(1.0)
    assert start.max == pytest.approx(16.5)
    depth = _slider(at, "水平部の PL からの距離")
    assert depth.min == 0.0 and depth.max == pytest.approx(0.35)
    _slider(at, "肉盗み終端").set_value(12.0).run()
    assert _slider(at, "水平部開始").max == pytest.approx(11.5)
    assert _recorded_spec_after_run(at)["island"]["weld"]["t_range"] == [7.0, 12.0]


def _recorded_spec_after_run(at: AppTest) -> dict:
    at.checkbox(key="two_phase_on").set_value(False)
    at.button[0].click().run()
    assert not at.exception
    return _recorded_spec(at)


def test_well_depth_cap_uses_this_drawings_wall_angle():
    """71.6°, not Film gate 1's 60°: half-width 4.5 → cap 13.5 (clamped to 15)."""
    at = _film_gate2_app()
    _slider(at, "井戸半幅").set_value(1.0).run()
    assert _slider(at, "井戸深さ").max == pytest.approx(3.0)  # 1.0 · tan(71.6°) = 3.0
