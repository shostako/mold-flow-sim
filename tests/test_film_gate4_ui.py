"""AppTest wiring checks for the parametric Film gate 4 (振り分け/ミニ扇×2).

Two mirrored mini fans (``sub_gates``) fed by a runner from the valve well,
with a full cut-out (steel at the PL, a deformed rhombus) between them. The
sliders default to the ``hamoko_gate_furiwake_twin_mini_20260824`` design
study; the derived quantities (fan walls from the tip axis / tip width,
runner path from the valve and the tip) are tied the way that drawing ties
them, so the assembled spec must match it exactly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from core import GateProfileSpec, ProfilePlateConfig, build_profile_gate_geometry

APP = Path(__file__).resolve().parent.parent / "app.py"
FILM_GATE4_LABEL = "Film gate 4 (振り分け/ミニ扇×2)"

TWIN_MINI_SPEC = {
    "name": "hamoko_gate_furiwake_twin_mini_20260824",
    "units": "mm",
    "symmetric": True,
    "gate_exit_width": 298.0,
    "land": {"depth": 0.35, "length": 1.0},
    "main_ramp": {"angle_deg": 10.95, "cap_depth": 2.5},
    "sub_gates": [
        {
            "inner_wall_line": [[1.0, 0.0], [14.0, 70.5]],
            "outer_wall_line": [[1.0, 149.0], [14.0, 78.5]],
            "tip_t": 14.0,
            "island": {
                "angle_deg": 2.5,
                "inner_line": [[1.0, 48.2], [10.0, 69.5]],
                "outer_line": [[1.0, 100.8], [10.0, 79.5]],
                "end_dist": 10.0,
            },
        }
    ],
    "runner": {"width": 8.0, "depth": 2.5, "path": [[21.5, 0.0], [14.0, 74.5]]},
    "well": {
        "shape": "obround",
        "t_range": [15.5, 27.5],
        "half_width": 4.5,
        "depth": 4.5,
        "floor_t_range": [18.1, 24.9],
        "wall_angle_deg": 60,
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


@pytest.fixture(scope="module")
def film_gate4_run() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=240.0)
    at.run()
    at.radio(key="geom_source").set_value(FILM_GATE4_LABEL).run()
    at.radio(key="wall_model").set_value("none")
    at.button[0].click().run()
    assert not at.exception
    return at


def _recorded_spec(at: AppTest) -> dict:
    geom = at.session_state["mfs_settings"]["geometry"]
    assert geom["input"] == FILM_GATE4_LABEL
    return geom["gate_profile"]


def _film_gate4_app() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=240.0)
    at.run()
    at.radio(key="geom_source").set_value(FILM_GATE4_LABEL).run()
    at.radio(key="wall_model").set_value("none")
    return at


def _slider(at: AppTest, label_prefix: str):
    hits = [s for s in at.slider if str(s.label).startswith(label_prefix)]
    assert len(hits) == 1, [str(s.label) for s in at.slider]
    return hits[0]


def test_default_sliders_reproduce_the_twin_mini_spec(film_gate4_run):
    rec = _recorded_spec(film_gate4_run)
    expected = GateProfileSpec.from_dict(TWIN_MINI_SPEC)
    got = GateProfileSpec.from_dict({**rec, "name": TWIN_MINI_SPEC["name"]})
    assert got.symmetric is True and got.outer_wall_line is None and got.island is None
    assert got.gate_exit_width == expected.gate_exit_width
    assert got.land == expected.land
    assert got.main_ramp == expected.main_ramp
    assert len(got.sub_gates) == 1
    fan, ref = got.sub_gates[0], expected.sub_gates[0]
    assert fan.tip_t == ref.tip_t
    assert np.asarray(fan.inner_wall_line) == pytest.approx(np.asarray(ref.inner_wall_line))
    assert np.asarray(fan.outer_wall_line) == pytest.approx(np.asarray(ref.outer_wall_line))
    assert fan.island.angle_deg == ref.island.angle_deg
    assert fan.island.end_dist == ref.island.end_dist
    assert np.asarray(fan.island.inner_line) == pytest.approx(np.asarray(ref.island.inner_line))
    assert np.asarray(fan.island.outer_line) == pytest.approx(np.asarray(ref.island.outer_line))
    assert got.runner.width == expected.runner.width
    assert got.runner.depth == expected.runner.depth
    assert np.asarray(got.runner.path) == pytest.approx(np.asarray(expected.runner.path))
    assert got.well.t_range == expected.well.t_range
    assert got.well.half_width == expected.well.half_width
    assert got.well.depth == expected.well.depth
    assert got.well.wall_angle_deg == expected.well.wall_angle_deg
    assert got.well.floor_t_range == pytest.approx(expected.well.floor_t_range, abs=0.05)
    assert got.valve == expected.valve


def test_default_geometry_matches_the_twin_mini_spec_built_directly(film_gate4_run):
    geom = film_gate4_run.session_state["mfs_geom"]
    ref = build_profile_gate_geometry(GateProfileSpec.from_dict(TWIN_MINI_SPEC), PLATE, 1.0)
    assert geom.mask.shape == ref.mask.shape
    assert np.array_equal(geom.mask, ref.mask)
    assert np.array_equal(geom.thickness_mm[geom.mask], ref.thickness_mm[ref.mask])
    assert geom.gates == ref.gates


def test_the_centre_between_the_fans_is_a_hole(film_gate4_run):
    """The design intent: at mid-fan depth the cells on the valve axis are
    steel, while the fans on both sides are cavity."""
    geom = film_gate4_run.session_state["mfs_geom"]
    spec = GateProfileSpec.from_dict(TWIN_MINI_SPEC)
    iy = int((PLATE.pad_mm + spec.t_max() - 7.0) / 1.0)  # t ≈ 7 (between land 1 and tip 14)
    cx = int((PLATE.pad_mm + PLATE.plate_w_mm / 2.0) / 1.0)
    assert not geom.mask[iy, cx - 20 : cx + 20].any()
    # inner wall at t=7: 70.5·6/13 ≈ 32.5, outer 149 − 70.5·6/13 ≈ 116.5
    assert geom.mask[iy, cx + 40 : cx + 110].all()
    assert geom.mask[iy, cx - 110 : cx - 40].all()


def test_island_and_well_can_be_switched_off():
    at = _film_gate4_app()
    at.checkbox(key="f4_island_on").set_value(False)
    at.checkbox(key="f4_well_on").set_value(False)
    at.button[0].click().run()
    assert not at.exception
    rec = _recorded_spec(at)
    assert rec["sub_gates"][0]["island"] is None
    assert rec["well"] is None
    assert "mfs_geom" in at.session_state


def test_runner_path_follows_the_valve_and_the_fan_tip():
    """The runner is derived, not dimensioned: moving the valve moves its
    start, moving the tip moves its end."""
    at = _film_gate4_app()
    _slider(at, "バルブ位置").set_value(24.0).run()
    _slider(at, "扇先端 t").set_value(16.0).run()
    _slider(at, "扇先端の中心半幅").set_value(60.0).run()
    at.button[0].click().run()
    assert not at.exception
    rec = _recorded_spec(at)
    assert rec["runner"]["path"] == [[24.0, 0.0], [16.0, 60.0]]
    assert rec["sub_gates"][0]["tip_t"] == 16.0
    assert rec["sub_gates"][0]["inner_wall_line"][1] == [16.0, 56.0]  # axis − width/2
    assert rec["sub_gates"][0]["outer_wall_line"][1] == [16.0, 64.0]
    assert rec["valve"]["t"] == 24.0


def test_tip_width_is_bounded_so_the_inner_wall_stays_at_or_past_the_axis():
    """The inner wall at the tip is axis − width/2; a width above 2·axis would
    push it below w = 0, which validate() rejects. The slider must not offer
    it."""
    at = _film_gate4_app()
    _slider(at, "扇先端の中心半幅").set_value(3.0).run()
    tip_w = _slider(at, "扇先端幅")
    assert tip_w.max <= 6.0 + 1e-9
    assert tip_w.value <= tip_w.max
    at.button[0].click().run()
    assert not at.exception
    rec = _recorded_spec(at)
    assert rec["sub_gates"][0]["inner_wall_line"][1][1] >= 0.0


def test_film_gate_4_sliders_do_not_leak_into_film_gate_1():
    at = _film_gate4_app()
    _slider(at, "製品幅").set_value(200.0).run()
    _slider(at, "ランド深さ").set_value(0.8).run()
    at.radio(key="geom_source").set_value("Film gate 1 (扇状/肉盗み1)").run()
    assert _slider(at, "製品幅").value == 300.0
    assert _slider(at, "ランド深さ").value == 0.35
