"""AppTest wiring checks for the parametric Film gate 5 (振り分け/L字ランナー).

The same twin mini fans as Film gate 4, but the runner is L-shaped: it runs
sideways from the valve at the valve's t and enters each fan tip from below,
perpendicular and centred (``hamoko_gate_furiwake_twin_mini_L_20260824``).
The centred entry keeps the flow path from the tip to both base corners of
the fan equal -- the straight runner of Film gate 4 grazes the fan's inner
wall and biases the fill toward the centre, which is what this variant
fixes. Every other dimension is identical to Film gate 4's defaults.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from core import GateProfileSpec, ProfilePlateConfig, build_profile_gate_geometry

APP = Path(__file__).resolve().parent.parent / "app.py"
FILM_GATE5_LABEL = "Film gate 5 (振り分け/L字ランナー)"

TWIN_MINI_L_SPEC = {
    "name": "hamoko_gate_furiwake_twin_mini_L_20260824",
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
    "runner": {
        "width": 8.0,
        "depth": 2.5,
        "path": [[21.5, 0.0], [21.5, 74.5], [14.0, 74.5]],
    },
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
def film_gate5_run() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=240.0)
    at.run()
    at.radio(key="geom_source").set_value(FILM_GATE5_LABEL).run()
    at.radio(key="wall_model").set_value("none")
    at.button[0].click().run()
    assert not at.exception
    return at


def _recorded_spec(at: AppTest) -> dict:
    geom = at.session_state["mfs_settings"]["geometry"]
    assert geom["input"] == FILM_GATE5_LABEL
    return geom["gate_profile"]


def _film_gate5_app() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=240.0)
    at.run()
    at.radio(key="geom_source").set_value(FILM_GATE5_LABEL).run()
    at.radio(key="wall_model").set_value("none")
    return at


def _slider(at: AppTest, label_prefix: str):
    hits = [s for s in at.slider if str(s.label).startswith(label_prefix)]
    assert len(hits) == 1, [str(s.label) for s in at.slider]
    return hits[0]


def test_default_sliders_reproduce_the_twin_mini_l_spec(film_gate5_run):
    rec = _recorded_spec(film_gate5_run)
    expected = GateProfileSpec.from_dict(TWIN_MINI_L_SPEC)
    got = GateProfileSpec.from_dict({**rec, "name": TWIN_MINI_L_SPEC["name"]})
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
    assert len(got.runner.path) == 3  # the L: valve -> sideways -> up into the tip
    assert np.asarray(got.runner.path) == pytest.approx(np.asarray(expected.runner.path))
    assert got.well.t_range == expected.well.t_range
    assert got.well.half_width == expected.well.half_width
    assert got.well.depth == expected.well.depth
    assert got.well.wall_angle_deg == expected.well.wall_angle_deg
    assert got.well.floor_t_range == pytest.approx(expected.well.floor_t_range, abs=0.05)
    assert got.valve == expected.valve


def test_default_geometry_matches_the_twin_mini_l_spec_built_directly(film_gate5_run):
    geom = film_gate5_run.session_state["mfs_geom"]
    ref = build_profile_gate_geometry(GateProfileSpec.from_dict(TWIN_MINI_L_SPEC), PLATE, 1.0)
    assert geom.mask.shape == ref.mask.shape
    assert np.array_equal(geom.mask, ref.mask)
    assert np.array_equal(geom.thickness_mm[geom.mask], ref.thickness_mm[ref.mask])
    assert geom.gates == ref.gates


def test_l_runner_path_follows_the_valve_and_the_fan_tip():
    """The L path is derived: its corner sits at (valve t, tip axis), so
    moving the valve moves the sideways trunk and moving the tip moves the
    vertical stub.

    The valve is set last: its slider floor follows the tip (Codex P2), and
    a widget whose bounds change is re-created by Streamlit with its default
    value -- setting the valve before the tip would silently discard it (the
    same trap CLAUDE.md records for dynamic labels, generalised to bounds).
    """
    at = _film_gate5_app()
    _slider(at, "扇先端 t").set_value(16.0).run()
    _slider(at, "扇先端の中心半幅").set_value(60.0).run()
    _slider(at, "バルブ位置").set_value(24.0).run()
    at.button[0].click().run()
    assert not at.exception
    rec = _recorded_spec(at)
    assert rec["runner"]["path"] == [[24.0, 0.0], [24.0, 60.0], [16.0, 60.0]]
    assert rec["sub_gates"][0]["tip_t"] == 16.0
    assert rec["valve"]["t"] == 24.0


def test_valve_on_the_tip_line_collapses_the_corner_not_the_build():
    """With the valve exactly on the tip line the L's corner coincides with
    its end -- a zero-length segment that ``validate()`` rejects. The path
    assembly must collapse the duplicate point instead of shipping it."""
    at = _film_gate5_app()
    _slider(at, "バルブ位置").set_value(14.0).run()  # == default 扇先端 t
    at.button[0].click().run()
    assert not at.exception
    assert not [str(e.value) for e in at.error]
    rec = _recorded_spec(at)
    assert rec["runner"]["path"] == [[14.0, 0.0], [14.0, 74.5]]
    assert "mfs_geom" in at.session_state


def test_runner_width_bound_holds_for_the_l_path():
    """The L path reaches the same ``中心半幅 + 幅/2`` as the straight one, so
    the shared width bound must keep the trunk inside the raster at the
    tip-axis slider's far end."""
    at = _film_gate5_app()
    axis = _slider(at, "扇先端の中心半幅")
    axis.set_value(axis.max).run()  # 149 − 1.0 = 148.0
    rw = _slider(at, "ランナー幅")
    assert rw.max == pytest.approx(14.0)  # pad(5) + Wp/2(150) − axis(148) = 7 each side
    rw.set_value(rw.max).run()
    at.button[0].click().run()
    assert not at.exception
    assert not [str(e.value) for e in at.error]
    assert "mfs_geom" in at.session_state


def test_film_gate_5_sliders_do_not_leak_into_film_gate_4():
    at = _film_gate5_app()
    _slider(at, "製品幅").set_value(200.0).run()
    _slider(at, "バルブ位置").set_value(30.0).run()
    at.radio(key="geom_source").set_value("Film gate 4 (振り分け/ミニ扇×2)").run()
    assert _slider(at, "製品幅").value == 300.0
    assert _slider(at, "バルブ位置").value == 21.5


def test_valve_position_cannot_go_below_the_fan_tip():
    """Codex P2: with the valve closer to the product than the fan tip, the
    sideways trunk would cut a deep stripe across the fan interiors and feed
    them mid-fan -- a materially different experiment from the advertised
    L-runner. The slider floor follows the tip; equality stays allowed (the
    corner collapses, covered above)."""
    at = _film_gate5_app()
    valve = _slider(at, "バルブ位置")
    assert valve.min == pytest.approx(14.0)  # == default 扇先端 t
    _slider(at, "扇先端 t").set_value(16.0).run()
    valve = _slider(at, "バルブ位置")
    assert valve.min == pytest.approx(16.0)
    # raising the tip past the current valve value must not crash the sidebar
    _slider(at, "扇先端 t").set_value(30.0).run()
    assert not at.exception
    valve = _slider(at, "バルブ位置")
    assert valve.min == pytest.approx(30.0)
    at.button[0].click().run()
    assert not at.exception
    assert not [str(e.value) for e in at.error]
    rec = _recorded_spec(at)
    assert rec["valve"]["t"] >= 30.0
