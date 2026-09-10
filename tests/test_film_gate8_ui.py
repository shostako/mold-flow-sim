"""AppTest wiring checks for the parametric Film gate 8 (T字/横ランナー+縦ランナー).

A T-shaped gate: the film land runs the full exit width, drops over a short
ramp into a full-width runner bar (the T's crossbar), and one centre stem of
the same depth runs from the bar to the valve well (the T's upright). Both
are ``sub_gates`` rectangles so the junction is square. The sliders default
to the ``hamoko_gate_T_20260910`` design study (the chairman's hand sketch),
and the derived quantities (ramp angle from the depth drop over the ramp
length, stem tip from the valve) are tied the way that spec ties them, so the
assembled spec must match it exactly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from core import GateProfileSpec, ProfilePlateConfig, build_profile_gate_geometry

APP = Path(__file__).resolve().parent.parent / "app.py"
FILM_GATE8_LABEL = "Film gate 8 (T字/横ランナー+縦ランナー)"
DX = 0.5

# 肉盗み in front of the vertical runner: flat plateau at the land depth that
# splits the flow -- base 5 wide at the ramp head (t=1, land side), apex on the
# centreline at the ramp foot (t=2, where the runner's resin arrives). In BOTH
# fans -- overlapping fans take the deeper value, so one alone is overwritten.
T_GATE_ISLAND = {
    "angle_deg": 0.0,
    "inner_line": [[1.0, 0.0], [2.0, 0.0]],
    "outer_line": [[1.0, 2.5], [2.0, 0.01]],
    "end_dist": 2.0,
    "floor_depth": 0.35,
}

T_GATE_SPEC = {
    "name": "hamoko_gate_T_20260910",
    "units": "mm",
    "symmetric": True,
    "gate_exit_width": 298.0,
    "land": {"depth": 0.35, "length": 1.0},
    "main_ramp": {"angle_deg": 58.78, "cap_depth": 2.0},
    "sub_gates": [
        {
            "inner_wall_line": [[1.0, 0.0], [4.0, 0.0]],
            "outer_wall_line": [[1.0, 149.0], [4.0, 149.0]],
            "tip_t": 4.0,
            "island": T_GATE_ISLAND,
        },
        {
            "inner_wall_line": [[1.0, 0.0], [23.0, 0.0]],
            "outer_wall_line": [[1.0, 2.5], [23.0, 2.5]],
            "tip_t": 23.0,
            "island": T_GATE_ISLAND,
        },
    ],
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
def film_gate8_run() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=240.0)
    at.run()
    at.radio(key="geom_source").set_value(FILM_GATE8_LABEL).run()
    at.radio(key="wall_model").set_value("none")
    at.button[0].click().run()
    assert not at.exception
    return at


def _recorded_spec(at: AppTest) -> dict:
    geom = at.session_state["mfs_settings"]["geometry"]
    assert geom["input"] == FILM_GATE8_LABEL
    return geom["gate_profile"]


def _film_gate8_app() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=240.0)
    at.run()
    at.radio(key="geom_source").set_value(FILM_GATE8_LABEL).run()
    at.radio(key="wall_model").set_value("none")
    return at


def _slider(at: AppTest, label_prefix: str):
    hits = [s for s in at.slider if str(s.label).startswith(label_prefix)]
    assert len(hits) == 1, [str(s.label) for s in at.slider]
    return hits[0]


def _cell(t: float, w: float, t_max: float = 27.5) -> tuple[int, int]:
    """Grid index of the cell whose centre is at (t, w); the request must hit
    a centre (t_max = 27.5 puts centres on k + 0.25 / k + 0.75 at 0.5 mm).
    ``t_max`` is the spec's: the well end, or the stem tip when the well is off."""
    y = PLATE.pad_mm + t_max - t
    x = PLATE.pad_mm + PLATE.plate_w_mm / 2.0 + w
    iy, ix = y / DX - 0.5, x / DX - 0.5
    assert abs(iy - round(iy)) < 1e-9 and abs(ix - round(ix)) < 1e-9, (t, w)
    return int(round(iy)), int(round(ix))


def test_default_sliders_reproduce_the_t_gate_spec(film_gate8_run):
    rec = _recorded_spec(film_gate8_run)
    expected = GateProfileSpec.from_dict(T_GATE_SPEC)
    got = GateProfileSpec.from_dict({**rec, "name": T_GATE_SPEC["name"]})
    assert got.symmetric is True and got.outer_wall_line is None and got.island is None
    assert got.runner is None
    assert got.gate_exit_width == expected.gate_exit_width
    assert got.land == expected.land
    assert got.main_ramp == expected.main_ramp
    assert len(got.sub_gates) == 2
    for fan, ref in zip(got.sub_gates, expected.sub_gates):
        assert fan.tip_t == ref.tip_t
        assert not fan.edge_channels
        assert fan.island.angle_deg == 0.0 and fan.island.floor_depth == 0.35
        assert fan.island.end_dist == ref.island.end_dist
        assert np.asarray(fan.island.inner_line) == pytest.approx(np.asarray(ref.island.inner_line))
        assert np.asarray(fan.island.outer_line) == pytest.approx(np.asarray(ref.island.outer_line))
        assert np.asarray(fan.inner_wall_line) == pytest.approx(np.asarray(ref.inner_wall_line))
        assert np.asarray(fan.outer_wall_line) == pytest.approx(np.asarray(ref.outer_wall_line))
    assert got.well.t_range == expected.well.t_range
    assert got.well.half_width == expected.well.half_width
    assert got.well.depth == expected.well.depth
    assert got.well.wall_angle_deg == expected.well.wall_angle_deg
    assert got.well.floor_t_range == pytest.approx(expected.well.floor_t_range, abs=0.05)
    assert got.valve == expected.valve


def test_default_geometry_matches_the_t_gate_spec_built_directly(film_gate8_run):
    geom = film_gate8_run.session_state["mfs_geom"]
    ref = build_profile_gate_geometry(GateProfileSpec.from_dict(T_GATE_SPEC), PLATE, DX)
    assert geom.cell_size_mm == DX
    assert geom.mask.shape == ref.mask.shape
    assert np.array_equal(geom.mask, ref.mask)
    assert np.array_equal(geom.thickness_mm[geom.mask], ref.thickness_mm[ref.mask])
    assert geom.gates == ref.gates


def test_the_pocket_is_a_t(film_gate8_run):
    """Bar across the full width to t=4, stem |w| ≤ 2.5 beyond it, steel
    elsewhere -- and the stem meets the bar at full width (square junction)."""
    geom = film_gate8_run.session_state["mfs_geom"]
    h = geom.thickness_mm
    # bar: land row, ramp row, floor rows, out at t=4
    assert h[_cell(0.25, 100.25)] == pytest.approx(0.35)
    assert h[_cell(1.75, 100.25)] == pytest.approx(
        0.35 + np.tan(np.radians(58.78)) * 0.75, abs=1e-6
    )
    assert h[_cell(3.75, 100.25)] == pytest.approx(2.0)
    assert h[_cell(3.75, -148.75)] == pytest.approx(2.0)
    assert not geom.mask[_cell(4.25, 100.25)]
    # 肉盗み: on the axis the ramp rows are the 0.35 plateau. The triangle is
    # wide at the land (half-width 1.875 at t=1.25: w=±1.75 in, ±2.25 out) and
    # narrows to the apex at the floor (half-width 0.625 at t=1.75: w=±0.25 in,
    # ±0.75 out) -- the point faces the vertical runner.
    tan_r = np.tan(np.radians(58.78))
    assert h[_cell(1.25, 0.25)] == pytest.approx(0.35)
    assert h[_cell(1.25, -1.75)] == pytest.approx(0.35)
    assert h[_cell(1.25, 2.25)] == pytest.approx(0.35 + tan_r * 0.25, abs=1e-6)
    assert h[_cell(1.75, 0.25)] == pytest.approx(0.35)
    assert h[_cell(1.75, 0.75)] == pytest.approx(0.35 + tan_r * 0.75, abs=1e-6)
    # stem: full 5 mm right behind the bar and half-way to the well, steel beside it
    for t in (4.25, 10.25):
        assert h[_cell(t, 2.25)] == pytest.approx(2.0)
        assert h[_cell(t, -2.25)] == pytest.approx(2.0)
        assert not geom.mask[_cell(t, 2.75)]
        assert not geom.mask[_cell(t, -2.75)]
    # well deeper than the stem at the valve
    assert h[_cell(21.25, 0.25)] == pytest.approx(4.5)


def test_well_can_be_switched_off_and_the_stem_still_reaches_the_valve():
    at = _film_gate8_app()
    at.checkbox(key="f8_well_on").set_value(False)
    at.button[0].click().run()
    assert not at.exception
    rec = _recorded_spec(at)
    assert rec["well"] is None
    geom = at.session_state["mfs_geom"]
    t_max = 23.0  # no well: the pocket ends at the stem tip = valve t + radius
    h = geom.thickness_mm
    assert h[_cell(21.25, 0.25, t_max)] == pytest.approx(2.0)
    assert h[_cell(22.75, 0.25, t_max)] == pytest.approx(2.0)
    assert h.shape[0] == int((PLATE.pad_mm * 2 + t_max + PLATE.plate_h_mm) / DX)
    assert geom.gates


def test_ramp_angle_and_stem_follow_the_sliders():
    """The ramp angle is derived from the drop over the ramp length; the stem
    tip from the valve; the bar end and stem width are dimensioned."""
    at = _film_gate8_app()
    _slider(at, "横ランナー深さ").set_value(2.5).run()
    _slider(at, "ランプ長").set_value(2.0).run()
    _slider(at, "横ランナー奥端").set_value(6.0).run()
    _slider(at, "縦ランナー幅").set_value(8.0).run()
    _slider(at, "バルブ位置").set_value(24.0).run()
    at.button[0].click().run()
    assert not at.exception
    rec = _recorded_spec(at)
    assert rec["main_ramp"] == {"angle_deg": 47.07, "cap_depth": 2.5}  # atan((2.5 − 0.35) / 2)
    bar, stem = rec["sub_gates"]
    # island follows the new ramp: base at land depth (t=1), apex where the
    # ramp reaches 2.5 = t 1 + 2 = 3; both fans carry the same island
    assert bar["island"] == stem["island"]
    assert bar["island"]["floor_depth"] == 0.35
    assert bar["island"]["inner_line"] == [[1.0, 0.0], [3.0, 0.0]]
    assert bar["island"]["outer_line"] == [[1.0, 2.5], [3.0, 0.01]]
    assert bar["island"]["end_dist"] == 3.0
    assert bar["tip_t"] == 6.0 and bar["outer_wall_line"] == [[1.0, 149.0], [6.0, 149.0]]
    assert stem["tip_t"] == 25.5 and stem["outer_wall_line"] == [[1.0, 4.0], [25.5, 4.0]]
    assert rec["valve"]["t"] == 24.0


def test_bar_end_and_valve_bounds_follow_the_ramp_and_the_bar():
    """The bar end cannot come before the ramp bottoms out; the valve cannot
    sit inside the bar (there the stem has no length)."""
    at = _film_gate8_app()
    _slider(at, "ランプ長").set_value(5.0).run()
    bar_end = _slider(at, "横ランナー奥端")
    assert bar_end.min == pytest.approx(6.1)
    assert bar_end.value >= bar_end.min
    _slider(at, "横ランナー奥端").set_value(12.0).run()
    valve = _slider(at, "バルブ位置")
    assert valve.min == pytest.approx(13.5)
    at.button[0].click().run()
    assert not at.exception
    rec = _recorded_spec(at)
    assert rec["sub_gates"][0]["tip_t"] == 12.0
    assert rec["valve"]["t"] >= 13.5


def test_stem_reaches_a_well_placed_beyond_the_valve():
    """The stem tip follows the valve, but a well past the valve must still
    be fed: otherwise it is an isolated cavity component and the solver
    rejects the geometry at analysis time (Codex P2 on PR #83)."""
    at = _film_gate8_app()
    _slider(at, "井戸開始 t").set_value(40.0).run()
    _slider(at, "井戸終端 t").set_value(50.0).run()
    _slider(at, "バルブ位置").set_value(10.0).run()
    at.button[0].click().run()
    assert not at.exception
    rec = _recorded_spec(at)
    assert rec["valve"]["t"] == 10.0 and rec["well"]["t_range"] == [40.0, 50.0]
    assert rec["sub_gates"][1]["tip_t"] == 44.5  # well start + half-width, not valve + radius
    assert "mfs_result" in at.session_state


def test_island_can_be_switched_off():
    at = _film_gate8_app()
    at.checkbox(key="f8_island_on").set_value(False)
    at.button[0].click().run()
    assert not at.exception
    rec = _recorded_spec(at)
    assert all(sg["island"] is None for sg in rec["sub_gates"])
    h = at.session_state["mfs_geom"].thickness_mm
    assert h[_cell(1.75, 0.25)] == pytest.approx(0.35 + np.tan(np.radians(58.78)) * 0.75, abs=1e-6)


def test_island_depths_and_width_follow_the_sliders():
    """Top depth 0.8 puts the base where the ramp reaches 0.8 (t = 1 + 0.45/1.65
    ≈ 1.2727), foot depth 1.5 puts the apex at t ≈ 1.697; the plateau is 0.8
    and the ramp stays untouched before the base and beyond the apex."""
    at = _film_gate8_app()
    _slider(at, "肉盗み 底辺幅").set_value(8.0).run()
    _slider(at, "肉盗み 天面深さ").set_value(0.8).run()
    _slider(at, "肉盗み 足の深さ").set_value(1.5).run()
    at.button[0].click().run()
    assert not at.exception
    isl = _recorded_spec(at)["sub_gates"][0]["island"]
    t_base, t_apex = 1.0 + 0.45 / 1.65, 1.0 + 1.15 / 1.65
    assert isl["floor_depth"] == 0.8
    assert np.asarray(isl["inner_line"]) == pytest.approx(
        np.asarray([[t_base, 0.0], [t_apex, 0.0]]), abs=1e-4
    )
    assert np.asarray(isl["outer_line"]) == pytest.approx(
        np.asarray([[t_base, 4.0], [t_apex, 0.01]]), abs=1e-4
    )
    assert isl["end_dist"] == pytest.approx(t_apex, abs=1e-4)
    # At 0.5 mm the ramp rows are t = 1.25 (before the base) and 1.75 (past
    # the apex): both must be the untouched ramp. The plateau itself falls
    # between the cell centres here; its cells are pinned at 0.1 mm in
    # tests/test_geometry_twin_fan.py.
    h = at.session_state["mfs_geom"].thickness_mm
    tan_r = np.tan(np.radians(58.78))
    assert h[_cell(1.75, 0.25)] == pytest.approx(0.35 + tan_r * 0.75, abs=1e-6)
    assert h[_cell(1.25, 0.25)] == pytest.approx(0.35 + tan_r * 0.25, abs=1e-6)


def test_island_sliders_keep_min_below_max_at_the_shallowest_bar():
    """bar depth min is land + 0.3 so top ∈ [land, bar − 0.2] and foot ∈
    [top + 0.1, bar] both keep a range; a min == max slider would raise and
    drop the rest of the sidebar."""
    at = _film_gate8_app()
    bar = _slider(at, "横ランナー深さ")
    assert bar.min == pytest.approx(0.65)
    bar.set_value(0.65).run()
    assert not at.exception
    top = _slider(at, "肉盗み 天面深さ")
    assert top.min < top.max
    top.set_value(top.max).run()
    foot = _slider(at, "肉盗み 足の深さ")
    assert foot.min < foot.max
    at.button[0].click().run()
    assert not at.exception


def test_island_that_the_mesh_cannot_resolve_is_an_error_not_a_silent_no_op():
    """At 1.0 mm the row centres sit on integer t and the splitter selects no
    cell (Codex P1 on PR #85). The build must refuse with a message, not
    solve the geometry without the restrictor; with the island off 1.0 mm is
    a legitimate (if coarse) mesh."""
    at = _film_gate8_app()
    _slider(at, "メッシュ粗さ").set_value(1.0).run()
    assert not at.exception
    errors = "\n".join(str(e.value) for e in at.error)
    assert "island" in errors and "zero cells" in errors
    assert "mfs_geom" not in at.session_state
    at.checkbox(key="f8_island_on").set_value(False).run()
    assert not at.exception
    assert not [e for e in at.error if "zero cells" in str(e.value)]
    at.button[0].click().run()
    assert not at.exception
    assert at.session_state["mfs_geom"].cell_size_mm == 1.0


def test_film_gate_8_sliders_do_not_leak_into_film_gate_1():
    at = _film_gate8_app()
    _slider(at, "製品幅").set_value(200.0).run()
    _slider(at, "ランド深さ").set_value(0.8).run()
    at.radio(key="geom_source").set_value("Film gate 1 (扇状/肉盗み1)").run()
    assert _slider(at, "製品幅").value == 300.0
    assert _slider(at, "ランド深さ").value == 0.35
