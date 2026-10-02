"""AppTest wiring checks for the parametric Film gate 15 (扇状/ランド長可変).

The 2026/10/02 proposal A3 (hamoko_gate_furiwake_landhanger_center5_20261002,
no CAD model): Film gate 14's kai02 frame -- outer wall, well, valve, the
11.06° ramp to 2.5 -- without the graded ends, the closure and the 肉盗み, and
with the land 0.35 deep but longer towards the centre:
L(w) = 1 + 4·(1 − |w|/149)², the ramp starting where it ends.

What the tests pin: the defaults reproduce the spec (bit-identical geometry),
against the same pocket without the profile only the ramp in front of the
centre gets shallower (the land row and the silhouette stay), the sliders
reach the spec, and none of it leaks into Film gate 14.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from core import GateProfileSpec, LandProfileSpec, ProfilePlateConfig, build_profile_gate_geometry

APP = Path(__file__).resolve().parent.parent / "app.py"
FILM_GATE15_LABEL = "Film gate 15 (扇状/ランド長可変)"
FILM_GATE14_LABEL = "Film gate 14 (扇状/斜面角度徐変3)"

CENTER_KEY = "f15_中央のランド長 [mm] (> ランド長さ)"
POWER_KEY = "f15_ランド長の形の指数（2 = 放物線）"

RAMP_DEG = math.degrees(math.atan(2.15 / 11.0))
TAN = 2.15 / 11.0

LANDHANGER_SPEC = {
    "name": "hamoko_gate_furiwake_landhanger_center5_20261002",
    "units": "mm",
    "symmetric": True,
    "gate_exit_width": 298.0,
    "land": {"depth": 0.35, "length": 1.0, "profile": {"center_length": 5.0, "power": 2.0}},
    "main_ramp": {"angle_deg": RAMP_DEG, "cap_depth": 2.5},
    "outer_wall_line": [[15.736241, 149.0], [23.309724, 4.489329]],
    "well": {
        "shape": "obround",
        "t_range": [15.5, 27.5],
        "half_width": 4.5,
        "depth": 4.5,
        "floor_t_range": [18.098076, 24.901924],
        "wall_angle_deg": 60.0,
    },
    "valve": {"t": 21.5, "w": 0.0, "orifice_diameter": 3.0},
}
PLAIN_SPEC = {**LANDHANGER_SPEC, "land": {"depth": 0.35, "length": 1.0}}

PLATE = ProfilePlateConfig(
    plate_w_mm=300.0,
    plate_h_mm=50.0,
    plate_thk_mm=0.35,
    plate_split_height_mm=20.0,
    plate_lower_thk_mm=0.35,
    plate_upper_thk_mm=0.50,
)


def _app() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=240.0)
    at.run()
    at.radio(key="geom_source").set_value(FILM_GATE15_LABEL).run()
    at.radio(key="wall_model").set_value("none")
    return at


@pytest.fixture(scope="module")
def film_gate15_run() -> AppTest:
    at = _app()
    at.button[0].click().run()
    assert not at.exception
    return at


def _recorded_spec(at: AppTest, label: str = FILM_GATE15_LABEL) -> dict:
    geom = at.session_state["mfs_settings"]["geometry"]
    assert geom["input"] == label
    return geom["gate_profile"]


def _recorded_spec_after_run(at: AppTest, label: str = FILM_GATE15_LABEL) -> dict:
    at.checkbox(key="two_phase_on").set_value(False)
    at.button[0].click().run()
    assert not at.exception
    return _recorded_spec(at, label)


def _build(d: dict, dx: float = 1.0):
    return build_profile_gate_geometry(GateProfileSpec.from_dict(d), PLATE, dx)


def _tw(geom, spec: GateProfileSpec):
    iy, ix = np.indices(geom.shape)
    pad = ProfilePlateConfig().pad_mm
    return (pad + spec.t_max()) - (iy + 0.5), np.abs((ix + 0.5) - (pad + PLATE.plate_w_mm / 2.0))


def test_default_sliders_reproduce_the_proposal_spec(film_gate15_run):
    rec = _recorded_spec(film_gate15_run)
    expected = GateProfileSpec.from_dict(LANDHANGER_SPEC)
    got = GateProfileSpec.from_dict({**rec, "name": expected.name})
    assert got.land == expected.land
    assert got.land.profile == LandProfileSpec(center_length=5.0, power=2.0)
    assert got.island is None and got.ramp_ends is None and got.land.closed_line is None
    assert got.outer_wall_corner_radius is None
    assert got.main_ramp.angle_deg == pytest.approx(RAMP_DEG, abs=1e-12)
    assert np.asarray(got.outer_wall_line) == pytest.approx(np.asarray(expected.outer_wall_line))
    assert got.valve == expected.valve
    assert got.edge_channels == () and got.ramp_cut is None and got.land_ends is None


def test_default_geometry_matches_the_spec_built_directly(film_gate15_run):
    geom = film_gate15_run.session_state["mfs_geom"]
    ref = _build(LANDHANGER_SPEC)
    assert np.array_equal(geom.mask, ref.mask)
    assert np.array_equal(geom.thickness_mm[geom.mask], ref.thickness_mm[ref.mask])
    assert geom.gates == ref.gates
    assert film_gate15_run.session_state["mfs_result"] is not None


def test_only_the_ramp_in_front_of_the_centre_gets_shallower(film_gate15_run):
    """Against the same pocket without the profile: the silhouette and the
    land row t=1 stay, the changed cells are ramp cells made shallower, and
    on the axis the land runs to t=5 and the ramp to 2.5 at t=16."""
    geom = film_gate15_run.session_state["mfs_geom"]
    spec = GateProfileSpec.from_dict(_recorded_spec(film_gate15_run))
    plain = _build(PLAIN_SPEC)
    assert np.array_equal(geom.mask, plain.mask)
    t, wa = _tw(geom, spec)
    changed = geom.mask & (geom.thickness_mm != plain.thickness_mm)
    assert changed.any()
    assert t[changed].min() > 1.0 and t[changed].max() < 16.0
    assert np.all(geom.thickness_mm[changed] < plain.thickness_mm[changed])
    on_axis = geom.mask & (wa == 0.5) & (t > 0)
    for tv, d in zip(t[on_axis], geom.thickness_mm[on_axis], strict=True):
        L = 1.0 + 4.0 * (1.0 - 0.5 / 149.0) ** 2
        want = 0.35 if tv <= L else min(0.35 + TAN * (tv - L), 2.5)
        if tv < 15.0:  # the well's wall starts at 15.5
            assert d == pytest.approx(want, abs=1e-9), tv


def test_sliders_reach_the_spec():
    at = _app()
    at.slider(key=CENTER_KEY).set_value(6.0)
    at.number_input(key=POWER_KEY).set_value(1.5).run()
    rec = _recorded_spec_after_run(at)
    assert rec["land"]["profile"] == {"center_length": 6.0, "power": 1.5}
    at.checkbox(key="f15_lp_on").set_value(False).run()
    rec = _recorded_spec_after_run(at)
    assert rec["land"].get("profile") is None
    geom = at.session_state["mfs_geom"]
    plain = _build(PLAIN_SPEC)
    assert np.array_equal(geom.mask, plain.mask)
    assert np.array_equal(geom.thickness_mm, plain.thickness_mm)


def test_the_defaults_do_not_leak_into_film_gate_14():
    at = _app()
    assert at.checkbox(key="f15_lp_on").value is True
    at.radio(key="geom_source").set_value(FILM_GATE14_LABEL).run()
    assert at.checkbox(key="f14_lp_on").value is False
    rec = _recorded_spec_after_run(at, FILM_GATE14_LABEL)
    assert rec["land"].get("profile") is None
    assert rec["ramp_ends"] is not None and rec["land"]["closed_line"] is not None
