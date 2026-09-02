"""AppTest wiring checks for the parametric Film gate 7 (扇状/縁部深彫り 0515).

The 2026/05/15 proposal「フィルムゲート(流動長150mm)」: Film gate 1's pocket
with the ends at t=4 and a groove of constant depth 2.4 along the outer wall
all the way down to the land (the detail draws its section as a trapezoid
3.0 at the floor / 4.0 at the opening; the plan draws the 3.0 floor). The
肉盗み exit is 100 like the rework. Everything the drawing does not dimension
(ramp, well, valve) is Film gate 1's. The band reaching the land is what
differs from Film gate 6, so the tests pin that: at the end the cell right
under the land is already 2.4 deep and the land itself is still 0.35.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from core import GateProfileSpec, ProfilePlateConfig, build_profile_gate_geometry

APP = Path(__file__).resolve().parent.parent / "app.py"
FILM_GATE7_LABEL = "Film gate 7 (扇状/縁部深彫り 0515)"
FILM_GATE6_LABEL = "Film gate 6 (扇状/縁部深彫り 0807)"

HAMOKO_EDGE_0515_SPEC = {
    "name": "hamoko_gate_furiwake_edge_20260515",
    "units": "mm",
    "symmetric": True,
    "gate_exit_width": 298.0,
    "land": {"depth": 0.35, "length": 1.0},
    "main_ramp": {"angle_deg": 10.95, "cap_depth": 2.5},
    "island": {
        "angle_deg": 2.5,
        "boundary_line": [[0.0, 50.0], [17.0, 10.0]],
        "end_dist": 17.0,
    },
    "outer_wall_line": [[4.0, 149.0], [23.3, 4.5]],
    "well": {
        "shape": "obround",
        "t_range": [15.5, 27.5],
        "half_width": 4.5,
        "depth": 4.5,
        "floor_t_range": [18.1, 24.9],
        "wall_angle_deg": 60,
    },
    "valve": {"t": 21.5, "w": 0.0, "orifice_diameter": 3.0},
    "edge_channels": [{"width": 3.0, "depth": 2.4, "t_range": [4.0, 23.3], "side": "outer"}],
}

PLATE = ProfilePlateConfig(
    plate_w_mm=300.0,
    plate_h_mm=50.0,
    plate_thk_mm=0.35,
    plate_split_height_mm=20.0,
    plate_lower_thk_mm=0.35,
    plate_upper_thk_mm=0.50,
)


def _film_gate7_app() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=240.0)
    at.run()
    at.radio(key="geom_source").set_value(FILM_GATE7_LABEL).run()
    at.radio(key="wall_model").set_value("none")
    return at


@pytest.fixture(scope="module")
def film_gate7_run() -> AppTest:
    at = _film_gate7_app()
    at.button[0].click().run()
    assert not at.exception
    return at


def _recorded_spec(at: AppTest, label: str = FILM_GATE7_LABEL) -> dict:
    geom = at.session_state["mfs_settings"]["geometry"]
    assert geom["input"] == label
    return geom["gate_profile"]


def _thickness_at(geom, spec: GateProfileSpec, t: float, w: float, dx: float = 1.0) -> float:
    """Thickness of the cell centred at (t, w); see test_film_gate6_ui."""
    pad = ProfilePlateConfig().pad_mm
    iy = (pad + spec.t_max() - t) / dx - 0.5
    ix = (pad + PLATE.plate_w_mm / 2.0 + w) / dx - 0.5
    assert abs(iy - round(iy)) < 1e-6 and abs(ix - round(ix)) < 1e-6, (t, w)
    iy, ix = int(round(iy)), int(round(ix))
    assert geom.mask[iy, ix], f"({t}, {w}) is not a cavity cell"
    return float(geom.thickness_mm[iy, ix])


def _ramp(t: float) -> float:
    return 0.35 + math.tan(math.radians(10.95)) * (t - 1.0)


def test_default_sliders_reproduce_the_0515_spec_with_its_groove(film_gate7_run):
    rec = _recorded_spec(film_gate7_run)
    expected = GateProfileSpec.from_dict(HAMOKO_EDGE_0515_SPEC)
    got = GateProfileSpec.from_dict({**rec, "name": HAMOKO_EDGE_0515_SPEC["name"]})
    assert got.symmetric is True
    assert got.gate_exit_width == expected.gate_exit_width
    assert got.land == expected.land
    assert got.main_ramp == expected.main_ramp
    assert got.island.angle_deg == expected.island.angle_deg
    assert got.island.end_dist == expected.island.end_dist
    assert got.island.weld is None
    (t1, w1), (t2, w2) = expected.island.boundary_line
    (g1, gw1), (g2, gw2) = got.island.boundary_line
    assert g1 == 1.0 and g2 == t2 and gw2 == w2
    assert gw1 == pytest.approx(w1 + (w2 - w1) * (g1 - t1) / (t2 - t1), abs=0.01)
    assert np.asarray(got.outer_wall_line) == pytest.approx(np.asarray(expected.outer_wall_line))
    assert got.well.t_range == expected.well.t_range
    assert got.well.half_width == expected.well.half_width
    assert got.well.depth == expected.well.depth
    assert got.well.wall_angle_deg == expected.well.wall_angle_deg
    assert got.well.floor_t_range == pytest.approx(expected.well.floor_t_range, abs=0.05)
    assert got.valve == expected.valve
    assert len(got.edge_channels) == 1
    (ec,) = got.edge_channels
    (ref,) = expected.edge_channels
    assert (ec.width, ec.depth, ec.side) == (ref.width, ref.depth, ref.side)
    assert ec.t_range == pytest.approx(ref.t_range)


def test_default_geometry_matches_the_spec_built_directly(film_gate7_run):
    geom = film_gate7_run.session_state["mfs_geom"]
    ref = build_profile_gate_geometry(GateProfileSpec.from_dict(HAMOKO_EDGE_0515_SPEC), PLATE, 1.0)
    assert np.array_equal(geom.mask, ref.mask)
    assert np.array_equal(geom.thickness_mm[geom.mask], ref.thickness_mm[ref.mask])
    assert geom.gates == ref.gates


def test_the_groove_reaches_the_land_and_runs_the_whole_wall(film_gate7_run):
    """At the end (w=148.5, wall at t≈4.07) the groove is 2.4 deep from the
    cell right after the land end (t=2) to the wall (t=4) while the land end
    (t=1) is still 0.35; two thirds of the way along the wall (w=100.5, wall
    at t≈10.5) the 3 mm band still reads 2.4 and the ramp inside it is the
    ramp. The 3 mm band around the wall corner (4, 149) reaches down to t≈1,
    so at the very end the groove meets the land instead of stopping 2 mm
    short as in Film gate 6 — while the land row itself (centre t=1, 3.04 from
    the corner) keeps its 0.35, as the drawing runs the land to the corner."""
    geom = film_gate7_run.session_state["mfs_geom"]
    rec = _recorded_spec(film_gate7_run)
    spec = GateProfileSpec.from_dict(rec)
    assert _thickness_at(geom, spec, 1.0, 148.5) == pytest.approx(0.35)
    for t in (2.0, 3.0, 4.0):
        assert _thickness_at(geom, spec, t, 148.5) == pytest.approx(2.4)
    assert _thickness_at(geom, spec, 9.0, 100.5) == pytest.approx(2.4)
    assert _thickness_at(geom, spec, 5.0, 100.5) == pytest.approx(_ramp(5.0), abs=1e-6)
    # the groove, unlike Film gate 6's band, is shallower than the cap: past
    # the cap line it raises nothing, before it every touched cell is 2.4
    bare = build_profile_gate_geometry(
        GateProfileSpec.from_dict({**rec, "edge_channels": []}), PLATE, 1.0
    )
    changed = geom.mask & (geom.thickness_mm != bare.thickness_mm)
    assert changed.sum() > 200
    assert np.all(bare.thickness_mm[changed] < 2.4)
    assert np.all(geom.thickness_mm[changed] == pytest.approx(2.4))


def test_groove_off_is_film_gate_1_with_the_ends_at_t4():
    at = _film_gate7_app()
    at.checkbox(key="f7_ec_on").set_value(False).run()
    at.button[0].click().run()
    assert not at.exception
    got = GateProfileSpec.from_dict({**_recorded_spec(at), "name": "x"})
    assert got.edge_channels == ()
    assert np.asarray(got.outer_wall_line) == pytest.approx(np.asarray([[4.0, 149.0], [23.3, 4.5]]))
    assert got.well.wall_angle_deg == 60.0


def test_film_gate_6_and_7_keep_their_own_band_defaults():
    """Switching 7 → 6 must show 6's band (2.0 × 2.5 along t∈[5, 23.28]), not
    7's (3.0 × 2.4 along t∈[4, 23.3]): same labels, different ``f6_`` / ``f7_``
    keys."""
    at = _film_gate7_app()
    at.radio(key="geom_source").set_value(FILM_GATE6_LABEL).run()
    at.button[0].click().run()
    assert not at.exception
    got = GateProfileSpec.from_dict({**_recorded_spec(at, FILM_GATE6_LABEL), "name": "x"})
    (ec,) = got.edge_channels
    assert (ec.width, ec.depth) == (2.0, 2.5)
    assert ec.t_range == pytest.approx((5.0, 23.28))
