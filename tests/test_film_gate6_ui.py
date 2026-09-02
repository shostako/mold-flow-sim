"""AppTest wiring checks for the parametric Film gate 6 (扇状/縁部深彫り 0807).

The 2026/08/07 rework of Film gate 1 (``hamoko_gate_furiwake_rework_20260807``
plus the band that drawing carries as the "2" dimension): the pocket ends
grew from t=3 to t=5, the 肉盗み exit narrowed to 100, the well wall
steepened to 71.6°, and a 2 mm band at the ramp cap depth (2.5) runs along
the outer wall from t≈3 at the end until it meets the cap line. What has to
be carried by the defaults, not the shared sliders: the outer wall start,
the well wall angle (floor = ``depth/tan(71.6°) = 1.5`` inside each end)
and — new with this input — the 縁部深彫り block starting *on* at the
drawing's numbers. The band is what distinguishes this input from Film
gate 2 minus its dam, so the tests pin the band itself in the built
thickness field, not just the recorded spec.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from core import GateProfileSpec, ProfilePlateConfig, build_profile_gate_geometry

APP = Path(__file__).resolve().parent.parent / "app.py"
FILM_GATE6_LABEL = "Film gate 6 (扇状/縁部深彫り 0807)"
FILM_GATE1_LABEL = "Film gate 1 (扇状/肉盗み1)"

HAMOKO_EDGE_0807_SPEC = {
    "name": "hamoko_gate_furiwake_edge_20260807",
    "units": "mm",
    "symmetric": True,
    "gate_exit_width": 298.0,
    "land": {"depth": 0.35, "length": 1.0},
    "main_ramp": {"angle_deg": 10.95, "cap_depth": 2.5},
    "island": {
        "angle_deg": 2.5,
        "boundary_line": [[0.0, 50.0], [17.0, 9.9]],
        "end_dist": 17.0,
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
    "edge_channels": [{"width": 2.0, "depth": 2.5, "t_range": [5.0, 23.28], "side": "outer"}],
}

PLATE = ProfilePlateConfig(
    plate_w_mm=300.0,
    plate_h_mm=50.0,
    plate_thk_mm=0.35,
    plate_split_height_mm=20.0,
    plate_lower_thk_mm=0.35,
    plate_upper_thk_mm=0.50,
)


def _film_gate6_app() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=240.0)
    at.run()
    at.radio(key="geom_source").set_value(FILM_GATE6_LABEL).run()
    at.radio(key="wall_model").set_value("none")
    return at


@pytest.fixture(scope="module")
def film_gate6_run() -> AppTest:
    at = _film_gate6_app()
    at.button[0].click().run()
    assert not at.exception
    return at


def _recorded_spec(at: AppTest, label: str = FILM_GATE6_LABEL) -> dict:
    geom = at.session_state["mfs_settings"]["geometry"]
    assert geom["input"] == label
    return geom["gate_profile"]


def _thickness_at(geom, spec: GateProfileSpec, t: float, w: float, dx: float = 1.0) -> float:
    """Built thickness of the cell whose centre is at gate coordinates (t, w).

    Same mapping as the builder: t = y_plate_bottom − y_centre with
    y_plate_bottom = pad + t_max, w from the plate centre (symmetric). With
    t_max = 27.5 the cell centres sit at integer t and half-integer w, so
    callers ask for those and the helper insists the request hits a centre.
    Assumes ``valve.w == 0`` (the UI always passes it), so the plate centre
    is the valve axis — a test-local helper, not a general mapping.
    """
    pad = ProfilePlateConfig().pad_mm
    iy = (pad + spec.t_max() - t) / dx - 0.5
    ix = (pad + PLATE.plate_w_mm / 2.0 + w) / dx - 0.5
    assert abs(iy - round(iy)) < 1e-6 and abs(ix - round(ix)) < 1e-6, (t, w)
    iy, ix = int(round(iy)), int(round(ix))
    assert geom.mask[iy, ix], f"({t}, {w}) is not a cavity cell"
    return float(geom.thickness_mm[iy, ix])


def _ramp(t: float) -> float:
    return 0.35 + math.tan(math.radians(10.95)) * (t - 1.0)


def test_default_sliders_reproduce_the_rework_spec_with_its_band(film_gate6_run):
    rec = _recorded_spec(film_gate6_run)
    expected = GateProfileSpec.from_dict(HAMOKO_EDGE_0807_SPEC)
    got = GateProfileSpec.from_dict({**rec, "name": HAMOKO_EDGE_0807_SPEC["name"]})
    assert got.symmetric is True
    assert got.gate_exit_width == expected.gate_exit_width
    assert got.land == expected.land
    assert got.main_ramp == expected.main_ramp
    assert got.island.angle_deg == expected.island.angle_deg
    assert got.island.end_dist == expected.island.end_dist
    assert got.island.weld is None
    # The drawing's boundary line starts at t=0; the UI pins it at t=land
    # length. Same line: compare where both are defined.
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
    assert got.valve == expected.valve  # 21.5 = the well centre
    # The band starts on, at the drawing's numbers.
    assert len(got.edge_channels) == 1
    (ec,) = got.edge_channels
    (ref,) = expected.edge_channels
    assert (ec.width, ec.depth, ec.side) == (ref.width, ref.depth, ref.side)
    assert ec.t_range == pytest.approx(ref.t_range)


def test_default_geometry_matches_the_spec_built_directly(film_gate6_run):
    geom = film_gate6_run.session_state["mfs_geom"]
    ref = build_profile_gate_geometry(GateProfileSpec.from_dict(HAMOKO_EDGE_0807_SPEC), PLATE, 1.0)
    assert np.array_equal(geom.mask, ref.mask)
    assert np.array_equal(geom.thickness_mm[geom.mask], ref.thickness_mm[ref.mask])
    assert geom.gates == ref.gates


def test_the_band_is_at_cap_depth_where_the_ramp_is_still_shallow(film_gate6_run):
    """At the pocket end (w≈148.5, wall at t≈5.06) the cells within 2 mm of the
    outer wall read 2.5 from t≈3 on, while the land end and the first 2 mm of
    ramp above the band are still 0.35 + tan(10.95°)·(t − 1). Where the ramp
    has reached the cap (t > 12.11) the band changes nothing — floor
    semantics — so every cell the band touches is one where the ramp was
    still below 2.5."""
    geom = film_gate6_run.session_state["mfs_geom"]
    rec = _recorded_spec(film_gate6_run)
    spec = GateProfileSpec.from_dict(rec)
    # band: cell centres t=4, 5 at the end are within 2 mm of the wall corner
    assert _thickness_at(geom, spec, 4.0, 148.5) == pytest.approx(2.5)
    assert _thickness_at(geom, spec, 5.0, 148.5) == pytest.approx(2.5)
    # the land end and the first 2 mm of ramp stay as drawn: the band's
    # t-range is the wall's own extent, so no stub runs along the end wall
    assert _thickness_at(geom, spec, 1.0, 148.5) == pytest.approx(0.35)
    assert _thickness_at(geom, spec, 2.0, 148.5) == pytest.approx(_ramp(2.0), abs=1e-6)
    # w=140.5: wall at t≈6.1, band inner limit ≈ 4.1, so t=3 is plain ramp
    assert _thickness_at(geom, spec, 3.0, 140.5) == pytest.approx(_ramp(3.0), abs=1e-6)
    # the band only ever raises cells the ramp had left below the cap
    bare = build_profile_gate_geometry(
        GateProfileSpec.from_dict({**rec, "edge_channels": []}), PLATE, 1.0
    )
    assert np.array_equal(geom.mask, bare.mask)
    changed = geom.mask & (geom.thickness_mm != bare.thickness_mm)
    assert changed.sum() > 50
    assert np.all(bare.thickness_mm[changed] < 2.5)
    assert np.all(geom.thickness_mm[changed] == pytest.approx(2.5))


def test_switching_the_band_off_drops_the_channel():
    at = _film_gate6_app()
    at.checkbox(key="f6_ec_on").set_value(False).run()
    at.button[0].click().run()
    assert not at.exception
    got = GateProfileSpec.from_dict({**_recorded_spec(at), "name": "x"})
    assert got.edge_channels == ()
    # everything else is still the rework pocket
    assert np.asarray(got.outer_wall_line[0]) == pytest.approx((5.0, 149.0))
    assert got.well.wall_angle_deg == 71.6


def _slider(at: AppTest, label_prefix: str):
    hits = [s for s in at.slider if str(s.label).startswith(label_prefix)]
    assert len(hits) == 1, [str(s.label) for s in at.slider]
    return hits[0]


def test_a_wall_shorter_than_the_drawings_band_still_builds():
    """Codex P2: with the wall end below the drawing's band start (5.0) the
    drawing's range clamped to [0, wall end] collapses to a zero-width range
    that ``validate()`` rejects, so an otherwise valid wall could not be
    built. The default must fall back to the live wall extent instead.

    The well is switched off and the valve moved into the short pocket so
    the only thing that could stop the build is the band's range — with the
    well on, a wall ending at t=4.5 strands the well and the gate is
    unreachable for a reason unrelated to this fix."""
    at = _film_gate6_app()
    at.checkbox(key="f6_well_on").set_value(False).run()
    _slider(at, "外壁開始 t").set_value(1.0).run()
    _slider(at, "外壁終端 t").set_value(4.5).run()
    _slider(at, "バルブ位置 t").set_value(2.5).run()
    assert _slider(at, "範囲 t").value == (1.0, 4.5)
    at.button[0].click().run()
    assert not at.exception
    assert not at.error
    assert at.session_state["mfs_geom"] is not None
    got = GateProfileSpec.from_dict({**_recorded_spec(at), "name": "x"})
    (ec,) = got.edge_channels
    lo, hi = ec.t_range
    assert hi > lo
    assert (lo, hi) == pytest.approx((1.0, 4.5))


def test_the_on_by_default_band_does_not_leak_into_film_gate_1():
    """Film gate 6 starts with 縁部深彫り on; Film gate 1 must still start off
    after visiting 6 (separate ``f6_`` / ``f1_`` widget keys)."""
    at = _film_gate6_app()
    assert at.checkbox(key="f6_ec_on").value is True
    at.radio(key="geom_source").set_value(FILM_GATE1_LABEL).run()
    assert at.checkbox(key="f1_ec_on").value is False
    at.button[0].click().run()
    assert not at.exception
    got = GateProfileSpec.from_dict({**_recorded_spec(at, FILM_GATE1_LABEL), "name": "x"})
    assert got.edge_channels == ()
    assert np.asarray(got.outer_wall_line[0]) == pytest.approx((3.0, 149.0))
