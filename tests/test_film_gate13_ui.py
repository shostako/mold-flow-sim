"""AppTest wiring checks for the parametric Film gate 13 (扇状/斜面角度徐変2).

The 2026/09/30 CAD model ``Runner-block_3D_kai01.igs``
(``hamoko_gate_furiwake_rampends2_20260930``): Film gate 12's pocket with
three changes read off the model's control points —

* the cap depth grades with the cap line: 2.5 at w=60 → 3.5 at the pocket end
  (cap line t=12 → 2.5), and the floor behind the line is that depth too;
* the centre of the land is steel at the PL, from w=50 at the exit to
  w=47.644 (the 肉盗み's start) at the land end — by design no resin enters
  the product through the centre;
* the corner where the exit width meets the 3° outer wall is rounded R10.

What the tests pin: the defaults reproduce the spec (bit-identical geometry),
the three changes are in the built field, against Film gate 12 nothing else
moves, each block switches off to its Film gate 12 form, and none of the
defaults leak into Film gate 12.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from core import GateProfileSpec, ProfilePlateConfig, RampEndsSpec, build_profile_gate_geometry

APP = Path(__file__).resolve().parent.parent / "app.py"
FILM_GATE13_LABEL = "Film gate 13 (扇状/斜面角度徐変2)"
FILM_GATE12_LABEL = "Film gate 12 (扇状/斜面角度徐変)"

DEPTH_END_KEY = "f13_ポケット端での上限深さ [mm] (≥ ランプ上限)"
CLOSED_KEY = "f13_閉鎖幅（中央、両側合計） [mm]"
CORNER_KEY = "f13_外壁の角 R [mm] (0 = 角のまま)"

RAMP_DEG = math.degrees(math.atan(2.15 / 11.0))

FG12_SPEC = {
    "name": "hamoko_gate_furiwake_rampends_20260928",
    "units": "mm",
    "symmetric": True,
    "gate_exit_width": 298.0,
    "land": {"depth": 0.35, "length": 1.0},
    "main_ramp": {"angle_deg": RAMP_DEG, "cap_depth": 2.5},
    "island": {
        "angle_deg": 2.5,
        "boundary_line": [[1.0, 47.644148], [17.0, 9.95051]],
        "end_dist": 17.0,
    },
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
    "ramp_ends": {"w_from": 60.0, "t_end": 4.0},
}
HAMOKO_RAMPENDS2_SPEC = {
    **FG12_SPEC,
    "name": "hamoko_gate_furiwake_rampends2_20260930",
    "land": {"depth": 0.35, "length": 1.0, "closed_line": [[0.0, 50.0], [1.0, 47.644148]]},
    "ramp_ends": {"w_from": 60.0, "t_end": 2.5, "depth_end": 3.5},
    "outer_wall_corner_radius": 10.0,
}

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
    at.radio(key="geom_source").set_value(FILM_GATE13_LABEL).run()
    at.radio(key="wall_model").set_value("none")
    return at


@pytest.fixture(scope="module")
def film_gate13_run() -> AppTest:
    at = _app()
    at.button[0].click().run()
    assert not at.exception
    return at


def _recorded_spec(at: AppTest, label: str = FILM_GATE13_LABEL) -> dict:
    geom = at.session_state["mfs_settings"]["geometry"]
    assert geom["input"] == label
    return geom["gate_profile"]


def _build(d: dict, dx: float = 1.0):
    return build_profile_gate_geometry(GateProfileSpec.from_dict(d), PLATE, dx)


def _cell(geom, spec: GateProfileSpec, t: float, w: float) -> float | None:
    """Thickness of the 1.0 mm cell centred at (t, w); None = steel."""
    pad = ProfilePlateConfig().pad_mm
    iy = (pad + spec.t_max() - t) - 0.5
    ix = (pad + PLATE.plate_w_mm / 2.0 + w) - 0.5
    assert abs(iy - round(iy)) < 1e-6 and abs(ix - round(ix)) < 1e-6, (t, w)
    iy, ix = int(round(iy)), int(round(ix))
    return float(geom.thickness_mm[iy, ix]) if geom.mask[iy, ix] else None


def _tw(geom, spec: GateProfileSpec):
    iy, ix = np.indices(geom.shape)
    pad = ProfilePlateConfig().pad_mm
    return (pad + spec.t_max()) - (iy + 0.5), np.abs((ix + 0.5) - (pad + PLATE.plate_w_mm / 2.0))


def _graded(t: float, w: float) -> float:
    """The ruled end ramp and floor: (1, 0.35) → (t_cap(w), D(w)), then D(w)."""
    if t <= 1.0:
        return 0.35
    frac = (min(w, 149.0) - 60.0) / 89.0
    t_cap, d_cap = 12.0 + (2.5 - 12.0) * frac, 2.5 + 1.0 * frac
    return min(d_cap, 0.35 + (d_cap - 0.35) * (t - 1.0) / (t_cap - 1.0))


def test_default_sliders_reproduce_the_cad_spec(film_gate13_run):
    rec = _recorded_spec(film_gate13_run)
    expected = GateProfileSpec.from_dict(HAMOKO_RAMPENDS2_SPEC)
    got = GateProfileSpec.from_dict({**rec, "name": expected.name})
    assert got.land == expected.land
    assert got.main_ramp.angle_deg == pytest.approx(RAMP_DEG, abs=1e-12)
    assert np.asarray(got.outer_wall_line) == pytest.approx(np.asarray(expected.outer_wall_line))
    assert got.outer_wall_corner_radius == 10.0
    assert got.ramp_ends == RampEndsSpec(w_from=60.0, t_end=2.5, depth_end=3.5)
    assert got.valve == expected.valve
    assert got.edge_channels == () and got.ramp_cut is None and got.land_ends is None


def test_default_geometry_matches_the_spec_built_directly(film_gate13_run):
    geom = film_gate13_run.session_state["mfs_geom"]
    ref = _build(HAMOKO_RAMPENDS2_SPEC)
    assert np.array_equal(geom.mask, ref.mask)
    assert np.array_equal(geom.thickness_mm[geom.mask], ref.thickness_mm[ref.mask])
    assert geom.gates == ref.gates
    # the default run solved: the product in front of the closed centre fills sideways
    assert film_gate13_run.session_state["mfs_result"] is not None


def test_the_three_changes_are_in_the_thickness_field(film_gate13_run):
    """At 1.0 mm the land is the row t=1: the closure reaches 47.644 there, so
    w=47.5 is steel and 48.5 is land. End columns follow the ruled surface
    up to D(w) and stay at D(w) behind it. The R10 corner takes w=148.5
    from t=7 on (the model's wall ends at t≈9.75 at w=148.25)."""
    geom = film_gate13_run.session_state["mfs_geom"]
    spec = GateProfileSpec.from_dict(_recorded_spec(film_gate13_run))
    for w in (0.5, 30.5, 47.5):
        assert _cell(geom, spec, 1.0, w) is None, w
    assert _cell(geom, spec, 1.0, 48.5) == pytest.approx(0.35)
    for w in (148.5, 120.5, 100.5, 60.5):
        for t in (2.0, 3.0, 5.0, 8.0):
            assert _cell(geom, spec, t, w) == pytest.approx(_graded(t, w), abs=1e-9), (t, w)
    assert _cell(geom, spec, 3.0, 148.5) == pytest.approx(2.5 + 88.5 / 89.0, abs=1e-9)
    assert _cell(geom, spec, 6.0, 148.5) is not None
    assert _cell(geom, spec, 11.0, 148.5) is None
    assert _cell(geom, spec, 11.0, 140.5) is not None


def test_against_film_gate_12_only_the_three_changes_move(film_gate13_run):
    geom = film_gate13_run.session_state["mfs_geom"]
    spec = GateProfileSpec.from_dict(_recorded_spec(film_gate13_run))
    fg12 = _build(FG12_SPEC)
    t, wa = _tw(geom, spec)
    assert not (geom.mask & ~fg12.mask).any()
    removed = fg12.mask & ~geom.mask
    land = removed & (t <= 1.0)
    corner = removed & (t > 1.0)
    assert land.any() and wa[land].max() < 50.0
    assert corner.any() and wa[corner].min() > 139.0 and t[corner].max() < 16.3
    both = geom.mask & fg12.mask
    diff = both & (geom.thickness_mm != fg12.thickness_mm)
    assert np.all(geom.thickness_mm[diff] > fg12.thickness_mm[diff])
    assert wa[diff].min() >= 60.0 and t[diff].min() > 1.0


def test_each_block_switches_off_to_its_film_gate_12_form():
    at = _app()
    at.checkbox(key="f13_lc_on").uncheck()
    at.number_input(key=CORNER_KEY).set_value(0.0)
    at.slider(key=DEPTH_END_KEY).set_value(2.5)
    at.slider(
        key="f13_ポケット端でランプが上限深さに達する t [mm] (< メインランプの到達 t)"
    ).set_value(4.0)
    at.run()
    at.button[0].click().run()
    assert not at.exception
    rec = _recorded_spec(at)
    assert rec["land"].get("closed_line") is None
    assert rec.get("outer_wall_corner_radius") is None
    assert rec["ramp_ends"].get("depth_end") is None
    geom = at.session_state["mfs_geom"]
    ref = _build(FG12_SPEC)
    assert np.array_equal(geom.mask, ref.mask)
    assert np.array_equal(geom.thickness_mm[geom.mask], ref.thickness_mm[ref.mask])


def test_the_closure_width_slider_moves_the_exit_edge():
    """Width 60 closes |w| < 30 at the exit; the 肉盗み (47.6) is wider, so
    the edge runs straight."""
    at = _app()
    at.slider(key=CLOSED_KEY).set_value(60.0).run()
    at.button[0].click().run()
    assert not at.exception
    spec = GateProfileSpec.from_dict(_recorded_spec(at))
    assert spec.land.closed_line == ((0.0, 30.0), (1.0, 30.0))
    geom = at.session_state["mfs_geom"]
    assert _cell(geom, spec, 1.0, 29.5) is None
    assert _cell(geom, spec, 1.0, 30.5) == pytest.approx(0.35)


def test_the_defaults_do_not_leak_into_film_gate_12():
    at = _app()
    assert at.checkbox(key="f13_lc_on").value is True
    assert at.number_input(key=CORNER_KEY).value == 10.0
    at.radio(key="geom_source").set_value(FILM_GATE12_LABEL).run()
    assert at.checkbox(key="f12_lc_on").value is False
    assert at.number_input(key="f12_外壁の角 R [mm] (0 = 角のまま)").value == 0.0
    assert at.slider(key="f12_ポケット端での上限深さ [mm] (≥ ランプ上限)").value == 2.5
    at.button[0].click().run()
    assert not at.exception
    rec = _recorded_spec(at, FILM_GATE12_LABEL)
    assert rec["land"].get("closed_line") is None
    assert rec.get("outer_wall_corner_radius") is None
    assert rec["ramp_ends"].get("depth_end") is None
