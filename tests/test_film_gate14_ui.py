"""AppTest wiring checks for the parametric Film gate 14 (扇状/斜面角度徐変3).

The 2026/09/30 CAD model ``Runner-block_3D_kai02.igs``
(``hamoko_gate_furiwake_rampends3_20260930``): Film gate 13's pocket with the
肉盗み and the corner R10 taken out. The land closure keeps the kai01 chamfer
(w=50 at the exit → 47.644 at the land end) although the 肉盗み it was aligned
with is gone, so the closure's land-end width is its own input here
(``lc_end_on``) instead of being derived from the 肉盗み.

What the tests pin: the defaults reproduce the spec (bit-identical geometry),
against Film gate 13 only the 肉盗み and the corner move, the land-end width
option is what keeps the chamfer, and none of it leaks into Film gate 13.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from core import GateProfileSpec, ProfilePlateConfig, RampEndsSpec, build_profile_gate_geometry

APP = Path(__file__).resolve().parent.parent / "app.py"
FILM_GATE14_LABEL = "Film gate 14 (扇状/斜面角度徐変3)"
FILM_GATE13_LABEL = "Film gate 13 (扇状/斜面角度徐変2)"

CLOSED_KEY = "f14_閉鎖幅（中央、両側合計） [mm]"
END_ON_KEY = "f14_lc_end_on"
END_KEY = "f14_ランド終端での閉鎖幅（両側合計） [mm]"
CORNER_KEY = "f14_外壁の角 R [mm] (0 = 角のまま)"

RAMP_DEG = math.degrees(math.atan(2.15 / 11.0))
TAN = 2.15 / 11.0

_WELL = {
    "shape": "obround",
    "t_range": [15.5, 27.5],
    "half_width": 4.5,
    "depth": 4.5,
    "floor_t_range": [18.098076, 24.901924],
    "wall_angle_deg": 60.0,
}
HAMOKO_RAMPENDS3_SPEC = {
    "name": "hamoko_gate_furiwake_rampends3_20260930",
    "units": "mm",
    "symmetric": True,
    "gate_exit_width": 298.0,
    "land": {"depth": 0.35, "length": 1.0, "closed_line": [[0.0, 50.0], [1.0, 47.644148]]},
    "main_ramp": {"angle_deg": RAMP_DEG, "cap_depth": 2.5},
    "outer_wall_line": [[15.736241, 149.0], [23.309724, 4.489329]],
    "well": _WELL,
    "valve": {"t": 21.5, "w": 0.0, "orifice_diameter": 3.0},
    "ramp_ends": {"w_from": 60.0, "t_end": 2.5, "depth_end": 3.5},
}
FG13_SPEC = {
    **HAMOKO_RAMPENDS3_SPEC,
    "name": "hamoko_gate_furiwake_rampends2_20260930",
    "island": {
        "angle_deg": 2.5,
        "boundary_line": [[1.0, 47.644148], [17.0, 9.95051]],
        "end_dist": 17.0,
    },
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
    at.radio(key="geom_source").set_value(FILM_GATE14_LABEL).run()
    at.radio(key="wall_model").set_value("none")
    return at


@pytest.fixture(scope="module")
def film_gate14_run() -> AppTest:
    at = _app()
    at.button[0].click().run()
    assert not at.exception
    return at


def _recorded_spec(at: AppTest, label: str = FILM_GATE14_LABEL) -> dict:
    geom = at.session_state["mfs_settings"]["geometry"]
    assert geom["input"] == label
    return geom["gate_profile"]


def _recorded_spec_after_run(at: AppTest, label: str = FILM_GATE14_LABEL) -> dict:
    at.checkbox(key="two_phase_on").set_value(False)
    at.button[0].click().run()
    assert not at.exception
    return _recorded_spec(at, label)


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


def test_default_sliders_reproduce_the_cad_spec(film_gate14_run):
    rec = _recorded_spec(film_gate14_run)
    expected = GateProfileSpec.from_dict(HAMOKO_RAMPENDS3_SPEC)
    got = GateProfileSpec.from_dict({**rec, "name": expected.name})
    assert got.land == expected.land  # the chamfer to 47.644148, with no 肉盗み
    assert got.island is None
    assert got.outer_wall_corner_radius is None
    assert got.main_ramp.angle_deg == pytest.approx(RAMP_DEG, abs=1e-12)
    assert np.asarray(got.outer_wall_line) == pytest.approx(np.asarray(expected.outer_wall_line))
    assert got.ramp_ends == RampEndsSpec(w_from=60.0, t_end=2.5, depth_end=3.5)
    assert got.valve == expected.valve
    assert got.edge_channels == () and got.ramp_cut is None and got.land_ends is None


def test_default_geometry_matches_the_spec_built_directly(film_gate14_run):
    geom = film_gate14_run.session_state["mfs_geom"]
    ref = _build(HAMOKO_RAMPENDS3_SPEC)
    assert np.array_equal(geom.mask, ref.mask)
    assert np.array_equal(geom.thickness_mm[geom.mask], ref.thickness_mm[ref.mask])
    assert geom.gates == ref.gates
    assert film_gate14_run.session_state["mfs_result"] is not None


def test_the_centre_is_one_plane_ramp_and_the_closure_keeps_its_chamfer(film_gate14_run):
    """No 肉盗み: on the axis the depth is the main ramp up to t=12 and the 2.5
    floor behind it. The land row t=1 is steel up to w=47.5 and land at 48.5
    (the chamfer's end 47.644 -- a straight closure would close 48.5 too)."""
    geom = film_gate14_run.session_state["mfs_geom"]
    spec = GateProfileSpec.from_dict(_recorded_spec(film_gate14_run))
    for t in (2.0, 5.0, 9.0, 11.0):
        assert _cell(geom, spec, t, 0.5) == pytest.approx(0.35 + TAN * (t - 1.0), abs=1e-9), t
    for t in (13.0, 14.0):
        assert _cell(geom, spec, t, 0.5) == pytest.approx(2.5), t
    for w in (0.5, 30.5, 47.5):
        assert _cell(geom, spec, 1.0, w) is None, w
    assert _cell(geom, spec, 1.0, 48.5) == pytest.approx(0.35)
    # square corner: w=148.5 is pocket where Film gate 13's R10 took it
    assert _cell(geom, spec, 11.0, 148.5) is not None


def test_against_film_gate_13_only_the_island_and_the_corner_move(film_gate14_run):
    geom = film_gate14_run.session_state["mfs_geom"]
    spec = GateProfileSpec.from_dict(_recorded_spec(film_gate14_run))
    fg13 = _build(FG13_SPEC)
    t, wa = _tw(geom, spec)
    assert not (fg13.mask & ~geom.mask).any()
    added = geom.mask & ~fg13.mask
    assert added.any() and wa[added].min() > 139.0 and t[added].min() > 1.0
    both = geom.mask & fg13.mask
    changed = both & (geom.thickness_mm != fg13.thickness_mm)
    assert changed.any()
    # the 肉盗み's band only, and the plane ramp is deeper than its 2.5° floor
    assert wa[changed].max() < 48.0 and t[changed].min() > 1.0 and t[changed].max() <= 17.0
    assert np.all(geom.thickness_mm[changed] > fg13.thickness_mm[changed])


def test_the_land_end_width_option_is_what_keeps_the_chamfer():
    """Off: with no 肉盗み the edge runs straight (w=50 at both ends). On with
    60: the edge runs 50 → 30."""
    at = _app()
    assert at.checkbox(key=END_ON_KEY).value is True
    assert at.slider(key=END_KEY).value == pytest.approx(95.288296)
    at.checkbox(key=END_ON_KEY).uncheck().run()
    rec = _recorded_spec_after_run(at)
    assert rec["land"]["closed_line"] == [[0.0, 50.0], [1.0, 50.0]]
    geom = at.session_state["mfs_geom"]
    spec = GateProfileSpec.from_dict(rec)
    assert _cell(geom, spec, 1.0, 48.5) is None and _cell(geom, spec, 1.0, 49.5) is None

    at.checkbox(key=END_ON_KEY).check().run()
    at.slider(key=END_KEY).set_value(60.0).run()
    rec = _recorded_spec_after_run(at)
    assert rec["land"]["closed_line"] == [[0.0, 50.0], [1.0, 30.0]]


def test_the_defaults_do_not_leak_into_film_gate_13():
    at = _app()
    assert at.checkbox(key="f14_island_on").value is False
    assert at.number_input(key=CORNER_KEY).value == 0.0
    at.radio(key="geom_source").set_value(FILM_GATE13_LABEL).run()
    assert at.checkbox(key="f13_island_on").value is True
    assert at.checkbox(key="f13_lc_end_on").value is False
    assert at.number_input(key="f13_外壁の角 R [mm] (0 = 角のまま)").value == 10.0
    rec = _recorded_spec_after_run(at, FILM_GATE13_LABEL)
    # Film gate 13's chamfer is still the one derived from its 肉盗み
    assert rec["land"]["closed_line"] == [[0.0, 50.0], [1.0, 47.644148]]
    assert rec["outer_wall_corner_radius"] == 10.0
    assert rec["island"] is not None
