"""AppTest wiring checks for the parametric Film gate 10 (扇状/肉厚調整 0923).

The 2026/09/23 drawing「ランナブロック 肉厚調整ゲート（振り分け）」
(``hamoko_gate_furiwake_stepcut_20260923``): Film gate 9's pocket with the
back of the ramp milled down to 2.5 beyond a line parallel to the 3° outer
wall — t=7.451 at the pocket end, meeting the depth-2.5 line t=12.11 at the
edge of an untouched central 120. Where the ramp has not reached 2.5 at that
line the cut leaves a step (0.9 mm at the end); the drawing's「斜面角度徐変」
asks for a slope instead, and the slope angle — cut into the product side —
is the parameter. The default is the angle whose slope at the end runs from
the land end to the line (no step, no cut into the land): 18.43°.

What the tests pin: the defaults reproduce the spec (line, depth, angle),
the built field carries the cut (the end column is a single 18.43° slope to
2.5 at t≈7.5, half way in the chamfer meets the ramp part way), against
Film gate 9's pocket it only deepens cells and only outside w=60, 90° gives
the bare step, and the block's default does not leak into Film gate 9.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from core import GateProfileSpec, ProfilePlateConfig, build_profile_gate_geometry

APP = Path(__file__).resolve().parent.parent / "app.py"
FILM_GATE10_LABEL = "Film gate 10 (扇状/肉厚調整 0923)"
FILM_GATE9_LABEL = "Film gate 9 (扇状/9月末試作)"

T_CAP = 1.0 + (2.5 - 0.35) / math.tan(math.radians(10.95))  # 12.1126, where the ramp hits 2.5
NO_STEP_DEG = math.degrees(math.atan((2.5 - 0.35) / (7.451 - 1.0)))  # 18.432°

HAMOKO_STEPCUT_0923_SPEC = {
    "name": "hamoko_gate_furiwake_stepcut_20260923",
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
    "outer_wall_line": [[15.736, 149.0], [23.28, 4.48]],
    "ramp_cut": {
        "line": [[7.451, 149.0], [T_CAP, 60.0]],
        "depth": 2.5,
        "slope_angle_deg": NO_STEP_DEG,
    },
    "well": {
        "shape": "obround",
        "t_range": [15.5, 27.5],
        "half_width": 4.5,
        "depth": 4.5,
        "floor_t_range": [18.1, 24.9],
        "wall_angle_deg": 60.0,
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


def _film_gate10_app() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=240.0)
    at.run()
    at.radio(key="geom_source").set_value(FILM_GATE10_LABEL).run()
    at.radio(key="wall_model").set_value("none")
    return at


@pytest.fixture(scope="module")
def film_gate10_run() -> AppTest:
    at = _film_gate10_app()
    at.button[0].click().run()
    assert not at.exception
    return at


def _recorded_spec(at: AppTest, label: str = FILM_GATE10_LABEL) -> dict:
    geom = at.session_state["mfs_settings"]["geometry"]
    assert geom["input"] == label
    return geom["gate_profile"]


def _fg9_geometry():
    d = {k: v for k, v in HAMOKO_STEPCUT_0923_SPEC.items() if k != "ramp_cut"}
    return build_profile_gate_geometry(GateProfileSpec.from_dict(d), PLATE, 1.0)


def _cell(geom, spec: GateProfileSpec, t: float, w: float, dx: float = 1.0) -> float | None:
    """Thickness of the cell centred at (t, w); None = steel (see test_film_gate9_ui)."""
    pad = ProfilePlateConfig().pad_mm
    iy = (pad + spec.t_max() - t) / dx - 0.5
    ix = (pad + PLATE.plate_w_mm / 2.0 + w) / dx - 0.5
    assert abs(iy - round(iy)) < 1e-6 and abs(ix - round(ix)) < 1e-6, (t, w)
    iy, ix = int(round(iy)), int(round(ix))
    return float(geom.thickness_mm[iy, ix]) if geom.mask[iy, ix] else None


def _ramp(t: float) -> float:
    return min(2.5, 0.35 + math.tan(math.radians(10.95)) * (t - 1.0))


def _t_cut(w: float) -> float:
    return 7.451 + (T_CAP - 7.451) * (149.0 - w) / (149.0 - 60.0)


def _chamfer(t: float, w: float, slope_deg: float = NO_STEP_DEG) -> float:
    return max(_ramp(t), 2.5 - max(_t_cut(w) - t, 0.0) * math.tan(math.radians(slope_deg)))


def test_default_sliders_reproduce_the_stepcut_spec(film_gate10_run):
    rec = _recorded_spec(film_gate10_run)
    expected = GateProfileSpec.from_dict(HAMOKO_STEPCUT_0923_SPEC)
    got = GateProfileSpec.from_dict({**rec, "name": expected.name})
    assert got.symmetric is True
    assert got.land == expected.land and got.main_ramp == expected.main_ramp
    assert np.asarray(got.outer_wall_line) == pytest.approx(np.asarray(expected.outer_wall_line))
    assert got.well.wall_angle_deg == 60.0
    assert got.valve == expected.valve
    assert got.edge_channels == ()
    # the one feature this drawing adds — exact: the line's start is the
    # drawing's "7.451" (not rounded to the slider step), its end is where
    # the ramp reaches 2.5 at the edge of the untouched 120
    rc = got.ramp_cut
    assert rc is not None
    assert rc.line[0] == (7.451, 149.0)
    assert rc.line[1] == pytest.approx((T_CAP, 60.0))
    assert rc.depth == 2.5
    assert rc.slope_angle_deg == pytest.approx(NO_STEP_DEG)
    # the line is the 3° wall shifted towards the exit
    (t1, w1), (t2, w2) = rc.line
    assert math.degrees(math.atan2(t2 - t1, w1 - w2)) == pytest.approx(3.0, abs=0.05)


def test_default_geometry_matches_the_spec_built_directly(film_gate10_run):
    geom = film_gate10_run.session_state["mfs_geom"]
    ref = build_profile_gate_geometry(
        GateProfileSpec.from_dict(HAMOKO_STEPCUT_0923_SPEC), PLATE, 1.0
    )
    assert np.array_equal(geom.mask, ref.mask)
    assert np.array_equal(geom.thickness_mm[geom.mask], ref.thickness_mm[ref.mask])
    assert geom.gates == ref.gates


def test_the_cut_is_in_the_thickness_field(film_gate10_run):
    """End column (w=148.5): one 18.43° slope from the land end to 2.5 at
    t≈7.5, then 2.5 up to the wall. Half way in (w=100.5, t_cut≈9.99): ramp
    up to t=7, the chamfer takes over at t=8–9, 2.5 from t=10. Inside the
    untouched 120 (w=59.5) it is Film gate 9's ramp unchanged."""
    geom = film_gate10_run.session_state["mfs_geom"]
    spec = GateProfileSpec.from_dict(_recorded_spec(film_gate10_run))
    assert _cell(geom, spec, 1.0, 148.5) == pytest.approx(0.35)
    for t in (2.0, 4.0, 7.0):
        got = _cell(geom, spec, t, 148.5)
        assert got == pytest.approx(_chamfer(t, 148.5), abs=1e-6)
        assert got > _ramp(t) + 0.1  # visibly deeper than Film gate 9
    for t in (8.0, 12.0, 15.0):
        assert _cell(geom, spec, t, 148.5) == pytest.approx(2.5)
    assert _cell(geom, spec, 16.0, 148.5) is None
    assert _cell(geom, spec, 7.0, 100.5) == pytest.approx(_ramp(7.0), abs=1e-6)
    for t in (8.0, 9.0):
        got = _cell(geom, spec, t, 100.5)
        assert got == pytest.approx(_chamfer(t, 100.5), abs=1e-6) and _ramp(t) < got < 2.5
    assert _cell(geom, spec, 10.0, 100.5) == pytest.approx(2.5)
    for t in (6.0, 10.0, 12.0):
        assert _cell(geom, spec, t, 59.5) == pytest.approx(_ramp(t), abs=1e-6)


def test_against_film_gate_9_the_cut_only_deepens_cells_outside_the_centre(film_gate10_run):
    geom = film_gate10_run.session_state["mfs_geom"]
    fg9 = _fg9_geometry()
    assert np.array_equal(geom.mask, fg9.mask)
    assert np.all(geom.thickness_mm >= fg9.thickness_mm)
    diff = geom.thickness_mm != fg9.thickness_mm
    assert diff.any()
    _iy, ix = np.indices(diff.shape)
    wa = np.abs((ix + 0.5) - (ProfilePlateConfig().pad_mm + PLATE.plate_w_mm / 2.0))
    assert wa[diff].min() >= 60.0 and wa[diff].max() <= 149.0
    # ≈ 0.3 cm³ more steel gone with the no-step slope (0.25 mm raster: +297 mm³)
    dv = (geom.volume_cm3() - fg9.volume_cm3()) * 1000.0
    assert dv == pytest.approx(297.0, rel=0.05)


def test_ninety_degrees_is_the_bare_step():
    """At 90° the end column is Film gate 9's ramp up to the line and 2.5
    beyond — a 0.9 mm step at t≈7.5 — and less steel goes than with the
    slope."""
    at = _film_gate10_app()
    at.number_input(key="f10_段差の傾斜角 [deg]（製品側に削る斜面、90 = 段差のまま）").set_value(
        90.0
    )
    at.button[0].click().run()
    assert not at.exception
    geom = at.session_state["mfs_geom"]
    spec = GateProfileSpec.from_dict(_recorded_spec(at))
    assert spec.ramp_cut.slope_angle_deg == 90.0
    for t in (2.0, 5.0, 7.0):
        assert _cell(geom, spec, t, 148.5) == pytest.approx(_ramp(t), abs=1e-6)
    assert _cell(geom, spec, 8.0, 148.5) == pytest.approx(2.5)
    assert 2.5 - _ramp(7.451) == pytest.approx(0.9, abs=0.01)
    dv = (geom.volume_cm3() - _fg9_geometry().volume_cm3()) * 1000.0
    assert 0.0 < dv < 297.0 * 0.6


def test_a_chamfer_that_would_cut_the_land_is_refused_not_built():
    at = _film_gate10_app()
    at.number_input(key="f10_段差の傾斜角 [deg]（製品側に削る斜面、90 = 段差のまま）").set_value(
        15.0
    )
    at.button[0].click().run()
    assert not at.exception
    assert any("too shallow" in e.value for e in at.error)
    assert "mfs_geom" not in at.session_state


def test_the_cut_default_does_not_leak_into_film_gate_9():
    at = _film_gate10_app()
    assert at.checkbox(key="f10_rc_on").value is True
    at.radio(key="geom_source").set_value(FILM_GATE9_LABEL).run()
    assert at.checkbox(key="f9_rc_on").value is False
    at.button[0].click().run()
    assert not at.exception
    assert (
        "ramp_cut" not in _recorded_spec(at, FILM_GATE9_LABEL)
        or _recorded_spec(at, FILM_GATE9_LABEL)["ramp_cut"] is None
    )
