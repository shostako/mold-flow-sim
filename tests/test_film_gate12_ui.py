"""AppTest wiring checks for the parametric Film gate 12 (扇状/斜面角度徐変).

The 2026/09/28 CAD model ``Runner-block_3D_00.igs``
(``hamoko_gate_furiwake_rampends_20260928``): Film gate 9's pocket with the
ramp angle graded over both ends. In |w| ≥ 60 the depth-2.5 line bends from
t=12 (the main ramp, 11.06°) to t=4 at the pocket end (atan(2.15/3) = 35.6°),
and at every w the ramp is the straight section from the land end to that
line — the bilinear patch the CAD model carries. The central 120 keeps the
plane ramp. The other dimensions are the CAD model's, which differ from the
9/14 PDF reading of Film gate 9 by < 0.11 mm.

What the tests pin: the defaults reproduce the spec (bit-identical geometry),
the built field carries the graded ramp (end columns against the ruled
surface written out here), against the same pocket without grading it only
deepens the ramp at |w| ≥ 60, the sliders move it, switching it off gives the
ungraded pocket, the default does not leak into Film gate 9 / 11 (neither the
grading nor the 11.06° ramp), and with no ramp to grade the block is disabled
instead of failing the run.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from core import GateProfileSpec, ProfilePlateConfig, RampEndsSpec, build_profile_gate_geometry

APP = Path(__file__).resolve().parent.parent / "app.py"
FILM_GATE12_LABEL = "Film gate 12 (扇状/斜面角度徐変)"
FILM_GATE9_LABEL = "Film gate 9 (扇状/9月末試作)"
FILM_GATE11_LABEL = "Film gate 11 (扇状/ランド両端肉厚可変)"

CENTRE_KEY = "f12_中央の一定角の幅（両側合計） [mm]"
T_END_KEY = "f12_ポケット端でランプが上限深さに達する t [mm] (< メインランプの到達 t)"

# The CAD model's main ramp: land end (t=1, 0.35) to the cap 2.5 at t=12.
RAMP_DEG = math.degrees(math.atan(2.15 / 11.0))

HAMOKO_RAMPENDS_SPEC = {
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

PLATE = ProfilePlateConfig(
    plate_w_mm=300.0,
    plate_h_mm=50.0,
    plate_thk_mm=0.35,
    plate_split_height_mm=20.0,
    plate_lower_thk_mm=0.35,
    plate_upper_thk_mm=0.50,
)


def _film_gate12_app() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=240.0)
    at.run()
    at.radio(key="geom_source").set_value(FILM_GATE12_LABEL).run()
    at.radio(key="wall_model").set_value("none")
    return at


@pytest.fixture(scope="module")
def film_gate12_run() -> AppTest:
    at = _film_gate12_app()
    at.button[0].click().run()
    assert not at.exception
    return at


def _recorded_spec(at: AppTest, label: str = FILM_GATE12_LABEL) -> dict:
    geom = at.session_state["mfs_settings"]["geometry"]
    assert geom["input"] == label
    return geom["gate_profile"]


def _ungraded_geometry():
    d = {k: v for k, v in HAMOKO_RAMPENDS_SPEC.items() if k != "ramp_ends"}
    return build_profile_gate_geometry(GateProfileSpec.from_dict(d), PLATE, 1.0)


def _cell(geom, spec: GateProfileSpec, t: float, w: float, dx: float = 1.0) -> float | None:
    """Thickness of the cell centred at (t, w); None = steel (see test_film_gate9_ui)."""
    pad = ProfilePlateConfig().pad_mm
    iy = (pad + spec.t_max() - t) / dx - 0.5
    ix = (pad + PLATE.plate_w_mm / 2.0 + w) / dx - 0.5
    assert abs(iy - round(iy)) < 1e-6 and abs(ix - round(ix)) < 1e-6, (t, w)
    iy, ix = int(round(iy)), int(round(ix))
    return float(geom.thickness_mm[iy, ix]) if geom.mask[iy, ix] else None


def _graded(t: float, w: float, *, w_from: float = 60.0, t_end: float = 4.0) -> float:
    """The ruled surface: straight ramp from (1, 0.35) to (t_cap(w), 2.5)."""
    if t <= 1.0:
        return 0.35
    t_cap = 12.0
    if w >= w_from:
        t_cap = 12.0 + (t_end - 12.0) * (min(w, 149.0) - w_from) / (149.0 - w_from)
    return min(2.5, 0.35 + 2.15 * (t - 1.0) / (t_cap - 1.0))


def _wa(geom) -> np.ndarray:
    _iy, ix = np.indices(geom.shape)
    return np.abs((ix + 0.5) - (ProfilePlateConfig().pad_mm + PLATE.plate_w_mm / 2.0))


def _t(geom, spec: GateProfileSpec) -> np.ndarray:
    iy, _ix = np.indices(geom.shape)
    return (ProfilePlateConfig().pad_mm + spec.t_max()) - (iy + 0.5)


def test_default_sliders_reproduce_the_cad_spec(film_gate12_run):
    rec = _recorded_spec(film_gate12_run)
    expected = GateProfileSpec.from_dict(HAMOKO_RAMPENDS_SPEC)
    got = GateProfileSpec.from_dict({**rec, "name": expected.name})
    assert got.symmetric is True
    assert got.land == expected.land
    assert got.main_ramp.angle_deg == pytest.approx(RAMP_DEG, abs=1e-12)
    assert got.ramp_cap_t() == pytest.approx(12.0, abs=1e-9)
    assert np.asarray(got.outer_wall_line) == pytest.approx(np.asarray(expected.outer_wall_line))
    assert np.asarray(got.island.boundary_line) == pytest.approx(
        np.asarray(expected.island.boundary_line)
    )
    assert got.well.t_range == expected.well.t_range
    assert got.well.wall_angle_deg == 60.0
    assert got.valve == expected.valve
    assert got.edge_channels == () and got.ramp_cut is None and got.land_ends is None
    # the one feature this drawing adds
    assert got.ramp_ends == RampEndsSpec(w_from=60.0, t_end=4.0)


def test_default_geometry_matches_the_spec_built_directly(film_gate12_run):
    geom = film_gate12_run.session_state["mfs_geom"]
    ref = build_profile_gate_geometry(GateProfileSpec.from_dict(HAMOKO_RAMPENDS_SPEC), PLATE, 1.0)
    assert np.array_equal(geom.mask, ref.mask)
    assert np.array_equal(geom.thickness_mm[geom.mask], ref.thickness_mm[ref.mask])
    assert geom.gates == ref.gates


def test_the_graded_ramp_is_in_the_thickness_field(film_gate12_run):
    """At 1.0 mm the land is the row t=1 and the ramp rows start at t=2.
    End column (w=148.5): the cap line is at t=4.045, so t=2, 3, 4 read
    1.056 / 1.762 / 2.468 (the main ramp there would be 0.545 / 0.741 /
    0.936) and t=5 is the cap. w=100.5 is part way; w=59.5 is the plane
    ramp."""
    geom = film_gate12_run.session_state["mfs_geom"]
    spec = GateProfileSpec.from_dict(_recorded_spec(film_gate12_run))
    for w in (148.5, 120.5, 100.5, 60.5, 59.5):
        assert _cell(geom, spec, 1.0, w) == pytest.approx(0.35)
        for t in (2.0, 3.0, 4.0, 5.0, 8.0):
            assert _cell(geom, spec, t, w) == pytest.approx(_graded(t, w), abs=1e-9), (t, w)
    assert _cell(geom, spec, 2.0, 148.5) == pytest.approx(1.0561, abs=1e-4)
    assert _cell(geom, spec, 2.0, 59.5) == pytest.approx(0.5455, abs=1e-4)


def test_against_the_ungraded_pocket_only_the_ramp_ends_deepen(film_gate12_run):
    geom = film_gate12_run.session_state["mfs_geom"]
    spec = GateProfileSpec.from_dict(_recorded_spec(film_gate12_run))
    flat = _ungraded_geometry()
    assert np.array_equal(geom.mask, flat.mask)
    assert np.all(geom.thickness_mm >= flat.thickness_mm)
    diff = geom.thickness_mm != flat.thickness_mm
    wa, t = _wa(geom), _t(geom, spec)
    assert diff.any()
    assert wa[diff].min() >= 60.0 and wa[diff].max() <= 149.0
    assert t[diff].min() > 1.0 and t[diff].max() < 12.0


def test_the_sliders_move_the_grading():
    """Centre 200 moves w_from to 100; t_end 6.0 puts the end angle at
    atan(2.15/5) = 23.3°."""
    at = _film_gate12_app()
    at.slider(key=CENTRE_KEY).set_value(200.0)
    at.slider(key=T_END_KEY).set_value(6.0)
    at.button[0].click().run()
    assert not at.exception
    spec = GateProfileSpec.from_dict(_recorded_spec(at))
    assert spec.ramp_ends == RampEndsSpec(w_from=100.0, t_end=6.0)
    geom = at.session_state["mfs_geom"]
    for w in (148.5, 120.5, 100.5, 99.5):
        for t in (2.0, 4.0, 6.0):
            want = _graded(t, w, w_from=100.0, t_end=6.0)
            assert _cell(geom, spec, t, w) == pytest.approx(want, abs=1e-9), (t, w)
    caption = " ".join(str(c.value) for c in at.caption)
    assert "23.27°（ポケット端）" in caption


def test_switched_off_it_is_the_ungraded_pocket():
    at = _film_gate12_app()
    at.checkbox(key="f12_re_on").uncheck().run()
    at.button[0].click().run()
    assert not at.exception
    assert _recorded_spec(at).get("ramp_ends") is None
    geom = at.session_state["mfs_geom"]
    flat = _ungraded_geometry()
    assert np.array_equal(geom.mask, flat.mask)
    assert np.array_equal(geom.thickness_mm[geom.mask], flat.thickness_mm[flat.mask])


def test_the_defaults_do_not_leak_into_film_gates_9_and_11():
    at = _film_gate12_app()
    assert at.checkbox(key="f12_re_on").value is True
    assert at.number_input(key="f12_ランプ角 [deg]").value == pytest.approx(RAMP_DEG)
    for label, tag in ((FILM_GATE9_LABEL, "f9"), (FILM_GATE11_LABEL, "f11")):
        at.radio(key="geom_source").set_value(label).run()
        assert at.checkbox(key=f"{tag}_re_on").value is False
        assert at.number_input(key=f"{tag}_ランプ角 [deg]").value == pytest.approx(10.95)
    at.radio(key="geom_source").set_value(FILM_GATE9_LABEL).run()
    at.button[0].click().run()
    assert not at.exception
    assert _recorded_spec(at, FILM_GATE9_LABEL).get("ramp_ends") is None


def test_with_the_cap_at_the_land_depth_the_block_is_disabled_not_broken():
    """With「ランプ上限深さ」=「ランド深さ」there is no ramp to grade: the
    checkbox is disabled, its sliders are gone, and the run succeeds without
    ramp_ends (a slider there would offer only values the build rejects)."""
    at = _film_gate12_app()
    at.slider(key="f12_ランプ上限深さ [mm] (≥ ランド深さ)").set_value(0.35).run()
    assert at.checkbox(key="f12_re_on").disabled is True
    assert not any(s.key == T_END_KEY for s in at.slider)
    at.button[0].click().run()
    assert not at.exception
    assert not at.error
    assert _recorded_spec(at).get("ramp_ends") is None
    at.slider(key="f12_ランプ上限深さ [mm] (≥ ランド深さ)").set_value(2.5).run()
    assert at.checkbox(key="f12_re_on").disabled is False
    assert at.slider(key=T_END_KEY).value == pytest.approx(4.0)
