"""AppTest wiring checks for the parametric Film gate 11 (扇状/ランド両端肉厚可変).

The 2026/09/23 candidate A「ランド端部増厚」
(``hamoko_gate_furiwake_cand_A_landends05_20260923``): Film gate 9's pocket
with both ends of the land, 49 each (|w| ≥ 100, the central 200 untouched),
milled with a flat 0.5 below the PL. The ramp face is not touched, so the
flat runs on to t = 1 + 0.15/tan(10.95°) = 1.78, where the ramp reaches 0.5,
and meets it without a step. The depth of the flat is the design variable.

What the tests pin: the defaults reproduce the spec, the built field carries
the flat (end column 0.5 on the land rows, ramp beyond; centre 0.35), against
Film gate 9 it only deepens cells and only at |w| ≥ 100, the sliders move the
flat (deeper → it reaches the next ramp row; wider centre → w_from moves),
switching it off gives Film gate 9, and the default does not leak into Film
gate 9 / 10.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from core import GateProfileSpec, LandEndsSpec, ProfilePlateConfig, build_profile_gate_geometry

APP = Path(__file__).resolve().parent.parent / "app.py"
FILM_GATE11_LABEL = "Film gate 11 (扇状/ランド両端肉厚可変)"
FILM_GATE9_LABEL = "Film gate 9 (扇状/9月末試作)"
FILM_GATE10_LABEL = "Film gate 10 (扇状/肉厚調整 0923)"

DEPTH_KEY = "f11_増厚後のランド深さ [mm] (> ランド深さ、≤ ランプ上限)"
CENTRE_KEY = "f11_中央の現状幅（両側合計） [mm]"

HAMOKO_CAND_A_SPEC = {
    "name": "hamoko_gate_furiwake_cand_A_landends05_20260923",
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
    "land_ends": {"w_from": 100.0, "depth": 0.5},
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


def _film_gate11_app() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=240.0)
    at.run()
    at.radio(key="geom_source").set_value(FILM_GATE11_LABEL).run()
    at.radio(key="wall_model").set_value("none")
    return at


@pytest.fixture(scope="module")
def film_gate11_run() -> AppTest:
    at = _film_gate11_app()
    at.button[0].click().run()
    assert not at.exception
    return at


def _recorded_spec(at: AppTest, label: str = FILM_GATE11_LABEL) -> dict:
    geom = at.session_state["mfs_settings"]["geometry"]
    assert geom["input"] == label
    return geom["gate_profile"]


def _fg9_geometry():
    d = {k: v for k, v in HAMOKO_CAND_A_SPEC.items() if k != "land_ends"}
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


def _wa(geom) -> np.ndarray:
    _iy, ix = np.indices(geom.shape)
    return np.abs((ix + 0.5) - (ProfilePlateConfig().pad_mm + PLATE.plate_w_mm / 2.0))


def test_default_sliders_reproduce_the_candidate_a_spec(film_gate11_run):
    rec = _recorded_spec(film_gate11_run)
    expected = GateProfileSpec.from_dict(HAMOKO_CAND_A_SPEC)
    got = GateProfileSpec.from_dict({**rec, "name": expected.name})
    assert got.symmetric is True
    assert got.land == expected.land and got.main_ramp == expected.main_ramp
    assert np.asarray(got.outer_wall_line) == pytest.approx(np.asarray(expected.outer_wall_line))
    # the island (pinned at t = land length, same line as the drawing's
    # (0, 50)) and the well floor (derived from the 60° wall, 18.098 vs the
    # drawing's rounded 18.1) are covered by the bit-identical geometry test
    assert got.well.t_range == expected.well.t_range
    assert got.well.wall_angle_deg == 60.0
    assert got.valve == expected.valve
    assert got.edge_channels == () and got.ramp_cut is None
    # the one feature this drawing adds
    assert got.land_ends == LandEndsSpec(w_from=100.0, depth=0.5)


def test_default_geometry_matches_the_spec_built_directly(film_gate11_run):
    geom = film_gate11_run.session_state["mfs_geom"]
    ref = build_profile_gate_geometry(GateProfileSpec.from_dict(HAMOKO_CAND_A_SPEC), PLATE, 1.0)
    assert np.array_equal(geom.mask, ref.mask)
    assert np.array_equal(geom.thickness_mm[geom.mask], ref.thickness_mm[ref.mask])
    assert geom.gates == ref.gates


def test_the_flat_is_in_the_thickness_field(film_gate11_run):
    """At 1.0 mm the land is the single row t=1 (t=0 is the plate's bottom
    row); the next row t=2 is ramp 0.543, already deeper than 0.5 (the flat
    ends at t=1.78). End column (w=148.5) and just outside the centre
    (w=100.5): 0.5 on the land row, the ramp beyond, the plate untouched.
    Just inside (w=99.5): the original 0.35."""
    geom = film_gate11_run.session_state["mfs_geom"]
    spec = GateProfileSpec.from_dict(_recorded_spec(film_gate11_run))
    for w in (148.5, 100.5):
        assert _cell(geom, spec, 0.0, w) == pytest.approx(0.35)  # plate
        assert _cell(geom, spec, 1.0, w) == pytest.approx(0.5)
        for t in (2.0, 5.0):
            assert _cell(geom, spec, t, w) == pytest.approx(_ramp(t), abs=1e-6)
    assert _cell(geom, spec, 1.0, 99.5) == pytest.approx(0.35)
    assert 1.0 + 0.15 / math.tan(math.radians(10.95)) == pytest.approx(1.78, abs=0.01)


def test_against_film_gate_9_only_the_land_ends_change(film_gate11_run):
    geom = film_gate11_run.session_state["mfs_geom"]
    fg9 = _fg9_geometry()
    assert np.array_equal(geom.mask, fg9.mask)
    assert np.all(geom.thickness_mm >= fg9.thickness_mm)
    diff = geom.thickness_mm != fg9.thickness_mm
    wa = _wa(geom)
    # 49 columns each side × the one land row, each 0.35 → 0.5
    assert wa[diff].min() >= 100.0 and wa[diff].max() <= 149.0
    assert int(diff.sum()) == 2 * 49
    assert np.allclose(fg9.thickness_mm[diff], 0.35) and np.allclose(geom.thickness_mm[diff], 0.5)


def test_the_sliders_move_the_flat():
    """0.6 deep: the flat reaches t = 1 + 0.25/tan(10.95°) = 2.29, so the
    t=2 row (ramp 0.543) is milled too. A 240 centre moves w_from to 120."""
    at = _film_gate11_app()
    at.slider(key=DEPTH_KEY).set_value(0.6)
    at.slider(key=CENTRE_KEY).set_value(240.0)
    at.button[0].click().run()
    assert not at.exception
    spec = GateProfileSpec.from_dict(_recorded_spec(at))
    assert spec.land_ends == LandEndsSpec(w_from=120.0, depth=0.6)
    geom = at.session_state["mfs_geom"]
    for t in (1.0, 2.0):
        assert _cell(geom, spec, t, 148.5) == pytest.approx(0.6)
    assert _cell(geom, spec, 3.0, 148.5) == pytest.approx(_ramp(3.0), abs=1e-6)
    assert _cell(geom, spec, 1.0, 119.5) == pytest.approx(0.35)
    assert _cell(geom, spec, 1.0, 120.5) == pytest.approx(0.6)


def test_switched_off_it_is_film_gate_9():
    at = _film_gate11_app()
    at.checkbox(key="f11_le_on").uncheck().run()
    at.button[0].click().run()
    assert not at.exception
    rec = _recorded_spec(at)
    assert rec.get("land_ends") is None
    geom = at.session_state["mfs_geom"]
    fg9 = _fg9_geometry()
    assert np.array_equal(geom.mask, fg9.mask)
    assert np.array_equal(geom.thickness_mm[geom.mask], fg9.thickness_mm[fg9.mask])


def test_the_default_does_not_leak_into_film_gates_9_and_10():
    at = _film_gate11_app()
    assert at.checkbox(key="f11_le_on").value is True
    for label, tag in ((FILM_GATE9_LABEL, "f9"), (FILM_GATE10_LABEL, "f10")):
        at.radio(key="geom_source").set_value(label).run()
        assert at.checkbox(key=f"{tag}_le_on").value is False
    at.radio(key="geom_source").set_value(FILM_GATE9_LABEL).run()
    at.button[0].click().run()
    assert not at.exception
    assert _recorded_spec(at, FILM_GATE9_LABEL).get("land_ends") is None


def test_with_the_cap_at_the_land_depth_the_block_is_disabled_not_broken():
    """「ランプ上限深さ」may equal「ランド深さ」. Then no flat depth is both
    deeper than the land and within the cap: the checkbox is disabled, the
    run succeeds without land_ends, and nothing fails validation (Codex P2
    on PR #93: the slider used to offer only values above the cap)."""
    at = _film_gate11_app()
    at.slider(key="f11_ランプ上限深さ [mm] (≥ ランド深さ)").set_value(0.35).run()
    box = at.checkbox(key="f11_le_on")
    assert box.disabled is True
    assert not any(s.key == DEPTH_KEY for s in at.slider)
    at.button[0].click().run()
    assert not at.exception
    assert not at.error
    assert _recorded_spec(at).get("land_ends") is None
    # raising the cap again brings the block back, still on by default
    at.slider(key="f11_ランプ上限深さ [mm] (≥ ランド深さ)").set_value(2.5).run()
    assert at.checkbox(key="f11_le_on").disabled is False
    assert at.slider(key=DEPTH_KEY).value == pytest.approx(0.5)
