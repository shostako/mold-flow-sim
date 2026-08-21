"""AppTest wiring checks for the parametric Film gate 1 (肉厚調整ゲート) input.

Film gate 1 assembles a ``GateProfileSpec`` from sliders and feeds it to the
same builder as the Profile gate JSON input. Its defaults reproduce the
``hamoko_gate_furiwake_20260703`` drawing, and the derived quantities (island
boundary-line t-endpoints, outer-wall start half-width, well floor range) are
tied to the major dimensions the way that drawing ties them -- so the
assembled spec must match the drawing's spec exactly, not just roughly.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from core import GateProfileSpec, ProfilePlateConfig, build_profile_gate_geometry

APP = Path(__file__).resolve().parent.parent / "app.py"
FILM_GATE1_LABEL = "Film gate 1 (肉厚調整ゲート)"

# The drawing-derived spec the sliders default to.
HAMOKO_SPEC = {
    "name": "hamoko_gate_furiwake_20260703",
    "units": "mm",
    "symmetric": True,
    "gate_exit_width": 298.0,
    "land": {"depth": 0.35, "length": 1.0},
    "main_ramp": {"angle_deg": 10.95, "cap_depth": 2.5},
    "island": {
        "angle_deg": 2.5,
        "boundary_line": [[1.0, 52.7], [17.0, 10.0]],
        "end_dist": 17.0,
    },
    "outer_wall_line": [[3.0, 149.0], [23.3, 4.5]],
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


@pytest.fixture(scope="module")
def film_gate1_run() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=240.0)
    at.run()
    at.radio(key="geom_source").set_value(FILM_GATE1_LABEL).run()
    at.radio(key="wall_model").set_value("none")
    at.button[0].click().run()
    assert not at.exception
    return at


def _recorded_spec(at: AppTest) -> dict:
    geom = at.session_state["mfs_settings"]["geometry"]
    assert geom["input"] == FILM_GATE1_LABEL
    return geom["gate_profile"]


def test_default_sliders_reproduce_the_hamoko_spec(film_gate1_run):
    rec = _recorded_spec(film_gate1_run)
    expected = GateProfileSpec.from_dict(HAMOKO_SPEC)
    got = GateProfileSpec.from_dict({**rec, "name": HAMOKO_SPEC["name"]})
    assert got.gate_exit_width == expected.gate_exit_width
    assert got.land == expected.land
    assert got.main_ramp == expected.main_ramp
    assert got.island.angle_deg == expected.island.angle_deg
    assert got.island.end_dist == expected.island.end_dist
    assert np.asarray(got.island.boundary_line) == pytest.approx(
        np.asarray(expected.island.boundary_line)
    )
    assert np.asarray(got.outer_wall_line) == pytest.approx(np.asarray(expected.outer_wall_line))
    assert got.well.t_range == expected.well.t_range
    assert got.well.half_width == expected.well.half_width
    assert got.well.depth == expected.well.depth
    assert got.well.wall_angle_deg == expected.well.wall_angle_deg
    # floor = t_range shrunk by depth / tan(60°) at each end
    eat = 4.5 / math.tan(math.radians(60.0))
    assert got.well.floor_t_range == pytest.approx((15.5 + eat, 27.5 - eat), abs=1e-9)
    assert got.well.floor_t_range == pytest.approx(expected.well.floor_t_range, abs=0.05)
    assert got.valve == expected.valve


def test_default_geometry_matches_the_hamoko_spec_built_directly(film_gate1_run):
    at = film_gate1_run
    geom = at.session_state["mfs_geom"]
    plate = ProfilePlateConfig(
        plate_w_mm=300.0,
        plate_h_mm=50.0,
        plate_thk_mm=0.35,
        plate_split_height_mm=20.0,
        plate_lower_thk_mm=0.35,
        plate_upper_thk_mm=0.50,
    )
    ref = build_profile_gate_geometry(GateProfileSpec.from_dict(HAMOKO_SPEC), plate, 1.0)
    assert geom.mask.shape == ref.mask.shape
    assert np.array_equal(geom.mask, ref.mask)
    # floor_t_range is reference metadata, so a 0.0x mm rounding difference in
    # it must not move the depth field at all
    assert np.array_equal(geom.thickness_mm[geom.mask], ref.thickness_mm[ref.mask])


def test_island_and_well_can_be_switched_off():
    at = AppTest.from_file(str(APP), default_timeout=240.0)
    at.run()
    at.radio(key="geom_source").set_value(FILM_GATE1_LABEL).run()
    at.checkbox(key="f1_island_on").set_value(False)
    at.checkbox(key="f1_well_on").set_value(False)
    at.radio(key="wall_model").set_value("none")
    at.button[0].click().run()
    assert not at.exception
    rec = _recorded_spec(at)
    assert rec["island"] is None
    assert rec["well"] is None


def _film_gate1_app() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=240.0)
    at.run()
    at.radio(key="geom_source").set_value(FILM_GATE1_LABEL).run()
    at.radio(key="wall_model").set_value("none")
    return at


def _slider(at: AppTest, label_prefix: str):
    hits = [s for s in at.slider if str(s.label).startswith(label_prefix)]
    assert len(hits) == 1, [str(s.label) for s in at.slider]
    return hits[0]


def test_well_depth_is_capped_by_what_the_sloped_wall_can_reach():
    """Codex P1: depth > half_width·tan(60°) is accepted by the spec, recorded
    in settings.json, and silently rasterised shallower. The slider's upper
    bound must follow the half-width."""
    at = _film_gate1_app()
    _slider(at, "井戸半幅").set_value(0.5).run()
    depth = _slider(at, "井戸深さ")
    assert depth.max <= 0.5 * math.tan(math.radians(60.0)) + 1e-9
    assert depth.value <= depth.max
    at.button[0].click().run()
    assert not at.exception
    rec = _recorded_spec(at)
    assert rec["well"]["depth"] <= 0.5 * math.tan(math.radians(60.0)) + 1e-9


def test_valve_position_is_bounded_by_the_pocket_end():
    """Codex P1: a valve placed past both the outer wall and the well misses
    the cavity, and the builder snaps the gate tens of mm away from the
    recorded ``t``. The slider must not offer such positions."""
    at = _film_gate1_app()
    valve = _slider(at, "バルブ位置")
    # defaults: outer wall ends at 23.3, well at 27.5, orifice Φ3 → ≤ 26.0
    assert valve.max == pytest.approx(27.5 - 1.5)
    assert valve.min == pytest.approx(1.5)
    # shrinking the well pulls the bound back to the outer wall end
    at.checkbox(key="f1_well_on").set_value(False).run()
    valve = _slider(at, "バルブ位置")
    assert valve.max == pytest.approx(23.3 - 1.5)
    at.button[0].click().run()
    assert not at.exception
    geom = at.session_state["mfs_geom"]
    rec = _recorded_spec(at)
    # the Dirichlet gate sits where the record says it does (no snapping)
    iy, ix = geom.gates[0]
    y_gate = (iy + 0.5) * geom.cell_size_mm
    t_gate = 5.0 + GateProfileSpec.from_dict({**rec, "name": "x"}).t_max() - y_gate
    assert abs(t_gate - rec["valve"]["t"]) <= rec["valve"]["orifice_diameter"] / 2.0 + 1.0


def test_uniform_plate_thickness_branch_records_no_split():
    at = _film_gate1_app()
    at.checkbox(key="f1_plate_2layer").set_value(False).run()
    at.button[0].click().run()
    assert not at.exception
    cfg = at.session_state["mfs_settings"]["geometry"]["config"]
    assert cfg["plate_split_height_mm"] == 0.0
    assert cfg["plate_lower_thk_mm"] is None and cfg["plate_upper_thk_mm"] is None
    geom = at.session_state["mfs_geom"]
    plate = geom.compression_mask & geom.mask
    assert np.unique(geom.thickness_mm[plate]).size == 1


def test_a_well_too_short_for_a_flat_floor_reports_no_floor_range():
    """floor = t_range shrunk by depth/tan(60°) at each end; when that
    inverts, the builder must not hand validate() an inverted range."""
    at = _film_gate1_app()
    _slider(at, "井戸開始").set_value(20.0).run()
    _slider(at, "井戸終端").set_value(21.0).run()  # 1 mm long, depth 4.5 eats 2.6 per end
    at.button[0].click().run()
    assert not at.exception
    rec = _recorded_spec(at)
    assert rec["well"]["t_range"] == [20.0, 21.0]
    assert rec["well"]["floor_t_range"] is None


def test_a_valve_centre_outside_the_pocket_is_rejected_not_snapped(monkeypatch):
    """The post-build guard in app.py. Slider bounds make this unreachable by
    hand (the centreline w=0 is always inside a pocket whose half-width is
    ≥ 0.5 mm), so the miss is injected: the real builder runs, then the
    valve-centre cell is cut out of the mask. ``app.py`` re-imports
    ``build_profile_gate_geometry`` from ``core`` on every AppTest run, so
    patching the module attribute reaches it."""
    import core

    real = core.build_profile_gate_geometry

    def cut_out_valve_cell(spec, plate, cell_size_mm=1.0):
        geom = real(spec, plate, cell_size_mm=cell_size_mm)
        iy = int((plate.pad_mm + spec.t_max() - spec.valve.t) / cell_size_mm)
        ix = int((plate.pad_mm + plate.plate_w_mm / 2.0) / cell_size_mm)
        assert geom.mask[iy, ix], "the injected miss must start from a hit"
        geom.mask[iy, ix] = False
        return geom

    monkeypatch.setattr(core, "build_profile_gate_geometry", cut_out_valve_cell)
    at = _film_gate1_app()
    at.button[0].click().run()
    assert not at.exception
    errors = "\n".join(str(e.value) for e in at.error)
    assert "バルブ位置" in errors and "ポケットの外" in errors
    assert "mfs_geom" not in at.session_state
