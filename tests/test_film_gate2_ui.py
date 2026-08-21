"""AppTest wiring checks for the parametric Film gate 2 (肉厚調整ゲート・片側).

The one-sided sibling of Film gate 1: same slider family, ``symmetric=False``
(valve at the w=0 edge, widths measured from that edge). Its defaults
reproduce ``hamoko_gate_2bai_20260703``; the derived quantities differ from
Film gate 1 in that the outer-wall start width is the *full* exit width and
the valve default is the drawing's literal ``t`` rather than the well centre.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from core import GateProfileSpec, ProfilePlateConfig, build_profile_gate_geometry

APP = Path(__file__).resolve().parent.parent / "app.py"
FILM_GATE2_LABEL = "Film gate 2 (肉厚調整ゲート・片側)"

HAMOKO_2BAI_SPEC = {
    "name": "hamoko_gate_2bai_20260703",
    "units": "mm",
    "symmetric": False,
    "gate_exit_width": 299.0,
    "land": {"depth": 0.35, "length": 1.0},
    "main_ramp": {"angle_deg": 10.95, "cap_depth": 2.5},
    "island": {
        "angle_deg": 2.5,
        "boundary_line": [[1.0, 95.3], [17.0, 20.0]],
        "end_dist": 17.0,
    },
    "outer_wall_line": [[3.0, 299.0], [23.6, 4.45]],
    "well": {
        "shape": "obround",
        "t_range": [15.5, 27.5],
        "half_width": 4.5,
        "depth": 4.5,
        "floor_t_range": [18.1, 24.9],
        "wall_angle_deg": 60,
    },
    "valve": {"t": 20.0, "w": 0.0, "orifice_diameter": 3.0},
}


@pytest.fixture(scope="module")
def film_gate2_run() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=240.0)
    at.run()
    assert at.radio(key="geom_source").value == FILM_GATE2_LABEL  # the UI default input
    at.button[0].click().run()
    assert not at.exception
    return at


def _recorded_spec(at: AppTest) -> dict:
    geom = at.session_state["mfs_settings"]["geometry"]
    assert geom["input"] == FILM_GATE2_LABEL
    return geom["gate_profile"]


def test_default_sliders_reproduce_the_2bai_spec(film_gate2_run):
    rec = _recorded_spec(film_gate2_run)
    expected = GateProfileSpec.from_dict(HAMOKO_2BAI_SPEC)
    got = GateProfileSpec.from_dict({**rec, "name": HAMOKO_2BAI_SPEC["name"]})
    assert got.symmetric is False
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
    assert got.well.floor_t_range == pytest.approx(expected.well.floor_t_range, abs=0.05)
    assert got.valve == expected.valve  # t=20.0 is the drawing's, not the well centre (21.5)


def test_default_geometry_matches_the_2bai_spec_built_directly(film_gate2_run):
    geom = film_gate2_run.session_state["mfs_geom"]
    plate = ProfilePlateConfig(
        plate_w_mm=300.0,
        plate_h_mm=50.0,
        plate_thk_mm=0.35,
        plate_split_height_mm=20.0,
        plate_lower_thk_mm=0.35,
        plate_upper_thk_mm=0.50,
    )
    ref = build_profile_gate_geometry(GateProfileSpec.from_dict(HAMOKO_2BAI_SPEC), plate, 1.0)
    assert np.array_equal(geom.mask, ref.mask)
    assert np.array_equal(geom.thickness_mm[geom.mask], ref.thickness_mm[ref.mask])
    assert geom.gates == ref.gates


def test_the_gate_sits_at_the_valve_side_edge(film_gate2_run):
    """One-sided: the Dirichlet cells cluster at the w=0 edge, i.e. the left
    end of the gate exit, not the plate centre."""
    geom = film_gate2_run.session_state["mfs_geom"]
    ixs = np.array([ix for _, ix in geom.gates])
    x_edge_cell = int((5.0 + 300.0 / 2.0 - 299.0 / 2.0) / 1.0)
    assert abs(ixs.mean() - x_edge_cell) < 2.0
