"""AppTest wiring checks for the parametric Film gate 9 (扇状/9月末試作).

The 2026/09/14 runner proposal (``hamoko_gate_furiwake_runner_20260914``):
the 08/07 rework pocket with one line moved. The outer wall used to leave
the pocket end near the land and run at 8° to the well; the proposal starts
it at t=15.736 and runs it at 3° to the same end point. Behind the ramp the
pocket is therefore a nearly full-width bar at the ramp cap depth (2.5) —
the depth-2.5 line t=12.11, which used to stop at the wall, now reaches the
pocket ends. What the defaults have to carry is just that wall start (and
the rework's well wall angle); the tests pin the bar itself in the built
field, against the rework pocket built directly, not only the recorded spec.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from core import GateProfileSpec, ProfilePlateConfig, build_profile_gate_geometry

APP = Path(__file__).resolve().parent.parent / "app.py"
FILM_GATE9_LABEL = "Film gate 9 (扇状/9月末試作)"
FILM_GATE6_LABEL = "Film gate 6 (扇状/縁部深彫り 0807)"

HAMOKO_RUNNER_0914_SPEC = {
    "name": "hamoko_gate_furiwake_runner_20260914",
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
    "well": {
        "shape": "obround",
        "t_range": [15.5, 27.5],
        "half_width": 4.5,
        "depth": 4.5,
        "floor_t_range": [17.0, 26.0],
        "wall_angle_deg": 71.6,
    },
    "valve": {"t": 21.5, "w": 0.0, "orifice_diameter": 3.0},
}
# The pocket this proposal starts from: same everything, wall from the end.
REWORK_WALL = [[5.0, 149.0], [23.28, 4.48]]

PLATE = ProfilePlateConfig(
    plate_w_mm=300.0,
    plate_h_mm=50.0,
    plate_thk_mm=0.35,
    plate_split_height_mm=20.0,
    plate_lower_thk_mm=0.35,
    plate_upper_thk_mm=0.50,
)


def _film_gate9_app() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=240.0)
    at.run()
    at.radio(key="geom_source").set_value(FILM_GATE9_LABEL).run()
    at.radio(key="wall_model").set_value("none")
    return at


@pytest.fixture(scope="module")
def film_gate9_run() -> AppTest:
    at = _film_gate9_app()
    at.button[0].click().run()
    assert not at.exception
    return at


def _recorded_spec(at: AppTest, label: str = FILM_GATE9_LABEL) -> dict:
    geom = at.session_state["mfs_settings"]["geometry"]
    assert geom["input"] == label
    return geom["gate_profile"]


def _cell(geom, spec: GateProfileSpec, t: float, w: float, dx: float = 1.0) -> float | None:
    """Thickness of the cell centred at gate coordinates (t, w); None = steel.

    Same mapping as the builder: t = y_plate_bottom − y_centre with
    y_plate_bottom = pad + t_max, w from the plate centre (``valve.w == 0``).
    With t_max = 27.5 the centres sit at integer t and half-integer w; the
    helper insists the request hits one.
    """
    pad = ProfilePlateConfig().pad_mm
    iy = (pad + spec.t_max() - t) / dx - 0.5
    ix = (pad + PLATE.plate_w_mm / 2.0 + w) / dx - 0.5
    assert abs(iy - round(iy)) < 1e-6 and abs(ix - round(ix)) < 1e-6, (t, w)
    iy, ix = int(round(iy)), int(round(ix))
    return float(geom.thickness_mm[iy, ix]) if geom.mask[iy, ix] else None


def _ramp(t: float) -> float:
    return min(2.5, 0.35 + math.tan(math.radians(10.95)) * (t - 1.0))


def test_default_sliders_reproduce_the_runner_proposal_spec(film_gate9_run):
    rec = _recorded_spec(film_gate9_run)
    expected = GateProfileSpec.from_dict(HAMOKO_RUNNER_0914_SPEC)
    got = GateProfileSpec.from_dict({**rec, "name": HAMOKO_RUNNER_0914_SPEC["name"]})
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
    # the one line this drawing moves — exact, not rounded to the slider step
    assert np.asarray(got.outer_wall_line) == pytest.approx(np.asarray(expected.outer_wall_line))
    assert got.outer_wall_line[0][0] == 15.736
    assert got.well.t_range == expected.well.t_range
    assert got.well.half_width == expected.well.half_width
    assert got.well.depth == expected.well.depth
    assert got.well.wall_angle_deg == expected.well.wall_angle_deg
    assert got.well.floor_t_range == pytest.approx(expected.well.floor_t_range, abs=0.05)
    assert got.valve == expected.valve  # 21.5 = the well centre
    assert got.edge_channels == ()


def test_the_new_wall_runs_at_three_degrees(film_gate9_run):
    """The drawing dimensions the new line as 3° to the gate edge (and 5° to
    the current 8° line). The recorded wall has to be that line."""
    (t1, w1), (t2, w2) = _recorded_spec(film_gate9_run)["outer_wall_line"]
    assert math.degrees(math.atan2(t2 - t1, w1 - w2)) == pytest.approx(3.0, abs=0.05)


def test_default_geometry_matches_the_spec_built_directly(film_gate9_run):
    geom = film_gate9_run.session_state["mfs_geom"]
    ref = build_profile_gate_geometry(
        GateProfileSpec.from_dict(HAMOKO_RUNNER_0914_SPEC), PLATE, 1.0
    )
    assert np.array_equal(geom.mask, ref.mask)
    assert np.array_equal(geom.thickness_mm[geom.mask], ref.thickness_mm[ref.mask])
    assert geom.gates == ref.gates


def test_the_back_of_the_pocket_is_a_full_width_bar_at_cap_depth(film_gate9_run):
    """At the pocket end (w=148.5) the wall sits at t≈15.76: the ramp runs
    its full length there and t=13..15 are 2.5 deep, t=16 is steel. Half way
    in (w=100.5) the wall is at t≈18.27. The rework pocket has steel at all
    of those end cells (its wall passes w=148.5 at t≈5.06)."""
    geom = film_gate9_run.session_state["mfs_geom"]
    spec = GateProfileSpec.from_dict(_recorded_spec(film_gate9_run))
    assert _cell(geom, spec, 1.0, 148.5) == pytest.approx(0.35)
    for t in (2.0, 6.0, 12.0):
        assert _cell(geom, spec, t, 148.5) == pytest.approx(_ramp(t), abs=1e-6)
    for t in (13.0, 14.0, 15.0):
        assert _cell(geom, spec, t, 148.5) == pytest.approx(2.5)
    assert _cell(geom, spec, 16.0, 148.5) is None
    assert _cell(geom, spec, 18.0, 100.5) == pytest.approx(2.5)
    assert _cell(geom, spec, 19.0, 100.5) is None

    rework = build_profile_gate_geometry(
        GateProfileSpec.from_dict({**HAMOKO_RUNNER_0914_SPEC, "outer_wall_line": REWORK_WALL}),
        PLATE,
        1.0,
    )
    assert _cell(rework, spec, 6.0, 148.5) is None
    assert _cell(rework, spec, 15.0, 148.5) is None


def test_the_proposal_only_adds_cells_to_the_rework_pocket(film_gate9_run):
    """Moving the wall outward must not touch a single cell the rework pocket
    already had, and every added cell is plain ramp/cap depth — the 肉盗み and
    the well sit inside the old wall, so neither can show up in the
    difference."""
    geom = film_gate9_run.session_state["mfs_geom"]
    rework = build_profile_gate_geometry(
        GateProfileSpec.from_dict({**HAMOKO_RUNNER_0914_SPEC, "outer_wall_line": REWORK_WALL}),
        PLATE,
        1.0,
    )
    assert geom.mask.shape == rework.mask.shape
    assert np.all(geom.mask[rework.mask])
    assert np.array_equal(geom.thickness_mm[rework.mask], rework.thickness_mm[rework.mask])
    added = geom.mask & ~rework.mask
    # two triangles between the 8° and the 3° line: ≈ 2 · ½ · 144.5 · 10.7 mm²
    assert added.sum() == pytest.approx(144.52 * (15.736 - 5.0), rel=0.03)
    depths = np.unique(np.round(geom.thickness_mm[added], 6))
    allowed = {round(_ramp(float(t)), 6) for t in range(5, 24)}
    assert set(depths.tolist()) <= allowed
    # most of the added steel-off is the 2.5 bar, not ramp
    assert np.mean(np.isclose(geom.thickness_mm[added], 2.5)) > 0.5


def test_the_wall_start_does_not_leak_into_film_gate_6():
    """Separate ``f9_`` / ``f6_`` widget keys: after visiting 9, Film gate 6
    still starts from its own wall (t=5) with its band on, and 9 started with
    the band off."""
    at = _film_gate9_app()
    assert at.checkbox(key="f9_ec_on").value is False
    at.radio(key="geom_source").set_value(FILM_GATE6_LABEL).run()
    assert at.checkbox(key="f6_ec_on").value is True
    at.button[0].click().run()
    assert not at.exception
    got = GateProfileSpec.from_dict({**_recorded_spec(at, FILM_GATE6_LABEL), "name": "x"})
    assert np.asarray(got.outer_wall_line[0]) == pytest.approx((5.0, 149.0))
    assert len(got.edge_channels) == 1
