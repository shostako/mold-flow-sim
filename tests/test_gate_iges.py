"""IGES of the gate block (``core/gate_iges.py``).

What the tests pin: the solid read back from above is the solver's own
builder cell for cell (outline and depth) for every feature the spec has,
alone and together; its volume is the raster's; the file reads back into
the same closed solid; the frame is that of the received CAD models; and
the check that guards the download actually tells a wrong solid from the
right one.

The solids need OCP (the ``cad`` extra); without it the module still
imports and says so.
"""

from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from core import gate_iges as gi
from core.gate_drawing import drawing_field
from core.profile_gate import GateProfileSpec
from tests.test_gate_drawing import FULL, LANDHANGER, PLATE

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "gate_profiles"

needs_ocp = pytest.mark.skipif(not gi.available(), reason="OCP (the cad extra) is not installed")

#: The landhanger block without the land profile (Film gate 9-like).
PLAIN = {k: v for k, v in LANDHANGER.items() if k != "land"} | {
    "land": {"depth": 0.35, "length": 1.0}
}

#: Film gate 3's one-sided block (the 0703 2bai drawing, the app's default).
ONE_SIDED = {
    "name": "one-sided",
    "units": "mm",
    "symmetric": False,
    "gate_exit_width": 299.0,
    "land": {"depth": 0.35, "length": 1.0},
    "main_ramp": {"angle_deg": 10.95, "cap_depth": 2.5},
    "island": {"angle_deg": 2.5, "boundary_line": [[1.0, 95.3], [17.0, 20.0]], "end_dist": 17.0},
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

_SPLITTER = {
    "angle_deg": 0.0,
    "inner_line": [[1.0, 0.0], [2.0, 0.0]],
    "outer_line": [[1.0, 2.5], [2.0, 0.01]],
    "end_dist": 2.0,
    "floor_depth": 0.35,
}
#: Film gate 8's T gate: two rectangular fans (bar and stem), a flat splitter on each.
T_GATE = {
    "name": "T",
    "units": "mm",
    "symmetric": True,
    "gate_exit_width": 298.0,
    "land": {"depth": 0.35, "length": 1.0},
    "main_ramp": {"angle_deg": 58.78, "cap_depth": 2.0},
    "sub_gates": [
        {
            "inner_wall_line": [[1.0, 0.0], [4.0, 0.0]],
            "outer_wall_line": [[1.0, 149.0], [4.0, 149.0]],
            "tip_t": 4.0,
            "island": _SPLITTER,
        },
        {
            "inner_wall_line": [[1.0, 0.0], [23.0, 0.0]],
            "outer_wall_line": [[1.0, 2.5], [23.0, 2.5]],
            "tip_t": 23.0,
            "island": _SPLITTER,
        },
    ],
    "well": LANDHANGER["well"],
    "valve": {"t": 21.5, "w": 0.0, "orifice_diameter": 3.0},
}


def _with(base: dict, **changes) -> dict:
    d = copy.deepcopy(base)
    for path, value in changes.items():
        node = d
        *head, last = path.split("__")
        for key in head:
            node = node[key]
        node[last] = value
    return d


def _twin_with_edges() -> dict:
    d = json.loads((DATA / "demo_twin_fan_gate.json").read_text())
    d["sub_gates"][0]["edge_channels"] = [
        {"width": 1.5, "depth": 2.0, "side": "outer"},
        {"width": 1.5, "depth": 2.0, "side": "inner"},
    ]
    return d


_CUT_LINE = [[7.451, 149.0], [12.0, 60.0]]
CASES = {
    "plain": PLAIN,
    "land profile (power 2)": LANDHANGER,
    "land profile (power 0.5)": _with(LANDHANGER, land__profile__power=0.5),
    "land profile (power 1.5)": _with(LANDHANGER, land__profile__power=1.5),
    "closure + profile + round + graded ends + island + weld + edge channel": FULL,
    "weld down to the PL": _with(FULL, island__weld__depth=0.0),
    "graded ends without the end depth": _with(FULL, ramp_ends__depth_end=None),
    "ramp cut, 20° chamfer": PLAIN
    | {"ramp_cut": {"line": _CUT_LINE, "depth": 2.5, "slope_angle_deg": 20.0}},
    "ramp cut, bare step": PLAIN | {"ramp_cut": {"line": _CUT_LINE, "depth": 2.5}},
    "land ends": PLAIN | {"land_ends": {"w_from": 100.0, "depth": 0.5}},
    "edge channel along the whole wall": PLAIN
    | {"edge_channels": [{"width": 2.0, "depth": 2.5, "side": "outer"}]},
    "well with vertical walls": _with(PLAIN, well__wall_angle_deg=90.0, well__depth=4.0),
    "valve off the plate centre": _with(PLAIN, valve__w=3.0),
    "one-sided": ONE_SIDED,
    "one-sided + profile + closure + land ends": ONE_SIDED
    | {
        "land": {
            "depth": 0.35,
            "length": 1.0,
            "profile": {"center_length": 4.0, "power": 2.0},
            "closed_line": [[0.0, 30.0], [1.0, 28.0]],
        },
        "land_ends": {"w_from": 200.0, "depth": 0.5},
    },
    "twin fan + runner + fan island + edge channels": _twin_with_edges(),
    "T gate (fans with flat splitters)": T_GATE,
}


def _spec(d: dict) -> GateProfileSpec:
    return GateProfileSpec.from_dict(d)


def test_the_module_imports_without_ocp():
    """Importing it must not pull OCP in (the app imports it on Streamlit Cloud)."""
    out = subprocess.run(
        [sys.executable, "-c", "import sys, core.gate_iges; print('OCP' in sys.modules)"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        check=True,
    )
    assert out.stdout.strip() == "False"


@needs_ocp
@pytest.mark.parametrize("name", list(CASES))
def test_the_solid_read_from_above_is_the_solvers_field(name):
    """Outline cell for cell, depth within the check tolerance, volume as the raster's."""
    spec = _spec(CASES[name])
    solid = gi.build_gate_solid(spec)
    field = drawing_field(spec, gi.plate_for(spec), cell_size_mm=0.25)
    c = gi.check_against_field(solid.shape, field.x, field.y, field.depth)
    assert c.cells > 1000
    assert c.outline_mismatch == 0, c
    assert c.max_depth_diff_mm <= gi.CHECK_TOL_MM, c
    v_field = float(np.nansum(field.depth)) * 0.25**2
    assert solid.volume_mm3 == pytest.approx(v_field, rel=2e-3)


@needs_ocp
def test_the_landhanger_volume_is_the_fine_raster_and_the_drawing():
    """0.1 mm raster to 0.5 mm³, and the 10/02 drawing's 9,731 mm³."""
    res = gi.export_gate_iges(_spec(LANDHANGER), PLATE)
    assert res.ok, res.check
    assert res.cell_mm == 0.1
    assert res.volume_mm3 == pytest.approx(res.field_volume_mm3, abs=0.5)
    assert res.volume_mm3 == pytest.approx(9731.0, abs=1.0)


@needs_ocp
def test_the_check_tells_a_wrong_solid_from_the_right_one():
    """The guard on the download: another shape's field must fail it, both ways."""
    solid = gi.build_gate_solid(_spec(LANDHANGER)).shape
    # Same outline, different depth (centre land 6 instead of 5).
    six = _spec(_with(LANDHANGER, land__profile__center_length=6.0))
    f6 = drawing_field(six, PLATE, cell_size_mm=0.25)
    c = gi.check_against_field(solid, f6.x, f6.y, f6.depth)
    # The ramp starts 1 mm later at the centre: tan(11.06°) × 1 = 0.195 mm, ~100× the tolerance.
    assert c.outline_mismatch == 0 and c.max_depth_diff_mm > 50 * gi.CHECK_TOL_MM
    # Different outline (a closure in the land).
    closed = _spec(_with(LANDHANGER, land__closed_line=[[0.0, 50.0], [1.0, 47.6]]))
    fc = drawing_field(closed, PLATE, cell_size_mm=0.25)
    assert gi.check_against_field(solid, fc.x, fc.y, fc.depth).outline_mismatch > 0


@needs_ocp
def test_the_file_reads_back_into_the_same_closed_solid(tmp_path):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeSolid, BRepBuilderAPI_Sewing
    from OCP.IGESControl import IGESControl_Reader
    from OCP.TopAbs import TopAbs_SHELL
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    solid = gi.build_gate_solid(_spec(FULL))
    data = gi.iges_bytes(solid.shape)
    path = tmp_path / "block.igs"
    path.write_bytes(data)
    # Global section: millimetres (unit flag 2, "MM").
    assert b"2HMM" in data.replace(b"\n", b"")
    # Faces only, as trimmed surfaces (entity 144), like the received models.
    d_section = [ln for ln in data.decode("ascii").splitlines() if ln[72:73] == "D"]
    types = {int(ln[0:8]) for ln in d_section[::2]}
    assert 144 in types and 186 not in types
    r = IGESControl_Reader()
    assert r.ReadFile(str(path)) == 1
    r.TransferRoots()
    sew = BRepBuilderAPI_Sewing(1e-6)
    sew.Add(r.OneShape())
    sew.Perform()
    assert sew.NbFreeEdges() == 0
    shell = TopoDS.Shell_s(TopExp_Explorer(sew.SewedShape(), TopAbs_SHELL).Current())
    back = BRepBuilderAPI_MakeSolid(shell).Solid()
    assert abs(gi.volume_mm3(back)) == pytest.approx(solid.volume_mm3, rel=1e-6)


@needs_ocp
def test_the_frame_is_the_received_cad_models():
    """x from the valve axis, y = 20 − t, z = −depth with the PL at z = 0."""
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    spec = _spec(LANDHANGER)
    box = Bnd_Box()
    BRepBndLib.AddOptimal_s(gi.build_gate_solid(spec).shape, box, False, False)
    x0, y0, z0, x1, y1, z1 = box.Get()
    tol = 1e-3 + box.GetGap()
    assert x0 == pytest.approx(-149.0, abs=tol) and x1 == pytest.approx(149.0, abs=tol)
    assert y1 == pytest.approx(gi.CAD_Y_AT_EXIT, abs=tol)  # the exit, t = 0
    assert y0 == pytest.approx(gi.CAD_Y_AT_EXIT - spec.t_max(), abs=tol)
    assert z1 == pytest.approx(0.0, abs=tol) and z0 == pytest.approx(-4.5, abs=tol)
    # One-sided: the valve axis is still x = 0, the block runs to the far edge.
    one = _spec(_with(ONE_SIDED, valve__w=10.0))
    box = Bnd_Box()
    BRepBndLib.AddOptimal_s(gi.build_gate_solid(one).shape, box, False, False)
    x0, _y0, _z0, x1, *_ = box.Get()
    assert x0 == pytest.approx(-10.0, abs=tol) and x1 == pytest.approx(289.0, abs=tol)


@needs_ocp
def test_a_symmetric_block_is_mirrored_about_the_valve_axis():
    spec = _spec(FULL)
    solid = gi.build_gate_solid(spec).shape
    x = np.arange(-148.875, 149.0, 0.25)
    y = gi.CAD_Y_AT_EXIT - np.arange(0.125, spec.t_max(), 0.25)[::-1]
    d = gi.raster_depth(solid, x, y)
    assert np.array_equal(np.isfinite(d), np.isfinite(d[:, ::-1]))
    both = np.isfinite(d)
    # Exact by construction; the mesh the read-back uses is not mirror-symmetric.
    assert np.max(np.abs(d[both] - d[:, ::-1][both])) < gi.CHECK_TOL_MM


def test_the_land_end_spline_is_exact_for_a_polynomial_and_close_otherwise():
    """Power 2 is one cubic span with no error; power 0.5 is reparametrised and still exact."""

    def land(p, center=5.0, edge=149.0):
        return lambda w: 1.0 + (center - 1.0) * np.clip(1.0 - w / edge, 0.0, 1.0) ** p

    sp, err = gi._w_spline(land(2.0), 0.0, 149.0, m=1, deg=3)
    assert len(sp.knots) == 8 and err < 1e-9
    m = math.ceil(3.0 / 0.5)  # (1 − v)^{mp}, mp = 3
    sp, err = gi._w_spline(land(0.5), 0.0, 149.0, m=m, deg=m)
    assert err < 1e-9
    w = sp.eval(sp.fit(lambda w: w))
    assert w[0] == pytest.approx(0.0, abs=1e-9) and w[-1] == pytest.approx(149.0, abs=1e-9)
    _sp, err = gi._w_spline(land(1.7), 0.0, 149.0, m=2, deg=3)
    assert err <= gi.CURVE_TOL_MM


@needs_ocp
def test_a_power_too_small_to_model_is_refused():
    with pytest.raises(ValueError, match="degree"):
        gi.build_gate_solid(_spec(_with(LANDHANGER, land__profile__power=0.05)))


@needs_ocp
def test_the_command_line_writes_the_file_from_a_spec_or_a_run(tmp_path):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(LANDHANGER))
    out = tmp_path / "a.igs"
    assert gi.main([str(spec), str(out), "--cell", "0.5"]) == 0
    assert out.read_bytes()[72:73] == b"S"
    run = tmp_path / "settings.json"
    run.write_text(json.dumps({"geometry": {"gate_profile": _spec(FULL).to_dict()}}))
    out2 = tmp_path / "b.igs"
    assert gi.main([str(run), str(out2), "--cell", "0.5"]) == 0
    assert out2.stat().st_size > out.stat().st_size  # FULL has more faces


@needs_ocp
def test_plate_for_fits_the_block_on_the_builder_grid():
    for d in CASES.values():
        spec = _spec(d)
        drawing_field(spec, gi.plate_for(spec), cell_size_mm=0.5)  # no overhang error
