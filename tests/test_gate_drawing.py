"""The A3 gate drawing (``core/gate_drawing.py``).

What the tests pin: the drawing is the solver's own builder on a finer mesh
(so the picture is the shape being solved), the sections read that field
(land, ramp, cap, steel where there is no pocket), the table carries every
value of the spec and names each one, and the sheet comes out as a one-page
PDF plus a preview PNG.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from core import gate_drawing as gd
from core.profile_gate import GateProfileSpec, ProfilePlateConfig, build_profile_gate_geometry

DATA = Path(__file__).resolve().parent.parent / "data" / "gate_profiles"
TAN = 2.15 / 11.0
RAMP_DEG = math.degrees(math.atan(TAN))

#: Film gate 15's default (the 10/02 proposal A3): land 5 at the centre, 1 at the edge.
LANDHANGER = {
    "name": "landhanger",
    "units": "mm",
    "symmetric": True,
    "gate_exit_width": 298.0,
    "land": {"depth": 0.35, "length": 1.0, "profile": {"center_length": 5.0, "power": 2.0}},
    "main_ramp": {"angle_deg": RAMP_DEG, "cap_depth": 2.5},
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
}
PLATE = ProfilePlateConfig(
    plate_w_mm=300.0,
    plate_h_mm=50.0,
    plate_thk_mm=0.35,
    plate_split_height_mm=20.0,
    plate_lower_thk_mm=0.35,
    plate_upper_thk_mm=0.50,
)


def _spec(d: dict | None = None) -> GateProfileSpec:
    return GateProfileSpec.from_dict(d or LANDHANGER)


def test_the_drawing_field_is_the_solvers_builder_on_the_fine_mesh():
    spec = _spec()
    f = gd.drawing_field(spec, PLATE)
    ref = build_profile_gate_geometry(spec, PLATE, cell_size_mm=gd.FINE_CELL_MM)
    assert f.geometry.cell_size_mm == gd.FINE_CELL_MM
    block = (f.y < 0)[:, None] & ref.mask
    assert np.array_equal(np.isfinite(f.depth), block)
    assert np.array_equal(f.depth[block], ref.thickness_mm[block])
    # Display frame: x = 0 on the valve axis, the block below y = 0.
    assert f.y[np.isfinite(f.depth).any(axis=1)].max() < 0
    assert abs(f.x[np.argmin(np.abs(f.x))]) <= gd.FINE_CELL_MM


def test_the_centre_and_edge_sections_read_the_land_profile():
    """Centre: land 0.35 to t=5, then 11.06° to 2.5 at t=16. Edge: land to t=1."""
    spec = _spec()
    f = gd.drawing_field(spec, PLATE)
    for w, land_end in ((0.0, 5.0), (148.9, 1.0 + 4.0 * (1 - 148.9 / 149.0) ** 2)):
        t, d = gd.section_profile(f, gd.w_to_x(spec, w))
        sel = t < 15.0
        want = np.where(t <= land_end, 0.35, np.minimum(0.35 + TAN * (t - land_end), 2.5))
        # One fine cell of slack across the land's end.
        assert np.max(np.abs(d[sel] - want[sel])) <= TAN * gd.FINE_CELL_MM + 1e-9, w
    t, d = gd.section_profile(f, 0.0)
    assert d[(t > 18.2) & (t < 24.8)] == pytest.approx(4.5)  # the well floor


def test_a_section_reads_steel_where_there_is_no_pocket():
    """The twin-fan demo has a steel diamond on the axis between the fans."""
    spec = GateProfileSpec.from_json((DATA / "demo_twin_fan_gate.json").read_text())
    f = gd.drawing_field(spec, PLATE)
    t, d = gd.section_profile(f, 0.0)
    assert np.any(d == 0.0) and np.any(d > 0.0)
    assert np.all(d >= 0.0) and np.all(np.isfinite(d))


def test_section_positions_spread_over_the_exit_width():
    sym = _spec()
    assert gd.section_positions(sym) == [("A", 0.0), ("B", 50.0), ("C", 100.0), ("D", 148.9)]
    one = _spec({**LANDHANGER, "symmetric": False, "land": {"depth": 0.35, "length": 1.0}})
    pos = [w for _, w in gd.section_positions(one)]
    assert pos[0] > 0 and pos[-1] < one.gate_exit_width and pos == sorted(pos)
    # The one-sided block measures w from the valve-side edge.
    assert gd.w_to_x(one, 0.0) == -one.valve.w


def _leaf_paths(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _leaf_paths(v, f"{path}.{k}" if path else k)
    else:
        yield path


#: A spec with most single-pocket features switched on (Film gate 13 + others).
FULL = {
    **LANDHANGER,
    "land": {
        "depth": 0.35,
        "length": 1.0,
        "closed_line": [[0.0, 50.0], [1.0, 47.644148]],
        "profile": {"center_length": 3.0, "power": 2.0},
    },
    "outer_wall_corner_radius": 10.0,
    "ramp_ends": {"w_from": 60.0, "t_end": 2.5, "depth_end": 3.5},
    "island": {
        "angle_deg": 2.5,
        "boundary_line": [[1.0, 47.644148], [17.0, 9.9505]],
        "end_dist": 17.0,
        "weld": {"t_range": [7.0, 17.0], "depth": 0.1, "w_max": 20.0},
    },
    "edge_channels": [{"width": 2.0, "depth": 2.5, "t_range": [16.0, 23.3], "side": "outer"}],
}


def test_the_table_lists_every_value_of_the_spec_under_a_name():
    """Every leaf of ``to_dict()`` becomes a row, and every row has a Japanese name.

    A feature added to the spec without a label here fails the second assert
    instead of showing up as an English path on the sheet.
    """
    spec = _spec(FULL)
    rows = gd.spec_rows(spec, PLATE)
    d = spec.to_dict()
    leaves = []
    for key, val in d.items():
        if key in gd._SKIP:
            continue
        if key in gd._LIST_LABELS:
            for item in val:
                leaves.extend(f"{key}:{p}" for p in _leaf_paths(item))
        else:
            leaves.extend(_leaf_paths(val, key))
    derived = [r for r in rows if r[0].startswith("（導出）") or r[0].startswith("製品")]
    assert len(rows) - len(derived) == len(leaves)
    labels = [k for k, _ in rows]
    assert not [k for k in labels if re.search(r"[a-z_]+\.[a-z_]+", k)], labels
    assert "中央のランド長" in labels and "ランドの閉鎖線 (t, w)" in labels
    assert any(k.startswith("縁部深彫り 1:") for k in labels)
    # The spec's own name is content, not geometry, and can name the part.
    assert all(v != spec.name for _, v in rows)


def test_every_known_spec_key_has_a_label():
    """The label table covers the spec dataclasses: walk a spec of each kind."""
    for spec in (
        _spec(FULL),
        GateProfileSpec.from_json((DATA / "demo_twin_fan_gate.json").read_text()),
    ):
        for k, _ in gd.spec_rows(spec, PLATE):
            assert not re.search(r"[A-Za-z_]+\.[A-Za-z_]+", k), k


def test_number_formatting():
    assert gd._fmt(0.35) == "0.35"
    assert gd._fmt(298.0) == "298"
    assert (
        gd._fmt([[15.736241, 149.0], [23.309724, 4.489329]]) == "(15.7362, 149) → (23.3097, 4.4893)"
    )
    assert gd._fmt([15.5, 27.5]) == "15.5〜27.5"
    assert gd._fmt(True) == "はい"


@pytest.fixture(scope="module")
def drawing():
    return gd.render_gate_drawing(
        _spec(), PLATE, title="Film gate 15 (扇状/ランド長可変)", solver_cell_mm=1.0
    )


def test_the_sheet_is_a_one_page_pdf_and_a_png(drawing):
    assert drawing.pdf.startswith(b"%PDF")
    assert len(re.findall(rb"/Type\s*/Page[^s]", drawing.pdf)) == 1
    assert drawing.png.startswith(b"\x89PNG")
    assert drawing.section_scale == 5.0
    assert [lab for lab, _ in drawing.sections] == ["A", "B", "C", "D"]


def test_the_reported_pocket_volume_matches_the_solver_mesh(drawing):
    """0.1 mm and 1.0 mm rasters of the same pocket agree to well under 1 %."""
    coarse = build_profile_gate_geometry(_spec(), PLATE, cell_size_mm=0.5)
    x0, y0 = coarse.display_origin_mm()
    y = (np.arange(coarse.ny) + 0.5) * coarse.cell_size_mm - y0
    block = (y < 0)[:, None] & coarse.mask
    v_coarse = float(coarse.thickness_mm[block].sum() * coarse.cell_size_mm**2)
    assert drawing.pocket_volume_mm3 == pytest.approx(v_coarse, rel=5e-3)
    # The 10/02 drawing's own number for this pocket.
    assert drawing.pocket_volume_mm3 == pytest.approx(9731.0, abs=2.0)


def test_a_deep_block_shrinks_the_sections_instead_of_overflowing():
    deep = {
        **LANDHANGER,
        "outer_wall_line": [[50.0, 149.0], [60.0, 4.5]],
        "well": {**LANDHANGER["well"], "t_range": [52.0, 64.0], "floor_t_range": [54.6, 61.4]},
        "valve": {"t": 58.0, "w": 0.0, "orifice_diameter": 3.0},
    }
    g = gd.render_gate_drawing(_spec(deep), PLATE, title="deep")
    assert g.section_scale < 5.0
    assert 2 * (_spec(deep).t_max() + 1.0) * g.section_scale + 22.0 <= 310.0


def test_the_japanese_font_prefers_truetype(monkeypatch, tmp_path):
    fake = tmp_path / "ipaexg.ttf"
    fake.write_bytes(b"")
    monkeypatch.setattr(gd, "_find_font_files", lambda names: iter([str(fake)]))
    f = gd.japanese_font()
    assert f.path == str(fake) and f.fonttype == 42
    # No TrueType Japanese font: a CFF one with Type 3 glyphs, or none at all.
    monkeypatch.setattr(gd, "_find_font_files", lambda names: iter([]))
    f = gd.japanese_font()
    assert (f.path is None and f.fonttype == 42) or f.fonttype == 3


def test_spec_key_is_stable_and_shape_sensitive():
    a = gd.spec_key(_spec(), PLATE)
    assert a == gd.spec_key(_spec(json.loads(json.dumps(LANDHANGER))), PLATE)
    other = {
        **LANDHANGER,
        "land": {**LANDHANGER["land"], "profile": {"center_length": 6.0, "power": 2.0}},
    }
    assert a != gd.spec_key(_spec(other), PLATE)
