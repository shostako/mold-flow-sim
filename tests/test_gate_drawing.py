"""The A3 gate drawing (``core/gate_drawing.py``).

What the tests pin: the drawing is the solver's own builder on a finer mesh
(so the picture is the shape being solved), the sections read that field
(land, ramp, cap, steel where there is no pocket), the table carries every
value of the spec and names each one, and the sheet comes out as a one-page
PDF plus a preview PNG.
"""

from __future__ import annotations

import dataclasses
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
    """Same builder, same block -- the product is only cut to a strip."""
    spec = _spec()
    f = gd.drawing_field(spec, PLATE)
    ref = build_profile_gate_geometry(spec, PLATE, cell_size_mm=gd.FINE_CELL_MM)
    assert f.geometry.cell_size_mm == gd.FINE_CELL_MM
    x0, y0 = ref.display_origin_mm()
    y_ref = (np.arange(ref.ny) + 0.5) * ref.cell_size_mm - y0
    rows_ref, rows = np.where(y_ref < 0)[0], np.where(f.y < 0)[0]
    assert np.allclose(y_ref[rows_ref], f.y[rows])
    want = np.where(ref.mask[rows_ref], ref.thickness_mm[rows_ref], np.nan)
    assert np.array_equal(np.isfinite(f.depth[rows]), np.isfinite(want))
    assert np.array_equal(f.depth[rows][np.isfinite(want)], want[np.isfinite(want)])
    # Display frame: x = 0 on the valve axis, the block below y = 0.
    assert f.y[np.isfinite(f.depth).any(axis=1)].max() < 0
    assert abs(f.x[np.argmin(np.abs(f.x))]) <= gd.FINE_CELL_MM


def test_a_large_product_is_not_rasterised_for_the_drawing():
    """A 400 × 200 plate at 0.1 mm would be ~10 M cells (Codex P2 on PR #104)."""
    big = dataclasses.replace(PLATE, plate_w_mm=400.0, plate_h_mm=200.0)
    f = gd.drawing_field(_spec(), big)
    assert f.geometry.mask.size < 2_000_000
    assert f.geometry.ny * f.geometry.cell_size_mm < _spec().t_max() + 2 * big.pad_mm + 2.0


def test_a_coarsened_mesh_still_cuts_the_edge_section_inside_the_pocket(monkeypatch):
    """The edge cut steps in by the drawing mesh, not by 0.1 mm (review on PR #104).

    Stepping in by one cell keeps the nearest column centre (at most half a
    cell away) inside the exit; stepping in by 0.1 mm on a 0.4 mm mesh does
    not, wherever the columns happen to fall.
    """
    monkeypatch.setattr(gd, "MAX_DRAWING_CELLS", 200_000)
    f = gd.drawing_field(_spec(), PLATE)
    dx = f.geometry.cell_size_mm
    assert dx > gd.FINE_CELL_MM
    g = gd.render_gate_drawing(_spec(), PLATE, title="coarse")
    lab, w = g.sections[-1]
    assert lab == "D" and w == pytest.approx(gd.exit_half_width(_spec()) - dx)
    _t, d = gd.section_profile(f, gd.w_to_x(_spec(), w))
    assert d.max() > 0.0


def test_a_sub_millimetre_exit_still_has_four_distinct_cuts():
    pos = [w for _, w in gd.section_positions(_spec({**LANDHANGER, "gate_exit_width": 0.5}), 0.05)]
    assert pos == sorted(set(pos)) and len(pos) == 4


def test_the_drawing_mesh_coarsens_past_the_cell_cap(monkeypatch):
    monkeypatch.setattr(gd, "MAX_DRAWING_CELLS", 200_000)
    dx = gd.drawing_cell_mm(_spec(), PLATE)
    assert dx > gd.FINE_CELL_MM
    w = 2 * PLATE.pad_mm + PLATE.plate_w_mm
    h = 2 * PLATE.pad_mm + _spec().t_max() + 1.0
    assert (w / dx) * (h / dx) <= 200_000


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


def test_a_narrow_exit_keeps_four_distinct_cuts_inside_the_pocket():
    """Rounding to 5 mm would give 0, 0, 5, 3.9 for a 4 mm half-width (Codex P2)."""
    narrow = {
        **LANDHANGER,
        "gate_exit_width": 8.0,
        "land": {"depth": 0.35, "length": 1.0},
        "outer_wall_line": [[15.0, 4.0], [23.0, 3.0]],
    }
    pos = [w for _, w in gd.section_positions(_spec(narrow))]
    assert pos == sorted(set(pos)) and len(pos) == 4
    assert 0.0 <= pos[0] and pos[-1] < 4.0


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


def test_a_wide_exit_drops_the_plan_to_one_half_instead_of_running_off_the_page():
    """A 400 mm exit at 1:1 is wider than A3 (Codex P2 on PR #104)."""
    wide = {
        **LANDHANGER,
        "gate_exit_width": 400.0,
        "outer_wall_line": [[15.736241, 200.0], [23.309724, 4.489329]],
    }
    plate = dataclasses.replace(PLATE, plate_w_mm=400.0)
    g = gd.render_gate_drawing(_spec(wide), plate, title="wide")
    assert g.plan_scale == 0.5
    assert (400.0 + 8.0) * g.plan_scale <= gd.A3_MM[0] - 28.0


def test_the_default_block_stays_at_one_to_one(drawing):
    assert drawing.plan_scale == 1.0


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


# ------------------------------ dimensions --------------------------------
def _dims(d: dict | None = None) -> dict[tuple[str, str], gd.Dim]:
    g = gd.render_gate_drawing(_spec(d), PLATE, title="dims")
    return {(x.view, x.name): x for x in g.dims}


def test_plan_dimensions_are_the_spec_values():
    dims = _dims()
    assert dims[("平面", "ゲート出口幅")].value == 298.0
    assert dims[("平面", "外壁の始点 t（端）")].value == pytest.approx(15.736241)
    assert dims[("平面", "ブロックの奥行き")].value == pytest.approx(27.5)
    assert dims[("平面", "バルブ位置 t")].value == 21.5
    assert all(x.exact for (v, _), x in dims.items() if v == "平面")
    assert ("平面", "閉鎖幅（製品側）") not in dims


def test_land_profile_sections_print_the_closed_form_values():
    """Centre: land 5, cap at 16, well 4.5 deep to 27.5. Edge: land 1, cap at 12."""
    dims = _dims()
    assert dims[("A", "ランド長")].value == pytest.approx(5.0, abs=1e-3)
    assert dims[("A", "上限深さに達する t")].value == pytest.approx(16.0, abs=1e-3)
    assert dims[("A", "ポケットの奥")].value == pytest.approx(27.5)
    assert dims[("A", "最大深さ")].value == 4.5
    assert dims[("D", "ランド長")].value == pytest.approx(1.0, abs=1e-3)
    assert dims[("D", "上限深さに達する t")].value == pytest.approx(12.0, abs=1e-3)
    for lab in "ABCD":
        assert dims[(lab, "斜面角度")].value == pytest.approx(RAMP_DEG, abs=0.01)
    assert all(x.exact for x in dims.values())


def test_graded_ends_and_a_closure_are_dimensioned():
    graded = {
        **LANDHANGER,
        "land": {"depth": 0.35, "length": 1.0, "closed_line": [[0.0, 50.0], [1.0, 47.644148]]},
        "ramp_ends": {"w_from": 60.0, "t_end": 2.5, "depth_end": 3.5},
    }
    dims = _dims(graded)
    assert dims[("平面", "閉鎖幅（製品側）")].value == pytest.approx(100.0)
    d_cap = dims[("D", "上限深さ")]
    t_cap = dims[("D", "上限深さに達する t")]
    assert d_cap.value == pytest.approx(3.5, abs=0.01) and t_cap.exact
    assert t_cap.value == pytest.approx(2.5, abs=0.05)
    # The centre is closed at the land: no land length there, the rest still reads.
    assert ("A", "ランド長") not in dims and ("A", "ポケットの奥") in dims


def test_a_feature_without_a_closed_form_prints_what_is_drawn_with_approx():
    """A ramp cut (Film gate 10) reshapes the edge: the cap is reached earlier."""
    cut = {
        **LANDHANGER,
        "land": {"depth": 0.35, "length": 1.0},
        "ramp_cut": {
            "line": [[7.451, 149.0], [12.1126, 60.0]],
            "depth": 2.5,
            "slope_angle_deg": 20.0,
        },
    }
    dims = _dims(cut)
    t_cap = dims[("D", "上限深さに達する t")]
    assert not t_cap.exact and t_cap.text().startswith("≈")
    assert t_cap.value < 12.0 - 1.0
    assert dims[("B", "上限深さに達する t")].exact  # inside the uncut 120 mm


def test_dim_text_and_snapping():
    assert gd.Dim("A", "x", 5.0, True).text() == "5"
    assert gd.Dim("A", "x", 7.456, False).text() == "≈7.46"
    assert gd._snap(5.04, [5.0, 6.0], 0.1) == (5.0, True)
    assert gd._snap(5.3, [5.0], 0.1) == (5.3, False)
    assert gd._snap(5.0, [None], 0.1) == (5.0, False)


def test_a_section_through_steel_has_no_dimensions():
    t = np.linspace(0.05, 27.45, 275)
    sd = gd.section_dims(_spec(), "X", 0.0, t, np.zeros_like(t), 0.1)
    assert all(v is None for v in sd.__dict__.values())


def test_a_section_with_a_steel_gap_measures_the_back_and_depth_over_all_runs():
    """The twin-fan centre: land, a steel diamond, then the well (Codex P1 on PR #106)."""
    spec = GateProfileSpec.from_json((DATA / "demo_twin_fan_gate.json").read_text())
    f = gd.drawing_field(spec, PLATE)
    t, d = gd.section_profile(f, 0.0)
    sd = gd.section_dims(spec, "A", 0.0, t, d, f.geometry.cell_size_mm)
    last_pocket = t[d > 0].max()
    assert sd.pocket_end.value == pytest.approx(last_pocket, abs=f.geometry.cell_size_mm)
    assert sd.max_depth is not None and sd.max_depth.value == pytest.approx(spec.well.depth)


def test_the_deepest_point_can_sit_in_the_first_run():
    """Synthetic: a deep first run, steel, then a shallow run -- the max is the first."""
    t = np.arange(0.05, 20.0, 0.1)
    d = np.where(
        t < 1.0, 0.35, np.where(t < 8.0, 3.0, np.where(t < 10.0, 0.0, np.where(t < 15.0, 1.0, 0.0)))
    )
    sd = gd.section_dims(_spec(), "X", 50.0, t, d, 0.1)
    assert sd.max_depth is not None and sd.max_depth.value == pytest.approx(3.0)
    assert sd.pocket_end.value == pytest.approx(15.0, abs=0.1)
    assert not sd.pocket_end.exact  # 15 is none of the spec's candidates at w = 50
