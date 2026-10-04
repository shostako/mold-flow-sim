"""AppTest wiring of the IGES panel (expander「3D モデル（IGES）」).

The panel exists for every spec-based shape and not for Direct gate, stays
off until asked, writes the shape the sidebar will solve, tells where OCP is
missing instead of failing, and holds the file back when the read-back
check does not match.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from core import gate_drawing, gate_iges

APP = Path(__file__).resolve().parent.parent / "app.py"
FG15 = "Film gate 15 (扇状/ランド長可変)"

needs_ocp = pytest.mark.skipif(
    not gate_iges.available(), reason="OCP (the cad extra) is not installed"
)


def _app(label: str = FG15) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=240.0)
    at.run()
    at.radio(key="geom_source").set_value(label).run()
    return at


def _iges(at: AppTest):
    """``(spec key, IGES bytes or None)`` of the panel, or None when it made nothing."""
    return at.session_state["mfs_gate_iges"] if "mfs_gate_iges" in at.session_state else None


@needs_ocp
def test_the_panel_is_off_until_asked_and_writes_the_sidebar_shape():
    st.cache_data.clear()
    at = _app()
    assert at.checkbox(key="gate_iges_on").value is False
    assert _iges(at) is None
    at.checkbox(key="gate_iges_on").set_value(True).run()
    assert not at.exception
    assert not at.error
    key, data = _iges(at)
    spec, plate, _dx, _tag = at.session_state["mfs_gate_spec"]
    assert key == gate_drawing.spec_key(spec, plate)
    assert data[72:73] == b"S" and b"2HMM" in data.replace(b"\n", b"")


@needs_ocp
def test_a_solid_that_does_not_match_is_not_offered(monkeypatch):
    st.cache_data.clear()
    real = gate_iges.export_gate_iges

    def off_by_a_cell(spec, plate, cell_size_mm=None):
        res = real(spec, plate, cell_size_mm)
        bad = gate_iges.FieldCheck(res.check.cells, 3, res.check.max_depth_diff_mm)
        return dataclasses.replace(res, check=bad)

    monkeypatch.setattr(gate_iges, "export_gate_iges", off_by_a_cell)
    at = _app()
    at.checkbox(key="gate_iges_on").set_value(True).run()
    assert not at.exception
    assert any("一致しない" in e.value for e in at.error)
    _key, data = _iges(at)
    assert data is None
    st.cache_data.clear()


def test_without_ocp_the_panel_says_so(monkeypatch):
    monkeypatch.setattr(gate_iges, "available", lambda: False)
    at = _app()
    assert not at.exception
    assert not [c for c in at.checkbox if c.key == "gate_iges_on"]
    assert any("OCP" in c.value for c in at.caption)
    assert _iges(at) is None


def test_direct_gate_has_no_iges_panel():
    at = _app("Direct gate (parametric)")
    assert not [c for c in at.checkbox if c.key == "gate_iges_on"]
    assert _iges(at) is None
