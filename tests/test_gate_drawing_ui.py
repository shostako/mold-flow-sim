"""AppTest wiring of the drawing panel (expander「図面（A3 PDF）」).

The panel exists for every spec-based shape and not for Direct gate, stays
off until asked (drawing costs seconds), draws the shape the sidebar will
solve, and follows the sidebar when it changes.
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from core import gate_drawing

APP = Path(__file__).resolve().parent.parent / "app.py"
FG15 = "Film gate 15 (扇状/ランド長可変)"
CENTER_KEY = "f15_中央のランド長 [mm] (> ランド長さ)"


def _app(label: str = FG15) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=240.0)
    at.run()
    at.radio(key="geom_source").set_value(label).run()
    return at


def _drawing(at: AppTest):
    """``(spec key, PDF bytes)`` of the drawing on screen, or None."""
    return at.session_state["mfs_gate_drawing"] if "mfs_gate_drawing" in at.session_state else None


def test_the_panel_is_off_until_asked_and_draws_the_sidebar_shape():
    at = _app()
    assert at.checkbox(key="gate_drawing_on").value is False
    assert _drawing(at) is None
    spec, plate, dx, tag = at.session_state["mfs_gate_spec"]
    assert tag == "film_gate_15_parametric" and dx == 1.0
    assert spec.land.profile.center_length == 5.0
    at.checkbox(key="gate_drawing_on").set_value(True).run()
    assert not at.exception
    key, pdf = _drawing(at)
    assert pdf.startswith(b"%PDF")
    assert gate_drawing.spec_key(spec, plate) == key


def test_the_drawing_follows_a_slider():
    at = _app()
    at.checkbox(key="gate_drawing_on").set_value(True).run()
    key5, _ = _drawing(at)
    at.slider(key=CENTER_KEY).set_value(6.0).run()
    assert not at.exception
    spec, plate, _dx, _tag = at.session_state["mfs_gate_spec"]
    assert spec.land.profile.center_length == 6.0
    key6, pdf6 = _drawing(at)
    assert key6 != key5 and key6 == gate_drawing.spec_key(spec, plate)
    assert pdf6.startswith(b"%PDF")


def test_direct_gate_has_no_drawing_panel():
    at = _app("Direct gate (parametric)")
    assert "mfs_gate_spec" not in at.session_state
    assert not [c for c in at.checkbox if c.key == "gate_drawing_on"]
    assert _drawing(at) is None
