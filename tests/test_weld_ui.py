"""AppTest wiring for the weld / meld threshold slider in ``app.py``.

The slider lives in the sidebar's output expander and is read before the
run button, so moving it after a run causes a rerun with ``do_run=False``.
The weld map must still follow it -- from the cached result's angle field,
not from a second solve -- and the ZIP handed out afterwards must carry the
re-thresholded image and setting (Codex P2 on PR #67).
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_moving_the_slider_after_a_run_rethresholds_without_resolving():
    at = AppTest.from_file(str(APP), default_timeout=240.0)
    at.run()
    at.radio(key="wall_model").set_value("none")
    at.checkbox(key="two_phase_on").set_value(False).run()
    assert at.slider(key="weld_min_angle").value == 0
    at.button[0].click().run()
    assert not at.exception
    result = at.session_state["mfs_result"]
    first_png = Path(at.session_state["mfs_weld_path"]).read_bytes()
    assert at.session_state["mfs_settings"]["output"]["weld_min_angle_deg"] == 0.0

    at.slider(key="weld_min_angle").set_value(40).run()
    assert not at.exception
    # same solve, different picture
    assert at.session_state["mfs_result"] is result
    second_png = Path(at.session_state["mfs_weld_path"]).read_bytes()
    assert second_png != first_png
    assert at.session_state["mfs_settings"]["output"]["weld_min_angle_deg"] == 40.0
    with zipfile.ZipFile(io.BytesIO(at.session_state["mfs_zip_bytes"])) as zf:
        names = zf.namelist()
        assert names.count("weld.png") == 1 and names.count("settings.json") == 1
        assert zf.read("weld.png") == second_png
        assert json.loads(zf.read("settings.json"))["output"]["weld_min_angle_deg"] == 40.0
        # nothing else fell out of the archive
        assert "fill.gif" in names and "metadata.json" in names and "player.html" in names
