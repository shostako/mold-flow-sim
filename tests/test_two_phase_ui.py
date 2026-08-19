"""AppTest wiring checks for the two-phase short-shot UI in ``app.py``."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "app.py"


def _app(timeout: float = 180.0) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    at.run()
    return at


def _texts(at: AppTest) -> str:
    parts = []
    for group in (at.caption, at.info, at.warning, at.error, at.markdown, at.exception):
        parts.extend(str(getattr(el, "value", "")) for el in group)
    return "\n".join(parts)


def test_the_toggle_is_off_by_default_and_nothing_two_phase_runs():
    at = _app()
    assert at.checkbox(key="two_phase_on").value is False
    at.radio(key="wall_model").set_value("none")
    at.button[0].click().run()
    assert not at.exception
    assert at.session_state["mfs_two_phase_path"] is None
    assert at.session_state["mfs_two_phase_result"] is None
    assert "二相ショートショット" not in _texts(at)


def test_the_two_phase_run_renders_the_map_and_packs_the_zip():
    at = _app()
    at.radio(key="wall_model").set_value("none")
    at.checkbox(key="icm_on").set_value(True)
    at.checkbox(key="two_phase_on").set_value(True).run()
    at.number_input(key="two_phase_shot_volume").set_value(4.5)
    at.button[0].click().run()
    assert not at.exception
    res = at.session_state["mfs_two_phase_result"]
    assert res is not None
    assert res.metadata["shot_volume_cm3"] == 4.5
    path = at.session_state["mfs_two_phase_path"]
    # Captions inside an expander are not surfaced by the flat AppTest
    # element lists (the pressure-map caption is equally absent), so the
    # rendered map is verified as a file, not as page text.
    assert path is not None and Path(path).exists() and Path(path).stat().st_size > 0
    # settings record and ZIP contents
    assert at.session_state["mfs_settings"]["two_phase_short_shot"] == {
        "enabled": True,
        "shot_volume_cm3": 4.5,
    }
    with zipfile.ZipFile(io.BytesIO(at.session_state["mfs_zip_bytes"])) as zf:
        names = set(zf.namelist())
    assert "two_phase_short_shot.png" in names
    assert "two_phase_metadata.json" in names


def test_a_wall_cooling_model_skips_two_phase_with_a_warning():
    at = _app()
    at.radio(key="wall_model").set_value("skin")
    at.checkbox(key="two_phase_on").set_value(True).run()
    at.button[0].click().run()
    assert not at.exception
    assert at.session_state["mfs_two_phase_result"] is None
    assert at.session_state["mfs_two_phase_path"] is None
    assert "壁面冷却モデル『なし』専用" in _texts(at)
    assert at.session_state["mfs_settings"]["two_phase_short_shot"] == {"enabled": False}
