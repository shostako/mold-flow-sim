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


def test_the_defaults_are_two_phase_on_with_icm_and_no_wall_model():
    """UI defaults (v0.29.0): two-phase ON, ICM ON at 0.50 mm stroke, wall
    model 'none' -- the combination the two-phase model actually runs in."""
    at = _app()
    assert at.checkbox(key="two_phase_on").value is True
    assert at.checkbox(key="icm_on").value is True
    assert at.radio(key="wall_model").value == "none"
    stroke = [s for s in at.slider if str(s.label).startswith("圧縮ストローク")]
    assert len(stroke) == 1 and stroke[0].value == 0.50


def test_switching_the_toggle_off_runs_nothing_two_phase():
    at = _app()
    at.checkbox(key="two_phase_on").set_value(False).run()
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
    assert "two_phase.gif" in names
    assert "two_phase_metadata.json" in names
    assert at.session_state["mfs_two_phase_gif_path"] is not None
    assert at.session_state["mfs_two_phase_skip"] is None


def test_a_rejected_shot_warns_instead_of_crashing(monkeypatch):
    """Codex P2 (round 2) on PR #62: the UI minimum of 0.01 cm3 can be below
    the gate region's open-gap volume on coarse meshes / large gates; the
    solver's ValueError must become a warning + skip, not an uncaught
    Streamlit exception.

    Whether a given geometry actually rejects a given shot depends on its
    gate-region volume (the default Film gate 2 sits at ~0.0094 cm3, just
    under the UI minimum), so the rejection is injected: ``app.py`` re-does
    ``from core.two_phase import ...`` on every AppTest run, which reads the
    patched module attribute — the except-path wiring is what is under test.
    """
    import core.two_phase as tp

    def _reject(solver, shot_volume_cm3):
        raise ValueError("injected rejection (gate region)")

    monkeypatch.setattr(tp, "solve_two_phase_short_shot", _reject)
    at = _app()
    at.radio(key="wall_model").set_value("none")
    at.checkbox(key="two_phase_on").set_value(True).run()
    at.button[0].click().run()
    assert not at.exception
    assert at.session_state["mfs_two_phase_result"] is None
    assert at.session_state["mfs_two_phase_path"] is None
    assert "二相ショートショット解析をスキップしました" in _texts(at)


def test_a_wall_cooling_model_skips_two_phase_with_a_warning():
    at = _app()
    at.radio(key="wall_model").set_value("skin")
    at.checkbox(key="two_phase_on").set_value(True).run()
    # The interference must be visible in the sidebar BEFORE any run — the
    # run-time warning alone washes away on the next rerun and the toggle
    # looks like it silently does nothing (the exact complaint that
    # motivated this: the default wall model is 層別, so out of the box the
    # checkbox appeared dead).
    assert "現在の設定では二相解析はスキップされる" in _texts(at)
    at.button[0].click().run()
    assert not at.exception
    assert at.session_state["mfs_two_phase_result"] is None
    assert at.session_state["mfs_two_phase_path"] is None
    assert "壁面冷却モデル『なし』専用" in _texts(at)
    # the skip reason survives in session_state for the results pane
    assert "併用不可" in at.session_state["mfs_two_phase_skip"]
    assert at.session_state["mfs_settings"]["two_phase_short_shot"] == {"enabled": False}
