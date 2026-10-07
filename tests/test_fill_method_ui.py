"""AppTest wiring for the fill-method choice (v0.62.0): "一発で解く" (τ, default) or "時間で追う" (march)."""

from __future__ import annotations

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "app.py"


def _app(timeout: float = 300.0) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    at.run()
    return at


def test_the_choice_sits_under_the_layered_model_and_defaults_to_the_tau_fill():
    """The page opens on the layered model with the τ fill -- the v0.61.0
    results, unchanged. The choice is a layered-model option: the other wall
    models have no march and do not show it."""
    at = _app()
    assert at.radio(key="wall_model").value == "multilayer"
    assert at.radio(key="fill_method").value == "tau"
    at.radio(key="wall_model").set_value("skin").run()
    assert not [r for r in at.radio if r.key == "fill_method"]
    at.radio(key="wall_model").set_value("none").run()
    assert not [r for r in at.radio if r.key == "fill_method"]


def test_the_march_reaches_the_solver_and_the_settings_record():
    """Choosing "時間で追う" runs the layered solver's march: the main result and
    the two-phase injection phase say so in their metadata, and settings.json
    records the choice next to the other wall-cooling inputs."""
    at = _app()
    at.radio(key="fill_method").set_value("march").run()
    at.button[0].click().run()
    assert not at.exception
    result = at.session_state["mfs_result"]
    assert result.metadata["fill_method"] == "march"
    assert result.metadata["march_steps"] > 0
    assert at.session_state["mfs_settings"]["wall_cooling"]["fill_method"] == "march"
    tp = at.session_state["mfs_two_phase_result"]
    assert tp is not None and tp.metadata["fill_method"] == "march"
    json.dumps(at.session_state["mfs_settings"])  # the record stays JSON


def test_the_default_run_records_the_tau_fill():
    at = _app()
    at.button[0].click().run()
    assert not at.exception
    assert at.session_state["mfs_result"].metadata["fill_method"] == "tau"
    assert at.session_state["mfs_settings"]["wall_cooling"]["fill_method"] == "tau"
