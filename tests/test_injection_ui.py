"""AppTest wiring for the screw-side injection block in ``app.py``.

The block replaces one number (a flat injection rate) with a machine
condition, so the checks are: the default really is the machine condition and
really reaches the solver, the direct-rate path still exists and still wins
when chosen, the stage widgets appear only when asked for, and an impossible
condition stops the run with a message instead of an exception.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "app.py"


def _app(timeout: float = 240.0) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    at.run()
    return at


def _labels(at: AppTest, prefix: str) -> list[str]:
    out = []
    for group in (at.number_input, at.slider):
        out.extend(str(e.label) for e in group if str(e.label).startswith(prefix))
    return out


def test_default_is_the_machine_condition_with_the_documented_numbers():
    """The machine as it is actually set up: 50 mm screw, 30 -> 28 -> 22 -> 18."""
    at = _app()
    assert at.radio(key="inj_mode").value == "machine"
    assert at.number_input(key="inj_screw_d").value == 50.0
    assert at.number_input(key="inj_meter_pos").value == 30.0
    assert at.number_input(key="inj_vp_pos").value == 18.0
    assert at.number_input(key="inj_num_stages").value == 3
    assert at.number_input(key="inj_switch_0").value == 28.0
    assert at.number_input(key="inj_switch_1").value == 22.0
    assert [at.number_input(key=f"inj_velocity_{i}").value for i in range(3)] == [
        200.0,
        200.0,
        200.0,
    ]
    # Three stages, so exactly two boundaries are asked for.
    assert not [n for n in at.number_input if n.key == "inj_switch_2"]


def test_the_equal_speed_default_is_arithmetically_one_stage():
    """Three steps at one speed: the map has kinks that do not bend anything."""
    at = _app()
    at.radio(key="wall_model").set_value("none")
    at.checkbox(key="two_phase_on").set_value(False).run()
    at.button[0].click().run()
    staged = at.session_state["mfs_result"]
    at.number_input(key="inj_num_stages").set_value(1).run()
    at.button[0].click().run()
    single = at.session_state["mfs_result"]
    assert staged.total_fill_time_s == pytest.approx(single.total_fill_time_s, rel=1e-12)
    assert staged.fill_time_s == pytest.approx(single.fill_time_s, rel=1e-9, nan_ok=True)


def test_dropping_to_one_stage_hides_the_switch_positions():
    at = _app()
    at.number_input(key="inj_num_stages").set_value(1).run()
    assert not [n for n in at.number_input if n.key == "inj_switch_0"]
    assert at.number_input(key="inj_velocity_0").value == 200.0
    assert _labels(at, "第1段 速度切替位置") == []


def test_the_representative_velocity_stays_its_own_slider():
    """Screw speed and gap-mean velocity are different quantities."""
    at = _app()
    assert at.slider(key="inj_rep_velocity").value == 200.0
    assert "代表流動速度" in str(at.slider(key="inj_rep_velocity").label)


def test_machine_condition_reaches_the_solver_and_the_settings_record():
    at = _app()
    at.radio(key="wall_model").set_value("none")
    at.checkbox(key="two_phase_on").set_value(False).run()
    at.button[0].click().run()
    assert not at.exception
    res = at.session_state["mfs_result"]
    rec = res.metadata["injection_profile"]
    assert rec is not None
    assert rec["screw_diameter_mm"] == 50.0
    assert rec["vp_position_mm"] == 18.0
    assert len(rec["stages"]) == 3
    # Q = pi D^2 / 4 * v, in cm^3/s
    expected_Q = math.pi * 625.0 * 200.0 / 1000.0
    assert rec["stages"][0]["rate_cm3s"] == pytest.approx(expected_Q)
    assert rec["mean_rate_cm3s"] == pytest.approx(expected_Q)
    # V/P displacement: pi*50^2/4 * (30 - 18) = 23.56 cm^3 in 12/200 = 0.06 s
    assert rec["total_volume_cm3"] == pytest.approx(math.pi * 625.0 * 12.0 / 1000.0)
    assert rec["total_time_s"] == pytest.approx(0.06)
    assert res.metadata["injection_Q_cm3s"] is None
    inj = at.session_state["mfs_settings"]["injection"]
    assert inj["rate_input_mode"] == "machine"
    assert inj["injection_volume_flow_cm3s"] is None
    assert inj["injection_profile"]["total_time_s"] > 0


def test_the_reported_fill_time_is_the_profile_time_for_the_cavity():
    """The acceptance criterion: the speed setting owns the time axis."""
    at = _app()
    at.radio(key="wall_model").set_value("none")
    at.checkbox(key="two_phase_on").set_value(False)
    at.checkbox(key="icm_on").set_value(False).run()
    at.button[0].click().run()
    assert not at.exception
    slow = at.session_state["mfs_result"].total_fill_time_s
    for i in range(3):
        at.number_input(key=f"inj_velocity_{i}").set_value(400.0)
    at.run()
    at.button[0].click().run()
    assert not at.exception
    fast = at.session_state["mfs_result"].total_fill_time_s
    assert fast == pytest.approx(slow / 2.0, rel=1e-9)


def test_direct_rate_mode_restores_the_slider_and_drops_the_profile():
    at = _app()
    at.radio(key="inj_mode").set_value("direct").run()
    assert at.slider(key="inj_Q_direct").value == 589.0
    # The screw widgets are gone, not merely ignored.
    assert not [n for n in at.number_input if n.key == "inj_screw_d"]
    at.radio(key="wall_model").set_value("none")
    at.checkbox(key="two_phase_on").set_value(False).run()
    at.button[0].click().run()
    assert not at.exception
    res = at.session_state["mfs_result"]
    assert res.metadata["injection_profile"] is None
    assert res.metadata["injection_Q_cm3s"] == 589.0
    assert at.session_state["mfs_settings"]["injection"]["rate_input_mode"] == "direct"


def test_adding_a_stage_keeps_the_positions_already_set():
    """Only the boundary that did not exist yet gets a default.

    Streamlit keeps a widget's value across reruns once it has one, and that
    is the behaviour to want here: adding a fourth stage must not move the
    switch points the user already dialled in. The even split is the default
    for the *new* widget only.
    """
    at = _app()
    at.number_input(key="inj_num_stages").set_value(4).run()
    assert at.number_input(key="inj_switch_0").value == 28.0
    assert at.number_input(key="inj_switch_1").value == 22.0
    span = 30.0 - 18.0
    assert at.number_input(key="inj_switch_2").value == pytest.approx(30.0 - 3 * span / 4.0)
    assert at.number_input(key="inj_velocity_3").value == 200.0
    assert not [n for n in at.number_input if n.key == "inj_switch_3"]


def test_multi_stage_volumes_and_times_reach_the_solver():
    at = _app()
    at.number_input(key="inj_num_stages").set_value(2).run()
    # Halve the stroke explicitly: switch_0 keeps the 3-stage default (28)
    # otherwise, which is a 2 mm / 10 mm split rather than 6 / 6.
    at.number_input(key="inj_switch_0").set_value(24.0)
    at.number_input(key="inj_velocity_0").set_value(20.0)
    at.number_input(key="inj_velocity_1").set_value(200.0)
    at.run()
    at.radio(key="wall_model").set_value("none")
    at.checkbox(key="two_phase_on").set_value(False).run()
    at.button[0].click().run()
    assert not at.exception
    rec = at.session_state["mfs_result"].metadata["injection_profile"]
    assert [s["velocity_mms"] for s in rec["stages"]] == [20.0, 200.0]
    # Equal stroke halves: equal volumes, and the slow stage takes 10x as long.
    v0, v1 = (s["volume_cm3"] for s in rec["stages"])
    t0, t1 = (s["time_s"] for s in rec["stages"])
    assert v0 == pytest.approx(v1)
    assert t0 == pytest.approx(t1 * 10.0)


def test_vp_above_the_metering_position_stops_the_run_with_a_message():
    at = _app()
    at.number_input(key="inj_vp_pos").set_value(35.0).run()
    at.button[0].click().run()
    assert not at.exception
    errors = "\n".join(str(e.value) for e in at.error)
    assert "V/P切替位置" in errors and "計量位置" in errors
    assert "mfs_result" not in at.session_state


def test_out_of_order_switch_positions_stop_the_run_with_a_message():
    at = _app()
    at.number_input(key="inj_num_stages").set_value(2).run()
    # Put the switch at V/P: the last stage becomes zero-length.
    at.number_input(key="inj_switch_0").set_value(18.0).run()
    at.button[0].click().run()
    assert not at.exception
    errors = "\n".join(str(e.value) for e in at.error)
    assert "射出条件が成立しません" in errors
    assert "mfs_result" not in at.session_state


def test_the_version_caption_survives_an_invalid_injection_condition():
    """The error must not take the build label off the screen with it."""
    at = _app()
    at.number_input(key="inj_vp_pos").set_value(35.0).run()
    at.button[0].click().run()
    captions = "\n".join(str(c.value) for c in at.caption)
    assert "v0." in captions


def test_a_shot_smaller_than_the_cavity_is_flagged_as_extrapolated():
    """Past V/P the map extrapolates; the pane has to say so.

    Default stroke displaces 23.6 cm^3, well over the default cavity, so the
    warning is provoked by shortening the stroke rather than by growing the
    part.
    """
    at = _app()
    at.radio(key="wall_model").set_value("none")
    at.checkbox(key="two_phase_on").set_value(False)
    at.number_input(key="inj_num_stages").set_value(1).run()
    at.number_input(key="inj_meter_pos").set_value(19.0).run()
    at.button[0].click().run()
    assert not at.exception
    warnings = "\n".join(str(w.value) for w in at.warning)
    assert "理論射出量" in warnings and "外挿" in warnings


def test_a_shot_that_covers_the_cavity_is_not_flagged():
    at = _app()
    at.radio(key="wall_model").set_value("none")
    at.checkbox(key="two_phase_on").set_value(False).run()
    at.button[0].click().run()
    assert not at.exception
    warnings = "\n".join(str(w.value) for w in at.warning)
    assert "外挿" not in warnings
    captions = "\n".join(str(c.value) for c in at.caption)
    assert "理論射出量" in captions


def test_the_default_clock_is_velocity_control():
    """v0.42.2: the screw profile is the default input, so the clock follows.

    Leaving the clock on "constant pressure" while the default injection
    input is a set of positions and speeds means the screen takes the
    machine's own injection time and hands back a different one.
    """
    at = _app()
    assert at.radio(key="wall_model").value == "skin"
    assert at.radio(key="skin_clock").value == "constant_rate"
    captions = "\n".join(str(c.value) for c in at.caption)
    assert "速度制御そのもの" not in captions


def test_constant_pressure_with_a_screw_profile_says_the_clock_will_stretch():
    """Positions and speeds are velocity control; holding pressure is not."""
    at = _app()
    at.radio(key="skin_clock").set_value("constant_pressure").run()
    captions = "\n".join(str(c.value) for c in at.caption)
    assert "速度制御そのもの" in captions
    at.radio(key="skin_clock").set_value("constant_rate").run()
    captions = "\n".join(str(c.value) for c in at.caption)
    assert "速度制御そのもの" not in captions


def test_the_stretch_notice_is_absent_in_direct_rate_mode():
    at = _app()
    at.radio(key="inj_mode").set_value("direct")
    at.radio(key="skin_clock").set_value("constant_pressure").run()
    captions = "\n".join(str(c.value) for c in at.caption)
    assert "速度制御そのもの" not in captions


def test_the_two_phase_panel_warns_when_the_shot_runs_past_vp():
    """Two-phase ON *and* machine conditions ON — the default combination.

    Every other two-phase test here turns the panel off, so this pair had no
    coverage together (@claude on PR #89). Shortening the stroke to 0.5 mm
    puts the theoretical displacement (0.98 cm^3) under both the cavity and
    the metered shot, which is exactly the silent-extrapolation case.
    """
    at = _app()
    at.radio(key="wall_model").set_value("none")
    at.checkbox(key="two_phase_on").set_value(True)
    at.checkbox(key="icm_on").set_value(True)
    at.number_input(key="inj_num_stages").set_value(1).run()
    at.number_input(key="inj_meter_pos").set_value(18.5).run()
    at.number_input(key="two_phase_shot_volume").set_value(4.5)
    at.button[0].click().run()
    assert not at.exception
    res = at.session_state["mfs_two_phase_result"]
    assert res is not None
    assert res.metadata["injection_extrapolated_past_vp"] is True
    warnings = "\n".join(str(w.value) for w in at.warning)
    assert "V/P より先を外挿" in warnings


def test_the_two_phase_panel_is_quiet_when_the_shot_fits_the_stroke():
    at = _app()
    at.radio(key="wall_model").set_value("none")
    at.checkbox(key="two_phase_on").set_value(True)
    at.checkbox(key="icm_on").set_value(True).run()
    at.number_input(key="two_phase_shot_volume").set_value(4.5)
    at.button[0].click().run()
    assert not at.exception
    res = at.session_state["mfs_two_phase_result"]
    assert res is not None and res.metadata["injection_extrapolated_past_vp"] is False
    warnings = "\n".join(str(w.value) for w in at.warning)
    assert "V/P より先を外挿" not in warnings


def test_adding_a_stage_below_an_edited_switch_names_the_stage_in_japanese():
    """The exact path called out in review: 2 段で switch_0=19 → 3 段。"""
    at = _app()
    at.number_input(key="inj_num_stages").set_value(2).run()
    at.number_input(key="inj_switch_0").set_value(19.0).run()
    at.number_input(key="inj_num_stages").set_value(3).run()
    at.button[0].click().run()
    assert not at.exception
    errors = "\n".join(str(e.value) for e in at.error)
    assert "第2段の速度切替位置" in errors and "第1段の速度切替位置" in errors
    # No internal field path survives into the message.
    assert "stages[" not in errors and "end_position_mm" not in errors
