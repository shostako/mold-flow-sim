"""AppTest wiring for the screw-side injection block in ``app.py``.

The block replaces one number (a flat injection rate) with a machine
condition, so the checks are: the default really is the machine condition and
really reaches the solver, the direct-rate path still exists and still wins
when chosen, the stage widgets appear only when asked for, and an impossible
condition stops the run with a message instead of an exception.

**The expected defaults live in one block below.** This file is shared with
the sibling repos (mold-flow-fangate / -fangate2), which run the same machine
on much larger parts and therefore meter a much longer stroke; porting the
tests then means rewriting one block instead of hunting literals through
twenty assertions. Everything else here is derived from those numbers, so a
default change is a one-line edit in ``app.py`` and a one-line edit here.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "app.py"

# --- the app's own injection defaults (keep in sync with app.py) -----------
DEF_SCREW_D = 50.0
DEF_METER = 30.0
DEF_VP = 18.0
DEF_STAGES = 3
#: Stage boundaries the app pre-fills at ``DEF_STAGES``. Empty means "no
#: machine-specific switch points known -- split the stroke evenly".
DEF_SWITCHES: tuple[float, ...] = (28.0, 22.0)
DEF_VELOCITY = 200.0

SPAN = DEF_METER - DEF_VP
AREA_MM2 = math.pi * DEF_SCREW_D * DEF_SCREW_D / 4.0


def _even_switch(i: int, n: int) -> float:
    """The app's fallback default for boundary ``i`` of ``n`` stages."""
    return DEF_METER - SPAN * (i + 1) / n


def _expected_switch(i: int) -> float:
    """What boundary ``i`` shows at the default stage count."""
    return DEF_SWITCHES[i] if i < len(DEF_SWITCHES) else _even_switch(i, DEF_STAGES)


def _app(timeout: float = 240.0) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    at.run()
    return at


def _labels(at: AppTest, prefix: str) -> list[str]:
    out = []
    for group in (at.number_input, at.slider):
        out.extend(str(e.label) for e in group if str(e.label).startswith(prefix))
    return out


def _captions(at: AppTest) -> str:
    return "\n".join(str(c.value) for c in at.caption)


def _warnings(at: AppTest) -> str:
    return "\n".join(str(w.value) for w in at.warning)


def _errors(at: AppTest) -> str:
    return "\n".join(str(e.value) for e in at.error)


def test_default_is_the_machine_condition_with_the_documented_numbers():
    at = _app()
    assert at.radio(key="inj_mode").value == "machine"
    assert at.number_input(key="inj_screw_d").value == DEF_SCREW_D
    assert at.number_input(key="inj_meter_pos").value == DEF_METER
    assert at.number_input(key="inj_vp_pos").value == DEF_VP
    assert at.number_input(key="inj_num_stages").value == DEF_STAGES
    for i in range(DEF_STAGES - 1):
        assert at.number_input(key=f"inj_switch_{i}").value == pytest.approx(_expected_switch(i))
    assert [at.number_input(key=f"inj_velocity_{i}").value for i in range(DEF_STAGES)] == [
        DEF_VELOCITY
    ] * DEF_STAGES
    # Exactly one boundary fewer than there are stages: the last ends at V/P.
    assert not [n for n in at.number_input if n.key == f"inj_switch_{DEF_STAGES - 1}"]


def test_the_equal_speed_default_is_arithmetically_one_stage():
    """Equal-speed steps: the map has kinks that do not bend anything."""
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
    assert at.number_input(key="inj_velocity_0").value == DEF_VELOCITY
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
    assert rec["screw_diameter_mm"] == DEF_SCREW_D
    assert rec["vp_position_mm"] == DEF_VP
    assert len(rec["stages"]) == DEF_STAGES
    expected_Q = AREA_MM2 * DEF_VELOCITY / 1000.0
    assert rec["stages"][0]["rate_cm3s"] == pytest.approx(expected_Q)
    assert rec["mean_rate_cm3s"] == pytest.approx(expected_Q)
    assert rec["total_volume_cm3"] == pytest.approx(AREA_MM2 * SPAN / 1000.0)
    assert rec["total_time_s"] == pytest.approx(SPAN / DEF_VELOCITY)
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
    for i in range(DEF_STAGES):
        at.number_input(key=f"inj_velocity_{i}").set_value(DEF_VELOCITY * 2.0)
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
    is the behaviour to want here: adding a stage must not move the switch
    points the user already dialled in. The even split is the default for the
    *new* widget only.
    """
    n = DEF_STAGES + 1
    at = _app()
    at.number_input(key="inj_num_stages").set_value(n).run()
    for i in range(DEF_STAGES - 1):
        assert at.number_input(key=f"inj_switch_{i}").value == pytest.approx(_expected_switch(i))
    assert at.number_input(key=f"inj_switch_{n - 2}").value == pytest.approx(_even_switch(n - 2, n))
    assert at.number_input(key=f"inj_velocity_{n - 1}").value == DEF_VELOCITY
    assert not [x for x in at.number_input if x.key == f"inj_switch_{n - 1}"]


def test_multi_stage_volumes_and_times_reach_the_solver():
    at = _app()
    at.number_input(key="inj_num_stages").set_value(2).run()
    # Halve the stroke explicitly: switch_0 keeps the default-stage-count
    # value otherwise, which is not an even split.
    at.number_input(key="inj_switch_0").set_value((DEF_METER + DEF_VP) / 2.0)
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
    at.number_input(key="inj_vp_pos").set_value(DEF_METER + 5.0).run()
    at.button[0].click().run()
    assert not at.exception
    assert "V/P切替位置" in _errors(at) and "計量位置" in _errors(at)
    assert "mfs_result" not in at.session_state


def test_out_of_order_switch_positions_stop_the_run_with_a_message():
    at = _app()
    at.number_input(key="inj_num_stages").set_value(2).run()
    # Put the switch at V/P: the last stage becomes zero-length.
    at.number_input(key="inj_switch_0").set_value(DEF_VP).run()
    at.button[0].click().run()
    assert not at.exception
    assert "射出条件が成立しません" in _errors(at)
    assert "mfs_result" not in at.session_state


def test_adding_a_stage_below_an_edited_switch_names_the_stage_in_japanese():
    """The path called out in review: edit a boundary down, then add a stage."""
    at = _app()
    at.number_input(key="inj_num_stages").set_value(2).run()
    at.number_input(key="inj_switch_0").set_value(DEF_VP + 1.0).run()
    at.number_input(key="inj_num_stages").set_value(3).run()
    at.button[0].click().run()
    assert not at.exception
    errors = _errors(at)
    assert "第2段の速度切替位置" in errors and "第1段の速度切替位置" in errors
    # No internal field path survives into the message.
    assert "stages[" not in errors and "end_position_mm" not in errors


def test_the_version_caption_survives_an_invalid_injection_condition():
    """The error must not take the build label off the screen with it."""
    at = _app()
    at.number_input(key="inj_vp_pos").set_value(DEF_METER + 5.0).run()
    at.button[0].click().run()
    assert "v0." in _captions(at)


def test_a_shot_smaller_than_the_cavity_is_flagged_as_extrapolated():
    """Past V/P the map extrapolates; the pane has to say so.

    The default stroke covers this part, so the warning is provoked by
    collapsing the stroke rather than by growing the part.
    """
    at = _app()
    at.radio(key="wall_model").set_value("none")
    at.checkbox(key="two_phase_on").set_value(False)
    at.number_input(key="inj_num_stages").set_value(1).run()
    at.number_input(key="inj_meter_pos").set_value(DEF_VP + 0.5).run()
    at.button[0].click().run()
    assert not at.exception
    assert "理論射出量" in _warnings(at) and "外挿" in _warnings(at)


def test_a_shot_that_covers_the_cavity_is_not_flagged():
    at = _app()
    at.radio(key="wall_model").set_value("none")
    at.checkbox(key="two_phase_on").set_value(False).run()
    at.button[0].click().run()
    assert not at.exception
    assert "外挿" not in _warnings(at)
    assert "理論射出量" in _captions(at)


def test_the_default_clock_is_velocity_control():
    """v0.42.2: the screw profile is the default input, so the clock follows.

    Leaving the clock on "constant pressure" while the default injection
    input is a set of positions and speeds means the screen takes the
    machine's own injection time and hands back a different one.
    """
    at = _app()
    assert at.radio(key="wall_model").value == "skin"
    assert at.radio(key="skin_clock").value == "constant_rate"
    assert "速度制御そのもの" not in _captions(at)


def test_constant_pressure_with_a_screw_profile_says_the_clock_will_stretch():
    """Positions and speeds are velocity control; holding pressure is not."""
    at = _app()
    at.radio(key="skin_clock").set_value("constant_pressure").run()
    assert "速度制御そのもの" in _captions(at)
    at.radio(key="skin_clock").set_value("constant_rate").run()
    assert "速度制御そのもの" not in _captions(at)


def test_the_stretch_notice_is_absent_in_direct_rate_mode():
    at = _app()
    at.radio(key="inj_mode").set_value("direct")
    at.radio(key="skin_clock").set_value("constant_pressure").run()
    assert "速度制御そのもの" not in _captions(at)


def test_the_two_phase_panel_warns_when_the_shot_runs_past_vp():
    """Two-phase ON *and* machine conditions ON — the default combination.

    Every other two-phase test here turns the panel off, so this pair had no
    coverage together (@claude on PR #89). Collapsing the stroke puts the
    theoretical displacement under both the cavity and the metered shot,
    which is exactly the silent-extrapolation case.
    """
    at = _app()
    at.radio(key="wall_model").set_value("none")
    at.checkbox(key="two_phase_on").set_value(True)
    at.checkbox(key="icm_on").set_value(True)
    at.number_input(key="inj_num_stages").set_value(1).run()
    at.number_input(key="inj_meter_pos").set_value(DEF_VP + 0.5).run()
    at.number_input(key="two_phase_shot_volume").set_value(4.5)
    at.button[0].click().run()
    assert not at.exception
    res = at.session_state["mfs_two_phase_result"]
    assert res is not None
    assert res.metadata["injection_extrapolated_past_vp"] is True
    assert "V/P より先を外挿" in _warnings(at)


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
    assert "V/P より先を外挿" not in _warnings(at)
