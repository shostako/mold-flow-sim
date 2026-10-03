"""AppTest wiring for the screw-side injection block in ``app.py``.

The block offers the injection rate two ways: typed in directly, or computed
from the screw diameter and speed (``Q = pi*D^2/4 * v``). Since v0.55.0 the
screw side opens on one stage, which asks for the diameter and the speed and
nothing else; the metering and V/P positions, the switch points and a speed
per stage appear once the stage count is 2 or more. The checks are: the
defaults are the documented numbers and really reach the solver, one stage
hands the solver a flat rate, the staged widgets appear only when asked for,
the direct-rate path still wins when chosen, and an impossible condition stops
the run with a message instead of an exception.

Since v0.50.0 the page opens on the direct rate (589 cm³/s) and the layered
wall model (user's call, 2026-09-30). ``_app()`` therefore switches to the
screw side and the skin wall model -- the setting every test below was written
against (the skin clock radio only exists under the skin model);
``_page_as_opened()`` is the untouched page.

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
DEF_VELOCITY = 200.0
DEF_STAGES = 1
#: Positions shown once there are two stages or more.
DEF_METER = 30.0
DEF_VP = 18.0
#: The machine's own switch points, pre-filled at ``len(DEF_SWITCHES) + 1``
#: stages. Empty means "none known -- split the stroke evenly".
DEF_SWITCHES: tuple[float, ...] = (28.0, 22.0)
MACHINE_STAGES = len(DEF_SWITCHES) + 1
#: A screw small enough that the default stroke displaces less than the
#: cavity -- the way the extrapolation tests provoke the warning.
SMALL_SCREW_D = 20.0

SPAN = DEF_METER - DEF_VP
AREA_MM2 = math.pi * DEF_SCREW_D * DEF_SCREW_D / 4.0
DEF_Q = AREA_MM2 * DEF_VELOCITY / 1000.0


def _even_switch(i: int, n: int) -> float:
    """The app's fallback default for boundary ``i`` of ``n`` stages."""
    return DEF_METER - SPAN * (i + 1) / n


def _expected_switch(i: int) -> float:
    """What boundary ``i`` shows at the machine's own stage count."""
    return DEF_SWITCHES[i] if i < len(DEF_SWITCHES) else _even_switch(i, MACHINE_STAGES)


def _page_as_opened(timeout: float = 240.0) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    at.run()
    return at


def _app(timeout: float = 240.0) -> AppTest:
    """The page on the screw side (one stage) with the skin wall model."""
    at = _page_as_opened(timeout)
    at.radio(key="inj_mode").set_value("machine")
    at.radio(key="wall_model").set_value("skin")
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


def _keys(at: AppTest) -> set[str]:
    return {str(n.key) for n in at.number_input}


def _staged(n: int = MACHINE_STAGES, timeout: float = 240.0) -> AppTest:
    at = _app(timeout)
    at.number_input(key="inj_num_stages").set_value(n).run()
    return at


def _quick(at: AppTest) -> None:
    """Isothermal, no two-phase: the fastest run that still reads the rate."""
    at.radio(key="wall_model").set_value("none")
    at.checkbox(key="two_phase_on").set_value(False).run()


STAGED_ONLY = {"inj_meter_pos", "inj_vp_pos", "inj_switch_0", "inj_velocity_0"}


def test_the_page_opens_on_the_direct_rate_589():
    """v0.50.0: the direct rate is the opening input (user's call)."""
    at = _page_as_opened()
    assert at.radio(key="inj_mode").value == "direct"
    assert at.slider(key="inj_Q_direct").value == 589.0
    assert "inj_screw_d" not in _keys(at)
    assert "inj_num_stages" not in _keys(at)


def test_the_screw_side_opens_on_one_stage_with_diameter_and_speed_only():
    """v0.55.0: one stage asks for two numbers; the positions stay hidden."""
    at = _app()
    assert at.radio(key="inj_mode").value == "machine"
    assert at.number_input(key="inj_screw_d").value == DEF_SCREW_D
    assert at.number_input(key="inj_num_stages").value == DEF_STAGES == 1
    assert at.number_input(key="inj_velocity").value == DEF_VELOCITY
    assert not (STAGED_ONLY & _keys(at))
    assert f"{DEF_Q:.1f} cm³/s" in _captions(at)


def test_one_stage_hands_the_solver_a_flat_rate():
    at = _app()
    _quick(at)
    at.button[0].click().run()
    assert not at.exception
    md = at.session_state["mfs_result"].metadata
    assert md["injection_profile"] is None
    assert md["injection_Q_cm3s"] == pytest.approx(DEF_Q)
    inj = at.session_state["mfs_settings"]["injection"]
    assert inj["rate_input_mode"] == "machine"
    assert inj["injection_volume_flow_cm3s"] == pytest.approx(DEF_Q)
    assert inj["injection_profile"] is None
    assert inj["machine"] == {
        "screw_diameter_mm": DEF_SCREW_D,
        "stages": 1,
        "velocity_mms": DEF_VELOCITY,
    }


def test_doubling_the_speed_halves_the_fill_time():
    """The acceptance criterion: the speed setting owns the time axis."""
    at = _app()
    at.checkbox(key="icm_on").set_value(False)
    _quick(at)
    at.button[0].click().run()
    assert not at.exception
    slow = at.session_state["mfs_result"].total_fill_time_s
    at.number_input(key="inj_velocity").set_value(DEF_VELOCITY * 2.0).run()
    at.button[0].click().run()
    assert not at.exception
    fast = at.session_state["mfs_result"].total_fill_time_s
    assert fast == pytest.approx(slow / 2.0, rel=1e-9)


def test_more_stages_reveal_the_positions_and_the_stage_speeds():
    at = _staged()
    assert at.number_input(key="inj_meter_pos").value == DEF_METER
    assert at.number_input(key="inj_vp_pos").value == DEF_VP
    for i in range(MACHINE_STAGES - 1):
        assert at.number_input(key=f"inj_switch_{i}").value == pytest.approx(_expected_switch(i))
    assert [at.number_input(key=f"inj_velocity_{i}").value for i in range(MACHINE_STAGES)] == [
        DEF_VELOCITY
    ] * MACHINE_STAGES
    # Exactly one boundary fewer than there are stages: the last ends at V/P.
    assert f"inj_switch_{MACHINE_STAGES - 1}" not in _keys(at)
    assert "inj_velocity" not in _keys(at)


def test_the_stage_speeds_start_from_the_single_stage_speed_and_back():
    """Switching the stage count carries the speed instead of resetting it."""
    at = _app()
    at.number_input(key="inj_velocity").set_value(150.0).run()
    at.number_input(key="inj_num_stages").set_value(2).run()
    assert [at.number_input(key=f"inj_velocity_{i}").value for i in range(2)] == [150.0, 150.0]
    at.number_input(key="inj_velocity_0").set_value(120.0).run()
    at.number_input(key="inj_num_stages").set_value(1).run()
    assert at.number_input(key="inj_velocity").value == 120.0
    assert not (STAGED_ONLY & _keys(at))


def test_equal_speed_stages_are_arithmetically_one_stage():
    """Equal-speed steps: the map has kinks that do not bend anything."""
    at = _app()
    _quick(at)
    at.button[0].click().run()
    single = at.session_state["mfs_result"]
    at.number_input(key="inj_num_stages").set_value(MACHINE_STAGES).run()
    at.button[0].click().run()
    staged = at.session_state["mfs_result"]
    assert staged.metadata["injection_profile"] is not None
    assert staged.total_fill_time_s == pytest.approx(single.total_fill_time_s, rel=1e-12)
    assert staged.fill_time_s == pytest.approx(single.fill_time_s, rel=1e-9, nan_ok=True)


def test_the_representative_velocity_stays_its_own_slider():
    """Screw speed and gap-mean velocity are different quantities."""
    at = _app()
    assert at.slider(key="inj_rep_velocity").value == 200.0
    assert "代表流動速度" in str(at.slider(key="inj_rep_velocity").label)


def test_a_staged_condition_reaches_the_solver_and_the_settings_record():
    at = _staged()
    _quick(at)
    at.button[0].click().run()
    assert not at.exception
    res = at.session_state["mfs_result"]
    rec = res.metadata["injection_profile"]
    assert rec is not None
    assert rec["screw_diameter_mm"] == DEF_SCREW_D
    assert rec["vp_position_mm"] == DEF_VP
    assert len(rec["stages"]) == MACHINE_STAGES
    assert rec["stages"][0]["rate_cm3s"] == pytest.approx(DEF_Q)
    assert rec["mean_rate_cm3s"] == pytest.approx(DEF_Q)
    assert rec["total_volume_cm3"] == pytest.approx(AREA_MM2 * SPAN / 1000.0)
    assert rec["total_time_s"] == pytest.approx(SPAN / DEF_VELOCITY)
    assert res.metadata["injection_Q_cm3s"] is None
    inj = at.session_state["mfs_settings"]["injection"]
    assert inj["rate_input_mode"] == "machine"
    assert inj["injection_volume_flow_cm3s"] is None
    assert inj["injection_profile"]["total_time_s"] > 0
    assert inj["machine"] == {"screw_diameter_mm": DEF_SCREW_D, "stages": MACHINE_STAGES}


def test_direct_rate_mode_restores_the_slider_and_drops_the_screw():
    at = _staged()
    at.radio(key="inj_mode").set_value("direct").run()
    assert at.slider(key="inj_Q_direct").value == 589.0
    # The screw widgets are gone, not merely ignored.
    assert not ({"inj_screw_d", "inj_num_stages"} | STAGED_ONLY) & _keys(at)
    _quick(at)
    at.button[0].click().run()
    assert not at.exception
    res = at.session_state["mfs_result"]
    assert res.metadata["injection_profile"] is None
    assert res.metadata["injection_Q_cm3s"] == 589.0
    inj = at.session_state["mfs_settings"]["injection"]
    assert inj["rate_input_mode"] == "direct"
    assert inj["machine"] is None


def test_adding_a_stage_keeps_the_positions_already_set():
    """Only the boundary that did not exist yet gets a default.

    Streamlit keeps a widget's value across reruns once it has one, and that
    is the behaviour to want here: adding a stage must not move the switch
    points the user already dialled in. The even split is the default for the
    *new* widget only.
    """
    n = MACHINE_STAGES + 1
    at = _staged()
    at.number_input(key="inj_num_stages").set_value(n).run()
    for i in range(MACHINE_STAGES - 1):
        assert at.number_input(key=f"inj_switch_{i}").value == pytest.approx(_expected_switch(i))
    assert at.number_input(key=f"inj_switch_{n - 2}").value == pytest.approx(_even_switch(n - 2, n))
    assert at.number_input(key=f"inj_velocity_{n - 1}").value == DEF_VELOCITY
    assert f"inj_switch_{n - 1}" not in _keys(at)


def test_multi_stage_volumes_and_times_reach_the_solver():
    at = _staged(2)
    # Halve the stroke explicitly rather than trust the two-stage default.
    at.number_input(key="inj_switch_0").set_value((DEF_METER + DEF_VP) / 2.0)
    at.number_input(key="inj_velocity_0").set_value(20.0)
    at.number_input(key="inj_velocity_1").set_value(200.0)
    at.run()
    _quick(at)
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
    at = _staged(2)
    at.number_input(key="inj_vp_pos").set_value(DEF_METER + 5.0).run()
    at.button[0].click().run()
    assert not at.exception
    assert "V/P切替位置" in _errors(at) and "計量位置" in _errors(at)
    assert "mfs_result" not in at.session_state


def test_out_of_order_switch_positions_stop_the_run_with_a_message():
    at = _staged(2)
    # Put the switch at V/P: the last stage becomes zero-length.
    at.number_input(key="inj_switch_0").set_value(DEF_VP).run()
    at.button[0].click().run()
    assert not at.exception
    assert "射出条件が成立しません" in _errors(at)
    assert "mfs_result" not in at.session_state


def test_adding_a_stage_below_an_edited_switch_names_the_stage_in_japanese():
    """The path called out in review: edit a boundary down, then add a stage."""
    at = _staged(2)
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
    at = _staged(2)
    at.number_input(key="inj_vp_pos").set_value(DEF_METER + 5.0).run()
    at.button[0].click().run()
    assert "v0." in _captions(at)


def test_a_staged_shot_smaller_than_the_cavity_is_flagged_as_extrapolated():
    """Past V/P the map extrapolates; the pane has to say so.

    The default stroke covers this part, so the warning is provoked by a
    small screw (less displacement per millimetre) rather than by growing
    the part.
    """
    at = _staged(2)
    at.number_input(key="inj_screw_d").set_value(SMALL_SCREW_D)
    _quick(at)
    at.button[0].click().run()
    assert not at.exception
    assert "理論射出量" in _warnings(at) and "外挿" in _warnings(at)
    assert at.session_state["mfs_result"].metadata["injection_extrapolated_past_vp"] is True


def test_a_staged_shot_that_covers_the_cavity_is_not_flagged():
    at = _staged(2)
    _quick(at)
    at.button[0].click().run()
    assert not at.exception
    assert "外挿" not in _warnings(at)
    assert "理論射出量" in _captions(at)
    assert at.session_state["mfs_result"].metadata["injection_extrapolated_past_vp"] is False


def test_one_stage_has_no_stroke_to_extrapolate_past():
    """A single speed defines no stroke, so the stroke wording never appears."""
    at = _app()
    at.number_input(key="inj_screw_d").set_value(SMALL_SCREW_D)
    _quick(at)
    at.button[0].click().run()
    assert not at.exception
    assert "理論射出量" not in _captions(at) and "外挿" not in _warnings(at)
    assert not at.session_state["mfs_result"].metadata.get("injection_extrapolated_past_vp")


def test_the_extrapolation_warning_reads_the_solver_flag():
    """Not the UI's own volume comparison (Codex P2).

    With ICM the profile map is read through the *open-gap* volume, which a
    stroke can miss while still clearing the cavity as drawn. The flag lives
    in the solver, which knows what it swept.
    """
    at = _staged(2)
    at.number_input(key="inj_screw_d").set_value(SMALL_SCREW_D)
    _quick(at)
    at.button[0].click().run()
    assert not at.exception
    md = at.session_state["mfs_result"].metadata
    assert md["injection_extrapolated_past_vp"] is True
    assert md["injection_swept_volume_cm3"] > 0
    assert "掃く体積" in _warnings(at)


def test_the_default_clock_is_velocity_control():
    """v0.42.2: the skin clock defaults to velocity control.

    Leaving the clock on "constant pressure" while the injection input is a
    screw speed means the screen takes the machine's own injection time and
    hands back a different one.
    """
    at = _app()
    assert at.radio(key="wall_model").value == "skin"
    assert at.radio(key="skin_clock").value == "constant_rate"
    assert "速度制御そのもの" not in _captions(at)


@pytest.mark.parametrize("stages", [1, 2])
def test_constant_pressure_with_a_screw_speed_says_the_clock_will_stretch(stages):
    """A screw speed is velocity control, one stage or several."""
    at = _staged(stages)
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
    """Two-phase ON *and* a staged screw condition ON.

    Every other two-phase test here turns the panel off, so this pair had no
    coverage together (@claude on PR #89). A small screw puts the theoretical
    displacement under the metered shot, which is exactly the
    silent-extrapolation case.
    """
    at = _staged(2)
    at.number_input(key="inj_screw_d").set_value(SMALL_SCREW_D)
    at.radio(key="wall_model").set_value("none")
    at.checkbox(key="two_phase_on").set_value(True)
    at.checkbox(key="icm_on").set_value(True).run()
    at.number_input(key="two_phase_shot_volume").set_value(4.5)
    at.button[0].click().run()
    assert not at.exception
    res = at.session_state["mfs_two_phase_result"]
    assert res is not None
    assert res.metadata["injection_extrapolated_past_vp"] is True
    assert "V/P より先を外挿" in _warnings(at)


def test_the_two_phase_panel_is_quiet_when_the_shot_fits_the_stroke():
    at = _staged(2)
    at.radio(key="wall_model").set_value("none")
    at.checkbox(key="two_phase_on").set_value(True)
    at.checkbox(key="icm_on").set_value(True).run()
    at.number_input(key="two_phase_shot_volume").set_value(4.5)
    at.button[0].click().run()
    assert not at.exception
    res = at.session_state["mfs_two_phase_result"]
    assert res is not None and res.metadata["injection_extrapolated_past_vp"] is False
    assert "V/P より先を外挿" not in _warnings(at)
