"""Tests for the run-settings record bundled with the results ZIP.

The point of this record is that a downloaded ZIP can be reproduced. Before
it existed, ``metadata.json`` held only what the solver computed, so the
geometry had to be recovered by measuring the rendered images and searching
for a config whose volume and tau_max matched. These tests pin the two
properties that make the record worth having: it must carry *every* config
field, and it must not carry the contents of an uploaded spec.
"""

from __future__ import annotations

import json

import pytest

from core.geometry import DirectGateConfig, FilmGateConfig
from core.profile_gate import ProfilePlateConfig
from core.settings_record import config_settings, file_fingerprint, settings_json


def test_every_config_field_is_recorded():
    """A field that is not recorded is a field the reader has to guess.

    Compared against the dataclass itself rather than a hand-written list, so
    a new geometry parameter cannot be added without appearing here.
    """
    cfg = DirectGateConfig(plate_w_mm=300.0, plate_h_mm=50.0, plate_thk_mm=0.35)
    rec = config_settings("Direct gate (parametric)", cfg)
    assert set(rec["config"]) == set(vars(cfg))
    assert rec["config"]["plate_w_mm"] == 300.0
    assert rec["input"] == "Direct gate (parametric)"


def test_none_fields_survive_as_null():
    """``None`` means "uniform plate", which is a setting, not a missing value."""
    cfg = DirectGateConfig(
        plate_w_mm=300.0, plate_h_mm=50.0, plate_thk_mm=0.4, plate_lower_thk_mm=None
    )
    rec = config_settings("Direct gate (parametric)", cfg)
    assert "plate_lower_thk_mm" in rec["config"]
    assert rec["config"]["plate_lower_thk_mm"] is None


def test_tuple_fields_become_lists_so_json_round_trips():
    """The balancer stages are tuples; JSON has no tuple."""
    cfg = FilmGateConfig(
        plate_w_mm=300.0,
        plate_h_mm=50.0,
        plate_thk_mm=0.35,
        runner_long_mm=300.0,
        runner_short_diameter_mm=10.0,
        runner_depth_mm=25.0,
        runner_thk_mm=2.0,
        runner_flat_depth_mm=10.0,
        runner_slope_depth_mm=15.0,
        valve_gate_diameter_mm=3.0,
        gate_width_mm=280.0,
        balancer_enabled=True,
        balancer_base_widths_mm=(20.0, 40.0),
        balancer_thicknesses_mm=(0.2, 0.3),
    )
    rec = config_settings("Film gate 1 (左右対称)", cfg)
    assert rec["config"]["balancer_base_widths_mm"] == [20.0, 40.0]
    assert json.loads(settings_json(rec)) == rec


def test_uploaded_spec_is_fingerprinted_not_embedded():
    """The ZIP is made to be forwarded; drawing dimensions must not ride along.

    Checked by looking for the actual numbers in the serialized record, not by
    inspecting which keys were set -- a future change that starts embedding
    the spec would keep the keys and still leak.
    """
    spec_text = json.dumps({"name": "real_part", "gate_exit_width": 299.0, "land": {"depth": 0.35}})
    rec = config_settings(
        "Profile gate (JSONスペック)",
        ProfilePlateConfig(plate_w_mm=300.0, plate_h_mm=50.0, plate_thk_mm=0.35),
        spec=file_fingerprint("real_part.json", spec_text),
    )
    blob = settings_json(rec)
    assert "299.0" not in blob
    assert "gate_exit_width" not in blob
    assert rec["spec"]["sha256"] == file_fingerprint("x", spec_text)["sha256"]
    assert rec["spec"]["name"] == "real_part.json"


def test_fingerprint_identifies_the_revision():
    """Same bytes -> same hash, one edited number -> different hash."""
    a = file_fingerprint("s.json", '{"depth": 0.35}')
    b = file_fingerprint("s.json", '{"depth": 0.35}')
    c = file_fingerprint("s.json", '{"depth": 0.36}')
    assert a["sha256"] == b["sha256"]
    assert a["sha256"] != c["sha256"]
    assert a["bytes"] == len('{"depth": 0.35}')


def test_fingerprint_accepts_bytes_and_text_alike():
    """Uploads arrive as bytes, pasted JSON as text; both must hash the same."""
    text = '{"depth": 0.35}'
    assert file_fingerprint("s", text)["sha256"] == file_fingerprint("s", text.encode())["sha256"]


def test_extra_fields_land_next_to_the_config():
    """Cell size is not a config field for the profile gate; it is a builder arg."""
    rec = config_settings(
        "Profile gate (JSONスペック)",
        ProfilePlateConfig(plate_w_mm=300.0, plate_h_mm=50.0, plate_thk_mm=0.35),
        cell_size_mm=0.8,
    )
    assert rec["cell_size_mm"] == 0.8


def test_config_is_optional_for_image_input():
    """The image importer has no config dataclass, only loose parameters."""
    rec = config_settings("画像から生成 (PNG/JPG)", None, threshold=128, invert=False)
    assert "config" not in rec
    assert rec["threshold"] == 128


def test_rejects_a_dataclass_type_passed_by_mistake():
    with pytest.raises(TypeError):
        config_settings("x", DirectGateConfig)


def test_rejects_an_empty_source():
    with pytest.raises(ValueError):
        config_settings("", None)
