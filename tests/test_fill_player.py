"""Tests for the fill-front HTML player and its shared frame-timing helpers.

The GIF, the PNG frame sequence and the HTML scrubber must all agree on
what frame ``k`` shows. That agreement now lives in one place
(``fill_frame_times``); these tests pin it so the three renderers cannot
drift apart again.
"""

from __future__ import annotations

import base64
import json
import re

import numpy as np
import pytest
from PIL import Image

from core import (
    CONTROLS_HEIGHT_PX,
    HeleShawSolver,
    MaterialDB,
    build_demo_geometry,
    build_fill_player_html,
    export_frames,
    fill_frame_fractions,
    fill_frame_times,
    fill_player_height_px,
    wrap_standalone_html,
)


@pytest.fixture(scope="module")
def result():
    geom = build_demo_geometry(cell_size_mm=3.0)
    mat = MaterialDB()["PP"]
    return HeleShawSolver(geom, mat).solve(num_frames=6)


# --- fill_frame_times -------------------------------------------------------


def test_frame_times_span_first_step_to_full_fill(result):
    n = 6
    t = fill_frame_times(result, n)
    t_max = float(np.nanmax(result.fill_time_s))
    assert t.shape == (n,)
    assert t[-1] == pytest.approx(t_max)
    assert t[0] == pytest.approx(t_max / n)
    assert np.all(np.diff(t) > 0)


def test_frame_times_rejects_non_positive_count(result):
    with pytest.raises(ValueError, match="num_frames"):
        fill_frame_times(result, 0)


def test_frame_times_match_exported_frame_count(result, tmp_path):
    """The PNG sequence and the timing helper must stay the same length."""
    n = 6
    paths = export_frames(result, tmp_path / "frames", num_frames=n)
    assert len(paths) == len(fill_frame_times(result, n)) == n


# --- fill_frame_fractions ---------------------------------------------------


def test_frame_fractions_rise_monotonically_to_full(result):
    f = fill_frame_fractions(result, 6)
    assert f.shape == (6,)
    assert np.all(f >= 0.0) and np.all(f <= 1.0)
    assert np.all(np.diff(f) >= -1e-12)
    assert f[-1] == pytest.approx(1.0)


# --- build_fill_player_html -------------------------------------------------


def _payload(html: str) -> dict:
    m = re.search(r"const D = (\{.*?\});", html, re.S)
    assert m, "player payload not found"
    return json.loads(m.group(1))


def test_player_embeds_every_frame_and_its_readout(result, tmp_path):
    n = 6
    paths = export_frames(result, tmp_path / "frames", num_frames=n)
    times = fill_frame_times(result, n)
    fills = fill_frame_fractions(result, n)
    html = build_fill_player_html(paths, times, fills, fps=8)

    d = _payload(html)
    assert len(d["frames"]) == n
    assert all(src.startswith("data:image/png;base64,") for src in d["frames"])
    assert d["times"] == pytest.approx(list(times))
    assert d["fills"] == pytest.approx(list(fills))
    assert d["fps"] == 8


def test_player_frame_payload_decodes_to_the_actual_png(result, tmp_path):
    paths = export_frames(result, tmp_path / "frames", num_frames=6)
    html = build_fill_player_html(
        paths, fill_frame_times(result, 6), fill_frame_fractions(result, 6)
    )
    src = _payload(html)["frames"][0]
    raw = base64.b64decode(src.split(",", 1)[1])
    assert raw == paths[0].read_bytes()
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"


def test_player_is_self_contained(result, tmp_path):
    """No external fetches: the CSP-sandboxed iframe would block them."""
    paths = export_frames(result, tmp_path / "frames", num_frames=6)
    html = build_fill_player_html(
        paths, fill_frame_times(result, 6), fill_frame_fractions(result, 6)
    )
    assert "http://" not in html
    assert "https://" not in html
    assert "<script>" in html and "seek" in html


def test_player_rejects_mismatched_or_empty_inputs(result, tmp_path):
    paths = export_frames(result, tmp_path / "frames", num_frames=6)
    times = fill_frame_times(result, 6)
    fills = fill_frame_fractions(result, 6)
    with pytest.raises(ValueError, match="must not be empty"):
        build_fill_player_html([], [], [])
    with pytest.raises(ValueError, match="equal length"):
        build_fill_player_html(paths, times[:-1], fills)
    with pytest.raises(ValueError, match="fps"):
        build_fill_player_html(paths, times, fills, fps=0)


# --- component sizing (regression: PR #47 review) -------------------------


def test_player_caps_the_image_at_its_native_width(result, tmp_path):
    """A column wider than the PNG must not stretch the image.

    ``st.components.v1.html`` gets a fixed height while the iframe width
    follows the column. Without the cap, a 900 px column renders the 7:5
    image 642 px tall, overflowing the fixed height — and with
    ``scrolling=False`` the controls become unreachable.
    """
    paths = export_frames(result, tmp_path / "frames", num_frames=6)
    with Image.open(paths[0]) as im:
        native_w, native_h = im.width, im.height
    html = build_fill_player_html(
        paths, fill_frame_times(result, 6), fill_frame_fractions(result, 6)
    )
    assert f"max-width:{native_w}px" in html
    assert "__MAXW__" not in html
    # the cap bounds the image height, so the declared component height fits
    assert fill_player_height_px(paths) == native_h + CONTROLS_HEIGHT_PX


def test_player_height_covers_image_plus_controls(result, tmp_path):
    paths = export_frames(result, tmp_path / "frames", num_frames=6)
    with Image.open(paths[0]) as im:
        native_h = im.height
    h = fill_player_height_px(paths)
    assert h > native_h  # controls are not clipped
    assert h - native_h == CONTROLS_HEIGHT_PX


def test_player_height_rejects_empty_input():
    with pytest.raises(ValueError, match="must not be empty"):
        fill_player_height_px([])


# --- wrap_standalone_html ---------------------------------------------------


@pytest.fixture(scope="module")
def fragment(result, tmp_path_factory):
    paths = export_frames(result, tmp_path_factory.mktemp("frames"), num_frames=6)
    return build_fill_player_html(
        paths, fill_frame_times(result, 6), fill_frame_fractions(result, 6)
    )


def test_standalone_declares_utf8_before_any_japanese_text(fragment):
    """The charset declaration must precede the first non-ASCII byte.

    A browser opening the file over ``file://`` has no HTTP header to go on.
    Without an early ``<meta charset>`` it falls back to the platform legacy
    encoding (CP932 on a Japanese Windows box) and the button labels turn
    into mojibake — the whole reason the fragment cannot be shipped as-is.
    """
    doc = wrap_standalone_html(fragment, title="充填アニメーション")
    raw = doc.encode("utf-8")
    charset_at = raw.index(b'<meta charset="utf-8">')
    first_non_ascii = next(i for i, b in enumerate(raw) if b > 0x7F)
    assert charset_at < first_non_ascii
    # and the declaration sits inside the first 1024 bytes browsers sniff
    assert charset_at < 1024


def test_standalone_is_a_complete_document_containing_the_fragment(fragment):
    doc = wrap_standalone_html(fragment, title="充填アニメーション")
    assert doc.startswith("<!doctype html>")
    assert '<html lang="ja">' in doc
    assert doc.rstrip().endswith("</html>")
    # the fragment is embedded verbatim: the shipped player is the same player
    assert fragment in doc


def test_standalone_stays_offline_self_contained(fragment):
    """No network reference may sneak in via the wrapper."""
    doc = wrap_standalone_html(fragment, title="充填アニメーション", note="v0.0.0")
    body = re.sub(r"data:image/png;base64,[A-Za-z0-9+/=]+", "", doc)
    assert "http://" not in body
    assert "https://" not in body


def test_standalone_note_is_optional_and_rendered(fragment):
    with_note = wrap_standalone_html(fragment, title="T", note="v0.15.0 (abc1234)")
    assert "v0.15.0 (abc1234)" in with_note
    without = wrap_standalone_html(fragment, title="T")
    assert "<small>" not in without


def test_standalone_escapes_title_and_note(fragment):
    doc = wrap_standalone_html(fragment, title="a<b>&c", note="<script>x</script>")
    assert "<title>a&lt;b&gt;&amp;c</title>" in doc
    assert "<script>x</script>" not in doc.replace(fragment, "")
    assert "&lt;script&gt;x&lt;/script&gt;" in doc


def test_standalone_rejects_empty_title(fragment):
    with pytest.raises(ValueError, match="title must not be empty"):
        wrap_standalone_html(fragment, title="")


def test_standalone_leaves_no_template_placeholder(fragment):
    doc = wrap_standalone_html(fragment, title="充填アニメーション", note="v0")
    assert "__TITLE__" not in doc
    assert "__HEADING__" not in doc
    assert "__BODY__" not in doc


def test_player_labels_replace_the_default_readout(result, tmp_path):
    """Per-frame labels ride in the payload; the default readout stays ``null``
    so the page script can tell the two modes apart."""
    n = 4
    paths = export_frames(result, tmp_path / "frames", num_frames=n)
    zeros = [0.0] * n
    plain = _payload(build_fill_player_html(paths, zeros, zeros))
    assert plain["labels"] is None
    labels = [f"step {i}" for i in range(n)]
    d = _payload(build_fill_player_html(paths, zeros, zeros, labels=labels))
    assert d["labels"] == labels
    with pytest.raises(ValueError, match="labels"):
        build_fill_player_html(paths, zeros, zeros, labels=labels[:-1])
