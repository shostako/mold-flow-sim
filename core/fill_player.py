"""Self-contained HTML5 player for the fill-front frame sequence.

``export_frames`` already writes one PNG per frame, so the scrubber costs no
extra rendering: the PNGs are inlined as data URIs and driven by a small
vanilla-JS player. Everything (play/pause, seek, speed) happens in the
browser, so dragging the scrubber never round-trips to the Streamlit server.

The markup is designed for ``st.components.v1.html``, which renders it inside
a sandboxed iframe with no access to the parent page's stylesheet — hence the
inline CSS and the transparent body that lets the host theme show through.

Glyphs are restricted to the Geometric Shapes block (U+25B6 / U+25C0) and
U+2759 bars. Emoji and the media-control block (U+23EE / U+23ED / U+1F501)
render as tofu boxes on machines without an emoji font, which is exactly
what happened on the first headless check.
"""

from __future__ import annotations

import base64
import html
import json
from collections.abc import Sequence
from pathlib import Path

from PIL import Image

# Vertical space the controls occupy under the image [px]. Callers sizing the
# component (``st.components.v1.html(..., height=...)``) add this to the
# image height so the controls are never clipped.
CONTROLS_HEIGHT_PX = 104


def _data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _native_size(path: str | Path) -> tuple[int, int]:
    """Pixel size of a rendered frame PNG."""
    with Image.open(Path(path)) as im:
        return int(im.width), int(im.height)


def fill_player_height_px(frame_paths: Sequence[str | Path]) -> int:
    """Component height [px] that always fits the player without clipping.

    ``st.components.v1.html`` takes a fixed height while the iframe width
    follows the surrounding column, so the height cannot be derived from the
    image's *rendered* width. The player therefore caps the image at its
    native width (upscaling a 700x500 matplotlib PNG only blurs it anyway),
    which bounds the image height by the native height — and that is what
    this returns, plus the controls. Without the cap a column wider than the
    native width grows the image past the fixed height and, with
    ``scrolling=False``, puts the controls out of reach.
    """
    if not frame_paths:
        raise ValueError("frame_paths must not be empty")
    return _native_size(frame_paths[0])[1] + CONTROLS_HEIGHT_PX


def build_fill_player_html(
    frame_paths: Sequence[str | Path],
    times_s: Sequence[float],
    fill_fractions: Sequence[float],
    *,
    fps: int = 8,
    autoplay: bool = True,
    labels: Sequence[str] | None = None,
) -> str:
    """Build a standalone HTML player for the fill-front frames.

    ``frame_paths``, ``times_s`` and ``fill_fractions`` must be the same
    length and in frame order (use ``visualizer.fill_frame_times`` /
    ``fill_frame_fractions`` so the readout matches the rendered PNG).

    ``labels`` replaces the default ``t = … s   充填 … %`` readout with one
    string per frame. The two-phase animation uses it: its second phase has
    no clock, only an order, so a time readout would be a lie there.
    """
    n = len(frame_paths)
    if n == 0:
        raise ValueError("frame_paths must not be empty")
    if len(times_s) != n or len(fill_fractions) != n:
        raise ValueError(
            f"frame_paths ({n}), times_s ({len(times_s)}) and "
            f"fill_fractions ({len(fill_fractions)}) must have equal length"
        )
    if labels is not None and len(labels) != n:
        raise ValueError(f"labels ({len(labels)}) must match frame_paths ({n})")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    frames = [_data_uri(Path(p)) for p in frame_paths]
    payload = json.dumps(
        {
            "frames": frames,
            "times": [float(t) for t in times_s],
            "fills": [float(f) for f in fill_fractions],
            "fps": int(fps),
            "autoplay": bool(autoplay),
            "labels": [str(x) for x in labels] if labels is not None else None,
        }
    )
    native_w = _native_size(frame_paths[0])[0]
    return _TEMPLATE.replace("__PAYLOAD__", payload).replace("__MAXW__", str(native_w))


_TEMPLATE = """
<style>
  :root { color-scheme: light dark; }
  body { margin:0; background:transparent;
         font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
  #wrap { display:flex; flex-direction:column; gap:6px; }
  #stage { width:100%; max-width:__MAXW__px; margin:0 auto;
           background:#fff; border-radius:6px; overflow:hidden; line-height:0; }
  #stage img { width:100%; height:auto; display:block; }
  #bar { display:flex; align-items:center; gap:10px; }
  #seek { flex:1; height:22px; cursor:pointer; accent-color:#2ecc71; }
  button { border:1px solid rgba(128,128,128,.45); background:rgba(128,128,128,.10);
           color:inherit; border-radius:6px; padding:3px 9px; font-size:13px;
           cursor:pointer; line-height:1.5; }
  button:hover { background:rgba(128,128,128,.24); }
  button.on { background:#2ecc71; border-color:#2ecc71; color:#08240f; font-weight:600; }
  #play { min-width:42px; font-size:15px; }
  #row2 { display:flex; align-items:center; gap:8px; flex-wrap:wrap;
          font-size:12.5px; color:#888; }
  #read { font-variant-numeric:tabular-nums; }
  .sp { margin-left:auto; }
</style>
<div id="wrap">
  <div id="stage"><img id="img" alt="fill front"></div>
  <div id="bar">
    <button id="play" title="再生 / 一時停止 (Space)">&#9654;</button>
    <input id="seek" type="range" min="0" value="0" step="1">
    <span id="cnt" style="font-size:12.5px;color:#888;font-variant-numeric:tabular-nums;"></span>
  </div>
  <div id="row2">
    <button id="prev" title="1コマ戻る (←)">&#9664; コマ</button>
    <button id="next" title="1コマ進む (→)">コマ &#9654;</button>
    <span id="read"></span>
    <span class="sp"></span>
    <span>速度</span>
    <button class="spd" data-s="0.25">0.25x</button>
    <button class="spd" data-s="0.5">0.5x</button>
    <button class="spd" data-s="1">1x</button>
    <button class="spd" data-s="2">2x</button>
    <button id="loop" class="on" title="末尾で先頭に戻る">ループ</button>
  </div>
</div>
<script>
(function(){
  const D = __PAYLOAD__;
  const img=document.getElementById('img'), seek=document.getElementById('seek');
  const play=document.getElementById('play'), read=document.getElementById('read');
  const cnt=document.getElementById('cnt'), loopBtn=document.getElementById('loop');
  const N=D.frames.length;
  seek.max=N-1;

  // Decode every frame up front so dragging never shows a blank gap.
  const pre=D.frames.map(function(src){ const i=new Image(); i.src=src; return i; });

  let idx=0, timer=null, speed=1, looping=true;

  function show(i){
    idx=Math.max(0,Math.min(N-1,i));
    img.src=D.frames[idx];
    seek.value=idx;
    cnt.textContent=(idx+1)+' / '+N;
    read.textContent=D.labels ? D.labels[idx]
                     : 't = '+D.times[idx].toFixed(3)+' s   充填 '
                       +(D.fills[idx]*100).toFixed(1)+' %';
  }
  function stop(){ if(timer){clearInterval(timer);timer=null;} play.innerHTML='&#9654;'; }
  function start(){
    stop();
    if(idx>=N-1) idx=0;
    timer=setInterval(function(){
      if(idx>=N-1){ if(looping){ show(0); } else { stop(); return; } }
      else { show(idx+1); }
    }, 1000/(D.fps*speed));
    play.innerHTML='&#10073;&#10073;';
  }
  function toggle(){ timer?stop():start(); }

  play.onclick=toggle;
  seek.oninput=function(){ stop(); show(parseInt(seek.value,10)); };
  document.getElementById('prev').onclick=function(){ stop(); show(idx-1); };
  document.getElementById('next').onclick=function(){ stop(); show(idx+1); };
  loopBtn.onclick=function(){ looping=!looping; loopBtn.classList.toggle('on',looping); };
  Array.prototype.forEach.call(document.querySelectorAll('.spd'), function(b){
    b.onclick=function(){
      speed=parseFloat(b.dataset.s);
      Array.prototype.forEach.call(document.querySelectorAll('.spd'),
        function(o){ o.classList.toggle('on', o===b); });
      if(timer) start();
    };
  });
  document.querySelector('.spd[data-s="1"]').classList.add('on');

  window.addEventListener('keydown', function(e){
    if(e.key===' '){ e.preventDefault(); toggle(); }
    else if(e.key==='ArrowLeft'){ e.preventDefault(); stop(); show(idx-1); }
    else if(e.key==='ArrowRight'){ e.preventDefault(); stop(); show(idx+1); }
  });

  show(0);
  if(D.autoplay) start();
})();
</script>
"""


_STANDALONE_TEMPLATE = """<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  body { padding:16px; box-sizing:border-box; }
  #page { max-width:1100px; margin:0 auto; }
  #hd { font:600 14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",
        "Hiragino Sans","Yu Gothic UI",Meiryo,sans-serif;
        margin:0 0 10px; opacity:.85; }
  #hd small { font-weight:400; opacity:.65; margin-left:8px; }
</style>
</head>
<body>
<div id="page">
<p id="hd">__HEADING__</p>
__BODY__
</div>
</body>
</html>
"""


def wrap_standalone_html(body_html: str, *, title: str, note: str | None = None) -> str:
    """Wrap the embed fragment into a complete document for offline viewing.

    ``build_fill_player_html`` returns a fragment (``<style>`` + markup +
    ``<script>``) because ``st.components.v1.html`` supplies the document
    around it. Shipping that fragment as a ``.html`` file needs a real
    document: without ``<meta charset="utf-8">`` a browser opening it over
    ``file://`` falls back to the platform's legacy encoding — CP932 on a
    Japanese Windows box — and the button labels come out as mojibake.

    The wrapper only adds the document shell, a page heading and outer
    padding; it never edits the fragment, so the in-app player and the
    downloadable file stay byte-identical where it matters.
    """
    if not title:
        raise ValueError("title must not be empty")
    heading = html.escape(title)
    if note:
        heading += f"<small>{html.escape(note)}</small>"
    return (
        _STANDALONE_TEMPLATE.replace("__TITLE__", html.escape(title))
        .replace("__HEADING__", heading)
        .replace("__BODY__", body_html)
    )
