"""Color helpers shared by the thickness-colormap tests.

These live in one place on purpose. The luminance formula was originally
written out by hand in two test files, and *both copies were wrong the same
way* — reason enough to stop having two copies. See :func:`relative_luminance`.
"""

from __future__ import annotations

import matplotlib.colors as mcolors


def css_rgb(value: str) -> tuple[float, float, float]:
    """Parse one Plotly colorscale stop into 0..1 (gamma-encoded) sRGB.

    Plotly hands back either ``#rrggbb`` or the CSS functional form
    ``rgb(r, g, b)`` depending on which named scale was asked for. Across its
    94 built-ins in both directions — 188 ramps, 1774 stops — the split is
    1442 functional to 332 hex, and ``matplotlib.colors.to_rgb`` rejects the
    functional form outright. Cividis happens to be one of the hex ones, so a
    hex-only parser passes today purely by luck of the colormap in force and
    would raise ``ValueError`` the moment anyone evaluated a different
    candidate — which is exactly what these tests exist to support. A crash
    there reads as "the test is broken" and invites deleting it rather than
    reconsidering the colormap, so both spellings are handled.
    """
    text = str(value).strip()
    if text.startswith("rgb"):  # covers rgb(...) and rgba(...)
        body = text[text.index("(") + 1 : text.rindex(")")]
        return tuple(float(n) / 255.0 for n in body.split(",")[:3])
    return mcolors.to_rgb(text)


def _linearize(channel: float) -> float:
    """Undo the sRGB transfer function on one 0..1 channel."""
    return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4


def relative_luminance(rgb: tuple[float, float, float]) -> float:
    """WCAG relative luminance of a *gamma-encoded* sRGB triple.

    The channels must be linearized before the weighted sum. Applying
    ``0.2126R + 0.7152G + 0.0722B`` to gamma-encoded values computes luma, not
    luminance, and the two disagree in sign when a palette trades intensity
    between channels: ``rgb(205,78,46) -> rgb(242,34,36)`` reads as *darker*
    under the naive sum (0.403 -> 0.307) and as *lighter* once linearized
    (0.186 -> 0.202). No Plotly built-in currently flips its monotonicity
    verdict between the two formulas, so this is not a live bug — but these
    tests exist to judge *replacement* palettes, hand-authored warm ramps
    included, and that is precisely where the wrong formula would wave through
    a ramp that actually gets lighter.
    """
    return 0.2126 * _linearize(rgb[0]) + 0.7152 * _linearize(rgb[1]) + 0.0722 * _linearize(rgb[2])
