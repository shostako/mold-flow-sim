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
    ``rgb(r, g, b)`` depending on which named scale was asked for, and
    ``matplotlib.colors.to_rgb`` rejects the functional form outright. Both
    spellings occurring is the whole reason this exists; how many scales use
    which is not, and is deliberately left unstated — ``plotly>=5.18`` is
    unpinned, so any count here would be a snapshot masquerading as a
    guarantee. (For scale: on plotly 6.7.0 the functional form was the large
    majority, and Cividis was one of the hex minority — which is why a
    hex-only parser passed at all.)

    A hex-only parser therefore survives only as long as the chosen colormap
    happens to be spelled in hex, and raises ``ValueError`` the moment anyone
    evaluates a different candidate — which is exactly what these tests exist
    to support. A crash there reads as "the test is broken" and invites
    deleting it rather than reconsidering the colormap, so both spellings are
    handled.
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
    (0.186 -> 0.202). The naive sum is wrong by definition, not merely
    imprecise, so do not swap it back on the grounds that no palette currently
    exposes the difference — a measurement of today's built-ins is not a
    licence. (It is true that none of them flip their monotonicity verdict
    between the two formulas as measured; that is a statement about which
    palettes ship, not about which formula is right. These tests exist to
    judge *replacement* palettes, hand-authored warm ramps included, and that
    is exactly where the difference bites.)
    """
    return 0.2126 * _linearize(rgb[0]) + 0.7152 * _linearize(rgb[1]) + 0.0722 * _linearize(rgb[2])


def contrast_ratio(rgb_a: tuple[float, float, float], rgb_b: tuple[float, float, float]) -> float:
    """WCAG contrast ratio between two gamma-encoded sRGB triples.

    ``(L_light + 0.05) / (L_dark + 0.05)``, running 1:1 (identical) to 21:1
    (black on white). Used instead of a raw luminance span because the ratio
    is an absolute, standard-anchored quantity: WCAG 1.4.11 asks 3:1 of
    graphical objects that carry meaning, which is exactly what a thickness
    ramp is.
    """
    lo, hi = sorted((relative_luminance(rgb_a), relative_luminance(rgb_b)))
    return (hi + 0.05) / (lo + 0.05)


def sample_ramp(
    stops: list[tuple[float, str]], per_segment: int = 64
) -> list[tuple[float, float, float]]:
    """Densely sample a Plotly colorscale, matching how it is rendered.

    Plotly interpolates **linearly between stops in gamma-encoded sRGB**.
    Relative luminance applies a nonlinear transfer function to those channels,
    so luminance is *not* linear along a segment and monotonicity at the stops
    does not imply monotonicity between them. (An earlier version of these
    tests claimed it did — true only while the luminance formula was itself a
    plain weighted sum of the encoded channels, and quietly false the moment
    that formula was corrected to linearize first.) Sampling removes the
    argument entirely. The transfer function is convex, so luminance along a
    segment is convex too and may sag below the darker endpoint before
    climbing back — a reversal the endpoints cannot show. That is a property
    of the color space, not of any one palette: ``bluered_r`` merely happens
    to exhibit it today (by 0.0023). If it stops, the reason for sampling is
    unchanged.
    """
    out: list[tuple[float, float, float]] = []
    colors = [css_rgb(color) for _offset, color in stops]
    for start, end in zip(colors, colors[1:]):
        for i in range(per_segment):
            t = i / per_segment
            out.append(tuple(s + t * (e - s) for s, e in zip(start, end)))
    out.append(colors[-1])
    return out
