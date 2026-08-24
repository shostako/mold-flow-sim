"""Single source of truth for the application version.

``__version__`` is the release version (kept in sync with
``pyproject.toml`` and ``CHANGELOG.md``). ``build_label()`` additionally
reports the git commit the running code came from, which is what tells
you whether a deployed instance is actually up to date.

To cut a release: bump ``__version__`` here and in ``pyproject.toml``,
then add a matching ``## [x.y.z]`` section to ``CHANGELOG.md``.
``tests/test_version.py`` enforces that the three stay in sync.
"""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path

__version__ = "0.36.0"

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _git(*args: str) -> str | None:
    """Run a git command in the repo, returning stripped stdout or ``None``."""
    if not (_REPO_ROOT / ".git").exists():
        return None
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - env dependent
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


@lru_cache(maxsize=1)
def git_build_info() -> tuple[str, str, bool] | None:
    """Return ``(short_sha, commit_date, is_dirty)`` for the checkout.

    ``None`` when git metadata is unavailable (install from a source
    archive with no ``.git``, or git not on PATH).

    Cached for the process lifetime: the label is meant to identify the
    deployed build, and both Streamlit Community Cloud and a local
    ``streamlit run`` start a fresh process per deploy/restart. An
    in-place ``git pull`` under a still-running server would keep
    reporting the old SHA until restart.
    """
    head = _git("log", "-1", "--format=%h %cd", "--date=short")
    if head is None:
        return None
    sha, _, date = head.partition(" ")
    if not sha:
        return None
    # Uncommitted edits mean the running code is not the commit named
    # above, so say so rather than advertising a clean build.
    dirty = bool(_git("status", "--porcelain"))
    return sha, date.strip(), dirty


def build_label() -> str:
    """Human-readable version string for the UI.

    ``v0.14.0 (ad8da46, 2026-08-07)`` when git metadata is available,
    with ``+dirty`` appended if the working tree has uncommitted changes;
    otherwise just ``v0.14.0``.
    """
    info = git_build_info()
    if info is None:
        return f"v{__version__}"
    sha, date, dirty = info
    suffix = "+dirty" if dirty else ""
    return f"v{__version__} ({sha}{suffix}, {date})" if date else f"v{__version__} ({sha}{suffix})"
