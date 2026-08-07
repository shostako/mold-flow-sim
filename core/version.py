"""Single source of truth for the application version.

``__version__`` is the release version (kept in sync with
``pyproject.toml`` and ``CHANGELOG.md``). ``build_label()`` additionally
reports the git commit the running code came from, which is what tells
you whether a deployed instance is actually up to date.
"""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path

__version__ = "0.14.0"

_REPO_ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def git_revision() -> str | None:
    """Short commit SHA of the checkout, or ``None`` if unavailable.

    Streamlit Community Cloud deploys by cloning the repository, so this
    normally resolves there too. Returns ``None`` for installs from a
    source archive (no ``.git``) or when git is missing.
    """
    if not (_REPO_ROOT / ".git").exists():
        return None
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "--short", "HEAD"],
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
def git_commit_date() -> str | None:
    """Commit date (YYYY-MM-DD) of the checkout, or ``None``."""
    if not (_REPO_ROOT / ".git").exists():
        return None
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "log", "-1", "--format=%cd", "--date=short"],
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


def build_label() -> str:
    """Human-readable version string for the UI footer.

    ``v0.14.0 (ad8da46, 2026-08-07)`` when git metadata is available,
    otherwise just ``v0.14.0``.
    """
    rev = git_revision()
    if rev is None:
        return f"v{__version__}"
    date = git_commit_date()
    return f"v{__version__} ({rev}, {date})" if date else f"v{__version__} ({rev})"
