"""Version metadata must stay consistent across the repo."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from core.version import __version__, build_label, git_build_info

REPO_ROOT = Path(__file__).parent.parent


def test_version_matches_pyproject() -> None:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["version"] == __version__


def test_changelog_documents_current_version() -> None:
    text = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"## [{__version__}]" in text, f"CHANGELOG.md has no entry for {__version__}"


def test_changelog_versions_are_descending() -> None:
    text = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    found = re.findall(r"^## \[(\d+)\.(\d+)\.(\d+)\]", text, flags=re.MULTILINE)
    assert found, "no version headings found in CHANGELOG.md"
    tuples = [tuple(int(p) for p in v) for v in found]
    assert tuples == sorted(tuples, reverse=True), "CHANGELOG entries must be newest-first"
    assert tuples[0] == tuple(int(p) for p in __version__.split("."))


def test_build_label_starts_with_version() -> None:
    label = build_label()
    assert label.startswith(f"v{__version__}")


def test_build_label_reports_git_metadata_when_available() -> None:
    info = git_build_info()
    if info is None:  # no .git / no git binary — bare version is correct
        assert build_label() == f"v{__version__}"
        return
    sha, date, dirty = info
    label = build_label()
    assert sha in label
    assert date in label
    # a modified working tree must be flagged, a clean one must not be
    assert ("+dirty" in label) is dirty
