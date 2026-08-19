"""Tests for locating local gate profile specs, and for the UI that uses them.

Two layers, on purpose. :mod:`core.spec_source` is pure and gets ordinary unit
tests. The precedence rule and the notice that reports it are *wiring*, and
wiring is exactly what a pure test passes while the app does something else, so
those are exercised through ``AppTest`` against the real ``app.py``.

Every test that involves a spec directory builds its own under ``tmp_path`` and
patches :func:`core.spec_source.spec_root` to point at it. None of them may
read the developer's real ``local_specs`` link: those files are customer
drawings, and a test whose behaviour depends on whether that link exists passes
on one machine and not on CI.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import core.settings_record as settings_record
import core.spec_source as spec_source
from core.spec_source import (
    SPEC_LINK_NAME,
    SpecMode,
    SpecOrigin,
    choose_spec_origin,
    list_spec_files,
    spec_link_exists,
    spec_root,
)

REPO = Path(__file__).resolve().parent.parent
APP = REPO / "app.py"
DEMO_SPEC = REPO / "data" / "gate_profiles" / "demo_profile_gate.json"

PROFILE_GATE_LABEL = "Profile gate (JSONスペック)"
LOCAL_LABEL = "ローカルから読込"
UNSELECTED = "— 未選択 —"

needs_non_root = pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="permission bits do not restrict root",
)


# --------------------------------------------------------------------------
# the ignore rule *is* the security boundary
# --------------------------------------------------------------------------


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def _ignored_in(repo: Path, path: str) -> bool:
    """Whether ``git`` ignores ``path`` inside ``repo``."""
    proc = _git("check-ignore", "-q", path, cwd=repo)
    if proc.returncode not in (0, 1):
        pytest.skip(f"git check-ignore unusable: {proc.stderr.strip()}")
    return proc.returncode == 0


def test_spec_link_is_ignored_in_this_repo() -> None:
    """The link itself must be ignored, here, in the real checkout.

    This is not housekeeping. Fixing the spec root at one gitignored name is
    what makes the local-load UI impossible to obtain on a deployed instance
    and what keeps ``git add -A`` from committing customer dimensions to a
    public repo. If the ignore rule goes, both properties go with it, and
    nothing else in the codebase would notice.

    Asked of ``git`` rather than by reading ``.gitignore``, because a later
    negation (``!local_specs``) or a rule in ``.git/info/exclude`` changes the
    answer without changing the line this would otherwise grep for.
    """
    assert _ignored_in(REPO, SPEC_LINK_NAME)


def test_ignore_rule_does_not_swallow_the_demo_spec(tmp_path: Path) -> None:
    """Counterweight: the rule must stay narrow.

    An over-broad ``*.json`` would satisfy every test above while quietly
    untracking the repo's own fictional demo spec.

    Asked in a scratch repo about an *untracked* copy of that path, because
    ``git check-ignore`` answers "not ignored" for any tracked file no matter
    what the rules say. Asking the real repo about the real demo spec is
    therefore vacuous -- it passes because the file is committed, and would go
    on passing under ``*.json``. Measured: with ``*.json`` in ``.gitignore``,
    the tracked path returns 1 and an untracked copy of it returns 0.
    """
    if _git("init", "-q", cwd=tmp_path).returncode != 0:
        pytest.skip("git unavailable")
    (tmp_path / ".gitignore").write_text(
        (REPO / ".gitignore").read_text(encoding="utf-8"), encoding="utf-8"
    )
    rel = DEMO_SPEC.relative_to(REPO)
    (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / rel).write_text("{}", encoding="utf-8")

    assert not _ignored_in(tmp_path, str(rel))


def test_demo_spec_is_tracked() -> None:
    """And it must actually be in the repo.

    Separate from the rule check above: a narrow ignore rule is no use if the
    file it was careful to spare was never committed.
    """
    rel = str(DEMO_SPEC.relative_to(REPO))
    assert _git("ls-files", "--error-unmatch", rel, cwd=REPO).returncode == 0


@pytest.mark.parametrize(
    "path", [SPEC_LINK_NAME, f"{SPEC_LINK_NAME}/spec.json", f"{SPEC_LINK_NAME}/sub/deep.json"]
)
def test_ignore_rule_covers_everything_under_the_link(tmp_path: Path, path: str) -> None:
    """Contents of the spec directory must be ignored too, not just the entry.

    Checked in a scratch repo built from this repo's ``.gitignore`` rather than
    against the checkout itself. Asking the real repo about a path *inside* the
    link fails with "beyond a symbolic link" on exactly the machines where the
    feature is set up -- so the check would pass on CI, skip for the person
    using it, and be worth nothing in the only place it matters.
    """
    if _git("init", "-q", cwd=tmp_path).returncode != 0:
        pytest.skip("git unavailable")
    (tmp_path / ".gitignore").write_text(
        (REPO / ".gitignore").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / SPEC_LINK_NAME / "sub").mkdir(parents=True)
    (tmp_path / SPEC_LINK_NAME / "spec.json").write_text("{}", encoding="utf-8")
    (tmp_path / SPEC_LINK_NAME / "sub" / "deep.json").write_text("{}", encoding="utf-8")

    assert _ignored_in(tmp_path, path)


def test_scratch_repo_check_would_notice_a_missing_rule(tmp_path: Path) -> None:
    """The scratch-repo check above has to be able to fail.

    Without this, a typo in the copied ``.gitignore`` path, or a scratch repo
    that never initialised, would leave the contents test passing vacuously.
    """
    if _git("init", "-q", cwd=tmp_path).returncode != 0:
        pytest.skip("git unavailable")
    (tmp_path / ".gitignore").write_text("# nothing ignored here\n", encoding="utf-8")
    (tmp_path / SPEC_LINK_NAME).mkdir()
    (tmp_path / SPEC_LINK_NAME / "spec.json").write_text("{}", encoding="utf-8")

    assert not _ignored_in(tmp_path, f"{SPEC_LINK_NAME}/spec.json")


# --------------------------------------------------------------------------
# spec_root
# --------------------------------------------------------------------------


def test_spec_root_absent_is_off(tmp_path: Path) -> None:
    assert spec_root(tmp_path) is None


def test_spec_root_accepts_a_real_directory(tmp_path: Path) -> None:
    (tmp_path / SPEC_LINK_NAME).mkdir()
    assert spec_root(tmp_path) == (tmp_path / SPEC_LINK_NAME).resolve()


def test_spec_root_follows_a_symlink_out_of_the_tree(tmp_path: Path) -> None:
    """The intended setup: the link points at a directory outside the repo."""
    outside = tmp_path / "elsewhere" / "specs"
    outside.mkdir(parents=True)
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / SPEC_LINK_NAME).symlink_to(outside)
    assert spec_root(app_dir) == outside.resolve()


def test_spec_root_is_fully_resolved(tmp_path: Path) -> None:
    """The returned root must contain no symlink components.

    Callers compare paths found through the root against the root itself. A
    root left unresolved while its contents resolve compares unequal to
    everything under it, so the feature would look broken rather than
    misconfigured.
    """
    real = tmp_path / "real"
    real.mkdir()
    hop = tmp_path / "hop"
    hop.symlink_to(real)
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / SPEC_LINK_NAME).symlink_to(hop)

    root = spec_root(app_dir)
    assert root == real.resolve()
    assert root == Path(os.path.realpath(root))


def test_spec_root_rejects_a_dangling_symlink(tmp_path: Path) -> None:
    (tmp_path / SPEC_LINK_NAME).symlink_to(tmp_path / "gone")
    assert spec_root(tmp_path) is None


def test_spec_root_rejects_a_plain_file(tmp_path: Path) -> None:
    (tmp_path / SPEC_LINK_NAME).write_text("{}", encoding="utf-8")
    assert spec_root(tmp_path) is None


def test_spec_root_rejects_a_symlink_loop(tmp_path: Path) -> None:
    """``Path.resolve`` raises ``OSError`` here; the feature must switch off
    rather than propagate it into the page."""
    link = tmp_path / SPEC_LINK_NAME
    link.symlink_to(link)
    assert spec_root(tmp_path) is None


# --------------------------------------------------------------------------
# spec_link_exists
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("make", "expected"),
    [
        (None, False),
        ("dir", True),
        ("file", True),
        ("dangling", True),
    ],
)
def test_spec_link_exists_distinguishes_absent_from_broken(
    tmp_path: Path, make: str | None, expected: bool
) -> None:
    """A dangling symlink is the case this exists for.

    ``Path.exists`` follows the link and answers ``False`` for a dangling one,
    which would make "set up but broken" indistinguishable from "never set up"
    -- and the person who set it up would get silence.
    """
    link = tmp_path / SPEC_LINK_NAME
    if make == "dir":
        link.mkdir()
    elif make == "file":
        link.write_text("{}", encoding="utf-8")
    elif make == "dangling":
        link.symlink_to(tmp_path / "gone")
    assert spec_link_exists(tmp_path) is expected


def test_dangling_link_is_visible_but_not_a_root(tmp_path: Path) -> None:
    """The pair that drives the "broken link" notice."""
    (tmp_path / SPEC_LINK_NAME).symlink_to(tmp_path / "gone")
    assert spec_link_exists(tmp_path) and spec_root(tmp_path) is None


# --------------------------------------------------------------------------
# list_spec_files
# --------------------------------------------------------------------------


def test_list_spec_files_empty(tmp_path: Path) -> None:
    assert list_spec_files(tmp_path) == []


def test_list_spec_files_selects_only_json_files(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text("{}", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    (tmp_path / "b.JSON").write_text("{}", encoding="utf-8")
    (tmp_path / "adir.json").mkdir()
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "deep.json").write_text("{}", encoding="utf-8")

    assert [p.name for p in list_spec_files(tmp_path)] == ["a.json"]


def test_list_spec_files_is_sorted_by_name_not_by_mtime(tmp_path: Path) -> None:
    """Order must not depend on the filesystem's idea of recency.

    mtime survives neither ``git checkout`` nor a copy between machines, so an
    mtime-ordered list silently reorders itself and the entry a user reaches
    for by position becomes a different spec. Written newest-first on disk so
    that an mtime sort would produce the reverse of the assertion.
    """
    names = ["c_20260818.json", "a_20260703.json", "b_20260807.json"]
    for i, name in enumerate(names):
        f = tmp_path / name
        f.write_text("{}", encoding="utf-8")
        os.utime(f, (1_700_000_000 - i, 1_700_000_000 - i))

    assert [p.name for p in list_spec_files(tmp_path)] == sorted(names)


@needs_non_root
def test_list_spec_files_raises_rather_than_reporting_empty(tmp_path: Path) -> None:
    """An unreadable directory must not render as "no specs here".

    Swallowing the error would make a broken mount look exactly like an empty
    folder, and the user would go looking for the missing files instead of the
    missing permission.
    """
    (tmp_path / "a.json").write_text("{}", encoding="utf-8")
    tmp_path.chmod(0o000)
    try:
        with pytest.raises(OSError):
            list_spec_files(tmp_path)
    finally:
        tmp_path.chmod(0o755)


# --------------------------------------------------------------------------
# choose_spec_origin
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "flags", "expected"),
    [
        (SpecMode.DEMO, {}, SpecOrigin.DEMO),
        # The demo is unconditional: no other source can displace it.
        (SpecMode.DEMO, {"has_upload": True, "has_local": True}, SpecOrigin.DEMO),
        (SpecMode.PASTE, {"has_paste": True}, SpecOrigin.PASTE),
        (SpecMode.PASTE, {}, SpecOrigin.NONE),
        (SpecMode.PASTE, {"has_upload": True}, SpecOrigin.NONE),
        (SpecMode.LOCAL, {}, SpecOrigin.NONE),
        (SpecMode.LOCAL, {"has_local": True}, SpecOrigin.LOCAL),
        (SpecMode.LOCAL, {"has_upload": True}, SpecOrigin.UPLOAD),
        # The conflict the notice exists to report.
        (SpecMode.LOCAL, {"has_upload": True, "has_local": True}, SpecOrigin.UPLOAD),
        (SpecMode.LOCAL, {"has_paste": True}, SpecOrigin.NONE),
    ],
)
def test_choose_spec_origin(mode: SpecMode, flags: dict, expected: SpecOrigin) -> None:
    assert choose_spec_origin(mode, **flags) is expected


def test_choose_spec_origin_rejects_an_unknown_mode() -> None:
    with pytest.raises(ValueError):
        choose_spec_origin("local")  # type: ignore[arg-type]


def test_choose_spec_origin_does_no_io(tmp_path: Path, monkeypatch) -> None:
    """It decides; it does not touch the disk.

    Keeping the decision free of IO is what lets the UI resolve the source in
    the sidebar -- before the geometry is built -- so the notice appears beside
    the controls that disagree instead of over in the results column.
    """

    def explode(*_a, **_k):
        raise AssertionError("choose_spec_origin touched the filesystem")

    monkeypatch.setattr(Path, "exists", explode)
    monkeypatch.setattr(Path, "is_dir", explode)
    monkeypatch.setattr(Path, "glob", explode)
    assert choose_spec_origin(SpecMode.LOCAL, has_upload=True) is SpecOrigin.UPLOAD


# --------------------------------------------------------------------------
# the wiring, against the real app
# --------------------------------------------------------------------------


@pytest.fixture
def spec_dir(tmp_path: Path, monkeypatch) -> Path:
    """A spec directory of our own, standing in for the local link.

    Patching :func:`core.spec_source.spec_root` rather than creating a link
    next to ``app.py`` keeps the tests off the developer's real specs and makes
    them behave the same on a machine that has no link at all.
    """
    root = tmp_path / "specs"
    root.mkdir()
    demo = DEMO_SPEC.read_text(encoding="utf-8")
    (root / "alpha.json").write_text(demo, encoding="utf-8")
    (root / "beta.json").write_text(demo, encoding="utf-8")
    monkeypatch.setattr(spec_source, "spec_root", lambda _app_dir: root)
    return root


def _profile_gate_app(timeout: float = 60.0) -> AppTest:
    """Run ``app.py`` with the Profile gate input and the local-load mode.

    Widgets are addressed by key rather than by position: the sidebar holds
    several radios and selectboxes, so an index would silently start pointing
    at the material picker the next time one is added above.
    """
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    at.run()
    at.radio("geom_source").set_value(PROFILE_GATE_LABEL).run()
    at.radio("spec_mode_pg").set_value(LOCAL_LABEL).run()
    return at


def _spec_pick(at: AppTest):
    return at.selectbox("spec_pick_pg")


def _has_spec_pick(at: AppTest) -> bool:
    try:
        at.selectbox("spec_pick_pg")
    except KeyError:
        return False
    return True


def _texts(at: AppTest) -> str:
    """Everything the page rendered as text, for substring checks."""
    parts = []
    for group in (at.caption, at.info, at.warning, at.error, at.markdown, at.exception):
        parts.extend(str(getattr(el, "value", "")) for el in group)
    return "\n".join(parts)


def test_no_local_link_means_no_dropdown(monkeypatch) -> None:
    """With the feature off, the local mode offers only the uploader.

    This is what a deployed instance and a fresh checkout see.
    """
    monkeypatch.setattr(spec_source, "spec_root", lambda _app_dir: None)
    monkeypatch.setattr(spec_source, "spec_link_exists", lambda _app_dir: False)
    at = _profile_gate_app()
    assert not _has_spec_pick(at)
    # The uploader stays: it is the path that works without any local setup.
    assert at.file_uploader("spec_upload_pg") is not None


def test_dropdown_starts_unselected_and_reads_nothing(spec_dir: Path) -> None:
    """Opening the page must not load a spec.

    The geometry is rebuilt on every rerun, not behind the run button, so a
    dropdown defaulting to the first entry would draw a customer part on page
    load. The sentinel is what keeps that from happening.
    """
    at = _profile_gate_app()
    assert _spec_pick(at).options == [UNSELECTED, "alpha.json", "beta.json"]
    assert _spec_pick(at).value == UNSELECTED
    assert any("スペック JSON" in w.value for w in at.warning)


def test_selecting_a_spec_loads_it(spec_dir: Path) -> None:
    at = _profile_gate_app()
    _spec_pick(at).set_value("beta.json").run()
    assert any("読込元: beta.json" in c.value for c in at.caption)
    assert not at.warning


def test_upload_wins_over_the_selection_and_says_so(spec_dir: Path) -> None:
    """The precedence rule, exercised end to end.

    A pure test of ``choose_spec_origin`` passes whether or not the app calls
    it, and whether or not the result reaches the reader. The point of this one
    is that the app resolves the conflict *and* reports it.
    """
    at = _profile_gate_app()
    _spec_pick(at).set_value("beta.json").run()
    at.file_uploader("spec_upload_pg").set_value(
        ("dropped.json", DEMO_SPEC.read_bytes(), "application/json")
    ).run()

    notice = "\n".join(i.value for i in at.info)
    assert "dropped.json" in notice
    assert "beta.json" not in notice
    # The losing control is disabled where it sits, not merely contradicted by
    # text somewhere else on the page.
    assert _spec_pick(at).disabled


def test_clearing_the_upload_falls_back_to_the_selection(spec_dir: Path) -> None:
    """Precedence must not be a trapdoor.

    Removing the dropped file has to hand control back to the dropdown, with
    the selection intact -- otherwise the only way out is a page reload.
    """
    at = _profile_gate_app()
    _spec_pick(at).set_value("beta.json").run()
    at.file_uploader("spec_upload_pg").set_value(
        ("dropped.json", DEMO_SPEC.read_bytes(), "application/json")
    ).run()
    at.file_uploader("spec_upload_pg").clear().run()

    assert not _spec_pick(at).disabled
    assert any("読込元: beta.json" in c.value for c in at.caption)


def test_broken_link_is_reported(monkeypatch) -> None:
    """Set up but not working must not look the same as never set up."""
    monkeypatch.setattr(spec_source, "spec_root", lambda _app_dir: None)
    monkeypatch.setattr(spec_source, "spec_link_exists", lambda _app_dir: True)
    at = _profile_gate_app()
    assert any(SPEC_LINK_NAME in c.value for c in at.caption)


@needs_non_root
def test_unreadable_spec_folder_reports_without_the_path(tmp_path: Path, monkeypatch) -> None:
    """An IO failure must not put the spec directory on screen.

    ``client.showErrorDetails`` defaults to "full", so an uncaught ``OSError``
    renders its message -- which is an absolute path naming the customer and
    the job -- plus a traceback. That is the same leak the fingerprint goes out
    of its way to avoid, arriving by a different door.
    """
    root = tmp_path / "acme_widget_specs"
    root.mkdir()
    (root / "part.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(spec_source, "spec_root", lambda _app_dir: root)
    root.chmod(0o000)
    try:
        at = _profile_gate_app()
        rendered = _texts(at)
        assert at.error
        assert str(root) not in rendered
        assert root.name not in rendered
    finally:
        root.chmod(0o755)


@needs_non_root
def test_unreadable_spec_file_reports_without_the_path(spec_dir: Path) -> None:
    """Same guarantee on the read, which happens after the listing succeeded."""
    at = _profile_gate_app()
    (spec_dir / "beta.json").chmod(0o000)
    try:
        _spec_pick(at).set_value("beta.json").run()
        rendered = _texts(at)
        assert at.error
        assert str(spec_dir) not in rendered
    finally:
        (spec_dir / "beta.json").chmod(0o644)


def test_selected_spec_is_fingerprinted_by_name_only(spec_dir: Path, monkeypatch) -> None:
    """The results ZIP records which file, never where it lived.

    The directory above a spec names the customer and the job, so the recorded
    name must be the bare file name even though the app read it by full path.
    Checked by intercepting the recording call rather than by unzipping a run:
    the ZIP is only produced behind the run button, and putting a full solve in
    the way of this assertion would make it slow enough to eventually be
    deleted.
    """
    seen: list[str] = []
    real = settings_record.file_fingerprint

    def spy(name, data):
        seen.append(name)
        return real(name, data)

    monkeypatch.setattr(settings_record, "file_fingerprint", spy)

    at = _profile_gate_app()
    _spec_pick(at).set_value("beta.json").run()

    assert seen, "the spec was never fingerprinted"
    assert seen[-1] == "beta.json"
    assert not any(str(spec_dir) in name for name in seen)
