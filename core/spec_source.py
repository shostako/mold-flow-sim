"""Finding gate profile specs that deliberately live outside this repository.

A real gate profile spec is traced off a customer drawing, so it may not enter
this repository (it is public) and may not enter a downloaded results ZIP.
Uploading one through the browser every time is the friction this module
removes -- and the only thing it is allowed to remove. Nothing here copies,
caches, or writes a spec anywhere: it locates files and reports which source a
run should read from. The reading itself stays in the caller.

**Why a gitignored symlink and not an environment variable.** The obvious
design is ``MFS_SPEC_DIR``, shown only when set. It fails open: a misconfigured
deployment *enables* the feature. That is not hypothetical for this app.
Streamlit Community Cloud has no environment-variable UI at all -- secrets are
the only channel -- and ``streamlit.runtime.secrets`` promotes every top-level
``str``/``int``/``float`` secret into ``os.environ``. So pasting a local
``secrets.toml`` into the Cloud settings box, which is exactly how secrets are
normally moved, would light up a filesystem reader on the public instance. An
exported shell variable has a second problem: it is set per machine, so it
would also apply to an unrelated checkout of this repo on the same machine.

A gitignored path fails closed instead. A Cloud deployment is a git checkout
plus secrets, and there is no way to make a gitignored path exist in one.
Turning the feature on there would require editing ``.gitignore`` and
committing the specs -- a visible diff, not a silent misconfiguration.

Fixing the root at one gitignored name also closes the worst leak by
construction rather than by check. With a configurable root, pointing it at a
directory inside the repo and running ``git add -A`` is the shortest path to
committing customer dimensions to a public repo, and neither an environment
gate nor a containment check does anything about it. Here the root is always
:data:`SPEC_LINK_NAME`, and that name is in ``.gitignore``.

Path containment has no equivalent here on purpose: there is no user-typed
path to contain. The dropdown offers what the link resolves to and nothing
else, which is why ``..`` handling, ``~`` expansion, prefix matching, and
extension filtering are all absent rather than merely unwritten. TOCTOU
between resolving and reading is likewise out of scope: the threat model is a
single operator on their own machine, plus a deployment where the feature does
not exist.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

__all__ = [
    "SPEC_LINK_NAME",
    "SpecOrigin",
    "choose_spec_origin",
    "list_spec_files",
    "spec_link_exists",
    "spec_root",
]

#: Name of the gitignored symlink (or directory) holding the local specs.
#: Kept in sync with the entry in ``.gitignore``; ``tests/test_spec_source.py``
#: checks that, because the ignore rule is the entire security boundary.
SPEC_LINK_NAME = "local_specs"


class SpecOrigin(Enum):
    """Where a run's spec text should be read from."""

    UPLOAD = "upload"
    LOCAL = "local"
    NONE = "none"


def spec_root(app_dir: Path | str) -> Path | None:
    """Resolve the local spec directory, or ``None`` if the feature is off.

    ``None`` is the normal state, not an error: it is what every deployment
    and every fresh checkout sees.

    The returned path is fully resolved. The link almost always *is* a symlink
    pointing outside the repo, so an unresolved root would compare unequal to
    anything found through it.
    """
    link = Path(app_dir) / SPEC_LINK_NAME
    try:
        resolved = link.resolve()
        return resolved if resolved.is_dir() else None
    except (OSError, RuntimeError):
        # A dead mount, or a symlink loop. ``RuntimeError`` is not redundant:
        # ``Path.resolve`` converts ``ELOOP`` into one, and its message quotes
        # the offending absolute path -- which, with Streamlit rendering error
        # details in full, would put the spec directory on screen.
        return None


def spec_link_exists(app_dir: Path | str) -> bool:
    """Whether something is at the link path, even if it does not resolve.

    Lets the caller tell "feature off" (nothing there -- stay silent) apart
    from "set up but broken" (a dangling symlink -- say so). Without this the
    second case is indistinguishable from the first, and a broken link would
    look like the feature simply not existing.

    ``is_symlink`` is checked first because ``exists`` follows the link and is
    ``False`` for a dangling one -- which is precisely the case this exists to
    catch.
    """
    link = Path(app_dir) / SPEC_LINK_NAME
    return link.is_symlink() or link.exists()


def list_spec_files(root: Path) -> list[Path]:
    """Spec files directly under ``root``, ordered by name.

    Flat rather than recursive: the dropdown shows bare filenames, and a
    recursive walk can surface two files with the same name and no way to tell
    them apart.

    Sorted by name rather than by modification time. Specs are revisions of one
    part and mtime looks like the useful order, but it does not survive a
    ``git checkout``, an ``rsync``, or a copy between machines, so the list
    would silently reorder itself. Name order is stable everywhere; with the
    date-suffixed names in use it also happens to read chronologically, but
    nothing here depends on that convention.

    Raises ``OSError`` if the directory cannot be read. Returning an empty list
    would render as "no specs here", which is the same thing an unreadable
    mount should not be allowed to look like.

    ``os.scandir`` rather than ``Path.glob`` for exactly that reason: glob
    swallows the ``PermissionError`` and yields nothing, so an unreadable
    directory and an empty one are indistinguishable to the caller. Hidden
    files stay excluded, matching what glob did.
    """
    with os.scandir(root) as entries:
        found = [
            Path(e.path)
            for e in entries
            if e.name.endswith(".json") and not e.name.startswith(".") and e.is_file()
        ]
    return sorted(found, key=lambda p: p.name)


def choose_spec_origin(
    *,
    has_upload: bool = False,
    has_local: bool = False,
) -> SpecOrigin:
    """Decide which source a run reads from. Performs no IO.

    Both a dropped file and a dropdown selection
    can be live at once, and an upload wins: it is the more recent explicit
    act. The caller must show which one was used -- a precedence rule that
    silently discards the other input is the failure this ordering invites.

    The dropdown carries an unselected sentinel so that ``has_local`` is
    ``False`` until the user picks something. Defaulting it to the first file
    would make the conflict permanent instead of occasional (every drop would
    hit this rule), and would read a customer spec on page load, since the
    geometry is built on every rerun rather than behind the run button.
    """
    if has_upload:
        return SpecOrigin.UPLOAD
    if has_local:
        return SpecOrigin.LOCAL
    return SpecOrigin.NONE
