"""Record of the inputs that produced a run, for the results ZIP.

``metadata.json`` holds what the solver *computed* (volumes, tau, iteration
counts). It never held what the user *entered*, so a downloaded ZIP could not
be traced back to the settings that made it -- the geometry had to be
reverse-engineered from the images and the volume. ``settings.json`` closes
that gap: it is the input side of the same run.

Field names are the dataclass / solver argument names rather than the UI
labels, because those are stable across UI wording changes and map onto the
CLI arguments directly.

**Uploaded spec files are fingerprinted, not embedded.** A gate profile JSON
is usually derived from a real drawing, and the ZIP is meant to be handed to
other people (that is why ``player.html`` is in there). Recording the name and
a SHA-256 lets the owner of the spec confirm which revision was used without
putting the dimensions into a file that travels.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, is_dataclass
from typing import Any

__all__ = ["config_settings", "file_fingerprint", "settings_json"]


def _jsonable(value: Any) -> Any:
    """Convert dataclass field values into JSON-safe types."""
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def config_settings(source: str, cfg: Any, **extra: Any) -> dict[str, Any]:
    """Describe the geometry input: which builder, and with what parameters.

    ``cfg`` is the geometry config dataclass (``FilmGateConfig`` etc.) and is
    required. It used to default to ``None`` for the image importer, which
    described itself with loose keyword arguments; that input was removed in
    v0.23.0 and every remaining caller passes a dataclass. Keeping the default
    would leave a record shape -- one with no ``config`` key -- that nothing
    produces and therefore nothing checks.

    ``extra`` carries what is not a config field: the builder's cell size, a
    spec fingerprint.
    """
    if not source:
        raise ValueError("source must not be empty")
    if not is_dataclass(cfg) or isinstance(cfg, type):
        raise TypeError("cfg must be a dataclass instance")
    record: dict[str, Any] = {
        "input": source,
        "config": {k: _jsonable(v) for k, v in asdict(cfg).items()},
    }
    for key, value in extra.items():
        record[key] = _jsonable(value)
    return record


def file_fingerprint(name: str, data: bytes | str) -> dict[str, Any]:
    """Identify an uploaded file without reproducing its contents.

    Enough to answer "was it this spec?" -- not enough to read the dimensions
    out of a ZIP that gets forwarded.
    """
    payload = data.encode("utf-8") if isinstance(data, str) else data
    return {
        "name": name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def settings_json(record: dict[str, Any]) -> str:
    """Serialize the settings record for the ZIP (UTF-8, readable)."""
    import json

    return json.dumps(record, indent=2, ensure_ascii=False, default=str)
