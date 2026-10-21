"""Wiring stub for "record your own line": hands the line to DubMaster's recordline backend once it lands.

The backend lives on its own branch (``feat/record-line``, package ``recordline``: ``main_character_comfy.recordline``
or a top-level ``mc.recordline``). This module does not implement it. Until it is merged, :func:`backend` returns None
and :func:`extras.run_redub` keeps its built-in chain (redub.py + optional lipdub.py).

Expected entry point (proposed contract, adjust here when the backend lands)::

    recordline.process_line(video: Path, line_wav: Path, *, start: float, work: Path, settings: dict) -> dict
        # returns {"output": <mp4 path>, "steps": [str, ...], **anything to record}

``MC_RECORDLINE=off`` forces the built-in chain; ``MC_RECORDLINE=<dotted.module>`` points at another module.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path
from types import ModuleType

CANDIDATES = ("main_character_comfy.recordline", "mc.recordline")
ENTRY = "process_line"


def backend() -> ModuleType | None:
    """The recordline module if one is installed and exposes :data:`ENTRY`, else None."""
    pick = os.environ.get("MC_RECORDLINE", "").strip()
    if pick.lower() in ("off", "0", "builtin"):
        return None
    for name in ((pick,) if pick else CANDIDATES):
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        if callable(getattr(mod, ENTRY, None)):
            return mod
    return None


def describe() -> str:
    mod = backend()
    return f"recordline backend ({mod.__name__})" if mod else "built-in chain (redub + lipdub)"


def process(mod: ModuleType, video: Path, line: Path, *, start: float, work: Path, settings: dict) -> dict:
    out = getattr(mod, ENTRY)(Path(video), Path(line), start=float(start), work=Path(work), settings=dict(settings))
    if not isinstance(out, dict) or not out.get("output"):
        raise RuntimeError(f"{mod.__name__}.{ENTRY} must return a dict with 'output'")
    out.setdefault("steps", [])
    return out
