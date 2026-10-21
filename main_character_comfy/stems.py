"""Vocals-only stem for lip sync (borrowed idea: SwapJutsu feeds the lip-sync model an isolated vocal, not the mix).

Lip sync on a song follows the loudest sound, so drums and synths make the mouth chatter. With a vocals-only stem the
mouth follows the singer. MC asks for an uploaded stem first (Sing mode). When none is given and **Demucs** (MIT,
https://github.com/facebookresearch/demucs, local, free) is installed, MC separates the song section itself.
Nothing is installed for you: ``pip install demucs`` in any Python (or point ``MC_DEMUCS_CMD`` at it), or set
``MC_AUTO_STEMS=0`` to turn it off. No separator found = the full mix, exactly as before.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

DEMUCS_MODEL = "htdemucs"  # the default 4-stem hybrid transformer model; --two-stems=vocals keeps vocals + the rest


def separator_command() -> list[str] | None:
    """The Demucs command line to run, or None when it isn't available (or MC_AUTO_STEMS=0)."""
    if os.environ.get("MC_AUTO_STEMS", "1").strip().lower() in ("0", "off", "false", "no"):
        return None
    custom = os.environ.get("MC_DEMUCS_CMD", "").strip()
    if custom:
        return shlex.split(custom, posix=os.name != "nt")
    exe = shutil.which("demucs")
    if exe:
        return [exe]
    try:  # installed into MC's own Python
        import importlib.util

        if importlib.util.find_spec("demucs") is not None:
            return [sys.executable, "-m", "demucs"]
    except Exception:  # noqa: BLE001
        pass
    return None


def separate_vocals(src: str | Path, out_dir: str | Path, *, timeout: float = 900.0) -> Path | None:
    """Vocals stem of ``src`` (a short song section) via Demucs, or None when no separator is available or it
    failed. Runs on whatever device Demucs picks (CPU is fine for a 20 s section, ~1 min)."""
    cmd = separator_command()
    if not cmd:
        return None
    src, out_dir = Path(src), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run([*cmd, "--two-stems=vocals", "-n", DEMUCS_MODEL, "-o", str(out_dir), str(src)],
                       check=True, capture_output=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    found = out_dir / DEMUCS_MODEL / src.stem / "vocals.wav"
    if not found.is_file():
        hits = sorted(out_dir.rglob("vocals.wav"))
        found = hits[0] if hits else found
    return found if found.is_file() and found.stat().st_size > 1000 else None
