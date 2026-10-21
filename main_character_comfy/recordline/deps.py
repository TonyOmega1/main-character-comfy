"""Optional dependencies of "record your own line", probed lazily with plain-English fallback messages.

Nothing here is a hard requirement of MC. Each helper says what is missing, what MC does without it and how to add
it. Nothing is ever installed for you.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


class OptionalDependencyMissing(RuntimeError):
    """An optional tool isn't installed. ``str(exc)`` is safe to show in the UI."""


class RecordLineError(ValueError):
    """Bad input (no takes, unreadable file, window outside the clip). ``extras`` maps ValueError to InputError."""


HINTS = {
    "faster_whisper": "Speech recognition (faster-whisper) isn't installed, so takes were picked by length and sound "
                      "quality only, not by the words. Add it with: pip install faster-whisper (CPU is fine).",
    "mediapipe": "Mouth tracking (MediaPipe) isn't installed, so the speaking window came from the sound and a rough "
                 "mouth-motion check only. Add it with: pip install mediapipe, and put face_landmarker.task in "
                 "main_character_comfy/assets/ (or set MC_FACE_LANDMARKER).",
    "demucs": "Voice separation (Demucs) isn't installed, so the original voice may be faintly audible under yours. "
              "Add it with: pip install demucs (or set MC_DEMUCS_CMD).",
    "deepfilternet": "Noise cleanup (DeepFilterNet) isn't installed, so your take was used without it. Add it with: "
                     "pip install deepfilternet (or set MC_DEEPFILTER).",
    "rubberband": "This ffmpeg has no rubberband filter, so the take was fitted with atempo (pitch kept, slightly "
                  "less natural). The imageio-ffmpeg build MC ships includes rubberband.",
}

FACE_LANDMARKER_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/"
                       "face_landmarker.task")


def has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover - broken installs
        return False


def require(name: str) -> None:
    if not has_module(name):
        raise OptionalDependencyMissing(HINTS.get(name, f"{name} isn't installed."))


def face_landmarker_model() -> Path | None:
    """MediaPipe FaceLandmarker model file (not shipped with MC: download it from FACE_LANDMARKER_URL)."""
    env = os.environ.get("MC_FACE_LANDMARKER", "").strip()
    cands = [Path(env)] if env else []
    cands.append(Path(__file__).resolve().parent.parent / "assets" / "face_landmarker.task")
    for c in cands:
        if c.is_file():
            return c
    return None


def mediapipe_ready() -> bool:
    return has_module("mediapipe") and face_landmarker_model() is not None


def status() -> dict:
    """What's available right now (for doctor / the UI)."""
    from .. import redub, stems

    return {"faster_whisper": has_module("faster_whisper"), "mediapipe": mediapipe_ready(),
            "demucs": stems.separator_command() is not None, "deepfilternet": redub.deepfilter_status()["installed"],
            "rubberband": redub.has_rubberband()}
