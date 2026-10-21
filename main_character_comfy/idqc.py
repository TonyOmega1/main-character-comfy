"""Post-render identity QC: does the person in the render still look like the reference?

Face-embedding cosine similarity (OpenCV SFace by default, any embedder can be passed in) of the render's face vs
the reference face, per sampled frame. Below the threshold the shot should be escalated (identity lock / tighter
reference crop) instead of rerolled. SFace's own same-person threshold is 0.363; sunglasses and drawn styles score
lower, so the verdict also has a softer "weak" band.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

SFACE_MODEL = Path(__file__).parent / "assets" / "face_recognition_sface_2021dec.onnx"  # OpenCV Zoo, Apache-2.0
SAME_PERSON = 0.363
WEAK = 0.25


def verdict(scores: Iterable[float], same: float = SAME_PERSON, weak: float = WEAK) -> dict[str, Any]:
    s = np.asarray([v for v in scores if v is not None], np.float64)
    if not s.size:
        return {"frames": 0, "verdict": "no face", "escalate": False}
    med = float(np.median(s))
    v = "pass" if med >= same else ("weak" if med >= weak else "drift")
    return {"frames": int(s.size), "mean": round(float(s.mean()), 3), "median": round(med, 3),
            "p10": round(float(np.percentile(s, 10)), 3), "verdict": v, "escalate": v == "drift"}


def escalation(framing: str, engine: str, face_lock: str | None) -> dict[str, str] | None:
    """Next step for a drifting shot: tighter crop first, then Animate with the facepose identity lock."""
    if framing in ("wide", "medium"):
        return {"reference": "tighter crop (upper body)", "engine": engine, "face": face_lock or ""}
    if engine != "chain" or face_lock != "facepose":
        return {"reference": "head crop", "engine": "chain", "face": "facepose"}
    return None


def sface_embedder() -> Callable[[np.ndarray, Sequence[float]], np.ndarray]:
    """OpenCV SFace embedder (BGR image + YuNet face row with landmarks). Raises if the model isn't installed."""
    import cv2

    if not SFACE_MODEL.exists():
        raise FileNotFoundError(f"{SFACE_MODEL.name} isn't installed (doctor --fix downloads it)")
    rec = cv2.FaceRecognizerSF.create(str(SFACE_MODEL), "")
    return lambda bgr, face: rec.feature(rec.alignCrop(bgr, np.asarray(face, np.float32)))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.ravel(a).astype(np.float64), np.ravel(b).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def identity_scores(ref_emb: np.ndarray, frame_embs: Iterable[np.ndarray | None]) -> list[float | None]:
    return [None if e is None else cosine(ref_emb, e) for e in frame_embs]
