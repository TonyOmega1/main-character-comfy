"""Reference photo quality check: runs on every selfie / avatar MC is handed (window and agent connector).

Checks resolution, sharpness (variance of the Laplacian, measured on the face when there is one), a face (YuNet,
when OpenCV is installed) and how big it is, and light. Never blocks a render: it returns friendly, lowercase
warnings the window shows under the photo and the agent connector returns in a ``reference_quality`` field so
agents can pass them on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .media import YUNET_MODEL, load_rgb

TIP = ("use the highest-quality reference photos you can: sharp, well lit, face and outfit clearly visible, ideally "
       "full body, no heavy filters, at least about 1024 px.")

MIN_SIDE_WEAK = 512      # short side under this: weak
GOOD_LONG_SIDE = 1024    # long side under this: worth a nudge
BLUR_WEAK = 40.0         # Laplacian variance (at a fixed working size) under this: blurry
BLUR_SOFT = 90.0         # ...under this: a bit soft
FACE_TINY = 0.07         # face height / image height under this: face too small to copy a likeness from
FACE_TIGHT = 0.55        # over this: a tight face crop, the outfit isn't in it
DARK_LUMA = 55           # mean luma under this: too dark
BRIGHT_LUMA = 225        # over this: blown out
GRAY_SAT = 0.08          # mean HSV saturation under this (or R~G~B): black and white
FACE_CLOSEUP = 0.15      # face height / image height under this (or under FACE_CLOSEUP_PX): too small for close-ups
FACE_CLOSEUP_PX = 256
EYES_DARK = 40           # eye band mean luma under this with low spread: covered (sunglasses)
POSTER_COLORS = 48       # fewer distinct 5-bit colours than this in a photo: heavy filter / posterized
CLIP_MIN_SIDE = 720      # clip short side under this: low-res source
CLIP_MIN_MBPS_720 = 2.5  # clip video bitrate at 720p under this (scaled by pixel count): over-compressed
CLIP_MIN_FPS = 24


def laplacian_var(gray: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian (no OpenCV needed). Higher = sharper."""
    g = gray.astype(np.float32)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var()) if lap.size else 0.0


def _gray(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def _resize(rgb: np.ndarray, long_side: int) -> np.ndarray:
    from PIL import Image

    h, w = rgb.shape[:2]
    k = long_side / max(h, w)
    if abs(k - 1) < 1e-3:
        return rgb
    return np.asarray(Image.fromarray(rgb).resize((max(1, round(w * k)), max(1, round(h * k))), Image.LANCZOS))


def saturation(rgb: np.ndarray) -> float:
    """Mean HSV saturation (0..1)."""
    x = rgb.astype(np.float32) / 255.0
    mx, mn = x.max(axis=2), x.min(axis=2)
    return float(np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0).mean())


def is_grayscale(rgb: np.ndarray) -> bool:
    """Black and white (or near enough that skin tone and eye/hair colour have to be guessed)."""
    x = rgb.astype(np.int16)
    chan = float(np.abs(x[..., 0] - x[..., 1]).mean() + np.abs(x[..., 1] - x[..., 2]).mean())
    return chan < 3.0 or saturation(rgb) < GRAY_SAT


def eyes_covered(rgb: np.ndarray, face: dict[str, Any]) -> bool:
    """Eye band of the face box is dark and flat: sunglasses (or a hand) over the eyes."""
    x, y, w, h = (int(face[k]) for k in ("x", "y", "w", "h"))
    band = rgb[max(0, y + int(0.25 * h)): max(0, y + int(0.5 * h)), max(0, x + int(0.12 * w)): max(0, x + int(0.88 * w))]
    if band.size == 0:
        return False
    g = _gray(band)
    return float(g.mean()) < EYES_DARK and float(g.std()) < 30


def posterized(rgb: np.ndarray) -> bool:
    """Few distinct colours (5 bits per channel) for its size: a heavy filter or a flat stylization."""
    small = _resize(rgb, 256) >> 3
    codes = (small[..., 0].astype(np.int32) << 10) | (small[..., 1].astype(np.int32) << 5) | small[..., 2]
    return len(np.unique(codes)) < POSTER_COLORS


def find_face(rgb: np.ndarray) -> dict[str, Any] | None:
    """Biggest face as {x, y, w, h, score} in image pixels, ``None`` if there's none, raises ImportError w/o cv2."""
    import cv2

    try:  # OpenCV 5's DNN prints a harmless "Targets are not supported" line per detector
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:  # noqa: BLE001
        try:
            cv2.setLogLevel(2)
        except Exception:  # noqa: BLE001
            pass
    bgr = rgb[:, :, ::-1].copy()
    h, w = bgr.shape[:2]
    det = cv2.FaceDetectorYN.create(str(YUNET_MODEL), "", (320, 320), 0.6, 0.3, 50)
    best = None
    for side in (640, 1024):
        k = side / max(h, w)
        small = cv2.resize(bgr, (max(1, int(w * k)), max(1, int(h * k))), interpolation=cv2.INTER_AREA)
        det.setInputSize((small.shape[1], small.shape[0]))
        _, faces = det.detect(small)
        for f in faces if faces is not None else []:
            box = {"x": float(f[0] / k), "y": float(f[1] / k), "w": float(f[2] / k), "h": float(f[3] / k),
                   "score": float(f[-1])}
            if best is None or box["w"] * box["h"] > best["w"] * best["h"]:
                best = box
    return best


def check_reference(src: str | Path, subject: str = "auto") -> dict[str, Any]:
    """Grade a reference photo: ``good`` / ``ok`` / ``weak``, with warnings and the numbers behind them.

    ``subject`` is photo / avatar / auto: a missing human face only counts against real photos (avatars,
    mascots and robots often have none, and that's fine).
    """
    rgb = np.asarray(load_rgb(src).convert("RGB"))
    h, w = rgb.shape[:2]
    warnings: list[str] = []
    weak = False
    out: dict[str, Any] = {"width": w, "height": h}

    if min(w, h) < MIN_SIDE_WEAK:
        weak = True
        warnings.append(f"it's only {w}x{h}. a photo of at least about 1024 px copies a likeness much better.")
    elif max(w, h) < GOOD_LONG_SIDE:
        warnings.append(f"it's {w}x{h}. at least about 1024 px gives a sharper likeness.")

    luma = float(_gray(rgb).mean())
    out["brightness"] = round(luma, 1)
    if luma < DARK_LUMA:
        warnings.append("it's dark. a well lit photo keeps the face and outfit details.")
    elif luma > BRIGHT_LUMA:
        warnings.append("it's blown out. softer, even light keeps the details.")

    face = None
    try:
        face = find_face(rgb)
        out["face"] = ({"found": True, "score": round(face["score"], 2),
                        "height_frac": round(float(face["h"]) / h, 3)} if face else {"found": False})
    except ImportError:
        out["face"] = {"found": None, "note": "opencv not installed, face not checked"}

    # sharpness: on the face when there is one (that's what has to match), else the whole frame
    if face:
        pad = 0.25
        x0, y0 = int(max(0, face["x"] - pad * face["w"])), int(max(0, face["y"] - pad * face["h"]))
        x1, y1 = int(min(w, face["x"] + (1 + pad) * face["w"])), int(min(h, face["y"] + (1 + pad) * face["h"]))
        crop = rgb[y0:y1, x0:x1]
        sharp = laplacian_var(_gray(_resize(crop, 256))) if crop.size else 0.0
        where = "the face"
    else:
        sharp = laplacian_var(_gray(_resize(rgb, 1024)))
        where = "the photo"
    out["sharpness"] = round(sharp, 1)
    if sharp < BLUR_WEAK:
        weak = True
        warnings.append(f"{where} looks blurry. use a sharp, in-focus photo with no heavy filters.")
    elif sharp < BLUR_SOFT:
        warnings.append(f"{where} is a bit soft. a sharper photo gives a closer likeness.")

    name = Path(str(src)).name if isinstance(src, (str, Path)) else "this photo"
    if is_grayscale(rgb):
        out["grayscale"] = True
        warnings.append(f"{name} is black and white, so MC has to guess skin tone and eye/hair colour. "
                        f"use a colour photo.")
    elif subject == "photo" and posterized(rgb):
        out["filtered"] = True
        warnings.append(f"{name} looks heavily filtered or stylized. use an unfiltered photo.")

    if face:
        frac = face["h"] / h
        if frac < FACE_TINY:
            weak = weak or frac < FACE_TINY / 2
            warnings.append("the face is tiny in this photo. crop closer, or use a sharper, bigger photo.")
        elif frac > FACE_TIGHT:
            warnings.append("it's a tight face crop. show the outfit too: upper body, ideally full body.")
        if frac < FACE_CLOSEUP or face["h"] < FACE_CLOSEUP_PX:
            out["closeup_ready"] = False
            warnings.append(f"{name}: the face is only {int(face['h'])} px tall. close-up shots need a front-facing "
                            f"face photo at least ~{FACE_CLOSEUP_PX} px (add one as a second reference).")
        else:
            out["closeup_ready"] = True
        if eyes_covered(rgb, face):
            out["eyes_covered"] = True
            warnings.append(f"{name}: the eyes are covered (sunglasses or a hand). that's fine for the look, but "
                            f"add a photo with the face clearly visible as a second reference for the likeness.")
    elif out["face"].get("found") is False and subject != "avatar":
        if subject == "photo":
            weak = True
        warnings.append("can't find a face. face the camera, nothing covering it. (avatars without a human face "
                        "are fine: set who's this to my avatar.)")

    out["grade"] = "weak" if weak else ("ok" if warnings else "good")
    out["ok"] = not weak
    out["warnings"] = warnings
    if warnings:
        out["tip"] = TIP
    return out


def summary_markdown(res: dict[str, Any] | None, with_tip: bool = True) -> str:
    """One friendly block for the window (``with_tip=False`` where the tip is already on screen)."""
    if not res or not res.get("warnings"):
        return "reference check: looks good." if res else ""
    head = "**weak reference.**" if res["grade"] == "weak" else "**reference could be better.**"
    return f"{head} " + " ".join(res["warnings"]) + (f"\n\n{TIP}" if with_tip else "")


def safe_check(src: Any, subject: str = "auto") -> dict[str, Any] | None:
    """check_reference that never raises (a broken image just gets no verdict)."""
    if not src:
        return None
    try:
        return check_reference(src, subject)
    except Exception:  # noqa: BLE001
        return None


def check_clip(width: int, height: int, fps: float | None = None, video_kbps: float | None = None,
               name: str = "this clip") -> dict[str, Any]:
    """Source-quality check of a clip (numbers from ffprobe / yt-dlp): low resolution, over-compression, low fps.

    Never blocks; the warnings suggest finding the original upload (clipfetch.rank_sources picks the best copy).
    """
    warnings: list[str] = []
    short = min(width, height)
    out: dict[str, Any] = {"width": width, "height": height, "fps": fps, "video_kbps": video_kbps}
    if short < CLIP_MIN_SIDE:
        warnings.append(f"{name} is only {width}x{height}: low-quality input means low-quality results. look for a "
                        f"720p+ copy (the original upload beats reposts).")
    if video_kbps:
        need = CLIP_MIN_MBPS_720 * 1000 * (width * height) / (1280 * 720)
        if video_kbps < need * 0.999:
            warnings.append(f"{name} is heavily compressed ({video_kbps / 1000:.1f} Mbps at {width}x{height}). "
                            f"a higher-bitrate copy keeps faces sharp.")
    if fps and fps < CLIP_MIN_FPS:
        warnings.append(f"{name} is {fps:g} fps; fast motion will look choppy.")
    out["warnings"] = warnings
    out["ok"] = not warnings
    return out
