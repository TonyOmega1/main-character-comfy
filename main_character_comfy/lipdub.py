"""Record your own line, then re-sync the lips (DubMaster's method, Oct 18 "Not You" dub; pending a real MC test).

The audio side is :mod:`redub` (denoise, fit, place, EQ toward the clip, short reverb, smooth duck, one gain).
This module does the picture side, only around the line:

1. find the face in the line's window (OpenCV YuNet: box + eyes, nose, mouth corners), fill gaps, smooth
2. one fixed square head crop for the whole window, its first frame = InfiniteTalk's start image (640x640)
3. InfiniteTalk (Wan2.1 I2V + InfiniteTalk, a normal MC lip-sync job) speaks the new line from that frame
4. paste ONLY the mouth and jaw back: the render is aligned to the original face every frame (similarity transform on
   the 5 landmarks, smoothed over time), the mask is a feathered ellipse around the mouth and chin, colours are
   matched to the original skin ring around it, and the patch fades in / out over ~80 ms at the window's edges
5. every frame outside the window is the original, byte for byte re-encoded; the mix from redub is the soundtrack

What DubMaster did by hand that this does automatically: crop choice, landmark alignment, lips + jaw mask, colour
offset, fade in / out. What it did that this MVP skips: MediaPipe's 478-point mesh (moustache-safe mask top), grain
and detail matching, picking the best of two seeds per word. CPU parts are unit tested; the GPU part is one job.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import media

GEN_SIZE = 640  # InfiniteTalk 480p model, square head crop (what the Rec5 dub used)
GEN_FPS = 25
CROP_SCALE = 2.6  # crop side = this x the median face height (head + a little neck / shoulders)
PAD_SECONDS = 0.25  # window = the line +- this (lips open a touch before the sound)
FADE_SECONDS = 0.08  # patch fade in / out at the window edges
MIN_FACE_FRAMES = 0.6  # need a face in at least this share of the window's frames
AUDIO_SCALE = 0.85  # InfiniteTalk audio strength (Rec5: 0.85 kept the cartoon's mouth shapes cleaner than 1.0)


@dataclass
class Window:
    t0: float
    t1: float
    f0: int  # first frame (clip fps)
    f1: int  # last frame + 1


def probe_fps(video: str | Path) -> float:
    proc = subprocess.run([media.ffmpeg_exe(), "-hide_banner", "-i", str(video)], capture_output=True, text=True)
    m = re.search(r"(\d+(?:\.\d+)?) fps", proc.stderr)
    if not m:
        raise RuntimeError(f"can't read the frame rate of {video}")
    return float(m.group(1))


def line_window(start: float, line_seconds: float, total: float, fps: float, pad: float = PAD_SECONDS) -> Window:
    t0 = max(0.0, start - pad)
    t1 = min(total, start + line_seconds + pad)
    f0 = int(round(t0 * fps))
    f1 = max(f0 + 1, min(int(round(total * fps)), int(round(t1 * fps))))
    return Window(f0 / fps, f1 / fps, f0, f1)


def read_frames(video: str | Path, f0: int, f1: int, size: tuple[int, int]) -> np.ndarray:
    """Frames [f0, f1) as uint8 RGB (n, h, w, 3) at the video's own size."""
    w, h = size
    proc = subprocess.run([media.ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-i", str(video), "-an",
                           "-vf", f"select=between(n\\,{int(f0)}\\,{int(f1) - 1}),setpts=N/TB", "-vsync", "0",
                           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
    if proc.returncode:
        raise RuntimeError(proc.stderr.decode(errors="replace")[-300:])
    data = np.frombuffer(proc.stdout, np.uint8)
    n = data.size // (w * h * 3)
    return data[: n * w * h * 3].reshape(n, h, w, 3).copy()


def write_frames(frames: np.ndarray, dest: str | Path, fps: float) -> Path:
    n, h, w, _ = frames.shape
    proc = subprocess.run([media.ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                           "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-", "-c:v", "libx264",
                           "-crf", "12", "-pix_fmt", "yuv420p", str(dest)], input=np.ascontiguousarray(frames).tobytes(),
                          capture_output=True)
    if proc.returncode:
        raise RuntimeError(proc.stderr.decode(errors="replace")[-300:])
    return Path(dest)


# --------------------------------------------------------------------------- faces
def detect_faces(frames: np.ndarray, score: float = 0.5) -> np.ndarray:
    """(n, 14) YuNet rows (x, y, w, h, right eye, left eye, nose, right / left mouth corner) per frame, NaN where no
    face. The biggest face, then the one nearest the previous frame's. Needs opencv (MC's requirements)."""
    import cv2

    out = np.full((len(frames), 14), np.nan, np.float32)
    if not len(frames):
        return out
    h, w = frames.shape[1:3]
    det = cv2.FaceDetectorYN.create(str(media.YUNET_MODEL), "", (w, h), score, 0.3, 20)
    prev = None
    for i, fr in enumerate(frames):
        bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
        _, faces = det.detect(bgr)
        if faces is None or not len(faces):  # dark / neon renders: retry brightened (what the Rec5 dub needed)
            _, faces = det.detect(cv2.convertScaleAbs(bgr, alpha=1.8, beta=10))
        if faces is None or not len(faces):
            continue
        if prev is None:
            k = int(np.argmax(faces[:, 2] * faces[:, 3]))
        else:
            c = faces[:, 0:2] + faces[:, 2:4] / 2
            k = int(np.argmin(((c - prev) ** 2).sum(1)))
        out[i] = faces[k, :14]
        prev = faces[k, 0:2] + faces[k, 2:4] / 2
    return out


def detect_faces_robust(frames: np.ndarray, need: float = MIN_FACE_FRAMES) -> np.ndarray:
    """``detect_faces`` at a normal threshold, lowered step by step until a face is found in ``need`` of the frames
    (stylised / dark faces score low: the Rec5 cartoon render was found in 1 of 81 frames at 0.5, 48 at 0.3, 75 at
    0.2)."""
    best = None
    for score in (0.5, 0.3, 0.2):
        t = detect_faces(frames, score)
        if best is None or (~np.isnan(t[:, 0])).sum() > (~np.isnan(best[:, 0])).sum():
            best = t
        if len(frames) and (~np.isnan(t[:, 0])).mean() >= need:
            return t
    return best if best is not None else detect_faces(frames)


def coverage(track: np.ndarray) -> float:
    return float((~np.isnan(track[:, 0])).mean()) if len(track) else 0.0


def fill_track(track: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Interpolate missing frames (NaN rows) and smooth a little over time."""
    t = np.array(track, np.float32)
    ok = ~np.isnan(t[:, 0])
    if not ok.any():
        return t
    idx = np.arange(len(t))
    for c in range(t.shape[1]):
        t[:, c] = np.interp(idx, idx[ok], t[ok, c])
    if sigma and len(t) > 2:
        t = smooth(t, sigma)
    return t


def smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing along axis 0 with edge padding (numpy only; scipy isn't an MC dependency)."""
    r = max(1, int(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    flat = x.reshape(len(x), -1).astype(np.float64)
    pad = np.pad(flat, ((r, r), (0, 0)), mode="edge")
    out = np.stack([np.convolve(pad[:, c], k, mode="valid") for c in range(flat.shape[1])], 1)
    return out.reshape(x.shape).astype(x.dtype)


def crop_box(track: np.ndarray, frame_size: tuple[int, int], scale: float = CROP_SCALE) -> tuple[int, int, int]:
    """(x, y, side): one square crop for the whole window, centred on the median face, clamped inside the frame."""
    w, h = frame_size
    cx = float(np.nanmedian(track[:, 0] + track[:, 2] / 2))
    cy = float(np.nanmedian(track[:, 1] + track[:, 3] / 2))
    side = int(min(w, h, max(64, scale * float(np.nanmedian(track[:, 3])))))
    side -= side % 2
    x = int(np.clip(cx - side / 2, 0, w - side))
    y = int(np.clip(cy - side * 0.45, 0, h - side))
    return x, y, side


def landmarks(row: np.ndarray) -> np.ndarray:
    """The 5 YuNet points (right eye, left eye, nose, right / left mouth corner) as (5, 2)."""
    return np.asarray(row[4:14], np.float32).reshape(5, 2)


def to_frame_coords(track: np.ndarray, box: tuple[int, int, int], gen_size: int = GEN_SIZE) -> np.ndarray:
    """A track found in the generated square crop -> original frame pixels."""
    x, y, side = box
    s = side / float(gen_size)
    t = np.array(track, np.float32)
    t[:, 0:14:2] = t[:, 0:14:2] * s + x
    t[:, 1:14:2] = t[:, 1:14:2] * s + y
    t[:, 2] = track[:, 2] * s
    t[:, 3] = track[:, 3] * s
    return t


def mouth_mask(pts: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Feathered 0..1 mask over the mouth and jaw (from the 5 points), (h, w) float32. Stops below the nose, so a
    moustache / the nose stay the original's."""
    import cv2

    h, w = shape
    eye_c = (pts[0] + pts[1]) / 2
    mouth_c = (pts[3] + pts[4]) / 2
    d = float(np.linalg.norm(mouth_c - eye_c)) or 1.0
    mw = float(np.linalg.norm(pts[3] - pts[4])) or d * 0.6
    ang = float(np.degrees(np.arctan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0])))
    centre = mouth_c + (mouth_c - eye_c) / d * (0.28 * d)
    axes = (int(max(4, 0.55 * mw + 0.18 * d)), int(max(4, 0.50 * d)))
    m = np.zeros((h, w), np.float32)
    cv2.ellipse(m, (int(centre[0]), int(centre[1])), axes, ang, 0, 360, 1.0, -1)
    nose_y = float(pts[2][1]) + 0.18 * d  # keep the nose and most of a moustache
    k = max(1.0, 0.12 * d)
    m = cv2.GaussianBlur(m, (0, 0), k)
    ys = np.arange(h, dtype=np.float32)[:, None]
    return m * np.clip((ys - nose_y) / (2 * k), 0.0, 1.0)  # soft top edge, applied after the blur


def align(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """2x3 similarity transform mapping the render's 5 points onto the original's."""
    import cv2

    M, _ = cv2.estimateAffinePartial2D(src_pts.astype(np.float32), dst_pts.astype(np.float32), method=cv2.LMEDS)
    if M is None:
        M = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
    return M.astype(np.float32)


def fade_weights(n: int, fps: float, fade: float = FADE_SECONDS) -> np.ndarray:
    k = max(1, int(round(fade * fps)))
    w = np.ones(n, np.float32)
    ramp = np.linspace(0, 1, k + 2, dtype=np.float32)[1:-1]
    w[:k] = np.minimum(w[:k], ramp[: min(k, n)])
    w[-k:] = np.minimum(w[-k:], ramp[::-1][-min(k, n):])
    return w


def paste_mouths(orig: np.ndarray, gen: np.ndarray, orig_track: np.ndarray, gen_track: np.ndarray, fps: float,
                 gen_fps: float = GEN_FPS, fade: float = FADE_SECONDS) -> tuple[np.ndarray, dict]:
    """``orig`` (n, h, w, 3) window frames with the render's mouth pasted in. ``gen`` are full-frame-sized renders
    (already placed back from the crop) or crop renders with ``gen_track`` in frame coordinates. Returns frames + a
    small report. Frame m of the window shows render frame round(m / fps * gen_fps)."""
    import cv2

    n, h, w, _ = orig.shape
    out = orig.copy()
    wts = fade_weights(n, fps, fade)
    js = np.clip(np.round(np.arange(n) / fps * gen_fps).astype(int), 0, len(gen) - 1)
    Ms = np.stack([align(landmarks(gen_track[j]), landmarks(orig_track[m])) for m, j in enumerate(js)])
    if n > 2:
        Ms = smooth(Ms, 1.0)
    shifts = []
    for m, j in enumerate(js):
        g = cv2.warpAffine(gen[j], Ms[m], (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        mask = mouth_mask(landmarks(orig_track[m]), (h, w))
        ring = (cv2.dilate((mask > 0.05).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0) & (mask < 0.05)
        go = cv2.cvtColor(g, cv2.COLOR_RGB2LAB).astype(np.float32)
        oo = cv2.cvtColor(orig[m], cv2.COLOR_RGB2LAB).astype(np.float32)
        if ring.sum() > 20:
            # colour: mean a/b offset; light: map the render's 5/50/95th luminance percentiles onto the original's
            # (DubMaster's exposure match; a plain mean shift lightened a dark beard in the Rec5 check)
            off = oo[ring].mean(0) - go[ring].mean(0)
            go[..., 1:] += off[1:]
            xs = np.r_[0.0, np.percentile(go[..., 0][ring], [5, 50, 95]), 255.0]
            ys = np.r_[0.0, np.percentile(oo[..., 0][ring], [5, 50, 95]), 255.0]
            xs = np.maximum.accumulate(xs + np.arange(5) * 1e-3)
            go[..., 0] = np.interp(go[..., 0], xs, ys)
            shifts.append([round(float(x), 2) for x in off])
        g = cv2.cvtColor(np.clip(go, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
        a = (mask * wts[m])[..., None]
        out[m] = np.clip(orig[m].astype(np.float32) * (1 - a) + g * a, 0, 255).astype(np.uint8)
    rep = {"frames": int(n), "render_frames": int(len(gen)), "fade_frames": int(round(fade * fps)),
           "mean_colour_shift_lab": [round(float(x), 2) for x in np.mean(shifts, 0)] if shifts else None}
    return out, rep


def place_crop_renders(gen_crop: np.ndarray, box: tuple[int, int, int], frame_size: tuple[int, int]) -> np.ndarray:
    """Resize the square renders back to the crop's size and drop them into blank full frames (only the pasted
    mouth region is ever used)."""
    import cv2

    x, y, side = box
    w, h = frame_size
    out = np.zeros((len(gen_crop), h, w, 3), np.uint8)
    for i, g in enumerate(gen_crop):
        out[i, y:y + side, x:x + side] = cv2.resize(g, (side, side), interpolation=cv2.INTER_AREA)
    return out
