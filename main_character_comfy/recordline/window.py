"""Where does the character speak? The original line's speaking window from the Demucs vocal stem (sound) plus
mouth openness (MediaPipe FaceLandmarker, 478-point mesh), with speech-to-text word times as the first guess.

Sound decides the timing (a stem onset is sample accurate); the mouth confirms and nudges it: when both agree within
0.25 s the window is a 70/30 blend (sound/mouth), otherwise sound wins and a note says they disagreed.
Without MediaPipe the mouth check falls back to YuNet face boxes + mouth-band motion energy (rougher).
Without Demucs the clip's own audio stands in for the vocal stem.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .. import stems
from ..media import run_ffmpeg
from .deps import HINTS, face_landmarker_model, has_module
from .dsp import SR, runs, speech_band_db

AGREE_S = 0.25
SOUND_WEIGHT = 0.7
MOUTH_PAD_S = 0.5  # frames read around the window for the mouth check


@dataclass
class Stems:
    clip_audio: Path            # 48 kHz stereo wav of the clip
    vocals: Path | None = None   # Demucs vocals (None = not separated)
    background: Path | None = None  # Demucs no_vocals (music + SFX)
    note: str | None = None


@dataclass
class SpeakingWindow:
    start: float
    end: float
    source: str
    sound: list | None = None   # [on, off] from the vocal stem
    mouth: list | None = None   # [on, off] from mouth openness
    words: list | None = None   # [first word start, last word end] from speech-to-text
    mouth_method: str | None = None
    notes: list = field(default_factory=list)

    @property
    def seconds(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("start", "end"):
            d[k] = round(d[k], 3)
        for k in ("sound", "mouth", "words"):
            if d[k]:
                d[k] = [round(v, 3) for v in d[k]]
        return d


def split_stems(video: str | Path, work: Path) -> Stems:
    """Clip audio + Demucs two-stem split (reuses :func:`stems.separate_vocals`)."""
    work.mkdir(parents=True, exist_ok=True)
    clip_audio = work / "clip_audio.wav"
    run_ffmpeg(["-i", str(video), "-vn", "-ac", "2", "-ar", str(SR), "-c:a", "pcm_f32le", str(clip_audio)])
    vocals = stems.separate_vocals(clip_audio, work / "stems")
    if vocals is None:
        return Stems(clip_audio, note=HINTS["demucs"] if stems.separator_command() is None else
                     "Voice separation failed on this clip, so the original voice may be faintly audible under yours.")
    bg = vocals.with_name("no_vocals.wav")
    return Stems(clip_audio, vocals, bg if bg.is_file() else None)


def sound_span(x: np.ndarray, sr: int = SR, around: tuple[float, float] | None = None,
               search_pad: float = 0.3, hint: float | None = None) -> tuple[float, float] | None:
    """Speech onset/offset in the speech band: frames above max(p95 - 24 dB, -60 dBFS), runs >= 50 ms, inside
    ``around`` +- ``search_pad`` (or the whole signal: the loudest cluster, or the one nearest ``hint`` s).
    Gaps < 0.35 s are bridged. (-30 dB caught Demucs bleed / breaths 0.7 s before a real line.)"""
    db, hop = speech_band_db(x, sr)
    t = np.arange(len(db)) * hop / sr
    sel = np.ones(len(db), bool) if around is None else (t >= around[0] - search_pad) & (t <= around[1] + search_pad)
    if not sel.any():
        return None
    ref = db[sel]
    thr = max(np.percentile(ref, 95) - 24, -60)
    act = (db > thr) & sel
    s, ln = runs(act)
    keep = [(a, n) for a, n in zip(s, ln) if n >= 5]
    if not keep:
        return None
    if around is None:  # the biggest cluster (gaps < 0.35 s bridged)
        groups, cur = [], [keep[0]]
        for a, n in keep[1:]:
            if (a - (cur[-1][0] + cur[-1][1])) * hop / sr < 0.35:
                cur.append((a, n))
            else:
                groups.append(cur)
                cur = [(a, n)]
        groups.append(cur)
        if hint is not None:  # the cluster nearest the hint (0 if it contains the hint)
            def dist(gr):
                a_, b_ = gr[0][0] * hop / sr, (gr[-1][0] + gr[-1][1]) * hop / sr
                return 0.0 if a_ - 0.3 <= hint <= b_ else min(abs(hint - a_), abs(hint - b_))
            g = min(groups, key=dist)
        else:
            g = max(groups, key=lambda gr: sum(10 ** (db[a:a + n] / 10).sum() for a, n in gr))
        a0, a1 = g[0][0], g[-1][0] + g[-1][1]
    else:
        a0, a1 = keep[0][0], keep[-1][0] + keep[-1][1]
    return float(a0 * hop / sr), float(a1 * hop / sr)


# --------------------------------------------------------------------------- mouth
def openness_from_landmarks(L: np.ndarray) -> np.ndarray:
    """Inner-lip gap (13-14) over face height (10-152) per frame, NaN gaps interpolated."""
    face = np.abs(L[:, 152, 1] - L[:, 10, 1])
    o = np.abs(L[:, 14, 1] - L[:, 13, 1]) / np.maximum(face, 1e-6)
    o = np.asarray(o, float)
    bad = np.isnan(o)
    if bad.any() and (~bad).any():
        o[bad] = np.interp(np.where(bad)[0], np.where(~bad)[0], o[~bad])
    return o


def _mediapipe_openness(frames: np.ndarray, fps: float, box) -> np.ndarray | None:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    x, y, side = box
    scale = max(1.0, 512 / max(side, 1))
    opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(face_landmarker_model())),
                                        running_mode=vision.RunningMode.VIDEO, num_faces=1,
                                        min_face_detection_confidence=0.2, min_tracking_confidence=0.2,
                                        min_face_presence_confidence=0.2)
    res = []
    with vision.FaceLandmarker.create_from_options(opts) as lm:
        for i, fr in enumerate(frames):
            c = fr[y:y + side, x:x + side]
            if scale > 1:
                c = cv2.resize(c, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            r = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(c)),
                                    int(i * 1000 / fps))
            if r.face_landmarks:
                res.append(np.array([[p.x * c.shape[1], p.y * c.shape[0]] for p in r.face_landmarks[0]], np.float32))
            else:
                res.append(np.full((478, 2), np.nan, np.float32))
    L = np.stack(res)
    if np.isnan(L[:, 0, 0]).mean() > 0.5:
        return None
    return openness_from_landmarks(L)


def mouth_track(video: str | Path, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray, str] | None:
    """(times, openness, method) around [t0, t1], or None when no face is found. MediaPipe when available,
    else YuNet mouth-band motion (sync.mouth_energy)."""
    from .. import lipdub, media, sync

    fps = lipdub.probe_fps(video)
    total = media.media_duration(video)
    a, b = max(0.0, t0 - MOUTH_PAD_S), min(total, t1 + MOUTH_PAD_S)
    f0, f1 = int(round(a * fps)), max(int(round(a * fps)) + 2, int(round(b * fps)))
    frames = lipdub.read_frames(video, f0, f1, media.video_size(video))
    if len(frames) < 3:
        return None
    track = lipdub.detect_faces_robust(frames)
    if lipdub.coverage(track) < lipdub.MIN_FACE_FRAMES:
        return None
    track = lipdub.fill_track(track)
    times = (f0 + np.arange(len(frames))) / fps
    if has_module("mediapipe") and face_landmarker_model() is not None:
        try:
            box = lipdub.crop_box(track, media.video_size(video), scale=1.6)
            o = _mediapipe_openness(frames, fps, box)
            if o is not None:
                return times, o, "mediapipe"
        except Exception:  # noqa: BLE001 - fall back to the rough check below
            pass
    faces = [None if np.isnan(r[0]) else r[:4] for r in track]
    e = sync.mouth_energy(list(frames), faces)
    return times, np.asarray(e, float), "motion"


def mouth_span(times: np.ndarray, o: np.ndarray, approx: tuple[float, float]) -> tuple[float, float] | None:
    """First / last frame near ``approx`` where the mouth is open (syncchk's clip-relative thresholds)."""
    if len(o) < 3 or not np.isfinite(o).all() or np.ptp(o) < 1e-6:
        return None
    closed = np.percentile(o, 15)
    opened = closed + 0.35 * (np.percentile(o, 90) - closed)
    sel = (times >= approx[0] - AGREE_S) & (times <= approx[1] + AGREE_S) & (o >= opened)
    if not sel.any():
        return None
    idx = np.where(sel)[0]
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.04
    return float(times[idx[0]]), float(times[idx[-1]] + dt)


def combine(sound: tuple[float, float] | None, mouth: tuple[float, float] | None,
            words: tuple[float, float] | None, method: str | None) -> SpeakingWindow:
    base = sound or words
    if base is None:
        raise ValueError("no speech found")
    src = "sound" if sound else "words"
    notes = []
    start, end = base
    if mouth:
        if abs(mouth[0] - base[0]) <= AGREE_S and abs(mouth[1] - base[1]) <= AGREE_S:
            w = SOUND_WEIGHT
            start = w * base[0] + (1 - w) * mouth[0]
            end = w * base[1] + (1 - w) * mouth[1]
            src += "+mouth"
        else:
            notes.append("The mouth and the sound disagree a little here, so the window follows the sound.")
    return SpeakingWindow(start, end, src, list(sound) if sound else None, list(mouth) if mouth else None,
                          list(words) if words else None, method, notes)
