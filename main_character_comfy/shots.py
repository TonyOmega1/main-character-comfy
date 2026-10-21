"""Hard cuts and who-is-who across them (CPU only).

Why: SAM 3.1 tracking follows a person frame to frame; at a hard cut everyone jumps, so a one-pass swap loses people
after the first cut (3-person dance test, Oct 19: only legs swapped in shot 1, shot 2 untouched). MC therefore swaps each shot on its
own. This module finds the cuts and keeps "person 1" the same person in every shot by appearance, instead of
re-numbering left to right per shot.

Method borrowed from SwapJutsu's documented pipeline (stages 02 and 05, MIT): a cut where the frame histogram jumps,
shots shorter than ``MIN_SHOT_FRAMES`` merged as false cuts (flashes / transition frames); identity by an appearance
distance (colour histogram inside each person's mask, plus a coarse spatial-colour layout), average-linkage, and two
people seen in the same shot can never be merged. Pending a real end-to-end test on the GPU.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from . import media

HIST_BINS = 8  # per channel (RGB) for the cut detector
# A cut needs the colours AND the picture's layout to jump. Measured on a 3-person dance clip (24 fps, 578 frames, 3 real
# cuts at 53 / 421 / 503): colour jump 0.39-0.49 at the cuts but also 0.29-0.34 at three stage-light flashes (162,
# 240, 332), so colour alone can't tell them apart; layout jump (1 - luma correlation) 0.67-0.74 at the cuts vs
# 0.25-0.28 at the flashes and < 0.1 inside a shot.
CUT_DISTANCE = 0.15  # half L1 distance between consecutive normalized RGB histograms (0..1)
STRUCT_CUT = 0.5  # 1 - correlation of consecutive low-res luma frames (0 same layout .. 1 unrelated)
HIST_ALONE = 0.6  # a colour jump this big is a cut even when the layout correlates (e.g. a cut to a colour flip)
CUT_RATIO = 4.0  # ...and the layout jump at least this many times the recent median (whip pans / shaky cam)
MIN_SHOT_FRAMES = 4  # a shorter "shot" is a flash or a transition frame: joined to its longer neighbour
ANALYSIS_SIZE = (64, 36)  # frames are read this small (fast; enough for histograms)
MERGE_DISTANCE = 0.33  # appearance distance below which two tracklets are the same person


@dataclass
class Shot:
    start: int  # first frame (inclusive)
    end: int  # last frame + 1

    @property
    def frames(self) -> int:
        return self.end - self.start


def read_small_frames(video: str | Path, size: tuple[int, int] = ANALYSIS_SIZE) -> np.ndarray:
    """All frames of ``video`` as uint8 RGB, (n, h, w, 3), scaled to ``size`` (one ffmpeg pass, raw pipe)."""
    w, h = size
    proc = subprocess.run([media.ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-i", str(video), "-an",
                           "-vf", f"scale={w}:{h}:flags=area", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                          capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"can't read {video}: {proc.stderr.decode(errors='replace')[-300:]}")
    data = np.frombuffer(proc.stdout, dtype=np.uint8)
    n = data.size // (w * h * 3)
    return data[: n * w * h * 3].reshape(n, h, w, 3)


def frame_histograms(frames: np.ndarray, bins: int = HIST_BINS) -> np.ndarray:
    """Normalized joint RGB histograms, (n, bins**3)."""
    q = (frames.astype(np.int32) * bins) // 256
    idx = (q[..., 0] * bins + q[..., 1]) * bins + q[..., 2]
    n = frames.shape[0]
    out = np.zeros((n, bins ** 3), dtype=np.float32)
    for i in range(n):
        out[i] = np.bincount(idx[i].ravel(), minlength=bins ** 3)
    out /= np.maximum(1.0, out.sum(axis=1, keepdims=True))
    return out


def cut_scores(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(colour jump, layout jump) between each frame and the previous one (0 for frame 0), both 0..1."""
    hist = frame_histograms(frames)
    n = len(hist)
    d = np.zeros(n, dtype=np.float32)
    s = np.zeros(n, dtype=np.float32)
    if n > 1:
        d[1:] = 0.5 * np.abs(hist[1:] - hist[:-1]).sum(axis=1)
        g = frames.astype(np.float32).mean(axis=3).reshape(n, -1)
        g -= g.mean(axis=1, keepdims=True)
        norm = np.sqrt((g * g).sum(axis=1)) + 1e-6
        corr = (g[1:] * g[:-1]).sum(axis=1) / (norm[1:] * norm[:-1])
        flat = (norm[1:] < 1e-3 * g.shape[1]) & (norm[:-1] < 1e-3 * g.shape[1])  # two flat frames: same layout
        s[1:] = np.where(flat, 0.0, np.clip(1.0 - corr, 0.0, 1.0))
    return d, s


def detect_cuts_from_scores(d: Sequence[float], struct: Sequence[float] | None = None, *,
                            threshold: float = CUT_DISTANCE, struct_cut: float = STRUCT_CUT,
                            hist_alone: float = HIST_ALONE, ratio: float = CUT_RATIO,
                            min_shot: int = MIN_SHOT_FRAMES, window: int = 12) -> list[Shot]:
    """Shots from per-frame jumps: a cut where (colour >= ``threshold`` and layout >= ``struct_cut`` and layout >=
    ``ratio`` x its recent median) or colour >= ``hist_alone``. Without ``struct`` only the colour test is used
    (threshold ``hist_alone``). Shots shorter than ``min_shot`` are merged into the longer neighbour."""
    d = list(map(float, d))
    st = list(map(float, struct)) if struct is not None else None
    n = len(d)
    if n == 0:
        return []
    cuts = []
    for i in range(1, n):
        if d[i] >= hist_alone:
            cuts.append(i)
            continue
        if st is None:
            continue
        recent = st[max(1, i - window):i]
        base = float(np.median(recent)) if recent else 0.0
        if d[i] >= threshold and st[i] >= struct_cut and st[i] >= ratio * max(base, 0.02):
            cuts.append(i)
    bounds = [0, *cuts, n]
    shots = [Shot(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    return merge_short(shots, min_shot)


def merge_short(shots: list[Shot], min_shot: int = MIN_SHOT_FRAMES) -> list[Shot]:
    shots = [Shot(s.start, s.end) for s in shots]
    changed = True
    while changed and len(shots) > 1:
        changed = False
        for k, s in enumerate(shots):
            if s.frames >= min_shot:
                continue
            if k == 0:
                j = 1
            elif k == len(shots) - 1:
                j = k - 1
            else:
                j = k - 1 if shots[k - 1].frames >= shots[k + 1].frames else k + 1
            a, b = (shots[j], s) if j < k else (s, shots[j])
            merged = Shot(a.start, b.end)
            lo = min(j, k)
            shots[lo:lo + 2] = [merged]
            changed = True
            break
    return shots


def detect_cuts(video: str | Path, **kw) -> list[Shot]:
    """Shots of ``video`` (frame ranges at its own frame rate). One shot = no hard cuts."""
    d, s = cut_scores(read_small_frames(video))
    return detect_cuts_from_scores(d, s, **kw)


# --------------------------------------------------------------------------- #
# Optional second opinion: PySceneDetect (BSD-3, pip install scenedetect==0.7.2)
# --------------------------------------------------------------------------- #
# Measured Oct 19 on our clips: on a 10-shot test video it agrees with MC 10/10; on the dance clip it also
# flags the 3 stage-light flashes (frames 162, 240, 332) as cuts (AdaptiveDetector and ContentDetector alike), which
# MC's layout test rightly ignores. So it is a REPORT, never a vote: it can't add or remove a split.
SECOND_TOLERANCE = 2  # frames: a PySceneDetect cut this close to an MC cut is "the same cut"


def scenedetect_cuts(video: str | Path) -> list[int] | None:
    """Cut frames from PySceneDetect's AdaptiveDetector, or None when it isn't installed or fails."""
    try:
        from scenedetect import AdaptiveDetector, detect
    except Exception:  # noqa: BLE001 - optional dependency
        return None
    try:
        scenes = detect(str(video), AdaptiveDetector(), show_progress=False)
    except Exception:  # noqa: BLE001
        return None
    return [int(s[0].get_frames()) for s in scenes[1:]]


def compare_cuts(mc: Sequence[int], other: Sequence[int], tol: int = SECOND_TOLERANCE) -> dict:
    """{'agree', 'mc_only', 'other_only'} between two cut lists (frame numbers, +/- ``tol``)."""
    other = list(other)
    agree, mc_only = [], []
    used: set[int] = set()
    for c in mc:
        hit = next((o for o in other if abs(o - c) <= tol and o not in used), None)
        if hit is None:
            mc_only.append(int(c))
        else:
            used.add(hit)
            agree.append(int(c))
    return {"agree": agree, "mc_only": mc_only, "other_only": [int(o) for o in other if o not in used]}


def second_opinion(video: str | Path, mc_cuts: Sequence[int], fps: float | None = None) -> dict | None:
    """PySceneDetect's view of ``video`` next to MC's cuts, for the run record and a one-line note. None = not
    installed. ``fps`` (of ``video``) turns frames into seconds in the note."""
    sd = scenedetect_cuts(video)
    if sd is None:
        return None
    cmp_ = compare_cuts(mc_cuts, sd)
    extra = cmp_["other_only"]
    note = ""
    if extra:
        at = ", ".join(f"{f / fps:.1f} s" if fps else f"frame {f}" for f in extra[:4])
        note = (f"PySceneDetect also flagged {len(extra)} spot(s) ({at}) that MC treats as one shot (usually a light "
                "flash or fast motion). If a person is lost there, set 'split at cuts' on (guide: #guide-cuts)")
    elif cmp_["mc_only"]:
        note = f"PySceneDetect didn't see {len(cmp_['mc_only'])} of MC's cuts; MC splits there anyway"
    return dict(cmp_, detector="PySceneDetect AdaptiveDetector", note=note)


# --------------------------------------------------------------------------- #
# Who is who across shots
# --------------------------------------------------------------------------- #
@dataclass
class Sighting:
    shot: int
    slot: int  # left-to-right index inside that shot's scan
    feature: np.ndarray = field(repr=False)


def appearance(frame_rgb: np.ndarray, mask: np.ndarray | None = None, box: tuple[int, int, int, int] | None = None,
               bins: int = 12) -> np.ndarray:
    """Appearance vector of one person: an HSV hue/saturation histogram + a value histogram inside the mask (or box),
    plus the mean colour of the top / middle / bottom thirds (hair, top, trousers). Normalized pieces, concatenated."""
    from PIL import Image

    img = frame_rgb
    if mask is None:
        mask = np.zeros(img.shape[:2], dtype=bool)
        if box is not None:
            x0, y0, x1, y1 = box
            mask[max(0, y0):y1 + 1, max(0, x0):x1 + 1] = True
        else:
            mask[:] = True
    mask = mask.astype(bool)
    if not mask.any():
        mask = np.ones(img.shape[:2], dtype=bool)
    hsv = np.asarray(Image.fromarray(img.astype(np.uint8)).convert("HSV"), dtype=np.float32)
    px = hsv[mask]
    hs, _, _ = np.histogram2d(px[:, 0], px[:, 1], bins=[bins, 4], range=[[0, 256], [0, 256]])
    hs = hs.ravel() / max(1.0, hs.sum())
    v, _ = np.histogram(px[:, 2], bins=8, range=(0, 256))
    v = v / max(1.0, v.sum())
    ys = np.nonzero(mask)[0]
    y0, y1 = ys.min(), ys.max() + 1
    thirds = []
    for k in range(3):
        a = y0 + (y1 - y0) * k // 3
        b = y0 + (y1 - y0) * (k + 1) // 3
        sub = mask.copy()
        sub[:a] = False
        sub[b:] = False
        part = img[sub] if sub.any() else img[mask]
        thirds.append(part.mean(axis=0) / 255.0)
    layout = np.concatenate(thirds)
    return np.concatenate([hs * 0.6, v * 0.2, layout * 0.2 / 3.0]).astype(np.float32)


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """0 (identical) .. ~1: half L1 of the histogram parts plus the layout colour gap."""
    return float(0.5 * np.abs(a - b).sum() / 0.5)


def link_identities(sightings: list[Sighting], merge_distance: float = MERGE_DISTANCE) -> list[int]:
    """Identity id per sighting (0, 1, 2 ...), average-linkage; two sightings from the same shot are never merged.
    Ids are numbered by the left-to-right slot of each identity's first appearance (first shot first)."""
    clusters: list[list[int]] = [[i] for i in range(len(sightings))]

    def link(a: list[int], b: list[int]) -> float:
        if {sightings[i].shot for i in a} & {sightings[j].shot for j in b}:
            return float("inf")
        return float(np.mean([distance(sightings[i].feature, sightings[j].feature) for i in a for j in b]))

    while True:
        best, pair = float("inf"), None
        for x in range(len(clusters)):
            for y in range(x + 1, len(clusters)):
                dxy = link(clusters[x], clusters[y])
                if dxy < best:
                    best, pair = dxy, (x, y)
        if pair is None or best >= merge_distance:
            break
        x, y = pair
        clusters[x] = clusters[x] + clusters[y]
        del clusters[y]
    order = sorted(range(len(clusters)), key=lambda c: min((sightings[i].shot, sightings[i].slot) for i in clusters[c]))
    ids = [0] * len(sightings)
    for new, c in enumerate(order):
        for i in clusters[c]:
            ids[i] = new
    return ids


def assign_across_shots(first_shot_refs: Sequence, sightings: list[Sighting],
                        merge_distance: float = MERGE_DISTANCE) -> dict[int, list]:
    """Per shot, the reference list in that shot's left-to-right slot order: the user assigned refs on shot 0's
    numbering (``first_shot_refs``); everyone else is matched by appearance. A person never seen in shot 0 gets None
    (left as filmed), and is reported by the caller."""
    ids = link_identities(sightings, merge_distance)
    ref_of_id: dict[int, object] = {}
    for s, ident in zip(sightings, ids):
        if s.shot == 0 and s.slot < len(first_shot_refs):
            ref_of_id[ident] = first_shot_refs[s.slot]
    per_shot: dict[int, list] = {}
    for s, ident in zip(sightings, ids):
        row = per_shot.setdefault(s.shot, [])
        while len(row) <= s.slot:
            row.append(None)
        row[s.slot] = ref_of_id.get(ident)
    return per_shot


def summary(shots: list[Shot], fps: float) -> str:
    if len(shots) <= 1:
        return "no hard cuts"
    marks = ", ".join(f"{s.start / fps:.2f}s" for s in shots[1:])
    return f"{len(shots) - 1} hard cut{'s' if len(shots) > 2 else ''} (at {marks}): {len(shots)} shots"
