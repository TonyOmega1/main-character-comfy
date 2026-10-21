"""Style lock for drawn references (cartoon, anime, flat illustration, painted avatars).

Why it exists (dance-clip test, Oct 19 2026): with a live-action clip, drawn references came out semi-photoreal, most of all
in the SCAIL-2 group pass. Causes found in MC itself:

* the group pass forced the relight LoRA on for avatars (it pulls the character toward the clip's photographic light),
* every sampler runs at CFG 1, where ComfyUI never evaluates the negative prompt, so "not photoreal" negatives did nothing,
* SeedVR2 (a real-world restorer) adds photo texture (pores, fabric weave) at upscale,
* the auto photo/avatar guess reads a painted picture with a gradient sky as a photo,
* the soft composite edge (mask grow + 8 px, blur 9) lets clip pixels fringe the character.

The settings below are the diagnosis's fixes. **PENDING A REAL TEST**: none of them has been A/B'd on a GPU yet, so the
lock is OFF by default (``Request.style_lock = "off"``). ``auto`` turns it on for drawn references only; photos are never
changed. Once the real A/B passes, flip DEFAULT_MODE (and update the README).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable
import os

MODES = ("off", "auto", "on")  # off = today's behaviour; auto = drawn refs only; on = treat every swap as drawn
DEFAULT_MODE = "off"  # pending a real GPU A/B (see module docstring)
PENDING_NOTE = "style lock: pending a real test (settings from the Oct 19 diagnosis, not yet A/B'd on a GPU)"

# Only used when the lock is active. CFG > 1 makes ComfyUI evaluate the negative branch (roughly 2x sampler time).
STYLE_CFG = 2.5
STYLE_NEGATIVE = ("photorealistic, photo, photograph, live action, realistic skin texture, skin pores, wrinkles, "
                  "3D render, CGI, film grain, DSLR, depth of field, realistic fabric texture")
SEEDVR2_BLEND = 0.4  # share of the SeedVR2 frame in the final (the rest is a plain lanczos upscale of the take)
COMPOSITE_GROW = 2  # px the soft paste-back mask grows past the tracked mask (today: MASK_GROW + 8)
COMPOSITE_BLUR = 3.0  # blur radius of that soft mask (today: 9)
MASK_THRESHOLD = 0.5  # mask hygiene: binarise the tracked SAM mask before grow (today: used as is)

# Prompt A/B knobs for supervised test renders (read only while the lock is active; unset = the defaults above).
# MC_STYLE_CFG: CFG (clamped 1.0-4.0). MC_STYLE_NEGATIVE: replaces STYLE_NEGATIVE. MC_STYLE_POSITIVE: style tags put
# FIRST in the positive prompt (Wan weights early tokens most) instead of the default flat-style line at the end.
# MC_STYLE_NEGATIVE_EXTRA: appended to the negative (e.g. Tony's shots: "visible eyes behind sunglasses, ...").
ENV_CFG, ENV_NEG, ENV_POS, ENV_NEG_EXTRA = "MC_STYLE_CFG", "MC_STYLE_NEGATIVE", "MC_STYLE_POSITIVE", "MC_STYLE_NEGATIVE_EXTRA"


def _env_overrides() -> dict[str, Any]:
    out: dict[str, Any] = {}
    raw = os.environ.get(ENV_CFG, "").strip()
    if raw:
        try:
            out["cfg"] = min(4.0, max(1.0, float(raw)))
        except ValueError:
            pass
    neg = os.environ.get(ENV_NEG, "").strip()
    extra = os.environ.get(ENV_NEG_EXTRA, "").strip()
    if neg or extra:
        out["negative_extra"] = merge_negative(neg or STYLE_NEGATIVE, extra)
    pos = os.environ.get(ENV_POS, "").strip()
    if pos:
        out["positive_prefix"] = pos.rstrip(". ")
    return out


@dataclass
class StyleSettings:
    active: bool = False
    mode: str = DEFAULT_MODE
    relight: bool | None = None  # None = leave MC's normal choice; False = force off
    cfg: float = 1.0
    negative_extra: str = ""
    upscale: str | None = None  # None = normal; "blend" = SeedVR2 blended with lanczos (SEEDVR2_BLEND)
    blend: float = SEEDVR2_BLEND
    composite_grow: int | None = None
    composite_blur: float | None = None
    mask_threshold: float | None = None
    why: str = ""
    pending: bool = True
    positive_prefix: str = ""  # style tags that lead the positive prompt (MC_STYLE_POSITIVE); "" = default line
    overrides: list[str] = field(default_factory=list)  # env knobs in effect (logged in the run record)

    def as_record(self) -> dict[str, Any]:
        d = asdict(self)
        if self.active:
            d["note"] = PENDING_NOTE
        return d


def normalize(mode: str | None) -> str:
    m = (mode or DEFAULT_MODE).strip().lower()
    return m if m in MODES else DEFAULT_MODE


def settings_for(mode: str | None, subjects: Iterable[str]) -> StyleSettings:
    """What the lock changes for these subjects ('photo' / 'avatar'). Inactive unless the mode asks for it AND (for
    ``auto``) at least one reference is drawn."""
    mode = normalize(mode)
    subjects = list(subjects)
    drawn = any(s == "avatar" for s in subjects)
    if mode == "off" or (mode == "auto" and not drawn):
        why = "off" if mode == "off" else "no drawn reference"
        return StyleSettings(active=False, mode=mode, why=why)
    ov = _env_overrides()
    cfg = ov.get("cfg", STYLE_CFG)
    return StyleSettings(active=True, mode=mode, relight=False, cfg=cfg, negative_extra=ov.get("negative_extra", STYLE_NEGATIVE),
                         upscale="blend", blend=SEEDVR2_BLEND, composite_grow=COMPOSITE_GROW,
                         composite_blur=COMPOSITE_BLUR, mask_threshold=MASK_THRESHOLD,
                         positive_prefix=ov.get("positive_prefix", ""), overrides=sorted(ov),
                         why="drawn reference: relight off, CFG %.1f + anti-photo negative, SeedVR2 blended %d%%, "
                             "tighter, binarised edge" % (cfg, round(SEEDVR2_BLEND * 100)))


def merge_negative(base: str, extra: str) -> str:
    base, extra = (base or "").strip().rstrip(","), (extra or "").strip()
    if not extra:
        return base
    return f"{base}, {extra}" if base else extra


# --------------------------------------------------------------------------- #
# Better photo / drawing guess (for the visible hint; the explicit choice always wins)
# --------------------------------------------------------------------------- #
LINE_DRAWN_MIN = 2.5  # % of pixels on thin dark lines (5 px black-hat > 45) that marks inked line art
NOISE_PHOTO_MIN = 0.65  # mean |pixel - 3x3 median| in smooth areas: camera noise / photo texture
COLOURS_DRAWN_MAX = 600  # same 5-bit colour count media.looks_like_avatar uses (160 px thumbnail)
COLOURS_UNSURE_MAX = 1500
GREY_SATURATION_MAX = 8.0  # mean max-min channel spread below this = black-and-white / near-greyscale
GREY_NOISE_DRAWN_MAX = 0.30
GREY_NOISE_PHOTO_MIN = 0.70
GREY_NOISE_LEAN = 0.45


@dataclass
class RefKind:
    label: str  # photo | avatar
    confident: bool
    why: str
    scores: dict[str, float] = field(default_factory=dict)


def _measure(src: str | Path) -> dict[str, float]:
    import numpy as np
    from PIL import Image, ImageFilter, ImageOps

    img = ImageOps.exif_transpose(Image.open(src))
    alpha = 0.0
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        a = np.asarray(img.convert("RGBA").getchannel("A").resize((128, 128)))
        alpha = float((a < 250).mean())
    rgb = img.convert("RGB")
    small = rgb.copy()
    small.thumbnail((160, 160))
    q = np.asarray(small.point(lambda v: v >> 3)).reshape(-1, 3).astype(np.int32)
    colours = float(len(np.unique(q[:, 0] * 1024 + q[:, 1] * 32 + q[:, 2])))
    arr = np.asarray(small, dtype=np.float32)
    sat = float((arr.max(axis=2) - arr.min(axis=2)).mean())
    mid = rgb.copy()
    mid.thumbnail((384, 384))
    g = mid.convert("L")
    closed = g.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))
    blackhat = np.asarray(closed, dtype=np.float32) - np.asarray(g, dtype=np.float32)
    lines = float((blackhat > 45).mean() * 100.0)
    a = np.asarray(g, dtype=np.float32)
    med = np.asarray(g.filter(ImageFilter.MedianFilter(3)), dtype=np.float32)
    res = np.abs(a - med)[1:-1, 1:-1]
    lap = np.abs(4 * a[1:-1, 1:-1] - a[:-2, 1:-1] - a[2:, 1:-1] - a[1:-1, :-2] - a[1:-1, 2:])
    smooth = lap < 20
    noise = float(res[smooth].mean()) if smooth.any() else 0.0
    return {"alpha": round(alpha, 4), "colours": colours, "saturation": round(sat, 2), "lines_pct": round(lines, 3), "noise": round(noise, 4)}


def classify_reference(src: str | Path) -> RefKind:
    """'photo' or 'avatar' with a confidence flag. Adds an inked-line + low-noise test to the colour count, so painted
    references (gradient skies, grass, soft shading) that have too many colours for the old test still read as
    drawings. Measured on our refs (Oct 19): a painted cartoon ref lines 5.1 % / noise 0.42 -> avatar; real
    photos 0.4-6.4 % / noise 0.21-1.40, so lines alone are NOT enough and noise rules out the busy photos. Real photos
    that are clean and low-noise can still look 'unsure': the UI then asks instead of guessing."""
    s = _measure(src)
    if s["alpha"] >= 0.02:
        return RefKind("avatar", True, "transparent background (cut-out character)", s)
    if s["saturation"] < GREY_SATURATION_MAX:  # black-and-white / desaturated: colour counts mean nothing here,
        # and our own refs overlap (a B&W video frame 0.35, a noir anime drawing 0.63), so only the extremes are sure
        if s["noise"] < GREY_NOISE_DRAWN_MAX:
            return RefKind("avatar", True, "clean, noise-free greyscale image (drawing)", s)
        if s["noise"] >= GREY_NOISE_PHOTO_MIN:
            return RefKind("photo", True, "greyscale image with heavy camera-like texture (photo)", s)
        lean = "avatar" if s["noise"] < GREY_NOISE_LEAN else "photo"
        return RefKind(lean, False, "greyscale image: could be a photo or a drawing", s)
    if s["colours"] < COLOURS_DRAWN_MAX:
        return RefKind("avatar", True, "few distinct colours (flat drawing)", s)
    if s["lines_pct"] >= LINE_DRAWN_MIN and s["noise"] < NOISE_PHOTO_MIN:
        return RefKind("avatar", s["lines_pct"] >= 4.0, "inked outlines with clean, noise-free fills (drawing)", s)
    if s["noise"] >= NOISE_PHOTO_MIN or s["colours"] >= COLOURS_UNSURE_MAX:
        return RefKind("photo", True, "camera-like texture and many colours (photo)", s)
    return RefKind("photo", False, "could be a clean photo or a soft painting", s)


def hint_text(kind: RefKind) -> str:
    """One line for the UI under the photo/avatar choice, or '' when MC is sure."""
    if kind.confident:
        return ""
    if kind.label == "avatar":
        return ("mc thinks this is a drawing (not sure). if it's a real photo of you, pick **me (photo)**.")
    return ("mc isn't sure if this is a photo or a painting. if it's a drawing, pick **my avatar** so it stays drawn.")
