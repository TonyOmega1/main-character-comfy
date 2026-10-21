"""The MC pipeline: your photo (+ voice) -> you, inside a meme, ready to post.

Headline modes:

* ``meme``  **Meme Recreate**: pick a format recipe (memes.py); FLUX.2 klein puts
  you in the scene, Wan 2.2 I2V animates it (or InfiniteTalk lip-syncs the
  meme line in your voice), the finisher adds top/bottom meme captions + effect.
* ``swap``  **Swap Me In**: bring a clip you have the rights to; Wan2.2 Animate
  replaces the person in it with you (SAM 3.1 mask + SDPose skeleton, all core
  ComfyUI nodes), original audio kept. Up to 30 s, chunked.
* ``vibe``  **Copy the Vibe**: bring a reference clip (up to 30 s); only its
  motion (SDPose skeleton + face crops) is used. FLUX.2 klein puts you in a NEW
  scene from your prompt, Wan2.2 Animate "move" mode makes you perform the
  motion there. No pixels of the reference footage end up in the output.

Hardware tiers (tiers.py): **Lite** (8-12 GB) swaps meme motion to Wan2.2 TI2V 5B
at 480p / ~3 s and gates the 16 GB modes with a friendly message, **Standard**
(16 GB, the reference RTX 5080) is the default, **Max** (24 GB+) renders bigger
and longer. Every ETA is per tier and labelled as an estimate.

Every run also writes ``mc_recipe.json`` (format, prompts, models, seed,
settings) next to the output so it can be remixed (``cli --remix``).

Talking-avatar modes (the original MC engine):

Stages (each model step is its own Comfy SDK job, so the UI can show per-stage
progress and preview every intermediate):

1. avatar     FLUX.2 klein 4B reference edit / text-to-image -> 9:16 portrait
2. voice      Chatterbox TTS cloning the user's voice (or built-in voice);
              dialogue mode voices every turn with that character's voice
3. revoice    optional Chatterbox VC into a character voice
4. lipsync    Wan2.1 I2V 14B + InfiniteTalk (single or two speakers), long
              scripts split into segments and stitched with crossfades
5. smooth     optional RIFE x2 frame interpolation (core ComfyUI nodes)
6. finish     local ffmpeg: 1080x1920 lanczos, burned-in captions, watermark

Modes: ``talk`` (one presenter), ``dialogue`` (two-character skit / podcast),
``sing`` (lip-sync a song: vocals stem drives the mouth, full mix is the audio).
"""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from PIL import Image

from . import dialogue, framing, graphs, longform, media, memes, shots, stylelock, system, tiers
from .backends import Backend, StageResult
from .config import Settings
from .errors import Cancelled
from .presets import DEFAULT_STYLE, STYLES, motion_prompt

STAGES = ["prepare", "avatar", "voice", "revoice", "lipsync", "motion", "swap", "vibe", "smooth", "finish"]
STAGE_LABELS = {
    "prepare": "Prep inputs",
    "avatar": "Avatar · FLUX.2 klein",
    "voice": "Voice · Chatterbox",
    "revoice": "Voice changer · Chatterbox VC",
    "lipsync": "Lip sync · InfiniteTalk",
    "motion": "Meme motion · Wan 2.2 I2V",
    "swap": "Swap me in · Wan2.2 Animate",
    "vibe": "Copy the vibe · Wan2.2 Animate (move)",
    "smooth": "Smooth motion · RIFE",
    "finish": "Captions + 9:16 export",
}
MODE_STAGES = {
    "meme": ["prepare", "avatar", "voice", "lipsync", "motion", "finish"],
    "swap": ["prepare", "avatar", "swap", "revoice", "finish"],
    "vibe": ["prepare", "avatar", "vibe", "revoice", "finish"],
}
MODES = {"meme", "swap", "vibe", "talk", "dialogue", "sing"}


def stages_for(mode: str) -> list[str]:
    return MODE_STAGES.get(mode, ["prepare", "avatar", "voice", "revoice", "lipsync", "smooth", "finish"])


# Timings on the reference RTX 5080 16 GB (= the Standard tier; per-tier values live in tiers.py).
# MEASURED on the reference PC (RTX 5080 16 GB), Oct 16 2026 (ComfyUI v0.41, default flags); they drive the "about N min left" countdown.
SCENE_SECONDS = 12  # FLUX.2 klein 4B edit, 4 steps 1088x1920: 10-12 s warm, 21-23 s right after a Wan job (reload)
MEME_MOTION_SECONDS = 115  # Wan2.2 I2V 14B Q5 x2 experts, 81 frames 480x832, 4 steps: 105-126 s incl. model swap
# Wan2.2 Animate Q4_K_M, one 77-frame window at 480x832, 6 steps, LightX2V: measured 188 s for a cold 1-window
# Copy the Vibe job = ~40 s load + ~18 s RT-DETR/SDPose prep + 120 s sampling (20 s/step) + ~10 s decode.
# 30 s clip = 480 frames = 7 windows in 3 chunks -> ~17 min (multi-window runs not yet measured).
ANIMATE_SECONDS_PER_WINDOW = 130
ANIMATE_PREP_SECONDS = {"replace": 25, "move": 18}  # per chunk: SAM 3.1 (replace only, estimate) + RT-DETR + SDPose
ANIMATE_LOAD_SECONDS = 40  # first chunk: loading the 11.5 GB Animate GGUF + CLIP Vision (measured)
SWAP_SECONDS_PER_WINDOW = ANIMATE_SECONDS_PER_WINDOW  # backwards-compatible name
# SCAIL-2 group swap (mxfp8 14B, 6 steps, 576x1024, 81 frames, 3 people + 3 extra views): 351 s incl. load,
# 14.8 GB peak; SeedVR2 7B int8 576x1024 -> 1080x1920, 81 frames: 190 s, 15.0 GB peak (RTX 5080, Oct 18 2026).
SCAIL_GROUP_SECONDS = 351
SEEDVR2_SECONDS_PER_FRAME = 190 / 81
ETA_SPREAD = (0.85, 1.15)  # shown as a range: these are estimates until measured on the PC

# Reference timing (RTX 5080 16 GB, Q5_K_M, 480x832, 6 steps): 212 s for 81+49 window frames.
SECONDS_PER_WINDOW_FRAME = 212 / (81 + 49)

from .safety import BLOCKLIST as _BLOCKLIST  # noqa: E402


class InputError(ValueError):
    """User-facing validation error."""


CLIP_AUDIO = ("original", "mute", "my_voice")
SUBJECTS = ("auto", "photo", "avatar")
LIPSYNC_AVATAR_NOTE = ("Heads-up: lip sync is tuned on human-like mouths, so a cartoon, mascot or non-human "
                       "character may mouth the words more loosely. The voice and timing stay exact.")


def resolve_subject(req: "Request", src: Path | None = None) -> str:
    """'photo' or 'avatar' for this request (``auto`` guesses from the uploaded image). With the style lock on, the
    guess uses stylelock.classify_reference (also catches painted drawings with many colours); off, it stays the
    original colour-count guess so existing results don't change."""
    if req.subject in ("photo", "avatar"):
        return req.subject
    src = src or req.photo
    if not src:
        return "photo"
    try:
        if stylelock.normalize(getattr(req, "style_lock", "off")) != "off":
            return stylelock.classify_reference(src).label
        return "avatar" if media.looks_like_avatar(src) else "photo"
    except Exception:  # unreadable image: the prep step reports it properly
        return "photo"


FACE_CHOICES = ("auto", "on", "off")
NONHUMAN_FACE_DRIVE = "blank"  # what 'off' / auto-non-human uses. Robot A/B on an RTX 5080 (Oct 17, 2026):
# clip = human head, facepose = human face, ref = white mask face, none = closest head but black
# leftover blobs around the body in every frame, blank = robot head (antenna, round eyes), clean frames.


def face_choice(req: "Request", slot: int = 0) -> str:
    return (req.face[slot] if slot < len(req.face) and req.face[slot] else "auto")


def resolve_face_drive(choice: str, ref: Path | None) -> tuple[str, str]:
    """(Animate face_drive, human-readable why) for one person's reference."""
    if choice in graphs.FACE_DRIVES:
        return choice, f"face drive set to {choice}"
    if choice == "on":
        return "clip", "copying the dancer's face (you chose 'on')"
    if choice == "off":
        return NONHUMAN_FACE_DRIVE, "not copying the dancer's face (you chose 'off')"
    if ref is None:
        return "clip", "no reference to check"
    try:
        nonhuman, why = media.face_check(ref)
    except Exception as exc:  # unreadable image / detector problem: keep the normal behaviour
        return "clip", f"face check failed ({exc})"
    if nonhuman:
        return NONHUMAN_FACE_DRIVE, f"non-human reference ({why}): not copying the dancer's face"
    return "clip", why


def use_relight(req: "Request", subject: str) -> bool:
    """Relight LoRA on/off: the user's explicit choice, else on for photos and off for avatars."""
    return bool(req.relight) if req.relight is not None else subject != "avatar"


PROPS_NEGATIVE = ("backpack, bag, tire, barrel, props, accessories, weapon, gear strapped to the body, "
                  "objects attached to the back, extra objects on the character")
MASK_GROW = 12  # px the tracked person mask is widened by
MASK_GROW_PROPS = 30  # "remove props": the original character's accessories fall inside the repainted area


def keep_avatar_look(req: "Request", subject: str) -> bool:
    """Avatar mode keeps the upload's own art style unless the user picked a different Look on purpose."""
    return subject == "avatar" and req.style in ("none", "real", DEFAULT_STYLE)


@dataclass
class Request:
    mode: str = "talk"  # meme | swap | talk | dialogue | sing
    script: str = ""  # talk: what to say; dialogue: "Name: line" per turn; sing: lyrics (captions only)
    photo: Path | None = None
    photo_b: Path | None = None  # dialogue: second person (optional)
    character: str = ""  # describe the character(s) instead of uploading a photo
    style: str = DEFAULT_STYLE
    style_extra: str = ""
    avatar: Path | None = None  # ready-made portrait: skips stage 1
    voice_mode: str = "clone"  # clone | builtin | my_recording
    voice_sample: Path | None = None
    voice_sample_b: Path | None = None  # dialogue: speaker B's sample
    recording: Path | None = None  # talk mode: you speak the script yourself
    character_voice: Path | None = None  # voice-changer target (talk mode)
    song: Path | None = None  # sing mode: full mix
    vocals: Path | None = None  # sing mode: vocals-only stem (e.g. Suno "stems" export)
    song_start: float = 0.0
    song_seconds: float = 30.0
    expressiveness: float = 0.55
    captions: bool = True
    smooth: str = "off"  # off | rife | ffmpeg
    hd: bool = True
    watermark: bool = True
    seed: int = 2026
    tier: str = tiers.DEFAULT_TIER  # lite | standard | max (see tiers.py; UI/CLI resolve it from your VRAM)
    allow_heavy: bool = False  # Lite: try the 16 GB modes anyway (experimental: smaller quants, may OOM)
    steps: int | None = None  # default: tier
    segment_seconds: float | None = None  # default: tier
    seam: str = "anchor"  # anchor (re-seed each segment from the avatar) | continue (from last frame)
    ext_frames: int = 81
    consent: bool = False
    # Meme Recreate
    meme: str = memes.DEFAULT_RECIPE  # recipe key, or "custom"
    meme_top: str | None = None  # None = recipe default
    meme_bottom: str | None = None
    say_line: bool = False  # lip-sync the meme line in your voice instead of silent motion
    line: str = ""  # override the recipe's line
    custom_scene: str = ""
    custom_motion: str = ""
    effect: str | None = None  # None = recipe default
    # Swap Me In
    clip: Path | None = None
    clip_start: float = 0.0
    clip_seconds: float = 10.0
    swap_prompt: str = ""
    # "match scene lighting" (Wan-Animate relight LoRA): None = auto (on for photos, OFF for avatars, because
    # relighting pulls a drawn character toward the clip's rendered look)
    relight: bool | None = None
    # remove props: widen the replaced area and steer away from the original character's accessories
    # (backpacks, weapons, gear) so they aren't redrawn on you
    remove_props: bool = False
    # Copy the Vibe
    scene_prompt: str = ""  # the NEW scene, e.g. "luxury hotel hallway at night"
    keep_audio: bool = False  # deprecated alias: True forces clip_audio="original"
    # Swap Me In / Copy the Vibe soundtrack: original (default: the uploaded clip's own audio, synced and trimmed)
    # | mute | my_voice (the clip's audio re-voiced into voice_sample with Chatterbox VC)
    clip_audio: str = "original"
    # Who is in the upload: photo (a real face) | avatar (cartoon / anime / 3D / VTuber / pixel art / mascot /
    # non-human: keeps its own art style) | auto (guess from the image)
    subject: str = "auto"
    # Multi-character Swap Me In: one reference per person in the clip, numbered left to right on the first
    # frame (None = leave that person unchanged). Empty = classic single-person swap of the main person.
    people: list = field(default_factory=list)
    # Swap Me In / Copy the Vibe render size: auto (tier default; 832x480 on Standard) | 480p | 720p.
    # Finishing still upscales to 1080p.
    resolution: str = "auto"
    # Swap Me In: erase these props from the clip first with VOID (e.g. "backpack"); empty = off. Experimental.
    erase_props: str = ""
    # Swap Me In face drive, one per person (single swap: face[0]): auto (non-human references keep their own
    # head, humans copy the dancer's expressions) | on (copy the dancer's face) | off (keep the reference's own
    # head). Advanced: clip | none | blank | ref (graphs.FACE_DRIVES) pick the exact Animate face input.
    face: list = field(default_factory=list)
    # Multi-character Swap Me In engine: auto (SCAIL-2 one pass when you swap EVERYONE, 2-6 people, in a clip of up
    # to ~5 s on Standard/Max; else chained Animate, which keeps people you leave alone pixel-identical) | scail | chain (one Animate pass per person, the pre-Oct-18 path)
    group_engine: str = "auto"
    # Multi-character Swap Me In: a short look description per person slot (same numbering as ``people``), e.g.
    # "anime girl, short lavender hair, red bomber jacket". SCAIL-2 needs these to copy the references exactly
    # (without text it drew generic people in the A/B), so auto only picks SCAIL-2 when every swapped person has one.
    people_desc: list = field(default_factory=list)
    # Upscale the swapped take before finishing: auto (SeedVR2 for SCAIL-2 group takes, which render at 576p;
    # lanczos otherwise) | seedvr2 (always, Standard/Max) | lanczos (fastest)
    upscale: str = "auto"
    # Render time (Oct 19, none of these change the look: same models, steps, sizes and seeds).
    # Multi-character: shots SCAIL-2 is known to fail (an extreme close-up, feet/legs only, a close shot with no
    # face; see framing.ROUTE_TO_ANIMATE) skip it and go straight to chained Animate. Normal close-ups still try
    # SCAIL-2 first. Only applies to group_engine "auto".
    route_framing: bool = True
    # auto: framing QC of a SCAIL-2 take (people re-found at the filmed size and place) BEFORE SeedVR2; a failed
    # take falls back to chained Animate and never gets upscaled | off
    qc: str = "auto"
    # Stop at the silent take (no SeedVR2, no export) and leave _work/pending_upscale.json; run
    # Pipeline.finish_pending (CLI --finish-pending) on the takes that pass review: SeedVR2 back to back, then export.
    defer_upscale: bool = False
    # Retake: a previous run's folder. Its prepared clip, VOID erase, people scan and soundtrack are reused
    # byte for byte (when the clip and prep settings match), so only the seed differs and ComfyUI can keep the
    # Animate mask / pose / face prep cached between consecutive takes.
    reuse_prep: Path | None = None
    # Style lock for drawn references (stylelock.py): off (default, today's behaviour) | auto (drawn refs only) | on.
    # PENDING A REAL TEST: relight off, CFG 2.5 + anti-photo negative, SeedVR2 blended 40 %, tighter paste-back edge.
    style_lock: str = stylelock.DEFAULT_MODE
    # Multi-character: whose moves each person follows, same numbering as ``people``. "" / "self" = their own (the
    # default); "person:N" = person N of the same clip; or a path to another clip (its main figure). Pending a real test.
    people_driver: list = field(default_factory=list)
    # Hard cuts: auto (detect and report; split when the clip has cuts and people are swapped) | on | off.
    # Splitting swaps every shot on its own and keeps who-is-who across cuts by appearance (shots.py).
    split_cuts: str = "auto"
    # Long clips (> 81 frames) in the SCAIL-2 group pass: one SCAIL-2 prompt per 81-frame chunk, chained on the last 5
    # frames (the official extend recipe), instead of falling back to one Animate pass per person. Off until the e2e
    # test (a 3-person test clip lost one person ~10.5 s in on the chained-Animate fallback).
    scail_chunks: bool = False
    # Extra clockwise turn of the clip (0 / 90 / 180 / 270) for phone clips whose rotation tag is missing or wrong;
    # a correct tag is already honoured automatically.
    rotate: int = 0
    # Face close-up from a full-body reference for close-up shots (framing.closeup_reference): off (default) | auto.
    # One-pass-per-person path only. Pending a real test.
    ref_closeup: str = "off"
    era: str = "classic"  # Meme Era look: classic | deep_fried | vaporwave | early_web | tiktok
    meme_frames: int | None = None  # None = format default (81 = ~5 s); 49 = ~3 s


@dataclass
class Event:
    stage: str
    status: str  # start | progress | done | skipped | error | complete
    fraction: float = 0.0
    message: str = ""
    artifact: Path | None = None
    data: dict[str, Any] = field(default_factory=dict)


def estimate_lipsync_seconds(audio_seconds: float, segment_seconds: float = 15.0, ext_frames: int = 81, fps: int = 25,
                             tier: str = tiers.DEFAULT_TIER) -> float:
    """Rough lip-sync render time (measured on the RTX 5080, scaled per tier: an ESTIMATE off Standard)."""
    total = 0.0
    remaining = audio_seconds
    while remaining > 0:
        seg = min(segment_seconds, remaining)
        plan = graphs.plan_windows(seg, fps=fps, ext_frames=ext_frames)
        total += (graphs.BASE_FRAMES + plan["extends"] * ext_frames) * SECONDS_PER_WINDOW_FRAME
        remaining -= seg
    return total * tiers.get(tier).lipsync_factor


def estimate_animate_seconds(frames: int, mode: str = "replace", tier: str = tiers.DEFAULT_TIER,
                             resolution: str = "auto") -> float:
    """Rough Wan2.2 Animate render time for ``frames`` frames (all chunks) on the tier's reference card.
    Non-default resolutions scale with the pixel count (720p: an ESTIMATE until measured)."""
    t = tiers.get(tier)
    segs = graphs.plan_animate_segments(frames)
    px = tiers.pixel_scale(t, resolution)
    scale = t.animate_window_seconds / ANIMATE_SECONDS_PER_WINDOW * px
    return (ANIMATE_LOAD_SECONDS + sum(x.windows for x in segs) * t.animate_window_seconds * px
            + len(segs) * ANIMATE_PREP_SECONDS.get(mode, 20) * scale)


VOID_SECONDS_PER_FRAME = 3.75  # VOID 2-pass at 672x384: MEASURED on the RTX 5080 (80 frames in ~6 min incl. loads)


def estimate_erase_seconds(frames: int) -> float:
    return 60 + frames * VOID_SECONDS_PER_FRAME


PEOPLE_SCAN_SECONDS = 15
# SAM 3.1 'person' confidence for the people scan (0.5 found both characters of a stylized 3D duo once the
# prompt asked for several detections; see graphs.people_scan_graph). order_people() drops duplicate masks.
PEOPLE_SCAN_THRESHOLD = float(os.environ.get("MC_PEOPLE_THRESHOLD", "0.5"))  # SAM 3.1 detect on one frame incl. load (ESTIMATE)


def estimate_people_seconds(frames: int, swapped: int, tier: str = tiers.DEFAULT_TIER,
                            resolution: str = "auto") -> float:
    """Multi-character Swap Me In: one full Animate pass per swapped person, plus a people scan per chunk."""
    segs = graphs.plan_animate_segments(frames)
    return max(1, swapped) * (estimate_animate_seconds(frames, "replace", tier, resolution)
                              + PEOPLE_SCAN_SECONDS * len(segs))


def estimate_scene_seconds(tier: str = tiers.DEFAULT_TIER) -> float:
    """FLUX.2 klein scene / avatar (warm), per tier. ESTIMATE."""
    return tiers.get(tier).scene_seconds


def estimate_motion_seconds(frames: int, tier: str = tiers.DEFAULT_TIER) -> float:
    """Silent meme motion (Wan2.2 I2V A14B, or TI2V 5B on Lite), per tier. ESTIMATE."""
    t = tiers.get(tier)
    return t.motion_seconds * frames / t.motion_ref_frames


def tier_record(tier: str) -> dict[str, Any]:
    d = asdict(tiers.get(tier))
    d["gated"] = sorted(d["gated"])
    return d


def eta_range_text(seconds: float) -> str:
    """Honest range for a long estimate, e.g. '~15-21 min'."""
    lo, hi = (seconds * f for f in ETA_SPREAD)
    if hi < 90:
        return f"~{lo:.0f}-{hi:.0f} s"
    return f"~{max(1, round(lo / 60))}-{max(1, round(hi / 60))} min"


def check_request(req: Request, settings: Settings) -> None:
    if req.mode not in MODES:
        raise InputError(f"unknown mode {req.mode!r}")
    try:
        req.tier = tiers.normalize(req.tier)
    except ValueError as exc:
        raise InputError(str(exc)) from None
    blocked = tiers.gate(req.tier, req.mode, say_line=req.say_line, allow_heavy=req.allow_heavy)
    if blocked:
        raise InputError(blocked)
    if req.style not in STYLES:
        raise InputError(f"unknown style {req.style!r}")
    if req.keep_audio:
        req.clip_audio = "original"
    if req.clip_audio not in CLIP_AUDIO:
        raise InputError(f"unknown clip audio option {req.clip_audio!r} (use {', '.join(CLIP_AUDIO)})")
    if req.subject not in SUBJECTS:
        raise InputError(f"unknown subject {req.subject!r} (use {', '.join(SUBJECTS)})")
    if req.ref_closeup not in ("off", "auto"):
        raise InputError(f"ref_closeup is off or auto (got {req.ref_closeup!r})")
    if int(req.rotate or 0) % 360 not in (0, 90, 180, 270):
        raise InputError(f"rotate must be 0, 90, 180 or 270 degrees (got {req.rotate})")
    if req.mode in ("swap", "vibe") and req.clip_audio == "my_voice" and not req.voice_sample:
        raise InputError("To re-voice the clip's audio as you, upload a 10-30 s voice sample (or keep the original "
                         "audio / mute it).")
    if req.people and req.mode != "swap":
        raise InputError("Per-person references are for Swap Me In.")
    if len(req.people) > tiers.PEOPLE_HARD_CAP:
        raise InputError(f"Swap Me In takes up to {tiers.PEOPLE_HARD_CAP} references, one per person "
                         f"(you gave {len(req.people)}).")
    for i, f in enumerate(req.face or []):
        if f and f not in FACE_CHOICES + graphs.FACE_DRIVES:
            raise InputError(f"face for person {i + 1}: {f!r} (use auto, on or off)")
    if len(req.face or []) > tiers.PEOPLE_HARD_CAP:
        raise InputError(f"face takes up to {tiers.PEOPLE_HARD_CAP} values, one per person")
    try:
        req.resolution = tiers.normalize_resolution(req.resolution)
    except ValueError as exc:
        raise InputError(str(exc)) from None
    if req.mode in ("swap", "vibe"):
        blocked = tiers.resolution_gate(req.tier, req.resolution, req.allow_heavy)
        if blocked:
            raise InputError(blocked)
    req.erase_props = (req.erase_props or "").strip()
    if req.erase_props and req.mode != "swap":
        raise InputError("'Erase props first' is for Swap Me In.")
    if len(req.erase_props) > 80:
        raise InputError("Keep 'erase props' short: a few words, e.g. 'backpack' or 'backpack, sword'.")
    if req.smooth not in {"off", "rife", "ffmpeg"}:
        raise InputError(f"unknown smoothing mode {req.smooth!r}")
    if req.seam not in {"anchor", "continue"}:
        raise InputError(f"unknown seam mode {req.seam!r}")
    if req.mode == "meme":
        if req.meme == "custom":
            if not (req.custom_scene.strip() and req.custom_motion.strip()):
                raise InputError("Describe your custom format: the scene and what you do in it.")
        elif req.meme not in memes.RECIPES:
            raise InputError(f"unknown meme format {req.meme!r}")
        if req.era not in media.ERAS:
            raise InputError(f"unknown meme era {req.era!r}")
        if req.effect is not None and req.effect not in memes.EFFECTS:
            raise InputError(f"unknown effect {req.effect!r}")
        if not req.photo and not req.avatar:
            raise InputError("Upload a selfie or your avatar: you're the star of the meme.")
        if req.say_line:
            line = req.line.strip() or (memes.RECIPES[req.meme].line if req.meme in memes.RECIPES else "")
            if req.voice_mode != "my_recording" and not line:
                raise InputError("This format has no line: type one, or turn off 'say a line'.")
            if len(line) > 200:
                raise InputError("Keep the meme line under 200 characters (~12 s).")
            if req.voice_mode == "clone" and not req.voice_sample:
                raise InputError("Upload a 10-30 s voice sample, or pick the built-in voice.")
            if req.voice_mode == "my_recording" and not req.recording:
                raise InputError("Upload your recording of the line.")
    elif req.mode == "swap":
        if not req.clip:
            raise InputError("Upload a clip you have the rights to (your own footage or licensed).")
        if req.people:
            swapped = [x for x in req.people if x]
            if not swapped:
                raise InputError("Pick a photo or avatar for at least one person (the others stay unchanged).")
            blocked = tiers.people_gate(req.tier, len(swapped), req.allow_heavy)
            if blocked:
                raise InputError(blocked)
        elif not (req.photo or req.avatar):
            raise InputError("Upload a selfie or your avatar to swap in.")
        cap = tiers.animate_cap(req.tier, settings.max_swap_seconds)
        if not 1 <= req.clip_seconds <= cap:
            raise InputError(f"Swap clips are 1-{cap:.0f} s on MC {tiers.get(req.tier).label}.")
        if not req.consent:
            raise InputError("Please tick the box: it's you or your own avatar (or one you have the rights to), and "
                             "you have the rights to the clip you uploaded.")
    elif req.mode == "vibe":
        if not req.clip:
            raise InputError("Upload a reference clip you have the rights to (your own, or generated). "
                             "Only its motion is used.")
        if not (req.photo or req.avatar):
            raise InputError("Upload a selfie or your avatar: you're the one doing the moves.")
        cap = tiers.animate_cap(req.tier, settings.max_vibe_seconds)
        if not 1 <= req.clip_seconds <= cap:
            raise InputError(f"Reference clips are 1-{cap:.0f} s on MC {tiers.get(req.tier).label}.")
        if len(req.scene_prompt) > 300:
            raise InputError("Keep the scene description under 300 characters.")
        if not req.consent:
            raise InputError("Please tick the box: it's you or your own avatar (or one you have the rights to), and "
                             "you have the rights to the reference clip.")
    elif req.mode == "talk":
        if req.voice_mode not in {"clone", "builtin", "my_recording"}:
            raise InputError(f"unknown voice mode {req.voice_mode!r}")
        if req.voice_mode != "my_recording":
            if not req.script.strip():
                raise InputError("Type a script for your avatar to say.")
            if len(req.script) > settings.max_script_chars * 2:
                raise InputError(f"Keep the script under {settings.max_script_chars * 2} characters (~80 s).")
        if req.voice_mode == "clone" and not req.voice_sample:
            raise InputError("Upload a 10-30 s voice sample, or pick the built-in voice.")
        if req.voice_mode == "my_recording" and not req.recording:
            raise InputError("Upload your recording of the script.")
    elif req.mode == "dialogue":
        try:
            dialogue.parse_dialogue(req.script)
        except ValueError as exc:
            raise InputError(str(exc)) from exc
    elif req.mode == "sing":
        if not (req.song or req.vocals):
            raise InputError("Upload a song (and ideally its vocals-only stem).")
        if not 3 <= req.song_seconds <= 90:
            raise InputError("Pick a song section between 3 and 90 seconds.")
    if not (req.photo or req.avatar or req.character.strip() or any(req.people)):
        raise InputError("Upload a selfie or your avatar, or describe a character.")
    uploads = [req.photo, req.photo_b, req.voice_sample, req.voice_sample_b, req.recording, req.character_voice,
               req.song, req.vocals, req.clip, *req.people]
    if any(uploads) and not req.consent:
        raise InputError(
            "Please tick the consent box: this is your face/voice or your own avatar, or you have permission / the "
            "rights to use it (and the rights to any song you upload)."
        )
    free_text = " ".join(str(x or "") for x in (req.script, req.character, req.style_extra, req.meme_top,
                                                req.meme_bottom, req.line, req.custom_scene, req.custom_motion,
                                                req.swap_prompt, req.scene_prompt))
    if _BLOCKLIST.search(free_text):
        raise InputError("MC is for SFW content only.")


def tts_chunks(script: str, max_chars: int = 280) -> list[str]:
    """Group sentences into passages of at most ``max_chars`` for TTS."""
    sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", script.strip()) if x.strip()]
    chunks: list[str] = []
    cur = ""
    for sent in sentences:
        if cur and len(cur) + 1 + len(sent) > max_chars:
            chunks.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        chunks.append(cur)
    return chunks or [script.strip()]


def _jsonable(v: Any) -> Any:
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:40].strip("-") or "clip"


HASHTAG = "#OpenSourceHiggsfield"
# Suggested post text leads with the group-chat hook and record-your-own-line (not hardware or speed claims).
POST_HOOK = "put the whole group chat in the meme. swap your homies in, record your own line. free, local, open weights."


def parse_driver(spec: Any, n_people: int, slot: int) -> tuple[str, Any]:
    """One ``Request.people_driver`` entry -> ("self", None) | ("person", zero-based index) | ("clip", Path).
    Raises InputError for a person number outside the clip or a missing clip file."""
    s = str(spec or "").strip()
    if not s or s.lower() in ("self", "own", "none", "-"):
        return "self", None
    m = re.fullmatch(r"(?:person|p)\s*[:#]?\s*(\d+)", s, flags=re.I) or re.fullmatch(r"(\d+)", s)
    if m:
        k = int(m.group(1)) - 1
        if not 0 <= k < n_people:
            raise InputError(f"Person {slot + 1} is set to follow person {k + 1}, but the clip has {n_people} "
                             f"{'person' if n_people == 1 else 'people'}.")
        return ("self", None) if k == slot else ("person", k)
    path = Path(s)
    if not path.is_file():
        raise InputError(f"Person {slot + 1}'s moves clip wasn't found: {s}")
    return "clip", path


def driver_label(kind: str, value: Any) -> str:
    return {"self": "own moves", "person": f"follows person {int(value) + 1 if value is not None else '?'}",
            "clip": f"follows {Path(str(value)).name}"}.get(kind, "own moves")


def slot_names(n: int) -> list[str]:
    """Human names for people numbered left to right."""
    return {1: ["solo"], 2: ["left", "right"], 3: ["left", "middle", "right"]}.get(n, [f"p{i + 1}" for i in range(n)])


def _stem(p: Any) -> str:
    return _slug(Path(str(p)).stem) if p else ""


def who_label(req: "Request") -> str:
    """Readable 'who' part of an output name: the uploaded photo/avatar's file name, plus which slot(s) for
    multi-character swaps (e.g. ``noir-avatar-left`` or ``noir-avatar-both``)."""
    if req.people:
        names = slot_names(len(req.people))
        used = [(names[i], _stem(x)) for i, x in enumerate(req.people) if x]
        stems = {st for _, st in used}
        if len(stems) == 1:
            st = stems.pop()
            if len(used) == len(req.people) and len(used) > 1:
                return f"{st}-{'both' if len(used) == 2 else 'all'}"
            return f"{st}-{'-'.join(n for n, _ in used)}"
        return "_".join(f"{st}-{n}" for n, st in used)[:60]
    return _stem(req.photo) or _stem(req.avatar) or _slug(req.character)[:20] if (req.photo or req.avatar or
                                                                                 req.character) else "me"


def readable_name(req: "Request", template: str, when: datetime | None = None, *, mode: str | None = None,
                  who: str | None = None) -> str:
    """``mode_template_who_YYYY-MM-DD_HHMMSS``: every output folder and file says what it is. A leading
    ``template_`` in an uploaded clip's file name is dropped (it's already the template slot)."""
    when = when or datetime.now()
    mode = mode or {"dialogue": "skit", "talk": "talk"}.get(req.mode, req.mode)
    template = re.sub(r"^templates?[-_ ]+", "", template or "", flags=re.I)
    parts = [mode, _slug(template) if template else "", who or who_label(req), when.strftime("%Y-%m-%d_%H%M%S")]
    return "_".join(p for p in parts if p)


def share_text(top: str, bottom: str) -> str:
    """Pre-filled post text (the user copies / edits it; MC never posts anything)."""
    cap = " ".join(x.strip() for x in (top, bottom) if x and x.strip())
    lead = f"{cap} 😂\n\n" if cap else ""
    return f"{lead}{POST_HOOK} {HASHTAG}"


# Output folder layout: <outputs>/<readable name>/ holds ONLY the results (final mp4 + gif, or the still);
# every intermediate (inputs, chunks, masks, previews, raw takes, project.json, mc_recipe.json) is written
# straight into its _work/ subfolder, so nobody mistakes an input chunk for the result.
WORK_SUBDIR = "_work"


def project_file(work: str | Path) -> Path:
    """Where a run's project.json lives (``_work/`` now; top level for runs made before the tidy-up)."""
    work = Path(work)
    new = work / WORK_SUBDIR / "project.json"
    return new if new.exists() or not (work / "project.json").exists() else work / "project.json"


def _exports(final: Path, work: Path) -> dict[str, str]:
    """Group-chat friendly extras: a small looping GIF next to the 9:16 MP4."""
    gif = media.make_gif(final, work / f"{final.stem}.gif")
    return {"mp4": str(final), "gif": str(gif)}


class Pipeline:
    def __init__(self, backend: Backend, settings: Settings) -> None:
        self.backend = backend
        self.settings = settings
        self._stop = False

    def cancel(self) -> bool:
        """Stop button: no new stages start, and the running ComfyUI job is cancelled."""
        self._stop = True
        stop = getattr(self.backend, "cancel", None)
        return bool(stop()) if stop else False

    # -- stage runner ----------------------------------------------------------
    def _run_stage(self, job_name: str, ui_stage: str, sg: graphs.StageGraph, files: dict[str, Path], work: Path,
                   scale: tuple[float, float] = (0.0, 1.0), prefix: str = "") -> Iterator[Event | StageResult]:
        """Run one backend job in a worker thread, yielding progress events.
        ``scale`` maps the job's 0..1 progress into a slice of the UI stage."""
        q: queue.Queue = queue.Queue()
        box: dict[str, Any] = {}
        lo, hi = scale

        def progress(frac: float, msg: str) -> None:
            f = -1.0 if frac < 0 else lo + (hi - lo) * frac
            q.put(Event(ui_stage, "progress", f, f"{prefix}{msg}"))

        def worker() -> None:
            try:
                box["result"] = self.backend.run_stage(job_name, sg, files, work, progress)
            except BaseException as exc:  # re-raised in the caller thread
                box["error"] = exc
            finally:
                q.put(None)

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        while (item := q.get()) is not None:
            yield item
        t.join()
        if "error" in box:
            raise box["error"]
        yield box["result"]

    def _job(self, job_name, ui_stage, sg, files, work, results, record, **kw) -> Iterator[Event]:
        if self._stop:
            raise Cancelled(f"stopped before {job_name}")
        prompts, used = graph_prompts_and_models(sg)
        record.setdefault("prompts", {})[job_name] = prompts
        record.setdefault("models", {})[job_name] = used
        for item in self._run_stage(job_name, ui_stage, sg, files, work, **kw):
            if isinstance(item, StageResult):
                results[job_name] = item
                record.setdefault("jobs", {})[job_name] = item.job_id  # extras callers may pass a bare record
            else:
                yield item

    # -- main entry --------------------------------------------------------------
    def run(self, req: Request, work: Path | None = None) -> Iterator[Event]:
        s = self.settings
        check_request(req, s)
        if req.mode == "meme":
            yield from self._run_meme(req, work)
            return
        if req.mode == "swap":
            yield from self._run_swap(req, work)
            return
        if req.mode == "vibe":
            yield from self._run_vibe(req, work)
            return
        tier = tiers.get(req.tier)
        m = tiers.models_for(tier, s.models)
        steps = req.steps or tier.steps
        seg_seconds = req.segment_seconds or tier.segment_seconds
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        top = work if work is not None else s.output_dir / readable_name(req, (req.script or req.character or "clip")[:40])
        work = top / WORK_SUBDIR
        (work / "inputs").mkdir(parents=True, exist_ok=True)
        results: dict[str, StageResult] = {}
        record: dict[str, Any] = {
            "request": {k: _jsonable(v) for k, v in asdict(req).items()},
            "backend": self.backend.name,
            "tier": tier_record(req.tier),
            "jobs": {},
            "started": stamp,
            "name": top.name,
            "top": str(top),
        }
        t_start = time.monotonic()
        two = req.mode == "dialogue"

        # 0. prepare --------------------------------------------------------------
        yield Event("prepare", "start", message="Preparing your inputs")
        files: dict[str, Path] = {}
        inp = work / "inputs"
        if req.photo:
            files["photo"] = media.prepare_photo(req.photo, inp / "photo.png")
        if req.photo_b:
            files["photo_b"] = media.prepare_photo(req.photo_b, inp / "photo_b.png")
        for key, src in (("voice_ref", req.voice_sample), ("voice_ref_b", req.voice_sample_b),
                         ("vc_target", req.character_voice)):
            if src and (req.mode != "talk" or req.voice_mode == "clone" or key == "vc_target"):
                dur = media.prepare_voice(src, inp / f"{key}.wav", s.max_voice_seconds)
                files[key] = inp / f"{key}.wav"
                if dur < 5:
                    yield Event("prepare", "progress", 0.5, f"{key}: only {dur:.1f}s of speech; 10-30 s clones better")
        if req.mode == "talk" and req.voice_mode == "my_recording" and req.recording:
            media.prepare_voice(req.recording, inp / "recording.wav", 90.0)
            files["recording"] = inp / "recording.wav"
        yield Event("prepare", "done", 1.0, "Inputs ready")

        # 1. avatar ------------------------------------------------------------------
        style = STYLES[req.style]
        if req.avatar:
            avatar = media.prepare_photo(req.avatar, work / "avatar.png")
            yield Event("avatar", "skipped", 1.0, "Using your ready-made avatar", artifact=avatar)
        elif req.photo and req.style == "none" and not req.photo_b:
            avatar = files["photo"]
            yield Event("avatar", "skipped", 1.0, "Keeping your photo as-is", artifact=avatar)
        else:
            n_refs = (1 if req.photo else 0) + (1 if req.photo_b else 0)
            if n_refs:
                prompt = style.edit_prompt if req.style != "none" else STYLES["real"].edit_prompt
                if two:
                    prompt = prompt.replace("the person in image 1", "the two people from the images" if n_refs == 2
                                            else "the person in image 1 and a second, matching character")
                    prompt += " Both people side by side facing the camera, podcast desk with two microphones."
            else:
                prompt = (style if req.style != "none" else STYLES["real"]).t2i_prompt.format(desc=req.character.strip())
                if two:
                    prompt += " Two people side by side facing the camera, podcast desk with two microphones."
            sg = graphs.avatar_graph(f"{prompt} {req.style_extra}".strip(), models=m, reference=n_refs, seed=req.seed,
                                     width=tier.scene_size[0], height=tier.scene_size[1], te_device=tier.te_device)
            yield Event("avatar", "start", message=f"Creating your {style.label} avatar",
                        data={"eta_seconds": estimate_scene_seconds(req.tier)})
            yield from self._job("avatar", "avatar", sg, files, work, results, record)
            avatar = results["avatar"].path
            yield Event("avatar", "done", 1.0, f"Avatar ready ({results['avatar'].seconds:.0f}s)", artifact=avatar)
        files["avatar"] = avatar

        # 2. voice --------------------------------------------------------------------
        captions: list[media.Caption] = []
        tracks: dict[str, Path] = {}
        final_audio: Path
        if req.mode == "talk":
            if req.voice_mode == "my_recording":
                speech = files["recording"]
                yield Event("voice", "skipped", 1.0, "Using your own recording", artifact=speech)
            else:
                clone = req.voice_mode == "clone"
                parts = tts_chunks(req.script)
                yield Event("voice", "start", message="Cloning your voice" if clone else "Using the built-in voice")
                if len(parts) == 1:
                    sg = graphs.tts_graph(parts[0], models=m, clone=clone, exaggeration=req.expressiveness, seed=req.seed)
                    yield from self._job("voice", "voice", sg, files, work, results, record)
                    speech = results["voice"].path
                else:  # Chatterbox is best on short passages: voice long scripts in sentence groups
                    pieces = []
                    for i, part in enumerate(parts):
                        sg = graphs.tts_graph(part, models=m, clone=clone, exaggeration=req.expressiveness, seed=req.seed)
                        yield from self._job(f"voice_{i:02d}", "voice", sg, files, work, results, record,
                                             scale=(i / len(parts), (i + 1) / len(parts)), prefix=f"part {i + 1}/{len(parts)}: ")
                        pieces.append(results[f"voice_{i:02d}"].path)
                    speech = dialogue.concat_audio(pieces, work / "speech_joined.wav", work / "tts_parts")
                yield Event("voice", "done", 1.0, f"Speech ready ({len(parts)} part(s))", artifact=speech)
            if req.character_voice:
                sg = graphs.voice_changer_graph(models=m)
                yield Event("revoice", "start", message="Switching to the character voice")
                yield from self._job("revoice", "revoice", sg, {"source": speech, "target": files["vc_target"]},
                                     work, results, record)
                speech = results["revoice"].path
                yield Event("revoice", "done", 1.0, "Character voice applied", artifact=speech)
            else:
                yield Event("revoice", "skipped", 1.0, "No character voice")
            speech_wav = media.to_wav(speech, work / "speech.wav")
            duration = media.media_duration(speech_wav)
            tracks = {"speech": speech_wav}
            final_audio = speech_wav
            if req.captions and req.script.strip():
                captions = media.time_captions(req.script, duration)

        elif req.mode == "dialogue":
            turns, names = dialogue.parse_dialogue(req.script)
            yield Event("voice", "start", message=f"Voicing {len(turns)} lines for {names['A']} and {names['B']}")
            clips: list[Path] = []
            for i, turn in enumerate(turns):
                ref_key = "voice_ref" if turn.speaker == "A" else "voice_ref_b"
                clone = ref_key in files
                sg = graphs.tts_graph(turn.text, models=m, clone=clone, seed=req.seed + i,
                                      exaggeration=req.expressiveness + (0.1 if turn.speaker == "B" and not clone else 0))
                job_files = {"voice_ref": files[ref_key]} if clone else {}
                yield from self._job(f"voice_{i:02d}", "voice", sg, job_files, work, results, record,
                                     scale=(i / len(turns), (i + 1) / len(turns)), prefix=f"{turn.name} line {i + 1}: ")
                clips.append(results[f"voice_{i:02d}"].path)
            built = dialogue.build_tracks(turns, clips, work / "dialogue")
            tracks = {"speech": built["a"], "speech_b": built["b"], "speech_mix": built["mix"]}
            final_audio = built["mix"]
            duration = built["duration"]
            captions = built["captions"] if req.captions else []
            yield Event("voice", "done", 1.0, f"Dialogue track ready ({duration:.1f}s)", artifact=final_audio)
            yield Event("revoice", "skipped", 1.0, "Not used in dialogue mode")

        else:  # sing
            yield Event("voice", "start", message="Cutting your song section")
            src_mix = req.song or req.vocals
            mix = longform.cut_audio(src_mix, work / "song_mix.wav", req.song_start, req.song_seconds)
            if req.vocals:
                vocals = longform.cut_audio(req.vocals, work / "song_vocals.wav", req.song_start, req.song_seconds)
                note = "Vocals stem drives the lips; full mix is the soundtrack"
            else:
                from . import stems

                sep = stems.separate_vocals(mix, work / "stems")
                if sep is not None:
                    vocals = sep
                    note = "Vocals separated locally (Demucs): they drive the lips; full mix is the soundtrack"
                    record["vocals_stem"] = {"source": "demucs", "path": str(sep)}
                else:
                    vocals = mix
                    note = "No vocals stem: lip sync follows the full mix (a stem syncs much better)"
            tracks = {"speech": vocals}
            final_audio = mix
            duration = media.media_duration(mix)
            if req.captions and req.script.strip():
                captions = media.time_captions(req.script, duration)
            yield Event("voice", "done", 1.0, note, artifact=mix)
            yield Event("revoice", "skipped", 1.0, "Not used in sing mode")

        if duration > 95:
            raise InputError(f"Audio is {duration:.0f}s long; keep it under 90 s.")

        # 4. lip sync (segmented) -------------------------------------------------------
        pauses = longform.detect_pauses(final_audio)
        segments = longform.plan_segments(duration, pauses, seg_seconds)
        eta = estimate_lipsync_seconds(duration, seg_seconds, req.ext_frames, s.fps, tier=req.tier)
        yield Event(
            "lipsync", "start",
            message=(f"Animating {duration:.1f}s in {len(segments)} segment(s), "
                     f"~{eta / 60:.0f} min on an RTX 5080 (free, local: retry as often as you like)"),
            data={"segments": len(segments), "eta_seconds": eta},
        )
        motion = motion_prompt(req.style, "Two people" if two else "The character")
        if two:
            motion = motion.replace("talks directly", "take turns talking, each one talks directly")
        seg_clips: list[Path] = []
        start_image = avatar
        ram_note = system.ram_preflight()
        if ram_note:
            yield Event("lipsync", "progress", -1, ram_note)
        for seg in segments:
            seg_files: dict[str, Path] = {"avatar": start_image}
            for key, track in tracks.items():
                if len(segments) == 1:
                    seg_files[key] = track
                else:
                    seg_files[key] = longform.cut_audio(track, work / f"seg{seg.index:02d}_{key}.wav", seg.start, seg.render_seconds)
            sg = graphs.lipsync_graph(
                motion, models=m, audio_seconds=seg.render_seconds, width=s.video_width, height=s.video_height,
                fps=s.fps, ext_frames=req.ext_frames, steps=steps, seed=req.seed + 100 * seg.index, two_speakers=two,
                te_device=tier.te_device,
            )
            name = "lipsync" if len(segments) == 1 else f"lipsync_{seg.index:02d}"
            n = len(segments)
            yield from self._job(name, "lipsync", sg, seg_files, work, results, record,
                                 scale=(seg.index / n, (seg.index + 1) / n),
                                 prefix=f"segment {seg.index + 1}/{n} · " if n > 1 else "")
            seg_clips.append(results[name].path)
            if req.seam == "continue" and seg.index < n - 1:
                start_image = longform.last_frame(results[name].path, work / f"seg{seg.index:02d}_last.png")
        if len(seg_clips) > 1:
            talking = longform.stitch(seg_clips, segments, work / "talking_stitched.mp4", fps=s.fps,
                                      size=(s.video_width, s.video_height))
            msg = f"{len(seg_clips)} segments stitched with crossfades"
        else:
            talking = seg_clips[0]
            msg = "Talking clip ready"
        yield Event("lipsync", "done", 1.0, msg, artifact=talking)

        # 5. smooth ------------------------------------------------------------------
        ffmpeg_fps = None
        if req.smooth == "rife":
            sg = graphs.smooth_graph(models=m, multiplier=2, fps=s.fps)
            yield Event("smooth", "start", message="Doubling the frame rate with RIFE")
            yield from self._job("smooth", "smooth", sg, {"video": talking}, work, results, record)
            talking = results["smooth"].path
            yield Event("smooth", "done", 1.0, "50 fps motion", artifact=talking)
        elif req.smooth == "ffmpeg":
            ffmpeg_fps = s.fps * 2
            yield Event("smooth", "skipped", 1.0, "Smoothing with ffmpeg during export")
        else:
            yield Event("smooth", "skipped", 1.0, "Off")

        # 6. finish ------------------------------------------------------------------
        yield Event("finish", "start", message="Burning in captions and exporting 9:16")
        w, h = (1080, 1920) if req.hd else (s.video_width, s.video_height)
        final = media.finish_video(
            talking, top / f"{top.name}.mp4", audio=final_audio, duration=duration, captions=captions,
            width=w, height=h, ffmpeg_smooth_fps=ffmpeg_fps, watermark=req.watermark, work_dir=work / "_finish",
        )
        tidy = work
        poster = media.poster_frame(final, tidy / "poster.jpg", at=min(1.0, duration / 2))
        record.update(
            final=str(final), duration=duration, segments=[asdict(x) for x in segments],
            captions=[asdict(c) for c in captions], elapsed_seconds=round(time.monotonic() - t_start, 1),
        )
        (tidy / "project.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        recipe_path = write_recipe(tidy, req, record)
        yield Event("finish", "done", 1.0, "Exported", artifact=final)
        yield Event(
            "complete", "complete", 1.0, f"Done in {record['elapsed_seconds'] / 60:.1f} min",
            artifact=final, data={"poster": str(poster), "work_dir": str(top), "duration": duration,
                                  "recipe": str(recipe_path)},
        )

    # -- shared setup ----------------------------------------------------------------
    def _begin(self, req: Request, work: Path | None, label: str, *, mode: str | None = None,
               who: str | None = None) -> tuple[Path, dict[str, Any], dict[str, StageResult]]:
        now = datetime.now()
        stamp = now.strftime("%Y%m%d-%H%M%S")
        name = readable_name(req, label, now, mode=mode, who=who)
        top = work if work is not None else self.settings.output_dir / name
        work = top / WORK_SUBDIR  # intermediates; results go to ``top`` in _complete
        (work / "inputs").mkdir(parents=True, exist_ok=True)
        record: dict[str, Any] = {
            "request": {k: _jsonable(v) for k, v in asdict(req).items()},
            "backend": self.backend.name,
            "tier": tier_record(req.tier),
            "jobs": {},
            "started": stamp,
            "t0": time.monotonic(),
            "_req": req,
            "name": name,
            "top": str(top),
        }
        return work, record, {}

    def _complete(self, work: Path, record: dict[str, Any], final: Path, duration: float,
                  extra: dict[str, Any] | None = None) -> Iterator[Event]:
        name = record.get("name")
        top = Path(record.get("top") or work)
        if final.exists() and final.parent == work:  # promote the result out of _work, with a readable name
            final = final.replace(top / f"{name or final.stem}{final.suffix}")
            if extra and extra.get("still"):
                extra["still"] = str(final)
        extra = dict(extra or {})
        if final.suffix == ".mp4":
            extra.update(_exports(final, final.parent))
        poster = media.poster_frame(final, work / "poster.jpg", at=min(1.0, duration / 2))
        req = record.pop("_req", None)
        record["exports"] = dict(extra)
        record["elapsed_seconds"] = round(time.monotonic() - record.pop("t0"), 1)
        record.update(final=str(final), duration=duration)
        (work / "project.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        if req is not None:
            extra["recipe"] = str(write_recipe(work, req, record))
        yield Event("finish", "done", 1.0, "Exported", artifact=final)
        yield Event(
            "complete", "complete", 1.0, f"Done in {record['elapsed_seconds'] / 60:.1f} min",
            artifact=final, data={"poster": str(poster), "work_dir": str(top), "duration": duration, **extra},
        )

    def _speech(self, req: Request, text: str, files: dict[str, Path], work: Path, results, record) -> Iterator[Event]:
        """Voice one short line (meme mode). Leaves the wav in ``files['speech']``."""
        if req.voice_mode == "my_recording":
            files["speech"] = media.to_wav(files["recording"], work / "speech.wav")
            yield Event("voice", "skipped", 1.0, "Using your own recording", artifact=files["speech"])
            return
        clone = req.voice_mode == "clone"
        yield Event("voice", "start", message="Saying the line in your voice" if clone else "Using the built-in voice")
        sg = graphs.tts_graph(text, models=self.settings.models, clone=clone, exaggeration=max(0.6, req.expressiveness),
                              seed=req.seed)
        yield from self._job("voice", "voice", sg, files, work, results, record)
        files["speech"] = media.to_wav(results["voice"].path, work / "speech.wav")
        yield Event("voice", "done", 1.0, "Line ready", artifact=files["speech"])

    # -- Meme Recreate ---------------------------------------------------------------
    def run_still(self, req: Request, work: Path | None = None) -> Iterator[Event]:
        """Fast path: just the scene + captions as a PNG (seconds, not minutes)."""
        check_request(req, self.settings)
        if req.mode != "meme":
            raise InputError("Instant stills are for Meme Recreate.")
        yield from self._run_meme(req, work, still_only=True)

    def _run_meme(self, req: Request, work: Path | None = None, still_only: bool = False) -> Iterator[Event]:
        s = self.settings
        tier = tiers.get(req.tier)
        m = tiers.models_for(tier, s.models)
        recipe = (memes.custom_recipe(req.custom_scene, req.custom_motion, req.meme_top or "", req.meme_bottom or "",
                                      req.line) if req.meme == "custom" else memes.RECIPES[req.meme])
        top = recipe.top if req.meme_top is None else req.meme_top
        bottom = recipe.bottom if req.meme_bottom is None else req.meme_bottom
        effect = recipe.effect if req.effect is None else req.effect
        work, record, results = self._begin(req, work, recipe.key + ("-still" if still_only else ""))
        record["recipe"] = memes.recipe_to_dict(recipe)

        subject = resolve_subject(req)
        record["subject"] = subject
        yield Event("prepare", "start", message=f"Setting up {recipe.title}" + (" (avatar mode: keeping its art style)"
                                                                               if subject == "avatar" else ""))
        files: dict[str, Path] = {}
        inp = work / "inputs"
        if req.photo:
            files["photo"] = media.prepare_photo(req.photo, inp / "photo.png", keep_whole=subject == "avatar")
        if req.say_line and req.voice_mode == "clone" and req.voice_sample:
            media.prepare_voice(req.voice_sample, inp / "voice_ref.wav", s.max_voice_seconds)
            files["voice_ref"] = inp / "voice_ref.wav"
        if req.say_line and req.voice_mode == "my_recording" and req.recording:
            media.prepare_voice(req.recording, inp / "recording.wav", 15.0)
            files["recording"] = inp / "recording.wav"
        yield Event("prepare", "done", 1.0, "Inputs ready")

        # 1. you, in the meme scene (FLUX.2 klein reference edit) ----------------------
        if req.avatar:
            avatar = media.prepare_photo(req.avatar, work / "avatar.png")
            yield Event("avatar", "skipped", 1.0, "Using your scene from the still", artifact=avatar)
        else:
            style = STYLES[req.style]
            hint = "" if req.style in ("none", "real") or keep_avatar_look(req, subject) else \
                f"{style.label} style, {style.motion_suffix}"
            prompt = f"{memes.edit_prompt(recipe, hint, subject)} {req.style_extra}".strip()
            yield Event("avatar", "start", message="Putting you in the scene",
                        data={"eta_seconds": estimate_scene_seconds(req.tier)})
            sg = graphs.avatar_graph(prompt, models=m, reference=1, seed=req.seed, prefix="mc/meme_scene",
                                     width=tier.scene_size[0], height=tier.scene_size[1], te_device=tier.te_device)
            yield from self._job("avatar", "avatar", sg, files, work, results, record)
            avatar = results["avatar"].path
            yield Event("avatar", "done", 1.0, f"Scene ready ({results['avatar'].seconds:.0f}s)", artifact=avatar)
        files["avatar"] = avatar

        if still_only:  # instant result: captioned still meme, then offer "Animate it"
            yield Event("finish", "start", message="Slapping the captions on")
            still = media.make_still_meme(avatar, work / "mc_still.png", top=top, bottom=bottom, era=req.era,
                                          watermark=req.watermark, work_dir=work / "_still",
                                          caption_layout=recipe.caption_layout)
            record.update(captions={"top": top, "bottom": bottom}, era=req.era, still=str(still))
            yield from self._complete(work, record, still, 0.0, {
                "still": str(still), "scene": str(avatar), "post_text": share_text(top, bottom),
                "animate_eta_seconds": estimate_motion_seconds(
                    tiers.motion_frames(tier, req.meme_frames or recipe.frames)[0], req.tier) + 10})
            return

        # 2a. talking version: your voice + InfiniteTalk ----------------------------------
        captions: list[media.Caption] = []
        audio: Path | None = None
        if req.say_line:
            line = req.line.strip() or recipe.line
            yield from self._speech(req, line, files, work, results, record)
            audio = files["speech"]
            duration = media.media_duration(audio)
            if duration > 15:
                raise InputError(f"The line is {duration:.0f}s long; memes hit harder under 12 s.")
            if subject == "avatar":
                yield Event("lipsync", "progress", -1, LIPSYNC_AVATAR_NOTE)
            ram_note = system.ram_preflight()
            if ram_note:
                yield Event("lipsync", "progress", -1, ram_note)
            sg = graphs.lipsync_graph(memes.motion_prompt(recipe, talking=True, subject=subject), models=m,
                                      audio_seconds=duration,
                                      width=s.video_width, height=s.video_height, fps=s.fps, seed=req.seed,
                                      steps=req.steps or tier.steps, te_device=tier.te_device)
            yield Event("lipsync", "start", message=f"Lip-syncing {duration:.1f}s (free, local: reroll any time)",
                        data={"eta_seconds": estimate_lipsync_seconds(duration, tier=req.tier)})
            yield from self._job("lipsync", "lipsync", sg, {"speech": audio, "avatar": avatar}, work, results, record)
            clip = results["lipsync"].path
            yield Event("lipsync", "done", 1.0, "Talking take ready", artifact=clip)
            yield Event("motion", "skipped", 1.0, "Lip sync drives the motion")
            fps = s.fps
            if req.captions and line and req.voice_mode != "my_recording":
                captions = media.time_captions(line, duration)
        # 2b. silent version: Wan 2.2 I2V acts out the format ---------------------------
        else:
            yield Event("voice", "skipped", 1.0, "No line (silent meme)")
            yield Event("lipsync", "skipped", 1.0, "No line")
            mw, mh = tier.motion_size
            neg = graphs.wan_negative(subject == "avatar", recipe.negative)
            if tier.motion_engine == "ti2v5b":  # MC Lite: one 5B model, 480p-ish, ~3 s
                frames, mfps = tiers.motion_frames(tier, req.meme_frames or recipe.frames)
                sg = graphs.ti2v_motion_graph(memes.motion_prompt(recipe, subject=subject), models=m, width=mw,
                                              height=mh, frames=frames, fps=mfps, seed=req.seed,
                                              steps=tier.motion_steps, te_device=tier.te_device,
                                              tiled_decode=tier.tiled_decode, negative=neg)
            else:
                sg = graphs.meme_motion_graph(memes.motion_prompt(recipe, subject=subject), models=m, width=mw,
                                              height=mh, frames=req.meme_frames or recipe.frames, fps=s.motion_fps,
                                              seed=req.seed, te_device=tier.te_device, negative=neg)
            fps = int(sg.meta["fps"])
            eta = estimate_motion_seconds(sg.meta["frames"], req.tier)
            yield Event("motion", "start",
                        message=(f"Acting it out on MC {tier.label} ({eta_range_text(eta)}, estimate; free, local: "
                                 f"reroll as often as you like)"),
                        data={"eta_seconds": eta})
            yield from self._job("meme_motion", "motion", sg, {"avatar": avatar}, work, results, record)
            clip = results["meme_motion"].path
            duration = sg.meta["frames"] / fps
            yield Event("motion", "done", 1.0, "Meme take ready", artifact=clip)

        # 3. meme captions + effect + 9:16 ---------------------------------------------
        yield Event("finish", "start", message="Adding meme captions and exporting 9:16")
        w, h = (1080, 1920) if req.hd else (s.video_width, s.video_height)
        final = media.finish_video(
            clip, work / "mc_final.mp4", audio=audio, duration=duration, captions=captions, width=w, height=h,
            watermark=req.watermark, work_dir=work / "_finish", meme_top=top, meme_bottom=bottom, effect=effect,
            effect_at=recipe.effect_at, fps=fps, era=req.era, caption_layout=recipe.caption_layout,
        )
        total = duration * (2 if effect == "boomerang" else 1)
        record.update(captions={"top": top, "bottom": bottom, "speech": [asdict(c) for c in captions]}, effect=effect,
                      era=req.era)
        yield from self._complete(work, record, final, total, {"post_text": share_text(top, bottom)})

    def run_pack(self, base: Request, recipe_keys: list[str]) -> Iterator[Event]:
        """One photo, many formats: a ready-to-post meme pack. The scene is made
        fresh per format; your photo and voice are reused."""
        keys = [k for k in recipe_keys if k in memes.RECIPES]
        if not keys:
            raise InputError("Pick at least one meme format.")
        root = self.settings.output_dir / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-meme-pack"
        finals: list[str] = []
        for i, key in enumerate(keys):
            req = replace(base, mode="meme", meme=key, meme_top=None, meme_bottom=None, effect=None)
            for ev in self.run(req, work=root / f"{i + 1:02d}-{key}"):
                ev.data = {**ev.data, "batch_index": i, "batch_total": len(keys)}
                if ev.status == "complete":
                    finals.append(str(ev.artifact))
                    ev = replace(ev, stage="batch_item", status="done")
                yield ev
        (root / "pack.json").write_text(json.dumps({"clips": finals}, indent=2), encoding="utf-8")
        yield Event("complete", "complete", 1.0, f"Meme pack done: {len(finals)} clips", artifact=Path(finals[-1]),
                    data={"clips": finals, "work_dir": str(root)})

    # -- Wan2.2 Animate, chunked (Swap Me In + Copy the Vibe) ------------------------------
    def _animate_chunks(self, mode: str, ui_stage: str, prompt: str, clip: media.PreparedClip, avatar: Path,
                        req: Request, work: Path, results, record, *, source: Path | None = None,
                        person: "media.Person | None" = None, tag: str = "", negative: str | None = None,
                        progress_span: tuple[float, float] = (0.0, 1.0), subject: str = "photo",
                        face_drive: str = "clip", driver: Path | None = None) -> Iterator[Event]:
        """Render ``clip.frames`` frames in chunks of <= 3 Animate windows (one ComfyUI
        job each) and join them. Seams: every chunk after the first re-reads the 5
        frames before its start and continues from the previous chunk's last 5
        rendered frames (``continue_motion``), the same mechanism Animate uses between
        its own windows, so motion flows across the cut. Leaves the joined, silent
        clip in ``record['animated']``.

        Multi-character: ``person`` (from the first-frame scan) picks who is replaced and
        ``source`` is the previous pass's output. Later chunks re-find that person at
        their first frame (re-scan + match), because SAM tracking can't cross jobs."""
        s = self.settings
        tier = tiers.get(req.tier)
        m = tiers.models_for(tier, s.models)
        style = stylelock.settings_for(req.style_lock, [subject])
        if style.active and mode == "replace":
            record.setdefault("style_lock", {})[tag.strip("_") or "solo"] = style.as_record()
        src_video = source or clip.video
        segs = graphs.plan_animate_segments(clip.frames)
        windows = sum(x.windows for x in segs)
        eta = estimate_animate_seconds(clip.frames, mode, req.tier, req.resolution)
        record["chunks"] = [asdict(x) for x in segs]
        span_lo, span_hi = progress_span
        if span_lo == 0.0:
            verb = "Swapping you in" if mode == "replace" else "Copying the vibe"
            yield Event(ui_stage, "start",
                        message=(f"{verb}: {clip.seconds:.1f}s = {windows} window(s) in {len(segs)} chunk(s), "
                                 f"{eta_range_text(eta)} on {tier.eta_card} (estimate; free, local)"),
                        data={"windows": windows, "chunks": len(segs), "eta_seconds": eta})
        done_w = 0
        pieces: list[Path] = []
        chunk_dir = work / f"chunks{tag}"
        chunk_dir.mkdir(exist_ok=True)
        for seg in segs:
            carry = seg.index > 0
            files: dict[str, Path] = {"avatar": avatar}
            if len(segs) == 1 and source is None:
                files["clip"] = src_video
            else:
                files["clip"] = media.cut_frames(src_video, chunk_dir / f"in_{seg.index:02d}.mp4",
                                                 seg.start - seg.lead, seg.frames + seg.lead, clip.fps)
            if person is not None:
                if seg.index == 0:
                    files["person_mask"] = person.mask
                else:  # re-find this person on the chunk's first frame
                    frame = media.grab_frame(files["clip"], chunk_dir / f"frame_{seg.index:02d}.png", 0, clip.fps)
                    found = yield from self._scan(frame, work, results, record, f"people{tag}_{seg.index:02d}",
                                                  (clip.width, clip.height), req.tier)
                    match = media.match_people([person], found)[0] if found else None
                    files["person_mask"] = match.mask if match else person.mask
            if carry:
                prev = pieces[-1]
                n_prev = segs[seg.index - 1].frames
                files["carry"] = media.cut_frames(prev, chunk_dir / f"carry_{seg.index:02d}.mp4",
                                                  n_prev - graphs.ANIMATE_MOTION, graphs.ANIMATE_MOTION, clip.fps)
            builder = graphs.swap_graph if mode == "replace" else graphs.vibe_graph
            kw: dict[str, Any] = {"person_mask": True} if person is not None else {}
            if mode == "replace":
                kw.update(isolate_ref=subject == "avatar", face_drive=face_drive,
                          mask_grow=MASK_GROW_PROPS if req.remove_props else MASK_GROW)
                if style.active:
                    kw["cfg"] = style.cfg
                    if style.mask_threshold is not None:
                        kw["mask_threshold"] = style.mask_threshold
                    if person is not None:
                        kw.update(composite_grow=style.composite_grow, composite_blur=style.composite_blur)
                if driver is not None and person is not None:
                    kw["driver"] = True
                    files["driver"] = (driver if len(segs) == 1 and source is None else
                                       media.cut_frames(driver, chunk_dir / f"drv_{seg.index:02d}.mp4",
                                                        seg.start - seg.lead, seg.frames + seg.lead, clip.fps))
            relight = use_relight(req, subject) and not (style.active and style.relight is False)
            sg = builder(prompt, models=m, frames=seg.frames, carry=carry, width=clip.width,
                         height=clip.height, fps=clip.fps, seed=req.seed, relight=relight,
                         animate_gguf=m.animate_gguf, te_device=tier.te_device, negative=negative, **kw)
            name = (ui_stage if len(segs) == 1 else f"{ui_stage}_{seg.index:02d}") + tag
            lo = span_lo + (span_hi - span_lo) * done_w / windows
            hi = span_lo + (span_hi - span_lo) * (done_w + seg.windows) / windows
            yield from self._job(name, ui_stage, sg, files, work, results, record, scale=(lo, hi),
                                 prefix=(f"{tag.strip('_')} · " if tag else "")
                                 + (f"chunk {seg.index + 1}/{len(segs)} · " if len(segs) > 1 else ""))
            done_w += seg.windows
            cut = results[name].extras.get("ref_cutout") if seg.index == 0 else None
            if cut and Path(cut).exists():  # what Animate actually saw as "you": check it if the style drifts
                kept = shutil.copyfile(cut, work / f"ref_cutout{tag}.png")
                record.setdefault("ref_cutouts", {})[tag.strip("_") or "solo"] = str(kept)
                yield Event(ui_stage, "progress", lo, "Saved the cut-out reference to _work", artifact=Path(kept))
            # keep exactly this chunk's new frames (the last window is padded with the final pose)
            pieces.append(media.cut_frames(results[name].path, chunk_dir / f"out_{seg.index:02d}.mp4", 0,
                                           seg.frames, clip.fps))
        joined = media.concat_frames(pieces, work / f"{ui_stage}{tag}_joined.mp4", clip.fps)
        record["animated"] = str(joined)
        if span_hi >= 1.0:
            msg = (f"{len(segs)} chunks joined with motion carried across the seams" if len(segs) > 1
                   else "Take ready")
            yield Event(ui_stage, "done", 1.0, msg, artifact=joined)

    def _scan(self, frame: Path, work: Path, results, record, name: str,
              size: tuple[int, int], tier: str = tiers.DEFAULT_TIER) -> Iterator[Event]:
        """SAM 3.1 people scan of one frame -> people numbered left to right (generator return value)."""
        m = tiers.models_for(tiers.get(tier), self.settings.models)
        sg = graphs.people_scan_graph(models=m, threshold=PEOPLE_SCAN_THRESHOLD)
        yield from self._job(name, "prepare", sg, {"frame": frame}, work, results, record)
        res = results[name]
        return media.order_people(res.paths or [res.path], size)

    def _soundtrack(self, req: Request, clip: media.PreparedClip, work: Path, results, record) -> Iterator[Event]:
        """The finished clip's audio: the upload's own track (default), silence, or that track re-voiced as
        you (Chatterbox VC). Same length as the clip (cut at the same start, padded/trimmed by the finisher)."""
        record["clip_audio"] = req.clip_audio
        record["_audio"] = None
        if req.clip_audio == "mute":
            yield Event("revoice", "skipped", 1.0, "Muted: exporting with a silent track")
            return
        if clip.audio is None:
            yield Event("revoice", "skipped", 1.0, "Your clip has no audio track: exporting silent")
            return
        if req.clip_audio == "original":
            record["_audio"] = clip.audio
            yield Event("revoice", "skipped", 1.0, "Keeping the clip's original audio, in sync")
            return
        s = self.settings
        m = tiers.models_for(tiers.get(req.tier), s.models)
        target = work / "inputs" / "voice_ref.wav"
        media.prepare_voice(req.voice_sample, target, s.max_voice_seconds)
        src = work / "inputs" / "clip_audio_vc.wav"
        media.to_wav(clip.audio, src)
        sg = graphs.voice_changer_graph(models=m, prefix="mc/clip_revoice")
        yield Event("revoice", "start", message="Re-voicing the clip's audio as you (works best on speech; music "
                                                "under it gets re-voiced too)")
        yield from self._job("revoice", "revoice", sg, {"source": src, "target": target}, work, results, record)
        record["_audio"] = results["revoice"].path
        yield Event("revoice", "done", 1.0, "Clip audio re-voiced", artifact=results["revoice"].path)

    def _finish_animated(self, req: Request, work: Path, record, clip: media.PreparedClip, video: Path,
                         audio: Path | None) -> Iterator[Event]:
        portrait = clip.height >= clip.width
        w, h = ((1080, 1920) if portrait else (1920, 1080)) if req.hd else (clip.width, clip.height)
        final = media.finish_video(
            video, work / "mc_final.mp4", audio=audio, duration=clip.seconds, width=w, height=h,
            watermark=req.watermark, work_dir=work / "_finish", meme_top=req.meme_top or "",
            meme_bottom=req.meme_bottom or "", fps=clip.fps, era=req.era,
        )
        record.update(captions={"top": req.meme_top or "", "bottom": req.meme_bottom or ""}, era=req.era,
                      audio=str(audio) if audio else None)
        yield from self._complete(work, record, final, clip.seconds,
                                  {"post_text": share_text(req.meme_top or "", req.meme_bottom or "")})

    def _prepare_clip(self, req: Request, work: Path, cap: float, record) -> Iterator[Event]:
        s = self.settings
        tier = tiers.get(req.tier)
        cap = tiers.animate_cap(tier, cap)
        size = tiers.animate_size_for(tier, req.resolution)
        yield Event("prepare", "start", message=f"Preparing your clip ({s.motion_fps} fps, {size[0]}p)")
        warn = tiers.resolution_warning(tier, req.resolution)
        if warn:
            yield Event("prepare", "progress", 0.1, f"Heads-up: {warn}")
        record["resolution"] = {"choice": req.resolution, "render": list(size)}
        clip = media.prepare_clip(req.clip, work / "inputs", start=req.clip_start,
                                  max_seconds=min(req.clip_seconds, cap), fps=s.motion_fps, portrait=size,
                                  rotate=req.rotate)
        record["clip"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(clip).items()}
        record["_clip"] = clip
        segs = graphs.plan_animate_segments(clip.frames)
        yield Event("prepare", "done", 1.0,
                    f"{clip.seconds:.1f}s at {clip.width}x{clip.height}: {sum(x.windows for x in segs)} window(s), "
                    f"{len(segs)} chunk(s)" + ("" if clip.audio else " · no audio track"))

    def _erase_props(self, req: Request, clip: media.PreparedClip, work: Path, results, record) -> Iterator[Event]:
        """Optional VOID pre-pass (experimental): erase ``req.erase_props`` from the prepared clip so Animate
        never sees the original character's backpack / props. Returns the (possibly) cleaned clip."""
        if not req.erase_props:
            return clip
        m = tiers.models_for(tiers.get(req.tier), self.settings.models)
        chunks = graphs.plan_void_chunks(clip.frames)
        yield Event("prepare", "start", message=(f"Erasing '{req.erase_props}' from your clip first (VOID, "
                                                 f"{len(chunks)} pass{'es' if len(chunks) > 1 else ''}, "
                                                 f"{eta_range_text(estimate_erase_seconds(clip.frames))}, experimental)"))
        edir = work / "erase"
        edir.mkdir(exist_ok=True)
        pieces: list[Path] = []
        for k, (first, count) in enumerate(chunks):
            src = clip.video if len(chunks) == 1 else media.cut_frames(clip.video, edir / f"in_{k:02d}.mp4", first,
                                                                         count, clip.fps)
            n = graphs.void_valid_length(count)
            if n != count:  # VOID needs 8k-3 frames: hold the last frame, then cut the padding off again
                padded = edir / f"in_{k:02d}_pad.mp4"
                media.run_ffmpeg(["-i", str(src), "-an", "-vf", f"tpad=stop_mode=clone:stop={n - count}",
                                  "-frames:v", str(n), "-r", str(clip.fps), "-c:v", "libx264", "-crf", "14",
                                  "-pix_fmt", "yuv420p", str(padded)])
                src = padded
            sg = graphs.void_erase_graph(req.erase_props, models=m, frames=n, width=clip.width,
                                         height=clip.height, fps=clip.fps, seed=req.seed)
            name = "erase" if len(chunks) == 1 else f"erase_{k:02d}"
            yield from self._job(name, "prepare", sg, {"clip": src}, work, results, record)
            pieces.append(media.cut_frames(results[name].path, edir / f"out_{k:02d}.mp4", 0, count, clip.fps))
        erased = media.concat_frames(pieces, work / "erased_clip.mp4", clip.fps)
        record["erase_props"] = {"what": req.erase_props, "clip": str(erased), "chunks": len(chunks)}
        yield Event("prepare", "done", 1.0, f"Erased '{req.erase_props}' (check _work/erased_clip.mp4)",
                    artifact=erased)
        return replace(clip, video=erased)

    # -- Multi-character: find the people ---------------------------------------------
    def scan_people(self, req: Request, work: Path | None = None) -> Iterator[Event]:
        """Preview for multi-character Swap Me In: the clip's first frame (at ``clip_start``) with every
        person numbered 1, 2, 3 left to right. ``complete`` carries ``people`` (count) and ``preview``."""
        if not req.clip:
            raise InputError("Upload the clip first.")
        s = self.settings
        req.tier = tiers.normalize(req.tier)
        work, record, results = self._begin(replace(req, mode="swap"), work, Path(str(req.clip)).stem,
                                            mode="people-scan", who="numbered")
        yield from self._prepare_clip(req, work, s.max_swap_seconds, record)
        clip: media.PreparedClip = record.pop("_clip")
        frame = media.grab_frame(clip.video, work / "first_frame.png", 0, clip.fps)
        people = yield from self._scan(frame, work, results, record, "people", (clip.width, clip.height), req.tier)
        if not people:
            raise InputError("No people found in the first frame. Try a start time where everyone is visible.")
        preview = media.people_preview(frame, people, work / "people_preview.png")
        cap = tiers.people_cap(req.tier, req.allow_heavy)
        note = (f"Found {len(people)} {'person' if len(people) == 1 else 'people'} (numbered left to right). "
                f"MC {tiers.get(req.tier).label} swaps up to {cap} per clip; {eta_range_text(estimate_people_seconds(clip.frames, 1, req.tier, req.resolution))} per person.")
        shown = top_preview = Path(record["top"]) / f"{record['name']}.png"
        shutil.copyfile(preview, top_preview)
        yield Event("complete", "complete", 1.0, note, artifact=shown,
                    data={"people": len(people), "preview": str(shown), "work_dir": record["top"],
                          "boxes": [p.box for p in people]})

    # -- Swap Me In ------------------------------------------------------------------
    def _run_swap(self, req: Request, work: Path | None = None) -> Iterator[Event]:
        if req.people:
            yield from self._run_swap_people(req, work)
            return
        s = self.settings
        tier = tiers.get(req.tier)
        m = tiers.models_for(tier, s.models)
        sw, sh = tier.scene_size
        work, record, results = self._begin(req, work, Path(str(req.clip)).stem)
        prep = yield from self._reused_prep(req, work, record, people=False)
        if prep is None:
            yield from self._prepare_clip(req, work, s.max_swap_seconds, record)
            clip: media.PreparedClip = record.pop("_clip")
            clip = yield from self._erase_props(req, clip, work, results, record)
        else:
            clip = prep["clip"]
        subject = resolve_subject(req, req.photo or req.avatar)
        record["subject"] = subject
        ref_size = (clip.width * 2, clip.height * 2)
        files: dict[str, Path] = {}
        if req.photo:
            files["photo"] = media.fit_photo(req.photo, work / "inputs" / "photo.png", ref_size)

        if req.avatar:
            avatar = media.fit_photo(req.avatar, work / "avatar.png", ref_size)
            yield Event("avatar", "skipped", 1.0, "Using your ready-made avatar", artifact=avatar)
        elif (req.style == "none" or keep_avatar_look(req, subject)) and not req.style_extra.strip():
            avatar = files["photo"]
            yield Event("avatar", "skipped", 1.0, "Swapping in your avatar as drawn" if subject == "avatar"
                        else "Swapping in your photo as-is", artifact=avatar)
        else:  # a look (style) and/or extra look details, e.g. "wearing a tuxedo"
            style = STYLES[req.style if req.style != "none" else "real"]
            if subject == "avatar" and keep_avatar_look(req, subject):
                prompt = ("Full body portrait of the character from image 1 as an exact copy: the same face, hair, "
                          "outfit, colors, proportions and art style, not redesigned, not photorealistic.")
            else:
                prompt = style.edit_prompt.replace(", centered, looking straight into the camera, mouth gently "
                                                   "closed", "")
            prompt += " Full body visible, plain neutral background."
            portrait = clip.height >= clip.width
            sg = graphs.avatar_graph(f"{prompt} {req.style_extra}".strip(), models=m, reference=1, seed=req.seed,
                                     width=sw if portrait else sh, height=sh if portrait else sw, prefix="mc/swap_ref",
                                     te_device=tier.te_device)
            yield Event("avatar", "start", message=f"Making your {style.label} look",
                        data={"eta_seconds": estimate_scene_seconds(req.tier)})
            yield from self._job("avatar", "avatar", sg, files, work, results, record)
            avatar = results["avatar"].path
            yield Event("avatar", "done", 1.0, "Look ready", artifact=avatar)

        audio = yield from self._clip_soundtrack(req, clip, work, results, record, prep)
        flat = subject == "avatar" and media.is_flat_art(req.photo or req.avatar)
        drive, why = resolve_face_drive(face_choice(req, 0), Path(str(avatar)) if avatar else None)
        record["swap_settings"] = dict(swap_settings(req, subject, flat), face_drive=drive, face_why=why)
        if drive != "clip":
            yield Event("swap", "progress", 0.0, f"Face: {why}")
        yield from self._animate_chunks("replace", "swap", swap_prompt(req, subject, flat), clip, avatar, req, work,
                                        results, record, negative=swap_negative(req, subject), subject=subject,
                                        face_drive=drive)
        swapped = Path(record.pop("animated"))
        yield Event("finish", "start", message="Re-adding the audio and exporting" if audio else "Exporting")
        yield from self._finish_animated(req, work, record, clip, swapped, audio)

    def _run_swap_people(self, req: Request, work: Path | None = None) -> Iterator[Event]:
        """Multi-character Swap Me In: scan the first frame, then one Animate pass per assigned person,
        each pass starting from the previous pass's output (the others stay pixel-identical)."""
        s = self.settings
        work, record, results = self._begin(req, work, Path(str(req.clip)).stem)
        prep = yield from self._reused_prep(req, work, record, people=True)
        if prep is None:
            yield from self._prepare_clip(req, work, s.max_swap_seconds, record)
            clip: media.PreparedClip = record.pop("_clip")
            clip = yield from self._erase_props(req, clip, work, results, record)
            frame = media.grab_frame(clip.video, work / "first_frame.png", 0, clip.fps)
            people = yield from self._scan(frame, work, results, record, "people", (clip.width, clip.height),
                                           req.tier)
        else:
            clip, frame, people = prep["clip"], prep["frame"], prep["people"]
        if not people:
            raise InputError("No people found in the first frame. Try a start time where everyone is visible.")
        slots = list(req.people) + [None] * max(0, len(people) - len(req.people))
        if len(slots) > len(people):
            extra = [i + 1 for i in range(len(people), len(slots)) if slots[i]]
            if extra:
                raise InputError(f"The clip has {len(people)} people, but you assigned person "
                                 f"{', '.join(map(str, extra))}. Re-run 'find people' and assign again.")
            slots = slots[:len(people)]
        jobs = [(i, ref) for i, ref in enumerate(slots) if ref]
        names = slot_names(len(people))
        preview = media.people_preview(frame, people, work / "people_preview.png",
                                       ["you" if ref else "" for ref in slots])
        specs = [parse_driver(req.people_driver[i] if i < len(req.people_driver) else "", len(people), i)
                 for i in range(len(people))]
        record["people"] = [{"slot": i + 1, "where": names[i], "box": p.box, "ref": str(slots[i] or ""),
                             "moves": driver_label(*specs[i])} for i, p in enumerate(people)]
        yield Event("avatar", "done", 1.0, f"{len(people)} people found; swapping {len(jobs)}", artifact=preview)
        audio = yield from self._clip_soundtrack(req, clip, work, results, record, prep, frame=frame, people=people)
        engine, why = group_engine_for(req, clip.frames, len(jobs), len(people))
        if engine == "scail":
            routed = self._route_framing(req, frame, people, jobs, record)
            if routed:
                engine, why = "chain", routed
        if any(k != "self" for k, _ in specs) and engine == "scail":  # SCAIL-2 has no per-person driver input
            engine, why = "chain", "someone follows another person's moves: one Animate pass per person"
        cuts = self._shots_for(req, clip, record)
        second_note = ((record.get("shots") or {}).get("second_opinion") or {}).get("note")
        if second_note:
            yield Event("prepare", "progress", 1.0, f"Heads-up: {second_note}")
        if cuts:
            summary = record["shots"]["summary"]
            if engine == "scail" and req.split_cuts != "on":
                yield Event("prepare", "progress", 1.0, f"Heads-up: {summary}. The group pass runs over the whole "
                            "clip; if people vanish after a cut, set 'split at cuts' on (guide: #guide-cuts)")
                cuts = []
            else:
                if engine == "scail":
                    engine, why = "chain", "split at hard cuts: one Animate pass per person, shot by shot"
                yield Event("prepare", "progress", 1.0, f"{summary[0].upper()}{summary[1:]}: swapping each shot on "
                            "its own and keeping who-is-who by appearance (guide: #guide-cuts)")
        record["group_engine"] = {"engine": engine, "why": why}
        if engine == "scail":
            current = yield from self._scail_group(req, clip, people, jobs, work, results, record)
            if current is not None:
                verdict = yield from self._take_qc(req, clip, people, jobs, current, work, results, record)
                if verdict is None or verdict["pass"]:
                    yield from self._deliver(req, work, record, results, clip, current, audio, group=True)
                    return
                record["group_engine"] = {"engine": "chain", "why": f"SCAIL-2 take failed framing QC "
                                                                    f"({verdict['why']}); fell back to chained Animate"}
                yield Event("swap", "progress", 0.0, f"SCAIL-2 take failed framing QC ({verdict['why']}); "
                                                     f"skipping SeedVR2 and using one Animate pass per person")
            else:
                record["group_engine"] = {"engine": "chain",
                                          "why": "SCAIL-2 pass failed; fell back to chained Animate"}
        elif req.group_engine != "chain" and len(jobs) > 1:
            yield Event("swap", "progress", 0.0, f"One pass per person: {why}")
        eta = estimate_people_seconds(clip.frames, len(jobs), req.tier, req.resolution)
        yield Event("swap", "start", message=(f"Swapping {len(jobs)} of {len(people)} people, one pass each: "
                                              f"{eta_range_text(eta)} on {tiers.get(req.tier).eta_card} (estimate)"),
                    data={"eta_seconds": eta})
        if cuts and engine != "scail":
            current = yield from self._swap_shots(req, clip, frame, people, slots, cuts, work, results, record)
        else:
            drivers = self._drivers(req, clip, people, [(i, ref, i) for i, ref in jobs], work, record)
            current = yield from self._chain_pass(req, clip, people, [(i, ref, i) for i, ref in jobs], work, results,
                                                  record, drivers=drivers)
        yield Event("swap", "done", 1.0, f"{len(jobs)} {'person' if len(jobs) == 1 else 'people'} swapped", artifact=current)
        yield from self._deliver(req, work, record, results, clip, current, audio, group=False)

    def _chain_pass(self, req: Request, clip: media.PreparedClip, people: list, jobs: list, work: Path, results,
                    record, *, drivers: dict | None = None, tag: str = "", span: tuple[float, float] = (0.0, 1.0),
                    source: Path | None = None) -> Iterator[Event]:
        """One Animate pass per job ``(slot in this clip, ref, original slot)``, each starting from the previous
        pass's output. ``original slot`` keys the face / prompt / driver choices (it differs from the slot after a
        hard cut). Returns the silent take (generator return value); ``source`` when there are no jobs."""
        current: Path | None = source
        ref_size = (clip.width * 2, clip.height * 2)
        lo, hi = span
        for n, (i, ref, orig) in enumerate(jobs):
            subject = resolve_subject(replace(req, photo=Path(str(ref))), Path(str(ref)))
            record.setdefault("subjects", []).append(subject)
            ref = self._closeup_ref(req, clip, people[i] if i < len(people) else None, ref, work,
                                    f"p{orig + 1}{tag}", record)
            avatar = media.fit_photo(ref, work / f"ref_p{orig + 1}.png", ref_size)
            sub = (lo + (hi - lo) * n / len(jobs), lo + (hi - lo) * (n + 1) / len(jobs))
            flat = subject == "avatar" and media.is_flat_art(ref)
            drive, why = resolve_face_drive(face_choice(req, orig), Path(str(ref)))
            record.setdefault("swap_settings", {})[f"p{orig + 1}{tag}"] = dict(swap_settings(req, subject, flat),
                                                                              face_drive=drive, face_why=why)
            if drive != "clip":
                yield Event("swap", "progress", sub[0], f"Person {orig + 1} face: {why}")
            yield from self._animate_chunks("replace", "swap", swap_prompt(req, subject, flat, slot=orig), clip,
                                            avatar, req, work, results, record, source=current, person=people[i],
                                            tag=f"{tag}_p{orig + 1}", negative=swap_negative(req, subject),
                                            progress_span=sub, subject=subject, face_drive=drive,
                                            driver=(drivers or {}).get(orig))
            current = Path(record.pop("animated"))
        return current if current is not None else clip.video

    def _closeup_ref(self, req: Request, clip: media.PreparedClip, person: Any, ref: Any, work: Path, key: str,
                     record) -> Any:
        """Request.ref_closeup: crop a full-body reference to the face for a close-up shot of this person (else
        the reference unchanged). Never fails the run."""
        if (req.ref_closeup or "off") != "auto" or person is None or ref is None:
            return ref
        try:
            import numpy as np

            frame = media.grab_frame(clip.video, work / f"closeup_frame_{key}.png", 0, clip.fps)
            shot = framing.classify_person(np.asarray(Image.open(frame).convert("RGB")), person.box)
            x0, y0, x1, y1 = person.box
            got = framing.closeup_reference(ref, work / f"ref_{key}_closeup.png", shot["framing"],
                                            aspect=(x1 - x0) / max(1, y1 - y0))
        except Exception as exc:  # noqa: BLE001 - an optimisation, never a failure
            record.setdefault("ref_closeup", {})[key] = {"error": str(exc)[:200]}
            return ref
        record.setdefault("ref_closeup", {})[key] = (dict(got, shot=shot["framing"], note="pending a real test")
                                                    if got else {"shot": shot["framing"], "used": False})
        return Path(got["path"]) if got else ref

    def _drivers(self, req: Request, clip: media.PreparedClip, people: list, jobs: list, work: Path, record, *,
                 tag: str = "", slot_of: dict | None = None, src_start: int = 0) -> dict:
        """``driver.mp4`` per original slot that follows someone else's moves (Request.people_driver).
        ``slot_of`` maps an original slot to its slot in this clip (after a cut); a followed person who isn't in
        this shot falls back to the person's own moves (noted in the record)."""
        out: dict[int, Path] = {}
        slot_of = slot_of if slot_of is not None else {i: i for i in range(len(people))}
        n_all = max([len(people), len(req.people), len(req.people_driver), *[o + 1 for o in slot_of]])
        for i, _ref, orig in jobs:
            spec = req.people_driver[orig] if orig < len(req.people_driver) else ""
            kind, value = parse_driver(spec, n_all, orig)
            if kind == "self":
                continue
            ddir = work / "drivers"
            ddir.mkdir(exist_ok=True)
            dest = ddir / f"driver{tag}_p{orig + 1}.mp4"
            if kind == "person":
                k = slot_of.get(value)
                if k is None or k >= len(people):
                    record.setdefault("driver_notes", []).append(
                        f"person {orig + 1}{tag}: person {value + 1} isn't in this shot, kept own moves")
                    continue
                media.compose_driver(clip.video, dest, dst_box=people[i].box, src_box=people[k].box,
                                     frame_size=(clip.width, clip.height), frames=clip.frames, fps=clip.fps)
            else:
                media.compose_driver(clip.video, dest, dst_box=people[i].box, driver_clip=value,
                                     frame_size=(clip.width, clip.height), frames=clip.frames, fps=clip.fps,
                                     src_start=src_start)
            out[orig] = dest
            record.setdefault("drivers", {})[f"p{orig + 1}{tag}"] = {"moves": driver_label(kind, value),
                                                                     "video": str(dest), "note": stylelock.PENDING_NOTE}
        return out

    def _shots_for(self, req: Request, clip: media.PreparedClip, record) -> list:
        """Hard cuts of the prepared clip (shots.py), recorded; [] when there are none or split_cuts is off."""
        mode = (req.split_cuts or "auto").strip().lower()
        if mode == "off":
            return []
        try:
            found = shots.detect_cuts(clip.video)
        except Exception as exc:  # noqa: BLE001 - detection is advisory: render the clip as one shot
            record["shots"] = {"mode": mode, "error": str(exc)[:200]}
            return []
        if found:
            found[-1] = shots.Shot(found[-1].start, max(found[-1].end, clip.frames))
        found = [shots.Shot(s.start, min(s.end, clip.frames)) for s in found if s.start < clip.frames]
        record["shots"] = {"mode": mode, "cuts": [s.start for s in found[1:]], "shots": [[s.start, s.end] for s in found],
                           "summary": shots.summary(found, clip.fps)}
        if os.environ.get("MC_CUT_SECOND", "1").strip().lower() not in ("0", "off", "false", "no"):
            try:  # optional PySceneDetect cross-check: recorded and noted, never changes the split
                second = shots.second_opinion(clip.video, record["shots"]["cuts"], clip.fps)
            except Exception as exc:  # noqa: BLE001
                second = {"error": str(exc)[:200]}
            if second is not None:
                record["shots"]["second_opinion"] = second
        return found if len(found) > 1 else []

    def _swap_shots(self, req: Request, clip: media.PreparedClip, frame: Path, people: list, slots: list, cuts: list,
                    work: Path, results, record) -> Iterator[Event]:
        """Hard cuts: every shot swapped on its own (SAM tracking can't follow people across a cut), people
        re-scanned at each shot's first frame and matched to shot 1's numbering by appearance."""
        import numpy as np

        sdir = work / "shots"
        sdir.mkdir(exist_ok=True)
        clips, people_by_shot, frames_by_shot = [], [], []
        for k, sh in enumerate(cuts):
            v = media.cut_frames(clip.video, sdir / f"shot_{k:02d}.mp4", sh.start, sh.frames, clip.fps)
            sc = replace(clip, video=v, frames=sh.frames, seconds=sh.frames / clip.fps, audio=None)
            if k == 0:
                fr, ppl = frame, people
            else:
                fr = media.grab_frame(v, sdir / f"shot_{k:02d}_frame.png", 0, clip.fps)
                ppl = yield from self._scan(fr, work, results, record, f"people_shot{k:02d}",
                                            (clip.width, clip.height), req.tier)
            clips.append(sc)
            people_by_shot.append(ppl)
            frames_by_shot.append(fr)
        sightings = []
        for k, (fr, ppl) in enumerate(zip(frames_by_shot, people_by_shot)):
            img = np.asarray(Image.open(fr).convert("RGB").resize((clip.width, clip.height)))
            for slot, p in enumerate(ppl):
                mask = np.asarray(Image.open(p.mask).convert("L").resize((clip.width, clip.height))) > 127
                sightings.append(shots.Sighting(k, slot, shots.appearance(img, mask)))
        per_shot = shots.assign_across_shots(list(range(len(people))), sightings)
        record["shots"]["who"] = {}
        pieces: list[Path] = []
        for k, (sc, ppl) in enumerate(zip(clips, people_by_shot)):
            row = per_shot.get(k, [])
            jobs = [(slot, slots[orig], orig) for slot, orig in enumerate(row)
                    if orig is not None and orig < len(slots) and slots[orig]]
            seen = {orig for orig in row if orig is not None}
            missing = [o + 1 for o, ref in enumerate(slots) if ref and o not in seen]
            record["shots"]["who"][f"shot{k + 1}"] = {"people": [None if o is None else o + 1 for o in row],
                                                      "missing": missing}
            if missing and k > 0:
                yield Event("swap", "progress", k / len(clips), f"Shot {k + 1}: person "
                            f"{', '.join(map(str, missing))} not recognised here (left as filmed)")
            slot_of = {orig: slot for slot, orig in enumerate(row) if orig is not None}
            drivers = self._drivers(req, sc, ppl, jobs, work, record, tag=f"_s{k + 1}", slot_of=slot_of,
                                    src_start=cuts[k].start)
            take = yield from self._chain_pass(req, sc, ppl, jobs, work, results, record, drivers=drivers,
                                               tag=f"_s{k + 1}", span=(k / len(clips), (k + 1) / len(clips)))
            pieces.append(media.cut_frames(take, sdir / f"take_{k:02d}.mp4", 0, sc.frames, clip.fps))
        return media.concat_frames(pieces, work / "swap_shots_joined.mp4", clip.fps)

    def _scail_group(self, req: Request, clip: media.PreparedClip, people: list, jobs: list, work: Path, results,
                     record) -> Iterator[Event]:
        """Everyone swapped in ONE SCAIL-2 pass (replacement mode). Returns the silent take (generator return
        value), or None if the pass failed (the caller falls back to chained Animate)."""
        m = tiers.models_for(tiers.get(req.tier), self.settings.models)
        gdir = work / "scail"
        gdir.mkdir(exist_ok=True)
        palette = graphs.SCAIL_PALETTE
        refs, sam_masks = [], []
        for n, (i, ref) in enumerate(jobs):  # find the character in each reference (busy photo backgrounds too)
            fitted = media.fit_photo(ref, gdir / f"ref_p{i + 1}.png", (clip.width * 2, clip.height * 2))
            refs.append(fitted)
            found: list = []
            if not media.has_transparency(Image.open(fitted)):
                try:
                    found = yield from self._scan(fitted, work, results, record, f"ref_scan_p{i + 1}",
                                                  Image.open(fitted).size, req.tier)
                except Exception:  # noqa: BLE001 - the plain-background matte still works
                    found = []
            sam_masks.append(max(found, key=lambda p: p.area).mask if found else None)
        made = media.scail_references(refs, [people[i].box for i, _ in jobs], (clip.width, clip.height), gdir,
                                      palette, sam_masks)
        files: dict[str, Path] = {"clip": clip.video, "ref": made["ref"], "ref_mask": made["ref_mask"]}
        for k, (v, vm) in enumerate(zip(made["views"], made["view_masks"])):
            files[f"view_{k}"], files[f"view_mask_{k}"] = v, vm
        subjects = [resolve_subject(replace(req, photo=Path(str(ref))), Path(str(ref))) for _, ref in jobs]
        relight = use_relight(req, "photo" if all(x == "photo" for x in subjects) else "avatar")
        # SCAIL-2's relight LoRA was part of the winning A/B for drawn characters too (the no-relight take had
        # more flicker), so the group path keeps it on unless you switch "match scene lighting" off yourself.
        relight = True if req.relight is None else relight
        record["subjects"] = subjects
        style = stylelock.settings_for(req.style_lock, subjects)
        if style.active and req.relight is None:  # style lock: the relight LoRA pulls drawings toward the clip's look
            relight = False
        prompt = group_swap_prompt(req, len(jobs), subjects, [i for i, _ in jobs])
        if style.active and style.positive_prefix:  # prompt A/B: style tags first (MC_STYLE_POSITIVE)
            prompt = f"{style.positive_prefix}. {prompt}"
        elif style.active and any(x == "avatar" for x in subjects):
            prompt = f"{prompt} {FLAT_STYLE[0].upper()}{FLAT_STYLE[1:]}."
        def build(sage: str, frames: int = clip.frames, carry: bool = False, seed_offset: int = 0):
            return graphs.scail_group_graph(prompt, models=m, frames=frames, width=clip.width,
                                            height=clip.height, people=len(people), views=len(jobs), fps=clip.fps,
                                            seed=req.seed + seed_offset, steps=req.steps or None,
                                            relight=1.0 if relight else 0.0, sage=sage,
                                            cfg=style.cfg if style.active else 1.0,
                                            negative=style.negative_extra if style.active else "", carry=carry)
        if style.active:
            record["style_lock"] = style.as_record()

        sage = graphs.SCAIL_SAGE
        sg = build(sage)
        record["swap_settings"] = {"engine": "scail-2", "subjects": subjects, "relight": relight,
                                   "render": sg.meta["size"], "people": [i + 1 for i, _ in jobs],
                                   "attention": sage or "comfy default"}
        yield Event("swap", "start", message=(f"Swapping {len(jobs)} of {len(people)} people in one SCAIL-2 pass: "
                                              f"{eta_range_text(SCAIL_GROUP_SECONDS)} on {tiers.get(req.tier).eta_card} "
                                              f"(estimate; free, local)"),
                    data={"eta_seconds": SCAIL_GROUP_SECONDS})
        plan = graphs.plan_scail_chunks(clip.frames) if clip.frames > graphs.SCAIL_MAX_FRAMES else None
        if plan is not None and not req.scail_chunks:
            plan = None  # (group_engine_for only lets long clips here with the flag on)
        state = {"sage": sage}

        def run(name: str, make, job_files: dict, scale=(0.0, 1.0), prefix_txt: str = "") -> Iterator[Event]:
            """One SCAIL-2 job; reruns once with normal attention when Sage isn't available. True on success."""
            g_ = make(state["sage"])
            while True:
                try:
                    yield from self._job(name, "swap", g_, job_files, work, results, record, scale=scale,
                                         prefix=prefix_txt)
                    return True
                except Cancelled:
                    raise
                except Exception as exc:  # noqa: BLE001 - e.g. SCAIL-2 not installed / out of memory: chain instead
                    if state["sage"] and graphs.sage_unavailable(exc):
                        yield Event("swap", "progress", 0.0, "Sage attention isn't available on this ComfyUI; "
                                                             "running SCAIL-2 with normal attention")
                        state["sage"] = ""
                        g_ = make("")
                        record["swap_settings"]["attention"] = "comfy default (sage unavailable)"
                        continue
                    yield Event("swap", "progress", 0.0, f"SCAIL-2 pass failed ({str(exc)[:160]}); using one "
                                                         f"Animate pass per person instead")
                    return False

        if plan is not None:  # long clip: one SCAIL-2 prompt per 81-frame chunk, chained (pending a real test)
            record["swap_settings"]["chunks"] = [list(x) for x in plan]
            yield Event("swap", "progress", 0.0, f"Long clip: {len(plan)} SCAIL-2 chunks, each continuing from the "
                                                 f"last {graphs.SCAIL_ANCHOR} frames of the one before (experimental)")
            pieces: list[Path] = []
            for k, (start, length, new) in enumerate(plan):
                n_k = graphs.scail_frames(length)
                win = media.cut_frames(clip.video, gdir / f"win_{k:02d}.mp4", start, length, clip.fps)
                if n_k > length:  # SCAIL-2 lengths are 4n+1: hold the last frame
                    padded = gdir / f"win_{k:02d}_pad.mp4"
                    media.run_ffmpeg(["-i", str(win), "-an", "-vf", f"tpad=stop_mode=clone:stop={n_k - length}",
                                      "-frames:v", str(n_k), "-r", str(clip.fps), "-c:v", "libx264", "-crf", "14",
                                      "-pix_fmt", "yuv420p", str(padded)])
                    win = padded
                cfiles = dict(files, clip=win)
                if k:
                    last = pieces[-1]
                    n_last = plan[k - 1][2]
                    cfiles["prev_chunk"] = media.cut_frames(last, gdir / f"prev_{k:02d}.mp4",
                                                            max(0, n_last - graphs.SCAIL_ANCHOR),
                                                            min(graphs.SCAIL_ANCHOR, n_last), clip.fps)
                name = f"swap_group_c{k + 1:02d}"
                ok = yield from run(name, lambda sg_sage, n_k=n_k, k=k: build(sg_sage, n_k, k > 0),
                                    cfiles, scale=(k / len(plan), (k + 1) / len(plan)),
                                    prefix_txt=f"chunk {k + 1}/{len(plan)} · ")
                if not ok:
                    return None
                pieces.append(media.cut_frames(results[name].path, gdir / f"new_{k:02d}.mp4", 0, new, clip.fps))
            joined = media.concat_frames(pieces, work / "swap_group_chunks.mp4", clip.fps)
            results["swap_group"] = StageResult(path=joined)
        else:
            ok = yield from run("swap_group", lambda sg_sage: build(sg_sage), files)
            if not ok:
                return None
        take = media.cut_frames(results["swap_group"].path, work / "swap_group_take.mp4", 0, clip.frames, clip.fps)
        yield Event("swap", "done", 1.0, f"{len(jobs)} people swapped in one pass", artifact=take)
        return take

    def _upscale(self, req: Request, clip: media.PreparedClip, video: Path, work: Path, results, record, *,
                 group: bool = False) -> Iterator[Event]:
        """SeedVR2 restore + upscale of the silent take to the export size (see Request.upscale). Returns the
        video to finish (the input unchanged when SeedVR2 is off, not available on this tier, or fails)."""
        if not wants_seedvr2(req, group):
            return video
        portrait = clip.height >= clip.width
        w, h = ((1080, 1920) if portrait else (1920, 1080)) if req.hd else (clip.width, clip.height)
        m = tiers.models_for(tiers.get(req.tier), self.settings.models)
        udir = work / "upscale"
        udir.mkdir(exist_ok=True)
        total = media.count_frames(video)
        chunks = [(a, min(graphs.SEEDVR2_MAX_FRAMES, total - a)) for a in range(0, total, graphs.SEEDVR2_MAX_FRAMES)]
        yield Event("finish", "start", message=(f"Sharpening to {w}x{h} with SeedVR2 "
                                                f"({eta_range_text(SEEDVR2_SECONDS_PER_FRAME * total)})"))
        pieces: list[Path] = []
        try:
            for k, (a, n) in enumerate(chunks):
                src = video if len(chunks) == 1 else media.cut_frames(video, udir / f"in_{k:02d}.mp4", a, n, clip.fps)
                sg = graphs.seedvr2_graph(models=m, width=w, height=h, fps=clip.fps, seed=req.seed)
                name = "upscale" if len(chunks) == 1 else f"upscale_{k:02d}"
                yield from self._job(name, "finish", sg, {"clip": src}, work, results, record,
                                     scale=(k / len(chunks), (k + 1) / len(chunks)))
                pieces.append(media.cut_frames(results[name].path, udir / f"out_{k:02d}.mp4", 0, n, clip.fps))
        except Cancelled:
            raise
        except Exception as exc:  # noqa: BLE001 - keep the take; the finisher's lanczos upscale still runs
            yield Event("finish", "progress", 0.0, f"SeedVR2 skipped ({str(exc)[:160]}); using a plain upscale")
            return video
        up = media.concat_frames(pieces, work / "upscaled_take.mp4", clip.fps)
        record["upscale"] = {"engine": "seedvr2", "size": [w, h], "chunks": len(chunks)}
        style = stylelock.settings_for(req.style_lock, record.get("subjects") or [record.get("subject", "photo")])
        if style.active and style.upscale == "blend":  # SeedVR2 adds photo texture: keep only part of it
            try:
                up = media.blend_upscale(video, up, work / "upscaled_blend.mp4", alpha=style.blend, size=(w, h),
                                         fps=clip.fps)
                record["upscale"].update(blend=style.blend, note=stylelock.PENDING_NOTE)
            except Exception as exc:  # noqa: BLE001 - the plain SeedVR2 take is still fine
                yield Event("finish", "progress", 0.0, f"Style blend skipped ({str(exc)[:120]})")
        return up

    # -- render time: deliver / defer SeedVR2, framing routing + QC, prep reuse (Oct 19) ---------------------
    def _deliver(self, req: Request, work: Path, record, results, clip: media.PreparedClip, video: Path,
                 audio: Path | None, *, group: bool) -> Iterator[Event]:
        """The accepted silent take -> SeedVR2 (when on) -> export; or, with Request.defer_upscale, stop at the
        silent take and leave it for finish_pending (SeedVR2 then runs only on the takes that pass review)."""
        if req.defer_upscale and wants_seedvr2(req, group):
            yield from self._defer(req, work, record, clip, video, audio, group)
            return
        video = yield from self._upscale(req, clip, video, work, results, record, group=group)
        yield Event("finish", "start", message="Re-adding the audio and exporting" if audio else "Exporting")
        yield from self._finish_animated(req, work, record, clip, video, audio)

    def _defer(self, req: Request, work: Path, record, clip: media.PreparedClip, video: Path, audio: Path | None,
               group: bool) -> Iterator[Event]:
        doc = {
            "version": 1,
            "request": {k: _jsonable(v) for k, v in asdict(req).items()},
            "clip": {k: _jsonable(v) for k, v in asdict(clip).items()},
            "video": str(video), "audio": str(audio) if audio else None, "group": group,
            "record": {k: v for k, v in record.items() if not k.startswith("_") and k != "t0"},
            "elapsed_before": round(time.monotonic() - record["t0"], 1),
        }
        path = work / PENDING_FILE
        path.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
        yield Event("finish", "skipped", 1.0, "SeedVR2 deferred: review the silent take, then finish the ones "
                                              "that pass (--finish-pending)", artifact=video)
        yield Event("complete", "complete", 1.0, "Silent take ready for review (not upscaled or exported yet)",
                    artifact=video, data={"pending": str(path), "work_dir": record["top"], "duration": clip.seconds,
                                          "deferred": True})

    def finish_pending(self, tops: list[Path] | list[str]) -> Iterator[Event]:
        """Finish deferred takes (Request.defer_upscale): every SeedVR2 pass back to back first, with the exact
        graph and seed the inline path would have used, then every export. One ``complete`` event per take."""
        items: list[dict[str, Any]] = []
        for t in tops:
            t = Path(t)
            path = next((p for p in (t / WORK_SUBDIR / PENDING_FILE, t / PENDING_FILE) if p.exists()), None)
            if path is None:
                raise InputError(f"No deferred take in {t} (expected {WORK_SUBDIR}/{PENDING_FILE})")
            doc = json.loads(path.read_text(encoding="utf-8"))
            req = _request_from_dict(doc["request"])
            c = doc["clip"]
            clip = media.PreparedClip(video=Path(c["video"]), audio=Path(c["audio"]) if c.get("audio") else None,
                                      width=int(c["width"]), height=int(c["height"]), frames=int(c["frames"]),
                                      fps=int(c["fps"]), seconds=float(c["seconds"]))
            record = dict(doc["record"], _req=req, t0=time.monotonic() - float(doc.get("elapsed_before", 0.0)))
            items.append({"path": path, "work": path.parent, "req": req, "clip": clip, "record": record,
                          "video": Path(doc["video"]), "audio": Path(doc["audio"]) if doc.get("audio") else None,
                          "group": bool(doc.get("group"))})
        yield Event("finish", "start", message=f"SeedVR2 on {len(items)} accepted take(s), back to back")
        for it in items:  # phase 1: all SeedVR2 passes (one model load, no swap jobs in between)
            it["up"] = yield from self._upscale(it["req"], it["clip"], it["video"], it["work"], {}, it["record"],
                                                group=it["group"])
        for it in items:  # phase 2: exports (CPU)
            yield Event("finish", "start", message="Re-adding the audio and exporting" if it["audio"] else "Exporting")
            yield from self._finish_animated(it["req"], it["work"], it["record"], it["clip"], it["up"], it["audio"])
            it["path"].replace(it["path"].with_name(PENDING_FILE.replace(".json", ".done.json")))

    def _route_framing(self, req: Request, frame: Path, people: list, jobs: list, record) -> str:
        """Why this shot skips SCAIL-2 (framing.ROUTE_TO_ANIMATE), or '' to keep it. Never raises: without
        OpenCV / the face model the shot keeps today's engine choice."""
        if not req.route_framing or req.group_engine != "auto":
            return ""
        try:
            import numpy as np

            rgb = np.asarray(Image.open(frame).convert("RGB"))
            found = [framing.classify_person(rgb, people[i].box) for i, _ in jobs]
        except Exception as exc:  # noqa: BLE001 - routing is an optimisation, never a failure
            record["routing"] = {"error": str(exc)[:200]}
            return ""
        go, why = framing.route_to_animate([f["framing"] for f in found])
        record["routing"] = {"people": [dict(f, slot=i + 1) for f, (i, _) in zip(found, jobs)], "animate": go,
                             "why": why}
        return why

    def _take_qc(self, req: Request, clip: media.PreparedClip, people: list, jobs: list, take: Path, work: Path,
                 results, record) -> Iterator[Event]:
        """Framing QC of a SCAIL-2 take before SeedVR2 (Request.qc). Returns the verdict, or None when QC is off
        or couldn't run (the take is then delivered as before)."""
        if req.qc == "off":
            return None
        try:
            frame = media.grab_frame(take, work / "take_first_frame.png", 0, clip.fps)
            size = media.video_size(take)
            found = yield from self._scan(frame, work, results, record, "qc_people", size, req.tier)
        except Cancelled:
            raise
        except Exception as exc:  # noqa: BLE001 - QC must never sink a take
            record["qc"] = {"error": str(exc)[:200]}
            return None
        src = [people[i] for i, _ in jobs]
        matched = media.match_people(src, found) if found else [None] * len(src)
        soft = self._nonhuman_slots(jobs)
        verdict = framing.framing_qc([p.box for p in src], (clip.width, clip.height),
                                     [m.box if m else None for m in matched], size, soft_missing=soft)
        record["qc"] = dict(verdict, stage="framing", take=str(take))
        yield Event("swap", "progress", 1.0, f"Framing QC {'passed' if verdict['pass'] else 'failed'}: "
                                             f"{verdict['why']}")
        return verdict

    @staticmethod
    def _nonhuman_slots(jobs: list) -> set[int]:
        """Indexes (into jobs) whose reference is non-human (media.face_check). The people detector often misses a
        drawn animal or mascot in the take (Oct 20: the bunny was there but 'person 3 missing' sent a good SCAIL-2
        take to a 40 min Animate fallback), so framing QC reports those as unverified instead of failing."""
        out: set[int] = set()
        for k, job in enumerate(jobs):
            ref = job[1] if len(job) > 1 else None
            if not ref:
                continue
            try:
                nonhuman, _ = media.face_check(Path(str(ref)))
            except Exception:  # noqa: BLE001 - unreadable ref: keep the strict check
                continue
            if nonhuman:
                out.add(k)
        return out

    def _prep_key(self, req: Request, people: bool) -> dict[str, Any]:
        src = Path(str(req.clip))
        st = src.stat()
        key: dict[str, Any] = {
            "clip": str(src.resolve()), "clip_bytes": st.st_size, "clip_mtime_ns": st.st_mtime_ns,
            "clip_start": req.clip_start, "clip_seconds": req.clip_seconds, "resolution": req.resolution,
            "tier": tiers.normalize(req.tier), "erase_props": req.erase_props, "clip_audio": req.clip_audio,
            "keep_audio": req.keep_audio, "motion_fps": self.settings.motion_fps, "rotate": int(req.rotate or 0),
            "max_swap_seconds": self.settings.max_swap_seconds, "people": people,
        }
        if req.clip_audio not in ("original", "mute") and req.voice_sample:
            v = Path(str(req.voice_sample))
            key["voice"] = [str(v.resolve()), v.stat().st_size if v.exists() else None]
        return key

    def _reused_prep(self, req: Request, work: Path, record, *, people: bool) -> Iterator[Event]:
        """Request.reuse_prep: the previous run's prepared clip, VOID erase, people scan and soundtrack, copied
        byte for byte into this run (generator return value), or None to prepare from scratch."""
        if not req.reuse_prep:
            return None
        top = Path(str(req.reuse_prep))
        path = next((p for p in (top / WORK_SUBDIR / PREP_FILE, top / PREP_FILE) if p.exists()), None)
        if path is None:
            yield Event("prepare", "progress", 0.0, f"No saved prep in {top}; preparing from scratch")
            return None
        doc = json.loads(path.read_text(encoding="utf-8"))
        key = self._prep_key(req, people)
        old = doc.get("key") or {}
        diff = [k for k in key if old.get(k) != key[k]]
        if diff or (people and not doc.get("people")):
            yield Event("prepare", "progress", 0.0, f"The saved prep in {top.name} doesn't match this run "
                                                    f"({', '.join(diff) or 'no people scan'}); preparing from scratch")
            return None
        dst = work / "inputs" / "reused"
        dst.mkdir(parents=True, exist_ok=True)
        n = [0]

        def copy(p: str | None) -> Path | None:
            if not p:
                return None
            n[0] += 1
            return Path(shutil.copyfile(p, dst / f"{n[0]:02d}_{Path(p).name}"))

        c = doc["clip"]
        clip = media.PreparedClip(video=copy(c["video"]), audio=copy(c.get("audio")), width=int(c["width"]),
                                  height=int(c["height"]), frames=int(c["frames"]), fps=int(c["fps"]),
                                  seconds=float(c["seconds"]))
        found = [media.Person(mask=copy(p["mask"]), box=tuple(int(v) for v in p["box"]), cx=float(p["cx"]),
                              cy=float(p["cy"]), area=float(p["area"])) for p in doc.get("people") or []]
        audio = copy(doc.get("audio"))
        frame = copy(doc.get("frame"))
        record.update(doc.get("record") or {})
        record["clip"] = {k: _jsonable(v) for k, v in asdict(clip).items()}
        record["reused_prep"] = str(top)
        what = "clip" + (", VOID erase" if req.erase_props else "") + (", people scan" if people else "")
        yield Event("prepare", "skipped", 1.0, f"Retake: reusing the {what} and soundtrack from {top.name} "
                                               f"(same bytes, so only the seed changes)")
        return {"clip": clip, "audio": audio, "frame": frame, "people": found, "reused": True}

    def _clip_soundtrack(self, req: Request, clip: media.PreparedClip, work: Path, results, record,
                         prep: dict[str, Any] | None, *, frame: Path | None = None,
                         people: list | None = None) -> Iterator[Event]:
        """The soundtrack (reused on a retake), then save this run's prep for later retakes. Returns the audio."""
        if prep is not None:
            record["clip_audio"] = req.clip_audio
            yield Event("revoice", "skipped", 1.0, "Retake: reusing the soundtrack")
            return prep["audio"]
        yield from self._soundtrack(req, clip, work, results, record)
        audio = record.pop("_audio")
        try:
            doc = {"version": 1, "key": self._prep_key(req, people is not None),
                   "clip": {k: _jsonable(v) for k, v in asdict(clip).items()},
                   "audio": str(audio) if audio else None, "frame": str(frame) if frame else None,
                   "people": [{k: _jsonable(v) for k, v in asdict(p).items()} for p in people or []],
                   "record": {k: record[k] for k in ("resolution", "erase_props") if k in record}}
            (work / PREP_FILE).write_text(json.dumps(doc, indent=2), encoding="utf-8")
        except OSError:  # retakes just prepare from scratch
            pass
        return audio

    # -- Copy the Vibe -----------------------------------------------------------------
    def _run_vibe(self, req: Request, work: Path | None = None) -> Iterator[Event]:
        """Reference clip -> motion only -> you, doing it, in a brand-new scene."""
        s = self.settings
        tier = tiers.get(req.tier)
        m = tiers.models_for(tier, s.models)
        sw, sh = tier.scene_size
        scene = req.scene_prompt.strip() or DEFAULT_VIBE_SCENE
        work, record, results = self._begin(req, work, scene)
        yield from self._prepare_clip(req, work, s.max_vibe_seconds, record)
        clip: media.PreparedClip = record.pop("_clip")
        portrait = clip.height >= clip.width
        subject = resolve_subject(req, req.photo or req.avatar)
        record["subject"] = subject
        files: dict[str, Path] = {}
        if req.photo:
            files["photo"] = media.prepare_photo(req.photo, work / "inputs" / "photo.png",
                                                 keep_whole=subject == "avatar")

        # 1. you, in the NEW scene (FLUX.2 klein reference edit). The reference clip is not an input here.
        if req.avatar:
            avatar = media.fit_photo(req.avatar, work / "avatar.png", (clip.width * 2, clip.height * 2))
            yield Event("avatar", "skipped", 1.0, "Using your ready-made scene", artifact=avatar)
        else:
            hint = "" if req.style in ("none", "real") or keep_avatar_look(req, subject) else \
                f"{STYLES[req.style].label} style"
            prompt = vibe_scene_prompt(scene, portrait, hint, req.style_extra, avatar=subject == "avatar")
            sg = graphs.avatar_graph(prompt, models=m, reference=1, seed=req.seed, width=sw if portrait else sh,
                                     height=sh if portrait else sw, prefix="mc/vibe_scene", te_device=tier.te_device)
            yield Event("avatar", "start", message=f"Building your new scene: {scene}",
                        data={"eta_seconds": estimate_scene_seconds(req.tier)})
            yield from self._job("avatar", "avatar", sg, files, work, results, record)
            avatar = results["avatar"].path
            yield Event("avatar", "done", 1.0, "Scene ready", artifact=avatar)

        yield from self._soundtrack(req, clip, work, results, record)
        audio = record.pop("_audio")
        # 2. Animate "move": the clip contributes pose + face motion only
        keep = ("consistent character design, colors and art style" if subject == "avatar"
                else "consistent face, identity and outfit")
        who = "The character" if subject == "avatar" else "The person"
        motion = (f"{scene}. {who} performs the movements of the pose sequence with the same timing and energy, "
                  f"full body, smooth natural motion, {keep}, the camera stays steady.")
        yield from self._animate_chunks("move", "vibe", motion, clip, avatar, req, work, results, record,
                                        negative=graphs.wan_negative(subject == "avatar"))
        video = Path(record.pop("animated"))
        record["scene_prompt"] = scene
        yield Event("finish", "start", message="Exporting" + (" with the clip's audio" if audio else " (silent)"))
        yield from self._finish_animated(req, work, record, clip, video, audio)

    # -- batch -------------------------------------------------------------------------
    def run_batch(self, base: Request, scripts: list[str]) -> Iterator[Event]:
        """Many scripts in, many clips out. The avatar is generated once and
        reused so every clip shows the same character."""
        scripts = [x.strip() for x in scripts if x.strip()]
        if not scripts:
            raise InputError("Add at least one script (one per line, or blocks separated by a blank line).")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        root = self.settings.output_dir / f"{stamp}-batch"
        avatar = base.avatar
        finals: list[str] = []
        for i, script in enumerate(scripts):
            req = replace(base, script=script, avatar=avatar)
            for ev in self.run(req, work=root / f"{i + 1:02d}-{_slug(script)}"):
                ev.data = {**ev.data, "batch_index": i, "batch_total": len(scripts)}
                if ev.stage == "avatar" and ev.artifact and avatar is None:
                    avatar = ev.artifact
                if ev.status == "complete":
                    finals.append(str(ev.artifact))
                    ev = replace(ev, stage="batch_item", status="done")
                yield ev
        (root / "batch.json").write_text(json.dumps({"clips": finals}, indent=2), encoding="utf-8")
        yield Event("complete", "complete", 1.0, f"Batch done: {len(finals)} clips", artifact=Path(finals[-1]),
                    data={"clips": finals, "work_dir": str(root)})


# --------------------------------------------------------------------------- #
# Copy the Vibe prompts
# --------------------------------------------------------------------------- #
DEFAULT_VIBE_SCENE = "a bright, clean photo studio with a soft colored backdrop"
VIBE_SCENE_IDEAS = [
    "luxury hotel hallway at night",
    "rooftop at golden hour with the city behind",
    "neon-lit arcade",
    "empty basketball court at sunset",
    "cozy kitchen on a rainy morning",
    "school gym with streamers and balloons",
    "retro 80s mall",
    "snowy forest clearing",
]


FLAT_STYLE = ("flat illustrated look, cel-shaded with clean outlines, keep the original colors of the reference, "
              "not photorealistic, not a 3D render")


# Exact-copy identity lines (default for every reference upload): the reference is copied, never reinterpreted.
EXACT_COPY_AVATAR = ("It is an exact copy of the reference: the same face, hair, outfit, colors, proportions and art "
                     "style, drawn the same way, nothing added, removed or redesigned.")
EXACT_COPY_PHOTO = ("They are an exact copy of the person in the reference: the same face, hair, skin tone, body "
                    "shape, outfit and colors.")


def swap_prompt(req: Request, subject: str = "photo", flat: bool = False, slot: int = 0) -> str:
    """Animate swap prompt for one person: motion line, that person's look (Request.people_desc[slot], optional)
    and an exact-copy identity line, so the reference is copied rather than reinterpreted."""
    descs = req.people_desc or []
    desc = str(descs[slot]).strip().rstrip(".") if slot < len(descs) and descs[slot] else ""
    if subject == "avatar":
        prompt = ("The character from the reference image moves naturally, matching the original performance, and "
                  "keeps its exact design, outfit, colors and art style from the reference.")
        if desc:
            prompt += f" The character is {desc}."
        prompt += f" {EXACT_COPY_AVATAR}"
        lead = stylelock.settings_for(req.style_lock, [subject]).positive_prefix
        if lead:  # prompt A/B (MC_STYLE_POSITIVE): style tags first, replacing the default flat line
            prompt = f"{lead}. {prompt}"
        elif flat:
            prompt += f" {FLAT_STYLE[0].upper()}{FLAT_STYLE[1:]}."
    else:
        prompt = "The person moves naturally, matching the original performance."
        if desc:
            prompt += f" The person is {desc}."
        prompt += f" {EXACT_COPY_PHOTO}"
    if req.remove_props:
        prompt += (" They wear only their own outfit from the reference: nothing on their back, no backpack, "
                   "no props or gear from the original character.")
    tweak = req.swap_prompt.strip()
    if tweak:  # scene/style tweak: lighting, mood, film look... (the background itself comes from your clip)
        prompt = f"{prompt} {tweak}."
    return prompt


PENDING_FILE = "pending_upscale.json"  # Request.defer_upscale: a silent take waiting for SeedVR2 + export
PREP_FILE = "prep.json"  # Request.reuse_prep: what a retake can reuse byte for byte


def wants_seedvr2(req: Request, group: bool = False) -> bool:
    """Whether the take gets SeedVR2 (see Request.upscale): auto = SCAIL-2 group takes only; never on Lite."""
    mode = req.upscale if req.upscale in ("seedvr2", "lanczos") else ("seedvr2" if group else "lanczos")
    return mode == "seedvr2" and tiers.get(req.tier).key != "lite"


def _request_from_dict(d: dict[str, Any]) -> Request:
    """Request back from ``asdict`` + _jsonable (pending_upscale.json): upload fields become Paths again."""
    names = {f for f in Request.__dataclass_fields__}
    kw = {k: v for k, v in d.items() if k in names}
    for k in (*_UPLOADS, "reuse_prep"):
        if kw.get(k):
            kw[k] = Path(kw[k])
    if kw.get("people"):
        kw["people"] = [Path(x) if x else None for x in kw["people"]]
    return Request(**kw)


def group_engine_for(req: Request, frames: int, swapped: int, found: int | None = None) -> tuple[str, str]:
    """('scail' | 'chain', why) for a multi-character swap (see Request.group_engine). ``found`` = people on the
    first frame: SCAIL-2 redraws the whole frame, so it only runs when everyone is swapped (people you leave
    alone must stay pixel-identical, which the masked Animate passes guarantee)."""
    if req.group_engine == "chain":
        return "chain", "you picked one pass per person"
    if tiers.get(req.tier).key == "lite":
        return "chain", "SCAIL-2 needs a 16 GB card (Standard or Max)"
    if swapped < (1 if req.group_engine == "scail" else 2):
        return "chain", "only one person to swap"
    if swapped > graphs.SCAIL_MAX_PEOPLE:
        return "chain", f"SCAIL-2 tells people apart for up to {graphs.SCAIL_MAX_PEOPLE} at once"
    descs = [d for d in req.people_desc if str(d or "").strip()]
    if req.group_engine != "scail" and len(descs) < swapped:
        return "chain", "describe each person's look to use the one-pass SCAIL-2 group swap"
    if found is not None and found > swapped:
        return "chain", "someone stays as filmed, and only masked passes keep them pixel-identical"
    if frames > graphs.SCAIL_MAX_FRAMES and not req.scail_chunks:
        return "chain", f"clips over ~{(graphs.SCAIL_MAX_FRAMES - 1) // 16} s use one pass per person for now"
    if frames > graphs.SCAIL_MAX_FRAMES:
        n = len(graphs.plan_scail_chunks(frames))
        return "scail", f"everyone in SCAIL-2, {n} chained chunks (long clip, experimental)"
    return "scail", "everyone in one SCAIL-2 pass"


def group_swap_prompt(req: Request, n: int, subjects: list[str], slots: list[int] | None = None) -> str:
    """SCAIL-2 prompt: the motion line, then each person's look left to right (Request.people_desc), then an
    identity line. The per-person looks are what make SCAIL-2 copy the references instead of inventing people."""
    who = "characters" if any(x == "avatar" for x in subjects) else "people"
    one = n == 1
    prompt = (("The character from the reference image moves" if one else f"The {n} {who} from the reference image move")
              + " naturally, matching the original performance in the original scene.")
    slots = slots if slots is not None else list(range(n))
    descs = [str(req.people_desc[i]).strip() if i < len(req.people_desc) and req.people_desc[i] else "" for i in slots]
    if any(descs):
        places = (["In the frame is"] if one else
                  ["On the left is"] + ["Next to them is"] * max(0, n - 2) + ["On the right is"])
        prompt += " " + " ".join(f"{places[k]} {d.rstrip('.')}." for k, d in enumerate(descs) if d)
    prompt += (" Each one is an exact copy of their reference: the same face, hair, outfit, colors, proportions and "
               "art style, drawn the same way" if any(x == "avatar" for x in subjects) else
               " Each one is an exact copy of their reference: the same face, hair, outfit and colors") + "."
    if req.remove_props:
        prompt += " They wear only their own outfits: nothing on their backs, no props from the original people."
    tweak = req.swap_prompt.strip()
    return f"{prompt} {tweak}." if tweak else prompt


def swap_negative(req: Request, subject: str) -> str:
    neg = graphs.wan_negative(subject == "avatar", PROPS_NEGATIVE if req.remove_props else "")
    style = stylelock.settings_for(req.style_lock, [subject])
    return stylelock.merge_negative(neg, style.negative_extra) if style.active else neg


def swap_settings(req: Request, subject: str, flat: bool) -> dict[str, Any]:
    """What the swap actually used (goes into project.json / the recipe for honest comparisons)."""
    return {"subject": subject, "flat_art": flat, "relight": use_relight(req, subject), "resolution": req.resolution,
            "erase_props": req.erase_props,
            "isolate_reference": subject == "avatar", "remove_props": bool(req.remove_props),
            "mask_grow": MASK_GROW_PROPS if req.remove_props else MASK_GROW}


def vibe_scene_prompt(scene: str, portrait: bool, style_hint: str = "", extra: str = "", avatar: bool = False) -> str:
    frame = "vertical 9:16 frame" if portrait else "horizontal 16:9 frame"
    if avatar:
        return (f"Put the character from image 1 in this scene: {scene.strip()}. Full body visible from head to feet, "
                f"standing in a relaxed neutral pose facing the camera, arms slightly away from the body, wide shot, "
                f"{frame}, room to move around them. The character is an exact copy of image 1: keep its exact face, design, art "
                f"style, proportions, colors and outfit, and render the whole scene in that same art style, not "
                f"photorealistic{', ' + style_hint if style_hint else ''}. No text, no watermark, the character is "
                f"the only one in the frame. {extra}").strip()
    look = style_hint or "Photorealistic, natural skin texture"
    return (f"Put the person from image 1 in this scene: {scene.strip()}. Full body visible from head to feet, "
            f"standing in a relaxed neutral pose facing the camera, arms slightly away from the body, wide shot, "
            f"{frame}, room to move around them. Keep the person's exact face, identity, hairstyle, skin tone and body "
            f"shape from image 1. {look}, no text, no watermark, the person is the only person in the frame. "
            f"{extra}").strip()


# --------------------------------------------------------------------------- #
# Remixable recipe JSON (saved next to every output)
# --------------------------------------------------------------------------- #
RECIPE_VERSION = 1
_MODEL_KEYS = ("unet_name", "ckpt_name", "lora_name", "clip_name", "vae_name", "name", "model_name")
_RECIPE_SETTINGS = ("style", "style_extra", "steps", "seed", "hd", "watermark", "captions", "smooth", "seam",
                    "segment_seconds", "ext_frames", "expressiveness", "meme", "meme_top", "meme_bottom", "say_line",
                    "line", "custom_scene", "custom_motion", "effect", "era", "meme_frames", "clip_start",
                    "clip_seconds", "swap_prompt", "relight", "remove_props", "resolution", "erase_props", "scene_prompt", "keep_audio", "clip_audio", "subject", "voice_mode", "script",
                    "character", "song_start", "song_seconds")
_UPLOADS = ("photo", "photo_b", "avatar", "voice_sample", "voice_sample_b", "recording", "character_voice", "song",
            "vocals", "clip")


def graph_prompts_and_models(sg: graphs.StageGraph) -> tuple[list[str], list[str]]:
    """Positive prompts and model files a job used (for the recipe; never file paths of your uploads)."""
    prompts: list[str] = []
    used: list[str] = []
    for n in sg.graph.values():
        title = n.get("_meta", {}).get("title", "")
        for k, v in n["inputs"].items():
            if not isinstance(v, str):
                continue
            if k in _MODEL_KEYS and "." in v:
                used.append(v)
            elif k in ("text", "prompt") and "negative" not in title.lower() and n["class_type"] != "ChatterboxTTS":
                prompts.append(v)
    return prompts, sorted(set(used))


def write_recipe(work: Path, req: Request, record: dict[str, Any]) -> Path:
    """``mc_recipe.json``: everything needed to remix this output except your own uploads
    (format, prompts, models, seed, settings). Paths of your files are NOT stored, only which kinds were used."""
    doc = {
        "mc_recipe": RECIPE_VERSION,
        "mode": req.mode,
        "format": record.get("recipe"),
        "settings": {k: getattr(req, k) for k in _RECIPE_SETTINGS},
        "seed": req.seed,
        "prompts": record.get("prompts", {}),
        "models": record.get("models", {}),
        "chunks": record.get("chunks"),
        "tier": req.tier,  # informational: a remix renders on YOUR tier (cli/ui resolve it from your GPU)
        "uploads_used": [k for k in _UPLOADS if getattr(req, k)],
        "comfy_jobs": record.get("jobs", {}),
        "backend": record.get("backend"),
        "created": record.get("started"),
        "remix": ("python -m main_character_comfy.cli --remix mc_recipe.json "
                  + " ".join(f"--{k.replace('_sample', '').replace('_', '-')} <yours>" for k in _UPLOADS
                             if getattr(req, k) and k in ("photo", "voice_sample", "clip"))
                  + " --i-have-rights").replace("  ", " "),
    }
    path = work / "mc_recipe.json"
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def request_from_recipe(doc: dict[str, Any], **overrides: Any) -> Request:
    """Rebuild a Request from ``mc_recipe.json`` (your uploads go in ``overrides``).
    A recipe whose format isn't in your library is replayed as a custom format."""
    if doc.get("mc_recipe") != RECIPE_VERSION:
        raise InputError("Not an MC recipe file (or a newer version).")
    known = {f.name for f in Request.__dataclass_fields__.values()}
    st = {k: v for k, v in (doc.get("settings") or {}).items() if k in known and k not in _UPLOADS}
    req = Request(mode=doc.get("mode", "meme"), **st)
    fmt = doc.get("format") or {}
    if req.mode == "meme" and req.meme != "custom" and req.meme not in memes.RECIPES and fmt.get("scene"):
        scene = fmt["scene"].replace("Put the person from image 1 in this scene: ", "")
        req = replace(req, meme="custom", custom_scene=scene, custom_motion=fmt.get("motion", ""))
    return replace(req, **overrides)


def split_batch_text(text: str) -> list[str]:
    """Blank-line separated blocks if present, else one script per line."""
    if re.search(r"\n\s*\n", text):
        return [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    return [ln.strip() for ln in text.splitlines() if ln.strip()]
