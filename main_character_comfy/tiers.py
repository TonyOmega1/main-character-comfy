"""Hardware tiers: what MC runs, and how, on your GPU.

* **Lite** (8-12 GB): FLUX.2 klein fp8 stills with the text encoder on the CPU,
  meme motion with Wan2.2 TI2V 5B (GGUF Q8_0, 5.4 GB) at 480x832 / 24 fps, ~3 s.
  Talking avatars, Swap Me In and Copy the Vibe need 16 GB+: they're gated with a
  friendly message (or run *experimental* with smaller quants if you insist).
* **Standard** (16 GB, the reference RTX 5080): the defaults everything was built on.
* **Max** (24 GB+): Q8_0 quants that fit without offload, 576x1024 motion,
  longer Swap / Vibe clips and longer lip-sync segments.

The tier is picked automatically from the GPU's VRAM (``nvidia-smi`` on the PC
running ComfyUI, or ``MC_VRAM_GB``), can be forced with ``MC_TIER`` /
``--tier``, and toggled in the UI. comfy-api-proxy 0.1.6 / comfy-sdk 0.5.0 don't
expose ComfyUI's system stats, and MC never calls ComfyUI's own API directly, so
the VRAM probe is local (MC and ComfyUI run on the same PC).

Every timing here is an ESTIMATE until measured (see PC_TODO.md).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace

from .config import Models

TIER_KEYS = ("lite", "standard", "max")
DEFAULT_TIER = "standard"
LEGACY_KEYS = {"12gb": "lite", "16gb": "standard", "24gb": "max"}  # old --profile names

# What needs a 16 GB+ card (Lite gates these unless allow_heavy is set).
HEAVY = {
    "talk": "Talking avatars (Wan2.1 + InfiniteTalk)",
    "dialogue": "Duets (Wan2.1 + InfiniteTalk)",
    "sing": "Singing (Wan2.1 + InfiniteTalk)",
    "swap": "Swap Me In (Wan2.2 Animate 14B)",
    "vibe": "Copy the Vibe (Wan2.2 Animate 14B)",
    "lipsync": "Saying the line (Wan2.1 + InfiniteTalk lip sync)",
}


@dataclass(frozen=True)
class Tier:
    key: str
    label: str
    vram: str
    gpus: str
    comfy_flags: str  # recommended ComfyUI launch flags (run_nvidia_gpu.bat / main.py)
    flags_note: str
    notes: str
    eta_card: str  # the card the ETAs below assume
    # stills: FLUX.2 klein 4B
    klein_fp8: bool = False  # use the 4.1 GB fp8 checkpoint instead of bf16 (7.8 GB)
    te_device: str = "default"  # "cpu": text encoders run on the CPU (CLIPLoader device), VRAM stays free
    scene_size: tuple[int, int] = (1088, 1920)  # 9:16, multiples of 16
    # meme motion
    motion_engine: str = "a14b"  # "a14b" = Wan2.2 I2V A14B (2 experts) | "ti2v5b" = Wan2.2 TI2V 5B
    motion_size: tuple[int, int] = (480, 832)
    motion_fps: int = 16
    motion_max_seconds: float = 5.0
    motion_steps: int = 4
    tiled_decode: bool = False  # VAEDecodeTiled (spatial + temporal tiles) to cap the decode spike
    # lip sync (Wan2.1 I2V 14B + InfiniteTalk)
    wan_gguf: str = "wan2.1-i2v-14b-480p-Q5_K_M.gguf"
    segment_seconds: float = 14.5  # segment + 0.24 s overlap fills whole 81-frame windows
    steps: int = 6
    # Swap Me In / Copy the Vibe (Wan2.2 Animate 14B)
    animate_gguf: str = "Wan2.2-Animate-14B-Q4_K_M.gguf"
    animate_size: tuple[int, int] = (480, 832)
    animate_cap_scale: float = 1.0  # x Settings.max_swap_seconds / max_vibe_seconds (30 s)
    # Wan2.2 I2V A14B expert quants ("" = Models default, Q5_K_M)
    wan22_quant: str = ""
    gated: frozenset[str] = field(default_factory=frozenset)
    # seconds on eta_card. Standard defaults MEASURED on the RTX 5080 (Oct 16 2026); see pipeline.py.
    scene_seconds: float = 12
    motion_seconds: float = 115  # one default-length motion clip (motion_ref_frames)
    motion_ref_frames: int = 81
    animate_window_seconds: float = 130
    lipsync_factor: float = 1.0  # x the measured RTX 5080 lip-sync speed

    @property
    def needs_allow_heavy(self) -> bool:
        return bool(self.gated)


TIERS: dict[str, Tier] = {
    t.key: t
    for t in [
        Tier(
            "lite", "Lite · 8-12 GB", "8-12 GB", "RTX 3060 12G / 4060 Ti 8G / 4070 / 5070",
            "--reserve-vram 1.0",
            "ComfyUI's Dynamic VRAM (on by default for NVIDIA in v0.41) streams what doesn't fit from RAM; --lowvram "
            "does nothing while it's on. GGUF loaders still use the classic partial offload. If you hit out-of-"
            "memory: add --disable-dynamic-vram --lowvram (classic offload, text encoders on CPU); with 16 GB of "
            "system RAM add --cache-none.",
            "Stills: FLUX.2 klein fp8 + text encoder on the CPU. Motion: Wan2.2 TI2V 5B Q8_0 at 480x832, ~3 s. "
            "Talking, Swap Me In and Copy the Vibe need 16 GB+ (experimental here: smaller quants, 10 s cap).",
            "a 12 GB card (RTX 4070 class)",
            klein_fp8=True, te_device="cpu", scene_size=(768, 1360),
            motion_engine="ti2v5b", motion_size=(480, 832), motion_fps=24, motion_max_seconds=3.0, motion_steps=20,
            tiled_decode=True,
            wan_gguf="wan2.1-i2v-14b-480p-Q3_K_M.gguf", segment_seconds=8.7,
            animate_gguf="Wan2.2-Animate-14B-Q3_K_M.gguf", animate_size=(384, 672), animate_cap_scale=1 / 3,
            gated=frozenset(HEAVY),
            # scene/motion MEASURED on the 5080 with --reserve-vram 4.5 (~12 GB card): klein fp8 16 s,
            # TI2V 5B Q8_0 73 frames 83-103 s. A real 12 GB card is slower (less bandwidth): +25% margin.
            # animate_window_seconds / lipsync_factor are still estimates.
            scene_seconds=20, motion_seconds=120, motion_ref_frames=73, animate_window_seconds=230,
            lipsync_factor=2.5,
        ),
        Tier(
            "standard", "Standard · 16 GB", "16 GB", "RTX 5080 (reference) / 4080 / 4070 Ti Super / 5060 Ti 16G",
            "--reserve-vram 1.5",
            "Dynamic VRAM handles partial offload of the Q5/Q4 GGUF blocks automatically.",
            "Everything on: FLUX.2 klein bf16, Wan2.2 I2V A14B Q5_K_M motion at 480x832, InfiniteTalk lip sync, "
            "Swap Me In / Copy the Vibe up to 30 s. Lip sync measured: ~212 s per 4.8 s window pair.",
            "a 16 GB RTX 5080",
        ),
        Tier(
            "max", "Max · 24 GB+", "24 GB+", "RTX 3090 / 4090 / 5090",
            "",
            "Defaults are right: don't add --highvram (it turns Dynamic VRAM off).",
            "Q8_0 quants that fit without offload, 576x1024 meme motion and Animate, Swap / Vibe up to 60 s, "
            "20 s lip-sync segments.",
            "a 24 GB RTX 4090",
            motion_size=(576, 1024), wan22_quant="Q8_0",
            wan_gguf="wan2.1-i2v-14b-480p-Q8_0.gguf", segment_seconds=20.0,
            animate_gguf="Wan2.2-Animate-14B-Q8_0.gguf", animate_size=(576, 1024), animate_cap_scale=2.0,
            # ESTIMATES (no 24 GB card to test): scaled from the measured Standard numbers.
            scene_seconds=9, motion_seconds=135, animate_window_seconds=150, lipsync_factor=0.8,
        ),
    ]
}


def normalize(key: str | None) -> str:
    """'Lite', '12gb' -> 'lite'. Raises ValueError for unknown names."""
    k = (key or "").strip().lower()
    k = LEGACY_KEYS.get(k, k)
    if k not in TIERS:
        raise ValueError(f"unknown tier {key!r} (use lite, standard or max)")
    return k


def get(key: str | None) -> Tier:
    return TIERS[normalize(key or DEFAULT_TIER)]


def tier_for_vram(gb: float | None) -> str:
    """16 GB cards report ~15.9 GiB, 12 GB ~12.0, 24 GB ~24.0."""
    if gb is None:
        return DEFAULT_TIER
    if gb < 14:
        return "lite"
    if gb < 22:
        return "standard"
    return "max"


@dataclass(frozen=True)
class TierChoice:
    key: str
    source: str  # "cli" | "MC_TIER" | "auto" | "default"
    vram_gb: float | None = None
    gpu: str = ""
    warning: str = ""

    @property
    def tier(self) -> Tier:
        return TIERS[self.key]

    def describe(self) -> str:
        t = self.tier
        if self.source == "auto":
            how = f"auto: {self.vram_gb:.0f} GB" + (f" {self.gpu}" if self.gpu else "")
        elif self.source == "default":
            how = "couldn't read VRAM, assuming 16 GB"
        else:
            how = f"forced with {'--tier' if self.source == 'cli' else 'MC_TIER'}"
        return f"MC {t.label} ({how})"


def resolve(explicit: str | None = None) -> TierChoice:
    """--tier beats MC_TIER beats auto-detect; unknown VRAM -> Standard."""
    from .system import gpu_name, vram_gb

    gb, gpu = vram_gb(), gpu_name()
    if explicit:
        return TierChoice(normalize(explicit), "cli", gb, gpu)
    env = os.environ.get("MC_TIER", "").strip()
    warning = ""
    if env and env.lower() != "auto":
        try:
            return TierChoice(normalize(env), "MC_TIER", gb, gpu)
        except ValueError:
            warning = f"Ignoring MC_TIER={env!r} (use lite, standard, max or auto)."
    if gb is None:
        return TierChoice(DEFAULT_TIER, "default", None, gpu, warning)
    if gb < 7.5:
        warning = (warning + " " if warning else "") + (
            f"Only {gb:.0f} GB of VRAM: below MC Lite's 8 GB minimum. Stills should work; motion may not.")
    return TierChoice(tier_for_vram(gb), "auto", gb, gpu, warning)


def gate(tier: str | Tier, mode: str, *, say_line: bool = False, allow_heavy: bool = False) -> str | None:
    """Friendly 'needs 16 GB+' message, or None if this tier can run it."""
    t = tier if isinstance(tier, Tier) else get(tier)
    what = "lipsync" if (mode == "meme" and say_line) else mode
    if what not in t.gated or allow_heavy:
        return None
    return (f"{HEAVY[what]} needs a 16 GB+ GPU, and you're on MC {t.label}. Silent memes and stills work great "
            f"on Lite. Got a 16 GB card? Switch the tier to Standard. Want to try anyway? Tick 'Try 16 GB modes "
            f"anyway' (experimental: smaller quants, slow, may run out of memory).")


_MODEL_ENV = {
    "wan_gguf": "MC_WAN_GGUF",
    "animate_gguf": "MC_ANIMATE_GGUF",
    "klein_unet": "MC_KLEIN_UNET",
    "wan22_high": "MC_WAN22_HIGH",
    "wan22_low": "MC_WAN22_LOW",
}


def models_for(tier: str | Tier, base: Models) -> Models:
    """The tier's model files. A filename you set yourself (MC_* env var) always wins."""
    t = tier if isinstance(tier, Tier) else get(tier)
    want: dict[str, str] = {"wan_gguf": t.wan_gguf, "animate_gguf": t.animate_gguf}
    if t.klein_fp8:
        want["klein_unet"] = base.klein_unet_lite
    if t.wan22_quant:
        want["wan22_high"] = f"Wan2.2-I2V-A14B-HighNoise-{t.wan22_quant}.gguf"
        want["wan22_low"] = f"Wan2.2-I2V-A14B-LowNoise-{t.wan22_quant}.gguf"
    changes = {k: v for k, v in want.items() if not os.environ.get(_MODEL_ENV[k], "").strip()}
    return replace(base, **changes) if changes else base


def motion_frames(tier: str | Tier, requested: int | None, default: int = 81) -> tuple[int, int]:
    """Meme-motion length for this tier as (frames, fps). ``requested`` is in 16 fps
    frames (49 = ~3 s, 81 = ~5 s, the UI's choices); Lite caps it at ~3 s at 24 fps."""
    t = tier if isinstance(tier, Tier) else get(tier)
    seconds = (int(requested or default) - 1) / 16
    seconds = min(seconds, t.motion_max_seconds)
    frames = max(5, int(round(seconds * t.motion_fps)) // 4 * 4 + 1)  # Wan wants 4n+1
    return frames, t.motion_fps


def animate_cap(tier: str | Tier, base_cap: float) -> float:
    t = tier if isinstance(tier, Tier) else get(tier)
    return round(base_cap * t.animate_cap_scale, 1)


# Multi-character Swap Me In (open-weights "Openjutsu"): one sequential Animate pass per swapped person, so the
# cost scales with the count. Max goes all the way to Genjutsu's 8 references.
MAX_PEOPLE = {"lite": 1, "standard": 2, "max": 8}
PEOPLE_HARD_CAP = 8  # person slots in the UI / CLI (Genjutsu parity: 1-8 reference images)
TRY_ANYWAY_PEOPLE = 4  # Lite / Standard with 'try anyway': each extra pass is another full render


def people_cap(tier: str | Tier, allow_heavy: bool = False) -> int:
    t = tier if isinstance(tier, Tier) else get(tier)
    base = MAX_PEOPLE.get(t.key, 1)
    return max(base, TRY_ANYWAY_PEOPLE) if allow_heavy else base


# Swap Me In / Copy the Vibe render resolution (short side x long side, multiples of 16).
# "auto" = the tier's default (Lite 384p, Standard 480p = 832x480, Max 576p). Finishing still upscales to 1080p.
RESOLUTIONS = ("auto", "480p", "720p")
RESOLUTION_SIZES = {"480p": (480, 832), "720p": (720, 1280)}
VRAM_720P_NOTE = ("720p on a 16 GB card is experimental: ~2.3x the pixels of 480p and measured at ~2.7x the render "
                  "time on an RTX 5080 (5 s clip: 15 min vs 6 min) and possible out-of-memory on long clips (ComfyUI's Dynamic VRAM offloads, which is slow). "
                  "Short clips first.")


def normalize_resolution(res: str | None) -> str:
    r = (res or "auto").strip().lower()
    r = {"480": "480p", "720": "720p", "default": "auto", "": "auto"}.get(r, r)
    if r not in RESOLUTIONS:
        raise ValueError(f"unknown resolution {res!r} (use {', '.join(RESOLUTIONS)})")
    return r


def animate_size_for(tier: str | Tier, resolution: str | None = "auto") -> tuple[int, int]:
    """(short, long) render size for Swap Me In / Copy the Vibe at this tier and resolution choice."""
    t = tier if isinstance(tier, Tier) else get(tier)
    r = normalize_resolution(resolution)
    return t.animate_size if r == "auto" else RESOLUTION_SIZES[r]


def resolution_gate(tier: str | Tier, resolution: str | None, allow_heavy: bool = False) -> str | None:
    """720p needs Standard (with a warning) or Max; Lite only with 'try anyway'."""
    t = tier if isinstance(tier, Tier) else get(tier)
    if normalize_resolution(resolution) == "720p" and t.key == "lite" and not allow_heavy:
        return ("720p needs a 16 GB+ card (Standard, experimental) or 24 GB+ (Max). On Lite use auto/480p, or tick "
                "'Try 16 GB modes anyway'.")
    return None


def resolution_warning(tier: str | Tier, resolution: str | None) -> str:
    """Non-blocking heads-up shown before a 720p render on a card under 24 GB."""
    t = tier if isinstance(tier, Tier) else get(tier)
    return VRAM_720P_NOTE if normalize_resolution(resolution) == "720p" and t.key != "max" else ""


def pixel_scale(tier: str | Tier, resolution: str | None) -> float:
    """Render-time multiplier vs the tier's default size (time ~ pixels; ESTIMATE until measured)."""
    t = tier if isinstance(tier, Tier) else get(tier)
    w, h = animate_size_for(t, resolution)
    bw, bh = t.animate_size
    return (w * h) / (bw * bh)


def people_gate(tier: str | Tier, n_swapped: int, allow_heavy: bool = False) -> str | None:
    """Friendly message when a tier can't swap ``n_swapped`` people (None = fine)."""
    t = tier if isinstance(tier, Tier) else get(tier)
    cap = people_cap(t, allow_heavy)
    if n_swapped <= cap:
        return None
    if allow_heavy or t.key == "max":
        return f"MC {t.label} swaps up to {cap} people per clip; leave the rest unchanged ('-')."
    return (f"MC {t.label} swaps up to {cap} {'person' if cap == 1 else 'people'} per clip (each one is another "
            f"full Animate pass). Leave some slots unchanged, switch to Max (up to {MAX_PEOPLE['max']}), or tick "
            f"'Try 16 GB modes anyway' (experimental: up to {TRY_ANYWAY_PEOPLE}, slow, may run out of memory).")
