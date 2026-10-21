"""ComfyUI API-format graph builders for each MC stage.

Each builder returns a :class:`StageGraph`: the raw graph plus the node ids the
pipeline needs (where to plug in uploaded assets, which node holds the output).
The graphs mirror the workflows that were hand-tested on an RTX 5080 16 GB with
ComfyUI v0.41 (see workflows/*.json, which are exported from these builders by
``python -m main_character_comfy.export_workflows``).
"""

from __future__ import annotations

import os

import math
from dataclasses import dataclass, field
from typing import Any

from .config import Models

# --------------------------------------------------------------------------- #
# Video decode: tiled by default (spatial + temporal tiles). A plain VAEDecode of a long
# video latent spikes VRAM and has crashed a 16 GB card's display driver, so every video
# graph decodes through core VAEDecodeTiled. Override with env vars (values snap to the
# node's limits): MC_DECODE_TILE (256), MC_DECODE_OVERLAP (32), MC_DECODE_TEMPORAL (32),
# MC_DECODE_TEMPORAL_OVERLAP (4). MC_DECODE_TILED=0 goes back to plain VAEDecode (big cards only).
# --------------------------------------------------------------------------- #
DECODE_DEFAULTS = {"tile_size": 256, "overlap": 32, "temporal_size": 32, "temporal_overlap": 4}
_DECODE_ENV = {"tile_size": "MC_DECODE_TILE", "overlap": "MC_DECODE_OVERLAP",
               "temporal_size": "MC_DECODE_TEMPORAL", "temporal_overlap": "MC_DECODE_TEMPORAL_OVERLAP"}
# (min, step) from core VAEDecodeTiled's object_info
_DECODE_LIMITS = {"tile_size": (64, 32), "overlap": (0, 32), "temporal_size": (8, 4), "temporal_overlap": (4, 4)}


def decode_settings() -> dict[str, int] | None:
    """Tile settings for video decodes, or None when MC_DECODE_TILED=0 (plain VAEDecode)."""
    if os.environ.get("MC_DECODE_TILED", "").strip().lower() in ("0", "off", "false", "no"):
        return None
    out_: dict[str, int] = {}
    for k, dflt in DECODE_DEFAULTS.items():
        raw = os.environ.get(_DECODE_ENV[k], "").strip()
        try:
            v = int(raw) if raw else dflt
        except ValueError:
            v = dflt
        lo, step = _DECODE_LIMITS[k]
        v = max(lo, (v // step) * step)
        out_[k] = v
    out_["overlap"] = min(out_["overlap"], out_["tile_size"] // 4 // 32 * 32)  # overlap at most a quarter tile
    out_["temporal_overlap"] = max(4, min(out_["temporal_overlap"], out_["temporal_size"] // 2 // 4 * 4))
    return out_


def video_decode(g: Any, samples: Any, vae: Any, title: str = "Tiled decode (low VRAM)") -> Any:
    """Decode a VIDEO latent: core VAEDecodeTiled with decode_settings(); plain VAEDecode only if turned off."""
    st = decode_settings()
    if st is None:
        return g.add("VAEDecode", samples=samples, vae=vae)
    return g.add("VAEDecodeTiled", title, samples=samples, vae=vae, **st)

Link = list  # [node_id, output_index]


class GraphBuilder:
    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self._next = 0

    def add(self, class_type: str, title: str | None = None, **inputs: Any) -> str:
        self._next += 1
        nid = str(self._next)
        node: dict[str, Any] = {"class_type": class_type, "inputs": inputs}
        if title:
            node["_meta"] = {"title": title}
        self.nodes[nid] = node
        return nid


def out(node_id: str, slot: int = 0) -> Link:
    return [node_id, slot]


@dataclass
class StageGraph:
    graph: dict[str, Any]
    # logical input name -> (node_id, input field); the pipeline uploads a file
    # through the Comfy SDK and plugs the asset handle in here.
    inputs: dict[str, tuple[str, str]] = field(default_factory=dict)
    output_node: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Stage 1: avatar (FLUX.2 klein 4B)
# --------------------------------------------------------------------------- #
AVATAR_W, AVATAR_H = 1088, 1920  # 9:16, multiples of 16


def _klein_loaders(g: GraphBuilder, m: Models, te_device: str = "default") -> tuple[str, str, str]:
    u = g.add("UNETLoader", "Load FLUX.2 klein 4B (distilled)", unet_name=m.klein_unet, weight_dtype="default")
    c = g.add("CLIPLoader", "Load Qwen3 4B" + (" (on CPU)" if te_device == "cpu" else ""), clip_name=m.klein_te,
              type="flux2", device=te_device)
    v = g.add("VAELoader", "Load FLUX.2 VAE", vae_name=m.klein_vae)
    return u, c, v


def avatar_graph(
    prompt: str,
    *,
    models: Models,
    reference: bool | int,
    seed: int = 77,
    width: int = AVATAR_W,
    height: int = AVATAR_H,
    steps: int = 4,
    prefix: str = "mc/avatar",
    te_device: str = "default",
) -> StageGraph:
    """FLUX.2 klein: text-to-image (``reference`` falsy) or reference edit with
    1 photo (``True``/``1``) or 2 photos (``2``, e.g. a two-person dialogue shot)."""
    g = GraphBuilder()
    u, c, v = _klein_loaders(g, models, te_device)
    pos = g.add("CLIPTextEncode", "Avatar prompt", text=prompt, clip=out(c))
    neg = g.add("ConditioningZeroOut", conditioning=out(pos))
    inputs: dict[str, tuple[str, str]] = {}
    for idx in range(int(reference)):
        key = "photo" if idx == 0 else "photo_b"
        li = g.add("LoadImage", "Your photo" if idx == 0 else "Second photo", image=f"{key}.png")
        inputs[key] = (li, "image")
        sc = g.add(
            "ImageScaleToTotalPixels",
            image=out(li),
            upscale_method="nearest-exact",
            megapixels=1.0,
            resolution_steps=1,
        )
        en = g.add("VAEEncode", pixels=out(sc), vae=out(v))
        pos = g.add("ReferenceLatent", conditioning=out(pos), latent=out(en))
        neg = g.add("ReferenceLatent", conditioning=out(neg), latent=out(en))
    gd = g.add("CFGGuider", model=out(u), positive=out(pos), negative=out(neg), cfg=1.0)
    nz = g.add("RandomNoise", noise_seed=int(seed))
    sm = g.add("KSamplerSelect", sampler_name="euler")
    sg = g.add("Flux2Scheduler", steps=steps, width=width, height=height)
    lt = g.add("EmptyFlux2LatentImage", "Size (9:16)", width=width, height=height, batch_size=1)
    sa = g.add(
        "SamplerCustomAdvanced", noise=out(nz), guider=out(gd), sampler=out(sm), sigmas=out(sg), latent_image=out(lt)
    )
    dec = g.add("VAEDecode", samples=out(sa), vae=out(v))
    save = g.add("SaveImage", "Save avatar", images=out(dec), filename_prefix=prefix)
    return StageGraph(g.nodes, inputs, save)


# --------------------------------------------------------------------------- #
# Stage 2: voice (Chatterbox, MIT)
# --------------------------------------------------------------------------- #
def tts_graph(
    text: str,
    *,
    models: Models,
    clone: bool = True,
    exaggeration: float = 0.55,
    cfg_weight: float = 0.5,
    temperature: float = 0.8,
    seed: int = 42,
    prefix: str = "mc/voice",
) -> StageGraph:
    """Chatterbox TTS. With ``clone=True`` the uploaded voice sample is the
    audio prompt; otherwise the model pack's built-in voice is used."""
    g = GraphBuilder()
    inputs: dict[str, tuple[str, str]] = {}
    extra: dict[str, Any] = {}
    if clone:
        ref = g.add("LoadAudio", "Voice sample to clone", audio="voice_ref.wav")
        inputs["voice_ref"] = (ref, "audio")
        extra["audio_prompt"] = out(ref)
    # ~25 tokens per second of speech; leave headroom for slow talkers.
    max_tokens = int(min(4000, max(400, len(text) * 2.2 + 200)))
    max_tokens -= max_tokens % 8
    tts = g.add(
        "ChatterboxTTS",
        "Chatterbox TTS",
        model_pack_name=models.chatterbox_pack,
        text=text,
        max_new_tokens=max_tokens,
        flow_cfg_scale=0.7,
        exaggeration=float(exaggeration),
        temperature=float(temperature),
        cfg_weight=float(cfg_weight),
        repetition_penalty=1.2,
        min_p=0.05,
        top_p=1.0,
        seed=int(seed),
        use_watermark=False,
        device="default",
        dtype="auto",
        attention_mode="sdpa",
        top_k=1000,
        normalize_loudness=True,
        num_steps=10,
        cfg_scale=1.0,
        **extra,
    )
    save = g.add("SaveAudio", "Save speech", audio=out(tts), filename_prefix=prefix)
    return StageGraph(g.nodes, inputs, save)


def voice_changer_graph(*, models: Models, prefix: str = "mc/voice_vc") -> StageGraph:
    """Chatterbox VC: re-voice ``source`` (e.g. your recording or the TTS line)
    with the timbre of ``target`` (a character voice you have rights to)."""
    g = GraphBuilder()
    src = g.add("LoadAudio", "Speech to convert", audio="vc_source.wav")
    tgt = g.add("LoadAudio", "Character voice sample", audio="vc_target.wav")
    vc = g.add(
        "ChatterboxVC",
        "Chatterbox voice changer",
        model_pack_name=models.chatterbox_pack,
        source_audio=out(src),
        n_timesteps=10,
        temperature=1.0,
        flow_cfg_scale=0.7,
        use_watermark=False,
        device="default",
        dtype="auto",
        attention_mode="sdpa",
        refinement_passes=1,
        max_chunk_duration=30.0,
        chunk_method="smart",
        target_voice_audio=out(tgt),
    )
    save = g.add("SaveAudio", "Save converted speech", audio=out(vc), filename_prefix=prefix)
    return StageGraph(g.nodes, {"source": (src, "audio"), "target": (tgt, "audio")}, save)


# --------------------------------------------------------------------------- #
# Stage 3: lip sync (Wan2.1 I2V 14B + InfiniteTalk)
# --------------------------------------------------------------------------- #
BASE_FRAMES = 81
MOTION_FRAMES = 9


def plan_windows(
    audio_seconds: float, fps: int = 25, base_frames: int = BASE_FRAMES, ext_frames: int = 81
) -> dict[str, int]:
    """How many InfiniteTalk extension windows cover ``audio_seconds``."""
    if ext_frames <= MOTION_FRAMES:
        raise ValueError("ext_frames must be larger than the motion context")
    needed = max(1, math.ceil(audio_seconds * fps))
    new_per_ext = ext_frames - MOTION_FRAMES
    extends = max(0, math.ceil((needed - base_frames) / new_per_ext))
    total = base_frames + extends * new_per_ext
    return {"extends": extends, "total_frames": total, "windows": extends + 1, "needed_frames": needed}


def lipsync_graph(
    motion: str,
    *,
    models: Models,
    audio_seconds: float,
    width: int = 480,
    height: int = 832,
    fps: int = 25,
    ext_frames: int = 81,
    steps: int = 6,
    seed: int = 2026,
    audio_scale: float = 1.0,
    two_speakers: bool = False,
    split: float = 0.5,
    prefix: str = "mc/talking",
    te_device: str = "default",
) -> StageGraph:
    """InfiniteTalk lip sync. ``two_speakers=True`` uses the multi-speaker patch
    with one audio track per speaker (silence while the other talks) and a
    left/right mask split at ``split`` (fraction of the width)."""
    plan = plan_windows(audio_seconds, fps=fps, ext_frames=ext_frames)
    g = GraphBuilder()
    unet = g.add("UnetLoaderGGUF", "Wan2.1 I2V 14B 480p (GGUF)", unet_name=models.wan_gguf)
    lora = g.add(
        "LoraLoaderModelOnly",
        "LightX2V step-distill LoRA",
        model=out(unet),
        lora_name=models.wan_lora,
        strength_model=1.0,
    )
    patch = g.add(
        "ModelPatchLoader",
        "InfiniteTalk patch (two speakers)" if two_speakers else "InfiniteTalk patch (single speaker)",
        name=models.infinitetalk_multi_patch if two_speakers else models.infinitetalk_patch,
    )
    clip = g.add("CLIPLoader", "Load umt5 (fp8)", clip_name=models.wan_te, type="wan", device=te_device)
    vae = g.add("VAELoader", "Load Wan 2.1 VAE", vae_name=models.wan_vae)
    aenc = g.add("AudioEncoderLoader", "Load wav2vec2 audio encoder", audio_encoder_name=models.audio_encoder)
    audio = g.add("LoadAudio", "Speech audio", audio="speech.wav")
    image = g.add("LoadImage", "Avatar image", image="avatar.png")
    enc = g.add("AudioEncoderEncode", audio_encoder=out(aenc), audio=out(audio))
    inputs = {"speech": (audio, "audio"), "avatar": (image, "image")}
    speaker_kw: dict[str, Any] = {}
    final_audio = audio
    if two_speakers:
        audio2 = g.add("LoadAudio", "Speaker B audio", audio="speech_b.wav")
        mixed = g.add("LoadAudio", "Mixed dialogue audio (for the video)", audio="speech_mix.wav")
        enc2 = g.add("AudioEncoderEncode", audio_encoder=out(aenc), audio=out(audio2))
        inputs["speech_b"] = (audio2, "audio")
        inputs["speech_mix"] = (mixed, "audio")
        final_audio = mixed
        cut = max(16, min(width - 16, int(width * split) // 16 * 16))
        empty_l = g.add("SolidMask", value=0.0, width=width, height=height)
        left_blk = g.add("SolidMask", value=1.0, width=cut, height=height)
        mask_a = g.add("MaskComposite", "Speaker A mask (left)", destination=out(empty_l), source=out(left_blk), x=0, y=0, operation="add")
        empty_r = g.add("SolidMask", value=0.0, width=width, height=height)
        right_blk = g.add("SolidMask", value=1.0, width=width - cut, height=height)
        mask_b = g.add("MaskComposite", "Speaker B mask (right)", destination=out(empty_r), source=out(right_blk), x=cut, y=0, operation="add")
        # ``mode`` is a V3 DynamicCombo: its sub-inputs MUST be keyed "mode.<name>". Un-dotted keys are silently
        # dropped by ComfyUI (comfy_api/latest/_io.py build_nested_inputs), so speaker B's audio and both masks never
        # reached the node before Oct 19 (found in a node study; pending one short duet confirmation render).
        speaker_kw = {"mode.audio_encoder_output_2": out(enc2), "mode.mask_1": out(mask_a), "mode.mask_2": out(mask_b)}
    pos = g.add("CLIPTextEncode", "Motion prompt", text=motion, clip=out(clip))
    neg = g.add("ConditioningZeroOut", conditioning=out(pos))
    sampler = g.add("KSamplerSelect", sampler_name="euler")

    frames_so_far: str | None = None
    for i in range(plan["windows"]):
        kw: dict[str, Any] = dict(
            model=out(lora),
            model_patch=out(patch),
            positive=out(pos),
            negative=out(neg),
            vae=out(vae),
            width=width,
            height=height,
            length=BASE_FRAMES if i == 0 else ext_frames,
            audio_encoder_output_1=out(enc),
            motion_frame_count=MOTION_FRAMES,
            audio_scale=float(audio_scale),
            start_image=out(image),
            mode="two_speakers" if two_speakers else "single_speaker",
            **speaker_kw,
        )
        if frames_so_far is not None:
            kw["previous_frames"] = out(frames_so_far)
        title = "InfiniteTalk base window" if i == 0 else f"InfiniteTalk extend window {i}"
        win = g.add("WanInfiniteTalkToVideo", title, **kw)
        guider = g.add("CFGGuider", model=out(win, 0), positive=out(win, 1), negative=out(win, 2), cfg=1.0)
        sched = g.add("BasicScheduler", model=out(win, 0), scheduler="normal", steps=int(steps), denoise=1.0)
        noise = g.add("RandomNoise", noise_seed=int(seed) + i)
        samp = g.add(
            "SamplerCustomAdvanced",
            noise=out(noise),
            guider=out(guider),
            sampler=out(sampler),
            sigmas=out(sched),
            latent_image=out(win, 3),
        )
        dec = video_decode(g, out(samp), out(vae))
        if frames_so_far is None:
            frames_so_far = dec
        else:
            trimmed = g.add(
                "ImageFromBatch", "Drop overlap frames", image=out(dec), batch_index=MOTION_FRAMES, length=4096
            )
            frames_so_far = g.add("ImageBatch", image1=out(frames_so_far), image2=out(trimmed))

    video = g.add("CreateVideo", images=out(frames_so_far), fps=float(fps), audio=out(final_audio))
    save = g.add(
        "SaveVideo",
        "Save talking clip",
        video=out(video),
        filename_prefix=prefix,
        **{"format": "mp4", "format.codec": "h264"},
    )
    return StageGraph(g.nodes, inputs, save, meta=plan)


# --------------------------------------------------------------------------- #
# Stage 4 (optional): smooth motion with RIFE frame interpolation (core nodes)
# --------------------------------------------------------------------------- #
def smooth_graph(*, models: Models, multiplier: int = 2, fps: int = 25, prefix: str = "mc/smooth") -> StageGraph:
    """Double the frame rate with RIFE (MIT) using ComfyUI's core
    FrameInterpolate node. Upscaling to 1080x1920 happens afterwards in ffmpeg
    (lanczos) to keep RAM use low on 32 GB machines."""
    g = GraphBuilder()
    lv = g.add("LoadVideo", "Talking clip", file="talking.mp4")
    parts = g.add("GetVideoComponents", video=out(lv))
    model = g.add("FrameInterpolationModelLoader", "Load RIFE", model_name=models.rife)
    interp = g.add("FrameInterpolate", interp_model=out(model), images=out(parts, 0), multiplier=int(multiplier))
    # Audio is re-muxed from the original speech track by the finisher.
    video = g.add("CreateVideo", images=out(interp), fps=float(fps * multiplier))
    save = g.add(
        "SaveVideo",
        "Save smoothed clip",
        video=out(video),
        filename_prefix=prefix,
        **{"format": "mp4", "format.codec": "h264"},
    )
    return StageGraph(g.nodes, {"video": (lv, "file")}, save)


# --------------------------------------------------------------------------- #
# Meme Recreate: motion with Wan2.2 I2V A14B (two GGUF experts, 4-step LoRAs)
# --------------------------------------------------------------------------- #
WAN_NEGATIVE = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，"
    "残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，"
    "三条腿，背景人很多，倒着走"
)  # Wan's recommended negative prompt (vivid colors, static frame, extra fingers, text ...)
# Avatar mode: Wan's stock negative pushes AWAY from "style, artwork, painting, picture" (风格，作品，画作，画面),
# which fights cartoon / anime / pixel-art characters. Stylized avatars drop those four terms.
WAN_NEGATIVE_STYLIZED = WAN_NEGATIVE.replace("风格，作品，画作，画面，", "")


def wan_negative(stylized: bool = False, extra: str = "") -> str:
    """Negative prompt for Wan jobs: the stock one (or the avatar-friendly one) plus a format's extras."""
    base = WAN_NEGATIVE_STYLIZED if stylized else WAN_NEGATIVE
    extra = (extra or "").strip().strip(",")
    return f"{base}，{extra}" if extra else base


MEME_FRAMES = 81  # ~5 s at 16 fps


def meme_motion_graph(
    motion: str,
    *,
    models: Models,
    width: int = 480,
    height: int = 832,
    frames: int = MEME_FRAMES,
    fps: int = 16,
    seed: int = 2026,
    steps: int = 4,
    split: int = 2,
    shift: float = 5.0,
    prefix: str = "mc/meme_motion",
    te_device: str = "default",
    negative: str | None = None,
) -> StageGraph:
    """Image-to-video for a meme recipe: the high-noise expert lays out the motion
    (steps 0..split), the low-noise expert adds detail (split..steps). Same
    settings as the Wan 2.2 I2V workflow already used on the reference PC."""
    frames = max(5, (int(frames) - 1) // 4 * 4 + 1)  # Wan needs 4n+1 frames
    g = GraphBuilder()
    hi = g.add("UnetLoaderGGUF", "Wan2.2 I2V high-noise expert (GGUF)", unet_name=models.wan22_high)
    lo = g.add("UnetLoaderGGUF", "Wan2.2 I2V low-noise expert (GGUF)", unet_name=models.wan22_low)
    hl = g.add("LoraLoaderModelOnly", "LightX2V 4-step (high)", model=out(hi), lora_name=models.wan22_lora_high, strength_model=1.0)
    ll = g.add("LoraLoaderModelOnly", "LightX2V 4-step (low)", model=out(lo), lora_name=models.wan22_lora_low, strength_model=1.0)
    hs = g.add("ModelSamplingSD3", model=out(hl), shift=float(shift))
    ls = g.add("ModelSamplingSD3", model=out(ll), shift=float(shift))
    clip = g.add("CLIPLoader", "Load umt5 (fp8)", clip_name=models.wan_te, type="wan", device=te_device)
    vae = g.add("VAELoader", "Load Wan 2.1 VAE", vae_name=models.wan_vae)
    pos = g.add("CLIPTextEncode", "Meme motion prompt", text=motion, clip=out(clip))
    neg = g.add("CLIPTextEncode", "Negative prompt", text=negative or WAN_NEGATIVE, clip=out(clip))
    img = g.add("LoadImage", "You, in the meme scene", image="avatar.png")
    wi = g.add("WanImageToVideo", "Size / length", positive=out(pos), negative=out(neg), vae=out(vae),
               width=width, height=height, length=frames, batch_size=1, start_image=out(img))
    k1 = g.add("KSamplerAdvanced", f"High-noise pass (steps 0-{split})", model=out(hs), add_noise="enable",
               noise_seed=int(seed), steps=int(steps), cfg=1.0, sampler_name="euler", scheduler="simple",
               positive=out(wi, 0), negative=out(wi, 1), latent_image=out(wi, 2), start_at_step=0,
               end_at_step=int(split), return_with_leftover_noise="enable")
    k2 = g.add("KSamplerAdvanced", f"Low-noise pass (steps {split}-{steps})", model=out(ls), add_noise="disable",
               noise_seed=0, steps=int(steps), cfg=1.0, sampler_name="euler", scheduler="simple",
               positive=out(wi, 0), negative=out(wi, 1), latent_image=out(k1), start_at_step=int(split),
               end_at_step=10000, return_with_leftover_noise="disable")
    dec = video_decode(g, out(k2), out(vae))
    video = g.add("CreateVideo", images=out(dec), fps=float(fps))
    save = g.add("SaveVideo", "Save meme clip", video=out(video), filename_prefix=prefix,
                 **{"format": "mp4", "format.codec": "h264"})
    return StageGraph(g.nodes, {"avatar": (img, "image")}, save, meta={"frames": frames, "fps": fps, "windows": 1})


TI2V_FPS = 24  # Wan2.2 TI2V 5B native frame rate


def ti2v_motion_graph(
    motion: str,
    *,
    models: Models,
    width: int = 480,
    height: int = 832,
    frames: int = 73,
    fps: int = TI2V_FPS,
    seed: int = 2026,
    steps: int = 20,
    cfg: float = 5.0,
    shift: float = 8.0,
    te_device: str = "cpu",
    tiled_decode: bool = True,
    prefix: str = "mc/meme_motion_lite",
    negative: str | None = None,
) -> StageGraph:
    """MC Lite meme motion: Wan2.2 TI2V 5B (GGUF) image-to-video, one model, no
    expert swap. Same settings as ComfyUI's Wan2.2 5B template (shift 8, 20 steps,
    CFG 5, uni_pc), just at 480p-ish and ~3 s. ``models.ti2v_lora`` (optional)
    adds a speed LoRA and drops to 6 steps at CFG 1. The umT5 text encoder runs on
    the CPU and the VAE decode is tiled, so an 8-12 GB card only holds the 5.4 GB
    model plus activations. Wan2.2's 5B VAE needs sizes in multiples of 32."""
    width, height = max(32, int(width) // 32 * 32), max(32, int(height) // 32 * 32)
    frames = max(5, (int(frames) - 1) // 4 * 4 + 1)
    g = GraphBuilder()
    if models.ti2v_model.endswith(".gguf"):
        unet = g.add("UnetLoaderGGUF", "Wan2.2 TI2V 5B (GGUF)", unet_name=models.ti2v_model)
    else:
        unet = g.add("UNETLoader", "Wan2.2 TI2V 5B", unet_name=models.ti2v_model, weight_dtype="default")
    model: str = unet
    if models.ti2v_lora:
        model = g.add("LoraLoaderModelOnly", "5B speed LoRA (optional)", model=out(unet), lora_name=models.ti2v_lora,
                      strength_model=1.0)
        steps, cfg = min(steps, 6), 1.0
    ms = g.add("ModelSamplingSD3", model=out(model), shift=float(shift))
    clip = g.add("CLIPLoader", "Load umt5 (fp8)" + (" on CPU" if te_device == "cpu" else ""), clip_name=models.wan_te,
                 type="wan", device=te_device)
    vae = g.add("VAELoader", "Load Wan 2.2 VAE (5B)", vae_name=models.wan22_vae)
    pos = g.add("CLIPTextEncode", "Meme motion prompt", text=motion, clip=out(clip))
    neg = g.add("CLIPTextEncode", "Negative prompt", text=negative or WAN_NEGATIVE, clip=out(clip))
    img = g.add("LoadImage", "You, in the meme scene", image="avatar.png")
    lat = g.add("Wan22ImageToVideoLatent", "Size / length", vae=out(vae), width=width, height=height, length=frames,
                batch_size=1, start_image=out(img))
    ks = g.add("KSampler", model=out(ms), seed=int(seed), steps=int(steps), cfg=float(cfg), sampler_name="uni_pc",
               scheduler="simple", positive=out(pos), negative=out(neg), latent_image=out(lat), denoise=1.0)
    # tiled_decode is kept for callers; every video decode is tiled now (see video_decode)
    dec = video_decode(g, out(ks), out(vae))
    video = g.add("CreateVideo", images=out(dec), fps=float(fps))
    save = g.add("SaveVideo", "Save meme clip", video=out(video), filename_prefix=prefix,
                 **{"format": "mp4", "format.codec": "h264"})
    return StageGraph(g.nodes, {"avatar": (img, "image")}, save,
                      meta={"frames": frames, "fps": fps, "windows": 1, "engine": "ti2v5b"})


# --------------------------------------------------------------------------- #
# Wan2.2 Animate 14B: Swap Me In ("replace") and Copy the Vibe ("move")
# --------------------------------------------------------------------------- #
ANIMATE_WINDOW = 77  # frames per Animate window (template default, ~4.8 s at 16 fps)
ANIMATE_MOTION = 5  # frames carried over between windows (and between chunks)
ANIMATE_SEGMENT_WINDOWS = 3  # windows per ComfyUI job: long clips are chunked into several jobs
ANIMATE_MODES = ("replace", "move")
_OFF = ("", "off", "none", "disabled", "0", "false", "default")
# A/B switches, both OFF by default (the default Animate graph is byte-identical to b007ac3; see
# tests/test_render_time.py). Read when animate_graph builds a graph, so a runner can flip them per job.
#  * MC_ANIMATE_WINDOW=shot: a single-window shot renders the smallest 4k+1 window that covers its frames (as
#    SCAIL-2 does) instead of the trained 77. Changes the sample, so it is a LOOK-RISK until the A/B passes.
#  * MC_ANIMATE_SAGE=on (or a kernel name): SageAttention on the Animate GGUF sampler (the SCAIL-2 patch node).


def animate_window_mode() -> str:
    """'shot' when MC_ANIMATE_WINDOW=shot, else 'fixed' (ANIMATE_WINDOW frames, the default)."""
    return "shot" if os.environ.get("MC_ANIMATE_WINDOW", "").strip().lower() == "shot" else "fixed"


def animate_sage() -> str:
    """Sage kernel for the Animate sampler from MC_ANIMATE_SAGE ('' = off, the default; 'on' = the SCAIL kernel)."""
    v = os.environ.get("MC_ANIMATE_SAGE", "").strip()
    if v.lower() in _OFF:
        return ""
    return "sageattn_qk_int8_pv_fp16_triton" if v.lower() in ("on", "1", "true", "yes") else v


def plan_animate_windows(frames: int, window: int = ANIMATE_WINDOW, carry: int = ANIMATE_MOTION,
                         carried: bool = False) -> int:
    """Windows needed for ``frames`` NEW frames. A carried chunk starts from the
    previous chunk's last ``carry`` frames, so even its first window only adds
    ``window - carry`` new frames."""
    if carried:
        return max(1, math.ceil(int(frames) / (window - carry)))
    return 1 + max(0, math.ceil((int(frames) - window) / (window - carry)))


def animate_output_frames(windows: int, carried: bool = False) -> int:
    """Frames a job actually returns (the last window is padded with the final pose)."""
    step = ANIMATE_WINDOW - ANIMATE_MOTION
    return windows * step if carried else ANIMATE_WINDOW + (windows - 1) * step


@dataclass
class AnimateSegment:
    index: int
    start: int  # first NEW frame (index into the prepared clip)
    frames: int  # new frames this chunk contributes
    lead: int  # extra frames before ``start`` fed to the job (the carried motion frames)
    windows: int


def plan_animate_segments(total_frames: int, max_windows: int = ANIMATE_SEGMENT_WINDOWS) -> list[AnimateSegment]:
    """Split a long clip into ComfyUI jobs of at most ``max_windows`` windows.

    Seam handling: chunk k>0 re-reads the 5 frames before its start (``lead``) and
    gets the previous chunk's last 5 *rendered* frames as ``continue_motion``,
    exactly like the windows inside one job, so motion carries across the cut
    instead of restarting. Windows are spread evenly over the chunks, so there's
    never a tiny trailing chunk, and the total window count is the same as for one
    giant job."""
    total = int(total_frames)
    n_windows = plan_animate_windows(total)
    n_chunks = math.ceil(n_windows / max_windows)
    per = [n_windows // n_chunks + (1 if i < n_windows % n_chunks else 0) for i in range(n_chunks)]
    segs: list[AnimateSegment] = []
    start = 0
    for i, w in enumerate(per):
        carried = i > 0
        n = min(animate_output_frames(w, carried), total - start)
        segs.append(AnimateSegment(i, start, n, ANIMATE_MOTION if carried else 0, plan_animate_windows(n, carried=carried)))
        start += n
    return segs


FACE_DRIVES = ("clip", "none", "blank", "ref", "facepose")


def animate_graph(
    prompt: str,
    *,
    models: Models,
    frames: int,
    mode: str = "replace",
    carry: bool = False,
    width: int = 480,
    height: int = 832,
    fps: int = 16,
    seed: int = 2026,
    steps: int = 6,
    relight: bool = True,
    subject: str = "person",
    animate_gguf: str | None = None,
    prefix: str | None = None,
    te_device: str = "default",
    negative: str | None = None,
    person_mask: bool = False,
    isolate_ref: bool = False,
    mask_grow: int = 12,
    face_drive: str = "clip",
    cfg: float = 1.0,
    composite_grow: int | None = None,
    composite_blur: float | None = None,
    driver: bool = False,
    mask_threshold: float | None = None,
) -> StageGraph:
    """One Wan2.2 Animate job rendering ``frames`` new frames.

    * ``replace`` (Swap Me In): SAM 3.1 masks the person in ``clip``; the
      background comes from the clip and only the person is re-rendered as you.
    * ``move`` (Copy the Vibe): only the *motion* is taken from ``clip`` (SDPose
      skeleton + face crops). No clip pixels reach the model: the scene comes
      from ``avatar`` (your photo already placed in a new scene by FLUX.2 klein)
      and the prompt.

    With ``carry`` the job continues a previous chunk: ``clip`` then starts
    ``ANIMATE_MOTION`` frames early and ``carry`` (the previous chunk's last
    rendered frames) seeds the first window's motion.

    ``person_mask`` (multi-character Swap Me In): instead of "whoever matches
    'person'", SAM 3.1 tracks exactly the one person drawn in the ``person_mask``
    input (a white-on-black first-frame mask from :func:`people_scan_graph`). Pose
    and face are read from that person only (everyone else blacked out), and the
    result is composited back over the input frames with a soft mask, so the
    other people stay pixel-identical. That lets MC swap people one pass at a time.

    ``isolate_ref`` (avatar mode, replace only): SAM 3.1 cuts the character out of the
    reference, crops it and centres it full-height on a neutral grey card, so Animate
    (and CLIP Vision) see the character big and clean instead of a small figure in its
    own dark scene: much stronger identity and art-style adherence. ``mask_grow``
    widens the replaced area (the "remove props" option uses a larger value so the
    original character's accessories fall inside the repainted region).

    ``face_drive`` picks what drives the face/expressions (Animate's ``face_video``):
    ``clip`` (default) = 512 px face crops of the person in the clip; ``none`` = no face
    input at all; ``blank`` = black frames (the same "no face" signal Animate's negative
    side gets); ``ref`` = a crop of the *reference's* own head, held still; ``facepose`` = the
    dancer's face *keypoints* drawn on black (expression timing without a human face to copy). A human face
    crop makes Animate draw a human head, so a robot, animal or mascot reference needs
    one of the other three (see FACE_DRIVES / pipeline.NONHUMAN_FACE_DRIVE).

    ``cfg`` (style lock, pending a real test): above 1.0 ComfyUI also evaluates the negative prompt (about 2x
    sampler time); at 1.0 the negative is never used. ``composite_grow`` / ``composite_blur`` tighten the soft
    paste-back edge of a multi-character pass (default: ``mask_grow`` + 8 px, blur 9).

    ``driver`` (per-person driver, pending a real test; needs ``person_mask``): pose, face and the tracked person
    come from a second input ``driver.mp4`` (the same frames with another person's moves placed where this person
    stands, see media.compose_driver), while the background and the paste-back use the untouched clip. The
    replaced area is this person's own mask in the clip PLUS the driver figure's mask, so neither shows through.

    Everything except the Animate model loader (ComfyUI-GGUF) is a core ComfyUI
    v0.41 node. ``clip`` must already be ``width`` x ``height`` at ``fps``."""
    if driver and not person_mask:
        raise ValueError("driver needs person_mask (it replaces one person's moves in a multi-character pass)")
    if person_mask and mode != "replace":
        raise ValueError("person_mask is for replace mode (multi-character Swap Me In)")
    if isolate_ref and mode != "replace":
        raise ValueError("isolate_ref is for replace mode (Swap Me In)")
    if mode not in ANIMATE_MODES:
        raise ValueError(f"unknown Animate mode {mode!r}")
    if face_drive not in FACE_DRIVES:
        raise ValueError(f"unknown face_drive {face_drive!r} (use {', '.join(FACE_DRIVES)})")
    replace_mode = mode == "replace"
    windows = plan_animate_windows(frames, carried=carry)
    lead = ANIMATE_MOTION if carry else 0
    window = ANIMATE_WINDOW
    if animate_window_mode() == "shot" and windows == 1 and not carry:  # A/B flag, off by default
        window = min(ANIMATE_WINDOW, max(5, (int(frames) - 1 + 3) // 4 * 4 + 1))
    sage = animate_sage()  # A/B flag, off by default
    prefix = prefix or ("mc/swap" if replace_mode else "mc/vibe")
    g = GraphBuilder()
    # inputs ---------------------------------------------------------------
    lv = g.add("LoadVideo", "Your clip (you must have the rights)" if replace_mode
               else "Reference motion clip (only its pose is used)", file="clip.mp4")
    parts = g.add("GetVideoComponents", video=out(lv))
    frames_img = out(parts, 0)
    drv_img = frames_img
    if driver:
        dv = g.add("LoadVideo", "Driver: whose moves this person follows", file="driver.mp4")
        drv_img = out(g.add("GetVideoComponents", video=out(dv)), 0)
    ref = g.add("LoadImage", "You (reference photo / avatar)" if replace_mode else "You, in the new scene",
                image="avatar.png")
    ref_sc = g.add("ImageScale", image=out(ref), upscale_method="lanczos", width=width, height=height, crop="center")
    carry_img = None
    if carry:
        cv_in = g.add("LoadVideo", "Previous chunk's last frames (seam)", file="carry.mp4")
        carry_img = out(g.add("GetVideoComponents", video=out(cv_in)), 0)
    char_mask = background = cutout_save = None
    if replace_mode:
        # person mask: SAM 3.1 tracks whoever matches the text prompt -----------
        sam = g.add("CheckpointLoaderSimple", "Load SAM 3.1", ckpt_name=models.sam3)
        if person_mask:
            pm_img = g.add("LoadImage", "Which person (first-frame mask)", image="person_mask.png")
            pm = g.add("ImageToMask", image=out(pm_img), channel="red")
            track = g.add("SAM3_VideoTrack", "Track this one person", images=frames_img, model=out(sam, 0),
                          detection_threshold=0.5, max_objects=1, detect_interval=1, initial_mask=out(pm))
        else:
            sam_txt = g.add("CLIPTextEncode", "Who to replace", text=subject, clip=out(sam, 1))
            track = g.add("SAM3_VideoTrack", "Track the person", images=frames_img, model=out(sam, 0),
                          detection_threshold=0.5, max_objects=1, detect_interval=1, conditioning=out(sam_txt))
        mask = g.add("SAM3_TrackToMask", track_data=out(track), object_indices="0")
        if driver:  # the figure placed in driver.mp4 (tracked from the same first-frame box) joins the replaced area
            d_track = g.add("SAM3_VideoTrack", "Track the driver figure", images=drv_img, model=out(sam, 0),
                            detection_threshold=0.5, max_objects=1, detect_interval=1, initial_mask=out(pm))
            d_mask = g.add("SAM3_TrackToMask", track_data=out(d_track), object_indices="0")
            mask = g.add("MaskComposite", "This person + the driver figure", destination=out(mask),
                         source=out(d_mask), x=0, y=0, operation="add")
        if mask_threshold is not None:  # mask hygiene: a hard 0/1 matte before grow, so soft SAM edges can't halo
            mask = g.add("ThresholdMask", "Binary person mask (hygiene)", mask=out(mask), value=float(mask_threshold))
        grown = g.add("GrowMask", mask=out(mask), expand=int(mask_grow), tapered_corners=True)
        # blocky mask (what Wan-Animate was trained with): down/up-scale + threshold
        m_img = g.add("MaskToImage", mask=out(grown))
        m_small = g.add("ImageScale", image=out(m_img), upscale_method="area", width=max(16, width // 16),
                        height=max(16, height // 16), crop="disabled")
        m_big = g.add("ImageScale", image=out(m_small), upscale_method="nearest-exact", width=width, height=height,
                      crop="disabled")
        m_blk = g.add("ImageToMask", image=out(m_big), channel="red")
        char_mask = g.add("ThresholdMask", "Character mask", mask=out(m_blk), value=0.01)
        black = g.add("EmptyImage", width=width, height=height, batch_size=int(frames) + lead, color=0)
        background = g.add("ImageCompositeMasked", "Background with the person cut out", destination=frames_img,
                           source=out(black), x=0, y=0, resize_source=False, mask=out(char_mask))
        if isolate_ref:  # character only, big and centred on neutral grey (style + identity adherence)
            r_txt = g.add("CLIPTextEncode", "Find the character in your reference", text="person", clip=out(sam, 1))
            r_det = g.add("SAM3_Detect", "Cut the character out", model=out(sam, 0), image=out(ref_sc), threshold=0.35,
                          refine_iterations=2, individual_masks=False, conditioning=out(r_txt))
            grey = g.add("EmptyImage", width=width, height=height, batch_size=1, color=0x808080)
            iso = g.add("ImageCompositeMasked", destination=out(grey), source=out(ref_sc), x=0, y=0,
                        resize_source=False, mask=out(r_det, 0))
            cw = max(64, min(width, int(round(height * 0.62 / 8)) * 8))
            c_img = g.add("CropByBBoxes", image=out(iso), bboxes=out(r_det, 1), output_width=cw, output_height=height,
                          padding=max(8, height // 40), keep_aspect="pad")
            c_msk = g.add("CropByBBoxes", image=out(g.add("MaskToImage", mask=out(r_det, 0))), bboxes=out(r_det, 1),
                          output_width=cw, output_height=height, padding=max(8, height // 40), keep_aspect="pad")
            ref_sc = g.add("ImageCompositeMasked", "Reference: your character, centred on grey", destination=out(grey),
                           source=out(c_img), x=(width - cw) // 2, y=0, resize_source=False,
                           mask=out(g.add("ImageToMask", image=out(c_msk), channel="red")))
            # saved next to the run (_work/ref_cutout*.png) so you can check what Animate actually saw
            cutout_save = g.add("SaveImage", "Save the cut-out reference (check it)", images=out(ref_sc),
                                filename_prefix="mc/ref_cutout")
    pose_src = frames_img
    if person_mask:  # pose + face from this person only (or their driver): everyone else is blacked out
        pose_src = out(g.add("ImageCompositeMasked", "Only this person (for pose)", destination=out(black),
                             source=drv_img, x=0, y=0, resize_source=False, mask=out(grown)))
    # pose + face: RT-DETR finds the person, SDPose extracts the keypoints -----
    det_model = g.add("UNETLoader", "Load RT-DETR v4", unet_name=models.rtdetr, weight_dtype="default")
    boxes = g.add("RTDETR_detect", image=pose_src, model=out(det_model), threshold=0.5, class_name="person",
                  max_detections=1)
    sdp = g.add("CheckpointLoaderSimple", "Load SDPose wholebody", ckpt_name=models.sdpose)
    kps = g.add("SDPoseKeypointExtractor", model=out(sdp, 0), vae=out(sdp, 2), image=pose_src, batch_size=16,
                bboxes=out(boxes))
    pose = g.add("SDPoseDrawKeypoints", "Pose video", keypoints=out(kps), draw_body=True, draw_hands=True,
                 draw_face=False, draw_feet=False, stick_width=4, face_point_size=3, score_threshold=0.3, draw_head=True)
    face_boxes = g.add("SDPoseFaceBBoxes", keypoints=out(kps), scale=1.5, force_square=True)
    face = g.add("CropByBBoxes", "Face video (512)", image=pose_src, bboxes=out(face_boxes), output_width=512,
                 output_height=512, padding=0, keep_aspect="stretch")
    face_in: Any = out(face)
    if face_drive == "none":
        face_in = None
    elif face_drive == "blank":
        face_in = out(g.add("EmptyImage", "No face drive (black)", width=512, height=512,
                            batch_size=int(frames) + lead, color=0))
    elif face_drive == "facepose":  # expressions as face keypoints only (no human skin/eyes to copy)
        fpose = g.add("SDPoseDrawKeypoints", "Face keypoints only", keypoints=out(kps), draw_body=False,
                      draw_hands=False, draw_face=True, draw_feet=False, stick_width=2, face_point_size=3,
                      score_threshold=0.3, draw_head=False)
        face_in = out(g.add("CropByBBoxes", "Face keypoints (512)", image=out(fpose), bboxes=out(face_boxes),
                            output_width=512, output_height=512, padding=0, keep_aspect="stretch"))
    elif face_drive == "ref":
        if not replace_mode:
            sam = g.add("CheckpointLoaderSimple", "Load SAM 3.1", ckpt_name=models.sam3)
        h_txt = g.add("CLIPTextEncode", "Find the reference's head", text="head", clip=out(sam, 1))
        h_det = g.add("SAM3_Detect", "Reference head", model=out(sam, 0), image=out(ref_sc), threshold=0.3,
                      refine_iterations=1, individual_masks=False, conditioning=out(h_txt))
        h_img = g.add("CropByBBoxes", "Reference head (512)", image=out(ref_sc), bboxes=out(h_det, 1), output_width=512,
                      output_height=512, padding=16, keep_aspect="pad")
        face_in = out(g.add("RepeatImageBatch", "Face drive: the reference's own head", image=out(h_img),
                            amount=int(frames) + lead))
    # model ------------------------------------------------------------------
    unet = g.add("UnetLoaderGGUF", "Wan2.2 Animate 14B (GGUF)", unet_name=animate_gguf or models.animate_gguf)
    model = g.add("LoraLoaderModelOnly", "LightX2V step-distill LoRA", model=out(unet), lora_name=models.wan_lora,
                  strength_model=1.0)
    if relight and replace_mode:
        model = g.add("LoraLoaderModelOnly", "Wan-Animate relight LoRA (match scene lighting)", model=out(model),
                      lora_name=models.animate_relight_lora, strength_model=1.0)
    if sage:
        model = g.add(SAGE_NODE, "SageAttention (Animate sampler, A/B)", model=out(model), sage_attention=sage,
                      allow_compile=False)
    model = g.add("ModelSamplingSD3", model=out(model), shift=8.0)
    clip = g.add("CLIPLoader", "Load umt5 (fp8)", clip_name=models.wan_te, type="wan", device=te_device)
    vae = g.add("VAELoader", "Load Wan 2.1 VAE", vae_name=models.wan_vae)
    cv = g.add("CLIPVisionLoader", "Load CLIP Vision H", clip_name=models.clip_vision)
    cv_out = g.add("CLIPVisionEncode", clip_vision=out(cv), image=out(ref_sc), crop="none")
    pos = g.add("CLIPTextEncode", "Prompt", text=prompt, clip=out(clip))
    neg = g.add("CLIPTextEncode", "Negative prompt", text=negative or WAN_NEGATIVE, clip=out(clip))
    # windows ----------------------------------------------------------------
    so_far: str | None = None
    prev_anim: str | None = None
    for i in range(windows):
        kw: dict[str, Any] = dict(
            positive=out(pos), negative=out(neg), vae=out(vae), width=width, height=height, length=window,
            batch_size=1, continue_motion_max_frames=ANIMATE_MOTION,
            video_frame_offset=(lead if prev_anim is None else out(prev_anim, 5)),
            clip_vision_output=out(cv_out), reference_image=out(ref_sc), pose_video=out(pose),
        )
        if face_in is not None:
            kw["face_video"] = face_in
        if replace_mode:
            kw.update(background_video=out(background), character_mask=out(char_mask))
        if so_far is not None:
            kw["continue_motion"] = out(so_far)
        elif carry_img is not None:
            kw["continue_motion"] = carry_img  # seam: motion continues from the previous chunk
        title = "Animate window 1" if i == 0 else f"Animate window {i + 1} (extend)"
        if i == 0 and carry:
            title = "Animate window 1 (continues the previous chunk)"
        anim = g.add("WanAnimateToVideo", title, **kw)
        ks = g.add("KSampler", model=out(model), seed=int(seed) + i, steps=int(steps), cfg=float(cfg), sampler_name="euler",
                   scheduler="simple", positive=out(anim, 0), negative=out(anim, 1), latent_image=out(anim, 2),
                   denoise=1.0)
        trim = g.add("TrimVideoLatent", samples=out(ks), trim_amount=out(anim, 3))
        dec = video_decode(g, out(trim), out(vae))
        new = g.add("ImageFromBatch", image=out(dec), batch_index=out(anim, 4), length=4096)
        so_far = new if so_far is None else g.add("ImageBatch", image1=out(so_far), image2=out(new))
        prev_anim = anim
    if person_mask:  # paste only this person back over the input frames (soft edge)
        n = int(frames)
        dst = g.add("ImageFromBatch", "Input frames (this chunk)", image=frames_img, batch_index=lead, length=n)
        src = g.add("ImageFromBatch", "Rendered frames", image=out(so_far), batch_index=0, length=n)
        c_grow = int(mask_grow) + 8 if composite_grow is None else int(composite_grow)
        c_blur = 9 if composite_blur is None else max(1, int(round(composite_blur)))
        soft = g.add("GrowMask", mask=out(mask), expand=c_grow, tapered_corners=True)
        soft_img = g.add("ImageBlur", image=out(g.add("MaskToImage", mask=out(soft))), blur_radius=c_blur,
                         sigma=4.0 if composite_blur is None else max(0.5, c_blur / 2))
        soft_img = g.add("ImageFromBatch", image=out(soft_img), batch_index=lead, length=n)
        soft_m = g.add("ImageToMask", "Soft person mask", image=out(soft_img), channel="red")
        so_far = g.add("ImageCompositeMasked", "Swapped person over the original frames", destination=out(dst),
                       source=out(src), x=0, y=0, resize_source=False, mask=out(soft_m))
    # Audio (if any) is re-muxed locally by the finisher.
    video = g.add("CreateVideo", images=out(so_far), fps=float(fps))
    save = g.add("SaveVideo", "Save swapped clip" if replace_mode else "Save your new-scene take",
                 video=out(video), filename_prefix=prefix, **{"format": "mp4", "format.codec": "h264"})
    inputs = {"clip": (lv, "file"), "avatar": (ref, "image")}
    if carry:
        inputs["carry"] = (cv_in, "file")
    if person_mask:
        inputs["person_mask"] = (pm_img, "image")
    if driver:
        inputs["driver"] = (dv, "file")
    meta: dict[str, Any] = {"frames": int(frames), "windows": windows, "fps": fps, "mode": mode, "lead": lead,
                            "out_frames": int(frames) if person_mask else (
                                window if window != ANIMATE_WINDOW else animate_output_frames(windows, carry))}
    if window != ANIMATE_WINDOW:
        meta["window"] = window
    if sage:
        meta["sage"] = sage
    if cfg != 1.0:
        meta["cfg"] = float(cfg)
    if driver:
        meta["driver"] = True
    if cutout_save:
        meta["extra_outputs"] = {"ref_cutout": cutout_save}
    return StageGraph(g.nodes, inputs, save, meta=meta)


def people_scan_graph(*, models: Models, threshold: float = 0.5, prefix: str = "mc/people",
                      max_people: int = 8) -> StageGraph:
    """Find everyone in one frame (multi-character Swap Me In): SAM 3.1 detects every 'person' and saves one
    white-on-black mask image per person. MC numbers them left to right locally.

    The prompt is ``person:8``: ComfyUI's SAM 3 text encoder keeps only the single best detection per
    prompt unless it carries a ``:N`` max-detections suffix (found on the reference PC: plain ``person``
    returned 1 of 2 characters at any threshold)."""
    g = GraphBuilder()
    img = g.add("LoadImage", "First frame of your clip", image="frame.png")
    sam = g.add("CheckpointLoaderSimple", "Load SAM 3.1", ckpt_name=models.sam3)
    txt = g.add("CLIPTextEncode", "Find people", text=f"person:{int(max_people)}", clip=out(sam, 1))
    det = g.add("SAM3_Detect", "Detect every person", model=out(sam, 0), image=out(img), threshold=float(threshold),
                refine_iterations=2, individual_masks=True, conditioning=out(txt))
    m_img = g.add("MaskToImage", mask=out(det, 0))
    save = g.add("SaveImage", "Save one mask per person", images=out(m_img), filename_prefix=prefix)
    return StageGraph(g.nodes, {"frame": (img, "image")}, save, meta={"all_outputs": True})


def swap_graph(prompt: str, *, models: Models, frames: int, **kw: Any) -> StageGraph:
    """Swap Me In: replace the main person in ``clip`` with the person in ``avatar``."""
    return animate_graph(prompt, models=models, frames=frames, mode="replace", **kw)


def vibe_graph(prompt: str, *, models: Models, frames: int, **kw: Any) -> StageGraph:
    """Copy the Vibe: you perform the clip's motion in a NEW scene (move mode, no clip pixels)."""
    kw.pop("relight", None)
    return animate_graph(prompt, models=models, frames=frames, mode="move", relight=False, **kw)


# --------------------------------------------------------------------------- #
# Optional pre-pass: erase props with VOID (Netflix video inpainting, Apache-2.0)
# --------------------------------------------------------------------------- #
VOID_MAX_FRAMES = 197  # VOID's longest clip per job; MC splits longer clips into chunks
VOID_CHUNK_FRAMES = 157  # 8k-3: no padding needed for full chunks
VOID_SIZE = (384, 672)  # VOID's native (short, long) working size (the template's 672x384)


def void_erase_graph(
    what: str,
    *,
    models: Models,
    frames: int,
    width: int,
    height: int,
    fps: int = 16,
    fill_prompt: str = "",
    seed: int = 2026,
    steps: int = 30,
    cfg: float = 6.0,
    two_pass: bool = True,
    mask_grow: int = 8,
    prefix: str = "mc/erased",
) -> StageGraph:
    """Erase ``what`` (a short SAM 3.1 phrase, e.g. ``backpack`` or ``backpack, tire``) from ``clip``
    before Swap Me In, so Animate never sees the original character's props.

    Mirrors Comfy's "VOID: Video Inpainting" template (core nodes since ComfyUI v0.41, PR #13403):
    SAM 3.1 masks the prop on every frame -> VOIDInpaintConditioning -> pass 1 (void_pass1, VOIDSampler,
    CFG 6, 30 steps) -> optional pass 2 (void_pass2 from optical-flow-warped noise of pass 1, steadier).
    VOID works at 672x384; only the masked area is pasted back over the full-resolution frames, so the rest
    of the clip stays pixel-identical. Measured on the RTX 5080 16 GB (Oct 17 2026): 5 s clip (80 frames) in
    ~6 min incl. model loads, peak ~16 GB with Dynamic VRAM; erased Junkrat-style tire pack cleanly."""
    vw, vh = (VOID_SIZE[1], VOID_SIZE[0]) if width >= height else VOID_SIZE
    g = GraphBuilder()
    lv = g.add("LoadVideo", "Your clip (props to erase)", file="clip.mp4")
    parts = g.add("GetVideoComponents", video=out(lv))
    frames_img = out(parts, 0)
    sam = g.add("CheckpointLoaderSimple", "Load SAM 3.1", ckpt_name=models.sam3)
    s_txt = g.add("CLIPTextEncode", "What to erase", text=what, clip=out(sam, 1))
    det = g.add("SAM3_Detect", "Mask the props on every frame", model=out(sam, 0), image=frames_img, threshold=0.4,
                refine_iterations=2, individual_masks=False, conditioning=out(s_txt))
    mask = g.add("GrowMask", mask=out(det, 0), expand=int(mask_grow), tapered_corners=True)
    te = g.add("CLIPLoader", "Load T5-XXL (CogVideoX)", clip_name=models.void_te, type="cogvideox", device="default")
    vae = g.add("VAELoader", "Load CogVideoX VAE", vae_name=models.void_vae)
    pos = g.add("CLIPTextEncode", "Fill (the scene without the props)", clip=out(te),
                text=fill_prompt or "the same person and background, natural clothing, nothing carried on the back")
    neg = g.add("CLIPTextEncode", "Negative", clip=out(te), text="")
    cond = g.add("VOIDInpaintConditioning", positive=out(pos), negative=out(neg), vae=out(vae), video=frames_img,
                 quadmask=out(mask), width=vw, height=vh, length=int(frames), batch_size=1)
    u1 = g.add("UNETLoader", "VOID pass 1", unet_name=models.void_pass1, weight_dtype="default")
    sig1 = g.add("BasicScheduler", model=out(u1), scheduler="simple", steps=int(steps), denoise=1.0)
    gd1 = g.add("CFGGuider", model=out(u1), positive=out(cond, 0), negative=out(cond, 1), cfg=float(cfg))
    noise = g.add("RandomNoise", noise_seed=int(seed))
    smp1 = g.add("VOIDSampler")
    s1 = g.add("SamplerCustomAdvanced", "Pass 1", noise=out(noise), guider=out(gd1), sampler=out(smp1),
               sigmas=out(sig1), latent_image=out(cond, 2))
    img = video_decode(g, out(s1, 0), out(vae))
    if two_pass:
        flow = g.add("OpticalFlowLoader", model_name=models.void_flow)
        wn = g.add("VOIDWarpedNoise", "Warped noise from pass 1", optical_flow=out(flow), video=out(img), width=vw,
                   height=vh, length=int(frames), batch_size=1)
        wns = g.add("VOIDWarpedNoiseSource", warped_noise=out(wn))
        u2 = g.add("UNETLoader", "VOID pass 2", unet_name=models.void_pass2, weight_dtype="default")
        sig2 = g.add("BasicScheduler", model=out(u2), scheduler="simple", steps=int(steps), denoise=1.0)
        gd2 = g.add("CFGGuider", model=out(u2), positive=out(cond, 0), negative=out(cond, 1), cfg=float(cfg))
        smp2 = g.add("VOIDSampler")
        s2 = g.add("SamplerCustomAdvanced", "Pass 2 (steadier)", noise=out(wns), guider=out(gd2), sampler=out(smp2),
                   sigmas=out(sig2), latent_image=out(cond, 2))
        img = video_decode(g, out(s2, 0), out(vae))
    up = g.add("ImageScale", image=out(img), upscale_method="lanczos", width=width, height=height, crop="disabled")
    up = g.add("ImageFromBatch", image=out(up), batch_index=0, length=int(frames))
    soft = g.add("ImageBlur", image=out(g.add("MaskToImage", mask=out(mask))), blur_radius=5, sigma=2.0)
    comp = g.add("ImageCompositeMasked", "Erased area over the original frames", destination=frames_img,
                 source=out(up), x=0, y=0, resize_source=False,
                 mask=out(g.add("ImageToMask", image=out(soft), channel="red")))
    video = g.add("CreateVideo", images=out(comp), fps=float(fps))
    save = g.add("SaveVideo", "Save the erased clip", video=out(video), filename_prefix=prefix,
                 **{"format": "mp4", "format.codec": "h264"})
    return StageGraph(g.nodes, {"clip": (lv, "file")}, save,
                      meta={"frames": int(frames), "fps": fps, "void": True, "two_pass": two_pass})


def void_valid_length(frames: int) -> int:
    """Smallest VOID-friendly frame count >= ``frames``. VOID (CogVideoX-Fun V1.5, 4x temporal compression,
    patch_size_t=2) needs an even latent length, i.e. 8k-3 frames (5, 13, ..., 45, ..., 197); ComfyUI
    rounds other lengths DOWN, which would drop frames, so MC pads the chunk up instead."""
    k = max(1, math.ceil((int(frames) + 3) / 8))
    return 8 * k - 3


def plan_void_chunks(frames: int) -> list[tuple[int, int]]:
    """(first, count) chunks of at most VOID_CHUNK_FRAMES (one VOID job each)."""
    if frames <= VOID_MAX_FRAMES:
        return [(0, int(frames))]
    n = math.ceil(frames / VOID_CHUNK_FRAMES)
    size = math.ceil(frames / n)
    return [(i, min(size, frames - i)) for i in range(0, frames, size)]


# --------------------------------------------------------------------------- #
# Multi-person Swap Me In in ONE pass: SCAIL-2 replacement mode (weights MIT, code Apache-2.0)
# --------------------------------------------------------------------------- #
# SCAIL-2 was trained on these identity colours, in this order (comfy_extras/nodes_scail.py DEFAULT_PALETTE):
# blue, red, green, magenta, cyan, yellow. Person k (left to right among the swapped people) gets colour k.
SCAIL_PALETTE = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0)]
SCAIL_MAX_PEOPLE = len(SCAIL_PALETTE)
SCAIL_MAX_FRAMES = 81  # one SCAIL-2 chunk (it was trained on 81-frame chunks); longer clips use chained Animate
SCAIL_ANCHOR = 5  # frames each extend chunk re-reads from the previous one (SCAIL-2 trained at 5)
SCAIL_CHUNK_STEP = SCAIL_MAX_FRAMES - SCAIL_ANCHOR  # 76 new frames per extend chunk (docs: pose offset 76 x (k-1))
# Render size on 16 GB: 704x1280 (81 frames + 4 reference views) fills the card and spills into Windows shared
# memory (12 min on step 1 of 6, RTX 5080, Oct 18 2026); 576x1024 fits at a 14.8 GB peak in 351 s.
SCAIL_STEPS = 6  # LightX2V 4-step distill at 0.8 strength: the official fast recipe uses 6
# SageAttention on the SCAIL-2 sampler (per graph, KJNodes "Patch Sage Attention", never the global
# --use-sage-attention flag). A/B on an RTX 5080 (Oct 19 2026): 97.0 -> 76.4 s per shot (-21%), no black/NaN
# frames, identity unchanged. The *_fp16_cuda kernel has no sm_120 build in sageattention 2.2.0 post6 (crash /
# garbage), so the default is the Triton fp16-PV kernel. MC_SCAIL_SAGE=off (or "") disables it; if the node or
# kernel is missing on a ComfyUI, the pipeline reruns the same pass with normal attention (sage_unavailable).
SCAIL_SAGE = os.environ.get("MC_SCAIL_SAGE", "sageattn_qk_int8_pv_fp16_triton").strip()
if SCAIL_SAGE.lower() in ("off", "none", "disabled", "0", "false"):
    SCAIL_SAGE = ""
SAGE_NODE = "PathchSageAttentionKJ"  # sic: KJNodes' class name


def sage_unavailable(exc: BaseException) -> bool:
    """True when a failed SCAIL-2 pass looks like the Sage node or kernel is missing (not a real render error)."""
    msg = str(exc).lower()
    return any(k in msg for k in (SAGE_NODE.lower(), "sageattention", "sageattn", "no kernel image", "triton"))


SCAIL_SIZE = (576, 1024)  # short side, long side; multiples of 32


def scail_size(width: int, height: int) -> tuple[int, int]:
    """SCAIL-2 render size for a clip: the clip's own size, capped at SCAIL_SIZE (same aspect, multiples of 32)."""
    short, long_ = min(width, height), max(width, height)
    scale = min(1.0, SCAIL_SIZE[0] / short, SCAIL_SIZE[1] / long_)
    s, l = max(32, int(short * scale) // 32 * 32), max(32, int(long_ * scale) // 32 * 32)
    return (s, l) if width <= height else (l, s)


def plan_scail_chunks(frames: int) -> list[tuple[int, int, int]]:
    """(window start, window length, new frames) per SCAIL-2 chunk for a clip of ``frames`` frames: 81-frame windows
    every 76 frames; chunk 1 keeps all its frames, the rest drop their 5 anchor frames. The last window can be shorter
    (the caller pads it by holding the last frame) and its new frames are trimmed to the clip."""
    frames = int(frames)
    if frames <= SCAIL_MAX_FRAMES:
        return [(0, frames, frames)]
    out = [(0, SCAIL_MAX_FRAMES, SCAIL_MAX_FRAMES)]
    done = SCAIL_MAX_FRAMES
    while done < frames:
        start = done - SCAIL_ANCHOR
        new = min(SCAIL_CHUNK_STEP, frames - done)
        out.append((start, min(SCAIL_MAX_FRAMES, frames - start), new))
        done += new
    return out


def scail_frames(frames: int) -> int:
    """SCAIL-2 lengths are 4n+1."""
    return max(5, (int(frames) - 1 + 3) // 4 * 4 + 1)


def scail_group_graph(
    prompt: str,
    *,
    models: Models,
    frames: int,
    width: int,
    height: int,
    people: int,
    views: int = 0,
    object_indices: str = "",
    fps: int = 16,
    seed: int = 2026,
    steps: int | None = None,
    relight: float = 1.0,
    cfg: float = 1.0,
    negative: str = "",
    dpo: float = 1.0,
    distill: float = 0.8,
    detect_threshold: float = 0.5,
    prefix: str = "mc/scail_group",
    sage: str | None = None,
    carry: bool = False,
    color_match: bool = True,
) -> StageGraph:
    """Swap several people at once. The clip's people are tracked with SAM 3.1 and coloured left to right; the
    reference is a composite of everyone's look at their positions (``ref``), with an identity-coloured mask
    (``ref_mask``), plus optional full-body extra views per person (``view_<k>`` / ``view_mask_<k>``).

    Validated on an RTX 5080 16 GB (Oct 18 2026): 3 people, 81 frames at 576x1024, 6 steps: 351 s, 14.8 GB peak;
    beat three chained Animate passes on identity (robot kept its colours and arm cannon) and temporal jitter.

    ``carry=True``: one chunk of a long clip (SCAIL_CHUNK_STEP-frame steps, the official SCAIL-2 extend recipe, one
    prompt per chunk). ``clip.mp4`` is then this chunk's 81-frame window (its first SCAIL_ANCHOR frames are the
    previous chunk's last ones) and ``prev_chunk.mp4`` the previous chunk's last SCAIL_ANCHOR output frames (only that
    tail is read: comfy_extras/nodes_scail.py). The anchors are dropped from the output, and with ``color_match``
    each chunk is colour-matched (Reinhard LAB) to the previous chunk's last frame. Pending a real test."""
    g = GraphBuilder()
    w, h = scail_size(width, height)
    n = scail_frames(frames)
    lv = g.add("LoadVideo", "Your clip", file="clip.mp4")
    comp = g.add("GetVideoComponents", video=out(lv))
    clipf = g.add("ImageFromBatch", image=out(comp, 0), batch_index=0, length=n)
    vid = g.add("ResizeImageMaskNode", "Clip at the SCAIL-2 size", input=out(clipf),
                **{"resize_type": "scale dimensions", "resize_type.width": w, "resize_type.height": h,
                   "resize_type.crop": "center"}, scale_method="area")
    sam = g.add("CheckpointLoaderSimple", "Load SAM 3.1", ckpt_name=models.sam3)
    samtxt = g.add("CLIPTextEncode", "Track people", clip=out(sam, 1), text="person")
    trk = g.add("SAM3_VideoTrack", "Track everyone", images=out(vid), model=out(sam, 0), conditioning=out(samtxt),
                detection_threshold=float(detect_threshold), max_objects=int(people), detect_interval=1)
    cm = g.add("SCAIL2ColoredMask", "Colour each person (left to right)", driving_track_data=out(trk),
               object_indices=object_indices, sort_by="left_to_right", replacement_mode=True)
    ref = g.add("LoadImage", "Everyone's look (composite)", image="ref.png")
    rmask = g.add("LoadImage", "Composite identity mask", image="ref_mask.png")
    inputs: dict[str, tuple[str, str]] = {"clip": (lv, "file"), "ref": (ref, "image"), "ref_mask": (rmask, "image")}
    refs, rmasks = [ref], [rmask]
    for k in range(int(views)):
        v = g.add("LoadImage", f"Extra view {k + 1}", image=f"view_{k}.png")
        vm = g.add("LoadImage", f"Extra view {k + 1} mask", image=f"view_mask_{k}.png")
        inputs[f"view_{k}"] = (v, "image")
        inputs[f"view_mask_{k}"] = (vm, "image")
        refs.append(v)
        rmasks.append(vm)
    if len(refs) > 1:
        ref_b = g.add("BatchImagesNode", **{f"images.image{i}": out(r) for i, r in enumerate(refs)})
        rmask_b = g.add("BatchImagesNode", **{f"images.image{i}": out(r) for i, r in enumerate(rmasks)})
    else:
        ref_b, rmask_b = ref, rmask
    unet = g.add("UNETLoader", "Load SCAIL-2 14B", unet_name=models.scail_unet, weight_dtype="default")
    m = g.add("LoraLoaderModelOnly", "SCAIL-2 DPO (hands + faces)", model=out(unet), lora_name=models.scail_dpo_lora,
              strength_model=float(dpo))
    if relight:
        m = g.add("LoraLoaderModelOnly", "SCAIL-2 relight", model=out(m), lora_name=models.scail_relight_lora,
                  strength_model=float(relight))
    m = g.add("LoraLoaderModelOnly", "LightX2V step distill", model=out(m), lora_name=models.wan_lora,
              strength_model=float(distill))
    sage = SCAIL_SAGE if sage is None else sage
    if sage:  # per-graph SageAttention (needs ComfyUI-KJNodes + sageattention); see SCAIL_SAGE
        m = g.add(SAGE_NODE, "SageAttention (SCAIL-2 sampler)", model=out(m), sage_attention=sage,
                  allow_compile=False)
    ms = g.add("ModelSamplingSD3", model=out(m), shift=5.0)
    te = g.add("CLIPLoader", "Load umT5", clip_name=models.wan_te, type="wan", device="default")
    pos = g.add("CLIPTextEncode", "Prompt", clip=out(te), text=prompt)
    neg = g.add("CLIPTextEncode", "Negative", clip=out(te), text=negative or "")  # only read when cfg > 1 (style lock)
    cvl = g.add("CLIPVisionLoader", clip_name=models.clip_vision)
    cve = g.add("CLIPVisionEncode", clip_vision=out(cvl), image=out(ref), crop="none")
    vae = g.add("VAELoader", vae_name=models.wan_vae)
    extra: dict[str, Any] = {}
    prev = None
    if carry:
        pv = g.add("LoadVideo", "Previous chunk (last frames)", file="prev_chunk.mp4")
        prev = g.add("GetVideoComponents", video=out(pv))
        inputs["prev_chunk"] = (pv, "file")
        extra["previous_frames"] = out(prev, 0)
    w2v = g.add("WanSCAILToVideo", positive=out(pos), negative=out(neg), vae=out(vae), width=w, height=h, length=n,
                batch_size=1, pose_strength=1.0, pose_start=0.0, pose_end=1.0, video_frame_offset=0,
                previous_frame_count=SCAIL_ANCHOR, pose_video=out(vid), pose_video_mask=out(cm, 0),
                replacement_mode=True, reference_image=out(ref_b), reference_image_mask=out(rmask_b),
                clip_vision_output=out(cve), **extra)
    ks = g.add("KSamplerSelect", sampler_name="euler")
    # sigmas from the SHIFTED model (shift 5, the official SCAIL-2 fast recipe); taking them from the unshifted
    # model silently used the checkpoint's default shift and blurred motion detail
    sig = g.add("BasicScheduler", model=out(ms), scheduler="simple", steps=int(steps or SCAIL_STEPS), denoise=1.0)
    sc = g.add("SamplerCustom", "Swap everyone (one pass)", model=out(ms), add_noise=True, noise_seed=int(seed),
               cfg=float(cfg), positive=out(w2v, 0), negative=out(w2v, 1), sampler=out(ks), sigmas=out(sig),
               latent_image=out(w2v, 2))
    dec = video_decode(g, out(sc, 1), out(vae))
    frames_out = dec
    if prev is not None:
        frames_out = g.add("ImageFromBatch", "Drop the anchor frames", image=out(dec), batch_index=SCAIL_ANCHOR,
                           length=4096)
        if color_match:
            last = g.add("ImageFromBatch", "Previous chunk's last frame", image=out(prev, 0), batch_index=-1, length=1)
            frames_out = g.add("ColorTransfer", "Match colours to the previous chunk", image_target=out(frames_out),
                               image_ref=out(last), method="reinhard_lab", source_stats="per_frame", strength=1.0)
    video = g.add("CreateVideo", images=out(frames_out), fps=float(fps))
    save = g.add("SaveVideo", "Save swapped clip", video=out(video), filename_prefix=prefix, **SAVE_HQ)
    return StageGraph(g.nodes, inputs, save, meta={"frames": n, "fps": fps, "size": [w, h], "people": int(people),
                                                   "views": int(views), "scail": True, "carry": bool(carry),
                                                   "new_frames": n - SCAIL_ANCHOR if carry else n})


SAVE_HQ = {"format": "mp4", "format.codec": "h264", "format.codec.encoding": "re-encode",
           "format.codec.encoding.crf": 12.0}
SEEDVR2_MAX_FRAMES = 96  # per job on 16 GB (7B int8, tiled VAE): 96 frames 720x1280 -> 1080x1920 at a 15.0 GB peak


def seedvr2_graph(*, models: Models, width: int, height: int, fps: int, seed: int = 2026, tile: int = 512,
                  tile_overlap: int = 128, prefix: str = "mc/seedvr2") -> StageGraph:
    """Upscale a finished take to ``width`` x ``height`` with SeedVR2 (one step; temporal chunks, tiled VAE,
    LAB colour match to the input). Measured on an RTX 5080 16 GB: 81 frames 576x1024 -> 1080x1920 in 190 s."""
    g = GraphBuilder()
    lv = g.add("LoadVideo", "Take to upscale", file="clip.mp4")
    comp = g.add("GetVideoComponents", video=out(lv))
    rs = g.add("ResizeImageMaskNode", input=out(comp, 0),
               **{"resize_type": "scale dimensions", "resize_type.width": int(width),
                  "resize_type.height": int(height), "resize_type.crop": "center"}, scale_method="lanczos")
    pre = g.add("SeedVR2Preprocess", resized_images=out(rs))
    vae = g.add("VAELoader", "Load SeedVR2 VAE", vae_name=models.seedvr2_vae)
    un = g.add("UNETLoader", "Load SeedVR2", unet_name=models.seedvr2_unet, weight_dtype="default")
    enc = g.add("VAEEncodeTiled", pixels=out(pre), vae=out(vae), tile_size=tile, overlap=tile_overlap,
                temporal_size=64, temporal_overlap=8)
    ch = g.add("SeedVR2TemporalChunk", latent=out(enc), temporal_overlap=1, chunking_mode="auto")
    cond = g.add("SeedVR2Conditioning", model=out(un), vae_conditioning=out(ch, 0))
    ks = g.add("KSampler", "Restore + upscale", model=out(un), seed=int(seed), steps=1, cfg=1.0, sampler_name="euler",
               scheduler="simple", positive=out(cond, 0), negative=out(cond, 1), latent_image=out(ch, 0), denoise=1.0)
    mg = g.add("SeedVR2TemporalMerge", latents=out(ks), temporal_overlap=out(ch, 1))
    dec = g.add("VAEDecodeTiled", samples=out(mg), vae=out(vae), tile_size=tile, overlap=tile_overlap,
                temporal_size=64, temporal_overlap=8)
    post = g.add("SeedVR2PostProcessing", images=out(dec), original_resized_images=out(rs),
                 color_correction_method="lab")
    video = g.add("CreateVideo", images=out(post), fps=float(fps))
    save = g.add("SaveVideo", "Save upscaled take", video=out(video), filename_prefix=prefix, **SAVE_HQ)
    return StageGraph(g.nodes, {"clip": (lv, "file")}, save, meta={"size": [int(width), int(height)], "fps": fps})
