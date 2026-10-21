"""Runtime configuration. Everything can be overridden with environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = ROOT / "workflows"


def _env(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


@dataclass(frozen=True)
class Models:
    """Model file names as they appear in ComfyUI's model folders.

    Defaults match the files listed in models.md. Override any of them with the
    matching MC_* environment variable if your files are named differently.
    """

    # FLUX.2 [klein] 4B (Apache-2.0): avatar / stylize step
    klein_unet: str = field(default_factory=lambda: _env("MC_KLEIN_UNET", "flux-2-klein-4b.safetensors"))
    klein_te: str = field(default_factory=lambda: _env("MC_KLEIN_TE", "qwen_3_4b.safetensors"))
    klein_vae: str = field(default_factory=lambda: _env("MC_KLEIN_VAE", "flux2-vae.safetensors"))
    # Wan2.1 I2V 14B 480p + InfiniteTalk (Apache-2.0): lip sync step
    wan_gguf: str = field(default_factory=lambda: _env("MC_WAN_GGUF", "wan2.1-i2v-14b-480p-Q5_K_M.gguf"))
    wan_lora: str = field(
        default_factory=lambda: _env(
            "MC_WAN_LORA", "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"
        )
    )
    infinitetalk_patch: str = field(
        default_factory=lambda: _env("MC_INFINITETALK", "wan2.1_infiniteTalk_single_fp16.safetensors")
    )
    infinitetalk_multi_patch: str = field(
        default_factory=lambda: _env("MC_INFINITETALK_MULTI", "wan2.1_infiniteTalk_multi_fp16.safetensors")
    )
    wan_te: str = field(default_factory=lambda: _env("MC_WAN_TE", "umt5_xxl_fp8_e4m3fn_scaled.safetensors"))
    wan_vae: str = field(default_factory=lambda: _env("MC_WAN_VAE", "wan_2.1_vae.safetensors"))
    audio_encoder: str = field(
        default_factory=lambda: _env("MC_AUDIO_ENCODER", "wav2vec2-chinese-base_fp16.safetensors")
    )
    # Chatterbox (MIT): voice clone TTS + voice changer
    chatterbox_pack: str = field(default_factory=lambda: _env("MC_CHATTERBOX_PACK", "resembleai_default_voice"))
    # RIFE (MIT): optional frame interpolation in the finishing step
    rife: str = field(default_factory=lambda: _env("MC_RIFE", "rife_v4.26.safetensors"))

    # --- Meme Recreate: Wan2.2 I2V A14B (Apache-2.0), two GGUF experts + LightX2V 4-step LoRAs ---
    wan22_high: str = field(default_factory=lambda: _env("MC_WAN22_HIGH", "Wan2.2-I2V-A14B-HighNoise-Q5_K_M.gguf"))
    wan22_low: str = field(default_factory=lambda: _env("MC_WAN22_LOW", "Wan2.2-I2V-A14B-LowNoise-Q5_K_M.gguf"))
    wan22_lora_high: str = field(
        default_factory=lambda: _env("MC_WAN22_LORA_HIGH", "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors")
    )
    wan22_lora_low: str = field(
        default_factory=lambda: _env("MC_WAN22_LORA_LOW", "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors")
    )

    # --- Swap Me In: Wan2.2 Animate 14B (Apache-2.0) + native ComfyUI preprocessors ---
    animate_gguf: str = field(default_factory=lambda: _env("MC_ANIMATE_GGUF", "Wan2.2-Animate-14B-Q4_K_M.gguf"))
    animate_relight_lora: str = field(
        default_factory=lambda: _env("MC_ANIMATE_RELIGHT", "WanAnimate_relight_lora_fp16.safetensors")
    )
    clip_vision: str = field(default_factory=lambda: _env("MC_CLIP_VISION", "clip_vision_h.safetensors"))
    sam3: str = field(default_factory=lambda: _env("MC_SAM3", "sam3.1_multiplex_fp16.safetensors"))
    # VOID video inpainting (Netflix, Apache-2.0; native ComfyUI nodes): the opt-in "erase props first" pre-pass
    void_pass1: str = field(default_factory=lambda: _env("MC_VOID_PASS1", "void_pass1.safetensors"))
    void_pass2: str = field(default_factory=lambda: _env("MC_VOID_PASS2", "void_pass2.safetensors"))
    void_vae: str = field(default_factory=lambda: _env("MC_VOID_VAE", "cogvideox_vae.safetensors"))
    void_te: str = field(default_factory=lambda: _env("MC_VOID_TE", "t5xxl_fp16.safetensors"))
    void_flow: str = field(default_factory=lambda: _env("MC_VOID_FLOW", "raft_large_C_T_SKHT_V2-ff5fadd5.safetensors"))
    # SCAIL-2 (zai-org, weights MIT / code Apache-2.0; native ComfyUI nodes): multi-person Swap Me In in ONE pass
    scail_unet: str = field(default_factory=lambda: _env("MC_SCAIL_UNET", "wan2.1_14B_SCAIL_2_mxfp8.safetensors"))
    scail_dpo_lora: str = field(default_factory=lambda: _env("MC_SCAIL_DPO", "wan2.1_SCAIL_2_DPO_lora_bf16.safetensors"))
    scail_relight_lora: str = field(
        default_factory=lambda: _env("MC_SCAIL_RELIGHT", "wan2.1_SCAIL_2_relight_lora_bf16.safetensors")
    )
    # SeedVR2 (ByteDance-Seed, Apache-2.0; native ComfyUI nodes): the finished swap upscaled to 1080p
    seedvr2_unet: str = field(default_factory=lambda: _env("MC_SEEDVR2_UNET", "seedvr2_7b_int8_convrot.safetensors"))
    seedvr2_vae: str = field(default_factory=lambda: _env("MC_SEEDVR2_VAE", "seedvr2_ema_vae_fp16.safetensors"))
    sdpose: str = field(default_factory=lambda: _env("MC_SDPOSE", "sdpose_wholebody_fp16.safetensors"))
    rtdetr: str = field(default_factory=lambda: _env("MC_RTDETR", "rt_detr_v4-x-hgnet_fp16.safetensors"))

    # --- MC Lite (8-12 GB): Wan2.2 TI2V 5B (Apache-2.0) for meme motion, FLUX.2 klein fp8 for stills ---
    # .gguf -> ComfyUI-GGUF loader (Q8_0 5.4 GB fits an 8 GB card whole); .safetensors -> core UNETLoader
    # (e.g. Comfy-Org's wan2.2_ti2v_5B_fp16.safetensors, 10 GB, streamed by Dynamic VRAM on small cards).
    ti2v_model: str = field(default_factory=lambda: _env("MC_TI2V_MODEL", "Wan2.2-TI2V-5B-Q8_0.gguf"))
    wan22_vae: str = field(default_factory=lambda: _env("MC_WAN22_VAE", "wan2.2_vae.safetensors"))
    # Optional speed LoRA for the 5B model (e.g. Kijai's FastWan extraction, Apache-2.0). Empty = off (20 steps, CFG 5).
    ti2v_lora: str = field(default_factory=lambda: _env("MC_TI2V_LORA", ""))
    klein_unet_lite: str = field(default_factory=lambda: _env("MC_KLEIN_UNET_LITE", "flux-2-klein-4b-fp8.safetensors"))


# Hardware tiers (lite / standard / max) live in tiers.py.


@dataclass(frozen=True)
class Settings:
    comfy_base_url: str = field(default_factory=lambda: _env("COMFY_BASE_URL", "http://127.0.0.1:8189"))
    # Static bearer token, only if you started comfy-api-proxy with --token.
    comfy_api_key: str | None = field(default_factory=lambda: os.environ.get("COMFY_API_KEY") or None)
    output_dir: Path = field(default_factory=lambda: Path(_env("MC_OUTPUT_DIR", str(ROOT / "outputs"))))
    # "comfy" (real ComfyUI through comfy-api-proxy) or "mock" (no GPU; placeholder media for UI/dev)
    backend: str = field(default_factory=lambda: _env("MC_BACKEND", "comfy"))
    # Lip-sync render size. 480x832 is what the Wan2.1 480p model + LightX2V LoRA are tuned for.
    video_width: int = field(default_factory=lambda: int(_env("MC_VIDEO_W", "480")))
    video_height: int = field(default_factory=lambda: int(_env("MC_VIDEO_H", "832")))
    fps: int = 25  # InfiniteTalk is trained at 25 fps
    motion_fps: int = 16  # Wan2.2 I2V / Animate native frame rate
    max_swap_seconds: float = 30.0  # Swap Me In clip cap (chunked: 3 Animate windows per job)
    max_vibe_seconds: float = 30.0  # Copy the Vibe reference clip cap (chunked the same way)
    max_script_chars: int = 600  # ~40 s of speech
    max_voice_seconds: float = 30.0
    models: Models = field(default_factory=Models)


def settings() -> Settings:
    return Settings()
