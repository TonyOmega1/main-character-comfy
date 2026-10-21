# Models and licenses

Every model MC uses is **open weights**, runs locally, and is called only through ComfyUI core nodes or open-source custom nodes. There are no partner nodes and no closed APIs anywhere in the pipeline.

MC doesn't ship any weights. Download them yourself from the links below into `ComfyUI/models/<folder>/`.

## Core (Meme Recreate + talking avatar)

| Stage | Model (file) | Folder | Size | License | Download |
|---|---|---|---|---|---|
| Scene / avatar | FLUX.2 [klein] 4B distilled (`flux-2-klein-4b.safetensors`) | `diffusion_models` | ~7.8 GB | Apache-2.0 | [Comfy-Org/flux2-klein](https://huggingface.co/Comfy-Org/flux2-klein/resolve/main/split_files/diffusion_models/flux-2-klein-4b.safetensors) (original: [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)) |
| Avatar | Qwen3 4B text encoder (`qwen_3_4b.safetensors`) | `text_encoders` | ~8 GB | Apache-2.0 | [Comfy-Org/flux2-klein](https://huggingface.co/Comfy-Org/flux2-klein/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors) |
| Avatar | FLUX.2 VAE (`flux2-vae.safetensors`) | `vae` | 0.3 GB | Apache-2.0 | [Comfy-Org/flux2-dev](https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors) |
| Voice | Chatterbox TTS + VC (`resembleai_default_voice` pack) | auto-downloaded by the node | ~3 GB | MIT | [ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox) |
| Lip sync | Wan2.1 I2V 14B 480p, GGUF Q5_K_M (`wan2.1-i2v-14b-480p-Q5_K_M.gguf`) | `unet` | 12.7 GB | Apache-2.0 | [city96/Wan2.1-I2V-14B-480P-gguf](https://huggingface.co/city96/Wan2.1-I2V-14B-480P-gguf/resolve/main/wan2.1-i2v-14b-480p-Q5_K_M.gguf) (original: [Wan-AI/Wan2.1-I2V-14B-480P](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P)) |
| Lip sync | InfiniteTalk single speaker (`wan2.1_infiniteTalk_single_fp16.safetensors`) | `model_patches` | 5.1 GB | Apache-2.0 | [Comfy-Org/Wan_2.1_ComfyUI_repackaged](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/model_patches/wan2.1_infiniteTalk_single_fp16.safetensors) (original: [MeiGen-AI/InfiniteTalk](https://huggingface.co/MeiGen-AI/InfiniteTalk)) |
| Duet (optional) | InfiniteTalk multi speaker (`wan2.1_infiniteTalk_multi_fp16.safetensors`) | `model_patches` | 5.1 GB | Apache-2.0 | [Comfy-Org/Wan_2.1_ComfyUI_repackaged](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/model_patches/wan2.1_infiniteTalk_multi_fp16.safetensors) |
| Lip sync | LightX2V step/CFG distill LoRA, rank 64 (`lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors`) | `loras` | 0.7 GB | Apache-2.0 | [Kijai/WanVideo_comfy](https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors) (original: [lightx2v](https://huggingface.co/lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v)) |
| Lip sync | umT5-XXL fp8 text encoder (`umt5_xxl_fp8_e4m3fn_scaled.safetensors`) | `text_encoders` | 6.7 GB | Apache-2.0 | [Comfy-Org/Wan_2.1_ComfyUI_repackaged](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors) |
| Lip sync | Wan 2.1 VAE (`wan_2.1_vae.safetensors`) | `vae` | 0.25 GB | Apache-2.0 | [Comfy-Org/Wan_2.1_ComfyUI_repackaged](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors) |
| Lip sync | wav2vec2 Chinese base audio encoder (`wav2vec2-chinese-base_fp16.safetensors`) | `audio_encoders` | 0.2 GB | MIT | [Kijai/wav2vec2_safetensors](https://huggingface.co/Kijai/wav2vec2_safetensors/resolve/main/wav2vec2-chinese-base_fp16.safetensors) (original: [TencentGameMate/chinese-wav2vec2-base](https://huggingface.co/TencentGameMate/chinese-wav2vec2-base)) |
| Meme motion | Wan2.2 I2V A14B **high-noise** expert, GGUF Q5_K_M (`Wan2.2-I2V-A14B-HighNoise-Q5_K_M.gguf`) | `unet` | 10.8 GB | Apache-2.0 | [QuantStack/Wan2.2-I2V-A14B-GGUF](https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF/resolve/main/HighNoise/Wan2.2-I2V-A14B-HighNoise-Q5_K_M.gguf) (original: [Wan-AI/Wan2.2-I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)) |
| Meme motion | Wan2.2 I2V A14B **low-noise** expert, GGUF Q5_K_M (`Wan2.2-I2V-A14B-LowNoise-Q5_K_M.gguf`) | `unet` | 10.8 GB | Apache-2.0 | [QuantStack/Wan2.2-I2V-A14B-GGUF](https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF/resolve/main/LowNoise/Wan2.2-I2V-A14B-LowNoise-Q5_K_M.gguf) |
| Meme motion | LightX2V 4-step I2V LoRAs, high + low (`wan2.2_i2v_lightx2v_4steps_lora_v1_{high,low}_noise.safetensors`) | `loras` | 2 × 1.2 GB | Apache-2.0 | [Comfy-Org/Wan_2.2_ComfyUI_Repackaged](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files/loras) |
| Smoothing (optional) | RIFE v4.26 (`rife_v4.26.safetensors`) | `frame_interpolation` | 23 MB | MIT | [Comfy-Org/frame_interpolation](https://huggingface.co/Comfy-Org/frame_interpolation/resolve/main/frame_interpolation/rife_v4.26.safetensors) (original: [hzwer/Practical-RIFE](https://github.com/hzwer/Practical-RIFE)) |

Meme motion reuses the umT5 text encoder and the Wan 2.1 VAE from the table above.

The core table is the **Standard (16 GB)** download list. Lite and Max swap some of these files: see [Downloads per hardware tier](#downloads-per-hardware-tier) below.

## Swap Me In / Copy the Vibe pack (optional, about 18 GB on Standard)

Copy the Vibe (Animate *move* mode) needs everything below **except SAM 3.1 and the relight LoRA**, about 15 GB.

Every node in this pack is a ComfyUI v0.41 **core node**, except `UnetLoaderGGUF` (ComfyUI-GGUF). The pack also reuses umT5, the Wan 2.1 VAE, and the Wan2.1 LightX2V rank-64 LoRA from the core table.

| Stage | Model (file) | Folder | Size | License | Download |
|---|---|---|---|---|---|
| Swap render | Wan2.2-Animate-14B GGUF Q4_K_M (`Wan2.2-Animate-14B-Q4_K_M.gguf`) | `unet` | 11.5 GB | Apache-2.0 | [QuantStack/Wan2.2-Animate-14B-GGUF](https://huggingface.co/QuantStack/Wan2.2-Animate-14B-GGUF/resolve/main/Wan2.2-Animate-14B-Q4_K_M.gguf) (original: [Wan-AI/Wan2.2-Animate-14B](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B)) |
| Swap render | CLIP Vision H (`clip_vision_h.safetensors`) | `clip_vision` | 1.26 GB | Apache-2.0 | [Comfy-Org/Wan_2.1_ComfyUI_repackaged](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors) |
| Swap relight (optional, `--no-relight` skips it) | Wan-Animate relighting LoRA (`WanAnimate_relight_lora_fp16.safetensors`) | `loras` | 1.44 GB | Apache-2.0 (Wan-AI original; Kijai's ComfyUI conversion) | [Kijai/WanVideo_comfy](https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/LoRAs/Wan22_relight/WanAnimate_relight_lora_fp16.safetensors) |
| Person mask | SAM 3.1 multiplex (`sam3.1_multiplex_fp16.safetensors`) | `checkpoints` | 1.75 GB | **SAM License** (Meta; royalty-free, commercial use allowed, with use restrictions; open weights but not OSI) | [Comfy-Org/sam3.1](https://huggingface.co/Comfy-Org/sam3.1/resolve/main/checkpoints/sam3.1_multiplex_fp16.safetensors) |
| Pose + face crop | SDPose whole-body (`sdpose_wholebody_fp16.safetensors`) | `checkpoints` | 1.92 GB | MIT | [Comfy-Org/SDPose](https://huggingface.co/Comfy-Org/SDPose/resolve/main/checkpoints/sdpose_wholebody_fp16.safetensors) |
| Person detection | RT-DETRv4-X (`rt_detr_v4-x-hgnet_fp16.safetensors`) | `diffusion_models` | 0.12 GB | Apache-2.0 | [Comfy-Org/RT-DETR](https://huggingface.co/Comfy-Org/RT-DETR/resolve/main/diffusion_models/rt_detr_v4-x-hgnet_fp16.safetensors) |

Animate quants by tier (same repo and license): Q3_K_M 8.63 GB (Lite, experimental only), **Q4_K_M 11.5 GB (Standard, default)**, Q5_K_S 12.35 GB / Q5_K_M 13.0 GB (Standard with more offload), Q8_0 18.72 GB (Max). You can override any filename with `MC_ANIMATE_GGUF`, `MC_SAM3`, and the other env vars listed in `main_character_comfy/config.py`.

## Group swap + 1080p sharpening pack (Standard / Max; strongly recommended for 2-6 people)

When you swap **everyone** in a clip of up to ~5 s (2-6 people), Swap Me In renders them all in **one SCAIL-2 pass** instead of one Animate pass per person, then sharpens the take to 1080p with **SeedVR2** (`--group-engine` / `--upscale` on the CLI). All nodes are ComfyUI core (`WanSCAILToVideo`, `SCAIL2ColoredMask`, `SAM3_VideoTrack`, `SeedVR2*`). It reuses SAM 3.1, CLIP Vision H, umT5, the Wan 2.1 VAE and the LightX2V LoRA. It needs a short look description per person (`--describe`). Without these files MC falls back to chained Animate passes and a plain upscale.

| Stage | Model (file) | Folder | Size | License | Download |
|---|---|---|---|---|---|
| Group swap | SCAIL-2 14B mxfp8 (`wan2.1_14B_SCAIL_2_mxfp8.safetensors`) | `diffusion_models` | 17.2 GB (offloads on 16 GB) | weights **MIT**, code Apache-2.0 | [ComfyUI SCAIL-2 tutorial](https://docs.comfy.org/tutorials/video/zai/scail2) (original: [zai-org/SCAIL-2](https://huggingface.co/zai-org/SCAIL-2)) |
| Group swap | SCAIL-2 DPO LoRA (`wan2.1_SCAIL_2_DPO_lora_bf16.safetensors`) | `loras` | 1.14 GB | MIT | same tutorial |
| Group swap | SCAIL-2 relight LoRA (`wan2.1_SCAIL_2_relight_lora_bf16.safetensors`) | `loras` | 1.14 GB | MIT | same tutorial |
| Sharpen to 1080p | SeedVR2 7B int8 (`seedvr2_7b_int8_convrot.safetensors`) | `diffusion_models` | 8.3 GB (3B int8: 3.5 GB) | Apache-2.0 | [ComfyUI SeedVR2 tutorial](https://docs.comfy.org/tutorials/utility/seedvr2) (original: [ByteDance-Seed/SeedVR2](https://huggingface.co/ByteDance-Seed)) |
| Sharpen to 1080p | SeedVR2 VAE (`seedvr2_ema_vae_fp16.safetensors`) | `vae` | 0.47 GB | Apache-2.0 | same tutorial |

Measured on the RTX 5080 16 GB: 3 people, 81 frames, rendered at 576x1024 in 351 s (14.8 GB peak; 704x1280 does not fit and spills into shared memory), SeedVR2 to 1080x1920 in 190 s (15.0 GB peak). Override names with `MC_SCAIL_UNET`, `MC_SCAIL_DPO`, `MC_SCAIL_RELIGHT`, `MC_SEEDVR2_UNET`, `MC_SEEDVR2_VAE`.

## Optional: VOID prop eraser ("erase props first", experimental)

Only needed if you use **erase props first** in Swap Me In. The nodes (`VOIDInpaintConditioning`, `VOIDSampler`, `VOIDWarpedNoise`, `OpticalFlowLoader`) are ComfyUI v0.41 core. SAM 3.1 is shared with the Swap pack.

| Role | File | Folder | Size | License | Download |
|---|---|---|---|---|---|
| VOID pass 1 (CogVideoX-Fun 5B based) | `void_pass1.safetensors` | `diffusion_models` | 10.4 GB | Apache-2.0 (Netflix) | [Comfy-Org/void-model](https://huggingface.co/Comfy-Org/void-model/resolve/main/diffusion_models/void_pass1.safetensors) |
| VOID pass 2 (temporal refinement) | `void_pass2.safetensors` | `diffusion_models` | 10.4 GB | Apache-2.0 (Netflix) | [Comfy-Org/void-model](https://huggingface.co/Comfy-Org/void-model/resolve/main/diffusion_models/void_pass2.safetensors) |
| CogVideoX VAE | `cogvideox_vae.safetensors` | `vae` | 0.4 GB | Apache-2.0 | [Comfy-Org/void-model](https://huggingface.co/Comfy-Org/void-model/resolve/main/vae/cogvideox_vae.safetensors) |
| Optical flow (RAFT large) | `raft_large_C_T_SKHT_V2-ff5fadd5.safetensors` | `optical_flow` | 20 MB | BSD-3-Clause (torchvision) | [Comfy-Org/void-model](https://huggingface.co/Comfy-Org/void-model/resolve/main/optical_flow/raft_large_C_T_SKHT_V2-ff5fadd5.safetensors) |
| Text encoder | `t5xxl_fp16.safetensors` | `text_encoders` | 9.1 GB | Apache-2.0 | [comfyanonymous/flux_text_encoders](https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors) |

File names can be overridden with `MC_VOID_PASS1`, `MC_VOID_PASS2`, `MC_VOID_VAE`, `MC_VOID_TE` and `MC_VOID_FLOW`.

## Downloads per hardware tier

MC picks a tier from your VRAM (see the README section "Runs on your gaming PC"). Every file below is open weights, and every size was read from the Hugging Face file listing (Oct 2026). Nothing here is bundled with MC.

### Lite · 8-12 GB (about 29 GB in total)

Stills, plus short silent meme videos. Voice is unchanged. Talking, Swap Me In and Copy the Vibe need 16 GB+.

| Stage | File | Folder | Size | License | Download |
|---|---|---|---|---|---|
| Scene / still | FLUX.2 [klein] 4B **fp8** (`flux-2-klein-4b-fp8.safetensors`) | `diffusion_models` | 4.07 GB | Apache-2.0 | [black-forest-labs/FLUX.2-klein-4b-fp8](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8/resolve/main/flux-2-klein-4b-fp8.safetensors) |
| Scene / still | Qwen3 4B text encoder (`qwen_3_4b.safetensors`), runs on the **CPU** on Lite | `text_encoders` | 8.04 GB | Apache-2.0 | core table |
| Scene / still | FLUX.2 VAE (`flux2-vae.safetensors`) | `vae` | 0.34 GB | Apache-2.0 | core table |
| Meme motion | **Wan2.2 TI2V 5B** GGUF Q8_0 (`Wan2.2-TI2V-5B-Q8_0.gguf`) | `unet` | 5.40 GB | Apache-2.0 | [QuantStack/Wan2.2-TI2V-5B-GGUF](https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF/resolve/main/Wan2.2-TI2V-5B-Q8_0.gguf) (original: [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)) |
| Meme motion | Wan 2.2 VAE (`wan2.2_vae.safetensors`; the 5B model uses this one, not the 2.1 VAE) | `vae` | 1.41 GB | Apache-2.0 | [Comfy-Org/Wan_2.2_ComfyUI_Repackaged](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan2.2_vae.safetensors) |
| Meme motion | umT5-XXL fp8 (`umt5_xxl_fp8_e4m3fn_scaled.safetensors`), runs on the **CPU** on Lite | `text_encoders` | 6.74 GB | Apache-2.0 | core table |
| Voice | Chatterbox pack | auto | ~3 GB | MIT | core table |

Smaller 5B quants from the same repo, if disk or VRAM is tight (set `MC_TI2V_MODEL`):
- Q6_K: 4.21 GB
- Q5_K_M: 3.81 GB
- Q4_K_M: 3.43 GB
- Q3_K_M: 2.55 GB (quality drops quickly below Q5)

You can also use Comfy-Org's full **`wan2.2_ti2v_5B_fp16.safetensors`** (10.0 GB, [download](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors)) by setting `MC_TI2V_MODEL=wan2.2_ti2v_5B_fp16.safetensors`. It loads with the core `UNETLoader`, so Dynamic VRAM can stream it. The reference PC already has this file.

There's also an optional, **untested** speed LoRA, off by default: Kijai's FastWan extraction `Wan2_2_5B_FastWanFullAttn_lora_rank_128_bf16.safetensors` (0.66 GB, `loras`). The original is FastVideo's FastWan2.2-TI2V-5B, Apache-2.0. Download it from [Kijai/WanVideo_comfy](https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/FastWan/Wan2_2_5B_FastWanFullAttn_lora_rank_128_bf16.safetensors). Set `MC_TI2V_LORA` to its filename and MC switches to 6 steps at CFG 1. We skipped the "TI2V-5B-Turbo" weights because their model card states no license.

Lite and Max are untested beyond one RTX 5080 16 GB.

**Experimental on Lite** (only if you tick *Try 16 GB modes anyway*):
- `wan2.1-i2v-14b-480p-Q3_K_M.gguf` (8.59 GB, [city96](https://huggingface.co/city96/Wan2.1-I2V-14B-480P-gguf/resolve/main/wan2.1-i2v-14b-480p-Q3_K_M.gguf)), for lip sync. It still needs the 5.13 GB InfiniteTalk patch, which Comfy-Org only publishes in fp16. Kijai's InfiniteTalk GGUF (1.4-2.7 GB) and fp8 files are made for the WanVideoWrapper nodes and haven't been verified with core ComfyUI's `ModelPatchLoader`. That's why lip sync is marked as needing 16 GB+.
- `Wan2.2-Animate-14B-Q3_K_M.gguf` (8.63 GB, [QuantStack](https://huggingface.co/QuantStack/Wan2.2-Animate-14B-GGUF/resolve/main/Wan2.2-Animate-14B-Q3_K_M.gguf)) plus the rest of the Swap / Vibe pack, for Swap Me In and Copy the Vibe (≤10 s at 384×672).

### Standard · 16 GB (the reference RTX 5080)

Use the **core table** above (about 75 GB), plus the optional Swap / Vibe pack below (about 18 GB).

### Max · 24 GB+

Untested: no 24 GB card has run MC yet (everything so far ran on one RTX 5080 16 GB). Use the core table and the Swap / Vibe pack, with these Q8_0 swaps (same repos and licenses; about 90 GB for Meme + Talk, about 115 GB with the Swap pack):

| Replaces | Max file | Size | Download |
|---|---|---|---|
| A14B experts Q5_K_M | `Wan2.2-I2V-A14B-HighNoise-Q8_0.gguf` + `Wan2.2-I2V-A14B-LowNoise-Q8_0.gguf` | 2 × 15.41 GB | [high](https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF/resolve/main/HighNoise/Wan2.2-I2V-A14B-HighNoise-Q8_0.gguf) · [low](https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF/resolve/main/LowNoise/Wan2.2-I2V-A14B-LowNoise-Q8_0.gguf) |
| Wan2.1 lip sync Q5_K_M | `wan2.1-i2v-14b-480p-Q8_0.gguf` | 18.14 GB | [city96](https://huggingface.co/city96/Wan2.1-I2V-14B-480P-gguf/resolve/main/wan2.1-i2v-14b-480p-Q8_0.gguf) |
| Animate Q4_K_M | `Wan2.2-Animate-14B-Q8_0.gguf` | 18.72 GB | [QuantStack](https://huggingface.co/QuantStack/Wan2.2-Animate-14B-GGUF/resolve/main/Wan2.2-Animate-14B-Q8_0.gguf) |

A filename you set yourself always wins over the tier's choice (`MC_WAN22_HIGH`, `MC_WAN22_LOW`, `MC_WAN_GGUF`, `MC_ANIMATE_GGUF`, `MC_KLEIN_UNET`). So you can, for example, keep Q5 experts on Max.

## Audio, face and helper models (run outside ComfyUI, mostly CPU)

These run in MC's own Python, not as ComfyUI nodes. All open weights. Licenses checked against each official repo or model card (Oct 19, 2026).

| Used for | Model | License | Source |
|---|---|---|---|
| Record your own line: speech to text, take scoring | Whisper `small.en` (CTranslate2 conversion) via faster-whisper | MIT (weights and code) | [openai/whisper](https://github.com/openai/whisper), [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper), [Systran/faster-whisper-small.en](https://huggingface.co/Systran/faster-whisper-small.en) |
| Vocal / music split (sing, re-dub stem bed) | Demucs `htdemucs` | MIT | [facebookresearch/demucs](https://github.com/facebookresearch/demucs) |
| Speaking window: mouth openness | MediaPipe Face Landmarker (`face_landmarker.task`) | Apache-2.0 | [google-ai-edge/mediapipe](https://github.com/google-ai-edge/mediapipe) |
| Voice cleanup | DeepFilterNet 3 | MIT or Apache-2.0 (dual) | [Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) |
| Face finding (lip re-sync crop, fallback mouth motion) | YuNet `face_detection_yunet_2023mar.onnx` | MIT | [opencv/opencv_zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) |
| Likeness check (idqc) | SFace `face_recognition_sface_2021dec.onnx` | Apache-2.0 | [opencv/opencv_zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) |

## Bundled asset

| Asset | License |
|---|---|
| Anton font (`main_character_comfy/assets/fonts/Anton-Regular.ttf`, from google/fonts) | SIL OFL 1.1 (`OFL.txt` alongside) |

## Software

| Component | License |
|---|---|
| ComfyUI (core nodes: InfiniteTalk, WanAnimateToVideo, SAM3, SDPose, RT-DETR, FrameInterpolate, video/audio I/O) | GPL-3.0 |
| [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) (city96) | Apache-2.0 |
| [ComfyUI-Chatterbox](https://github.com/wildminder/ComfyUI-Chatterbox) (wildminder) | Apache-2.0 |
| [comfy-sdk](https://pypi.org/project/comfy-sdk/) (official Comfy SDK) | MIT |
| [comfy-api-proxy](https://github.com/Comfy-Org/comfy-api-proxy) | MIT |
| Gradio | Apache-2.0 |
| ffmpeg (via imageio-ffmpeg) | LGPL/GPL |
| [yt-dlp](https://github.com/yt-dlp/yt-dlp) (bring-your-own-clip link fetch) | Unlicense |
| [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) (optional second cut detector) | BSD-3-Clause |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [Demucs](https://github.com/facebookresearch/demucs), [MediaPipe](https://github.com/google-ai-edge/mediapipe), [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet), OpenCV | MIT, MIT, Apache-2.0, MIT/Apache-2.0, Apache-2.0 |

## Optional: FLUX 3 Image Klein (not a default dependency)

[FLUX 3 Image Klein](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy) is also open weights, but its license is the **FLUX Kommunity License**. That license allows non-commercial use, plus commercial use of outputs for users with less than US$5M in annual revenue, with conditions. It also explicitly bans biometric processing and any violation of publicity or digital-replica rights. Swapping it in for the avatar step would still be Open Weights Only, but the stack would no longer be all permissive licenses. See the README section "FLUX 3 Klein (optional)".
