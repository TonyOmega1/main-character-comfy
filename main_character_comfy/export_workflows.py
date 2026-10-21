"""Write the API-format workflows the app builds into workflows/ so they can be
opened in ComfyUI (Workflow > Open) or reused from any Comfy SDK client.

    python -m main_character_comfy.export_workflows
"""

from __future__ import annotations

import json

from . import graphs, memes, tiers
from .config import WORKFLOWS_DIR, Models
from .pipeline import vibe_scene_prompt
from .presets import STYLES, motion_prompt


def export() -> list[str]:
    m = Models()
    WORKFLOWS_DIR.mkdir(exist_ok=True)
    items = {
        "01_avatar_from_photo_flux2_klein.json": graphs.avatar_graph(STYLES["3d"].edit_prompt, models=m, reference=True),
        "01b_avatar_from_text_flux2_klein.json": graphs.avatar_graph(
            STYLES["anime"].t2i_prompt.format(desc="a cheerful barista with curly red hair"), models=m, reference=False),
        "01c_duet_avatar_two_photos_flux2_klein.json": graphs.avatar_graph(
            STYLES["3d"].edit_prompt.replace("the person in image 1", "the two people from the images")
            + " Both people side by side facing the camera, podcast desk with two microphones.", models=m, reference=2),
        "02_voice_clone_tts_chatterbox.json": graphs.tts_graph(
            "Hey, welcome back! Today I'm showing you how I made this with open models.", models=m, clone=True),
        "03_voice_changer_chatterbox_vc.json": graphs.voice_changer_graph(models=m),
        "04_lipsync_infinitetalk_5s.json": graphs.lipsync_graph(motion_prompt("3d"), models=m, audio_seconds=4.8, ext_frames=49),
        "04b_lipsync_infinitetalk_14s_segment.json": graphs.lipsync_graph(motion_prompt("3d"), models=m, audio_seconds=14.74),
        "04c_duet_lipsync_infinitetalk_two_speakers.json": graphs.lipsync_graph(
            motion_prompt("3d", "Two people"), models=m, audio_seconds=10.0, two_speakers=True),
        "05_smooth_rife_x2.json": graphs.smooth_graph(models=m),
        "06_meme_scene_flux2_klein_side_eye.json": graphs.avatar_graph(
            memes.edit_prompt(memes.RECIPES["side_eye"]), models=m, reference=1, prefix="mc/meme_scene"),
        "06b_meme_motion_wan22_i2v_side_eye.json": graphs.meme_motion_graph(
            memes.motion_prompt(memes.RECIPES["side_eye"]), models=m),
        "07_swap_me_in_wan22_animate_5s.json": graphs.swap_graph(
            "The person moves naturally, matching the original performance.", models=m, frames=77),
        "07b_swap_me_in_wan22_animate_10s_landscape.json": graphs.swap_graph(
            "The person moves naturally, matching the original performance.", models=m, frames=160, width=832,
            height=480),
        "07c_swap_me_in_continue_chunk_wan22_animate.json": graphs.swap_graph(
            "The person moves naturally, matching the original performance.", models=m, frames=144, carry=True),
        # Openjutsu: one person of a group (first-frame mask), avatar reference cut out on grey (saved for checking)
        "07d_swap_me_in_one_person_avatar_cutout_wan22_animate.json": graphs.swap_graph(
            "The character from the reference image moves naturally, matching the original performance, and keeps "
            "its exact design, outfit, colors and art style from the reference.", models=m, frames=77, width=832,
            height=480, person_mask=True, isolate_ref=True, relight=False),
        "07e_swap_me_in_720p_wan22_animate.json": graphs.swap_graph(
            "The person moves naturally, matching the original performance.", models=m, frames=77, width=720,
            height=1280),
        # optional pre-pass: erase the original character's props with VOID (Netflix, Apache-2.0)
        "07f_erase_props_void_inpainting.json": graphs.void_erase_graph(
            "backpack", models=m, frames=77, width=832, height=480),
        "08_copy_the_vibe_scene_flux2_klein.json": graphs.avatar_graph(
            vibe_scene_prompt("luxury hotel hallway at night", portrait=True), models=m, reference=1,
            prefix="mc/vibe_scene"),
        "08b_copy_the_vibe_move_wan22_animate_13s.json": graphs.vibe_graph(
            "luxury hotel hallway at night. The person performs the movements of the pose sequence with the same "
            "timing and energy, full body, smooth natural motion, consistent face, identity and outfit, the camera "
            "stays steady.", models=m, frames=221),
        "08c_copy_the_vibe_continue_chunk_wan22_animate.json": graphs.vibe_graph(
            "luxury hotel hallway at night. The person performs the movements of the pose sequence.", models=m,
            frames=144, carry=True),
        # MC Lite (8-12 GB): klein fp8 + text encoder on CPU, Wan2.2 TI2V 5B motion at 480x832, ~3 s
        "09_lite_meme_scene_flux2_klein_fp8.json": graphs.avatar_graph(
            memes.edit_prompt(memes.RECIPES["side_eye"]), models=tiers.models_for("lite", m), reference=1,
            width=tiers.TIERS["lite"].scene_size[0], height=tiers.TIERS["lite"].scene_size[1], te_device="cpu",
            prefix="mc/meme_scene"),
        "09b_lite_meme_motion_wan22_ti2v_5b_side_eye.json": graphs.ti2v_motion_graph(
            memes.motion_prompt(memes.RECIPES["side_eye"]), models=m,
            frames=tiers.motion_frames("lite", 81)[0]),
        # MC Max (24 GB+): Q8_0 experts at 576x1024
        "10_max_meme_motion_wan22_i2v_q8_576x1024.json": graphs.meme_motion_graph(
            memes.motion_prompt(memes.RECIPES["side_eye"]), models=tiers.models_for("max", m), width=576,
            height=1024),
    }
    written = []
    for name, sg in items.items():
        path = WORKFLOWS_DIR / name
        path.write_text(json.dumps(sg.graph, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append(name)
    return written


if __name__ == "__main__":
    for n in export():
        print("wrote workflows/" + n)
