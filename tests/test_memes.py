"""MC meme modes: recipe library, graphs, media helpers, mock pipeline."""

import json
import re

import pytest

from main_character_comfy import copy as C
from main_character_comfy import graphs, memes, themes
from main_character_comfy.backends import MockBackend
from main_character_comfy.config import Settings
from main_character_comfy.tiers import TIERS
from main_character_comfy.media import (fit_photo, has_audio_stream, media_duration, prepare_clip, render_meme_caption_png,
                            run_ffmpeg, video_size)
from main_character_comfy.pipeline import InputError, Pipeline, Request, stages_for
from main_character_comfy.validate import validate_graph

from .conftest import FIX

# Names of specific meme sources, franchises or real people that must never appear in recipes.
DENY = re.compile(
    r"boyfriend|drake|kermit|this is fine|dicaprio|leonardo|pepe|doge|shiba|wojak|harold|chipmunk|prairie dog|"
    r"galaxy brain|stonks|rick ?roll|gigachad|morpheus|spongebob|pikachu|office|steve|obama|trump|elon|musk|"
    r"citizen kane|orson|grumpy cat|success kid|disaster girl|bad luck brian|overly attached",
    re.I,
)


@pytest.fixture
def pipe(tmp_path):
    return Pipeline(MockBackend(), Settings(output_dir=tmp_path / "out"))


@pytest.fixture
def landscape_clip(tmp_path):
    p = tmp_path / "my_clip.mp4"
    run_ffmpeg(["-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=6", "-f", "lavfi", "-i",
                "sine=frequency=300:duration=6", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                "-shortest", str(p)])
    return p


# --------------------------------------------------------------------------- recipes
def test_recipe_library_is_original_and_complete():
    assert len(memes.RECIPES) >= 10
    for r in memes.RECIPES.values():
        text = " ".join([r.label, r.scene, r.motion, r.camera, r.top, r.bottom, r.line])
        # "office" is a generic setting word; only flag it as part of a title-like name
        hits = [m for m in DENY.findall(text) if m.lower() != "office"]
        assert not hits, (r.key, hits)
        assert "image 1" in r.scene and r.motion and r.camera
        assert r.effect in memes.EFFECTS
        assert len(r.top) <= 40 and len(r.bottom) <= 40 and len(r.line) <= 200
        assert r.frames % 4 == 1


def test_edit_prompt_keeps_identity_and_bans_text():
    p = memes.edit_prompt(memes.RECIPES["side_eye"])
    assert "exact face" in p and "no text" in p and "only person" in p
    styled = memes.edit_prompt(memes.RECIPES["side_eye"], "Anime style")
    assert "Anime style" in styled and "Photorealistic" not in styled


def test_formats_listed_in_brief_are_covered():
    keys = set(memes.RECIPES)
    for k in ["long_stare", "look_back", "slow_clap", "side_eye", "dramatic_zoom", "calm_chaos", "dramatic_turn"]:
        assert k in keys


# --------------------------------------------------------------------------- graphs
def test_meme_motion_graph_validates_and_matches_localcomfy_workflow(models, object_info, monkeypatch):
    sg = graphs.meme_motion_graph("x", models=models, seed=123)
    assert validate_graph(sg.graph, object_info) == []  # default (tiled decode) validates
    # with tiling off it matches the tested workflow exactly, so the decode node is the only difference
    monkeypatch.setenv("MC_DECODE_TILED", "0")
    sg = graphs.meme_motion_graph("x", models=models, seed=123)
    ref = json.loads((FIX / "tested_workflows" / "localcomfy_wan22_14b_i2v.json").read_text())
    strip = lambda g: sorted((n["class_type"], json.dumps({k: v for k, v in n["inputs"].items()
                                                            if not isinstance(v, list) and k not in ("text", "filename_prefix", "image", "noise_seed")},
                                                           sort_keys=True)) for n in g.values())
    assert strip(sg.graph) == strip(ref)
    assert sg.meta["frames"] == 81 and set(sg.inputs) == {"avatar"}


def test_meme_frames_rounded_to_4n_plus_1(models):
    assert graphs.meme_motion_graph("x", models=models, frames=50).meta["frames"] == 49


@pytest.mark.parametrize("frames,windows", [(16, 1), (77, 1), (78, 2), (149, 2), (150, 3), (160, 3)])
def test_animate_window_plan(frames, windows):
    assert graphs.plan_animate_windows(frames) == windows


def test_swap_graph_only_misses_the_swap_model_pack(models, object_info):
    sg = graphs.swap_graph("x", models=models, frames=160)
    problems = validate_graph(sg.graph, object_info)
    # every node type, input and link exists in ComfyUI v0.41 core; only the downloads are missing
    assert len(problems) == 6, problems
    assert all("not in available options" in p or "no model files" in p for p in problems)
    blob = " ".join(problems)
    for f in ["rt_detr_v4", "Wan2.2-Animate-14B-Q4_K_M", "WanAnimate_relight"]:
        assert f in blob
    assert set(sg.inputs) == {"clip", "avatar"} and sg.meta["windows"] == 3


def test_swap_graph_chains_windows(models):
    sg = graphs.swap_graph("x", models=models, frames=160, relight=False)
    anims = [(k, n) for k, n in sg.graph.items() if n["class_type"] == "WanAnimateToVideo"]
    assert len(anims) == 3
    assert "continue_motion" not in anims[0][1]["inputs"]
    for (prev_id, _), (_, n) in zip(anims, anims[1:]):
        assert n["inputs"]["video_frame_offset"] == [prev_id, 5]
        assert "continue_motion" in n["inputs"]
    for _, n in anims:
        for key in ("pose_video", "face_video", "background_video", "character_mask", "reference_image", "clip_vision_output"):
            assert key in n["inputs"]
    assert not any(n["inputs"].get("lora_name", "").startswith("WanAnimate_relight") for n in sg.graph.values())


def test_tiers_pick_animate_quants():
    assert TIERS["standard"].animate_gguf.endswith("Q4_K_M.gguf")
    assert TIERS["lite"].animate_gguf.endswith("Q3_K_M.gguf")
    assert TIERS["max"].animate_gguf.endswith("Q8_0.gguf")


# --------------------------------------------------------------------------- media
def test_meme_caption_png(tmp_path):
    from PIL import Image

    p = render_meme_caption_png("when the code", "works on the first try " * 3, tmp_path / "c.png", 1080, 1920)
    img = Image.open(p)
    assert img.size == (1080, 1920) and img.mode == "RGBA"
    alpha = img.getchannel("A")
    assert alpha.crop((0, 0, 1080, 400)).getbbox() is not None  # top text
    assert alpha.crop((0, 1000, 1080, 1600)).getbbox() is not None  # bottom text above the app-UI zone
    assert alpha.crop((0, 1700, 1080, 1920)).getbbox() is None


def test_prepare_clip_landscape(tmp_path, landscape_clip):
    c = prepare_clip(landscape_clip, tmp_path / "prep", max_seconds=4, fps=16)
    assert (c.width, c.height) == (832, 480) and c.frames == 64 and c.audio is not None
    assert video_size(c.video) == (832, 480) and not has_audio_stream(c.video)


def test_fit_photo_keeps_whole_photo(tmp_path, sample_photo):
    out = fit_photo(sample_photo, tmp_path / "f.png", (1664, 960))
    from PIL import Image

    assert Image.open(out).size == (1664, 960)


# --------------------------------------------------------------------------- pipeline (mock)
def _run(pipe, req):
    evs = list(pipe.run(req))
    assert evs[-1].status == "complete"
    return evs, evs[-1]


def test_meme_silent_end_to_end(pipe, sample_photo):
    evs, last = _run(pipe, Request(mode="meme", meme="dramatic_zoom", photo=sample_photo, style="none", consent=True))
    by = {(e.stage, e.status) for e in evs}
    assert ("motion", "done") in by and ("lipsync", "skipped") in by
    assert video_size(last.artifact) == (1080, 1920) and has_audio_stream(last.artifact)
    assert abs(media_duration(last.artifact) - 81 / 16) < 0.15
    proj = json.loads((last.artifact.parent / "_work" / "project.json").read_text())
    assert proj["captions"]["top"] == "when you remember" and proj["effect"] == "punch_in"
    assert set(proj["jobs"]) == {"avatar", "meme_motion"}
    assert last.data["gif"].endswith(".gif") and "#OpenSourceHiggsfield" in last.data["post_text"]


def test_meme_boomerang_doubles_length(pipe, sample_photo):
    _, last = _run(pipe, Request(mode="meme", meme="victory_dance", photo=sample_photo, consent=True))
    assert abs(media_duration(last.artifact) - 2 * 81 / 16) < 0.2


def test_meme_with_line_lipsyncs(pipe, sample_photo, sample_voice):
    evs, last = _run(pipe, Request(mode="meme", meme="calm_chaos", photo=sample_photo, say_line=True,
                                   voice_sample=sample_voice, consent=True, meme_top="custom top"))
    by = {(e.stage, e.status) for e in evs}
    assert ("lipsync", "done") in by and ("motion", "skipped") in by
    proj = json.loads((last.artifact.parent / "_work" / "project.json").read_text())
    assert proj["captions"]["top"] == "custom top" and proj["captions"]["speech"]


def test_custom_format(pipe, sample_photo):
    _run(pipe, Request(mode="meme", meme="custom", custom_scene="at a bus stop in the rain",
                       custom_motion="opens an umbrella that instantly flips inside out", photo=sample_photo,
                       consent=True, meme_bottom="every single time"))


def test_meme_pack(pipe, sample_photo):
    evs = list(pipe.run_pack(Request(photo=sample_photo, consent=True), ["side_eye", "slow_clap", "nope"]))
    assert evs[-1].status == "complete" and len(evs[-1].data["clips"]) == 2


def test_swap_end_to_end_keeps_audio_and_orientation(pipe, sample_photo, landscape_clip):
    evs, last = _run(pipe, Request(mode="swap", clip=landscape_clip, photo=sample_photo, consent=True, clip_seconds=5,
                                   style="none"))
    assert video_size(last.artifact) == (1920, 1080) and has_audio_stream(last.artifact)
    assert abs(media_duration(last.artifact) - 5.0) < 0.15
    proj = json.loads((last.artifact.parent / "_work" / "project.json").read_text())
    assert proj["clip"]["frames"] == 80 and set(proj["jobs"]) == {"swap"}
    assert [e.stage for e in evs if e.status == "start"] == ["prepare", "swap", "finish"]


@pytest.mark.parametrize("kwargs,msg", [
    (dict(mode="swap", consent=True), "clip"),
    (dict(mode="swap", clip="CLIP"), "rights"),
    (dict(mode="swap", clip="CLIP", consent=True, clip_seconds=31), "1-30"),
    (dict(mode="vibe", consent=True), "reference clip"),
    (dict(mode="vibe", clip="CLIP"), "rights"),
    (dict(mode="vibe", clip="CLIP", consent=True, clip_seconds=45), "1-30"),
    (dict(mode="vibe", clip="CLIP", consent=True, scene_prompt="explicit hotel"), "SFW"),
    (dict(mode="meme", meme="custom", consent=True), "custom format"),
    (dict(mode="meme", meme="not_a_meme", consent=True), "unknown meme"),
    (dict(mode="meme", consent=True, meme_top="nsfw edition"), "SFW"),
    (dict(mode="meme", meme="side_eye", say_line=True, voice_mode="builtin", consent=True), "no line"),
    (dict(mode="meme"), "consent"),
])
def test_meme_and_swap_validation(pipe, sample_photo, landscape_clip, kwargs, msg):
    if kwargs.get("clip") == "CLIP":
        kwargs["clip"] = landscape_clip
    with pytest.raises(InputError, match=msg):
        list(pipe.run(Request(photo=sample_photo, **kwargs)))


def test_stage_lists_per_mode():
    assert stages_for("swap") == ["prepare", "avatar", "swap", "revoice", "finish"]
    assert "motion" in stages_for("meme") and "swap" not in stages_for("meme")


# --------------------------------------------------------------------------- copy + theme layer
def test_microcopy_and_easter_eggs():
    assert C.loading_joke("motion")
    assert C.easter_egg(top="Top text", bottom="BOTTOM TEXT")
    assert C.easter_egg(seed=69420) == "nice."
    assert C.easter_egg(rerolls=10) and C.easter_egg(top="hi") is None


def test_theme_layer_is_separate():
    theme, css, _ = themes.load("does-not-exist")
    assert ".mc-format-grid" in css and ".mc-primary" in css
    assert themes.theme_name("nope") == "noir"
    from pathlib import Path

    ui_src = (Path(themes.__file__).parent.parent / "ui.py").read_text()
    assert not re.search(r"#[0-9a-fA-F]{6}", ui_src)  # no hard-coded colors in the layout layer
    assert "css=" not in ui_src.replace("css=css", "")


# --------------------------------------------------------------------------- research "do first" items
def test_library_is_json_and_community_extensible(tmp_path):
    doc = json.loads((memes.FORMATS_DIR / "memes.json").read_text())
    assert len(doc["formats"]) == len(memes.RECIPES)
    for r in memes.RECIPES.values():
        assert len(r.captions) >= 5 and memes.check_recipe(r) == []
    extra = tmp_path / "community.json"
    extra.write_text(json.dumps({"formats": [{"key": "plot_twist", "label": "Plot Twist", "emoji": "🌀", "vibe": "panic",
                                              "scene": "Put the person from image 1 on a park bench reading.",
                                              "motion": "The person looks up, shocked.", "camera": "static camera",
                                              "captions": [{"top": "me", "bottom": "reading the last page"}]}]}))
    lib = memes.load_library([memes.FORMATS_DIR / "memes.json", extra])
    assert "plot_twist" in lib and lib["plot_twist"].top == "me"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"formats": [{"key": "x", "label": "x", "scene": "no photo ref", "motion": "m"}]}))
    with pytest.raises(ValueError, match="image 1"):
        memes.load_library([bad])


def test_caption_reroll_and_surprise():
    first = (memes.RECIPES["side_eye"].top, memes.RECIPES["side_eye"].bottom)
    assert memes.next_caption("side_eye", first) != first
    key, cap = memes.surprise()
    assert key in memes.RECIPES and cap in memes.RECIPES[key].captions


def test_instant_still_then_animate(pipe, sample_photo):
    evs = list(pipe.run_still(Request(mode="meme", meme="side_eye", photo=sample_photo, consent=True, era="deep_fried")))
    last = evs[-1]
    assert last.status == "complete" and last.artifact.suffix == ".png"
    assert last.data["scene"] and 90 <= last.data["animate_eta_seconds"] <= 200
    assert not any(e.stage == "motion" for e in evs)  # no video work on the fast path
    # Animate reuses the scene: no second FLUX job
    evs2, last2 = _run(pipe, Request(mode="meme", meme="side_eye", photo=sample_photo, consent=True,
                                     avatar=__import__("pathlib").Path(last.data["scene"])))
    proj = json.loads((last2.artifact.parent / "_work" / "project.json").read_text())
    assert set(proj["jobs"]) == {"meme_motion"}
    starts = [e for e in evs2 if e.status == "start" and e.stage == "motion"]
    assert starts[0].data["eta_seconds"] > 0


def test_meme_length_option(pipe, sample_photo):
    _, last = _run(pipe, Request(mode="meme", meme="side_eye", photo=sample_photo, consent=True, meme_frames=49))
    assert abs(media_duration(last.artifact) - 49 / 16) < 0.15


@pytest.mark.parametrize("era", ["classic", "deep_fried", "vaporwave", "early_web", "tiktok"])
def test_meme_eras_render(tmp_path, sample_photo, era):
    from main_character_comfy.media import make_still_meme

    out = make_still_meme(sample_photo, tmp_path / f"{era}.png", top="me at 3am", bottom="nobody: / me:", era=era)
    from PIL import Image

    assert Image.open(out).size == (1080, 1920)


def test_watermark_text_and_anton_font():
    from main_character_comfy.media import FONTS_DIR, WATERMARK

    assert WATERMARK == "made with MC: Main Character · 0 credits"
    assert (FONTS_DIR / "Anton-Regular.ttf").exists() and "Open Font License" in (FONTS_DIR / "OFL.txt").read_text()


def test_share_text_and_bar():
    from main_character_comfy.pipeline import share_text
    from main_character_comfy.ui import share_html

    t = share_text("me at 3am", "")
    assert t.startswith("me at 3am") and t.endswith("#OpenSourceHiggsfield")
    assert "put the whole group chat in the meme" in t and "record your own line" in t and "GB" not in t
    assert share_text("", "").startswith("put the whole group chat in the meme")
    bar = share_html({"mp4": "/x/a.mp4", "gif": "/x/a.gif", "post_text": t}, "/x/a.mp4")
    order = [bar.index(k) for k in ("Copy / send", "9:16 MP4", "GIF", "Post on X")]
    assert order == sorted(order)  # group-chat order: copy, MP4, GIF, then post helpers
    assert "%23OpenSourceHiggsfield" in bar


def test_privacy_badge(monkeypatch):
    from main_character_comfy import system

    system.gpu_name.cache_clear()
    monkeypatch.setenv("MC_GPU_NAME", "NVIDIA GeForce RTX 5080")
    assert system.privacy_badge() == "🔒 Runs on your GPU: NVIDIA GeForce RTX 5080. Your face never leaves your PC."
    system.gpu_name.cache_clear()


def test_ui_builds_and_streams_still(pipe, sample_photo):
    from main_character_comfy.ui import OUT_KEYS, _stream, build_ui

    build_ui("mock")
    outs = [dict(zip(OUT_KEYS, o)) for o in _stream(pipe.run_still(
        Request(mode="meme", meme="slow_clap", photo=sample_photo, consent=True)), "meme")]
    last = outs[-1]
    assert last["animate"]["visible"] is True and C.ANIMATE_CTA in last["animate"]["value"]
    assert last["still"]["visible"] is True and "mc-share" in last["share"]
