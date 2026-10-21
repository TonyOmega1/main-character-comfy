"""Copy the Vibe, chunked Animate (Swap Me In up to 30 s), Save as format, remix recipes."""

from __future__ import annotations

import json

import pytest

from main_character_comfy import graphs, memes
from main_character_comfy.backends import MockBackend
from main_character_comfy.config import Settings
from main_character_comfy.media import count_frames, media_duration, run_ffmpeg
from main_character_comfy.pipeline import (InputError, Pipeline, Request, estimate_animate_seconds, eta_range_text,
                               request_from_recipe, stages_for, vibe_scene_prompt)
from main_character_comfy.validate import validate_graph


@pytest.fixture
def pipe(tmp_path):
    return Pipeline(MockBackend(), Settings(output_dir=tmp_path / "out"))


def _clip(path, seconds, size="720x1280"):
    run_ffmpeg(["-f", "lavfi", "-i", f"testsrc2=size={size}:rate=30:duration={seconds}", "-f", "lavfi", "-i",
                f"sine=frequency=300:duration={seconds}", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                "-shortest", str(path)])
    return path


@pytest.fixture
def ref_clip(tmp_path):  # generated test pattern: no real footage anywhere in the tests
    return _clip(tmp_path / "reference_move.mp4", 15)


# --------------------------------------------------------------------------- chunk planning
@pytest.mark.parametrize("frames", [5, 77, 78, 160, 221, 222, 300, 480])
def test_segments_cover_every_frame_once(frames):
    segs = graphs.plan_animate_segments(frames)
    assert sum(x.frames for x in segs) == frames
    assert segs[0].start == 0 and segs[0].lead == 0
    for a, b in zip(segs, segs[1:]):
        assert b.start == a.start + a.frames and b.lead == graphs.ANIMATE_MOTION
    assert all(x.windows <= graphs.ANIMATE_SEGMENT_WINDOWS for x in segs)
    assert all(graphs.animate_output_frames(x.windows, x.lead > 0) >= x.frames for x in segs)
    # chunking costs no extra windows compared with one giant job
    assert sum(x.windows for x in segs) == graphs.plan_animate_windows(frames)


def test_thirty_seconds_is_three_balanced_chunks():
    segs = graphs.plan_animate_segments(480)
    assert [x.windows for x in segs] == [3, 2, 2]
    assert min(x.frames for x in segs) > 100  # no tiny trailing chunk


def test_eta_is_honest_for_30_seconds():
    eta = estimate_animate_seconds(480, "move")
    assert 15 * 60 <= eta <= 20 * 60
    assert eta_range_text(eta) == "~14-19 min"
    assert estimate_animate_seconds(480, "replace") > eta  # SAM 3.1 tracking per chunk
    assert eta_range_text(40) == "~34-46 s"


# --------------------------------------------------------------------------- graphs
def test_vibe_graph_is_move_mode_without_clip_pixels(models, object_info):
    sg = graphs.vibe_graph("luxury hotel hallway at night", models=models, frames=160)
    classes = [n["class_type"] for n in sg.graph.values()]
    assert "SAM3_VideoTrack" not in classes and "ImageCompositeMasked" not in classes
    for n in sg.graph.values():
        if n["class_type"] == "WanAnimateToVideo":
            assert "background_video" not in n["inputs"] and "character_mask" not in n["inputs"]
            assert {"pose_video", "face_video", "reference_image"} <= set(n["inputs"])
    assert not any(str(n["inputs"].get("lora_name", "")).startswith("WanAnimate_relight") for n in sg.graph.values())
    problems = validate_graph(sg.graph, object_info)
    # move mode needs only 4 of the pack's files (no SAM 3.1, no relight LoRA)
    assert len(problems) == 4, problems
    assert sg.meta["mode"] == "move" and sg.meta["windows"] == 3


def test_carried_chunk_continues_previous_motion(models, object_info):
    sg = graphs.swap_graph("x", models=models, frames=144, carry=True, relight=False)
    assert set(sg.inputs) == {"clip", "avatar", "carry"}
    anims = [n for n in sg.graph.values() if n["class_type"] == "WanAnimateToVideo"]
    assert len(anims) == 2 and sg.meta["out_frames"] == 144 and sg.meta["lead"] == 5
    first = anims[0]["inputs"]
    assert first["video_frame_offset"] == graphs.ANIMATE_MOTION and "continue_motion" in first
    carry_node = sg.inputs["carry"][0]
    get = first["continue_motion"][0]
    assert sg.graph[get]["inputs"]["video"] == [carry_node, 0]
    # the background batch also covers the lead frames
    empty = next(n for n in sg.graph.values() if n["class_type"] == "EmptyImage")
    assert empty["inputs"]["batch_size"] == 149
    assert all("not in available options" in p or "no model files" in p
               for p in validate_graph(sg.graph, object_info))


def test_vibe_scene_prompt_is_new_scene_with_your_photo():
    p = vibe_scene_prompt("luxury hotel hallway at night", portrait=True)
    assert "image 1" in p and "luxury hotel hallway at night" in p and "9:16" in p
    assert "16:9" in vibe_scene_prompt("rooftop", portrait=False)


# --------------------------------------------------------------------------- pipeline (mock)
def test_copy_the_vibe_end_to_end_chunked(pipe, ref_clip, sample_photo):
    req = Request(mode="vibe", clip=ref_clip, photo=sample_photo, consent=True, clip_seconds=14,
                  scene_prompt="luxury hotel hallway at night", style="none", seed=7)
    evs = list(pipe.run(req))
    done = evs[-1]
    assert done.status == "complete"
    work = done.data["work_dir"]
    proj = json.loads(open(f"{work}/_work/project.json").read())
    assert [c["windows"] for c in proj["chunks"]] == [2, 2]  # 224 frames
    assert set(proj["jobs"]) == {"avatar", "vibe_00", "vibe_01"}
    assert "luxury hotel hallway at night" in proj["prompts"]["avatar"][0]
    assert count_frames(f"{work}/_work/vibe_joined.mp4") == 224
    assert abs(media_duration(done.artifact) - 14.0) < 0.2
    starts = [e.stage for e in evs if e.status == "start"]
    assert starts == ["prepare", "avatar", "vibe", "finish"]
    assert stages_for("vibe") == ["prepare", "avatar", "vibe", "revoice", "finish"]
    vibe_start = next(e for e in evs if e.stage == "vibe" and e.status == "start")
    assert "2 chunk(s)" in vibe_start.message and "estimate" in vibe_start.message
    # the reference clip never goes into the scene job: only the selfie does
    assert proj["models"]["avatar"] and "clip" not in json.dumps(proj["prompts"]["avatar"])


def test_vibe_keeps_clip_audio_by_default_and_can_mute(pipe, ref_clip, sample_photo, tmp_path):
    from main_character_comfy.media import has_audio_stream

    base = dict(mode="vibe", clip=ref_clip, photo=sample_photo, consent=True, clip_seconds=2, style="none")
    kept = list(pipe.run(Request(**base)))[-1]
    muted = list(pipe.run(Request(**base, clip_audio="mute")))[-1]
    rec_k = json.loads(open(f"{kept.data['work_dir']}/_work/mc_recipe.json").read())
    rec_m = json.loads(open(f"{muted.data['work_dir']}/_work/mc_recipe.json").read())
    assert rec_k["settings"]["clip_audio"] == "original" and rec_m["settings"]["clip_audio"] == "mute"
    proj_k = json.loads(open(f"{kept.data['work_dir']}/_work/project.json").read())
    proj_m = json.loads(open(f"{muted.data['work_dir']}/_work/project.json").read())
    assert proj_k["audio"] and proj_k["audio"].endswith("clip_audio.wav") and proj_m["audio"] is None
    assert has_audio_stream(kept.artifact) and has_audio_stream(muted.artifact)  # muted = silent track


def test_swap_thirty_seconds_chunks_and_keeps_audio(pipe, tmp_path, sample_photo):
    clip = _clip(tmp_path / "mine.mp4", 31, "1280x720")
    req = Request(mode="swap", clip=clip, photo=sample_photo, consent=True, clip_seconds=30, style="none",
                  swap_prompt="warm golden-hour light, film grain")
    done = list(pipe.run(req))[-1]
    work = done.data["work_dir"]
    proj = json.loads(open(f"{work}/_work/project.json").read())
    assert len(proj["chunks"]) == 3 and set(proj["jobs"]) == {"swap_00", "swap_01", "swap_02"}
    assert "golden-hour" in proj["prompts"]["swap_00"][-1]
    assert count_frames(f"{work}/_work/swap_joined.mp4") == 480
    assert abs(media_duration(done.artifact) - 30.0) < 0.2




# --------------------------------------------------------------------------- recipe JSON + remix
def test_every_output_has_a_remixable_recipe(pipe, sample_photo):
    req = Request(mode="meme", meme="dramatic_zoom", photo=sample_photo, consent=True, style="none", seed=4242,
                  meme_top="when you remember", meme_bottom="you left the oven on", era="vaporwave")
    done = list(pipe.run(req))[-1]
    recipe = json.loads(open(done.data["recipe"]).read())
    assert recipe["mc_recipe"] == 1 and recipe["mode"] == "meme" and recipe["seed"] == 4242
    assert recipe["format"]["key"] == "dramatic_zoom"
    assert recipe["settings"]["era"] == "vaporwave"
    assert any("flux-2-klein" in m for m in recipe["models"]["avatar"])
    assert any(m.startswith("Wan2.2-I2V-A14B") for m in recipe["models"]["meme_motion"])
    assert recipe["prompts"]["meme_motion"]
    assert recipe["uploads_used"] == ["photo"]
    assert str(sample_photo) not in json.dumps(recipe)  # your file paths are never stored
    again = request_from_recipe(recipe, photo=sample_photo, consent=True)
    assert (again.meme, again.seed, again.era, again.meme_top) == ("dramatic_zoom", 4242, "vaporwave", "when you remember")
    assert list(pipe.run(again))[-1].status == "complete"


def test_still_and_talk_outputs_also_get_a_recipe(pipe, sample_photo):
    still = list(pipe.run_still(Request(mode="meme", photo=sample_photo, consent=True, style="none")))[-1]
    assert json.loads(open(still.data["recipe"]).read())["mode"] == "meme"
    talk = list(pipe.run(Request(mode="talk", character="a friendly robot", voice_mode="builtin", script="Hi there.")))[-1]
    assert json.loads(open(talk.data["recipe"]).read())["mode"] == "talk"


def test_unknown_format_in_recipe_replays_as_custom(sample_photo):
    doc = {"mc_recipe": 1, "mode": "meme",
           "format": {"key": "someone_elses", "scene": "Put the person from image 1 in this scene: a tiny boat",
                      "motion": "rows dramatically"},
           "settings": {"meme": "someone_elses", "seed": 3}}
    req = request_from_recipe(doc, photo=sample_photo)
    assert req.meme == "custom" and req.custom_scene == "a tiny boat" and req.custom_motion == "rows dramatically"
    with pytest.raises(InputError):
        request_from_recipe({"mc_recipe": 99})


# --------------------------------------------------------------------------- Save as format
def test_save_custom_format_appears_in_library(tmp_path):
    r = memes.custom_recipe("at a drive-through holding a giant soda", "slowly lowers sunglasses",
                            "me at the drive-through", "when they say pull forward")
    key, path = memes.save_format(r, "Drive-Thru Stare", emoji="🥤")
    assert key == "drive_thru_stare" and path.parent == tmp_path / "community"
    assert key in memes.RECIPES and memes.RECIPES[key].title == "🥤 Drive-Thru Stare"
    assert memes.RECIPES[key].captions[0] == ("me at the drive-through", "when they say pull forward")
    doc = json.loads(path.read_text())
    assert doc["formats"][0]["fresh"] is True
    key2, _ = memes.save_format(r, "Drive-Thru Stare")
    assert key2 == "drive_thru_stare_2"  # never overwrites


def test_save_never_overwrites_builtins_and_stays_sfw():
    key, _ = memes.save_format(memes.RECIPES["side_eye"], "side eye", captions=[("my top", "my bottom")])
    assert key == "side_eye_2" and memes.RECIPES["side_eye"].top != "my top"
    assert memes.RECIPES[key].captions[0] == ("my top", "my bottom")
    with pytest.raises(ValueError, match="SFW"):
        memes.save_format(memes.RECIPES["side_eye"], "nsfw side eye")
    with pytest.raises(ValueError):
        memes.save_format(memes.RECIPES["side_eye"], "x", captions=[("y" * 80, "")])


def test_broken_community_file_is_skipped(tmp_path):
    (tmp_path / "community").mkdir(exist_ok=True)
    (tmp_path / "community" / "broken.json").write_text('{"formats": [{"key": "bad"}]}')
    lib = memes.reload()
    assert "side_eye" in lib and "bad" not in lib


def test_ui_saves_finished_meme_as_format(pipe, sample_photo):
    from main_character_comfy.ui import save_format_from

    done = list(pipe.run(Request(mode="meme", meme="slow_clap", photo=sample_photo, consent=True, style="none",
                                 meme_top="congrats on", meme_bottom="finally reading the group chat")))[-1]
    key, msg = save_format_from("Late Clap", "👏", done.data["work_dir"], "slow_clap", "", "", "", "")
    r = memes.RECIPES[key]
    assert r.motion == memes.RECIPES["slow_clap"].motion
    assert r.captions[0] == ("congrats on", "finally reading the group chat") and "Saved" in msg
    with pytest.raises(ValueError, match="custom"):
        save_format_from("x", "", None, "custom", "", "", "", "")


def test_ui_builds_with_vibe_tab():
    from main_character_comfy.ui import animate_eta_hint, build_ui

    assert build_ui("mock") is not None
    assert "~14-19 min" in animate_eta_hint(30, "move") and "3 chunks" in animate_eta_hint(30, "move")
