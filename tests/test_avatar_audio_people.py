"""Avatar mode, clip audio carry-over, multi-character Swap Me In, readable names, QoL (errors, stop)."""

from __future__ import annotations

import json
from pathlib import Path
import re
from datetime import datetime

import pytest
from PIL import Image, ImageDraw

from main_character_comfy import errors, graphs, media, memes, tiers
from main_character_comfy.backends import MockBackend
from main_character_comfy.config import Settings
from main_character_comfy.media import count_frames, has_audio_stream, media_duration, run_ffmpeg
from main_character_comfy.pipeline import (InputError, Pipeline, Request, estimate_people_seconds, readable_name,
                                           resolve_subject, who_label)
from main_character_comfy.validate import validate_graph


@pytest.fixture
def pipe(tmp_path):
    return Pipeline(MockBackend(), Settings(output_dir=tmp_path / "out"))


def _clip(path, seconds, size="1280x720", audio=True):
    args = ["-f", "lavfi", "-i", f"testsrc2=size={size}:rate=30:duration={seconds}"]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-c:a", "aac", "-shortest"]
    run_ffmpeg([*args, "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)])
    return path


@pytest.fixture
def cartoon(tmp_path):
    """A flat, transparent-background 'mascot' (generated: no real character)."""
    img = Image.new("RGBA", (600, 400), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((220, 60, 380, 220), fill=(255, 200, 0, 255), outline=(0, 0, 0, 255), width=6)
    d.rectangle((250, 220, 350, 380), fill=(40, 90, 220, 255))
    p = tmp_path / "my_mascot.png"
    img.save(p)
    return p


# --------------------------------------------------------------------------- avatar input
def test_auto_detects_avatars_and_photos(cartoon, sample_photo, tmp_path):
    assert media.looks_like_avatar(cartoon)  # transparent background
    flat = tmp_path / "flat_toon.jpg"
    Image.open(cartoon).convert("RGB").save(flat)  # no alpha, but flat colours
    assert media.looks_like_avatar(flat)
    assert not media.looks_like_avatar(sample_photo)
    assert resolve_subject(Request(photo=cartoon)) == "avatar"
    assert resolve_subject(Request(photo=sample_photo)) == "photo"
    assert resolve_subject(Request(photo=cartoon, subject="photo")) == "photo"  # explicit choice wins


def test_transparent_png_is_flattened_on_neutral_grey_not_black(cartoon, tmp_path):
    rgb = media.load_rgb(cartoon)
    assert rgb.mode == "RGB" and rgb.getpixel((5, 5)) == media.NEUTRAL_BG
    out = media.fit_photo(cartoon, tmp_path / "fit.png", (480, 832))
    assert Image.open(out).getpixel((240, 5)) != (0, 0, 0)


def test_avatar_mode_never_crops_a_wide_image(cartoon, tmp_path):
    out = Image.open(media.prepare_photo(cartoon, tmp_path / "p.png", (1088, 1920), keep_whole=True))
    assert out.size == (1088, 1920)
    # the whole 600x400 design fits: its yellow head and blue body are both still in frame
    px = set(out.resize((136, 240)).getdata())
    assert any(r > 200 and g > 150 and b < 80 for r, g, b in px) and any(b > 180 and r < 90 for r, g, b in px)


def test_avatar_prompts_keep_the_art_style():
    r = memes.RECIPES["side_eye"]
    a, p = memes.edit_prompt(r, subject="avatar"), memes.edit_prompt(r)
    assert "art style" in a and "not photorealistic" in a and "Photorealistic" not in a
    assert "Photorealistic" in p
    assert "art style" in memes.motion_prompt(r, subject="avatar")
    stylized = graphs.wan_negative(True)
    assert "风格" not in stylized and "画作" not in stylized and "风格" in graphs.WAN_NEGATIVE


def test_meme_in_avatar_mode_end_to_end(pipe, cartoon):
    req = Request(mode="meme", meme="side_eye", photo=cartoon, consent=True, style="none", seed=3)
    done = list(pipe.run(req))[-1]
    proj = json.loads(open(f"{done.data['work_dir']}/_work/project.json").read())
    assert proj["subject"] == "avatar"
    assert "art style" in proj["prompts"]["avatar"][0]
    neg = next(n["inputs"]["text"] for n in _graph_of(pipe, req, "meme_motion") if n["class_type"] == "CLIPTextEncode"
               and n["_meta"]["title"] == "Negative prompt")
    assert "风格" not in neg and "hand entering the frame" in neg  # avatar negative + side_eye's own extras


def _graph_of(pipe, req, job):
    """Rebuild the job's graph the way the pipeline does (prompts are recorded, graphs aren't)."""
    m = tiers.models_for(tiers.get(req.tier), pipe.settings.models)
    r = memes.RECIPES[req.meme]
    sg = graphs.meme_motion_graph(memes.motion_prompt(r, subject="avatar"), models=m,
                                  negative=graphs.wan_negative(True, r.negative))
    return sg.graph.values()


def test_rights_copy_covers_avatars():
    from main_character_comfy import copy as C

    assert "avatar" in C.CONSENT and "rights" in C.CONSENT and "avatar" in C.RIGHTS_NOTE
    assert "avatar" in C.FACE_LABEL


# --------------------------------------------------------------------------- clip audio
def test_swap_keeps_original_audio_in_sync(pipe, tmp_path, sample_photo):
    clip = _clip(tmp_path / "dance.mp4", 4)
    done = list(pipe.run(Request(mode="swap", clip=clip, photo=sample_photo, consent=True, clip_seconds=3,
                                 clip_start=0.5, style="none")))[-1]
    final = done.artifact
    assert has_audio_stream(final)
    assert abs(media_duration(final) - 3.0) < 0.15
    proj = json.loads(open(f"{done.data['work_dir']}/_work/project.json").read())
    assert proj["clip_audio"] == "original" and proj["audio"].endswith("clip_audio.wav")
    assert abs(media_duration(proj["audio"]) - 3.0) < 0.1  # cut at the same start/length as the video


def test_swap_can_mute_or_revoice(pipe, tmp_path, sample_photo):
    clip = _clip(tmp_path / "dance.mp4", 2)
    base = dict(mode="swap", clip=clip, photo=sample_photo, consent=True, clip_seconds=2, style="none")
    muted = list(pipe.run(Request(**base, clip_audio="mute")))[-1]
    assert json.loads(open(f"{muted.data['work_dir']}/_work/project.json").read())["audio"] is None
    with pytest.raises(InputError, match="voice sample"):
        list(pipe.run(Request(**base, clip_audio="my_voice")))
    voice = tmp_path / "me.wav"
    run_ffmpeg(["-f", "lavfi", "-i", "sine=frequency=200:duration=12", str(voice)])
    rv = list(pipe.run(Request(**base, clip_audio="my_voice", voice_sample=voice)))
    proj = json.loads(open(f"{rv[-1].data['work_dir']}/_work/project.json").read())
    assert "revoice" in proj["jobs"] and has_audio_stream(rv[-1].artifact)


def test_clip_without_audio_exports_a_silent_track(pipe, tmp_path, sample_photo):
    clip = _clip(tmp_path / "silent.mp4", 2, audio=False)
    evs = list(pipe.run(Request(mode="swap", clip=clip, photo=sample_photo, consent=True, clip_seconds=2,
                                style="none")))
    assert any("no audio track" in e.message for e in evs)
    assert has_audio_stream(evs[-1].artifact)


def test_short_soundtrack_is_padded_to_the_video(tmp_path):
    video = _clip(tmp_path / "v.mp4", 3, audio=False)
    short = tmp_path / "a.wav"
    run_ffmpeg(["-f", "lavfi", "-i", "sine=frequency=300:duration=1.5", str(short)])
    out = media.finish_video(video, tmp_path / "o.mp4", audio=short, duration=3.0, width=320, height=180,
                             watermark=False, fps=30)
    assert abs(media_duration(out) - 3.0) < 0.15


# --------------------------------------------------------------------------- multi-character swap
def test_people_are_numbered_left_to_right(tmp_path):
    frame = tmp_path / "f.png"
    Image.new("RGB", (640, 360), "orange").save(frame)
    masks = media_masks = __import__("main_character_comfy.mock_media", fromlist=["x"]).fake_people_masks(
        frame, tmp_path, "people", 3)
    people = media.order_people(media_masks, (640, 360))
    assert [round(p.cx, 2) for p in people] == sorted(round(p.cx, 2) for p in people)
    assert len(people) == 3 and people[0].box[0] < people[1].box[0] < people[2].box[0]
    prev = media.people_preview(frame, people, tmp_path / "prev.png")
    assert Image.open(prev).size == (640, 360) and masks


def test_match_people_keeps_identity_across_chunks(tmp_path):
    P = media.Person
    a, b = P(tmp_path / "a", (0, 0, 1, 1), 0.3, 0.5, 0.1), P(tmp_path / "b", (0, 0, 1, 1), 0.7, 0.5, 0.1)
    a2, b2 = P(tmp_path / "a2", (0, 0, 1, 1), 0.35, 0.5, 0.1), P(tmp_path / "b2", (0, 0, 1, 1), 0.68, 0.5, 0.1)
    assert media.match_people([a, b], [a2, b2]) == [a2, b2]
    assert media.match_people([b], [a2, b2]) == [b2]  # nearest centroid


def test_scan_people_preview(pipe, tmp_path):
    clip = _clip(tmp_path / "duo.mp4", 2)
    done = list(pipe.scan_people(Request(mode="swap", clip=clip, clip_seconds=2)))[-1]
    assert done.status == "complete" and done.data["people"] == 2
    assert Image.open(done.data["preview"]).size[0] > Image.open(done.data["preview"]).size[1]  # 16:9 kept


def test_multi_swap_one_person_leaves_the_other(pipe, tmp_path, sample_photo, cartoon):
    clip = _clip(tmp_path / "duo.mp4", 3)
    req = Request(mode="swap", clip=clip, consent=True, clip_seconds=3, people=[cartoon, None], style="none")
    done = list(pipe.run(req))[-1]
    work = done.data["work_dir"]
    proj = json.loads(open(f"{work}/_work/project.json").read())
    swap_jobs = [j for j in proj["jobs"] if j.startswith("swap")]
    assert swap_jobs == ["swap_p1"] and "people" in proj["jobs"]
    assert [p["slot"] for p in proj["people"]] == [1, 2] and proj["people"][1]["ref"] == ""
    assert has_audio_stream(done.artifact) and abs(media_duration(done.artifact) - 3.0) < 0.15
    assert "my-mascot-left" in done.artifact.name


def test_multi_swap_both_people_runs_two_passes_in_order(pipe, tmp_path, cartoon):
    clip = _clip(tmp_path / "duo.mp4", 3)
    req = Request(mode="swap", clip=clip, consent=True, clip_seconds=3, people=[cartoon, cartoon], style="none",
                  tier="standard")
    evs = list(pipe.run(req))
    work = evs[-1].data["work_dir"]
    proj = json.loads(open(f"{work}/_work/project.json").read())
    assert [j for j in proj["jobs"] if j.startswith("swap")] == ["swap_p1", "swap_p2"]
    assert count_frames(f"{work}/_work/swap_p2_joined.mp4") == 48
    assert "my-mascot-both" in evs[-1].artifact.name
    start = next(e for e in evs if e.stage == "swap" and e.status == "start")
    assert "2 of 2 people" in start.message and "estimate" in start.message


def test_multi_swap_graph_tracks_one_person_and_validates(models):
    import json as _json
    from pathlib import Path as _P

    oi = _json.loads((_P(__file__).parent / "fixtures" / "object_info_reference_live.json").read_text())
    sg = graphs.swap_graph("x", models=models, frames=77, person_mask=True)
    classes = [n["class_type"] for n in sg.graph.values()]
    track = next(n for n in sg.graph.values() if n["class_type"] == "SAM3_VideoTrack")
    assert "initial_mask" in track["inputs"] and "conditioning" not in track["inputs"]
    assert "person_mask" in sg.inputs and classes.count("ImageCompositeMasked") == 3
    assert sg.meta["out_frames"] == 77
    assert validate_graph(sg.graph, oi, check_files=False) == []
    assert validate_graph(graphs.people_scan_graph(models=models).graph, oi, check_files=False) == []


def test_people_caps_by_tier(pipe, tmp_path, cartoon):
    clip = _clip(tmp_path / "trio.mp4", 1)
    with pytest.raises(InputError, match="Lite"):
        list(pipe.run(Request(mode="swap", clip=clip, consent=True, clip_seconds=1, people=[cartoon, cartoon],
                              tier="lite")))
    with pytest.raises(InputError, match="up to 2"):
        list(pipe.run(Request(mode="swap", clip=clip, consent=True, clip_seconds=1,
                              people=[cartoon, cartoon, cartoon], tier="standard")))
    assert tiers.people_gate("standard", 3, allow_heavy=True) is None
    assert tiers.people_cap("max") == 8 == tiers.PEOPLE_HARD_CAP
    assert tiers.people_cap("lite", True) == tiers.TRY_ANYWAY_PEOPLE == 4
    assert estimate_people_seconds(368, 2, "standard") > 2 * estimate_people_seconds(368, 1, "standard") * 0.99


# --------------------------------------------------------------------------- readable names
def test_readable_output_names(tmp_path, cartoon):
    when = datetime(2026, 10, 17, 9, 5, 7)
    clip = tmp_path / "Duo Dance Take.mp4"
    r = Request(mode="swap", clip=clip, people=[tmp_path / "noir-avatar.jpg", None])
    assert readable_name(r, clip.stem, when) == "swap_duo-dance-take_noir-avatar-left_2026-10-17_090507"
    both = Request(mode="swap", people=[tmp_path / "noir-avatar.jpg"] * 2)
    assert who_label(both) == "noir-avatar-both"
    meme = Request(mode="meme", photo=tmp_path / "Me At The Beach.JPG")
    assert readable_name(meme, "side_eye", when) == "meme_side-eye_me-at-the-beach_2026-10-17_090507"


def test_outputs_are_named_readably_by_default(pipe, sample_photo):
    done = list(pipe.run_still(Request(mode="meme", meme="side_eye", photo=sample_photo, consent=True,
                                       style="none")))[-1]
    name = done.artifact.name
    assert re.fullmatch(r"meme_side-eye-still_photo_\d{4}-\d{2}-\d{2}_\d{6}\.png", name), name
    assert done.artifact.parent.name == done.artifact.stem


# --------------------------------------------------------------------------- caption layouts
def _ink_rows(path):
    img = Image.open(path).getchannel("A")
    w, h = img.size
    return [y / h for y in range(h) if max(img.crop((0, y, w, y + 1)).getdata()) > 0]


def test_caption_layouts_keep_captions_off_the_subject(tmp_path):
    long_bottom = "works on the first try and nobody touched it"
    top_rows = _ink_rows(media.render_meme_caption_png("when the code", long_bottom, tmp_path / "t.png", 540, 960,
                                                       layout="top"))
    assert max(top_rows) < 0.45  # both lines stacked in the top band: the body below stays clear
    edge_rows = _ink_rows(media.render_meme_caption_png("when you remember", "you left the oven on",
                                                        tmp_path / "e.png", 540, 960, layout="edges"))
    assert not [y for y in edge_rows if 0.15 < y < 0.70]  # the crash-zoomed face in the middle stays clear
    assert memes.RECIPES["dramatic_zoom"].caption_layout == "edges"
    assert memes.RECIPES["victory_dance"].caption_layout == "top"
    assert "hand" in memes.RECIPES["side_eye"].negative


# --------------------------------------------------------------------------- QoL
def test_friendly_errors():
    oom = errors.friendly(RuntimeError("swap failed in ComfyUI: CUDA error: out of memory"), "standard")
    assert "ran out of memory" in oom and "Lite" in oom
    assert "Lite" not in errors.friendly(RuntimeError("Allocation on device"), "lite").split("<details>")[0]
    miss = errors.friendly(RuntimeError("Value not in list: unet_name: 'Wan2.2-Animate-14B-Q4_K_M.gguf'"))
    assert "Wan2.2-Animate-14B-Q4_K_M.gguf" in miss and "check my models" in miss
    assert "Start it" in errors.friendly(ConnectionError("Connection refused"))
    assert errors.kind(errors.Cancelled("x")) == "cancelled"


def test_stop_cancels_before_the_next_stage(pipe, sample_photo):
    pipe.cancel()
    with pytest.raises(errors.Cancelled):
        list(pipe.run(Request(mode="meme", meme="side_eye", photo=sample_photo, consent=True, style="none")))


def test_people_scan_drops_duplicate_masks_and_template_prefix(tmp_path):
    from PIL import Image as _I
    from main_character_comfy.media import order_people
    from main_character_comfy.pipeline import Request, readable_name
    from datetime import datetime as _dt
    paths = []
    for i, box in enumerate([(10, 10, 60, 90), (20, 30, 55, 70), (120, 10, 180, 90)]):  # body, same torso, other
        im = _I.new("L", (200, 100), 0)
        im.paste(255, box)
        paths.append(tmp_path / f"m{i}.png"); im.save(paths[-1])
    people = order_people(paths)
    assert len(people) == 2 and people[0].cx < people[1].cx and people[0].mask == paths[0]
    r = Request(mode="swap", photo=tmp_path / "noir-avatar.jpg")
    when = _dt(2026, 10, 17, 9, 5, 7)
    assert readable_name(r, "template_duo-dance-take", when) == "swap_duo-dance-take_noir-avatar_2026-10-17_090507"
    assert readable_name(r, "duo", when, mode="people-scan", who="numbered").startswith("people-scan_duo_numbered_")


def test_people_scan_asks_sam3_for_several_detections():
    from main_character_comfy import graphs as _g, tiers as _t
    from main_character_comfy.pipeline import Settings as _S
    sg = _g.people_scan_graph(models=_t.models_for(_t.get("standard"), _S().models))
    texts = [n["inputs"]["text"] for n in sg.graph.values() if n["class_type"] == "CLIPTextEncode"]
    assert texts == ["person:8"]  # plain "person" = top-1 detection only in ComfyUI's SAM 3 encoder


def test_output_folder_has_only_results_on_top(pipe, tmp_path, sample_photo):
    """Swap: only the final mp4 + gif at the top; inputs, chunks, masks, previews, json in _work/."""
    from main_character_comfy.pipeline import Request, WORK_SUBDIR
    from main_character_comfy import mock_media
    clip = tmp_path / "duo.mp4"
    mock_media.fake_clip(clip, seconds=3) if hasattr(mock_media, "fake_clip") else None
    if not clip.exists():
        import subprocess
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=s=320x240:d=3:r=16",
                        "-f", "lavfi", "-i", "sine=d=3", "-shortest", str(clip)], check=True)
    req = Request(mode="swap", clip=clip, photo=sample_photo, consent=True, clip_seconds=3)
    evs = list(pipe.run(req))
    done = evs[-1]
    work = Path(done.data["work_dir"])
    top = sorted(p.name for p in work.iterdir())
    assert top == sorted([done.artifact.name, done.artifact.with_suffix(".gif").name, WORK_SUBDIR])
    assert (work / WORK_SUBDIR / "project.json").exists() and (work / WORK_SUBDIR / "mc_recipe.json").exists()
    assert Path(done.data["poster"]).parent.name == WORK_SUBDIR and Path(done.data["recipe"]).exists()


def test_avatar_swap_relight_off_isolated_reference_and_props(pipe, tmp_path):
    from PIL import Image as _I
    from main_character_comfy import graphs as _g
    from main_character_comfy.pipeline import (Request, use_relight, swap_prompt, swap_negative, MASK_GROW_PROPS)
    flat = tmp_path / "flat_avatar.png"
    im = _I.new("RGB", (300, 400), (20, 20, 60)); im.paste((200, 30, 30), (100, 80, 200, 380)); im.save(flat)
    av = Request(mode="swap", subject="avatar", remove_props=True)
    ph = Request(mode="swap", subject="photo")
    assert use_relight(av, "avatar") is False and use_relight(ph, "photo") is True
    assert use_relight(Request(relight=True), "avatar") is True and use_relight(Request(relight=False), "photo") is False
    pr = swap_prompt(av, "avatar", flat=True)
    assert "cel-shaded" in pr and "original colors" in pr and "no backpack" in pr
    assert "backpack" in swap_negative(av, "avatar") and "backpack" not in swap_negative(ph, "photo")
    from main_character_comfy import tiers as _t
    from main_character_comfy.pipeline import Settings as _S
    m = _t.models_for(_t.get("standard"), _S().models)
    sg = _g.swap_graph("x", models=m, frames=33, width=832, height=480, relight=False, isolate_ref=True,
                       mask_grow=MASK_GROW_PROPS)
    kinds = [n["class_type"] for n in sg.graph.values()]
    assert kinds.count("SAM3_Detect") == 1 and "CropByBBoxes" in kinds
    assert not any(n["inputs"].get("lora_name", "") == m.animate_relight_lora for n in sg.graph.values())
    grow = [n["inputs"]["expand"] for n in sg.graph.values() if n["class_type"] == "GrowMask"]
    assert MASK_GROW_PROPS in grow
    # the isolated reference feeds both CLIP Vision and Animate
    ref_id = next(k for k, n in sg.graph.items() if n.get("_meta", {}).get("title", "").startswith("Reference: your character"))
    cve = next(n for n in sg.graph.values() if n["class_type"] == "CLIPVisionEncode")
    assert cve["inputs"]["image"][0] == ref_id
    import json as _json
    oi = _json.loads((Path(__file__).parent / "fixtures" / "object_info_reference_live.json").read_text())
    assert validate_graph(sg.graph, oi, check_files=False) == []
