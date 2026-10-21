"""Style lock (pending a real test), hard cuts + who-is-who across them, per-person drivers. CPU only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from main_character_comfy import graphs, media, shots, stylelock
from main_character_comfy.backends import MockBackend
from main_character_comfy.config import Models, Settings
from main_character_comfy.media import count_frames, media_duration, run_ffmpeg
from main_character_comfy.pipeline import InputError, Pipeline, Request, driver_label, parse_driver, swap_negative
from main_character_comfy.validate import validate_graph

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def live():
    return json.loads((FIX / "object_info_reference_live.json").read_text())


@pytest.fixture
def pipe(tmp_path):
    return Pipeline(MockBackend(), Settings(output_dir=tmp_path / "out"))


def _drawn(path: Path) -> Path:
    img = Image.new("RGB", (400, 600), (245, 240, 230))
    d = ImageDraw.Draw(img)
    d.ellipse((130, 40, 270, 180), fill=(250, 210, 170), outline=(0, 0, 0), width=6)
    d.rectangle((140, 190, 260, 420), fill=(200, 30, 40), outline=(0, 0, 0), width=6)
    for x in (150, 210):
        d.rectangle((x, 420, x + 40, 580), fill=(30, 40, 90), outline=(0, 0, 0), width=6)
    img.save(path)
    return path


def _photo(path: Path) -> Path:
    rnd = np.random.default_rng(3)
    y, x = np.mgrid[0:600, 0:400]
    base = np.stack([90 + x * 0.3, 70 + y * 0.15, 110 + (x + y) * 0.05], axis=2)
    img = np.clip(base + rnd.normal(0, 14, base.shape), 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=92)
    return path


def _two_shot_clip(path: Path, a: float = 1.5, b: float = 1.5) -> Path:
    run_ffmpeg(["-f", "lavfi", "-i", f"testsrc2=size=1280x720:rate=30:duration={a}",
                "-f", "lavfi", "-i", f"smptehdbars=size=1280x720:rate=30:duration={b}",
                "-f", "lavfi", "-i", f"sine=frequency=440:duration={a + b}",
                "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]", "-map", "2:a",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)])
    return path


# --------------------------------------------------------------------------- style lock
def test_style_lock_is_off_by_default_and_pending():
    assert Request().style_lock == "off" == stylelock.DEFAULT_MODE
    off = stylelock.settings_for("off", ["avatar"])
    assert not off.active
    auto = stylelock.settings_for("auto", ["photo", "avatar"])
    assert auto.active and auto.relight is False and auto.cfg > 1.0 and auto.pending
    assert not stylelock.settings_for("auto", ["photo"]).active
    assert stylelock.settings_for("on", ["photo"]).active
    assert stylelock.normalize("bogus") == "off"
    rec = auto.as_record()
    assert rec["pending"] and "real test" in stylelock.PENDING_NOTE.lower()


def test_style_negative_only_when_the_lock_is_on(tmp_path):
    ref = _drawn(tmp_path / "d.png")
    plain = swap_negative(Request(photo=ref), "avatar")
    locked = swap_negative(Request(photo=ref, style_lock="on"), "avatar")
    assert "photorealistic" in locked and len(locked) > len(plain or "")


def test_classifier_tells_flat_drawings_from_photos(tmp_path):
    k = stylelock.classify_reference(_drawn(tmp_path / "d.png"))
    assert k.label == "avatar" and k.why
    p = stylelock.classify_reference(_photo(tmp_path / "p.jpg"))
    assert p.label == "photo"
    assert stylelock.hint_text(k) == ""  # sure: no nag
    assert stylelock.hint_text(stylelock.RefKind("photo", False, "", {}))  # unsure: ask


def test_classifier_is_unsure_on_mid_grey_images(tmp_path):
    rnd = np.random.default_rng(1)
    g = np.clip(128 + rnd.normal(0, 2.0, (300, 200)), 0, 255).astype(np.uint8)
    Image.fromarray(np.stack([g, g, g], 2)).save(tmp_path / "g.png")
    k = stylelock.classify_reference(tmp_path / "g.png")
    assert k.label in ("photo", "avatar")  # a lean only; the UI asks
    assert isinstance(k.confident, bool)


def test_animate_graph_style_knobs_validate(live):
    sg = graphs.swap_graph("x", models=Models(), frames=77, person_mask=True, cfg=2.5, negative="photo",
                           composite_grow=2, composite_blur=3.0)
    ks = next(n for n in sg.graph.values() if n["class_type"] == "KSampler")
    assert ks["inputs"]["cfg"] == 2.5 and sg.meta["cfg"] == 2.5
    grows = [n["inputs"].get("expand") for n in sg.graph.values() if n["class_type"] == "GrowMask"]
    assert 2 in grows
    assert validate_graph(sg.graph, live, check_files=False) == []
    default = graphs.swap_graph("x", models=Models(), frames=77, person_mask=True)
    assert next(n for n in default.graph.values() if n["class_type"] == "KSampler")["inputs"]["cfg"] == 1.0


def test_scail_group_cfg_and_negative(live):
    sg = graphs.scail_group_graph("p", models=Models(), frames=81, width=1280, height=720, people=2, views=2,
                                  cfg=2.5, negative="photorealistic")
    neg = [n for n in sg.graph.values() if n["class_type"] == "CLIPTextEncode" and n["inputs"]["text"] == "photorealistic"]
    assert neg
    assert any(n["class_type"] == "SamplerCustom" and n["inputs"]["cfg"] == 2.5 for n in sg.graph.values())


def test_blend_upscale_mixes_two_takes(tmp_path):
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    run_ffmpeg(["-f", "lavfi", "-i", "color=c=black:size=320x180:rate=16:duration=0.5", "-pix_fmt", "yuv420p", str(a)])
    run_ffmpeg(["-f", "lavfi", "-i", "color=c=white:size=640x360:rate=16:duration=0.5", "-pix_fmt", "yuv420p", str(b)])
    out = media.blend_upscale(a, b, tmp_path / "m.mp4", alpha=0.4, size=(640, 360), fps=16)
    fr = media.grab_frame(out, tmp_path / "m.png", 2, 16)
    v = np.asarray(Image.open(fr).convert("L")).mean()
    assert 80 < v < 125  # ~0.4 * 255
    assert media.probe_size(out) == (640, 360) and count_frames(out) == 8


# --------------------------------------------------------------------------- hard cuts
def test_flashes_are_not_cuts_but_layout_jumps_are():
    d = [0, .03, .04, .33, .03, .03, .45, .03, .03, .03, .03, .03]
    s = [0, .02, .02, .27, .02, .02, .70, .02, .02, .02, .02, .02]
    found = shots.detect_cuts_from_scores(d, s, min_shot=2)
    assert [(x.start, x.end) for x in found] == [(0, 6), (6, 12)]
    assert len(shots.detect_cuts_from_scores(d, None)) == 1  # colour alone below HIST_ALONE: no cut


def test_tiny_shots_are_merged():
    merged = shots.merge_short([shots.Shot(0, 30), shots.Shot(30, 32), shots.Shot(32, 60)], 4)
    assert [(x.start, x.end) for x in merged] == [(0, 32), (32, 60)]
    assert shots.merge_short([shots.Shot(0, 2), shots.Shot(2, 40)], 4)[0].start == 0


def test_detect_cuts_on_a_real_two_shot_clip(tmp_path):
    clip = _two_shot_clip(tmp_path / "cut.mp4")
    found = shots.detect_cuts(clip)
    assert len(found) == 2 and abs(found[1].start - 45) <= 1
    assert "1 hard cut" in shots.summary(found, 30)


def test_cuts_match_hand_labelled_shots():
    """Optional local check: MC_CUT_REFS=/path/clip.mp4:53,421,503;/other.mp4:29,60 (clips are not in the repo)."""
    import os

    refs = [r for r in os.environ.get("MC_CUT_REFS", "").split(";") if ":" in r]
    if not refs:
        pytest.skip("no local hand-labelled clips (set MC_CUT_REFS)")
    for r in refs:
        video, cuts = r.rsplit(":", 1)
        if not Path(video).exists():
            pytest.skip(f"{video} not on this machine")
        assert [s.start for s in shots.detect_cuts(video)[1:]] == [int(c) for c in cuts.split(",")]


def test_identity_links_across_shots_and_never_inside_one():
    red, blue = np.zeros((60, 40, 3), np.uint8), np.zeros((60, 40, 3), np.uint8)
    red[..., 0], blue[..., 2] = 200, 200
    fr, fb = shots.appearance(red), shots.appearance(blue)
    sig = [shots.Sighting(0, 0, fr), shots.Sighting(0, 1, fb), shots.Sighting(1, 0, fb), shots.Sighting(1, 1, fr)]
    assert shots.link_identities(sig) == [0, 1, 1, 0]
    per = shots.assign_across_shots(["A", "B"], sig)
    assert per[0] == ["A", "B"] and per[1] == ["B", "A"]  # they swapped sides after the cut
    twins = [shots.Sighting(0, 0, fr), shots.Sighting(0, 1, fr)]
    assert len(set(shots.link_identities(twins))) == 2  # same look, same shot: still two people


def test_new_person_after_a_cut_is_left_as_filmed():
    red, green = np.zeros((60, 40, 3), np.uint8), np.zeros((60, 40, 3), np.uint8)
    red[..., 0], green[..., 1] = 200, 200
    sig = [shots.Sighting(0, 0, shots.appearance(red)), shots.Sighting(1, 0, shots.appearance(green))]
    assert shots.assign_across_shots(["A"], sig)[1] == [None]


def test_swap_splits_at_hard_cuts(pipe, tmp_path):
    clip = _two_shot_clip(tmp_path / "cut.mp4")
    ref = _drawn(tmp_path / "me.png")
    req = Request(mode="swap", clip=clip, consent=True, clip_seconds=3, people=[ref, None], style="none",
                  group_engine="chain")
    evs = list(pipe.run(req))
    assert evs[-1].status == "complete"
    proj = json.loads((Path(evs[-1].data["work_dir"]) / "_work" / "project.json").read_text())
    assert len(proj["shots"]["shots"]) == 2 and "who" in proj["shots"]
    assert any(j.startswith("people_shot01") for j in proj["jobs"])
    assert any("_s1_p1" in j for j in proj["jobs"])
    assert abs(media_duration(evs[-1].artifact) - 3.0) < 0.2
    assert any("hard cut" in e.message for e in evs)


def test_split_cuts_off_keeps_one_pass(pipe, tmp_path):
    clip = _two_shot_clip(tmp_path / "cut.mp4")
    req = Request(mode="swap", clip=clip, consent=True, clip_seconds=3, people=[_drawn(tmp_path / "me.png"), None],
                  style="none", group_engine="chain", split_cuts="off")
    evs = list(pipe.run(req))
    proj = json.loads((Path(evs[-1].data["work_dir"]) / "_work" / "project.json").read_text())
    assert "shots" not in proj and not any("_s1" in j for j in proj["jobs"])


# --------------------------------------------------------------------------- per-person drivers
def test_parse_driver():
    assert parse_driver("", 2, 0) == ("self", None)
    assert parse_driver("person:2", 2, 0) == ("person", 1)
    assert parse_driver("p1", 2, 0) == ("self", None)  # following yourself = own moves
    assert parse_driver("2", 3, 0) == ("person", 1)
    with pytest.raises(InputError, match="has 2 people"):
        parse_driver("person:5", 2, 0)
    with pytest.raises(InputError, match="wasn't found"):
        parse_driver("/nope/moves.mp4", 2, 0)
    assert driver_label("person", 1) == "follows person 2" and driver_label("self", None) == "own moves"


def test_driver_layout_stays_inside_the_frame():
    L = media.driver_layout((900, 100, 1100, 700), (50, 200, 250, 710), (1280, 720), (1280, 720))
    assert L["x"] >= 0 and L["y"] >= 0 and L["x"] + L["w"] <= 1280 and L["y"] + L["h"] <= 720
    assert L["w"] % 2 == 0 and L["h"] % 2 == 0
    assert L["crop_x"] + L["crop_w"] <= 1280 and L["crop_y"] + L["crop_h"] <= 720


def test_compose_driver_puts_the_other_figure_in_place(tmp_path):
    src = tmp_path / "duo.mp4"
    # left half grey, a white "dancer" on the right
    run_ffmpeg(["-f", "lavfi", "-i", "color=c=0x202020:size=320x180:rate=16:duration=1",
                "-vf", "drawbox=x=220:y=40:w=40:h=120:color=white:t=fill", "-pix_fmt", "yuv420p", str(src)])
    out = media.compose_driver(src, tmp_path / "drv.mp4", dst_box=(40, 40, 80, 160), src_box=(220, 40, 260, 160),
                               frame_size=(320, 180), frames=16, fps=16)
    assert count_frames(out) == 16
    a = np.asarray(Image.open(media.grab_frame(out, tmp_path / "f.png", 3, 16)).convert("L")).astype(float)
    assert a[60:140, 50:70].mean() > 180  # the dancer now stands where person 1 was


def test_person_follows_another_persons_moves(pipe, tmp_path, live):
    from main_character_comfy.media import run_ffmpeg as rf

    clip = tmp_path / "duo.mp4"
    rf(["-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(clip)])
    ref = _drawn(tmp_path / "me.png")
    req = Request(mode="swap", clip=clip, consent=True, clip_seconds=2, people=[ref, ref], style="none",
                  people_driver=["person:2", ""], tier="standard")
    evs = list(pipe.run(req))
    work = Path(evs[-1].data["work_dir"]) / "_work"
    proj = json.loads((work / "project.json").read_text())
    assert proj["people"][0]["moves"] == "follows person 2" and proj["people"][1]["moves"] == "own moves"
    assert "p1" in proj["drivers"] and Path(proj["drivers"]["p1"]["video"]).exists()
    assert proj["group_engine"]["engine"] == "chain"
    sg = graphs.swap_graph("x", models=Models(), frames=33, person_mask=True, driver=True)
    assert "driver" in sg.inputs and sg.meta["driver"]
    assert any(n["class_type"] == "MaskComposite" for n in sg.graph.values())
    assert validate_graph(sg.graph, live, check_files=False) == []
    with pytest.raises(ValueError):
        graphs.swap_graph("x", models=Models(), frames=33, driver=True)


# --------------------------------------------------------------------------- InfiniteTalk duet keys (Oct 19)
def test_duet_sends_speaker_b_under_the_dynamic_combo_prefix(live):
    sg = graphs.lipsync_graph("two people talk", models=Models(), audio_seconds=6.0, two_speakers=True)
    talk = [n for n in sg.graph.values() if n["class_type"] == "WanInfiniteTalkToVideo"]
    assert talk
    for n in talk:
        ins = n["inputs"]
        assert ins["mode"] == "two_speakers"
        assert {"mode.audio_encoder_output_2", "mode.mask_1", "mode.mask_2"} <= set(ins)
        assert not {"audio_encoder_output_2", "mask_1", "mask_2"} & set(ins)
    assert validate_graph(sg.graph, live, check_files=False) == []


def test_validator_catches_undotted_dynamic_combo_inputs(live):
    sg = graphs.lipsync_graph("two people talk", models=Models(), audio_seconds=3.0, two_speakers=True)
    bad = json.loads(json.dumps(sg.graph))
    for n in bad.values():
        if n["class_type"] == "WanInfiniteTalkToVideo":
            n["inputs"]["mask_1"] = n["inputs"].pop("mode.mask_1")
    problems = validate_graph(bad, live, check_files=False)
    assert any("must be sent as 'mode.mask_1'" in p for p in problems)
    single = graphs.lipsync_graph("one talks", models=Models(), audio_seconds=3.0)
    wrong = json.loads(json.dumps(single.graph))
    for n in wrong.values():
        if n["class_type"] == "WanInfiniteTalkToVideo":
            n["inputs"]["mode.mask_1"] = ["1", 0]
    assert any("not an input of mode='single_speaker'" in p for p in validate_graph(wrong, live, check_files=False))


# ------------------------------------------- cut cases ported from the node-study practice tests (Oct 19)
def _segments_clip(path, segments=(("red", 1.0), ("blue", 2.0), ("0x208020", 1.5)), fps=24, size="320x180"):
    """Hard cuts between coloured segments, with a moving box + temporal noise inside each (practice/conftest)."""
    inputs, filters = [], []
    for i, (color, dur) in enumerate(segments):
        inputs += ["-f", "lavfi", "-i", f"color=c={color}:s={size}:r={fps}:d={dur}"]
        filters.append(f"[{i}:v]drawbox=x='mod(t*120\\,280)':y=60:w=40:h=40:color=white@0.9:t=fill,"
                       f"noise=alls=12:allf=t[v{i}]")
    n = len(segments)
    filt = ";".join(filters) + ";" + "".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[v]"
    run_ffmpeg([*inputs, "-filter_complex", filt, "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-r", str(fps), str(path)])
    return path


def test_ported_three_shot_clip_exact_cuts(tmp_path):
    found = shots.detect_cuts(_segments_clip(tmp_path / "three.mp4"))
    assert [(s.start, s.end) for s in found] == [(0, 24), (24, 72), (72, 108)]


def test_ported_motion_inside_a_shot_is_not_a_cut():
    rng = np.random.default_rng(0)
    frames = np.full((60, 36, 64, 3), 100, np.float32)
    for i in range(60):  # a bright box sweeping across + noise
        frames[i, 10:20, (i % 50):(i % 50) + 10] = 250
        frames[i] += rng.normal(0, 4, frames[i].shape)
    d, s = shots.cut_scores(np.clip(frames, 0, 255).astype(np.uint8))
    assert len(shots.detect_cuts_from_scores(d, s)) == 1


def test_ported_one_frame_flash_after_a_cut_is_one_cut():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 255, (36, 64, 3)).astype(np.uint8)
    b = rng.integers(0, 255, (36, 64, 3)).astype(np.uint8)
    frames = np.stack([a] * 20 + [b] * 20)
    frames[22] = 0  # 1-frame black flash right after the cut
    d, s = shots.cut_scores(frames)
    found = shots.detect_cuts_from_scores(d, s, min_shot=6)
    assert len(found) == 2 and found[1].start in (20, 23)


# ---------------- optional second detector (PySceneDetect) ----------------
def test_compare_cuts_tolerance_and_leftovers():
    from main_character_comfy import shots as S

    got = S.compare_cuts([53, 421, 503], [52, 162, 240, 332, 422, 503])
    assert got == {"agree": [53, 421, 503], "mc_only": [], "other_only": [162, 240, 332]}
    assert S.compare_cuts([10, 40], [11])["mc_only"] == [40]


def test_second_opinion_is_a_report_not_a_vote(tmp_path, monkeypatch):
    from main_character_comfy import shots as S

    monkeypatch.setattr(S, "scenedetect_cuts", lambda v: [35, 108])
    rep = S.second_opinion("x.mp4", [35], fps=16)
    assert rep["agree"] == [35] and rep["other_only"] == [108] and "6.8 s" in rep["note"] and "#guide-cuts" in rep["note"]
    monkeypatch.setattr(S, "scenedetect_cuts", lambda v: None)
    assert S.second_opinion("x.mp4", [35]) is None


def test_scenedetect_on_a_real_cut_if_installed(tmp_path):
    pytest.importorskip("scenedetect")
    from main_character_comfy import shots as S
    from main_character_comfy.media import run_ffmpeg

    a, b, v = tmp_path / "a.mp4", tmp_path / "b.mp4", tmp_path / "v.mp4"
    run_ffmpeg(["-f", "lavfi", "-i", "testsrc=size=320x240:rate=16:duration=2", "-pix_fmt", "yuv420p", str(a)])
    run_ffmpeg(["-f", "lavfi", "-i", "mandelbrot=size=320x240:rate=16", "-t", "2", "-pix_fmt", "yuv420p", str(b)])
    run_ffmpeg(["-i", str(a), "-i", str(b), "-filter_complex", "[0:v][1:v]concat=n=2:v=1[v]", "-map", "[v]",
                "-pix_fmt", "yuv420p", str(v)])
    got = S.scenedetect_cuts(v)
    assert got is not None and any(abs(c - 32) <= 2 for c in got)


def test_style_env_overrides_only_apply_while_the_lock_is_on(monkeypatch, tmp_path):
    """Prompt A/B knobs (MC_STYLE_*): no effect with the lock off; CFG clamped; tags lead the prompt; Tony's eye line
    is appended to the negative; unset env = the shipped defaults."""
    from main_character_comfy import pipeline as P, stylelock as S
    base = S.settings_for("on", ["avatar"])
    assert base.cfg == S.STYLE_CFG and base.negative_extra == S.STYLE_NEGATIVE and base.positive_prefix == ""
    monkeypatch.setenv(S.ENV_CFG, "9")
    monkeypatch.setenv(S.ENV_NEG, "photo, 3D render")
    monkeypatch.setenv(S.ENV_NEG_EXTRA, "visible eyes behind sunglasses, transparent lenses")
    monkeypatch.setenv(S.ENV_POS, "flat 2D cartoon, cel shading.")
    on = S.settings_for("on", ["avatar"])
    assert on.cfg == 4.0
    assert on.negative_extra == "photo, 3D render, visible eyes behind sunglasses, transparent lenses"
    assert on.positive_prefix == "flat 2D cartoon, cel shading"
    assert set(on.overrides) == {"cfg", "negative_extra", "positive_prefix"}
    off = S.settings_for("off", ["avatar"])
    assert not off.active and off.cfg == 1.0 and off.positive_prefix == ""
    req = P.Request(mode="swap", style_lock="on")
    assert P.swap_prompt(req, "avatar", True).startswith("flat 2D cartoon, cel shading. The character")
    assert P.FLAT_STYLE not in P.swap_prompt(req, "avatar", True)
    assert not P.swap_prompt(P.Request(mode="swap"), "avatar", True).startswith("flat 2D")
    assert "transparent lenses" in P.swap_negative(req, "avatar")
    monkeypatch.setenv(S.ENV_CFG, "abc")
    assert S.settings_for("on", ["avatar"]).cfg == S.STYLE_CFG
