"""Open-weights Openjutsu: up to 8 references, 480p / 720p renders, saved reference cut-out, VOID prop eraser."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from main_character_comfy import cli, copy as C, graphs, tiers
from main_character_comfy.backends import MockBackend
from main_character_comfy.config import Settings
from main_character_comfy.media import count_frames, run_ffmpeg, video_size
from main_character_comfy.pipeline import (InputError, Pipeline, Request, estimate_animate_seconds,
                                           estimate_people_seconds, slot_names, who_label)

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture
def pipe(tmp_path):
    return Pipeline(MockBackend(), Settings(output_dir=tmp_path / "out"))


def _clip(path, seconds, size="1280x720"):
    run_ffmpeg(["-f", "lavfi", "-i", f"testsrc2=size={size}:rate=30:duration={seconds}", "-f", "lavfi", "-i",
                f"sine=frequency=440:duration={seconds}", "-c:a", "aac", "-shortest", "-c:v", "libx264",
                "-pix_fmt", "yuv420p", str(path)])
    return path


@pytest.fixture
def mascot(tmp_path):
    img = Image.new("RGBA", (400, 600), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((120, 40, 280, 200), fill=(0, 180, 255, 255), outline=(0, 0, 0, 255), width=6)
    d.rectangle((140, 200, 260, 560), fill=(255, 90, 0, 255), outline=(0, 0, 0, 255), width=6)
    p = tmp_path / "blue_mascot.png"
    img.save(p)
    return p


# ------------------------------------------------------------------ up to 8 people
def test_tier_people_caps_match_genjutsu_on_max():
    assert [tiers.people_cap(k) for k in ("lite", "standard", "max")] == [1, 2, 8]
    assert tiers.PEOPLE_HARD_CAP == 8
    assert tiers.people_gate("max", 8) is None
    assert "up to 8" in tiers.people_gate("max", 9)
    assert "Max (up to 8)" in tiers.people_gate("standard", 3)
    assert tiers.people_cap("standard", allow_heavy=True) == 4


def test_more_than_8_references_rejected(pipe, tmp_path, mascot):
    clip = _clip(tmp_path / "crowd.mp4", 1)
    with pytest.raises(InputError, match="up to 8 references"):
        list(pipe.run(Request(mode="swap", clip=clip, consent=True, clip_seconds=1, people=[mascot] * 9,
                              tier="max")))


def test_max_tier_swaps_five_people_one_pass_each(pipe, tmp_path, mascot, monkeypatch):
    monkeypatch.setenv("MC_MOCK_PEOPLE", "6")
    clip = _clip(tmp_path / "group.mp4", 1)
    people = [mascot, None, mascot, mascot, mascot, mascot]  # '-' for person 2
    done = list(pipe.run(Request(mode="swap", clip=clip, consent=True, clip_seconds=1, people=people,
                                 tier="max", style="none")))[-1]
    proj = json.loads((Path(done.data["work_dir"]) / "_work" / "project.json").read_text())
    assert [j for j in proj["jobs"] if j.startswith("swap")] == ["swap_p1", "swap_p3", "swap_p4", "swap_p5", "swap_p6"]
    assert len(proj["people"]) == 6 and proj["people"][1]["ref"] == ""
    assert done.artifact.exists()


def test_slot_names_and_who_label_scale_to_8(tmp_path):
    assert slot_names(8) == [f"p{i}" for i in range(1, 9)]
    r = Request(mode="swap", people=[tmp_path / "me.png"] * 8)
    assert who_label(r) == "me-all"
    assert estimate_people_seconds(80, 8, "max") > 7 * estimate_people_seconds(80, 1, "max")


def test_cli_parses_8_people_with_dashes(tmp_path):
    a = cli.build_parser().parse_args(["--mode", "swap", "--people", "a.png,-,b.png,-,-,-,-,c.png",
                                       "--resolution", "720p", "--erase-props", "backpack"])
    assert a.people.count(",") == 7 and a.resolution == "720p" and a.erase_props == "backpack"


# ------------------------------------------------------------------ 480p / 720p
def test_resolution_sizes_and_defaults():
    assert tiers.animate_size_for("standard") == (480, 832)  # 832x480 stays the default
    assert tiers.animate_size_for("standard", "720p") == (720, 1280)
    assert tiers.animate_size_for("max", "480p") == (480, 832)
    assert tiers.animate_size_for("max") == (576, 1024)
    assert tiers.normalize_resolution("720") == "720p"
    with pytest.raises(ValueError):
        tiers.normalize_resolution("4k")
    assert tiers.resolution_gate("lite", "720p") and tiers.resolution_gate("lite", "720p", allow_heavy=True) is None
    assert tiers.resolution_gate("standard", "720p") is None and tiers.resolution_warning("standard", "720p")
    assert tiers.resolution_warning("max", "720p") == ""
    assert 2.2 < tiers.pixel_scale("standard", "720p") < 2.4
    assert estimate_animate_seconds(80, "replace", "standard", "720p") > 2 * estimate_animate_seconds(80, "replace", "standard")


def test_720p_swap_renders_at_720_and_warns_on_16gb(pipe, tmp_path, mascot):
    clip = _clip(tmp_path / "wide.mp4", 1)
    evs = list(pipe.run(Request(mode="swap", clip=clip, photo=mascot, consent=True, clip_seconds=1,
                                resolution="720p", tier="standard", style="none")))
    assert any("720p on a 16 GB card" in e.message for e in evs)
    work = Path(evs[-1].data["work_dir"]) / "_work"
    assert video_size(work / "inputs" / "clip.mp4") == (1280, 720)
    proj = json.loads((work / "project.json").read_text())
    assert proj["resolution"] == {"choice": "720p", "render": [720, 1280]}
    assert video_size(evs[-1].artifact) == (1920, 1080)  # finishing still upscales to 1080p


def test_720p_blocked_on_lite_and_bad_resolution_rejected(pipe, tmp_path, mascot):
    clip = _clip(tmp_path / "wide.mp4", 1)
    with pytest.raises(InputError, match="16 GB"):
        list(pipe.run(Request(mode="swap", clip=clip, photo=mascot, consent=True, clip_seconds=1,
                              resolution="720p", tier="lite", allow_heavy=False)))
    with pytest.raises(InputError, match="unknown resolution"):
        list(pipe.run(Request(mode="swap", clip=clip, photo=mascot, consent=True, clip_seconds=1, resolution="8k")))


def test_default_swap_still_832x480(pipe, tmp_path, mascot):
    clip = _clip(tmp_path / "wide.mp4", 1)
    done = list(pipe.run(Request(mode="swap", clip=clip, photo=mascot, consent=True, clip_seconds=1,
                                 style="none")))[-1]
    assert video_size(Path(done.data["work_dir"]) / "_work" / "inputs" / "clip.mp4") == (832, 480)


# ------------------------------------------------------------------ cut-out reference saved to _work
def test_avatar_swap_saves_the_cutout_reference(pipe, tmp_path, mascot):
    clip = _clip(tmp_path / "wide.mp4", 1)
    evs = list(pipe.run(Request(mode="swap", clip=clip, photo=mascot, consent=True, clip_seconds=1,
                                subject="avatar", style="none")))
    work = Path(evs[-1].data["work_dir"]) / "_work"
    assert (work / "ref_cutout.png").exists()
    proj = json.loads((work / "project.json").read_text())
    assert proj["ref_cutouts"]["solo"].endswith("ref_cutout.png")
    assert any("cut-out reference" in e.message for e in evs)


def test_cutout_save_node_only_when_isolating(models):
    iso = graphs.swap_graph("x", models=models, frames=77, isolate_ref=True)
    plain = graphs.swap_graph("x", models=models, frames=77)
    node = iso.meta["extra_outputs"]["ref_cutout"]
    assert iso.graph[node]["class_type"] == "SaveImage"
    assert "extra_outputs" not in plain.meta
    assert sum(n["class_type"] == "SaveImage" for n in plain.graph.values()) == 0


def test_multi_swap_saves_one_cutout_per_avatar(pipe, tmp_path, mascot):
    clip = _clip(tmp_path / "duo.mp4", 1)
    done = list(pipe.run(Request(mode="swap", clip=clip, consent=True, clip_seconds=1, people=[mascot, mascot],
                                 subject="avatar", style="none")))[-1]
    work = Path(done.data["work_dir"]) / "_work"
    assert (work / "ref_cutout_p1.png").exists() and (work / "ref_cutout_p2.png").exists()


# ------------------------------------------------------------------ VOID "erase props first"
def test_void_graph_shape(models):
    sg = graphs.void_erase_graph("backpack", models=models, frames=80, width=480, height=832)
    classes = [n["class_type"] for n in sg.graph.values()]
    for c in ("VOIDInpaintConditioning", "VOIDSampler", "VOIDWarpedNoise", "VOIDWarpedNoiseSource",
              "OpticalFlowLoader", "SAM3_Detect", "SaveVideo"):
        assert c in classes
    cond = next(n for n in sg.graph.values() if n["class_type"] == "VOIDInpaintConditioning")
    assert (cond["inputs"]["width"], cond["inputs"]["height"]) == (384, 672)  # portrait clip -> portrait VOID
    det_txt = next(n for n in sg.graph.values() if n.get("_meta", {}).get("title") == "What to erase")
    assert det_txt["inputs"]["text"] == "backpack"
    one = graphs.void_erase_graph("x", models=models, frames=80, width=832, height=480, two_pass=False)
    assert "VOIDWarpedNoise" not in [n["class_type"] for n in one.graph.values()]
    c1 = next(n for n in one.graph.values() if n["class_type"] == "VOIDInpaintConditioning")
    assert (c1["inputs"]["width"], c1["inputs"]["height"]) == (672, 384)
    assert {models.void_pass1, models.void_pass2} <= {n["inputs"].get("unet_name") for n in sg.graph.values()}


def test_void_lengths_are_8k_minus_3():
    assert [graphs.void_valid_length(n) for n in (1, 5, 6, 32, 45, 46, 157, 197)] == [5, 5, 13, 37, 45, 53, 157, 197]
    assert all((graphs.void_valid_length(n) + 3) % 8 == 0 for n in range(1, 300))


def test_void_chunks_cover_long_clips():
    assert graphs.plan_void_chunks(150) == [(0, 150)]
    chunks = graphs.plan_void_chunks(480)
    assert sum(c for _, c in chunks) == 480 and all(c <= graphs.VOID_MAX_FRAMES for _, c in chunks)
    assert [f for f, _ in chunks] == sorted(f for f, _ in chunks)


def test_erase_props_prepass_feeds_the_swap(pipe, tmp_path, mascot):
    clip = _clip(tmp_path / "wide.mp4", 2)
    evs = list(pipe.run(Request(mode="swap", clip=clip, photo=mascot, consent=True, clip_seconds=2,
                                erase_props="backpack", style="none")))
    work = Path(evs[-1].data["work_dir"]) / "_work"
    proj = json.loads((work / "project.json").read_text())
    assert "erase" in proj["jobs"] and proj["erase_props"]["what"] == "backpack"
    assert count_frames(work / "erased_clip.mp4") == 32
    jobs = list(proj["jobs"])
    assert jobs.index("erase") < jobs.index("swap")
    assert any("Erasing 'backpack'" in e.message for e in evs)


def test_erase_props_validation(pipe, tmp_path, mascot):
    clip = _clip(tmp_path / "wide.mp4", 1)
    with pytest.raises(InputError, match="Swap Me In"):
        list(pipe.run(Request(mode="vibe", clip=clip, photo=mascot, consent=True, clip_seconds=1,
                              erase_props="backpack")))
    with pytest.raises(InputError, match="short"):
        list(pipe.run(Request(mode="swap", clip=clip, photo=mascot, consent=True, clip_seconds=1,
                              erase_props="x" * 81)))


def test_off_by_default_no_erase_job(pipe, tmp_path, mascot):
    clip = _clip(tmp_path / "wide.mp4", 1)
    done = list(pipe.run(Request(mode="swap", clip=clip, photo=mascot, consent=True, clip_seconds=1,
                                 style="none")))[-1]
    proj = json.loads((Path(done.data["work_dir"]) / "_work" / "project.json").read_text())
    assert "erase" not in proj["jobs"]


# ------------------------------------------------------------------ pitch copy
def test_openjutsu_copy_no_hashtags_no_real_name():
    root = Path(__file__).resolve().parent.parent
    readme = (root / "README.md").read_text(encoding="utf-8")
    for text in (C.SWAP_INTRO, readme):
        assert "Openjutsu" in text and "#" + "OpenSourceHiggsfield" not in text
    assert "Why it's innovative" in readme or "Why MC is innovative" in readme
    assert "Oroku Saki" in readme and "@Tony_Omega" in readme
    assert "1-8" in C.SWAP_INTRO


def test_void_graph_validates_against_live_comfyui_schema(models):
    oi = json.loads((FIX / "object_info_void_live.json").read_text())
    oi.pop("$comment", None)
    # core VAEDecodeTiled (video decodes are tiled now) from the full live snapshot of the same ComfyUI
    oi.setdefault("VAEDecodeTiled", json.loads((FIX / "object_info_reference_live.json").read_text())["VAEDecodeTiled"])
    from main_character_comfy.validate import validate_graph

    for two in (True, False):
        sg = graphs.void_erase_graph("backpack", models=models, frames=45, width=832, height=480, two_pass=two)
        assert validate_graph(sg.graph, oi, check_files=False) == []
