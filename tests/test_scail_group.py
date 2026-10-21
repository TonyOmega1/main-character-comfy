"""SCAIL-2 one-pass group swap + SeedVR2 upscale (A/B-verified on an RTX 5080, Oct 18 2026)."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from main_character_comfy import graphs, media, tiers
from main_character_comfy.config import Models
from main_character_comfy.pipeline import Request, group_engine_for, group_swap_prompt


def test_scail_size_and_frames():
    assert graphs.scail_size(720, 1280) == (576, 1024)
    assert graphs.scail_size(480, 832) == (480, 832)
    assert graphs.scail_size(1280, 720) == (1024, 576)
    assert graphs.scail_frames(80) == 81 and graphs.scail_frames(81) == 81 and graphs.scail_frames(78) == 81


def test_scail_group_graph_wiring():
    sg = graphs.scail_group_graph("p", models=Models(), frames=81, width=720, height=1280, people=3, views=3,
                                  object_indices="0,2")
    types = [n["class_type"] for n in sg.graph.values()]
    for t in ("SAM3_VideoTrack", "SCAIL2ColoredMask", "WanSCAILToVideo", "BatchImagesNode", "SaveVideo"):
        assert t in types
    assert set(sg.inputs) == {"clip", "ref", "ref_mask"} | {f"view_{k}" for k in range(3)} | {
        f"view_mask_{k}" for k in range(3)}
    w2v = next(n for n in sg.graph.values() if n["class_type"] == "WanSCAILToVideo")["inputs"]
    assert (w2v["width"], w2v["height"], w2v["length"], w2v["replacement_mode"]) == (576, 1024, 81, True)
    cm = next(n for n in sg.graph.values() if n["class_type"] == "SCAIL2ColoredMask")["inputs"]
    assert cm["object_indices"] == "0,2" and cm["sort_by"] == "left_to_right"
    loras = [n["inputs"]["lora_name"] for n in sg.graph.values() if n["class_type"] == "LoraLoaderModelOnly"]
    assert len(loras) == 3  # DPO + relight + LightX2V


def test_seedvr2_graph():
    sg = graphs.seedvr2_graph(models=Models(), width=1080, height=1920, fps=16)
    types = [n["class_type"] for n in sg.graph.values()]
    assert "SeedVR2Conditioning" in types and "VAEDecodeTiled" in types and set(sg.inputs) == {"clip"}


def test_group_engine_rules():
    assert group_engine_for(Request(mode="swap", tier="standard"), 81, 3)[0] == "chain"  # no looks described
    r = Request(mode="swap", tier="standard", people_desc=["a", "b", "c"])
    assert group_engine_for(r, 81, 3)[0] == "scail"
    assert group_engine_for(r, 81, 1)[0] == "chain"
    assert group_engine_for(r, 161, 3)[0] == "chain"
    assert group_engine_for(r, 81, 7)[0] == "chain"
    assert group_engine_for(Request(mode="swap", tier="lite"), 81, 2)[0] == "chain"
    assert group_engine_for(Request(mode="swap", group_engine="chain"), 81, 3)[0] == "chain"
    assert group_engine_for(r, 81, 2, found=3)[0] == "chain"  # someone stays as filmed
    assert group_engine_for(r, 81, 3, found=3)[0] == "scail"
    p = group_swap_prompt(r, 3, ["avatar", "photo", "avatar"])
    assert "3 characters" in p and "On the left is a." in p and "On the right is c." in p and "exact copy" in p
    assert group_engine_for(Request(mode="swap", group_engine="scail"), 81, 1)[0] == "scail"


def _figure(path: Path, colour, size=(400, 800)):
    im = Image.new("RGB", size, (200, 200, 200))
    d = ImageDraw.Draw(im)
    d.ellipse((150, 60, 250, 160), fill=colour)
    d.rectangle((130, 160, 270, 700), fill=colour)
    im.save(path)
    return path


def test_scail_references(tmp_path):
    refs = [_figure(tmp_path / f"r{k}.png", c) for k, c in enumerate([(200, 30, 30), (30, 120, 30), (30, 30, 200)])]
    boxes = [(60, 400, 260, 1180), (260, 420, 460, 1170), (460, 410, 660, 1180)]
    pal = graphs.SCAIL_PALETTE
    out = media.scail_references(refs, boxes, (720, 1280), tmp_path / "s", pal)
    m = np.asarray(Image.open(out["ref_mask"]))
    xs = []
    for k in range(3):
        sel = (m == np.array(pal[k])).all(2)
        assert sel.sum() > 1000  # everyone is in the composite, in their own colour
        xs.append(np.nonzero(sel)[1].mean())
    assert xs == sorted(xs)  # left to right, like the clip
    for k in range(3):
        vm = np.asarray(Image.open(out["view_masks"][k]))
        assert (vm == np.array(pal[k])).all(2).mean() > 0.1


def test_scail_references_enlarges_far_away_people(tmp_path):
    """People filmed far from the camera (~20% of the frame) still get big, readable cut-outs."""
    refs = [_figure(tmp_path / f"r{k}.png", c) for k, c in enumerate([(200, 30, 30), (30, 120, 30), (30, 30, 200)])]
    boxes = [(250, 700, 300, 960), (310, 700, 360, 955), (370, 700, 420, 960)]
    out = media.scail_references(refs, boxes, (720, 1280), tmp_path / "s", graphs.SCAIL_PALETTE)
    m = np.asarray(Image.open(out["ref_mask"])).any(2)
    ys = np.nonzero(m)[0]
    assert ys.max() - ys.min() > 0.4 * 1280 and ys.max() < 1280 - 10  # big and fully inside the frame


def test_scail_sigmas_come_from_the_shifted_model_and_steps_are_configurable():
    from main_character_comfy import tiers
    from main_character_comfy.pipeline import Settings
    m = tiers.models_for(tiers.get("standard"), Settings().models)
    sg = graphs.scail_group_graph("x", models=m, frames=81, width=576, height=1024, people=1, views=1)
    g = sg.graph
    shift = next(k for k, n in g.items() if n["class_type"] == "ModelSamplingSD3")
    sched = next(n for n in g.values() if n["class_type"] == "BasicScheduler")
    assert sched["inputs"]["model"][0] == shift and sched["inputs"]["steps"] == graphs.SCAIL_STEPS
    sg8 = graphs.scail_group_graph("x", models=m, frames=81, width=576, height=1024, people=1, views=1, steps=8)
    assert next(n for n in sg8.graph.values() if n["class_type"] == "BasicScheduler")["inputs"]["steps"] == 8
