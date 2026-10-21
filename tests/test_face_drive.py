"""Non-human references keep their own head: face_drive in the Animate graph, YuNet face check, per-person toggle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from main_character_comfy import cli, graphs, media, pipeline as P, ui
from main_character_comfy.backends import MockBackend
from main_character_comfy.config import Models, Settings
from main_character_comfy.media import run_ffmpeg
from main_character_comfy.pipeline import InputError, Pipeline, Request

FIX = Path(__file__).parent / "fixtures"
cv2 = pytest.importorskip("cv2")


def _classes(sg):
    return [n["class_type"] for n in sg.graph.values()]


def _anim_inputs(sg):
    return [n["inputs"] for n in sg.graph.values() if n["class_type"] == "WanAnimateToVideo"]


@pytest.mark.parametrize("drive", graphs.FACE_DRIVES)
def test_face_drive_graph_shapes(drive):
    sg = graphs.swap_graph("p", models=Models(), frames=77, face_drive=drive, person_mask=True)
    ins = _anim_inputs(sg)
    assert ins
    if drive == "none":
        assert all("face_video" not in i for i in ins)
    else:
        src = {sg.graph[i["face_video"][0]]["class_type"] for i in ins}
        assert src == {{"clip": "CropByBBoxes", "blank": "EmptyImage", "ref": "RepeatImageBatch",
                        "facepose": "CropByBBoxes"}[drive]}
    if drive == "blank":
        e = [n for n in sg.graph.values() if n["class_type"] == "EmptyImage" and n["inputs"]["width"] == 512][0]
        assert e["inputs"]["batch_size"] >= 77 and e["inputs"]["color"] == 0
    if drive == "ref":
        rep = [n for n in sg.graph.values() if n["class_type"] == "RepeatImageBatch"][0]
        assert rep["inputs"]["amount"] >= 77


def test_face_drive_carry_covers_the_lead_frames():
    sg = graphs.swap_graph("p", models=Models(), frames=73, carry=True, face_drive="blank")
    e = [n for n in sg.graph.values() if n["class_type"] == "EmptyImage" and n["inputs"]["width"] == 512][0]
    assert e["inputs"]["batch_size"] == 73 + graphs.ANIMATE_MOTION


def test_bad_face_drive_rejected():
    with pytest.raises(ValueError):
        graphs.swap_graph("p", models=Models(), frames=77, face_drive="mask")


def test_face_drive_graph_validates_against_live_schema():
    from main_character_comfy.validate import validate_graph
    live = json.loads((FIX / "object_info_reference_live.json").read_text())
    live.pop("$comment", None)
    for drive in graphs.FACE_DRIVES:
        for kw in ({"isolate_ref": True}, {"person_mask": True}, {"carry": True}):
            sg = graphs.swap_graph("p", models=Models(), frames=73, face_drive=drive, **kw)
            assert validate_graph(sg.graph, live, check_files=False) == [], (drive, kw)


def test_face_check_human_vs_non_human():
    assert media.face_check(FIX / "face_human_cartoon.jpg")[0] is False
    nh, why = media.face_check(FIX / "face_robot_teal.jpg")
    assert nh is True and "skin" in why  # has eyes + mouth, but a teal face
    nh, why = media.face_check(FIX / "face_robot_boxy.jpg")
    assert nh is True and "no human face" in why


def test_resolve_face_drive_choices():
    assert P.resolve_face_drive("on", FIX / "face_robot_teal.jpg")[0] == "clip"
    assert P.resolve_face_drive("off", FIX / "face_human_cartoon.jpg")[0] == P.NONHUMAN_FACE_DRIVE
    assert P.resolve_face_drive("auto", FIX / "face_robot_boxy.jpg")[0] == P.NONHUMAN_FACE_DRIVE
    assert P.resolve_face_drive("auto", FIX / "face_human_cartoon.jpg")[0] == "clip"
    assert P.resolve_face_drive("ref", None)[0] == "ref"
    assert P.resolve_face_drive("auto", Path("/nope/missing.png"))[0] == "clip"  # unreadable: keep normal


def test_face_check_without_opencv_assumes_human(monkeypatch):
    import builtins
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "cv2":
            raise ImportError
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake)
    assert media.face_check(FIX / "face_robot_boxy.jpg")[0] is None
    assert P.resolve_face_drive("auto", FIX / "face_robot_boxy.jpg")[0] == "clip"


def _clip(path, seconds):
    run_ffmpeg(["-f", "lavfi", "-i", f"testsrc2=size=1280x720:rate=30:duration={seconds}", "-c:v", "libx264",
                "-pix_fmt", "yuv420p", str(path)])
    return path


@pytest.fixture
def pipe(tmp_path):
    return Pipeline(MockBackend(), Settings(output_dir=tmp_path / "out"))


def test_single_swap_robot_keeps_its_head(pipe, tmp_path):
    evs = list(pipe.run(Request(mode="swap", clip=_clip(tmp_path / "c.mp4", 1), photo=FIX / "face_robot_teal.jpg",
                                consent=True, clip_seconds=1, style="none", clip_audio="mute")))
    proj = json.loads((Path(evs[-1].data["work_dir"]) / "_work" / "project.json").read_text())
    assert proj["swap_settings"]["face_drive"] == P.NONHUMAN_FACE_DRIVE
    assert any("not copying the dancer's face" in e.message for e in evs)


def test_multi_swap_per_person_face(pipe, tmp_path):
    req = Request(mode="swap", clip=_clip(tmp_path / "c.mp4", 1), consent=True, clip_seconds=1, clip_audio="mute",
                  people=[FIX / "face_human_cartoon.jpg", FIX / "face_human_cartoon.jpg"], face=["auto", "off"],
                  tier="max")
    evs = list(pipe.run(req))
    st = json.loads((Path(evs[-1].data["work_dir"]) / "_work" / "project.json").read_text())["swap_settings"]
    assert st["p1"]["face_drive"] == "clip" and st["p2"]["face_drive"] == P.NONHUMAN_FACE_DRIVE


def test_face_validation(pipe, tmp_path):
    with pytest.raises(InputError, match="face for person 1"):
        list(pipe.run(Request(mode="swap", clip=_clip(tmp_path / "c.mp4", 1), photo=FIX / "face_robot_teal.jpg",
                              consent=True, clip_seconds=1, face=["sideways"])))


def test_cli_and_ui_face_lists():
    a = cli.build_parser().parse_args(["--mode", "swap", "--clip", "c.mp4", "--people", "a.png,b.png,-",
                                       "--face", "auto,off,-"])
    assert a.face == "auto,off,-"
    assert ui.face_list("auto", ["person 2"], 3) == ["auto", "off", "auto"]
    assert ui.face_list("copy the dancer's face", [], 1) == ["on"]
