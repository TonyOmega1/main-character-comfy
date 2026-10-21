import json

import pytest

from main_character_comfy import graphs
from main_character_comfy.validate import validate_graph

from .conftest import FIX

TESTED = FIX / "tested_workflows"


def _strip(graph, drop=("text", "filename_prefix", "max_new_tokens", "noise_seed", "seed", "image", "audio")):
    return [
        (n["class_type"], {k: v for k, v in n["inputs"].items() if k not in drop})
        for _, n in sorted(graph.items(), key=lambda kv: int(kv[0]))
    ]


def test_lipsync_matches_hand_tested_workflow(models, monkeypatch):
    # The workflow tested on the RTX 5080: base 81 + one 49-frame extension. Compared with tiling off, so the
    # tiled decode (default since Oct 21, see test_tiled_decode) is the only difference from the tested graph.
    monkeypatch.setenv("MC_DECODE_TILED", "0")
    sg = graphs.lipsync_graph("x", models=models, audio_seconds=4.8, ext_frames=49)
    tested = json.loads((TESTED / "lipsync_infinitetalk.json").read_text())
    assert sg.meta["windows"] == 2
    assert _strip(sg.graph) == _strip(tested)


def test_portrait_t2i_matches_hand_tested_workflow(models):
    sg = graphs.avatar_graph("x", models=models, reference=False)
    tested = json.loads((TESTED / "lipsync_portrait_klein.json").read_text())
    assert _strip(sg.graph) == _strip(tested)


def test_tts_and_vc_match_hand_tested_workflows(models):
    tts = graphs.tts_graph("hello there", models=models, clone=True, exaggeration=0.55)
    assert _strip(tts.graph) == _strip(json.loads((TESTED / "voice_clone_tts.json").read_text()))
    vc = graphs.voice_changer_graph(models=models)
    assert _strip(vc.graph) == _strip(json.loads((TESTED / "voice_convert.json").read_text()))


def test_reference_edit_uses_same_nodes_as_tested_klein_edit(models):
    sg = graphs.avatar_graph("x", models=models, reference=2)
    tested = json.loads((TESTED / "flux2_klein_ref_edit.json").read_text())
    assert {n["class_type"] for n in sg.graph.values()} == {n["class_type"] for n in tested.values()}
    assert [k for k in sg.inputs] == ["photo", "photo_b"]


@pytest.mark.parametrize(
    "build",
    [
        lambda m: graphs.avatar_graph("p", models=m, reference=False),
        lambda m: graphs.avatar_graph("p", models=m, reference=True),
        lambda m: graphs.avatar_graph("p", models=m, reference=2),
        lambda m: graphs.tts_graph("hi", models=m, clone=True),
        lambda m: graphs.tts_graph("hi", models=m, clone=False),
        lambda m: graphs.voice_changer_graph(models=m),
        lambda m: graphs.lipsync_graph("p", models=m, audio_seconds=3.0),
        lambda m: graphs.lipsync_graph("p", models=m, audio_seconds=14.9),
    ],
)
def test_graphs_validate_against_reference_install(build, models, object_info):
    """Every node type, input name, link and model file exists on the reference PC."""
    assert validate_graph(build(models).graph, object_info) == []


def test_dialogue_graph_only_misses_the_multi_speaker_patch(models, object_info):
    sg = graphs.lipsync_graph("p", models=models, audio_seconds=8, two_speakers=True)
    problems = validate_graph(sg.graph, object_info)
    assert len(problems) == 1 and "wan2.1_infiniteTalk_multi_fp16" in problems[0]
    assert set(sg.inputs) == {"speech", "speech_b", "speech_mix", "avatar"}
    win = next(n for n in sg.graph.values() if n["class_type"] == "WanInfiniteTalkToVideo")
    assert win["inputs"]["mode"] == "two_speakers" and "mode.mask_2" in win["inputs"]  # V3 DynamicCombo sub-input


def test_smooth_graph_only_misses_the_rife_weights(models, object_info):
    problems = validate_graph(graphs.smooth_graph(models=models).graph, object_info)
    assert len(problems) == 1 and "no model files" in problems[0]


@pytest.mark.parametrize("secs,windows,frames", [(1.0, 1, 81), (3.24, 1, 81), (3.3, 2, 153), (14.0, 5, 369), (14.74, 5, 369), (15.0, 6, 441)])
def test_plan_windows(secs, windows, frames):
    p = graphs.plan_windows(secs)
    assert (p["windows"], p["total_frames"]) == (windows, frames)
    assert p["total_frames"] >= p["needed_frames"]


def test_validator_catches_problems(models, object_info):
    g = graphs.tts_graph("hi", models=models).graph
    g["2"]["inputs"]["model_pack_name"] = "nope"
    g["3"]["inputs"]["audio"] = ["99", 0]
    g["9"] = {"class_type": "NotARealNode", "inputs": {}}
    probs = validate_graph(g, object_info)
    assert any("nope" in p for p in probs)
    assert any("missing node 99" in p for p in probs)
    assert any("NotARealNode" in p for p in probs)
