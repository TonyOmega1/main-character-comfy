"""Every VIDEO graph decodes with core VAEDecodeTiled (a plain VAEDecode of a long video latent crashed a 16 GB
card's display driver). Also: doctor treats a not-yet-started proxy as a note, and agent_e2e writes outside the repo."""
import json
from pathlib import Path

import pytest

from main_character_comfy import doctor, export_workflows, graphs
from main_character_comfy.config import Models

FIX = Path(__file__).parent / "fixtures" / "object_info_reference_live.json"
IMAGE_WORKFLOWS = ("01_", "01b_", "01c_", "06_", "08_", "09_")  # FLUX.2 stills: plain VAEDecode is fine


def _limits():
    req = json.loads(FIX.read_text(encoding="utf-8"))["VAEDecodeTiled"]["input"]["required"]
    return req


def _check_tiled(node):
    req = _limits()
    ins = node["inputs"]
    assert set(ins) == set(req), "VAEDecodeTiled inputs must match the live ComfyUI schema"
    for k in ("tile_size", "overlap", "temporal_size", "temporal_overlap"):
        lo, step = req[k][1]["min"], req[k][1]["step"]
        assert ins[k] >= lo and ins[k] % step == 0, (k, ins[k])


@pytest.fixture
def exported(tmp_path, monkeypatch):
    for k in ("MC_DECODE_TILED", "MC_DECODE_TILE", "MC_DECODE_OVERLAP", "MC_DECODE_TEMPORAL",
              "MC_DECODE_TEMPORAL_OVERLAP"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(export_workflows, "WORKFLOWS_DIR", tmp_path)
    export_workflows.export()
    return {p.name: json.loads(p.read_text(encoding="utf-8")) for p in tmp_path.glob("*.json")}


def test_default_decode_settings():
    assert graphs.DECODE_DEFAULTS == {"tile_size": 256, "overlap": 32, "temporal_size": 32, "temporal_overlap": 4}


def test_every_video_workflow_uses_tiled_decode(exported):
    video = {n: wf for n, wf in exported.items() if not n.startswith(IMAGE_WORKFLOWS)}
    assert any(n.startswith("04_") for n in video) and any(n.startswith("07_") for n in video)
    for name, wf in video.items():
        types = [node["class_type"] for node in wf.values() if isinstance(node, dict) and "class_type" in node]
        assert "VAEDecode" not in types, f"{name} still uses plain VAEDecode"
        tiled = [node for node in wf.values() if isinstance(node, dict) and node.get("class_type") == "VAEDecodeTiled"]
        if any(t in types for t in ("WanInfiniteTalkToVideo", "WanImageToVideo", "Wan22ImageToVideoLatent",
                                     "WanAnimateToVideo", "VOIDInpaintConditioning")):
            assert tiled, name
        for node in tiled:
            _check_tiled(node)


def test_lipsync_graph_is_tiled_with_defaults(monkeypatch):
    monkeypatch.delenv("MC_DECODE_TILED", raising=False)
    sg = graphs.lipsync_graph("talk", models=Models(), audio_seconds=4.8, ext_frames=49)
    dec = [n for n in sg.graph.values() if n["class_type"] in ("VAEDecode", "VAEDecodeTiled")]
    assert dec and all(n["class_type"] == "VAEDecodeTiled" for n in dec)
    assert {k: dec[0]["inputs"][k] for k in graphs.DECODE_DEFAULTS} == graphs.DECODE_DEFAULTS


def test_scail_group_graph_is_tiled():
    sg = graphs.scail_group_graph("p", models=Models(), frames=81, width=720, height=1280, people=2, views=2,
                                  object_indices="0,1")
    types = [n["class_type"] for n in sg.graph.values()]
    assert "VAEDecode" not in types and "VAEDecodeTiled" in types


def test_decode_settings_env_override_and_snap(monkeypatch):
    monkeypatch.setenv("MC_DECODE_TILE", "300")      # snaps down to a multiple of 32
    monkeypatch.setenv("MC_DECODE_OVERLAP", "200")   # clamped to a quarter tile
    monkeypatch.setenv("MC_DECODE_TEMPORAL", "6")    # below the node's minimum (8)
    monkeypatch.setenv("MC_DECODE_TEMPORAL_OVERLAP", "nope")
    st = graphs.decode_settings()
    assert st == {"tile_size": 288, "overlap": 64, "temporal_size": 8, "temporal_overlap": 4}
    monkeypatch.setenv("MC_DECODE_TILED", "0")
    assert graphs.decode_settings() is None
    sg = graphs.lipsync_graph("talk", models=Models(), audio_seconds=4.8, ext_frames=49)
    assert "VAEDecodeTiled" not in [n["class_type"] for n in sg.graph.values()]


def test_doctor_missing_proxy_is_a_note_on_fresh_install(capsys, monkeypatch):
    monkeypatch.setenv("COMFY_BASE_URL", "http://127.0.0.1:9")  # nothing listens there
    rc = doctor.main(["--object-info", str(FIX), "--tier", "standard"])
    out = capsys.readouterr().out
    assert "isn't running yet" in out and "!!  proxy" not in out
    assert rc == (0 if "  !!  " not in out else 1)
    assert doctor.main(["--tier", "standard"]) == 1  # nothing else checked: proxy down is a real failure


def test_agent_e2e_writes_outside_the_repo():
    src = (Path(__file__).parents[1] / "scripts" / "agent_e2e.py").read_text(encoding="utf-8")
    assert 'ROOT / "agent_e2e' not in src and 'ROOT / "library"' not in src and 'ROOT / "outputs"' not in src
    assert 'Path.home() / "mc_agent_e2e"' in src and '"--redub"' in src and "ensure_proxy(" in src
