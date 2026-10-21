"""Pipeline quality fixes (fix/pipeline-quality): Sage on SCAIL-2, timestamp conform + sync check, shot-adaptive
reference crops, reference/clip quality checks, identity QC verdicts, best-source-first ranking."""
import importlib

import numpy as np
import pytest

from main_character_comfy import clipfetch, framing, graphs, idqc, refcheck, sync
from main_character_comfy.config import Models


def _types(sg):
    return [n["class_type"] for n in sg.graph.values()]


def test_scail_graph_patches_sage_by_default_and_can_turn_it_off():
    sg = graphs.scail_group_graph("p", models=Models(), frames=81, width=1280, height=720, people=1, views=1)
    node = next(n for n in sg.graph.values() if n["class_type"] == graphs.SAGE_NODE)
    assert node["inputs"]["sage_attention"] == "sageattn_qk_int8_pv_fp16_triton"
    assert node["inputs"]["allow_compile"] is False
    # the sampler's model goes through the Sage patch, then the shift
    shift = next(n for n in sg.graph.values() if n["class_type"] == "ModelSamplingSD3")
    src = sg.graph[shift["inputs"]["model"][0]]
    assert src["class_type"] == graphs.SAGE_NODE
    off = graphs.scail_group_graph("p", models=Models(), frames=81, width=1280, height=720, people=1, views=1, sage="")
    assert graphs.SAGE_NODE not in _types(off)


def test_scail_sage_env_switch(monkeypatch):
    monkeypatch.setenv("MC_SCAIL_SAGE", "off")
    g = importlib.reload(graphs)
    try:
        assert g.SCAIL_SAGE == ""
        sg = g.scail_group_graph("p", models=Models(), frames=81, width=1280, height=720, people=1, views=1)
        assert g.SAGE_NODE not in _types(sg)
    finally:
        monkeypatch.delenv("MC_SCAIL_SAGE")
        importlib.reload(graphs)


def test_animate_graph_has_no_sage():
    # Animate stays on the GGUF Q4_K_M path with normal attention (Sage there is not adopted)
    import inspect
    assert "sage" not in inspect.signature(graphs.animate_graph).parameters


def test_sage_unavailable_detection():
    assert graphs.sage_unavailable(RuntimeError("Cannot execute because node PathchSageAttentionKJ does not exist"))
    assert graphs.sage_unavailable(RuntimeError("CUDA error: no kernel image is available for execution"))
    assert not graphs.sage_unavailable(RuntimeError("CUDA out of memory"))


def test_conform_is_timestamp_accurate_and_proportional_drifts():
    # a 2 s, 60-frame 30 fps shot; RIFE take at 32 fps has 63 frames (2k-1 from 32 generated)
    idx = sync.conform_indices(60, 30.0, 32.0, 63)
    assert idx[0] == 0 and idx[30] == 32 and idx[-1] == min(62, round(59 / 30 * 32))
    prop = sync.proportional_indices(60, 63)
    drift = max(abs(a - b) for a, b in zip(idx, prop))
    assert drift >= 1  # the old stretch is off by a frame or more somewhere
    # ping-pong padded shots render 3x and use the first third
    assert max(sync.conform_indices(18, 30.0, 32.0, 117, used_frac=1 / 3)) <= 38


def test_sync_check_finds_lag():
    rng = np.random.default_rng(0)
    src = rng.random(60)
    assert sync.sync_check(src, src)["best_lag"] == 0 and sync.sync_check(src, src)["ok"]
    late = np.concatenate([[0, 0], src[:-2]])
    r = sync.sync_check(src, late)
    assert r["best_lag"] == 2 and not r["ok"] and "late" in r["note"]


def test_mouth_energy_sees_mouth_motion_only():
    frames = [np.zeros((100, 100, 3), np.uint8) for _ in range(4)]
    frames[2][70:95, 30:70] = 255  # mouth band changes on frame 2
    frames[1][5:20, 5:20] = 255    # motion outside the face band is ignored
    e = sync.mouth_energy(frames, [(20, 10, 60, 80)] * 4)
    assert e[2] > 0 and e[1] == 0 and e[0] == 0


@pytest.mark.parametrize("frac,expect", [(0.8, "ecu"), (0.3, "cu"), (0.12, "medium"), (0.05, "wide"), (None, "wide")])
def test_shot_framing(frac, expect):
    assert framing.shot_framing(frac) == expect


def test_reference_crop_per_framing():
    W, H, face = 1000, 1500, (450, 200, 100, 120)
    ecu = framing.reference_crop(W, H, face, "ecu")
    cu = framing.reference_crop(W, H, face, "cu")
    med = framing.reference_crop(W, H, face, "medium")
    assert framing.reference_crop(W, H, face, "wide") == (0, 0, W, H)
    heights = [b[3] - b[1] for b in (ecu, cu, med)]
    assert heights == sorted(heights)                      # tighter for closer shots
    for b in (ecu, cu, med):
        assert 0 <= b[0] < b[2] <= W and 0 <= b[1] < b[3] <= H
        assert b[1] <= face[1] and b[3] >= face[1] + face[3]  # the whole face stays in
    feet = framing.reference_crop(W, H, face, "feet")
    assert feet[1] > face[1] + face[3] and feet[3] == H  # no face in a feet crop
    wide_aspect = framing.reference_crop(W, H, face, "cu", aspect=16 / 9)
    assert (wide_aspect[2] - wide_aspect[0]) >= (cu[2] - cu[0])
    with pytest.raises(ValueError):
        framing.reference_crop(W, H, face, "macro")


def test_refcheck_grayscale_tiny_face_and_sunglasses(tmp_path, monkeypatch):
    from PIL import Image

    rng = np.random.default_rng(1)
    g = (rng.random((1200, 900)) * 255).astype(np.uint8)
    bw = np.stack([g, g, g], axis=2)
    p = tmp_path / "tony_bw.png"
    Image.fromarray(bw).save(p)
    monkeypatch.setattr(refcheck, "find_face", lambda rgb: {"x": 400, "y": 100, "w": 60, "h": 70, "score": 0.9})
    r = refcheck.check_reference(p, "photo")
    text = " ".join(r["warnings"])
    assert r["grayscale"] and "tony_bw.png is black and white" in text
    assert r["closeup_ready"] is False and "only 70 px" in text

    col = (rng.random((1200, 900, 3)) * 255).astype(np.uint8)
    col[100 + 18:100 + 35, 400:460] = 10  # dark, flat eye band = sunglasses
    p2 = tmp_path / "shades.png"
    Image.fromarray(col).save(p2)
    monkeypatch.setattr(refcheck, "find_face", lambda rgb: {"x": 400, "y": 100, "w": 60, "h": 70, "score": 0.9})
    r2 = refcheck.check_reference(p2, "photo")
    assert r2.get("eyes_covered") and not r2.get("grayscale")


def test_refcheck_posterized_photo(tmp_path, monkeypatch):
    from PIL import Image

    flat = np.zeros((1024, 1024, 3), np.uint8)
    flat[:, :512] = (200, 40, 40)
    flat[:, 512:] = (30, 30, 200)
    p = tmp_path / "filtered.png"
    Image.fromarray(flat).save(p)
    monkeypatch.setattr(refcheck, "find_face", lambda rgb: None)
    r = refcheck.check_reference(p, "photo")
    assert r.get("filtered") and any("filtered" in w for w in r["warnings"])
    assert not refcheck.check_reference(p, "avatar").get("filtered")  # drawn avatars are flat on purpose


def test_check_clip():
    low = refcheck.check_clip(854, 480, 30, 1100, name="clip.mp4")
    assert not low["ok"] and any("854x480" in w for w in low["warnings"])
    assert any("compressed" in w for w in low["warnings"])
    good = refcheck.check_clip(1920, 1080, 30, 9000)
    assert good["ok"] and good["warnings"] == []
    assert any("fps" in w for w in refcheck.check_clip(1920, 1080, 15, 9000)["warnings"])


def test_identity_verdict_and_escalation():
    assert idqc.verdict([0.5, 0.45, 0.6])["verdict"] == "pass"
    assert idqc.verdict([0.3, 0.28, 0.31])["verdict"] == "weak"
    d = idqc.verdict([0.1, 0.12, 0.2])
    assert d["verdict"] == "drift" and d["escalate"]
    assert idqc.verdict([])["verdict"] == "no face"
    assert idqc.escalation("cu", "scail", None) == {"reference": "head crop", "engine": "chain", "face": "facepose"}
    assert idqc.escalation("cu", "chain", "facepose") is None
    a = np.array([1.0, 0, 0]); b = np.array([1.0, 1.0, 0])
    assert idqc.identity_scores(a, [a, b, None])[0] == pytest.approx(1.0)
    assert idqc.identity_scores(a, [a, b, None])[2] is None


def test_rank_sources_best_first():
    cands = [{"url": "repost", "best_height": 480, "fps": 30, "tbr": 1100},
             {"url": "ig", "best_height": 1078, "fps": 30, "tbr": 3150},
             {"url": "hls", "best_height": 1078, "fps": 30, "tbr": 2000}]
    r = clipfetch.rank_sources(cands)
    assert [c["url"] for c in r] == ["ig", "hls", "repost"]
    assert r[0]["rank"] == 1 and not r[0]["low"] and r[-1]["low"]
    assert "br" in clipfetch.FORMAT_SORT
