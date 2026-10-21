"""Render-time fixes that keep the look (Oct 19): SeedVR2 only after framing QC and batched after the swaps,
ECU / feet / no-face close shots routed straight to Animate, Animate prep reused on retakes, plus the two
off-by-default A/B flags (MC_ANIMATE_WINDOW=shot, MC_ANIMATE_SAGE=on)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from main_character_comfy import cli, framing, graphs, media, mock_media
from main_character_comfy.backends import MockBackend, StageResult
from main_character_comfy.config import Models, Settings
from main_character_comfy.media import run_ffmpeg
from main_character_comfy.validate import validate_graph
from main_character_comfy.pipeline import (
    PENDING_FILE,
    PREP_FILE,
    Pipeline,
    Request,
    wants_seedvr2,
)


# --------------------------------------------------------------------------- helpers
def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class ScailMock(MockBackend):
    """MockBackend that also renders SCAIL-2 (a resize of the clip to the SCAIL size) and logs every job:
    (stage, graph JSON, {input: sha256})."""

    def __init__(self):
        super().__init__()
        self.log: list[tuple[str, str, dict[str, str]]] = []

    def run_stage(self, stage, sg, files, out_dir, progress):
        self.log.append((stage, json.dumps(sg.graph, sort_keys=True),
                         {k: _sha(v) for k, v in files.items() if Path(v).is_file()}))
        if any(n["class_type"] == "WanSCAILToVideo" for n in sg.graph.values()):
            out_dir.mkdir(parents=True, exist_ok=True)
            w, h = sg.meta.get("size", [576, 1024])
            dest = mock_media.fake_upscale(files["clip"], int(w), int(h), int(sg.meta.get("fps", 16)),
                                           out_dir / f"{stage}.mp4")
            progress(1.0, "done (mock)")
            return StageResult(dest, None, 0.0, [dest])
        return super().run_stage(stage, sg, files, out_dir, progress)

    def stages(self) -> list[str]:
        return [s for s, _, _ in self.log]


def _clip(path, seconds=3, size="720x1280"):
    run_ffmpeg(["-f", "lavfi", "-i", f"testsrc2=size={size}:rate=30:duration={seconds}", "-f", "lavfi", "-i",
                f"sine=frequency=440:duration={seconds}", "-c:a", "aac", "-shortest", "-c:v", "libx264",
                "-pix_fmt", "yuv420p", str(path)])
    return path


@pytest.fixture
def mock(tmp_path):
    return ScailMock()


@pytest.fixture
def pipe(mock, tmp_path):
    return Pipeline(mock, Settings(output_dir=tmp_path / "out"))


@pytest.fixture
def duo(tmp_path, sample_photo):
    clip = _clip(tmp_path / "duo.mp4")
    return dict(mode="swap", clip=clip, consent=True, clip_seconds=3, people=[sample_photo, sample_photo],
                people_desc=["red jacket", "blue hoodie"], style="none", tier="standard")


@pytest.fixture
def wide(monkeypatch):
    """Everyone classifies as a wide shot (no real face detector on the synthetic clip)."""
    monkeypatch.setattr(framing, "classify_person",
                        lambda rgb, box, **kw: {"framing": "wide", "face_frac": None, "face": None})


def _run(pipe, **kw):
    evs = list(pipe.run(Request(**kw)))
    done = evs[-1]
    proj_path = Path(done.data["work_dir"]) / "_work" / "project.json"
    proj = json.loads(proj_path.read_text()) if proj_path.exists() else None
    return evs, done, proj


# --------------------------------------------------------------------------- default graphs unchanged
B007AC3 = {  # sha256 of the default Animate graphs (graph + meta) at b007ac3, before this change
    "solo_720": "6b65820e719239a60bdefdde626e4d96ea58ca91d4b55112094a419689e93fde",
    "carry": "6bfd5e75baf206ddb94268232aa5ea5ff0816a4e5473b52f70bf4d940428fdd0",
    "person": "73d990bd70a0d5b080d8b67838492b7f31667a6d9523daee950422bad0d357d3",
    "short": "0d43735d46a8399a302c81a6cd79e1f7379e16b886eb3048038bef77a0d40a13",
    "vibe": "30ece976485e5d4728776c9e85f015247d97034a00253fc8524cf26b0c930e38",
}
CASES = {
    "solo_720": dict(frames=81, width=720, height=1280),
    "carry": dict(frames=150, carry=True, width=720, height=1280, seed=7),
    "person": dict(frames=40, width=720, height=1280, person_mask=True, isolate_ref=True, face_drive="ref"),
    "short": dict(frames=33, width=480, height=832, seed=3),
}


def _graph_sha(sg) -> str:
    return hashlib.sha256(json.dumps({"g": sg.graph, "m": sg.meta}, sort_keys=True, default=str).encode()).hexdigest()


@pytest.mark.parametrize("unset", [True, False])
def test_default_animate_graphs_are_byte_identical_to_b007ac3(monkeypatch, unset):
    # Oct 21: video decodes are tiled by default (VAEDecodeTiled). With tiling off the graphs must still be
    # byte identical to b007ac3, so the decode node is the ONLY change; test_tiled_decode covers the tiled default.
    monkeypatch.setenv("MC_DECODE_TILED", "0")
    if unset:
        monkeypatch.delenv("MC_ANIMATE_WINDOW", raising=False)
        monkeypatch.delenv("MC_ANIMATE_SAGE", raising=False)
    else:  # explicit "off" values are the default too
        monkeypatch.setenv("MC_ANIMATE_WINDOW", "fixed")
        monkeypatch.setenv("MC_ANIMATE_SAGE", "off")
    for k, kw in CASES.items():
        assert _graph_sha(graphs.swap_graph("p", models=Models(), **kw)) == B007AC3[k], k
    assert _graph_sha(graphs.vibe_graph("p", models=Models(), frames=81)) == B007AC3["vibe"]


def _w2v(sg):
    return [n["inputs"] for n in sg.graph.values() if n["class_type"] == "WanAnimateToVideo"]


def test_shot_window_flag_is_opt_in_and_single_window_only(monkeypatch):
    monkeypatch.setenv("MC_ANIMATE_WINDOW", "shot")
    sg = graphs.swap_graph("p", models=Models(), frames=30, width=720, height=1280)
    assert [w["length"] for w in _w2v(sg)] == [33] and sg.meta["out_frames"] == 33 and sg.meta["window"] == 33
    pm = graphs.swap_graph("p", models=Models(), frames=30, width=720, height=1280, person_mask=True)
    assert [w["length"] for w in _w2v(pm)] == [33] and pm.meta["out_frames"] == 30
    assert graphs.plan_animate_windows(30) == 1
    # multi-window and carried chunks keep the trained 77
    assert {w["length"] for w in _w2v(graphs.swap_graph("p", models=Models(), frames=150))} == {77}
    assert {w["length"] for w in _w2v(graphs.swap_graph("p", models=Models(), frames=40, carry=True))} == {77}
    assert [w["length"] for w in _w2v(graphs.swap_graph("p", models=Models(), frames=77))] == [77]
    monkeypatch.delenv("MC_ANIMATE_WINDOW")
    assert [w["length"] for w in _w2v(graphs.swap_graph("p", models=Models(), frames=30))] == [77]


def test_animate_sage_flag_is_opt_in(monkeypatch):
    monkeypatch.delenv("MC_ANIMATE_SAGE", raising=False)
    base = graphs.swap_graph("p", models=Models(), frames=81)
    assert graphs.SAGE_NODE not in {n["class_type"] for n in base.graph.values()}
    monkeypatch.setenv("MC_ANIMATE_SAGE", "on")
    sg = graphs.swap_graph("p", models=Models(), frames=81)
    sage = [(k, n) for k, n in sg.graph.items() if n["class_type"] == graphs.SAGE_NODE]
    assert len(sage) == 1 and sage[0][1]["inputs"]["sage_attention"] == "sageattn_qk_int8_pv_fp16_triton"
    msd3 = next(n for n in sg.graph.values() if n["class_type"] == "ModelSamplingSD3")
    assert msd3["inputs"]["model"][0] == sage[0][0]  # patched right before the sampler shift
    assert sg.meta["sage"] == "sageattn_qk_int8_pv_fp16_triton"
    ks = lambda g: [(n["inputs"]["seed"], n["inputs"]["steps"]) for n in g.graph.values()
                    if n["class_type"] == "KSampler"]
    assert ks(sg) == ks(base)  # same seeds and steps
    monkeypatch.setenv("MC_ANIMATE_SAGE", "sageattn_qk_int8_pv_fp8_cuda")
    assert graphs.animate_sage() == "sageattn_qk_int8_pv_fp8_cuda"


# --------------------------------------------------------------------------- framing classifier + routing
def _frame(h=1280, w=720):
    return np.zeros((h, w, 3), np.uint8)


def test_classify_person_uses_the_face_inside_the_persons_box():
    seen = {}

    def face_at(frac):
        def f(crop):
            seen["shape"] = crop.shape
            return {"x": 10.0, "y": 5.0, "w": 100.0, "h": frac * 1280, "score": 0.9}
        return f

    box = (100, 200, 500, 1200)
    r = framing.classify_person(_frame(), box, find_face=face_at(0.5))
    assert r["framing"] == "ecu" and r["face_frac"] == 0.5
    assert r["face"]["x"] == 70.0 and r["face"]["y"] == 105.0  # crop offset added back (10% pad)
    assert seen["shape"][:2] == (1280 - 100, 540 - 60)
    assert framing.classify_person(_frame(), box, find_face=face_at(0.3))["framing"] == "cu"
    assert framing.classify_person(_frame(), box, find_face=face_at(0.12))["framing"] == "medium"
    nf = lambda crop: None
    assert framing.classify_person(_frame(), (100, 0, 500, 1100), find_face=nf)["framing"] == "close_noface"
    assert framing.classify_person(_frame(), (100, 0, 500, 500), find_face=nf)["framing"] == "wide"  # short
    assert framing.classify_person(_frame(), (100, 200, 500, 1280), find_face=nf)["framing"] == "wide"  # no top cut


def test_route_set_is_ecu_feet_noface_but_not_closeups():
    assert framing.ROUTE_TO_ANIMATE == {"ecu", "feet", "close_noface"}
    assert framing.route_to_animate(["cu", "wide", "medium"]) == (False, "")
    go, why = framing.route_to_animate(["cu", "ecu"])
    assert go and "person 2" in why and "close-up" in why
    assert framing.route_to_animate(["feet"])[0] and framing.route_to_animate(["close_noface"])[0]
    assert framing.reference_crop(400, 600, (100, 50, 80, 100), "close_noface") == (0, 0, 400, 600)


def test_framing_qc():
    src = [(100, 200, 300, 1200), (400, 200, 600, 1200)]
    scale = lambda b, k: tuple(int(v * k) for v in b)
    ok = framing.framing_qc(src, (720, 1280), [scale(b, 0.8) for b in src], (576, 1024))  # same box at SCAIL size
    assert ok["pass"] and all(abs(p["height_ratio"] - 1) < 0.01 for p in ok["people"])
    big = framing.framing_qc(src, (720, 1280), [src[0], (350, 0, 650, 1280)], (720, 1280))
    assert not big["pass"] and "person 2" in big["why"]
    moved = framing.framing_qc(src, (720, 1280), [(400, 200, 600, 1200), src[1]], (720, 1280))
    assert not moved["pass"] and "moved" in moved["why"]
    gone = framing.framing_qc(src, (720, 1280), [src[0], None], (720, 1280))
    assert not gone["pass"] and "missing" in gone["why"]


def test_ecu_shot_skips_scail_and_goes_to_animate(pipe, mock, duo, monkeypatch):
    calls = []

    def fake(rgb, box, **kw):
        calls.append(box)
        return {"framing": "ecu" if len(calls) == 1 else "cu", "face_frac": 0.5, "face": None}

    monkeypatch.setattr(framing, "classify_person", fake)
    _, done, proj = _run(pipe, **duo)
    assert "swap_group" not in mock.stages() and {"swap_p1", "swap_p2"} <= set(mock.stages())
    assert proj["group_engine"]["engine"] == "chain" and "extreme close-up" in proj["group_engine"]["why"]
    assert proj["routing"]["animate"] and [p["framing"] for p in proj["routing"]["people"]] == ["ecu", "cu"]


def test_normal_closeups_still_try_scail(pipe, mock, duo, monkeypatch):
    monkeypatch.setattr(framing, "classify_person",
                        lambda rgb, box, **kw: {"framing": "cu", "face_frac": 0.3, "face": None})
    _, _, proj = _run(pipe, **duo)
    assert "swap_group" in mock.stages() and proj["group_engine"]["engine"] == "scail"
    assert proj["routing"]["animate"] is False


def test_routing_can_be_turned_off_and_respects_a_forced_engine(pipe, mock, duo, monkeypatch):
    monkeypatch.setattr(framing, "classify_person",
                        lambda rgb, box, **kw: {"framing": "ecu", "face_frac": 0.5, "face": None})
    _run(pipe, **duo, route_framing=False)
    assert mock.stages().count("swap_group") == 1
    _run(pipe, **dict(duo, group_engine="scail"))
    assert mock.stages().count("swap_group") == 2


def test_routing_never_breaks_a_run(pipe, mock, duo, monkeypatch):
    def boom(rgb, box, **kw):
        raise ImportError("no cv2")

    monkeypatch.setattr(framing, "classify_person", boom)
    _, _, proj = _run(pipe, **duo)
    assert "swap_group" in mock.stages() and "no cv2" in proj["routing"]["error"]


# --------------------------------------------------------------------------- SeedVR2 after QC
def test_seedvr2_runs_only_after_the_take_passes_framing_qc(pipe, mock, duo, wide):
    _, done, proj = _run(pipe, **duo)
    st = mock.stages()
    assert st.index("swap_group") < st.index("qc_people") < st.index("upscale")
    assert proj["qc"]["pass"] and proj["upscale"]["engine"] == "seedvr2"
    assert done.artifact.exists()


def test_failed_qc_take_is_never_upscaled_and_falls_back_to_animate(pipe, mock, duo, wide, monkeypatch):
    monkeypatch.setattr(framing, "framing_qc", lambda *a, **k: {"pass": False, "why": "person 1 drawn at 1.60x",
                                                                "people": []})
    evs, done, proj = _run(pipe, **duo)
    st = mock.stages()
    assert "upscale" not in st and st.index("qc_people") < st.index("swap_p1") < st.index("swap_p2")
    assert proj["group_engine"]["engine"] == "chain" and "framing QC" in proj["group_engine"]["why"]
    assert not proj["qc"]["pass"]
    assert any("skipping SeedVR2" in e.message for e in evs)


def test_qc_off_keeps_the_old_path(pipe, mock, duo, wide):
    _, _, proj = _run(pipe, **duo, qc="off")
    st = mock.stages()
    assert "qc_people" not in st and st.index("swap_group") < st.index("upscale") and "qc" not in proj


def test_wants_seedvr2_matches_the_inline_rule():
    assert wants_seedvr2(Request(), group=True) and not wants_seedvr2(Request(), group=False)
    assert wants_seedvr2(Request(upscale="seedvr2")) and not wants_seedvr2(Request(upscale="lanczos"), True)
    assert not wants_seedvr2(Request(upscale="seedvr2", tier="lite"), True)


def test_deferred_seedvr2_is_batched_and_identical_to_inline(tmp_path, duo, wide):
    inline = ScailMock()
    _run(Pipeline(inline, Settings(output_dir=tmp_path / "a")), **duo)
    deferred = ScailMock()
    p = Pipeline(deferred, Settings(output_dir=tmp_path / "b"))
    tops = []
    for seed in (2026, 7):
        evs = list(p.run(Request(**duo, seed=seed, defer_upscale=True)))
        done = evs[-1]
        assert done.status == "complete" and done.data["deferred"] and done.artifact.name == "swap_group_take.mp4"
        assert Path(done.data["pending"]).name == PENDING_FILE and Path(done.data["pending"]).exists()
        assert not (Path(done.data["work_dir"]) / "_work" / "project.json").exists()  # not exported yet
        tops.append(Path(done.data["work_dir"]))
    assert "upscale" not in deferred.stages()
    n_before = len(deferred.log)
    fin = list(p.finish_pending(tops))
    after = deferred.log[n_before:]
    assert [s for s, _, _ in after] == ["upscale", "upscale"]  # back to back, before any export
    first_export = next(i for i, e in enumerate(fin) if e.stage == "finish" and e.status == "start"
                        and "export" in e.message.lower())
    assert all(e.stage != "complete" for e in fin[:first_export])
    completes = [e for e in fin if e.status == "complete"]
    assert len(completes) == 2 and all(e.artifact.exists() and e.artifact.suffix == ".mp4" for e in completes)
    for t in tops:
        assert not (t / "_work" / PENDING_FILE).exists() and (t / "_work" / "pending_upscale.done.json").exists()
        proj = json.loads((t / "_work" / "project.json").read_text())
        assert proj["upscale"]["engine"] == "seedvr2" and proj["qc"]["pass"]
    # same seed: the deferred SeedVR2 job is the inline one (same graph, same input bytes)
    up_inline = next(x for x in inline.log if x[0] == "upscale")
    assert after[0][1] == up_inline[1] and after[0][2] == up_inline[2]
    seeds = {n["inputs"].get("seed") for n in json.loads(after[1][1]).values() if "seed" in n["inputs"]}
    assert seeds == {7}


def test_finish_pending_needs_a_deferred_take(pipe, tmp_path):
    from main_character_comfy.pipeline import InputError

    with pytest.raises(InputError, match="No deferred take"):
        list(pipe.finish_pending([tmp_path]))


def test_defer_is_a_no_op_without_seedvr2(pipe, mock, duo):
    _, done, proj = _run(pipe, **dict(duo, group_engine="chain"), defer_upscale=True)
    assert not done.data.get("deferred") and proj is not None and "upscale" not in mock.stages()


# --------------------------------------------------------------------------- prep reuse on retakes
def _strip_seeds(graph_json: str) -> dict:
    g = json.loads(graph_json)
    for n in g.values():
        if n["class_type"] == "KSampler":
            n["inputs"]["seed"] = None
    return g


def test_retake_reuses_prep_so_only_the_seed_changes(tmp_path, duo, monkeypatch):
    req = dict(duo, group_engine="chain")
    a = ScailMock()
    _, done1, _ = _run(Pipeline(a, Settings(output_dir=tmp_path / "a")), **req, seed=1)
    top1 = Path(done1.data["work_dir"])
    assert (top1 / "_work" / PREP_FILE).exists()

    def no_prepare(*a_, **k):
        raise AssertionError("retake must not re-prepare the clip")

    monkeypatch.setattr(media, "prepare_clip", no_prepare)
    b = ScailMock()
    evs, done2, proj2 = _run(Pipeline(b, Settings(output_dir=tmp_path / "b")), **req, seed=2, reuse_prep=top1)
    assert "people" not in b.stages() and proj2["reused_prep"] == str(top1)
    assert any("Retake: reusing" in e.message for e in evs)
    one = dict((s, (g, f)) for s, g, f in a.log)
    two = dict((s, (g, f)) for s, g, f in b.log)
    g1, f1 = one["swap_p1"]
    g2, f2 = two["swap_p1"]
    assert f1 == f2  # clip, person mask and reference: same bytes -> same ComfyUI cache keys
    assert g1 != g2 and _strip_seeds(g1) == _strip_seeds(g2)  # the KSampler seed is the only difference
    ks = [n["inputs"]["seed"] for n in json.loads(g2).values() if n["class_type"] == "KSampler"]
    assert ks == [2]
    assert done2.artifact.exists()


def test_retake_with_changed_prep_settings_prepares_from_scratch(tmp_path, duo):
    req = dict(duo, group_engine="chain")
    a = ScailMock()
    _, done1, _ = _run(Pipeline(a, Settings(output_dir=tmp_path / "a")), **req)
    b = ScailMock()
    evs, _, proj = _run(Pipeline(b, Settings(output_dir=tmp_path / "b")), **dict(req, clip_seconds=2),
                        reuse_prep=Path(done1.data["work_dir"]))
    assert "people" in b.stages() and "reused_prep" not in proj
    assert any("doesn't match" in e.message and "clip_seconds" in e.message for e in evs)


def test_solo_retake_reuses_the_clip(tmp_path, sample_photo, monkeypatch):
    clip = _clip(tmp_path / "solo.mp4", 2)
    req = dict(mode="swap", clip=clip, consent=True, clip_seconds=2, photo=sample_photo, style="none")
    a = ScailMock()
    _, done1, _ = _run(Pipeline(a, Settings(output_dir=tmp_path / "a")), **req, seed=1)
    monkeypatch.setattr(media, "prepare_clip", lambda *x, **k: (_ for _ in ()).throw(AssertionError("re-prepared")))
    b = ScailMock()
    _, _, proj = _run(Pipeline(b, Settings(output_dir=tmp_path / "b")), **req, seed=2,
                      reuse_prep=Path(done1.data["work_dir"]))
    (g1, f1), (g2, f2) = dict((s, (g, f)) for s, g, f in a.log)["swap"], dict((s, (g, f)) for s, g, f in b.log)["swap"]
    assert f1 == f2 and _strip_seeds(g1) == _strip_seeds(g2) and g1 != g2


# --------------------------------------------------------------------------- CLI
def test_cli_flags(tmp_path):
    a = cli.build_parser().parse_args(["--mode", "swap", "--defer-upscale", "--retake", str(tmp_path), "--no-route",
                                       "--qc", "off"])
    assert a.defer_upscale and a.retake == tmp_path and a.no_route and a.qc == "off"
    b = cli.build_parser().parse_args(["--finish-pending", "x", "y"])
    assert b.finish_pending == [Path("x"), Path("y")]
    d = cli.build_parser().parse_args([])
    assert not d.defer_upscale and d.retake is None and not d.no_route and d.qc == "auto"
    assert Request().route_framing and Request().qc == "auto" and not Request().defer_upscale


# --------------------------------------------------------------------------- long clips: chained SCAIL-2 chunks
def test_scail_chunk_plan_covers_the_clip_exactly():
    for n in (81, 82, 112, 157, 240, 401):
        plan = graphs.plan_scail_chunks(n)
        assert sum(x[2] for x in plan) == n
        for (s0, l0, n0), (s1, l1, n1) in zip(plan, plan[1:]):
            assert s1 == s0 + (n0 if s0 == 0 else n0 + graphs.SCAIL_ANCHOR) - graphs.SCAIL_ANCHOR
            assert l1 <= graphs.SCAIL_MAX_FRAMES


def test_scail_carry_chunk_graph_validates(models):
    live = json.loads((Path(__file__).parent / "fixtures" / "object_info_reference_live.json").read_text())
    sg = graphs.scail_group_graph("p", models=models, frames=81, width=720, height=1280, people=2, views=2,
                                  carry=True, sage="")  # the KJ Sage node isn't in the Oct 18 snapshot
    w2v = next(n for n in sg.graph.values() if n["class_type"] == "WanSCAILToVideo")
    assert "previous_frames" in w2v["inputs"] and w2v["inputs"]["video_frame_offset"] == 0
    assert "prev_chunk" in sg.inputs and sg.meta["new_frames"] == 76
    assert any(n["class_type"] == "ColorTransfer" for n in sg.graph.values())
    assert validate_graph(sg.graph, live, check_files=False) == []
    first = graphs.scail_group_graph("p", models=models, frames=81, width=720, height=1280, people=2, views=2)
    assert "prev_chunk" not in first.inputs and not any(n["class_type"] == "ColorTransfer" for n in first.graph.values())


def test_long_group_clip_uses_scail_chunks_only_with_the_flag(pipe, mock, duo, wide, tmp_path):
    long_clip = _clip(tmp_path / "long.mp4", seconds=7)
    kw = dict(duo, clip=long_clip, clip_seconds=7)
    _, _, proj = _run(pipe, **kw)
    assert proj["group_engine"]["engine"] == "chain"  # default unchanged
    mock.log.clear()
    _, done, proj = _run(pipe, **dict(kw, scail_chunks=True))
    stages = mock.stages()
    assert [s for s in stages if s.startswith("swap_group")] == ["swap_group_c01", "swap_group_c02"]
    assert proj["group_engine"]["engine"] == "scail" and len(proj["swap_settings"]["chunks"]) == 2
    c2 = next(g for s, g, _ in mock.log if s == "swap_group_c02")
    assert "previous_frames" in c2 and "ColorTransfer" in c2
    assert abs(media.media_duration(done.artifact) - 7.0) < 0.2
