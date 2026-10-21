"""Hardware tiers: detection, overrides, per-tier graphs/ETAs, and Lite gating."""

from __future__ import annotations

import json

import pytest

from main_character_comfy import cli, graphs, memes, tiers
from main_character_comfy.backends import MockBackend
from main_character_comfy.config import Models, Settings
from main_character_comfy.media import media_duration, run_ffmpeg, video_size
from main_character_comfy.pipeline import (InputError, Pipeline, Request, check_request, estimate_animate_seconds,
                                           estimate_lipsync_seconds, estimate_motion_seconds, request_from_recipe)
from main_character_comfy.validate import validate_graph


@pytest.fixture
def pipe(tmp_path):
    return Pipeline(MockBackend(), Settings(output_dir=tmp_path / "out"))


def _clip(path, seconds):
    run_ffmpeg(["-f", "lavfi", "-i", f"testsrc2=size=720x1280:rate=30:duration={seconds}", "-c:v", "libx264",
                "-pix_fmt", "yuv420p", str(path)])
    return path


# --------------------------------------------------------------------------- selection
@pytest.mark.parametrize("gb,key", [(6.0, "lite"), (8.0, "lite"), (11.99, "lite"), (15.9, "standard"),
                                    (16.0, "standard"), (20.0, "standard"), (23.99, "max"), (32.0, "max")])
def test_tier_for_vram(gb, key):
    assert tiers.tier_for_vram(gb) == key


def test_auto_detect_from_vram(monkeypatch):
    monkeypatch.setenv("MC_VRAM_GB", "12")
    c = tiers.resolve()
    assert (c.key, c.source, c.vram_gb) == ("lite", "auto", 12.0)
    assert "Lite" in c.describe() and "auto: 12 GB" in c.describe()


def test_unknown_vram_defaults_to_standard():
    c = tiers.resolve()
    assert (c.key, c.source) == ("standard", "default") and "assuming 16 GB" in c.describe()


def test_nvidia_smi_probe_is_used(monkeypatch):
    from main_character_comfy import system

    monkeypatch.setattr(system, "_smi_vram_gb", lambda: 23.99)
    assert tiers.resolve().key == "max"


def test_override_order_cli_beats_env_beats_auto(monkeypatch):
    monkeypatch.setenv("MC_VRAM_GB", "24")
    monkeypatch.setenv("MC_TIER", "lite")
    assert tiers.resolve().key == "lite" and tiers.resolve().source == "MC_TIER"
    assert tiers.resolve("standard").key == "standard" and tiers.resolve("standard").source == "cli"
    monkeypatch.setenv("MC_TIER", "auto")
    assert tiers.resolve().key == "max"


def test_bad_env_tier_warns_and_falls_back(monkeypatch):
    monkeypatch.setenv("MC_TIER", "potato")
    monkeypatch.setenv("MC_VRAM_GB", "16")
    c = tiers.resolve()
    assert c.key == "standard" and "MC_TIER" in c.warning


def test_below_minimum_warns(monkeypatch):
    monkeypatch.setenv("MC_VRAM_GB", "6")
    c = tiers.resolve()
    assert c.key == "lite" and "8 GB minimum" in c.warning


def test_normalize_accepts_legacy_profile_names():
    assert [tiers.normalize(x) for x in ("12gb", "16GB", "24gb", "Lite", " max ")] == ["lite", "standard", "max",
                                                                                    "lite", "max"]
    with pytest.raises(ValueError):
        tiers.normalize("8gb")


def test_cli_show_tier(monkeypatch, capsys):
    monkeypatch.setenv("MC_VRAM_GB", "12")
    assert cli.main(["--show-tier"]) == 0
    out = capsys.readouterr().out
    assert "Lite" in out and "--reserve-vram 1.0" in out and "swap" in out
    assert cli.main(["--show-tier", "--tier", "max"]) == 0
    assert "Max" in capsys.readouterr().out


# --------------------------------------------------------------------------- models per tier
def test_models_for_each_tier():
    base = Models()
    lite, std, mx = (tiers.models_for(k, base) for k in tiers.TIER_KEYS)
    assert std == base  # Standard = the defaults the app was built on
    assert lite.klein_unet == "flux-2-klein-4b-fp8.safetensors"
    assert lite.wan_gguf.endswith("Q3_K_M.gguf") and lite.animate_gguf.endswith("Q3_K_M.gguf")
    assert mx.wan22_high == "Wan2.2-I2V-A14B-HighNoise-Q8_0.gguf" and mx.animate_gguf.endswith("Q8_0.gguf")
    assert mx.wan_gguf == "wan2.1-i2v-14b-480p-Q8_0.gguf"


def test_your_own_model_filename_wins(monkeypatch):
    monkeypatch.setenv("MC_ANIMATE_GGUF", "my-animate.gguf")
    base = Models()
    assert tiers.models_for("lite", base).animate_gguf == "my-animate.gguf"


def test_lite_motion_is_short_480p_at_24fps():
    assert tiers.motion_frames("lite", 81) == (73, 24)  # ~3 s cap
    assert tiers.motion_frames("lite", 49) == (73, 24)
    assert tiers.motion_frames("standard", 81) == (81, 16)
    assert tiers.motion_frames("max", 49) == (49, 16)


def test_ti2v_graph_validates_on_v041_core(models, object_info_v041):
    sg = graphs.ti2v_motion_graph("x", models=models, width=480, height=832, frames=73)
    nodes = {n["class_type"]: n for n in sg.graph.values()}
    assert nodes["Wan22ImageToVideoLatent"]["inputs"]["length"] == 73
    assert nodes["CLIPLoader"]["inputs"]["device"] == "cpu" and "VAEDecodeTiled" in nodes
    assert nodes["KSampler"]["inputs"]["steps"] == 20 and nodes["KSampler"]["inputs"]["cfg"] == 5.0
    assert sg.meta["fps"] == 24
    problems = validate_graph(sg.graph, object_info_v041)
    # the reference PC already has the Wan 2.2 VAE; the only gap is the 5.4 GB Lite GGUF (see models.md)
    assert problems == ["node 1 (UnetLoaderGGUF).unet_name: 'Wan2.2-TI2V-5B-Q8_0.gguf' not in available options"]


def test_ti2v_fp16_safetensors_uses_core_loader_and_is_on_the_pc(models, object_info_v041):
    from dataclasses import replace

    sg = graphs.ti2v_motion_graph("x", models=replace(models, ti2v_model="wan2.2_ti2v_5B_fp16.safetensors"))
    assert sg.graph["1"]["class_type"] == "UNETLoader"
    assert validate_graph(sg.graph, object_info_v041) == []


def test_ti2v_graph_rounds_to_multiples_of_32_and_optional_lora(models):
    from dataclasses import replace

    sg = graphs.ti2v_motion_graph("x", models=replace(models, ti2v_lora="fastwan.safetensors"), width=500,
                                  height=850, frames=70)
    lat = next(n for n in sg.graph.values() if n["class_type"] == "Wan22ImageToVideoLatent")["inputs"]
    assert (lat["width"], lat["height"], lat["length"]) == (480, 832, 69)
    ks = next(n for n in sg.graph.values() if n["class_type"] == "KSampler")["inputs"]
    assert ks["cfg"] == 1.0 and ks["steps"] == 6


def test_lite_scene_graph_offloads_text_encoder(models, object_info):
    sg = graphs.avatar_graph("x", models=tiers.models_for("lite", models), reference=1, width=768, height=1360,
                             te_device="cpu")
    clip = next(n for n in sg.graph.values() if n["class_type"] == "CLIPLoader")
    assert clip["inputs"]["device"] == "cpu"
    assert validate_graph(sg.graph, object_info) == [
        "node 1 (UNETLoader).unet_name: 'flux-2-klein-4b-fp8.safetensors' not in available options"]


# --------------------------------------------------------------------------- gating
@pytest.mark.parametrize("mode", ["talk", "dialogue", "sing", "swap", "vibe"])
def test_lite_gates_16gb_modes_with_a_friendly_message(mode):
    msg = tiers.gate("lite", mode)
    assert msg and "16 GB+" in msg and "Standard" in msg
    assert tiers.gate("lite", mode, allow_heavy=True) is None
    assert tiers.gate("standard", mode) is None and tiers.gate("max", mode) is None


def test_lite_allows_silent_memes_but_gates_the_line():
    assert tiers.gate("lite", "meme") is None
    assert "Saying the line" in tiers.gate("lite", "meme", say_line=True)


def test_gated_request_fails_validation_not_the_gpu(sample_photo, tmp_path):
    clip = _clip(tmp_path / "c.mp4", 4)
    req = Request(mode="vibe", tier="lite", clip=clip, photo=sample_photo, clip_seconds=4, consent=True)
    with pytest.raises(InputError, match="16 GB"):
        check_request(req, Settings())
    with pytest.raises(InputError, match="16 GB"):
        check_request(Request(mode="talk", tier="lite", script="hi", voice_mode="builtin", character="a robot"),
                      Settings())
    check_request(Request(mode="talk", tier="lite", allow_heavy=True, script="hi", voice_mode="builtin",
                          character="a robot"), Settings())


def test_animate_caps_per_tier(sample_photo, tmp_path):
    clip = _clip(tmp_path / "c.mp4", 2)
    base = dict(mode="swap", clip=clip, photo=sample_photo, consent=True)
    with pytest.raises(InputError, match="1-10 s on MC Lite"):
        check_request(Request(**base, tier="lite", allow_heavy=True, clip_seconds=12), Settings())
    with pytest.raises(InputError, match="1-30 s"):
        check_request(Request(**base, tier="standard", clip_seconds=31), Settings())
    check_request(Request(**base, tier="max", clip_seconds=60), Settings())
    with pytest.raises(InputError, match="1-60 s"):
        check_request(Request(**{**base, "mode": "vibe"}, tier="max", clip_seconds=61), Settings())


def test_legacy_profile_name_and_unknown_tier():
    req = Request(mode="meme", tier="12gb", photo=None, avatar=None, character="x")
    with pytest.raises(InputError):  # still needs a photo, but the tier itself was accepted
        check_request(req, Settings())
    assert req.tier == "lite"
    with pytest.raises(InputError, match="unknown tier"):
        check_request(Request(mode="meme", tier="huge"), Settings())


def test_cli_gate_is_a_friendly_exit(monkeypatch, capsys, tmp_path, sample_photo):
    code = cli.main(["--backend", "mock", "--tier", "lite", "--mode", "talk", "--character", "a robot",
                     "--builtin-voice", "--script", "hi", "--output-dir", str(tmp_path)])
    assert code == 2 and "16 GB+" in capsys.readouterr().out


# --------------------------------------------------------------------------- ETAs
def test_etas_scale_per_tier():
    for frames in (77, 160, 480):
        lite, std, mx = (estimate_animate_seconds(frames, "move", k) for k in tiers.TIER_KEYS)
        assert lite > std and mx > 0
    assert estimate_animate_seconds(480, "move", "standard") == estimate_animate_seconds(480, "move")
    assert estimate_lipsync_seconds(10, tier="lite") > estimate_lipsync_seconds(10) > estimate_lipsync_seconds(10, tier="max")
    assert estimate_motion_seconds(73, "lite") == pytest.approx(120)
    assert estimate_motion_seconds(81, "standard") == pytest.approx(115)


# --------------------------------------------------------------------------- end to end (mock)
def test_lite_meme_end_to_end_uses_5b(pipe, sample_photo):
    evs = list(pipe.run(Request(mode="meme", meme="side_eye", photo=sample_photo, consent=True, tier="lite")))
    last = evs[-1]
    assert last.status == "complete"
    assert abs(media_duration(last.artifact) - 73 / 24) < 0.2 and video_size(last.artifact) == (1080, 1920)
    proj = json.loads((last.artifact.parent / "_work" / "project.json").read_text())
    assert proj["tier"]["key"] == "lite" and proj["tier"]["motion_engine"] == "ti2v5b"
    assert "Wan2.2-TI2V-5B-Q8_0.gguf" in proj["models"]["meme_motion"]
    assert "flux-2-klein-4b-fp8.safetensors" in proj["models"]["avatar"]
    motion_start = next(e for e in evs if e.stage == "motion" and e.status == "start")
    assert "Lite" in motion_start.message and "estimate" in motion_start.message
    recipe = json.loads((last.artifact.parent / "_work" / "mc_recipe.json").read_text())
    assert recipe["tier"] == "lite" and "tier" not in recipe["settings"]


def test_max_meme_uses_q8_at_576(pipe, sample_photo):
    evs = list(pipe.run(Request(mode="meme", meme="side_eye", photo=sample_photo, consent=True, tier="max")))
    proj = json.loads((evs[-1].artifact.parent / "_work" / "project.json").read_text())
    assert "Wan2.2-I2V-A14B-HighNoise-Q8_0.gguf" in proj["models"]["meme_motion"]
    wf = proj["prompts"]["meme_motion"]
    assert wf  # prompts recorded
    assert proj["tier"]["motion_size"] == [576, 1024]


def test_lite_experimental_swap_is_capped_and_small(pipe, sample_photo, tmp_path):
    clip = _clip(tmp_path / "c.mp4", 12)
    req = Request(mode="swap", tier="lite", allow_heavy=True, clip=clip, photo=sample_photo, clip_seconds=10,
                  consent=True, style="none")
    evs = list(pipe.run(req))
    last = evs[-1]
    assert last.status == "complete"
    proj = json.loads((last.artifact.parent / "_work" / "project.json").read_text())
    assert (proj["clip"]["width"], proj["clip"]["height"]) == (384, 672)
    assert "Wan2.2-Animate-14B-Q3_K_M.gguf" in proj["models"]["swap"]
    start = next(e for e in evs if e.stage == "swap" and e.status == "start")
    assert "12 GB" in start.message


def test_remix_renders_on_your_tier(pipe, sample_photo):
    evs = list(pipe.run(Request(mode="meme", meme="side_eye", photo=sample_photo, consent=True, tier="max")))
    doc = json.loads((evs[-1].artifact.parent / "_work" / "mc_recipe.json").read_text())
    req = request_from_recipe(doc, photo=sample_photo, consent=True, tier="lite")
    assert req.tier == "lite"
    assert request_from_recipe(doc, photo=sample_photo).tier == tiers.DEFAULT_TIER
    legacy = {**doc, "settings": {**doc["settings"], "profile": "24gb"}}  # pre-tier recipe files
    assert request_from_recipe(legacy, photo=sample_photo).tier == tiers.DEFAULT_TIER


# --------------------------------------------------------------------------- UI
def test_ui_tier_badge_toggle_and_hints(monkeypatch):
    from main_character_comfy.ui import animate_eta_hint, build_ui, more_modes_gate, tier_info

    monkeypatch.setenv("MC_VRAM_GB", "12")
    assert build_ui("mock") is not None
    c = tiers.resolve()
    assert "Lite" in tier_info(c, "lite") and "Needs 16 GB+" in tier_info(c, "lite")
    assert "you picked" in tier_info(c, "standard")
    assert animate_eta_hint(10, "move", "lite").startswith("🔒") and "16 GB+" in animate_eta_hint(10, "move", "lite")
    assert "Experimental" in animate_eta_hint(10, "move", "lite", True)
    assert "24 GB" in animate_eta_hint(60, "move", "max") and "16 GB RTX 5080" in animate_eta_hint(30, "move")
    assert more_modes_gate("lite", False) and more_modes_gate("standard", False) == ""
