"""The real recordline backend, reached exactly the way the Re-dub tab does: recordline_bridge -> process_line.
CPU only: Demucs and faster-whisper are mocked (or simply absent), the mock backend stands in for ComfyUI."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from main_character_comfy import extras, recordline_bridge as rb, redub
from main_character_comfy.backends import MockBackend
from main_character_comfy.config import Settings
from main_character_comfy.media import media_duration
from main_character_comfy.pipeline import Pipeline
from main_character_comfy.recordline import asr, core, window as W
from tests._recordline_synth import clip, take
from tests.test_recordline_e2e import LINE, _fake_asr


@pytest.fixture
def scene(tmp_path, monkeypatch):
    monkeypatch.delenv("MC_RECORDLINE", raising=False)
    monkeypatch.setattr(core, "mouth_track", lambda *a, **k: None)
    monkeypatch.setattr(redub, "_deepfilter_cmd", lambda: None)
    c = clip(tmp_path / "clip", seconds=4.0, line=(1.0, 2.4))

    def fake_sep(src, out_dir, **kw):
        d = out_dir / "htdemucs" / src.stem
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(c["vocals"], d / "vocals.wav")
        shutil.copy(c["no_vocals"], d / "no_vocals.wav")
        return d / "vocals.wav"

    monkeypatch.setattr(W.stems, "separate_vocals", fake_sep)
    return c


def test_bridge_finds_the_real_backend():
    import os

    if os.environ.get("MC_RECORDLINE"):
        pytest.skip("MC_RECORDLINE is set in this environment")
    mod = rb.backend()
    assert mod is not None and mod.__name__ == "main_character_comfy.recordline"
    assert "recordline backend" in rb.describe()


def test_process_line_contract(scene, tmp_path, monkeypatch):
    _fake_asr(monkeypatch, {})
    t1 = take(tmp_path / "a.wav", 1.45, seed=1)
    t2 = take(tmp_path / "b.wav", 2.7, seed=2)
    out = rb.process(rb.backend(), scene["video"], t2, start=0.9, work=tmp_path / "work",
                     settings={"takes": [t2, t1], "duck": 0.28, "denoise": "off", "voice_level_db": 0.0,
                               "lipsync": False, "seed": 1, "fit_seconds": None})
    assert Path(out["output"]).suffix == ".mp4" and abs(media_duration(out["output"]) - 4.0) < 0.1
    assert out["chosen_take"] == "take2" and out["aligned_voice_wav"] and out["final_mix_wav"]
    assert all(isinstance(s, str) for s in out["steps"]) and out["steps"][0].startswith("Picked take 2 of 2")
    assert out["line_seconds"] > 1.0


def test_redub_tab_path_end_to_end(scene, tmp_path, monkeypatch):
    """extras.run_redub (what the Re-dub tab and the agent tool call) -> bridge -> recordline, no ASR installed."""
    monkeypatch.setattr(asr, "available", lambda: False)
    line = take(tmp_path / "line.wav", 1.4, seed=3)
    pipe = Pipeline(MockBackend(), Settings(output_dir=tmp_path / "out"))
    evs = list(extras.run_redub(pipe, scene["video"], recording=line, start=1.0, denoise="off", consent=True))
    assert evs[-1].status == "complete" and Path(evs[-1].artifact).exists()
    msgs = " ".join(e.message or "" for e in evs)
    assert "recordline backend" in msgs and "pip install faster-whisper" in msgs
    assert abs(media_duration(evs[-1].artifact) - 4.0) < 0.1
    mix = redub.load(next((tmp_path / "out").rglob("final_mix.wav")), 2)
    assert np.sqrt((mix[int(1.2 * 48000):int(2.2 * 48000)] ** 2).mean()) > 0.01
