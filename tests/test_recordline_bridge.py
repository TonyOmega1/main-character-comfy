"""Wiring stub for DubMaster's recordline backend (feat/record-line). CPU / mock only; no recordline package here."""

from __future__ import annotations

import shutil
import sys
import types
from pathlib import Path

from main_character_comfy import extras, recordline_bridge as rb
from main_character_comfy.backends import MockBackend
from main_character_comfy.config import Settings
from main_character_comfy.media import run_ffmpeg
from main_character_comfy.pipeline import Pipeline


def test_no_backend_until_it_lands(monkeypatch):
    monkeypatch.delenv("MC_RECORDLINE", raising=False)
    for name in rb.CANDIDATES:
        monkeypatch.setitem(sys.modules, name, None)  # import -> ImportError
    assert rb.backend() is None and "built-in" in rb.describe()


def test_fake_backend_gets_the_line(tmp_path, monkeypatch):
    clip = tmp_path / "talk.mp4"
    run_ffmpeg(["-f", "lavfi", "-i", "testsrc2=size=160x120:rate=24:duration=2", "-f", "lavfi", "-i",
                "sine=frequency=300:duration=2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
                str(clip)])
    line = tmp_path / "line.wav"
    run_ffmpeg(["-f", "lavfi", "-i", "sine=frequency=500:duration=0.8", "-ar", "48000", str(line)])
    seen = {}

    def process_line(video, line_wav, *, start, work, settings):
        seen.update(video=video, line=line_wav, start=start, settings=settings)
        work.mkdir(parents=True, exist_ok=True)
        out = work / "out.mp4"
        shutil.copy(video, out)
        return {"output": out, "steps": ["fake mix"], "aligned_voice_wav": str(line_wav), "final_mix_wav": "mix.wav",
                "voice_start": 0.5, "voice_end": 1.3}

    fake = types.ModuleType("fake_recordline")
    fake.process_line = process_line
    monkeypatch.setitem(sys.modules, "fake_recordline", fake)
    monkeypatch.setenv("MC_RECORDLINE", "fake_recordline")
    assert rb.backend() is fake
    pipe = Pipeline(MockBackend(), Settings(output_dir=tmp_path / "out"))
    lips = {}

    def fake_lipsync(pipe_, video, mixed, voice, start, line_seconds, *a):
        lips.update(mixed=mixed, voice=voice, start=start, line_seconds=line_seconds)
        return None
        yield  # noqa: unreachable (makes this a generator like the real one)

    monkeypatch.setattr(extras, "_lipsync_line", fake_lipsync)
    evs = list(extras.run_redub(pipe, clip, recording=line, start=0.5, denoise="off", consent=True, lipsync=True))
    assert evs[-1].status == "complete" and Path(evs[-1].artifact).exists()
    assert seen["start"] == 0.5 and seen["settings"]["lipsync"] is True and seen["line"].suffix == ".wav"
    assert any("fake mix" in (e.message or "") for e in evs)
    monkeypatch.setenv("MC_RECORDLINE", "off")
    assert rb.backend() is None
    # the bridge path still lip-syncs (was skipped before): no line_seconds in the dict -> voice_end - voice_start
    assert lips["start"] == 0.5 and abs(lips["line_seconds"] - 0.8) < 1e-9 and lips["mixed"].suffix == ".mp4"


def test_bridge_gets_original_takes_and_lipsyncs(tmp_path, monkeypatch):
    clip = tmp_path / "talk.mp4"
    run_ffmpeg(["-f", "lavfi", "-i", "testsrc2=size=160x120:rate=24:duration=2", "-f", "lavfi", "-i",
                "sine=frequency=300:duration=2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
                str(clip)])
    t1, t2 = tmp_path / "take1.webm", tmp_path / "take2.wav"
    run_ffmpeg(["-f", "lavfi", "-i", "sine=frequency=500:duration=0.8", "-c:a", "libopus", str(t1)])
    run_ffmpeg(["-f", "lavfi", "-i", "sine=frequency=600:duration=0.8", "-ar", "48000", str(t2)])
    seen, lips = {}, []

    def process_line(video, line_wav, *, start, work, settings):
        seen.update(settings)
        work.mkdir(parents=True, exist_ok=True)
        out = work / "out.mp4"
        shutil.copy(video, out)
        return {"output": out, "steps": [], "aligned_voice_wav": str(line_wav), "final_mix_wav": "m.wav",
                "voice_start": 0.2, "voice_end": 1.0, "line_seconds": 0.8}

    def fake_lipsync(pipe_, video, mixed, voice, start, line_seconds, *a):
        lips.append(line_seconds)
        synced = tmp_path / "synced.mp4"
        shutil.copy(mixed, synced)
        return synced
        yield  # noqa

    fake = types.ModuleType("fake_recordline2")
    fake.process_line = process_line
    monkeypatch.setitem(sys.modules, "fake_recordline2", fake)
    monkeypatch.setenv("MC_RECORDLINE", "fake_recordline2")
    monkeypatch.setattr(extras, "_lipsync_line", fake_lipsync)
    pipe = Pipeline(MockBackend(), Settings(output_dir=tmp_path / "out"))
    evs = list(extras.run_redub(pipe, clip, takes=[t1, t2], consent=True, lipsync=True))
    assert seen["takes"] == [str(t1), str(t2)]  # the original uploads, not 24 kHz copies
    assert lips == [0.8] and evs[-1].status == "complete"
