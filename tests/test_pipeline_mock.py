import json
from dataclasses import replace

import pytest

from main_character_comfy.backends import MockBackend
from main_character_comfy.config import Settings
from main_character_comfy.media import media_duration
from main_character_comfy.pipeline import InputError, Pipeline, Request, split_batch_text


@pytest.fixture
def pipe(tmp_path):
    return Pipeline(MockBackend(), Settings(output_dir=tmp_path / "out"))


def _final(events):
    evs = list(events)
    assert evs[-1].status == "complete"
    return evs, evs[-1]


def test_talk_clone_end_to_end(pipe, sample_photo, sample_voice):
    req = Request(script="Hey! Welcome back to the channel, let's build something fun today.", photo=sample_photo,
                  voice_sample=sample_voice, style="anime", consent=True)
    evs, last = _final(pipe.run(req))
    stages = {e.stage for e in evs}
    assert {"prepare", "avatar", "voice", "lipsync", "finish"} <= stages
    proj = json.loads((last.artifact.parent / "_work" / "project.json").read_text())
    assert proj["captions"] and proj["backend"] == "mock"
    assert abs(media_duration(last.artifact) - proj["duration"]) < 0.15


def test_consent_required_for_uploads(pipe, sample_photo, sample_voice):
    with pytest.raises(InputError, match="consent"):
        list(pipe.run(Request(script="hi", photo=sample_photo, voice_sample=sample_voice)))


def test_description_and_builtin_voice_need_no_consent(pipe):
    _final(pipe.run(Request(script="Hello world, I am new here.", character="a friendly robot chef", voice_mode="builtin")))


def test_sfw_filter(pipe):
    with pytest.raises(InputError, match="SFW"):
        list(pipe.run(Request(script="something nsfw", character="robot", voice_mode="builtin")))


def test_long_script_is_segmented(pipe):
    script = " ".join(["This is a long sentence for the segmentation test."] * 9)
    req = Request(script=script, character="a robot", voice_mode="builtin", segment_seconds=6.0, seam="continue")
    evs, last = _final(pipe.run(req))
    proj = json.loads((last.artifact.parent / "_work" / "project.json").read_text())
    assert len(proj["segments"]) >= 3
    assert sum(k.startswith("lipsync_") for k in proj["jobs"]) == len(proj["segments"])
    assert abs(media_duration(last.artifact) - proj["duration"]) < 0.15


def test_dialogue_mode(pipe):
    req = Request(mode="dialogue", character="two robots", script="Ava: Hello!\nBo: Hi Ava.\nAva: Ready?")
    evs, last = _final(pipe.run(req))
    proj = json.loads((last.artifact.parent / "_work" / "project.json").read_text())
    assert {c["speaker"] for c in proj["captions"]} == {"A", "B"}
    assert sum(k.startswith("voice_") for k in proj["jobs"]) == 3


def test_sing_mode_uses_mix_as_soundtrack(pipe, sample_photo, sample_voice):
    req = Request(mode="sing", photo=sample_photo, song=sample_voice, vocals=sample_voice, song_start=1, song_seconds=6,
                  script="la la la", consent=True, style="none")
    evs, last = _final(pipe.run(req))
    assert abs(media_duration(last.artifact) - 6.0) < 0.15


def test_batch(pipe):
    base = Request(character="a cheerful robot", voice_mode="builtin")
    evs = list(pipe.run_batch(base, split_batch_text("First clip here.\nSecond clip here.")))
    assert evs[-1].status == "complete" and len(evs[-1].data["clips"]) == 2
    # avatar generated once, reused for clip 2
    assert sum(1 for e in evs if e.stage == "avatar" and e.status == "start") == 1


def test_split_batch_text():
    assert split_batch_text("a\nb\n") == ["a", "b"]
    assert split_batch_text("a line\nstill a\n\nb") == ["a line\nstill a", "b"]


def test_long_script_tts_is_chunked(pipe):
    from main_character_comfy.pipeline import tts_chunks

    script = " ".join(f"Sentence number {i} is here to make this script long enough." for i in range(12))
    parts = tts_chunks(script)
    assert len(parts) >= 3 and all(len(p) <= 280 for p in parts) and " ".join(parts) == script
    evs, last = _final(pipe.run(Request(script=script, character="a robot", voice_mode="builtin")))
    proj = json.loads((last.artifact.parent / "_work" / "project.json").read_text())
    assert sum(k.startswith("voice_") for k in proj["jobs"]) == len(parts)
