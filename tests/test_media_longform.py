import numpy as np
import pytest
from PIL import Image

from main_character_comfy import dialogue, longform, media


def test_prepare_photo_is_9_16(sample_photo, tmp_path):
    out = media.prepare_photo(sample_photo, tmp_path / "p.png")
    assert Image.open(out).size == (1088, 1920)


def test_prepare_voice_trims_and_resamples(sample_voice, tmp_path):
    dur = media.prepare_voice(sample_voice, tmp_path / "v.wav", max_seconds=10)
    assert 9.5 <= dur <= 10.05
    import wave

    with wave.open(str(tmp_path / "v.wav")) as w:
        assert (w.getframerate(), w.getnchannels()) == (24000, 1)


def test_caption_chunks_and_timing():
    script = "Hey, welcome back! Today we build a talking avatar with open models only. Let's go."
    caps = media.time_captions(script, 6.0)
    assert caps[0].start >= 0 and caps[-1].end <= 6.0
    assert all(a.end <= b.start + 1e-6 for a, b in zip(caps, caps[1:]))
    assert all(len(c.text) <= 30 for c in caps)
    assert " ".join(c.text for c in caps).split() == script.split()


def _tone(tmp_path, name, segments):
    """segments: list of (seconds, on) -> wav with tone/silence."""
    sr = 24000
    parts = []
    for secs, on in segments:
        t = np.arange(int(secs * sr)) / sr
        parts.append((np.sin(2 * np.pi * 220 * t) * 8000 if on else np.zeros_like(t)))
    return dialogue._write(tmp_path / name, np.concatenate(parts))


def test_pause_detection_and_segment_plan(tmp_path):
    # 3 phrases of ~7 s separated by 0.4 s pauses -> ~22 s
    wav = _tone(tmp_path, "s.wav", [(7, 1), (0.4, 0), (7, 1), (0.4, 0), (7, 1)])
    pauses = longform.detect_pauses(wav)
    assert len(pauses) == 2 and abs(pauses[0] - 7.2) < 0.1
    segs = longform.plan_segments(media.media_duration(wav), pauses, segment_seconds=15.0)
    assert len(segs) == 2
    assert abs(segs[0].end - 14.6) < 0.1  # cut at the second pause, not mid-word
    assert segs[1].start == segs[0].end and segs[-1].end == segs[-1].render_end
    assert abs(segs[0].render_end - segs[0].end - 0.24) < 1e-6


def test_segment_plan_without_pauses_and_tiny_tail():
    segs = longform.plan_segments(30.0, [], segment_seconds=14.5)
    assert [round(s.start, 2) for s in segs] == [0.0, 14.5, 27.0]  # tail pulled back to >= 3 s
    assert all(s.end - s.start <= 14.5 + 1e-6 for s in segs)
    assert longform.plan_segments(10.0, [], 14.5)[0].render_end == 10.0


def test_stitch_keeps_sync(tmp_path):
    segs = longform.plan_segments(9.0, [], segment_seconds=4.0)
    clips = []
    for s in segs:
        p = tmp_path / f"c{s.index}.mp4"
        media.run_ffmpeg(["-f", "lavfi", "-i", f"testsrc=size=480x832:rate=25:duration={s.render_seconds + 1:.2f}",
                          "-pix_fmt", "yuv420p", str(p)])
        clips.append(p)
    out = longform.stitch(clips, segs, tmp_path / "st.mp4", fps=25, size=(480, 832))
    assert abs(media.media_duration(out) - 9.0) < 0.1


def test_finish_video_burns_captions_and_trims(tmp_path):
    v = tmp_path / "v.mp4"
    a = tmp_path / "a.wav"
    media.run_ffmpeg(["-f", "lavfi", "-i", "testsrc=size=480x832:rate=25:duration=5", "-pix_fmt", "yuv420p", str(v)])
    media.run_ffmpeg(["-f", "lavfi", "-i", "sine=duration=3.5:sample_rate=24000", str(a)])
    caps = media.time_captions("Hello there friends, this is a test.", 3.5)
    out = media.finish_video(v, tmp_path / "f.mp4", audio=a, duration=3.5, captions=caps, ffmpeg_smooth_fps=50)
    assert abs(media.media_duration(out) - 3.5) < 0.1
    assert media.has_audio_stream(out)
    frame = media.poster_frame(out, tmp_path / "f.png", at=1.0)
    assert Image.open(frame).size == (1080, 1920)


def test_dialogue_parse_and_tracks(tmp_path):
    turns, names = dialogue.parse_dialogue("Maya: Hi there!\nTheo: Hey.\n\nMaya: Bye.")
    assert [t.speaker for t in turns] == ["A", "B", "A"] and names == {"A": "Maya", "B": "Theo"}
    with pytest.raises(ValueError):
        dialogue.parse_dialogue("Maya: only me")
    with pytest.raises(ValueError):
        dialogue.parse_dialogue("A: x\nB: y\nC: z")
    clips = [_tone(tmp_path, f"t{i}.wav", [(d, 1)]) for i, d in enumerate([1.0, 0.5, 0.8])]
    built = dialogue.build_tracks(turns, clips, tmp_path / "d")
    a, b = dialogue._read(built["a"]), dialogue._read(built["b"])
    assert len(a) == len(b) and abs(built["duration"] - (2.3 + 4 * 0.25)) < 0.01
    # never both talking at once
    assert not np.any((np.abs(a) > 0) & (np.abs(b) > 0))
    assert {c.speaker for c in built["captions"]} == {"A", "B"}
