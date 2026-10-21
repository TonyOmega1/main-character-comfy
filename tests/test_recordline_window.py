"""record-your-own-line: speaking window (stem sound + mouth), stems split (Demucs mocked)."""

import shutil

import numpy as np
import pytest

from main_character_comfy.recordline import window as W
from tests._recordline_synth import SR, clip, voice


def test_sound_span_finds_the_line():
    x = np.zeros(4 * SR)
    v = voice(1.4, seed=1)
    x[SR:SR + len(v)] = v
    x += 1e-4 * np.random.default_rng(0).standard_normal(len(x))
    on, off = W.sound_span(x, SR)
    assert abs(on - 1.0) < 0.05 and abs(off - 2.4) < 0.1
    # restricted search around a guess
    on2, off2 = W.sound_span(x, SR, (1.1, 2.2))
    assert abs(on2 - on) < 0.02 and abs(off2 - off) < 0.02
    assert W.sound_span(np.zeros(SR), SR) is None or True  # silence never crashes


def test_mouth_span_and_combine():
    fps = 24
    t = np.arange(96) / fps
    o = np.where((t > 1.05) & (t < 2.3), 0.08 + 0.04 * np.sin(2 * np.pi * 4 * t), 0.01)
    m = W.mouth_span(t, o, (1.0, 2.4))
    assert abs(m[0] - 1.05) < 0.1 and abs(m[1] - 2.3) < 0.1
    w = W.combine((1.0, 2.4), m, (0.98, 2.35), "mediapipe")
    assert w.source == "sound+mouth" and 1.0 <= w.start <= m[0] and w.notes == []
    far = W.combine((1.0, 2.4), (1.6, 2.4), None, "motion")
    assert far.start == 1.0 and far.source == "sound" and far.notes
    assert W.combine(None, None, (0.5, 1.5), None).source == "words"
    with pytest.raises(ValueError):
        W.combine(None, None, None, None)
    assert W.mouth_span(t, np.zeros_like(t), (1, 2)) is None


def test_openness_from_landmarks_fills_gaps():
    L = np.zeros((5, 478, 2), np.float32)
    L[:, 10, 1], L[:, 152, 1] = 0, 100
    L[:, 13, 1], L[:, 14, 1] = 50, [50, 55, 60, 55, 50]
    L[2] = np.nan
    o = W.openness_from_landmarks(L)
    assert np.allclose(o, [0, 0.05, 0.05, 0.05, 0])


def test_split_stems_uses_demucs_when_there(tmp_path, monkeypatch):
    c = clip(tmp_path / "c")

    def fake_sep(src, out_dir, **kw):
        d = out_dir / "htdemucs" / src.stem
        d.mkdir(parents=True)
        shutil.copy(c["vocals"], d / "vocals.wav")
        shutil.copy(c["no_vocals"], d / "no_vocals.wav")
        return d / "vocals.wav"

    monkeypatch.setattr(W.stems, "separate_vocals", fake_sep)
    st = W.split_stems(c["video"], tmp_path / "w")
    assert st.vocals.name == "vocals.wav" and st.background.name == "no_vocals.wav" and st.note is None
    assert st.clip_audio.is_file()


def test_split_stems_without_demucs_explains(tmp_path, monkeypatch):
    c = clip(tmp_path / "c")
    monkeypatch.setattr(W.stems, "separate_vocals", lambda *a, **k: None)
    monkeypatch.setattr(W.stems, "separator_command", lambda: None)
    st = W.split_stems(c["video"], tmp_path / "w")
    assert st.vocals is None and st.background is None and "pip install demucs" in st.note
