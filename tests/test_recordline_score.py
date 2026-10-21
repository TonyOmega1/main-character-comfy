"""record-your-own-line: ASR helpers, scoring, DSP (fast, no models)."""

import numpy as np
import pytest

from main_character_comfy.recordline import asr, deps, dsp, score


def test_wer_and_normalisation():
    assert asr.wer("You're not you when you're gay.", "you're not you when you're gay") == 0
    assert asr.wer("hello biological ladies", "hello logical ladies") == pytest.approx(1 / 3)
    assert asr.wer("I have 2 dogs", "i have two dogs") == 0
    assert asr.wer("", "") == 0 and asr.wer("", "x") == 1
    assert asr.wer("a b", "") == 1


def test_locate_finds_the_line_inside_a_longer_transcript():
    W = asr.Word
    words = [W(w, i * 0.4, i * 0.4 + 0.3, 0.9) for i, w in enumerate("so anyway hello biological ladies and then".split())]
    tr = asr.Transcript("", words)
    i0, i1, e = asr.locate(tr, "Hello, biological ladies.")
    assert (i0, i1, e) == (2, 5, 0.0)
    assert tr.span(i0, i1).start == pytest.approx(0.8)
    assert asr.locate(asr.Transcript(), "x") is None


def test_transcribe_without_faster_whisper_says_how_to_fix(monkeypatch):
    monkeypatch.setattr(deps, "has_module", lambda name: False)
    with pytest.raises(deps.OptionalDependencyMissing, match="pip install faster-whisper"):
        asr.transcribe(np.zeros(16000, np.float32))


def test_fit_score_and_ranking_prefers_the_right_words():
    assert score.fit_score(1.0) == 1 and score.fit_score(0.85) == 1 and score.fit_score(1.15) == 1
    assert 0 < score.fit_score(1.3) < 1 and score.fit_score(2.0) == 0 and score.fit_score(None) == 0
    good = score.compute(score.TakeScore("take1", "a", wer=0.0, min_word_prob=0.8, mean_word_prob=0.9,
                                         duration_fit=1.2, speech_seconds=1))
    wrong = score.compute(score.TakeScore("take2", "b", wer=0.6, min_word_prob=0.9, mean_word_prob=0.95,
                                          duration_fit=1.0, speech_seconds=1))
    clipped = score.compute(score.TakeScore("take3", "c", wer=0.0, min_word_prob=0.8, mean_word_prob=0.9,
                                            duration_fit=1.2, speech_seconds=1, qc={"clipping": True}))
    assert [t.take_id for t in score.rank([wrong, clipped, good])] == ["take1", "take3", "take2"]


def test_take_warnings_are_plain_english():
    t = score.TakeScore("take2", "x", heard="hello logical ladies", wer=0.33, min_word_prob=0.3,
                        weakest_word="logical", duration_fit=1.4, speech_seconds=2)
    w = " ".join(score.take_warnings(t, "Take 2"))
    assert "doesn't quite match" in w and "closer to the mic" in w and "runs long" in w
    assert score.take_warnings(score.TakeScore("t", "x"), "Take 1") == \
        ["Take 1 sounds silent. Check that the right mic was selected."]


def test_slow_peak_control_is_capped_and_leaves_normal_speech_alone():
    sr = dsp.SR
    t = np.arange(sr) / sr
    x = 0.1 * np.sin(2 * np.pi * 200 * t)
    y, cut = dsp.slow_peak_control(x, sr, 4.0)
    assert cut == 0 and np.allclose(x, y)
    spike = x.copy()
    spike[sr // 2:sr // 2 + 200] *= 9  # a +19 dB burst
    y, cut = dsp.slow_peak_control(spike, sr, 4.0)
    assert 0 < cut <= 4.0 + 1e-6
    idx = np.where(np.abs(spike) > 0.05)[0]
    g = y[idx] / spike[idx]
    step = np.abs(np.diff(g))[np.diff(idx) == 1]
    assert g.min() < 0.9 and step.max() < 0.01  # smooth gain, no steps


def test_fft_filter_and_resample():
    sr = dsp.SR
    t = np.arange(sr) / sr
    lo, hi = np.sin(2 * np.pi * 50 * t), np.sin(2 * np.pi * 2000 * t)
    y = dsp.fft_filter(lo + hi, sr, 300, 4000)
    assert np.std(y[1000:-1000] - hi[1000:-1000]) < 0.05
    r = dsp.resample(hi, sr, 16000)
    assert len(r) == 16000 and abs(np.std(r) - np.std(hi)) < 0.01


def test_consensus_drops_loops_and_picks_the_medoid():
    T = lambda text, p=0.9: asr.Transcript(text, [asr.Word(w, i * .2, i * .2 + .1, p) for i, w in enumerate(text.split())])  # noqa: E731
    readings = [T("and in and and and and and and and and"), T("You're you're not you when you're gay."),
                T("Not you when you're gay"), T("you're not you when you're gay", 0.8), T("You're"),
                T("You're not you when you're... Sssssssssss"), asr.Transcript()]
    assert asr.hallucinated(readings[0]) and asr.hallucinated(readings[5]) and not asr.hallucinated(readings[3])
    assert asr.consensus(readings).text == "you're not you when you're gay"
    assert asr.consensus([asr.Transcript()]).words == []
