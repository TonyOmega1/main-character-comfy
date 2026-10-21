"""record-your-own-line: pick_and_align end to end on synthetic clips (ASR and Demucs mocked, CPU, seconds)."""

import json
import shutil

import numpy as np
import pytest

from main_character_comfy import redub
from main_character_comfy.media import media_duration, run_ffmpeg
from main_character_comfy.recordline import RecordLineError, asr, core, mux, pick_and_align, window as W
from tests._recordline_synth import SR, clip, take

LINE = "you're not you when you're gay"


def _fake_asr(monkeypatch, by_seconds: dict, default=LINE, prob=0.9):
    """Stand-in for faster-whisper: the text is looked up by the audio's length; words are spread over the part
    of the audio that's above the noise, like real word timestamps."""
    monkeypatch.setattr(asr, "available", lambda: True)

    def fake(a16, model=None, language="en"):
        secs = round(len(a16) / 16000, 1)
        text = by_seconds.get(secs, default)
        a = np.abs(np.asarray(a16, float))
        n = 160
        fr = np.sqrt(np.mean(a[: len(a) // n * n].reshape(-1, n) ** 2, 1))
        act = np.where(fr > fr.max() * 0.05)[0] if len(fr) and fr.max() > 0 else []
        if not len(act):
            return asr.Transcript()
        t0, t1 = act[0] * 0.01, (act[-1] + 1) * 0.01
        words = text.split()
        edges = np.linspace(t0, t1, len(words) + 1)
        p = prob if not isinstance(prob, dict) else prob.get(secs, 0.9)
        return asr.Transcript(text, [asr.Word(w, round(a_, 3), round(b_, 3), p)
                                     for w, a_, b_ in zip(words, edges[:-1], edges[1:])])

    monkeypatch.setattr(asr, "transcribe", fake)


def _fake_demucs(monkeypatch, c):
    def fake_sep(src, out_dir, **kw):
        d = out_dir / "htdemucs" / src.stem
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(c["vocals"], d / "vocals.wav")
        shutil.copy(c["no_vocals"], d / "no_vocals.wav")
        return d / "vocals.wav"

    monkeypatch.setattr(W.stems, "separate_vocals", fake_sep)


@pytest.fixture
def scene(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "mouth_track", lambda *a, **k: None)  # test pattern has no face
    monkeypatch.setattr(redub, "_deepfilter_cmd", lambda: None)
    c = clip(tmp_path / "clip", seconds=4.0, line=(1.0, 2.4))
    _fake_demucs(monkeypatch, c)
    return c


def test_picks_the_take_that_says_the_line_and_fits(scene, tmp_path, monkeypatch):
    seen = []
    real = redub.run_ffmpeg
    import main_character_comfy.recordline.chain as chain_mod
    import main_character_comfy.recordline.window as win_mod

    def spy(args, *a, **k):
        seen.append(" ".join(map(str, args)))
        return real(args, *a, **k)

    for mod in (chain_mod, win_mod, core):
        monkeypatch.setattr(mod, "run_ffmpeg", spy)
    t1 = take(tmp_path / "t1.wav", 1.5, seed=1, lead=0.6)         # 2.5 s file: says the wrong thing
    t2 = take(tmp_path / "t2.wav", 1.55, seed=2, lead=0.35)      # says the line, fits
    t3 = take(tmp_path / "t3.wav", 2.6, seed=3)                   # way too long
    webm = tmp_path / "t2.webm"                                   # as a browser recorder sends it (opus)
    run_ffmpeg(["-i", str(t2), "-c:a", "libopus", "-b:a", "128k", str(webm)])
    _fake_asr(monkeypatch, {2.5: "you're not shoe when you're gray"}, default=LINE)
    r = pick_and_align(scene["video"], [t1, webm, t3], LINE, out_dir=tmp_path / "out")
    json.dumps(r)  # JSON-able
    assert r["chosen_take"] == "take2", [(t["take_id"], t["score"], t["wer"], t["duration_fit"]) for t in r["takes"]]
    by = {t["take_id"]: t for t in r["takes"]}
    assert by["take1"]["wer"] > 0.3 and by["take2"]["wer"] == 0
    assert 0.85 <= by["take2"]["duration_fit"] <= 1.25 and by["take3"]["duration_fit"] > 1.5
    assert abs(r["window_start"] - 1.0) < 0.08 and abs(r["window_end"] - 2.4) < 0.15
    assert 0.85 <= r["stretch_ratio"] <= 1.15
    assert "Picked take 2 of 3." in r["warnings"]
    # outputs: full clip length, onset on the window
    for k in ("final_mix_wav", "aligned_voice_wav", "voice_wet_wav"):
        assert abs(media_duration(r[k]) - 4.0) < 0.03, k
    v = redub.load(r["aligned_voice_wav"])
    on = np.argmax(np.abs(v) > np.abs(v).max() * 0.02) / SR
    assert abs(on - r["window_start"]) < 0.06
    # InfiniteTalk wav: 0.6 s silent head pad, 16 kHz
    it = redub.load(r["infinitetalk_wav"])
    assert np.abs(it[: int(0.55 * SR)]).max() < 1e-3 and np.abs(it).max() > 0.01
    assert abs(r["infinitetalk_clip_t0"] - (r["window_start"] - 0.6)) < 1e-6
    # never loudnorm / limiters / low-bitrate AAC
    bad = ("loudnorm", "alimiter", "dynaudnorm", "speechnorm", "asoftclip", "aexciter", "crystalizer", "anoisesrc")
    assert seen and not any(b in cmd for cmd in seen for b in bad)
    # true peak ceiling and one static gain
    mix = redub.load(r["final_mix_wav"], 2)
    assert redub.true_peak_db(mix) <= -1.4
    # no noise-floor holes: the music keeps running under the line (no frame drops to digital silence)
    seg = mix[int(0.5 * SR):int(3.5 * SR)].mean(1)
    fr = np.sqrt(np.mean(seg[: len(seg) // 480 * 480].reshape(-1, 480) ** 2, 1))
    assert fr.min() > 10 ** (-50 / 20)
    bed = redub.load(scene["no_vocals"], 2)
    ratio_under = np.sqrt((mix[int(1.5 * SR):int(2 * SR)] ** 2).mean()) / np.sqrt((bed[int(1.5 * SR):int(2 * SR)] ** 2).mean())
    assert ratio_under > 0.3  # voice + bed, not voice alone
    assert (tmp_path / "out" / "recordline.json").is_file()
    assert r["verify"]["wer"] == 0
    out = mux(scene["video"], r["final_mix_wav"], tmp_path / "out.mp4")
    assert abs(media_duration(out) - 4.0) < 0.1


def test_single_long_take_is_trimmed_with_a_warning(scene, tmp_path, monkeypatch):
    _fake_asr(monkeypatch, {})
    t = take(tmp_path / "long.wav", 2.6, seed=4)
    r = pick_and_align(scene["video"], [t], out_dir=tmp_path / "o")
    assert r["chosen_take"] == "take1" and r["stretch_ratio"] == pytest.approx(0.85)
    assert any("too long to fit, even sped up" in w for w in r["warnings"])
    assert r["voice_end"] <= r["window_end"] + 0.31
    assert r["target_text"] == LINE  # from speech-to-text of the original line


def test_low_clarity_warning(scene, tmp_path, monkeypatch):
    _fake_asr(monkeypatch, {}, prob={2.3: 0.3})
    t = take(tmp_path / "t.wav", 1.5, seed=6)
    r = pick_and_align(scene["video"], [t], LINE, out_dir=tmp_path / "o")
    assert any("Low clarity" in w and "closer to the mic" in w for w in r["warnings"])


def test_given_window_and_no_asr_no_stems_still_work(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "mouth_track", lambda *a, **k: None)
    monkeypatch.setattr(redub, "_deepfilter_cmd", lambda: None)
    monkeypatch.setattr(asr, "available", lambda: False)
    monkeypatch.setattr(W.stems, "separate_vocals", lambda *a, **k: None)
    monkeypatch.setattr(W.stems, "separator_command", lambda: None)
    c = clip(tmp_path / "c", seconds=4.0, line=(1.0, 2.4))
    good = take(tmp_path / "a.wav", 1.45, seed=1)
    short = take(tmp_path / "b.wav", 0.6, seed=2)
    r = pick_and_align(c["video"], [short, good], out_dir=tmp_path / "o", window=(1.0, 2.4))
    assert r["chosen_take"] == "take2"  # picked on fit alone
    assert r["window"]["source"] == "given" and r["window_start"] == 1.0
    w = " ".join(r["warnings"])
    assert "pip install faster-whisper" in w and "pip install demucs" in w
    assert all(t["wer"] is None for t in r["takes"])
    mix = redub.load(r["final_mix_wav"], 2)
    assert np.sqrt((mix[int(1.2 * SR):int(2.2 * SR)] ** 2).mean()) > 0.01


def test_bad_input(scene, tmp_path):
    with pytest.raises(RecordLineError, match="at least one take"):
        pick_and_align(scene["video"], [], out_dir=tmp_path / "o")
    with pytest.raises(RecordLineError, match="Can't find"):
        pick_and_align(scene["video"], [tmp_path / "nope.wav"], out_dir=tmp_path / "o")
    t = take(tmp_path / "t.wav", 1.0)
    with pytest.raises(RecordLineError, match="outside the clip"):
        pick_and_align(scene["video"], [t], out_dir=tmp_path / "o", window=(3.0, 9.0))
    silent = redub.save(tmp_path / "s.wav", np.zeros(SR))
    with pytest.raises(RecordLineError, match="no speech|None of the takes"):
        pick_and_align(scene["video"], [silent], out_dir=tmp_path / "o2", window=(1.0, 2.4))
    assert isinstance(RecordLineError("x"), ValueError)  # extras maps ValueError to InputError
