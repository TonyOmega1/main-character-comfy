"""pick_and_align: score 1-N recorded takes of a line, pick the best, fit it to the character's speaking window and
mix it into the clip. CPU only (no GPU, no ComfyUI). See the package docstring for the hook into the re-dub tab."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..media import has_audio_stream, media_duration, run_ffmpeg
from . import asr, score as scoring, takeqc
from .chain import ChainSettings, build, denoise
from .deps import HINTS, OptionalDependencyMissing, RecordLineError
from .dsp import SR, decode, speech_band_db, to16
from .window import SpeakingWindow, combine, mouth_span, mouth_track, sound_span, split_stems

MAX_TAKES = 20


def jsonable(o):
    """numpy scalars / arrays and Paths -> plain JSON types, recursively."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, Path):
        return str(o)
    return o


@dataclass
class RecordLineResult:
    chosen_take: str
    chosen_take_path: str
    takes: list                      # per-take scores (TakeScore.to_dict), best first
    window_start: float
    window_end: float
    window: dict
    target_text: str | None
    stretch_ratio: float
    aligned_voice_wav: str           # dry new line on the clip timeline, full clip length, 48 kHz mono float
    infinitetalk_wav: str            # 16 kHz mono, 0.6 s silent head pad + the window
    infinitetalk_clip_t0: float      # clip time of the InfiniteTalk wav's sample 0 (may be negative)
    final_mix_wav: str               # full clip soundtrack, 48 kHz stereo 24-bit
    voice_wet_wav: str
    voice_start: float
    voice_end: float
    warnings: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    room: dict = field(default_factory=dict)
    levels: dict = field(default_factory=dict)
    verify: dict = field(default_factory=dict)
    available: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("window_start", "window_end", "stretch_ratio", "infinitetalk_clip_t0", "voice_start", "voice_end"):
            d[k] = round(float(d[k]), 3)
        return jsonable(d)

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), **({"indent": 2} | kw))


def _speech_region(x48: np.ndarray, tr: asr.Transcript | None) -> tuple[float, float]:
    """Speech start / end inside a take. The speech-band level decides (sample accurate); word times only say where
    to look. Whisper pins the first word to 0.0 s when a take opens with gated silence (Rec5 take: words "from" 0.05 s,
    real onset 0.53 s), so its times must not stretch the region."""
    snd = sound_span(x48, SR, (tr.start, tr.end) if tr and tr.words else None, search_pad=0.15)
    if snd:
        return snd
    if tr and tr.words:
        return tr.start, tr.end
    return 0.0, 0.0


def _drop_silent_words(tr: asr.Transcript, x48: np.ndarray) -> asr.Transcript:
    """Remove words that sit on silence (Whisper's "You." in front of a take, "Thank you." on room tone)."""
    if not tr.words:
        return tr
    db, hop = speech_band_db(x48, SR)
    thr = max(float(np.percentile(db, 95)) - 30, -60)
    keep = []
    for w in tr.words:
        seg = db[int(w.start * SR / hop):max(int(w.start * SR / hop) + 1, int(w.end * SR / hop))]
        if len(seg) and seg.max() > thr:
            keep.append(w)
    return asr.Transcript(" ".join(w.w for w in keep), keep)


def pick_and_align(video: Path, takes: list[Path], original_text: str | None = None, *, out_dir: Path,
                   window: tuple[float, float] | None = None, start_hint: float | None = None,
                   settings: ChainSettings | None = None, whisper_model: str | None = None,
                   verify: bool = True) -> dict:
    """Pick the best of ``takes`` (wav / webm / m4a / any ffmpeg-readable file) for the line in ``video`` and mix it in.

    ``original_text``: the line the takes should say (typed by the user). None = speech-to-text of the original line.
    ``window``: the character's speaking window (start, end) in clip seconds. None = found from the Demucs vocal stem,
    speech-to-text and mouth openness; ``start_hint`` (s) picks the line nearest that time when the clip has several.
    ``out_dir`` is keyword-only (it follows a defaulted parameter).
    Returns a JSON-able dict (``RecordLineResult.to_dict``); also written to ``out_dir/recordline.json``.
    Raises RecordLineError (a ValueError) for bad input; optional tools only add warnings.
    """
    s = settings or ChainSettings()
    video, out_dir = Path(video), Path(out_dir)
    takes = [Path(t) for t in (takes or [])]
    if not takes:
        raise RecordLineError("Record or upload at least one take of the line.")
    if len(takes) > MAX_TAKES:
        raise RecordLineError(f"That's {len(takes)} takes; keep it to {MAX_TAKES} or fewer.")
    for t in [video, *takes]:
        if not t.is_file():
            raise RecordLineError(f"Can't find {t.name}.")
    if not has_audio_stream(video):
        raise RecordLineError("This clip has no sound, so there's no original line to replace.")
    out_dir.mkdir(parents=True, exist_ok=True)
    total = media_duration(video)
    warnings: list[str] = []
    have_asr = asr.available()
    if not have_asr:
        warnings.append(HINTS["faster_whisper"])

    # stems + the original line
    st = split_stems(video, out_dir / "stems")
    if st.note:
        warnings.append(st.note)
    line_src = st.vocals or st.clip_audio
    line48 = decode(line_src, SR)
    # the original line: the speech cluster (nearest the hint), transcribed with 0.4 s around it (Whisper invents
    # words in long silences, e.g. "0.0.0." after the Rec5 line)
    cluster = sound_span(line48, SR, hint=start_hint)
    if window is not None:
        cluster = (float(window[0]), float(window[1]))
    orig_tr = None
    if have_asr and cluster:
        a0 = max(0.0, cluster[0] - 0.4)
        a1 = min(len(line48) / SR, cluster[1] + 0.4)
        try:
            orig_tr = asr.transcribe(to16(line48[int(a0 * SR):int(a1 * SR)]), whisper_model).shifted(a0)
        except OptionalDependencyMissing as exc:  # pragma: no cover - raced uninstall
            warnings.append(str(exc))
            have_asr = False

    # speaking window
    if window is not None:
        w0, w1 = float(window[0]), float(window[1])
        if not (0 <= w0 < w1 <= total + 0.05):
            raise RecordLineError(f"The speaking window {w0:.2f}-{w1:.2f}s is outside the clip (0-{total:.2f}s).")
        sw = SpeakingWindow(w0, min(w1, total), "given")
        line_tr = None
        if orig_tr and orig_tr.words:
            inside = [i for i, w in enumerate(orig_tr.words) if w.end > w0 - 0.1 and w.start < w1 + 0.1]
            line_tr = orig_tr.span(inside[0], inside[-1] + 1) if inside else None
    else:
        line_tr, words_span = None, None
        if orig_tr and orig_tr.words:
            loc = asr.locate(orig_tr, original_text) if original_text else None
            line_tr = orig_tr.span(loc[0], loc[1]) if loc else orig_tr
            words_span = (line_tr.start, line_tr.end)
            if loc and loc[2] > 0.5:
                warnings.append("Couldn't find your typed line in the clip's audio, so the window is a best guess. "
                                "Set the start and end times if it looks off.")
        snd = sound_span(line48, SR, words_span) if words_span else cluster
        if snd is None and words_span is None:
            raise RecordLineError("Couldn't hear anyone speaking in this clip. Set the line's start and end times.")
        approx = snd or words_span
        mouth, method = None, None
        try:
            mt = mouth_track(video, *approx)
        except Exception:  # noqa: BLE001 - mouth is a refinement only
            mt = None
        if mt is not None:
            mouth, method = mouth_span(mt[0], mt[1], approx), mt[2]
            if method != "mediapipe":
                warnings.append(HINTS["mediapipe"])
        sw = combine(snd, mouth, words_span, method)
        sw.start, sw.end = max(0.0, sw.start), min(total, sw.end)
        warnings += sw.notes
    target = (original_text or "").strip() or (line_tr.text if line_tr and line_tr.words else None)

    # score every take
    scores: list[scoring.TakeScore] = []
    trs = {}
    for i, path in enumerate(takes):
        tid = f"take{i + 1}"
        label = f"Take {i + 1}"
        try:
            x48 = decode(path, SR)
        except RuntimeError:
            sc = scoring.TakeScore(tid, str(path), warnings=[f"{label} couldn't be read. Try recording it again."])
            sc.score = -1.0
            scores.append(sc)
            continue
        qc = takeqc.analyse(x48, SR, label)
        tr = None
        if have_asr:
            tr = _drop_silent_words(asr.transcribe(to16(x48), whisper_model), x48)
            trs[tid] = tr
        a, b = _speech_region(x48, tr)
        sc = scoring.TakeScore(tid, str(path), speech_start=a, speech_end=b, speech_seconds=max(0.0, b - a),
                               qc={k: qc.get(k) for k in ("clipping", "dropouts", "crackle", "quiet",
                                                          "sample_peak_dbfs", "digital_silence_pct", "snr_db")})
        if tr is not None:
            sc.heard = tr.text
            if target:
                sc.wer = asr.wer(target, tr.text)
            sc.min_word_prob, sc.mean_word_prob = tr.min_prob, tr.mean_prob
            wk = tr.weakest()
            sc.weakest_word = wk.w if wk else None
        sc.duration_fit = sc.speech_seconds / sw.seconds if sw.seconds > 0 and sc.speech_seconds > 0 else None
        scoring.compute(sc)
        sc.warnings = scoring.take_warnings(sc, label)
        if sc.speech_seconds > 0:
            sc.warnings += qc.get("warnings", [])
        if qc.get("silent"):
            sc.score = -1.0
        scores.append(sc)
    ranked = scoring.rank(scores)
    best = ranked[0]
    if best.score < 0 or best.speech_seconds <= 0:
        raise RecordLineError("None of the takes has any speech in it. Check the mic and record again.")
    if len(takes) > 1:
        warnings.append(f"Picked take {best.take_id[4:]} of {len(takes)}.")
    warnings += best.warnings
    if best.wer is not None and best.wer > 0.5:
        warnings.append("Even the best take doesn't say the line clearly. Record it again a bit slower and closer "
                        "to the mic.")
    fits = [t.duration_fit for t in scores if t.duration_fit]
    if len(takes) > 1 and fits and min(fits) > scoring.FIT_HI:
        warnings.append("All takes ran long; the best one is sped up 15% at most.")

    # the chain
    work = out_dir / "chain"
    src, note = denoise(Path(best.path), work / "denoise", s)
    cr = build(src, (best.speech_start, best.speech_end), (sw.start, sw.end), st.clip_audio, st.vocals,
               st.background, work, s)
    steps = [note] + cr.steps
    if cr.trimmed_s > 0.05:
        warnings.append(f"This take is too long to fit, even sped up: trimmed {cr.trimmed_s:.1f}s off the end. "
                        "Try saying it a little faster.")
    if best.duration_fit and best.duration_fit * cr.stretch_ratio < 0.85:
        warnings.append("This take is shorter than the original line, so the mouth will move a little longer than "
                        "your voice. Try saying it a little slower.")
    warnings += cr.room.get("notes", [])

    # speech-to-text check of the aligned line (DubMaster's delivery rule)
    ver = {}
    if verify and have_asr:
        al = decode(cr.aligned_voice_wav, SR)
        a0, a1 = max(0.0, sw.start - 0.4), min(total, max(sw.end, cr.voice_end) + 0.4)
        vtr = asr.transcribe(to16(al[int(a0 * SR):int(a1 * SR)]), whisper_model)
        ver = {"heard": vtr.text, "wer": round(asr.wer(target, vtr.text), 3) if target else None,
               "min_word_prob": vtr.min_prob}
        if ver["wer"] is not None and ver["wer"] > max(0.2, (best.wer or 0) + 0.15):
            warnings.append("After fitting, the line is harder to understand than your raw take. Try a take closer "
                            "to the original length.")

    from .deps import status

    res = RecordLineResult(
        chosen_take=best.take_id, chosen_take_path=best.path, takes=[t.to_dict() for t in ranked],
        window_start=sw.start, window_end=sw.end, window=sw.to_dict(), target_text=target,
        stretch_ratio=cr.stretch_ratio, aligned_voice_wav=str(cr.aligned_voice_wav),
        infinitetalk_wav=str(cr.infinitetalk_wav), infinitetalk_clip_t0=cr.infinitetalk_t0,
        final_mix_wav=str(cr.final_mix_wav), voice_wet_wav=str(cr.voice_wet_wav), voice_start=cr.voice_start,
        voice_end=cr.voice_end, warnings=list(dict.fromkeys(warnings)), steps=steps, room=cr.room, levels=cr.levels,
        verify=ver, available=status())
    d = res.to_dict()
    (out_dir / "recordline.json").write_text(json.dumps(d, indent=2), encoding="utf-8")
    return d


def mux(video: Path, final_mix_wav: Path, out: Path) -> Path:
    """Video stream-copied + the new soundtrack as AAC 320 kb/s 48 kHz (same encode as redub), clip length kept."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(["-i", str(video), "-i", str(final_mix_wav), "-map", "0:v:0?", "-map", "1:a", "-c:v", "copy",
                "-c:a", "aac", "-b:a", "320k", "-ar", str(SR), "-t", f"{media_duration(video):.3f}",
                "-movflags", "+faststart", str(out)])
    return out


def process_line(video: Path, line_wav: Path, *, start: float, work: Path, settings: dict | None = None) -> dict:
    """The Re-dub tab's entry point (via recordline_bridge). One take in ``line_wav``, or several in
    ``settings["takes"]`` (original files: browser webm / wav uploads at full quality). ``start`` is the speaking-window
    start hint; ``settings["fit_seconds"]`` (with start) fixes the window; ``settings["window"]`` = (start, end) too.
    Other settings: ``text`` / ``original_text`` (the line), ``denoise``, ``duck`` (full-mix duck when there are no
    stems), ``voice_level_db``.

    Returns pick_and_align's dict plus ``output`` (the clip with the new soundtrack, mp4: extras hands it to
    _finish as the video) and ``steps`` (plain-English warnings first, then what was done)."""
    st = dict(settings or {})
    takes = [Path(t) for t in (st.get("takes") or []) if t] or [Path(line_wav)]
    cs = ChainSettings()
    if st.get("denoise") in ("auto", "deepfilternet", "off"):
        cs.denoise = st["denoise"]
    if st.get("duck") is not None:
        cs.duck_mix = float(st["duck"])
    cs.voice_level_db = float(st.get("voice_level_db") or 0.0)
    win = st.get("window")
    if win is None and st.get("fit_seconds"):
        win = (float(start), float(start) + float(st["fit_seconds"]))
    work = Path(work)
    r = pick_and_align(Path(video), takes, st.get("original_text") or st.get("text") or None, out_dir=work,
                       window=tuple(win) if win else None, start_hint=float(start) if start is not None else None,
                       settings=cs, whisper_model=st.get("whisper_model"))
    r["output"] = str(mux(Path(video), Path(r["final_mix_wav"]), work / "redubbed.mp4"))
    r["line_seconds"] = round(r["voice_end"] - r["voice_start"], 3)
    r["steps"] = list(r["warnings"]) + list(r["steps"])
    (work / "recordline.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
    return r
