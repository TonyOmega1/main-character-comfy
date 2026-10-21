"""Speech-to-text for record-your-own-line: faster-whisper (optional, CPU int8) on numpy audio, word timestamps,
word error rate and "find this line in a longer transcript".

faster-whisper is always given a float32 16 kHz numpy array decoded by ffmpeg, never a file path: its own decoder
needs a PyAV version that clashes with other packages, and numpy input skips it entirely.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field

import numpy as np

from .deps import require

DEFAULT_MODEL = "small.en"  # what the Rec5 / Val Venis dubs were checked with; MC_WHISPER_MODEL overrides
_MODELS: dict = {}
PADS = (0.0, 0.3, 1.0)  # silence around the audio per decode; small.en on short takes flips between readings
NO_SPEECH = (0.6, None)  # Whisper's default no-speech skip, and none


@dataclass
class Word:
    w: str
    start: float
    end: float
    p: float


@dataclass
class Transcript:
    text: str = ""
    words: list = field(default_factory=list)  # list[Word]

    @property
    def start(self) -> float | None:
        return self.words[0].start if self.words else None

    @property
    def end(self) -> float | None:
        return self.words[-1].end if self.words else None

    @property
    def min_prob(self) -> float | None:
        return min(w.p for w in self.words) if self.words else None

    @property
    def mean_prob(self) -> float | None:
        return float(np.mean([w.p for w in self.words])) if self.words else None

    def weakest(self) -> Word | None:
        return min(self.words, key=lambda w: w.p) if self.words else None

    def shifted(self, dt: float) -> "Transcript":
        return Transcript(self.text, [Word(w.w, round(w.start + dt, 3), round(w.end + dt, 3), w.p) for w in self.words])

    def span(self, i0: int, i1: int) -> "Transcript":
        ws = self.words[i0:i1]
        return Transcript(" ".join(w.w for w in ws), list(ws))

    def to_dict(self) -> dict:
        return {"text": self.text, "words": [asdict(w) for w in self.words]}


def available() -> bool:
    from .deps import has_module

    return has_module("faster_whisper")


def _model(name: str):
    if name not in _MODELS:
        from faster_whisper import WhisperModel

        _MODELS[name] = WhisperModel(name, device="cpu", compute_type="int8")
    return _MODELS[name]


def _decode_once(m, a: np.ndarray, language: str, no_speech: float | None = 0.6) -> Transcript:
    segs, _ = m.transcribe(a, word_timestamps=True, language=language, vad_filter=False, beam_size=5,
                           temperature=0.0, condition_on_previous_text=False, no_speech_threshold=no_speech)
    words, txt = [], []
    for s in segs:
        txt.append(s.text.strip())
        for w in s.words or []:
            words.append(Word(w.word.strip(), round(float(w.start), 3), round(float(w.end), 3),
                              round(float(w.probability), 3)))
    return Transcript(" ".join(t for t in txt if t), words)


def _clean(tr: Transcript, pad: float, dur: float) -> Transcript:
    """Shift back by ``pad``, drop hallucinated tokens ("0", ".0", words inside the silent padding)."""
    keep = []
    for w in tr.words:
        if not re.search(r"[A-Za-z]", w.w) or w.p < 0.02:
            continue
        if w.end <= pad + 0.02 or w.start >= pad + dur - 0.02:
            continue
        keep.append(Word(w.w, round(max(0.0, w.start - pad), 3), round(min(dur, w.end - pad), 3), w.p))
    return Transcript(" ".join(w.w for w in keep), keep)


def hallucinated(tr: Transcript) -> bool:
    """Whisper's failure modes on short clips: one word looping ("and and and ...", "that that ..."), or a stretched
    token ("Sssssss...")."""
    ws = [w for w in (norm_words(tr.text)) if w]
    if not ws:
        return False
    if any(re.search(r"(.)\1{4,}", w) for w in ws):
        return True
    run = best = 1
    for a, b in zip(ws, ws[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    if best >= 3:  # the same word 3+ times in a row
        return True
    return len(ws) >= 6 and max(ws.count(w) for w in set(ws)) / len(ws) > 0.5


def consensus(readings: list[Transcript]) -> Transcript:
    """The medoid reading: the one closest (summed WER) to all the others; ties go to the more confident one."""
    ok = [r for r in readings if r.words and not hallucinated(r)]
    if not ok:
        return Transcript()
    def dist(a: Transcript, b: Transcript) -> float:  # symmetric: edits / the longer reading
        x, y = norm_words(a.text), norm_words(b.text)
        return wer(x, y) * len(x) / max(len(x), len(y), 1)

    best = min(ok, key=lambda r: (sum(dist(o, r) for o in ok if o is not r), -(r.mean_prob or 0)))
    return best


def transcribe(audio16: np.ndarray, model: str | None = None, language: str = "en") -> Transcript:
    """Words with timestamps (s, relative to the array start) and probabilities. Raises OptionalDependencyMissing.

    small.en (int8, CPU) on a 1-3 s take is unstable: the same audio can come back complete, truncated, garbled or as
    a looping hallucination depending on the silence around it and the no-speech skip (measured on the Rec5 takes).
    So: temperature 0 (no random sampling fallback), one decode per pad in PADS and per no-speech setting, looping
    readings dropped, and the consensus (medoid) reading returned. ~6 decodes per call; MC_WHISPER_MODEL overrides."""
    require("faster_whisper")
    a = np.ascontiguousarray(audio16, np.float32)
    if len(a) < 1600 or not np.any(np.abs(a) > 1e-4):
        return Transcript()
    m = _model(model or os.environ.get("MC_WHISPER_MODEL", "").strip() or DEFAULT_MODEL)
    dur = len(a) / 16000
    readings = []
    for no_speech in NO_SPEECH:
        for pad in PADS:
            p = np.zeros(int(pad * 16000), np.float32)
            readings.append(_clean(_decode_once(m, np.concatenate([p, a, p]), language, no_speech), pad, dur))
    return consensus(readings)


_NUM = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven",
        "8": "eight", "9": "nine", "10": "ten"}


def norm_words(text: str) -> list[str]:
    t = text.lower().replace("\u2019", "'")
    t = re.sub(r"[^a-z0-9' ]", " ", t)
    out = []
    for w in t.split():
        w = w.strip("'")
        if w:
            out.append(_NUM.get(w, w))
    return out


def wer(ref: str | list[str], hyp: str | list[str]) -> float:
    """Word error rate (Levenshtein over normalised words). Empty reference: 0 if hyp empty, else 1."""
    r = norm_words(ref) if isinstance(ref, str) else ref
    h = norm_words(hyp) if isinstance(hyp, str) else hyp
    if not r:
        return 0.0 if not h else 1.0
    d = np.arange(len(h) + 1, dtype=float)
    for i in range(1, len(r) + 1):
        prev, d[0] = d.copy(), i
        for j in range(1, len(h) + 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
    return float(d[-1] / len(r))


def locate(tr: Transcript, text: str) -> tuple[int, int, float] | None:
    """Best-matching word span [i0, i1) of ``text`` inside ``tr`` (lowest WER, then the longest). None if no words."""
    ref = norm_words(text)
    if not tr.words or not ref:
        return None
    hw = [(norm_words(w.w) or [""])[0] for w in tr.words]
    best = None
    n = len(ref)
    for i0 in range(len(hw)):
        for i1 in range(i0 + 1, min(len(hw), i0 + 2 * n + 2) + 1):
            e = wer(ref, hw[i0:i1])
            key = (e, -(i1 - i0))
            if best is None or key < best[0]:
                best = (key, i0, i1)
    return best[1], best[2], best[0][0]
