"""Take scoring: does it say the line (WER), is every word clear (minimum word probability), does it fit the
character's speaking window (duration fit 0.85-1.15), and is the recording itself clean (takeqc)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

FIT_LO, FIT_HI = 0.85, 1.15  # take length / window length that a +-15 % stretch can absorb
WER_OK = 0.2
CLARITY_OK = 0.5
WEIGHTS = {"words": 0.55, "clarity": 0.25, "fit": 0.20}
QC_PENALTY = {"clipping": 0.15, "dropouts": 0.10, "crackle": 0.05, "quiet": 0.05}


@dataclass
class TakeScore:
    take_id: str
    path: str
    heard: str | None = None          # what speech-to-text heard (None = no ASR)
    wer: float | None = None          # vs the target line
    min_word_prob: float | None = None
    mean_word_prob: float | None = None
    weakest_word: str | None = None
    speech_start: float = 0.0         # inside the take (s)
    speech_end: float = 0.0
    speech_seconds: float = 0.0
    duration_fit: float | None = None  # speech_seconds / window seconds (1.0 = exact, 0.85-1.15 = fits)
    qc: dict = field(default_factory=dict)
    score: float = 0.0
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("wer", "min_word_prob", "mean_word_prob", "duration_fit", "score", "speech_start", "speech_end",
                  "speech_seconds"):
            if d[k] is not None:
                d[k] = round(float(d[k]), 3)
        return d


def fit_score(fit: float | None) -> float:
    if fit is None:
        return 0.0
    if FIT_LO <= fit <= FIT_HI:
        return 1.0
    dist = (FIT_LO - fit) if fit < FIT_LO else (fit - FIT_HI)
    return max(0.0, 1.0 - dist / 0.30)


def qc_penalty(qc: dict) -> float:
    return sum(QC_PENALTY[k] for k in QC_PENALTY if qc.get(k))


def compute(t: TakeScore) -> TakeScore:
    words = 1.0 - min(t.wer, 1.0) if t.wer is not None else 0.5
    clarity = t.min_word_prob if t.min_word_prob is not None else 0.5
    t.score = (WEIGHTS["words"] * words + WEIGHTS["clarity"] * clarity + WEIGHTS["fit"] * fit_score(t.duration_fit)
               - qc_penalty(t.qc))
    return t


def take_warnings(t: TakeScore, label: str) -> list[str]:
    w = []
    if t.speech_seconds <= 0:
        return [f"{label} sounds silent. Check that the right mic was selected."]
    if t.wer is not None and t.wer > WER_OK:
        w.append(f"{label} doesn't quite match the line (heard: \"{t.heard or '...'}\").")
    if t.min_word_prob is not None and t.min_word_prob < CLARITY_OK:
        word = f" on \"{t.weakest_word}\"" if t.weakest_word else ""
        w.append(f"Low clarity{word} in {label.lower()}, try recording closer to the mic.")
    if t.duration_fit is not None:
        if t.duration_fit > FIT_HI:
            w.append(f"{label} runs long ({t.duration_fit:.2f}x the original line), so it's sped up by at most 15%.")
        elif t.duration_fit < FIT_LO:
            w.append(f"{label} is short ({t.duration_fit:.2f}x the original line), so it's slowed by at most 15%.")
    return w


def rank(scores: list[TakeScore]) -> list[TakeScore]:
    """Best first: score, then the clearer take, then the better fit."""
    return sorted(scores, key=lambda t: (-round(t.score, 6), -(t.mean_word_prob or 0),
                                         abs((t.duration_fit or 9) - 1)))
