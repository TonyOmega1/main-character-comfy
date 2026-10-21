"""Two-character dialogue: parse a skit script and build per-speaker tracks.

Script format (one line per turn; the first two distinct names become A and B)::

    Maya: Did you see the new update?
    Theo: I did. I'm still recovering.

Each turn is voiced separately (Chatterbox, cloning that speaker's sample), then
laid end to end. InfiniteTalk's two-speaker mode needs one track per speaker
that is silent while the other one talks, plus the mixed track for the video.
"""

from __future__ import annotations

import re
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .media import Caption, split_caption_chunks, to_wav

SR = 24000
_LINE = re.compile(r"^\s*([^:]{1,24}):\s*(.+?)\s*$")


@dataclass
class Turn:
    speaker: str  # "A" or "B"
    name: str
    text: str


def parse_dialogue(script: str) -> tuple[list[Turn], dict[str, str]]:
    names: dict[str, str] = {}
    turns: list[Turn] = []
    for raw in script.splitlines():
        if not raw.strip():
            continue
        m = _LINE.match(raw)
        if not m:
            raise ValueError(f"Dialogue lines look like 'Name: what they say'. Can't read: {raw.strip()!r}")
        name, text = m.group(1).strip(), m.group(2)
        if name not in names:
            if len(names) == 2:
                raise ValueError("Dialogue mode supports two characters.")
            names[name] = "A" if not names else "B"
        turns.append(Turn(names[name], name, text))
    if len(names) < 2:
        raise ValueError("Dialogue mode needs lines from two different characters.")
    return turns, {v: k for k, v in names.items()}


def _read(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)


def _write(path: Path, data: np.ndarray) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(np.clip(data, -32768, 32767).astype(np.int16).tobytes())
    return path


def build_tracks(turns: list[Turn], clips: list[Path], work: Path, gap: float = 0.25) -> dict:
    """Lay out voiced turns; returns paths of A, B and mix tracks plus captions."""
    work.mkdir(parents=True, exist_ok=True)
    pieces = [_read(to_wav(c, work / f"turn_{i:02d}.wav")) for i, c in enumerate(clips)]
    gap_n = int(gap * SR)
    total = sum(len(p) for p in pieces) + gap_n * (len(pieces) + 1)
    a = np.zeros(total, np.float32)
    b = np.zeros(total, np.float32)
    caps: list[Caption] = []
    pos = gap_n
    for turn, pcm in zip(turns, pieces):
        (a if turn.speaker == "A" else b)[pos : pos + len(pcm)] = pcm
        start, end = pos / SR, (pos + len(pcm)) / SR
        chunks = split_caption_chunks(turn.text)
        weights = [len(c) for c in chunks]
        t = start
        for ch, wt in zip(chunks, weights):
            span = (end - start) * wt / sum(weights)
            caps.append(Caption(ch, round(t, 3), round(t + span, 3), speaker=turn.speaker))
            t += span
        pos += len(pcm) + gap_n
    return {
        "a": _write(work / "speech_a.wav", a),
        "b": _write(work / "speech_b.wav", b),
        "mix": _write(work / "speech_mix.wav", a + b),
        "captions": caps,
        "duration": total / SR,
    }


def concat_audio(clips: list[Path], dst: Path, work: Path, gap: float = 0.2) -> Path:
    """Join TTS passages with a short pause between them."""
    work.mkdir(parents=True, exist_ok=True)
    gap_pcm = np.zeros(int(gap * SR), np.float32)
    parts: list[np.ndarray] = []
    for i, c in enumerate(clips):
        if parts:
            parts.append(gap_pcm)
        parts.append(_read(to_wav(c, work / f"part_{i:02d}.wav")))
    return _write(dst, np.concatenate(parts))
