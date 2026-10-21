"""Synthetic audio for the record-your-own-line tests (no models, no fixtures on disk)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from main_character_comfy import redub
from main_character_comfy.media import run_ffmpeg

SR = 48000


def voice(seconds: float, words: int = 4, f0: float = 140.0, seed: int = 0, level: float = 0.25) -> np.ndarray:
    """Speech-like: a harmonic buzz through two formant-ish bumps, ``words`` syllable bursts with short gaps."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    vib = 1 + 0.02 * np.sin(2 * np.pi * 5 * t)
    ph = 2 * np.pi * np.cumsum(f0 * vib) / SR
    x = sum((1 / k) * np.sin(k * ph) * (1.6 if k in (4, 5, 12, 13) else 1.0) for k in range(1, 25))
    env = np.zeros(n)
    edges = np.linspace(0, n, words + 1).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        g = int(0.06 * SR)  # gap between words
        L = max(1, b - a - g)
        env[a:a + L] = np.sin(np.pi * np.arange(L) / L) ** 0.6
    x = x * env
    x += 0.002 * rng.standard_normal(n) * (env > 0)
    return level * x / (np.abs(x).max() + 1e-12)


def music(seconds: float, level: float = 0.05, seed: int = 1) -> np.ndarray:
    n = int(seconds * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(seed)
    m = sum(np.sin(2 * np.pi * f * t) for f in (110, 165, 220, 277))
    m = m / 4 + 0.15 * rng.standard_normal(n) * (np.sin(2 * np.pi * 2 * t) > 0.9)  # chord + hats
    return level * np.stack([m, np.roll(m, 37)], 1)


def take(path: Path, speech_s: float, lead: float = 0.4, tail: float = 0.4, noise: float = 0.0008, seed: int = 0,
         **kw) -> Path:
    rng = np.random.default_rng(seed + 10)
    v = np.concatenate([np.zeros(int(lead * SR)), voice(speech_s, seed=seed, **kw), np.zeros(int(tail * SR))])
    v += noise * rng.standard_normal(len(v))  # room tone (a real take is never digital silence)
    return redub.save(path, v)


def clip(dir_: Path, seconds: float = 4.0, line=(1.0, 2.4)) -> dict:
    """A test video whose audio = music + an original line in ``line``; also writes the true stems."""
    dir_.mkdir(parents=True, exist_ok=True)
    n = int(seconds * SR)
    v = np.zeros(n)
    a = int(line[0] * SR)
    seg = voice(line[1] - line[0], seed=5, f0=120)
    v[a:a + len(seg)] = seg
    m = music(seconds)
    mix = m + v[:, None] * 0.8
    redub.save(dir_ / "mix.wav", mix)
    redub.save(dir_ / "true_vocals.wav", np.stack([v, v], 1) * 0.8)
    redub.save(dir_ / "true_no_vocals.wav", m)
    vid = dir_ / "clip.mp4"
    run_ffmpeg(["-f", "lavfi", "-i", f"testsrc2=size=320x240:rate=24:duration={seconds}", "-i", str(dir_ / "mix.wav"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "pcm_s24le", "-shortest", str(dir_ / "clip.mov")])
    run_ffmpeg(["-i", str(dir_ / "clip.mov"), "-c:v", "copy", "-c:a", "aac", "-b:a", "320k", str(vid)])
    return {"video": vid, "vocals": dir_ / "true_vocals.wav", "no_vocals": dir_ / "true_no_vocals.wav",
            "line": line, "seconds": seconds}
