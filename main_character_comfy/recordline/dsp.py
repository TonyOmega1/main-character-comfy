"""Small numpy-only DSP helpers for record-your-own-line (MC has no scipy requirement, so nothing here needs it).

Decoding, resampling, filters and metering run through ffmpeg (the same static build MC uses everywhere); loudness and
true peak reuse :mod:`redub`.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from ..media import ffmpeg_exe

SR = 48000
ASR_SR = 16000


def decode(path: str | Path, sr: int = SR, channels: int = 1) -> np.ndarray:
    """Any audio/video file (wav, webm/opus, m4a, mp4...) -> float64 at ``sr``. (n,) mono or (n, ch)."""
    proc = subprocess.run([ffmpeg_exe(), "-v", "error", "-i", str(path), "-vn", "-ac", str(channels), "-ar", str(sr),
                           "-f", "f32le", "-"], capture_output=True)
    if proc.returncode:
        raise RuntimeError(f"couldn't read the audio in {Path(path).name}: {proc.stderr.decode(errors='replace')[-300:]}")
    a = np.frombuffer(proc.stdout, np.float32).astype(np.float64)
    return a.reshape(-1, channels) if channels > 1 else a


def decode16(path: str | Path) -> np.ndarray:
    """Mono float32 16 kHz numpy audio for faster-whisper (pass numpy, never the file: its PyAV pin clashes)."""
    return decode(path, ASR_SR, 1).astype(np.float32)


def to16(x: np.ndarray, sr: int = SR) -> np.ndarray:
    """Mono float -> float32 16 kHz via ffmpeg's resampler (what Whisper reads best; an FFT resample's edge ringing
    made it drop or invent words on short takes)."""
    if len(x) == 0:
        return np.zeros(0, np.float32)
    proc = subprocess.run([ffmpeg_exe(), "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-",
                           "-ar", str(ASR_SR), "-f", "f32le", "-"], input=np.ascontiguousarray(x, np.float32).tobytes(),
                          capture_output=True)
    if proc.returncode:
        raise RuntimeError(proc.stderr.decode(errors="replace")[-300:])
    return np.frombuffer(proc.stdout, np.float32).copy()


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """FFT resample (band-limited). Fine for the short segments used here."""
    if sr_in == sr_out or len(x) == 0:
        return x.copy()
    n = int(round(len(x) * sr_out / sr_in))
    spec = np.fft.rfft(x, axis=0)
    m = n // 2 + 1
    out = np.zeros((m,) + spec.shape[1:], complex)
    k = min(m, spec.shape[0])
    out[:k] = spec[:k]
    return np.fft.irfft(out, n, axis=0) * (n / len(x))


def butter_mag(f: np.ndarray, lo: float | None = None, hi: float | None = None, order: int = 4) -> np.ndarray:
    """Magnitude response of a Butterworth high/low/band-pass at frequencies ``f``."""
    g = np.ones_like(f, dtype=float)
    fz = np.maximum(f, 1e-6)
    if lo:
        g /= np.sqrt(1 + (lo / fz) ** (2 * order))
    if hi:
        g /= np.sqrt(1 + (fz / hi) ** (2 * order))
    return g


def fft_filter(x: np.ndarray, sr: int, lo: float | None = None, hi: float | None = None, order: int = 4,
               zero_phase: bool = True) -> np.ndarray:
    """Zero-phase Butterworth-shaped filter in the frequency domain (like scipy's sosfiltfilt: magnitude squared)."""
    n = len(x)
    if n == 0:
        return x.copy()
    size = 1 << int(np.ceil(np.log2(n + 1024)))
    f = np.fft.rfftfreq(size, 1 / sr)
    g = butter_mag(f, lo, hi, order)
    if zero_phase:
        g = g ** 2
    spec = np.fft.rfft(x, size, axis=0)
    shape = g if x.ndim == 1 else g[:, None]
    return np.fft.irfft(spec * shape, size, axis=0)[:n]


def frame_db(x: np.ndarray, sr: int, hop: float = 0.01) -> tuple[np.ndarray, int]:
    """Per-frame RMS level in dBFS and the frame length in samples."""
    n = max(1, int(hop * sr))
    k = len(x) // n
    if k == 0:
        return np.array([-200.0]), n
    r = np.sqrt(np.mean(x[: k * n].reshape(k, n) ** 2, 1) + 1e-20)
    return 20 * np.log10(r), n


def speech_band_db(x: np.ndarray, sr: int, hop: float = 0.01) -> tuple[np.ndarray, int]:
    """Frame levels of the 300 Hz - 4 kHz band (where speech lives; ignores rumble and hiss)."""
    return frame_db(fft_filter(x, sr, 300, 4000), sr, hop)


def runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(start, length) of True runs."""
    m = np.concatenate([[0], np.asarray(mask, np.int8), [0]])
    d = np.diff(m)
    s = np.where(d == 1)[0]
    e = np.where(d == -1)[0]
    return s, e - s


def sliding_min(x: np.ndarray, w: int) -> np.ndarray:
    w = max(1, int(w)) | 1
    p = np.pad(x, w // 2, mode="edge")
    return np.lib.stride_tricks.sliding_window_view(p, w).min(1)


def smooth(x: np.ndarray, w: int) -> np.ndarray:
    w = max(1, int(w)) | 1
    hw = np.hanning(w + 2)[1:-1]
    hw /= hw.sum()
    return np.convolve(np.pad(x, w, mode="edge"), hw, "same")[w:-w]


def oversampled_peak(x: np.ndarray, factor: int = 4) -> np.ndarray:
    """Per-sample |peak| of the ``factor``x oversampled signal (inter-sample peaks), max over channels."""
    xs = x if x.ndim == 2 else x[:, None]
    n = xs.shape[0]
    up = np.fft.irfft(np.fft.rfft(xs, axis=0), n * factor, axis=0) * factor
    return np.abs(up).max(1)[: n * factor].reshape(n, factor).max(1)


def slow_peak_control(x: np.ndarray, sr: int, max_cut_db: float = 4.0, crest_db: float = 12.0,
                      lookahead: float = 0.006, release: float = 0.080) -> tuple[np.ndarray, float]:
    """The approved chain's slow, clean peak control: only peaks more than ``crest_db`` above the active speech RMS
    are pulled down, never by more than ``max_cut_db``; smooth look-ahead gain with an 80 ms release. Not a limiter:
    no hard ceiling, no fast attack, so syllables don't buzz. Returns (out, max reduction dB)."""
    if not np.any(np.abs(x) > 1e-9):
        return x.copy(), 0.0
    pk = oversampled_peak(x)
    mono = x if x.ndim == 1 else x.mean(1)
    db, hop = frame_db(mono, sr, 0.02)
    act = db > max(np.percentile(db, 95) - 30, -70)
    rms_db = 10 * np.log10(np.mean(10 ** (db[act] / 10))) if act.any() else -70.0
    ceil = max(10 ** ((rms_db + crest_db) / 20), pk.max() * 10 ** (-max_cut_db / 20))
    g = np.minimum(1.0, ceil / np.maximum(pk, 1e-12))
    if g.min() >= 1.0:
        return x.copy(), 0.0
    w = int(lookahead * sr) | 1
    g = sliding_min(g, w)
    g = np.minimum(g, smooth(g, w))
    g = smooth(sliding_min(g, w), w)
    a = np.exp(-1 / (release * sr))
    o = g.copy()
    for i in range(1, len(o)):  # release: recover slowly
        o[i] = min(g[i], o[i - 1] * a + (1 - a))
    out = x * (o if x.ndim == 1 else o[:, None])
    return out, float(-20 * np.log10(max(o.min(), 1e-12)))


def fades(x: np.ndarray, sr: int, fade_in: float = 0.010, fade_out: float = 0.050) -> np.ndarray:
    y = x.copy()
    fi, fo = min(len(y), int(fade_in * sr)), min(len(y), int(fade_out * sr))
    if fi:
        y[:fi] *= np.linspace(0, 1, fi)
    if fo:
        y[len(y) - fo:] *= np.linspace(1, 0, fo)
    return y
