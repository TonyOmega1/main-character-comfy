"""Make a dry take sit in the scene (DubMaster's study tool roommatch.py, ported to numpy only).

Blind RT60 from free speech decays in the ORIGINAL line (vocal stem), wetness matched on the late-to-early energy
ratio after speech offsets (grid search, 0 allowed for a dry voice-over), and a gentle 1/3-octave smoothed EQ match
toward the original line (30 %, +/-3 dB, boosts <= +2 dB, none above 12 kHz). No fixed preset: every value comes from
the clip, with the approved defaults (0.45 s, 15 %) only when the reference has too few decays to measure.
No loudnorm, limiter, saturation, exciter or synthetic room tone here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..redub import fft_convolve, spectrum
from .dsp import SR, butter_mag, fft_filter

DEFAULT_RT, DEFAULT_WET, MAX_WET = 0.45, 0.15, 0.25


def env_db(x: np.ndarray, hop: float = 0.005, lo: float = 300, hi: float = 4000) -> tuple[np.ndarray, float]:
    y = fft_filter(x, SR, lo, hi)
    n = int(hop * SR)
    k = len(y) // n
    if k < 3:
        return np.full(3, -200.0), hop
    e = np.mean(y[: k * n].reshape(k, n) ** 2, 1)
    e = np.convolve(e, np.ones(3) / 3, "same")
    return 10 * np.log10(e + 1e-20), hop


def decays(x: np.ndarray) -> tuple[list, float]:
    """Free-decay segments after speech offsets: list of (rt60, start_frame, end_frame)."""
    d, hop = env_db(x, 0.005)
    d = np.convolve(d, np.ones(4) / 4, "same")
    top = np.percentile(d, 95)
    floor = max(np.percentile(d, 5), top - 65)
    out = []
    i, W = 10, int(0.05 / hop)
    while i < len(d) - 6:
        if d[i] == d[max(0, i - W):i + W].max() and d[i] > top - 25 and d[i] > floor + 20:
            j, rmin = i, d[i]
            while j + 1 < len(d) and d[j + 1] < rmin + 2.0 and d[j + 1] > floor + 3 and (j - i) * hop < 1.5:
                j += 1
                rmin = min(rmin, d[j])
            j = i + int(np.argmin(d[i:j + 1]))
            drop = d[i] - d[j]
            if drop >= 10:
                seg = d[i:j + 1]
                a = int(np.argmax(seg <= d[i] - 5))
                b = int(np.argmax(seg <= d[i] - min(drop, 35) + 1))
                if b - a >= max(3, int(0.03 / hop)):
                    t = np.arange(a, b + 1) * hop
                    sl = np.polyfit(t, seg[a:b + 1], 1)[0]
                    if sl < 0:
                        out.append((min(60 / -sl, 3.0), i, j))
            i = max(j, i + W)
        else:
            i += 1
    return out, hop


def offsets(x: np.ndarray, drop_db: float = 15.0) -> tuple[list, float]:
    """Speech offsets (sharp or slow): a local level peak near the top that falls >= ``drop_db`` within 80 ms.
    Same (rt, start, end) tuples as :func:`decays` (rt unused), so both feed :func:`late_ratio`. Works on dry
    takes too, where :func:`decays` finds nothing (their decays are too fast to fit a slope)."""
    d, hop = env_db(x, 0.005)
    top = np.percentile(d, 95)
    out, i, W, L = [], 10, int(0.05 / hop), int(0.08 / hop)
    while i < len(d) - L:
        if d[i] == d[max(0, i - W):i + W].max() and d[i] > top - 25:
            j = i + int(np.argmax(d[i:i + L] < d[i] - drop_db)) if (d[i:i + L] < d[i] - drop_db).any() else -1
            if j > 0:
                k = i + int(np.argmax(d[i:j + 1] < d[i] - 3))  # where the fall starts
                out.append((0.0, k, j))
                i = j + W
                continue
        i += 1
    return out, hop


def late_ratio(x: np.ndarray, offs: list, hop: float) -> float | None:
    r = []
    for _, i, _ in offs:
        c = i * int(hop * SR)
        pre, post = x[max(0, c - int(.04 * SR)):c], x[c + int(.03 * SR):c + int(.2 * SR)]
        if len(pre) > 10 and len(post) > 10:  # stop before the next syllable starts (back within 6 dB of pre)
            n5 = int(0.005 * SR)
            k = len(post) // n5
            if k:
                fe = np.mean(post[: k * n5].reshape(k, n5) ** 2, 1)
                hot = np.where(fe > 0.25 * np.mean(pre ** 2))[0]
                if len(hot) and hot[0] >= 4:
                    post = post[: hot[0] * n5]
                elif len(hot):
                    continue
            r.append(10 * np.log10(np.mean(post ** 2) + 1e-20) - 10 * np.log10(np.mean(pre ** 2) + 1e-20))
    return float(np.median(r)) if r else None


def rt_estimate(x: np.ndarray) -> tuple[float | None, int, list, float]:
    offs, hop = decays(x)
    rts = np.array([o[0] for o in offs])
    if len(rts) == 0:
        return None, 0, offs, hop
    est = float(np.percentile(rts, 25)) if len(rts) >= 4 else float(rts.min())
    return est * 1.10, len(rts), offs, hop  # 25th pct under-reads simulated rooms by ~10 %


def ltas(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d, _ = env_db(x, 0.02, 100, 8000)
    n = int(0.02 * SR)
    act = d > np.percentile(d, 95) - 30
    m = np.repeat(act, n)[: len(x)]
    xa = x[: len(m)][m]
    return spectrum(xa if len(xa) > 4096 else x)


def third_oct_smooth(f: np.ndarray, g: np.ndarray) -> np.ndarray:
    return np.array([np.mean(g[(f > x / 2 ** (1 / 6)) & (f < x * 2 ** (1 / 6))]) if x > 20 else 0 for x in f])


def eq_curve(ref: np.ndarray, take: np.ndarray, match: float = 0.30, clamp: float = 3.0,
             cap: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    fr, pr = ltas(ref)
    _, pt = ltas(take)
    dd = 10 * np.log10(pr + 1e-20) - 10 * np.log10(pt + 1e-20)
    band = (fr > 150) & (fr < 12000)
    dd -= np.average(dd[band], weights=1 / fr[band])
    g = np.clip(third_oct_smooth(fr, dd) * match, -clamp, clamp)
    g[~band] = np.interp(fr[~band], fr[band][[0, -1]], g[band][[0, -1]])
    g = np.minimum(g, cap)
    g = np.where(g > 0, g * np.clip((12000 - fr) / 4000, 0, 1), g)
    return fr, g


def apply_eq(x: np.ndarray, g: np.ndarray) -> np.ndarray:
    h = np.fft.irfft(10 ** (g / 20))
    h = np.roll(h, len(h) // 2) * np.hanning(len(h))
    dl = len(h) // 2
    return fft_convolve(x, h)[dl:dl + len(x)]


def make_ir(rt: float, pre: float = 0.012, seed: int = 7, hf_ratio: float = 0.65) -> np.ndarray:
    """Stereo IR: early reflections + a noise tail whose highs decay faster (like real rooms)."""
    rng = np.random.default_rng(seed)
    n = int(SR * (rt * 1.2 + pre))
    t = np.arange(n) / SR
    ir = np.zeros((n, 2))
    for de, gg in [(0.006, .45), (0.011, .33), (0.017, .26), (0.024, .19), (0.033, .13)]:
        sign = 1 if rng.random() > .5 else -1
        for ch in (0, 1):
            ir[min(n - 1, int((pre + de + 0.0013 * ch) * SR)), ch] += gg * sign
    f = np.fft.rfftfreq(n, 1 / SR)
    tail = np.zeros((n, 2))
    for ch in (0, 1):
        spec = np.fft.rfft(rng.standard_normal(n))
        for (lo, hi), k in (((None, 1000), 1.0), ((1000, 4000), 0.85), ((4000, None), hf_ratio)):
            band = np.fft.irfft(spec * butter_mag(f, lo, hi, 4), n)
            tail[:, ch] += band * np.exp(-6.9 * np.clip(t - pre - 0.025, 0, None) / (rt * k))
    tail[: int((pre + 0.010) * SR)] = 0
    tail = fft_filter(tail, SR, 200, 7000, order=2, zero_phase=False) * 0.07
    ir += tail
    return ir / np.sqrt((ir ** 2).sum(0)).max()


def reverb(x: np.ndarray, ir: np.ndarray, wet: float) -> np.ndarray:
    """Mono in, stereo out: dry * (1 - wet) + loudness-matched wet * wet."""
    w = np.stack([fft_convolve(x, ir[:, c])[: len(x)] for c in (0, 1)], 1)
    w *= np.sqrt((x ** 2).mean()) / (np.sqrt((w ** 2).mean()) + 1e-12)
    return np.stack([x, x], 1) * (1 - wet) + w * wet


@dataclass
class RoomMatch:
    rt60: float = DEFAULT_RT
    wet: float = DEFAULT_WET
    rt60_reason: str = "approved default"
    wet_reason: str = "approved default"
    eq_f: np.ndarray | None = None
    eq_g: np.ndarray | None = None
    notes: list = field(default_factory=list)

    def summary(self) -> dict:
        eq = {}
        if self.eq_f is not None:
            eq = {str(int(q)): round(float(np.interp(q, self.eq_f, self.eq_g)), 2)
                  for q in (100, 200, 400, 800, 1600, 3150, 6300, 10000)}
        return {"rt60": round(self.rt60, 3), "rt60_reason": self.rt60_reason, "wet": round(self.wet, 3),
                "wet_reason": self.wet_reason, "eq_db": eq}


def analyse(ref: np.ndarray, take: np.ndarray, match: float = 0.30, max_wet: float = MAX_WET,
            default_rt: float = DEFAULT_RT) -> RoomMatch:
    """Estimate the scene's room from ``ref`` (original line, vocal stem) for the dry ``take`` (mono, 48 kHz)."""
    rm = RoomMatch()
    rt_r, n_r, offs_r, hop = rt_estimate(ref)
    if rt_r is None or n_r < 2:
        rm.rt60, rm.rt60_reason = default_rt, "too few decays in the original line, approved default"
    else:
        rm.rt60, rm.rt60_reason = float(np.clip(rt_r, 0.2, 1.2)), f"estimated from {n_r} decays"
    rm.eq_f, rm.eq_g = eq_curve(ref, take, match)
    teq = apply_eq(take, rm.eq_g)
    lr_r = late_ratio(ref, offsets(ref)[0] or offs_r, hop)
    offs_t = offsets(teq)[0]
    if lr_r is None or not offs_t:
        rm.wet, rm.wet_reason = DEFAULT_WET, "not enough pauses to measure, approved default"
        return rm
    ir = make_ir(rm.rt60)
    best = None
    for w in np.arange(0, max_wet + 1e-9, 0.01):
        lr = late_ratio(reverb(teq, ir, w).mean(1), offs_t, hop)
        if lr is None:
            continue
        err = abs(lr - lr_r)
        if best is None or err < best[0] - 1e-9:
            best = (err, float(w), lr)
    if best is None:
        rm.wet, rm.wet_reason = DEFAULT_WET, "not enough pauses to measure, approved default"
    else:
        rm.wet = round(best[1], 2)
        rm.wet_reason = f"matched the scene's late sound ({best[2]:.1f} dB vs {lr_r:.1f} dB)"
        if rm.wet >= max_wet - 1e-6:
            rm.notes.append("The scene is very echoey; the voice reverb was capped so it stays clear.")
    return rm
