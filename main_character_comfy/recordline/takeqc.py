"""Raw-take checks before scoring (DubMaster's study tool takeqc.py, ported to numpy only).

* clipping: runs of >= 3 samples within 0.1 dB of full scale, or repeated flat tops at the take's own max (clipped
  upstream, then turned down)
* dropouts: 1-30 ms runs of exact zeros / frozen samples inside speech, and > 30 dB level holes (buffer or USB trouble)
* clicks / crackle: outliers in the linear-prediction residual (Godsill & Rayner, "Digital Audio Restoration", 1998),
  tagged LOUD (gain driven) or QUIET (cable / buffer)
* digital silence share (a gate or noise removal cutting the room to zero), level vs a -12 dBFS peak target, SNR

Run on the take exactly as recorded (before any cleanup). Warnings are short plain English for the UI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .dsp import SR, decode, fft_filter, frame_db, oversampled_peak, runs

K_MAD, K_CTX = 16, 8  # click thresholds (validated on injected clicks in the study tool)


def _toeplitz_solve(r: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = len(r)
    idx = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    return np.linalg.solve(r[idx], b)


def lpc_residual(x: np.ndarray, order: int = 24, blk: int = 960) -> np.ndarray:
    e = np.zeros_like(x)
    for i in range(0, len(x) - blk, blk):
        seg = x[max(0, i - order):i + blk]
        w = seg * np.hanning(len(seg))
        r = np.correlate(w, w, "full")[len(w) - 1:len(w) + order].copy()
        if r[0] < 1e-12:
            continue
        r[0] *= 1.0001
        try:
            a = _toeplitz_solve(r[:order], r[1:order + 1])
        except np.linalg.LinAlgError:
            continue
        res = np.convolve(seg, np.concatenate([[1], -a]))[: len(seg)]
        e[i:i + blk] = res[-blk:]
    return e


def analyse(src: str | Path | np.ndarray, sr: int = SR, label: str = "This take") -> dict:
    x = decode(src, SR) if not isinstance(src, np.ndarray) else np.asarray(src, float)
    if isinstance(src, np.ndarray) and sr != SR:
        from .dsp import resample

        x = resample(x, sr, SR)
    sr = SR
    rep: dict = {"duration_s": round(len(x) / sr, 3)}
    if len(x) < int(0.1 * sr) or not np.any(np.abs(x) > 1e-6):
        rep.update(silent=True, warnings=[f"{label} is silent or too short."], clipping=False, dropouts=False,
                   crackle=False, quiet=True)
        return rep
    pk = np.abs(x).max()
    rep["sample_peak_dbfs"] = round(20 * np.log10(pk + 1e-12), 2)
    rep["true_peak_dbtp"] = round(20 * np.log10(oversampled_peak(x).max() + 1e-12), 2)
    db, hop = frame_db(x, sr)
    act = db > max(np.percentile(db, 90) - 30, -60)
    live = db > -140
    rep["digital_silence_pct"] = round(100 * float(1 - live.mean()), 1)
    rep["noise_floor_dbfs"] = round(float(np.percentile(db[live], 10)), 1) if live.any() else -200.0
    rep["speech_level_dbfs_p90"] = round(float(np.percentile(db, 90)), 1)
    rep["snr_db"] = round(rep["speech_level_dbfs_p90"] - rep["noise_floor_dbfs"], 1)
    loud_thr = np.percentile(db[act], 70) if act.any() else 0
    # clipping
    a = np.abs(x)
    clip = []
    for lvl, tag in ((10 ** (-0.1 / 20), "full_scale"), (pk * 0.9995, "flat_top")):
        if tag == "flat_top" and pk < 0.05:
            continue
        s, ln = runs(a >= lvl)
        clip += [(st, tag) for st, n in zip(s, ln) if n >= 3]
    if sum(1 for c in clip if c[1] == "flat_top") < 3:
        clip = [c for c in clip if c[1] != "flat_top"]
    rep["clip_events"] = len(clip)
    # dropouts
    drops = []
    for mask in (x == 0, np.concatenate([[False], np.diff(x) == 0])):
        s, ln = runs(mask)
        g = int(0.02 * sr)
        for st, n in zip(s, ln):
            if n < int(0.001 * sr) or n > int(0.03 * sr):
                continue
            pre, post = x[max(0, st - g):st], x[st + n:st + n + g]
            if len(pre) and len(post) and np.abs(pre).max() > 1e-4 and np.abs(post).max() > 1e-4:
                drops.append(st / sr)
    for i in range(1, len(db) - 3):
        if act[i - 1] and (act[i + 1] or act[i + 2]) and db[i] < min(db[i - 1], max(db[i + 1], db[i + 2])) - 30:
            drops.append(i * hop / sr)
    rep["dropout_events"] = len({round(t * 50) for t in drops})
    # clicks on a 1 kHz high-passed copy
    y = fft_filter(x, sr, lo=1000, order=2)
    e = lpc_residual(y, blk=int(0.02 * sr))
    W = int(0.05 * sr)
    k = len(e) // W
    loud = quiet = 0
    if k:
        mad = np.median(np.abs(e[: k * W].reshape(k, W)), 1) + 1e-9
        mad = np.maximum(mad, np.percentile(mad, 20))
        thr = np.maximum(np.repeat(mad, W) * K_MAD, 10 ** (-60 / 20))
        thr = np.pad(thr, (0, len(e) - len(thr)), mode="edge")
        s, ln = runs(np.abs(e) > thr)
        q = int(0.002 * sr)
        last = -10 ** 9
        for st, n in zip(s, ln):
            if n > q or st - last < int(0.003 * sr):
                continue
            peak = np.abs(e[st:st + n]).max()
            ctx = np.concatenate([e[max(0, st - 4 * q):max(0, st - q)], e[st + n + q:st + n + 4 * q]])
            if len(ctx) == 0 or peak < K_CTX * np.sqrt(np.mean(ctx ** 2) + 1e-20):
                continue
            last = st
            fi = min(st // hop, len(db) - 1)
            w5 = x[max(0, st - int(0.005 * sr)):st + int(0.005 * sr)]
            if len(w5) and (np.abs(w5) < 3e-6).sum() > int(0.001 * sr):
                continue  # gate edge, reported via digital silence
            if act[fi] and db[fi] >= loud_thr:
                loud += 1
            elif not act[fi]:
                quiet += 1
    rep["clicks_loud"], rep["clicks_quiet"] = loud, quiet
    # verdicts
    w = []
    rep["clipping"] = bool(rep["clip_events"])
    rep["dropouts"] = bool(rep["dropout_events"])
    rep["crackle"] = loud + quiet >= 3
    rep["quiet"] = rep["sample_peak_dbfs"] < -30
    if rep["clipping"]:
        w.append(f"{label} is clipping (distorted on loud words). Turn the mic gain down so loud lines peak "
                 "around -12 dB.")
    elif rep["sample_peak_dbfs"] > -3:
        w.append(f"{label} is very hot. Back the mic gain off a little (aim for -12 dB peaks).")
    if rep["quiet"]:
        w.append(f"{label} is very quiet. Raise the mic gain or move closer.")
    if rep["dropouts"]:
        w.append(f"{label} has tiny gaps in the audio, which is a computer or USB hiccup, not your voice. Close heavy "
                 "apps and record again.")
    if rep["crackle"]:
        if loud >= 2 * max(quiet, 1):
            w.append(f"{label} crackles on loud words, which means the gain is too hot. Turn it down a bit.")
        elif quiet > loud:
            w.append(f"{label} has random clicks in quiet parts, which points to a loose cable or buffer dropouts.")
        else:
            w.append(f"{label} has some clicks. Check the gain and the cable.")
    if rep["digital_silence_pct"] > 5:
        w.append(f"{label} cuts to dead silence between words (a noise gate or noise removal is on). Turn it off "
                 "for dubbing so your voice blends into the scene.")
    elif rep["snr_db"] < 30 and not rep["quiet"]:
        w.append(f"{label} has a lot of background noise. A quieter room or closer mic will help.")
    rep["warnings"] = w
    return rep
