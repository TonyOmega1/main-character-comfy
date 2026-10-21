"""Timestamp-accurate frame conform + a mouth-sync self-check.

Why: a swap renders at 16 fps (Wan) and is RIFE-interpolated to 32 fps, then has to land on the source clip's own
frame grid (24/25/30/60 fps). Stretching the interpolated take proportionally onto the shot length drifts by 1-2
frames in longer shots (the mouth leads the audio). ``conform_indices`` maps every output frame to the generated
frame at the same *time* instead. ``sync_check`` then measures it: it correlates mouth-motion energy of the source
and the render over small lags; a healthy render peaks at lag 0.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

SYNC_MAX_LAG = 3          # frames each way
SYNC_MIN_CORR = 0.5       # under this the mouth motion doesn't follow the source well enough to trust the lag


def conform_indices(n_out: int, out_fps: float, gen_fps: float, n_gen: int, used_frac: float = 1.0) -> list[int]:
    """Index into the generated take for each of ``n_out`` output frames at ``out_fps``.

    Output frame j sits at t = j / out_fps; it takes the generated frame nearest that time, round(t * gen_fps).
    ``used_frac`` < 1 keeps only the first part of the take (ping-pong padded shots render 3x and use 1/3).
    """
    if n_out <= 0 or n_gen <= 0:
        return []
    usable = max(1, int(round(n_gen * used_frac)))
    return [min(usable - 1, int(round(j / float(out_fps) * float(gen_fps)))) for j in range(n_out)]


def proportional_indices(n_out: int, n_gen: int) -> list[int]:
    """The old way (stretch the take onto the shot length); kept for comparison in tests."""
    if n_out <= 0 or n_gen <= 0:
        return []
    return [min(n_gen - 1, int(j * n_gen / n_out)) for j in range(n_out)]


def mouth_box(face: Sequence[float]) -> tuple[int, int, int, int]:
    """Lower-face box (x0, y0, x1, y1) from a face box (x, y, w, h): the mouth/jaw band."""
    x, y, w, h = (float(v) for v in face[:4])
    return int(x + 0.2 * w), int(y + 0.6 * h), int(x + 0.8 * w), int(y + 1.05 * h)


def mouth_energy(frames: Sequence[np.ndarray], faces: Sequence[Sequence[float] | None]) -> np.ndarray:
    """Per-frame mouth motion: mean |frame_t - frame_t-1| inside the mouth band (grey, 64x32 resample).

    ``faces`` is one face box per frame (None where none was found; that step counts as 0 motion).
    """
    out = np.zeros(len(frames), np.float32)
    prev = None
    for i, (f, b) in enumerate(zip(frames, faces)):
        if b is None:
            prev = None
            continue
        x0, y0, x1, y1 = mouth_box(b)
        h, w = f.shape[:2]
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if x1 - x0 < 4 or y1 - y0 < 4:
            prev = None
            continue
        crop = f[y0:y1, x0:x1].astype(np.float32)
        g = crop.mean(axis=2) if crop.ndim == 3 else crop
        g = _resample(g, 64, 32)
        if prev is not None:
            out[i] = float(np.abs(g - prev).mean())
        prev = g
    return out


def _resample(g: np.ndarray, w: int, h: int) -> np.ndarray:
    ys = np.linspace(0, g.shape[0] - 1, h).astype(int)
    xs = np.linspace(0, g.shape[1] - 1, w).astype(int)
    return g[ys][:, xs]


def lag_correlations(src: Iterable[float], out: Iterable[float], max_lag: int = SYNC_MAX_LAG) -> dict[int, float]:
    """Pearson correlation of two equal-rate signals at lags -max_lag..max_lag (positive lag: render is late)."""
    a = np.asarray(list(src), np.float64)
    b = np.asarray(list(out), np.float64)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    res: dict[int, float] = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x, y = a[: n - lag], b[lag:]
        else:
            x, y = a[-lag:], b[: n + lag]
        if len(x) < 4 or x.std() < 1e-9 or y.std() < 1e-9:
            res[lag] = 0.0
            continue
        res[lag] = float(np.corrcoef(x, y)[0, 1])
    return res


def sync_check(src_energy: Iterable[float], out_energy: Iterable[float], max_lag: int = SYNC_MAX_LAG,
               min_corr: float = SYNC_MIN_CORR) -> dict[str, Any]:
    """Best lag + its correlation, ``ok`` when the best lag is 0 and the motion follows the source."""
    c = lag_correlations(src_energy, out_energy, max_lag)
    best = max(c, key=lambda k: (c[k], -abs(k)))
    ok = best == 0 and c[best] >= min_corr
    note = ("in sync" if ok else (f"mouth is {abs(best)} frame(s) {'late' if best > 0 else 'early'}" if best != 0
                                  else "mouth motion doesn't follow the source closely"))
    return {"best_lag": best, "corr": round(c[best], 3), "corr_at_0": round(c.get(0, 0.0), 3), "ok": ok,
            "note": note, "by_lag": {k: round(v, 3) for k, v in c.items()}}
