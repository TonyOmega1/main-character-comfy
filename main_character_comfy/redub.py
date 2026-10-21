"""Redub one line of a finished clip with MC's clean voice chain (the one approved for the PULL UP demo).

The new line (your recording, or a Chatterbox line in your cloned voice) is laid over the clip's own audio:

1. speech enhancement with **DeepFilterNet3** (open source, CPU; optional install, see ``deepfilter_status``)
2. retime to fit a target length (rubberband if this ffmpeg has it, else atempo; pitch kept)
3. high-pass 80 Hz, then light compression (-18 dB, 2:1, 10/150 ms, 6 dB knee)
4. gentle linear-phase EQ toward the clip's own soundtrack (30 % match, +/-3 dB, boosts capped at +2 dB, none
   above 12 kHz), so the voice sits in the scene
5. a short clean convolution reverb (12 ms pre-delay, RT60 0.45 s, 200 Hz-5 kHz tail, 15 % wet)
6. the clip's audio ducked under the line with a smooth cosine envelope (0.25 s ramps, to 0.28 = -11 dB)
7. a float sum, then ONE static gain (loudness matched to the original clip, backed off for -1.5 dBTP)

On purpose **not** used: ffmpeg ``loudnorm`` (it silently goes dynamic when the input peak is over the target),
fast bus limiters (they crush syllables into a buzz), noise beds, exciters, saturation. Video is stream-copied;
the result is AAC 320 kb/s, 48 kHz stereo.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .media import ffmpeg_exe, has_audio_stream, media_duration, run_ffmpeg

SR = 48000


@dataclass
class RedubSettings:
    start: float = 0.0  # where the line starts in the clip (s)
    fit_seconds: float | None = None  # retime the line to this length (None = keep its natural length)
    denoise: str = "auto"  # auto (DeepFilterNet3 if installed) | deepfilternet (required) | off
    hp_hz: float = 80.0
    comp: str = "threshold=-18dB:ratio=2:attack=10:release=150:knee=6:makeup=1"
    eq_match: float = 0.30
    eq_clamp_db: float = 3.0
    eq_boost_cap_db: float = 2.0
    reverb_wet: float = 0.15
    reverb_rt60: float = 0.45
    reverb_predelay: float = 0.012
    duck: float = 0.28  # clip audio gain under the line (0 = replace it, 1 = no duck)
    duck_ramp: float = 0.25
    voice_level_db: float = 0.0  # voice loudness relative to the original clip's loudness
    tp_max: float = -1.5  # true-peak ceiling for the one static gain (dBTP)
    target_lufs: float = -16.0  # only used when the clip has no audio of its own
    notes: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# small DSP helpers (numpy only; ffmpeg does resampling, filters and metering)
# --------------------------------------------------------------------------- #
def load(path: str | Path, channels: int = 1) -> np.ndarray:
    raw = subprocess.check_output([ffmpeg_exe(), "-v", "error", "-i", str(path), "-vn", "-ac", str(channels),
                                   "-ar", str(SR), "-f", "f32le", "-"])
    a = np.frombuffer(raw, np.float32).astype(np.float64)
    return a.reshape(-1, channels) if channels > 1 else a


def save(path: str | Path, x: np.ndarray, subtype: str = "f32le") -> Path:
    path = Path(path)
    ch = 1 if x.ndim == 1 else x.shape[1]
    codec = {"f32le": "pcm_f32le", "s24le": "pcm_s24le"}[subtype]
    proc = subprocess.run([ffmpeg_exe(), "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", str(ch), "-i", "-",
                           "-c:a", codec, str(path)], input=np.ascontiguousarray(x, np.float32).tobytes(),
                          capture_output=True)
    if proc.returncode:
        raise RuntimeError(f"ffmpeg could not write {path.name}: {proc.stderr.decode()[-400:]}")
    return path


def lufs(x: np.ndarray, work: Path) -> float:
    """Integrated loudness (EBU R128), measured by ffmpeg's ebur128 meter (measurement only)."""
    if not np.any(np.abs(x) > 1e-6):
        return -70.0
    tmp = save(work / "_meter.wav", x)
    proc = subprocess.run([ffmpeg_exe(), "-hide_banner", "-nostats", "-i", str(tmp), "-af", "ebur128", "-f", "null",
                           "-"], capture_output=True, text=True)
    tail = proc.stderr[proc.stderr.rfind("Summary:"):]
    m = re.search(r"I:\s+(-?[\d.]+)", tail)
    return float(m.group(1)) if m else -70.0


def true_peak_db(x: np.ndarray) -> float:
    """4x-oversampled peak (FFT zero-padding)."""
    xs = x if x.ndim == 2 else x[:, None]
    n = xs.shape[0]
    if n == 0:
        return -120.0
    spec = np.fft.rfft(xs, axis=0)
    up = np.fft.irfft(spec, n * 4, axis=0) * 4
    return float(20 * np.log10(np.abs(up).max() + 1e-12))


def spectrum(x: np.ndarray, nfft: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Welch-style average power spectrum."""
    x = x if x.ndim == 1 else x.mean(1)
    hop, win = nfft // 2, np.hanning(nfft)
    frames = [x[i:i + nfft] * win for i in range(0, max(1, len(x) - nfft), hop) if len(x[i:i + nfft]) == nfft]
    if not frames:
        frames = [np.pad(x, (0, nfft - len(x))) * win]
    p = np.mean([np.abs(np.fft.rfft(f)) ** 2 for f in frames], axis=0)
    return np.fft.rfftfreq(nfft, 1 / SR), p


def fft_convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = len(a) + len(b) - 1
    size = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(a, size) * np.fft.rfft(b, size), size)[:n]


def match_eq(voice: np.ndarray, bed: np.ndarray, s: RedubSettings) -> tuple[np.ndarray, dict]:
    """Gentle linear-phase EQ that nudges the voice toward the bed's long-term spectrum."""
    f, pb = spectrum(bed)
    _, pv = spectrum(voice[np.abs(voice) > 1e-3] if np.any(np.abs(voice) > 1e-3) else voice)
    band = (f > 150) & (f < 12000)
    d = 10 * np.log10(pb + 1e-20) - 10 * np.log10(pv + 1e-20)
    d -= np.average(d[band], weights=1 / f[band])
    sm = np.array([np.mean(d[(f > x / 2 ** (1 / 6)) & (f < x * 2 ** (1 / 6))]) if x > 20 else 0.0 for x in f])
    g = np.clip(sm * s.eq_match, -s.eq_clamp_db, s.eq_clamp_db)
    g[~band] = np.interp(f[~band], f[band][[0, -1]], g[band][[0, -1]])
    g = np.minimum(g, s.eq_boost_cap_db)
    g = np.where(g > 0, g * np.clip((12000 - f) / 4000, 0, 1), g)
    h = np.fft.irfft(10 ** (g / 20))
    h = np.roll(h, len(h) // 2) * np.hanning(len(h))
    dly = len(h) // 2
    out = fft_convolve(voice, h)[dly:dly + len(voice)]
    return out, {"max_boost_db": round(float(g.max()), 2), "max_cut_db": round(float(g.min()), 2)}


def reverb_ir(s: RedubSettings, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(SR * (s.reverb_rt60 * 1.2 + s.reverb_predelay))
    t = np.arange(n) / SR
    ir = np.zeros((n, 2))
    for dd, gg in [(0.006, .45), (0.011, .33), (0.017, .26), (0.024, .19), (0.033, .13)]:
        for ch in (0, 1):
            k = int((s.reverb_predelay + dd + 0.0013 * ch) * SR)
            ir[min(k, n - 1), ch] += gg * (1 if rng.random() > .5 else -1)
    tail = rng.standard_normal((n, 2)) * np.exp(-6.9 * np.clip(t - s.reverb_predelay - 0.025, 0, None)
                                                / s.reverb_rt60)[:, None]
    tail[: int((s.reverb_predelay + 0.010) * SR)] = 0
    spec = np.fft.rfft(tail, axis=0)  # 200 Hz - 5 kHz tail (smooth band-pass in the frequency domain)
    fr = np.fft.rfftfreq(n, 1 / SR)
    shape = 1 / np.sqrt(1 + (200 / np.maximum(fr, 1)) ** 4) / np.sqrt(1 + (fr / 5000) ** 4)
    tail = np.fft.irfft(spec * shape[:, None], n, axis=0) * 0.07
    ir += tail
    return ir / np.sqrt((ir ** 2).sum(0)).max()


def cosine_duck(n: int, a0: float, a1: float, depth: float, ramp: float) -> np.ndarray:
    t = np.arange(n) / SR
    r = lambda x: 0.5 - 0.5 * np.cos(np.pi * np.clip(x, 0, 1))  # noqa: E731
    return 1 - (1 - depth) * np.minimum(r((t - (a0 - ramp)) / ramp), r(((a1 + ramp) - t) / ramp))


# --------------------------------------------------------------------------- #
# DeepFilterNet3 (optional) and retiming
# --------------------------------------------------------------------------- #
def _deepfilter_cmd() -> list[str] | None:
    exe = os.environ.get("MC_DEEPFILTER", "").strip()
    if exe and Path(exe).exists():
        return [exe]
    found = shutil.which("deepFilter")
    if found:
        return [found]
    try:  # the deepfilternet package installed into MC's own Python
        import importlib.util

        if importlib.util.find_spec("df") is not None:
            import sys

            return [sys.executable, "-m", "df.enhance"]
    except Exception:  # pragma: no cover
        pass
    return None


def deepfilter_status() -> dict:
    cmd = _deepfilter_cmd()
    return {"installed": bool(cmd), "command": " ".join(cmd) if cmd else None,
            "how_to_install": "pip install deepfilternet (pulls in CPU torch), or point MC_DEEPFILTER at a "
                              "deepFilter executable from its own venv"}


def deepfilter(src: Path, out_dir: Path) -> Path:
    cmd = _deepfilter_cmd()
    if not cmd:
        raise RuntimeError("DeepFilterNet isn't installed. " + deepfilter_status()["how_to_install"])
    out_dir.mkdir(parents=True, exist_ok=True)
    wav48 = out_dir / "line_48k.wav"
    run_ffmpeg(["-i", str(src), "-vn", "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(wav48)])
    proc = subprocess.run([*cmd, "-m", "DeepFilterNet3", "--no-suffix", "-o", str(out_dir / "df"), str(wav48)],
                          capture_output=True, text=True, timeout=600)
    out = out_dir / "df" / wav48.name
    if proc.returncode or not out.exists():
        raise RuntimeError(f"DeepFilterNet3 failed: {(proc.stderr or proc.stdout)[-600:]}")
    return out


def has_rubberband() -> bool:
    proc = subprocess.run([ffmpeg_exe(), "-hide_banner", "-filters"], capture_output=True, text=True)
    return " rubberband " in proc.stdout


def retime(src: Path, dst: Path, ratio: float) -> tuple[Path, str]:
    """Stretch ``src`` by ``ratio`` (output length / input length), pitch kept."""
    tempo = 1.0 / ratio
    if has_rubberband():
        run_ffmpeg(["-i", str(src), "-af", f"rubberband=tempo={tempo:.5f}:pitchq=quality:formant=preserved",
                    "-c:a", "pcm_f32le", str(dst)])
        return dst, "rubberband"
    chain, t = [], tempo
    while t > 2.0:
        chain.append("atempo=2.0")
        t /= 2.0
    while t < 0.5:
        chain.append("atempo=0.5")
        t /= 0.5
    chain.append(f"atempo={t:.5f}")
    run_ffmpeg(["-i", str(src), "-af", ",".join(chain), "-c:a", "pcm_f32le", str(dst)])
    return dst, "atempo"


# --------------------------------------------------------------------------- #
# the chain
# --------------------------------------------------------------------------- #
def redub(video: str | Path, line: str | Path, out: str | Path, work: str | Path,
          s: RedubSettings | None = None) -> dict:
    """Lay ``line`` over ``video`` at ``s.start`` with the clean chain. Returns a report dict (also saved)."""
    s = s or RedubSettings()
    video, line, out, work = Path(video), Path(line), Path(out), Path(work)
    work.mkdir(parents=True, exist_ok=True)
    report: dict = {"settings": {k: v for k, v in asdict(s).items() if k != "notes"}, "steps": []}
    total = media_duration(video)
    if not 0 <= s.start < total:
        raise ValueError(f"start {s.start:.2f}s is outside the clip (0-{total:.2f}s)")
    n_total = int(round(total * SR))

    # 1. speech enhancement
    src = line
    if s.denoise != "off":
        try:
            src = deepfilter(line, work / "denoise")
            report["steps"].append("deepfilternet3")
        except RuntimeError as exc:
            if s.denoise == "deepfilternet":
                raise
            report["steps"].append("deepfilternet3 skipped (not installed)")
            report["denoise_note"] = str(exc)
    # trim leading/trailing silence so the line starts where you asked
    trimmed = work / "line_trim.wav"
    run_ffmpeg(["-i", str(src), "-af", "silenceremove=start_periods=1:start_threshold=-55dB,areverse,"
                "silenceremove=start_periods=1:start_threshold=-55dB,areverse,afade=t=in:d=0.01",
                "-ac", "1", "-ar", str(SR), "-c:a", "pcm_f32le", str(trimmed)])
    src = trimmed
    natural = media_duration(src)

    # 2. retime
    if s.fit_seconds:
        ratio = float(s.fit_seconds) / max(natural, 1e-3)
        if not 0.5 <= ratio <= 2.0:
            raise ValueError(f"fitting a {natural:.2f}s line into {s.fit_seconds:.2f}s is more than 2x: re-record or "
                             "change fit_seconds")
        if abs(ratio - 1) > 0.01:
            src, how = retime(src, work / "line_retimed.wav", ratio)
            report["steps"].append(f"retime x{ratio:.3f} ({how})")

    # 3. high-pass + light compression (float)
    comp = work / "line_comp.wav"
    run_ffmpeg(["-i", str(src), "-af", f"highpass=f={s.hp_hz}:p=2,acompressor={s.comp}", "-ac", "1", "-ar", str(SR),
                "-c:a", "pcm_f32le", str(comp)])
    report["steps"].append("highpass 80 Hz + compressor 2:1")
    v = load(comp)
    v = v[: max(0, n_total - int(s.start * SR))]
    if len(v) == 0:
        raise ValueError("the line ends up empty after trimming: is it silent?")

    # the clip's own audio (the bed)
    has_bed = has_audio_stream(video)
    bed = load(video, 2)[:n_total] if has_bed else np.zeros((n_total, 2))
    bed = np.pad(bed, ((0, n_total - len(bed)), (0, 0)))
    report["clip_has_audio"] = has_bed

    # 4. gentle EQ toward the bed
    if has_bed and np.any(np.abs(bed) > 1e-4):
        v, eq = match_eq(v, bed, s)
        report["steps"].append("linear-phase match eq")
        report["eq"] = eq

    # 5. short reverb
    ir = reverb_ir(s)
    dry = np.pad(v, (0, int(0.6 * SR)))
    wet = np.stack([fft_convolve(dry, ir[:, c])[: len(dry)] for c in (0, 1)], 1)
    wet *= np.sqrt((dry ** 2).mean()) / (np.sqrt((wet ** 2).mean()) + 1e-12)
    vr = np.stack([dry, dry], 1) * (1 - s.reverb_wet) + wet * s.reverb_wet
    report["steps"].append(f"convolution reverb {int(s.reverb_wet * 100)}% wet")

    # level: voice as loud as the clip's program (+ voice_level_db)
    ref = lufs(bed, work) if has_bed else s.target_lufs
    if ref <= -60:
        ref = s.target_lufs
    gv = 10 ** ((ref + s.voice_level_db - lufs(vr, work)) / 20)
    vr *= gv

    # 6. smooth duck + float sum
    k0 = int(round(s.start * SR))
    line_len = len(v) / SR
    env = cosine_duck(n_total, s.start - 0.15, s.start + line_len + 0.10, s.duck, s.duck_ramp) if has_bed else \
        np.ones(n_total)
    voice = np.zeros((n_total, 2))
    seg = vr[: n_total - k0]
    voice[k0:k0 + len(seg)] = seg
    mix = bed * env[:, None] + voice
    report["steps"].append(f"cosine duck to {s.duck:.2f} with {s.duck_ramp:.2f}s ramps")

    # 7. one static gain
    l0, tp0 = lufs(mix, work), true_peak_db(mix)
    gain = (ref - l0) if has_bed else (s.target_lufs - l0)
    gain = min(gain, s.tp_max - tp0)
    mix *= 10 ** (gain / 20)
    report["steps"].append("one static gain (no loudnorm, no limiter)")
    wav = save(work / "mix_final.wav", mix, "s24le")
    stem = save(work / "voice_placed.wav", voice * 10 ** (gain / 20), "f32le")  # the line alone (lip-sync drive)
    out.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(["-i", str(video), "-i", str(wav), "-map", "0:v:0?", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
                "-b:a", "320k", "-ar", str(SR), "-t", f"{total:.3f}", "-movflags", "+faststart", str(out)])
    report.update(output=str(out), clip_seconds=round(total, 3), line_seconds=round(line_len, 3),
                  line_natural_seconds=round(natural, 3), start=s.start, clip_lufs=round(ref, 2),
                  final_lufs=round(l0 + gain, 2), final_true_peak_db=round(tp0 + gain, 2),
                  static_gain_db=round(gain, 2), voice_stem=str(stem))
    (work / "redub_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
