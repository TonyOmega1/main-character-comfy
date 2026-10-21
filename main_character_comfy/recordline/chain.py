"""Voice chain + mix for the chosen take.

Which chain, and why: this follows DubMaster's approved Test 6 chain (``voice_chain.py``) rather than ``redub.py``,
keeping two of redub's parts. Reasons, all audible:

* EQ target: voice_chain matches toward the ORIGINAL LINE (Demucs vocal stem in the window); redub matches toward the
  whole soundtrack (music + SFX + voice), which drags the voice toward the music's tilt.
* Room: reverb time and wetness are measured from the original line (roommatch, wet may be 0); redub always adds a
  fixed 0.45 s / 15 % room.
* Level: the voice is set to the original line's loudness (what the character sounded like); redub sets it to the
  whole programme's loudness, which is too hot in loud scenes and too quiet in quiet ones.
* Peaks: voice_chain has slow, capped peak control (<= 4 dB) on the voice; redub has none.
* Bed: voice_chain MUTED the original under the line (a noise-floor hole, now banned). Here the Demucs background stem
  (music + SFX, no voice) keeps running under the new line with a gentle cosine duck (-3 dB, 0.25 s ramps). Without
  stems it falls back to redub's duck of the full mix (-11 dB, never to silence). From redub: the duck envelope and
  the final ONE static gain with a -1.5 dBTP ceiling.

Order: DeepFilterNet3 (attenuation-limited, never to digital silence) -> cut the speech -> rubberband stretch
(pitch + formant kept, max +-15 %) -> place on the window onset -> 80 Hz high-pass + 2:1 compression -> roommatch EQ
-> slow peak control -> one voice gain (= original line loudness) -> roommatch reverb -> bed duck -> float sum -> one
static gain -> 24-bit WAV. Never: loudnorm, fast limiters, saturation, exciters, synthetic room tone, looped gap fill,
low-bitrate AAC.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import redub
from ..media import run_ffmpeg
from . import roommatch
from .deps import HINTS
from .dsp import ASR_SR, SR, decode, fades, slow_peak_control, speech_band_db


@dataclass
class ChainSettings:
    denoise: str = "auto"            # auto (DeepFilterNet3 if installed) | deepfilternet (required) | off
    atten_lim_db: float = 24.0       # DeepFilterNet attenuation limit: keeps the take's own room, no digital silence
    hp_hz: float = 80.0
    comp: str = "threshold=-18dB:ratio=2:attack=10:release=150:knee=6:makeup=1"
    eq_match: float = 0.30
    max_wet: float = roommatch.MAX_WET
    peak_cut_db: float = 4.0
    duck_stem: float = 0.71          # background stem under the line (-3 dB): the original voice is gone from it
    duck_mix: float = 0.28           # full mix under the line when there are no stems (-11 dB, the approved duck)
    duck_ramp: float = 0.25
    duck_pre: float = 0.15
    duck_post: float = 0.10
    tp_max: float = -1.5
    max_stretch: float = 0.15
    tail_grace: float = 0.30         # a take may run this far past the window before its end is trimmed
    voice_level_db: float = 0.0      # new line vs the original line's loudness
    head_pad: float = 0.6            # silence before the InfiniteTalk wav (lets the model settle the mouth)
    it_post: float = 0.25            # audio kept after the window in the InfiniteTalk wav


def denoise(src: Path, work: Path, s: ChainSettings) -> tuple[Path, str]:
    """DeepFilterNet3 with an attenuation limit, through redub's command lookup. Returns (path, step note)."""
    cmd = redub._deepfilter_cmd() if s.denoise != "off" else None
    if s.denoise == "off":
        return src, "noise cleanup off"
    if not cmd:
        if s.denoise == "deepfilternet":
            raise RuntimeError(HINTS["deepfilternet"])
        return src, "noise cleanup skipped (DeepFilterNet not installed)"
    work.mkdir(parents=True, exist_ok=True)
    wav48 = work / "take_48k.wav"
    run_ffmpeg(["-i", str(src), "-vn", "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(wav48)])
    proc = subprocess.run([*cmd, "-m", "DeepFilterNet3", "--atten-lim", str(int(s.atten_lim_db)), "--no-suffix",
                           "-o", str(work / "df"), str(wav48)], capture_output=True, text=True, timeout=600)
    out = work / "df" / wav48.name
    if proc.returncode or not out.exists():
        if s.denoise == "deepfilternet":
            raise RuntimeError(f"DeepFilterNet3 failed: {(proc.stderr or proc.stdout)[-400:]}")
        return src, "noise cleanup skipped (DeepFilterNet failed on this take)"
    return out, f"DeepFilterNet3 (limited to {int(s.atten_lim_db)} dB, keeps your room tone)"


def onset(x: np.ndarray) -> float:
    """First speech-band frame within 35 dB of the loudest (s)."""
    db, hop = speech_band_db(x, SR, 0.005)
    idx = np.where(db > db.max() - 35)[0]
    return float(idx[0] * hop / SR) if len(idx) else 0.0


def offset(x: np.ndarray) -> float:
    db, hop = speech_band_db(x, SR, 0.005)
    idx = np.where(db > db.max() - 35)[0]
    return float((idx[-1] + 1) * hop / SR) if len(idx) else len(x) / SR


def lufs_or_rms(x: np.ndarray, work: Path) -> float:
    """EBU R128 integrated loudness (measurement only); RMS of active frames for very short lines."""
    v = redub.lufs(x, work)
    if v > -69.9:
        return v
    mono = x if x.ndim == 1 else x.mean(1)
    db, _ = speech_band_db(mono, SR, 0.02)
    act = db > db.max() - 20
    return float(10 * np.log10(np.mean(10 ** (db[act] / 10)))) if act.any() else -70.0


@dataclass
class ChainResult:
    aligned_voice_wav: Path
    infinitetalk_wav: Path
    infinitetalk_t0: float
    voice_wet_wav: Path
    final_mix_wav: Path
    stretch_ratio: float
    voice_start: float
    voice_end: float
    trimmed_s: float
    steps: list
    room: dict
    levels: dict


def build(take: Path, speech: tuple[float, float], window_t: tuple[float, float], clip_audio: Path,
          vocals: Path | None, background: Path | None, work: Path, s: ChainSettings,
          stretch_mode: str = "rubberband") -> ChainResult:
    """Run the chain on ``take`` (already denoised or raw), speech region ``speech`` (s, inside the take), placed on
    ``window_t`` of the clip. All outputs are full clip length on the clip's timeline except the InfiniteTalk wav."""
    work.mkdir(parents=True, exist_ok=True)
    steps: list[str] = []
    clip = decode(clip_audio, SR, 2)
    n_total = len(clip)
    total = n_total / SR
    w0, w1 = window_t
    # 1 cut the speech (10 / 50 ms fades), true silence outside
    raw = decode(take, SR)
    a, b = int(max(0, speech[0] - 0.03) * SR), int(min(len(raw) / SR, speech[1] + 0.08) * SR)
    cut = fades(raw[a:b], SR)
    cut_wav = redub.save(work / "cut.wav", cut)
    # 2 stretch (+-15 % max), pitch and formant kept
    sp = max(offset(cut) - onset(cut), 1e-3)
    want = (w1 - w0) / sp
    ratio = float(np.clip(want, 1 - s.max_stretch, 1 + s.max_stretch))
    src = cut_wav
    if abs(ratio - 1) > 0.005:
        src, how = redub.retime(cut_wav, work / "stretched.wav", ratio)
        steps.append(f"retime x{ratio:.3f} to fit the window ({how}, pitch and formant kept)")
        if how != "rubberband":
            steps.append(HINTS["rubberband"])
    v = decode(src, SR)
    # 3 place: speech onset on the window start; trim anything past window end + grace (fade 60 ms)
    on = onset(v)
    k0 = int(round((w0 - on) * SR))
    if k0 < 0:
        v, k0 = v[-k0:], 0
    limit = int(min(total, w1 + s.tail_grace) * SR)
    end_speech = k0 + int(offset(v) * SR)
    trimmed = 0.0
    if end_speech > limit:
        trimmed = (end_speech - limit) / SR
        v = v[: max(0, limit - k0)]
        v = fades(v, SR, 0.0, min(0.06, len(v) / SR))
        steps.append(f"trimmed {trimmed:.2f}s off the end")
    line = np.zeros(n_total)
    seg = v[: max(0, n_total - k0)]
    line[k0:k0 + len(seg)] = seg
    voice_start = k0 / SR + on
    voice_end = min(total, (k0 + len(seg)) / SR)
    # 4 high-pass + gentle compression (ffmpeg, float)
    placed = redub.save(work / "placed_raw.wav", line)
    comp = work / "placed_comp.wav"
    run_ffmpeg(["-i", str(placed), "-af", f"highpass=f={s.hp_hz}:p=2,acompressor={s.comp}", "-c:a", "pcm_f32le",
                str(comp)])
    v = decode(comp, SR)[:n_total]
    v = np.pad(v, (0, n_total - len(v)))
    steps.append("80 Hz high-pass + 2:1 compression")
    # 5 EQ + room toward the original line (vocal stem in the window), else the clip audio in the window
    r0, r1 = int(max(0, w0 - 0.3) * SR), int(min(total, w1 + 0.3) * SR)
    if vocals is not None:
        ref = decode(vocals, SR)[r0:r1]
        ref_kind = "the original line (vocal stem)"
    else:
        ref = clip.mean(1)[r0:r1]
        ref_kind = "the clip's audio around the line (no vocal stem)"
    act = slice(int(voice_start * SR), int(voice_end * SR))
    has_ref = len(ref) > SR // 4 and np.any(np.abs(ref) > 1e-4)
    if has_ref:
        rm = roommatch.analyse(ref, v[act], s.eq_match, s.max_wet)
        v = roommatch.apply_eq(v, rm.eq_g)
        steps.append(f"EQ matched 30% toward {ref_kind}")
    else:
        rm = roommatch.RoomMatch()
        rm.notes.append("No original line to measure, so the approved default room was used.")
    # 6 slow peak control
    v, cut_db = slow_peak_control(v, SR, s.peak_cut_db)
    steps.append(f"slow peak control ({cut_db:.1f} dB max)")
    # 7 one voice gain: the new line as loud as the original line
    work_m = work / "meter"
    work_m.mkdir(exist_ok=True)
    target = lufs_or_rms(ref, work_m) if has_ref else -20.0
    lv = lufs_or_rms(v[act], work_m)
    gv = 10 ** ((target + s.voice_level_db - lv) / 20)
    v = v * gv
    # 8 room (wet may be 0)
    vr = roommatch.reverb(v, roommatch.make_ir(rm.rt60), rm.wet) if rm.wet > 0 else np.stack([v, v], 1)
    steps.append(f"room matched: {rm.rt60:.2f}s, {rm.wet * 100:.0f}% wet ({rm.wet_reason})")
    # 9 bed: background stem ducked gently (never muted), or the full mix ducked when there's no stem
    if background is not None:
        bed = decode(background, SR, 2)[:n_total]
        depth = s.duck_stem
        steps.append(f"music and effects (no-vocals stem) kept under the line, ducked {20 * np.log10(depth):.0f} dB")
    else:
        bed = clip
        depth = s.duck_mix
        steps.append(f"clip audio ducked {20 * np.log10(depth):.0f} dB under the line (no stems)")
    bed = np.pad(bed, ((0, n_total - len(bed)), (0, 0)))
    env = redub.cosine_duck(n_total, voice_start - s.duck_pre, voice_end + s.duck_post, depth, s.duck_ramp)
    mix = bed * env[:, None] + vr
    # 10 one static gain: the clip's original loudness, true peak <= -1.5 dBTP
    l_clip = lufs_or_rms(clip, work_m)
    l_mix = lufs_or_rms(mix, work_m)
    tp = redub.true_peak_db(mix)
    gain = min(l_clip - l_mix, s.tp_max - tp)
    g = 10 ** (gain / 20)
    mix *= g
    steps.append(f"one static gain {gain:+.1f} dB (no loudnorm, no limiter)")
    final = redub.save(work / "final_mix.wav", mix, "s24le")
    aligned = redub.save(work / "aligned_voice.wav", v * g)
    wet = redub.save(work / "voice_wet.wav", vr * g)
    # InfiniteTalk drive: the dry aligned voice from the window start, 0.6 s silent head pad, 16 kHz mono
    it0 = max(0.0, w0)
    it1 = min(total, max(w1, voice_end) + s.it_post)
    seg16 = _to16(v[int(it0 * SR):int(it1 * SR)] * g)
    it = np.concatenate([np.zeros(int(s.head_pad * ASR_SR)), seg16])
    it_wav = work / "infinitetalk_voice.wav"
    _save16(it_wav, it)
    levels = {"original_line_lufs": round(target, 2), "voice_gain_db": round(20 * np.log10(gv), 2),
              "clip_lufs": round(l_clip, 2), "final_gain_db": round(gain, 2),
              "final_true_peak_db": round(tp + gain, 2), "peak_control_db": round(cut_db, 2)}
    return ChainResult(aligned, it_wav, it0 - s.head_pad, wet, final, ratio, voice_start, voice_end, trimmed, steps,
                       {**rm.summary(), "notes": rm.notes, "reference": ref_kind}, levels)


def _to16(x: np.ndarray) -> np.ndarray:
    from .dsp import resample

    return resample(x, SR, ASR_SR)


def _save16(path: Path, x: np.ndarray) -> None:
    proc = subprocess.run([redub.ffmpeg_exe(), "-v", "error", "-y", "-f", "f32le", "-ar", str(ASR_SR), "-ac", "1",
                           "-i", "-", "-c:a", "pcm_s16le", str(path)],
                          input=np.ascontiguousarray(x, np.float32).tobytes(), capture_output=True)
    if proc.returncode:
        raise RuntimeError(proc.stderr.decode(errors="replace")[-300:])
