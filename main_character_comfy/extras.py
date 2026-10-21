"""Single-purpose MC operations that the UI tabs only use inside bigger runs, exposed on their own so the CLI and
agents can call them headlessly: voice clone / TTS, voice changer, image restyle (look or background), video
upscale (SeedVR2) and redub a line (the clean voice chain in :mod:`redub`).

Each one is a generator of :class:`pipeline.Event` like ``Pipeline.run`` and ends with a ``complete`` event, and
each GPU step is one Comfy SDK job through ``Pipeline._job`` (so stop / cancel, progress and recipes all work).
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from . import dialogue, graphs, media, redub as redub_mod, tiers
from .pipeline import (Event, InputError, Pipeline, Request, _BLOCKLIST, estimate_scene_seconds, resolve_subject,
                       tts_chunks)
from .presets import STYLES

AUDIO_EXT = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".opus"}


def _sfw(*texts: str) -> None:
    if _BLOCKLIST.search(" ".join(t or "" for t in texts)):
        raise InputError("MC is for SFW content only.")


def _need_consent(consent: bool, what: str) -> None:
    if not consent:
        raise InputError(f"Confirm the rights first (i_have_rights): {what}")


def _begin(pipe: Pipeline, req: Request, label: str, mode: str, who: str | None = None):
    work, record, results = pipe._begin(req, None, label, mode=mode, who=who)
    record.pop("_req", None)
    return work, record, results


def _finish(work: Path, record: dict[str, Any], final: Path, message: str, extra: dict[str, Any] | None = None,
            kind: str = "file") -> Iterator[Event]:
    top = Path(record["top"])
    name = record["name"]
    dest = top / f"{name}{final.suffix}"
    if final.resolve() != dest.resolve():
        shutil.copyfile(final, dest)
    record["elapsed_seconds"] = round(time.monotonic() - record.pop("t0"), 1)
    record.update(final=str(dest), kind=kind, **(extra or {}))
    (work / "project.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    yield Event("finish", "done", 1.0, "Saved", artifact=dest)
    yield Event("complete", "complete", 1.0, f"{message} ({record['elapsed_seconds']:.0f}s)", artifact=dest,
                data={"work_dir": str(top), **(extra or {})})


# --------------------------------------------------------------------------- #
# voice
# --------------------------------------------------------------------------- #
def run_tts(pipe: Pipeline, text: str, *, voice_sample: Path | None = None, expressiveness: float = 0.55,
            seed: int = 2026, consent: bool = False) -> Iterator[Event]:
    """Say ``text`` in a cloned voice (``voice_sample``, 10-30 s) or Chatterbox's built-in voice. Audio only."""
    text = (text or "").strip()
    if not text:
        raise InputError("Type what the voice should say.")
    if len(text) > pipe.settings.max_script_chars * 2:
        raise InputError(f"Keep it under {pipe.settings.max_script_chars * 2} characters (~80 s).")
    _sfw(text)
    if voice_sample:
        _need_consent(consent, "the voice sample is yours, or you have the speaker's OK.")
    req = Request(mode="talk", script=text, voice_sample=voice_sample, seed=seed, consent=consent,
                  voice_mode="clone" if voice_sample else "builtin")
    work, record, results = _begin(pipe, req, text[:40], "voice", who=Path(str(voice_sample)).stem if voice_sample
                                   else "builtin")
    files: dict[str, Path] = {}
    if voice_sample:
        dur = media.prepare_voice(voice_sample, work / "inputs" / "voice_ref.wav", pipe.settings.max_voice_seconds)
        files["voice_ref"] = work / "inputs" / "voice_ref.wav"
        if dur < 5:
            yield Event("prepare", "progress", 0.5, f"only {dur:.1f}s of speech in the sample; 10-30 s clones better")
    parts = tts_chunks(text)
    m = tiers.models_for(tiers.get(tiers.resolve().key), pipe.settings.models)
    yield Event("voice", "start", message="Cloning the voice" if voice_sample else "Using the built-in voice")
    pieces = []
    for i, part in enumerate(parts):
        sg = graphs.tts_graph(part, models=m, clone=bool(voice_sample), exaggeration=expressiveness, seed=seed)
        name = "voice" if len(parts) == 1 else f"voice_{i:02d}"
        yield from pipe._job(name, "voice", sg, files, work, results, record,
                             scale=(i / len(parts), (i + 1) / len(parts)))
        pieces.append(results[name].path)
    speech = pieces[0] if len(pieces) == 1 else dialogue.concat_audio(pieces, work / "speech_joined.wav",
                                                                       work / "tts_parts")
    wav = media.to_wav(speech, work / "speech.wav")
    dur = media.media_duration(wav)
    yield Event("voice", "done", 1.0, f"Speech ready ({dur:.1f}s)", artifact=wav)
    yield from _finish(work, record, wav, "Voice line ready", {"duration": round(dur, 2), "text": text}, "audio")


def run_voice_convert(pipe: Pipeline, source: Path, target_voice: Path, *, consent: bool = False) -> Iterator[Event]:
    """Re-voice ``source`` (speech audio, or a video's audio track) with the timbre of ``target_voice``."""
    _need_consent(consent, "you have the rights to both voices.")
    req = Request(mode="talk", character_voice=target_voice, consent=consent)
    work, record, results = _begin(pipe, req, Path(str(source)).stem, "voice-convert",
                                   who=Path(str(target_voice)).stem)
    src = media.to_wav(source, work / "inputs" / "vc_source.wav")
    tgt = work / "inputs" / "vc_target.wav"
    media.prepare_voice(target_voice, tgt, pipe.settings.max_voice_seconds)
    m = tiers.models_for(tiers.get(tiers.resolve().key), pipe.settings.models)
    yield Event("revoice", "start", message="Converting the voice")
    yield from pipe._job("revoice", "revoice", graphs.voice_changer_graph(models=m), {"source": src, "target": tgt},
                         work, results, record)
    wav = media.to_wav(results["revoice"].path, work / "converted.wav")
    yield Event("revoice", "done", 1.0, "Voice converted", artifact=wav)
    yield from _finish(work, record, wav, "Converted voice ready", {"duration": round(media.media_duration(wav), 2)},
                       "audio")


# --------------------------------------------------------------------------- #
# images
# --------------------------------------------------------------------------- #
def restyle_prompt(style: str, background: str, extra: str, avatar: bool) -> str:
    keep = ("Keep the exact character design from image 1: the same face, hair, outfit, colors, proportions and art "
            "style." if avatar else "Keep the person's exact face, identity, hairstyle, skin tone, body and outfit "
            "from image 1.")
    parts: list[str] = []
    if style and style not in ("none", "real"):
        parts.append(STYLES[style].edit_prompt)
    else:
        parts.append(f"The same person from image 1, same framing and pose. {keep}")
    if background.strip():
        parts.append(f"Replace only the background with: {background.strip()}. Match the lighting and the art style "
                     f"of the person to the new background so it doesn't look pasted in.")
    if extra.strip():
        parts.append(extra.strip())
    if style and style not in ("none", "real"):
        parts.append(keep)
    return " ".join(parts)


def run_restyle(pipe: Pipeline, image: Path, *, style: str = "none", background: str = "", extra: str = "",
                subject: str = "auto", portrait: bool = True, seed: int = 2026, consent: bool = False,
                tier: str | None = None) -> Iterator[Event]:
    """FLUX.2 klein reference edit of one image: a new look (style preset), a new background, or both."""
    if style not in STYLES:
        raise InputError(f"unknown style {style!r} (use one of {', '.join(sorted(STYLES))})")
    if not (background.strip() or extra.strip() or style not in ("none", "real")):
        raise InputError("Say what to change: a style, a new background, or a short instruction.")
    if len(background) + len(extra) > 400:
        raise InputError("Keep the restyle text under 400 characters.")
    _sfw(background, extra)
    _need_consent(consent, "it's you or your own avatar, or you have their OK.")
    t = tiers.get(tiers.normalize(tier or tiers.resolve().key))
    m = tiers.models_for(t, pipe.settings.models)
    req = Request(mode="meme", photo=image, style=style, consent=consent, subject=subject, seed=seed)
    sub = resolve_subject(req, image)
    work, record, results = _begin(pipe, req, background or style or "restyle", "restyle")
    files = {"photo": media.prepare_photo(image, work / "inputs" / "photo.png", keep_whole=sub == "avatar")}
    prompt = restyle_prompt(style, background, extra, sub == "avatar")
    w, h = t.scene_size
    if not portrait:
        w, h = h, w
    sg = graphs.avatar_graph(prompt, models=m, reference=1, seed=seed, width=w, height=h, prefix="mc/restyle",
                             te_device=t.te_device)
    yield Event("avatar", "start", message="Restyling", data={"eta_seconds": estimate_scene_seconds(t.key)})
    yield from pipe._job("avatar", "avatar", sg, files, work, results, record)
    out = results["avatar"].path
    yield Event("avatar", "done", 1.0, "Restyled", artifact=out)
    yield from _finish(work, record, out, "Restyled image ready", {"prompt": prompt, "subject": sub}, "image")


# --------------------------------------------------------------------------- #
# video
# --------------------------------------------------------------------------- #
def run_upscale(pipe: Pipeline, video: Path, *, engine: str = "seedvr2", width: int | None = None,
                height: int | None = None, seed: int = 2026, tier: str | None = None) -> Iterator[Event]:
    """Sharpen a finished clip to 1080p (or ``width`` x ``height``): SeedVR2 restore + upscale, or plain lanczos.
    The clip's audio is kept."""
    if engine not in ("seedvr2", "lanczos"):
        raise InputError("engine is seedvr2 or lanczos")
    t = tiers.get(tiers.normalize(tier or tiers.resolve().key))
    if engine == "seedvr2" and t.key == "lite":
        raise InputError("SeedVR2 needs a 16 GB+ GPU (Standard or Max). Use engine='lanczos' on Lite.")
    sw, sh = media.video_size(video)
    if not (width and height):
        width, height = (1080, 1920) if sh >= sw else (1920, 1080)
    width, height = int(width) // 2 * 2, int(height) // 2 * 2
    if not (64 <= width <= 3840 and 64 <= height <= 3840):
        raise InputError("Pick an output size between 64 and 3840 px per side.")
    req = Request(mode="swap", clip=video, seed=seed)
    work, record, results = _begin(pipe, req, Path(str(video)).stem, "upscale", who=f"{width}x{height}")
    fps = _fps(video)
    total = media.count_frames(video)
    up = work / "upscaled_take.mp4"
    if engine == "seedvr2":
        m = tiers.models_for(t, pipe.settings.models)
        chunks = [(a, min(graphs.SEEDVR2_MAX_FRAMES, total - a)) for a in range(0, total, graphs.SEEDVR2_MAX_FRAMES)]
        yield Event("finish", "start", message=f"Sharpening {total} frames to {width}x{height} with SeedVR2")
        pieces = []
        for k, (a, n) in enumerate(chunks):
            src = video if len(chunks) == 1 else media.cut_frames(video, work / f"in_{k:02d}.mp4", a, n, fps)
            sg = graphs.seedvr2_graph(models=m, width=width, height=height, fps=fps, seed=seed)
            name = "upscale" if len(chunks) == 1 else f"upscale_{k:02d}"
            yield from pipe._job(name, "finish", sg, {"clip": src}, work, results, record,
                                 scale=(k / len(chunks), (k + 1) / len(chunks)))
            pieces.append(media.cut_frames(results[name].path, work / f"out_{k:02d}.mp4", 0, n, fps))
        media.concat_frames(pieces, up, fps)
    else:
        yield Event("finish", "start", message=f"Resizing to {width}x{height} (lanczos)")
        media.run_ffmpeg(["-i", str(video), "-an", "-vf", f"scale={width}:{height}:flags=lanczos", "-c:v", "libx264",
                          "-crf", "16", "-pix_fmt", "yuv420p", str(up)])
    final = work / "upscaled.mp4"
    if media.has_audio_stream(video):
        media.run_ffmpeg(["-i", str(up), "-i", str(video), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                          "-c:a", "aac", "-b:a", "320k", "-shortest", "-movflags", "+faststart", str(final)])
    else:
        shutil.copyfile(up, final)
    yield from _finish(work, record, final, "Upscaled clip ready", {"engine": engine, "size": [width, height]},
                       "video")


def _fps(video: Path) -> int:
    import re
    import subprocess

    proc = subprocess.run([media.ffmpeg_exe(), "-hide_banner", "-i", str(video)], capture_output=True, text=True)
    m = re.search(r"(\d+(?:\.\d+)?) fps", proc.stderr)
    return max(1, round(float(m.group(1)))) if m else 16


# --------------------------------------------------------------------------- #
# redub
# --------------------------------------------------------------------------- #
def run_redub(pipe: Pipeline, video: Path, *, recording: Path | None = None, text: str = "",
              voice_sample: Path | None = None, start: float = 0.0, fit_seconds: float | None = None,
              duck: float = 0.28, denoise: str = "auto", voice_level_db: float = 0.0, seed: int = 2026,
              consent: bool = False, lipsync: bool = False,
              takes: list[Path] | None = None) -> Iterator[Event]:
    """Replace or add one spoken line in a finished clip. The line is your ``recording``, or ``text`` said in your
    cloned voice (``voice_sample``) or the built-in voice. The approved clean chain does the rest (see redub.py).
    ``lipsync=True`` also re-syncs the mouth to the new line (lipdub.py: one InfiniteTalk job on a head crop of the
    line's window, only the mouth and jaw pasted back; pending a real test). No face found = audio-only result."""
    takes = [Path(str(t)) for t in (takes or []) if t]
    if takes and not recording:
        recording = takes[0]
    if recording and Path(str(recording)) not in takes:
        takes.insert(0, Path(str(recording)))
    if not (recording or text.strip()):
        raise InputError("Give the new line: a recording, or the text to say.")
    if recording and text.strip():
        raise InputError("Use either a recording or text, not both.")
    if denoise not in ("auto", "deepfilternet", "off"):
        raise InputError("denoise is auto, deepfilternet or off")
    if not 0.0 <= duck <= 1.0:
        raise InputError("duck is 0-1 (0 replaces the clip's audio under the line, 1 leaves it alone)")
    _sfw(text)
    _need_consent(consent, "the clip and the voice are yours, or you have the rights.")
    req = Request(mode="talk", clip=video, script=text, recording=recording, voice_sample=voice_sample,
                  consent=consent, seed=seed)
    work, record, results = _begin(pipe, req, Path(str(video)).stem, "redub")
    if recording:
        line = media.to_wav(recording, work / "inputs" / "line.wav")
        yield Event("voice", "skipped", 1.0, "Using your recording", artifact=line)
    else:
        files: dict[str, Path] = {}
        if voice_sample:
            media.prepare_voice(voice_sample, work / "inputs" / "voice_ref.wav", pipe.settings.max_voice_seconds)
            files["voice_ref"] = work / "inputs" / "voice_ref.wav"
        m = tiers.models_for(tiers.get(tiers.resolve().key), pipe.settings.models)
        yield Event("voice", "start", message="Saying the new line" + (" in your voice" if voice_sample else ""))
        sg = graphs.tts_graph(text.strip(), models=m, clone=bool(voice_sample), seed=seed)
        yield from pipe._job("voice", "voice", sg, files, work, results, record)
        line = media.to_wav(results["voice"].path, work / "inputs" / "line.wav")
        yield Event("voice", "done", 1.0, "Line ready", artifact=line)
    from . import recordline_bridge

    rl = recordline_bridge.backend()
    if rl is not None:  # DubMaster's backend (feat/record-line) once merged; see recordline_bridge.py
        yield Event("finish", "start", message="Fitting and mixing the line (" + recordline_bridge.describe() + ")")
        # the original uploads (browser webm / wav at full quality), not the 24 kHz media.to_wav copy
        orig_takes = [str(t) for t in takes] if recording else [str(line)]
        out = recordline_bridge.process(rl, Path(video), line, start=float(start), work=work / "recordline",
                                        settings={"fit_seconds": fit_seconds, "duck": float(duck), "denoise": denoise,
                                                  "voice_level_db": float(voice_level_db), "lipsync": bool(lipsync),
                                                  "seed": seed, "takes": orig_takes,
                                                  **({"text": text.strip()} if text.strip() else {})})
        for step in out["steps"]:
            yield Event("finish", "progress", 0.5, step)
        final = Path(out["output"])  # the mp4 (final_mix_wav muxed); final_mix_wav stays in the dict
        if lipsync:
            line_s = out.get("line_seconds")
            if line_s is None:
                line_s = float(out["voice_end"]) - float(out["voice_start"])
            synced = yield from _lipsync_line(pipe, Path(video), final, Path(out["aligned_voice_wav"]),
                                              float(out["voice_start"]), float(line_s), work, record, results, seed)
            if synced is not None:
                final = synced
                out["steps"].append("lip re-sync (InfiniteTalk, mouth + jaw pasted back)")
        yield from _finish(work, record, final, "Redubbed clip ready",
                           {"redub": {"backend": rl.__name__, "steps": out["steps"], "takes": len(orig_takes),
                                      "final_mix_wav": str(out.get("final_mix_wav", ""))}}, "video")
        return
    yield Event("finish", "start", message="Cleaning, fitting and mixing the line (deepfilternet3, eq, light "
                                           "compression, short reverb, smooth duck)")
    s = redub_mod.RedubSettings(start=float(start), fit_seconds=fit_seconds, duck=float(duck), denoise=denoise,
                                voice_level_db=float(voice_level_db))
    try:
        report = redub_mod.redub(video, line, work / "redubbed.mp4", work / "redub", s)
    except ValueError as exc:
        raise InputError(str(exc)) from None
    for step in report["steps"]:
        yield Event("finish", "progress", 0.5, step)
    final = Path(report["output"])
    if lipsync:
        synced = yield from _lipsync_line(pipe, Path(video), final, Path(report["voice_stem"]), float(start),
                                          float(report["line_seconds"]), work, record, results, seed)
        if synced is not None:
            final = synced
            report["steps"].append("lip re-sync (InfiniteTalk, mouth + jaw pasted back)")
    yield from _finish(work, record, final, "Redubbed clip ready",
                       {"redub": {k: report[k] for k in ("steps", "start", "line_seconds", "final_lufs",
                                                         "final_true_peak_db", "static_gain_db")}}, "video")


def _lipsync_line(pipe: Pipeline, video: Path, mixed: Path, voice: Path, start: float, line_seconds: float,
                  work: Path, record: dict, results: dict, seed: int) -> Iterator[Event]:
    """Picture side of 'record your own line' (lipdub.py). Returns the re-synced video, or None (kept audio-only)."""
    from . import lipdub

    ldir = work / "lipdub"
    ldir.mkdir(exist_ok=True)
    size = media.video_size(video)
    fps = lipdub.probe_fps(video)
    win = lipdub.line_window(start, line_seconds, media.media_duration(video), fps)
    orig = lipdub.read_frames(video, win.f0, win.f1, size)
    track = lipdub.detect_faces_robust(orig)
    cov = lipdub.coverage(track)
    record["lipdub"] = {"window": [round(win.t0, 3), round(win.t1, 3)], "frames": [win.f0, win.f1],
                        "face_coverage": round(cov, 2), "note": "pending a real test"}
    if cov < lipdub.MIN_FACE_FRAMES:
        yield Event("lipsync", "skipped", 1.0, f"Lip re-sync skipped: a face was found in only {cov:.0%} of the "
                                               "line's frames (kept the new audio)")
        return None
    track = lipdub.fill_track(track)
    box = lipdub.crop_box(track, size)
    from PIL import Image

    x, y, side = box
    start_img = Image.fromarray(orig[0, y:y + side, x:x + side]).resize((lipdub.GEN_SIZE, lipdub.GEN_SIZE),
                                                                        Image.LANCZOS)
    start_png = ldir / "start.png"
    start_img.save(start_png)
    speech = ldir / "speech.wav"
    media.run_ffmpeg(["-ss", f"{win.t0:.3f}", "-i", str(voice), "-t", f"{win.t1 - win.t0:.3f}", "-ac", "1",
                      "-ar", "16000", str(speech)])
    m = tiers.models_for(tiers.get(tiers.resolve().key), pipe.settings.models)
    from .system import ram_preflight

    ram_note = ram_preflight()
    if ram_note:
        yield Event("lipsync", "progress", -1, ram_note)
    sg = graphs.lipsync_graph(
        "Close-up of the same person speaking to the camera with clear natural lip movements, head steady, the same "
        "face, hair and art style, static camera", models=m, audio_seconds=win.t1 - win.t0, width=lipdub.GEN_SIZE,
        height=lipdub.GEN_SIZE, fps=lipdub.GEN_FPS, seed=seed, audio_scale=lipdub.AUDIO_SCALE, prefix="mc/lipdub")
    yield Event("lipsync", "start", message=f"Re-syncing the lips for {win.t1 - win.t0:.1f}s (one InfiniteTalk job)")
    yield from pipe._job("lipdub", "lipsync", sg, {"avatar": start_png, "speech": speech}, work, results, record)
    gsize = media.video_size(results["lipdub"].path)
    gen = lipdub.read_frames(results["lipdub"].path, 0, 100000, gsize)
    if gsize != (lipdub.GEN_SIZE, lipdub.GEN_SIZE):
        import cv2

        gen = np.stack([cv2.resize(f, (lipdub.GEN_SIZE, lipdub.GEN_SIZE)) for f in gen]) if len(gen) else gen
    gtrack = lipdub.detect_faces_robust(gen)
    if not len(gen) or lipdub.coverage(gtrack) < lipdub.MIN_FACE_FRAMES:
        yield Event("lipsync", "skipped", 1.0, "Lip re-sync skipped: no steady face in the render (kept the new "
                                               "audio)")
        return None
    gtrack = lipdub.to_frame_coords(lipdub.fill_track(gtrack), box)
    pasted, rep = lipdub.paste_mouths(orig, lipdub.place_crop_renders(gen, box, size), track, gtrack, fps)
    record["lipdub"].update(rep)
    win_mp4 = lipdub.write_frames(pasted, ldir / "window.mp4", fps)
    total = media.count_frames(mixed)
    parts = []
    if win.f0 > 0:
        parts.append(media.cut_frames(mixed, ldir / "pre.mp4", 0, win.f0, int(round(fps))))
    parts.append(win_mp4)
    if win.f1 < total:
        parts.append(media.cut_frames(mixed, ldir / "post.mp4", win.f1, total - win.f1, int(round(fps))))
    silent = media.concat_frames(parts, ldir / "video.mp4", int(round(fps)))
    out = work / "redubbed_lipsync.mp4"
    media.run_ffmpeg(["-i", str(silent), "-i", str(mixed), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                      "-c:a", "copy", "-shortest", "-movflags", "+faststart", str(out)])
    yield Event("lipsync", "done", 1.0, "Lips re-synced to your line", artifact=out)
    return out


__all__ = ["run_tts", "run_voice_convert", "run_restyle", "run_upscale", "run_redub", "restyle_prompt"]
