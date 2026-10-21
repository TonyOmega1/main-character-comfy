"""Placeholder media for the mock backend (clearly labelled MOCK in-frame)."""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageOps

from .media import _font, prepare_photo, run_ffmpeg


def fake_avatar(prompt: str, photo: Path | None, dest: Path, size: tuple[int, int] = (1088, 1920)) -> Path:
    if photo is not None:
        prepare_photo(photo, dest, size)
        img = Image.open(dest).convert("RGB")
        p = prompt.lower()
        if "comic" in p:
            img = ImageOps.posterize(img, 3).filter(ImageFilter.EDGE_ENHANCE_MORE)
        elif "anime" in p:
            img = ImageOps.posterize(img.filter(ImageFilter.SMOOTH_MORE), 4)
        elif "3d" in p or "clay" in p:
            img = img.filter(ImageFilter.GaussianBlur(3)).filter(ImageFilter.DETAIL)
    else:
        seed = int(hashlib.sha1(prompt.encode()).hexdigest()[:6], 16)
        c1 = ((seed >> 16) & 255, (seed >> 8) & 255, seed & 255)
        img = Image.new("RGB", size, c1)
        d = ImageDraw.Draw(img)
        w, h = size
        d.ellipse((w * 0.25, h * 0.18, w * 0.75, h * 0.48), fill=(250, 220, 190))  # head
        d.rectangle((w * 0.15, h * 0.5, w * 0.85, h), fill=tuple(255 - c for c in c1))  # body
        d.ellipse((w * 0.37, h * 0.29, w * 0.43, h * 0.32), fill="black")
        d.ellipse((w * 0.57, h * 0.29, w * 0.63, h * 0.32), fill="black")
        d.arc((w * 0.42, h * 0.36, w * 0.58, h * 0.42), 10, 170, fill="black", width=8)
    d = ImageDraw.Draw(img)
    d.text((30, 30), "MOCK AVATAR", font=_font(48), fill="white", stroke_width=3, stroke_fill="black")
    img.save(dest)
    return dest


def fake_speech(text: str, dest: Path) -> Path:
    # ~2.6 words per second, minimum 2 s; a soft modulated tone stands in for speech.
    seconds = max(2.0, len(text.split()) / 2.6)
    run_ffmpeg([
        "-f", "lavfi", "-i", f"sine=frequency=180:sample_rate=24000:duration={seconds:.2f}",
        "-af", "tremolo=f=4:d=0.8,volume=0.4", "-ac", "1", str(dest),
    ])
    return dest


def fake_voice_change(src: Path, dest: Path) -> Path:
    run_ffmpeg(["-i", str(src), "-af", "asetrate=24000*1.25,aresample=24000,atempo=0.8", "-ac", "1", str(dest)])
    return dest


def fake_talking(avatar: Path, speech: Path, frames: int, dest: Path, size: tuple[int, int] = (480, 832)) -> Path:
    w, h = size
    seconds = frames / 25
    run_ffmpeg([
        "-loop", "1", "-i", str(avatar), "-i", str(speech),
        "-filter_complex",
        f"[0:v]scale={w * 2}:{h * 2},zoompan=z='1.03+0.02*sin(on/6)':x='iw/2-(iw/zoom/2)':"
        f"y='ih/2-(ih/zoom/2)+6*sin(on/3)':d=1:s={w}x{h}:fps=25[v]",
        "-map", "[v]", "-map", "1:a", "-t", f"{seconds:.2f}", "-af", "apad",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(dest),
    ])
    return dest


def fake_smooth(video: Path, dest: Path) -> Path:
    run_ffmpeg(["-i", str(video), "-vf", "fps=50", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "copy", str(dest)])
    return dest


def fake_motion(avatar: Path, frames: int, fps: int, dest: Path, size: tuple[int, int] = (480, 832)) -> Path:
    """Stand-in for Wan 2.2 I2V: a slow push-in with a little sway, no audio."""
    w, h = size
    run_ffmpeg([
        "-loop", "1", "-i", str(avatar),
        "-vf", f"scale={w * 2}:{h * 2},zoompan=z='1+0.0025*on':x='iw/2-(iw/zoom/2)+8*sin(on/5)':"
               f"y='ih/2-(ih/zoom/2)':d=1:s={w}x{h}:fps={fps}",
        "-frames:v", str(frames), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(dest),
    ])
    return dest




def fake_animate(mode: str, clip: Path, avatar: Path, out_frames: int, lead: int, fps: int, dest: Path) -> Path:
    """Stand-in for a Wan2.2 Animate job, with the real job's frame accounting:
    the carried ``lead`` frames are dropped and the output is ``out_frames`` long
    (the last pose is held, like the padded final window).

    * replace: your clip, desaturated, with your photo pasted in a corner
    * move:    your new-scene still with a gentle sway, plus a small edge-detected
               "pose" thumbnail (stands in for the SDPose skeleton)"""
    from .media import video_size

    w, h = video_size(clip)
    hold = f"trim=start_frame={lead},setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop={out_frames}"
    if mode == "replace":
        graph = f"[0:v]{hold},hue=s=0.6[b];[1:v]scale=144:-2[a];[b][a]overlay=W-w-12:12[v]"
        args = ["-i", str(clip), "-i", str(avatar)]
    else:
        graph = (f"[1:v]scale={w * 2}:{h * 2}:force_original_aspect_ratio=increase,crop={w * 2}:{h * 2},"
                 f"zoompan=z='1.02':x='iw/2-(iw/zoom/2)+10*sin(on/4)':y='ih/2-(ih/zoom/2)+6*sin(on/3)':d=1:"
                 f"s={w}x{h}:fps={fps}[s];"
                 f"[0:v]{hold},edgedetect=low=0.1:high=0.3,scale=-2:{h // 4}[p];[s][p]overlay=12:H-h-12[v]")
        args = ["-i", str(clip), "-loop", "1", "-i", str(avatar)]
    run_ffmpeg([*args, "-filter_complex", graph, "-map", "[v]", "-r", str(fps), "-frames:v", str(out_frames),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(dest)])
    return dest


def fake_people_masks(frame: Path, out_dir: Path, stage: str, n: int = 2) -> list[Path]:
    """Stand-in for the SAM 3.1 people scan: ``n`` upright white blobs, deliberately saved right-to-left
    (SAM's order is arbitrary, so MC must sort them itself)."""
    from PIL import Image, ImageDraw

    w, h = Image.open(frame).size
    paths = []
    for k, i in enumerate(reversed(range(max(1, n)))):
        img = Image.new("RGB", (w, h), "black")
        cx = int(w * (i + 0.5) / max(1, n))
        bw = max(8, w // (3 * max(1, n)))
        ImageDraw.Draw(img).ellipse((cx - bw, int(h * 0.15), cx + bw, int(h * 0.95)), fill="white")
        p = Path(out_dir) / f"{stage}_{k}.png"
        img.save(p)
        paths.append(p)
    return paths


def fake_cutout(avatar: Path, dest: Path, size: tuple[int, int] = (480, 832)) -> Path:
    """Stand-in for the SAM 3.1 reference cut-out: the reference shrunk onto a neutral grey card."""
    ref = Image.open(avatar).convert("RGB")
    ref.thumbnail((int(size[0] * 0.62), size[1]))
    card = Image.new("RGB", size, (128, 128, 128))
    card.paste(ref, ((size[0] - ref.width) // 2, size[1] - ref.height))
    ImageDraw.Draw(card).text((12, 12), "MOCK CUT-OUT", font=_font(28), fill="white", stroke_width=2,
                              stroke_fill="black")
    card.save(dest)
    return dest


def fake_erase(clip: Path, frames: int, fps: int, dest: Path) -> Path:
    """Stand-in for the VOID prop-erase pre-pass: the clip, slightly blurred, labelled."""
    run_ffmpeg(["-i", str(clip), "-vf", "gblur=sigma=1.5", "-r", str(fps), "-frames:v", str(frames),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(dest)])
    return dest


def fake_upscale(clip: Path, width: int, height: int, fps: int, dest: Path) -> Path:
    """Stand-in for the SeedVR2 restore + upscale pass: a plain lanczos resize to the target size."""
    run_ffmpeg(["-i", str(clip), "-an", "-vf", f"scale={width}:{height}:flags=lanczos", "-r", str(fps),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(dest)])
    return dest
