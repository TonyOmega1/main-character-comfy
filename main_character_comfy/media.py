"""Local media helpers (no models): input prep, captions, finishing encode.

Uses the static ffmpeg build shipped by ``imageio-ffmpeg`` so Windows users don't
need to install ffmpeg separately; falls back to ``ffmpeg`` on PATH.
"""

from __future__ import annotations

import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # pragma: no cover - fallback path
        exe = shutil.which("ffmpeg")
        if not exe:
            raise RuntimeError("ffmpeg not found: pip install imageio-ffmpeg (or put ffmpeg on PATH)")
        return exe


def run_ffmpeg(args: list[str]) -> None:
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {proc.stderr.strip()[-1500:]}")


_DUR_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


def media_duration(path: str | Path) -> float:
    """Duration in seconds, parsed from ffmpeg's stream summary."""
    proc = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    m = _DUR_RE.search(proc.stderr)
    if not m:
        raise RuntimeError(f"could not read duration of {path}")
    h, mnt, s = m.groups()
    return int(h) * 3600 + int(mnt) * 60 + float(s)


def has_video_stream(path: str | Path) -> bool:
    proc = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    return "Video:" in proc.stderr


def has_audio_stream(path: str | Path) -> bool:
    proc = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    return "Audio:" in proc.stderr


# --------------------------------------------------------------------------- #
# Input preparation
# --------------------------------------------------------------------------- #
def prepare_voice(src: str | Path, dst: str | Path, max_seconds: float = 30.0) -> float:
    """Any audio/video file -> mono 24 kHz WAV, trimmed to ``max_seconds``.
    Returns the resulting duration."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        [
            "-i", str(src), "-vn", "-t", f"{max_seconds:.2f}",
            "-af", "highpass=f=70,silenceremove=start_periods=1:start_threshold=-50dB",
            "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(dst),
        ]
    )
    return media_duration(dst)


def to_wav(src: str | Path, dst: str | Path) -> Path:
    dst = Path(dst)
    run_ffmpeg(["-i", str(src), "-vn", "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(dst)])
    return dst


NEUTRAL_BG = (200, 200, 200)  # light neutral grey behind transparent avatars (no colour cast, no halo)


def has_transparency(img: Image.Image, min_fraction: float = 0.02) -> bool:
    """True when at least ``min_fraction`` of the pixels are (partly) transparent."""
    if img.mode == "P" and "transparency" in img.info:
        img = img.convert("RGBA")
    if img.mode not in ("RGBA", "LA", "PA"):
        return False
    a = img.getchannel("A").resize((128, 128))
    hist = a.histogram()
    return sum(hist[:250]) / (128 * 128) >= min_fraction


def load_rgb(src: str | Path, bg: tuple[int, int, int] = NEUTRAL_BG) -> Image.Image:
    """Open any upload as RGB: EXIF-rotated, and transparent PNG/WebP avatars flattened onto a
    neutral grey (a plain ``convert("RGB")`` turns transparency black or noisy)."""
    img = ImageOps.exif_transpose(Image.open(src))
    if img.mode == "P" and "transparency" in img.info:
        img = img.convert("RGBA")
    if img.mode in ("RGBA", "LA", "PA"):
        rgba = img.convert("RGBA")
        flat = Image.new("RGB", rgba.size, bg)
        flat.paste(rgba, mask=rgba.getchannel("A"))
        return flat
    return img.convert("RGB")


AVATAR_MAX_COLOURS = 600  # 5-bit colours in a 160 px thumbnail: flat drawings ~100-500, photos well above


def looks_like_avatar(src: str | Path) -> bool:
    """Cheap 'is this a drawn/rendered character rather than a camera photo?' guess for
    ``subject='auto'``: a transparent background, or very few distinct colours (flat
    cartoon / anime / pixel art / vector mascots). Smooth 3D renders can read as photos;
    the explicit 'my avatar' choice always wins."""
    img = ImageOps.exif_transpose(Image.open(src))
    if has_transparency(img):
        return True
    small = img.convert("RGB")
    small.thumbnail((160, 160))
    q = small.point(lambda v: v >> 3)  # 5 bits per channel: ignores JPEG noise
    return len(set(q.getdata())) < AVATAR_MAX_COLOURS


def is_flat_art(src: str | Path) -> bool:
    """Flat illustration / cel-shaded / vector look (few distinct colours), as opposed to a photo or a smooth
    3D render. Avatar mode uses it to ask Wan for a flat, illustrated look instead of a rendered one."""
    try:
        small = load_rgb(src)
    except Exception:
        return False
    small.thumbnail((160, 160))
    q = small.point(lambda v: v >> 3)
    return len(set(q.getdata())) < AVATAR_MAX_COLOURS


def prepare_photo(src: str | Path, dst: str | Path, size: tuple[int, int] = (1088, 1920),
                  keep_whole: bool = False) -> Path:
    """Orient, crop to 9:16 (biased toward the top where faces usually are) and resize.
    ``keep_whole`` (avatar mode) never crops: the whole image is fitted onto a neutral
    background, so wide, square or tiny avatars keep their full design."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img = load_rgb(src)
    tw, th = size
    if keep_whole:
        canvas = Image.new("RGB", size, _edge_colour(img))
        fg = ImageOps.contain(img, (int(tw * 0.92), int(th * 0.92)), Image.LANCZOS)
        canvas.paste(fg, ((tw - fg.width) // 2, (th - fg.height) // 2))
        canvas.save(dst, "PNG")
        return dst
    target = tw / th
    w, h = img.size
    if w / h > target:  # too wide: crop sides, centered
        nw = int(h * target)
        left = (w - nw) // 2
        img = img.crop((left, 0, left + nw, h))
    elif w / h < target:  # too tall: crop bottom more than top
        nh = int(w / target)
        top = int((h - nh) * 0.25)
        img = img.crop((0, top, w, top + nh))
    img = img.resize(size, Image.LANCZOS)
    img.save(dst, "PNG")
    return dst


def _edge_colour(img: Image.Image) -> tuple[int, int, int]:
    """Average colour of the image border: the padding blends into the avatar's own background."""
    w, h = img.size
    strips = [img.crop((0, 0, w, max(1, h // 50))), img.crop((0, h - max(1, h // 50), w, h)),
              img.crop((0, 0, max(1, w // 50), h)), img.crop((w - max(1, w // 50), 0, w, h))]
    px = [p for st in strips for p in st.resize((16, 16)).getdata()]
    return tuple(int(sum(c[i] for c in px) / len(px)) for i in range(3))  # type: ignore[return-value]


def fit_photo(src: str | Path, dst: str | Path, size: tuple[int, int]) -> Path:
    """Fit the whole photo inside ``size`` (no cropping) on a blurred copy of
    itself. Used for Swap Me In references, where cutting off the body hurts.
    Transparent avatars are flattened onto neutral grey first."""
    from PIL import ImageFilter

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img = load_rgb(src)
    bg = ImageOps.fit(img, size, Image.LANCZOS).filter(ImageFilter.GaussianBlur(28))
    fg = ImageOps.contain(img, size, Image.LANCZOS)
    bg.paste(fg, ((size[0] - fg.width) // 2, (size[1] - fg.height) // 2))
    bg.save(dst, "PNG")
    return dst


@dataclass
class PreparedClip:
    video: Path  # silent, width x height, constant fps, exactly ``frames`` frames
    audio: Path | None  # original soundtrack (wav), if the clip had one
    width: int
    height: int
    frames: int
    fps: int
    seconds: float


def video_size(path: str | Path) -> tuple[int, int]:
    proc = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    m = re.search(r"Video:.*?\b(\d{2,5})x(\d{2,5})\b", proc.stderr)
    if not m:
        raise RuntimeError(f"no video stream in {path}")
    w, h = int(m.group(1)), int(m.group(2))
    rot = re.search(r"rotat(?:e|ion)(?:\s+of)?\s*:?\s*(-?\d+)", proc.stderr)
    if rot and abs(int(rot.group(1))) in (90, 270):
        w, h = h, w
    return w, h


ROTATE_FILTERS = {0: "", 90: "transpose=1,", 180: "hflip,vflip,", 270: "transpose=2,"}


def rotate_filter(rotate: int) -> str:
    """Extra clockwise turn on top of the phone's own rotation tag (ffmpeg already applies that one on decode)."""
    r = int(rotate or 0) % 360
    if r not in ROTATE_FILTERS:
        raise ValueError(f"rotate must be 0, 90, 180 or 270 (got {rotate})")
    return ROTATE_FILTERS[r]


def prepare_clip(src: str | Path, out_dir: str | Path, *, start: float = 0.0, max_seconds: float = 10.0,
                 fps: int = 16, portrait: tuple[int, int] = (480, 832), rotate: int = 0) -> PreparedClip:
    """Bring-your-own clip -> what Wan2.2 Animate wants: center-cropped to 480x832
    (portrait/square sources) or 832x480 (landscape), constant ``fps``, at most
    ``max_seconds``. The soundtrack is kept separately and re-muxed at the end.
    Phone clips are auto-rotated (ffmpeg honours the rotation tag); ``rotate`` turns the picture a further
    90/180/270 degrees clockwise for clips whose tag is missing or wrong."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    turn = rotate_filter(rotate)
    sw, sh = video_size(src)
    if int(rotate or 0) % 180 == 90:
        sw, sh = sh, sw
    w, h = portrait if sh >= sw else (portrait[1], portrait[0])
    total = media_duration(src)
    seconds = max(0.5, min(max_seconds, total - start))
    frames = max(5, int(seconds * fps))
    video = out_dir / "clip.mp4"
    run_ffmpeg([
        "-ss", f"{start:.3f}", "-i", str(src), "-an",
        "-vf", f"{turn}fps={fps},scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},setsar=1",
        "-frames:v", str(frames), "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(video),
    ])
    audio = None
    if has_audio_stream(src):
        audio = out_dir / "clip_audio.wav"
        run_ffmpeg(["-ss", f"{start:.3f}", "-i", str(src), "-vn", "-t", f"{frames / fps:.3f}", "-ac", "2",
                    "-ar", "44100", "-c:a", "pcm_s16le", str(audio)])
    return PreparedClip(video, audio, w, h, frames, fps, frames / fps)


def cut_frames(src: str | Path, dest: str | Path, first: int, count: int, fps: int = 16) -> Path:
    """Frame-exact cut [first, first+count) of a constant-fps silent clip (re-encoded, near lossless)."""
    run_ffmpeg([
        "-i", str(src), "-an",
        # by frame INDEX, then re-time to exactly ``fps`` (robust to odd input timestamps)
        "-vf", f"trim=start_frame={int(first)}:end_frame={int(first) + int(count)},setpts=N/({fps}*TB)",
        "-r", str(fps), "-frames:v", str(int(count)), "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", str(dest),
    ])
    return Path(dest)


def count_frames(path: str | Path) -> int:
    proc = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"],
                          capture_output=True, text=True)
    found = re.findall(r"frame=\s*(\d+)", proc.stderr)
    if not found:
        raise RuntimeError(f"can't count frames in {path}")
    return int(found[-1])


def concat_frames(clips: list[Path], dest: str | Path, fps: int = 16) -> Path:
    """Butt-join chunk renders (each already trimmed to its new frames) into one clip.
    The seam is handled at render time (motion carried into the next chunk), so a
    hard cut here lands on continuous motion."""
    if len(clips) == 1:
        run_ffmpeg(["-i", str(clips[0]), "-an", "-vf", f"setpts=N/({fps}*TB)", "-r", str(fps), "-c:v", "libx264",
                    "-crf", "14", "-pix_fmt", "yuv420p", str(dest)])
        return Path(dest)
    args: list[str] = []
    for c in clips:
        args += ["-i", str(c)]
    chain = "".join(f"[{i}:v]setsar=1[v{i}];" for i in range(len(clips)))
    chain += "".join(f"[v{i}]" for i in range(len(clips))) + f"concat=n={len(clips)}:v=1:a=0,setpts=N/({fps}*TB)[out]"
    run_ffmpeg([*args, "-filter_complex", chain, "-map", "[out]", "-r", str(fps), "-c:v", "libx264", "-crf", "14",
                "-pix_fmt", "yuv420p", str(dest)])
    return Path(dest)


# --------------------------------------------------------------------------- #
# Captions
# --------------------------------------------------------------------------- #
@dataclass
class Caption:
    text: str
    start: float
    end: float
    speaker: str = ""  # "A"/"B" in dialogue mode (colours the caption)


def split_caption_chunks(script: str, max_chars: int = 26, max_words: int = 5) -> list[str]:
    words = script.replace("\n", " ").split()
    chunks: list[str] = []
    cur: list[str] = []
    for w in words:
        trial = " ".join(cur + [w])
        if cur and (len(trial) > max_chars or len(cur) >= max_words):
            chunks.append(" ".join(cur))
            cur = [w]
        else:
            cur.append(w)
        if w[-1:] in ".!?;:" and cur:
            chunks.append(" ".join(cur))
            cur = []
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def time_captions(script: str, duration: float, lead_in: float = 0.15, tail: float = 0.25) -> list[Caption]:
    """Spread caption chunks over the speech, weighted by character count plus a
    small pause after sentence punctuation. Approximate (no forced alignment)."""
    chunks = split_caption_chunks(script)
    if not chunks or duration <= 0:
        return []
    weights = [len(c) + (6 if c[-1:] in ".!?" else 2 if c[-1:] in ",;:" else 0) for c in chunks]
    usable = max(0.1, duration - lead_in - tail)
    t = lead_in
    caps: list[Caption] = []
    total = sum(weights)
    for chunk, wt in zip(chunks, weights):
        span = usable * wt / total
        caps.append(Caption(chunk, round(t, 3), round(t + span, 3)))
        t += span
    caps[-1].end = round(min(duration, caps[-1].end + tail), 3)
    return caps


def _font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/seguibl.ttf",  # Segoe UI Black
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "DejaVuSans-Bold.ttf",
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


SPEAKER_COLORS = {"": "#FFE14D", "A": "#FFE14D", "B": "#5CE1FF"}


def render_caption_png(text: str, path: Path, width: int, height: int, highlight: str = "#FFE14D",
                       y_frac: float = 0.70) -> Path:
    """Full-frame transparent PNG with a bold, outlined caption at ``y_frac`` height."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    size = max(28, int(width * 0.085))
    font = _font(size)
    text = text.upper()
    # wrap to two lines max if needed
    lines = [text]
    if draw.textlength(text, font=font) > width * 0.88 and " " in text:
        words = text.split()
        mid = math.ceil(len(words) / 2)
        lines = [" ".join(words[:mid]), " ".join(words[mid:])]
    stroke = max(3, size // 9)
    y = int(height * y_frac)
    for i, line in enumerate(lines):
        tw = draw.textlength(line, font=font)
        x = (width - tw) / 2
        fill = highlight if i == 0 else "white"
        draw.text((x, y), line, font=font, fill=fill, stroke_width=stroke, stroke_fill="black")
        y += int(size * 1.15)
    img.save(path)
    return path


WATERMARK = "made with MC: Main Character · 0 credits"


def render_watermark_png(path: Path, width: int, height: int, text: str = WATERMARK) -> Path:
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = _font(max(14, width // 40))
    tw = draw.textlength(text, font=font)
    draw.text((width - tw - width * 0.04, height * 0.035), text, font=font, fill=(255, 255, 255, 170),
              stroke_width=2, stroke_fill=(0, 0, 0, 140))
    img.save(path)
    return path


FONTS_DIR = Path(__file__).parent / "assets" / "fonts"
ERAS = {
    "classic": "Classic (white outline, 2000s macro)",
    "deep_fried": "Deep fried",
    "vaporwave": "Vaporwave",
    "early_web": "Early web (pixelated)",
    "tiktok": "TikTok text box",
}


def _meme_font(size: int) -> ImageFont.ImageFont:
    # Anton (SIL OFL 1.1, bundled) is the open stand-in for the classic condensed meme face.
    for c in (FONTS_DIR / "Anton-Regular.ttf", "C:/Windows/Fonts/impact.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(str(c), size)
        except OSError:
            continue
    return _font(size)


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_w: float) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for wd in words:
        trial = f"{cur} {wd}".strip()
        if cur and draw.textlength(trial, font=font) > max_w:
            lines.append(cur)
            cur = wd
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


# (top anchor, bottom anchor, bottom grows up?, size scale, max lines) per caption layout
CAPTION_SLOTS = {
    "classic": (0.06, 0.80, True, 1.0, 3),
    "edges": (0.025, 0.84, True, 0.82, 2),  # crash zooms: keep the middle (the face) clear
    "top": (0.04, None, False, 0.82, 2),  # body/action in the lower half: stack both lines at the top
}


def render_meme_caption_png(top: str, bottom: str, path: Path, width: int, height: int, era: str = "classic",
                            layout: str = "classic") -> Path:
    """Meme caption layer. ``classic``: white all-caps condensed type with a heavy
    black outline; ``vaporwave``: wide-spaced pastel caps; ``tiktok``: sentence-case
    text on white boxes. The bottom caption sits above the area that TikTok /
    Reels / Shorts cover with their buttons. ``layout`` (see CAPTION_SLOTS) moves the
    captions off the subject for formats where the classic slots would cover it."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    top_y, bottom_y, bottom_up, scale, max_lines = CAPTION_SLOTS.get(layout, CAPTION_SLOTS["classic"])
    slots = [(top, top_y, False), (bottom, bottom_y, bottom_up)]
    if bottom_y is None:  # stacked: the bottom line goes right under the top block
        slots = [(" \n ".join(t for t in ((top or "").strip(), (bottom or "").strip()) if t), top_y, False)]
    for text, anchor_y, grow_up in slots:
        text = (text or "").strip()
        if not text:
            continue
        if era == "tiktok":
            size = int(width * 0.058 * max(scale, 0.9))
            font = _font(size)
            lines = [ln for part in text.split(" \n ") for ln in _wrap(draw, part, font, width * 0.80)]
            line_h = int(size * 1.35)
            y = int(height * anchor_y) - (line_h * len(lines) if grow_up else 0)
            for line in lines:
                tw = draw.textlength(line, font=font)
                x = (width - tw) / 2
                pad = size * 0.35
                draw.rounded_rectangle((x - pad, y - pad * 0.4, x + tw + pad, y + size + pad * 0.6), radius=pad,
                                       fill="white")
                draw.text((x, y), line, font=font, fill="black")
                y += line_h
            continue
        text = text.upper()
        track_k = 0.28 if era == "vaporwave" else 0.0  # "A E S T H E T I C" letter spacing
        size = int(width * (0.085 if track_k else 0.105) * scale)
        parts = text.split(" \n ")
        limit = max_lines * len(parts)
        while True:
            font = _meme_font(size)
            lines = [ln for part in parts for ln in _wrap(draw, part, font, width * 0.92 / (1 + track_k * 1.6))]
            if len(lines) <= limit or size <= 24:
                break
            size = int(size * 0.88)
        line_h = int(size * 1.12)
        y = int(height * anchor_y) - (line_h * len(lines) if grow_up else 0)
        stroke = max(3, size // 12)
        for line in lines:
            tw = draw.textlength(line, font=font)
            x = (width - tw) / 2
            if era == "vaporwave":
                track = size * track_k
                tw = sum(draw.textlength(ch, font=font) for ch in line) + track * (len(line) - 1)
                cx = (width - tw) / 2
                for ch in line:
                    draw.text((cx + size * 0.06, y + size * 0.06), ch, font=font, fill="#00e5ff")
                    draw.text((cx, y), ch, font=font, fill="#ff9de6", stroke_width=max(2, stroke // 2),
                              stroke_fill="#5b2a86")
                    cx += draw.textlength(ch, font=font) + track
            else:
                draw.text((x, y), line, font=font, fill="white", stroke_width=stroke, stroke_fill="black")
            y += line_h
    img.save(path)
    return path


def era_filter(era: str, width: int, height: int) -> str:
    """ffmpeg video look for a 'Meme Era' (applied before the caption overlay)."""
    if era == "deep_fried":
        return ",eq=saturation=2.6:contrast=1.6:brightness=0.03,unsharp=7:7:2.5,noise=alls=18:allf=t"
    if era == "vaporwave":
        return ",hue=h=-25:s=1.3,colorbalance=rs=0.25:gs=-0.1:bs=0.3"
    if era == "early_web":
        return f",scale={max(64, width // 8)}:-2:flags=bilinear,scale={width}:{height}:flags=neighbor,eq=saturation=1.2"
    return ""


def era_image(img: Image.Image, era: str) -> Image.Image:
    """Same looks as :func:`era_filter`, for the instant still."""
    from PIL import ImageEnhance, ImageFilter

    if era == "deep_fried":
        img = ImageEnhance.Color(img).enhance(2.6)
        img = ImageEnhance.Contrast(img).enhance(1.6)
        img = img.filter(ImageFilter.UnsharpMask(radius=4, percent=250, threshold=1))
        return img
    if era == "vaporwave":
        r, g, b = img.split()
        r = r.point(lambda v: min(255, int(v * 1.2 + 20)))
        b = b.point(lambda v: min(255, int(v * 1.25 + 30)))
        g = g.point(lambda v: int(v * 0.9))
        return Image.merge("RGB", (r, g, b))
    if era == "early_web":
        w, h = img.size
        return img.resize((max(64, w // 8), max(64, h // 8)), Image.BILINEAR).resize((w, h), Image.NEAREST)
    return img


def make_still_meme(scene: str | Path, out_path: str | Path, *, top: str, bottom: str, era: str = "classic",
                    watermark: bool = True, size: tuple[int, int] = (1080, 1920), work_dir: str | Path | None = None,
                    caption_layout: str = "classic") -> Path:
    """The instant result: you-in-the-scene + captions + era look, as a PNG."""
    out_path = Path(out_path)
    work = Path(work_dir or out_path.parent / "_still")
    work.mkdir(parents=True, exist_ok=True)
    img = ImageOps.fit(Image.open(scene).convert("RGB"), size, Image.LANCZOS)
    img = era_image(img, era).convert("RGBA")
    cap = Image.open(render_meme_caption_png(top, bottom, work / "still_caption.png", *size, era=era,
                                             layout=caption_layout))
    img.alpha_composite(cap)
    if watermark:
        img.alpha_composite(Image.open(render_watermark_png(work / "still_wm.png", *size)))
    img.convert("RGB").save(out_path)
    return out_path


GIF_MAX_BYTES = 3 * 1024 * 1024  # group chats / Discord free tier: keep reaction GIFs under ~3 MB


def make_gif(video: str | Path, out_path: str | Path, *, width: int = 320, fps: int = 10, max_seconds: float = 6.0,
             max_bytes: int = GIF_MAX_BYTES) -> Path:
    """Small looping GIF for reactions / group chats (palette-optimised).

    Busy motion (punch-ins, shakes) can blow past ``max_bytes`` at 320 px / 128 colours (measured 4.4 MB on the
    RTX 5080 dramatic_zoom run), so step down colours, fps and width until it fits.
    """
    out_path = Path(out_path)
    ladder = [(width, fps, 128), (width, fps, 64), (int(width * 0.85) // 2 * 2, max(8, fps - 2), 64),
              (int(width * 0.75) // 2 * 2, 8, 48)]
    for w, f, colors in ladder:
        run_ffmpeg([
            "-t", f"{max_seconds:.2f}", "-i", str(video),
            "-filter_complex", f"fps={f},scale={w}:-2:flags=lanczos,split[a][b];[a]palettegen=max_colors={colors}[p];"
                               f"[b][p]paletteuse=dither=bayer:bayer_scale=4",
            "-loop", "0", str(out_path),
        ])
        if not max_bytes or out_path.stat().st_size <= max_bytes:
            break
    return out_path


def effect_filter(effect: str, at: float, width: int, height: int, fps: int) -> str:
    """ffmpeg filter (applied after the 9:16 upscale) for a recipe's post effect."""
    if effect == "punch_in":  # crash zoom toward the upper third (where faces are)
        z = f"if(lt(it,{at:.2f}),1,min(1.55,1+(it-{at:.2f})*4))"
        return (f",zoompan=z='{z}':d=1:x='iw/2-(iw/zoom/2)':y='ih/3-(ih/zoom/3)':s={width}x{height}:fps={fps}")
    if effect == "shake":
        on = f"between(t,{at:.2f},{at + 0.7:.2f})"
        return (f",crop=iw*0.94:ih*0.94:x='(iw-ow)/2+if({on},(iw-ow)/2*sin(t*71),0)':"
                f"y='(ih-oh)/2+if({on},(ih-oh)/2*cos(t*53),0)',scale={width}:{height}:flags=lanczos")
    return ""


# --------------------------------------------------------------------------- #
# Finishing: upscale, optional motion smoothing, captions, mux, trim
# --------------------------------------------------------------------------- #
def finish_video(
    video: str | Path,
    out_path: str | Path,
    *,
    audio: str | Path | None = None,
    duration: float | None = None,
    captions: list[Caption] | None = None,
    width: int = 1080,
    height: int = 1920,
    ffmpeg_smooth_fps: int | None = None,
    watermark: bool = True,
    work_dir: str | Path | None = None,
    meme_top: str = "",
    meme_bottom: str = "",
    effect: str = "none",
    effect_at: float = 0.0,
    fps: int = 16,
    era: str = "classic",
    caption_layout: str = "classic",
) -> Path:
    """Final encode for sharing: lanczos upscale to ``width``x``height``,
    optional ffmpeg motion interpolation (fallback when RIFE isn't installed),
    burned-in captions, small watermark, H.264 + AAC, trimmed to the speech."""
    out_path = Path(out_path)
    work = Path(work_dir or out_path.parent / "_finish")
    work.mkdir(parents=True, exist_ok=True)
    args: list[str] = ["-i", str(video)]
    overlays: list[tuple[Path, float | None, float | None]] = []
    if meme_top.strip() or meme_bottom.strip():
        overlays.append((render_meme_caption_png(meme_top, meme_bottom, work / "meme_caption.png", width, height, era=era,
                                                 layout=caption_layout),
                         None, None))
    for i, cap in enumerate(captions or []):
        color = SPEAKER_COLORS.get(cap.speaker, SPEAKER_COLORS[""])
        overlays.append(
            (render_caption_png(cap.text, work / f"cap_{i:03d}.png", width, height, color,
                                y_frac=0.56 if meme_bottom.strip() else 0.70), cap.start, cap.end)
        )
    if watermark:
        overlays.append((render_watermark_png(work / "wm.png", width, height), None, None))
    audio_idx = None
    for png, _, _ in overlays:
        args += ["-i", str(png)]
    silent = audio is None and not has_audio_stream(video)
    if audio is not None:
        args += ["-i", str(audio)]
        audio_idx = 1 + len(overlays)
    elif silent:  # platforms handle clips with an audio track more reliably
        args += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"]
        audio_idx = 1 + len(overlays)

    chain = "[0:v]"
    if effect == "boomerang":
        chain = "[0:v]split[bf][bb];[bb]reverse[br];[bf][br]concat=n=2:v=1:a=0[bm];[bm]"
        if duration:
            duration = duration * 2
    if ffmpeg_smooth_fps:  # interpolate at native size first: far cheaper than at 1080p
        chain += f"minterpolate=fps={ffmpeg_smooth_fps}:mi_mode=mci:mc_mode=aobmc:vsbmc=1,"
    chain += f"scale={width}:{height}:flags=lanczos{effect_filter(effect, effect_at, width, height, fps)}{era_filter(era, width, height)},setsar=1[v0]"
    filters = [chain]
    last = "v0"
    for k, (_, start, end) in enumerate(overlays, start=1):
        enable = f":enable='between(t,{start:.3f},{end:.3f})'" if start is not None else ""
        filters.append(f"[{last}][{k}:v]overlay=0:0{enable}[v{k}]")
        last = f"v{k}"
    args += ["-filter_complex", ";".join(filters), "-map", f"[{last}]"]
    if audio_idx is not None:
        args += ["-map", f"{audio_idx}:a"]
        if audio is not None and duration:  # a short soundtrack is padded with silence, a long one is cut at -t
            args += ["-af", "apad"]
    else:
        args += ["-map", "0:a?"]
    if duration:
        args += ["-t", f"{duration:.3f}"]
    elif silent:
        args += ["-shortest"]
    args += [
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out_path),
    ]
    run_ffmpeg(args)
    return out_path


def poster_frame(video: str | Path, out_path: str | Path, at: float = 0.5) -> Path:
    run_ffmpeg(["-ss", f"{at:.2f}", "-i", str(video), "-frames:v", "1", str(out_path)])
    return Path(out_path)


def open_folder(path: str | Path) -> None:  # pragma: no cover - UI nicety
    p = str(path)
    if sys.platform.startswith("win"):
        subprocess.Popen(["explorer", p])
    elif sys.platform == "darwin":
        subprocess.Popen(["open", p])
    else:
        subprocess.Popen(["xdg-open", p])


# --------------------------------------------------------------------------- #
# Multi-character Swap Me In: number the people, draw the preview
# --------------------------------------------------------------------------- #
@dataclass
class Person:
    mask: Path  # white-on-black first-frame mask (clip resolution)
    box: tuple[int, int, int, int]  # x0, y0, x1, y1
    cx: float  # centroid x (0..1)
    cy: float
    area: float  # fraction of the frame


def grab_frame(video: str | Path, dst: str | Path, frame: int = 0, fps: int = 16) -> Path:
    """One frame of ``video`` as PNG (frame index at ``fps``)."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(["-i", str(video), "-vf", f"select=eq(n\\,{int(frame)})", "-frames:v", "1", "-vsync", "0", str(dst)])
    return dst


def order_people(masks: list[Path], size: tuple[int, int] | None = None, min_area: float = 0.004,
                 overlap: float = 0.6) -> list[Person]:
    """Turn the scan's masks into people numbered left to right (by centroid). Masks smaller than
    ``min_area`` of the frame (a hand, a poster face) are dropped; masks are resized to ``size``."""
    import numpy as np

    people: list[Person] = []
    bits: list = []
    for m in masks:
        img = Image.open(m).convert("L")
        if size and img.size != tuple(size):
            img = img.resize(tuple(size), Image.NEAREST)
            img.save(m)
        a = np.asarray(img) > 127
        frac = float(a.mean())
        if frac < min_area:
            continue
        ys, xs = np.nonzero(a)
        h, w = a.shape
        people.append(Person(Path(m), (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                             float(xs.mean() / w), float(ys.mean() / h), frac))
        bits.append(a)
    # A low detection threshold (needed for stylized / 3D characters) can return the same person twice
    # (whole body + torso): drop a mask mostly covered by a bigger one already kept.
    keep: list[int] = []
    for i in sorted(range(len(people)), key=lambda i: -people[i].area):
        if all((bits[i] & bits[j]).sum() < overlap * bits[i].sum() for j in keep):
            keep.append(i)
    people = [people[i] for i in keep]
    people.sort(key=lambda p: p.cx)
    return people


def match_people(ref: list[Person], found: list[Person]) -> list[Person | None]:
    """Re-identify ``ref`` people in a later frame's ``found`` scan. Same count: keep the left-to-right
    order (people rarely cross in a dance). Otherwise: nearest centroid, each found person used once."""
    if len(found) == len(ref):
        return list(found)
    out: list[Person | None] = []
    free = list(found)
    for r in ref:
        if not free:
            out.append(None)
            continue
        best = min(free, key=lambda f: (f.cx - r.cx) ** 2 + (f.cy - r.cy) ** 2)
        free.remove(best)
        out.append(best)
    return out


def people_preview(frame: str | Path, people: list[Person], dst: str | Path, labels: list[str] | None = None) -> Path:
    """First frame with each person outlined and numbered 1, 2, 3 (left to right), Noir style:
    thin bone outline, red number tag."""
    img = Image.open(frame).convert("RGB")
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    w, h = img.size
    size = max(18, int(min(w, h) * 0.09))
    font = _meme_font(size)
    for i, p in enumerate(people):
        m = Image.open(p.mask).convert("L").resize(img.size, Image.NEAREST)
        tint = Image.new("RGBA", img.size, (255, 42, 42, 70))
        over.paste(tint, mask=m.point(lambda v: 255 if v > 127 else 0))
        x0, y0, x1, y1 = p.box
        d.rectangle((x0, y0, x1, y1), outline=(239, 236, 230, 255), width=max(2, size // 12))
        tag = str(i + 1) + (f" {labels[i]}" if labels and i < len(labels) and labels[i] else "")
        tw = d.textlength(tag, font=font)
        tx, ty = max(0, min(w - tw - size * 0.6, (x0 + x1) / 2 - tw / 2 - size * 0.3)), max(0, y0 - size * 1.3)
        d.rectangle((tx, ty, tx + tw + size * 0.6, ty + size * 1.2), fill=(255, 42, 42, 255))
        d.text((tx + size * 0.3, ty + size * 0.05), tag, font=font, fill=(10, 10, 10, 255))
    out = Image.alpha_composite(img.convert("RGBA"), over).convert("RGB")
    dst = Path(dst)
    out.save(dst)
    return dst


# --------------------------------------------------------------------------- #
# Non-human reference check (robot / animal / mascot -> don't drive its head with a human face)
# --------------------------------------------------------------------------- #
YUNET_MODEL = Path(__file__).parent / "assets" / "face_detection_yunet_2023mar.onnx"  # OpenCV Zoo YuNet, MIT
FACE_SCORE_MIN = 0.6  # YuNet confidence for "there is a face"
NONHUMAN_GREEN_RATIO = 1.15  # face centre clearly greener than red (teal / green / blue skin) -> not human skin
NONHUMAN_MIN_LUMA = 50  # ...but only judge colour on a lit face (dark noir avatars read as purple-black)


def face_check(src: str | Path) -> tuple[bool | None, str]:
    """``(is_non_human, why)`` for a reference image. Non-human = YuNet finds no face, or the biggest face
    has clearly non-skin colour (a teal robot face still has 'eyes and a mouth', so a face detector alone
    passes it). ``None`` when OpenCV isn't installed: callers then keep the normal human behaviour.
    Measured on MC's test refs: invented robot (face found, green>red) and a boxy robot / cat mascot (no
    face) -> non-human; real people, a 3D cartoon woman, the astronaut and a dark noir avatar -> human."""
    try:
        import cv2  # optional: opencv-python-headless
        import numpy as np
    except ImportError:
        return None, "OpenCV not installed (pip install opencv-python-headless) - assuming a human face"
    try:  # OpenCV 5's DNN prints a harmless "Targets are not supported" warning per detector
        cv2.setLogLevel(2)
    except Exception:
        pass
    img = np.asarray(load_rgb(src))[:, :, ::-1].copy()  # RGB -> BGR
    h, w = img.shape[:2]
    det = cv2.FaceDetectorYN.create(str(YUNET_MODEL), "", (320, 320), FACE_SCORE_MIN, 0.3, 50)
    best = None
    for side in (640, 1024):
        k = side / max(h, w)
        small = cv2.resize(img, (max(1, int(w * k)), max(1, int(h * k))), interpolation=cv2.INTER_AREA)
        det.setInputSize((small.shape[1], small.shape[0]))
        _, faces = det.detect(small)
        if faces is not None:
            for f in faces:
                if best is None or f[2] * f[3] / k ** 2 > best[0][2] * best[0][3] / best[1] ** 2:
                    best = (f, k, small)
    if best is None:
        return True, "no human face found in the reference"
    f, _, small = best
    x, y, fw, fh = (int(v) for v in f[:4])
    roi = small[max(0, y + fh // 3): y + 2 * fh // 3, max(0, x + fw // 4): x + 3 * fw // 4]
    if roi.size == 0:
        return False, f"face found ({float(f[-1]):.2f})"
    b, g, r = (float(v) for v in roi.reshape(-1, 3).mean(0))
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    if luma >= NONHUMAN_MIN_LUMA and g > r * NONHUMAN_GREEN_RATIO:
        return True, f"face found but its colour isn't skin (rgb {r:.0f},{g:.0f},{b:.0f})"
    return False, f"human face found ({float(f[-1]):.2f})"


# --------------------------------------------------------------------------- #
# SCAIL-2 group swap references (one composite of everyone + identity-coloured masks)
# --------------------------------------------------------------------------- #
def _keep_blobs(m, min_frac: float = 0.0, largest_only: bool = False):
    import cv2
    import numpy as np

    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n <= 2:
        return m
    areas = stats[1:, cv2.CC_STAT_AREA]
    if largest_only:
        return lab == (int(np.argmax(areas)) + 1)
    return np.isin(lab, [i + 1 for i, a in enumerate(areas) if a > min_frac * areas.max()])


def _fill_holes(m):
    import cv2
    import numpy as np

    inv = (~m).astype(np.uint8)
    h, w = inv.shape
    flood = inv.copy()
    cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 2)
    return m | (flood == 1)


def matte_plain_background(img: Image.Image):
    """Character mask for a reference on a plain / studio background (MC avatars are made on one): pixels that
    differ from the row's edge colour, minus soft floor shadows, largest blob, holes filled. Returns a bool array."""
    import cv2
    import numpy as np

    a = np.asarray(img.convert("RGB")).astype(np.float32)
    h, w, _ = a.shape
    edge = max(4, w // 40)
    rowbg = (np.median(a[:, :edge], axis=1) + np.median(a[:, -edge:], axis=1)) / 2
    d = np.linalg.norm(a - rowbg[:, None, :], axis=2)
    sat = a.max(2) - a.min(2)
    m = (d > 18) | (sat > 40)
    darker = rowbg.mean(1)[:, None] - a.mean(2)
    shadow = (sat < 22) & (darker > 4) & (darker < 70) & (d < 75)
    shadow[: int(h * 0.75)] = False
    m &= ~shadow
    k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, k, iterations=2) > 0
    m = _keep_blobs(m, 0.03)  # drop specks before closing, so they can't bridge onto the floor
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, k, iterations=4) > 0
    m = _fill_holes(m)
    if m.any():  # below the shins, cut anything wider than the legs (floor shadow / reflections)
        ys = np.nonzero(m.any(1))[0]
        fy0, fh = ys.min(), ys.max() - ys.min()
        bx = np.nonzero(m[int(fy0 + 0.76 * fh):int(fy0 + 0.84 * fh)].any(0))[0]
        if len(bx):
            lo, hi = bx.min(), bx.max()
            pad, cut = int(0.12 * (hi - lo)), int(fy0 + 0.84 * fh)
            m[cut:, :max(0, lo - pad)] = False
            m[cut:, hi + pad:] = False
    return _keep_blobs(m, largest_only=True)


def reference_matte(src: str | Path, sam_mask: str | Path | None = None):
    """(RGB image, bool mask) of the character in a reference. Order: the PNG's own transparency, then the SAM 3.1
    person mask (works on busy photo backgrounds), then the plain-background matte."""
    import numpy as np

    raw = Image.open(src)
    if has_transparency(raw):
        rgba = raw.convert("RGBA")
        return load_rgb(src), np.asarray(rgba)[:, :, 3] > 127
    img = raw.convert("RGB")
    if sam_mask:
        m = Image.open(sam_mask).convert("L")
        if m.size != img.size:
            m = m.resize(img.size, Image.NEAREST)
        a = np.asarray(m) > 127
        if a.mean() > 0.01:
            return img, a
    return img, matte_plain_background(img)


def scail_references(refs: list, people_boxes: list, size: tuple[int, int], out_dir: str | Path,
                     palette: list, sam_masks: list | None = None) -> dict:
    """Build SCAIL-2's replacement-mode references for a group swap.

    ``refs[k]`` is the look for swapped person k (left to right), ``people_boxes[k]`` that person's first-frame box
    in clip pixels, ``palette[k]`` their identity colour. Writes:
      - ``ref.png``: everyone cut out and placed at their spot in the clip (scaled to the person's height) on grey,
      - ``ref_mask.png``: the same layout in identity colours on black,
      - ``view_<k>.png`` / ``view_mask_<k>.png``: each full reference with its mask in that person's colour,
    and returns their paths ({"ref", "ref_mask", "views": [...], "view_masks": [...]})."""
    import numpy as np

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    W, H = size
    comp = np.full((H, W, 3), 127, np.uint8)
    cmask = np.zeros((H, W, 3), np.uint8)
    cuts = []
    views, view_masks = [], []
    for k, ref in enumerate(refs):
        img, m = reference_matte(ref, (sam_masks or [None] * len(refs))[k])
        a = np.asarray(img)
        ev = np.zeros_like(a)
        ev[m] = palette[k]
        vp, vmp = out_dir / f"view_{k}.png", out_dir / f"view_mask_{k}.png"
        img.save(vp)
        Image.fromarray(ev).save(vmp)
        views.append(vp)
        view_masks.append(vmp)
        ys, xs = np.nonzero(m)
        x0, x1 = int(np.percentile(xs, 0.5)), int(np.percentile(xs, 99.5))
        cuts.append((a[ys.min():ys.max() + 1, x0:x1 + 1], m[ys.min():ys.max() + 1, x0:x1 + 1]))
    # Layout (what won the Oct 18 A/B): everyone at their own height relation, side by side WITHOUT overlapping
    # (SCAIL-2 separates identities by colour, so touching cut-outs blur who is who), left to right like the clip.
    order = sorted(range(len(refs)), key=lambda i: (people_boxes[i][0] + people_boxes[i][2]) / 2)
    heights = [max(1, people_boxes[i][3] - people_boxes[i][1]) for i in range(len(refs))]
    widths = [cuts[i][0].shape[1] * heights[i] / cuts[i][0].shape[0] for i in range(len(refs))]
    gap = 0.02 * W
    # 0.8x the filmed height, but people far from the camera are enlarged to ~3/4 of the frame height: tiny
    # cut-outs lose faces, hats and patterns (Oct 18 PULL UP lineup: a beanie read as hair at 20% height).
    s = min(max(0.8, 0.75 * H / max(heights)), (0.94 * W - gap * (len(refs) - 1)) / max(1.0, sum(widths)))
    x = (W - (s * sum(widths) + gap * (len(refs) - 1))) / 2
    tallest = s * max(heights)
    floor = min(H - int(0.02 * H), max(max(b[3] for b in people_boxes) - int(0.01 * H), int(tallest + 0.03 * H)))
    for k in order:
        crop, mc = cuts[k]
        nh = max(1, int(round(heights[k] * s)))
        nw = max(1, int(round(widths[k] * s)))
        c = np.asarray(Image.fromarray(crop).resize((nw, nh), Image.LANCZOS))
        mm = np.asarray(Image.fromarray(mc.astype(np.uint8) * 255).resize((nw, nh), Image.BILINEAR)) > 127
        px0, py0 = int(round(x)), min(H, floor) - nh
        x += nw + gap
        sx0, sy0 = max(0, -px0), max(0, -py0)
        dx0, dy0, dx1, dy1 = max(0, px0), max(0, py0), min(W, px0 + nw), min(H, py0 + nh)
        if dx1 <= dx0 or dy1 <= dy0:
            continue
        sub = mm[sy0:sy0 + dy1 - dy0, sx0:sx0 + dx1 - dx0]
        comp[dy0:dy1, dx0:dx1][sub] = c[sy0:sy0 + dy1 - dy0, sx0:sx0 + dx1 - dx0][sub]
        cmask[dy0:dy1, dx0:dx1][sub] = palette[k]
    rp, rmp = out_dir / "ref.png", out_dir / "ref_mask.png"
    Image.fromarray(comp).save(rp)
    Image.fromarray(cmask).save(rmp)
    return {"ref": rp, "ref_mask": rmp, "views": views, "view_masks": view_masks}


# --------------------------------------------------------------------------- #
# Style lock + per-person drivers (pending a real test; see stylelock.py / pipeline.Request.people_driver)
# --------------------------------------------------------------------------- #
def blend_upscale(take: str | Path, upscaled: str | Path, dest: str | Path, *, alpha: float = 0.4,
                  size: tuple[int, int], fps: int = 16) -> Path:
    """``alpha`` x the SeedVR2 frame + (1 - alpha) x a plain lanczos upscale of the take. SeedVR2 restores toward
    photographs (pores, fabric weave), which pulls drawn characters toward realism; a partial blend keeps its
    de-noising and sharper edges with less invented texture."""
    a = max(0.0, min(1.0, float(alpha)))
    w, h = size
    chain = (f"[0:v]scale={w}:{h}:flags=lanczos,format=gbrp,setsar=1[base];"
             f"[1:v]scale={w}:{h}:flags=lanczos,format=gbrp,setsar=1[sr];"
             f"[sr][base]blend=all_mode=normal:all_opacity={a:.3f},format=yuv420p,setpts=N/({fps}*TB)[out]")
    run_ffmpeg(["-i", str(take), "-i", str(upscaled), "-filter_complex", chain, "-map", "[out]", "-an",
                "-r", str(fps), "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", str(dest)])
    return Path(dest)


def _even(v: float) -> int:
    return max(2, int(round(v / 2.0)) * 2)


def driver_layout(src_box: tuple[int, int, int, int] | None, dst_box: tuple[int, int, int, int],
                  frame_size: tuple[int, int], src_size: tuple[int, int], margin: float = 0.12) -> dict[str, int]:
    """Where a driver figure goes: crop ``src_box`` (+``margin``) from the driver frames (``None`` = the whole driver
    frame), scale it to the height of ``dst_box`` (+``margin``), bottom-aligned and centred on ``dst_box``.
    Returns integer crop / scale / overlay numbers (all clamped inside the frames)."""
    fw, fh = frame_size
    sw, sh = src_size

    def grow(box, W, H):
        x0, y0, x1, y1 = box
        mx, my = (x1 - x0) * margin, (y1 - y0) * margin
        return (max(0, int(x0 - mx)), max(0, int(y0 - my)), min(W, int(x1 + mx)), min(H, int(y1 + my)))

    cx0, cy0, cx1, cy1 = grow(src_box, sw, sh) if src_box else (0, 0, sw, sh)
    dx0, dy0, dx1, dy1 = grow(dst_box, fw, fh)
    cw, ch = max(2, cx1 - cx0), max(2, cy1 - cy0)
    th = _even(min(fh, dy1 - dy0))
    tw = _even(min(fw, cw * th / ch))
    ox = int(max(0, min(fw - tw, (dx0 + dx1) / 2 - tw / 2)))
    oy = int(max(0, min(fh - th, dy1 - th)))
    return {"crop_x": cx0, "crop_y": cy0, "crop_w": cw - cw % 2, "crop_h": ch - ch % 2, "w": tw, "h": th,
            "x": ox, "y": oy,
            "erase_x": dx0, "erase_y": dy0, "erase_w": dx1 - dx0, "erase_h": dy1 - dy0}


def compose_driver(clip: str | Path, dest: str | Path, *, dst_box: tuple[int, int, int, int],
                   frame_size: tuple[int, int], frames: int, fps: int = 16,
                   src_box: tuple[int, int, int, int] | None = None, driver_clip: str | Path | None = None,
                   driver_size: tuple[int, int] | None = None, src_start: int = 0) -> Path:
    """``driver.mp4`` for one person: the clip with that person's area greyed out and another figure's moves placed
    there (``src_box`` of the same clip = "follow person N", or ``driver_clip`` = "follow this other clip"). Only
    Animate's pose / face / tracking read it; the background and the final paste-back still use the real clip."""
    src = Path(driver_clip) if driver_clip else Path(clip)
    ssize = driver_size or (frame_size if driver_clip is None else probe_size(src))
    L = driver_layout(src_box, dst_box, frame_size, ssize)
    fw, fh = frame_size
    chain = (f"[0:v]scale={fw}:{fh},setsar=1,drawbox=x={L['erase_x']}:y={L['erase_y']}:w={L['erase_w']}:"
             f"h={L['erase_h']}:color=gray:t=fill[bg];"
             f"[1:v]fps={fps},trim=start_frame={int(src_start)},setpts=PTS-STARTPTS,crop={L['crop_w']}:{L['crop_h']}:{L['crop_x']}:{L['crop_y']},"
             f"scale={L['w']}:{L['h']}:flags=lanczos,setsar=1,tpad=stop_mode=clone:stop={int(frames)}[fig];"
             f"[bg][fig]overlay=x={L['x']}:y={L['y']}:shortest=0:eof_action=repeat,setpts=N/({fps}*TB)[out]")
    run_ffmpeg(["-i", str(clip), "-i", str(src), "-filter_complex", chain, "-map", "[out]", "-an",
                "-frames:v", str(int(frames)), "-r", str(fps), "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p",
                str(dest)])
    return Path(dest)


def probe_size(video: str | Path) -> tuple[int, int]:
    """(width, height) of a video's first stream (via ffmpeg's banner; no ffprobe needed)."""
    proc = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(video)], capture_output=True, text=True)
    m = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", proc.stderr)
    if not m:
        raise RuntimeError(f"can't read the size of {video}")
    return int(m.group(1)), int(m.group(2))


def wipe_compare(before: str | Path, after: str | Path, dest: str | Path, *, period: float = 4.0,
                 seconds: float | None = None) -> Path:
    """Before/after wipe for sharing: the original clip and the result on one frame, a white divider sweeping
    left and right every ``period`` seconds (original on the left of it). ``before`` is scaled and cropped to the
    result's size; the result's audio is kept. Both should start at the same moment (pass the prepared clip)."""
    w, h = video_size(after)
    dur = media_duration(after) if seconds is None else float(seconds)
    pos = f"(W*(0.5+0.42*sin(2*PI*T/{float(period):.3f})))"
    pick = f"if(lt(X,{pos}),B,A)"
    luma, chroma = f"if(lt(abs(X-{pos}),2),235,{pick})", f"if(lt(abs(X-{pos}),2),128,{pick})"  # white line in YUV
    args = ["-i", str(after), "-i", str(before), "-filter_complex",
            f"[1:v]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},setsar=1,format=yuv444p[b];"
            f"[0:v]setsar=1,format=yuv444p[a];[a][b]blend=c0_expr='{luma}':c1_expr='{chroma}':c2_expr='{chroma}'"
            ":shortest=1,format=yuv420p[v]",
            "-map", "[v]"]
    if has_audio_stream(after):
        args += ["-map", "0:a:0", "-c:a", "aac", "-b:a", "160k"]
    args += ["-t", f"{dur:.3f}", "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-movflags", "+faststart",
             str(dest)]
    run_ffmpeg(args)
    return Path(dest)
