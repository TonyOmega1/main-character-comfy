"""Long clips: split speech into segments, render each, stitch with crossfades.

InfiniteTalk drifts (soft/blurry faces, colour shift) as one generation grows
past ~20 s, and hard cuts between separate generations flash. MC keeps
every generation short and hides the joins:

* the speech is cut at natural pauses (ffmpeg ``silencedetect``) into segments
  of at most ``segment_seconds``;
* each segment is its own InfiniteTalk job, re-seeded either from the clean
  avatar (``anchor``, default: no accumulated drift) or from the previous
  segment's last frame (``continue``: smoother pose continuity);
* every segment except the last is rendered ``overlap`` seconds longer, and the
  segments are joined with a video crossfade exactly over that overlap, so the
  picture stays frame-accurate with the original, uncut speech track (audio is
  never stitched, so there are no audio seams at all).
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .media import ffmpeg_exe, media_duration, run_ffmpeg


@dataclass
class Segment:
    index: int
    start: float
    end: float  # speech end of this segment (next segment starts here)
    render_end: float  # end including crossfade overlap

    @property
    def render_seconds(self) -> float:
        return self.render_end - self.start


_SIL_RE = re.compile(r"silence_(start|end): (-?\d+(?:\.\d+)?)")


def detect_pauses(audio: str | Path, noise_db: float = -35.0, min_silence: float = 0.18) -> list[float]:
    """Midpoints of silent gaps, in seconds."""
    proc = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-i", str(audio), "-af",
         f"silencedetect=noise={noise_db}dB:d={min_silence}", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    points: list[float] = []
    start = None
    for kind, val in _SIL_RE.findall(proc.stderr):
        v = float(val)
        if kind == "start":
            start = v
        elif start is not None:
            points.append((start + v) / 2)
            start = None
    return points


def plan_segments(
    duration: float, pauses: list[float], segment_seconds: float = 15.0, overlap: float = 0.24, min_seconds: float = 3.0
) -> list[Segment]:
    """Greedy cut at the last pause before each ``segment_seconds`` boundary;
    hard cut if a stretch has no pause."""
    if duration <= segment_seconds:
        return [Segment(0, 0.0, duration, duration)]
    cuts: list[float] = []
    cursor = 0.0
    while duration - cursor > segment_seconds:
        limit = cursor + segment_seconds
        candidates = [p for p in pauses if cursor + min_seconds <= p <= limit]
        cut = candidates[-1] if candidates else limit
        cuts.append(round(cut, 3))
        cursor = cut
    # Avoid a tiny last segment: pull the last cut back (to a pause if one fits).
    if cuts and duration - cuts[-1] < min_seconds:
        prev = cuts[-2] if len(cuts) > 1 else 0.0
        lo, hi = prev + min_seconds, duration - min_seconds
        if lo <= hi:
            fits = [p for p in pauses if lo <= p <= hi]
            cuts[-1] = round(fits[-1] if fits else hi, 3)
        else:
            cuts.pop()
    bounds = [0.0, *cuts, duration]
    segs = []
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        last = i == len(bounds) - 2
        segs.append(Segment(i, s, e, e if last else min(duration, e + overlap)))
    return segs


def cut_audio(src: str | Path, dst: str | Path, start: float, seconds: float) -> Path:
    """Exact cut, padded with silence if the source ends early."""
    run_ffmpeg([
        "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{seconds:.3f}",
        "-af", f"apad=whole_dur={seconds:.3f}", "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(dst),
    ])
    return Path(dst)


def last_frame(video: str | Path, dst: str | Path) -> Path:
    run_ffmpeg(["-sseof", "-0.2", "-i", str(video), "-update", "1", "-frames:v", "1", str(dst)])
    return Path(dst)


def stitch(
    clips: list[Path], segments: list[Segment], dst: str | Path, *, fps: int = 25, size: tuple[int, int] | None = None
) -> Path:
    """Crossfade-join segment videos so segment i+1 starts exactly at its speech
    start time. Output has no audio (the finisher muxes the full speech track)."""
    dst = Path(dst)
    if len(clips) == 1:
        run_ffmpeg(["-i", str(clips[0]), "-an", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(dst)])
        return dst
    args: list[str] = []
    for c in clips:
        args += ["-i", str(c)]
    norm = []
    for i, (c, seg) in enumerate(zip(clips, segments)):
        scale = f",scale={size[0]}:{size[1]}" if size else ""
        # trim each clip to exactly its planned length (pad by cloning the last frame if short)
        norm.append(
            f"[{i}:v]fps={fps}{scale},setsar=1,tpad=stop_mode=clone:stop_duration=1,"
            f"trim=duration={seg.render_seconds:.3f},setpts=PTS-STARTPTS,fps={fps},format=yuv420p[n{i}]"
        )
    chain = list(norm)
    prev = "n0"
    for i in range(1, len(clips)):
        overlap = segments[i - 1].render_end - segments[i - 1].end
        offset = segments[i].start  # == accumulated length - overlap
        out = f"x{i}"
        chain.append(f"[{prev}][n{i}]xfade=transition=fade:duration={max(0.04, overlap):.3f}:offset={offset:.3f}[{out}]")
        prev = out
    args += ["-filter_complex", ";".join(chain), "-map", f"[{prev}]", "-an",
             "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(dst)]
    run_ffmpeg(args)
    return dst


__all__ = ["Segment", "detect_pauses", "plan_segments", "cut_audio", "last_frame", "stitch", "media_duration"]
