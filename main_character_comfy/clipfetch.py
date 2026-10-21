"""Bring your own clip: fetch the sharpest copy of a clip from a link, locally, into the user's own folder.

MC ships no footage. This only downloads what the user points it at, on their own PC, into their own library
(``library/clips/`` by default), with yt-dlp picking the best video stream plus the best audio stream and merging
them with MC's bundled ffmpeg. Plain direct file links work too. The user is responsible for having the rights to
the clips they use.
"""

from __future__ import annotations

import ipaddress
import json
import re
import socket
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from .pipeline import InputError

BEST_RESULTS_TIP = ("for the best results, use the highest-quality version of your clip you can find: 1080p or 4K, "
                    "steady camera, good lighting, clean audio.")
RIGHTS_LINE = "you're responsible for having the rights to any clip you use."

# best resolution first, then frame rate; among equals prefer codecs that decode everywhere.
FORMAT = "bv*+ba/b"
FORMAT_SORT = ["res", "fps", "br", "vcodec:h264", "acodec:aac", "size"]
MAX_MB = 2000


class FetchError(InputError):
    pass


def available() -> bool:
    try:
        import yt_dlp  # noqa: F401
        return True
    except ImportError:
        return False


def _need() -> Any:
    try:
        import yt_dlp
    except ImportError:
        raise FetchError("clip links need yt-dlp: pip install yt-dlp (it's in requirements.txt)") from None
    return yt_dlp


def check_url(url: str) -> str:
    """Only public http(s) links: no files on disk, no local or private network addresses."""
    url = (url or "").strip()
    u = urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise FetchError("paste a public http(s) link to the clip")
    try:
        infos = socket.getaddrinfo(u.hostname, u.port or (443 if u.scheme == "https" else 80))
    except socket.gaierror:
        raise FetchError(f"can't find {u.hostname}: check the link") from None
    for info in infos:
        if blocked_ip(info[4][0]):
            raise FetchError("MC won't fetch from local or private network addresses")
    return url


# 198.18.0.0/15 is where "fake-ip" proxies and VPN apps (clash, surge, some corporate tunnels) map every public
# name; it's never a real LAN, so it isn't treated as one.
_FAKE_IP = ipaddress.ip_network("198.18.0.0/15")


def blocked_ip(addr: str) -> bool:
    ip = ipaddress.ip_address(addr)
    if ip.version == 4 and ip in _FAKE_IP:
        return False
    return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified


def _ffmpeg() -> str | None:
    import shutil

    if shutil.which("ffmpeg"):  # a system ffmpeg handles yt-dlp's network section grabs best
        return shutil.which("ffmpeg")
    try:
        from .media import ffmpeg_exe
        return ffmpeg_exe()
    except Exception:
        return None


def _base_opts() -> dict[str, Any]:
    return {"quiet": True, "no_warnings": True, "noplaylist": True, "noprogress": True,
            "ffmpeg_location": _ffmpeg(), "socket_timeout": 30, "retries": 3, "cachedir": False}


def _quality(height: int | None) -> str:
    if not height:
        return "unknown"
    if height >= 2000:
        return "4k"
    if height >= 1400:
        return "1440p"
    if height >= 1000:
        return "1080p"
    if height >= 700:
        return "720p"
    return f"{height}p (low: look for a sharper copy)"


def _summary(info: dict[str, Any]) -> dict[str, Any]:
    fmts = [f for f in info.get("formats") or [] if (f.get("vcodec") or "none") != "none"]
    heights = sorted({int(f["height"]) for f in fmts if f.get("height")}, reverse=True)
    best = heights[0] if heights else info.get("height")
    w = None
    for f in fmts:
        if f.get("height") == best and f.get("width"):
            w = f["width"]
            break
    return {"title": info.get("title"), "url": info.get("webpage_url") or info.get("url"),
            "site": info.get("extractor_key") or info.get("extractor"), "uploader": info.get("uploader"),
            "seconds": info.get("duration"), "best_height": best, "best_width": w or info.get("width"),
            "best_quality": _quality(best), "heights": heights[:8],
            "tbr": max([f.get("tbr") or 0 for f in fmts if f.get("height") == best] or [0]) or None,
            "fps": max([f.get("fps") or 0 for f in fmts] or [info.get("fps") or 0]) or None}


def probe(url: str) -> dict[str, Any]:
    """What's at the link and the sharpest copy available, without downloading."""
    yt = _need()
    url = check_url(url)
    try:
        with yt.YoutubeDL(_base_opts()) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:  # yt-dlp raises DownloadError with a long message
        raise FetchError(f"couldn't read that link: {_short(e)}") from None
    if info.get("_type") == "playlist":
        raise FetchError("that's a playlist; paste a link to one clip")
    return _summary(info)


def search(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Find candidate copies of a clip by text (youtube search), sharpest first."""
    yt = _need()
    query = (query or "").strip()
    if not query:
        raise FetchError("give something to search for")
    limit = max(1, min(int(limit), 10))
    opts = {**_base_opts(), "noplaylist": False, "extract_flat": "in_playlist"}
    try:
        with yt.YoutubeDL(opts) as ydl:
            res = ydl.extract_info(f"ytsearch{limit}:{query}", download=False)
    except Exception as e:
        raise FetchError(f"search failed: {_short(e)}") from None
    out = []
    for entry in (res or {}).get("entries") or []:
        link = entry.get("url") or entry.get("webpage_url")
        if not link:
            continue
        try:
            out.append(probe(link))
        except FetchError:
            continue
    out.sort(key=lambda r: (r.get("best_height") or 0, r.get("fps") or 0), reverse=True)
    return out


def source_score(summary: dict[str, Any]) -> tuple:
    """Sort key for candidate copies of one clip: resolution, then fps, then bitrate (bigger is better)."""
    return (summary.get("best_height") or 0, summary.get("fps") or 0, summary.get("tbr") or 0)


def rank_sources(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Best-source-first: candidate copies (the post, its reposts/quotes, search hits) sharpest first, each with a
    plain verdict; ``low`` when even the best is under 720p so the caller can warn before rendering."""
    ranked = sorted(summaries, key=source_score, reverse=True)
    for i, s in enumerate(ranked):
        s["rank"] = i + 1
        s["low"] = (s.get("best_height") or 0) < 720
    return ranked


def _download(yt: Any, url: str, opts: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    with yt.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if info.get("_type") == "playlist":
            raise FetchError("that's a playlist; paste a link to one clip")
        files = [Path(r["filepath"]) for r in info.get("requested_downloads") or [] if r.get("filepath")]
        return info, (files[0] if files else Path(ydl.prepare_filename(info)).with_suffix(".mp4"))


def _cut(src: Path, dst: Path, start: float | None, end: float | None) -> None:
    """Frame-accurate local cut, re-encoded at high quality (crf 14) so nothing visible is lost."""
    from .media import run_ffmpeg

    args = ["-ss", f"{float(start or 0):.3f}", "-i", str(src)]
    if end is not None:
        args += ["-t", f"{float(end) - float(start or 0):.3f}"]
    run_ffmpeg(args + ["-c:v", "libx264", "-crf", "14", "-preset", "medium", "-pix_fmt", "yuv420p",
                       "-c:a", "aac", "-b:a", "320k", "-movflags", "+faststart", str(dst)])


def _short(e: Exception) -> str:
    msg = re.sub(r"\x1b\[[0-9;]*m", "", str(e)).replace("ERROR: ", "").strip()
    return msg.splitlines()[0][:300] if msg else type(e).__name__


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "clip").lower()).strip("-")
    return (s or "clip")[:60]


def fetch(url: str, dest_dir: str | Path, *, name: str | None = None, start: float | None = None,
          end: float | None = None, max_height: int | None = None,
          progress: Callable[[float, str], None] | None = None,
          cancelled: Callable[[], bool] | None = None) -> dict[str, Any]:
    """Download the best video + best audio at ``url`` into ``dest_dir`` as one mp4.

    ``start`` / ``end`` (seconds) grab just that part. ``max_height`` caps the resolution (default: no cap).
    Returns the file path, its quality and the source, and writes a small ``<file>.source.json`` next to it.
    """
    yt = _need()
    url = check_url(url)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    if start is not None and end is not None and end <= start:
        raise FetchError("end has to be after start")

    def hook(d: dict[str, Any]) -> None:
        if cancelled and cancelled():
            raise FetchError("cancelled")
        if progress and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            frac = (d.get("downloaded_bytes") or 0) / total if total else 0.0
            progress(min(0.95, frac), f"downloading {d.get('_percent_str', '').strip() or ''}".strip())

    fmt_sort = list(FORMAT_SORT)
    if max_height:
        fmt_sort[0] = f"res:{int(max_height)}"
    stem = _slug(name) if name else None
    opts = {**_base_opts(), "format": FORMAT, "format_sort": fmt_sort, "merge_output_format": "mp4",
            "outtmpl": str(dest_dir / f"{stem or '%(title).60B'}-%(id)s.%(ext)s"), "restrictfilenames": True,
            "max_filesize": MAX_MB * 1024 * 1024, "progress_hooks": [hook], "overwrites": False}
    if start is not None or end is not None:
        from yt_dlp.utils import download_range_func

        opts["download_ranges"] = download_range_func(None, [(float(start or 0), float(end) if end else float("inf"))])
        opts["force_keyframes_at_cuts"] = True
    t0 = time.time()
    try:
        info, path = _download(yt, url, opts)
    except FetchError:
        raise
    except Exception as e:
        if "download_ranges" not in opts:
            raise FetchError(f"couldn't fetch that clip: {_short(e)}") from None
        # some ffmpeg builds can't cut straight from the network: grab it whole, then cut locally
        opts = {k: v for k, v in opts.items() if k not in ("download_ranges", "force_keyframes_at_cuts")}
        try:
            info, full = _download(yt, url, opts)
        except FetchError:
            raise
        except Exception as e2:
            raise FetchError(f"couldn't fetch that clip: {_short(e2)}") from None
        path = full.with_name(full.stem + f"-{int(start or 0)}s.mp4")
        _cut(full, path, start, end)
        full.unlink(missing_ok=True)
    if not path.exists():
        raise FetchError("the download finished but no file showed up")
    summary = _summary(info)
    got_w, got_h = info.get("width"), info.get("height")
    try:  # what actually landed (direct links don't report it up front)
        from .media import video_size

        got_w, got_h = video_size(path)
    except Exception:
        got_h = got_h or summary["best_height"]
    meta = {"source_url": url, "title": summary["title"], "site": summary["site"], "uploader": summary["uploader"],
            "height": got_h, "width": got_w, "fps": info.get("fps"), "vcodec": info.get("vcodec"),
            "acodec": info.get("acodec"), "quality": _quality(got_h), "best_available_height": summary["best_height"] or got_h,
            "section": [start, end] if (start is not None or end is not None) else None,
            "fetched": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds_to_fetch": round(time.time() - t0, 1),
            "note": RIGHTS_LINE}
    path.with_name(path.name + ".source.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if progress:
        progress(1.0, f"got it: {meta['quality']}")
    tips = []
    if got_h and got_h < 1000:
        tips.append(f"this copy is only {got_h}p. " + BEST_RESULTS_TIP)
    return {"path": path, **meta, "tips": tips}
