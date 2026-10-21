"""MC as a headless service: everything the UI can do, as plain function calls that return JSON-friendly dicts.

This is what the agent connector (``main_character_comfy.agent``) wraps, and it's usable from any Python script:

    from main_character_comfy.service import MCService
    mc = MCService()
    job = mc.make_meme(photo="references/me.png", meme="side_eye", i_have_rights=True)
    mc.wait(job["id"], 600)

Renders go through one :class:`jobs.JobQueue` (FIFO, one at a time) inside the shared GPU lane, so agent jobs and
renders started in the MC window never fight over the GPU. Quick, non-GPU calls (templates, uploads, status,
settings) answer immediately.

**File sandbox.** The service only reads and writes inside MC's own folders: the upload library
(``MC_LIBRARY_DIR``, default ``<repo>/library``), the outputs (``MC_OUTPUT_DIR``), the repo's ``examples/`` (read
only), plus any folders listed in ``MC_AGENT_EXTRA_DIRS`` (``os.pathsep``-separated, read only). Paths are given
as ``library/references/me.png``, ``outputs/<run>/<file>``, ``examples/<file>``, a bare name found in the library,
or an absolute path inside one of those folders. Anything else is refused.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import socket
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib.parse import urlparse

from . import clipfetch, extras, memes, media, refcheck, tiers
from .backends import Backend, make_backend
from .config import ROOT, Settings, settings as load_settings
from .jobs import ACTIVE, JobContext, JobQueue, JobStore, agent_dir, lane_owner
from .pipeline import (CLIP_AUDIO, FACE_CHOICES, SUBJECTS, VIBE_SCENE_IDEAS, InputError, Pipeline, Request,
                       check_request, request_from_recipe, split_batch_text)
from .presets import STYLES

VERSION = "0.3.0"
KINDS = {
    "references": {".png", ".jpg", ".jpeg", ".webp", ".bmp"},
    "clips": {".mp4", ".mov", ".webm", ".mkv", ".m4v", ".avi", ".gif"},
    "voices": {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".opus", ".mp4", ".mov", ".webm"},
    "audio": {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".opus"},
    "recipes": {".json"},
}
KIND_ALIASES = {"reference": "references", "ref": "references", "image": "references", "photo": "references",
                "avatar": "references", "selfie": "references", "clip": "clips", "video": "clips",
                "driving_clip": "clips", "voice": "voices", "voice_sample": "voices", "song": "audio",
                "recording": "audio", "line": "audio", "recipe": "recipes"}
OUTPUT_EXT = {".mp4", ".gif", ".png", ".jpg", ".wav", ".flac", ".mp3", ".json"}
DEFAULT_AGENT_SETTINGS = {
    "tier": "auto",  # auto | lite | standard | max
    "allow_heavy": False,  # Lite: try the 16 GB modes anyway
    "watermark": True,
    "hd": True,  # 1080p export
    "seed": 2026,
    "wait_for_comfy_idle": True,  # wait until ComfyUI's own queue is empty before an agent render starts
    "comfy_idle_timeout_min": 180,
}
SETTINGS_HELP = {
    "tier": "hardware tier: auto (from VRAM), lite (8-12 GB), standard (16 GB), max (24 GB+)",
    "allow_heavy": "on Lite, try the 16 GB modes anyway (may run out of memory)",
    "watermark": "small 'made with MC' watermark on exports",
    "hd": "export 1080p (off = render size)",
    "seed": "default seed for new jobs (each job can override it)",
    "wait_for_comfy_idle": "before an agent render starts, wait until ComfyUI's own queue is empty (someone else "
                           "may be using ComfyUI directly)",
    "comfy_idle_timeout_min": "give up waiting for an idle ComfyUI after this many minutes",
}
MAX_UPLOAD_BYTES = int(float(os.environ.get("MC_AGENT_MAX_UPLOAD_MB", "500")) * 1024 * 1024)


class AccessError(PermissionError):
    """A path outside MC's folders, or a URL MC won't fetch."""


def _safe_name(name: str) -> str:
    base = Path(str(name).replace("\\", "/")).name
    base = re.sub(r"[^A-Za-z0-9._ -]+", "_", base).strip(" .")
    return base[:120] or "upload"


class MCService:
    def __init__(self, settings: Settings | None = None, *, backend_kind: str | None = None,
                 backend_factory: Callable[[], Backend] | None = None, library_dir: str | Path | None = None,
                 extra_dirs: list[str | Path] | None = None, comfyui_url: str | None = None) -> None:
        self.settings = settings or load_settings()
        self.backend_kind = backend_kind or self.settings.backend
        self._backend_factory = backend_factory
        self.output_dir = Path(self.settings.output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.library_dir = Path(library_dir or os.environ.get("MC_LIBRARY_DIR") or ROOT / "library").resolve()
        for k in KINDS:
            (self.library_dir / k).mkdir(parents=True, exist_ok=True)
        env_extra = [p for p in os.environ.get("MC_AGENT_EXTRA_DIRS", "").split(os.pathsep) if p.strip()]
        self.extra_dirs = [Path(p).resolve() for p in (extra_dirs or []) + env_extra]
        self.examples_dir = (ROOT / "examples").resolve()
        self.comfyui_url = (comfyui_url or os.environ.get("COMFYUI_URL") or "http://127.0.0.1:8188").rstrip("/")
        self.queue = JobQueue(self.output_dir, before_gpu=self._before_gpu)
        self.store: JobStore = self.queue.store

    # ------------------------------------------------------------------ helpers
    def _backend(self) -> Backend:
        if self._backend_factory:
            return self._backend_factory()
        return make_backend(self.backend_kind, self.settings.comfy_base_url, self.settings.comfy_api_key)

    def _pipeline(self, ctx: JobContext) -> Pipeline:
        pipe = Pipeline(self._backend(), self.settings)
        ctx.on_cancel(pipe.cancel)
        return pipe

    def roots(self) -> dict[str, Path]:
        r = {"library": self.library_dir, "outputs": self.output_dir, "examples": self.examples_dir}
        for i, d in enumerate(self.extra_dirs):
            r[f"extra{i + 1}" if d.name in r else d.name] = d
        return r

    def writable_roots(self) -> list[Path]:
        return [self.library_dir, self.output_dir]

    def resolve(self, path: str | Path | None, *, must_exist: bool = True, kinds: set[str] | None = None) -> Path | None:
        """Map an agent-supplied path into MC's sandbox (see the module docstring)."""
        if path is None or str(path).strip() in ("", "-", "none", "null"):
            return None
        raw = str(path).strip().replace("\\", "/")
        roots = self.roots()
        cands: list[Path] = []
        p = Path(raw)
        if p.is_absolute() or re.match(r"^[A-Za-z]:/", raw):
            cands.append(Path(raw))
        else:
            head, _, rest = raw.partition("/")
            if head in roots and rest:
                cands.append(roots[head] / rest)
            cands += [self.library_dir / raw, self.output_dir / raw, self.examples_dir / raw]
            if "/" not in raw:  # bare file name: look through the library kinds
                cands += [self.library_dir / k / raw for k in KINDS]
        for c in cands:
            try:
                rc = c.resolve()
            except OSError:
                continue
            if not any(rc == r or r in rc.parents for r in roots.values()):
                continue
            if must_exist and not rc.is_file():
                continue
            if kinds and rc.suffix.lower() not in kinds:
                raise InputError(f"{rc.name}: expected one of {', '.join(sorted(kinds))}")
            return rc
        inside = any(_inside(c, roots.values()) for c in cands)
        if not inside:
            raise AccessError(f"'{raw}' is outside MC's folders (library, outputs, examples). Upload it first with "
                              f"upload_file.")
        raise InputError(f"'{raw}' not found. Use list_library or list_outputs to see what's there.")

    def display(self, p: str | Path | None) -> str | None:
        """Short sandbox-relative name for a path (what agents should pass back in)."""
        if not p:
            return None
        rp = Path(p).resolve()
        for name, r in self.roots().items():
            if rp == r or r in rp.parents:
                return f"{name}/{rp.relative_to(r).as_posix()}"
        return str(p)

    def _agent_settings_path(self) -> Path:
        return agent_dir(self.output_dir) / "settings.json"

    def get_settings(self) -> dict[str, Any]:
        cur = dict(DEFAULT_AGENT_SETTINGS)
        try:
            cur.update({k: v for k, v in json.loads(self._agent_settings_path().read_text("utf-8")).items()
                        if k in DEFAULT_AGENT_SETTINGS})
        except (OSError, ValueError):
            pass
        return cur

    def update_settings(self, **changes: Any) -> dict[str, Any]:
        cur = self.get_settings()
        for k, v in changes.items():
            if v is None:
                continue
            if k not in DEFAULT_AGENT_SETTINGS:
                raise InputError(f"unknown setting {k!r} (settings: {', '.join(DEFAULT_AGENT_SETTINGS)})")
            if k == "tier" and v != "auto":
                v = tiers.normalize(v)
            want = type(DEFAULT_AGENT_SETTINGS[k])
            if want is bool and not isinstance(v, bool):
                v = str(v).lower() in ("1", "true", "yes", "on")
            elif want in (int, float):
                v = want(v)
            cur[k] = v
        self._agent_settings_path().write_text(json.dumps(cur, indent=2), encoding="utf-8")
        return cur

    def tier_key(self, tier: str | None = None) -> str:
        t = tier or self.get_settings()["tier"]
        return tiers.resolve(None if t in (None, "", "auto") else t).key

    def _base(self, tier: str | None, seed: int | None) -> dict[str, Any]:
        st = self.get_settings()
        return dict(tier=self.tier_key(tier), allow_heavy=bool(st["allow_heavy"]), watermark=bool(st["watermark"]),
                    hd=bool(st["hd"]), seed=int(seed if seed is not None else st["seed"]))

    # ------------------------------------------------------------- GPU courtesy
    def comfy_queue(self) -> dict[str, Any] | None:
        """ComfyUI's own queue (read-only diagnostic: someone may be using ComfyUI directly). None if unreachable."""
        if self.backend_kind != "comfy":
            return None
        try:
            import httpx

            q = httpx.get(f"{self.comfyui_url}/queue", timeout=3).json()
            return {"running": len(q.get("queue_running", [])), "pending": len(q.get("queue_pending", []))}
        except Exception:
            return None

    def _before_gpu(self, ctx: JobContext) -> None:
        st = self.get_settings()
        if not st["wait_for_comfy_idle"] or self.backend_kind != "comfy":
            return
        end = time.monotonic() + float(st["comfy_idle_timeout_min"]) * 60
        while True:
            q = self.comfy_queue()
            if not q or (q["running"] == 0 and q["pending"] == 0):
                return
            if ctx.cancelled:
                from .errors import Cancelled

                raise Cancelled("cancelled while waiting for ComfyUI")
            if time.monotonic() > end:
                raise RuntimeError("ComfyUI stayed busy with someone else's jobs; try again later or turn off "
                                   "wait_for_comfy_idle")
            ctx.tracker.set(message=f"waiting for ComfyUI to finish other work ({q['running']} running, "
                                    f"{q['pending']} queued outside mc)")
            time.sleep(5)

    # ------------------------------------------------------------------- submit
    def _submit(self, kind: str, title: str, params: dict[str, Any], gen: Callable[[Pipeline], Iterator[Any]],
                gpu: bool = True) -> dict[str, Any]:
        def run(ctx: JobContext) -> Iterator[Any]:
            yield from gen(self._pipeline(ctx))

        clean = {k: (self.display(v) if isinstance(v, Path) else v) for k, v in params.items()}
        rec = self.queue.submit(kind, title, clean, run, gpu=gpu)
        return self.job_summary(rec)

    def _submit_request(self, kind: str, title: str, req: Request, method: str = "run",
                        extra_args: tuple = ()) -> dict[str, Any]:
        if method in ("run", "run_still"):
            check_request(req, self.settings)
        params = {k: v for k, v in req.__dict__.items() if v not in (None, "", [], False) or k in ("consent",)}
        params = {k: ([self.display(x) if isinstance(x, Path) else x for x in v] if isinstance(v, list) else v)
                  for k, v in params.items()}

        def gen(pipe: Pipeline) -> Iterator[Any]:
            yield from getattr(pipe, method)(req, *extra_args)

        summary = self._submit(kind, title, params, gen)
        warn = self._reference_warnings(req)
        if warn:  # never blocks: the render runs, the agent relays the warning
            summary["reference_quality"] = warn
        return summary

    # ============================================================== catalog
    def list_templates(self) -> dict[str, Any]:
        recipes = [{"key": r.key, "title": r.title, "top": r.top, "bottom": r.bottom, "line": r.line or "",
                    "effect": r.effect, "seconds": round(r.frames / 16, 1), "builtin": r.key in memes.BUILTIN_KEYS}
                   for r in memes.RECIPES.values()]
        examples = [self.display(p) for p in sorted(self.examples_dir.glob("*"))
                    if p.suffix.lower() in KINDS["references"] | KINDS["clips"] | KINDS["audio"]]
        return {
            "meme_formats": recipes,
            "styles": [{"key": k, "label": s.label} for k, s in STYLES.items()],
            "meme_eras": list(media.ERAS),
            "effects": list(memes.EFFECTS),
            "vibe_scene_ideas": VIBE_SCENE_IDEAS,
            "clip_audio": list(CLIP_AUDIO),
            "subjects": list(SUBJECTS),
            "face_options": list(FACE_CHOICES),
            "group_engines": ["auto", "scail", "chain"],
            "style_lock": ["off", "auto", "on"],
            "split_cuts": ["auto", "on", "off"],
            "upscale": ["auto", "seedvr2", "lanczos"],
            "resolutions": list(tiers.RESOLUTIONS),
            "tiers": {k: {"label": t.label, "gpus": t.gpus, "notes": t.notes, "gated": sorted(t.gated)}
                      for k, t in tiers.TIERS.items()},
            "builtin_examples": examples,
            "note": "Built-in formats are original recipes generated around your photo; MC ships no meme footage. "
                    "Driving clips for swap / copy the vibe are yours: upload footage you have the rights to.",
        }

    def list_library(self, kind: str | None = None) -> dict[str, Any]:
        kinds = [self._kind(kind)] if kind else list(KINDS)
        out: dict[str, Any] = {}
        for k in kinds:
            files = sorted((p for p in (self.library_dir / k).glob("*") if p.is_file()),
                           key=lambda p: p.stat().st_mtime, reverse=True)
            out[k] = [{"path": self.display(p), "bytes": p.stat().st_size} for p in files[:200]]
        out["examples"] = [self.display(p) for p in sorted(self.examples_dir.glob("*")) if p.is_file()
                           and p.suffix.lower() in KINDS["references"] | KINDS["clips"] | KINDS["audio"]]
        for name, d in self.roots().items():
            if d in self.extra_dirs:
                out[name] = [self.display(p) for p in sorted(d.rglob("*"))[:300] if p.is_file()]
        return out

    def _kind(self, kind: str | None) -> str:
        k = (kind or "").strip().lower()
        k = KIND_ALIASES.get(k, k)
        if k not in KINDS:
            raise InputError(f"unknown kind {kind!r} (use references, clips, voices, audio or recipes)")
        return k

    def _dest(self, kind: str, filename: str) -> Path:
        k = self._kind(kind)
        name = _safe_name(filename)
        ext = Path(name).suffix.lower()
        if ext not in KINDS[k]:
            raise InputError(f"{name}: {k} takes {', '.join(sorted(KINDS[k]))}")
        dest = self.library_dir / k / name
        i = 2
        while dest.exists():
            dest = dest.with_name(f"{Path(name).stem}-{i}{ext}")
            i += 1
        return dest

    def save_upload(self, kind: str, filename: str, data: bytes) -> dict[str, Any]:
        if len(data) > MAX_UPLOAD_BYTES:
            raise InputError(f"file is over the {MAX_UPLOAD_BYTES // 2 ** 20} MB upload limit")
        if not data:
            raise InputError("empty file")
        dest = self._dest(kind, filename)
        dest.write_bytes(data)
        info = self._file_info(dest)
        if self._kind(kind) == "references":
            q = refcheck.safe_check(dest)
            if q:
                info["reference_quality"] = q
        return info

    def check_reference(self, path: str, subject: str = "auto") -> dict[str, Any]:
        p = self.resolve(path)
        if p.suffix.lower() not in KINDS["references"]:
            raise InputError(f"{path}: not an image")
        return {"path": self.display(p), **refcheck.check_reference(p, subject)}

    def _reference_warnings(self, req: Request) -> dict[str, Any]:
        """Quality warnings for every reference a render uses, keyed by path (only the ones with warnings)."""
        refs = [req.photo, req.photo_b, req.avatar, *[x for x in (req.people or []) if isinstance(x, (str, Path))]]
        out: dict[str, Any] = {}
        for r in refs:
            if not r or str(r) == "-":
                continue
            q = refcheck.safe_check(r, getattr(req, "subject", "auto"))
            if q and q.get("warnings"):
                out[self.display(r)] = {"grade": q["grade"], "warnings": q["warnings"]}
        if out:
            out["tip"] = refcheck.TIP
        return out

    def upload(self, kind: str, filename: str = "", content_base64: str | None = None, url: str | None = None,
               source_path: str | None = None) -> dict[str, Any]:
        given = [x for x in (content_base64, url, source_path) if x]
        if len(given) != 1:
            raise InputError("give exactly one of content_base64, url or source_path")
        if content_base64:
            b64 = content_base64.split(",", 1)[1] if content_base64.startswith("data:") else content_base64
            try:
                data = base64.b64decode(b64, validate=False)
            except Exception:
                raise InputError("content_base64 isn't valid base64") from None
            return self.save_upload(kind, filename or "upload", data)
        if source_path:
            src = self.resolve(source_path)
            return self.save_upload(kind, filename or src.name, src.read_bytes())
        return self.save_upload(kind, filename or _safe_name(urlparse(url).path) or "download", self._fetch(url))

    # ---------------------------------------------------------- bring your own clip
    def probe_clip(self, url: str) -> dict[str, Any]:
        """What's at a link and the sharpest copy available (nothing is downloaded)."""
        from . import clipfetch

        return {**clipfetch.probe(url), "tip": clipfetch.BEST_RESULTS_TIP}

    def find_clips(self, query: str, limit: int = 5) -> dict[str, Any]:
        from . import clipfetch

        return {"candidates": clipfetch.search(query, limit), "tip": clipfetch.BEST_RESULTS_TIP,
                "rights": clipfetch.RIGHTS_LINE}

    def fetch_clip(self, *, url: str, name: str | None = None, start_seconds: float | None = None,
                   end_seconds: float | None = None, max_height: int | None = None,
                   i_have_rights: bool = False) -> dict[str, Any]:
        """Queue a download of the best video + audio at ``url`` into library/clips (a cpu job, no gpu)."""
        from . import clipfetch
        from .pipeline import Event

        if not i_have_rights:
            raise InputError("confirm with the user that they have the rights to this clip (i_have_rights=true). "
                             + clipfetch.RIGHTS_LINE)
        clipfetch._need()
        clipfetch.check_url(url)
        dest = self.library_dir / "clips"

        def run(ctx: JobContext) -> Iterator[Any]:
            def progress(frac: float, msg: str) -> None:
                ctx.tracker.set(fraction=round(frac, 3), message=msg)

            got = clipfetch.fetch(url, dest, name=name, start=start_seconds, end=end_seconds, max_height=max_height,
                                  progress=progress, cancelled=lambda: ctx.cancelled)
            path = got.pop("path")
            data = {**got, "path": self.display(path), "file": self._file_info(path)}
            yield Event("complete", "complete", 1.0, f"fetched {data['quality']} into {self.display(path)}",
                        artifact=path, data=data)

        params = {"url": url, "name": name, "start_seconds": start_seconds, "end_seconds": end_seconds,
                  "max_height": max_height}
        rec = self.queue.submit("fetch_clip", f"fetch clip {name or url}"[:80],
                                {k: v for k, v in params.items() if v is not None}, run, gpu=False)
        return self.job_summary(rec)

    def _fetch(self, url: str) -> bytes:
        """Download a public http(s) file (no local / private network addresses)."""
        import httpx

        u = urlparse(url)
        if u.scheme not in ("http", "https") or not u.hostname:
            raise AccessError("only http(s) URLs")
        for info in socket.getaddrinfo(u.hostname, u.port or (443 if u.scheme == "https" else 80)):
            if clipfetch.blocked_ip(info[4][0]):
                raise AccessError("MC won't fetch from local or private network addresses")
        buf = bytearray()
        with httpx.stream("GET", url, follow_redirects=False, timeout=60) as r:
            r.raise_for_status()
            for chunk in r.iter_bytes():
                buf += chunk
                if len(buf) > MAX_UPLOAD_BYTES:
                    raise InputError(f"download is over the {MAX_UPLOAD_BYTES // 2 ** 20} MB limit")
        return bytes(buf)

    def _file_info(self, p: Path) -> dict[str, Any]:
        info: dict[str, Any] = {"path": self.display(p), "bytes": p.stat().st_size,
                                "mime": mimetypes.guess_type(p.name)[0] or "application/octet-stream"}
        try:
            if p.suffix.lower() in KINDS["clips"] | KINDS["audio"]:
                info["seconds"] = round(media.media_duration(p), 2)
            if p.suffix.lower() in KINDS["clips"] and p.suffix.lower() != ".gif":
                info["size"] = list(media.video_size(p))
            elif p.suffix.lower() in KINDS["references"]:
                from PIL import Image

                with Image.open(p) as im:
                    info["size"] = list(im.size)
        except Exception:  # noqa: BLE001 - info is best effort
            pass
        return info

    # =============================================================== system
    def system_info(self) -> dict[str, Any]:
        from . import redub, system

        choice = tiers.resolve(None)
        st = self.get_settings()
        active = self.tier_key()
        t = tiers.get(active)
        health: dict[str, Any] = {"backend": self.backend_kind}
        if self.backend_kind == "comfy":
            try:
                import httpx

                r = httpx.get(f"{self.settings.comfy_base_url}/api/v2/health", timeout=4)
                health.update(proxy="ok" if r.status_code == 200 else f"http {r.status_code}")
            except Exception as exc:  # noqa: BLE001
                health.update(proxy=f"unreachable ({type(exc).__name__}): start ComfyUI, then run.ps1 / run.sh")
        jobs = self.store.list(limit=50)
        return {
            "mc_version": VERSION,
            "gpu": system.gpu_name() or "unknown",
            "vram_gb": system.vram_gb(),
            "detected_tier": {"key": choice.key, "label": choice.tier.label, "how": choice.describe()},
            "active_tier": {"key": active, "label": t.label, "notes": t.notes, "needs_16gb": sorted(t.gated),
                            "comfy_flags": t.comfy_flags},
            "tiers": {"lite": "8-12 GB: stills, short silent memes; 16 GB modes gated",
                      "standard": "16 GB (reference RTX 5080): every mode",
                      "max": "24 GB+: bigger, longer renders"},
            "health": health,
            "comfyui_queue": self.comfy_queue(),
            "gpu_lane": lane_owner(self.output_dir),
            "agent_queue": {"queued": sum(1 for j in jobs if j["status"] == "queued"),
                            "running": sum(1 for j in jobs if j["status"] == "running")},
            "deepfilternet": redub.deepfilter_status(),
            "folders": {k: str(v) for k, v in self.roots().items()},
            "settings": st,
        }

    def check_models(self, tier: str | None = None) -> str:
        from .doctor import live_report

        return live_report(self.tier_key(tier), self.comfyui_url)

    # ================================================================= jobs
    def job_summary(self, rec: dict[str, Any], events: int = 8) -> dict[str, Any]:
        out = {k: rec.get(k) for k in ("id", "kind", "title", "source", "status", "stage", "fraction", "message",
                                       "error", "position")}
        out = {k: v for k, v in out.items() if v not in (None, "")}
        for k in ("created", "started", "finished"):
            if rec.get(k):
                out[k] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(rec[k]))
        if rec.get("started"):
            out["elapsed_s"] = round((rec.get("finished") or time.time()) - rec["started"], 1)
        if rec.get("result"):
            res = rec["result"]
            files = self._result_files(res)
            out["result"] = {"message": res.get("message"), "final": self.display(res.get("artifact")),
                             "files": files}
            data = res.get("data") or {}
            if rec.get("kind") == "fetch_clip":  # the clip lands in the library, not in a run folder
                out["result"]["files"] = [data["path"]] if data.get("path") else []
                for k in ("path", "quality", "width", "height", "fps", "best_available_height", "title", "site",
                          "source_url", "section", "tips", "note"):
                    if data.get(k) not in (None, "", []):
                        out["result"][k] = data[k]
            for k in ("post_text", "people", "duration", "redub", "prompt", "boxes", "clips"):
                if k in data:
                    out["result"][k] = data[k] if k != "clips" else [self.display(c) for c in data[k]]
        if events:
            out["recent_events"] = [f"[{e['stage']}] {e['status']}: {e['message']}" for e in rec.get("events", [])
                                    [-events:]]
        return out

    def _result_files(self, res: dict[str, Any]) -> list[str]:
        art = res.get("artifact")
        files: list[str] = []
        if art and Path(art).exists():
            top = Path(art).parent
            files = [self.display(p) for p in sorted(top.iterdir()) if p.is_file() and p.suffix.lower() in OUTPUT_EXT]
        for k in ("gif", "still", "recipe", "preview"):
            v = (res.get("data") or {}).get(k)
            if v and Path(v).exists() and self.display(v) not in files:
                files.append(self.display(v))
        return files

    def job(self, job_id: str) -> dict[str, Any]:
        try:
            return self.job_summary(self.queue.status(job_id), events=15)
        except KeyError:
            raise InputError(f"no job {job_id!r} (list_jobs shows recent ones)") from None

    def list_jobs(self, status: str | None = None, limit: int = 20) -> dict[str, Any]:
        recs = self.store.list(limit=limit, status=status)
        for r in recs:
            if r["status"] == "queued" and r["id"] in self.queue.pending():
                r["position"] = self.queue.pending().index(r["id"]) + 1
        return {"jobs": [self.job_summary(r, events=0) for r in recs], "gpu_lane": lane_owner(self.output_dir),
                "comfyui_queue": self.comfy_queue()}

    def cancel(self, job_id: str) -> dict[str, Any]:
        try:
            rec = self.store.read(job_id)
        except KeyError:
            raise InputError(f"no job {job_id!r}") from None
        if rec.get("source") == "ui" and rec.get("status") in ACTIVE:
            self.store.update(job_id, cancel_requested=True)
            return {**self.job_summary(rec, 0), "note": "asked the mc window to stop this render"}
        return self.job_summary(self.queue.cancel(job_id), 0)

    def wait(self, job_id: str, timeout: float = 600) -> dict[str, Any]:
        try:
            return self.job_summary(self.queue.wait(job_id, timeout), events=15)
        except KeyError:
            raise InputError(f"no job {job_id!r}") from None

    # ============================================================== outputs
    def list_outputs(self, limit: int = 20) -> dict[str, Any]:
        runs = []
        dirs = [d for d in self.output_dir.iterdir() if d.is_dir() and not d.name.startswith("_")]
        for d in sorted(dirs, key=lambda d: d.stat().st_mtime, reverse=True)[:limit]:
            files = [self.display(p) for p in sorted(d.iterdir()) if p.is_file() and p.suffix.lower() in OUTPUT_EXT]
            if files:
                runs.append({"run": d.name, "files": files,
                             "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(d.stat().st_mtime))})
        return {"outputs_dir": str(self.output_dir), "runs": runs}

    def file_info(self, path: str) -> dict[str, Any]:
        p = self.resolve(path)
        return {**self._file_info(p), "absolute_path": str(p)}

    def read_file(self, path: str, max_bytes: int = 8 * 2 ** 20) -> tuple[Path, bytes]:
        p = self.resolve(path)
        size = p.stat().st_size
        if size > max_bytes:
            raise InputError(f"{p.name} is {size / 2 ** 20:.1f} MB; that's over the {max_bytes / 2 ** 20:.0f} MB "
                             "inline limit. Use the download link (HTTP mode) or the path on the PC.")
        return p, p.read_bytes()

    # =========================================================== GPU jobs
    def make_meme(self, *, photo: str | None = None, meme: str = memes.DEFAULT_RECIPE, animate: bool = True,
                  top: str | None = None, bottom: str | None = None, style: str = "none", era: str = "classic",
                  effect: str | None = None, seconds: int | None = None, say_line: bool = False, line: str = "",
                  voice: str = "builtin", voice_sample: str | None = None, recording: str | None = None,
                  custom_scene: str = "", custom_action: str = "", scene_image: str | None = None,
                  subject: str = "auto", i_have_rights: bool = False, tier: str | None = None,
                  seed: int | None = None) -> dict[str, Any]:
        b = self._base(tier, seed)
        if voice not in ("builtin", "clone", "my_recording"):
            raise InputError("voice is builtin, clone or my_recording")
        req = Request(mode="meme", photo=self.resolve(photo, kinds=KINDS["references"]), meme=meme,
                      meme_top=top, meme_bottom=bottom, style=style, era=era, effect=effect,
                      meme_frames={3: 49, 5: 81}.get(int(seconds)) if seconds else None, say_line=say_line,
                      line=line, voice_mode=voice, voice_sample=self.resolve(voice_sample),
                      recording=self.resolve(recording), custom_scene=custom_scene, custom_motion=custom_action,
                      avatar=self.resolve(scene_image, kinds=KINDS["references"]), subject=subject,
                      consent=i_have_rights, **b)
        title = f"meme {meme}" + (" (still)" if not animate else "")
        return self._submit_request("meme", title, req, "run" if animate else "run_still")

    def make_meme_pack(self, *, photo: str, memes_list: list[str], style: str = "none",
                       i_have_rights: bool = False, tier: str | None = None, seed: int | None = None) -> dict[str, Any]:
        unknown = [k for k in memes_list if k not in memes.RECIPES]
        if unknown or not memes_list:
            raise InputError(f"unknown formats: {unknown} (list_templates shows them)" if unknown else
                             "pick at least one format")
        req = Request(mode="meme", photo=self.resolve(photo, kinds=KINDS["references"]), style=style,
                      consent=i_have_rights, **self._base(tier, seed))
        check_request(replace(req, meme=memes_list[0]), self.settings)
        return self._submit_request("pack", f"meme pack ({len(memes_list)})", req, "run_pack", (list(memes_list),))

    def swap(self, *, clip: str, photo: str | None = None, people: list[str | None] | None = None,
             describe: list[str] | None = None, face: list[str] | None = None, clip_start: float = 0.0,
             clip_seconds: float = 5.0, clip_audio: str = "original", voice_sample: str | None = None,
             group_engine: str = "auto", upscale: str = "auto", resolution: str = "auto", erase_props: str = "",
             relight: bool | None = None, remove_props: bool = False, style: str = "none", look_tweak: str = "",
             swap_prompt: str = "", top: str = "", bottom: str = "", subject: str = "auto",
             i_have_rights: bool = False, tier: str | None = None, seed: int | None = None,
             steps: int | None = None, style_lock: str = "off", drivers: list[str | None] | None = None,
             split_cuts: str = "auto", scail_chunks: bool = False, rotate: int = 0,
             ref_closeup: str = "off") -> dict[str, Any]:
        if style_lock not in ("off", "auto", "on"):
            raise InputError("style_lock is off, auto or on")
        if split_cuts not in ("auto", "on", "off"):
            raise InputError("split_cuts is auto, on or off")
        moves = []
        for x in drivers or []:
            s = "" if x is None or str(x).strip() in ("", "-", "self") else str(x).strip()
            if s and not re.fullmatch(r"(?:person|p)?\s*[:#]?\s*\d+", s, flags=re.I):
                s = str(self.resolve(s, kinds=KINDS["clips"]))
            moves.append(s)
        refs = [None if (x is None or str(x).strip() in ("", "-")) else self.resolve(x, kinds=KINDS["references"])
                for x in (people or [])]
        req = Request(mode="swap", clip=self.resolve(clip, kinds=KINDS["clips"]),
                      photo=self.resolve(photo, kinds=KINDS["references"]) if not refs else None, people=refs,
                      people_desc=[d.strip() for d in (describe or [])], face=[f or "auto" for f in (face or [])],
                      clip_start=float(clip_start), clip_seconds=float(clip_seconds), clip_audio=clip_audio,
                      voice_sample=self.resolve(voice_sample), group_engine=group_engine, upscale=upscale,
                      resolution=resolution, erase_props=erase_props, relight=relight, remove_props=remove_props,
                      style=style, style_extra=look_tweak, swap_prompt=swap_prompt, meme_top=top, meme_bottom=bottom,
                      subject=subject, consent=i_have_rights, steps=steps, style_lock=style_lock,
                      people_driver=moves, split_cuts=split_cuts, scail_chunks=bool(scail_chunks), rotate=int(rotate or 0),
                      ref_closeup=ref_closeup,
                      **self._base(tier, seed))
        if group_engine not in ("auto", "scail", "chain"):
            raise InputError("group_engine is auto, scail or chain")
        if upscale not in ("auto", "seedvr2", "lanczos"):
            raise InputError("upscale is auto, seedvr2 or lanczos")
        n = len([r for r in refs if r]) or 1
        return self._submit_request("swap", f"swap {n} into {Path(clip).stem}", req)

    def find_people(self, *, clip: str, clip_start: float = 0.0, clip_seconds: float = 5.0,
                    tier: str | None = None) -> dict[str, Any]:
        req = Request(mode="swap", clip=self.resolve(clip, kinds=KINDS["clips"]), clip_start=float(clip_start),
                      clip_seconds=float(clip_seconds), **self._base(tier, None))
        return self._submit_request("find_people", f"find people in {Path(clip).stem}", req, "scan_people")

    def copy_the_vibe(self, *, clip: str, photo: str, scene: str = "", clip_start: float = 0.0,
                      clip_seconds: float = 5.0, clip_audio: str = "original", voice_sample: str | None = None,
                      style: str = "none", look_tweak: str = "", resolution: str = "auto", top: str = "",
                      bottom: str = "", subject: str = "auto", scene_image: str | None = None,
                      i_have_rights: bool = False, tier: str | None = None, seed: int | None = None) -> dict[str, Any]:
        req = Request(mode="vibe", clip=self.resolve(clip, kinds=KINDS["clips"]),
                      photo=self.resolve(photo, kinds=KINDS["references"]), scene_prompt=scene,
                      clip_start=float(clip_start), clip_seconds=float(clip_seconds), clip_audio=clip_audio,
                      voice_sample=self.resolve(voice_sample), style=style, style_extra=look_tweak,
                      resolution=resolution, meme_top=top, meme_bottom=bottom, subject=subject,
                      avatar=self.resolve(scene_image, kinds=KINDS["references"]), consent=i_have_rights,
                      **self._base(tier, seed))
        return self._submit_request("vibe", f"copy the vibe: {scene or 'studio'}"[:60], req)

    def talk(self, *, photo: str | None = None, character: str = "", avatar: str | None = None, script: str = "",
             voice: str = "builtin", voice_sample: str | None = None, recording: str | None = None,
             character_voice: str | None = None, style: str = "none", captions: bool = True,
             smooth: str = "off", i_have_rights: bool = False, tier: str | None = None,
             seed: int | None = None) -> dict[str, Any]:
        if voice not in ("builtin", "clone", "my_recording"):
            raise InputError("voice is builtin, clone or my_recording")
        req = Request(mode="talk", photo=self.resolve(photo, kinds=KINDS["references"]), character=character,
                      avatar=self.resolve(avatar, kinds=KINDS["references"]), script=script, voice_mode=voice,
                      voice_sample=self.resolve(voice_sample), recording=self.resolve(recording),
                      character_voice=self.resolve(character_voice), style=style, captions=captions, smooth=smooth,
                      consent=i_have_rights, **self._base(tier, seed))
        return self._submit_request("talk", f"talk: {script[:40] or 'recording'}", req)

    def sing(self, *, song: str, photo: str | None = None, character: str = "", avatar: str | None = None,
             vocals: str | None = None, song_start: float = 0.0, song_seconds: float = 20.0, lyrics: str = "",
             style: str = "none", i_have_rights: bool = False, tier: str | None = None,
             seed: int | None = None) -> dict[str, Any]:
        req = Request(mode="sing", song=self.resolve(song, kinds=KINDS["audio"] | KINDS["clips"]),
                      vocals=self.resolve(vocals), photo=self.resolve(photo, kinds=KINDS["references"]),
                      character=character, avatar=self.resolve(avatar, kinds=KINDS["references"]),
                      song_start=float(song_start), song_seconds=float(song_seconds), script=lyrics, style=style,
                      consent=i_have_rights, **self._base(tier, seed))
        return self._submit_request("sing", f"sing {Path(song).stem}", req)

    def dialogue(self, *, script: str, photo: str | None = None, photo_b: str | None = None, character: str = "",
                 voice_sample: str | None = None, voice_sample_b: str | None = None, style: str = "3d",
                 i_have_rights: bool = False, tier: str | None = None, seed: int | None = None) -> dict[str, Any]:
        req = Request(mode="dialogue", script=script, photo=self.resolve(photo, kinds=KINDS["references"]),
                      photo_b=self.resolve(photo_b, kinds=KINDS["references"]), character=character,
                      voice_sample=self.resolve(voice_sample), voice_sample_b=self.resolve(voice_sample_b),
                      style=style, consent=i_have_rights, **self._base(tier, seed))
        return self._submit_request("dialogue", "duet skit", req)

    def batch_talk(self, *, scripts: str, photo: str | None = None, character: str = "",
                   voice_sample: str | None = None, style: str = "none", i_have_rights: bool = False,
                   tier: str | None = None, seed: int | None = None) -> dict[str, Any]:
        items = split_batch_text(scripts)
        if not items:
            raise InputError("add at least one script")
        req = Request(mode="talk", photo=self.resolve(photo, kinds=KINDS["references"]), character=character,
                      voice_mode="clone" if voice_sample else "builtin", voice_sample=self.resolve(voice_sample),
                      style=style, consent=i_have_rights, **self._base(tier, seed))
        check_request(replace(req, script=items[0]), self.settings)
        return self._submit_request("batch", f"batch of {len(items)}", req, "run_batch", (items,))

    def remix(self, *, recipe: str, photo: str | None = None, clip: str | None = None,
              voice_sample: str | None = None, i_have_rights: bool = False, tier: str | None = None) -> dict[str, Any]:
        doc = json.loads(self.resolve(recipe, kinds=KINDS["recipes"]).read_text(encoding="utf-8"))
        uploads = {k: v for k, v in dict(photo=self.resolve(photo), clip=self.resolve(clip),
                                         voice_sample=self.resolve(voice_sample)).items() if v}
        req = request_from_recipe(doc, consent=i_have_rights, tier=self.tier_key(tier), **uploads)
        return self._submit_request("remix", f"remix {Path(recipe).stem}", req)

    def tts(self, *, text: str, voice_sample: str | None = None, expressiveness: float = 0.55,
            i_have_rights: bool = False, seed: int | None = None) -> dict[str, Any]:
        vs = self.resolve(voice_sample)
        seed_ = self._base(None, seed)["seed"]
        if not (text or "").strip():
            raise InputError("type what the voice should say")
        if vs and not i_have_rights:
            raise InputError("confirm i_have_rights: the voice sample is yours or you have the speaker's OK")
        return self._submit("tts", f"voice: {text[:40]}", {"text": text, "voice_sample": vs},
                            lambda p: extras.run_tts(p, text, voice_sample=vs, expressiveness=expressiveness,
                                                     seed=seed_, consent=i_have_rights))

    def convert_voice(self, *, source: str, target_voice: str, i_have_rights: bool = False) -> dict[str, Any]:
        src, tgt = self.resolve(source), self.resolve(target_voice)
        if not i_have_rights:
            raise InputError("confirm i_have_rights: you have the rights to both voices")
        return self._submit("voice_convert", f"convert {src.stem}", {"source": src, "target_voice": tgt},
                            lambda p: extras.run_voice_convert(p, src, tgt, consent=True))

    def restyle(self, *, image: str, style: str = "none", background: str = "", instruction: str = "",
                subject: str = "auto", landscape: bool = False, i_have_rights: bool = False,
                tier: str | None = None, seed: int | None = None) -> dict[str, Any]:
        img = self.resolve(image, kinds=KINDS["references"])
        b = self._base(tier, seed)
        if style not in STYLES:
            raise InputError(f"unknown style {style!r}")
        if not i_have_rights:
            raise InputError("confirm i_have_rights: it's you or your own avatar, or you have their OK")
        if not (background.strip() or instruction.strip() or style not in ("none", "real")):
            raise InputError("say what to change: a style, a new background, or an instruction")
        return self._submit("restyle", f"restyle: {background or style}"[:60],
                            {"image": img, "style": style, "background": background, "instruction": instruction},
                            lambda p: extras.run_restyle(p, img, style=style, background=background, extra=instruction,
                                                         subject=subject, portrait=not landscape, seed=b["seed"],
                                                         consent=True, tier=b["tier"]))

    def upscale_video(self, *, video: str, engine: str = "seedvr2", width: int | None = None,
                      height: int | None = None, tier: str | None = None, seed: int | None = None) -> dict[str, Any]:
        v = self.resolve(video, kinds=KINDS["clips"])
        b = self._base(tier, seed)
        if engine == "seedvr2" and b["tier"] == "lite":
            raise InputError("SeedVR2 needs a 16 GB+ GPU. Use engine='lanczos' on Lite.")
        return self._submit("upscale", f"upscale {v.stem}", {"video": v, "engine": engine},
                            lambda p: extras.run_upscale(p, v, engine=engine, width=width, height=height,
                                                         seed=b["seed"], tier=b["tier"]), gpu=engine == "seedvr2")

    def redub(self, *, video: str, recording: str | None = None, text: str = "", voice_sample: str | None = None,
              start: float = 0.0, fit_seconds: float | None = None, duck: float = 0.28, denoise: str = "auto",
              voice_level_db: float = 0.0, i_have_rights: bool = False, seed: int | None = None,
              lipsync: bool = False) -> dict[str, Any]:
        v = self.resolve(video, kinds=KINDS["clips"])
        rec, vs = self.resolve(recording), self.resolve(voice_sample)
        if not (rec or text.strip()) or (rec and text.strip()):
            raise InputError("give either a recording or the text of the new line")
        if not i_have_rights:
            raise InputError("confirm i_have_rights: the clip and the voice are yours, or you have the rights")
        seed_ = self._base(None, seed)["seed"]
        return self._submit("redub", f"redub {v.stem}", {"video": v, "recording": rec, "text": text, "start": start},
                            lambda p: extras.run_redub(p, v, recording=rec, text=text, voice_sample=vs, start=start,
                                                       fit_seconds=fit_seconds, duck=duck, denoise=denoise,
                                                       voice_level_db=voice_level_db, seed=seed_, consent=True,
                                                       lipsync=bool(lipsync)),
                            gpu=bool(text.strip()) or bool(lipsync))

    def save_format(self, *, name: str, meme: str = "custom", scene: str = "", action: str = "", top: str = "",
                    bottom: str = "", line: str = "", emoji: str = "") -> dict[str, Any]:
        base = memes.custom_recipe(scene, action, top, bottom, line) if meme == "custom" else memes.RECIPES.get(meme)
        if base is None:
            raise InputError(f"unknown format {meme!r}")
        try:
            key, path = memes.save_format(base, name, emoji=emoji, captions=[(top, bottom)] if (top or bottom) else [],
                                          source=f"agent:{meme}")
        except ValueError as exc:
            raise InputError(str(exc)) from None
        return {"key": key, "path": str(path)}


def _inside(p: Path, roots) -> bool:
    try:
        rp = p.resolve()
    except OSError:
        return False
    return any(rp == r or r in rp.parents for r in roots)
