"""Execution backends.

* :class:`ComfyBackend` runs every stage through the official **Comfy SDK**
  (``comfy-sdk``) against a local ComfyUI exposed as Comfy API v2 by
  ``comfy-api-proxy`` (127.0.0.1:8188 -> 127.0.0.1:8189).
* :class:`MockBackend` produces placeholder media with Pillow + ffmpeg so the UI
  and pipeline can be developed and tested on machines without a GPU.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from .graphs import StageGraph

ProgressFn = Callable[[float, str], None]


@dataclass
class StageResult:
    path: Path
    job_id: str | None = None
    seconds: float = 0.0
    paths: list[Path] = field(default_factory=list)  # every output (people scan: one mask per person)
    extras: dict[str, Path] = field(default_factory=dict)  # side outputs (e.g. the cut-out reference image)


class Backend(Protocol):
    name: str

    def run_stage(
        self, stage: str, sg: StageGraph, files: dict[str, Path], out_dir: Path, progress: ProgressFn
    ) -> StageResult: ...


# --------------------------------------------------------------------------- #
# Real backend: Comfy SDK -> comfy-api-proxy -> ComfyUI
# --------------------------------------------------------------------------- #
class ComfyBackend:
    name = "comfy"

    def __init__(self, base_url: str | None = None, api_key: str | None = None, timeout: float = 60.0) -> None:
        # The SDK reads its target from COMFY_BASE_URL each time a client is built.
        if base_url:
            os.environ["COMFY_BASE_URL"] = base_url
        os.environ.setdefault("COMFY_BASE_URL", "http://127.0.0.1:8189")
        from comfy_sdk import Comfy  # imported lazily so the mock path needs no SDK

        self.base_url = os.environ["COMFY_BASE_URL"].rstrip("/")
        self.client = Comfy(api_key=api_key, timeout=timeout, client_info="main-character-comfy/0.2")
        self._job = None
        self._cancelled = False

    def cancel(self) -> bool:
        """Stop button: ask ComfyUI to cancel the running job. Returns True if one was running."""
        self._cancelled = True
        job = self._job
        if job is None:
            return False
        try:
            job.cancel()
        except Exception:  # already finished, or the proxy is gone: nothing left to stop
            pass
        return True

    def health(self) -> dict:
        import httpx

        r = httpx.get(f"{self.base_url}/api/v2/health", timeout=5)
        r.raise_for_status()
        return r.json()

    def run_stage(
        self, stage: str, sg: StageGraph, files: dict[str, Path], out_dir: Path, progress: ProgressFn
    ) -> StageResult:
        from comfy_sdk import JobFailed, Progress, StatusChange

        t0 = time.monotonic()
        wf = self.client.workflows.from_json(sg.graph)
        for logical, (node_id, field) in sg.inputs.items():
            if logical not in files:
                raise ValueError(f"stage {stage}: missing input file '{logical}'")
            # Lazy, content-addressed handle: hashed locally, uploaded only if new.
            wf.set_input(node_id, field, self.client.assets.from_file(files[logical]))

        from .errors import Cancelled

        self._cancelled = False
        progress(0.0, "queued")
        job = self.client.submit(wf, metadata=None)
        self._job = job
        windows = int(sg.meta.get("windows", 1) or 1)
        sampler_runs = 0
        last_step = -1
        for ev in job.events():
            if isinstance(ev, Progress):
                if ev.steps:
                    # Each lip-sync window restarts the sampler step counter.
                    if ev.step is not None and ev.step < last_step:
                        sampler_runs += 1
                    last_step = ev.step if ev.step is not None else last_step
                    win = min(sampler_runs, windows - 1)
                    frac = (win + (ev.step or 0) / ev.steps) / windows
                    label = f"step {ev.step}/{ev.steps}"
                    if windows > 1:
                        label = f"window {win + 1}/{windows} · " + label
                    progress(frac, label)
                else:
                    progress(ev.value, ev.message or "running")
            elif isinstance(ev, StatusChange):
                progress(0.0 if ev.status == "queued" else -1, ev.status)
            if self._cancelled:
                break
        if self._cancelled:
            self._job = None
            raise Cancelled(f"{stage} stopped")
        try:
            job = job.result()  # poll-authoritative; raises JobFailed with node detail
        except JobFailed as exc:
            self._job = None
            err = exc.error
            detail = ""
            if err is not None:
                node = getattr(err, "node_id", None)
                detail = f" (node {node}: {getattr(err, 'message', '')})" if node else f" ({getattr(err, 'message', '')})"
            raise RuntimeError(f"{stage} failed in ComfyUI: {exc}{detail}") from exc
        self._job = None
        outputs = job.get_outputs(sg.output_node)
        if not outputs:
            raise RuntimeError(f"{stage}: job {job.id} finished without outputs on node {sg.output_node}")
        out_dir.mkdir(parents=True, exist_ok=True)
        if sg.meta.get("all_outputs"):
            paths = []
            for i, o in enumerate(outputs):
                dest = out_dir / f"{stage}_{i}{Path(o.name).suffix or ''}"
                o.to_file(dest)
                paths.append(dest)
            progress(1.0, "done")
            return StageResult(paths[0], job.id, time.monotonic() - t0, paths)
        o = outputs[0]
        dest = out_dir / f"{stage}{Path(o.name).suffix or ''}"
        o.to_file(dest)
        extras: dict[str, Path] = {}
        for name, node in (sg.meta.get("extra_outputs") or {}).items():
            try:  # best effort: a missing side output never fails the render
                side = job.get_outputs(node)
                if side:
                    extras[name] = out_dir / f"{stage}_{name}{Path(side[0].name).suffix or '.png'}"
                    side[0].to_file(extras[name])
            except Exception:
                pass
        progress(1.0, "done")
        return StageResult(dest, job.id, time.monotonic() - t0, [dest], extras)


# --------------------------------------------------------------------------- #
# Mock backend: no GPU, deterministic placeholder media
# --------------------------------------------------------------------------- #
class MockBackend:
    name = "mock"

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self._cancelled = False

    def cancel(self) -> bool:
        self._cancelled = True
        return True

    def _tick(self, progress: ProgressFn, label: str, n: int = 4) -> None:
        from .errors import Cancelled

        for i in range(n):
            if self._cancelled:
                self._cancelled = False
                raise Cancelled("stopped")
            progress(i / n, f"{label} {i + 1}/{n} (mock)")
            if self.delay:
                time.sleep(self.delay / n)

    def run_stage(
        self, stage: str, sg: StageGraph, files: dict[str, Path], out_dir: Path, progress: ProgressFn
    ) -> StageResult:
        from . import mock_media

        self._cancelled = False
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.monotonic()
        classes = {n["class_type"] for n in sg.graph.values()}
        self._tick(progress, "step")
        if "SAM3_Detect" in classes and "frame" in sg.inputs:  # the people scan (not an avatar-ref cut-out)
            paths = mock_media.fake_people_masks(files["frame"], out_dir, stage, int(os.environ.get("MC_MOCK_PEOPLE", 2)))
            progress(1.0, "done (mock)")
            return StageResult(paths[0], None, time.monotonic() - t0, paths)
        if "WanAnimateToVideo" in classes:
            dest = mock_media.fake_animate(sg.meta.get("mode", "replace"), files["clip"], files["avatar"],
                                           int(sg.meta.get("out_frames", 77)), int(sg.meta.get("lead", 0)),
                                           int(sg.meta.get("fps", 16)), out_dir / f"{stage}.mp4")
            extras = {}
            if "ref_cutout" in (sg.meta.get("extra_outputs") or {}):
                extras["ref_cutout"] = mock_media.fake_cutout(files["avatar"], out_dir / f"{stage}_ref_cutout.png")
            progress(1.0, "done (mock)")
            return StageResult(dest, None, time.monotonic() - t0, [dest], extras)
        if "SeedVR2Conditioning" in classes:
            w, h = sg.meta.get("size", [1080, 1920])
            dest = mock_media.fake_upscale(files["clip"], int(w), int(h), int(sg.meta.get("fps", 16)),
                                           out_dir / f"{stage}.mp4")
            progress(1.0, "done (mock)")
            return StageResult(dest, None, time.monotonic() - t0, [dest])
        if "VOIDSampler" in classes:
            dest = mock_media.fake_erase(files["clip"], int(sg.meta.get("frames", 16)), int(sg.meta.get("fps", 16)),
                                         out_dir / f"{stage}.mp4")
            progress(1.0, "done (mock)")
            return StageResult(dest, None, time.monotonic() - t0, [dest])
        if "SaveImage" in classes:
            prompt = next(n["inputs"]["text"] for n in sg.graph.values() if n["class_type"] == "CLIPTextEncode")
            dest = mock_media.fake_avatar(prompt, files.get("photo"), out_dir / f"{stage}.png")
        elif "ChatterboxTTS" in classes:
            text = next(n["inputs"]["text"] for n in sg.graph.values() if n["class_type"] == "ChatterboxTTS")
            dest = mock_media.fake_speech(text, out_dir / f"{stage}.flac")
        elif "ChatterboxVC" in classes:
            dest = mock_media.fake_voice_change(files["source"], out_dir / f"{stage}.flac")
        elif "WanInfiniteTalkToVideo" in classes:
            frames = int(sg.meta.get("total_frames", 81))
            dest = mock_media.fake_talking(files["avatar"], files["speech"], frames, out_dir / f"{stage}.mp4")
        elif "WanImageToVideo" in classes or "Wan22ImageToVideoLatent" in classes:
            dest = mock_media.fake_motion(files["avatar"], int(sg.meta.get("frames", 81)), int(sg.meta.get("fps", 16)),
                                          out_dir / f"{stage}.mp4")
        elif "WanAnimateToVideo" in classes:
            dest = mock_media.fake_animate(sg.meta.get("mode", "replace"), files["clip"], files["avatar"],
                                           int(sg.meta.get("out_frames", 77)), int(sg.meta.get("lead", 0)),
                                           int(sg.meta.get("fps", 16)), out_dir / f"{stage}.mp4")
        elif "FrameInterpolate" in classes:
            dest = mock_media.fake_smooth(files["video"], out_dir / f"{stage}.mp4")
        else:
            raise ValueError(f"mock backend can't run stage {stage}")
        progress(1.0, "done (mock)")
        return StageResult(dest, None, time.monotonic() - t0)


def make_backend(kind: str, base_url: str | None = None, api_key: str | None = None) -> Backend:
    if kind == "mock":
        return MockBackend()
    if kind == "comfy":
        return ComfyBackend(base_url=base_url, api_key=api_key)
    raise ValueError(f"unknown backend '{kind}' (use 'comfy' or 'mock')")
