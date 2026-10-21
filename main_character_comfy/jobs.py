"""One GPU lane and one job list, shared by the web UI, the CLI and the agent connector.

* :func:`gpu_lane` is a context manager every render holds while it talks to ComfyUI. It combines an in-process
  lock with a lock file in ``<outputs>/_agent/gpu.lock``, so the UI (``app.py``) and an agent server running in
  another process never interleave GPU jobs. Whoever holds it is written next to it (``gpu.owner.json``), so a
  waiting render can say who it is waiting for.
* :class:`JobStore` keeps one small JSON record per job in ``<outputs>/_agent/jobs/``. UI and CLI runs register
  there too (:func:`laned_events`), so an agent asking "what's in the queue?" sees the whole picture.
* :class:`JobQueue` is the agent side: jobs are queued in order and a single worker thread runs them one at a
  time inside the GPU lane. Cancel works for queued and running jobs, also across processes (a cancel flag in
  the record that the owning process watches).

No network, no ComfyUI calls here: the optional "wait until ComfyUI's own queue is empty" courtesy check lives in
:mod:`main_character_comfy.service`.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from .errors import Cancelled

AGENT_SUBDIR = "_agent"
_LANE = threading.Lock()  # in-process half of the GPU lane


def agent_dir(output_dir: str | Path) -> Path:
    d = Path(output_dir) / AGENT_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    if pid == os.getpid():
        return True
    if os.name == "nt":  # pragma: no cover - exercised on the Windows PC
        import ctypes

        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class FileLock:
    """Exclusive, non-blocking lock on a file (msvcrt on Windows, fcntl elsewhere). Released if the process dies."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fh = None

    def try_acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+b")
        try:
            if os.name == "nt":  # pragma: no cover
                import msvcrt

                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            return False
        self._fh = fh
        return True

    def release(self) -> None:
        fh, self._fh = self._fh, None
        if fh is None:
            return
        try:
            if os.name == "nt":  # pragma: no cover
                import msvcrt

                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()


def lane_owner(output_dir: str | Path) -> dict[str, Any] | None:
    """Who holds the GPU lane right now (None if free or the holder is gone)."""
    p = agent_dir(output_dir) / "gpu.owner.json"
    try:
        info = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return info if pid_alive(info.get("pid")) else None


class GpuLane:
    """The one GPU lane: an in-process lock plus ``<outputs>/_agent/gpu.lock`` (cross-process)."""

    def __init__(self, output_dir: str | Path, owner: dict[str, Any]) -> None:
        self.dir = agent_dir(output_dir)
        self.owner = owner
        self._file = FileLock(self.dir / "gpu.lock")
        self.held = False

    def try_acquire(self) -> bool:
        if self.held:
            return True
        if not _LANE.acquire(blocking=False):
            return False
        if not self._file.try_acquire():
            _LANE.release()
            return False
        self.held = True
        with contextlib.suppress(OSError):
            (self.dir / "gpu.owner.json").write_text(json.dumps({**self.owner, "pid": os.getpid(),
                                                                 "since": time.time()}), encoding="utf-8")
        return True

    def release(self) -> None:
        if not self.held:
            return
        self.held = False
        with contextlib.suppress(OSError):
            (self.dir / "gpu.owner.json").unlink()
        self._file.release()
        _LANE.release()


@contextlib.contextmanager
def gpu_lane(output_dir: str | Path, owner: dict[str, Any], *, on_wait: Callable[[dict | None], None] | None = None,
             cancelled: Callable[[], bool] | None = None, poll: float = 0.5) -> Iterator[None]:
    """Hold the one GPU lane for the duration of a render (blocks until it's free).

    ``on_wait(holder)`` is called about every 5 s while waiting; ``cancelled()`` returning True while waiting
    raises :class:`Cancelled` without ever touching the GPU."""
    lane = GpuLane(output_dir, owner)
    last = 0.0
    while not lane.try_acquire():
        if cancelled and cancelled():
            raise Cancelled("cancelled while waiting for the gpu")
        if on_wait and time.monotonic() - last > 5:
            last = time.monotonic()
            on_wait(lane_owner(output_dir))
        time.sleep(poll)
    try:
        yield
    finally:
        lane.release()


def describe_owner(who: dict[str, Any] | None) -> str:
    if not who:
        return "another render"
    src = {"ui": "the mc window", "agent": "an agent", "cli": "the command line"}.get(who.get("source", ""), "another app")
    return f"{src} ('{who.get('title', '')}')" if who.get("title") else src


def laned_events(output_dir: str | Path, events: Iterator[Any], kind: str, title: str, source: str = "ui",
                 cancel: Callable[[], Any] | None = None, wait_event: Callable[[str], Any] | None = None,
                 poll: float = 2.0) -> Iterator[Any]:
    """Run a pipeline event stream inside the GPU lane and mirror it into the shared job list.

    Used by the MC window and the CLI. While another render holds the GPU it yields ``wait_event(message)`` (if
    given) every ``poll`` seconds instead of starting. An agent's cancel_job on this run calls ``cancel()``."""
    lane = GpuLane(output_dir, {"source": source, "title": title, "kind": kind})
    store = JobStore(output_dir)
    tr = JobTracker(store, store.new(kind, title, source))
    try:
        while not lane.try_acquire():
            if tr.cancel_requested():
                raise Cancelled("cancelled while waiting for the gpu")
            msg = f"waiting for the gpu: {describe_owner(lane_owner(output_dir))} is rendering"
            tr.set(message=msg)
            if wait_event:
                yield wait_event(msg)
            time.sleep(poll)
        tr.set(status="running", started=time.time(), message="running")
        for ev in events:
            tr.event(ev)
            if tr.cancel_requested() and cancel:
                cancel()
            yield ev
            if getattr(ev, "status", "") == "complete":
                tr.set(status="done", fraction=1.0, finished=time.time(),
                       result={"artifact": str(ev.artifact) if ev.artifact else None, "message": ev.message,
                               "data": _jsonable(ev.data)})
        if tr.rec.get("status") == "running":
            tr.set(status="done", finished=time.time())
    except Cancelled:
        tr.set(status="cancelled", finished=time.time())
        raise
    except GeneratorExit:
        if tr.rec.get("status") in ACTIVE:
            tr.set(status="cancelled", finished=time.time(), message="stopped")
        raise
    except BaseException as exc:
        tr.set(status="error", error=str(exc)[:500], finished=time.time())
        raise
    finally:
        lane.release()


# --------------------------------------------------------------------------- #
# Job records
# --------------------------------------------------------------------------- #
ACTIVE = ("queued", "running")
FINAL = ("done", "error", "cancelled", "lost")
MAX_EVENTS = 60


class JobStore:
    """``<outputs>/_agent/jobs/<id>.json``, written atomically. Safe to read from any process."""

    def __init__(self, output_dir: str | Path) -> None:
        self.dir = agent_dir(output_dir) / "jobs"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def path(self, job_id: str) -> Path:
        if not job_id or any(c in job_id for c in "/\\.:"):
            raise KeyError(job_id)
        return self.dir / f"{job_id}.json"

    def write(self, rec: dict[str, Any]) -> None:
        p = self.path(rec["id"])
        tmp = p.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        with self._lock:
            tmp.write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
            for _ in range(20):  # Windows: a reader can hold the file for a moment
                try:
                    os.replace(tmp, p)
                    break
                except PermissionError:  # pragma: no cover
                    time.sleep(0.05)

    def read(self, job_id: str) -> dict[str, Any]:
        p = self.path(job_id)
        for _ in range(20):
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
                break
            except FileNotFoundError:
                raise KeyError(job_id) from None
            except (OSError, ValueError):  # mid-replace on Windows
                time.sleep(0.05)
        else:  # pragma: no cover
            raise KeyError(job_id)
        if rec.get("status") in ACTIVE and not pid_alive(rec.get("pid")):
            rec["status"] = "lost"
            rec["error"] = "the MC process running this job stopped before it finished"
        return rec

    def update(self, job_id: str, **changes: Any) -> dict[str, Any]:
        rec = self.read(job_id)
        rec.update(changes)
        self.write(rec)
        return rec

    def list(self, limit: int = 30, status: str | None = None) -> list[dict[str, Any]]:
        out = []
        files = sorted(self.dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in files:
            try:
                rec = self.read(p.stem)
            except KeyError:
                continue
            if status and rec.get("status") != status:
                continue
            out.append(rec)
            if len(out) >= limit:
                break
        return out

    def new(self, kind: str, title: str, source: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        rec = {
            "id": f"{time.strftime('%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}",
            "kind": kind, "title": title, "source": source, "status": "queued",
            "params": params or {}, "pid": os.getpid(), "created": time.time(), "started": None, "finished": None,
            "stage": "", "fraction": 0.0, "message": "queued", "events": [], "result": None, "error": None,
            "cancel_requested": False,
        }
        self.write(rec)
        return rec


def _event_dict(ev: Any) -> dict[str, Any]:
    return {"t": round(time.time(), 1), "stage": getattr(ev, "stage", ""), "status": getattr(ev, "status", ""),
            "fraction": round(float(getattr(ev, "fraction", 0.0) or 0.0), 3), "message": getattr(ev, "message", ""),
            "artifact": str(ev.artifact) if getattr(ev, "artifact", None) else None}


class JobTracker:
    """Mirrors a stream of pipeline events into a job record (throttled writes). Used by the UI and the queue."""

    def __init__(self, store: JobStore, rec: dict[str, Any]) -> None:
        self.store, self.rec = store, rec
        self._last_write = 0.0

    def set(self, flush: bool = True, **changes: Any) -> None:
        self.rec.update(changes)
        if flush:
            self.flush()

    def flush(self) -> None:
        try:  # keep a cancel request written by another process
            disk = self.store.read(self.rec["id"])
            if disk.get("cancel_requested"):
                self.rec["cancel_requested"] = True
        except KeyError:
            pass
        self.store.write(self.rec)
        self._last_write = time.monotonic()

    def event(self, ev: Any) -> None:
        d = _event_dict(ev)
        evs = self.rec.setdefault("events", [])
        if d["status"] == "progress" and evs and evs[-1]["status"] == "progress" and evs[-1]["stage"] == d["stage"]:
            evs[-1] = d  # keep the log short: one progress line per stage
        else:
            evs.append(d)
        del evs[:-MAX_EVENTS]
        if d["stage"] != "complete":
            self.rec.update(stage=d["stage"], message=d["message"])
            if d["fraction"] >= 0:
                self.rec["fraction"] = d["fraction"]
        important = d["status"] in ("start", "done", "skipped", "error", "complete")
        if important or time.monotonic() - self._last_write > 1.0:
            self.flush()

    def cancel_requested(self) -> bool:
        if self.rec.get("cancel_requested"):
            return True
        try:
            return bool(self.store.read(self.rec["id"]).get("cancel_requested"))
        except KeyError:
            return False


# --------------------------------------------------------------------------- #
# Agent queue: one worker, FIFO, inside the GPU lane
# --------------------------------------------------------------------------- #
@dataclass
class _Queued:
    rec: dict[str, Any]
    run: Callable[["JobContext"], Iterator[Any]]
    gpu: bool = True


@dataclass
class JobContext:
    """What a running job gets: its tracker, and a hook to register the thing to cancel (a Pipeline)."""

    tracker: JobTracker
    cancel_hooks: list[Callable[[], Any]] = field(default_factory=list)

    @property
    def cancelled(self) -> bool:
        return self.tracker.cancel_requested()

    def on_cancel(self, fn: Callable[[], Any]) -> None:
        self.cancel_hooks.append(fn)


class JobQueue:
    def __init__(self, output_dir: str | Path, *, before_gpu: Callable[[JobContext], None] | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.store = JobStore(output_dir)
        self.before_gpu = before_gpu
        self._q: list[_Queued] = []
        self._cv = threading.Condition()
        self._running: dict[str, JobContext] = {}
        self._worker: threading.Thread | None = None
        self._stop = False

    # -- public -----------------------------------------------------------------
    def submit(self, kind: str, title: str, params: dict[str, Any], run: Callable[[JobContext], Iterator[Any]],
               *, source: str = "agent", gpu: bool = True) -> dict[str, Any]:
        rec = self.store.new(kind, title, source, params)
        if not gpu:  # downloads and other cpu-only work don't wait behind renders
            threading.Thread(target=self._run, args=(_Queued(rec, run, gpu),), name=f"mc-job-{rec['id']}",
                             daemon=True).start()
            return self.status(rec["id"])
        with self._cv:
            self._q.append(_Queued(rec, run, gpu))
            self._ensure_worker()
            self._cv.notify_all()
        return self.status(rec["id"])

    def status(self, job_id: str) -> dict[str, Any]:
        rec = self.store.read(job_id)
        if rec.get("status") == "queued":
            ids = [q.rec["id"] for q in self._q]
            if job_id in ids:
                rec["position"] = ids.index(job_id) + 1 + len(self._running)
        return rec

    def cancel(self, job_id: str) -> dict[str, Any]:
        rec = self.store.read(job_id)
        if rec.get("status") in FINAL or rec.get("status") == "done":
            return rec
        with self._cv:
            for q in list(self._q):
                if q.rec["id"] == job_id:
                    self._q.remove(q)
                    return self.store.update(job_id, status="cancelled", finished=time.time(),
                                             message="cancelled before it started")
        rec = self.store.update(job_id, cancel_requested=True)
        ctx = self._running.get(job_id)
        if ctx:
            self._fire_cancel(ctx)
        return rec

    def wait(self, job_id: str, timeout: float) -> dict[str, Any]:
        end = time.monotonic() + max(0.0, timeout)
        while True:
            rec = self.status(job_id)
            if rec.get("status") not in ACTIVE or time.monotonic() >= end:
                return rec
            time.sleep(min(1.0, max(0.05, end - time.monotonic())))

    def pending(self) -> list[str]:
        return [q.rec["id"] for q in self._q]

    def shutdown(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()

    # -- worker -------------------------------------------------------------------
    def _ensure_worker(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._loop, name="mc-agent-jobs", daemon=True)
            self._worker.start()

    def _loop(self) -> None:
        while True:
            with self._cv:
                while not self._q and not self._stop:
                    self._cv.wait(timeout=5)
                if self._stop:
                    return
                item = self._q.pop(0)
            self._run(item)

    @staticmethod
    def _fire_cancel(ctx: JobContext) -> None:
        for fn in ctx.cancel_hooks:
            with contextlib.suppress(Exception):
                fn()

    def _run(self, item: _Queued) -> None:
        tr = JobTracker(self.store, item.rec)
        ctx = JobContext(tr)
        jid = item.rec["id"]
        self._running[jid] = ctx
        stop_watch = threading.Event()

        def watch() -> None:  # cancel requested from another process (or a client that only writes the flag)
            while not stop_watch.wait(2.0):
                if tr.cancel_requested():
                    tr.rec["cancel_requested"] = True
                    self._fire_cancel(ctx)
                    return

        threading.Thread(target=watch, daemon=True).start()
        try:
            if tr.cancel_requested():
                raise Cancelled("cancelled before it started")
            tr.set(status="queued", message="waiting for the gpu" if item.gpu else "starting")
            lane = (gpu_lane(self.output_dir, {"job": jid, "source": item.rec.get("source", "agent"),
                                                "title": item.rec.get("title", "")},
                             on_wait=lambda who: tr.set(message=f"waiting for the gpu: {describe_owner(who)} is rendering"),
                             cancelled=lambda: ctx.cancelled)
                    if item.gpu else contextlib.nullcontext())
            with lane:
                if item.gpu and self.before_gpu:
                    self.before_gpu(ctx)
                tr.set(status="running", started=time.time(), message="running")
                final = None
                for ev in item.run(ctx):
                    if ctx.cancelled:
                        self._fire_cancel(ctx)
                        raise Cancelled("cancelled")
                    tr.event(ev)
                    if getattr(ev, "status", "") == "complete":
                        final = ev
                if final is None:
                    raise RuntimeError("the job ended without a result")
                result = {"artifact": str(final.artifact) if final.artifact else None, "message": final.message,
                          "data": _jsonable(final.data)}
                tr.set(status="done", fraction=1.0, finished=time.time(), result=result, message=final.message)
        except Cancelled as exc:
            tr.set(status="cancelled", finished=time.time(), message=str(exc) or "cancelled")
        except Exception as exc:  # noqa: BLE001 - recorded for the agent
            from .errors import friendly

            msg = friendly(exc, str(item.rec.get("params", {}).get("tier", "")))
            msg = msg.replace("<details><summary>details</summary>", " details: ").replace("</details>", "")
            tr.set(status="error", finished=time.time(), error=msg[:1500], message="failed")
        finally:
            stop_watch.set()
            self._running.pop(jid, None)


def _jsonable(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)
