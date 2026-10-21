"""A tiny fake ComfyUI server for integration tests (no GPU).

Implements just enough of ComfyUI's HTTP/WS surface for comfy-api-proxy:
/prompt, /history/{id}, /queue, /ws, /upload/image, /view, /api/jobs/{id}/cancel.
"Executing" a prompt checks that every LoadImage/LoadAudio/LoadVideo input was
uploaded first (i.e. the SDK -> proxy asset substitution worked), then writes a
placeholder output for each Save* node.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import uuid
from pathlib import Path

from aiohttp import WSMsgType, web
from PIL import Image

try:
    import imageio_ffmpeg

    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:  # pragma: no cover
    FFMPEG = "ffmpeg"


class Fake:
    def __init__(self, root: Path) -> None:
        self.input = root / "input"
        self.output = root / "output"
        self.input.mkdir(parents=True, exist_ok=True)
        self.output.mkdir(parents=True, exist_ok=True)
        self.history: dict[str, dict] = {}
        self.running: list[str] = []
        self.pending: list[str] = []
        self.sockets: dict[str, web.WebSocketResponse] = {}
        self.prompts: list[dict] = []
        self.counter = 0

    async def send(self, client_id: str, msg: dict) -> None:
        ws = self.sockets.get(client_id)
        if ws is not None and not ws.closed:
            try:
                await ws.send_str(json.dumps(msg))
            except Exception:
                pass

    def _make_output(self, ct: str, inputs: dict) -> tuple[str, dict]:
        self.counter += 1
        prefix = inputs.get("filename_prefix", "out").replace("/", "_")
        if ct == "SaveImage":
            name = f"{prefix}_{self.counter:05d}_.png"
            Image.new("RGB", (1088, 1920), (120, 80, 200)).save(self.output / name)
            return "images", {"filename": name, "subfolder": "", "type": "output"}
        if ct == "SaveAudio":
            name = f"{prefix}_{self.counter:05d}_.flac"
            subprocess.run([FFMPEG, "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                            "sine=frequency=200:sample_rate=24000:duration=2.5", str(self.output / name)], check=True)
            return "audio", {"filename": name, "subfolder": "", "type": "output"}
        if ct == "SaveVideo":
            name = f"{prefix}_{self.counter:05d}_.mp4"
            subprocess.run([FFMPEG, "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=480x832:rate=25:duration=4",
                            "-pix_fmt", "yuv420p", str(self.output / name)], check=True)
            return "images", {"filename": name, "subfolder": "", "type": "output"}
        raise ValueError(ct)

    async def execute(self, pid: str, prompt: dict, client_id: str) -> None:
        await asyncio.sleep(0.6)  # give the proxy time to attach its websocket
        self.pending.remove(pid)
        self.running.append(pid)
        await self.send(client_id, {"type": "execution_start", "data": {"prompt_id": pid}})
        messages = [["execution_start", {"prompt_id": pid}]]
        outputs: dict[str, dict] = {}
        error = None
        for nid, node in prompt.items():
            ct, inputs = node["class_type"], node["inputs"]
            field = {"LoadImage": "image", "LoadAudio": "audio", "LoadVideo": "file"}.get(ct)
            if field:
                val = inputs.get(field)
                if not isinstance(val, str) or not (self.input / val).exists():
                    error = {"prompt_id": pid, "node_id": nid, "node_type": ct,
                             "exception_message": f"input file not uploaded: {val!r}"}
                    break
        if error is None:
            for step in range(1, 5):
                await self.send(client_id, {"type": "progress", "data": {"prompt_id": pid, "value": step, "max": 4, "node": "x"}})
                await asyncio.sleep(0.05)
            for nid, node in prompt.items():
                if node["class_type"] in ("SaveImage", "SaveAudio", "SaveVideo"):
                    key, item = self._make_output(node["class_type"], node["inputs"])
                    outputs[nid] = {key: [item]}
                    await self.send(client_id, {"type": "executed", "data": {"prompt_id": pid, "node": nid, "output": outputs[nid]}})
        self.running.remove(pid)
        if error:
            messages.append(["execution_error", error])
            self.history[pid] = {"prompt": [0, pid, prompt, {}, []], "outputs": {},
                                 "status": {"status_str": "error", "completed": False, "messages": messages}}
            await self.send(client_id, {"type": "execution_error", "data": error})
        else:
            messages.append(["execution_success", {"prompt_id": pid}])
            self.history[pid] = {"prompt": [0, pid, prompt, {}, list(outputs)], "outputs": outputs,
                                 "status": {"status_str": "success", "completed": True, "messages": messages}}
            await self.send(client_id, {"type": "execution_success", "data": {"prompt_id": pid}})

    # -- routes -----------------------------------------------------------
    async def prompt(self, request: web.Request) -> web.Response:
        body = await request.json()
        pid = body.get("prompt_id") or str(uuid.uuid4())
        self.prompts.append(body["prompt"])
        self.pending.append(pid)
        asyncio.get_running_loop().create_task(self.execute(pid, body["prompt"], body.get("client_id", "")))
        return web.json_response({"prompt_id": pid, "number": len(self.prompts), "node_errors": {}})

    async def get_history(self, request: web.Request) -> web.Response:
        pid = request.match_info["id"]
        return web.json_response({pid: self.history[pid]} if pid in self.history else {})

    async def queue(self, request: web.Request) -> web.Response:
        return web.json_response({
            "queue_running": [[0, p, {}, {}, []] for p in self.running],
            "queue_pending": [[1, p, {}, {}, []] for p in self.pending],
        })

    async def ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        cid = request.query.get("clientId", str(uuid.uuid4()))
        self.sockets[cid] = ws
        await ws.send_str(json.dumps({"type": "status", "data": {"status": {"exec_info": {"queue_remaining": 0}}, "sid": cid}}))
        async for msg in ws:
            if msg.type == WSMsgType.ERROR:
                break
        self.sockets.pop(cid, None)
        return ws

    async def upload(self, request: web.Request) -> web.Response:
        data = await request.post()
        f = data["image"]
        sub = data.get("subfolder", "") or ""
        dest_dir = self.input / sub
        dest_dir.mkdir(parents=True, exist_ok=True)
        (dest_dir / f.filename).write_bytes(f.file.read())
        return web.json_response({"name": f.filename, "subfolder": sub, "type": "input"})

    async def view(self, request: web.Request) -> web.StreamResponse:
        q = request.query
        base = self.output if q.get("type", "output") == "output" else self.input
        path = base / q.get("subfolder", "") / q["filename"]
        if not path.exists():
            raise web.HTTPNotFound()
        return web.FileResponse(path)

    async def cancel(self, request: web.Request) -> web.Response:
        return web.json_response({})

    async def prompts_dump(self, request: web.Request) -> web.Response:
        return web.json_response(self.prompts)


def make_app(root: Path) -> web.Application:
    fake = Fake(root)
    app = web.Application(client_max_size=200 * 1024 * 1024)
    app.add_routes([
        web.post("/prompt", fake.prompt),
        web.get("/history/{id}", fake.get_history),
        web.get("/queue", fake.queue),
        web.get("/ws", fake.ws),
        web.post("/upload/image", fake.upload),
        web.get("/view", fake.view),
        web.post("/api/jobs/{id}/cancel", fake.cancel),
        web.get("/_test/prompts", fake.prompts_dump),
    ])
    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18188)
    ap.add_argument("--root", type=Path, required=True)
    a = ap.parse_args()
    web.run_app(make_app(a.root), host="127.0.0.1", port=a.port, print=None)
