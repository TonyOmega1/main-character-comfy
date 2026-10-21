"""End-to-end through the *real* Comfy SDK and the *real* comfy-api-proxy.

    MC --comfy-sdk--> comfy-api-proxy (:port2) --> fake ComfyUI (:port1)

Only the GPU work is faked. This proves asset upload + core/ASSET substitution,
submit, SSE progress, output download and multi-job chaining all go through
the Comfy Developer Platform surface (no raw ComfyUI calls from the app).
"""

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

HERE = Path(__file__).parent


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait(url, timeout=20):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if httpx.get(url, timeout=1).status_code < 500:
                return
        except httpx.HTTPError:
            time.sleep(0.2)
    raise RuntimeError(f"{url} did not come up")


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    proxy_bin = shutil.which("comfy-api-proxy") or str(Path(sys.executable).parent / "comfy-api-proxy")
    if not Path(proxy_bin).exists():
        pytest.skip("comfy-api-proxy not installed")
    root = tmp_path_factory.mktemp("fakecomfy")
    p1, p2 = _free_port(), _free_port()
    fake = subprocess.Popen([sys.executable, str(HERE / "fake_comfyui.py"), "--port", str(p1), "--root", str(root)])
    proxy = subprocess.Popen([proxy_bin, "run", "--comfyui", f"http://127.0.0.1:{p1}", "--port", str(p2)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        _wait(f"http://127.0.0.1:{p1}/queue")
        _wait(f"http://127.0.0.1:{p2}/api/v2/health")
        yield {"fake": f"http://127.0.0.1:{p1}", "proxy": f"http://127.0.0.1:{p2}", "root": root}
    finally:
        proxy.terminate()
        fake.terminate()


def test_talk_pipeline_through_comfy_sdk(stack, tmp_path, sample_photo, sample_voice):
    from main_character_comfy.backends import ComfyBackend
    from main_character_comfy.config import Settings
    from main_character_comfy.pipeline import Pipeline, Request

    backend = ComfyBackend(base_url=stack["proxy"])
    assert backend.health()["status"] == "healthy"
    pipe = Pipeline(backend, Settings(output_dir=tmp_path / "out"))
    req = Request(script="Hello from the integration test, running through the Comfy SDK.", photo=sample_photo,
                  voice_sample=sample_voice, style="comic", consent=True, smooth="rife")
    events = list(pipe.run(req))
    assert events[-1].status == "complete"
    progress = [e for e in events if e.status == "progress" and e.stage == "avatar"]
    assert progress, "expected SSE progress events relayed by the proxy"
    proj = json.loads((events[-1].artifact.parent / "_work" / "project.json").read_text())
    assert set(proj["jobs"]) == {"avatar", "voice", "lipsync", "smooth"}
    assert all(len(j) == 36 for j in proj["jobs"].values())  # proxy job ids are UUIDs

    prompts = httpx.get(stack["fake"] + "/_test/prompts").json()
    assert len(prompts) == 4
    blob = json.dumps(prompts)
    assert "core/ASSET" not in blob  # every asset handle was resolved to an uploaded filename
    uploaded = {p.name for p in (stack["root"] / "input").rglob("*") if p.is_file()}
    assert len(uploaded) >= 4  # photo, voice ref, avatar, speech, talking clip


def test_dialogue_segments_through_comfy_sdk(stack, tmp_path):
    from main_character_comfy.backends import ComfyBackend
    from main_character_comfy.config import Settings
    from main_character_comfy.pipeline import Pipeline, Request

    pipe = Pipeline(ComfyBackend(base_url=stack["proxy"]), Settings(output_dir=tmp_path / "out"))
    req = Request(mode="dialogue", character="two cheerful robots", segment_seconds=4.0,
                  script="Ava: Hello there!\nBo: Hi Ava, ready to film?\nAva: Always.")
    events = list(pipe.run(req))
    assert events[-1].status == "complete"
    proj = json.loads((events[-1].artifact.parent / "_work" / "project.json").read_text())
    assert len(proj["segments"]) >= 2
    assert sum(k.startswith("lipsync_") for k in proj["jobs"]) == len(proj["segments"])


def test_failed_job_surfaces_node_error(stack, tmp_path):
    from main_character_comfy import graphs
    from main_character_comfy.backends import ComfyBackend
    from main_character_comfy.config import Models

    sg = graphs.avatar_graph("x", models=Models(), reference=True)  # photo never uploaded
    node, field = sg.inputs.pop("photo")
    sg.graph[node]["inputs"][field] = "never_uploaded.png"
    with pytest.raises(RuntimeError, match="not uploaded"):
        ComfyBackend(base_url=stack["proxy"]).run_stage("avatar", sg, {}, tmp_path, lambda f, m: None)
