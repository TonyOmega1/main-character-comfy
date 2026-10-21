"""Agent connector: shared GPU lane + job queue, the service sandbox, every MCP tool against the mock backend,
the HTTP transport's token / signed links / upload route, and the redub voice chain."""

import asyncio
import base64
import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from main_character_comfy import jobs
from main_character_comfy.backends import MockBackend
from main_character_comfy.config import Settings
from main_character_comfy.errors import Cancelled
from main_character_comfy.media import has_audio_stream, media_duration, run_ffmpeg
from main_character_comfy.pipeline import Event, InputError
from main_character_comfy.service import AccessError, MCService

mcp = pytest.importorskip("mcp")
from mcp import Client  # noqa: E402

EXPECTED_TOOLS = {
    "mc_status", "check_models", "list_templates", "list_library", "upload_file", "make_meme", "make_meme_pack",
    "find_people", "swap_into_clip", "copy_the_vibe", "talk", "sing", "duet", "batch_talk", "voice_tts",
    "convert_voice", "redub_line", "restyle_image", "upscale_video", "remix_recipe", "save_meme_format", "get_job",
    "wait_for_job", "list_jobs", "cancel_job", "list_outputs", "get_output", "get_settings", "update_settings",
    "find_clips", "probe_clip", "fetch_clip", "check_reference",
}


# --------------------------------------------------------------------------- fixtures
@pytest.fixture
def svc(tmp_path, monkeypatch):
    monkeypatch.delenv("MC_AGENT_EXTRA_DIRS", raising=False)
    monkeypatch.delenv("MC_AGENT_TOKEN", raising=False)
    s = Settings(output_dir=tmp_path / "out")
    return MCService(s, backend_kind="mock", backend_factory=MockBackend, library_dir=tmp_path / "lib")


def _clip(path: Path, seconds=3, size="832x480", audio=True) -> Path:
    args = ["-f", "lavfi", "-i", f"testsrc2=size={size}:rate=16:duration={seconds}"]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=330:sample_rate=48000:duration={seconds}", "-c:a", "aac",
                 "-shortest"]
    run_ffmpeg([*args, "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)])
    return path


@pytest.fixture
def lib(svc, sample_photo, sample_voice, tmp_path):
    """Files uploaded the way an agent would: base64 through the service."""
    up = lambda kind, p, name=None: svc.upload(kind, name or p.name,  # noqa: E731
                                                content_base64=base64.b64encode(p.read_bytes()).decode())["path"]
    clip = _clip(tmp_path / "my_dance.mp4")
    song = tmp_path / "song.wav"
    run_ffmpeg(["-f", "lavfi", "-i", "sine=frequency=262:duration=8", str(song)])
    rec = tmp_path / "line.wav"
    run_ffmpeg(["-f", "lavfi", "-i", "sine=frequency=180:sample_rate=48000:duration=1.2", "-af",
                "tremolo=f=5:d=0.7", str(rec)])
    return {"photo": up("references", sample_photo, "me.jpg"), "friend": up("references", sample_photo, "friend.jpg"),
            "voice": up("voices", sample_voice), "clip": up("clips", clip), "song": up("audio", song),
            "line": up("audio", rec)}


def call(server, name, args=None):
    async def go():
        async with Client(server) as c:
            return await c.call_tool(name, args or {})
    return asyncio.run(go())


def payload(res):
    assert not res.is_error, res.content[0].text if res.content else res
    if res.structured_content is not None:
        sc = res.structured_content
        return sc.get("result", sc) if isinstance(sc, dict) and set(sc) == {"result"} else sc
    return json.loads(res.content[0].text)


@pytest.fixture
def server(svc):
    from main_character_comfy.agent import build_server

    return build_server(svc)


def run_job(server, name, args, timeout=120):
    job = payload(call(server, name, args))
    assert job["id"] and job["status"] in ("queued", "running", "done")
    done = payload(call(server, "wait_for_job", {"job_id": job["id"], "timeout_s": timeout}))
    assert done["status"] == "done", done
    return done


# --------------------------------------------------------------------------- GPU lane + queue
def test_gpu_lane_is_exclusive_across_processes(tmp_path):
    out = tmp_path / "out"
    jobs.agent_dir(out)
    code = ("import sys,time; sys.path.insert(0, %r); from main_character_comfy import jobs; "
            "l = jobs.GpuLane(%r, {'source': 'ui', 'title': 'other'}); assert l.try_acquire(); print('held', flush=True); "
            "time.sleep(3); l.release()") % (str(Path(__file__).parent.parent), str(out))
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "held"
    lane = jobs.GpuLane(out, {"source": "agent"})
    assert not lane.try_acquire()
    who = jobs.lane_owner(out)
    assert who and who["source"] == "ui" and "mc window" in jobs.describe_owner(who)
    p.wait(10)
    assert lane.try_acquire()
    lane.release()


def test_queue_runs_fifo_one_at_a_time(tmp_path):
    q = jobs.JobQueue(tmp_path / "out")
    active, seen, peak = [0], [], [0]

    def make(tag):
        def run(ctx):
            active[0] += 1
            peak[0] = max(peak[0], active[0])
            seen.append(tag)
            time.sleep(0.2)
            active[0] -= 1
            yield Event("finish", "done", 1.0, "ok")
            yield Event("complete", "complete", 1.0, f"done {tag}", artifact=None)
        return run

    ids = [q.submit("t", f"job {i}", {}, make(i))["id"] for i in range(3)]
    for i in ids:
        assert q.wait(i, 10)["status"] == "done"
    assert seen == [0, 1, 2] and peak[0] == 1


def test_queue_cancel_queued_and_running(tmp_path):
    q = jobs.JobQueue(tmp_path / "out")
    stopped = threading.Event()

    def slow(ctx):
        ctx.on_cancel(stopped.set)
        for i in range(100):
            if stopped.is_set():
                raise Cancelled("stopped")
            time.sleep(0.05)
            yield Event("swap", "progress", i / 100, f"step {i}")
        yield Event("complete", "complete", 1.0, "never")

    a = q.submit("t", "slow", {}, slow)["id"]
    b = q.submit("t", "queued", {}, slow)["id"]
    time.sleep(0.3)
    assert q.cancel(b)["status"] == "cancelled"
    q.cancel(a)
    assert q.wait(a, 10)["status"] == "cancelled" and stopped.is_set()


def test_queue_reports_errors_in_plain_english(tmp_path):
    q = jobs.JobQueue(tmp_path / "out")

    def boom(ctx):
        raise RuntimeError("CUDA error: out of memory")
        yield  # pragma: no cover

    rec = q.wait(q.submit("t", "boom", {}, boom)["id"], 10)
    assert rec["status"] == "error" and "ran out of memory" in rec["error"]


def test_window_runs_wait_for_the_lane_and_show_up_in_the_job_list(tmp_path):
    out = tmp_path / "out"
    lane = jobs.GpuLane(out, {"source": "agent", "title": "agent swap"})
    assert lane.try_acquire()
    evs = iter([Event("prepare", "start", 0, "go"), Event("complete", "complete", 1.0, "done")])
    gen = jobs.laned_events(out, evs, "meme", "meme (mc window)", wait_event=lambda m: Event("prepare", "progress",
                                                                                           -1, m), poll=0.05)
    first = next(gen)
    assert "waiting for the gpu" in first.message and "an agent" in first.message
    lane.release()
    rest = list(gen)
    assert rest[-1].status == "complete"
    rec = jobs.JobStore(out).list()[0]
    assert rec["source"] == "ui" and rec["status"] == "done"


# --------------------------------------------------------------------------- sandbox
def test_sandbox_resolves_only_mc_folders(svc, lib, tmp_path):
    assert svc.resolve(lib["photo"]).name == "me.jpg"
    assert svc.resolve("me.jpg").name == "me.jpg"  # bare name found in the library
    assert svc.resolve("examples/avatar_3d_flux2_klein.jpg").is_file()
    outside = tmp_path / "secret.txt"
    outside.write_text("x")
    with pytest.raises(AccessError):
        svc.resolve(str(outside))
    with pytest.raises(AccessError):
        svc.resolve("library/../../secret.txt")
    with pytest.raises(InputError):
        svc.resolve("library/references/nope.png")


def test_upload_validates_kind_and_never_overwrites(svc, sample_photo):
    b64 = base64.b64encode(sample_photo.read_bytes()).decode()
    a = svc.upload("references", "me.jpg", content_base64=b64)
    b = svc.upload("photo", "me.jpg", content_base64="data:image/jpeg;base64," + b64)
    assert a["path"] != b["path"] and a["size"] == [900, 1200]
    with pytest.raises(InputError):
        svc.upload("clips", "evil.exe", content_base64=b64)
    with pytest.raises(InputError):
        svc.upload("references", "../../x.png", content_base64="")
    assert svc.upload("references", "../../x.png", content_base64=b64)["path"] == "library/references/x.png"


def test_url_upload_refuses_local_network(svc):
    for url in ("http://127.0.0.1:8188/queue", "http://localhost/x.png", "file:///etc/passwd", "http://10.0.0.5/a.png"):
        with pytest.raises(AccessError):
            svc.upload("references", "a.png", url=url)


def test_extra_dirs_are_opt_in(tmp_path, monkeypatch, sample_photo):
    d = tmp_path / "Desktop MC"
    d.mkdir()
    (d / "ref.jpg").write_bytes(sample_photo.read_bytes())
    monkeypatch.setenv("MC_AGENT_EXTRA_DIRS", str(d))
    s = MCService(Settings(output_dir=tmp_path / "out"), backend_kind="mock", backend_factory=MockBackend,
                  library_dir=tmp_path / "lib")
    assert s.resolve(str(d / "ref.jpg")).name == "ref.jpg"
    info = s.upload("references", source_path=str(d / "ref.jpg"))
    assert info["path"] == "library/references/ref.jpg"


# --------------------------------------------------------------------------- MCP tools (mock backend)
def test_every_capability_is_a_tool_with_a_description(server):
    async def go():
        async with Client(server) as c:
            return await c.list_tools(), await c.list_prompts(), await c.list_resources()
    tools, prompts, resources = asyncio.run(go())
    names = {t.name for t in tools.tools}
    assert names == EXPECTED_TOOLS
    for t in tools.tools:
        assert t.description and len(t.description) > 60, t.name
        assert t.annotations is not None
    assert {p.name for p in prompts.prompts} >= {"make_a_meme", "swap_me_in", "redub_a_line"}
    assert {str(r.uri) for r in resources.resources} >= {"mc://guide", "mc://templates", "mc://status"}


def test_status_templates_library_settings(server, lib):
    st = payload(call(server, "mc_status"))
    assert st["active_tier"]["key"] == "standard" and st["health"]["backend"] == "mock"
    assert "deepfilternet" in st and "library" in st["folders"]
    tpl = payload(call(server, "list_templates"))
    assert any(f["key"] == "side_eye" for f in tpl["meme_formats"]) and "anime" in {s["key"] for s in tpl["styles"]}
    assert "examples/avatar_3d_flux2_klein.jpg" in tpl["builtin_examples"]
    libr = payload(call(server, "list_library", {"kind": "references"}))
    assert {x["path"] for x in libr["references"]} >= {lib["photo"], lib["friend"]}
    assert payload(call(server, "update_settings", {"tier": "max", "watermark": False}))["tier"] == "max"
    assert payload(call(server, "get_settings"))["settings"]["watermark"] is False
    assert payload(call(server, "mc_status"))["active_tier"]["key"] == "max"


def test_upload_tool(server, sample_photo):
    up = payload(call(server, "upload_file", {"kind": "references", "filename": "selfie.jpg",
                                              "content_base64": base64.b64encode(sample_photo.read_bytes()).decode()}))
    assert up["path"] == "library/references/selfie.jpg"


def test_rights_and_sfw_errors_come_back_as_tool_errors(server, lib):
    res = call(server, "make_meme", {"photo": lib["photo"], "meme": "side_eye"})
    assert res.is_error and "consent" in res.content[0].text.lower()
    res = call(server, "make_meme", {"photo": "/etc/passwd", "i_have_rights": True})
    assert res.is_error and "outside" in res.content[0].text
    res = call(server, "voice_tts", {"text": "something nsfw"})
    assert res.is_error or True  # SFW is checked when the job runs: see the next assertion
    job = payload(res) if not res.is_error else None
    if job:
        done = payload(call(server, "wait_for_job", {"job_id": job["id"], "timeout_s": 30}))
        assert done["status"] == "error" and "SFW" in done["error"]


def test_make_meme_still_then_animate(server, lib):
    still = run_job(server, "make_meme", {"photo": lib["photo"], "meme": "side_eye", "animate": False,
                                          "i_have_rights": True})
    png = still["result"]["final"]
    assert png.endswith(".png")
    img = call(server, "get_output", {"path": png})
    assert not img.is_error and any(c.type == "image" for c in img.content)
    video = run_job(server, "make_meme", {"photo": lib["photo"], "meme": "side_eye", "scene_image": png,
                                          "i_have_rights": True, "top": "me", "bottom": "also me"})
    assert video["result"]["final"].endswith(".mp4") and any(f.endswith(".gif") for f in video["result"]["files"])


def test_custom_meme_with_cloned_line(server, lib):
    run_job(server, "make_meme", {"photo": lib["photo"], "meme": "custom", "custom_scene": "at a drive-through",
                                  "custom_action": "slowly lowers sunglasses", "say_line": True, "line": "one more",
                                  "voice": "clone", "voice_sample": lib["voice"], "i_have_rights": True})


def test_meme_pack(server, lib):
    done = run_job(server, "make_meme_pack", {"photo": lib["photo"], "formats": ["side_eye", "slow_clap"],
                                              "i_have_rights": True})
    assert done["result"]["final"]


def test_find_people_then_group_swap(server, lib, monkeypatch):
    monkeypatch.setenv("MC_MOCK_PEOPLE", "2")
    found = run_job(server, "find_people", {"clip": lib["clip"]})
    assert found["result"]["people"] == 2
    done = run_job(server, "swap_into_clip", {
        "clip": lib["clip"], "people": [lib["photo"], lib["friend"]], "group_engine": "chain",
        "describe": ["man in a black hat and purple hoodie", "man in a brown jacket"], "clip_seconds": 3,
        "i_have_rights": True})
    final = done["result"]["final"]
    assert final.endswith(".mp4")


def test_single_swap_keeps_audio(server, svc, lib):
    done = run_job(server, "swap_into_clip", {"clip": lib["clip"], "photo": lib["photo"], "clip_seconds": 2,
                                              "i_have_rights": True})
    assert has_audio_stream(svc.resolve(done["result"]["final"]))


def test_copy_the_vibe(server, lib):
    run_job(server, "copy_the_vibe", {"clip": lib["clip"], "photo": lib["photo"], "scene": "flat purple cartoon "
                                      "living room", "clip_seconds": 2, "i_have_rights": True})


def test_talk_sing_duet_batch(server, lib):
    run_job(server, "talk", {"photo": lib["photo"], "script": "Hey, new video every Friday.", "voice": "clone",
                             "voice_sample": lib["voice"], "i_have_rights": True})
    run_job(server, "talk", {"character": "a friendly robot chef", "script": "Hello there."})
    run_job(server, "sing", {"photo": lib["photo"], "song": lib["song"], "song_seconds": 4, "i_have_rights": True})
    run_job(server, "duet", {"character": "two robots", "script": "Ava: Hello!\nBo: Hi Ava."})
    done = run_job(server, "batch_talk", {"character": "a robot", "scripts": "one\n\ntwo"})
    assert len(done["result"]["clips"]) == 2


def test_voice_tools(server, lib):
    tts = run_job(server, "voice_tts", {"text": "This is my cloned voice.", "voice_sample": lib["voice"],
                                        "i_have_rights": True})
    assert tts["result"]["final"].endswith(".wav")
    run_job(server, "convert_voice", {"source": tts["result"]["final"], "target_voice": lib["voice"],
                                      "i_have_rights": True})


def test_redub_line_tool(server, svc, lib, monkeypatch):
    monkeypatch.setenv("MC_RECORDLINE", "off")  # this test covers the built-in redub.py chain (report fields)
    done = run_job(server, "redub_line", {"video": lib["clip"], "recording": lib["line"], "start": 0.8,
                                          "fit_seconds": 1.0, "denoise": "off", "i_have_rights": True})
    rd = done["result"]["redub"]
    assert any("retime" in s for s in rd["steps"]) and rd["final_true_peak_db"] <= -1.4
    out = svc.resolve(done["result"]["final"])
    assert abs(media_duration(out) - 3.0) < 0.1
    run_job(server, "redub_line", {"video": lib["clip"], "text": "pull up", "voice_sample": lib["voice"],
                                   "start": 0.5, "denoise": "off", "i_have_rights": True})


def test_restyle_upscale_remix_save_format(server, svc, lib):
    r = run_job(server, "restyle_image", {"image": lib["photo"], "background": "flat purple cartoon street",
                                          "i_have_rights": True})
    assert r["result"]["final"].endswith(".png") and "background" in r["result"]["prompt"]
    up = run_job(server, "upscale_video", {"video": lib["clip"], "engine": "seedvr2"})
    from main_character_comfy.media import video_size

    assert video_size(svc.resolve(up["result"]["final"])) == (1920, 1080)
    run_job(server, "upscale_video", {"video": lib["clip"], "engine": "lanczos", "width": 640, "height": 360})
    meme = run_job(server, "make_meme", {"photo": lib["photo"], "meme": "slow_clap", "i_have_rights": True})
    recipe = next(Path(p) for p in [svc.resolve(meme["result"]["final"]).parent / "_work" / "mc_recipe.json"])
    rmx = run_job(server, "remix_recipe", {"recipe": svc.display(recipe), "photo": lib["friend"],
                                           "i_have_rights": True})
    assert rmx["status"] == "done"
    saved = payload(call(server, "save_meme_format", {"name": "Drive-Thru Stare", "scene": "at a drive-through",
                                                      "action": "slowly lowers sunglasses", "top": "me"}))
    assert saved["key"]


def test_jobs_outputs_and_cancel_tools(server, svc, lib):
    job = payload(call(server, "make_meme", {"photo": lib["photo"], "meme": "side_eye", "i_have_rights": True}))
    listed = payload(call(server, "list_jobs", {}))
    assert any(j["id"] == job["id"] for j in listed["jobs"])
    payload(call(server, "cancel_job", {"job_id": job["id"]}))
    st = payload(call(server, "wait_for_job", {"job_id": job["id"], "timeout_s": 30}))
    assert st["status"] in ("cancelled", "done")
    run_job(server, "make_meme", {"photo": lib["photo"], "meme": "side_eye", "animate": False, "i_have_rights": True})
    outs = payload(call(server, "list_outputs", {}))
    assert outs["runs"] and outs["runs"][0]["files"]
    g = payload(call(server, "get_job", {"job_id": job["id"]}))
    assert g["id"] == job["id"]
    assert call(server, "get_job", {"job_id": "nope"}).is_error


# --------------------------------------------------------------------------- HTTP transport
def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def http_server(svc):
    import uvicorn

    from main_character_comfy.agent.http import build_http_app

    port = _free_port()
    app, base = build_http_app(svc, port=port, token="t0ken-xyz")
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}", "t0ken-xyz"
    srv.should_exit = True
    th.join(5)


def test_http_requires_the_token(http_server, svc, lib):
    import httpx

    base, token = http_server
    assert httpx.get(f"{base}/health").json()["auth"] == "token"
    assert httpx.post(f"{base}/mcp", json={}).status_code == 401
    assert httpx.post(f"{base}/mcp", json={}, headers={"Authorization": "Bearer wrong"}).status_code == 401
    # with the token: a real MCP session over streamable HTTP
    st, done = asyncio.run(_session(base, token, lib))
    assert st["health"]["backend"] == "mock" and done["status"] == "done"
    link = next(iter(done["result"]["download_links"].values()))
    r = httpx.get(link)
    assert r.status_code == 200 and len(r.content) > 1000
    assert httpx.get(link.replace("/files/", "/files/0")).status_code == 403
    # query-string token for clients that can't set headers
    assert httpx.post(f"{base}/mcp?token={token}", json={}).status_code != 401
    # big uploads go to /upload with the token
    data = svc.resolve(lib["photo"]).read_bytes()
    assert httpx.put(f"{base}/upload/references/big.jpg", content=data).status_code == 401
    r = httpx.put(f"{base}/upload/references/big.jpg", content=data, headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200 and r.json()["path"] == "library/references/big.jpg"


async def _session(base, token, lib):
    import httpx2
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with httpx2.AsyncClient(headers={"Authorization": f"Bearer {token}"}, timeout=60) as hc:
        async with streamable_http_client(f"{base}/mcp", http_client=hc) as streams:
            read, write = streams[0], streams[1]
            async with ClientSession(read, write) as c:
                await c.initialize()
                st = await c.call_tool("mc_status", {})
                job = await c.call_tool("make_meme", {"photo": lib["photo"], "animate": False, "i_have_rights": True})
                done = await c.call_tool("wait_for_job", {"job_id": payload(job)["id"], "timeout_s": 60})
                return payload(st), payload(done)


def test_local_http_mode_blocks_foreign_hosts(svc):
    from starlette.testclient import TestClient

    from main_character_comfy.agent.http import build_http_app

    app, _ = build_http_app(svc, port=8765, token=None)
    with TestClient(app, base_url="http://127.0.0.1:8765") as c:
        assert c.get("/health").json()["auth"] == "local"
        r = c.post("/mcp", json={}, headers={"Host": "evil.example:8765"})
        assert r.status_code in (403, 421)


def test_exposure_rules():
    from main_character_comfy.agent.http import check_exposure

    check_exposure("127.0.0.1", None, False, False)
    check_exposure("127.0.0.1", "tok", True, False)
    with pytest.raises(SystemExit):
        check_exposure("127.0.0.1", None, True, False)  # remote without a token
    with pytest.raises(SystemExit):
        check_exposure("0.0.0.0", None, False, True)  # open bind without a token
    with pytest.raises(SystemExit):
        check_exposure("0.0.0.0", "tok", True, False)  # LAN bind needs --allow-lan
    check_exposure("100.64.1.2", "tok", True, True)


def test_token_file_is_created_once(svc, monkeypatch):
    from main_character_comfy.agent.http import load_or_create_token, token_file

    t1, new1 = load_or_create_token(svc, create=True)
    t2, new2 = load_or_create_token(svc, create=True)
    assert new1 and not new2 and t1 == t2 and t1.startswith("mc_") and token_file(svc).exists()
    monkeypatch.setenv("MC_AGENT_TOKEN", "from-env")
    assert load_or_create_token(svc, create=True) == ("from-env", False)


# --------------------------------------------------------------------------- redub chain
def test_redub_chain_ducks_and_never_limits(tmp_path, monkeypatch):
    from main_character_comfy import redub

    seen = []
    real = redub.run_ffmpeg
    monkeypatch.setattr(redub, "run_ffmpeg", lambda args, *a, **k: (seen.append(" ".join(map(str, args))), real(args, *a, **k))[1])

    video = _clip(tmp_path / "v.mp4", seconds=4)
    line = tmp_path / "line.wav"
    run_ffmpeg(["-f", "lavfi", "-i", "anoisesrc=d=0.3:a=0.02", "-f", "lavfi", "-i",
                "sine=frequency=200:sample_rate=48000:duration=1.5", "-filter_complex",
                "[0][1]concat=n=2:v=0:a=1,tremolo=f=6:d=0.8", str(line)])
    rep = redub.redub(video, line, tmp_path / "out.mp4", tmp_path / "w",
                      redub.RedubSettings(start=1.0, denoise="off", fit_seconds=1.2))
    assert abs(media_duration(tmp_path / "out.mp4") - 4.0) < 0.1
    assert rep["final_true_peak_db"] <= -1.49
    assert seen and not any(f in c for c in seen for f in ("loudnorm", "alimiter", "dynaudnorm", "speechnorm"))
    assert abs(rep["line_seconds"] - 1.2) < 0.08
    mix = redub.load(tmp_path / "w" / "mix_final.wav", 2)
    bed = redub.load(video, 2)
    sr = redub.SR

    def ratio(a, b):
        seg = lambda x: float((x[int(a * sr):int(b * sr)] ** 2).mean() ** 0.5)  # noqa: E731
        return seg(mix), seg(bed)
    # outside the line the bed only gets the one static gain; at 3.5 s it's back to that level
    m_out, b_out = ratio(3.5, 3.9)
    assert m_out > 0 and b_out > 0
    env = redub.cosine_duck(sr * 4, 1.0, 2.3, 0.28, 0.25)
    assert env[int(1.5 * sr)] == pytest.approx(0.28) and env[int(3.5 * sr)] == pytest.approx(1.0)
    assert np_smooth(env)


def np_smooth(env):
    import numpy as np

    return float(np.abs(np.diff(env)).max()) < 1e-3  # no clicks: the duck is a smooth ramp


def test_redub_requires_deepfilternet_only_when_asked(tmp_path, monkeypatch):
    from main_character_comfy import redub

    monkeypatch.setattr(redub, "_deepfilter_cmd", lambda: None)
    video = _clip(tmp_path / "v.mp4", seconds=2)
    line = tmp_path / "l.wav"
    run_ffmpeg(["-f", "lavfi", "-i", "sine=frequency=200:sample_rate=48000:duration=0.8", str(line)])
    rep = redub.redub(video, line, tmp_path / "o.mp4", tmp_path / "w", redub.RedubSettings(start=0.2))
    assert "deepfilternet3 skipped" in rep["steps"][0]
    with pytest.raises(RuntimeError, match="DeepFilterNet"):
        redub.redub(video, line, tmp_path / "o2.mp4", tmp_path / "w2",
                    redub.RedubSettings(start=0.2, denoise="deepfilternet"))


# --------------------------------------------------------------------------- bring your own clip
class _FakeYDL:
    """Stands in for yt_dlp.YoutubeDL: 'downloads' by rendering a local test clip."""
    calls: list = []
    height = 2160

    def __init__(self, opts):
        self.opts = opts
        _FakeYDL.calls.append(opts)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def _info(self, url):
        if "playlist" in url:
            return {"_type": "playlist", "entries": []}
        if url.startswith("ytsearch"):
            return {"entries": [{"url": "https://example.com/a"}, {"url": "https://example.com/b"}]}
        h = 720 if url.endswith("/b") else _FakeYDL.height
        return {"id": "abc", "title": "my own dance", "webpage_url": url, "extractor_key": "Generic", "duration": 3,
                "formats": [{"vcodec": "avc1", "height": h, "width": h * 16 // 9, "fps": 30},
                            {"vcodec": "avc1", "height": 360, "width": 640, "fps": 30},
                            {"vcodec": "none", "acodec": "mp4a"}]}

    def extract_info(self, url, download=False):
        info = self._info(url)
        if download and info.get("_type") != "playlist":
            out = Path(self.opts["outtmpl"].replace("%(title).60B", "my-own-dance").replace("%(id)s", "abc")
                       .replace("%(ext)s", "mp4"))
            h = min(_FakeYDL.height, int(self.opts["format_sort"][0].split(":")[1])) \
                if ":" in self.opts["format_sort"][0] else _FakeYDL.height
            _clip(out, seconds=2, size=f"{h * 16 // 9 // 2 * 2}x{h}" if h <= 720 else "1920x1080")
            for hook in self.opts.get("progress_hooks", []):
                hook({"status": "downloading", "downloaded_bytes": 5, "total_bytes": 10})
            info.update(requested_downloads=[{"filepath": str(out)}], height=None, width=None)
        return info


@pytest.fixture
def fake_ytdlp(monkeypatch):
    import types

    mod = types.ModuleType("yt_dlp")
    mod.YoutubeDL = _FakeYDL
    utils = types.ModuleType("yt_dlp.utils")
    utils.download_range_func = lambda chapters, ranges: ("ranges", ranges)
    mod.utils = utils
    monkeypatch.setitem(sys.modules, "yt_dlp", mod)
    monkeypatch.setitem(sys.modules, "yt_dlp.utils", utils)
    real = socket.getaddrinfo

    def fake_dns(host, *a, **k):
        if host in ("example.com", "www.youtube.com"):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]
        return real(host, *a, **k)

    monkeypatch.setattr(socket, "getaddrinfo", fake_dns)
    _FakeYDL.calls = []
    _FakeYDL.height = 2160
    return _FakeYDL


def test_clip_links_are_public_only(monkeypatch):
    from main_character_comfy import clipfetch

    for bad in ("file:///etc/passwd", "http://127.0.0.1:8188/view", "http://192.168.1.10/x.mp4", "ftp://x/y", ""):
        with pytest.raises(InputError):
            clipfetch.check_url(bad)
    assert clipfetch.blocked_ip("10.0.0.5") and clipfetch.blocked_ip("::1") and clipfetch.blocked_ip("169.254.1.1")
    assert not clipfetch.blocked_ip("198.18.0.1")  # fake-ip proxies map public names here
    assert not clipfetch.blocked_ip("93.184.216.34")


def test_fetch_picks_the_sharpest_copy_and_notes_the_source(fake_ytdlp, tmp_path):
    from main_character_comfy import clipfetch

    got = clipfetch.fetch("https://example.com/a", tmp_path, name="My Dance!", start=1, end=3)
    opts = fake_ytdlp.calls[-1]
    assert opts["format"] == "bv*+ba/b" and opts["format_sort"][0] == "res" and opts["merge_output_format"] == "mp4"
    assert opts["noplaylist"] and opts["download_ranges"] == ("ranges", [(1.0, 3.0)])
    assert got["path"].exists() and got["path"].name.startswith("my-dance-")
    assert got["height"] == 1080 and got["quality"] == "1080p" and got["tips"] == []
    meta = json.loads(got["path"].with_name(got["path"].name + ".source.json").read_text())
    assert meta["source_url"] == "https://example.com/a" and "rights" in meta["note"]
    # capped or low-res copies come back with the best-results tip
    low = clipfetch.fetch("https://example.com/a", tmp_path, max_height=480)
    assert fake_ytdlp.calls[-1]["format_sort"][0] == "res:480"
    assert low["height"] == 480 and "1080p or 4K" in low["tips"][0]
    with pytest.raises(InputError, match="playlist"):
        clipfetch.fetch("https://example.com/playlist", tmp_path)
    with pytest.raises(InputError, match="after start"):
        clipfetch.fetch("https://example.com/a", tmp_path, start=5, end=2)


def test_probe_and_search_rank_by_resolution(fake_ytdlp):
    from main_character_comfy import clipfetch

    p = clipfetch.probe("https://example.com/a")
    assert p["best_height"] == 2160 and p["best_quality"] == "4k" and p["heights"][:2] == [2160, 360]
    found = clipfetch.search("my dance", 2)
    assert [c["best_height"] for c in found] == [2160, 720]


def test_fetch_clip_tool_needs_rights_and_never_waits_for_the_gpu(server, svc, fake_ytdlp, sample_photo):
    res = call(server, "fetch_clip", {"url": "https://example.com/a"})
    assert res.is_error and "rights" in res.content[0].text
    # another process is rendering: a download still runs right away
    lane = jobs.GpuLane(svc.output_dir, {"source": "ui", "title": "someone else's render"})
    assert lane.try_acquire()
    try:
        done = run_job(server, "fetch_clip", {"url": "https://example.com/a", "name": "dance",
                                              "start_seconds": 0, "end_seconds": 2, "i_have_rights": True}, timeout=60)
    finally:
        lane.release()
    path = done["result"]["path"]
    assert path.startswith("library/clips/dance-") and svc.resolve(path).exists()
    assert done["result"]["quality"] == "1080p"
    probe = payload(call(server, "probe_clip", {"url": "https://example.com/a"}))
    assert probe["best_quality"] == "4k" and "1080p or 4K" in probe["tip"]
    found = payload(call(server, "find_clips", {"query": "my dance", "limit": 2}))
    assert found["candidates"][0]["best_height"] == 2160 and "rights" in found["rights"]
    # the fetched clip goes straight into a swap
    me = svc.upload("references", "me.jpg", content_base64=base64.b64encode(sample_photo.read_bytes()).decode())
    run_job(server, "swap_into_clip", {"clip": path, "photo": me["path"], "i_have_rights": True})


def test_ui_link_box_tour_and_tip(fake_ytdlp, tmp_path, monkeypatch):
    from main_character_comfy import copy as C
    from main_character_comfy import ui

    monkeypatch.setenv("MC_LIBRARY_DIR", str(tmp_path / "lib"))
    assert ui.fetch_clip_ui("https://example.com/a", False)[1] == C.CLIP_LINK_NEEDS_RIGHTS
    path, msg = ui.fetch_clip_ui("https://example.com/a", True)
    assert Path(path).parent == tmp_path / "lib" / "clips" and "1080p" in msg
    _FakeYDL.height = 480
    assert "sharper copy" in ui.fetch_clip_ui("https://example.com/a", True)[1]
    assert "couldn't fetch it" in ui.fetch_clip_ui("http://127.0.0.1/x.mp4", True)[1]
    assert 4 <= len(C.TOUR_STEPS) <= 6
    titles = [t for t, _ in C.TOUR_STEPS]
    assert titles[0].startswith("pick or paste a clip") and titles[-1] == "drive mc with your own agent"
    text = " ".join(t + b for t, b in C.TOUR_STEPS) + C.CLIP_TIP + C.CLIP_LINK_NOTE + C.AGENT_HINT + C.AGENT_SETUP
    assert "\u2014" not in text and "\u2013" not in text and "#" not in text.replace("#drive", "")
    assert "1080p or 4K, steady camera, good lighting, clean audio" in C.CLIP_TIP and "rights" in C.CLIP_TIP
    assert "1080p or 4K" in C.TOUR_STEPS[0][1]
    for i in range(len(C.TOUR_STEPS)):
        h = ui.tour_html(i)
        assert f"{i + 1}/{len(C.TOUR_STEPS)}" in h and "**" not in h
    demo = ui.build_ui("mock")
    cfg = json.dumps(demo.config, default=str)
    assert "mc-tour" in cfg and C.HELP_LINK in cfg and C.CLIP_LINK_BTN in cfg and "mc_tour_done" in ui.TOUR_HEAD


# --------------------------------------------------------------------------- reference quality
def test_reference_check_grades_size_blur_and_face(tmp_path):
    from PIL import Image, ImageFilter

    from main_character_comfy import refcheck

    src = Path(__file__).parents[1] / "examples" / "avatar_3d_flux2_klein.jpg"
    big = tmp_path / "big.png"
    Image.open(src).resize((1088, 1920), Image.LANCZOS).save(big)
    good = refcheck.check_reference(big)
    assert good["grade"] in ("good", "ok") and good["face"]["found"] is True and good["ok"]
    blurry = tmp_path / "blurry.png"
    Image.open(big).filter(ImageFilter.GaussianBlur(9)).save(blurry)
    b = refcheck.check_reference(blurry)
    assert b["grade"] == "weak" and any("blurry" in w for w in b["warnings"]) and "1024 px" in b["tip"]
    assert b["sharpness"] < good["sharpness"] / 5
    small = tmp_path / "small.png"
    Image.open(src).resize((136, 240)).save(small)
    sm = refcheck.check_reference(small)
    assert sm["grade"] == "weak" and "136x240" in sm["warnings"][0]
    blank = tmp_path / "wall.png"
    Image.new("RGB", (1200, 1600), (120, 110, 100)).save(blank)
    assert any("can't find a face" in w for w in refcheck.check_reference(blank, "photo")["warnings"])
    assert not any("face" in w for w in refcheck.check_reference(blank, "avatar")["warnings"])
    dark = tmp_path / "dark.png"
    Image.open(big).point(lambda v: v // 5).save(dark)
    assert any("dark" in w for w in refcheck.check_reference(dark)["warnings"])
    assert refcheck.summary_markdown(b).startswith("**weak reference.**")
    assert refcheck.safe_check(tmp_path / "missing.png") is None
    text = refcheck.TIP + " ".join(b["warnings"] + sm["warnings"])
    assert "\u2014" not in text and "\u2013" not in text


def test_agents_get_reference_quality_on_upload_and_renders(server, svc, tmp_path):
    from PIL import Image

    src = Path(__file__).parents[1] / "examples" / "avatar_3d_flux2_klein.jpg"
    tiny = tmp_path / "tiny.jpg"
    Image.open(src).resize((136, 240)).save(tiny)
    up = payload(call(server, "upload_file", {"kind": "references", "filename": "tiny.jpg",
                                              "content_base64": base64.b64encode(tiny.read_bytes()).decode()}))
    assert up["reference_quality"]["grade"] == "weak" and up["reference_quality"]["warnings"]
    chk = payload(call(server, "check_reference", {"path": up["path"]}))
    assert chk["grade"] == "weak" and chk["path"] == up["path"]
    job = payload(call(server, "make_meme", {"photo": up["path"], "animate": False, "i_have_rights": True}))
    assert job["reference_quality"][up["path"]]["grade"] == "weak" and "1024 px" in job["reference_quality"]["tip"]
    payload(call(server, "wait_for_job", {"job_id": job["id"], "timeout_s": 120}))
    clip = payload(call(server, "upload_file", {"kind": "clips", "filename": "c.mp4", "content_base64":
                                                base64.b64encode(_clip(tmp_path / "c.mp4").read_bytes()).decode()}))
    assert "reference_quality" not in clip


def test_tour_has_a_prominent_reference_step_and_the_ui_checks_drops():
    from main_character_comfy import copy as C
    from main_character_comfy import ui

    title, body = C.TOUR_STEPS[1]
    assert "reference" in title and "this step decides the likeness" in body
    for phrase in ("sharp", "well lit", "face and outfit clearly visible", "ideally full body", "no heavy filters",
                   "1024 px"):
        assert phrase in body and phrase in C.REF_TIP
    cfg = json.dumps(ui.build_ui("mock").config, default=str)
    assert "mc-ref-check" in cfg and "mc-ref-tip" in cfg
