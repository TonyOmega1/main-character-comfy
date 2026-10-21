"""End-to-end check of the agent connector on a real MC + ComfyUI install.

    python scripts/agent_e2e.py            # no GPU: remote HTTP mode + token, stdio, uploads, reference check
    python scripts/agent_e2e.py --gpu      # also a short real talk job (built-in example avatar + built-in voice)
    python scripts/agent_e2e.py --gpu --redub   # plus one short redub with lip re-sync on that talk clip

Everything it writes (report, server log, outputs, library) goes under --out (default: ~/mc_agent_e2e), never
into the repo folder. With --gpu it needs ComfyUI running; it starts its own comfy-api-proxy if none answers at
--proxy and stops it again at the end (or start MC once with run.ps1 / run.sh, which starts the proxy).

It starts its own connector processes (HTTP on --port with a fresh token, plus a stdio one), talks to them with the
official MCP client, and writes everything it sees to <out>/agent_e2e_report.json. With --gpu it first checks ComfyUI's queue
ONCE and stops (exit 2) if anything is running or pending: it never waits for idle or submits later on its own (team rule:
no wait-for-idle submitters during a GPU hold). --wait-for-idle brings back the old behaviour for a supervised run. Then it drives
mc_status -> list_templates -> upload_file -> talk -> wait_for_job -> get_output and downloads the signed link.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main_character_comfy.system import RAM_PREFLIGHT_GB, free_ram_gb, ram_preflight  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORT: dict = {"started": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "steps": []}


def log(step: str, **data) -> None:
    rec = {"t": time.strftime("%H:%M:%S"), "step": step, **data}
    REPORT["steps"].append(rec)
    print(json.dumps(rec, default=str)[:900], flush=True)


def payload(res):
    if res.is_error:
        raise RuntimeError(res.content[0].text if res.content else "tool error")
    sc = res.structured_content
    if sc is not None:
        return sc.get("result", sc) if isinstance(sc, dict) and set(sc) == {"result"} else sc
    return json.loads(res.content[0].text)


def comfy_queue(url: str) -> tuple[int, int] | None:
    try:
        q = httpx.get(f"{url}/queue", timeout=5).json()
        return len(q.get("queue_running", [])), len(q.get("queue_pending", []))
    except Exception:
        return None


def require_comfy_idle(url: str) -> None:
    """One look at the queue: busy or unreachable = stop now (exit 2), never wait."""
    q = comfy_queue(url)
    if q is None:
        log("comfy unreachable; GPU test skipped")
        raise SystemExit(2)
    if q != (0, 0):
        log("comfy busy; GPU test skipped (no waiting)", queue=q)
        raise SystemExit(2)
    log("comfy idle", checked="once")


def wait_comfy_idle(url: str, idle_min: float, max_hours: float) -> None:
    need, idle_since, end = idle_min * 60, None, time.time() + max_hours * 3600
    while time.time() < end:
        q = comfy_queue(url)
        if q == (0, 0):
            idle_since = idle_since or time.time()
            if time.time() - idle_since >= need:
                log("comfy idle", minutes=idle_min)
                return
        else:
            if idle_since:
                log("comfy busy again", queue=q)
            idle_since = None
        time.sleep(15)
    raise SystemExit("ComfyUI never went idle; GPU test skipped")


def proxy_ok(url: str) -> bool:
    try:
        return httpx.get(f"{url}/api/v2/health", timeout=3).status_code == 200
    except Exception:
        return False


def ensure_proxy(proxy: str, comfy: str, out: Path):
    """Use a running comfy-api-proxy, else start one for this test (returned so it can be stopped)."""
    if proxy_ok(proxy):
        log("proxy", url=proxy, started=False)
        return None
    exe = shutil.which("comfy-api-proxy") or shutil.which("comfy-api-proxy", path=str(Path(sys.executable).parent))
    if not exe:
        log("proxy missing", hint="pip install -r requirements.txt, or start MC once with run.ps1 / run.sh")
        raise SystemExit(2)
    port = proxy.rsplit(":", 1)[-1].split("/")[0]
    plog = open(out / "agent_e2e_proxy.log", "w", encoding="utf-8")
    p = subprocess.Popen([exe, "run", "--comfyui", comfy, "--port", port], stdout=plog, stderr=plog,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    for _ in range(60):
        if proxy_ok(proxy):
            log("proxy", url=proxy, started=True)
            return p
        time.sleep(0.5)
    p.terminate()
    log("proxy did not come up", url=proxy, log=str(out / "agent_e2e_proxy.log"))
    raise SystemExit(2)


async def http_session(base: str, token: str, fn):
    import httpx2
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with httpx2.AsyncClient(headers={"Authorization": f"Bearer {token}"}, timeout=120) as hc:
        async with streamable_http_client(f"{base}/mcp", http_client=hc) as streams:
            async with ClientSession(streams[0], streams[1]) as c:
                await c.initialize()
                return await fn(c)


async def stdio_session(env: dict, fn):
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    params = StdioServerParameters(command=sys.executable, args=["-m", "main_character_comfy.agent"], env=env,
                                   cwd=str(ROOT))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as c:
            await c.initialize()
            return await fn(c)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--comfy", default="http://127.0.0.1:8188")
    ap.add_argument("--proxy", default=os.environ.get("COMFY_BASE_URL", "http://127.0.0.1:8189"),
                    help="comfy-api-proxy URL (started for the test if nothing answers)")
    ap.add_argument("--redub", action="store_true",
                    help="with --gpu: also redub one short line on the talk clip with lip re-sync (one more GPU job)")
    ap.add_argument("--wait-for-idle", action="store_true",
                    help="old behaviour: poll until the queue has been empty for --idle-min (supervised runs only)")
    ap.add_argument("--idle-min", type=float, default=3.0)
    ap.add_argument("--max-wait-hours", type=float, default=6.0)
    ap.add_argument("--out", default=str(Path.home() / "mc_agent_e2e"),
                    help="where the report, logs, outputs and library go (default ~/mc_agent_e2e, not the repo)")
    a = ap.parse_args()
    if a.redub and not a.gpu:
        ap.error("--redub needs --gpu")
    out = Path(a.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    a.out = str(out)

    env = {**os.environ, "MC_OUTPUT_DIR": str(out), "MC_LIBRARY_DIR": str(out / "library"), "PYTHONPATH": str(ROOT),
           "COMFY_BASE_URL": a.proxy, "PYTHONUNBUFFERED": "1"}
    token = "mc_e2e_" + secrets.token_urlsafe(24)
    base = f"http://127.0.0.1:{a.port}"
    errlog = open(out / "agent_e2e_server.log", "w", encoding="utf-8")
    proxy_proc = None
    srv = subprocess.Popen([sys.executable, "-m", "main_character_comfy.agent", "--http", "--remote", "--port",
                            str(a.port), "--token", token], cwd=ROOT, env=env, stdout=errlog, stderr=errlog,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    ok = False
    try:
        for _ in range(120):
            try:
                if httpx.get(f"{base}/health", timeout=2).status_code == 200:
                    break
            except Exception:
                time.sleep(0.5)
        log("health", body=httpx.get(f"{base}/health").json())
        no_tok = httpx.post(f"{base}/mcp", json={}).status_code
        bad_tok = httpx.post(f"{base}/mcp", json={}, headers={"Authorization": "Bearer nope"}).status_code
        qs_tok = httpx.post(f"{base}/mcp?token={token}", json={}).status_code
        log("auth", no_token=no_tok, wrong_token=bad_tok, query_token=qs_tok)
        assert no_tok == 401 and bad_tok == 401 and qs_tok != 401

        async def basics(c):
            tools = [t.name for t in (await c.list_tools()).tools]
            st = payload(await c.call_tool("mc_status", {}))
            tpl = payload(await c.call_tool("list_templates", {}))
            up = payload(await c.call_tool("upload_file", {"kind": "references", "filename": "e2e_avatar.jpg",
                                                           "source_path": "examples/avatar_3d_flux2_klein.jpg"}))
            chk = payload(await c.call_tool("check_reference", {"path": up["path"], "subject": "avatar"}))
            return tools, st, tpl, up, chk

        tools, st, tpl, up, chk = asyncio.run(http_session(base, token, basics))
        log("http tools", count=len(tools))
        log("mc_status", status=st)
        log("list_templates", formats=len(tpl["meme_formats"]), examples=tpl["builtin_examples"])
        log("upload_file", path=up["path"], reference_quality=up.get("reference_quality"))
        log("check_reference", grade=chk["grade"], warnings=chk["warnings"])

        async def stdio_basics(c):
            return len((await c.list_tools()).tools), payload(await c.call_tool("list_jobs", {"limit": 5}))

        n, jobs = asyncio.run(stdio_session(env, stdio_basics))
        log("stdio", tools=n, jobs=jobs)

        if a.gpu:
            if a.wait_for_idle:
                wait_comfy_idle(a.comfy, a.idle_min, a.max_wait_hours)
            else:
                require_comfy_idle(a.comfy)
            proxy_proc = ensure_proxy(a.proxy, a.comfy, out)
            ram = free_ram_gb()
            note = ram_preflight()
            log("ram preflight", free_gb=None if ram is None else round(ram, 1), need_gb=RAM_PREFLIGHT_GB,
                warning=note or None)

            async def talk(c):
                job = payload(await c.call_tool("talk", {"photo": up["path"], "script": "pull up. main character energy.",
                                                         "voice": "builtin", "captions": True, "i_have_rights": True}))
                log("talk submitted", job=job)
                t0 = time.time()
                while True:
                    done = payload(await c.call_tool("wait_for_job", {"job_id": job["id"], "timeout_s": 120}))
                    log("wait_for_job", status=done["status"], stage=done.get("stage"), message=done.get("message"),
                        fraction=done.get("fraction"))
                    if done["status"] in ("done", "error", "cancelled", "lost"):
                        break
                    if time.time() - t0 > 3600:
                        raise RuntimeError("talk job took over an hour")
                got = None
                if done["status"] == "done":
                    got = payload(await c.call_tool("get_output", {"path": done["result"]["final"],
                                                                   "include_data": False}))
                return done, got

            # NOTE: not "out": that name is the output folder (report, logs, project.json search below)
            done, talk_out = asyncio.run(http_session(base, token, talk))
            log("talk done", status=done["status"], result=done.get("result"), error=done.get("error"))
            assert done["status"] == "done", done
            final = Path(a.out) / Path(done["result"]["final"]).relative_to("outputs")
            log("output file", path=str(final), exists=final.exists(), bytes=final.stat().st_size if final.exists() else 0)
            assert final.exists() and final.stat().st_size > 10_000
            link = talk_out.get("download_url")
            r = httpx.get(link, timeout=60)
            log("signed download", status=r.status_code, bytes=len(r.content), same_size=len(r.content) == final.stat().st_size)
            assert r.status_code == 200 and len(r.content) == final.stat().st_size
            tampered = httpx.get(link.replace("/files/", "/files/x"), timeout=10).status_code
            log("tampered link", status=tampered)

            async def stdio_sees(c):
                return payload(await c.call_tool("list_jobs", {"limit": 3}))

            log("stdio sees the http job", jobs=asyncio.run(stdio_session(env, stdio_sees)))

            if a.redub:
                async def redub(c):
                    job = payload(await c.call_tool("redub_line", {
                        "video": done["result"]["final"], "text": "main character.", "start": 0.4,
                        "fit_seconds": 1.2, "lipsync": True, "i_have_rights": True}))
                    log("redub submitted", job=job)
                    t0 = time.time()
                    while True:
                        res = payload(await c.call_tool("wait_for_job", {"job_id": job["id"], "timeout_s": 120}))
                        log("wait_for_job (redub)", status=res["status"], stage=res.get("stage"),
                            message=res.get("message"))
                        if res["status"] in ("done", "error", "cancelled", "lost"):
                            return res
                        if time.time() - t0 > 3600:
                            raise RuntimeError("redub job took over an hour")

                rd = asyncio.run(http_session(base, token, redub))
                log("redub done", status=rd["status"], result=rd.get("result"), error=rd.get("error"))
                assert rd["status"] == "done", rd
                rfinal = Path(a.out) / Path(rd["result"]["final"]).relative_to("outputs")
                log("redub output", path=str(rfinal), exists=rfinal.exists())
                assert rfinal.exists() and rfinal.stat().st_size > 10_000
                # did the mouth really get re-synced, or did MC keep it audio-only (no face found)?
                recs = sorted(out.rglob("project.json"), key=lambda q: q.stat().st_mtime)
                newest = recs[-1].read_text(encoding="utf-8", errors="replace") if recs else ""
                synced = "lip re-sync" in newest or any((recs[-1].parent / "lipdub").rglob("*.mp4")) if recs else False
                log("redub lip re-sync", project=str(recs[-1]) if recs else None, synced=bool(synced))
                assert synced, "redub finished audio-only: the lip re-sync step did not run"
        ok = True
    finally:
        srv.terminate()
        try:
            srv.wait(10)
        except Exception:
            srv.kill()
        errlog.close()
        if proxy_proc is not None:
            proxy_proc.terminate()
        REPORT["ok"] = ok
        REPORT["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
        (out / "agent_e2e_report.json").write_text(json.dumps(REPORT, indent=2, default=str), encoding="utf-8")
        print("OK" if ok else "FAILED", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
