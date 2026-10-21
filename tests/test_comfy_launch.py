"""ComfyUI launch flags (--max-upload-size / --reserve-vram) and the 413 message. CPU only."""

from pathlib import Path

import pytest

from main_character_comfy import comfy_launch as cl
from main_character_comfy import errors


def test_comfy_args_and_validation():
    assert cl.comfy_args(max_upload_mb=500, reserve_vram=1.5) == ["--max-upload-size", "500", "--reserve-vram", "1.5"]
    assert cl.comfy_args(defaults=False) == []
    with pytest.raises(ValueError):
        cl.comfy_args(max_upload_mb=0)
    with pytest.raises(ValueError):
        cl.comfy_args(reserve_vram=-1)


def test_launch_command_portable_and_checkout(tmp_path):
    port = tmp_path / "portable"
    (port / "ComfyUI").mkdir(parents=True)
    (port / "ComfyUI" / "main.py").write_text("")
    (port / "python_embeded").mkdir()
    (port / "python_embeded" / "python.exe").write_text("")
    cmd = cl.launch_command(port, ["--max-upload-size", "500"])
    assert cmd[0].endswith("python.exe") and cmd[1] == "-s" and cmd[2].endswith("main.py")
    assert "--windows-standalone-build" in cmd and cmd[-2:] == ["--max-upload-size", "500"]
    co = tmp_path / "checkout"
    co.mkdir()
    (co / "main.py").write_text("")
    assert cl.launch_command(co, [], python="py")[:2] == ["py", str(co / "main.py")]
    with pytest.raises(FileNotFoundError):
        cl.launch_command(tmp_path / "nope", [])


def test_print_mode(tmp_path, capsys):
    (tmp_path / "main.py").write_text("")
    assert cl.main(["--root", str(tmp_path), "--max-upload-size", "600", "--reserve-vram", "2", "--print"]) == 0
    out = capsys.readouterr().out
    assert "--max-upload-size 600" in out and "--reserve-vram 2" in out


def test_parse_running_flags():
    assert cl.parse_flags(["main.py"]) == {"max_upload_mb": 100.0, "reserve_vram": None}
    f = cl.parse_flags(["main.py", "--max-upload-size", "500", "--reserve-vram=1.5"])
    assert f == {"max_upload_mb": 500.0, "reserve_vram": 1.5}


def test_upload_warning(tmp_path):
    big = tmp_path / "big.mp4"
    with open(big, "wb") as fh:
        fh.truncate(120 * 2 ** 20)
    assert "--max-upload-size" in cl.upload_warning(big, {"max_upload_mb": 100})
    assert cl.upload_warning(big, {"max_upload_mb": 500}) == ""


def test_413_gets_a_plain_fix():
    e = RuntimeError("HTTP 413: Request Entity Too Large")
    assert errors.kind(e) == "upload_too_big"
    assert "--max-upload-size" in errors.friendly(e)


def test_launchers_expose_the_flags():
    root = Path(__file__).resolve().parent.parent
    ps = (root / "run.ps1").read_text(encoding="utf-8")
    assert "-MaxUploadMB" in ps and "-ReserveVram" in ps and "comfy_launch" in ps
    sh = (root / "run.sh").read_text(encoding="utf-8")
    assert "MC_MAX_UPLOAD_MB" in sh and "MC_RESERVE_VRAM" in sh


# ---------------- CoS: every MC start of ComfyUI carries --reserve-vram 1.5 --max-upload-size 500 ----------------
def _flags(args):
    return cl.parse_flags(args)


def test_default_launch_args_include_both_flags(monkeypatch):
    monkeypatch.delenv("MC_MAX_UPLOAD_MB", raising=False)
    monkeypatch.delenv("MC_RESERVE_VRAM", raising=False)
    args = cl.default_launch_args()
    assert args == ["--max-upload-size", "500", "--reserve-vram", "1.5"]
    assert cl.comfy_args() == args
    assert cl.comfy_args(defaults=False) == []
    assert cl.comfy_args(reserve_vram=2) == ["--max-upload-size", "500", "--reserve-vram", "2"]  # streaming


def test_start_uses_mc_flags_on_main_py_and_prefers_a_correct_bat(tmp_path, monkeypatch):
    monkeypatch.delenv("MC_MAX_UPLOAD_MB", raising=False)
    monkeypatch.delenv("MC_RESERVE_VRAM", raising=False)
    monkeypatch.delenv("MC_COMFY_LAUNCHER", raising=False)
    root = tmp_path / "ComfyUI_windows_portable"
    (root / "ComfyUI").mkdir(parents=True)
    (root / "ComfyUI" / "main.py").write_text("")
    cmd = cl.start(root, dry_run=True)
    f = _flags(cmd)
    assert cmd[-4:] == ["--max-upload-size", "500", "--reserve-vram", "1.5"] and f["reserve_vram"] == 1.5
    bat = root / cl.MC_BAT
    bat.write_text("@echo off\nrem copy\n.\\python_embeded\\python.exe -s ComfyUI\\main.py --windows-standalone-build "
                   "--disable-auto-launch --reserve-vram 1.5 --max-upload-size 500\nrem .\\python_embeded\\python.exe -s ComfyUI\\main.py "
                   "--reserve-vram 2\npause\n")
    monkeypatch.setattr(cl, "_is_windows", lambda: True)
    cmd = cl.start(root, dry_run=True)
    assert cmd[:3] == ["cmd.exe", "/c", "start"] and cmd[-1] == str(bat)
    # explicit non-default flags (streaming) -> main.py with exactly those
    cmd = cl.start(root, reserve_vram=2, dry_run=True)
    assert cmd[-4:] == ["--max-upload-size", "500", "--reserve-vram", "2"]
    # a .bat WITHOUT the flags is never trusted
    bat.write_text(".\\python_embeded\\python.exe -s ComfyUI\\main.py --windows-standalone-build\n")
    cmd = cl.start(root, dry_run=True)
    assert "cmd.exe" not in cmd and cmd[-4:] == ["--max-upload-size", "500", "--reserve-vram", "1.5"]


def test_comfy_launcher_env_points_at_a_script(tmp_path, monkeypatch):
    s = tmp_path / "run_nvidia_gpu_mc.bat"
    s.write_text("python -s ComfyUI\\main.py --disable-auto-launch --reserve-vram 1.5 --max-upload-size 500\n")
    monkeypatch.setenv("MC_COMFY_LAUNCHER", str(s))
    assert cl.launcher_script(None) == s
    assert cl.start(None, dry_run=True)[-1] == str(s)


def test_cli_print_defaults(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("MC_COMFY_LAUNCHER", raising=False)
    (tmp_path / "main.py").write_text("")
    assert cl.main(["--root", str(tmp_path), "--print"]) == 0
    out = capsys.readouterr().out
    assert "--max-upload-size 500" in out and "--reserve-vram 1.5" in out


def test_flags_advice():
    assert cl.flags_advice({"max_upload_mb": 500, "reserve_vram": 1.5}) == ""
    tip = cl.flags_advice({"max_upload_mb": 100, "reserve_vram": None})
    assert "413" in tip and "run_nvidia_gpu_mc.bat" in tip and "-StartComfy" in tip


def test_every_start_path_carries_the_flags():
    root = Path(__file__).resolve().parent.parent
    ps = (root / "run.ps1").read_text(encoding="utf-8")
    assert "[double]$MaxUploadMB = 500" in ps and "[double]$ReserveVram = 1.5" in ps and "$ComfyLauncher" in ps
    agent = (root / "main_character_comfy" / "agent" / "__main__.py").read_text(encoding="utf-8")
    assert "--start-comfy" in agent and "ensure_running" in agent
    guide = (root / "docs" / "GUIDE.md").read_text(encoding="utf-8")
    readme = (root / "README.md").read_text(encoding="utf-8")
    for doc in (guide, readme):
        assert "run_nvidia_gpu_mc.bat" in doc and "-StartComfy" in doc


def test_write_mc_bat_copies_the_original_with_the_flags(tmp_path, monkeypatch):
    monkeypatch.delenv("MC_MAX_UPLOAD_MB", raising=False)
    monkeypatch.delenv("MC_RESERVE_VRAM", raising=False)
    monkeypatch.delenv("MC_COMFY_LAUNCHER", raising=False)
    orig = tmp_path / "run_nvidia_gpu.bat"
    orig.write_text(".\\python_embeded\\python.exe -s ComfyUI\\main.py --windows-standalone-build\r\n"
                    "echo If you see this...\r\npause\r\n")
    before = orig.read_bytes()
    path, made = cl.write_mc_bat(tmp_path)
    assert made and path.name == cl.MC_BAT and orig.read_bytes() == before
    f = cl.script_flags(path)
    assert f["reserve_vram"] == 1.5 and f["max_upload_mb"] == 500 and "--windows-standalone-build" in f["line"]
    assert "pause" in path.read_text() and "--reserve-vram 2" in path.read_text()
    monkeypatch.setattr(cl, "_is_windows", lambda: True)
    assert cl.launcher_script(tmp_path) == path  # MC then starts it
    path.write_text("custom")
    assert cl.write_mc_bat(tmp_path) == (path, False) and path.read_text() == "custom"  # never overwrites


def _stats_server(argv):
    """Tiny local ComfyUI stand-in: only /system_stats with the given argv."""
    import http.server
    import json as _json
    import threading

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body = _json.dumps({"system": {"argv": argv}}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


class _Proxy:
    def health(self):
        return {"status": "healthy", "upstream": "0.4.x"}


def test_app_status_line_warns_when_launch_flags_missing(monkeypatch):
    """The web app's status line (#mc-conn) carries the same flags check as doctor / comfy_launch / the agent."""
    from types import SimpleNamespace

    from main_character_comfy import copy as C, ui
    from tests.test_copy_lint import lint

    monkeypatch.setattr(ui, "make_backend", lambda *a, **k: _Proxy())
    s = SimpleNamespace(comfy_base_url="http://127.0.0.1:8189", comfy_api_key=None)
    srv, url = _stats_server(["main.py", "--listen"])  # plain ComfyUI: no MC flags
    try:
        line = ui.conn_status("comfy", s, url)
    finally:
        srv.shutdown()
    assert line.startswith("🟢 Connected")
    warn = line.split("\n", 1)[1]
    assert warn == C.LAUNCH_FLAGS_WARN.format(missing=", ".join([C.LAUNCH_FLAG_RESERVE, C.LAUNCH_FLAG_UPLOAD.format(mb=100)]))
    assert "reserve vram" in warn and "max upload size" in warn and "run_nvidia_gpu_mc.bat" in warn
    # terse lowercase, no dashes (not even a hyphen), no hashtags, copy lint clean
    assert warn == warn.lower() and "-" not in warn and "#" not in warn and lint(warn, "status") == []
    # only one flag missing -> only that one is named
    one = ui.launch_flags_note({"max_upload_mb": 500.0, "reserve_vram": None})
    assert C.LAUNCH_FLAG_RESERVE in one and "upload" not in one


def test_app_status_line_quiet_when_launch_flags_present(monkeypatch):
    from types import SimpleNamespace

    from main_character_comfy import ui

    monkeypatch.setattr(ui, "make_backend", lambda *a, **k: _Proxy())
    s = SimpleNamespace(comfy_base_url="http://127.0.0.1:8189", comfy_api_key=None)
    srv, url = _stats_server(["main.py", "--reserve-vram", "1.5", "--max-upload-size=500"])
    try:
        line = ui.conn_status("comfy", s, url)
    finally:
        srv.shutdown()
    assert line.startswith("🟢 Connected") and "\n" not in line and "launch flags" not in line
    # ComfyUI unreadable (proxy up, 8188 closed) -> no false alarm; mock mode never checks
    assert "launch flags" not in ui.conn_status("comfy", s, "http://127.0.0.1:9")
    assert ui.conn_status("mock", s).startswith("🧪 **Mock mode**")


def test_comfyui_never_opens_a_browser_tab(tmp_path, monkeypatch):
    """Every MC start of ComfyUI passes --disable-auto-launch; an older MC .bat without it is not trusted."""
    monkeypatch.delenv("MC_COMFY_LAUNCHER", raising=False)
    root = tmp_path / "portable"
    (root / "ComfyUI").mkdir(parents=True)
    (root / "ComfyUI" / "main.py").write_text("")
    cmd = cl.start(root, dry_run=True)
    assert cmd.count("--disable-auto-launch") == 1
    co = tmp_path / "checkout"
    co.mkdir()
    (co / "main.py").write_text("")
    assert "--disable-auto-launch" in cl.launch_command(co, [], python="py")
    assert cl.launch_command(co, ["--disable-auto-launch"], python="py").count("--disable-auto-launch") == 1
    old = root / cl.MC_BAT
    old.write_text(".\\python_embeded\\python.exe -s ComfyUI\\main.py --windows-standalone-build "
                   "--reserve-vram 1.5 --max-upload-size 500\n")
    monkeypatch.setattr(cl, "_is_windows", lambda: True)
    assert cl.launcher_script(root) is None
    assert "cmd.exe" not in cl.start(root, dry_run=True)
