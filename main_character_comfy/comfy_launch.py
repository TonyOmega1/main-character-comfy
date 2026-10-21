"""Start ComfyUI with the flags MC needs, and read which flags a running ComfyUI was started with.

Two ComfyUI server flags matter for MC (both are ComfyUI's own, comfy/cli_args.py):

* ``--max-upload-size MB`` (default 100): MC uploads clips, driver videos and audio through ComfyUI's upload
  route; anything bigger comes back as HTTP 413. Long or 4K BYO clips need 300-500.
* ``--reserve-vram GB``: keep that much VRAM free for the OS / OBS / a stream encoder while rendering. With
  dynamic VRAM (ComfyUI 0.4x) ``--lowvram`` does nothing; ``--reserve-vram`` is the knob that still works.

    python -m main_character_comfy.comfy_launch --root C:\\AI\\ComfyUI_windows_portable --max-upload-size 500 \\
        --reserve-vram 1.5          # start it (portable layout or a git checkout with main.py)
    python -m main_character_comfy.comfy_launch --root ... --print     # just print the command
    python -m main_character_comfy.comfy_launch --flags                 # what is the running ComfyUI using?

The launchers wrap this: ``run.ps1 -StartComfy <root>`` / ``COMFYUI_DIR=<root> ./run.sh`` / the agent connector's
``--start-comfy``. **Every MC start of ComfyUI uses --reserve-vram 1.5 --max-upload-size 500 unless you pass other
values** (DEFAULT_RESERVE_VRAM / DEFAULT_MAX_UPLOAD_MB). In a portable folder that has ``run_nvidia_gpu_mc.bat`` (a
copy of run_nvidia_gpu.bat with those flags), MC starts that .bat, as long as it really carries both flags; point
``MC_COMFY_LAUNCHER`` at another launcher script to use that instead. Nothing here changes attention / Sage.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

DEFAULT_URL = "http://127.0.0.1:8188"
COMFY_DEFAULT_UPLOAD_MB = 100.0  # ComfyUI's own default (comfy/cli_args.py)
DEFAULT_MAX_UPLOAD_MB = 500.0  # what MC always starts ComfyUI with (long / 4K clips; HTTP 413 above the limit)
DEFAULT_RESERVE_VRAM = 1.5  # GB kept free for the Windows desktop (~1.3-1.5 GB measured on the RTX 5080)
MC_BAT = "run_nvidia_gpu_mc.bat"
NO_AUTO_LAUNCH = "--disable-auto-launch"  # ComfyUI's own flag: no browser tab on start  # portable-folder copy of run_nvidia_gpu.bat with the two flags


def comfy_args(*, max_upload_mb: float | None = None, reserve_vram: float | None = None, port: int | None = None,
               extra: list[str] | None = None, defaults: bool = True) -> list[str]:
    """ComfyUI command-line flags (validated). ``None`` = MC's default (500 MB / 1.5 GB, or $MC_MAX_UPLOAD_MB /
    $MC_RESERVE_VRAM); ``defaults=False`` leaves unset flags out (ComfyUI's own defaults)."""
    if defaults:
        if max_upload_mb is None:
            max_upload_mb = float(os.environ.get("MC_MAX_UPLOAD_MB") or DEFAULT_MAX_UPLOAD_MB)
        if reserve_vram is None:
            reserve_vram = float(os.environ.get("MC_RESERVE_VRAM") or DEFAULT_RESERVE_VRAM)
    out: list[str] = []
    if max_upload_mb is not None:
        if not 1 <= float(max_upload_mb) <= 10240:
            raise ValueError("max upload size is 1-10240 MB")
        out += ["--max-upload-size", f"{float(max_upload_mb):g}"]
    if reserve_vram is not None:
        if not 0 <= float(reserve_vram) <= 64:
            raise ValueError("reserve VRAM is 0-64 GB")
        out += ["--reserve-vram", f"{float(reserve_vram):g}"]
    if port:
        out += ["--port", str(int(port))]
    return out + list(extra or [])


def launch_command(root: str | Path, flags: list[str], python: str | None = None) -> list[str]:
    """The ComfyUI command for a portable folder (python_embeded + ComfyUI/main.py) or a checkout (main.py)."""
    root = Path(root)
    emb = root / "python_embeded" / "python.exe"
    if not emb.exists():
        emb = root / "python_embeded" / "python"
    # --disable-auto-launch: MC opens its own window; ComfyUI must not pop a browser tab on start
    nal = [] if NO_AUTO_LAUNCH in flags else [NO_AUTO_LAUNCH]
    if (root / "ComfyUI" / "main.py").exists():
        py = python or (str(emb) if emb.exists() else sys.executable)
        return [py, "-s", str(root / "ComfyUI" / "main.py"), "--windows-standalone-build", *nal, *flags]
    if (root / "main.py").exists():
        return [python or sys.executable, str(root / "main.py"), *nal, *flags]
    raise FileNotFoundError(f"no ComfyUI found in {root} (expected ComfyUI/main.py or main.py)")


def default_launch_args() -> list[str]:
    """The flags every MC start of ComfyUI carries (CoS requirement, Oct 19 2026)."""
    return comfy_args()


def script_flags(path: str | Path) -> dict | None:
    """Flags on the (first, un-commented) ``main.py`` line of a launcher .bat / .sh, or None if there is none."""
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in lines:
        s = line.strip()
        if not s or s.lower().startswith(("rem ", "::", "#", "@rem")) or "main.py" not in s:
            continue
        return {**parse_flags(shlex.split(s, posix=False)), "line": s}
    return None


def _is_windows() -> bool:
    return os.name == "nt"


def launcher_script(root: str | Path | None) -> Path | None:
    """The launcher script MC should start: $MC_COMFY_LAUNCHER, else ``<root>/run_nvidia_gpu_mc.bat`` (Windows). Only
    returned when its main.py line carries both MC flags at least at MC's values plus --disable-auto-launch; else None
    (MC runs main.py)."""
    env = os.environ.get("MC_COMFY_LAUNCHER", "").strip()
    cand = Path(env) if env else (Path(root) / MC_BAT if root and _is_windows() else None)
    if cand is None or not cand.is_file():
        return None
    f = script_flags(cand)
    if not f or f["reserve_vram"] is None or f["reserve_vram"] < DEFAULT_RESERVE_VRAM \
            or f["max_upload_mb"] < DEFAULT_MAX_UPLOAD_MB or NO_AUTO_LAUNCH not in f["line"]:
        return None  # an older MC .bat without --disable-auto-launch: run main.py with the full flags instead
    return cand


def script_command(script: Path) -> list[str]:
    """Command that runs a launcher script in its own window (Windows) / session."""
    if script.suffix.lower() in (".bat", ".cmd"):
        return ["cmd.exe", "/c", "start", "ComfyUI (MC)", "/D", str(script.parent), str(script)]
    return ["bash", str(script)]


def start(root: str | Path | None, *, max_upload_mb: float | None = None, reserve_vram: float | None = None,
          port: int | None = None, extra: list[str] | None = None, python: str | None = None,
          dry_run: bool = False) -> list[str]:
    """Start ComfyUI the MC way and return the command. With MC's default flags (and no port / extra args), a
    launcher script that carries them (run_nvidia_gpu_mc.bat / $MC_COMFY_LAUNCHER) wins; else main.py + flags."""
    custom = (max_upload_mb not in (None, DEFAULT_MAX_UPLOAD_MB) or reserve_vram not in (None, DEFAULT_RESERVE_VRAM)
              or port or extra or os.environ.get("MC_MAX_UPLOAD_MB") or os.environ.get("MC_RESERVE_VRAM"))
    script = None if custom else launcher_script(root)
    if script is not None:
        cmd, cwd = script_command(script), script.parent
    else:
        if not root:
            raise FileNotFoundError("no ComfyUI folder given (--root / COMFYUI_DIR / MC_COMFYUI_DIR)")
        cmd = launch_command(root, comfy_args(max_upload_mb=max_upload_mb, reserve_vram=reserve_vram, port=port,
                                              extra=extra), python)
        cwd = Path(root)
    if dry_run:
        return cmd
    kw: dict = {"cwd": str(cwd)}
    if os.name == "nt":
        kw["creationflags"] = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    else:
        kw["start_new_session"] = True
    subprocess.Popen(cmd, **kw)
    return cmd


def ensure_running(root: str | Path | None, url: str = DEFAULT_URL, wait: float = 180.0) -> str:
    """Start ComfyUI (MC flags) if nothing answers at ``url``; then wait for it. Returns a one-line status."""
    import time

    if running_flags(url, timeout=2.0) is not None:
        return f"ComfyUI already running at {url}"
    cmd = start(root)
    t0 = time.monotonic()
    while time.monotonic() - t0 < wait:
        if running_flags(url, timeout=2.0) is not None:
            return f"started ComfyUI in {time.monotonic() - t0:.0f} s: {' '.join(cmd[-4:])}"
        time.sleep(1.0)
    return f"started ComfyUI ({' '.join(cmd[-4:])}) but it isn't answering at {url} yet; check its window"


def parse_flags(argv: list[str]) -> dict:
    """{'max_upload_mb': float, 'reserve_vram': float | None} from a ComfyUI argv (defaults when absent)."""
    out = {"max_upload_mb": COMFY_DEFAULT_UPLOAD_MB, "reserve_vram": None}
    args = list(argv or [])
    for i, a in enumerate(args):
        key, _, val = a.partition("=")
        if key in ("--max-upload-size", "--reserve-vram"):
            v = val or (args[i + 1] if i + 1 < len(args) else "")
            try:
                out["max_upload_mb" if key == "--max-upload-size" else "reserve_vram"] = float(v)
            except ValueError:
                pass
    return out


def running_flags(url: str = DEFAULT_URL, timeout: float = 4.0) -> dict | None:
    """Flags of the running ComfyUI (from /system_stats 'argv'), or None when it isn't reachable."""
    import urllib.request

    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/system_stats", timeout=timeout) as r:
            stats = json.loads(r.read().decode("utf-8"))
    except Exception:  # noqa: BLE001
        return None
    argv = (stats.get("system") or {}).get("argv") or []
    return {**parse_flags(argv), "argv": argv}


def write_mc_bat(root: str | Path) -> tuple[Path, bool]:
    """Create ``<root>/run_nvidia_gpu_mc.bat`` from the portable's run_nvidia_gpu.bat with MC's flags (and a
    commented streaming line). Never overwrites; returns (path, created). The original .bat is left untouched."""
    root = Path(root)
    dst = root / MC_BAT
    if dst.exists():
        return dst, False
    src = root / "run_nvidia_gpu.bat"
    lines = src.read_text(encoding="utf-8", errors="replace").splitlines() if src.is_file() else [
        r".\python_embeded\python.exe -s ComfyUI\main.py --windows-standalone-build", "pause"]
    flags = " ".join(default_launch_args()).replace("--max-upload-size 500 --reserve-vram 1.5",
                                                     "--reserve-vram 1.5 --max-upload-size 500")
    out = ["@echo off", "rem MC copy of run_nvidia_gpu.bat (made by main_character_comfy.comfy_launch --write-bat;",
           "rem the original is untouched). Keeps 1.5 GB VRAM free for the desktop and allows 500 MB uploads.",
           "rem STREAMING (OBS / NVENC): comment the first python line and uncomment the second (reserve 2 GB)."]
    done = False
    for line in lines:
        s = line.strip()
        if s.lower() == "@echo off":
            continue
        if not done and "main.py" in s and not s.lower().startswith(("rem", "::")):
            nal = "" if NO_AUTO_LAUNCH in s else f" {NO_AUTO_LAUNCH}"
            out += [f"{s}{nal} {flags}", f"rem {s}{nal} --reserve-vram 2 --max-upload-size 500"]
            done = True
        else:
            out.append(line)
    dst.write_text("\r\n".join(out) + "\r\n", encoding="utf-8")
    return dst, True


def flags_advice(flags: dict | None) -> str:
    """'' when a running ComfyUI has MC's flags, else how to restart it with them."""
    if not flags:
        return ""
    low = []
    if flags.get("max_upload_mb", COMFY_DEFAULT_UPLOAD_MB) < DEFAULT_MAX_UPLOAD_MB:
        low.append("uploads over %g MB fail with HTTP 413" % flags.get("max_upload_mb", COMFY_DEFAULT_UPLOAD_MB))
    if flags.get("reserve_vram") is None or flags["reserve_vram"] < DEFAULT_RESERVE_VRAM:
        low.append("no VRAM kept free for the desktop")
    if not low:
        return ""
    return ("; ".join(low) + ": restart ComfyUI with run_nvidia_gpu_mc.bat or run.ps1 -StartComfy <folder> "
            "(--reserve-vram 1.5 --max-upload-size 500)")


def upload_warning(path: str | Path, flags: dict | None) -> str:
    """'' or a one-line heads-up when ``path`` is bigger than the running ComfyUI accepts."""
    try:
        mb = Path(path).stat().st_size / 2 ** 20
    except OSError:
        return ""
    limit = (flags or {}).get("max_upload_mb", COMFY_DEFAULT_UPLOAD_MB)
    if mb <= limit * 0.98:
        return ""
    return (f"{Path(path).name} is {mb:.0f} MB, over ComfyUI's {limit:g} MB upload limit: restart ComfyUI with "
            f"--max-upload-size {max(500, int(mb * 1.5) // 100 * 100)} (run.ps1 -MaxUploadMB) or trim the clip")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="main_character_comfy.comfy_launch", description=__doc__.split("\n")[0])
    ap.add_argument("--root", help="ComfyUI portable folder or checkout")
    ap.add_argument("--python", default=None, help="Python for ComfyUI (default: the portable's python_embeded)")
    ap.add_argument("--max-upload-size", type=float, default=None, metavar="MB")
    ap.add_argument("--reserve-vram", type=float, default=None, metavar="GB")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--print", action="store_true", help="print the command instead of starting ComfyUI")
    ap.add_argument("--flags", action="store_true", help="show the running ComfyUI's upload / VRAM flags")
    ap.add_argument("--write-bat", action="store_true",
                    help=f"create {MC_BAT} in --root (a copy of run_nvidia_gpu.bat with MC's flags), then exit")
    ap.add_argument("--url", default=os.environ.get("COMFYUI_URL", DEFAULT_URL))
    a, extra = ap.parse_known_args(argv)
    if a.flags:
        f = running_flags(a.url)
        if f is None:
            print(f"ComfyUI not reachable at {a.url}")
            return 1
        rv = "default" if f["reserve_vram"] is None else "%g GB" % f["reserve_vram"]
        print("max upload %g MB · reserve VRAM %s" % (f["max_upload_mb"], rv))
        return 0
    root = a.root or os.environ.get("MC_COMFYUI_DIR") or os.environ.get("COMFYUI_DIR")
    if a.write_bat:
        if not root:
            ap.error("--write-bat needs --root (your ComfyUI_windows_portable folder)")
        path, made = write_mc_bat(root)
        print(("created " if made else "already there (left as is): ") + str(path))
        return 0
    if not root and not os.environ.get("MC_COMFY_LAUNCHER"):
        ap.error("--root (or COMFYUI_DIR / MC_COMFY_LAUNCHER) is required to start ComfyUI")
    cmd = start(root, max_upload_mb=a.max_upload_size, reserve_vram=a.reserve_vram, port=a.port, extra=extra,
                python=a.python, dry_run=a.print)
    line = subprocess.list2cmdline(cmd) if os.name == "nt" else shlex.join(cmd)
    print(line if a.print else f"Started ComfyUI: {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
