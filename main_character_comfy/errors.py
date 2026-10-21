"""Turn raw ComfyUI / network failures into a plain-English fix.

Beginners mostly hit the same three walls (out of VRAM, a missing model file,
ComfyUI not running). The raw text stays available underneath for bug reports.
"""

from __future__ import annotations

import re

OOM_PATTERNS = ("out of memory", "allocation on device", "cuda error: out of memory", "outofmemoryerror",
                "not enough memory", "cublas_status_alloc_failed", "hip out of memory", "mps backend out of memory")
MISSING_PATTERNS = ("value not in list", "value_not_in_list", "not in list", "no such file", "file not found",
                    "could not find model", "missing model")
UPLOAD_PATTERNS = ("http 413", "status 413", "error 413", "413 request", "request entity too large", "payload too large", "httprequestentitytoolarge",
                   "client_max_size", "maximum request body size")
OFFLINE_PATTERNS = ("connection refused", "connecterror", "failed to establish", "actively refused",
                    "name or service not known", "timed out", "connection reset", "all connection attempts failed")


class Cancelled(RuntimeError):
    """The user pressed Stop."""


def kind(exc: BaseException | str) -> str:
    """``oom`` | ``missing_model`` | ``offline`` | ``upload_too_big`` | ``cancelled`` | ``other``."""
    if isinstance(exc, Cancelled):
        return "cancelled"
    text = f"{type(exc).__name__}: {exc}".lower() if isinstance(exc, BaseException) else str(exc).lower()
    if any(p in text for p in OOM_PATTERNS):
        return "oom"
    if any(p in text for p in UPLOAD_PATTERNS):
        return "upload_too_big"
    if any(p in text for p in OFFLINE_PATTERNS):
        return "offline"
    if any(p in text for p in MISSING_PATTERNS):
        return "missing_model"
    if "cancel" in text or "interrupt" in text:
        return "cancelled"
    return "other"


def missing_file(exc: BaseException | str) -> str:
    """Best guess at the model file name ComfyUI complained about ('' if none)."""
    m = re.search(r"([\w.\-]+\.(?:safetensors|gguf|ckpt|pt|pth|bin))", str(exc))
    return m.group(1) if m else ""


def friendly(exc: BaseException | str, tier: str = "") -> str:
    """One short, actionable message (Markdown) for the status box / console."""
    k = kind(exc)
    raw = f"{type(exc).__name__}: {exc}" if isinstance(exc, BaseException) else str(exc)
    raw = raw.strip().splitlines()[0][:300] if raw.strip() else ""
    if k == "cancelled":
        return "⏹️ Stopped. Nothing else is running; hit the button again whenever."
    if k == "oom":
        lite = "" if tier == "lite" else " Switch to **MC Lite** (More options → GPU tier) or"
        return ("🧠 Your GPU ran out of memory." + lite + " pick a shorter length, close other GPU apps "
                "(games, browsers with video), then try again. If ComfyUI froze, restart it. "
                f"\n\n<details><summary>details</summary>{raw}</details>")
    if k == "missing_model":
        f = missing_file(exc)
        what = f"`{f}`" if f else "a model file"
        return (f"📦 ComfyUI is missing {what}. Click **check my models** (or run "
                "`python -m main_character_comfy.doctor --live`) to see every missing file, and get it from "
                f"`models.md`. Then restart ComfyUI.\n\n<details><summary>details</summary>{raw}</details>")
    if k == "upload_too_big":
        return ("📤 ComfyUI refused the upload: it's over ComfyUI's upload limit (100 MB unless ComfyUI was started "
                "with `--max-upload-size`). Restart ComfyUI the MC way: `run_nvidia_gpu_mc.bat` in the ComfyUI folder, or "
                "`run.ps1 -StartComfy C:\\AI\\ComfyUI_windows_portable` (both use `--max-upload-size 500 "
                "--reserve-vram 1.5`; add `-MaxUploadMB 1024` for even bigger clips), or trim the clip first."
                f"\n\n<details><summary>details</summary>{raw}</details>")
    if k == "offline":
        return ("🔌 Can't reach ComfyUI. Start it (port 8188), then relaunch MC with `run.ps1` / `run.sh`, which "
                f"also starts the local API proxy (port 8189).\n\n<details><summary>details</summary>{raw}</details>")
    return f"❌ {raw}"
