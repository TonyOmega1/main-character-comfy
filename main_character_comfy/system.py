"""Local machine facts for the privacy badge and the hardware tier (no network, no ComfyUI calls)."""

from __future__ import annotations

import functools
import os
import shutil
import subprocess


@functools.lru_cache(maxsize=1)
def gpu_name() -> str:
    """GPU model from MC_GPU_NAME or ``nvidia-smi`` on this PC; '' if unknown."""
    env = os.environ.get("MC_GPU_NAME", "").strip()
    if env:
        return env
    exe = shutil.which("nvidia-smi")
    if not exe:
        return ""
    try:
        out = subprocess.run([exe, "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True,
                             timeout=5).stdout.strip().splitlines()
        return out[0].strip() if out else ""
    except Exception:
        return ""


def vram_gb() -> float | None:
    """Total VRAM in GB (GiB) from MC_VRAM_GB, else ``nvidia-smi`` on this PC; None if unknown.

    MC runs on the same PC as ComfyUI. comfy-api-proxy doesn't expose ComfyUI's
    system stats, and MC never talks to ComfyUI's own API, so the probe is local."""
    env = os.environ.get("MC_VRAM_GB", "").strip()
    if env:
        try:
            return float(env)
        except ValueError:
            return None
    return _smi_vram_gb()


@functools.lru_cache(maxsize=1)
def _smi_vram_gb() -> float | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=memory.total", "--format=csv,noheader,nounits"], capture_output=True,
                             text=True, timeout=5).stdout.strip().splitlines()
        return round(float(out[0]) / 1024, 1) if out else None  # MiB -> GiB
    except Exception:
        return None


def privacy_badge(mock: bool = False) -> str:
    if mock:
        return "🔒 Mock mode: nothing leaves this PC (no GPU renders)."
    name = gpu_name() or "your GPU"
    return f"🔒 Runs on your GPU: {name}. Your face never leaves your PC."


# --------------------------------------------------------------------------- system RAM preflight
# Measured on a 31 GB RAM test PC: loading InfiniteTalk for a lip sync job adds about 21 GB of system RAM in a
# few seconds; with a normal desktop open it hit 31 of 31 GB. So warn (never block) under about 22 GB free.
RAM_PREFLIGHT_GB = 22.0


def free_ram_gb() -> float | None:
    """Available system RAM in GB (GiB) on this PC: MC_FREE_RAM_GB, else the OS (Linux, Windows); None if unknown."""
    env = os.environ.get("MC_FREE_RAM_GB", "").strip()
    if env:
        try:
            return float(env)
        except ValueError:
            return None
    try:
        if os.name == "nt":
            import ctypes

            class _MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = _MS()
            st.dwLength = ctypes.sizeof(_MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return st.ullAvailPhys / 2**30
            return None
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 2**20
    except Exception:
        return None
    return None


def ram_preflight(need_gb: float = RAM_PREFLIGHT_GB) -> str:
    """'' when there's room for the lip sync model load, else a plain warning (MC still runs the job)."""
    free = free_ram_gb()
    if free is None or free >= need_gb:
        return ""
    return (f"Low free RAM: about {free:.0f} GB free. Loading the lip sync model (InfiniteTalk) briefly needs about "
            f"21 GB more system RAM, so close browsers and other heavy apps (or add a page file) before this render, or the PC can run "
            f"out of memory.")
