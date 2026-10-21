"""Pre-flight checks before a long render.

    python -m main_character_comfy.doctor                       # proxy reachable through Comfy API v2?
    python -m main_character_comfy.doctor --live                # + ask the running ComfyUI what's installed
    python -m main_character_comfy.doctor --object-info oi.json # same check, offline, from a saved catalogue

The app itself only talks to ComfyUI through the Comfy SDK / comfy-api-proxy.
For the optional deep check, save ComfyUI's node catalogue once from your
browser (http://127.0.0.1:8188/object_info -> Save as oi.json); the doctor then
checks offline that every node type and model file the workflows need exists.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import graphs, memes, tiers
from .config import settings
from .presets import motion_prompt
from .validate import validate_graph


DEFAULT_COMFYUI = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")


def fetch_object_info(url: str = DEFAULT_COMFYUI, timeout: float = 30.0) -> dict:
    """Read-only: ComfyUI's own node catalogue (lists every installed model file per loader)."""
    import httpx

    r = httpx.get(url.rstrip("/") + "/object_info", timeout=timeout)
    r.raise_for_status()
    return r.json()


def check_models(oi: dict, t: "tiers.Tier", models) -> list[tuple[str, list[str], bool]]:
    """Validate every MC workflow for tier ``t`` against ``oi``: (name, problems, optional?)."""
    m = tiers.models_for(t, models)
    lite = t.motion_engine == "ti2v5b"
    heavy = " (needs 16 GB+: experimental on Lite)" if t.gated else ""
    scene_w, scene_h = t.scene_size
    motion = memes.motion_prompt(memes.RECIPES["side_eye"])
    checks = {
        "meme scene (FLUX.2 klein edit)": graphs.avatar_graph("x", models=m, reference=True, width=scene_w,
                                                              height=scene_h, te_device=t.te_device),
        ("meme motion (Wan2.2 TI2V 5B, Lite)" if lite else "meme motion (Wan 2.2 I2V A14B)"): (
            graphs.ti2v_motion_graph(motion, models=m) if lite
            else graphs.meme_motion_graph(motion, models=m, width=t.motion_size[0], height=t.motion_size[1])),
        f"Swap Me In (Wan2.2 Animate pack){heavy}": graphs.swap_graph("x", models=m, frames=77,
                                                                       animate_gguf=m.animate_gguf),
        f"Swap Me In, chunked (pack){heavy}": graphs.swap_graph("x", models=m, frames=144, carry=True,
                                                                 animate_gguf=m.animate_gguf),
        f"Swap Me In, multi-character (pack){heavy}": graphs.swap_graph("x", models=m, frames=77, person_mask=True,
                                                                         animate_gguf=m.animate_gguf),
        "people scan (SAM 3.1, pack)": graphs.people_scan_graph(models=m),
        f"Copy the Vibe (Animate move mode, pack minus SAM/relight){heavy}": graphs.vibe_graph(
            "x", models=m, frames=221, animate_gguf=m.animate_gguf),
        "avatar (text)": graphs.avatar_graph("x", models=m, reference=False, te_device=t.te_device),
        "voice clone": graphs.tts_graph("x", models=m, clone=True),
        "voice changer": graphs.voice_changer_graph(models=m),
        f"lip sync{heavy}": graphs.lipsync_graph(motion_prompt("3d"), models=m, audio_seconds=10),
        "duet lip sync (optional)": graphs.lipsync_graph(motion_prompt("3d"), models=m, audio_seconds=10,
                                                         two_speakers=True),
        "RIFE smoothing (optional)": graphs.smooth_graph(models=m),
    }
    out = []
    for name, sg in checks.items():
        problems = validate_graph(sg.graph, oi)
        soft = "optional" in name or "pack" in name or "16 GB+" in name
        out.append((name, problems, soft))
    return out


def live_report(tier: str = tiers.DEFAULT_TIER, url: str = DEFAULT_COMFYUI) -> str:
    """Markdown for the app's 'check my models' button."""
    t = tiers.get(tier)
    oi = fetch_object_info(url)
    rows = check_models(oi, t, settings().models)
    lines = [f"**MC {t.label}** vs the ComfyUI at `{url}`:"]
    for name, problems, soft in rows:
        if not problems:
            lines.append(f"- ✅ {name}")
        else:
            lines.append(f"- {'➖' if soft else '❌'} {name}: " + "; ".join(problems[:4])
                         + (" …" if len(problems) > 4 else ""))
    if any(p and not soft for _, p, soft in rows):
        lines.append("\nGet the missing files from `models.md` (links + licenses), drop them in ComfyUI's `models` "
                     "folders, restart ComfyUI and check again.")
    else:
        lines.append("\nCore modes are ready. ➖ = optional pack (Swap / Vibe / extras).")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from .cli import _utf8_console
    _utf8_console()
    ap = argparse.ArgumentParser(prog="main_character_comfy.doctor")
    ap.add_argument("--object-info", type=Path, help="saved /object_info JSON from your ComfyUI")
    ap.add_argument("--live", nargs="?", const=DEFAULT_COMFYUI, default=None, metavar="URL",
                    help=f"read the node/model list straight from a running ComfyUI (default {DEFAULT_COMFYUI})")
    ap.add_argument("--tier", choices=["auto", *tiers.TIER_KEYS], default=None,
                    help="check the model files for this tier (default: MC_TIER, else auto from your VRAM)")
    a = ap.parse_args(argv)
    s = settings()
    choice = tiers.resolve(a.tier if a.tier and a.tier != "auto" else None)
    t = choice.tier
    print(f"Hardware tier: {choice.describe()}")
    if choice.warning:
        print(f"  ⚠️ {choice.warning}")
    ok = True
    print(f"Comfy API v2 base URL: {s.comfy_base_url}")
    proxy_down = None
    try:
        from .backends import ComfyBackend

        info = ComfyBackend(s.comfy_base_url, s.comfy_api_key).health()
        print(f"  OK  comfy-api-proxy healthy, upstream ComfyUI {info.get('upstream')}")
    except Exception as exc:
        proxy_down = f"{type(exc).__name__}: {exc}"
    if proxy_down is not None:
        if a.live or a.object_info:
            # fresh install: run.ps1 / run.sh start the proxy themselves after this check, so it's a note, not a fail
            print("  --  comfy-api-proxy isn't running yet (fine on a fresh install: run.ps1 / run.sh start it "
                  "for you when MC starts)")
        else:
            ok = False
            print(f"  !!  proxy not reachable ({proxy_down}). Start ComfyUI, then: comfy-api-proxy run "
                  "(or use .\\run.ps1 -Check / ./run.sh --check to check ComfyUI directly)")

    if a.live:
        from .comfy_launch import flags_advice, running_flags

        fl = running_flags(a.live)
        if fl is not None:
            rv = "default" if fl["reserve_vram"] is None else "%g GB" % fl["reserve_vram"]
            tip = flags_advice(fl)
            print("  %s  ComfyUI flags: max upload %g MB, reserve VRAM %s%s" % (
                "!!" if tip else "OK", fl["max_upload_mb"], rv, f" ({tip})" if tip else ""))
    oi = None
    if a.object_info:
        oi = json.loads(a.object_info.read_text(encoding="utf-8"))
    elif a.live:
        try:
            oi = fetch_object_info(a.live)
            print(f"Read the installed nodes/models from ComfyUI at {a.live}")
        except Exception as exc:
            ok = False
            print(f"  !!  couldn't read {a.live}/object_info ({type(exc).__name__}: {exc}). Is ComfyUI running?")
    if oi is not None:
        for name, problems, soft in check_models(oi, t, s.models):
            if problems:
                if not soft:
                    ok = False
                print(f"  {'--' if soft else '!!'}  {name}:")
                for p in problems:
                    print(f"        {p}")
            else:
                print(f"  OK  {name}")
    print(f"\nMC {t.label}: launch ComfyUI with {t.comfy_flags or '(no extra flags)'}\n  {t.flags_note}\n  {t.notes}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
