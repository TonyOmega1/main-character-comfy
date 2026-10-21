"""Headless runner.

Examples::

    python -m main_character_comfy.cli --mode meme --meme side_eye --photo me.jpg --i-have-rights
    python -m main_character_comfy.cli --mode meme --meme calm_chaos --say-line --voice me.wav --photo me.jpg --i-have-rights
    python -m main_character_comfy.cli --pack side_eye,dramatic_zoom,slow_clap,calm_chaos --photo me.jpg --i-have-rights
    python -m main_character_comfy.cli --mode swap --clip my_dance.mp4 --photo me.jpg --clip-seconds 20 --i-have-rights
    python -m main_character_comfy.cli --mode vibe --clip my_dance.mp4 --photo me.jpg --scene-prompt "luxury hotel hallway at night" --i-have-rights
    python -m main_character_comfy.cli --remix outputs/<run>/mc_recipe.json --photo me.jpg --i-have-rights
    python -m main_character_comfy.cli --meme custom --scene "at a drive-through" --action "lowers sunglasses" --top "me" --save-format "Drive-Thru Stare"
    python -m main_character_comfy.cli --list-memes
    python -m main_character_comfy.cli --show-tier          # what MC detected for your GPU
    python -m main_character_comfy.cli --tier lite --meme side_eye --photo me.jpg --i-have-rights
    python -m main_character_comfy.cli --mode talk --photo me.jpg --voice me.wav --script "Hi!" --i-have-rights
    python -m main_character_comfy.cli --mode dialogue --character "two friendly robots" --builtin-voice \
        --script-file skit.txt
    python -m main_character_comfy.cli --mode sing --photo me.jpg --song mix.mp3 --vocals vocals.wav --song-start 42 \
        --song-seconds 20 --i-have-rights
    python -m main_character_comfy.cli --batch scripts.txt --photo me.jpg --voice me.wav --i-have-rights
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from .backends import make_backend
from .config import settings
from . import media, memes, tiers
from .pipeline import InputError, Pipeline, Request, request_from_recipe, split_batch_text
from .presets import STYLES


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mc", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", default="meme", choices=["meme", "swap", "vibe", "talk", "dialogue", "sing"])
    p.add_argument("--meme", default=memes.DEFAULT_RECIPE, choices=sorted(memes.RECIPES) + ["custom"])
    p.add_argument("--list-memes", action="store_true", help="print the meme format library and exit")
    p.add_argument("--pack", help="comma-separated meme formats to render from one photo")
    p.add_argument("--top", help="top caption (default: the format's)")
    p.add_argument("--bottom", help="bottom caption (default: the format's)")
    p.add_argument("--effect", choices=list(memes.EFFECTS))
    p.add_argument("--era", default="classic", choices=list(media.ERAS), help="Meme Era caption filter")
    p.add_argument("--length", choices=["3", "5"], help="meme length in seconds (default: the format's)")
    p.add_argument("--say-line", action="store_true", help="lip-sync the meme line instead of silent motion")
    p.add_argument("--line", default="", help="custom meme line")
    p.add_argument("--scene", default="", help="custom format: scene")
    p.add_argument("--action", default="", help="custom format: what you do")
    p.add_argument("--clip", type=Path, help="swap/vibe mode: a clip you have the rights to")
    p.add_argument("--clip-start", type=float, default=0.0)
    p.add_argument("--clip-seconds", type=float, default=10.0)
    p.add_argument("--swap-prompt", default="")
    p.add_argument("--no-relight", action="store_true", help="swap: don't match the clip's lighting")
    p.add_argument("--relight", action="store_true",
                   help="swap: match the clip's lighting (default: on for photos, off for avatars)")
    p.add_argument("--remove-props", action="store_true",
                   help="swap: don't redraw the original character's backpack / props on you (wider repaint area)")
    p.add_argument("--scene-prompt", default="", help="vibe mode: the NEW scene, e.g. 'luxury hotel hallway at night'")
    p.add_argument("--keep-audio", action="store_true", help="(old flag) same as --clip-audio original")
    p.add_argument("--clip-audio", choices=["original", "mute", "my_voice"], default="original",
                   help="swap/vibe soundtrack: the clip's own audio (default), mute, or re-voiced as you (--voice)")
    p.add_argument("--subject", choices=["auto", "photo", "avatar"], default="auto",
                   help="photo = a real face; avatar = cartoon/anime/3D/VTuber/pixel/mascot (keeps its art style)")
    p.add_argument("--people", default="",
                   help="multi-character swap (open-weights Openjutsu): one image per person, left to right, up to 8; "
                        "'-' = leave unchanged (e.g. 'me.png,-' swaps only the left person). Tier caps: Lite 1, "
                        "Standard 2, Max 8")
    p.add_argument("--resolution", choices=list(tiers.RESOLUTIONS), default="auto",
                   help="swap/vibe render size: auto (tier default: 832x480 on Standard), 480p, or 720p (Max; "
                        "experimental on 16 GB). The finished clip is still upscaled to 1080p")
    p.add_argument("--erase-props", default="", metavar="WHAT",
                   help="swap: erase these props from the clip first with VOID video inpainting, e.g. 'backpack' "
                        "(experimental, adds a render pass; needs the VOID models, see models.md)")
    p.add_argument("--group-engine", choices=["auto", "scail", "chain"], default="auto",
                   help="swap mode with 2+ people: auto (default: ONE SCAIL-2 pass when you swap everyone, 2-6 "
                        "people, clip up to ~5 s, Standard/Max) | scail | chain (one Wan Animate pass per person)")
    p.add_argument("--style-lock", choices=["off", "auto", "on"], default="off",
                   help="swap: keep drawn characters drawn (pending a real test): relight off, CFG 2.5 + anti-photo "
                        "negative, SeedVR2 blended 40%%, tighter binarised edge. auto = only for drawn references (default off)")
    p.add_argument("--drivers", default="",
                   help="swap with 2+ people, one per person like --people, separated by ',': whose moves each "
                        "person follows: '-' own (default), 'person:N', or a path to another clip (pending a real test)")
    p.add_argument("--split-cuts", choices=["auto", "on", "off"], default="auto",
                   help="swap: hard cuts. auto (default) = detect; swap each shot on its own on the one-pass-per-"
                        "person path, keeping who-is-who by appearance | on = always split | off = one pass")
    p.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0,
                   help="swap / vibe: turn the clip a further N degrees clockwise (phone clips are already "
                        "auto-rotated from their rotation tag; use this when the tag is missing or wrong)")
    p.add_argument("--ref-closeup", choices=["off", "auto"], default="off",
                   help="swap, one pass per person: crop a full-body reference to the face for close-up shots "
                        "(pending a real test)")
    p.add_argument("--scail-chunks", action="store_true",
                   help="swap, group pass: clips over ~5 s render as chained 81-frame SCAIL-2 chunks instead of one "
                        "Animate pass per person (experimental, pending a real test)")
    p.add_argument("--describe", default="",
                   help="swap mode, one per person like --people, separated by '|': each person's look, e.g. "
                        "\"anime girl, lavender bob, red bomber jacket|bearded trucker, olive bucket hat\". Lets MC "
                        "swap everyone in one SCAIL-2 pass and copy each reference exactly (single swaps use it too)")
    p.add_argument("--upscale", choices=["auto", "seedvr2", "lanczos"], default="auto",
                   help="swap mode: sharpen the take to 1080p with SeedVR2 before export. auto (default) = on for "
                        "SCAIL-2 group takes; seedvr2 = always (+~2.3 s per frame); lanczos = plain resize")
    p.add_argument("--defer-upscale", action="store_true",
                   help="swap mode: stop at the silent take (no SeedVR2, no export) for review; finish the takes "
                        "that pass with --finish-pending (SeedVR2 runs back to back, same graph and seed)")
    p.add_argument("--finish-pending", nargs="+", type=Path, metavar="DIR",
                   help="run SeedVR2 + export on deferred takes (the run folders from --defer-upscale), then exit")
    p.add_argument("--retake", type=Path, metavar="DIR",
                   help="swap mode: re-render a previous run (with a new --seed) reusing its prepared clip, erase, "
                        "people scan and soundtrack byte for byte, so ComfyUI can keep the Animate prep cached")
    p.add_argument("--no-route", action="store_true",
                   help="swap mode, 2+ people: don't send extreme close-ups / feet-only / no-face close shots "
                        "straight to Animate (they try SCAIL-2 first)")
    p.add_argument("--qc", choices=["auto", "off"], default="auto",
                   help="swap mode: framing QC of a SCAIL-2 take before SeedVR2; a failed take falls back to "
                        "Animate (auto, default) | off")
    p.add_argument("--face", default="",
                   help="swap mode, one per person like --people: auto (default: robots, animals and mascots don't copy "
                        "the dancer's face) | on (copy the dancer's face and expressions) | off (never copy the dancer's "
                        "face). e.g. --face auto,off,auto")
    p.add_argument("--find-people", action="store_true", help="swap mode: just number the people in --clip and save "
                                                               "a preview")
    p.add_argument("--remix", type=Path, help="re-run an mc_recipe.json with your own uploads")
    p.add_argument("--save-format", metavar="NAME", help="save --meme (or the custom --scene/--action) with --top/--bottom "
                                                        "as a community format, then exit")
    p.add_argument("--format-emoji", default="", help="emoji for --save-format")
    p.add_argument("--script", default="", help="what to say (dialogue: 'Name: line' per line; sing: lyrics)")
    p.add_argument("--script-file", type=Path)
    p.add_argument("--batch", type=Path, help="text file of scripts: one per line, or blocks separated by blank lines")
    p.add_argument("--photo", type=Path)
    p.add_argument("--photo-b", type=Path, help="dialogue: second person's photo")
    p.add_argument("--character", default="", help="describe the character(s) instead of a photo")
    p.add_argument("--avatar", type=Path, help="ready-made portrait (skips the avatar stage)")
    p.add_argument("--style", choices=sorted(STYLES), help="default: none (real you) for meme/swap, 3d otherwise")
    p.add_argument("--voice", type=Path, help="10-30 s sample of your voice")
    p.add_argument("--voice-b", type=Path, help="dialogue: speaker B's voice sample")
    p.add_argument("--builtin-voice", action="store_true", help="use Chatterbox's built-in voice")
    p.add_argument("--recording", type=Path, help="talk mode: your own recording of the script")
    p.add_argument("--character-voice", type=Path, help="voice changer target (needs rights)")
    p.add_argument("--song", type=Path)
    p.add_argument("--vocals", type=Path, help="vocals-only stem of the song")
    p.add_argument("--song-start", type=float, default=0.0)
    p.add_argument("--song-seconds", type=float, default=30.0)
    p.add_argument("--tier", choices=["auto", *tiers.TIER_KEYS], default=None,
                   help="hardware tier: lite (8-12 GB), standard (16 GB), max (24 GB+). Default: MC_TIER, else auto "
                        "from your GPU's VRAM")
    p.add_argument("--profile", choices=sorted(tiers.LEGACY_KEYS), help=argparse.SUPPRESS)  # old name for --tier
    p.add_argument("--allow-heavy", action="store_true",
                   help="Lite tier: try the 16 GB modes anyway (experimental: smaller quants, may run out of memory)")
    p.add_argument("--show-tier", action="store_true", help="print the detected hardware tier and exit")
    p.add_argument("--seam", default="anchor", choices=["anchor", "continue"])
    p.add_argument("--segment-seconds", type=float)
    p.add_argument("--smooth", default="off", choices=["off", "rife", "ffmpeg"])
    p.add_argument("--no-captions", action="store_true")
    p.add_argument("--sd", action="store_true", help="keep 480x832 instead of 1080x1920")
    p.add_argument("--no-watermark", action="store_true")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--steps", type=int)
    p.add_argument("--ext-frames", type=int, default=81)
    p.add_argument("--backend", choices=["comfy", "mock"])
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--i-have-rights", action="store_true",
                   help="confirm this is your face/voice or you have permission (required for uploads)")
    return p


def _utf8_console() -> None:
    """Windows consoles default to cp1252; meme titles carry emoji. Never crash on print."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass


# Per-person list options whose value may start with '-' (e.g. --drivers -,-,person:2 = "own moves, own moves,
# follow person 2"). argparse reads a leading '-' as a new flag and stops with "expected one argument" (Oct 20
# overnight crash), so "--opt VALUE" is joined to "--opt=VALUE" before parsing.
DASH_VALUE_OPTS = ("--drivers", "--people", "--face")


def _join_dash_values(argv: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        nxt = argv[i + 1] if i + 1 < len(argv) else None
        if tok in DASH_VALUE_OPTS and nxt is not None and nxt.startswith("-") and not nxt.startswith("--"):
            out.append(f"{tok}={nxt}")
            i += 2
            continue
        out.append(tok)
        i += 1
    return out


def main(argv: list[str] | None = None) -> int:
    _utf8_console()
    a = build_parser().parse_args(_join_dash_values(list(sys.argv[1:] if argv is None else argv)))
    if a.list_memes:
        for r in memes.RECIPES.values():
            extra = f"  line: \"{r.line}\"" if r.line else ""
            print(f"{r.key:22s} {r.title}  [{r.top} / {r.bottom}]{extra}")
        return 0
    if a.save_format:
        base = (memes.custom_recipe(a.scene, a.action, a.top or "", a.bottom or "", a.line) if a.meme == "custom"
                else memes.RECIPES[a.meme])
        caps = [(a.top or "", a.bottom or "")] if (a.top or a.bottom) else []
        key, path = memes.save_format(base, a.save_format, emoji=a.format_emoji, captions=caps, source=f"cli:{a.meme}")
        print(f"Saved format '{key}' -> {path}")
        return 0
    explicit = a.tier if a.tier and a.tier != "auto" else (tiers.LEGACY_KEYS[a.profile] if a.profile else None)
    choice = tiers.resolve(explicit)
    if a.show_tier:
        t = choice.tier
        print(choice.describe())
        print(f"  {t.notes}")
        print(f"  ComfyUI flags: {t.comfy_flags or '(none needed)'}")
        if t.gated:
            print(f"  Needs 16 GB+ (use --allow-heavy to try): {', '.join(sorted(t.gated))}")
        if choice.warning:
            print(f"  ⚠️ {choice.warning}")
        return 0
    if choice.warning:
        print(f"⚠️ {choice.warning}")
    s = settings()
    if a.output_dir:
        s = replace(s, output_dir=a.output_dir)
    backend = make_backend(a.backend or s.backend, s.comfy_base_url, s.comfy_api_key)
    script = a.script_file.read_text(encoding="utf-8") if a.script_file else a.script
    mode = "my_recording" if a.recording else "builtin" if a.builtin_voice else "clone"
    req = Request(
        mode=a.mode, script=script, photo=a.photo, photo_b=a.photo_b, character=a.character, avatar=a.avatar,
        style=a.style or ("none" if a.mode in ("meme", "swap", "vibe") or a.pack else "3d"), voice_mode=mode, voice_sample=a.voice, voice_sample_b=a.voice_b, recording=a.recording,
        character_voice=a.character_voice, song=a.song, vocals=a.vocals, song_start=a.song_start,
        song_seconds=a.song_seconds, captions=not a.no_captions, smooth=a.smooth, hd=not a.sd,
        watermark=not a.no_watermark, seed=a.seed, steps=a.steps, ext_frames=a.ext_frames,
        tier=choice.key, allow_heavy=a.allow_heavy,
        seam=a.seam, segment_seconds=a.segment_seconds, consent=a.i_have_rights,
        meme=a.meme, meme_top=a.top, meme_bottom=a.bottom, effect=a.effect, era=a.era,
        meme_frames={"3": 49, "5": 81}.get(a.length or ""), say_line=a.say_line, line=a.line,
        custom_scene=a.scene, custom_motion=a.action, clip=a.clip, clip_start=a.clip_start,
        clip_seconds=a.clip_seconds, swap_prompt=a.swap_prompt, relight=True if a.relight else (False if a.no_relight else None), remove_props=a.remove_props,
        scene_prompt=a.scene_prompt, keep_audio=a.keep_audio, clip_audio=a.clip_audio, subject=a.subject,
        people=[None if x.strip() in ("", "-") else Path(x.strip()) for x in a.people.split(",")] if a.people else [],
        face=[x.strip() if x.strip() not in ("", "-") else "auto" for x in a.face.split(",")] if a.face else [],
        resolution=a.resolution, erase_props=a.erase_props, group_engine=a.group_engine, upscale=a.upscale,
        people_desc=[x.strip() for x in a.describe.split('|')] if a.describe else [],
        route_framing=not a.no_route, qc=a.qc, defer_upscale=a.defer_upscale, reuse_prep=a.retake,
        style_lock=a.style_lock, split_cuts=a.split_cuts, scail_chunks=a.scail_chunks, rotate=a.rotate,
        ref_closeup=a.ref_closeup,
        people_driver=["" if x.strip() in ("", "-") else x.strip() for x in a.drivers.split(",")] if a.drivers else [],
    )
    if a.remix:
        doc = json.loads(a.remix.read_text(encoding="utf-8"))
        uploads = {k: v for k, v in dict(photo=a.photo, avatar=a.avatar, voice_sample=a.voice, recording=a.recording,
                                         clip=a.clip, song=a.song, vocals=a.vocals).items() if v}
        req = request_from_recipe(doc, consent=a.i_have_rights, tier=choice.key, allow_heavy=a.allow_heavy,
                                  **uploads)
        req = replace(req, route_framing=not a.no_route, qc=a.qc, defer_upscale=a.defer_upscale,
                      reuse_prep=a.retake)
    print(f"Hardware: {choice.describe()}")
    pipe = Pipeline(backend, s)
    if a.finish_pending:
        events = pipe.finish_pending(a.finish_pending)
    elif a.find_people:
        events = pipe.scan_people(req)
    elif a.pack:
        events = pipe.run_pack(req, [k.strip() for k in a.pack.split(",")])
    elif a.batch:
        events = pipe.run_batch(replace(req, mode="talk"), split_batch_text(a.batch.read_text(encoding="utf-8")))
    else:
        events = pipe.run(req)
    from .jobs import laned_events
    from .pipeline import Event

    title = (a.pack and f"pack {a.pack}" or (a.find_people and "find people")
             or (a.finish_pending and f"finish {len(a.finish_pending)} take(s)")
             or f"{a.mode} {a.meme if a.mode == 'meme' else ''}")
    events = laned_events(s.output_dir, events, "find_people" if a.find_people else a.mode, title.strip(),
                          source="cli", cancel=pipe.cancel,
                          wait_event=lambda msg: Event("prepare", "progress", -1.0, msg))
    final = None
    finals: list = []
    try:
        for ev in _print_events(events):
            if ev.status == "complete":
                final = ev.artifact
                finals.append(ev.artifact)
    except InputError as exc:
        print(f"⚠️ {exc}")
        return 2
    except Exception as exc:  # the fix first; the raw error is in the details line
        from .errors import friendly

        print(friendly(exc, choice.key).replace("<details><summary>details</summary>", "\n  details: ")
              .replace("</details>", ""))
        return 1
    if len(finals) > 1:
        print("\nFinal videos:\n" + "\n".join(f"  {f}" for f in finals))
        return 0
    print(f"\nFinal video: {final}")
    return 0


def _print_events(events):
    for ev in events:
        if ev.status == "progress":
            pct = f"{ev.fraction:5.0%} " if ev.fraction >= 0 else ""
            print(f"  [{ev.stage}] {pct}{ev.message}")
        else:
            print(f"[{ev.stage}] {ev.status}: {ev.message}" + (f" -> {ev.artifact}" if ev.artifact else ""))
        yield ev


if __name__ == "__main__":
    sys.exit(main())
