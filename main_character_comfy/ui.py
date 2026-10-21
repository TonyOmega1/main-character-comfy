"""Gradio interface for MC.

Layout only: microcopy lives in copy.py, visuals in themes/ (swap with
MC_THEME). Hierarchy: one obvious primary action per tab, a fast path
(selfie -> format -> button, captions prefilled), everything else behind
"More options" / "Advanced".
"""

from __future__ import annotations

import html
import json
from urllib.parse import quote
import random
from pathlib import Path
from typing import Any, Iterator

import gradio as gr

from . import copy as C
from . import media, memes, refcheck, stylelock, themes, tiers, uibits
from .backends import make_backend
from .errors import friendly as friendly_error
from .media import ERAS
from .system import privacy_badge
from .config import settings
from .jobs import laned_events
from .pipeline import (STAGE_LABELS, STAGES, VIBE_SCENE_IDEAS, Event, InputError, Pipeline, Request,
                       estimate_animate_seconds, eta_range_text, split_batch_text, stages_for)
from .presets import SCRIPT_STARTERS, STYLES

DEFAULT_PACK = ["side_eye", "dramatic_zoom", "slow_clap", "calm_chaos"]
# The pipeline the stop button (and an agent's cancel_job on a window render) stops: whatever started last.
_RUNNING: dict[str, Pipeline | None] = {"p": None}


def _cancel_running() -> None:
    p = _RUNNING.get("p")
    if p is not None:
        p.cancel()


def in_gpu_lane(events: Iterator, mode: str, title: str = "") -> Iterator:
    """A window render inside the shared GPU lane (agents queue behind it, it queues behind them), listed in the
    shared job list so agents can see it."""
    return laned_events(settings().output_dir, events, mode, title or f"{mode} (mc window)", source="ui",
                        cancel=_cancel_running, wait_event=lambda msg: Event("prepare", "progress", -1.0, msg))


ICONS = {"pending": "⚪", "start": "🟡", "progress": "🟡", "done": "✅", "skipped": "⏭️", "error": "❌"}


def _stages_html(state: dict[str, dict[str, Any]], mode: str = "meme") -> str:
    rows = []
    for st in stages_for(mode):
        s = state.get(st, {"status": "pending", "fraction": 0.0, "message": ""})
        pct = int(max(0.0, min(1.0, s["fraction"])) * 100)
        rows.append(
            f"<div class='mc-stage is-{html.escape(s['status'])}'><div>{ICONS.get(s['status'], '⚪')}</div>"
            f"<div><b>{html.escape(STAGE_LABELS[st])}</b> <span class='mc-msg'>{html.escape(s['message'])}</span>"
            f"<div class='mc-bar'><div style='width:{pct}%'></div></div></div><div>{pct}%</div></div>"
        )
    idle = " data-idle='1'" if all(state.get(st, {}).get("status", "pending") == "pending" for st in stages_for(mode)) else ""
    return f"<div class='mc-stages'{idle}>" + "".join(rows) + "</div>"


def _style_choices() -> list[tuple[str, str]]:
    return [(f"{s.emoji} {s.label}", s.key) for s in STYLES.values()]


def _p(file: Any) -> Path | None:
    if not file:
        return None
    return Path(file if isinstance(file, str) else getattr(file, "name", file))


def _recipe_choices() -> list[tuple[str, str]]:
    return [(r.title, r.key) for r in memes.RECIPES.values()] + [("✍️ Custom format", "custom")]


def _recipe_card(key: str) -> str:
    r = memes.RECIPES.get(key)
    if r is None:
        return "Write your own scene and action under 'More options'. Same rules: original idea, you as the star, no real people."
    line = f" · 🗣️ line: *“{r.line}”*" if r.line else ""
    fx = f" · ✨ {r.effect.replace('_', ' ')}" if r.effect != "none" else ""
    return f"**{r.title}**: {r.motion} 🎥 *{r.camera}*{fx}{line}"


def face_list(face_all: str | None, face_keep: list | None, n: int) -> list[str]:
    """UI face controls -> Request.face: the radio sets everyone, ticked 'person N' boxes force 'off'
    (keep that reference's own head). The single-person swap uses person 1."""
    base = dict(zip(C.FACE_DRIVE_CHOICES, ("auto", "off", "on"))).get(face_all or "", "auto")
    keep = {int(str(x).split()[-1]) for x in (face_keep or [])}
    return ["off" if i + 1 in keep else base for i in range(max(n, max(keep, default=0)))]


def animate_eta_hint(seconds: float, mode: str, tier: str = tiers.DEFAULT_TIER, allow_heavy: bool = False,
                     resolution: str = "auto") -> str:
    """Live estimate under the length slider (Swap Me In / Copy the Vibe), for the selected tier."""
    from .graphs import plan_animate_segments

    t = tiers.get(tier)
    blocked = tiers.gate(t, "swap" if mode == "replace" else "vibe", allow_heavy=allow_heavy)
    if blocked:
        return f"🔒 {blocked}"
    cap = tiers.animate_cap(t, settings().max_swap_seconds)
    seconds = min(float(seconds or 1), cap)
    frames = max(5, int(seconds * 16))
    segs = plan_animate_segments(frames)
    res_block = tiers.resolution_gate(t, resolution, allow_heavy)
    if res_block:
        return f"🔒 {res_block}"
    eta = estimate_animate_seconds(frames, mode, t.key, resolution)
    chunks = f" in {len(segs)} chunks" if len(segs) > 1 else ""
    warn = " ⚠️ Experimental on Lite: may run out of memory." if t.gated else ""
    if tiers.resolution_warning(t, resolution):
        warn += " ⚠️ 720p on a 16 GB card is experimental (estimate scaled by pixel count, not measured)."
    return (f"⏱️ {seconds:.0f} s clip → {sum(x.windows for x in segs)} Animate window(s){chunks}: "
            f"**{eta_range_text(eta)}** on {t.eta_card} (measured per window on an RTX 5080; long clips are an estimate). Free and local, so reroll away.{warn}")


TIER_CHOICES = [("🪶 Lite · 8-12 GB", "lite"), ("⚡ Standard · 16 GB", "standard"), ("🚀 Max · 24 GB+", "max")]


def tier_info(choice: tiers.TierChoice, selected: str) -> str:
    """Tier badge: what was detected, what the selected tier does."""
    t = tiers.get(selected)
    detected = choice.describe()
    if selected != choice.key:
        detected += f" · you picked **{t.label}**"
    gated = f" 🔒 Needs 16 GB+: {', '.join(sorted({tiers.HEAVY[g].split(' (')[0] for g in t.gated}))}." if t.gated else ""
    warn = f"  \n⚠️ {choice.warning}" if choice.warning else ""
    return f"🖥️ **{detected}**  \n{t.notes}{gated}{warn}"


def step_label(n: int, text: str) -> str:
    """Deadpan numbered label from the v3 mockup: '01 face'."""
    return f"<div class='mc-lab'><b>{n:02d}</b>{html.escape(text)}</div>"


def launch_flags_note(flags: dict | None) -> str:
    """'' when ComfyUI runs with MC's flags (or can't be read), else the status line warning."""
    from .comfy_launch import COMFY_DEFAULT_UPLOAD_MB, DEFAULT_MAX_UPLOAD_MB, DEFAULT_RESERVE_VRAM

    if not flags:
        return ""
    missing = []
    if flags.get("reserve_vram") is None or flags["reserve_vram"] < DEFAULT_RESERVE_VRAM:
        missing.append(C.LAUNCH_FLAG_RESERVE)
    mb = flags.get("max_upload_mb", COMFY_DEFAULT_UPLOAD_MB)
    if mb < DEFAULT_MAX_UPLOAD_MB:
        missing.append(C.LAUNCH_FLAG_UPLOAD.format(mb=mb))
    return C.LAUNCH_FLAGS_WARN.format(missing=", ".join(missing)) if missing else ""


def conn_status(kind: str, s: Any, comfyui_url: str | None = None) -> str:
    """The connection status line (#mc-conn): mock banner, proxy up/down, plus the launch flags warning."""
    if kind == "mock":
        return C.CONN_MOCK
    try:
        info = make_backend("comfy", s.comfy_base_url, s.comfy_api_key).health()  # type: ignore[attr-defined]
    except Exception as exc:
        return C.CONN_DOWN.format(url=s.comfy_base_url, err=type(exc).__name__)
    line = C.CONN_UP.format(url=s.comfy_base_url, upstream=info.get("upstream", "?"))
    import os

    from . import comfy_launch

    try:
        note = launch_flags_note(comfy_launch.running_flags(
            comfyui_url or os.environ.get("COMFYUI_URL") or comfy_launch.DEFAULT_URL, timeout=2.0))
    except Exception:  # noqa: BLE001  (the warning is a nicety; never break the status line)
        note = ""
    return line + (f"  \n{note}" if note else "")


def header_html(tier: str = tiers.DEFAULT_TIER) -> str:
    """Logo + creator credit | gpu tier · local note · dark/light toggle."""
    mark, _ = themes.logo()
    gpu = html.escape(C.GPU_LABEL.format(tier=tiers.get(tier).key))
    jokes = html.escape(json.dumps(C.LEGAL_JOKES), quote=True)
    legal = (f"<span class='mc-legal' data-jokes=\"{jokes}\" title='{html.escape(C.LEGAL_DISCLAIMER, quote=True)}'>"
             f"{html.escape(C.LEGAL_JOKE)}</span>"
             f"<span class='mc-patent' title='{html.escape(C.LEGAL_DISCLAIMER, quote=True)}'>"
             f"{html.escape(C.PATENT_JOKE)}</span>")
    # Credit sits under the wordmark, above the joke legal lines: front-and-center, small.
    return (f"<header class='mc-header'><div class='mc-logo'>{mark}{credit_html()}{legal}"
            f"<span class='mc-sr'>{html.escape(C.EXPANSION_LINE)}</span></div>"
            f"<div class='mc-header-right'><span class='mc-gpu'><span class='mc-dot'></span><b>{gpu}</b></span>"
            f"<span class='mc-local'>{html.escape(C.LOCAL_NOTE)}</span>{tip_html()}{themes.toggle_html()}</div></header>")


def hero_html() -> str:
    return (f"<div class='mc-kicker'>{html.escape(C.KICKER)}</div><div class='mc-rule'></div>"
            f"<h1 class='mc-h1'>{html.escape(C.HEADLINE)}<em>.</em></h1>"
            f"<div class='mc-tagline'>{html.escape(C.APP_TAGLINE)}</div>")


def footer_html() -> str:
    return (f"<div class='mc-footline'><span>{html.escape(C.FOOTER_LEFT)}</span>"
            f"<span>{html.escape(C.FOOTER_RIGHT)}</span></div>"
            f"<div class='mc-fineprint'>{feedback_html()}</div>"
            # always-on use disclaimer (LEGAL_DISCLAIMER stays the header tooltip; tests keep it out of the footer)
            f"<p class='mc-disclaim'>{html.escape(C.USE_DISCLAIMER)}</p>{tip_link_html()}")


def feedback_mailto() -> str:
    """mailto: for the footer 'send feedback' link (subject + body prefilled)."""
    return (f"mailto:{C.FEEDBACK_EMAIL}"
            f"?subject={quote(C.FEEDBACK_SUBJECT)}"
            f"&body={quote(C.FEEDBACK_BODY)}")


def feedback_html() -> str:
    """Deadpan 'send feedback' mailto in the footer."""
    return (f"<a class='mc-feedback' href='{html.escape(feedback_mailto(), quote=True)}'>"
            f"{html.escape(C.FEEDBACK_LABEL)}</a>")


def tip_card_url() -> str:
    """Future 'tip with card' link from ``MC_TIP_CARD_URL``; empty (slot hidden) unless it's an https URL."""
    import os

    url = os.environ.get(C.TIP_CARD_ENV, "").strip()
    return url if url.startswith("https://") else ""


def _tip_qr_svg() -> str:
    try:
        return (Path(__file__).parent / "assets" / "brand" / "xmoney-qr.svg").read_text(encoding="utf-8")
    except OSError:
        return ""


def tip_html() -> str:
    """Header pill 'tip with x money' (plain link, new tab) plus a hidden slot for a future card tip link
    (shown only when MC_TIP_CARD_URL is set). Empty when tips are off (contest build)."""
    if not C.TIPS_ENABLED or not C.TIP_URL:
        return ""
    url = html.escape(C.TIP_URL, quote=True)
    pill = (f"<a class='mc-tip-pill' href='{url}' target='_blank' rel='noopener' "
            f"title='{html.escape(C.TIP_QUOTE, quote=True)}' aria-label='{html.escape(C.TIP_TITLE, quote=True)}'>"
            f"<span class='mc-tip-mark' aria-hidden='true'>$</span>{html.escape(C.TIP_LABEL)}</a>")
    return f"<span class='mc-tip'>{pill}{tip_card_html()}</span>"


def tip_card_html() -> str:
    card_url = tip_card_url()
    if card_url:
        return (f"<a class='mc-tip-card' data-slot='tip-card' href='{html.escape(card_url, quote=True)}' "
                f"target='_blank' rel='noopener'>{html.escape(C.TIP_CARD_LABEL)}</a>")
    return "<span class='mc-tip-card' data-slot='tip-card' hidden aria-hidden='true'></span>"


def tip_link_html() -> str:
    """Footer tip area. Desktop (wide + fine pointer, CSS): the X Money QR is always visible on a white card
    with 'Scan to tip @Tony_Omega', the quote under it and a text link as fallback. Mobile / touch: no QR,
    the link becomes a tap pill, plus the quote. The optional card tip (MC_TIP_CARD_URL) sits under the quote, so the
    quote stays right with the X Money tip. Pure CSS media queries: no popovers, no layout shift.
    Empty when tips are off (contest build)."""
    if not C.TIPS_ENABLED or not C.TIP_URL:
        return ""
    url = html.escape(C.TIP_URL, quote=True)
    svg = _tip_qr_svg()
    # A data-URI <img> (not inline <svg>) so no HTML sanitizer can strip it.
    import base64

    src = "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")
    qr = (f"<figure class='mc-tip-qrbox'><span class='mc-tip-qrcard'><img src='{src}' width='112' height='112' "
          f"alt='QR code: tip @Tony_Omega with X Money'></span>"
          f"<figcaption>{html.escape(C.TIP_QR_CAPTION)}</figcaption></figure>") if svg else ""
    # ONE wrapper (.mc-tip-unit) holds QR + link + quote, so the quote never drifts away from the tip.
    return (f"<div class='mc-tip-foot mc-tip-unit'>{qr}<div class='mc-tip-foot-text'>"
            f"<a class='mc-tip-link' href='{url}' target='_blank' rel='noopener'>"
            f"<span class='mc-tip-mark' aria-hidden='true'>$</span>{html.escape(C.TIP_LABEL)}</a>"
            f"<i class='mc-tip-quote'>{html.escape(C.TIP_QUOTE)}</i>{tip_card_html()}</div></div>")


def credit_html() -> str:
    """'made by Oroku Saki · @Tony_Omega' (header, under the logo)."""
    return (f"<span class='mc-credit'>{html.escape(C.MADE_BY)} {html.escape(C.CREATOR_NAME)} · "
            f"<a href='{html.escape(C.CREATOR_URL, quote=True)}' target='_blank' rel='noopener'>"
            f"{html.escape(C.CREATOR_HANDLE)}</a></span>")


def credit_head() -> str:
    """Creator meta for link previews + a tiny schema.org JSON-LD (local app, so nothing more)."""
    ld = {"@context": "https://schema.org", "@type": "SoftwareApplication", "name": C.APP_NAME,
          "applicationCategory": "MultimediaApplication", "license": "https://www.apache.org/licenses/LICENSE-2.0",
          "author": {"@type": "Person", "name": C.CREATOR_NAME, "alternateName": C.CREATOR_HANDLE,
                     "sameAs": C.CREATOR_URL}}
    return (f'\n<meta name="author" content="{html.escape(C.CREATOR_NAME, quote=True)}">'
            f'\n<meta name="twitter:creator" content="{html.escape(C.CREATOR_HANDLE, quote=True)}">'
            f'\n<script type="application/ld+json">{json.dumps(ld)}</script>'
            # Gradio's own page template ships twitter:creator=@Gradio/@teamGradio; point those at the creator too
            f"\n<script>document.querySelectorAll('meta[name=\"twitter:creator\"]').forEach(function(m){{"
            f"m.setAttribute('content', {json.dumps(C.CREATOR_HANDLE)});}});</script>\n")


def more_modes_gate(tier: str, allow_heavy: bool) -> str:
    msg = tiers.gate(tier, "talk", allow_heavy=allow_heavy)
    return f"🔒 {msg}" if msg else ""


def save_format_from(name: str, emoji: str, work: str | None, key: str, top: str, bottom: str, scene: str,
                     motion: str) -> tuple[str, str]:
    """Save as format: custom format > the finished meme's recipe > the selected format. Returns (key, message)."""
    import json

    caps = [(top or "", bottom or "")] if (top or bottom) else []
    if key == "custom":
        if not (scene or "").strip() or not (motion or "").strip():
            raise ValueError("Describe the custom scene and what you do (More options) first.")
        base: Any = memes.custom_recipe(scene, motion, top or "", bottom or "")
        source = "custom"
    else:
        base = memes.RECIPES.get(key)
        source = f"format:{key}"
        from .pipeline import project_file

        proj = project_file(work) if work else None
        if proj and proj.exists():
            rec = json.loads(proj.read_text(encoding="utf-8"))
            if rec.get("recipe"):
                base, source = rec["recipe"], f"meme:{Path(work).name}"
                c = rec.get("captions") or {}
                if c.get("top") or c.get("bottom"):
                    caps = [(c.get("top", ""), c.get("bottom", "")), *caps]
        if base is None:
            raise ValueError("Pick a format first.")
    new_key, path = memes.save_format(base, name, emoji=emoji, captions=caps, source=source)
    return new_key, f"💾 Saved **{memes.RECIPES[new_key].title}** as a format (`{path.name}`). It's in the grid now."


def _fmt_left(seconds: float) -> str:
    seconds = max(1, int(round(seconds)))
    m, sec = divmod(seconds, 60)
    return f"about {m} min {sec:02d} s left" if m else f"about {sec} s left"


def _file_url(path: str) -> str:
    from urllib.parse import quote

    return "/gradio_api/file=" + quote(str(Path(path).resolve()).replace("\\", "/"))


def share_html(data: dict[str, Any], final: str) -> str:
    """Group-chat-first share bar: copy/share, MP4, GIF, PNG, then post helpers."""
    from urllib.parse import quote

    mp4, gif, still = data.get("mp4"), data.get("gif"), data.get("still")
    main = mp4 or still or final
    mime = "video/mp4" if mp4 else "image/png"
    btns = [
        f"<button class='mc-share-btn mc-copy' onclick=\"mcShare(this)\" data-url='{_file_url(main)}' data-mime='{mime}' "
        f"data-name='{Path(main).name}'>📋 {C.SHARE_COPY}</button>"
    ]
    if mp4:
        btns.append(f"<a class='mc-share-btn' download href='{_file_url(mp4)}'>⬇️ {C.SHARE_MP4}</a>")
    if gif:
        btns.append(f"<a class='mc-share-btn' download href='{_file_url(gif)}'>🎞️ {C.SHARE_GIF}</a>")
    if still:
        btns.append(f"<a class='mc-share-btn' download href='{_file_url(still)}'>🖼️ {C.SHARE_PNG}</a>")
    text = data.get("post_text", "")
    if text:
        btns.append(f"<a class='mc-share-btn mc-post' target='_blank' rel='noopener' "
                    f"href='https://x.com/intent/post?text={quote(text)}'>𝕏 {C.SHARE_X}</a>")
        btns.append(f"<a class='mc-share-btn mc-post' target='_blank' rel='noopener' "
                    f"href='https://www.reddit.com/submit?title={quote(text[:280])}'>👽 {C.SHARE_REDDIT}</a>")
    return f"<div class='mc-share'>{''.join(btns)}<div class='mc-share-msg'></div></div>"


# Client-side share helper: Web Share API with the file (Windows/Android/iOS share sheet -> Discord, WhatsApp...),
# else copy the image to the clipboard (browsers can't put video files on the clipboard).
SHARE_JS = """
<script>
async function mcShare(btn){
  const msg = btn.parentElement.querySelector('.mc-share-msg');
  try {
    const r = await fetch(btn.dataset.url); const blob = await r.blob();
    const file = new File([blob], btn.dataset.name, {type: btn.dataset.mime});
    if (navigator.canShare && navigator.canShare({files:[file]})) { await navigator.share({files:[file]}); msg.textContent='Sent to the share sheet ✅'; return; }
    if (btn.dataset.mime === 'image/png' && window.ClipboardItem) { await navigator.clipboard.write([new ClipboardItem({'image/png': blob})]); msg.textContent='Copied! Paste it in the group chat ✅'; return; }
    msg.textContent='This browser cannot copy videos: grab the MP4 or GIF button and drag it into the chat.';
  } catch(e) { msg.textContent='Could not share (' + e.name + '): use the download buttons.'; }
}
</script>
"""

OUT_KEYS = ["stages", "avatar", "audio", "video", "still", "download", "status", "share", "post", "animate", "scene", "work"]


def _stream(events: Iterator, mode: str = "meme") -> Iterator[tuple]:
    """Turn pipeline events into Gradio updates (order: OUT_KEYS)."""
    import time

    state: dict[str, dict[str, Any]] = {}
    o: dict[str, Any] = {k: gr.update() for k in OUT_KEYS}
    o.update(avatar=None, audio=None, download=None, share="", post="",
             animate=gr.update(visible=False))
    status = "Starting…"
    timers: dict[str, tuple[float, float]] = {}  # stage -> (t0, eta)
    notes: list[str] = []  # heads-ups that link the useful guide stay under the status for the whole run
    events = in_gpu_lane(events, mode)

    def emit():
        pinned = "".join(f"\n\n> {n}" for n in notes)
        o["stages"], o["status"] = _stages_html(state, mode), status + pinned
        return tuple(o[k] for k in OUT_KEYS)

    try:
        for ev in events:
            if "#guide-" in (ev.message or "") and uibits.link_guide(ev.message) not in notes:
                notes.append(uibits.link_guide(ev.message))
            if ev.stage in STAGES:
                cur = state.setdefault(ev.stage, {"status": "pending", "fraction": 0.0, "message": ""})
                cur["status"] = ev.status
                if ev.fraction >= 0:
                    cur["fraction"] = ev.fraction if ev.status != "start" else 0.0
                if ev.status in ("done", "skipped"):
                    cur["fraction"] = 1.0
                cur["message"] = ev.message
                if ev.status == "start" and ev.data.get("eta_seconds"):
                    timers[ev.stage] = (time.monotonic(), float(ev.data["eta_seconds"]))
                if ev.artifact:
                    if ev.stage == "avatar":
                        o["avatar"] = str(ev.artifact)
                    elif ev.stage in ("voice", "revoice"):
                        o["audio"] = str(ev.artifact)
                    elif ev.stage in ("lipsync", "smooth", "motion", "swap", "vibe"):
                        o["video"] = gr.update(value=str(ev.artifact), visible=True)
                        o["still"] = gr.update(visible=False)
            if "batch_index" in ev.data:
                status = f"Clip {ev.data['batch_index'] + 1} of {ev.data['batch_total']}: {ev.message}"
                if ev.stage == "prepare" and ev.status == "start":
                    state = {}
            else:
                status = ev.message
            if ev.status in ("start", "progress") and ev.stage in STAGES:
                if ev.stage in timers:
                    t0, eta = timers[ev.stage]
                    el = time.monotonic() - t0
                    f = state[ev.stage]["fraction"]
                    left = eta - el if f <= 0.02 else 0.5 * (eta - el) + 0.5 * el * (1 - f) / f
                    status += f" · ⏱️ {_fmt_left(max(5.0, left))}"
                status += f"  \n<span class='mc-joke'>{html.escape(C.loading_joke(ev.stage))}…</span>"
            if ev.stage == "batch_item":
                o["video"] = gr.update(value=str(ev.artifact), visible=True)
                o["download"] = str(ev.artifact)
            if ev.status == "complete":
                final = str(ev.artifact)
                clips = ev.data.get("clips")
                if final.endswith(".png"):
                    o["still"] = gr.update(value=final, visible=True)
                    o["video"] = gr.update(value=None, visible=False)
                    o["download"] = final
                    o["scene"] = ev.data.get("scene")
                    eta = ev.data.get("animate_eta_seconds", 210)
                    o["animate"] = gr.update(visible=True, value=f"{C.ANIMATE_CTA} ({eta_range_text(eta)})")
                    status = f"{C.STILL_DONE}  \n{C.ANIMATE_HINT}"
                else:
                    o["video"] = gr.update(value=final, visible=True)
                    o["still"] = gr.update(visible=False)
                    o["download"] = final
                    o["animate"] = gr.update(visible=False)
                    status = f"{C.done_line()} {ev.message}. {C.COMPLETE_HINT}  \nSaved in `{ev.data.get('work_dir')}`"
                if clips:
                    status += "\n\n" + "\n".join(f"- `{c}`" for c in clips)
                    o["download"] = clips
                o["work"] = ev.data.get("work_dir")
                o["share"] = share_html(ev.data, final)
                o["post"] = ev.data.get("post_text", "")
            yield emit()
    except InputError as exc:
        status = f"⚠️ {exc}"
        yield emit()
    except Exception as exc:  # show backend errors instead of a stack trace: the fix first, the joke never
        for st, s_ in state.items():
            if s_["status"] in ("start", "progress"):
                s_["status"] = "error"
        status = friendly_error(exc)
        yield emit()


def _common(adv: dict) -> dict:
    return dict(
        style=adv["style"], style_extra=adv["style_extra"] or "", captions=adv["captions"], smooth=adv["smooth"],
        hd=adv["hd"], watermark=adv["watermark"], seed=int(adv["seed"]), tier=adv["tier"] or tiers.DEFAULT_TIER,
        allow_heavy=bool(adv["heavy"]),
        seam=adv["seam"], segment_seconds=float(adv["segment"]) or None, subject=adv.get("subject") or "auto",
    )


def subject_radio() -> gr.Radio:
    """Selfie vs avatar: avatar mode keeps a cartoon / anime / 3D / VTuber / pixel-art / mascot look as drawn."""
    return gr.Radio(C.SUBJECT_CHOICES, value="auto", label=C.SUBJECT_LABEL, elem_classes="mc-subject")


def clip_audio_controls() -> tuple[gr.Radio, gr.Audio]:
    radio = gr.Radio(C.CLIP_AUDIO_CHOICES, value="original", label=C.CLIP_AUDIO_LABEL, elem_classes="mc-clip-audio")
    voice = gr.Audio(type="filepath", label="Your voice sample (10 to 30 s) for re-voicing", visible=False,
                     sources=["upload", "microphone"])
    return radio, voice


def models_report(tier: str) -> str:
    """'check my models' button: ask ComfyUI what's installed and list what this tier still needs."""
    from .doctor import live_report

    try:
        return live_report(tier)
    except Exception as exc:
        return friendly_error(exc, tier)


def library_dir() -> Path:
    """The user's own files (fetched clips land in ``library/clips``). Same folder the agent connector uses."""
    import os

    from .config import ROOT

    return Path(os.environ.get("MC_LIBRARY_DIR") or (ROOT / "library")).resolve()


def fetch_clip_ui(url: str, consent: bool, progress: Any = None) -> tuple[Any, str]:
    """Bring your own clip: fetch the sharpest copy at ``url`` into library/clips and load it into the clip box."""
    from . import clipfetch

    url = (url or "").strip()
    if not url:
        return gr.update(), ""
    if not consent:
        return gr.update(), C.CLIP_LINK_NEEDS_RIGHTS
    try:
        def prog(frac: float, msg: str) -> None:
            if progress is not None:
                progress(frac, desc=msg)

        got = clipfetch.fetch(url, library_dir() / "clips", progress=prog)
    except Exception as exc:  # noqa: BLE001 (shown to the user, friendly)
        return gr.update(), f"couldn't fetch it: {exc}"
    path = got["path"]
    try:
        seconds = media.media_duration(path)
    except Exception:  # noqa: BLE001
        seconds = 0.0
    shown = path.relative_to(library_dir().parent) if path.is_relative_to(library_dir().parent) else path
    msg = C.CLIP_LINK_DONE.format(quality=got["quality"], seconds=seconds, path=shown.as_posix())
    if got.get("height") and got["height"] < 1000:
        msg += C.CLIP_LINK_LOW.format(height=got["height"])
    return str(path), msg


def _mini_md(text: str) -> str:
    """Just enough markdown for the tour cards: paragraphs, **bold**, *italic*, `code`, [links](https://...)."""
    import re

    out = []
    for para in text.split("\n\n"):
        t = html.escape(para)
        t = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r"<a href='\2' target='_blank' rel='noopener'>\1</a>", t)
        t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
        t = re.sub(r"\*(.+?)\*", r"<i>\1</i>", t)
        t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
        out.append(f"<p>{t}</p>")
    return "".join(out)


def tour_html(i: int) -> str:
    i = max(0, min(i, len(C.TOUR_STEPS) - 1))
    title, body = C.TOUR_STEPS[i]
    dots = "".join(f"<span class='mc-tour-dot{' on' if k == i else ''}'></span>" for k in range(len(C.TOUR_STEPS)))
    return (f"<div class='mc-tour-card'><div class='mc-tour-kicker'>{html.escape(C.TOUR_TITLE)} · "
            f"{i + 1}/{len(C.TOUR_STEPS)}</div><h3 class='mc-tour-title'>{html.escape(title)}<span "
            f"class='mc-red'>.</span></h3><div class='mc-tour-body'>{_mini_md(body)}</div>"
            f"<div class='mc-tour-dots'>{dots}</div></div>")


# first visit: show the tour; once skipped or finished, this browser remembers (localStorage), like the theme switch.
TOUR_HEAD = """<script>(function(){try{if(localStorage.getItem('mc_tour_done')==='1'){document.documentElement.classList.add('mc-tour-done');}}catch(e){}})();</script>"""
TOUR_DONE_JS = "(i) => { try { localStorage.setItem('mc_tour_done', '1'); } catch (e) {} " \
               "document.documentElement.classList.add('mc-tour-done'); return i; }"
TOUR_OPEN_JS = "(i) => { document.documentElement.classList.remove('mc-tour-done'); " \
               "window.scrollTo({top: 0, behavior: 'smooth'}); return i; }"



def pending_take(work: Any) -> Path | None:
    """The deferred-upscale record of a run folder (Request.defer_upscale), or None."""
    from .pipeline import PENDING_FILE, WORK_SUBDIR

    if not work:
        return None
    top = Path(str(work))
    return next((p for p in (top / WORK_SUBDIR / PENDING_FILE, top / PENDING_FILE) if p.is_file()), None)


def prepared_clip_of(work: Any) -> Path | None:
    """The run's prepared input clip (the 'before' of the before/after wipe), or None."""
    from .pipeline import WORK_SUBDIR

    if not work:
        return None
    p = Path(str(work)) / WORK_SUBDIR / "inputs" / "clip.mp4"
    return p if p.is_file() else None


def after_video_run(work: Any, video: Any) -> tuple[Any, Any]:
    """After a swap / vibe run: show 'make it HD' for a deferred take, 'before / after' for a finished one."""
    pending = pending_take(work) is not None
    return (gr.update(visible=pending),
            gr.update(visible=bool(video) and not pending and prepared_clip_of(work) is not None))


def wipe_for(work: Any, video: Any) -> tuple[Any, str]:
    """Before/after wipe of the shown result (media.wipe_compare), saved next to it."""
    before = prepared_clip_of(work)
    if not before or not video:
        return gr.update(), "⚠️ Nothing to compare yet: run a swap first."
    dest = Path(str(work)) / f"{Path(str(video)).stem}_before_after.mp4"
    try:
        media.wipe_compare(before, video, dest)
    except Exception as exc:  # noqa: BLE001 - shown, never a crash
        return gr.update(), f"⚠️ Couldn't make the before/after ({str(exc)[:160]})"
    return str(dest), f"↔️ Before/after saved: `{dest.name}`"


def swap_button_label(length: float, tier: str, allow: bool, res: str, n_people: int = 1, mode: str = "replace",
                      cta: str | None = None) -> str:
    """'swap me in · $0 · ~N min' (one Animate pass per swapped person; SeedVR2 not counted)."""
    cta = cta or (C.SWAP_CTA if mode == "replace" else C.VIBE_CTA)
    t = tiers.get(tier)
    if tiers.gate(t, "swap" if mode == "replace" else "vibe", allow_heavy=bool(allow)):
        return uibits.money_time_label(cta, None)
    cap = tiers.animate_cap(t, settings().max_swap_seconds if mode == "replace" else settings().max_vibe_seconds)
    frames = max(5, int(min(float(length or 1), cap) * 16))
    try:
        eta = estimate_animate_seconds(frames, mode, t.key, res or "auto") * max(1, int(n_people or 1))
    except Exception:  # noqa: BLE001
        eta = None
    return uibits.money_time_label(cta, eta)


def build_ui(backend_kind: str | None = None) -> gr.Blocks:
    s = settings()
    kind = backend_kind or s.backend
    choice = tiers.resolve()
    t0 = choice.tier
    swap_cap = tiers.animate_cap(t0, s.max_swap_seconds)
    vibe_cap = tiers.animate_cap(t0, s.max_vibe_seconds)

    running = _RUNNING

    def pipeline() -> Pipeline:
        p = Pipeline(make_backend(kind, s.comfy_base_url, s.comfy_api_key), s)
        running["p"] = p  # the stop button cancels whatever started last
        return p

    def health() -> str:
        return conn_status(kind, s)

    theme, css, head = themes.load()

    def consent_box() -> gr.Checkbox:
        # the short rights line is always visible right above the (still required) consent box
        gr.HTML(uibits.rights_html(C.RIGHTS_SHORT), elem_classes="mc-rights-wrap")
        return gr.Checkbox(label="✅ " + C.CONSENT, value=False, elem_classes="mc-consent")

    css = (css or "") + uibits.GUIDE_CSS + uibits.WHO_CSS + uibits.PREVIEW_CSS
    with gr.Blocks(title=C.APP_NAME, css=css, theme=theme,
                   head=head + SHARE_JS + credit_head() + TOUR_HEAD + uibits.GUIDE_HEAD + uibits.APP_HEAD,
                   elem_classes="mc-root") as demo:
        header = gr.HTML(header_html(t0.key), elem_id="mc-header")
        gr.HTML(hero_html(), elem_classes="mc-hero")
        with gr.Row(elem_classes="mc-statusbar"):
            if kind != "mock":  # mock: the privacy facts live in the one mock line (#mc-conn, C.CONN_MOCK)
                gr.Markdown(privacy_badge(), elem_classes="mc-privacy mc-chip")
            # mock: the one mock line is there from the first paint (no network call); comfy fills it on load
            conn = gr.Markdown(C.CONN_MOCK if kind == "mock" else "", elem_classes="mc-conn mc-chip", elem_id="mc-conn")
            with gr.Row(elem_classes="mc-help-row"):
                help_btn = gr.Button(C.HELP_LINK, size="sm", elem_classes="mc-help", elem_id="mc-help-tour")
                guide_btn = gr.Button(uibits.GUIDE_LABEL, size="sm", elem_classes="mc-help mc-guide-btn", elem_id="mc-guide-open")
        with gr.Column(elem_id="mc-guide", elem_classes="mc-guide"):  # slide-in panel (docs/GUIDE.md)
            guide_close = gr.Button("close ✕", size="sm", elem_classes="mc-guide-close", elem_id="mc-guide-close")
            gr.HTML(uibits.guide_html())
        tour_i = gr.Number(0, visible=False, precision=0)  # client-side, so the js hooks can read the step
        with gr.Column(elem_id="mc-tour", elem_classes="mc-tour") as tour_box:  # floating coach card
            tour_md = gr.HTML(tour_html(0))
            with gr.Row():
                tour_skip = gr.Button(C.TOUR_SKIP, size="sm", elem_classes="mc-tour-skip", elem_id="mc-tour-skip")
                tour_back = gr.Button(C.TOUR_BACK, size="sm", elem_classes="mc-tour-back", elem_id="mc-tour-back")
                tour_next = gr.Button(C.TOUR_NEXT, size="sm", variant="primary", elem_classes="mc-tour-next", elem_id="mc-tour-next")
        with gr.Accordion(C.SETUP_ACC, open=False, elem_classes="mc-more mc-setup-acc"):
            with gr.Row(elem_classes="mc-tier"):
                tier_md = gr.Markdown(tier_info(choice, choice.key), elem_classes="mc-tier-info")
                with gr.Column(scale=0, min_width=470):
                    tier_radio = gr.Radio(TIER_CHOICES, value=choice.key, label="Hardware tier", elem_classes="mc-tier-toggle", elem_id="mc-tier")
                    heavy = gr.Checkbox(value=False, label="Try 16 GB modes anyway (experimental)", visible=bool(t0.gated),
                                        elem_classes="mc-tier-heavy", elem_id="mc-tier-heavy")
            gr.Markdown(C.AGENT_HINT, elem_classes="mc-note mc-agent-hint")
        demo.load(health, outputs=conn)

        with gr.Row(elem_classes="mc-main"):
            with gr.Column(scale=9, elem_classes="mc-work"):
                with gr.Tabs(elem_classes="mc-modes"):
                    # ---------------- headline: Meme Recreate ----------------
                    with gr.Tab(C.TAB_MEME, id="meme"):
                        with gr.Row():
                            with gr.Column(scale=4, elem_classes="mc-steps"):
                                gr.HTML(step_label(1, C.STEP_FACE), elem_classes="mc-step-title")
                                # image_mode=None hands MC the original file (alpha kept): gradio 5 crashes turning a .jpg into RGBA
                                m_photo = gr.Image(type="filepath", label=C.FACE_LABEL, height=220, elem_classes="mc-frame",
                                                   image_mode=None, elem_id="mc-meme-photo")
                                m_subject = subject_radio()
                                m_refq = gr.Markdown("", elem_classes="mc-note mc-ref-check")
                                gr.HTML(uibits.info_card(C.SUBJECT_CARD, C.SUBJECT_NOTE))
                            with gr.Column(scale=5, elem_classes="mc-formats"):
                                gr.HTML(step_label(2, C.STEP_FORMAT.format(n=len(memes.RECIPES))), elem_classes="mc-step-title")
                                m_recipe = gr.Radio(_recipe_choices(), value=memes.DEFAULT_RECIPE, show_label=False,
                                                    elem_classes="mc-format-grid", elem_id="mc-meme-format")
                                m_card = gr.Markdown(_recipe_card(memes.DEFAULT_RECIPE), elem_classes="mc-format-card")
                                surprise = gr.Button(C.SURPRISE, size="sm", elem_classes="mc-surprise")
                                gr.HTML(uibits.info_card(C.LIBRARY_CARD, C.LIBRARY_NOTE))
                        # 03 caption + the big button: full width under face | format, so phones see 01, 02, 03 in order
                        gr.HTML(step_label(3, C.STEP_CAPTIONS), elem_classes="mc-step-title")
                        d0 = memes.RECIPES[memes.DEFAULT_RECIPE]
                        with gr.Row(elem_classes="mc-caption-row"):
                            m_top = gr.Textbox(label="Top text", value=d0.top)
                            m_bottom = gr.Textbox(label="Bottom text", value=d0.bottom)
                        cap_reroll = gr.Button(C.CAPTION_REROLL, size="sm", elem_classes="mc-caption-reroll")
                        m_consent = consent_box()
                        go_meme = gr.Button(C.PRIMARY_CTA, variant="primary", size="lg", elem_classes="mc-primary", elem_id="mc-go-meme")
                        with gr.Accordion(C.MORE_OPTIONS, open=False, elem_classes="mc-more"):
                            style = gr.Radio(_style_choices(), value="none", label="Look")
                            style_extra = gr.Textbox(label="Extra look details", placeholder="e.g. wearing a tuxedo, neon lighting")
                            with gr.Row():
                                m_length = gr.Radio([("~3 s", 49), ("~5 s", 81)], value=81, label="Length")
                                m_era = gr.Radio([(v, k) for k, v in ERAS.items()], value="classic", label="Meme Era")
                            m_straight = gr.Checkbox(label=C.STRAIGHT_TO_VIDEO, value=False)
                            m_effect = gr.Radio([("Format default", "default"), ("None", "none"), ("Crash zoom", "punch_in"),
                                                 ("Shake", "shake"), ("Boomerang", "boomerang")], value="default", label="Effect")
                            m_say = gr.Checkbox(label="🗣️ Say a line (lip-synced, in your voice) instead of a silent reaction", value=False)
                            m_line = gr.Textbox(label="Line (empty = the format's line)", max_lines=2)
                            m_voice_mode = gr.Radio([("Clone my voice", "clone"), ("Built-in voice", "builtin"), ("My own recording", "my_recording")], value="clone", label="Voice")
                            m_voice = gr.Audio(type="filepath", label="Voice sample (10 to 30 s) or your recording of the line", sources=["upload", "microphone"])
                            gr.Markdown("**✍️ Custom format** (pick *Custom format* above). Original ideas only, you as the star.")
                            m_scene = gr.Textbox(label="Scene", placeholder="sitting in a car at a drive-through, holding a giant soda")
                            m_motion = gr.Textbox(label="What you do", placeholder="slowly lowers sunglasses and stares into the camera")
                    # ---------------- Swap Me In ----------------
                    with gr.Tab(C.TAB_SWAP, id="swap"):
                        gr.HTML(uibits.lead_html(C.SWAP_LEAD))
                        with gr.Row(elem_classes="mc-cards"):
                            gr.HTML(uibits.info_card(C.SWAP_INFO, C.SWAP_INTRO), elem_classes="mc-intro")
                            gr.HTML(uibits.info_card(C.RULES_CARD, C.RIGHTS_NOTE + "\n\n" + C.CLIP_TIP, "mc-info-warn"), elem_classes="mc-warn-card mc-clip-tip")
                        gr.HTML(step_label(1, C.STEP_CLIP), elem_classes="mc-step-title")
                        with gr.Row(elem_classes="mc-clip-link"):
                            w_link = gr.Textbox(label=C.CLIP_LINK_LABEL, placeholder=C.CLIP_LINK_PLACEHOLDER, scale=4,
                                                 info=C.CLIP_LINK_NOTE, elem_id="mc-swap-link")
                            w_fetch = gr.Button(C.CLIP_LINK_BTN, scale=1, elem_classes="mc-surprise", elem_id="mc-swap-fetch")
                        w_link_md = gr.Markdown("", elem_classes="mc-note")
                        with gr.Row():
                            w_clip = gr.Video(label=f"① Your clip (up to {s.max_swap_seconds * 2:.0f} s on Max, {s.max_swap_seconds:.0f} s on Standard)", height=240, elem_id="mc-swap-clip")
                            with gr.Column():
                                gr.HTML(step_label(2, C.STEP_YOU), elem_classes="mc-step-title")
                                w_photo = gr.Image(type="filepath", label="② Your selfie or avatar (upper body works best)", height=240,
                                                   elem_classes="mc-frame", image_mode=None, elem_id="mc-swap-photo")
                                w_subject = subject_radio()
                                gr.HTML(uibits.info_card(C.REF_CARD, C.REF_TIP), elem_classes="mc-ref-tip")
                                w_refq = gr.Markdown("", elem_classes="mc-note mc-ref-check", elem_id="mc-swap-refcheck")
                        gr.HTML(step_label(3, C.STEP_EXTRAS), elem_classes="mc-step-title")
                        with gr.Accordion(C.PEOPLE_TITLE, open=False, elem_classes="mc-more mc-people"):
                            gr.Markdown(C.PEOPLE_NOTE, elem_classes="mc-note")
                            w_find = gr.Button(C.PEOPLE_FIND, size="sm", elem_classes="mc-surprise", elem_id="mc-people-find")
                            w_people_md = gr.Markdown("", elem_classes="mc-note")
                            w_people_img = gr.Image(label="who's who", height=260, interactive=False, visible=False,
                                                    elem_classes="mc-frame", elem_id="mc-people-img")
                            w_count = gr.State(0)
                            w_slots = []
                            for row in range(0, tiers.PEOPLE_HARD_CAP, 4):  # 8 slots in rows of 4
                                with gr.Row():
                                    w_slots += [gr.Image(type="filepath", label=f"person {i + 1}: selfie / avatar (empty = unchanged)",
                                                         height=160, visible=False, elem_classes="mc-frame", image_mode=None)
                                                for i in range(row, min(row + 4, tiers.PEOPLE_HARD_CAP))]
                            w_slots_refq = gr.Markdown("", elem_classes="mc-note mc-ref-check")
                            w_moves, w_descs = [], []
                            for row in range(0, tiers.PEOPLE_HARD_CAP, 4):
                                with gr.Row():
                                    for i in range(row, min(row + 4, tiers.PEOPLE_HARD_CAP)):
                                        with gr.Column(min_width=160):
                                            w_moves.append(gr.Dropdown(uibits.driver_choices(tiers.PEOPLE_HARD_CAP, i),
                                                                       value="", label=f"person {i + 1}: moves",
                                                                       visible=False, elem_classes="mc-moves"))
                                            w_descs.append(gr.Textbox(label=f"person {i + 1} looks like (optional)",
                                                                      placeholder="lavender bob, red bomber jacket",
                                                                      max_lines=1, visible=False))
                            w_who = gr.HTML("", elem_classes="mc-who-wrap", elem_id="mc-who")
                        w_prompt = gr.Textbox(label="✨ Tweak the scene / style (optional)",
                                              placeholder="warm golden-hour light, film grain, slow-motion feel")
                        w_audio, w_voice = clip_audio_controls()
                        gr.HTML(step_label(4, C.STEP_GO), elem_classes="mc-step-title")
                        w_eta = gr.Markdown(animate_eta_hint(5, "replace", t0.key), elem_classes="mc-note mc-eta", elem_id="mc-swap-eta")
                        w_consent = consent_box()
                        go_swap = gr.Button(swap_button_label(5, t0.key, False, "auto"), variant="primary", size="lg",
                                            elem_classes="mc-primary", elem_id="mc-go-swap")
                        with gr.Accordion("More options", open=False, elem_classes="mc-more"):
                            gr.Markdown("MC finds the person automatically (SAM 3.1), copies their moves and expressions (SDPose) and re-renders the shot with **you** (or your avatar, in its own art style) in it (Wan2.2 Animate). The clip's original audio is kept, in sync, unless you mute or re-voice it. *Needs the Swap model pack: see Setup.*")
                            with gr.Row():
                                w_start = gr.Number(value=0, label="Start at (s)")
                                w_len = gr.Slider(1, swap_cap, value=5, step=0.5, label="Length (s)", elem_id="mc-swap-len")
                            gr.Markdown("The background comes from your clip; the tweak changes light, mood and look. "
                                        "For a whole new scene, use **Copy the Vibe**. For a new outfit, use *Extra look "
                                        "details* (Meme tab → More options).")
                            with gr.Row():
                                w_top = gr.Textbox(label="Top text")
                                w_bottom = gr.Textbox(label="Bottom text")
                            w_relight = gr.Radio(C.RELIGHT_CHOICES, value=C.RELIGHT_CHOICES[0], label=C.RELIGHT_LABEL)
                            w_props = gr.Checkbox(value=False, label=C.REMOVE_PROPS_LABEL)
                            w_erase = gr.Textbox(label=C.ERASE_PROPS_LABEL, placeholder=C.ERASE_PROPS_PLACEHOLDER,
                                                 info=C.ERASE_PROPS_NOTE)
                            w_res = gr.Radio(C.RESOLUTION_CHOICES, value="auto", label=C.RESOLUTION_LABEL,
                                             info=C.RESOLUTION_NOTE, elem_id="mc-swap-res")
                            w_face = gr.Radio(list(C.FACE_DRIVE_CHOICES), value=C.FACE_DRIVE_CHOICES[0], label=C.FACE_DRIVE_LABEL,
                                              info=C.FACE_DRIVE_NOTE)
                            w_face_keep = gr.CheckboxGroup([f"person {i + 1}" for i in range(tiers.PEOPLE_HARD_CAP)],
                                                           value=[], label=C.FACE_DRIVE_KEEP_LABEL)
                            w_defer = gr.Checkbox(value=False, label="⚡ preview first, HD later (skip the HD pass; "
                                                  "hit 'make it HD' on the takes you like)", elem_id="mc-swap-defer")
                            w_engine = gr.Radio([("auto", "auto"), ("all at once (SCAIL-2)", "scail"),
                                                 ("one at a time", "chain")], value="auto", label="people engine",
                                                info="one at a time holds drawn styles better and splits at cuts", elem_id="mc-swap-engine")
                            w_split = gr.Radio([("auto", "auto"), ("on", "on"), ("off", "off")], value="auto",
                                               label="split at cuts", info="swap each shot on its own (see the guide: cuts)", elem_id="mc-swap-split")
                            w_stylelock = gr.Radio([("off", "off"), ("drawn refs only", "auto"), ("on", "on")],
                                                   value=stylelock.DEFAULT_MODE, label="style lock · pending a real test",
                                                   info="keeps drawn characters drawn: relight off, anti-photo negative, "
                                                        "lighter HD pass, tighter edge. Not A/B tested yet, so off by default.", elem_id="mc-stylelock")
                            with gr.Row():
                                w_closeup = gr.Checkbox(value=False, label="close-ups: crop a full-body ref to the face "
                                                        "(pending a real test)", elem_id="mc-closeup")
                                w_chunks = gr.Checkbox(value=False, label="long group clips: chained SCAIL-2 chunks "
                                                       "(experimental)", elem_id="mc-chunks")
                            w_rotate = gr.Radio([("as filmed", 0), ("90°", 90), ("180°", 180), ("270°", 270)], value=0,
                                                label="rotate clip", info="phone clips turn upright on their own; use "
                                                                          "this only if yours comes out sideways")
                    # ---------------- Copy the Vibe ----------------
                    with gr.Tab(C.TAB_VIBE, id="vibe"):
                        gr.HTML(uibits.lead_html(C.VIBE_LEAD))
                        with gr.Row(elem_classes="mc-cards"):
                            gr.HTML(uibits.info_card(C.RULES_CARD, C.RIGHTS_NOTE + "\n\n" + C.CLIP_TIP, "mc-info-warn"), elem_classes="mc-warn-card mc-clip-tip")
                        gr.HTML(step_label(1, C.STEP_CLIP), elem_classes="mc-step-title")
                        with gr.Row(elem_classes="mc-clip-link"):
                            v_link = gr.Textbox(label=C.CLIP_LINK_LABEL, placeholder=C.CLIP_LINK_PLACEHOLDER, scale=4,
                                                 info=C.CLIP_LINK_NOTE)
                            v_fetch = gr.Button(C.CLIP_LINK_BTN, scale=1, elem_classes="mc-surprise", elem_id="mc-vibe-fetch")
                        v_link_md = gr.Markdown("", elem_classes="mc-note")
                        with gr.Row():
                            v_clip = gr.Video(label=f"① Reference move (up to {s.max_vibe_seconds * 2:.0f} s on Max, {s.max_vibe_seconds:.0f} s on Standard, one person)", height=240, elem_id="mc-vibe-clip")
                            with gr.Column():
                                gr.HTML(step_label(2, C.STEP_YOU), elem_classes="mc-step-title")
                                v_photo = gr.Image(type="filepath", label="② Your selfie or avatar (full body works best)", height=240,
                                                   elem_classes="mc-frame", image_mode=None)
                                v_subject = subject_radio()
                                gr.HTML(uibits.info_card(C.REF_CARD, C.REF_TIP), elem_classes="mc-ref-tip")
                                v_refq = gr.Markdown("", elem_classes="mc-note mc-ref-check")
                        v_scene = gr.Textbox(label="③ Your new scene (optional)", placeholder=VIBE_SCENE_IDEAS[0])
                        v_ideas = gr.Button("🎲 Scene idea", size="sm", elem_classes="mc-surprise")
                        gr.HTML(step_label(3, C.STEP_GO), elem_classes="mc-step-title")
                        v_eta = gr.Markdown(animate_eta_hint(10, "move", t0.key), elem_classes="mc-note mc-eta", elem_id="mc-swap-eta")
                        v_audio, v_voice = clip_audio_controls()
                        v_consent = consent_box()
                        go_vibe = gr.Button(swap_button_label(min(10, vibe_cap), t0.key, False, "auto", mode="move"),
                                            variant="primary", size="lg", elem_classes="mc-primary", elem_id="mc-go-vibe")
                        with gr.Accordion("More options", open=False, elem_classes="mc-more"):
                            gr.Markdown(C.VIBE_NOTE + "\n\n*Needs the Swap model pack (minus SAM 3.1): see Setup.*")
                            with gr.Row():
                                v_start = gr.Number(value=0, label="Start at (s)")
                                v_len = gr.Slider(1, vibe_cap, value=min(10, vibe_cap), step=0.5, label="Length (s)")
                            with gr.Row():
                                v_top = gr.Textbox(label="Top text")
                                v_bottom = gr.Textbox(label="Bottom text")
                    # ---------------- Meme Pack ----------------
                    with gr.Tab(C.TAB_PACK, id="pack"):
                        gr.Markdown("One selfie (or avatar), a week of posts. Default captions; tweak any of them later in the Meme tab.")
                        p_photo = gr.Image(type="filepath", label="Your selfie or avatar", height=200, image_mode=None)
                        p_refq = gr.Markdown("", elem_classes="mc-note mc-ref-check")
                        p_recipes = gr.CheckboxGroup([(r.title, r.key) for r in memes.RECIPES.values()], value=DEFAULT_PACK,
                                                     label="Formats", elem_classes="mc-format-grid")
                        p_consent = consent_box()
                        go_pack = gr.Button(C.PACK_CTA, variant="primary", size="lg", elem_classes="mc-primary", elem_id="mc-go-pack")
                    with gr.Tab(C.TAB_MORE, id="more"):
                        gr.Markdown(C.MORE_INTRO)
                        talk_gate = gr.Markdown(more_modes_gate(t0.key, False), elem_classes="mc-note mc-warn")
                        with gr.Tabs():
                            # ---------------- talking modes ----------------
                            with gr.Tab("🎤 Talk", id="talk"):
                                with gr.Row():
                                    photo = gr.Image(type="filepath", label="Your photo or avatar (or describe a character)", height=200,
                                                     image_mode=None)
                                    with gr.Column():
                                        character = gr.Textbox(label="…or describe a character", placeholder="a cheerful barista with curly red hair")
                                        voice_mode = gr.Radio([("Clone my voice", "clone"), ("Built-in voice", "builtin"), ("I'll record it myself", "my_recording")], value="clone", label="Voice")
                                voice = gr.Audio(type="filepath", label="Voice sample: 10 to 30 s of you talking", sources=["upload", "microphone"])
                                recording = gr.Audio(type="filepath", label="Your recording of the script", sources=["upload", "microphone"], visible=False)
                                starter = gr.Dropdown(list(SCRIPT_STARTERS), label="Need an idea? Pick a starter", value=None)
                                script = gr.Textbox(label="Script", lines=4, placeholder="Hey, welcome back! …", max_length=s.max_script_chars * 2)
                                t_consent = consent_box()
                                go_talk = gr.Button(C.TALK_CTA, variant="primary", size="lg", elem_classes="mc-primary", elem_id="mc-go-talk")
                                with gr.Accordion("🎭 Character voice (voice changer)", open=False, elem_classes="mc-more"):
                                    char_voice = gr.Audio(type="filepath", label="Target voice sample (yours, or one you have rights to)")
                            with gr.Tab("🎭 Duet", id="duet"):
                                gr.Markdown("Two characters, one shot. One line per turn as `Name: line`.")
                                with gr.Row():
                                    d_photo = gr.Image(type="filepath", label="Person A (left)", height=180)
                                    d_photo_b = gr.Image(type="filepath", label="Person B (right, optional)", height=180)
                                d_character = gr.Textbox(label="…or describe both characters", placeholder="two friendly robots hosting a tech podcast")
                                with gr.Row():
                                    d_voice = gr.Audio(type="filepath", label="Voice A sample (optional)")
                                    d_voice_b = gr.Audio(type="filepath", label="Voice B sample (optional)")
                                d_script = gr.Textbox(label="Dialogue", lines=5, value="Maya: Did you see the new update?\nTheo: I did. I'm still recovering.\nMaya: Okay chat, be honest. Good or bad?")
                                d_consent = consent_box()
                                go_duet = gr.Button("Make the skit", variant="primary", size="lg", elem_classes="mc-primary", elem_id="mc-go-duet")
                            with gr.Tab("🎵 Sing", id="sing"):
                                gr.Markdown("Lip-sync yourself to your song. Full mix **and** vocals-only stem work best.")
                                s_photo = gr.Image(type="filepath", label="Your photo", height=200)
                                s_character = gr.Textbox(label="…or describe the singer")
                                with gr.Row():
                                    song = gr.Audio(type="filepath", label="Song (full mix)")
                                    vocals = gr.Audio(type="filepath", label="Vocals stem (recommended)")
                                with gr.Row():
                                    song_start = gr.Number(value=0, label="Start at (s)")
                                    song_len = gr.Slider(5, 60, value=20, step=1, label="Length (s)")
                                lyrics = gr.Textbox(label="Lyrics for captions (optional)", lines=3)
                                s_consent = consent_box()
                                go_sing = gr.Button("Make the music clip", variant="primary", size="lg", elem_classes="mc-primary", elem_id="mc-go-sing")
                            with gr.Tab("🎙️ Re-dub a line", id="redub"):
                                gr.Markdown(C.REDUB_INTRO)
                                r_video = gr.Video(label="Finished clip", height=200, elem_id="mc-redub-video")
                                with gr.Row():
                                    with gr.Column():
                                        r_rec = gr.Audio(type="filepath", label="Your line (record or upload)",
                                                         sources=["microphone", "upload"], elem_id="mc-redub-rec")
                                        r_takes = gr.File(file_count="multiple", file_types=["audio"], height=90,
                                                          label="more takes (optional): mc picks the best one",
                                                          elem_id="mc-redub-takes")
                                    with gr.Column():
                                        r_text = gr.Textbox(label="…or type it (said in your voice)", max_lines=2, elem_id="mc-redub-text")
                                        r_voice = gr.Audio(type="filepath", label="Voice sample for typed lines (optional)")
                                with gr.Row():
                                    r_start = gr.Number(value=0, label="Line starts at (s)")
                                    r_fit = gr.Number(value=0, label="Fit into (s, 0 = as recorded)")
                                r_lips = gr.Checkbox(value=False, label="👄 lip re-sync (one short GPU job; pending a real test)", elem_id="mc-redub-lips")
                                r_consent = consent_box()
                                go_redub = gr.Button("re-dub it", variant="primary", size="lg", elem_classes="mc-primary", elem_id="mc-go-redub")
                            with gr.Tab("📚 Batch", id="batch"):
                                gr.Markdown("Many scripts in, many talking clips out, same face and voice. One script per line, or blank-line separated blocks.")
                                b_photo = gr.Image(type="filepath", label="Your photo", height=180)
                                b_character = gr.Textbox(label="…or describe a character")
                                b_voice = gr.Audio(type="filepath", label="Voice sample (empty = built-in voice)")
                                b_scripts = gr.Textbox(label="Scripts", lines=7, placeholder="Script one…\nScript two…")
                                b_consent = consent_box()
                                go_batch = gr.Button("Render the batch", variant="primary", size="lg", elem_classes="mc-primary", elem_id="mc-go-batch")
                    with gr.Tab(C.TAB_SETUP, id="setup"):
                        gr.Markdown(
                            "**Hardware tiers.** MC picks one from your GPU's VRAM (override: `MC_TIER=lite|standard|max` "
                            "or the toggle at the top). ComfyUI streams what doesn't fit from RAM (Dynamic VRAM, the native "
                            "*block swap*). Add the tier's launch flags to ComfyUI.\n\n"
                            + "\n".join(f"- **{t.label}** ({t.gpus}): {t.notes} Flags: `{t.comfy_flags or 'none'}`. {t.flags_note}"
                                         for t in tiers.TIERS.values())
                            + "\n\n**Swap / Vibe model pack** (one-time, ~18 GB, all open weights; Copy the Vibe skips SAM 3.1 and the relight LoRA): Wan2.2 Animate 14B GGUF, CLIP Vision H, "
                              "Wan-Animate relight LoRA, SAM 3.1, SDPose + RT-DETR v4. Links and licenses in `models.md`; "
                              "`python -m main_character_comfy.doctor --live` tells you what's missing."
                        )
                        gr.Markdown(C.AGENT_SETUP, elem_classes="mc-note mc-agent-setup")
                        check_btn = gr.Button(C.CHECK_MODELS, size="sm", elem_classes="mc-surprise", elem_id="mc-check-models")
                        check_md = gr.Markdown("", elem_classes="mc-note")

                with gr.Accordion("Advanced", open=False, elem_classes="mc-advanced"):
                    with gr.Row():
                        seam = gr.Radio([("Anchor (sharpest)", "anchor"), ("Continue (smoothest)", "continue")], value="anchor", label="Long-clip seams")
                        segment = gr.Number(value=0, label="Segment length s (0 = tier default)")
                    with gr.Row():
                        smooth = gr.Radio([("Off", "off"), ("RIFE 50 fps", "rife"), ("ffmpeg 50 fps", "ffmpeg")], value="off", label="Smooth motion (talk modes)")
                        captions = gr.Checkbox(value=True, label="Speech captions")
                        hd = gr.Checkbox(value=True, label="Export 1080p")
                    seed = gr.Number(value=2026, label="Seed", precision=0)

            # ---------------- preview column (shared) ----------------
            with gr.Column(scale=4, elem_classes="mc-preview"):
                gr.HTML(f"<div class='mc-lab'>{html.escape(C.OUTPUT_LABEL)}</div>", elem_classes="mc-step-title")
                gr.HTML(uibits.empty_state_html(C.EMPTY_TITLE, C.TAG, C.EMPTY_HINT), elem_classes="mc-empty-wrap")
                still_out = gr.Image(label="preview", height=520, interactive=False, elem_classes="mc-still mc-frame", elem_id="mc-still")
                video_out = gr.Video(label="preview", height=520, autoplay=True, visible=False, elem_classes="mc-video mc-frame", elem_id="mc-video")
                animate_btn = gr.Button(C.ANIMATE_CTA, variant="secondary", size="lg", visible=False, elem_classes="mc-animate", elem_id="mc-animate")
                with gr.Row():
                    hd_btn = gr.Button("✨ make it HD", variant="secondary", visible=False, elem_classes="mc-hd", elem_id="mc-hd")
                    wipe_btn = gr.Button("↔️ before / after", visible=False, elem_classes="mc-wipe", elem_id="mc-wipe")
                status = gr.Markdown(C.READY, elem_classes="mc-status", elem_id="mc-status")
                toast = gr.Markdown("", elem_classes="mc-toast")
                stages_box = gr.HTML(_stages_html({}, "meme"))
                share_box = gr.HTML("", elem_classes="mc-share-wrap", elem_id="mc-share")
                post_text = gr.Textbox(label=C.POST_TEXT_LABEL, lines=3, show_copy_button=True, elem_classes="mc-post-text",
                                       placeholder=C.POST_TEXT_EMPTY, elem_id="mc-post-text")
                with gr.Row():
                    reroll = gr.Button(C.REROLL, elem_classes="mc-reroll", elem_id="mc-reroll")
                    stop_btn = gr.Button(C.STOP, elem_classes="mc-stop", elem_id="mc-stop")
                    folder_btn = gr.Button(C.OPEN_FOLDER, elem_classes="mc-folder", elem_id="mc-folder")
                watermark = gr.Checkbox(value=True, label=C.WATERMARK_TOGGLE, elem_classes="mc-watermark", elem_id="mc-watermark")
                download = gr.File(label="Download", elem_classes="mc-download", elem_id="mc-download")
                scene_state = gr.State(None)
                work_state = gr.State(None)
                with gr.Accordion(C.SAVE_FORMAT, open=False, elem_classes="mc-save-format"):
                    gr.Markdown(C.SAVE_FORMAT_HINT, elem_classes="mc-note")
                    with gr.Row():
                        f_name = gr.Textbox(label="Format name", placeholder="Drive-Thru Stare", max_length=40, scale=3)
                        f_emoji = gr.Textbox(label="Emoji", placeholder="🥤", max_length=4, scale=1)
                    f_save = gr.Button(C.SAVE_FORMAT, size="sm")
                with gr.Accordion("Behind the scenes", open=False, elem_classes="mc-bts"):
                    avatar_out = gr.Image(label="You, in the scene", height=240, interactive=False)
                    audio_out = gr.Audio(label="Voice", interactive=False)

        with gr.Accordion(C.ABOUT_ACC, open=False, elem_classes="mc-more mc-about"):
            gr.Markdown(
                "Built on the **Comfy Developer Platform** (official Comfy SDK → comfy-api-proxy → your local ComfyUI). "
                "Models: FLUX.2 [klein] 4B · Wan 2.2 I2V A14B · Wan2.2 Animate 14B · SAM 3.1 · SDPose · Chatterbox · "
                "Wan2.1 + InfiniteTalk · RIFE. All open weights. Formats are original recipes; no meme footage is "
                "bundled. Only upload clips you have the rights to, your own face or avatar (or one you have the rights "
                "to), and don't upload anyone's face, voice or video without their permission. Every output folder includes a remixable recipe JSON.",
                elem_classes="mc-footer",
            )
        gr.HTML(footer_html(), elem_classes="mc-footer")

        outputs = [stages_box, avatar_out, audio_out, video_out, still_out, download, status, share_box, post_text,
                   animate_btn, scene_state, work_state]

        def bad(msg: str, mode: str = "meme") -> tuple:
            vals = {k: gr.update() for k in OUT_KEYS}
            vals.update(stages=_stages_html({}, mode), status=f"⚠️ {msg}")
            return tuple(vals[k] for k in OUT_KEYS)
        adv_inputs = [style, style_extra, captions, smooth, hd, watermark, seed, tier_radio, heavy, seam, segment, m_subject]
        adv_keys = ["style", "style_extra", "captions", "smooth", "hd", "watermark", "seed", "tier", "heavy", "seam", "segment",
                    "subject"]

        def adv_of(vals: tuple) -> dict:
            return dict(zip(adv_keys, vals))

        def on_recipe(key: str):
            r = memes.RECIPES.get(key)
            return (_recipe_card(key), gr.update(value=r.top if r else ""), gr.update(value=r.bottom if r else ""),
                    C.easter_egg(top=r.top, bottom=r.bottom) if r else "")

        m_recipe.change(on_recipe, m_recipe, [m_card, m_top, m_bottom, toast])
        def do_surprise():
            key, (t, b) = memes.surprise()
            return key, t, b

        surprise.click(do_surprise, None, [m_recipe, m_top, m_bottom])
        cap_reroll.click(lambda k, t, b: memes.next_caption(k, (t, b)), [m_recipe, m_top, m_bottom], [m_top, m_bottom])
        m_top.change(lambda t, b: C.easter_egg(top=t, bottom=b) or "", [m_top, m_bottom], toast)
        m_bottom.change(lambda t, b: C.easter_egg(top=t, bottom=b) or "", [m_top, m_bottom], toast)
        voice_mode.change(lambda m: gr.update(visible=m == "my_recording"), voice_mode, recording)
        starter.change(lambda k: SCRIPT_STARTERS.get(k, gr.update()) if k else gr.update(), starter, script)

        def meme_request(photo, recipe, top, bottom, consent, length, era, effect, say, line, vmode, voice, scene,
                         motion, adv) -> Request:
            return Request(mode="meme", meme=recipe, photo=_p(photo), meme_top=top or "", meme_bottom=bottom or "",
                           effect=None if effect == "default" else effect, say_line=bool(say), line=line or "",
                           voice_mode=vmode, voice_sample=_p(voice) if vmode == "clone" else None,
                           recording=_p(voice) if vmode == "my_recording" else None, custom_scene=scene or "",
                           custom_motion=motion or "", consent=bool(consent), meme_frames=int(length), era=era,
                           **_common(adv_of(adv)))

        def run_meme(photo, recipe, top, bottom, consent, length, era, straight, effect, say, line, vmode, voice,
                     scene, motion, *adv):
            """Main button: instant captioned still (or straight to video if asked)."""
            req = meme_request(photo, recipe, top, bottom, consent, length, era, effect, say, line, vmode, voice, scene,
                               motion, adv)
            p = pipeline()
            yield from _stream(p.run(req) if straight else p.run_still(req), "meme")

        def animate(scene, photo, recipe, top, bottom, consent, length, era, straight, effect, say, line, vmode, voice,
                    sc, motion, *adv):
            req = meme_request(photo, recipe, top, bottom, consent, length, era, effect, say, line, vmode, voice, sc,
                               motion, adv)
            if scene:
                req.avatar = Path(scene)  # reuse the still's scene: no second FLUX pass
            yield from _stream(pipeline().run(req), "meme")

        def run_swap(clip, photo, consent, start, length, prompt, top, bottom, relight, props, erase, res, face_all,
                     face_keep, subj, audio_mode, avoice, defer, engine, split, slock, closeup, chunks, rotate, count,
                     *rest):
            cap_ = tiers.PEOPLE_HARD_CAP
            slots, moves, descs, adv = rest[:cap_], rest[cap_:2 * cap_], rest[2 * cap_:3 * cap_], rest[3 * cap_:]
            n_ = int(count or 0)
            people = [_p(x) for x in slots[:n_]] if any(slots[:n_]) else []
            face = face_list(face_all, face_keep, max(1, len(people)))
            common = {**_common(adv_of(adv)), "subject": subj or "auto"}
            req = Request(mode="swap", clip=_p(clip), photo=_p(photo), clip_start=float(start or 0),
                          clip_seconds=float(length), swap_prompt=prompt or "", meme_top=top or "",
                          meme_bottom=bottom or "", relight={C.RELIGHT_CHOICES[1]: True, C.RELIGHT_CHOICES[2]: False}.get(relight), remove_props=bool(props), erase_props=erase or "", resolution=res or "auto", consent=bool(consent), people=people, face=face,
                          clip_audio=audio_mode or "original", voice_sample=_p(avoice),
                          defer_upscale=bool(defer), group_engine=engine or "auto", split_cuts=split or "auto",
                          style_lock=slock or stylelock.DEFAULT_MODE, ref_closeup="auto" if closeup else "off",
                          scail_chunks=bool(chunks), rotate=int(rotate or 0),
                          people_driver=[str(m or "") for m in moves[:n_]] if people else [],
                          people_desc=[str(d or "").strip() for d in descs[:n_]] if people else [], **common)
            yield from _stream(pipeline().run(req), "swap")

        def find_people(clip, start, length, tier, allow):
            hide = [gr.update(visible=False)] * tiers.PEOPLE_HARD_CAP * 3
            if not clip:
                return ("⚠️ Upload the clip first.", gr.update(visible=False), 0, *hide, "")
            req = Request(mode="swap", clip=_p(clip), clip_start=float(start or 0), clip_seconds=float(length),
                          tier=tier or tiers.DEFAULT_TIER, allow_heavy=bool(allow))
            try:
                done = None
                for ev in in_gpu_lane(pipeline().scan_people(req), "find_people"):
                    if ev.status == "complete":
                        done = ev
            except Exception as exc:
                return (friendly_error(exc, tier), gr.update(visible=False), 0, *hide, "")
            n = int(done.data["people"]) if done else 0
            cap = tiers.people_cap(tier, bool(allow))
            slots = [gr.update(visible=i < n, label=f"person {i + 1}: selfie / avatar (empty = unchanged)")
                     for i in range(tiers.PEOPLE_HARD_CAP)]
            moves = [gr.update(visible=i < n and n > 1, choices=uibits.driver_choices(n, i), value="")
                     for i in range(tiers.PEOPLE_HARD_CAP)]
            descs = [gr.update(visible=i < n) for i in range(tiers.PEOPLE_HARD_CAP)]
            note = done.message if done else "No people found."
            if n > cap:
                note += f" Assign up to {cap}; leave the rest empty."
            return (note, gr.update(value=str(done.artifact), visible=True) if done else gr.update(), n, *slots,
                    *moves, *descs, uibits.who_strip_html(n, []))

        def run_vibe(clip, photo, scene, consent, start, length, top, bottom, subj, audio_mode, avoice, *adv):
            common = {**_common(adv_of(adv)), "subject": subj or "auto"}
            req = Request(mode="vibe", clip=_p(clip), photo=_p(photo), scene_prompt=scene or "",
                          clip_start=float(start or 0), clip_seconds=float(length), clip_audio=audio_mode or "original",
                          voice_sample=_p(avoice), meme_top=top or "", meme_bottom=bottom or "", consent=bool(consent),
                          **common)
            yield from _stream(pipeline().run(req), "vibe")

        def run_pack(photo, keys, consent, *adv):
            base = Request(mode="meme", photo=_p(photo), consent=bool(consent), **_common(adv_of(adv)))
            try:
                yield from _stream(pipeline().run_pack(base, list(keys or [])), "meme")
            except InputError as exc:
                yield bad(str(exc), "meme")

        def run_talk(photo, character, voice_mode, voice, recording, script, consent, char_voice, *adv):
            req = Request(mode="talk", script=script or "", photo=_p(photo), character=character or "",
                          voice_mode=voice_mode, voice_sample=_p(voice), recording=_p(recording),
                          character_voice=_p(char_voice), consent=bool(consent), **_common(adv_of(adv)))
            yield from _stream(pipeline().run(req), "talk")

        def run_duet(p_a, p_b, character, v_a, v_b, script, consent, *adv):
            req = Request(mode="dialogue", script=script or "", photo=_p(p_a), photo_b=_p(p_b), character=character or "",
                          voice_sample=_p(v_a), voice_sample_b=_p(v_b), consent=bool(consent), **_common(adv_of(adv)))
            yield from _stream(pipeline().run(req), "dialogue")

        def run_sing(p, character, song, vocals, start, length, lyrics, consent, *adv):
            req = Request(mode="sing", script=lyrics or "", photo=_p(p), character=character or "", song=_p(song),
                          vocals=_p(vocals), song_start=float(start or 0), song_seconds=float(length),
                          consent=bool(consent), **_common(adv_of(adv)))
            yield from _stream(pipeline().run(req), "sing")

        def run_batch(p, character, v, scripts, consent, *adv):
            base = Request(mode="talk", photo=_p(p), character=character or "", consent=bool(consent),
                           voice_mode="clone" if v else "builtin", voice_sample=_p(v), **_common(adv_of(adv)))
            try:
                items = split_batch_text(scripts or "")
                yield from _stream(pipeline().run_batch(base, items), "talk")
            except InputError as exc:
                yield bad(str(exc), "talk")

        def make_hd(work):
            """'make it HD': SeedVR2 + export of the deferred take shown (Pipeline.finish_pending)."""
            if pending_take(work) is None:
                yield bad("No preview take waiting for HD here.", "swap")
                return
            yield from _stream(pipeline().finish_pending([Path(str(work))]), "swap")

        def run_redub_ui(video, rec, takes, text, voice, start, fit, lips, consent):
            from .extras import run_redub

            if not video:
                yield bad("Add the finished clip first.", "talk")
                return
            try:
                more = [Path(str(getattr(t, "name", t))) for t in (takes or []) if t]
                ev = run_redub(pipeline(), Path(str(video)), recording=_p(rec), text=text or "", takes=more,
                               voice_sample=_p(voice), start=float(start or 0),
                               fit_seconds=float(fit) if fit else None, consent=bool(consent), lipsync=bool(lips))
                yield from _stream(ev, "talk")
            except InputError as exc:
                yield bad(str(exc), "talk")

        meme_inputs = [m_photo, m_recipe, m_top, m_bottom, m_consent, m_length, m_era, m_straight, m_effect, m_say, m_line,
                       m_voice_mode, m_voice, m_scene, m_motion]
        swap_inputs = [w_clip, w_photo, w_consent, w_start, w_len, w_prompt, w_top, w_bottom, w_relight, w_props, w_erase, w_res, w_face, w_face_keep, w_subject, w_audio,
                       w_voice, w_defer, w_engine, w_split, w_stylelock, w_closeup, w_chunks, w_rotate, w_count,
                       *w_slots, *w_moves, *w_descs]
        vibe_inputs = [v_clip, v_photo, v_scene, v_consent, v_start, v_len, v_top, v_bottom, v_subject, v_audio, v_voice]
        talk_inputs = [photo, character, voice_mode, voice, recording, script, t_consent, char_voice]
        last = gr.State("meme")
        rerolls = gr.State(0)
        runs = [
            go_meme.click(run_meme, [*meme_inputs, *adv_inputs], outputs),
            animate_btn.click(animate, [scene_state, *meme_inputs, *adv_inputs], outputs),
            go_swap.click(run_swap, [*swap_inputs, *adv_inputs], outputs),
            go_vibe.click(run_vibe, [*vibe_inputs, *adv_inputs], outputs),
        ]
        for ev_, name_ in zip(runs, ("meme", "animate", "swap", "vibe")):
            ev_.then(lambda n=name_: n, None, last)
        for ev_ in runs[2:4]:  # swap / vibe: offer "make it HD" (deferred take) and the before/after wipe
            ev_.then(after_video_run, [work_state, video_out], [hd_btn, wipe_btn])
        w_find.click(find_people, [w_clip, w_start, w_len, tier_radio, heavy],
                     [w_people_md, w_people_img, w_count, *w_slots, *w_moves, *w_descs, w_who])

        def _who(count, *vals):
            cap_ = tiers.PEOPLE_HARD_CAP
            return uibits.who_strip_html(int(count or 0), vals[:cap_], vals[cap_:2 * cap_], vals[2 * cap_:])

        for ctl_ in (*w_slots, *w_moves, *w_descs):
            ctl_.change(_who, [w_count, *w_slots, *w_moves, *w_descs], w_who)
        for ctl, voice_box in ((w_audio, w_voice), (v_audio, v_voice)):
            ctl.change(lambda m: gr.update(visible=m == "my_voice"), ctl, voice_box)

        def do_stop():
            p = running["p"]
            if p is not None:
                p.cancel()
            return "⏹️ Stopping… the current ComfyUI job is being cancelled."

        def do_folder(work):
            target = Path(work) if work else s.output_dir
            target.mkdir(parents=True, exist_ok=True)
            try:
                media.open_folder(target)
                return f"📂 Opened `{target}`"
            except Exception:
                return f"📂 Your files are in `{target}`"

        folder_btn.click(do_folder, work_state, toast)
        guide_btn.click(None, None, None, js=uibits.GUIDE_OPEN_JS)
        guide_close.click(None, None, None, js=uibits.GUIDE_CLOSE_JS)
        wipe_btn.click(wipe_for, [work_state, video_out], [download, toast])
        hd_run = hd_btn.click(make_hd, work_state, outputs)
        hd_run.then(after_video_run, [work_state, video_out], [hd_btn, wipe_btn])
        redub_run = go_redub.click(run_redub_ui, [r_video, r_rec, r_takes, r_text, r_voice, r_start, r_fit, r_lips, r_consent],
                                   outputs)
        runs += [hd_run, redub_run]

        def _swap_label(length, tier, allow, res, count, *slots):
            n_ = sum(1 for x in slots[:int(count or 0)] if x) or 1
            return swap_button_label(length, tier, allow, res, n_)

        swap_lbl_in = [w_len, tier_radio, heavy, w_res, w_count, *w_slots]
        for ctl_ in (w_len, tier_radio, heavy, w_res, w_count, *w_slots):
            ctl_.change(_swap_label, swap_lbl_in, go_swap)
        for ctl_ in (v_len, tier_radio, heavy):
            ctl_.change(lambda x, t_, h_: swap_button_label(x, t_, h_, "auto", mode="move"), [v_len, tier_radio, heavy],
                        go_vibe)
        check_btn.click(lambda t: models_report(t), tier_radio, check_md)

        # reference quality check on every selfie / avatar drop (never blocks, just says so)
        def _refq(path, subject="auto"):
            return refcheck.summary_markdown(refcheck.safe_check(path, subject or "auto"))

        def _refq_short(path, subject="auto"):  # swap / vibe already show the tip right above
            md_ = refcheck.summary_markdown(refcheck.safe_check(path, subject or "auto"), with_tip=False)
            if path and (subject or "auto") == "auto":  # the drawing-vs-photo guess, when MC isn't sure
                try:
                    hint_ = stylelock.hint_text(stylelock.classify_reference(path))
                except Exception:  # noqa: BLE001
                    hint_ = ""
                if hint_:
                    md_ = (md_ + "\n\n" if md_ else "") + f"🎨 {hint_} ([why](#guide-cartoon))"
            return md_

        for img_, subj_, md_, fn_ in ((m_photo, m_subject, m_refq, _refq), (w_photo, w_subject, w_refq, _refq_short),
                                      (v_photo, v_subject, v_refq, _refq_short)):
            img_.change(fn_, [img_, subj_], md_)
            subj_.change(fn_, [img_, subj_], md_)
        p_photo.change(_refq, p_photo, p_refq)

        def _slots_refq(*paths):
            lines = []
            for i_, p_ in enumerate(paths):
                q_ = refcheck.safe_check(p_)
                if q_ and q_.get("warnings"):
                    lines.append(f"- **person {i_ + 1}:** " + " ".join(q_["warnings"]))
            return ("\n".join(lines) + f"\n\n{refcheck.TIP}") if lines else ""

        for slot_ in w_slots:
            slot_.change(_slots_refq, w_slots, w_slots_refq)

        # bring your own clip: link -> sharpest local copy -> the clip box
        def _fetch(url: str, ok: bool, progress=gr.Progress()):  # noqa: B008 (gradio injects it)
            return fetch_clip_ui(url, ok, progress)

        w_fetch.click(_fetch, [w_link, w_consent], [w_clip, w_link_md])
        w_link.submit(_fetch, [w_link, w_consent], [w_clip, w_link_md])
        v_fetch.click(_fetch, [v_link, v_consent], [v_clip, v_link_md])
        v_link.submit(_fetch, [v_link, v_consent], [v_clip, v_link_md])

        # first-run tour (skippable, re-open with the help link)
        tour_last = len(C.TOUR_STEPS) - 1

        def _tour(i: int):
            i = int(i or 0)
            return i, tour_html(i), gr.update(value=C.TOUR_DONE if i == tour_last else C.TOUR_NEXT), gr.update(interactive=i > 0)

        def _next(i: int):
            i = int(i or 0)
            if i >= tour_last:
                return (gr.update(visible=False), *_tour(0))
            return (gr.update(visible=True), *_tour(i + 1))

        tour_outs = [tour_box, tour_i, tour_md, tour_next, tour_back]
        tour_next.click(_next, tour_i, tour_outs, js=(
            f"(i) => {{ if (Number(i) >= {tour_last}) {{ try {{ localStorage.setItem('mc_tour_done', '1'); }} "
            "catch (e) {} document.documentElement.classList.add('mc-tour-done'); } return i; }"))
        tour_back.click(lambda i: (gr.update(visible=True), *_tour(max(0, int(i or 0) - 1))), tour_i, tour_outs)
        tour_skip.click(lambda i: (gr.update(visible=False), *_tour(0)), tour_i, tour_outs, js=TOUR_DONE_JS)
        help_btn.click(lambda i: (gr.update(visible=True), *_tour(0)), tour_i, tour_outs, js=TOUR_OPEN_JS)
        demo.load(lambda: gr.update(interactive=False), None, tour_back)
        w_len.change(lambda x, t, h, r: animate_eta_hint(x, "replace", t, h, r), [w_len, tier_radio, heavy, w_res], w_eta)
        w_res.change(lambda x, t, h, r: animate_eta_hint(x, "replace", t, h, r), [w_len, tier_radio, heavy, w_res], w_eta)
        v_len.change(lambda x, t, h: animate_eta_hint(x, "move", t, h), [v_len, tier_radio, heavy], v_eta)

        def on_tier(tier, allow, wl, vl):
            t = tiers.get(tier)
            allow = bool(allow) and bool(t.gated)
            wc, vc = tiers.animate_cap(t, s.max_swap_seconds), tiers.animate_cap(t, s.max_vibe_seconds)
            wl, vl = min(float(wl or 5), wc), min(float(vl or 10), vc)
            return (header_html(t.key), tier_info(choice, t.key), gr.update(visible=bool(t.gated), value=allow),
                    gr.update(maximum=wc, value=wl), gr.update(maximum=vc, value=vl),
                    animate_eta_hint(wl, "replace", t.key, allow), animate_eta_hint(vl, "move", t.key, allow),
                    more_modes_gate(t.key, allow))

        tier_outs = [header, tier_md, heavy, w_len, v_len, w_eta, v_eta, talk_gate]
        tier_radio.change(on_tier, [tier_radio, heavy, w_len, v_len], tier_outs)
        heavy.change(lambda t, h, wl, vl: on_tier(t, h, wl, vl)[3:], [tier_radio, heavy, w_len, v_len], tier_outs[3:])
        v_ideas.click(lambda cur: random.choice([x for x in VIBE_SCENE_IDEAS if x != cur]), v_scene, v_scene)

        def do_save_format(name, emoji, work, key, top, bottom, scene, motion):
            try:
                new_key, msg = save_format_from(name, emoji, work, key, top, bottom, scene, motion)
            except ValueError as exc:
                return gr.update(), gr.update(), f"⚠️ {exc}"
            choices = [(r.title, r.key) for r in memes.RECIPES.values()]
            return gr.update(choices=_recipe_choices(), value=new_key), gr.update(choices=choices), msg

        f_save.click(do_save_format, [f_name, f_emoji, work_state, m_recipe, m_top, m_bottom, m_scene, m_motion],
                     [m_recipe, p_recipes, toast])
        more_runs = [
            (go_pack.click(run_pack, [p_photo, p_recipes, p_consent, *adv_inputs], outputs), "pack"),
            (go_talk.click(run_talk, [*talk_inputs, *adv_inputs], outputs), "talk"),
            (go_duet.click(run_duet, [d_photo, d_photo_b, d_character, d_voice, d_voice_b, d_script, d_consent, *adv_inputs], outputs), "duet"),
            (go_sing.click(run_sing, [s_photo, s_character, song, vocals, song_start, song_len, lyrics, s_consent, *adv_inputs], outputs), "sing"),
            (go_batch.click(run_batch, [b_photo, b_character, b_voice, b_scripts, b_consent, *adv_inputs], outputs), "batch"),
        ]
        for ev_, name_ in more_runs:
            ev_.then(lambda n=name_: n, None, last)
        runs += [ev_ for ev_, _ in more_runs]

        def reroll_seed(n: int):
            new = random.randint(1, 2**31 - 1)
            return new, n + 1, C.easter_egg(seed=new, rerolls=n + 1) or ""

        def rerun(which, *vals):
            """Free retry: re-run whatever you ran last with the new seed."""
            nm, nw, nv, nt = len(meme_inputs), len(swap_inputs), len(vibe_inputs), len(talk_inputs)
            vals = list(vals)
            meme_v, vals = vals[:nm], vals[nm:]
            swap_v, vals = vals[:nw], vals[nw:]
            vibe_v, vals = vals[:nv], vals[nv:]
            talk_v, adv = vals[:nt], vals[nt:]
            if which == "swap":
                yield from run_swap(*swap_v, *adv)
            elif which == "vibe":
                yield from run_vibe(*vibe_v, *adv)
            elif which == "animate":
                yield from animate(None, *meme_v, *adv)
            elif which == "talk":
                yield from run_talk(*talk_v, *adv)
            else:
                yield from run_meme(*meme_v, *adv)

        runs.append(reroll.click(reroll_seed, rerolls, [seed, rerolls, toast]).then(
            rerun, [last, *meme_inputs, *swap_inputs, *vibe_inputs, *talk_inputs, *adv_inputs], outputs))
        stop_btn.click(do_stop, None, status, cancels=runs)
    return demo
