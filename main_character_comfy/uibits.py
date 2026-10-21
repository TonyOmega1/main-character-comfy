"""Small, testable pieces of the web UI (ui.py wires them up): the useful-guide panel, the who's-who strip, the
"$0 · ~N min" render buttons and the per-person "moves" choices."""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any, Sequence

from .config import ROOT

GUIDE_FILE = ROOT / "docs" / "GUIDE.md"
GUIDE_LABEL = "useful guide"
GUIDE_TITLE = "Useful guide"
GUIDE_ANCHORS = ("source", "refs", "cartoon", "cuts", "duplicates", "roles", "lipsync", "details", "flicker", "speed",
                 "rights")
_HEAD_RE = re.compile(r"^###\s+(.*?)\s*·\s*`#guide-([a-z0-9-]+)`\s*$")


def guide_sections(text: str | None = None) -> list[tuple[str, str, str]]:
    """docs/GUIDE.md -> [(anchor, title, body markdown)], in order. Headings look like ``### 4. Hard cuts · `#guide-cuts```."""
    if text is None:
        try:
            text = GUIDE_FILE.read_text(encoding="utf-8")
        except OSError:
            return []
    out: list[tuple[str, str, list[str]]] = []
    for line in text.splitlines():
        m = _HEAD_RE.match(line.strip())
        if m:
            out.append((m.group(2), m.group(1).strip(), []))
        elif out:
            out[-1][2].append(line)
    return [(a, t, "\n".join(b).strip()) for a, t, b in out]


def _inline(s: str) -> str:
    t = html.escape(s, quote=False)
    t = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r"<a href='\2' target='_blank' rel='noopener'>\1</a>", t)
    t = re.sub(r"\[([^\]]+)\]\((#guide-[a-z0-9-]+)\)", r"<a href='\2'>\1</a>", t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", t)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", t)


def md_html(md: str) -> str:
    """Just enough markdown for the guide: paragraphs, '- ' lists, **bold**, *italic*, `code`, links."""
    out: list[str] = []
    for block in re.split(r"\n\s*\n", md.strip()):
        lines = [x.strip() for x in block.splitlines() if x.strip() and not x.strip().startswith("<!--")]
        if not lines:
            continue
        para: list[str] = []
        items: list[str] = []
        for x in lines + [""]:  # a paragraph line, then list items, can share one block ("**Tips:**" + "- ...")
            if x.startswith("- "):
                if para:
                    out.append(f"<p>{_inline(' '.join(para))}</p>")
                    para = []
                items.append(x[2:])
                continue
            if items:
                out.append("<ul>" + "".join(f"<li>{_inline(i)}</li>" for i in items) + "</ul>")
                items = []
            if x:
                para.append(x)
        if para:
            out.append(f"<p>{_inline(' '.join(para))}</p>")
    return "".join(out)


def guide_html(text: str | None = None) -> str:
    """The guide panel's body: a contents line, then one <section id="guide-..."> per topic (warnings deep-link)."""
    secs = guide_sections(text)
    if not secs:
        return "<div class='mc-guide-body'><p>The guide is in <code>docs/GUIDE.md</code>.</p></div>"
    short = [re.sub(r"^\d+\.\s*", "", t) for _, t, _ in secs]  # (no backslash inside an f-string: py3.10/3.11)
    toc = " · ".join(f"<a href='#guide-{a}'>{html.escape(s)}</a>" for (a, _, _), s in zip(secs, short))
    body = "".join(f"<section id='guide-{a}' class='mc-guide-sec'><h4>{html.escape(t)}</h4>{md_html(b)}</section>"
                   for a, t, b in secs)
    return (f"<div class='mc-guide-body'><h3 class='mc-guide-title'>{html.escape(GUIDE_TITLE)}<span "
            f"class='mc-red'>.</span></h3><p class='mc-guide-toc'>{toc}</p>{body}</div>")


# Any '#guide-*' link (a warning, the status line, the contents line) opens the panel and scrolls to the topic.
GUIDE_HEAD = """<script>(function(){
function openGuide(id){var root=document.documentElement;root.classList.add('mc-guide-open');
 var t=id?document.getElementById(id):null;
 setTimeout(function(){var p=document.getElementById('mc-guide');if(t){t.scrollIntoView({behavior:'smooth',block:'start'});
 t.classList.add('mc-guide-hit');setTimeout(function(){t.classList.remove('mc-guide-hit');},1600);}else if(p){p.scrollTop=0;}},60);}
window.mcOpenGuide=openGuide;
document.addEventListener('click',function(e){var a=e.target.closest&&e.target.closest('a[href^="#guide-"]');
 if(!a)return;e.preventDefault();e.stopPropagation();openGuide(a.getAttribute('href').slice(1));},true);
document.addEventListener('keydown',function(e){if(e.key==='Escape')document.documentElement.classList.remove('mc-guide-open');});
})();</script>"""
GUIDE_OPEN_JS = "() => { window.mcOpenGuide && window.mcOpenGuide(''); }"
GUIDE_CLOSE_JS = "() => { document.documentElement.classList.remove('mc-guide-open'); }"
GUIDE_CSS = """
#mc-guide{position:fixed;top:0;right:0;height:100vh;width:min(460px,92vw);overflow-y:auto;z-index:1000;
 transform:translateX(105%);transition:transform .25s ease;box-shadow:-8px 0 24px rgba(0,0,0,.25);
 background:var(--background-fill-primary);padding:18px 18px 40px;border-left:1px solid var(--border-color-primary)}
.mc-guide-open #mc-guide{transform:none !important}  /* !important: gradio re-emits every rule scoped
 under .gradio-container .contain, which out-ranks a plain html-class selector (same trick as .mc-tour-done) */
.mc-help-row{justify-content:flex-start !important;gap:24px !important}  /* left, where 'help: quick tour' always sat */
.mc-help-row > *{flex:0 0 auto !important;min-width:0 !important;width:auto !important}
#mc-guide .mc-guide-sec{scroll-margin-top:56px;border-radius:0;padding:2px 10px;transition:background .4s,box-shadow .4s}
#mc-guide .mc-guide-toc{font-size:.85em;line-height:1.7}
/* fit the noir look: same title as the quick tour card, a compact ghost "close" that stays in reach while you
   scroll, mono contents line, and a visible red flash on the topic a link jumped to */
#mc-guide{gap:10px}
#mc-guide .mc-guide-close{position:sticky;top:0;z-index:2;align-self:flex-end;flex:0 0 auto !important;width:auto !important;
 min-width:0 !important;padding:4px 10px !important;background:var(--bg,var(--background-fill-primary)) !important}
#mc-guide .mc-guide-title{font-family:var(--f-display,inherit);font-size:26px;margin:0 0 8px;text-transform:lowercase;
 color:var(--ink,inherit)}
#mc-guide .mc-red{color:var(--accent,var(--color-accent))}
#mc-guide .mc-guide-toc{font-family:var(--f-mono,monospace);font-size:11px;color:var(--muted,inherit);text-transform:lowercase}
#mc-guide .mc-guide-toc a{color:var(--ink,inherit);text-decoration-color:var(--accent,currentColor);text-underline-offset:2px}
#mc-guide .mc-guide-sec h4{font-family:var(--f-display,inherit);font-weight:800;color:var(--ink,inherit)}
#mc-guide .mc-guide-sec li::marker{color:var(--accent,var(--color-accent))}
#mc-guide .mc-guide-sec a{color:var(--ink,inherit);text-decoration-color:var(--accent,currentColor)}
/* body copy in the app's mono at a readable size (was the softer sans); red bullets stay */
#mc-guide .mc-guide-sec, #mc-guide .mc-guide-sec p, #mc-guide .mc-guide-sec li, #mc-guide .mc-guide-sec strong{
 font-family:var(--f-mono,ui-monospace,monospace) !important;font-size:12.5px !important;line-height:1.65 !important}
#mc-guide .mc-guide-sec h4{font-size:15px !important}
#mc-guide .mc-guide-hit{background:var(--surface,var(--background-fill-secondary));box-shadow:inset 2px 0 0 var(--accent,red)}
"""


def money_time_label(base: str, seconds: float | None) -> str:
    """Render button text with the price and the time it takes, e.g. 'swap me in · $0 · ~11-15 min'.

    The time comes from pipeline.eta_range_text, the same function the hint under the length slider uses, so the
    button and the hint always show the same range."""
    if not seconds or seconds <= 0:
        return f"{base} · $0"
    from .pipeline import eta_range_text  # lazy: pipeline is heavy and imports the UI-free modules only
    return f"{base} · $0 · {eta_range_text(seconds)}"


def driver_choices(n_people: int, slot: int) -> list[tuple[str, str]]:
    """'moves' dropdown for one person: their own moves, or anyone else's in the same clip."""
    return [("own moves", "")] + [(f"follow person {k + 1}", f"person:{k + 1}") for k in range(n_people) if k != slot]


def who_strip_html(n: int, refs: Sequence[Any], drivers: Sequence[str] = (), descs: Sequence[str] = ()) -> str:
    """Who's-who strip under the people preview: one chip per person found, saying who they become and whose moves
    they follow, so a mix-up is caught before a 20-minute render (guide: #guide-roles)."""
    if n <= 0:
        return ""
    chips = []
    for i in range(n):
        ref = refs[i] if i < len(refs) else None
        drv = (drivers[i] if i < len(drivers) else "") or ""
        desc = (descs[i] if i < len(descs) else "") or ""
        who = html.escape(Path(str(ref)).name) if ref else "unchanged"
        moves = "own moves"
        m = re.fullmatch(r"person:(\d+)", drv.strip())
        if m:
            moves = f"follows person {m.group(1)}"
        extra = f"<div class='mc-who-desc'>{html.escape(desc[:60])}</div>" if desc.strip() else ""
        cls = "mc-who on" if ref else "mc-who"
        chips.append(f"<div class='{cls}'><div class='mc-who-n'>{i + 1}</div><div><b>{who}</b>"
                     f"<div class='mc-who-moves'>{moves}</div>{extra}</div></div>")
    return ("<div class='mc-who-strip'>" + "".join(chips) + "</div><div class='mc-note'>numbered left to right on the "
            "first frame. wrong person? <a href='#guide-roles'>how numbering works</a></div>")


WHO_CSS = """
.mc-who-strip{display:flex;flex-wrap:wrap;gap:8px;margin:6px 0}
.mc-who{display:flex;gap:8px;align-items:center;border:1px solid var(--border-color-primary);border-radius:10px;
 padding:6px 10px;opacity:.65;font-size:.9em}
.mc-who.on{opacity:1;border-color:var(--color-accent)}
.mc-who-n{font-weight:800;font-size:1.2em;min-width:1.2em;text-align:center}
.mc-who-moves,.mc-who-desc{font-size:.85em;opacity:.8}
/* noir chrome: square chips, mono text, the number in the accent red once that person has a reference */
.mc-who{border-radius:0;font-family:var(--f-mono,monospace);font-size:11px;color:var(--muted,inherit);
 border-color:var(--line-strong,var(--border-color-primary))}
.mc-who b{color:var(--ink,inherit);font-weight:600}
.mc-who-n{font-family:var(--f-display,inherit);color:var(--muted,inherit)}
.mc-who.on .mc-who-n{color:var(--accent,var(--color-accent))}
/* the who's-who preview: grow the frame by its label so the picture stays inside the crop marks instead of
   spilling over the person slots under it */
.mc-people .block.mc-frame:has(img):not(:has(input[type=file])){height:auto !important}  /* the read-only one */
.mc-people .block.mc-frame:not(:has(input[type=file])) img{max-height:260px;object-fit:contain}  /* height=260 in ui.py */
"""


_GUIDE_REF = re.compile(r"\(guide: #guide-([a-z0-9-]+)\)|(?<![\w(/'\"])#guide-([a-z0-9-]+)")


def link_guide(text: str) -> str:
    """Pipeline messages say '(guide: #guide-cuts)'; in the UI that becomes a markdown link that opens the panel."""
    def sub(m: re.Match) -> str:
        a = m.group(1) or m.group(2)
        return f"([useful guide: {a}](#guide-{a}))" if m.group(1) else f"[useful guide: {a}](#guide-{a})"

    return _GUIDE_REF.sub(sub, text or "")


# The preview column's video is also an input (the before/after wipe reads it), so Gradio gives it an upload/webcam
# picker. It hangs below the fixed-height frame and sat on top of "make it HD" / "before / after", eating their
# clicks. Nothing gets uploaded into the output preview, so hide the picker there (the clip inputs keep theirs).
PREVIEW_CSS = """
.mc-preview .mc-video [data-testid="source-select"]{display:none !important}
"""


def info_card(summary: str, body_md: str, cls: str = "") -> str:
    """A compact, collapsible info card: one line visible, the long text one tap away (nothing deleted)."""
    return (f"<details class='mc-info {html.escape(cls, quote=True)}'><summary>{_inline(summary)}</summary>"
            f"<div class='mc-info-body'>{md_html(body_md)}</div></details>")


def rights_html(text: str) -> str:
    """The always-on one-line rights warning that sits right above every consent box (never folded away)."""
    return f"<p class='mc-rights' role='note'>{_inline(text)}</p>"


def lead_html(text: str) -> str:
    """One short line that says what a tab does."""
    return f"<p class='mc-lead'>{_inline(text)}</p>"


def empty_state_html(title: str, steps: str, hint: str) -> str:
    """The result area before the first render: what lands here and how to get it."""
    return (f"<div class='mc-empty'><div class='mc-empty-mark' aria-hidden='true'>MC<i>.</i></div>"
            f"<div class='mc-empty-title'>{html.escape(title)}<span class='mc-red'>.</span></div>"
            f"<div class='mc-empty-steps'>{html.escape(steps)}</div>"
            f"<div class='mc-empty-hint'>{html.escape(hint)}</div></div>")


# Gradio decides which tabs fit (the rest go behind a "..." menu) when a tab row first renders; a row inside a
# hidden tab (More: talk, duet, sing, re-dub, batch) gets measured at width 0. Re-measure after every tab switch.
APP_HEAD = """<script>document.addEventListener('click',function(e){
 if(e.target.closest&&e.target.closest('[role="tab"]')){setTimeout(function(){window.dispatchEvent(new Event('resize'));},60);}
},true);</script>"""
