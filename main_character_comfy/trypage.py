"""Static "try it" page for GitHub Pages (docs/try/index.html): what MC makes, a before/after wipe per example, the
useful guide, and how to run it. No server, no GPU, no tracking; everything is a relative link into docs/.

    python -m main_character_comfy.trypage                     # docs/try/index.html from docs/try/examples.json
    python -m main_character_comfy.trypage --examples my.json

examples.json: [{"title": "...", "after": "../demo/x.mp4", "before": "../demo/x_src.mp4" (optional), "note": "..."}].
A "before" turns the card into a drag-to-compare wipe. Only post clips you have the rights to (README: Responsible use).
Publishing the page (enabling GitHub Pages) is up to the repo owner; this only writes files.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import re
from pathlib import Path
from typing import Any

from . import copy as C
from . import themes, uibits
from .config import ROOT

OUT = ROOT / "docs" / "try"
DEFAULT_EXAMPLES = [
    {"title": "pull up: the 45 s demo", "after": "../demo/MC_demo_pullup_9x16.mp4",
     "note": "made with mc on one rtx 5080 (16 gb). free. local. open weights."},
]

PAGE_CSS = """
body{margin:0;font:13px/1.65 var(--f-mono);background:var(--bg);color:var(--ink)}
main{max-width:980px;margin:0 auto;padding:18px 20px 80px}
.mc-header{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap}
.mc-header-right{display:flex;align-items:center;gap:18px;flex-wrap:wrap}
.mc-logo{display:flex;flex-direction:column;gap:4px}
.mc-credit{font-family:var(--f-mono);font-size:11px;color:var(--muted)}
.mc-credit a{color:var(--ink)}
.hero{padding:28px 0 8px}.hero .mc-rule{margin:14px 0 18px}.hero h1{margin:0 0 14px}
.mc-h1{font-size:clamp(34px,5.4vw,64px)}
.lede{color:var(--muted);max-width:68ch}.lede b{color:var(--ink);font-weight:600}
h2{font-family:var(--f-mono);font-weight:600;font-size:13px;text-transform:lowercase;color:var(--ink);margin-top:2.4em}
h2::before{content:"";display:inline-block;width:6px;height:6px;background:var(--accent);border-radius:50%;margin-right:8px;vertical-align:1px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:18px;margin:24px 0}
.card{background:var(--surface);border:1px solid var(--line);border-radius:0;padding:12px}
.card h3{font-family:var(--f-mono);font-size:13px;font-weight:600;text-transform:lowercase;margin:4px 0 10px}
.card p{color:var(--muted)}
.wipe{position:relative;overflow:hidden;border-radius:0;background:#000;aspect-ratio:9/16}
.wipe video{position:absolute;inset:0;width:100%;height:100%;object-fit:contain}
.wipe .before{clip-path:inset(0 calc(100% - var(--x,50%)) 0 0)}
.wipe .bar{position:absolute;top:0;bottom:0;left:var(--x,50%);width:3px;background:#fff;transform:translateX(-50%)}
.wipe input{position:absolute;inset:0;width:100%;height:100%;opacity:0;cursor:ew-resize}
.solo video{width:100%;border-radius:0;background:#000}
.tag{font-family:var(--f-mono);font-size:11px;color:var(--muted)}
code{background:var(--tile);padding:1px 5px;border-radius:0;font-family:var(--f-mono);font-size:.95em}
a{color:var(--ink);text-decoration-color:var(--accent);text-underline-offset:2px}a:hover{color:var(--accent)}
.toc{font-size:11px;line-height:1.8;color:var(--muted);text-transform:lowercase}
.mc-guide-sec{background:var(--surface);border-radius:0;padding:6px 16px;margin:10px 0;border:1px solid var(--line);scroll-margin-top:12px}
.mc-guide-sec h4{font-family:var(--f-display);font-weight:800;font-size:15px}
.mc-guide-sec:target{border-color:var(--accent);box-shadow:inset 2px 0 0 var(--accent)}
.mc-guide-sec li::marker{color:var(--accent)}
"""

# Placeholder / test-pattern "before" clips never get a wipe (a fake before is worse than none).
PLACEHOLDER_BEFORE = re.compile(r"(placeholder|test[_-]?pattern|testsrc|smpte|colou?r[_-]?bars|\bbars\b|mock|dummy)", re.I)

THEME_JS = """<script>
(function(){var K='mc-try-theme';function p(m){document.documentElement.setAttribute('data-theme',m);
document.querySelectorAll('.mc-toggle [data-m]').forEach(function(b){var on=b.getAttribute('data-m')===m;
b.classList.toggle('on',on);b.setAttribute('aria-pressed',on?'true':'false');b.textContent=(on?'\\u25CF ':'\\u25CB ')+b.getAttribute('data-m');});}
window.mcSetTheme=function(m){try{localStorage.setItem(K,m);}catch(e){}p(m);};
var s=null;try{s=localStorage.getItem(K);}catch(e){}p(s==='light'?'light':'dark');})();
</script>"""

QR_SVG = Path(__file__).parent / "assets" / "brand" / "xmoney-qr.svg"  # tip QR (file only on the extras/tip-qr branch)
# Rules reused verbatim from the app theme (themes/noir.css) so the page's header, tip area and palette match the app.
_KEEP = (":root", "[data-theme", ".mc-header", ".mc-dot", ".mc-toggle", ".mc-tip", "a.mc-tip", ".mc-kicker", ".mc-rule",
         ".mc-h1", ".mc-tagline", ".mc-logo", ".mc-credit", "@media (max-width:700px)")


def app_css() -> str:
    """Top-level rules from the app's noir.css whose selector starts with one of _KEEP (palettes, tip area, header)."""
    src = re.sub(r"/\*.*?\*/", "", (themes.HERE / "noir.css").read_text(encoding="utf-8"), flags=re.S)
    out, i = [], 0
    while True:
        j = src.find("{", i)
        if j < 0:
            break
        depth, k = 1, j + 1
        while depth and k < len(src):
            depth += {"{": 1, "}": -1}.get(src[k], 0)
            k += 1
        sel = src[i:j].strip()
        if sel.startswith(_KEEP) and "gradio" not in sel:
            out.append(src[i:k].strip())
        i = k
    return "\n".join(out)


def _qr_src() -> str:
    for p in (QR_SVG,):
        if p.is_file():
            return "data:image/svg+xml;base64," + base64.b64encode(p.read_bytes()).decode("ascii")
    return ""


def tip_html() -> str:
    """Standard tip area, same markup/classes as the app footer (ui.tip_link_html): desktop = QR card + quote beside it,
    phones = tap pill + quote (pure CSS media query, no popups). Card tip = Stripe link. Empty when tips are off."""
    if not C.TIPS_ENABLED or not C.TIP_URL:
        return ""
    src = _qr_src()
    qr = (f"<figure class='mc-tip-qrbox'><span class='mc-tip-qrcard'><img src='{src}' width='112' height='112' "
          f"alt='QR code: tip @Tony_Omega with X Money'></span>"
          f"<figcaption>{html.escape(C.TIP_QR_CAPTION)}</figcaption></figure>") if src else ""
    card = (f"<a class='mc-tip-card' href='{html.escape(C.TIP_CARD_URL, quote=True)}' target='_blank' "
            f"rel='noopener'>{html.escape(C.TIP_CARD_LABEL)}</a>")
    return (f"<div class='mc-tip-foot mc-tip-unit' id='tip'>{qr}<div class='mc-tip-foot-text'>"
            f"<a class='mc-tip-link' href='{html.escape(C.TIP_URL, quote=True)}' target='_blank' rel='noopener' "
            f"aria-label='{html.escape(C.TIP_TITLE, quote=True)}'><span class='mc-tip-mark' aria-hidden='true'>$</span>"
            f"{html.escape(C.TIP_LABEL)}</a><i class='mc-tip-quote'>{html.escape(C.TIP_QUOTE)}</i>{card}</div></div>")


WIPE_JS = """<script>
document.querySelectorAll('.wipe').forEach(function(w){
  var r=w.querySelector('input'),v=w.querySelectorAll('video');
  r.addEventListener('input',function(){w.style.setProperty('--x',r.value+'%');});
  v[0].addEventListener('play',function(){v[1].currentTime=v[0].currentTime;v[1].play();});
  v[0].addEventListener('pause',function(){v[1].pause();});
  w.addEventListener('click',function(){v[0].paused?v[0].play():v[0].pause();});
});
</script>"""


def card_html(ex: dict[str, Any]) -> str:
    t, a, b = (html.escape(str(ex.get(k) or ""), quote=True) for k in ("title", "after", "before"))
    note = html.escape(str(ex.get("note") or ""))
    if b and PLACEHOLDER_BEFORE.search(b):
        b = ""  # placeholder / color-bars before: show the result alone, no fake wipe
    if b:
        media = (f"<div class='wipe'><video class='after' src='{a}' muted loop playsinline preload='metadata'></video>"
                 f"<video class='before' src='{b}' muted loop playsinline preload='metadata'></video>"
                 "<div class='bar'></div><input type='range' min='0' max='100' value='50' "
                 "aria-label='drag to compare'></div><div class='tag'>left: the original · right: mc · drag "
                 "to compare, click to play</div>")
    else:
        media = f"<div class='solo'><video src='{a}' controls muted loop playsinline preload='metadata'></video></div>"
    return f"<div class='card'><h3>{t}</h3>{media}<p>{note}</p></div>"


def build(examples: list[dict[str, Any]] | None = None, out_dir: Path | None = None,
          guide_text: str | None = None, repo_url: str = "") -> Path:
    out_dir = Path(out_dir or OUT)
    out_dir.mkdir(parents=True, exist_ok=True)
    exs = examples if examples is not None else DEFAULT_EXAMPLES
    repo = (f"<a href='{html.escape(repo_url, quote=True)}'>the repo</a>" if repo_url.startswith("https://")
            else "the repo")
    secs = uibits.guide_sections(guide_text)
    short = [re.sub(r"^\d+\.\s*", "", t) for _, t, _ in secs]  # no backslash inside an f-string (py3.10/3.11)
    toc = " · ".join(f"<a href='#guide-{a}'>{html.escape(n)}</a>" for (a, _, _), n in zip(secs, short))
    guide = "".join(f"<section id='guide-{a}' class='mc-guide-sec'><h4>{html.escape(t)}</h4>{uibits.md_html(b)}</section>"
                    for a, t, b in secs)
    page = f"""<!doctype html><html lang="en" data-theme="dark"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MC: Main Character · try it</title><style>{app_css()}{themes.logo()[1]}{PAGE_CSS}</style></head><body><main>
<header class="mc-header"><div class="mc-logo">{themes.logo()[0]}<span class="mc-credit">made by {html.escape(C.CREATOR_NAME)} · <a href='{html.escape(C.CREATOR_URL, quote=True)}'>{html.escape(C.CREATOR_HANDLE)}</a></span></div>
<div class="mc-header-right"><span><span class="mc-dot"></span><b>try it</b></span><span>{html.escape(C.LOCAL_NOTE)}</span>
<span class="mc-toggle" role="group" aria-label="color mode"><button type="button" data-m="dark" class="on" aria-pressed="true" onclick="mcSetTheme('dark')">● dark</button><button type="button" data-m="light" aria-pressed="false" onclick="mcSetTheme('light')">○ light</button></span></div></header>
<div class="hero"><div class="mc-kicker">{html.escape(C.KICKER)} · runs on your gaming pc</div><div class="mc-rule"></div>
<h1 class="mc-h1">{html.escape(C.HEADLINE)}<em>.</em></h1><div class="mc-tagline">{html.escape(C.TAG)}</div></div>
<p class="lede">you (or your avatar) in memes, dance clips, music videos. open weights. your own gpu. comfyui.
<b>no credits. no queue. nothing uploaded.</b> these are real outputs.</p>
<div class="cards">{''.join(card_html(e) for e in exs)}</div>
<h2>no gpu? try it anyway</h2>
<p>clone {repo}, run <code>./run.sh --mock</code> (windows: <code>.\\run.ps1 -Mock</code>). whole app,
placeholder media, every button real. nvidia gpu (tested on one rtx 5080 16 gb)? follow the readme setup.</p>
<h2>useful guide</h2><p class="toc">{toc}</p>{guide}
<p class="tag">only clips you have the rights to. only faces of people who are cool with it.</p>
<p class="tag">{html.escape(C.USE_DISCLAIMER)}</p>
{tip_html()}
<p class="tag">made by {html.escape(C.CREATOR_NAME)} · <a href='{html.escape(C.CREATOR_URL, quote=True)}'>{html.escape(C.CREATOR_HANDLE)}</a></p>
</main>{WIPE_JS}{THEME_JS}</body></html>"""
    path = out_dir / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--examples", type=Path, help="examples JSON (default: docs/try/examples.json if present)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--repo", default="", help="https link to the public repo (shown on the page)")
    a = ap.parse_args(argv)
    src = a.examples or (OUT / "examples.json")
    exs = json.loads(src.read_text(encoding="utf-8")) if src.is_file() else None
    print(build(exs, a.out, repo_url=a.repo))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
