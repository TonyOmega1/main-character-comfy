"""Swappable visual themes.

The UI (ui.py) only emits semantic ``mc-*`` class names; everything visual
lives here. A theme is ``themes/<name>.css`` plus an entry in ``THEMES``.
``base.css`` holds layout-only rules shared by every theme.

* ``noir`` (default): the approved v3 "Noir Editorial" look. Red as the only
  accent, film grain, crop marks, halftone frames, deadpan lowercase mono labels.
  Dark and light palettes are CSS variables on ``html[data-theme=dark|light]``;
  the app always opens DARK (whatever the OS or browser prefers); light is opt-in
  via the header toggle, which stores the choice in ``localStorage`` (key ``mc-theme``).
* The logo is one swappable file, ``logo.html`` (or ``MC_LOGO=<path>``): the silhouette of the
  author's own portrait as a CSS mask (recolored by the palette) beside "MC." Brand assets
  (favicon, icons, title cards) live in ``assets/brand/``; see scripts/make_brand.py.

Pick a theme with ``MC_THEME=<name>`` or ``python app.py --theme <name>``. The
pre-mockup "default" look and the earlier sticker / chaos explorations are
retired; their names fall back to ``noir``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).parent
FONTS = HERE.parent / "assets" / "fonts"
BRAND = HERE.parent / "assets" / "brand"
FAVICON = BRAND / "favicon.ico"
DEFAULT_THEME = "noir"
RETIRED = {"default", "sticker", "chaos", "loud", "retro", "studio", "bright"}
STORAGE_KEY = "mc-theme"


@dataclass(frozen=True)
class ThemeSpec:
    name: str
    label: str
    modes: tuple[str, ...] = ("dark", "light")
    fonts: tuple[tuple[str, str], ...] = ()  # (family, file in assets/fonts)


THEMES: dict[str, ThemeSpec] = {
    "noir": ThemeSpec("noir", "Noir Editorial (v3, approved)",
                      fonts=(("Archivo", "Archivo-Variable.ttf"), ("Inter", "Inter-Variable.ttf"),
                             ("JetBrains Mono", "JetBrainsMono-Variable.ttf"), ("Anton", "Anton-Regular.ttf"))),
}


def theme_name(name: str | None = None) -> str:
    n = (name or os.environ.get("MC_THEME") or DEFAULT_THEME).strip().lower()
    return n if n in THEMES and (HERE / f"{n}.css").exists() else DEFAULT_THEME


def _file_url(path: Path) -> str:
    from urllib.parse import quote

    return "/gradio_api/file=" + quote(str(path.resolve()).replace("\\", "/"))


def font_css(spec: ThemeSpec) -> str:
    """@font-face rules for the bundled (SIL OFL) fonts, served locally: no font CDN, nothing leaves the PC."""
    rules = []
    for family, file in spec.fonts:
        if (FONTS / file).exists():
            rules.append(f"@font-face{{font-family:'{family}';src:url('{_file_url(FONTS / file)}') format('truetype');"
                         f"font-weight:100 900;font-display:swap}}")
    return "\n".join(rules)


def _data_uri(path: Path) -> str:
    import base64
    import mimetypes

    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def icon_links() -> str:
    """Browser/app icons beyond the favicon (served from assets/brand, which app.py allows)."""
    out = []
    for rel, n in (("icon", 32), ("icon", 192), ("apple-touch-icon", 180)):
        p = BRAND / f"icon-{n}.png"
        if p.exists():
            out.append(f'<link rel="{rel}" sizes="{n}x{n}" type="image/png" href="{_file_url(p)}">')
    return "\n".join(out)


def logo(path: str | Path | None = None) -> tuple[str, str]:
    """The swappable logo file -> (markup, css). Default: themes/logo.html; override with MC_LOGO."""
    p = Path(path or os.environ.get("MC_LOGO") or HERE / "logo.html")
    if not p.exists():
        p = HERE / "logo.html"
    raw = p.read_text(encoding="utf-8")
    raw = re.sub(r"<!--.*?-->", "", raw, flags=re.S)
    raw = re.sub(r"\{\{brand:([\w.-]+)\}\}", lambda m: _data_uri(BRAND / m.group(1)), raw)
    css = "\n".join(re.findall(r"<style[^>]*>(.*?)</style>", raw, flags=re.S))
    markup = re.sub(r"<style[^>]*>.*?</style>", "", raw, flags=re.S).strip()
    return markup, css


# Light/dark: dark by default, never follows the OS/browser. Light is opt-in via the header toggle (saved in localStorage).
TOGGLE_JS = """
<script>
(function(){
  var KEY = '%(key)s';
  function stored(){ try { var v = localStorage.getItem(KEY); return (v === 'dark' || v === 'light') ? v : null; } catch (e) { return null; } }
  var DEFAULT = 'dark';  // Noir Editorial: dark unless the user picked light on the toggle
  function current(){ return stored() || DEFAULT; }
  function paint(m){
    var root = document.documentElement;
    root.setAttribute('data-theme', m);
    root.style.colorScheme = m;
    if (document.body) { document.body.classList.toggle('dark', m === 'dark'); }
    document.querySelectorAll('.mc-toggle [data-m]').forEach(function(b){
      var on = b.getAttribute('data-m') === m;
      b.classList.toggle('on', on);
      b.setAttribute('aria-pressed', on ? 'true' : 'false');
      b.textContent = (on ? '\\u25CF ' : '\\u25CB ') + b.getAttribute('data-m');
    });
  }
  window.mcSetTheme = function(m){ try { localStorage.setItem(KEY, m); } catch (e) {} paint(m); };
  window.mcTheme = current;
  paint(current());
  // Gradio renders the header (and toggles body.dark itself) after load: re-paint when it shows up.
  var mo = new MutationObserver(function(){
    var t = document.querySelector('.mc-toggle [data-m]');
    if (t && !t.dataset.mcPainted) { t.dataset.mcPainted = '1'; paint(current()); }
    rotateLegal();
    if (document.body && document.body.classList.contains('dark') !== (current() === 'dark')) { paint(current()); }
  });
  // Fake legal line under the logo: one of the jokes from copy.py, picked per page load.
  function rotateLegal(){
    var el = document.querySelector('.mc-legal[data-jokes]');
    if (!el || el.dataset.mcRotated) { return; }
    el.dataset.mcRotated = '1';
    try { var j = JSON.parse(el.getAttribute('data-jokes')); el.textContent = j[Math.floor(Math.random() * j.length)]; } catch (e) {}
  }
  function start(){
    paint(current());
    rotateLegal();
    mo.observe(document.body, {childList: true, subtree: true, attributes: true, attributeFilter: ['class']});
  }
  // Gradio injects head scripts after the page loaded, so DOMContentLoaded may be long gone.
  if (document.readyState === 'loading') { document.addEventListener('DOMContentLoaded', start); } else { start(); }
})();
</script>
""" % {"key": STORAGE_KEY}


def toggle_html() -> str:
    """The header's dark/light switch (JS above keeps it in sync)."""
    return ("<span class='mc-toggle' role='group' aria-label='color mode'>"
            "<button type='button' data-m='dark' aria-pressed='true' onclick=\"mcSetTheme('dark')\">● dark</button>"
            "<button type='button' data-m='light' aria-pressed='false' onclick=\"mcSetTheme('light')\">○ light</button>"
            "</span>")


def load(name: str | None = None):
    """Return ``(gradio_theme, css, js_head)`` for the chosen theme."""
    import gradio as gr

    spec = THEMES[theme_name(name)]
    theme = gr.themes.Base(
        primary_hue="red", neutral_hue="stone", radius_size=gr.themes.sizes.radius_none,
        font=[gr.themes.Font("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=[gr.themes.Font("JetBrains Mono"), "ui-monospace", "monospace"],
    )
    _, logo_css = logo()
    css = "\n".join([font_css(spec), (HERE / "base.css").read_text(encoding="utf-8"),
                     (HERE / f"{spec.name}.css").read_text(encoding="utf-8"), logo_css,
                     # app-feel layer (mockup): structure and interactions only, same palette and fonts
                     (HERE / "app.css").read_text(encoding="utf-8") if (HERE / "app.css").exists() else ""])
    return theme, css, TOGGLE_JS + icon_links()
