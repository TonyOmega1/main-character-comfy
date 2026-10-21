"""v3 "Noir Editorial" theme: default theme, dark/light toggle, swappable logo, deadpan copy."""

import re
from pathlib import Path

from main_character_comfy import copy as C
from main_character_comfy import themes, ui


def test_noir_is_default_and_old_themes_retired(monkeypatch):
    monkeypatch.delenv("MC_THEME", raising=False)
    assert themes.DEFAULT_THEME == "noir" and themes.theme_name() == "noir"
    assert set(themes.THEMES) == {"noir"}
    for old in ("default", "sticker", "chaos"):
        assert themes.theme_name(old) == "noir"
        assert not (themes.HERE / f"{old}.css").exists()
    monkeypatch.setenv("MC_THEME", "sticker")
    assert themes.theme_name() == "noir"


def test_both_palettes_and_single_red_accent():
    _, css, _ = themes.load()
    for mode in ("dark", "light"):
        block = re.search(r'\[data-theme="%s"\]\{(.*?)\}' % mode, css, re.S).group(1)
        assert "--bg:" in block and "--ink:" in block and "--accent:" in block
    assert "--accent:#ff2a2a" in css and "--accent:#e01e1e" in css
    # grain, crop marks / halftone frames, picked tile state
    assert "body::after" in css and "feTurbulence" in css
    assert ".mc-frame::before" in css and ".mc-frame::after" in css and "radial-gradient" in css
    assert "picked" in css and ".mc-format-grid label:hover" in css
    # Gradio's own variables follow the palette
    assert "--body-background-fill:var(--bg)" in css and "--button-primary-background-fill:var(--accent)" in css


def test_toggle_persists_and_defaults_to_dark():
    _, css, head = themes.load()
    assert "localStorage.setItem(KEY" in head and "localStorage.getItem(KEY" in head
    assert f"var KEY = '{themes.STORAGE_KEY}'" in head
    # dark by default whatever the OS/browser prefers; light only when picked on the toggle
    assert "var DEFAULT = 'dark'" in head and "stored() || DEFAULT" in head
    assert "prefers-color-scheme" not in head and "matchMedia" not in head
    # the CSS default (no data-theme yet, before the script runs) is the dark palette too
    root = re.search(r':root,\[data-theme="dark"\]\{(.*?)\}', css, re.S).group(1)
    assert "--bg:#0a0a0a" in root and "color-scheme:dark" in root
    assert "prefers-color-scheme" not in css
    assert "window.mcSetTheme" in head and "setAttribute('data-theme'" in head
    assert "classList.toggle('dark'" in head  # keeps Gradio's own dark class in sync
    t = themes.toggle_html()
    assert "mcSetTheme('dark')" in t and "mcSetTheme('light')" in t
    assert "data-m='dark'" in t and "data-m='light'" in t and "aria-pressed" in t
    hdr = ui.header_html("lite")
    assert "mc-toggle" in hdr and "gpu: lite" in hdr and C.LOCAL_NOTE in hdr


def test_toggle_runs_in_a_dom(monkeypatch):
    """Execute the head script against a tiny fake DOM: stored choice wins, else DARK (even on a light OS)."""
    import shutil
    import subprocess

    node = shutil.which("node")
    if not node:
        import pytest

        pytest.skip("node not installed")
    js = themes.TOGGLE_JS.replace("<script>", "").replace("</script>", "")
    harness = r"""
    function run(stored, prefersLight){
      var store = {}; if (stored) store['mc-theme'] = stored;
      var attrs = {}, bodyCls = new Set(), btns = ['dark','light'].map(function(m){
        return {a:{'data-m':m}, cls:new Set(), textContent:'', dataset:{},
                getAttribute:function(k){return this.a[k]}, setAttribute:function(k,v){this.a[k]=v},
                classList:{toggle:function(c,on){}}};
      });
      global.localStorage = {getItem:function(k){return k in store ? store[k] : null}, setItem:function(k,v){store[k]=v}};
      global.window = global;
      global.matchMedia = function(q){ return {matches: prefersLight, addEventListener:function(){}}; };
      global.MutationObserver = function(){ this.observe = function(){}; };
      global.document = {documentElement:{style:{}, setAttribute:function(k,v){attrs[k]=v}},
        body:{classList:{toggle:function(c,on){ on ? bodyCls.add(c) : bodyCls.delete(c); }, contains:function(c){return bodyCls.has(c)}}},
        querySelectorAll:function(){return btns}, querySelector:function(){return btns[0]}, addEventListener:function(){}};
      eval(JS);
      var first = attrs['data-theme'];
      window.mcSetTheme('light');
      return [first, attrs['data-theme'], store['mc-theme'], bodyCls.has('dark'), btns[1].a['aria-pressed'], btns[1].textContent];
    }
    console.log(JSON.stringify([run(null, true), run(null, false), run('dark', true), run('light', false)]));
    """
    out = subprocess.run([node, "-e", "var JS = " + repr(js) + ";\n" + harness], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    import json

    sys_light, sys_dark, stored_dark, stored_light = json.loads(out.stdout)
    assert sys_light[0] == "dark" and sys_dark[0] == "dark" and stored_dark[0] == "dark"  # light OS still opens dark
    assert stored_light[0] == "light"  # opt-in light survives a reload
    for r in (sys_light, sys_dark, stored_dark, stored_light):
        assert r[1:5] == ["light", "light", False, "true"] and r[5].endswith("light")


def test_logo_is_one_swappable_file(tmp_path, monkeypatch):
    mark, css = themes.logo()
    assert "MC" in mark and "mc-wordmark" in mark and "<style" not in mark and "{{brand:" not in css
    # the figure is a single-color mask painted with the palette's ink, so the toggle flips it
    assert "mask:url(\"data:image/png;base64," in css and "background:currentColor" in css
    assert ".mc-wordmark" in themes.load()[1] and "var(--accent)" in css  # the red period
    alt = tmp_path / "logo.html"
    alt.write_text("<style>.x{color:var(--ink)}</style><img class='x' alt='MC' src='logo.svg'>", encoding="utf-8")
    monkeypatch.setenv("MC_LOGO", str(alt))
    assert "logo.svg" in ui.header_html() and ".x{" in themes.load()[1]


def test_ui_says_what_mc_means():
    hdr = ui.header_html()
    assert ">main character<" in hdr and C.EXPANSION == "main character"
    assert C.EXPANSION_LINE == "mc = main character." and C.EXPANSION_LINE in hdr
    assert 'aria-label="MC: Main Character"' in hdr
    demo = ui.build_ui("mock")
    assert demo.title == C.APP_NAME == "MC: Main Character"
    from pathlib import Path

    readme = (Path(themes.__file__).parents[2] / "README.md").read_text(encoding="utf-8")
    assert "MC = Main Character" in readme.splitlines()[0]


def test_fake_legal_line_is_a_labelled_joke():
    import html as h
    import json

    assert 5 <= len(C.LEGAL_JOKES) <= 6 and C.LEGAL_JOKE in C.LEGAL_JOKES
    assert C.PATENT_JOKE == "Patent pending. (It is not.)"
    assert not any("mit" in j.split() or "mit licensed" in j for j in C.LEGAL_JOKES)  # the real license is Apache-2.0
    assert "patent pending" not in " ".join(C.LEGAL_JOKES).lower()  # patent line is fixed, not in the rotator
    assert "joke" in C.LEGAL_DISCLAIMER and "apache-2.0" in C.LEGAL_DISCLAIMER
    hdr = ui.header_html()
    assert h.escape(C.LEGAL_JOKE) in hdr and h.escape(C.PATENT_JOKE) in hdr
    assert h.escape(C.LEGAL_DISCLAIMER, quote=True) in hdr  # tooltip on the under-logo lines
    assert "class='mc-patent'" in hdr
    attr = hdr.split('data-jokes="')[1].split('"')[0]
    assert json.loads(h.unescape(attr)) == C.LEGAL_JOKES
    foot = ui.footer_html()
    assert C.LEGAL_DISCLAIMER not in foot  # not duplicated in the footer
    assert C.PATENT_JOKE not in foot
    assert "made by Oroku Saki" in hdr  # credit lives under the logo
    assert "made by Oroku Saki" not in foot
    assert "rotateLegal" in themes.TOGGLE_JS and "Math.random()" in themes.TOGGLE_JS


def test_brand_assets_and_favicon():
    from PIL import Image

    b = themes.BRAND
    for f in ("mc_logo_dark.png", "mc_logo_light.png", "mc_figure_mask.png", "mc_figure_mask_160.png",
              "title_card_dark.png", "title_card_light.png", "icon-512.png"):
        assert (b / f).exists(), f
    ico = Image.open(themes.FAVICON)
    assert {(16, 16), (32, 32)} <= set(ico.info["sizes"])
    m = Image.open(b / "mc_figure_mask_160.png")
    assert m.mode == "RGBA" and m.height == 160
    for n in (16, 32):  # the head-and-shoulders icon is readable: real figure pixels, not a sliver
        px = Image.open(b / f"icon-{n}.png").convert("L")
        light = sum(1 for v in px.getdata() if v > 160) / (n * n)
        assert 0.25 < light < 0.75
    _, _, head = themes.load()
    assert 'rel="apple-touch-icon"' in head and "icon-180.png" in head
    import inspect

    import app  # noqa: F401  (launch args)

    src = inspect.getsource(app)
    assert "favicon_path=str(themes.FAVICON)" in src and "themes.BRAND" in src


def test_fonts_are_bundled_and_served_locally():
    _, css, _ = themes.load()
    for fam, f in themes.THEMES["noir"].fonts:
        assert (themes.FONTS / f).stat().st_size > 50_000
        assert f"font-family:'{fam}'" in css
    assert "fonts.googleapis" not in css and "/gradio_api/file=" in css
    for d in ("archivo", "inter", "jetbrainsmono"):
        assert "SIL OPEN FONT LICENSE" in (themes.FONTS / f"OFL-{d}.txt").read_text(encoding="utf-8").upper()


def test_deadpan_copy_and_every_feature_on_the_main_page():
    assert C.PRIMARY_CTA == "make it" and C.STEP_FACE == "face"
    assert ui.step_label(1, C.STEP_FACE) == "<div class='mc-lab'><b>01</b>face</div>"
    assert "<em>.</em>" in ui.hero_html()
    demo = ui.build_ui("mock")
    labels = {getattr(b, "label", None) for b in demo.blocks.values()}
    for tab in (C.TAB_MEME, C.TAB_VIBE, C.TAB_SWAP, C.TAB_PACK, C.TAB_MORE, C.TAB_SETUP):
        assert tab in labels
    values = [getattr(b, "value", None) for b in demo.blocks.values()]
    assert C.PRIMARY_CTA in values and C.PACK_CTA in values
    # swap / vibe buttons carry the price and time: "swap me in · $0 · ~5 min"
    texts = [v for v in values if isinstance(v, str)]
    for cta in (C.SWAP_CTA, C.VIBE_CTA):
        assert any(v.startswith(cta + " · $0") for v in texts), cta


def test_creator_credit_link():
    import html as h

    hdr = ui.header_html()
    foot = ui.footer_html()
    assert "made by Oroku Saki · " in hdr
    assert ("<a href='https://x.com/Tony_Omega' target='_blank' rel='noopener'>@Tony_Omega</a>") in hdr
    assert hdr.count("x.com/Tony_Omega") == 1
    assert "made by Oroku Saki" not in foot  # not duplicated in the footer
    # credit sits with the logo block, ahead of the joke legal lines
    assert hdr.index("mc-credit") < hdr.index("mc-legal")
    assert hdr.index("mc-wordmark") < hdr.index("mc-credit")
    head = ui.credit_head()
    assert '<meta name="twitter:creator" content="@Tony_Omega">' in head
    import json

    ld = json.loads(head.split('application/ld+json">')[1].split("</script>")[0])
    assert ld["author"]["@type"] == "Person" and ld["author"]["name"] == "Oroku Saki"
    assert ld["author"]["sameAs"] == "https://x.com/Tony_Omega"
    _, css, _ = themes.load()
    assert ".mc-credit a, a.mc-feedback{color:var(--ink)" in css  # readable in dark and light
    assert ".mc-logo .mc-credit{" in css
    demo = ui.build_ui("mock")
    assert "twitter:creator" in demo.head
    from main_character_comfy import __author__

    assert __author__.startswith("Oroku Saki (@Tony_Omega")
    assert h.escape("Oroku Saki") in hdr

def test_feedback_mailto_in_footer_and_readme():
    import html as h
    from pathlib import Path
    from urllib.parse import quote, urlparse, parse_qs

    assert C.FEEDBACK_LABEL == "send feedback"
    assert C.FEEDBACK_EMAIL == "oroku.sakis.homies+mc@mail.grokbot.com"
    assert C.FEEDBACK_SUBJECT == "MC (main-character-comfy) feedback"
    body_l = C.FEEDBACK_BODY.lower()
    for needle in ("mc version", "gpu tier", "what happened", "suggested fix", "personal info"):
        assert needle in body_l

    href = ui.feedback_mailto()
    assert href.startswith(f"mailto:{C.FEEDBACK_EMAIL}?")
    assert f"subject={quote(C.FEEDBACK_SUBJECT)}" in href
    assert f"body={quote(C.FEEDBACK_BODY)}" in href
    qs = parse_qs(urlparse(href).query)
    assert qs["subject"] == [C.FEEDBACK_SUBJECT] and qs["body"] == [C.FEEDBACK_BODY]

    foot = ui.footer_html()
    hdr = ui.header_html()
    assert C.FEEDBACK_LABEL in foot and "mc-feedback" in foot
    assert h.escape(href, quote=True) in foot
    assert "made by Oroku Saki" not in foot
    assert C.FEEDBACK_LABEL not in hdr  # feedback stays in the footer only
    _, css, _ = themes.load()
    assert "a.mc-feedback" in css and "var(--ink)" in css  # readable in dark and light

    readme = (Path(themes.__file__).parents[2] / "README.md").read_text(encoding="utf-8")
    assert "## Feedback" in readme and C.FEEDBACK_EMAIL in readme



# --------------------------------------------------------------------------- tips (X Money)
def test_tips_are_off_in_the_contest_build(monkeypatch):
    """Contest build: no tip pill, QR, X Money or card link anywhere (the extras/tip-qr branch has them)."""
    from main_character_comfy import ui
    from main_character_comfy import copy as C
    monkeypatch.setenv("MC_TIP_CARD_URL", "https://buy.stripe.com/test_123")
    assert C.TIPS_ENABLED is False and C.TIP_URL == "" and C.TIP_CARD_URL == "" and C.TIP_QUOTE == ""
    assert ui.tip_html() == "" and ui.tip_link_html() == ""
    hdr, foot = ui.header_html(), ui.footer_html()
    for part in (hdr, foot):
        assert "mc-tip" not in part and "x.com/i/money" not in part and "stripe" not in part.lower()
        assert "Mewtwo" not in part
    assert "mc-feedback" in foot and "mc-toggle" in hdr
    assert not (Path(ui.__file__).parent / "assets" / "brand" / "xmoney-qr.svg").exists()





def test_tip_styles_are_theme_aware():
    css = (Path(__file__).parent.parent / "main_character_comfy" / "themes" / "noir.css").read_text()
    block = css[css.index(".mc-tip{"):css.index("@media (max-width:700px)")]
    assert "var(--accent)" in block and "var(--ink)" in block
    assert "var(--tip-muted)" in block and "var(--tip-quote)" in block and "var(--tip-accent)" in block
    assert "var(--tip-on-accent)" in block
    # palette variables only (dark/light both work), except the QR's white card so it scans in dark mode
    assert block.replace("background:#ffffff", "").count("#") == 0


def test_tip_css_kept_for_later_but_nothing_floats():
    """Contest build has no tip markup; the tip CSS stays (for the extras/tip-qr branch) and never floats."""
    css = (Path(__file__).parent.parent / "main_character_comfy" / "themes" / "noir.css").read_text()
    desk = css[css.index("@media (max-width:700px), (hover:none) and (pointer:coarse)"):]
    desk = desk[:desk.index("\n}") + 2]
    tips = css[css.index(".mc-tip{"):css.index("@media (max-width:700px)")] + desk
    assert "position:fixed" not in tips and "position:absolute" not in tips  # nothing floats over content



def test_no_tip_quote_or_dash_in_copy():
    from main_character_comfy import copy as C
    assert "Mewtwo" not in Path(C.__file__).read_text(encoding="utf-8")




def test_tip_text_meets_wcag_aa_contrast():
    """Tip pill mark, tip links/caption, and tip quote hit WCAG AA 4.5:1 on --bg (light + dark)."""
    import re as _re

    def _lin(c):
        c = c / 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    def _lum(hx):
        hx = hx.lstrip("#")
        r, g, b = int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)
        return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)

    def _ratio(fg, bg):
        a, b = _lum(fg), _lum(bg)
        return (max(a, b) + 0.05) / (min(a, b) + 0.05)

    css = (Path(__file__).parent.parent / "main_character_comfy" / "themes" / "noir.css").read_text()

    def _palette(mode):
        block = _re.search(r'\[data-theme="%s"\]\{(.*?)\}' % mode, css, _re.S).group(1)
        out = {}
        for m in _re.finditer(r"--([a-z0-9-]+):#([0-9a-fA-F]{6})", block):
            out[m.group(1)] = m.group(2).lower()
        return out

    # Before (legacy faint / light accent) failed AA; after tip-* tokens pass.
    before = {
        "dark": {"quote": ("55534e", "0a0a0a", 2.58), "mark": ("ff2a2a", "0a0a0a", 5.30)},
        "light": {"quote": ("8f897e", "efebe3", 2.92), "mark": ("e01e1e", "efebe3", 4.04)},
    }
    for mode, cases in before.items():
        for name, (fg, bg, expected) in cases.items():
            assert abs(_ratio(fg, bg) - expected) < 0.05, (mode, name, _ratio(fg, bg))

    for mode in ("dark", "light"):
        pal = _palette(mode)
        bg = pal["bg"]
        # tip muted (QR caption, tip-with-card, desktop tip link)
        assert _ratio(pal["tip-muted"], bg) >= 4.5
        # tip quote (extras/tip-qr branch)
        assert _ratio(pal["tip-quote"], bg) >= 4.5
        # $ mark on tip pill
        assert _ratio(pal["tip-accent"], bg) >= 4.5
        # tip pill label uses ink
        assert _ratio(pal["ink"], bg) >= 4.5
        # hover fill still uses --accent; tip-on-accent text must clear 4.5:1 on it
        assert _ratio(pal["tip-on-accent"], pal["accent"]) >= 4.5
        # tip tokens stay the same hue family (warm grey / red)
        assert pal["tip-muted"] == pal["muted"]
        assert "tip-quote" in pal and "tip-accent" in pal
