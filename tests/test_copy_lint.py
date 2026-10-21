"""Copy lint: user-facing text has no em dash or en dash, and no hashtag except the contest's #OpenSourceHiggsfield.

Scanned: every string constant in copy.py, string constants in uibits.py, the suggested post text (share_text), the generated try-it page's visible text,
docs/GUIDE.md, docs/RECORDLINE.md, models.md, examples/*.md and README.md. No dash is allowed anywhere (the old tip quote is gone)."""

from __future__ import annotations

import html
import re
from pathlib import Path

from main_character_comfy import copy as C, trypage, uibits
from main_character_comfy.config import ROOT

DASHES = re.compile("[\u2014\u2013]")
HASHTAG = re.compile(r"(?<![\w&/#(`'\"=-])#([A-Za-z][A-Za-z0-9_]*)(?![\w*-])")  # not anchors like #guide-x
HEX = re.compile(r"^[0-9a-fA-F]{3,8}$")
ALLOWED_TAGS = {"OpenSourceHiggsfield"}
ALLOWED_PHRASES: tuple[str, ...] = ()


def lint(text: str, where: str) -> list[str]:
    for ok in ALLOWED_PHRASES:
        text = text.replace(ok, "")
    bad = [f"{where}: dash in ...{text[max(0, m.start() - 25):m.end() + 25]!r}" for m in DASHES.finditer(text)]
    bad += [f"{where}: hashtag #{m.group(1)}" for m in HASHTAG.finditer(text) if m.group(1) not in ALLOWED_TAGS and not HEX.match(m.group(1))]
    return bad


def _module_strings(mod) -> str:
    out = []
    for name, v in vars(mod).items():
        if name.startswith("_") or name.endswith("_CSS") or name.endswith("_JS"):
            continue
        vals = v if isinstance(v, (list, tuple)) else (v.values() if isinstance(v, dict) else [v])
        out += [x if isinstance(x, str) else " ".join(map(str, x)) for x in vals if isinstance(x, (str, tuple, list))]
    return "\n".join(out)


def _visible(page: str) -> str:
    page = re.sub(r"<(style|script)[^>]*>.*?</\1>", " ", page, flags=re.S)
    return html.unescape(re.sub(r"<[^>]+>", " ", page))


def test_lint_catches_known_bad_samples():
    assert lint("about 14\u201319 min", "s") and lint("fast \u2014 free", "s") and lint("post it #ad", "s")
    assert not lint("post it #OpenSourceHiggsfield", "s") and not lint(C.TIP_QUOTE, "s")
    assert not lint("see (#guide-source) and color #fff; `#guide-x`", "s")


def test_user_facing_copy_is_clean(tmp_path):
    bad = lint(_module_strings(C), "copy.py") + lint(_module_strings(uibits), "uibits.py")
    bad += lint(_visible(trypage.build(None, tmp_path).read_text(encoding="utf-8")), "try-it page")
    from main_character_comfy.pipeline import POST_HOOK, share_text
    bad += lint(POST_HOOK, "pipeline.POST_HOOK") + lint(share_text("me at 3am", "every group chat"), "share_text")
    for rel in ("docs/GUIDE.md", "README.md", "docs/RECORDLINE.md", "models.md", "examples/README.md", "examples/reference_motion.md"):
        bad += lint((ROOT / rel).read_text(encoding="utf-8"), rel)
    assert bad == []


def test_public_docs_never_name_private_test_clips(tmp_path):
    """docs/, README and the try-it page must not reference the private test clips or refs (contest rights audit)."""
    banned = re.compile(r"migos|tiptoe|tony_flag_chair", re.I)
    texts = {"try-it page": trypage.build(None, tmp_path).read_text(encoding="utf-8"),
             "README.md": (ROOT / "README.md").read_text(encoding="utf-8")}
    for p in (ROOT / "docs").rglob("*"):
        if p.is_file() and p.suffix.lower() in (".md", ".html", ".json", ".txt"):
            texts[str(p.relative_to(ROOT))] = p.read_text(encoding="utf-8", errors="ignore")
    assert [k for k, t in texts.items() if banned.search(t)] == []
