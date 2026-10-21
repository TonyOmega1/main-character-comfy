"""QoL + borrow items (Oct 19): useful-guide panel, who's-who strip, '$0 · ~N min' buttons, preview -> HD, before/after
wipe, auto-rotate, face close-up refs, mask hygiene, vocals stem. CPU only."""

import json
import re
import stat
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from main_character_comfy import framing, graphs, media, stems, stylelock, uibits, ui
from main_character_comfy.media import run_ffmpeg, video_size
from main_character_comfy.pipeline import InputError, Request, check_request

ROOT = Path(__file__).resolve().parent.parent
LIVE = ROOT / "tests" / "fixtures" / "object_info_reference_live.json"


# ---------------- useful guide ----------------
def test_guide_has_every_anchor_and_every_warning_link_resolves():
    secs = uibits.guide_sections()
    assert [a for a, _, _ in secs] == list(uibits.GUIDE_ANCHORS)
    used = set()
    for p in (ROOT / "main_character_comfy").rglob("*.py"):
        used |= set(re.findall(r"#guide-([a-z0-9-]+)", p.read_text(encoding="utf-8")))
    assert used and used <= set(uibits.GUIDE_ANCHORS), used - set(uibits.GUIDE_ANCHORS)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/GUIDE.md" in readme


def test_guide_html_sections_lists_and_links():
    h = uibits.guide_html()
    for a in uibits.GUIDE_ANCHORS:
        assert f"id='guide-{a}'" in h and f"href='#guide-{a}'" in h
    assert "<ul><li>" in h and "<b>" in h and "<script" not in h.lower()
    # what the guide promises exists in the UI copy (no claims about controls that aren't there)
    for words in ("people engine", "split at cuts", "style lock", "make it HD", "record your own line"):
        assert words.lower() in h.lower()


def test_guide_markdown_is_escaped():
    secs = "### 1. X  ·  `#guide-x`\n\nhi <img src=x onerror=alert(1)> **b** - not a list\n\n- a\n- b *c*\n"
    h = uibits.guide_html(secs)
    assert "<img" not in h and "&lt;img" in h and "<ul><li>a</li><li>b <i>c</i></li></ul>" in h


def test_link_guide_turns_pipeline_refs_into_panel_links():
    m = "Heads-up: 2 cuts. set 'split at cuts' on (guide: #guide-cuts)"
    assert uibits.link_guide(m).endswith("([useful guide: cuts](#guide-cuts))")
    assert uibits.link_guide("see #guide-roles now") == "see [useful guide: roles](#guide-roles) now"
    assert uibits.link_guide("[x](#guide-roles)") == "[x](#guide-roles)"  # already a link: untouched


def test_stream_pins_guide_heads_ups(monkeypatch):
    from main_character_comfy.pipeline import Event

    monkeypatch.setattr(ui, "in_gpu_lane", lambda ev, mode, title="": ev)
    evs = [Event("prepare", "progress", 1.0, "Heads-up: one cut (guide: #guide-cuts)"),
           Event("swap", "start", message="Swapping"), Event("swap", "progress", 0.5, "half")]
    last = list(ui._stream(iter(evs), "swap"))[-1]
    status = last[ui.OUT_KEYS.index("status")]
    assert "](#guide-cuts)" in status and status.count("guide-cuts") == 1


# ---------------- who's who + moves + button labels ----------------
def test_driver_choices_never_offer_yourself():
    ch = uibits.driver_choices(3, 1)
    assert ch[0] == ("own moves", "") and [v for _, v in ch[1:]] == ["person:1", "person:3"]


def test_who_strip_says_who_becomes_who_and_whose_moves():
    h = uibits.who_strip_html(3, ["/x/friend.png", None, "/y/<me>.jpg"], ["", "", "person:1"], ["", "", "red cap"])
    assert h.count("mc-who-n") == 3 and "friend.png" in h and "unchanged" in h and "&lt;me&gt;.jpg" in h
    assert "follows person 1" in h and "red cap" in h and "#guide-roles" in h
    assert uibits.who_strip_html(0, []) == ""


def test_money_time_label():
    assert uibits.money_time_label("swap me in", 760) == "swap me in · $0 · ~11-15 min"
    assert uibits.money_time_label("go", 42) == "go · $0 · ~36-48 s"


def test_button_and_hint_share_one_estimate():
    """The swap button and the hint under the slider must show the same range (was '~5 min' vs '~5-6 min')."""
    from main_character_comfy.pipeline import eta_range_text
    for secs in (42, 300, 330, 760, 1500):
        assert uibits.money_time_label("swap me in", secs).endswith(eta_range_text(secs))
    assert uibits.money_time_label("go", None) == "go · $0"


def test_swap_button_label_scales_with_people():
    one = ui.swap_button_label(5, "standard", False, "auto", 1)
    two = ui.swap_button_label(5, "standard", False, "auto", 2)
    assert "$0" in one and "min" in one
    n1, n2 = (int(re.search(r"~(\d+)-\d+ min", s).group(1)) for s in (one, two))
    assert n2 >= 2 * n1 - 1
    assert ui.swap_button_label(5, "lite", False, "auto").endswith("· $0")  # gated tier: no fake ETA


def test_ui_builds_with_guide_panel_and_new_controls():
    demo = ui.build_ui("mock")
    cfg = json.dumps(demo.config, default=str, ensure_ascii=False)
    for s in ("mc-guide", "useful guide", "help: quick tour", "style lock · pending a real test", "split at cuts",
              "people engine", "make it HD", "before / after", "Re-dub a line", "person 2: moves", "$0"):
        assert s in cfg, s


# ---------------- preview -> HD, before/after ----------------
def _clip(path, size="320x240", seconds=1.0, colour=None, audio=False):
    src = f"color=c={colour}:size={size}:rate=16:duration={seconds}" if colour else f"testsrc=size={size}:rate=16:duration={seconds}"
    args = ["-f", "lavfi", "-i", src]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-c:a", "aac", "-shortest"]
    run_ffmpeg([*args, "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)])
    return path


def test_pending_and_wipe_buttons(tmp_path):
    top = tmp_path / "run"
    (top / "_work" / "inputs").mkdir(parents=True)
    assert ui.pending_take(top) is None and ui.prepared_clip_of(top) is None
    hd, wipe = ui.after_video_run(str(top), None)
    assert hd["visible"] is False and wipe["visible"] is False
    (top / "_work" / "pending_upscale.json").write_text("{}")
    hd, wipe = ui.after_video_run(str(top), "x.mp4")
    assert hd["visible"] is True and wipe["visible"] is False
    (top / "_work" / "pending_upscale.json").unlink()
    _clip(top / "_work" / "inputs" / "clip.mp4")
    hd, wipe = ui.after_video_run(str(top), "x.mp4")
    assert hd["visible"] is False and wipe["visible"] is True


def test_wipe_compare_keeps_result_size_audio_and_shows_both(tmp_path):
    before = _clip(tmp_path / "b.mp4", "240x320", 2.0, colour="red")
    after = _clip(tmp_path / "a.mp4", "480x640", 2.0, colour="blue", audio=True)
    out = media.wipe_compare(before, after, tmp_path / "w.mp4", period=4.0)
    assert video_size(out) == (480, 640) and media.has_audio_stream(out)
    assert abs(media.media_duration(out) - 2.0) < 0.25
    fr = tmp_path / "f.png"
    run_ffmpeg(["-ss", "0.0", "-i", str(out), "-frames:v", "1", str(fr)])
    a = np.asarray(Image.open(fr).convert("RGB")).astype(int)
    left, right = a[:, 20].mean(0), a[:, -20].mean(0)  # t=0: divider mid-frame, original (red) on the left
    assert left[0] > 150 and left[2] < 100 and right[2] > 150 and right[0] < 100
    assert a[:, 239:242].min(axis=2).mean() > 200  # white divider


def test_wipe_for_without_inputs_explains(tmp_path):
    _, msg = ui.wipe_for(str(tmp_path), None)
    assert msg.startswith("⚠️")


# ---------------- rotate ----------------
def test_phone_rotation_tag_is_honoured_and_manual_rotate_works(tmp_path):
    land = _clip(tmp_path / "land.mp4", "320x240")
    tagged = tmp_path / "tagged.mp4"
    run_ffmpeg(["-display_rotation:v:0", "90", "-i", str(land), "-c", "copy", str(tagged)])
    assert video_size(tagged) == (240, 320)
    c = media.prepare_clip(tagged, tmp_path / "p1", max_seconds=1)
    assert (c.width, c.height) == (480, 832) and video_size(c.video) == (480, 832)
    c2 = media.prepare_clip(land, tmp_path / "p2", max_seconds=1, rotate=270)
    assert (c2.width, c2.height) == (480, 832)
    c3 = media.prepare_clip(land, tmp_path / "p3", max_seconds=1, rotate=180)
    assert (c3.width, c3.height) == (832, 480)
    with pytest.raises(ValueError):
        media.rotate_filter(45)


def test_request_validates_rotate_and_closeup():
    from main_character_comfy.config import settings

    with pytest.raises(InputError, match="rotate"):
        check_request(Request(mode="swap", rotate=45), settings())
    with pytest.raises(InputError, match="ref_closeup"):
        check_request(Request(mode="swap", ref_closeup="always"), settings())
    assert Request().rotate == 0 and Request().ref_closeup == "off"


# ---------------- face close-up from full-body refs ----------------
def test_closeup_reference_crops_full_body_refs_only(tmp_path):
    ref = tmp_path / "full.png"
    Image.new("RGBA", (600, 1500), (10, 20, 30, 255)).save(ref)
    small = lambda rgb: {"x": 260.0, "y": 100.0, "w": 80.0, "h": 100.0}  # face = 6.7 % of the height
    got = framing.closeup_reference(ref, tmp_path / "cu.png", "cu", aspect=0.6, find_face=small)
    assert got and got["face_frac"] < framing.CLOSEUP_FACE_MAX
    im = Image.open(got["path"])
    assert im.mode == "RGBA" and im.height < 600 and im.height >= 300  # head and shoulders, alpha kept
    assert framing.closeup_reference(ref, tmp_path / "w.png", "wide", find_face=small) is None
    assert framing.closeup_reference(ref, tmp_path / "m.png", "medium", find_face=small) is None
    big = lambda rgb: {"x": 100.0, "y": 100.0, "w": 400.0, "h": 600.0}  # already a close-up
    assert framing.closeup_reference(ref, tmp_path / "b.png", "cu", find_face=big) is None
    assert framing.closeup_reference(ref, tmp_path / "n.png", "ecu", find_face=lambda rgb: None) is None


# ---------------- mask hygiene ----------------
def _models():
    from main_character_comfy.config import settings

    return settings().models


def test_mask_hygiene_is_off_by_default_and_on_with_style_lock():
    base = graphs.swap_graph("p", models=_models(), frames=33, person_mask=True)
    assert not any(n.get("_meta", {}).get("title", "").startswith("Binary person mask")
                   for n in base.graph.values())
    g = graphs.swap_graph("p", models=_models(), frames=33, person_mask=True, mask_threshold=0.5)
    th = [n for n in g.graph.values() if n.get("_meta", {}).get("title", "") == "Binary person mask (hygiene)"]
    assert len(th) == 1 and th[0]["class_type"] == "ThresholdMask" and th[0]["inputs"]["value"] == 0.5
    grow = [n for n in g.graph.values() if n["class_type"] == "GrowMask"]
    tid = next(k for k, n in g.graph.items() if n is th[0])
    assert any(n["inputs"]["mask"][0] == tid for n in grow)  # threshold feeds the grow (before, not after)
    from main_character_comfy.validate import validate_graph

    assert not validate_graph(g.graph, json.loads(LIVE.read_text(encoding="utf-8")))
    assert stylelock.settings_for("on", ["photo"]).mask_threshold == stylelock.MASK_THRESHOLD
    assert stylelock.settings_for("off", ["avatar"]).mask_threshold is None


# ---------------- vocals stem ----------------
def test_vocals_stem_via_demucs_command(tmp_path, monkeypatch):
    fake = tmp_path / "fake_demucs.py"
    fake.write_text(
        "import sys, pathlib, shutil\n"
        "a = sys.argv[1:]\n"
        "assert '--two-stems=vocals' in a\n"
        "out = pathlib.Path(a[a.index('-o') + 1]); src = pathlib.Path(a[-1]); n = a[a.index('-n') + 1]\n"
        "d = out / n / src.stem; d.mkdir(parents=True)\n"
        "shutil.copy(src, d / 'vocals.wav'); shutil.copy(src, d / 'no_vocals.wav')\n")
    song = tmp_path / "song_mix.wav"
    run_ffmpeg(["-f", "lavfi", "-i", "sine=frequency=330:duration=1", str(song)])
    monkeypatch.setenv("MC_AUTO_STEMS", "1")
    monkeypatch.setenv("MC_DEMUCS_CMD", f'"{sys.executable}" "{fake}"')
    got = stems.separate_vocals(song, tmp_path / "stems")
    assert got is not None and got.name == "vocals.wav" and got.parent.name == "song_mix"
    monkeypatch.setenv("MC_AUTO_STEMS", "0")
    assert stems.separator_command() is None and stems.separate_vocals(song, tmp_path / "s2") is None


def test_vocals_stem_failure_falls_back(tmp_path, monkeypatch):
    monkeypatch.setenv("MC_AUTO_STEMS", "1")
    monkeypatch.setenv("MC_DEMUCS_CMD", f'"{sys.executable}" -c "import sys; sys.exit(3)"')
    song = tmp_path / "s.wav"
    run_ffmpeg(["-f", "lavfi", "-i", "sine=frequency=330:duration=1", str(song)])
    assert stems.separate_vocals(song, tmp_path / "o") is None


# ---------------- try-it page ----------------
def test_try_page_builds_static_with_wipe_and_guide(tmp_path):
    from main_character_comfy import trypage

    exs = [{"title": "Duo <swap>", "after": "../demo/a.mp4", "before": "../demo/b.mp4", "note": "n"},
           {"title": "Solo", "after": "../demo/c.mp4"}]
    page = trypage.build(exs, tmp_path, repo_url="javascript:alert(1)").read_text(encoding="utf-8")
    assert page.count("class='wipe'") == 1 and "Duo &lt;swap&gt;" in page and "type='range'" in page
    assert "javascript:" not in page and "the repo" in page
    for a in uibits.GUIDE_ANCHORS:
        assert f"id='guide-{a}'" in page
    assert "http://" not in page.replace("https://", "")  # no external trackers / mixed content
    ok = trypage.build(exs, tmp_path / "b", repo_url="https://example.org/mc").read_text(encoding="utf-8")
    assert "href='https://example.org/mc'" in ok


def test_pipeline_closeup_hook_only_when_asked(tmp_path, monkeypatch):
    from main_character_comfy import refcheck
    from main_character_comfy.pipeline import Pipeline

    clip_path = _clip(tmp_path / "c.mp4", "240x320")
    clip = media.PreparedClip(clip_path, None, 240, 320, 16, 16, 1.0)
    ref = tmp_path / "full.png"
    Image.new("RGB", (600, 1500), (90, 90, 90)).save(ref)
    person = media.Person(mask=tmp_path / "m.png", box=(40, 10, 200, 310), cx=0.5, cy=0.5, area=0.4)
    monkeypatch.setattr(framing, "classify_person", lambda rgb, box: {"framing": "cu", "face_frac": 0.3, "face": None})
    monkeypatch.setattr(refcheck, "find_face", lambda rgb: {"x": 260.0, "y": 100.0, "w": 80.0, "h": 100.0})
    p = object.__new__(Pipeline)
    rec: dict = {}
    assert p._closeup_ref(Request(mode="swap"), clip, person, ref, tmp_path, "p1", rec) == ref and not rec
    got = p._closeup_ref(Request(mode="swap", ref_closeup="auto"), clip, person, ref, tmp_path, "p1", rec)
    assert got != ref and Path(got).is_file() and rec["ref_closeup"]["p1"]["shot"] == "cu"
    monkeypatch.setattr(framing, "classify_person", lambda rgb, box: (_ for _ in ()).throw(RuntimeError("no cv2")))
    rec2: dict = {}
    assert p._closeup_ref(Request(mode="swap", ref_closeup="auto"), clip, person, ref, tmp_path, "p2", rec2) == ref
    assert "error" in rec2["ref_closeup"]["p2"]


def test_try_page_has_no_tip_area(tmp_path):
    from main_character_comfy import trypage

    page = trypage.build([], tmp_path).read_text(encoding="utf-8")
    assert "mc-tip-unit" not in page and "x.com/i/money" not in page and "stripe" not in page.lower()
    assert "Mewtwo" not in page and "class='mc-tip-qrcard'" not in page
    assert "Oroku Saki" in page and "@Tony_Omega" in page and "popover" not in page



def test_try_page_noir_default_and_no_fake_wipe(tmp_path):
    from main_character_comfy import trypage

    exs = [{"title": "bars", "after": "../demo/a.mp4", "before": "../demo/color_bars.mp4"},
           {"title": "real", "after": "../demo/b.mp4", "before": "../demo/b_src.mp4"}]
    page = trypage.build(exs, tmp_path).read_text(encoding="utf-8")
    assert page.count("class='wipe'") == 1 and 'data-theme="dark"' in page and "mc-wordmark" in page
    assert "no credits. no queue. nothing uploaded." in page and "mc-toggle" in page
