"""Feature-parity guard: no redesign may silently drop a user-facing feature.

Builds the UI in mock mode and checks that every user-facing feature still has its component
(by stable elem_id) and, where it is interactive, an event handler wired to it; plus the backend function behind it.
A new layout (e.g. feat/app-feel-mock) passes only if it keeps every elem_id below and wires the same triggers."""

from __future__ import annotations

import importlib
import json

import pytest

from main_character_comfy import ui

# feature -> (elem_id, trigger or None for display-only)
COMPONENTS = {
    "header (logo, credit, tip pill, dark/light toggle)": ("mc-header", None),
    "quick tour: open": ("mc-help-tour", "click"),
    "quick tour: next": ("mc-tour-next", "click"),
    "quick tour: back": ("mc-tour-back", "click"),
    "quick tour: skip": ("mc-tour-skip", "click"),
    "quick tour card": ("mc-tour", None),
    "useful guide: open": ("mc-guide-open", "click"),
    "useful guide: close": ("mc-guide-close", "click"),
    "useful guide panel": ("mc-guide", None),
    "gpu tier picker": ("mc-tier", "change"),
    "gpu tier: try 16 GB anyway": ("mc-tier-heavy", "change"),
    "connection / mock banner": ("mc-conn", None),
    "meme: face photo": ("mc-meme-photo", "change"),
    "meme: format picker": ("mc-meme-format", "change"),
    "meme: make it": ("mc-go-meme", "click"),
    "swap: fetch clip by link (byo clip)": ("mc-swap-fetch", "click"),
    "swap: clip": ("mc-swap-clip", None),
    "swap: photo + reference check": ("mc-swap-photo", "change"),
    "swap: reference check note": ("mc-swap-refcheck", None),
    "who's-who: find people": ("mc-people-find", "click"),
    "who's-who strip": ("mc-who", None),
    "$0 · ~N min swap button": ("mc-go-swap", "click"),
    "eta hint under the slider": ("mc-swap-eta", None),
    "swap length (drives eta + button)": ("mc-swap-len", "change"),
    "preview first, HD later": ("mc-swap-defer", None),
    "people engine": ("mc-swap-engine", None),
    "split at cuts": ("mc-swap-split", None),
    "style lock": ("mc-stylelock", None),
    "close-up refs": ("mc-closeup", None),
    "long group chunks": ("mc-chunks", None),
    "vibe: fetch clip by link": ("mc-vibe-fetch", "click"),
    "vibe: go ($0 · ~N min)": ("mc-go-vibe", "click"),
    "pack: go": ("mc-go-pack", "click"),
    "talk: go": ("mc-go-talk", "click"),
    "duet: go": ("mc-go-duet", "click"),
    "sing: go": ("mc-go-sing", "click"),
    "batch: go": ("mc-go-batch", "click"),
    "re-dub: finished clip": ("mc-redub-video", None),
    "re-dub: record / upload a take": ("mc-redub-rec", None),
    "re-dub: more takes (record-line multi-take)": ("mc-redub-takes", None),
    "re-dub: lip re-sync": ("mc-redub-lips", None),
    "re-dub: go": ("mc-go-redub", "click"),
    "setup: check my models (doctor)": ("mc-check-models", "click"),
    "preview -> HD": ("mc-hd", "click"),
    "before / after wipe": ("mc-wipe", "click"),
    "still -> animate": ("mc-animate", "click"),
    "post text": ("mc-post-text", None),
    "share buttons": ("mc-share", None),
    "reroll": ("mc-reroll", "click"),
    "stop": ("mc-stop", "click"),
    "open folder": ("mc-folder", "click"),
    "watermark toggle": ("mc-watermark", None),
    "download": ("mc-download", None),
}

# inputs that must feed a given handler (elem_id of the trigger -> elem_ids among its inputs)
INPUTS = {"mc-go-redub": ("mc-redub-rec", "mc-redub-takes", "mc-redub-lips"),
          "mc-go-swap": ("mc-swap-clip", "mc-swap-photo", "mc-stylelock", "mc-swap-split", "mc-closeup")}

# strings that must appear somewhere in the built config (HTML blocks, labels)
STRINGS = ("mc-toggle", "Mock mode", "agent control", "person 2: moves",
           "person 2 looks like", "person 2: selfie", "Re-dub a line", "useful guide", "help: quick tour", "$0")

BACKEND = {"guide sections + deep links": ("main_character_comfy.uibits", "guide_sections"),
           "deep-link status notes": ("main_character_comfy.uibits", "link_guide"),
           "who's-who strip html": ("main_character_comfy.uibits", "who_strip_html"),
           "per-person moves": ("main_character_comfy.uibits", "driver_choices"),
           "$0 · ~N min label": ("main_character_comfy.uibits", "money_time_label"),
           "shared eta range": ("main_character_comfy.pipeline", "eta_range_text"),
           "preview -> HD": ("main_character_comfy.pipeline", "Pipeline.finish_pending"),
           "re-dub": ("main_character_comfy.extras", "run_redub"),
           "lip re-sync": ("main_character_comfy.extras", "_lipsync_line"),
           "record-line bridge": ("main_character_comfy.recordline_bridge", "process"),
           "record-line backend": ("main_character_comfy.recordline", "process_line"),
           "style lock": ("main_character_comfy.stylelock", None),
           "cuts": ("main_character_comfy.shots", None),
           "close-up refs": ("main_character_comfy.refcheck", None),
           "byo clip fetch": ("main_character_comfy.clipfetch", None),
           "gpu tiers": ("main_character_comfy.tiers", "gate"),
           "agent control": ("main_character_comfy.agent", None),
           "doctor": ("main_character_comfy.doctor", "live_report"),
           "launch-flag warning": ("main_character_comfy.comfy_launch", "flags_advice"),
           "dark/light toggle": ("main_character_comfy.themes", "toggle_html"),
           "try-it page tip area": ("main_character_comfy.trypage", "tip_html")}


@pytest.fixture(scope="module")
def cfg():
    return ui.build_ui("mock").config


def _ids(cfg):
    return {c.get("props", {}).get("elem_id"): c["id"] for c in cfg["components"] if c.get("props", {}).get("elem_id")}


def check_parity(cfg) -> list[str]:
    """Every missing feature, as plain text (empty list = parity)."""
    ids, missing = _ids(cfg), []
    deps = cfg["dependencies"]
    for feat, (eid, trig) in COMPONENTS.items():
        if eid not in ids:
            missing.append(f"{feat}: no component with elem_id {eid!r}")
            continue
        if trig and not any([ids[eid], trig] in [list(t) for t in d.get("targets", [])]
                            and (d.get("backend_fn") or d.get("js")) for d in deps):
            missing.append(f"{feat}: no {trig} handler on {eid!r}")
    for eid, need in INPUTS.items():
        feeds = [d for d in deps if eid in ids and [ids[eid], "click"] in [list(t) for t in d.get("targets", [])]]
        ins = {i for d in feeds for i in d.get("inputs", [])}
        missing += [f"{eid} handler does not take {n!r}" for n in need if ids.get(n) not in ins]
    blob = json.dumps(cfg, default=str, ensure_ascii=False)
    missing += [f"text {s!r} gone from the UI" for s in STRINGS if s not in blob]
    return missing


def test_every_feature_has_component_and_handler(cfg):
    assert check_parity(cfg) == []


@pytest.mark.parametrize("feat", sorted(BACKEND))
def test_backend_function_exists(feat):
    mod, attr = BACKEND[feat]
    obj = importlib.import_module(mod)
    for part in (attr.split(".") if attr else []):
        obj = getattr(obj, part)
    assert obj is not None


def test_parity_catches_a_dropped_feature(cfg):
    """The guard itself works: drop the multi-take input and the re-dub handler, and it complains."""
    import copy

    bad = copy.deepcopy(cfg)
    bad["components"] = [c for c in bad["components"] if c.get("props", {}).get("elem_id") != "mc-redub-takes"]
    wipe = _ids(cfg)["mc-wipe"]
    bad["dependencies"] = [d for d in bad["dependencies"] if [wipe, "click"] not in [list(t) for t in d.get("targets", [])]]
    msgs = check_parity(bad)
    assert any("mc-redub-takes" in m for m in msgs) and any("no click handler on 'mc-wipe'" in m for m in msgs)
