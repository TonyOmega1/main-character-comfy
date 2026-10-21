"""record-your-own-line: raw-take checks (numpy port of takeqc)."""

import numpy as np

from main_character_comfy.recordline import takeqc
from tests._recordline_synth import SR, voice


def _take(level=0.25, seed=0):
    rng = np.random.default_rng(seed)
    v = np.concatenate([np.zeros(SR // 3), voice(2.0, level=level, seed=seed), np.zeros(SR // 3)])
    return v + 0.0005 * rng.standard_normal(len(v))


def test_clean_take_has_no_warnings():
    r = takeqc.analyse(_take(), SR)
    assert not (r["clipping"] or r["dropouts"] or r["crackle"]), r
    assert r["warnings"] == []


def test_clipping_is_caught():
    x = np.clip(_take(level=0.9) * 3, -1, 1)
    r = takeqc.analyse(x, SR)
    assert r["clipping"] and "clipping" in r["warnings"][0]


def test_dropouts_are_caught():
    x = _take()
    for c in (0.8, 1.3, 1.7):
        x[int(c * SR):int(c * SR) + 240] = 0.0  # 5 ms buffer holes inside speech
    r = takeqc.analyse(x, SR)
    assert r["dropouts"] and any("tiny gaps" in w for w in r["warnings"])


def test_gate_digital_silence_is_flagged():
    x = _take()
    x[np.abs(x) < 0.003] = 0.0  # a gate snapping the room to zero
    x[: SR // 3] = 0.0
    r = takeqc.analyse(x, SR)
    assert r["digital_silence_pct"] > 5 and any("dead silence" in w for w in r["warnings"])


def test_random_clicks_in_quiet_parts_point_to_the_cable():
    x = _take()
    x = np.concatenate([x, 0.0005 * np.random.default_rng(3).standard_normal(SR)])
    for c in np.linspace(2.75, 3.5, 6):
        x[int(c * SR)] += 0.2
    r = takeqc.analyse(x, SR)
    assert r["crackle"] and r["clicks_quiet"] > r["clicks_loud"]
    assert any("cable" in w for w in r["warnings"])


def test_silent_take():
    r = takeqc.analyse(np.zeros(SR), SR)
    assert r["silent"] and r["warnings"]
