"""record-your-own-line: room match (numpy port of roommatch)."""

import numpy as np

from main_character_comfy import redub
from main_character_comfy.recordline import roommatch
from tests._recordline_synth import SR, voice


def _speech(seconds=4.0):
    """Syllable bursts with sharp (5 ms) offsets and pauses, like real speech, so room decays are measurable."""
    v = voice(seconds, words=1, seed=2)
    env = np.zeros(len(v))
    r = int(0.005 * SR)
    for a in np.arange(0.1, seconds - 0.3, 0.36):
        i0, i1 = int(a * SR), int((a + 0.2) * SR)
        env[i0:i1] = 1
        env[i0:i0 + r] = np.linspace(0, 1, r)
        env[i1 - r:i1] = np.linspace(1, 0, r)
    v = voice(seconds, words=1, seed=2) * 0 + _buzz(len(v)) * env * 0.25
    return np.concatenate([v, np.zeros(SR // 2)]) + 1e-5 * np.random.default_rng(0).standard_normal(len(v) + SR // 2)


def _buzz(n):
    t = np.arange(n) / SR
    ph = 2 * np.pi * 140 * t
    x = sum((1 / k) * np.sin(k * ph) for k in range(1, 25))
    return x / np.abs(x).max()


def test_blind_rt60_tracks_the_room():
    dry = _speech()
    for rt in (0.4, 0.8):
        wet = redub.fft_convolve(dry, roommatch.make_ir(rt, seed=3).mean(1))[: len(dry)]
        est, n, _, _ = roommatch.rt_estimate(wet)
        assert n >= 2 and 0.5 * rt < est < 1.8 * rt, (rt, est, n)


def test_wet_level_is_measured_not_a_preset():
    dry = _speech()
    rm_dry = roommatch.analyse(dry, dry)  # dry voice-over scene: little or no reverb
    wet_ref = roommatch.reverb(dry, roommatch.make_ir(0.6, seed=5), 0.15).mean(1)
    rm_wet = roommatch.analyse(wet_ref, dry)
    assert rm_dry.wet <= 0.03
    assert 0.08 <= rm_wet.wet <= 0.25 and "matched" in rm_wet.wet_reason
    very = roommatch.analyse(roommatch.reverb(dry, roommatch.make_ir(0.9, seed=5), 0.6).mean(1), dry)
    assert roommatch.DEFAULT_WET <= very.wet <= roommatch.MAX_WET  # capped, never a wash


def test_eq_is_gentle_and_capped():
    dry = _speech()
    dark = redub.fft_convolve(dry, np.ones(24) / 24)[: len(dry)]  # a much darker reference
    f, g = roommatch.eq_curve(dark, dry)
    assert g.max() <= 2.0 + 1e-9 and g.min() >= -3.0 - 1e-9
    assert np.all(g[f > 12000] <= 1e-9)  # no boosts up top


def test_too_few_decays_uses_the_approved_default():
    rm = roommatch.analyse(np.sin(np.arange(SR) / 10) * 0.1, _speech())
    assert rm.rt60 == roommatch.DEFAULT_RT and "default" in rm.rt60_reason
