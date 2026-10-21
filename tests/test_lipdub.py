"""Record your own line: the lip re-sync geometry (lipdub.py) and the redub flow with lipsync on. CPU / mock only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from main_character_comfy import extras, lipdub, media
from main_character_comfy.backends import MockBackend
from main_character_comfy.config import Settings
from main_character_comfy.media import count_frames, has_audio_stream, media_duration, run_ffmpeg
from main_character_comfy.pipeline import Pipeline


def _row(cx=100.0, cy=100.0, s=60.0):
    """A YuNet-style row for a face centred at (cx, cy) of height s."""
    return np.array([cx - s / 2, cy - s / 2, s, s, cx - s * .2, cy - s * .15, cx + s * .2, cy - s * .15,
                     cx, cy + s * .05, cx - s * .15, cy + s * .25, cx + s * .15, cy + s * .25, 0.9], np.float32)


def test_line_window_pads_and_clamps():
    w = lipdub.line_window(1.0, 1.5, 3.0, 24)
    assert (w.f0, w.f1) == (18, 66) and abs(w.t0 - 0.75) < 1e-6
    edge = lipdub.line_window(0.1, 5.0, 3.0, 24)
    assert edge.f0 == 0 and edge.f1 == 72


def test_fill_track_and_crop_box():
    t = np.stack([_row(), _row(), _row()])
    t[1] = np.nan
    f = lipdub.fill_track(t)
    assert not np.isnan(f).any() and abs(f[1, 0] - t[0, 0]) < 1e-3
    x, y, side = lipdub.crop_box(f, (320, 240))
    assert side % 2 == 0 and 0 <= x <= 320 - side and 0 <= y <= 240 - side and side >= 2 * 60


def test_mouth_mask_covers_mouth_not_eyes():
    pts = lipdub.landmarks(_row())
    m = lipdub.mouth_mask(pts, (200, 200))
    mouth = ((pts[3] + pts[4]) / 2).astype(int)
    assert m[mouth[1], mouth[0]] > 0.5
    assert m[int(pts[0][1]), int(pts[0][0])] < 0.01 and m[int(pts[2][1]) - 2, int(pts[2][0])] < 0.01


def test_align_recovers_a_similarity_transform():
    src = lipdub.landmarks(_row(100, 100, 60))
    dst = lipdub.landmarks(_row(130, 110, 90))
    M = lipdub.align(src, dst)
    assert abs(np.hypot(M[0, 0], M[1, 0]) - 1.5) < 0.02


def test_paste_changes_only_the_mouth_and_fades():
    n, h, w = 12, 200, 200
    orig = np.full((n, h, w, 3), 90, np.uint8)
    gen = np.full((n, h, w, 3), 90, np.uint8)
    gen[:, 122:138, 90:110] = 250  # a bright "open mouth" in the render (inside the mask, not the skin ring)
    track = np.stack([_row()] * n)
    out, rep = lipdub.paste_mouths(orig, gen, track, track, 24.0, gen_fps=24.0)
    assert rep["frames"] == n
    mid = out[n // 2].astype(int)
    assert mid[125:135, 90:110].mean() > 150  # mouth region came from the render
    assert np.abs(mid[:60] - 90).max() <= 1  # eyes / forehead untouched
    assert out[0, 125:135, 90:110].mean() < mid[125:135, 90:110].mean()  # faded in at the window edge


def test_redub_with_lipsync_runs_one_job_and_keeps_length(tmp_path, monkeypatch):
    monkeypatch.setenv("MC_RECORDLINE", "off")  # built-in redub + lipdub path (the bridge path is covered in test_recordline_bridge.py)
    clip = tmp_path / "talk.mp4"
    run_ffmpeg(["-f", "lavfi", "-i", "testsrc2=size=320x240:rate=24:duration=3", "-f", "lavfi", "-i",
                "sine=frequency=300:duration=3", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
                str(clip)])
    line = tmp_path / "line.wav"
    run_ffmpeg(["-f", "lavfi", "-i", "sine=frequency=500:duration=1.2", "-ar", "48000", str(line)])
    monkeypatch.setattr(lipdub, "detect_faces_robust",
                        lambda frames, need=0.6: np.stack([_row(160, 110, 70)] * len(frames)) if len(frames) else
                        np.zeros((0, 15), np.float32))
    pipe = Pipeline(MockBackend(), Settings(output_dir=tmp_path / "out"))
    evs = list(extras.run_redub(pipe, clip, recording=line, start=1.0, denoise="off", consent=True, lipsync=True))
    done = evs[-1]
    assert done.status == "complete"
    assert any(e.stage == "lipsync" and e.status == "done" for e in evs)
    assert has_audio_stream(done.artifact) and abs(media_duration(done.artifact) - 3.0) < 0.15
    assert count_frames(done.artifact) == 72


def test_redub_lipsync_without_a_face_falls_back_to_audio_only(tmp_path, monkeypatch):
    monkeypatch.setenv("MC_RECORDLINE", "off")  # built-in redub + lipdub path (the bridge path is covered in test_recordline_bridge.py)
    clip = tmp_path / "noface.mp4"
    run_ffmpeg(["-f", "lavfi", "-i", "color=c=gray:size=320x240:rate=24:duration=2", "-f", "lavfi", "-i",
                "sine=frequency=300:duration=2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
                str(clip)])
    line = tmp_path / "line.wav"
    run_ffmpeg(["-f", "lavfi", "-i", "sine=frequency=500:duration=0.8", "-ar", "48000", str(line)])
    pipe = Pipeline(MockBackend(), Settings(output_dir=tmp_path / "out"))
    evs = list(extras.run_redub(pipe, clip, recording=line, start=0.5, denoise="off", consent=True, lipsync=True))
    assert evs[-1].status == "complete"
    assert any(e.stage == "lipsync" and e.status == "skipped" for e in evs)
