"""Oct 20 overnight lessons turned into regression tests (box, no GPU):
- argparse read '--drivers -,-,person:2' as a flag and crashed the beat A job;
- extras callers passing a bare record crashed with KeyError 'jobs' after a good InfiniteTalk render;
- framing QC failed a good SCAIL-2 take because the people detector missed the (non-human) bunny."""
from __future__ import annotations

from main_character_comfy import cli, framing
from main_character_comfy.pipeline import Pipeline


def test_dash_values_are_joined_for_per_person_options():
    got = cli._join_dash_values(["--clip", "c.mp4", "--drivers", "-,-,person:2", "--people", "-,me.png", "--face", "-"])
    assert got == ["--clip", "c.mp4", "--drivers=-,-,person:2", "--people=-,me.png", "--face=-"]
    # a real flag after the option is left alone, and other options are untouched
    assert cli._join_dash_values(["--drivers", "--qc", "off"]) == ["--drivers", "--qc", "off"]
    assert cli._join_dash_values(["--seed", "-1"]) == ["--seed", "-1"]


def test_parser_accepts_drivers_starting_with_dash():
    a = cli.build_parser().parse_args(cli._join_dash_values(["--drivers", "-,-,person:2"]))
    assert a.drivers == "-,-,person:2"


def test_framing_qc_soft_missing_for_non_human_people():
    src = [(100, 100, 300, 1200), (400, 100, 600, 1200), (650, 300, 700, 1200)]
    take = [src[0], src[1], None]
    strict = framing.framing_qc(src, (720, 1280), take, (720, 1280))
    assert not strict["pass"] and "person 3 missing" in strict["why"]
    soft = framing.framing_qc(src, (720, 1280), take, (720, 1280), soft_missing={2})
    assert soft["pass"] and soft["unverified"] == [3] and "check by eye" in soft["why"]
    # a soft slot never hides a real framing fail of a human
    moved = framing.framing_qc(src, (720, 1280), [(400, 200, 600, 1200), src[1], None], (720, 1280), soft_missing={2})
    assert not moved["pass"]


def test_nonhuman_slots_only_flags_non_human_refs(monkeypatch):
    from main_character_comfy import media
    monkeypatch.setattr(media, "face_check", lambda p: (str(p).endswith("bunny.png"), "test"))
    assert Pipeline._nonhuman_slots([(0, "tony.png"), (1, "homie.png"), (2, "bunny.png")]) == {2}
    def boom(p):
        raise OSError("unreadable")
    monkeypatch.setattr(media, "face_check", boom)
    assert Pipeline._nonhuman_slots([(0, "x.png")]) == set()


def test_job_record_without_jobs_key_does_not_crash(monkeypatch, tmp_path):
    from pathlib import Path
    from main_character_comfy import pipeline as pl
    from main_character_comfy.backends import StageResult
    pipe = Pipeline.__new__(Pipeline)  # no backend needed: _run_stage is stubbed
    pipe._stop = False
    monkeypatch.setattr(pl, "graph_prompts_and_models", lambda sg: (["p"], ["m"]))
    monkeypatch.setattr(Pipeline, "_run_stage", lambda self, *a, **k: iter([StageResult(Path(tmp_path / "x.mp4"), "abc")]))
    record: dict = {}  # what extras._lipsync_line callers (lipdub wrapper) pass
    results: dict = {}
    list(pipe._job("lipdub", "lipsync", object(), {}, tmp_path, results, record))
    assert record["jobs"] == {"lipdub": "abc"} and results["lipdub"].job_id == "abc"
