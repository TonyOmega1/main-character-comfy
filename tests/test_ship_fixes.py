"""Fixes from the Oct 21 PC fresh-install test: agent_e2e's output folder, README order, RAM preflight."""
import ast
from pathlib import Path

from main_character_comfy import system

ROOT = Path(__file__).parents[1]


def _main_fn():
    tree = ast.parse((ROOT / "scripts" / "agent_e2e.py").read_text(encoding="utf-8"))
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")


def test_agent_e2e_never_rebinds_the_output_folder():
    """`out` is the output folder; a talk result once overwrote it and the report write crashed."""
    stores = [n for n in ast.walk(_main_fn()) if isinstance(n, ast.Name) and n.id == "out" and isinstance(n.ctx, ast.Store)]
    assert len(stores) == 1, [n.lineno for n in stores]
    src = (ROOT / "scripts" / "agent_e2e.py").read_text(encoding="utf-8")
    assert '(out / "agent_e2e_report.json")' in src and 'open(out / "agent_e2e_server.log"' in src
    assert 'out.rglob("project.json")' in src and "talk_out.get(\"download_url\")" in src
    assert '"MC_LIBRARY_DIR": str(out / "library")' in src


def test_readme_setup_comes_before_the_tour_and_names_the_self_test():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert readme.index("\n## Setup\n") < readme.index("\n## Quick start\n") < readme.index("six-step tour")
    qs = readme[readme.index("\n## Quick start\n"):readme.index("six-step tour")]
    assert "scripts/agent_e2e.py" in qs and "run.ps1 -Check" in qs
    assert "about 22 GB free" in readme and "page file" in readme and "not passed on a 32 GB machine" in readme


def test_ram_preflight(monkeypatch):
    monkeypatch.setenv("MC_FREE_RAM_GB", "12")
    msg = system.ram_preflight()
    assert "12 GB free" in msg and "close browsers" in msg and "\u2014" not in msg and "\u2013" not in msg
    monkeypatch.setenv("MC_FREE_RAM_GB", "20")  # the Oct 21 PC retest: 18 GB was too low, InfiniteTalk adds ~21 GB
    assert "20 GB free" in system.ram_preflight()
    assert system.RAM_PREFLIGHT_GB == 22.0
    monkeypatch.setenv("MC_FREE_RAM_GB", "23")
    assert system.ram_preflight() == ""
    monkeypatch.setenv("MC_FREE_RAM_GB", "x")
    assert system.ram_preflight() == ""  # unknown: never nag
    monkeypatch.delenv("MC_FREE_RAM_GB")
    real = system.free_ram_gb()
    assert real is None or real > 0


def test_lipsync_paths_run_the_preflight():
    for f in ("pipeline.py", "extras.py"):
        assert "ram_preflight()" in (ROOT / "main_character_comfy" / f).read_text(encoding="utf-8"), f


def test_talk_run_warns_when_free_ram_is_low(tmp_path, monkeypatch, sample_photo, sample_voice):
    from main_character_comfy.backends import MockBackend
    from main_character_comfy.config import Settings
    from main_character_comfy.pipeline import Pipeline, Request

    monkeypatch.setenv("MC_FREE_RAM_GB", "10")
    pipe = Pipeline(MockBackend(), Settings(output_dir=tmp_path / "out"))
    req = Request(script="Hey! Welcome back to the channel.", photo=sample_photo, voice_sample=sample_voice,
                  style="anime", consent=True)
    evs = list(pipe.run(req))
    assert evs[-1].status == "complete"  # a warning, never a block
    warn = [e for e in evs if e.stage == "lipsync" and "Low free RAM" in (e.message or "")]
    assert len(warn) == 1 and "10 GB free" in warn[0].message
