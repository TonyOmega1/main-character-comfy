import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def object_info():
    """Reference snapshot + core VAEDecodeTiled (every video graph decodes tiled; the node is core ComfyUI)."""
    oi = json.loads((FIX / "object_info_reference.json").read_text())
    extra = json.loads((FIX / "object_info_v041_core_extra.json").read_text(encoding="utf-8"))
    oi.setdefault("VAEDecodeTiled", extra["VAEDecodeTiled"])
    return oi


@pytest.fixture(scope="session")
def object_info_v041(object_info):
    """Reference snapshot + the two core v0.41 nodes MC Lite adds (transcribed from source, see the fixture)."""
    extra = json.loads((FIX / "object_info_v041_core_extra.json").read_text(encoding="utf-8"))
    extra.pop("$comment", None)
    return {**object_info, **extra}


@pytest.fixture(autouse=True)
def _no_tier_env(monkeypatch):
    """Tier detection must not depend on the machine running the tests."""
    for var in ("MC_TIER", "MC_VRAM_GB"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MC_FREE_RAM_GB", "64")  # RAM preflight: plenty free unless a test says otherwise
    from main_character_comfy import system

    monkeypatch.setattr(system, "_smi_vram_gb", lambda: None)


@pytest.fixture(scope="session")
def models():
    from main_character_comfy.config import Models

    return Models()


@pytest.fixture
def sample_photo(tmp_path):
    from PIL import Image

    import random

    p = tmp_path / "photo.jpg"
    # camera-photo-like (thousands of colours), so subject="auto" reads it as a photo, not an avatar
    rnd = random.Random(1)
    img = Image.new("RGB", (90, 120))
    img.putdata([(min(255, 60 + x * 2 + rnd.randint(-25, 25)), min(255, 40 + y + rnd.randint(-25, 25)),
                  max(0, 120 + rnd.randint(-50, 50))) for y in range(120) for x in range(90)])
    img.resize((900, 1200)).save(p)
    return p


@pytest.fixture
def sample_voice(tmp_path):
    from main_character_comfy.media import run_ffmpeg

    p = tmp_path / "voice.wav"
    run_ffmpeg(["-f", "lavfi", "-i", "sine=frequency=220:sample_rate=44100:duration=12", "-ac", "2", str(p)])
    return p


@pytest.fixture(autouse=True)
def _community_formats_in_tmp(tmp_path, monkeypatch):
    """'Save as format' must never write into the repo during tests."""
    monkeypatch.setenv("MC_COMMUNITY_DIR", str(tmp_path / "community"))
    monkeypatch.setenv("MC_AUTO_STEMS", "0")
    monkeypatch.setenv("MC_CUT_SECOND", "0")  # PySceneDetect cross-check off unless a test turns it on  # a dev machine's Demucs must not run in tests (stems.py)
    yield
    from main_character_comfy import memes

    monkeypatch.delenv("MC_COMMUNITY_DIR")
    memes.reload()
