"""Make rights-free test material for Swap Me In with MC's own models (no third-party footage, no real people).

* ``template_trio-park-dance_generated_5s.mp4``: three invented people dancing side by side (FLUX.2 klein still ->
  Wan 2.2 I2V A14B, 832x480, 5 s) for multi-character / Openjutsu tests and the demo.
* ``ref_invented-<name>.png``: invented characters to cast into it (alongside your own selfie / avatar).

    python scripts/make_test_clips.py --out "C:/path/to/MC/2 Uploads"

Needs ComfyUI + comfy-api-proxy running (same as the app) and the core + meme model files.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main_character_comfy import graphs  # noqa: E402
from main_character_comfy.backends import ComfyBackend  # noqa: E402
from main_character_comfy.config import Models  # noqa: E402

TRIO = ("Wide shot, horizontal 16:9 frame, three invented adult friends standing side by side in a sunny city park, "
        "full bodies visible head to feet with space between them, facing the camera, happy, ready to dance: on the "
        "left a tall woman in a bright yellow hoodie and jeans, in the middle a short man in a green tracksuit, on "
        "the right a woman with curly hair in a red jacket and black trousers. Natural daylight, photorealistic, no "
        "text, no logos.")
DANCE = ("The three friends dance together side by side to an upbeat song, bouncing, swinging their arms and stepping "
         "left and right in sync, smiling. The camera stays still. Full bodies stay in frame.")
REFS = {
    "invented-robot": "Full body portrait of a friendly invented cartoon robot mascot with a round teal head and orange "
                      "arms, flat illustrated style with clean outlines, standing, plain neutral grey background.",
    "invented-astronaut": "Full body portrait of an invented adult astronaut in a white and purple suit with the visor "
                          "up, smiling, 3D animated film style, standing, plain neutral grey background.",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    work = a.out / "_make_test_clips"
    m = Models()
    be = ComfyBackend()

    def run(name, sg, files=None):
        res = be.run_stage(name, sg, files or {}, work, lambda f, msg: print(f"  {name}: {msg}", flush=True))
        return res.path

    for key, prompt in REFS.items():
        p = run(key, graphs.avatar_graph(prompt, models=m, reference=0, seed=a.seed, width=768, height=1344))
        shutil.copyfile(p, a.out / f"ref_{key}.png")
    still = run("trio_still", graphs.avatar_graph(TRIO, models=m, reference=0, seed=a.seed, width=1344, height=768))
    shutil.copyfile(still, a.out / "template_trio-park-dance_still.png")
    clip = run("trio_dance", graphs.meme_motion_graph(DANCE, models=m, width=832, height=480, frames=81,
                                                      seed=a.seed), {"avatar": still})
    shutil.copyfile(clip, a.out / "template_trio-park-dance_generated_5s.mp4")
    print("done:", *sorted(p.name for p in a.out.iterdir() if p.is_file()), sep="\n  ")


if __name__ == "__main__":
    main()
