# Reference moves without anyone else's footage

Copy the Vibe and Swap Me In need a clip **you have the rights to**. For demos and examples we never use third-party clips (movies, TV, music videos, other creators' posts). There are two clean sources.

## 1. Generate the move with MC itself (invented character)

`examples/avatar_3d_flux2_klein.jpg` is a fictional character generated with FLUX.2 klein. Render it doing the move as a custom Meme Recreate format, with no captions and no watermark:

```bash
python -m main_character_comfy.cli --meme custom \
  --photo examples/avatar_3d_flux2_klein.jpg \
  --scene "standing in an empty white dance studio, full body visible, wide shot" \
  --action "does a simple, upbeat two-step dance with a little spin and finishes with a point at the camera" \
  --top "" --bottom "" --no-watermark --sd --i-have-rights
```

The output (`outputs/<run>/mc_final.mp4`, about 5 s) is a generated reference move. Feed it to Copy the Vibe:

```bash
python -m main_character_comfy.cli --mode vibe --clip outputs/<run>/mc_final.mp4 --photo me.jpg \
  --scene-prompt "luxury hotel hallway at night" --i-have-rights
```

For longer references, render 2 to 3 moves and join them, or use option 2.

## 2. Film it yourself

A 10 to 30 s phone clip of you, or of a friend who agrees, doing the move. Full body in frame, one person, steady camera, plain background. Copy the Vibe only uses the motion, so a messy room is fine.
