# First real GPU run: the reference PC (RTX 5080 16 GB), Oct 16 2026

RTX 5080 16 GB, 32 GB RAM, Windows 11, ComfyUI portable v0.41.0 (default flags unless noted), MC at commit after 469fe7b.
Stand-in face: the author's own illustrated portrait (full body, cap + shades). Times are wall clock (CT).
Peak VRAM is whole-GPU `nvidia-smi` and includes ~2.7-3.3 GB used by the Windows desktop/browser.

| Test | Result | Wall time | Stages | Peak VRAM |
|---|---|---|---|---|
| a) side_eye | pass | 145 s | scene 23 s (cold), motion ~118 s | 15.6 GB |
| b) dramatic_zoom | pass | 118 s | scene 10 s, motion ~105 s | 15.4 GB |
| e) 4-meme pack | pass | 513 s (8.6 min) | 2.0-2.3 min per meme, scenes 10-12 s | 15.7 GB |
| f) generated reference move (--sd) | pass | 141 s | | 15.5 GB |
| g) Copy the Vibe 4.8 s (1 window) | pass, first ever GPU run | 214 s | scene 21 s, Animate job 188 s (sampling 6x20 s) | 15.6 GB |
| k1) Lite side_eye (reserve-vram 4.5) | pass | 122 s | scene 17 s, TI2V 5B Q8_0 motion 103 s (cold) | 13.3 GB |
| k2) Lite dramatic_zoom --length 3 | pass | 107 s | scene 16 s, motion 88 s | 13.5 GB |
| k3) Lite vibe without --allow-heavy | pass (friendly exit 2) | 1 s | | |
| k4) Lite side_eye, TI2V fp16 | pass | 101 s | scene 16 s, motion 83 s | 15.4 GB (fp16 overshoots a 12 GB budget: keep Q8_0) |

Not run yet: say-a-line / talk (needs the author's voice), Swap Me In (needs a clip the author filmed), 30 s Copy the Vibe (chunk seams),
Lite --allow-heavy Animate (needs the Q3_K_M file), Max tier (no 24 GB card).

Bugs found and fixed: CLI/doctor crashed printing emoji on cp1252 consoles; GIFs of busy clips were 3.0-4.4 MB (now
stepped down to fit 3 MB); PowerShell 5 drops `--top ""` (use `--top=`).

Quality notes: the scene step drops the cap/shades in some formats (side_eye) because the stand-in is an illustration;
side_eye (seed 2026) has a stray hand entering the last frame; victory_dance's bottom caption wraps to 2 lines and
the motion is nearly static; Copy the Vibe kept outfit/cap and the reference timing with none of the reference background.
