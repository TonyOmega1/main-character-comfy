# Useful guide

<!-- Shown in the app (the 'useful guide' panel next to 'help: quick tour'); warnings link to the #guide-* anchors. Keep the anchors stable. -->

We made a lot of videos with MC on a normal gaming PC (RTX 5080, 16 GB). These are the lessons we learned along the way, shared so your first takes come out better. Results can still be a little janky sometimes, and none of this blocks you. Rerolls are free.

### 1. Start with the best clip you can get  ·  `#guide-source`

**What happens:** A blurry or small clip gives a blurry, smeary result. MC can sharpen the final result, but it cannot invent detail that was never filmed.

**Why:** The model copies motion, light and framing from your clip. When the clip is a 480p re-upload of a meme, there is less to copy, and the upscale step sometimes "invents" skin or teeth to fill the gaps.

**Tips:**
- Use **fetch the sharpest copy** with the original post's link, not a repost. (Our dance test topped out at 720p because that was the best copy available anywhere.)
- Look for a steady camera, good light, and people who are not too small in the frame.
- Start with shorter clips, around 5 to 10 seconds.

### 2. Your reference photo decides the likeness  ·  `#guide-refs`

**What happens:** The swapped person looks only somewhat like you. The face drifts, or a cap gains a logo that is not on yours.

**Why:** The model only knows what is in your photo. A small, dark, black-and-white or heavily filtered photo leaves it guessing, and it fills the gaps with "a generic person".

**Tips:**
- Use one sharp, well-lit, colour photo per person, at least about 1024 px, with the face and outfit clearly visible. A full-body photo is ideal.
- For close-up clips, add a face close-up as a second photo.
- Sunglasses hide the eyes, so likeness scores drop. This is expected.
- MC checks every photo you add and tells you when it is weak. Take that warning seriously.

### 3. Cartoon and anime refs can drift toward "real"  ·  `#guide-cartoon`

**What happens:** You give MC a drawn character, and it comes back looking like a real person in a costume, with real skin shading, real fabric texture and no outlines.

**Why:** Your clip is live action. The model copies the motion and the lighting from it, and a real-looking clip pulls the drawing toward "real". The sharpening step makes this worse by adding photo-style texture.

**Tips:**
- Set the reference to **my avatar (drawing)** instead of auto, so MC uses its drawing-friendly settings. Auto can mistake a painted picture (sky, grass, gradients) for a photo.
- Turn **match scene lighting** off for drawn characters.
- The one-person-at-a-time path holds a drawing style better than the all-at-once group pass. Choose **people engine: one at a time** (Swap → More options) for drawn characters if the first take looks too real.
- **Style lock** (Swap → More options) bundles our drawing-friendly settings. It is new and *pending a real test*, so it stays off unless you turn it on.
- A clean character on a plain background works best, such as a flat-colour full-body drawing.
- The talking-photo modes (no video driver) keep drawing styles best of all.

*We are improving this now, and this section will be updated.*

### 4. Hard cuts: why MC swaps shot by shot  ·  `#guide-cuts`

**What happens:** In a clip with camera cuts, the swap works in the first shot, then the original person "comes back" after the cut, or only some of the people are swapped.

**Why:** MC follows each person frame by frame. At a hard cut, everyone jumps to a new position, so the tracker loses them.

**Tips:**
- MC finds hard cuts and tells you. On the one-at-a-time path it then swaps each shot separately and stitches them back together with the original audio, keeping each person matched across cuts by their look. **Split at cuts: on** forces this for any clip. It is slower, but the result holds.
- If a shot still loses someone, set **Start at** and **Length** to just that shot and redo only that part, instead of the whole clip.
- Cut points where people turn their backs are where looks can shift slightly between shots.

### 5. Extra or doubled people  ·  `#guide-duplicates`

**What happens:** A third person appears who is not in the clip, usually a copy of one of your characters.

**Why:** When a reference has very different proportions from the person in the clip (for example a full-body cartoon on a tall dancer, or a wide shot with tiny people), the model sometimes draws the character twice to "fit" both.

**Tips:**
- Reroll with a new seed. That fixed it for us on the next take.
- Give each person a description of what they look like. This tells the model exactly how many people to draw.
- Choose a reference with framing similar to the clip (full body for wide shots, upper body for close-ups).

### 6. The right person doing the right moves  ·  `#guide-roles`

**What happens:** In a group clip, two people's looks swap places, or someone performs the other person's moves.

**Why:** MC numbers people **left to right on the first frame**. If they cross paths or the shot cuts, the numbering can get mixed up later in the clip.

**Tips:**
- After **find people**, check the numbers on the preview before rendering.
- Clips where people stay on their own side work best.
- For clips with cuts, check the first frame of each shot.

### 7. Lip sync: mostly good, sometimes soft  ·  `#guide-lipsync`

**What happens:** The mouth follows the words, but on some syllables it looks soft or half-closed for a moment.

**Why:** Fast speech, close-ups with sunglasses, and music under the voice give the model less to work with. A short silence is also *supposed* to look closed, so check the words before treating it as a sync error.

**Tips:**
- For talking, use a clear voice recording with little music under it.
- A still photo plus your line (talk mode) gives the steadiest mouth and face.
- If one moment is soft, redo that line with **record your own line** (More → Re-dub a line) instead of the whole clip. Its lip re-sync option is new and pending a real test.

### 8. Hands, glasses and small details  ·  `#guide-details`

**What happens:** Fingers merge together, glasses or earrings flicker, floppy ears melt, text on a book or shirt turns to mush, and small objects float next to the head.

**Why:** Small, thin details are the hardest part for video models, especially when they move fast or the face is small in the frame.

**Tips:**
- Reroll first. It is free.
- Keep hands and props large in the frame if they matter.
- Do not count on readable text on props.
- For a talking shot, start from one clean still (talk mode). That kept glasses, beard and gauges steady for us.

### 9. Flicker and look changes over time  ·  `#guide-flicker`

**What happens:** The outfit or face shifts a little over a long clip, or the look changes at a seam.

**Why:** Long clips are rendered in pieces, and each piece re-reads your reference. The further from the first frame, the more room there is for drift.

**Tips:**
- Shorter clips drift less.
- Retaking the bad stretch is better than re-rendering everything.
- "Faster" settings that change the window size also change the look. MC keeps the slower, steadier default on purpose.

### 10. It's slow, and that's normal  ·  `#guide-speed`

**What happens:** A 10 second two-person swap can take 15 to 50 minutes on a 16 GB card.

**Why:** Every swapped person is a full render pass, plus sharpening at the end. Everything runs on your GPU. There are no credits and no queue, but there is no datacenter either.

**Tips:**
- Start with a short preview (**preview first, HD later**), then hit **make it HD** only on the takes you like.
- Run one render at a time. Running two at once makes both slower.
- The ETA above the button is measured on an RTX 5080. Treat it as a good estimate, not a promise.

**Pre-render checklist (long renders):** nothing here is forced, but each item frees GPU memory and stops another app from competing with the render.
- Close browsers, Discord, game launchers (Steam, Epic), overlays (Discord, Steam, NVIDIA, Xbox Game Bar) and other GPU apps (games, Blender, Unreal, VRoid, wallpaper apps). Leave OBS open only if you are streaming.
- If you need a browser open, turn off its hardware acceleration: Chrome or Edge, `chrome://settings/system` (or `edge://settings/system`), turn off *Use graphics acceleration when available*, then Relaunch. Turn it back on afterwards.
- Check what is still on the GPU: `nvidia-smi` (or `nvitop --once`). On Windows (WDDM) both list every app that has a window on the GPU but show *N/A* per-process memory, so look at total memory used (about 1.3-1.5 GB is just the desktop) and make sure no other `python.exe`, Blender, Unreal or VRoid is running. One GPU job at a time.
- Start ComfyUI with `run_nvidia_gpu_mc.bat` (make it once with `python -m main_character_comfy.comfy_launch --root <your ComfyUI_windows_portable folder> --write-bat`) or with `run.ps1 -StartComfy <that folder>`. Both use `--reserve-vram 1.5 --max-upload-size 500`, so 1.5 GB stays free for the desktop and clips up to 500 MB upload. If you are streaming, use the .bat's commented `--reserve-vram 2` line, or `-ReserveVram 2`.

### 11. Rights and respect  ·  `#guide-rights`

Use clips you have the rights to, and only put people in a video if they are okay with it. MC runs on your PC and uploads nothing, but you are responsible for what you post.
