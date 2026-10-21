# Record your own line: audio backend (`main_character_comfy.recordline`)

CPU only. No GPU and no ComfyUI. It takes 1-N recorded takes of a line, scores them, picks the best one, fits it
to the character's speaking window and mixes it into the clip. The picture side (lip re-sync) stays in `lipdub.py`.
This module gives it `aligned_voice_wav`.

## API

```python
from main_character_comfy.recordline import pick_and_align, process_line, mux

r = pick_and_align(video, [take1, take2], original_text=None, *, out_dir, window=None, start_hint=None,
                   settings=None, whisper_model=None, verify=True)  # -> JSON-able dict

out = process_line(video, line_wav, *, start, work, settings)       # Re-dub tab entry (recordline_bridge)
# settings: takes=[...], text/original_text, window=(s, e), fit_seconds, denoise, duck, voice_level_db
# -> pick_and_align's dict + output (mp4, stream-copied video + AAC 320k) + steps (warnings first)
```

`out_dir` is keyword-only. It has no default and comes after a defaulted parameter, so making it keyword-only
keeps the spec's order. The result keys are `chosen_take`, `takes` (one entry per take, best first, with `wer`,
`min_word_prob`, `duration_fit`, `score`, `heard`, `qc` and `warnings`), `window_start` / `window_end` (s),
`window` (how it was found), `stretch_ratio` (output/input, capped at 0.85-1.15), `aligned_voice_wav`,
`infinitetalk_wav`, `infinitetalk_clip_t0`, `final_mix_wav`, `voice_start` / `voice_end`, `warnings`
(plain English), `steps`, `room`, `levels`, `verify` (a speech-to-text check of the fitted line) and `available`.

## Outputs

| key | what |
| --- | --- |
| `aligned_voice_wav` | The new line alone and dry (EQ, level, no reverb). It's on the clip timeline at full clip length, 48 kHz mono float. Pass it as `voice` to `extras._lipsync_line`, which cuts its own window. |
| `infinitetalk_wav` | 16 kHz mono with a 0.6 s silent head pad, starting at the window start. Render time t lands on clip time `infinitetalk_clip_t0 + t`. |
| `final_mix_wav` | The finished soundtrack, 48 kHz stereo 24-bit. `mux()` turns it into the mp4. |
| `voice_wet_wav` | The new line with its matched room (for remixing). |

## Steps

1. **Stems**: Demucs two-stem through `stems.separate_vocals` (vocals and no_vocals).
2. **Original line**: finds the speech cluster in the vocal stem (the one nearest `start_hint`). Speech-to-text runs
   on the cluster plus 0.4 s on each side, and `original_text` is located inside that transcript.
3. **Speaking window**: the stem onset and offset (speech band, p95 - 24 dB). Mouth openness from MediaPipe nudges
   it: when the two agree within 0.25 s, the result is 70% sound and 30% mouth. Without MediaPipe it falls back to
   YuNet motion. A window you pass in always wins.
4. **Each take**: takeqc (clipping, dropouts, crackle, gate silence, level), then speech-to-text, then the speech
   region (from the level, not Whisper's word times). Score = 0.55 x (1 - WER) + 0.25 x min word probability +
   0.20 x duration fit (0.85-1.15), minus QC penalties.
5. **Chain** (chosen take): DeepFilterNet3 with a 24 dB attenuation limit, cut, rubberband retime (pitch and formant
   kept, max 15%), placed on the onset, HP 80 Hz + 2:1 compression, roommatch EQ toward the original line, slow peak
   control (<= 4 dB), one voice gain (= the original line's loudness), roommatch reverb (wet can be 0), the
   no-vocals stem ducked -3 dB with 0.25 s cosine ramps, one static gain (the clip's loudness, <= -1.5 dBTP),
   then a 24-bit WAV.

Never used: loudnorm, fast limiters, saturation, exciters, synthetic room tone, looped gap fill or low-bitrate AAC.
The bed is never muted, so there are no noise-floor holes.

### Why this chain and not redub.py

The chain follows DubMaster's approved Test 6 chain:

- EQ targets the original line (vocal stem), not the whole soundtrack.
- The room is measured, not a fixed 0.45 s / 15%.
- The voice level matches the original line, not the programme.
- It adds capped peak control.

It keeps two things from redub.py: the cosine duck and the final static gain with a TP ceiling. It drops voice_chain's
mute-under-the-line in favour of ducking the no-vocals stem. Without stems it ducks the full mix by -11 dB, as
redub.py does.

## Optional tools (none required)

faster-whisper, Demucs, MediaPipe (plus `face_landmarker.task` in `assets/` or `MC_FACE_LANDMARKER`), DeepFilterNet
and rubberband (the ffmpeg filter). A missing tool adds one plain-English warning that says what was skipped and
how to add it. `recordline.status()` lists what's there right now.

## Validated on

The Rec5 clip, with DubMaster's three cut takes, run on CPU with every tool installed (about 50 s warm). It
picked take 5, the one approved by hand for the delivered dub. The window came out as 3.72-5.39 s (by hand: 3.76-5.40). The
fitted line's speech-to-text check returned the line exactly (WER 0, min p 0.90).
