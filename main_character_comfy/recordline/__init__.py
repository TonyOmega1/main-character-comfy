"""Record your own line: the audio backend (take scoring, speaking window, fit, voice chain, mix). CPU only.

    from main_character_comfy.recordline import pick_and_align
    r = pick_and_align(video, [take1, take2], "You're not you when you're gay.", out_dir=work / "recordline")
    r["chosen_take"], r["takes"], r["window_start"], r["window_end"], r["stretch_ratio"],
    r["aligned_voice_wav"], r["final_mix_wav"], r["warnings"]

Outputs (all on the clip's own timeline, full clip length, unless noted):
* ``aligned_voice_wav``: the new line alone, dry (EQ + level, no reverb), 48 kHz mono float. This is the lip-sync
  drive: pass it as ``voice`` to ``extras._lipsync_line``, which cuts the window itself.
* ``infinitetalk_wav``: a ready InfiniteTalk input, 16 kHz mono, with DubMaster's 0.6 s silent head pad, starting at
  the window start. Render time t maps to clip time ``infinitetalk_clip_t0 + t``.
* ``final_mix_wav``: the finished soundtrack (new line + the clip's music and effects), 48 kHz stereo 24-bit.
  ``mux(video, final_mix_wav, out_mp4)`` makes the video (stream copy, AAC 320k).

Hook (for LocalComfyMaster): in ``extras.run_redub`` when ``len(takes) > 1``:
``r = recordline.pick_and_align(video, takes, text or None, out_dir=work / "recordline")``, then
``final = recordline.mux(video, Path(r["final_mix_wav"]), work / "redubbed.mp4")`` and, if lipsync,
``_lipsync_line(pipe, video, final, Path(r["aligned_voice_wav"]), r["voice_start"], r["voice_end"] -
r["voice_start"], work, record, results, seed)``; surface ``r["warnings"]`` as UI notes.
"""

from .chain import ChainSettings
from .core import RecordLineResult, mux, pick_and_align, process_line
from .deps import OptionalDependencyMissing, RecordLineError, status

__all__ = ["pick_and_align", "process_line", "mux", "RecordLineResult", "ChainSettings", "RecordLineError",
           "OptionalDependencyMissing", "status"]
