"""The MC MCP server: every MC capability as a tool, plus a few resources and prompts.

Tools are thin wrappers over :class:`main_character_comfy.service.MCService`; renders are queued jobs (one GPU lane
shared with the MC window), so a tool that starts a render returns a job id at once. Agents then call
``wait_for_job`` (blocks up to N seconds) or ``get_job``, and ``get_output`` for the files.
"""



import base64
import hashlib
import hmac
import json
import time
from pathlib import Path
from typing import Annotated, Any, Callable, Literal
from urllib.parse import quote

from pydantic import Field

from ..pipeline import InputError
from ..service import KINDS, MCService, AccessError, SETTINGS_HELP, VERSION

INSTRUCTIONS = """\
MC (Main Character) puts the user's face or avatar, and their own cloned voice, into memes and clips. It runs on
the user's own GPU through local ComfyUI: free, open weights, unlimited retries.

How to drive it:
1. mc_status: GPU, hardware tier (lite / standard / max) and whether ComfyUI is up. list_templates: meme formats,
   styles, eras, options.
2. Files: list_library shows what's already uploaded; upload_file adds a selfie / avatar (references), a driving
   clip (clips), a voice sample (voices) or audio (audio). Pass files back as the 'path' strings MC returns
   (e.g. "library/references/me.png"). Uploading a reference returns reference_quality: if it has warnings, tell the
   user and suggest a better photo (sharp, well lit, face and outfit visible, ideally full body, ~1024 px+).
   Clip from a link: probe_clip / find_clips to find the sharpest copy, then
   fetch_clip (best video + audio, downloaded locally into library/clips). Sharper sources give better results:
   1080p or 4K, steady camera, good lighting, clean audio.
3. Start a render with the tool that matches the request (make_meme, swap_into_clip, copy_the_vibe, talk, sing,
   duet, redub_line, restyle_image, upscale_video, voice_tts, ...). It returns a job id immediately.
4. wait_for_job (repeat until status is done / error / cancelled), then get_output for the file (inline preview
   and, over HTTP, a download link). Renders take ~20 s (stills) to several minutes (video); tell the user.

Rules: ask the user before setting i_have_rights=true (it means: it's their own face / voice / footage, or they
have permission). SFW only. MC never posts anything anywhere. One GPU job runs at a time; jobs queue in order.
"""

ReadOnly = {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False}
Render = {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False}

Path_ = Annotated[str, Field(description="a path MC returned, e.g. 'library/references/me.png' or 'examples/x.jpg'")]
OptPath = Annotated[str | None, Field(description="a path MC returned (library/..., outputs/..., examples/...)")]
Rights = Annotated[bool, Field(description="true only after the user confirms it's their own face / voice / footage "
                                           "(or they have permission). Required for anything uploaded.")]
Tier = Annotated[Literal["auto", "lite", "standard", "max"] | None,
                 Field(description="hardware tier override; omit to use the setting (auto = from VRAM)")]
Seed = Annotated[int | None, Field(description="seed; change it to reroll a take")]


def sign(secret: bytes, rel: str, exp: int) -> str:
    return hmac.new(secret, f"{rel}|{exp}".encode(), hashlib.sha256).hexdigest()[:32]


def build_server(service: MCService | None = None, *, public_url: Callable[[], str | None] | None = None,
                 link_secret: bytes | None = None):
    """Create the MCPServer. ``public_url()`` (HTTP mode) is used to hand out signed download links."""
    from mcp.server.mcpserver import Image, MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp.types import ToolAnnotations

    svc = service or MCService()
    mcp = MCPServer("main-character", title="MC: Main Character", version=VERSION, instructions=INSTRUCTIONS,
                    website_url="https://github.com/TonyOmega1/main-character-comfy")
    secret = link_secret or b""

    def tool(name: str, title: str, hints: dict, desc: str):
        def deco(fn):
            def wrapped(*a, **kw):
                try:
                    return fn(*a, **kw)
                except (InputError, AccessError, ValueError, KeyError) as exc:
                    raise ToolError(str(exc).strip("'\"")) from None
            wrapped.__name__, wrapped.__doc__ = fn.__name__, fn.__doc__
            wrapped.__signature__ = __import__("inspect").signature(fn)
            wrapped.__annotations__ = fn.__annotations__
            mcp.add_tool(wrapped, name=name, title=title, description=desc,
                         annotations=ToolAnnotations(title=title, **hints))
            return fn
        return deco

    def link(path: str) -> str | None:
        base = public_url() if public_url else None
        if not base or not secret:
            return None
        exp = int(time.time()) + 24 * 3600
        return f"{base.rstrip('/')}/files/{sign(secret, path, exp)}/{exp}/{quote(path)}"

    def with_links(job: dict[str, Any]) -> dict[str, Any]:
        res = job.get("result")
        if res and res.get("files") and public_url and public_url():
            res["download_links"] = {f: link(f) for f in res["files"]}
        return job

    # ------------------------------------------------------------------ status & catalog
    @tool("mc_status", "MC status", ReadOnly,
          "Start here. GPU name, VRAM, hardware tier (lite 8-12 GB / standard 16 GB / max 24 GB+) and what it "
          "unlocks, whether ComfyUI and the API proxy are up, what's rendering now, the agent queue, DeepFilterNet, "
          "MC's folders and current settings.")
    def mc_status() -> dict:
        return svc.system_info()

    @tool("check_models", "Check installed models", ReadOnly,
          "Ask ComfyUI which nodes and model files are installed and list what's missing for a tier (with the "
          "folder each file goes in). Use when a render fails with a missing model.")
    def check_models(tier: Tier = None) -> str:
        return svc.check_models(tier)

    @tool("list_templates", "List templates and options", ReadOnly,
          "Built-in meme formats (key, title, default captions, line), look styles, meme eras, effects, scene "
          "ideas for copy the vibe, swap options and tiers. Use the 'key' values in make_meme / make_meme_pack.")
    def list_templates() -> dict:
        return svc.list_templates()

    @tool("list_library", "List uploaded files", ReadOnly,
          "Files the user already gave MC: references (selfies, avatars), clips (driving videos), voices (voice "
          "samples), audio (songs, recordings), recipes, plus built-in examples. Returns paths to pass to other tools.")
    def list_library(kind: Annotated[Literal["references", "clips", "voices", "audio", "recipes"] | None,
                                     Field(description="only this kind")] = None) -> dict:
        return svc.list_library(kind)

    @tool("upload_file", "Upload a file to MC", {**Render, "idempotentHint": False},
          "Add a file to MC's library: a selfie or avatar (kind=references), a driving clip you have the rights to "
          "(clips), a 10-30 s voice sample (voices), or a song / recording / line (audio). Give ONE of: "
          "content_base64 (small files, a few MB), url (public http(s) link), or source_path (a file already in "
          "MC's folders). Over HTTP, big files can also be PUT to /upload/<kind>/<filename> with the bearer token. "
          "Returns the path to use in other tools.")
    def upload_file(kind: Annotated[Literal["references", "clips", "voices", "audio", "recipes"],
                                    Field(description="what the file is")],
                    filename: Annotated[str, Field(description="file name with extension, e.g. me.png")] = "",
                    content_base64: Annotated[str | None, Field(description="file bytes, base64 (data: URLs ok)")] = None,
                    url: Annotated[str | None, Field(description="public http(s) URL to download")] = None,
                    source_path: OptPath = None) -> dict:
        return svc.upload(kind, filename, content_base64, url, source_path)

    @tool("check_reference", "Check a reference photo's quality", ReadOnly,
          "Grade a selfie / avatar before a render: good, ok or weak, with friendly warnings (low resolution, "
          "blurry face, no face found, face too small, tight crop with no outfit, too dark). upload_file runs this "
          "automatically for references (field reference_quality) and render tools add reference_quality when a "
          "reference is weak. Always pass the warnings on to the user. Best: sharp, well lit, face and outfit "
          "clearly visible, ideally full body, no heavy filters, at least about 1024 px.")
    def check_reference(path: Path_, subject: Annotated[Literal["auto", "photo", "avatar"],
                                                       Field(description="avatars may have no human face")] = "auto") -> dict:
        return svc.check_reference(path, subject)

    # --------------------------------------------------------- bring your own clip
    Web = {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True}

    @tool("find_clips", "Find the sharpest copy of a clip", Web,
          "Search the web (youtube search via yt-dlp) for copies of a clip the user describes, and report each "
          "candidate's best available resolution, fps and length, sharpest first. Nothing is downloaded. Use it to "
          "pick the highest-quality source before fetch_clip. Only for clips the user has the rights to use.")
    def find_clips(query: Annotated[str, Field(description="what the clip is, e.g. 'my channel dance tutorial 4k'")],
                   limit: Annotated[int, Field(description="how many candidates (1-10)", ge=1, le=10)] = 5) -> dict:
        return svc.find_clips(query, limit)

    @tool("probe_clip", "Check a clip link's best quality", Web,
          "Look at a link (youtube, tiktok, instagram, x, vimeo, reddit, a direct .mp4, and the ~1800 other sites "
          "yt-dlp knows) and report the title, length and every available resolution without downloading. Compare "
          "links with it to find the sharpest copy.")
    def probe_clip(url: Annotated[str, Field(description="public http(s) link to one clip")]) -> dict:
        return svc.probe_clip(url)

    @tool("fetch_clip", "Bring your own clip from a link",
          {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
          "Download the best available video + best audio at a link, locally on this PC, merged into one mp4 in "
          "the user's library (library/clips). Returns a job id at once: wait_for_job, then use result.data.path "
          "as the clip in swap_into_clip / copy_the_vibe / redub_line. start_seconds / end_seconds grab just "
          "the part needed (faster). Warns when the copy is under 1080p. Ask the user first: they must have the "
          "rights to the clip (i_have_rights). Runs on the cpu, so it never waits behind a render.")
    def fetch_clip(url: Annotated[str, Field(description="public http(s) link to one clip")],
                   i_have_rights: Rights = False,
                   name: Annotated[str | None, Field(description="short file name, e.g. 'my-dance'")] = None,
                   start_seconds: Annotated[float | None, Field(description="grab from here (s)", ge=0)] = None,
                   end_seconds: Annotated[float | None, Field(description="grab up to here (s)", gt=0)] = None,
                   max_height: Annotated[int | None, Field(description="cap resolution, e.g. 1080; default "
                                                                       "is the sharpest available", ge=144)] = None) -> dict:
        return svc.fetch_clip(url=url, name=name, start_seconds=start_seconds, end_seconds=end_seconds,
                              max_height=max_height, i_have_rights=i_have_rights)

    # ------------------------------------------------------------------ renders
    @tool("make_meme", "Make a meme starring you", Render,
          "Meme Recreate: put the person (selfie) or avatar into a meme format. animate=false gives a captioned "
          "still in ~20 s; animate=true (default) a 3-5 s 9:16 video in ~2 min. say_line=true lip-syncs a line in "
          "the user's cloned voice (voice='clone' + voice_sample) or the built-in voice. For an idea that isn't "
          "a built-in format, use meme='custom' with custom_scene + custom_action. To animate a still you already "
          "made, pass it as scene_image.")
    def make_meme(photo: OptPath = None,
                  meme: Annotated[str, Field(description="format key from list_templates, or 'custom'")] = "side_eye",
                  animate: Annotated[bool, Field(description="false = instant still only")] = True,
                  top: Annotated[str | None, Field(description="top caption (default: the format's)")] = None,
                  bottom: Annotated[str | None, Field(description="bottom caption")] = None,
                  style: Annotated[str, Field(description="look: none keeps the photo; 3d, anime, comic, ...")] = "none",
                  era: Annotated[str, Field(description="caption era: classic, deep_fried, vaporwave, early_web, tiktok")] = "classic",
                  effect: Annotated[str | None, Field(description="none, punch_in, shake, boomerang")] = None,
                  seconds: Annotated[Literal[3, 5] | None, Field(description="length")] = None,
                  say_line: bool = False,
                  line: Annotated[str, Field(description="the line to say (default: the format's)")] = "",
                  voice: Annotated[Literal["builtin", "clone", "my_recording"], Field(description="voice for say_line")] = "builtin",
                  voice_sample: OptPath = None, recording: OptPath = None,
                  custom_scene: Annotated[str, Field(description="custom format: where it happens")] = "",
                  custom_action: Annotated[str, Field(description="custom format: what you do")] = "",
                  scene_image: OptPath = None,
                  subject: Annotated[Literal["auto", "photo", "avatar"], Field(description="avatar keeps a drawn art style")] = "auto",
                  i_have_rights: Rights = False, tier: Tier = None, seed: Seed = None) -> dict:
        return svc.make_meme(photo=photo, meme=meme, animate=animate, top=top, bottom=bottom, style=style, era=era,
                             effect=effect, seconds=seconds, say_line=say_line, line=line, voice=voice,
                             voice_sample=voice_sample, recording=recording, custom_scene=custom_scene,
                             custom_action=custom_action, scene_image=scene_image, subject=subject,
                             i_have_rights=i_have_rights, tier=tier, seed=seed)

    @tool("make_meme_pack", "Make a meme pack", Render,
          "Several meme formats from one selfie in one go (a week of posts). Each format becomes its own video.")
    def make_meme_pack(photo: Path_, formats: Annotated[list[str], Field(description="format keys")],
                       style: str = "none", i_have_rights: Rights = False, tier: Tier = None,
                       seed: Seed = None) -> dict:
        return svc.make_meme_pack(photo=photo, memes_list=formats, style=style, i_have_rights=i_have_rights,
                                  tier=tier, seed=seed)

    @tool("find_people", "Find and number the people in a clip", Render,
          "Before a multi-person swap: scan the clip's first frame (at clip_start), number everyone 1, 2, 3 left to "
          "right and save a preview image. The result says how many people there are; then call swap_into_clip "
          "with one reference per person in that order.")
    def find_people(clip: Path_, clip_start: float = 0.0, clip_seconds: float = 5.0, tier: Tier = None) -> dict:
        return svc.find_people(clip=clip, clip_start=clip_start, clip_seconds=clip_seconds, tier=tier)

    @tool("swap_into_clip", "Swap you (and friends) into a clip", Render,
          "Swap Me In (open-weights Openjutsu): recast the people in a clip the user has the rights to, keeping its "
          "motion, camera, timing and audio. Single swap: photo=<reference>. Group swap: people=[ref1, ref2, ...] "
          "numbered left to right ('-' leaves that person as filmed) plus describe=[look1, look2, ...] (short "
          "outfit/colour descriptions: with every person swapped and described, 2-6 people, clip <= ~5 s, MC "
          "renders the whole group in ONE SCAIL-2 pass and sharpens it with SeedVR2). Several minutes per take.")
    def swap_into_clip(clip: Path_, photo: OptPath = None,
                       people: Annotated[list[str] | None, Field(description="one reference per person, left to right; '-' = keep")] = None,
                       describe: Annotated[list[str] | None, Field(description="one short look description per person, same order")] = None,
                       face: Annotated[list[str] | None, Field(description="per person: auto, on (copy the dancer's face), off")] = None,
                       clip_start: float = 0.0,
                       clip_seconds: Annotated[float, Field(description="how much of the clip, 1-30 s (tier caps)")] = 5.0,
                       clip_audio: Annotated[Literal["original", "mute", "my_voice"], Field(description="soundtrack")] = "original",
                       voice_sample: OptPath = None,
                       group_engine: Literal["auto", "scail", "chain"] = "auto",
                       upscale: Literal["auto", "seedvr2", "lanczos"] = "auto",
                       resolution: Literal["auto", "480p", "720p"] = "auto",
                       erase_props: Annotated[str, Field(description="erase these props first (VOID), e.g. 'backpack'")] = "",
                       relight: Annotated[bool | None, Field(description="match scene lighting (default auto)")] = None,
                       remove_props: bool = False, style: str = "none",
                       look_tweak: Annotated[str, Field(description="extra look detail, e.g. 'wearing a tuxedo'")] = "",
                       top: str = "", bottom: str = "",
                       subject: Literal["auto", "photo", "avatar"] = "auto",
                       i_have_rights: Rights = False, tier: Tier = None, seed: Seed = None,
                       steps: Annotated[int | None, Field(description="sampler steps (advanced)")] = None,
                       style_lock: Annotated[Literal["off", "auto", "on"], Field(description="keep drawn characters drawn (pending a real test; default off)")] = "off",
                       drivers: Annotated[list[str] | None, Field(description="per person, whose moves they follow: '-' own, 'person:N', or another clip")] = None,
                       split_cuts: Annotated[Literal["auto", "on", "off"], Field(description="hard cuts: swap each shot on its own")] = "auto",
                       scail_chunks: Annotated[bool, Field(description="group pass on clips over ~5 s: chained SCAIL-2 chunks (experimental)")] = False,
                       rotate: Annotated[Literal[0, 90, 180, 270], Field(description="extra clockwise turn when a phone clip's rotation tag is wrong")] = 0,
                       ref_closeup: Annotated[Literal["off", "auto"], Field(description="close-up shots: crop a full-body reference to the face (pending a real test)")] = "off") -> dict:
        return svc.swap(clip=clip, photo=photo, people=people, describe=describe, face=face, clip_start=clip_start,
                        clip_seconds=clip_seconds, clip_audio=clip_audio, voice_sample=voice_sample,
                        group_engine=group_engine, upscale=upscale, resolution=resolution, erase_props=erase_props,
                        relight=relight, remove_props=remove_props, style=style, look_tweak=look_tweak, top=top,
                        bottom=bottom, subject=subject, i_have_rights=i_have_rights, tier=tier, seed=seed,
                        steps=steps, style_lock=style_lock, drivers=drivers, split_cuts=split_cuts,
                        scail_chunks=scail_chunks, rotate=rotate, ref_closeup=ref_closeup)

    @tool("copy_the_vibe", "Copy a move into a new scene", Render,
          "Copy the Vibe: take only the MOTION of a reference clip (pose + face) and render the user doing it in a "
          "brand-new scene / background from a prompt (e.g. 'luxury hotel hallway at night'). No pixels of the "
          "reference end up in the result. Use this for 'me doing <move> in <place>' or to restyle the background.")
    def copy_the_vibe(clip: Path_, photo: Path_,
                      scene: Annotated[str, Field(description="the new scene / background")] = "",
                      clip_start: float = 0.0, clip_seconds: float = 5.0,
                      clip_audio: Literal["original", "mute", "my_voice"] = "original", voice_sample: OptPath = None,
                      style: str = "none", look_tweak: str = "", resolution: Literal["auto", "480p", "720p"] = "auto",
                      top: str = "", bottom: str = "", subject: Literal["auto", "photo", "avatar"] = "auto",
                      scene_image: Annotated[str | None, Field(description="use this ready-made scene still instead of generating one")] = None,
                      i_have_rights: Rights = False, tier: Tier = None, seed: Seed = None) -> dict:
        return svc.copy_the_vibe(clip=clip, photo=photo, scene=scene, clip_start=clip_start, clip_seconds=clip_seconds,
                                 clip_audio=clip_audio, voice_sample=voice_sample, style=style, look_tweak=look_tweak,
                                 resolution=resolution, top=top, bottom=bottom, subject=subject,
                                 scene_image=scene_image, i_have_rights=i_have_rights, tier=tier, seed=seed)

    @tool("talk", "Talking avatar with lip sync", Render,
          "Lip sync (InfiniteTalk): the photo / avatar / described character says a script, in a cloned voice "
          "(voice='clone' + voice_sample), the built-in voice, or the user's own recording (voice='my_recording'). "
          "Captioned 9:16 video. Optional character_voice re-voices it. Needs a 16 GB+ GPU.")
    def talk(photo: OptPath = None, character: Annotated[str, Field(description="describe a character instead of a photo")] = "",
             avatar: Annotated[str | None, Field(description="ready-made portrait (skips the look step)")] = None,
             script: str = "", voice: Literal["builtin", "clone", "my_recording"] = "builtin",
             voice_sample: OptPath = None, recording: OptPath = None, character_voice: OptPath = None,
             style: Annotated[str, Field(description="none keeps the photo; 3d, anime, ...")] = "none",
             captions: bool = True, smooth: Literal["off", "rife", "ffmpeg"] = "off",
             i_have_rights: Rights = False, tier: Tier = None, seed: Seed = None) -> dict:
        return svc.talk(photo=photo, character=character, avatar=avatar, script=script, voice=voice,
                        voice_sample=voice_sample, recording=recording, character_voice=character_voice, style=style,
                        captions=captions, smooth=smooth, i_have_rights=i_have_rights, tier=tier, seed=seed)

    @tool("sing", "Lip-sync to a song", Render,
          "Sing: the photo / avatar lip-syncs a section of a song (3-90 s). A vocals-only stem syncs much better; "
          "the full mix is the soundtrack. Needs a 16 GB+ GPU.")
    def sing(song: Path_, photo: OptPath = None, character: str = "", avatar: OptPath = None,
             vocals: OptPath = None, song_start: float = 0.0, song_seconds: float = 20.0,
             lyrics: Annotated[str, Field(description="optional, for captions")] = "", style: str = "none",
             i_have_rights: Rights = False, tier: Tier = None, seed: Seed = None) -> dict:
        return svc.sing(song=song, photo=photo, character=character, avatar=avatar, vocals=vocals,
                        song_start=song_start, song_seconds=song_seconds, lyrics=lyrics, style=style,
                        i_have_rights=i_have_rights, tier=tier, seed=seed)

    @tool("duet", "Two-character skit", Render,
          "Duet / podcast skit: two characters in one shot, one line per turn written as 'Name: line'. Each speaker "
          "can have a photo and a voice sample (else built-in voices).")
    def duet(script: str, photo: OptPath = None, photo_b: OptPath = None, character: str = "",
             voice_sample: OptPath = None, voice_sample_b: OptPath = None, style: str = "3d",
             i_have_rights: Rights = False, tier: Tier = None, seed: Seed = None) -> dict:
        return svc.dialogue(script=script, photo=photo, photo_b=photo_b, character=character,
                            voice_sample=voice_sample, voice_sample_b=voice_sample_b, style=style,
                            i_have_rights=i_have_rights, tier=tier, seed=seed)

    @tool("batch_talk", "Many talking clips", Render,
          "Batch: many scripts in (one per line, or blocks separated by a blank line), many talking clips out with "
          "the same face and voice.")
    def batch_talk(scripts: str, photo: OptPath = None, character: str = "", voice_sample: OptPath = None,
                   style: str = "none", i_have_rights: Rights = False, tier: Tier = None, seed: Seed = None) -> dict:
        return svc.batch_talk(scripts=scripts, photo=photo, character=character, voice_sample=voice_sample,
                              style=style, i_have_rights=i_have_rights, tier=tier, seed=seed)

    @tool("voice_tts", "Say text in a cloned voice", Render,
          "Voice clone + TTS (Chatterbox): say any text in the voice of a 10-30 s voice_sample, or the built-in "
          "voice. Audio only (wav). Use for voice lines, or to make a line for redub_line.")
    def voice_tts(text: str, voice_sample: OptPath = None,
                  expressiveness: Annotated[float, Field(ge=0.25, le=2.0, description="0.5 calm, 1+ dramatic")] = 0.55,
                  i_have_rights: Rights = False, seed: Seed = None) -> dict:
        return svc.tts(text=text, voice_sample=voice_sample, expressiveness=expressiveness,
                       i_have_rights=i_have_rights, seed=seed)

    @tool("convert_voice", "Voice changer", Render,
          "Re-voice existing speech (an audio file or a video's audio) with the timbre of target_voice (Chatterbox "
          "VC). Needs rights to both voices.")
    def convert_voice(source: Path_, target_voice: Path_, i_have_rights: Rights = False) -> dict:
        return svc.convert_voice(source=source, target_voice=target_voice, i_have_rights=i_have_rights)

    @tool("redub_line", "Redub a line in a finished clip", Render,
          "Replace or add one spoken line in a finished video, mixed to sound produced: the line is the user's "
          "recording, or text said in their cloned voice. MC's clean chain: DeepFilterNet3 speech cleanup, retime "
          "to fit_seconds, gentle EQ toward the scene, light compression, short reverb, the clip's audio smoothly "
          "ducked under the line, one static gain (no loudnorm, no fast limiter). Video is untouched unless "
          "lipsync=true (the mouth is re-synced to the line around its window only).")
    def redub_line(video: Path_, recording: OptPath = None,
                   text: Annotated[str, Field(description="say this instead of a recording")] = "",
                   voice_sample: OptPath = None,
                   start: Annotated[float, Field(description="where the line starts in the clip (s)")] = 0.0,
                   fit_seconds: Annotated[float | None, Field(description="retime the line to this length (s)")] = None,
                   duck: Annotated[float, Field(ge=0, le=1, description="clip audio under the line: 0 replaces it, 0.28 = -11 dB")] = 0.28,
                   denoise: Literal["auto", "deepfilternet", "off"] = "auto",
                   voice_level_db: Annotated[float, Field(ge=-12, le=12, description="voice level vs the clip")] = 0.0,
                   i_have_rights: Rights = False, seed: Seed = None,
                   lipsync: Annotated[bool, Field(description="also re-sync the mouth to the new line (one GPU job; pending a real test)")] = False) -> dict:
        return svc.redub(video=video, recording=recording, text=text, voice_sample=voice_sample, start=start,
                         fit_seconds=fit_seconds, duck=duck, denoise=denoise, voice_level_db=voice_level_db,
                         i_have_rights=i_have_rights, seed=seed, lipsync=lipsync)

    @tool("restyle_image", "Restyle a photo or swap its background", Render,
          "FLUX.2 klein reference edit of one image: give it a new look (style: 3d, anime, comic, clay, ...), put "
          "the person in a new background ('background'), or both, while keeping their identity. Use it to match "
          "references to one art style or to make a scene still for make_meme / copy_the_vibe (scene_image).")
    def restyle_image(image: Path_, style: str = "none",
                      background: Annotated[str, Field(description="the new background, e.g. 'flat purple cartoon city at night'")] = "",
                      instruction: Annotated[str, Field(description="any extra edit instruction")] = "",
                      subject: Literal["auto", "photo", "avatar"] = "auto", landscape: bool = False,
                      i_have_rights: Rights = False, tier: Tier = None, seed: Seed = None) -> dict:
        return svc.restyle(image=image, style=style, background=background, instruction=instruction,
                           subject=subject, landscape=landscape, i_have_rights=i_have_rights, tier=tier, seed=seed)

    @tool("upscale_video", "Upscale a video", Render,
          "Sharpen a finished clip to 1080p (or width x height): SeedVR2 restore + upscale (16 GB+, about 2.3 s per "
          "frame) or plain lanczos (instant). Audio is kept.")
    def upscale_video(video: Path_, engine: Literal["seedvr2", "lanczos"] = "seedvr2", width: int | None = None,
                      height: int | None = None, tier: Tier = None, seed: Seed = None) -> dict:
        return svc.upscale_video(video=video, engine=engine, width=width, height=height, tier=tier, seed=seed)

    @tool("remix_recipe", "Remix an mc_recipe.json", Render,
          "Re-run a finished take's mc_recipe.json (in its _work folder) with new uploads, e.g. someone else's selfie.")
    def remix_recipe(recipe: Path_, photo: OptPath = None, clip: OptPath = None, voice_sample: OptPath = None,
                     i_have_rights: Rights = False, tier: Tier = None) -> dict:
        return svc.remix(recipe=recipe, photo=photo, clip=clip, voice_sample=voice_sample,
                         i_have_rights=i_have_rights, tier=tier)

    @tool("save_meme_format", "Save a meme format", {**Render, "idempotentHint": True},
          "Save a custom format (scene + action + captions) or a tweaked built-in one as a reusable community "
          "format. It shows up in list_templates and the MC window. SFW, original ideas only.")
    def save_meme_format(name: str, meme: str = "custom", scene: str = "", action: str = "", top: str = "",
                         bottom: str = "", line: str = "", emoji: str = "") -> dict:
        return svc.save_format(name=name, meme=meme, scene=scene, action=action, top=top, bottom=bottom, line=line,
                               emoji=emoji)

    # ------------------------------------------------------------------ jobs & outputs
    @tool("get_job", "Job status", ReadOnly,
          "Status of one job: queued (with position), running (stage, progress, message), done (result files), "
          "error (a plain-English fix) or cancelled.")
    def get_job(job_id: str) -> dict:
        return with_links(svc.job(job_id))

    @tool("wait_for_job", "Wait for a job", ReadOnly,
          "Block until the job finishes or timeout_s passes (max 600), then return its status. Call again while it "
          "is still queued / running. Renders take ~20 s (stills) to 10+ minutes (group swaps).")
    def wait_for_job(job_id: str, timeout_s: Annotated[float, Field(ge=0, le=600)] = 120) -> dict:
        return with_links(svc.wait(job_id, timeout_s))

    @tool("list_jobs", "List jobs and the queue", ReadOnly,
          "Recent jobs from agents AND the MC window (newest first), what holds the GPU right now, and ComfyUI's own "
          "queue (other apps may be using ComfyUI).")
    def list_jobs(status: Literal["queued", "running", "done", "error", "cancelled"] | None = None,
                  limit: Annotated[int, Field(ge=1, le=100)] = 20) -> dict:
        return svc.list_jobs(status, limit)

    @tool("cancel_job", "Cancel a job", {**Render, "destructiveHint": True, "idempotentHint": True},
          "Cancel a queued job, or stop a running one (the current ComfyUI job is interrupted). Finished files are "
          "never deleted.")
    def cancel_job(job_id: str) -> dict:
        return svc.cancel(job_id)

    @tool("list_outputs", "List finished renders", ReadOnly,
          "Recent output folders and their result files (mp4, gif, png, wav), newest first.")
    def list_outputs(limit: Annotated[int, Field(ge=1, le=100)] = 20) -> dict:
        return svc.list_outputs(limit)

    @tool("get_output", "Get an output file", ReadOnly,
          "Info for one file MC made (size, length, absolute path on the PC), a 24 h download link in HTTP mode, "
          "and the file itself inline when it's small: images and GIFs as an image, other files as base64 when "
          "include_data=true.")
    def get_output(path: Path_, include_data: Annotated[bool, Field(description="return small files inline")] = True):
        info = svc.file_info(path)
        url = link(info["path"])
        if url:
            info["download_url"] = url
        out: list[Any] = [json.dumps(info, indent=1)]
        if include_data:
            p = Path(info["absolute_path"])
            suf = p.suffix.lower()
            if suf in (".png", ".jpg", ".jpeg", ".gif", ".webp") and info["bytes"] <= 6 * 2 ** 20:
                out.append(Image(data=p.read_bytes(), format={"jpg": "jpeg"}.get(suf[1:], suf[1:])))
            elif info["bytes"] <= 4 * 2 ** 20:
                out.append(json.dumps({"base64": base64.b64encode(p.read_bytes()).decode()}))
            elif suf == ".mp4":
                gif = p.with_suffix(".gif")
                if gif.exists() and gif.stat().st_size <= 6 * 2 ** 20:
                    out.append(Image(data=gif.read_bytes(), format="gif"))
        return out

    # ------------------------------------------------------------------ settings
    @tool("get_settings", "Get MC settings", ReadOnly,
          "Agent defaults: tier, allow_heavy, watermark, hd (1080p), seed, wait_for_comfy_idle, with what each does.")
    def get_settings() -> dict:
        return {"settings": svc.get_settings(), "help": SETTINGS_HELP}

    @tool("update_settings", "Change MC settings", {**Render, "idempotentHint": True},
          "Change agent defaults (only the ones you pass). Security settings (token, bind address, folders) can't "
          "be changed from here.")
    def update_settings(tier: Literal["auto", "lite", "standard", "max"] | None = None,
                        allow_heavy: bool | None = None, watermark: bool | None = None, hd: bool | None = None,
                        seed: int | None = None, wait_for_comfy_idle: bool | None = None,
                        comfy_idle_timeout_min: Annotated[int | None, Field(ge=1, le=1440)] = None) -> dict:
        return svc.update_settings(tier=tier, allow_heavy=allow_heavy, watermark=watermark, hd=hd, seed=seed,
                                   wait_for_comfy_idle=wait_for_comfy_idle,
                                   comfy_idle_timeout_min=comfy_idle_timeout_min)

    # ------------------------------------------------------------------ resources
    @mcp.resource("mc://guide", name="guide", title="How to drive MC", mime_type="text/markdown")
    def guide() -> str:
        return INSTRUCTIONS

    @mcp.resource("mc://templates", name="templates", title="Meme formats and options", mime_type="application/json")
    def templates_res() -> str:
        return json.dumps(svc.list_templates(), indent=1)

    @mcp.resource("mc://status", name="status", title="GPU, tier and health", mime_type="application/json")
    def status_res() -> str:
        return json.dumps(svc.system_info(), indent=1, default=str)

    @mcp.resource("mc://jobs/{job_id}", name="job", title="One job", mime_type="application/json")
    def job_res(job_id: str) -> str:
        return json.dumps(svc.job(job_id), indent=1)

    # ------------------------------------------------------------------ prompts
    @mcp.prompt(name="make_a_meme", title="Make a meme",
                description="Turn a plain-language idea into an MC meme, step by step.")
    def make_a_meme(idea: str, who: str = "me") -> str:
        return (f"Make an MC meme: {idea}. Star: {who}.\n"
                "1. Call list_library (references) and list_templates. If there's no reference for the star, ask the "
                "user for a selfie or avatar and upload_file it.\n"
                "2. If a built-in format fits, use make_meme with its key; otherwise meme='custom' with a short "
                "custom_scene and custom_action, and write punchy lowercase top/bottom captions.\n"
                "3. Ask the user to confirm the face is theirs (or they have permission) before i_have_rights=true.\n"
                "4. Start with animate=false for a quick still, show it with get_output, then animate (scene_image = "
                "the still) once they like it. wait_for_job until done and hand over the mp4 and gif.")

    @mcp.prompt(name="swap_me_in", title="Swap me (and friends) into a clip",
                description="Recast the people in a clip the user owns, one reference per person.")
    def swap_me_in(request: str) -> str:
        return (f"Do this with MC's Swap Me In: {request}\n"
                "1. Find the clip and references in list_library (upload_file anything missing; clips must be the "
                "user's own or licensed).\n"
                "2. More than one person in the clip? Run find_people first and show the numbered preview, then map "
                "each person (left to right) to a reference, or '-' to keep them.\n"
                "3. Write one short look description per swapped person (outfit, colours, hair) for describe=[...].\n"
                "4. Confirm rights with the user, call swap_into_clip, wait_for_job, and return the result.")

    @mcp.prompt(name="redub_a_line", title="Redub a line",
                description="Replace a spoken line in a finished clip so it sounds produced.")
    def redub_a_line(video: str, line: str = "") -> str:
        return (f"Redub a line in {video}. New line: {line or '(ask the user for a recording or the text)'}.\n"
                "Ask for the start time and roughly how long the line should last. Use the user's recording if they "
                "have one (upload_file kind=audio), otherwise text + their voice sample. Call redub_line with start "
                "and fit_seconds, then wait_for_job and share the result.")

    return mcp
