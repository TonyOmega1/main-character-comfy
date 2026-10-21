"""All user-facing microcopy in one place.

Tone: terminally online, never mean. Since the v3 "Noir Editorial" theme the chrome
(labels, buttons, header) is deadpan lowercase; jokes stay in the loading lines.

Kept separate from layout (ui.py) and styling (themes/) so copy can be tuned
without touching either.
"""

from __future__ import annotations

import random

APP_NAME = "MC: Main Character"
SHORT_NAME = "MC"
TAGLINE = "Be the main character of every meme."
SUBLINE = "NPC to MC in one click. 0 credits, unlimited retries, runs on your own GPU."
BADGES = ["main character energy", "0 credits", "unlimited retries", "100% open weights"]

# Deadpan editorial chrome (v3 mockup): small lowercase labels, numbered steps.
KICKER = "v1 · free · open source"
HEADLINE = "Be the main character of every meme"   # the theme adds the red period
TAG = "face + format + caption = meme. that's it."
EXPANSION = "main character"            # small, right under the MC. logo
EXPANSION_LINE = "mc = main character."  # spelled out for screen readers / tooltips
# Fake legal lines under the logo (directly under "main character"). Meme only: real license is Apache-2.0.
# LEGAL_JOKES: one copyright gag is picked at random per page load; LEGAL_JOKE is the no-JS default.
# PATENT_JOKE: always shown under the logo with the copyright gag (Tony: "Patent pending. (It is not.)").
LEGAL_JOKES = [
    "™ © you, probably. all rights reserved by the group chat.",
    "© 2026 main character energy llc (not a real llc).",
    "™ & © your mom. jk. it's apache-2.0 licensed.",
    "trademark registered in the group chat only.",
    "© nobody. side characters may not reproduce this.",
]
LEGAL_JOKE = LEGAL_JOKES[0]
PATENT_JOKE = "Patent pending. (It is not.)"
LEGAL_DISCLAIMER = "the ™/© line up top is a joke. mc is open source: apache-2.0."
USE_DISCLAIMER = "user discretion advised. you're responsible for what you make and share with mc. the makers aren't responsible for misuse."
LOCAL_NOTE = "local. nothing leaves your pc."
GPU_LABEL = "gpu: {tier}"
TOGGLE_DARK = "dark"
TOGGLE_LIGHT = "light"
# Public creator credit (alias + X handle). One place; the UI, head meta and package metadata read it.
CREATOR_NAME = "Oroku Saki"
CREATOR_HANDLE = "@Tony_Omega"
CREATOR_URL = "https://x.com/Tony_Omega"
MADE_BY = "made by"
# Community feedback loop: footer mailto (every MC build gets one).
FEEDBACK_EMAIL = "oroku.sakis.homies+mc@mail.grokbot.com"
FEEDBACK_LABEL = "send feedback"
# Tips: OFF in the contest build (no money links in a judged entry). The tip UI code stays; the
# extras/tip-qr branch has the X Money QR, links and quote for after judging.
TIPS_ENABLED = False
TIP_URL = ""
TIP_LABEL = "tip"
TIP_TITLE = "Tip the creator (opens in a new tab)"
TIP_QR_CAPTION = ""
TIP_CARD_LABEL = "tip with card"
TIP_QUOTE = ""
TIP_CARD_URL = ""  # static try-it page card tip link (empty = hidden)
TIP_CARD_ENV = "MC_TIP_CARD_URL"  # future card tip link (e.g. a Stripe payment link); hidden while empty
FEEDBACK_SUBJECT = "MC (main-character-comfy) feedback"
# Agent control (MCP connector): one quiet line under the privacy badge + a Setup section. Deadpan, lowercase.
AGENT_HINT = ("agent control: on. plug in your own agent (grok bot, claude, chatgpt, cursor) and just tell it what "
              "to make. see setup → drive mc with your own agent.")
AGENT_SETUP = """**drive mc with your own agent.** everything in this window is also an MCP connector, so grok bot, claude \
desktop, chatgpt, cursor or any MCP client can run it for you: "make me strolling into the meeting late with my friend \
looking shocked." it shares this window's gpu queue, so nothing fights.

- this pc: `python -m main_character_comfy.agent` (stdio) or `.\\run.ps1 -Agent` (http://127.0.0.1:8765/mcp)
- from your phone: `.\\run.ps1 -Agent -Remote`, then reach it through tailscale or a cloudflare tunnel. token \
required. never open a port on your router.

copy-paste configs for each app are in the readme, under *drive mc with your own agent*."""
# Bring your own clip: paste a link, MC grabs the sharpest copy locally (yt-dlp) into library/clips.
CLIP_TIP = ("**for the best results, use the highest-quality version of your clip you can find: 1080p or 4K, steady "
            "camera, good lighting, clean audio.** you're responsible for having the rights to any clip you use.")
CLIP_LINK_LABEL = "or paste a link to your clip"
CLIP_LINK_PLACEHOLDER = "https://... (youtube, tiktok, instagram, x, vimeo, a direct .mp4)"
CLIP_LINK_BTN = "fetch the sharpest copy"
CLIP_LINK_NOTE = ("mc grabs the best video and the best audio it can find at that link, on this pc, into "
                  "`library/clips`. nothing ships with mc and nothing gets uploaded anywhere.")
CLIP_LINK_BUSY = "fetching the sharpest copy..."
CLIP_LINK_DONE = "got it: {quality}, {seconds:.1f} s, saved as `{path}`."
CLIP_LINK_LOW = " only {height}p though. a sharper copy swaps sharper."
CLIP_LINK_NEEDS_RIGHTS = "tick the rights box below first. it's your clip, or you have the ok."

REF_TIP = ("**reference quality matters most.** use the highest-quality photo you can: sharp, well lit, face and outfit "
           "clearly visible, ideally full body, no heavy filters, at least about 1024 px.")

# First-run tour: skippable, re-openable from the help link. Deadpan, lowercase.
HELP_LINK = "help: quick tour"
TOUR_TITLE = "quick tour"
TOUR_SKIP = "skip"
TOUR_BACK = "back"
TOUR_NEXT = "next"
TOUR_DONE = "got it"
TOUR_STEPS = [
    ("pick or paste a clip",
     "open **swap me in** (or **copy the vibe**). drop a clip, or paste a link and hit *fetch the sharpest copy*. "
     "mc pulls the best video and audio it can find, locally.\n\n"
     "for the best results, use the highest-quality version of your clip you can find: 1080p or 4K, steady camera, "
     "good lighting, clean audio. you're responsible for having the rights to any clip you use."),
    ("add your best reference photos",
     "**this step decides the likeness.** one photo per person. use the highest-quality reference photos you can: "
     "sharp, well lit, face and outfit clearly visible, ideally full body, no heavy filters, at least about 1024 px. "
     "face the camera, no sunglasses, nothing covering the face.\n\n"
     "mc checks every photo you drop (size, blur, face) and says so if it's weak. avatars keep their own art "
     "style. one good photo beats five blurry ones."),
    ("assign who plays who",
     "open *more than one person in the clip?* and hit *find people*. mc numbers everyone left to right. "
     "drop a photo on each person you want swapped and leave the rest empty. up to 8 people, depending on "
     "your tier."),
    ("choose a tier and render",
     "the tier up top comes from your gpu: lite, standard or max. max is sharpest, lite fits 8 GB. pick the "
     "render size under *more options*, tick the rights box, hit the big button. the eta shows above it."),
    ("review and download",
     "the preview lands on the right. don't like the take? reroll, it's free. happy? download, or *open folder*. "
     "every run is saved in `outputs/` with a readable name."),
    ("drive mc with your own agent",
     "rather just say it? plug grok bot, claude, chatgpt or cursor into mc and tell it what to make, even from "
     "your phone. setup is in the **setup** tab under *drive mc with your own agent*, and in the "
     "[readme](https://github.com/TonyOmega1/main-character-comfy#drive-mc-with-your-own-agent)."),
]
WAITING_FOR_GPU = "waiting for the gpu: {who} is rendering. you're next."
FEEDBACK_BODY = (
    "MC feedback\n"
    "-----------\n"
    "MC version: \n"
    "GPU tier (lite / standard / max): \n"
    "What happened: \n"
    "Suggested fix: \n"
    "\n"
    "(Please don't include personal info, selfies, voice clips, or anything you wouldn't post publicly.)\n"
)
FOOTER_LEFT = "original formats. no borrowed footage."
FOOTER_RIGHT = "you, but in the meme."
OUTPUT_LABEL = "→ output"

STEP_FACE = "face"
STEP_FORMAT = "format · {n} originals"
STEP_CAPTIONS = "caption"
PRIMARY_CTA = "make it"
SWAP_CTA = "swap me in"
VIBE_CTA = "copy the vibe"
SAVE_FORMAT = "save as format"
SAVE_FORMAT_HINT = ("Turn this format (or your custom one) with your captions into a reusable format. "
                    "It's saved to `formats/community/` and shows up in the grid right away; share it with a pull request.")
PACK_CTA = "make the pack"
TALK_CTA = "make the clip"
SURPRISE = "surprise me"
REROLL = "reroll (free)"
MORE_OPTIONS = "more: look, length, voice, meme era, effects, custom format"
CONSENT = ("This is me (my face, voice) or my own avatar, or one I have the rights to / their OK. I have the rights "
           "to any clip I upload, and I won't upload other people's faces without permission. SFW only.")
READY = "ready. selfie (or avatar), format, make it."
FACE_LABEL = "drop a selfie or your avatar"
SUBJECT_LABEL = "who's this?"
SUBJECT_CHOICES = [("auto", "auto"), ("me (photo)", "photo"), ("my avatar", "avatar")]
SUBJECT_NOTE = ("**my avatar** = cartoon, anime, 3D, VTuber, pixel art, a mascot, even non-human. MC keeps its art "
                "style instead of turning it into a photo. Transparent PNGs and any aspect ratio are fine.")
CLIP_AUDIO_LABEL = "sound"
CLIP_AUDIO_CHOICES = [("original audio (synced)", "original"), ("mute", "mute"), ("my voice (re-voice it)", "my_voice")]
PEOPLE_TITLE = "more than one person in the clip?"
PEOPLE_NOTE = ("Hit **find people**: MC numbers everyone on the first frame, left to right. Drop a selfie or avatar "
               "on each person you want swapped and leave the rest empty (they stay exactly as they are). One full "
               "pass per swapped person, so two people take about twice as long. Lite swaps 1, Standard 2, Max up to "
               "8, like Genjutsu's 8 references ('try anyway' lets Lite/Standard go to 4).")
PEOPLE_FIND = "find people"
CHECK_MODELS = "check my models"
STOP = "stop"
OPEN_FOLDER = "open folder"

# Top-level nav (every feature stays one click from the main page)
TAB_MEME = "Meme"
TAB_VIBE = "Copy the Vibe"
TAB_SWAP = "Swap Me In"
# Swap Me In = an open-weights "Openjutsu" (open rebuild of Higgsfield Genjutsu's motion transfer).
SWAP_INTRO = ("**Swap Me In is an open-weights Openjutsu.** Drop in a 1-30 s clip you have the rights to and 1-8 "
              "selfies or avatars. MC recasts the people in it and keeps the clip's motion, camera, timing and "
              "audio. Or re-voice it in your own cloned voice. Runs on your GPU with open weights: no credits, no "
              "per-second billing, no upload, unlimited rerolls.")
TAB_PACK = "Meme Pack"
TAB_MORE = "More"
TAB_SETUP = "Setup"

LIBRARY_NOTE = (
    "Every format here is an **original recipe**: a pose, a camera move and a caption layout. MC ships "
    "**no meme footage or images** and no real-person subjects. The scene gets generated fresh around *you*."
)
RIGHTS_NOTE = (
    "⚠️ **Only upload clips you have the rights to**: footage you filmed yourself, footage licensed for remixing, "
    "or a reference move you generated. No movies, TV, music videos or other creators' posts. "
    "**Don't upload other people's faces without their permission**: only put in yourself (or your own avatar, or "
    "one you have the rights to), or someone who said yes."
)
VIBE_NOTE = (
    "Only the **motion** of your reference clip is used (a pose skeleton and face crops); its pixels never reach the "
    "video model. The scene, the outfit and the lighting are generated new from your selfie + prompt. "
    "No reference clip handy? Make one: render an invented character doing the move in Meme Recreate "
    "(custom format), or film yourself."
)

# Shown next to the live stage message while a job runs (one picked per update).
LOADING_JOKES: dict[str, list[str]] = {
    "prepare": ["stretching before the bit", "finding your good angle", "lint-rolling the pixels"],
    "avatar": ["casting you as the main character", "applying main-character lighting",
               "convincing the GPU you're photogenic (easy)"],
    "voice": ["warming up your vocal cords (the GPU's, technically)", "rehearsing the line in the mirror"],
    "lipsync": ["teaching your mouth the words", "syncing lips, not vibes"],
    "motion": ["directing your Oscar-bait reaction", "rendering peak cinema", "the side-eye is loading… slowly"],
    "vibe": ["stealing the choreography (legally: just the stick figure)", "building your new set",
             "teaching you the moves, frame by frame", "main character walk loading"],
    "swap": ["learning your best moves", "politely asking the old person to leave the shot",
             "stunt double has been dismissed"],
    "finish": ["adding the IMPACT font energy", "cropping to 9:16 for the algorithm", "putting the bottom text on the bottom"],
}
GENERIC_JOKES = ["no credits were harmed in the making of this meme", "your GPU fans say hi",
                 "certified fresh (rendered seconds ago)", "this is the part where you hydrate"]

DONE = ["🎉 It's giving viral.", "🎉 Fresh meme, zero credits.", "🎉 Post it before someone else does."]
COMPLETE_HINT = "Not funny enough? Reroll. It's free, forever."


def loading_joke(stage: str, rng: random.Random | None = None) -> str:
    r = rng or random
    return r.choice(LOADING_JOKES.get(stage, []) + GENERIC_JOKES)


def done_line(rng: random.Random | None = None) -> str:
    return (rng or random).choice(DONE)


# Easter eggs: harmless, cosmetic only. Return a toast message or None.
def easter_egg(*, top: str = "", bottom: str = "", seed: int = 0, rerolls: int = 0) -> str | None:
    t, b = top.strip().lower(), bottom.strip().lower()
    if t == "top text" and b == "bottom text":
        return "A classic. Respect. 🫡"
    if seed in (69, 420, 69420, 1337):
        return "nice."
    if rerolls >= 10:
        return f"Reroll #{rerolls}. On HeyGen that'd be a lot of credits. Here it's free. Maybe touch grass after this one? 🌱"
    if rerolls == 5:
        return "5 rerolls deep. You'd be out of credits on most apps by now. 😌"
    return None

# Instant still -> animate
ANIMATE_CTA = "animate it →"
STILL_DONE = "✓ still rendered. already postable."
ANIMATE_HINT = "Want it moving? Hit **animate it →**: same scene, now with motion."
STRAIGHT_TO_VIDEO = "Skip the still and go straight to video"
CAPTION_REROLL = "new caption"

# Share bar (group-chat order)
SHARE_COPY = "Copy / send to chat"
SHARE_MP4 = "9:16 MP4"
SHARE_GIF = "GIF"
SHARE_PNG = "Still (PNG)"
SHARE_X = "Post on X"
SHARE_REDDIT = "Post on Reddit"
POST_TEXT_LABEL = "Post text (edit, copy, paste; includes #OpenSourceHiggsfield)"
WATERMARK_TOGGLE = "Small 'made with MC: Main Character · 0 credits' tag (untick to remove)"

# Swap Me In: lighting + props
RELIGHT_LABEL = "match scene lighting"
RELIGHT_CHOICES = ["auto (on for photos, off for avatars)", "on", "off"]
REMOVE_PROPS_LABEL = "remove props (don't redraw their backpack / gear on you)"

# Swap Me In / Copy the Vibe: render resolution
RESOLUTION_LABEL = "render size"
RESOLUTION_CHOICES = [("auto (tier default, 832x480 on 16 GB)", "auto"), ("480p", "480p"), ("720p (Max; slow on 16 GB)", "720p")]
RESOLUTION_NOTE = ("Every result is upscaled to 1080p at the end. 720p renders more real detail (faces, text on shirts) "
                   "but has ~2.3x the pixels: fine on 24 GB, experimental on 16 GB.")
# Swap Me In: VOID prop eraser (opt-in pre-pass)
ERASE_PROPS_LABEL = "erase props first (VOID)"
ERASE_PROPS_PLACEHOLDER = "e.g. backpack   (empty = off)"
ERASE_PROPS_NOTE = ("Names something the original character carries (backpack, sword, guitar). MC erases it from the clip "
                    "with Netflix's open VOID inpainting model before swapping you in, so it isn't redrawn on you. Adds "
                    "a render pass (~6 min per 5 s on a 16 GB card) and needs the VOID model pack (~30 GB, see Setup).")

FACE_DRIVE_LABEL = "face"
FACE_DRIVE_CHOICES = ("auto", "no human face", "copy the dancer's face")
FACE_DRIVE_NOTE = ("Auto: robots, animals and mascots don't get the dancer's face (their head stays non-human, colours can drift); people copy the dancer's expressions. "
             "Wan Animate otherwise draws the dancer's human face onto a non-human character.")
FACE_DRIVE_KEEP_LABEL = "never copy the dancer's face for (multi-character)"

# Status line under the tier picker when the running ComfyUI lacks MC's launch flags
# (same check as doctor / comfy_launch / the agent entry). Terse, lowercase, no dashes.
LAUNCH_FLAGS_WARN = "⚠️ comfyui is missing mc's launch flags: {missing}. restart it with `run_nvidia_gpu_mc.bat`."
LAUNCH_FLAG_RESERVE = "no vram kept free (reserve vram 1.5)"
LAUNCH_FLAG_UPLOAD = "uploads capped at {mb:g} mb (max upload size 500)"

# Connection status line (#mc-conn). Terse, lowercase, no dashes. The first words "Mock mode" / "Connected" stay as
# written because tests pin them; #mc-conn renders lowercase in CSS (code spans keep their case).
# mock: one line for both facts (it replaces the old separate "nothing leaves this PC" privacy chip)
CONN_MOCK = ("🧪 **Mock mode**: nothing leaves this pc. no gpu calls or renders, placeholder media. "
             "real renders: start with `--backend comfy`.")
CONN_DOWN = ("🔴 can't reach comfy-api-proxy at `{url}` ({err}). start comfyui, then `comfy-api-proxy run` "
             "(or run.ps1 / run.sh).")
CONN_UP = "🟢 Connected: comfy api v2 at `{url}` → comfyui `{upstream}`"

# More tab intros
MORE_INTRO = "the original mc talking-avatar engine. talk, duet, sing, batch."
REDUB_INTRO = ("**record your own line**: swap or add one spoken line in a finished clip. your take gets cleaned, "
               "fit to the slot and mixed under the original sound. *lip re-sync* redraws the mouth to match "
               "(new, pending a real test).")

# ---- app-feel mockup (feat/app-feel-mock): short leads up front, the long text one tap away ----
APP_TAGLINE = "maximum fun, zero learning curve."   # Tony's tagline (creative brief)
SETUP_ACC = "your setup · gpu tier, agent control"
ABOUT_ACC = "about mc · models, licenses, the rules"
RIGHTS_SHORT = "⚠️ only clips you have the rights to, and only faces that said yes."
RULES_CARD = "the rules + how to get the best clip"
SWAP_LEAD = "recast the people in a clip. the motion, camera, timing and audio stay."
SWAP_INFO = "how swap me in works"
VIBE_LEAD = "borrow the moves, not the pixels: you do the move in a brand new scene."
REF_CARD = "📸 reference quality matters most"
SUBJECT_CARD = "🎨 avatars keep their art style"
LIBRARY_CARD = "🧾 every format is an original recipe"
STEP_CLIP = "clip"
STEP_YOU = "you"
STEP_EXTRAS = "extras · optional"
STEP_GO = "go"
EMPTY_TITLE = "your meme lands here"
EMPTY_HINT = "free and local, so reroll as much as you like."
POST_TEXT_EMPTY = "your post text shows up here after a render."
