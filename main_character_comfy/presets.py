"""Style presets, motion prompts and script starters.

Every preset is an original description (no named artists, studios, franchises or
characters) so outputs stay clear of third-party IP.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Style:
    key: str
    label: str
    emoji: str
    # Used when restyling an uploaded photo (FLUX.2 klein reference edit).
    edit_prompt: str
    # Used when the user only describes a character (FLUX.2 klein text-to-image).
    t2i_prompt: str
    # Appended to the InfiniteTalk motion prompt so the animation keeps the look.
    motion_suffix: str


_FRAMING = (
    "upper body medium close-up, centered, looking straight into the camera, mouth gently closed, "
    "relaxed natural expression, vertical 9:16 composition, soft key light, clean uncluttered background"
)

STYLES: dict[str, Style] = {
    s.key: s
    for s in [
        Style(
            "3d",
            "3D Animated",
            "🧸",
            "Turn the person in image 1 into a stylized 3D animated movie character: big expressive eyes, "
            "smooth subsurface-scattered skin, soft rounded features, keep their hairstyle, hair color, skin tone, "
            f"outfit and recognizable face shape. {_FRAMING}, cozy colorful room background with warm bokeh.",
            "Stylized 3D animated movie character render of {desc}, big expressive eyes, smooth stylized skin, "
            f"soft rounded features, {_FRAMING}, cozy colorful room background with warm bokeh, high detail.",
            "stylized 3D animation, smooth expressive facial animation",
        ),
        Style(
            "anime",
            "Anime",
            "🌸",
            "Redraw the person in image 1 as a hand-drawn anime character: clean line art, cel shading, "
            "large luminous eyes, keep their hairstyle, hair color, skin tone and outfit. "
            f"{_FRAMING}, pastel sky and city rooftop background.",
            "Hand-drawn anime illustration of {desc}, clean line art, cel shading, large luminous eyes, "
            f"{_FRAMING}, pastel sky and city rooftop background.",
            "2D anime style, cel shaded animation",
        ),
        Style(
            "comic",
            "Comic Book",
            "💥",
            "Redraw the person in image 1 as a bold comic book hero portrait: thick ink outlines, halftone dot "
            "shading, saturated flat colors, keep their hairstyle, skin tone, outfit and face shape. "
            f"{_FRAMING}, dynamic pop-art background.",
            "Bold comic book portrait of {desc}, thick ink outlines, halftone dot shading, saturated flat colors, "
            f"{_FRAMING}, dynamic pop-art background.",
            "comic book style, bold ink lines, halftone shading",
        ),
        Style(
            "real",
            "Realistic",
            "📸",
            "Make a polished, realistic portrait photo of the person in image 1, keep their exact face, hairstyle, "
            f"skin tone and outfit, natural skin texture. {_FRAMING}, modern creator studio with soft ring light.",
            "Realistic portrait photo of {desc}, natural skin texture, 50mm lens, "
            f"{_FRAMING}, modern creator studio with soft ring light.",
            "photorealistic, natural lighting",
        ),
        Style(
            "clay",
            "Claymation",
            "🧱",
            "Turn the person in image 1 into a handmade clay stop-motion puppet: visible fingerprints in the clay, "
            f"matte textures, keep their hairstyle, colors and outfit. {_FRAMING}, miniature handmade set.",
            "Handmade clay stop-motion puppet of {desc}, visible fingerprints in the clay, matte textures, "
            f"{_FRAMING}, miniature handmade set.",
            "claymation stop-motion style",
        ),
        Style(
            "none",
            "Real me (no restyle)",
            "🙂",
            "",
            "",
            "natural lighting",
        ),
    ]
}

DEFAULT_STYLE = "3d"


def motion_prompt(style_key: str, subject: str = "The character") -> str:
    style = STYLES.get(style_key, STYLES[DEFAULT_STYLE])
    return (
        f"{subject} talks directly to the camera with natural expressive lip movements, subtle head nods "
        f"and small hand gestures, blinking naturally, static camera, {style.motion_suffix}"
    )


SCRIPT_STARTERS: dict[str, str] = {
    "👋 Channel intro": "Hey, welcome back! If you're new here, I make short videos every week, so hit follow and let's get into it.",
    "🔥 Hot take": "Okay, hot take: breakfast for dinner is the best meal of the week, and I will not be taking questions.",
    "📦 Product hype": "I've been testing this for a whole week, and honestly? It changed my morning routine. Here's why.",
    "🎵 New music drop": "My new song drops this Friday! I worked on it for months, and I can't wait for you to hear it.",
    "😂 Story time": "So you won't believe what happened to me today. I walked into the wrong meeting and just stayed.",
    "💡 Quick tip": "Here's a quick tip: record your voice in a closet full of clothes. It sounds like a real studio.",
}
