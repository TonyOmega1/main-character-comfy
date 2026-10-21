"""Derive the web/brand assets from the MC logo sprite.

Source: main_character_comfy/assets/brand/mc_logo_dark.png (and _light), the flat
silhouette sprite (figure + "MC." with a red period) made from the author's own
portrait. Outputs (all in assets/brand/):

* mc_figure_mask.png   white figure on transparent: a CSS mask, recolored with currentColor
* mc_figure_mask_160.png  the same, 160 px tall, for the app header
* mc_logo_mask.png     full sprite (figure + MC.) as a single-color mask
* favicon.ico (16/32/48), icon-{16,32,48,180,192,512}.png
                       head-and-shoulders crop (the full body is unreadable at 16/32 px)
* title_card_{dark,light}.png  1920x1080 demo title card

Run: python scripts/make_brand.py   (needs Pillow + numpy only)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

BRAND = Path(__file__).resolve().parent.parent / "main_character_comfy" / "assets" / "brand"
DARK_BG, DARK_FG, RED = (10, 10, 10), (239, 236, 230), (255, 42, 42)
LIGHT_BG = (239, 235, 227)


def split_rows(alpha: np.ndarray) -> int:
    """First empty row between the figure and the 'MC.' wordmark."""
    rows = alpha.max(1) > 10
    top = int(np.argmax(rows))
    run = 0
    for y in range(top, len(rows)):
        run = run + 1 if not rows[y] else 0
        if run >= 20:
            return y - run + 1
    raise ValueError("no gap between figure and wordmark")


def trim(im: Image.Image) -> Image.Image:
    return im.crop(im.getchannel("A").point(lambda v: 255 if v > 6 else 0).getbbox())


def mask_of(alpha: np.ndarray) -> Image.Image:
    white = np.full(alpha.shape + (3,), 255, np.uint8)
    return Image.fromarray(np.dstack([white, alpha]))


# Head-and-shoulders crop on the 566 px wide master sprite (x0, y0, side): hat, face, raised hand and
# shoulders. Checked by eye at 16/32/48 px: the full figure is a thin sliver at those sizes.
ICON_CROP = (70, 120, 440)


def icon(alpha_fig: np.ndarray, size: int, master_width: int = 566) -> Image.Image:
    """Head and shoulders, off-white on the noir black, the red period in the empty top-right corner."""
    k = alpha_fig.shape[1] / master_width
    x0, y0, side = (int(v * k) for v in ICON_CROP)
    crop = alpha_fig[y0:y0 + side, x0:x0 + side]
    if size <= 32:  # the faint wisps turn to mush at tiny sizes: keep only solid shapes
        crop = np.where(crop > 150, 255, 0).astype(np.uint8)
    big = 512
    fig = Image.fromarray(crop).resize((big, big), Image.LANCZOS)
    canvas = Image.new("RGBA", (big, big), DARK_BG + (255,))
    canvas.paste(Image.new("RGBA", (big, big), DARK_FG + (255,)), (0, 0), fig)
    dot, m = big // 6, big // 12
    ImageDraw.Draw(canvas).rectangle([big - dot - m, m, big - m, m + dot], fill=RED + (255,))
    return canvas.resize((size, size), Image.LANCZOS)


def title_card(sprite: Image.Image, bg: tuple[int, int, int]) -> Image.Image:
    card = Image.new("RGB", (1920, 1080), bg)
    s = trim(sprite)
    h = 860
    s = s.resize((int(s.width * h / s.height), h), Image.LANCZOS)
    card.paste(s, ((1920 - s.width) // 2, (1080 - h) // 2), s)
    return card


def main() -> None:
    dark = Image.open(BRAND / "mc_logo_dark.png").convert("RGBA")
    light = Image.open(BRAND / "mc_logo_light.png").convert("RGBA")
    alpha = np.array(dark.getchannel("A"))
    cut = split_rows(alpha)
    fig = alpha[:cut]
    fmask = trim(mask_of(fig))
    fmask.save(BRAND / "mc_figure_mask.png", optimize=True)
    # small copy for the header (inlined into the CSS as a data URI): 2x of the ~72 px tall header figure
    fmask.resize((round(fmask.width * 160 / fmask.height), 160), Image.LANCZOS).save(BRAND / "mc_figure_mask_160.png", optimize=True)
    trim(mask_of(alpha)).save(BRAND / "mc_logo_mask.png", optimize=True)
    for n in (16, 32, 48, 180, 192, 512):
        icon(fig, n).save(BRAND / f"icon-{n}.png", optimize=True)
    icon(fig, 48).save(BRAND / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48)],
                       append_images=[icon(fig, 16), icon(fig, 32)])
    title_card(dark, DARK_BG).save(BRAND / "title_card_dark.png", optimize=True)
    title_card(light, LIGHT_BG).save(BRAND / "title_card_light.png", optimize=True)
    print("wrote", sorted(p.name for p in BRAND.iterdir()))


if __name__ == "__main__":
    main()
