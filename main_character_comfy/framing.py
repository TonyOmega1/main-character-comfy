"""Shot-adaptive reference crops: the reference handed to the swap should match the shot's framing.

A full-body photo scaled into a close-up leaves a ~40 px face for the model to copy (likeness lost), and a tight
face crop in a wide shot loses the outfit. The rule from the Oct 18-19 tests (it fixed every close-up failure):
head crop for CU/ECU, upper body for medium shots, full body for wide shots, legs for feet-only shots.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence, Iterable

# face height / frame height in the *shot* -> framing
ECU_AT = 0.45
CU_AT = 0.22
MEDIUM_AT = 0.10

FRAMINGS = ("ecu", "cu", "medium", "wide", "feet", "close_noface")

# Engine routing (Oct 19 HQ evidence): SCAIL-2 failed 6/6 extreme close-up and feet/legs-only shots, which all
# ended up on Animate anyway, while normal close-ups passed SCAIL-2 4/5. So only these framings skip SCAIL-2;
# "cu" deliberately still tries SCAIL-2 first.
ROUTE_TO_ANIMATE = frozenset({"ecu", "feet", "close_noface"})
# No face found: the person counts as a close no-face shot (feet/legs, back of a torso) only when the body is cut
# by the top edge AND fills most of the frame height. Anything else stays "wide" (YuNet misses many cartoon faces,
# and a missed face in a wide shot must not cost a SCAIL-2 take).
NOFACE_TOP_EDGE = 0.02  # box top within 2% of the frame top
NOFACE_MIN_HEIGHT = 0.6  # box height >= 60% of the frame height


def shot_framing(face_frac: float | None, feet_only: bool = False) -> str:
    """Classify a shot from its tracked face height fraction (None: no face visible -> wide, or feet)."""
    if feet_only:
        return "feet"
    if face_frac is None:
        return "wide"
    if face_frac >= ECU_AT:
        return "ecu"
    if face_frac >= CU_AT:
        return "cu"
    if face_frac >= MEDIUM_AT:
        return "medium"
    return "wide"


def classify_person(rgb: Any, box: Sequence[float], find_face: Callable[[Any], dict | None] | None = None,
                    pad: float = 0.1) -> dict[str, Any]:
    """Framing of one person in a shot frame: {"framing", "face_frac", "face"}.

    ``rgb`` is the frame (H x W x 3 array), ``box`` the person's (x0, y0, x1, y1) from the people scan. The face is
    looked for inside the person's (padded) box only, so another person's face can't decide this one's framing.
    ``face_frac`` = face height / frame height. ``find_face`` defaults to refcheck.find_face (YuNet)."""
    import numpy as np

    if find_face is None:
        from .refcheck import find_face as yunet  # lazy: cv2 is optional

        find_face = yunet
    arr = np.asarray(rgb)
    fh, fw = arr.shape[:2]
    x0, y0, x1, y1 = (float(v) for v in box[:4])
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    cx0, cy0 = max(0, int(x0 - px)), max(0, int(y0 - py))
    cx1, cy1 = min(fw, int(round(x1 + px))), min(fh, int(round(y1 + py)))
    face = find_face(arr[cy0:cy1, cx0:cx1]) if cx1 > cx0 and cy1 > cy0 else None
    if face is not None:
        face = dict(face, x=float(face["x"]) + cx0, y=float(face["y"]) + cy0)
        frac = float(face["h"]) / fh
        return {"framing": shot_framing(frac), "face_frac": round(frac, 3), "face": face}
    if y0 <= NOFACE_TOP_EDGE * fh and (y1 - y0) >= NOFACE_MIN_HEIGHT * fh:
        return {"framing": "close_noface", "face_frac": None, "face": None}
    return {"framing": "wide", "face_frac": None, "face": None}


def route_to_animate(framings: Sequence[str]) -> tuple[bool, str]:
    """(True, why) when any swapped person's framing is one SCAIL-2 is known to fail (see ROUTE_TO_ANIMATE)."""
    bad = [(i + 1, f) for i, f in enumerate(framings) if f in ROUTE_TO_ANIMATE]
    if not bad:
        return False, ""
    label = {"ecu": "an extreme close-up", "feet": "a feet-only shot", "close_noface": "a close shot with no face"}
    who = ", ".join(f"person {i} is {label[f]}" for i, f in bad)
    return True, f"{who}: SCAIL-2 fails these, so straight to Animate"


# Framing QC of a SCAIL-2 take against the source frame (the failure seen on Oct 19: the swapped person drawn
# zoomed or shifted). Boxes are compared as fractions of their own frames (the take renders at the SCAIL-2 size).
QC_HEIGHT_RATIO = (0.8, 1.25)
QC_MAX_SHIFT = 0.15  # centre shift, as a fraction of the frame


def framing_qc(src: Sequence[Sequence[float]], src_size: Sequence[int], take: Sequence[Sequence[float] | None],
               take_size: Sequence[int], soft_missing: Iterable[int] = ()) -> dict[str, Any]:
    """Compare each source person box with its matched take box. ``take[i]`` None = that person wasn't found.
    Returns {"pass": bool, "why": str, "people": [...], "unverified": [...]}; it fails on a missing person, a
    height ratio outside QC_HEIGHT_RATIO, or a centre shift over QC_MAX_SHIFT. A missing person whose index is in
    ``soft_missing`` (a non-human reference the people detector can't be trusted to find) is listed in
    "unverified" instead of failing; the other people still decide the verdict."""
    soft = set(soft_missing)
    unverified: list[int] = []
    sw, sh = (float(v) for v in src_size[:2])
    tw, th = (float(v) for v in take_size[:2])
    rows: list[dict[str, Any]] = []
    fails: list[str] = []
    for i, (a, b) in enumerate(zip(src, take)):
        if b is None:
            rows.append({"person": i + 1, "found": False})
            if i in soft:
                unverified.append(i + 1)
            else:
                fails.append(f"person {i + 1} missing from the take")
            continue
        ha, hb = (a[3] - a[1]) / sh, (b[3] - b[1]) / th
        ratio = hb / ha if ha > 0 else 0.0
        dx = (b[0] + b[2]) / 2 / tw - (a[0] + a[2]) / 2 / sw
        dy = (b[1] + b[3]) / 2 / th - (a[1] + a[3]) / 2 / sh
        shift = (dx * dx + dy * dy) ** 0.5
        rows.append({"person": i + 1, "found": True, "height_ratio": round(ratio, 3), "shift": round(shift, 3)})
        if not QC_HEIGHT_RATIO[0] <= ratio <= QC_HEIGHT_RATIO[1]:
            fails.append(f"person {i + 1} drawn at {ratio:.2f}x the filmed size")
        elif shift > QC_MAX_SHIFT:
            fails.append(f"person {i + 1} moved {shift:.0%} of the frame")
    why = "; ".join(fails) or "framing matches the clip"
    if unverified:
        why += "; " + ", ".join(f"person {n}" for n in unverified) + " (non-human) not found by the people detector: check by eye"
    return {"pass": not fails, "why": why, "people": rows, "unverified": unverified}


def reference_crop(img_w: int, img_h: int, face: Sequence[float], framing: str,
                   aspect: float | None = None) -> tuple[int, int, int, int]:
    """Crop box (x0, y0, x1, y1) of a reference photo for a shot framing.

    ``face`` is the reference's face box (x, y, w, h). ``aspect`` (w/h) optionally matches the shot's tracked box.
    The crop never leaves the image; wide shots get the whole photo.
    """
    x, y, w, h = (float(v) for v in face[:4])
    cx = x + w / 2
    if framing in ("wide", "close_noface"):  # no face to anchor on: the whole reference
        return 0, 0, img_w, img_h
    if framing == "ecu":
        top, bottom, half_w = y - 0.45 * h, y + 1.35 * h, 0.95 * w
    elif framing == "cu":
        top, bottom, half_w = y - 0.7 * h, y + 2.3 * h, 1.45 * w
    elif framing == "medium":
        top, bottom, half_w = y - 0.8 * h, y + 4.6 * h, 2.4 * w
    elif framing == "feet":
        top, bottom, half_w = y + 4.4 * h, float(img_h), img_w / 2
        cx = img_w / 2
    else:
        raise ValueError(f"unknown framing {framing!r}; one of {FRAMINGS}")
    if aspect and framing != "feet":
        half_w = max(half_w, aspect * (bottom - top) / 2)
    x0, x1 = cx - half_w, cx + half_w
    y0, y1 = top, bottom
    # shift inside the image instead of shrinking where possible
    if x0 < 0:
        x1, x0 = x1 - x0, 0
    if x1 > img_w:
        x0, x1 = max(0.0, x0 - (x1 - img_w)), img_w
    if y0 < 0:
        y1, y0 = y1 - y0, 0
    if y1 > img_h:
        y0, y1 = max(0.0, y0 - (y1 - img_h)), img_h
    return int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))


CLOSEUP_FRAMINGS = ("ecu", "cu")
CLOSEUP_FACE_MAX = 0.35  # a reference whose face is under 35 % of its height is "full body / wide" for a close-up


def closeup_reference(src: Any, dst: Any, shot: str, aspect: float | None = None,
                      find_face: Callable[[Any], dict | None] | None = None) -> dict[str, Any] | None:
    """Face close-up from a full-body reference, for close-up shots (Request.ref_closeup = "auto").

    A full-body photo used for a close-up gives the model a tiny face to copy, so the likeness drops. When ``shot`` is
    a close-up framing and the reference's face is under CLOSEUP_FACE_MAX of its height, the reference is cropped to
    that framing (reference_crop) and saved to ``dst``. Returns {"path", "box", "face_frac"} or None (left as is)."""
    from pathlib import Path

    import numpy as np
    from PIL import Image

    if shot not in CLOSEUP_FRAMINGS:
        return None
    if find_face is None:
        from .refcheck import find_face as yunet

        find_face = yunet
    img = Image.open(src)
    img.load()
    rgb = np.asarray(img.convert("RGB"))
    face = find_face(rgb)
    if not face:
        return None
    h_img, w_img = rgb.shape[:2]
    frac = float(face["h"]) / h_img
    if frac >= CLOSEUP_FACE_MAX:
        return None
    box = reference_crop(w_img, h_img, (face["x"], face["y"], face["w"], face["h"]), shot, aspect)
    crop = img.crop(box)  # keeps alpha (cut-out avatars stay cut out)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    crop.save(dst)
    return {"path": str(dst), "box": list(box), "face_frac": round(frac, 3)}
