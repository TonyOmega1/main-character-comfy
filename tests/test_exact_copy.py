"""Reference handling copies the upload exactly: identity lines and per-person looks in every swap prompt."""
from main_character_comfy.pipeline import (EXACT_COPY_AVATAR, EXACT_COPY_PHOTO, Request, swap_prompt,
                                            use_relight, vibe_scene_prompt)


def test_avatar_swap_prompt_has_exact_copy_line():
    p = swap_prompt(Request(mode="swap", subject="avatar"), "avatar")
    assert EXACT_COPY_AVATAR in p and "exact copy" in p and "art style" in p


def test_photo_swap_prompt_has_exact_copy_line():
    p = swap_prompt(Request(mode="swap", subject="photo"), "photo")
    assert EXACT_COPY_PHOTO in p and "same face" in p


def test_single_swap_uses_description():
    r = Request(mode="swap", people_desc=["a gray ragdoll bunny with long ears, striped sweater."])
    p = swap_prompt(r, "avatar")
    assert "The character is a gray ragdoll bunny with long ears, striped sweater." in p


def test_chained_passes_use_their_own_description():
    r = Request(mode="swap", people_desc=["man in a black cap", "", "bunny in a striped sweater"])
    assert "black cap" in swap_prompt(r, "photo", slot=0) and "bunny" not in swap_prompt(r, "photo", slot=0)
    assert "black cap" not in swap_prompt(r, "avatar", slot=1) and "The character is" not in swap_prompt(r, "avatar", slot=1)
    assert "striped sweater" in swap_prompt(r, "avatar", slot=2)
    assert "The person is" not in swap_prompt(r, "photo", slot=5)


def test_avatar_defaults_keep_the_drawing():
    assert use_relight(Request(), "avatar") is False  # no relighting of drawn references by default
    assert "exact copy of image 1" in vibe_scene_prompt("garage", True, avatar=True)
