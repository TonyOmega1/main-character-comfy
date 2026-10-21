"""Shared content rules (SFW keyword blocklist) used by the pipeline and the format library."""

from __future__ import annotations

import re

BLOCKLIST = re.compile(r"\b(nude|naked|nsfw|porn|sex(?:ual)?|explicit)\b", re.I)


def is_sfw(*texts: str | None) -> bool:
    return not BLOCKLIST.search(" ".join(t or "" for t in texts))
