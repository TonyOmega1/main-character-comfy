"""MC supports Python 3.10+. Python 3.12 relaxed f-strings (PEP 701); two things it now allows are SyntaxErrors on
3.10 / 3.11 and slipped in twice: a backslash inside an f-string's {expression}, and reusing the enclosing quote
inside it. Checked with the tokenizer (3.12+ emits FSTRING_* tokens), so no 3.10 interpreter is needed."""

import sys
import tokenize
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parent.parent / "main_character_comfy"


def _problems(path: Path) -> list[str]:
    out: list[str] = []
    stack: list[str] = []  # quote of each open f-string
    with path.open("rb") as fh:
        for tok in tokenize.tokenize(fh.readline):
            name = tokenize.tok_name[tok.type]
            if name == "FSTRING_START":
                q = tok.string.lstrip("rRfFbBuU")
                if stack and q[0] in "".join(stack):
                    out.append(f"{path.name}:{tok.start[0]} nested f-string reuses an enclosing quote")
                stack.append(q)
            elif name == "FSTRING_END":
                stack.pop()
            elif stack and name != "FSTRING_MIDDLE":
                if "\\" in tok.string:
                    out.append(f"{path.name}:{tok.start[0]} backslash inside an f-string expression")
                if name == "STRING" and tok.string.lstrip("rRbBuU")[:1] in "".join(stack):
                    out.append(f"{path.name}:{tok.start[0]} string inside an f-string reuses its quote")
    return out


@pytest.mark.skipif(sys.version_info < (3, 12), reason="needs the 3.12+ f-string tokens")
def test_no_py312_only_fstrings():
    bad = [p for f in sorted(PKG.rglob("*.py")) for p in _problems(f)]
    assert not bad, "\n".join(bad)
