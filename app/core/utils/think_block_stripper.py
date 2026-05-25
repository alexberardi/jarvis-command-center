"""Strip and detect think-block scaffolding around model chain-of-thought output.

Used by the streaming voice paths to (a) remove complete think spans before
they reach TTS, (b) pause sentence emission while inside an unclosed block,
and (c) skip think content during sentinel pre-checks.

Delimiters come from the active prompt provider's ``think_delimiters`` property,
so model families with non-standard markers (e.g. Llama 3.3 thinking variants
that emit ``[[[thinking start]]] ... [[[thinking end]]]``) work without touching
the streaming code.
"""

import re
from typing import Tuple


class ThinkBlockStripper:
    """Strip and detect think blocks bounded by a (start, end) delimiter pair."""

    def __init__(self, start: str, end: str):
        self.start = start
        self.end = end
        esc_start = re.escape(start)
        esc_end = re.escape(end)
        self._block_re = re.compile(
            esc_start + r".*?" + esc_end + r"\s*", re.DOTALL
        )
        self._unclosed_re = re.compile(esc_start + r".*", re.DOTALL)

    @classmethod
    def from_pair(cls, pair: Tuple[str, str]) -> "ThinkBlockStripper":
        return cls(pair[0], pair[1])

    def strip_complete_blocks(self, text: str) -> str:
        """Remove all balanced ``start...end`` spans from text."""
        return self._block_re.sub("", text)

    def has_open_block(self, text: str) -> bool:
        """True if a start marker has appeared with no matching end yet."""
        return self.start in text and self.end not in text

    def strip_all(self, text: str) -> str:
        """Strip complete blocks first, then any remaining unclosed open span."""
        return self._unclosed_re.sub("", self._block_re.sub("", text))


DEFAULT_THINK_DELIMITERS: Tuple[str, str] = ("<think>", "</think>")
