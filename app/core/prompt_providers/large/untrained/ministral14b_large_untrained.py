"""Prompt provider for Ministral-3-14B-Instruct-2512 (Mistral, Dec 2025).

Reuses the Mistral v0.3 text tool-calling prompt+parse verbatim from
:class:`Mistral7bMediumUntrained` (``<tools>`` in the system prompt,
``<tool_call>{...}</tool_call>`` out, with a ``[TOOL_CALLS]`` fallback that
also matches this model's native tekken tool tokens). Only the selectable
name/capabilities differ — plus a chatml-token sanitizer (see ``sanitize_text``).

Serving note: the model is served on the proxy under ``--chat-template chatml``,
NOT its native tekken template, because the embedded tekken template rejects the
``system`` role (``Only user, assistant and tool roles are supported``) and CC
puts the tool schemas in a system message. chatml supports a system role, which
is exactly why the Mistral family is documented to run under chatml here.
"""

import re
from typing import Any, Dict

from app.core.prompt_providers.medium.untrained.mistral7b_medium_untrained import (
    Mistral7bMediumUntrained,
)

# The model's real EOS is </s>; <|im_end|> is not a token in its vocab, so under
# chatml llama.cpp can't stop on the chatml turn-end and the model emits it (and
# sometimes a stray <|im_start|>assistant turn) as LITERAL text. Cut the reply at
# the first chatml marker and strip any residual BOS/EOS so TTS never speaks them.
_CHATML_CUT_RE = re.compile(r"<\|im_(?:start|end)\|>")
_RESIDUAL_SPECIAL_RE = re.compile(r"</?s>")


class Ministral14bLargeUntrained(Mistral7bMediumUntrained):
    """Ministral-3-14B-2512 (arch=mistral3), served under chatml."""

    @property
    def name(self) -> str:
        return "Ministral14bLargeUntrained"

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "provider_name": self.name,
            "model_family": "ministral",
            "size_tier": "large",
            "training_tier": "untrained",
            "use_tool_classifier": self.use_tool_classifier,
            "supports_native_tools": self.supports_native_tools,
        }

    def sanitize_text(self, text: str) -> str:
        """Strip leaked chatml turn markers from user-facing text (see module
        docstring). Tool-call replies are unaffected — ``parse_response`` has
        already extracted ``<tool_call>`` before this runs on plain-text answers.
        """
        text = _CHATML_CUT_RE.split(text, maxsplit=1)[0]
        text = _RESIDUAL_SPECIAL_RE.sub("", text)
        return text.strip()
