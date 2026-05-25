"""
Llama33ThinkingMediumUntrained — Prompt provider for DavidAU's Llama 3.3 8B
fine-tuned on Claude Haiku 4.5 reasoning traces (1700x dataset).

Inherits Llama 3.1's ``<function=name>{args}</function>`` calling format —
the underlying base is Meta's Llama 3.x Instruct, so the same Llama-3 chat
template and function-tag training carries through. Overrides:

- ``think_delimiters``: model emits ``[[[thinking start]]] ... [[[thinking end]]]``
  rather than Qwen3's ``<think>...</think>``. The streaming code reads this
  from the provider to keep reasoning prose out of TTS.
- ``parse_response``: strips thinking content before extracting function
  tags so any ``<function=...>`` the model hallucinates inside the reasoning
  block doesn't poison the parser.
- ``sanitize_text``: also strips thinking markers from user-facing strings
  in addition to the parent's function-tag scrubbers.

⚠️ This model was NOT fine-tuned on function-calling data — the reasoning
LoRA sits on top of an unofficial Llama 3.3 8B base, and tool-routing
accuracy is empirical. Benchmark against ``Llama31MediumUntrained`` before
relying on it for production traffic.
"""

import json
import logging
import re
from typing import Any, Dict, Optional

from app.core.prompt_providers.medium.untrained.llama31_medium_untrained import (
    Llama31MediumUntrained,
)
from app.core.utils.think_block_stripper import ThinkBlockStripper

logger = logging.getLogger("uvicorn")

_THINK_DELIMITERS: tuple[str, str] = (
    "[[[thinking start]]]",
    "[[[thinking end]]]",
)
_THINK_STRIPPER = ThinkBlockStripper.from_pair(_THINK_DELIMITERS)

# The Claude-Haiku LoRA biases output toward <tool_call>{...}</tool_call>
# (the IJarvisPromptProvider default training format) rather than Llama 3.1's
# <function=name>{args}</function>. Recognize both so model preference doesn't
# break tool routing. JSON inside is the Jarvis call shape:
#   {"name": "...", "arguments": {...}, "failure_message": "..."}
_TOOL_CALL_TAG_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
# Strip pattern for sanitize_text — keep any <tool_call> scaffolding out of TTS.
_STRIP_TOOL_CALL_RE = re.compile(r"<tool_call>.*?</tool_call>\s*", re.DOTALL)
# Truncation case: model emitted opening tag but no closer (max_tokens cut).
_STRIP_TOOL_CALL_UNCLOSED_RE = re.compile(r"<tool_call>.*", re.DOTALL)


class Llama33ThinkingMediumUntrained(Llama31MediumUntrained):
    """Llama 3.3 8B with Claude Haiku reasoning LoRA. Same tool format as Llama 3.1."""

    @property
    def name(self) -> str:
        return "Llama33ThinkingMediumUntrained"

    @property
    def think_delimiters(self) -> tuple[str, str]:
        return _THINK_DELIMITERS

    def parse_response(self, raw_content: str) -> Optional[str]:
        # Strip reasoning prose first so any tool-call scaffolding hallucinated
        # inside the thinking block doesn't get extracted as a real call.
        without_think = _THINK_STRIPPER.strip_all(raw_content).strip()

        # Hermes/Qwen-style <tool_call>{...}</tool_call> — this model prefers
        # it over Llama's <function=...>. Try it first; fall through to the
        # parent for any other format the model might emit.
        tool_call_matches = _TOOL_CALL_TAG_RE.findall(without_think)
        if tool_call_matches:
            parsed_calls: list[Dict[str, Any]] = []
            for match in tool_call_matches:
                try:
                    call_obj = json.loads(match.strip())
                except json.JSONDecodeError:
                    logger.warning(
                        "Llama33Thinking: failed to parse <tool_call> JSON: %s",
                        match[:120],
                    )
                    continue
                parsed_calls.append(call_obj)
            if parsed_calls:
                return json.dumps({
                    "message": "",
                    "tool_calls": parsed_calls,
                    "error": None,
                })

        return super().parse_response(without_think)

    def sanitize_text(self, text: str) -> str:
        # Strip thinking markers AND any leaked <tool_call> scaffolding,
        # then delegate to parent for <function=...> scrubs.
        cleaned = _THINK_STRIPPER.strip_all(text)
        cleaned = _STRIP_TOOL_CALL_RE.sub("", cleaned)
        cleaned = _STRIP_TOOL_CALL_UNCLOSED_RE.sub("", cleaned)
        return super().sanitize_text(cleaned)

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "provider_name": self.name,
            "model_family": "llama",
            "size_tier": "medium",
            "training_tier": "untrained",
            "use_tool_classifier": self.use_tool_classifier,
            "supports_native_tools": self.supports_native_tools,
            "thinking_mode": "auto",
        }
