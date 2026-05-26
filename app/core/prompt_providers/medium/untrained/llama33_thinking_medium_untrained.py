"""
Llama33ThinkingMediumUntrained — Prompt provider for DavidAU's Llama 3.3 8B
fine-tuned on Claude Haiku 4.5 reasoning traces (1700x dataset).

Inherits Llama 3.1's ``<function=name>{args}</function>`` calling format —
the underlying base is Meta's Llama 3.x Instruct, so the same Llama-3 chat
template and function-tag training carries through. Overrides:

- ``parse_response``: strips thinking content before extracting function
  tags so any ``<function=...>`` the model hallucinates inside the reasoning
  block doesn't poison the parser. Also catches the model's tendency to emit
  ``{"message": "<think>...</think>actual answer"}`` — the reasoning ends up
  trapped inside the JSON ``message`` field and must be stripped there before
  the JSON parses.
- ``sanitize_text``: strips thinking markers from user-facing strings, and
  recognises broken-JSON scaffolding left when the model gets cut off
  mid-think (``finish_reason=length``) so TTS doesn't speak ``{"message": "``.
- ``user_message_suffix``: appends ``/no_think`` to user turns to keep voice
  latency in budget. The reasoning LoRA respects this control token; without
  it the model regularly chews through the 256-token cap before producing
  any usable output.

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

# The reasoning LoRA emits standard <think>...</think> markers (same shape as
# Qwen3), not the [[[thinking start]]] form claimed in earlier model notes —
# this was verified against live llm-proxy traffic.
_THINK_DELIMITERS: tuple[str, str] = ("<think>", "</think>")
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

# Broken-JSON scaffolding: model wrapped its reasoning inside the JSON
# `message` field and ran out of tokens before closing the string. Once the
# unclosed <think> tail is stripped, what's left is the opening of the
# wrapper — typically `{"message": "` with no closer. Speaking that aloud
# produces "open brace quote message colon" so detect it and discard.
_BROKEN_JSON_WRAPPER_RE = re.compile(
    r'^\s*\{\s*"message"\s*:\s*"?\s*$', re.DOTALL
)

# Empty Jarvis-JSON envelope returned when the model produced nothing usable —
# caller will treat this as a no-op (202 JSON / silent abort) instead of
# letting raw scaffolding reach TTS.
_EMPTY_RESPONSE_JSON: str = json.dumps({
    "message": "",
    "tool_calls": [],
    "error": None,
})


class Llama33ThinkingMediumUntrained(Llama31MediumUntrained):
    """Llama 3.3 8B with Claude Haiku reasoning LoRA. Same tool format as Llama 3.1."""

    @property
    def name(self) -> str:
        return "Llama33ThinkingMediumUntrained"

    @property
    def user_message_suffix(self) -> str:
        # /no_think keeps voice turns in the 256-token budget. The reasoning
        # LoRA otherwise spends the whole budget inside <think>...</think>
        # and the response gets truncated mid-thought.
        return "/no_think"

    def parse_response(self, raw_content: str) -> Optional[str]:
        # Strip reasoning prose first so any tool-call scaffolding hallucinated
        # inside the thinking block doesn't get extracted as a real call.
        without_think = _THINK_STRIPPER.strip_all(raw_content).strip()

        # Model frequently does `{"message": "<think>...</think>"}` and
        # hits max_tokens mid-think. After stripping the unclosed think
        # tail, only the JSON wrapper opening remains — return an empty
        # envelope so downstream sees a clean no-op instead of speaking
        # the broken JSON.
        if _BROKEN_JSON_WRAPPER_RE.match(without_think):
            logger.info(
                "Llama33Thinking: truncated mid-think detected — "
                "returning empty response (raw=%r)", raw_content[:120],
            )
            return _EMPTY_RESPONSE_JSON

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
        # If all that's left is the JSON wrapper opening (model got cut off
        # mid-think and parse_response didn't intercept — e.g. when this is
        # called on tool_call_parser's plain-text fallback), drop it before
        # TTS speaks "open brace quote message colon".
        if _BROKEN_JSON_WRAPPER_RE.match(cleaned.strip()):
            return ""
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
