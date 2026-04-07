"""
Qwen3_8B_Compressed - Prompt provider for Qwen3-8B Instruct (Q4_K_M GGUF).

8B dense model using ChatML format and <tool_call> tags like Qwen 2.5.
Strips Qwen3's <think> blocks and injects /nothink in user messages to
disable chain-of-thought reasoning on simple voice commands.

Based on Qwen25_7B_Compressed (95%+ accuracy) — compressed tools block,
4 rules, CRITICAL DT_KEYS, force_tool_calls. Adds Qwen3-specific thinking
mode handling from Qwen3LargeUntrained.

Inherits parse_response, build_tools, get_response_format, and
build_training_completion from Qwen25MediumUntrained via Qwen25_7B_Compressed.
"""

import re
from typing import Any, Dict, Optional

from app.core.prompt_providers.medium.untrained.qwen25_7b_compressed import (
    Qwen25_7B_Compressed,
)

# Strip <think>...</think> blocks (Qwen3 thinking mode output)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


class Qwen3_8B_Compressed(Qwen25_7B_Compressed):
    """
    Prompt provider for Qwen3-8B Instruct (Q4_K_M, untrained).

    Same compressed prompt as Qwen 2.5 7B Compressed (4 rules, stripped
    param descriptions, Direct Answer section). Adds /nothink suffix
    and <think> block stripping for Qwen3's thinking mode.
    """

    @property
    def name(self) -> str:
        return "Qwen3_8B_Compressed"

    @property
    def user_message_suffix(self) -> str:
        """Append /nothink to user messages to disable Qwen3 thinking mode.

        Qwen3 ignores /no_think in the system prompt — the model was
        trained to recognize this token in user turns only.
        """
        return "/nothink"

    def parse_response(self, raw_content: str) -> Optional[str]:
        """Strip <think> blocks, then delegate to Qwen 2.5 parser."""
        cleaned: str = _THINK_BLOCK_RE.sub("", raw_content)
        return super().parse_response(cleaned)

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "provider_name": self.name,
            "model_family": "qwen",
            "size_tier": "medium",
            "training_tier": "untrained",
            "use_tool_classifier": self.use_tool_classifier,
            "supports_native_tools": self.supports_native_tools,
        }
