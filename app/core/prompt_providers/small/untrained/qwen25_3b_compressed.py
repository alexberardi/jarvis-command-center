"""
Qwen25_3B_Compressed - Prompt provider for Qwen 2.5 3B Instruct.

Inherits from Qwen25Compressed (the best-performing Qwen provider at 95.1%
on the 7B model). The 3B model uses the same ChatML format and <tool_call>
tags as the 7B variant but with less capacity.

Uses the compressed prompt strategy: compact tool listings with first-sentence
descriptions, full <tools> JSON block for param schemas, and standard Qwen
function-calling preamble.

Inherits parse_response, build_tools, and tool-calling format from
Qwen25MediumUntrained via Qwen25Compressed.
"""

from typing import Any, Dict

from app.core.prompt_providers.medium.untrained.qwen25_compressed import (
    Qwen25Compressed,
)


class Qwen25_3B_Compressed(Qwen25Compressed):
    """
    Prompt provider for Qwen 2.5 3B Instruct (untrained).

    Same ChatML format and <tool_call> tags as the 7B variant.
    Compressed prompt keeps context usage low for the smaller model.
    """

    @property
    def name(self) -> str:
        return "Qwen25_3B_Compressed"

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "provider_name": self.name,
            "model_family": "qwen",
            "size_tier": "small",
            "training_tier": "untrained",
            "use_tool_classifier": self.use_tool_classifier,
            "supports_native_tools": self.supports_native_tools,
        }
