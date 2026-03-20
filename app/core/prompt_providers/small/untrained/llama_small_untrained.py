"""
LlamaSmallUntrained - Prompt provider for small untrained Llama models.

Optimized for Llama 3.x 1B-8B without LoRA adapters. Uses the same
prompt strategy as JarvisToolModel: full tool descriptions, all examples,
explicit JSON formatting rules, and fastText classifier routing hints.

This is the reference implementation showing how to build a prompt provider
from the IJarvisPromptProvider interface using shared utilities.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from app.core.interfaces.ijarvis_prompt_provider import IJarvisPromptProvider
from app.core.prompt_providers.shared.context_builders import (
    build_direct_answer_section,
)
from app.core.prompt_providers.shared.tool_formatters import format_tools_for_prompt

logger = logging.getLogger("uvicorn")


class LlamaSmallUntrained(IJarvisPromptProvider):
    """
    Prompt provider for small untrained Llama models (1B-8B).

    Strategy:
    - Rich system prompt with full tool descriptions and ALL examples
    - Explicit JSON formatting rules (small models need more guidance)
    - fastText classifier enabled for routing hints
    - No agent context (HA devices) to keep prompt within context window
    """

    @property
    def name(self) -> str:
        return "LlamaSmallUntrained"

    @property
    def use_tool_classifier(self) -> bool:
        return True

    def build_system_prompt(
        self,
        node_context: Dict[str, Any],
        timezone: Optional[str],
        tools: List[Dict[str, Any]],
        available_commands: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Build a rich system prompt for small untrained Llama models.

        Includes full tool descriptions, all examples, and explicit JSON
        rules. Equivalent to JarvisToolModel._build_system_prompt.
        """
        available_commands = available_commands or []
        node_context = node_context or {}

        room = node_context.get("room", "unknown")
        user = node_context.get("user", "default")
        voice_mode = node_context.get("voice_mode", "brief")

        direct_answer_section = build_direct_answer_section(available_commands)
        tools_section = format_tools_for_prompt(tools, available_commands)

        system_prompt = f"""You are Jarvis, a voice assistant that uses tools.
Context: room={room}, user={user}, style={voice_mode}

Rules:
- Use tools to fulfill requests. Call ONE tool at a time.
- Pick the tool that best matches intent/keywords/examples; otherwise use get_command_utterance_examples.
- Extract parameters from the user's words; request validation only if required params are missing/ambiguous.
{direct_answer_section}

Response format: JSON ONLY (no extra text).
- Tool call: {{"message":"brief natural ack of what you're doing (REQUIRED, never empty)","tool_calls":[{{"name":"<tool>","arguments":{{...}}}}],"error":null}}
- Final: {{"message":"<concise spoken reply>","tool_calls":[],"error":null}}

Tools:
{tools_section}
"""

        logger.info(
            "Built LlamaSmallUntrained system prompt: %d chars, %d tools",
            len(system_prompt),
            len(tools),
        )

        if os.getenv("LOG_FULL_SYSTEM_PROMPT", "false").lower() in {"1", "true", "yes"}:
            logger.info("System prompt (full):\n%s", system_prompt)

        return system_prompt

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "provider_name": self.name,
            "model_family": "llama",
            "size_tier": "small",
            "training_tier": "untrained",
            "use_tool_classifier": self.use_tool_classifier,
        }
