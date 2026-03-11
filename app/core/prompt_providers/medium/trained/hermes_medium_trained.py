"""
HermesMediumTrained - Prompt provider for date-key-trained Hermes 3 Llama 3.1 8B.

Inherits from HermesMediumUntrained. The only difference is a simplified date rule
in the system prompt — the model has date/time key extraction baked in via LoRA
merge, so it handles resolved_datetimes reliably without verbose formatting rules.

Model: Hermes-3-Llama-3.1-8B-jarvis-Q4_K_M.gguf
Training: 2128 date/time extraction examples, 3 epochs, final loss 0.097
"""

import logging
import os
from typing import Any, Dict, List, Optional

from app.core.prompt_providers.medium.untrained.hermes_medium_untrained import (
    HermesMediumUntrained,
)
from app.core.prompt_providers.shared.context_builders import (
    build_agent_context_summary,
    build_direct_answer_section,
)
from app.core.prompt_providers.shared.tool_formatters import format_tools_for_prompt

logger = logging.getLogger("uvicorn")


class HermesMediumTrained(HermesMediumUntrained):
    """
    Prompt provider for date-key-trained Hermes 3 Llama 3.1 8B.

    Identical to HermesMediumUntrained except:
    - Simplified date parameter rule (model already knows the format)
    - training_tier: "trained" in capabilities
    - use_tool_classifier still True (date training doesn't improve routing)
    """

    @property
    def name(self) -> str:
        return "HermesMediumTrained"

    def build_system_prompt(
        self,
        node_context: Dict[str, Any],
        timezone: Optional[str],
        tools: List[Dict[str, Any]],
        available_commands: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Build system prompt with simplified date rule.

        Same structure as HermesMediumUntrained, but the verbose date formatting
        instruction is replaced with a brief reinforcement since the model has
        date key extraction baked into its weights.
        """
        available_commands = available_commands or []
        node_context = node_context or {}

        room: str = node_context.get("room", "unknown")
        user: str = node_context.get("user", "default")
        voice_mode: str = node_context.get("voice_mode", "brief")

        # Shared sections
        direct_answer_section: str = build_direct_answer_section(available_commands)
        agent_context_section: str = build_agent_context_summary(node_context)

        # Build <tools> XML block for Hermes's fine-tuned format
        tools_xml: str = HermesMediumUntrained._build_tools_xml(tools)

        system_prompt: str = f"""You are Jarvis, a function calling voice assistant.
Context: room={room}, user={user}, style={voice_mode}

You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions.

{tools_xml}

Rules:
- Call ONE tool at a time to fulfill requests.
- Pick the tool that best matches intent; use get_command_utterance_examples if unsure.
- Extract parameters from the user's words; only request validation if required params are truly missing/ambiguous.
- For date parameters, use natural words (today, tomorrow, this_weekend, etc.).
- Always populate required tool parameters from the user's request.
- For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
<tool_call>
{{"name": "<function-name>", "arguments": {{"<arg-name>": "<arg-value>"}}, "failure_message": "<brief spoken response if this call fails>"}}
</tool_call>
{direct_answer_section}
{agent_context_section}
For final answers with no tool needed, respond with a brief spoken reply.
"""

        logger.info(
            "Built HermesMediumTrained system prompt: %d chars, %d tools",
            len(system_prompt),
            len(tools),
        )

        if os.getenv("LOG_FULL_SYSTEM_PROMPT", "false").lower() in {"1", "true", "yes"}:
            logger.info("System prompt (full):\n%s", system_prompt)

        return system_prompt

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            **super().get_capabilities(),
            "provider_name": self.name,
            "training_tier": "trained",
        }
