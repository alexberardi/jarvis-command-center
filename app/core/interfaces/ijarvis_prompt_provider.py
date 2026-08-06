"""
IJarvisPromptProvider - Interface for model-specific prompt construction.

Separates prompt construction from tool execution. Each provider knows how to
build the optimal system prompt for a specific model family + training tier
combination (e.g., Llama-small-untrained, Mistral-medium-trained).

Existing models (JarvisToolModel, JarvisAdapterModel, JarvisTrainedAdapterModel)
continue working via duck-typing on _build_system_prompt. New providers implement
this interface directly for cleaner separation.
"""

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class IJarvisPromptProvider(ABC):
    """
    Abstract interface for model-specific prompt construction.

    Implementations provide prompt-building logic decoupled from the
    tool execution loop. The factory discovers providers by scanning
    app/core/prompt_providers/ and matching by the `name` property.
    """

    # Set by ConversationHandler from the model.include_thinking setting.
    # Providers that support thinking mode use this to decide /think vs /no_think.
    include_thinking: bool = False

    # ── Shared context builders ──────────────────────────────────────
    # Room and user context are automatically extracted from node_context
    # and injected into the prompt. Providers call these helpers instead
    # of manually pulling fields out of node_context.

    def build_context_header(self, node_context: Dict[str, Any] | None) -> str:
        """Build the SPEAKER-AGNOSTIC identity + room/style header.

        This is the LEADING (cached) segment of the system prompt — it must be
        identical across all household members so the llama.cpp prefix cache
        hits regardless of who is speaking. Speaker name + memories are NOT
        included here; they are injected per-turn via :meth:`build_speaker_context`
        as a trailing system message after the cached prefix.

        When the household set a speaking-voice persona, a ``<personality>`` block
        is appended after the identity/context lines. The persona is node-level
        (one voice per household), so it rides the cached prefix and stays
        byte-identical across speakers. It's loaded into ``node_context`` once at
        warmup (see ``ConversationHandler._get_household_persona``); absent → the
        output is byte-identical to the pre-persona header.

        Returns a string like::

            You are Jarvis, a function calling voice assistant.
            Context: room=kitchen, style=brief

            <personality>
            The household chose this as your speaking style — ...
            </personality>
        """
        from app.core.prompt_providers.shared.core_rules import (
            build_identity_header,
            build_personality_block,
        )

        ctx = node_context or {}
        room: str = ctx.get("room", "unknown")
        voice_mode: str = ctx.get("voice_mode", "brief")
        header = build_identity_header(room, voice_mode)

        personality = build_personality_block(ctx.get("household_persona", ""))
        if personality:
            header = f"{header}\n\n{personality}"

        # Ambient situational context (time/weather/calendar) is DELIBERATELY not
        # injected here. This header is the byte-stable cached prefix sent at warmup
        # (messages[0]); the llama.cpp prefix cache is a single household-shared
        # sequence, and ambient content is situational (it differs between two
        # separate conversations — clock bucket, weather refresh, today's calendar).
        # Putting it at the top of the prefix invalidated the cache on every new
        # conversation → cold ~1.8s warmup re-prefill (prod TTFS 3s→5s regression,
        # 2026-08-06). It is now injected per-turn as a TRAILING system message after
        # the cached prefix — exactly like the speaker block — see
        # ConversationHandler._process_voice_command.
        return header

    def build_speaker_context(self, node_context: Dict[str, Any]) -> str:
        """Build the per-turn SPEAKER-SPECIFIC block (name + memories) from
        node_context, returned for injection as a trailing system message
        after the cached prefix. Returns "" when there is nothing
        speaker-specific to say (unknown speaker / no memories).
        """
        from app.core.prompt_providers.shared.core_rules import build_speaker_block

        ctx = node_context or {}
        user: str = ctx.get("speaker_name") or ctx.get("user", "default")
        user_memories: str = ctx.get("user_memories", "")
        return build_speaker_block(user, user_memories)

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique identifier for factory matching.

        Convention: <Family><Size><Tier>, e.g. "LlamaSmallUntrained".
        Matched case-insensitively by PromptProviderFactory.
        """
        ...

    @abstractmethod
    def build_system_prompt(
        self,
        node_context: Dict[str, Any],
        timezone: Optional[str],
        tools: List[Dict[str, Any]],
        available_commands: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Build the complete system prompt for an LLM call.

        Args:
            node_context: Runtime context (room, user, voice_mode, agents, etc.)
            timezone: User's IANA timezone (e.g. "America/New_York")
            tools: Tool definitions in OpenAI function-calling format
            available_commands: Command flag dicts with command_name,
                allow_direct_answer, keywords, etc.

        Returns:
            Full system prompt string ready for the messages array.
        """
        ...

    @property
    def use_tool_classifier(self) -> bool:
        """
        Whether the fastText tool classifier should provide routing hints.

        Untrained providers typically return True (needs help routing).
        Trained providers return False (adapter handles routing).
        """
        return True

    def get_response_format(self) -> Optional[Dict[str, Any]]:
        """
        Override the default JSON response format schema.

        Return None to use the shared default from system_prompt_builder.
        Override for model families that need a different JSON schema
        (e.g., stricter schemas for smaller models).
        """
        return None

    def parse_response(self, raw_content: str) -> Optional[str]:
        """
        Transform raw LLM output into Jarvis JSON format string.

        Each provider can override this to handle its model's native output
        format (e.g., XML tool_call tags, markdown fences, trailing commas)
        and transform it into the Jarvis JSON shape that ToolCallParser expects.

        Return the transformed string, or None to pass raw content to
        ToolCallParser as-is.

        Args:
            raw_content: Raw LLM response text

        Returns:
            Transformed Jarvis JSON string, or None to use raw_content unchanged.
        """
        return None

    def sanitize_text(self, text: str) -> str:
        """Clean text emitted to the user (wake responses, direct answers, TTS).

        Separate from parse_response (which only runs on tool-calling paths):
        this runs on ANY user-facing string the provider produces, so think
        blocks, XML artifacts, or control tokens don't leak into TTS.

        Default is identity. Models with thinking modes (Qwen3) or other
        scaffolding override this to strip before the text hits the TTS
        pipeline.

        Args:
            text: Raw text destined for the user.

        Returns:
            Cleaned text safe to speak.
        """
        return text

    @property
    def user_message_suffix(self) -> str:
        """Optional suffix appended to every user message.

        Override in providers that need control tokens in the user turn
        (e.g., Qwen3 ``/nothink`` to disable chain-of-thought).
        """
        return ""

    @property
    def think_delimiters(self) -> tuple[str, str]:
        """Open/close markers wrapping the model's chain-of-thought output.

        Streaming code uses these to strip complete think spans before TTS,
        pause sentence emission while inside an unclosed span, and skip
        think content during sentinel pre-checks.

        Default ``("<think>", "</think>")`` matches Qwen3-style thinking tokens.
        Override in providers whose thinking format differs (e.g. Llama 3.3
        thinking variants emit ``[[[thinking start]]] ... [[[thinking end]]]``).
        """
        return ("<think>", "</think>")

    @property
    def lazy_tool_loading(self) -> bool:
        """Whether this provider uses lazy tool loading.

        When True: system prompt contains only a compact capability list
        and a ``get_tools`` meta-tool. Full tool schemas are returned on
        demand when the model calls ``get_tools``. This dramatically
        reduces prompt tokens for context-answerable queries.

        Default: False (all tool schemas in the system prompt).
        """
        return False

    @property
    def supports_native_tools(self) -> bool:
        """
        Whether to pass tools natively to the LLM proxy.

        When True: tools passed via API 'tools' parameter, tool_calls read
        from structured response (finish_reason="tool_calls").
        When False: tools embedded in system prompt, parsed from text output.

        Default: False (backward compatible).
        """
        return False

    def build_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Build OpenAI-format tool definitions for native tool calling.

        Override to customize tool schemas for a specific model family
        (e.g., strip descriptions, adjust parameter schemas).

        Only called when supports_native_tools is True.

        Args:
            tools: Raw tool definitions from tool registry.

        Returns:
            OpenAI-format tool definitions ready for the API.
        """
        return tools

    def build_training_completion(self, tool_call: Dict[str, Any]) -> str:
        """Format a tool call as this model expects to output it during inference.

        Default: raw Jarvis JSON (backward compatible).
        Override in providers that use native formats (XML tags, function tags, etc.)
        """
        return " " + json.dumps({
            "message": "",
            "tool_calls": [tool_call],
            "error": None,
        })

    def build_training_system_prompt(self) -> str:
        """Return the system message used for training examples.

        This is used with the tokenizer's chat template to build properly
        formatted training data (system/user/assistant messages with correct
        special tokens). The training script wraps this as the system message,
        the voice command as the user message, and build_training_completion()
        output as the assistant message.

        Default: minimal tool router system prompt.
        Override if the model needs specific formatting cues.
        """
        return (
            "You are a function calling AI model. "
            "For each function call return a json object with function name and arguments "
            "within <tool_call></tool_call> XML tags as follows:\n"
            "<tool_call>\n"
            '{"name": "<function-name>", "arguments": {"<arg-name>": "<arg-value>"}, "failure_message": "<brief spoken response if this call fails>"}\n'
            "</tool_call>"
        )

    def build_training_prompt(self, voice_command: str) -> str:
        """Build the training prompt for a voice command.

        DEPRECATED: Use build_training_system_prompt() + voice_command with
        the tokenizer's chat template instead. Kept for backward compatibility
        with training scripts that don't support chat templates.
        """
        system = self.build_training_system_prompt()
        return (
            f"{system}\n"
            f"User: {voice_command}\n"
            "Assistant:"
        )

    def get_capabilities(self) -> Dict[str, Any]:
        """
        Metadata about this provider for health checks and admin UI.

        Returns:
            Dict with at minimum:
            - "provider_name": str
            - "model_family": str
            - "size_tier": str  (small / medium / large)
            - "training_tier": str  (untrained / trained)
            - "use_tool_classifier": bool
        """
        return {
            "provider_name": self.name,
            "model_family": "unknown",
            "size_tier": "unknown",
            "training_tier": "unknown",
            "use_tool_classifier": self.use_tool_classifier,
        }
