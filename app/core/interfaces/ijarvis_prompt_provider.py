"""
IJarvisPromptProvider - Interface for model-specific prompt construction.

Separates prompt construction from tool execution. Each provider knows how to
build the optimal system prompt for a specific model family + training tier
combination (e.g., Llama-small-untrained, Mistral-medium-trained).

Existing models (JarvisToolModel, JarvisAdapterModel, JarvisTrainedAdapterModel)
continue working via duck-typing on _build_system_prompt. New providers implement
this interface directly for cleaner separation.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class IJarvisPromptProvider(ABC):
    """
    Abstract interface for model-specific prompt construction.

    Implementations provide prompt-building logic decoupled from the
    tool execution loop. The factory discovers providers by scanning
    app/core/prompt_providers/ and matching by the `name` property.
    """

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

    def parse_response_quirks(self, raw_content: str) -> Optional[str]:
        """
        Hook for model-specific JSON cleanup before parsing.

        Some models emit trailing commas, comments, or markdown fences.
        Return cleaned content, or None to skip (use raw_content as-is).

        Args:
            raw_content: Raw LLM response text

        Returns:
            Cleaned text, or None to use raw_content unchanged.
        """
        return None

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
