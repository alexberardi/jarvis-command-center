"""
Model Service for Jarvis Voice Assistant.

This service provides a clean interface to the model system and can
gradually replace the existing prompt provider architecture.
"""

import logging
from typing import Any, Dict, List, Optional

from app.core.model_factory import ModelFactory
from app.core.interfaces.imodel_interface import IModelInterface
from app.core.interfaces.ijarvis_prompt_provider import IJarvisPromptProvider
from app.core.llm_proxy_client import LLMProxyClient
from app.core.conversation_handler import ConversationHandler
from app.core.prompt_provider_factory import PromptProviderFactory
from app.core.system_prompt_builder import build_tool_system_message
from app.request_models.voice_command_request import CommandDefinition

logger = logging.getLogger("uvicorn")


class ModelService:
    """
    Service for managing model interactions.

    This service provides a clean interface to the model system and handles:
    1. Model instantiation and management
    2. Conversation warmup and cleanup (via ConversationHandler)
    3. Inference requests
    4. Error handling and logging
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the model service.

        Tries PromptProviderFactory first for new-style providers, then
        falls back to ModelFactory for legacy IModelInterface classes.

        Args:
            model_name: Specific model to use. If None, uses environment variable.
        """
        # Try new prompt provider first
        self.prompt_provider: Optional[IJarvisPromptProvider] = None
        try:
            self.prompt_provider = PromptProviderFactory.create_provider(model_name)
        except Exception as e:
            logger.debug("PromptProviderFactory lookup failed: %s", e)

        # Always create legacy model (needed for perform_warmup, perform_inference, etc.)
        # If a new-style prompt provider was found, the legacy model may not exist
        # under the same name — fall back to default legacy model.
        if self.prompt_provider:
            try:
                self.model: IModelInterface = ModelFactory.create_model(model_name)
            except ValueError:
                logger.info(
                    "Legacy model '%s' not found; using default legacy model "
                    "alongside prompt provider '%s'",
                    self.prompt_provider.name,
                    self.prompt_provider.name,
                )
                self.model = ModelFactory.create_model("JarvisToolModel")
        else:
            self.model = ModelFactory.create_model(model_name)
        self.llm_client = LLMProxyClient()
        self._conversation_handler = ConversationHandler(
            model=self.model,
            llm_client=self.llm_client,
            prompt_provider=self.prompt_provider,
        )

        if self.prompt_provider:
            logger.info(
                "ModelService initialized with provider=%s, model=%s",
                self.prompt_provider.name,
                self.model.name,
            )
        else:
            logger.info("ModelService initialized with model=%s (legacy)", self.model.name)

    # =========================================================================
    # Legacy Methods (delegate to model interface)
    # =========================================================================

    async def warmup_conversation(
        self,
        node_context: Dict[str, Any],
        available_commands: List[CommandDefinition],
        conversation_id: str,
        timezone: Optional[str] = None,
    ) -> None:
        """
        Warm up a conversation with the model (legacy method).

        Args:
            node_context: Node information (room, user, device, etc.)
            available_commands: List of commands available to this node
            conversation_id: Unique conversation identifier
            timezone: User's timezone for date calculations
        """
        logger.info(f"🚀 Warming up conversation {conversation_id[:8]} with {self.model.name}")

        try:
            await self.model.perform_warmup(
                node_context=node_context,
                available_commands=available_commands,
                conversation_id=conversation_id,
                timezone=timezone,
            )
            logger.info(f"✅ Warmup completed for {conversation_id[:8]}")

        except Exception as e:
            logger.error(f"❌ Warmup failed for {conversation_id[:8]}: {e}")
            raise

    async def process_voice_command(
        self,
        voice_command: str,
        conversation_id: str,
        node_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process a voice command and return the result (legacy method).

        Args:
            voice_command: The user's voice command text
            conversation_id: Conversation ID (must match warmup call)
            node_context: Optional additional node context

        Returns:
            Standard Jarvis response format:
            {
                "s": bool,           # Success flag
                "n": str,            # Command name (or null)
                "p": Dict[str, Any], # Extracted parameters
                "e": Optional[Dict]  # Error object (if s=False)
            }
        """
        logger.info(f"🎯 Processing command with {self.model.name}: '{voice_command}'")

        try:
            result = await self.model.perform_inference(
                voice_command=voice_command,
                conversation_id=conversation_id,
                node_context=node_context,
            )

            logger.info("✅ Command processed successfully")
            return result

        except Exception as e:
            logger.error(f"❌ Command processing failed: {e}")
            return {
                "s": False,
                "n": None,
                "p": {},
                "e": {"type": "service_error", "message": str(e)},
            }

    async def cleanup_conversation(self, conversation_id: str) -> None:
        """
        Clean up a conversation.

        Args:
            conversation_id: Conversation to clean up
        """
        logger.info(f"🧹 Cleaning up conversation {conversation_id[:8]}")

        try:
            # Clean up via legacy model interface
            await self.model.cleanup_conversation(conversation_id)
            # Also clean up via conversation handler (for tool-based conversations)
            await self._conversation_handler.cleanup_conversation(conversation_id)
            logger.info(f"✅ Cleanup completed for {conversation_id[:8]}")

        except Exception as e:
            logger.error(f"❌ Cleanup failed for {conversation_id[:8]}: {e}")
            # Don't raise - cleanup failures shouldn't break the system

    # =========================================================================
    # Model Info & Health Methods
    # =========================================================================

    async def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the model service.

        Returns:
            Health status information
        """
        try:
            model_health = await self.model.health_check()

            return {
                "status": "healthy",
                "service": "ModelService",
                "model": model_health,
                "capabilities": self.model.get_capabilities(),
            }

        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return {
                "status": "unhealthy",
                "service": "ModelService",
                "error": str(e),
            }

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.

        Returns:
            Model information
        """
        return {
            "name": self.model.name,
            "class": self.model.__class__.__name__,
            "capabilities": self.model.get_capabilities(),
        }

    @staticmethod
    def get_available_models() -> List[str]:
        """
        Get list of available model names.

        Returns:
            List of available model names
        """
        return ModelFactory.get_available_models()

    @staticmethod
    def get_model_details(model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Model details or None if not found
        """
        return ModelFactory.get_model_info(model_name)

    # =========================================================================
    # Tool-Based Architecture Methods (delegate to ConversationHandler)
    # =========================================================================

    async def warmup_conversation_with_tools(
        self,
        node_context: Dict[str, Any],
        conversation_id: str,
        timezone: Optional[str] = None,
        client_tools: Optional[List[Dict[str, Any]]] = None,
        available_commands: Optional[List[CommandDefinition]] = None,
        adapter_settings: Optional[Dict[str, Any]] = None,
        skip_warmup_inference: bool = False,
    ) -> None:
        """
        Initialize a tool-based conversation with warmup.

        Sets up the conversation cache with system prompt, tools, and available commands.
        Optionally performs a warmup inference call to reduce first-response latency.

        Args:
            node_context: Context about the node (room, user, etc.)
            conversation_id: Unique conversation identifier
            timezone: User's timezone
            client_tools: Tools provided by the client
            available_commands: Command definitions for examples/antipatterns
            adapter_settings: Optional adapter configuration
            skip_warmup_inference: If True, skip the warmup LLM call
        """
        await self._conversation_handler.warmup_conversation_with_tools(
            conversation_id=conversation_id,
            node_context=node_context,
            timezone=timezone,
            client_tools=client_tools,
            available_commands=available_commands,
            adapter_settings=adapter_settings,
            skip_warmup_inference=skip_warmup_inference,
        )

    async def process_voice_command_with_tools(
        self,
        voice_command: str,
        conversation_id: str,
        speaker_user_id: int | None = None,
    ) -> Dict[str, Any]:
        """
        Process a voice command using tool-based architecture.

        Args:
            voice_command: The user's voice command text
            conversation_id: Conversation ID
            speaker_user_id: Actual speaker from STT (for mismatch detection
                            when warmup used a cached/predicted speaker ID)

        Returns:
            Response dict with:
            {
                "stop_reason": str,  # "complete", "tool_calls", "validation_required"
                "assistant_message": Optional[str],
                "tool_calls": Optional[List[Dict]],  # Client tool calls
                "validation_request": Optional[Dict]
            }
        """
        return await self._conversation_handler.process_voice_command_with_tools(
            voice_command=voice_command,
            conversation_id=conversation_id,
            speaker_user_id=speaker_user_id,
        )

    async def try_stream_voice_response(
        self,
        conversation_id: str,
        voice_command: str,
        tts_client,
        speaker_user_id: int | None = None,
    ):
        """Attempt router-gated streaming LLM → TTS.

        Returns an async audio generator if the command is streaming-eligible,
        or ``None`` to fall back to the blocking pipeline.
        """
        return await self._conversation_handler.stream_voice_response(
            conversation_id=conversation_id,
            voice_command=voice_command,
            tts_client=tts_client,
            speaker_user_id=speaker_user_id,
        )

    async def try_stream_voice_response_with_tools(
        self,
        conversation_id: str,
        voice_command: str,
        tts_client,
        speaker_user_id: int | None = None,
    ):
        """Attempt streaming LLM → TTS for commands needing server-side tools.

        Runs iter 1 (tool decision) and tool execution blocking, then streams
        iter 2 (final natural-language response) sentence-by-sentence to TTS.
        Falls back (returns None) for client tools, validation requests, and
        cases where iter 1 didn't actually produce a tool call.

        Gated by JARVIS_STREAM_TOOL_RESPONSES env flag.
        """
        return await self._conversation_handler.stream_voice_response_with_tools(
            conversation_id=conversation_id,
            voice_command=voice_command,
            tts_client=tts_client,
            speaker_user_id=speaker_user_id,
        )

    async def try_stream_continue_with_tool_results(
        self,
        conversation_id: str,
        tool_results: List[Dict[str, Any]],
        tts_client,
    ):
        """Attempt streaming LLM → TTS for the post-tool-results iteration.

        Returns an async generator of PCM bytes when applicable, or None
        to fall back to the blocking continue path (which returns JSON).
        """
        return await self._conversation_handler.stream_continue_with_tool_results(
            conversation_id=conversation_id,
            tool_results=tool_results,
            tts_client=tts_client,
        )

    async def continue_conversation_with_tool_results(
        self,
        conversation_id: str,
        tool_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Continue a conversation by providing tool execution results.

        Args:
            conversation_id: Conversation ID
            tool_results: List of tool results in format:
                [{"tool_call_id": str, "output": Any}, ...]

        Returns:
            Response dict (same format as process_voice_command_with_tools)
        """
        return await self._conversation_handler.continue_conversation_with_tool_results(
            conversation_id=conversation_id,
            tool_results=tool_results,
        )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _build_tool_system_message(
        self,
        node_context: Dict[str, Any],
        timezone: Optional[str],
        tools: List[Dict[str, Any]],
        available_commands: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Build system message for tool-based conversations."""
        return build_tool_system_message(node_context, timezone, tools, available_commands)
