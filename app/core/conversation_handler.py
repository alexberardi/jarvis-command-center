"""
Conversation Handler for Jarvis Voice Assistant.

This module orchestrates conversation flow including warmup, command processing,
and continuation with tool results. Extracted from ModelService for better
separation of concerns.
"""

import json
import logging
import os
import re
import uuid
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Set

from app.core.conversation_cache import conversation_cache
from app.core.tts_text import clean_for_tts
from app.services.settings_service import get_settings_service
from app.core.interfaces.ijarvis_prompt_provider import IJarvisPromptProvider
from app.core.interfaces.imodel_interface import IModelInterface
from app.core.llm_proxy_client import LLMProxyClient
from app.core.tool_execution_engine import ToolExecutionEngine
from app.core.tool_registry import tool_registry
from app.core.tool_routing import (
    get_shared_tool_classifier,
    route_tool,
    filter_tools_for_utterance,
)
from app.core.utils.latency_logger import latency_logger
from app.core.voice_command_helpers import (
    get_tool_name,
    build_available_command_flags,
    apply_tool_routing,
    prune_tools_by_router_decision,
)
from app.core.warmup_service import warmup_service

logger = logging.getLogger("uvicorn")


def get_server_tool_names(tools: Optional[List[Dict[str, Any]]]) -> Set[str]:
    """
    Extract server tool names from a tools list.

    Server tools are those registered in the tool_registry.

    Args:
        tools: List of tool definitions

    Returns:
        Set of server tool names
    """
    if not tools:
        return set()

    return {
        name
        for name in (get_tool_name(t) for t in tools)
        if name and tool_registry.has_tool(name)
    }


class ConversationHandler:
    """
    Orchestrates conversation flow for voice commands.

    Handles:
    - Conversation warmup (initializing state)
    - Voice command processing
    - Continuation with tool results
    - Conversation cleanup

    This class coordinates between the conversation cache, tool execution engine,
    and model interface to process voice commands through the tool-based architecture.
    """

    def __init__(
        self,
        model: IModelInterface,
        llm_client: LLMProxyClient,
        prompt_provider: Optional[IJarvisPromptProvider] = None,
    ):
        """
        Initialize the conversation handler.

        Args:
            model: The model interface for building prompts and configuration
            llm_client: The LLM proxy client for API calls
            prompt_provider: Optional new-style prompt provider. When set,
                build_system_prompt is used instead of model._build_system_prompt.
        """
        self.model = model
        self.llm_client = llm_client
        self.prompt_provider = prompt_provider

    async def cleanup_conversation(self, conversation_id: str) -> None:
        """
        Clean up a conversation by removing it from the cache.

        Args:
            conversation_id: The conversation ID to clean up
        """
        logger.info(f"🧹 Cleaning up conversation {conversation_id[:8]}...")
        conversation_cache.remove(conversation_id)

    async def warmup_conversation_with_tools(
        self,
        conversation_id: str,
        node_context: Dict[str, Any],
        timezone: Optional[str],
        client_tools: Optional[List[Dict[str, Any]]],
        available_commands: Optional[List[Any]],
        adapter_settings: Optional[Dict[str, Any]] = None,
        skip_warmup_inference: bool = False,
    ) -> None:
        """
        Initialize a tool-based conversation with warmup.

        Sets up the conversation cache with system prompt, tools, and available commands.
        Optionally performs a warmup inference call to reduce first-response latency.

        Args:
            conversation_id: Unique conversation identifier
            node_context: Context about the node (room, user, etc.)
            timezone: User's timezone
            client_tools: Tools provided by the client
            available_commands: Command definitions for examples/antipatterns
            adapter_settings: Optional adapter configuration
            skip_warmup_inference: If True, skip the warmup LLM call
        """
        logger.info(f"🔥 Warming up tool-based conversation {conversation_id[:8]}...")

        # Check memory settings (used for both tool gating and memory loading)
        _memory_enabled, _recall_enabled = self._get_memory_settings(node_context)

        # Get server tools from registry
        # Text-based providers should only see client/command tools in the prompt.
        # Server tools (get_command_utterance_examples, request_validation) confuse
        # text-based models — they call them when unsure, triggering runaway loops.
        use_text_path: bool = (
            self.prompt_provider is not None
            and not self.prompt_provider.supports_native_tools
        )
        if use_text_path:
            server_tools = []
            # Whitelist safe server tools for the text-based path.
            # Only tools that are simple CRUD / one-shot operations belong
            # here.  Avoid tools that trigger follow-up LLM calls or loops
            # (e.g. get_command_utterance_examples, request_validation).
            _safe_tool_names: list[str] = ["answer_question", "deep_research", "quick_search"]
            # Only include tools that require speaker identification when
            # a speaker has actually been identified for this conversation.
            _has_speaker = bool(
                node_context and node_context.get("speaker_user_id")
            )

            # Gate memory tools on settings
            if _has_speaker and _memory_enabled:
                _safe_tool_names.extend(["remember", "forget"])
                if _recall_enabled:
                    _safe_tool_names.append("recall")

            for tool_name in _safe_tool_names:
                tool = tool_registry.get_tool(tool_name)
                if tool:
                    server_tools.append(tool.to_openai_format())
        else:
            server_tools = tool_registry.get_tools_for_model(self.model.name)
        server_tool_names = {get_tool_name(t) for t in server_tools if get_tool_name(t)}

        # Merge server and client tools
        all_tools = warmup_service.merge_tools(server_tools, client_tools)

        # Get available command names for filtering
        available_command_names = warmup_service.get_available_command_names(
            all_tools, available_commands
        )

        # Process client tools (filter antipatterns, etc.)
        if client_tools:
            warmup_service.process_client_tools(client_tools, available_command_names)

        # Build examples map from client tools
        examples_map = warmup_service.build_examples_map(client_tools or [])

        # Merge available commands into examples
        command_examples = warmup_service.merge_available_commands(
            examples_map, available_commands, available_command_names
        )

        # Build available command flags for system prompt
        available_command_flags = build_available_command_flags(
            [cmd for cmd in command_examples if cmd.get("command_name")]
        )

        # Load user memories for identified speaker (only if memory feature is enabled)
        speaker_user_id = node_context.get("speaker_user_id") if node_context else None
        household_id = node_context.get("household_id") if node_context else None
        if speaker_user_id and household_id and _memory_enabled:
            try:
                from app.db import get_session_local
                from app.services.memory_service import MemoryService

                # Get pinned_max_chars from settings
                pinned_max_chars = 500
                try:
                    from app.services.settings_service import get_settings_service
                    settings = get_settings_service()
                    val = settings.get("memory.pinned_max_chars")
                    if val is not None:
                        pinned_max_chars = int(val)
                except Exception:
                    pass

                SessionLocal = get_session_local()
                db = SessionLocal()
                try:
                    svc = MemoryService(db)
                    memories_text = svc.get_memories_for_prompt(
                        speaker_user_id, household_id, max_chars=pinned_max_chars
                    )
                    if memories_text:
                        node_context["user_memories"] = memories_text
                        logger.info(f"📝 Loaded {len(memories_text)} chars of user memories for speaker {speaker_user_id}")
                finally:
                    db.close()
            except Exception as e:
                logger.warning(f"⚠️ Failed to load user memories: {e}")

        # Fetch date keys for prompt providers that need them.
        # We INTENTIONALLY trim to a small high-frequency subset before
        # injecting into the prompt — the full list is ~60 keys (~250 prompt
        # tokens). The backend resolver still accepts the full list, so any
        # less-common key the model emits ("next_friday" etc.) still
        # resolves; the trim just stops sending all 60 as a hint every call.
        try:
            date_keys = await self.llm_client.get_date_keys()
            if date_keys:
                _COMMON_DATE_KEYS = {
                    "today", "tomorrow", "yesterday",
                    "tonight", "this_weekend", "last_weekend", "next_weekend",
                    "morning", "afternoon", "evening", "night",
                    "this_week", "last_week", "next_week",
                    "this_month", "last_month",
                }
                trimmed = [k for k in date_keys if k in _COMMON_DATE_KEYS]
                node_context["date_keys"] = trimmed if trimmed else date_keys
        except Exception as e:
            logger.warning("Failed to fetch date keys for prompt: %s", e)

        # Build system prompt (new provider first, then legacy model)
        system_prompt = self._get_system_prompt(
            node_context, timezone, all_tools, available_command_flags
        )

        messages = [{"role": "system", "content": system_prompt}]

        # Store in cache
        conversation_cache.set(
            conversation_id=conversation_id,
            messages=messages,
            available_commands=command_examples,
            timezone=timezone,
            tools=all_tools,
            node_context=node_context,
        )

        # If prompt provider mandates tool calls, store the flag so the
        # tool execution engine can enforce it (retry on finish_reason=stop).
        if self.prompt_provider is not None and getattr(self.prompt_provider, 'force_tool_calls', False):
            conversation_cache.set_force_tool_calls(conversation_id, True)

        # Optional warmup inference (reduces first-response latency)
        # For llama.cpp: warmup processes the system prompt and caches the
        # KV state. The actual voice command then only needs to process the
        # new user message, dramatically reducing inference time.
        use_native_tools: bool = (
            self.prompt_provider is not None
            and self.prompt_provider.supports_native_tools
        )
        if not skip_warmup_inference:
            try:
                if use_native_tools:
                    # Native tools path: full warmup with tool definitions.
                    # IMPORTANT: use the same tool transformation as the inference
                    # path (strip_jarvis_extensions + prompt_provider.build_tools)
                    # so the token sequence matches and llama.cpp's prefix cache
                    # is reused on the first real inference call.
                    from app.core.tool_builder import ToolBuilder
                    warmup_tools = self.prompt_provider.build_tools(
                        ToolBuilder.strip_jarvis_extensions(all_tools)
                    )
                    await self.llm_client.chat_completion(
                        conversation_id=conversation_id,
                        messages=messages,
                        tools=warmup_tools,
                        adapter_settings=adapter_settings,
                    )
                else:
                    # Text-based path: send system prompt for KV caching
                    # with max_tokens=1 to avoid generating useless output.
                    # llama.cpp caches the prompt prefix; the voice command
                    # inference then only processes the new user message.
                    allows_caching: bool = await self.llm_client.allows_warmup_caching()
                    if allows_caching:
                        await self.llm_client.chat_completion(
                            conversation_id=conversation_id,
                            messages=messages,
                            adapter_settings=adapter_settings,
                            max_tokens=1,
                        )
                    else:
                        logger.info(f"⏭️ Engine doesn't benefit from warmup ({conversation_id[:8]})")
                logger.info(f"✅ Warmup inference complete for {conversation_id[:8]}...")
            except Exception as e:
                logger.warning(f"⚠️ Warmup inference failed (non-fatal): {e}")

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
        import time

        _t0 = time.time()
        timing = latency_logger.get_request(conversation_id)
        logger.info(f"🎯 Processing tool-based command: '{voice_command}'")

        # Get conversation state from cache
        with timing.measure("cache_lookups") if timing else nullcontext():
            messages = conversation_cache.get_messages(conversation_id)
            tools = conversation_cache.get_tools(conversation_id)
            available_commands = conversation_cache.get_available_commands(conversation_id) or []

        # --- Speaker mismatch detection (parallel warmup) ---
        # Warmup may have used a cached/predicted speaker_user_id. If STT
        # identified a different speaker, reload memories and rebuild the
        # system prompt so the LLM sees the correct user context.
        if speaker_user_id is not None and messages:
            node_context = conversation_cache.get_node_context(conversation_id)
            warmup_speaker = node_context.get("speaker_user_id") if node_context else None
            if warmup_speaker != speaker_user_id:
                logger.info(
                    "Speaker mismatch detected, reloading memories: warmup=%s actual=%s",
                    warmup_speaker, speaker_user_id,
                )
                await self._reload_memories_for_speaker(
                    conversation_id, speaker_user_id, node_context, messages,
                )

        logger.debug(f"⏱️  [T+{(time.time()-_t0)*1000:.0f}ms] Cache lookups done")

        if not messages:
            raise ValueError(f"Conversation {conversation_id} not found or expired")

        # Get server tool names for filtering
        server_tool_names = get_server_tool_names(tools)

        # Apply keyword-based tool filtering (if enabled)
        tools = self._apply_tool_filtering(
            voice_command, tools, available_commands
        )
        logger.debug(f"⏱️  [T+{(time.time()-_t0)*1000:.0f}ms] Tool filtering done")

        # Apply tool routing classifier
        _router_t0 = time.time()
        router_decision = self._apply_tool_routing_with_cache(
            voice_command, tools or [], conversation_id
        )
        logger.debug(f"⏱️  [T+{(time.time()-_t0)*1000:.0f}ms] Tool routing done (router took {(time.time()-_router_t0)*1000:.0f}ms)")

        # Apply high-confidence tool pruning — also rebuilds the system
        # prompt with only the pruned tools, dramatically reducing token count.
        tools = self._apply_high_confidence_pruning(
            tools, router_decision, server_tool_names,
            messages=messages, conversation_id=conversation_id,
            available_commands=available_commands,
        )

        # Add router hint if decision was used
        if router_decision and router_decision.get("used"):
            hint_tool = router_decision.get("tool_name")
            messages.append({
                "role": "system",
                "content": f"Router hint: likely tool is '{hint_tool}'. Use it if it matches intent; otherwise choose the best tool."
            })

        # Pre-execute quick_search for search-intent utterances so the LLM
        # gets web results in context without needing to call the tool itself.
        # Smaller models (e.g. Qwen 14B) don't reliably call tools on their own.
        search_results = await self._maybe_quick_search(voice_command)
        if search_results:
            messages.append({
                "role": "system",
                "content": search_results,
            })

        # Add user message (with optional provider suffix, e.g. /nothink for Qwen3)
        suffix: str = (
            self.prompt_provider.user_message_suffix
            if self.prompt_provider else ""
        )
        user_content: str = f"{voice_command}\n{suffix}" if suffix else voice_command
        messages.append({"role": "user", "content": user_content})
        logger.debug(f"⏱️  [T+{(time.time()-_t0)*1000:.0f}ms] Starting tool execution loop")

        # Execute tool loop
        # Text-based providers should resolve in 1-2 iterations; cap at 3
        # to avoid runaway loops that cause 30s+ timeouts.
        use_native: bool = (
            self.prompt_provider is not None
            and self.prompt_provider.supports_native_tools
        )
        max_iters: int = 10 if use_native else 3
        with timing.measure("tool_execution_loop") if timing else nullcontext():
            engine = ToolExecutionEngine(self.llm_client, prompt_provider=self.prompt_provider)
            result = await engine.execute(
                conversation_id=conversation_id,
                messages=messages,
                tools=tools or [],
                user_utterance=voice_command,
                max_iterations=max_iters,
            )

        logger.debug(f"⏱️  [T+{(time.time()-_t0)*1000:.0f}ms] Tool execution loop completed")

        # Text-based models: server tools completed but the model can't
        # process role="tool" messages.  Format results with a clean call.
        if result.get("stop_reason") == "server_tool_complete":
            server_results = result.get("server_tool_results", [])
            tool_results_for_format = [
                {
                    "tool_call_id": r.get("tool_call_id", ""),
                    "output": r.get("content", ""),
                }
                for r in server_results
            ]
            result = await self._format_tool_result_text_mode(
                conversation_id, messages, tool_results_for_format
            )

        # Update cache and return
        conversation_cache.update_messages(conversation_id, messages)
        if result.get("stop_reason") == "error":
            logger.error("❌ Tool loop returned stop_reason=error: %s", result)
        return result

    # --- Streaming-eligible server tools (bypass tool execution loop) ---
    _STREAMING_ELIGIBLE_TOOLS: set[str] = {"answer_question", "quick_search"}
    _STREAMING_MIN_CONFIDENCE: float = 0.8

    async def stream_voice_response(
        self,
        conversation_id: str,
        voice_command: str,
        tts_client,
        speaker_user_id: int | None = None,
    ):
        """Attempt to stream LLM tokens directly to TTS for eligible queries.

        Uses the tool router to predict whether the command is conversational
        (no tool execution needed). If so, streams LLM tokens → sentence
        detection → TTS, yielding audio bytes as each sentence completes.

        Returns:
            An async generator of PCM audio bytes if streaming is eligible,
            or ``None`` if the caller should fall back to the blocking pipeline.
        """
        import time as _time
        from app.core.streaming_handler import extract_sentences

        _t0 = _time.time()

        # Get conversation state from cache
        messages = conversation_cache.get_messages(conversation_id)
        tools = conversation_cache.get_tools(conversation_id)
        available_commands = conversation_cache.get_available_commands(conversation_id) or []

        if not messages:
            return None

        # Apply tool filtering + routing (same as blocking path)
        tools = self._apply_tool_filtering(voice_command, tools, available_commands)
        router_decision = self._apply_tool_routing_with_cache(
            voice_command, tools or [], conversation_id,
        )

        # Check if streaming-eligible
        if not router_decision:
            logger.debug("No router decision — falling back to blocking path")
            return None

        predicted_tool = router_decision.get("tool_name", "")
        confidence = router_decision.get("score", 0.0)

        if predicted_tool not in self._STREAMING_ELIGIBLE_TOOLS:
            logger.debug(
                "Router predicted %s (not streaming-eligible) — blocking path",
                predicted_tool,
            )
            return None

        if confidence < self._STREAMING_MIN_CONFIDENCE:
            logger.debug(
                "Router confidence %.2f < %.2f threshold — blocking path",
                confidence, self._STREAMING_MIN_CONFIDENCE,
            )
            return None

        logger.info(
            "🚀 Streaming path: router=%s confidence=%.2f",
            predicted_tool, confidence,
        )

        # Speaker mismatch detection (same as blocking path)
        if speaker_user_id is not None:
            node_context = conversation_cache.get_node_context(conversation_id)
            warmup_speaker = node_context.get("speaker_user_id") if node_context else None
            if warmup_speaker != speaker_user_id:
                await self._reload_memories_for_speaker(
                    conversation_id, speaker_user_id, node_context, messages,
                )
                # Re-fetch messages after reload
                messages = conversation_cache.get_messages(conversation_id) or messages

        # Pre-execute quick_search if applicable
        search_results = await self._maybe_quick_search(voice_command)
        if search_results:
            messages.append({"role": "system", "content": search_results})

        # Add user message
        suffix: str = (
            self.prompt_provider.user_message_suffix
            if self.prompt_provider else ""
        )
        user_content: str = f"{voice_command}\n{suffix}" if suffix else voice_command
        messages.append({"role": "user", "content": user_content})

        # Override: tell the LLM to respond in plain text (no JSON, no tools)
        messages.append({
            "role": "system",
            "content": (
                "Respond naturally in plain text. Do not use JSON format "
                "or call any tools. Answer the user's question directly."
            ),
        })

        logger.info("🎙️ Starting streaming LLM → TTS (T+%dms)", (_time.time() - _t0) * 1000)

        # Build adapter settings
        node_context = conversation_cache.get_node_context(conversation_id) or {}
        adapter_settings = None
        adapter_hash = node_context.get("adapter_hash")
        if adapter_hash:
            adapter_settings = {"hash": adapter_hash, "enabled": True}

        # Stream LLM tokens → accumulate sentences → TTS each sentence
        #
        # Sentence boundary heuristic: a period/question-mark/exclamation
        # followed by whitespace indicates a complete sentence. We split
        # on that boundary, send the complete part to TTS immediately, and
        # keep the remainder in the buffer for the next sentence.
        _SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

        # Strip model-specific scaffolding (Qwen3's <think>...</think>
        # blocks most notably) from the stream before it reaches TTS.
        # The /no_think control token usually prevents thinking mode from
        # firing, but empty <think></think> wrappers still come through on
        # Qwen3 and get spoken as "think slash think" if we don't strip.
        _THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

        async def _audio_generator():
            token_buffer = ""
            full_response = ""
            sentences_sent = 0

            try:
                async for event in self.llm_client.chat_completion_stream(
                    messages=messages,
                    adapter_settings=adapter_settings,
                    max_tokens=512,
                ):
                    if event.get("done"):
                        break

                    delta = event.get("delta", "")
                    if not delta:
                        continue

                    token_buffer += delta
                    full_response += delta

                    # Strip any complete <think>...</think> spans from the
                    # buffer. DOTALL so the regex spans newlines (Qwen3
                    # emits `<think>\n\n</think>\n\n...` even under
                    # /no_think). Incomplete opens are left intact for
                    # the next iteration when </think> arrives.
                    token_buffer = _THINK_BLOCK_RE.sub("", token_buffer)

                    # If there's an unclosed <think> in the buffer, pause
                    # sentence emission until </think> arrives. Otherwise
                    # anything we flush now would speak raw XML tags.
                    if "<think>" in token_buffer:
                        continue

                    # Check for sentence boundaries in the buffer.
                    # Split yields [complete1, complete2, ..., partial].
                    # If there's at least one split point, everything before
                    # the last element is a complete sentence.
                    parts = _SENTENCE_BOUNDARY.split(token_buffer)
                    if len(parts) >= 2:
                        complete_parts = parts[:-1]
                        token_buffer = parts[-1]

                        for sentence in complete_parts:
                            sentence = clean_for_tts(sentence.strip())
                            if not sentence:
                                continue
                            sentences_sent += 1
                            try:
                                audio_iter, _meta = await tts_client.speak_stream(sentence)
                                async for chunk in audio_iter:
                                    yield chunk
                            except Exception as e:
                                logger.warning("TTS stream error for sentence %d: %s", sentences_sent, e)

                # Flush remaining buffer — strip any lingering complete
                # think block one more time (in case the whole response
                # came back as a single final chunk), then run the
                # universal TTS-safe scrub.
                remaining_text = clean_for_tts(_THINK_BLOCK_RE.sub("", token_buffer).strip())
                if remaining_text:
                    sentences_sent += 1
                    try:
                        audio_iter, _meta = await tts_client.speak_stream(remaining_text)
                        async for chunk in audio_iter:
                            yield chunk
                    except Exception as e:
                        logger.warning("TTS stream error for final sentence: %s", e)

            except Exception as e:
                logger.error("Streaming LLM error: %s", e)
                return

            # Update conversation cache with the full response — strip
            # think blocks so context windows don't accumulate scaffolding
            # that the model didn't actually "say".
            clean_response = _THINK_BLOCK_RE.sub("", full_response).strip()
            messages.append({"role": "assistant", "content": clean_response})
            conversation_cache.update_messages(conversation_id, messages)

            logger.info(
                "✅ Streaming complete: %d sentences, %d chars (T+%dms)",
                sentences_sent, len(full_response), (_time.time() - _t0) * 1000,
            )

        return _audio_generator()

    async def stream_voice_response_with_tools(
        self,
        conversation_id: str,
        voice_command: str,
        tts_client,
        speaker_user_id: int | None = None,
    ):
        """Stream LLM → TTS for commands that need server-side tool execution.

        Strategy: blocking iter 1 (tool decision + server-tool execution),
        then streaming iter 2 (final natural-language response, sentence
        by sentence to TTS). Saves ~2-3s of perceived latency on chatty
        answers (news, weather summaries, etc.) by overlapping LLM
        generation with TTS synthesis.

        Returns an async generator of PCM bytes when applicable, or None
        to signal the caller should fall through to the blocking pipeline.
        """
        # Gate behind env flag; default off until validated under load.
        if os.getenv("JARVIS_STREAM_TOOL_RESPONSES", "false").lower() != "true":
            return None

        import time as _time
        from app.core.tool_executor import tool_executor
        from app.core.tool_execution_engine import _normalize_native_tool_calls
        from app.core.tool_builder import ToolBuilder
        from app.core.tool_registry import tool_registry

        _t0 = _time.time()

        # Get conversation state — work on a COPY so fall-back doesn't
        # pollute the cache (blocking path appends its own user message).
        cached_messages = conversation_cache.get_messages(conversation_id)
        tools = conversation_cache.get_tools(conversation_id)
        available_commands = conversation_cache.get_available_commands(conversation_id) or []

        if not cached_messages or not tools:
            return None

        # Native tool calling required for this path
        if (
            self.prompt_provider is None
            or not self.prompt_provider.supports_native_tools
        ):
            return None

        # Apply tool filtering + routing
        tools = self._apply_tool_filtering(voice_command, tools, available_commands)
        router_decision = self._apply_tool_routing_with_cache(
            voice_command, tools or [], conversation_id,
        )

        if not router_decision:
            return None  # No confident prediction

        predicted_tool = router_decision.get("tool_name", "")
        try:
            min_conf = float(os.getenv("JARVIS_STREAM_TOOL_MIN_CONFIDENCE", "0.85"))
        except ValueError:
            min_conf = 0.85
        confidence = router_decision.get("score", 0.0)

        if confidence < min_conf:
            return None

        # Streaming-eligible tools have their own (faster, no-tool) path.
        if predicted_tool in self._STREAMING_ELIGIBLE_TOOLS:
            return None

        # Server-side tool only — client tools need a node round-trip
        # which the blocking path handles via 202 JSON.
        if not tool_registry.has_tool(predicted_tool):
            logger.debug(
                "Tool-stream: predicted tool '%s' is client-side — fall back",
                predicted_tool,
            )
            return None

        logger.info(
            "🚀 Tool-streaming path: router=%s confidence=%.2f",
            predicted_tool, confidence,
        )

        # Speaker mismatch handling (mirrors blocking path)
        if speaker_user_id is not None:
            node_context = conversation_cache.get_node_context(conversation_id)
            warmup_speaker = node_context.get("speaker_user_id") if node_context else None
            if warmup_speaker != speaker_user_id:
                await self._reload_memories_for_speaker(
                    conversation_id, speaker_user_id, node_context, cached_messages,
                )
                cached_messages = conversation_cache.get_messages(conversation_id) or cached_messages

        # Work on a copy so fall-back doesn't pollute the cache.
        messages = list(cached_messages)

        # Adapter settings
        node_context = conversation_cache.get_node_context(conversation_id) or {}
        adapter_settings: Optional[Dict[str, Any]] = None
        adapter_hash = node_context.get("adapter_hash")
        if adapter_hash:
            adapter_settings = {"hash": adapter_hash, "enabled": True}

        # Build native tool definitions for iter 1
        native_tools = self.prompt_provider.build_tools(
            ToolBuilder.strip_jarvis_extensions(tools or [])
        )

        # Append user message
        suffix = self.prompt_provider.user_message_suffix or ""
        user_content = f"{voice_command}\n{suffix}" if suffix else voice_command
        messages.append({"role": "user", "content": user_content})

        # === Iter 1: blocking tool call decision ===
        try:
            response = await self.llm_client.chat_completion(
                messages=messages,
                conversation_id=conversation_id,
                tools=native_tools,
                tool_choice="auto",
                include_date_context=True,
                adapter_settings=adapter_settings,
                max_tokens=256,
            )
        except Exception as e:
            logger.warning("Tool-stream iter 1 failed: %s — fall back", e)
            return None

        try:
            choice = response["choices"][0]
            msg = choice.get("message")
            finish_reason = choice.get("finish_reason", "stop")
            raw_content = (
                msg.get("content") if isinstance(msg, dict) else msg
            ) or ""
            tool_calls_raw = msg.get("tool_calls") if isinstance(msg, dict) else None
        except (KeyError, IndexError, TypeError, AttributeError) as e:
            logger.warning("Tool-stream iter 1 parse failed: %s — fall back", e)
            return None

        # If no tool calls came back, the LLM gave a direct answer in iter 1.
        # Fall back so the blocking path returns it without re-invoking.
        if finish_reason != "tool_calls" or not tool_calls_raw:
            logger.debug("Tool-stream: iter 1 returned no tool calls — fall back")
            return None

        tool_calls = _normalize_native_tool_calls(tool_calls_raw)

        # Append assistant message with tool calls
        messages.append({
            "role": "assistant",
            "content": raw_content,
            "tool_calls": tool_calls,
        })

        # === Execute server tools synchronously ===
        try:
            server_results, client_calls = tool_executor.execute_tool_calls(
                tool_calls,
                conversation_id=conversation_id,
                user_utterance=voice_command,
            )
        except Exception as e:
            logger.warning("Tool-stream tool exec failed: %s — fall back", e)
            return None

        # Client tools need a node round-trip — fall back so blocking
        # path returns 202 with the tool calls.
        if client_calls:
            logger.debug("Tool-stream: client tools needed — fall back")
            return None

        # Validation requests need special handling — fall back.
        if any(
            tc.get("function", {}).get("name") == "request_validation"
            for tc in tool_calls
        ):
            return None

        if not server_results:
            # Tool returned nothing — let blocking path handle it.
            logger.debug("Tool-stream: no server results — fall back")
            return None

        messages.extend(server_results)

        logger.info(
            "🎙️ Tool-stream iter 1+exec done at T+%dms; starting streaming iter 2",
            (_time.time() - _t0) * 1000,
        )

        # === Iter 2: streaming response ===
        # Reuse the same regexes / pattern as stream_voice_response so behavior
        # matches (think-block strip, sentence detection, etc.).
        _SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
        _THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
        _TOOL_CALL_TAG_RE = re.compile(r"</?tool_call>")

        async def _audio_generator():
            token_buffer = ""
            full_response = ""
            sentences_sent = 0

            try:
                async for event in self.llm_client.chat_completion_stream(
                    messages=messages,
                    adapter_settings=adapter_settings,
                    max_tokens=512,
                ):
                    if event.get("done"):
                        break
                    delta = event.get("delta", "")
                    if not delta:
                        continue
                    token_buffer += delta
                    full_response += delta

                    token_buffer = _THINK_BLOCK_RE.sub("", token_buffer)
                    if "<think>" in token_buffer:
                        continue

                    parts = _SENTENCE_BOUNDARY.split(token_buffer)
                    if len(parts) >= 2:
                        complete_parts = parts[:-1]
                        token_buffer = parts[-1]
                        for sentence in complete_parts:
                            sentence = clean_for_tts(sentence.strip())
                            if not sentence:
                                continue
                            sentences_sent += 1
                            try:
                                audio_iter, _meta = await tts_client.speak_stream(sentence)
                                async for chunk in audio_iter:
                                    yield chunk
                            except Exception as e:
                                logger.warning(
                                    "TTS error sentence %d: %s", sentences_sent, e,
                                )

                # Flush trailing partial
                remaining_text = clean_for_tts(
                    _THINK_BLOCK_RE.sub("", token_buffer).strip()
                )
                if remaining_text:
                    sentences_sent += 1
                    try:
                        audio_iter, _meta = await tts_client.speak_stream(remaining_text)
                        async for chunk in audio_iter:
                            yield chunk
                    except Exception as e:
                        logger.warning("TTS error final sentence: %s", e)

            except Exception as e:
                logger.error("Tool-stream iter 2 error: %s", e)
                return

            # Commit the full conversation (iter 1 user/assistant/tool +
            # iter 2 final assistant) to the cache. Strip think blocks from
            # the recorded assistant message so future turns don't see
            # accumulated scaffolding.
            clean_response = _THINK_BLOCK_RE.sub("", full_response).strip()
            if clean_response:
                messages.append({"role": "assistant", "content": clean_response})
                conversation_cache.update_messages(conversation_id, messages)

            logger.info(
                "✅ Tool-stream complete: %d sentences, %d chars (T+%dms)",
                sentences_sent, len(full_response), (_time.time() - _t0) * 1000,
            )

        return _audio_generator()

    async def stream_continue_with_tool_results(
        self,
        conversation_id: str,
        tool_results: List[Dict[str, Any]],
        tts_client,
    ):
        """Continue a conversation with tool results, streaming the LLM
        response sentence-by-sentence to TTS.

        This is the streaming twin of continue_conversation_with_tool_results
        for the case where the post-tool-results LLM call will produce a
        natural-language response (the common case after a single client tool
        like get_news / get_weather / get_sports / set_timer).

        Compared to the blocking variant, this overlaps LLM token generation
        with TTS synthesis: audio starts ~700ms into iter 2 instead of after
        the full ~3s completion.

        Returns:
            An async generator of PCM audio bytes if the streaming path is
            applicable, or ``None`` to signal the caller should fall back to
            the blocking continue endpoint (which returns JSON text).
        """
        import time as _time

        _t0 = _time.time()

        cached_messages = conversation_cache.get_messages(conversation_id)
        if not cached_messages:
            logger.info("Continue-stream skip: no cached messages for %s", conversation_id[:8])
            return None

        # Work on a copy so fall-back doesn't pollute the cache.
        messages = list(cached_messages)

        # Two modes for adding tool results to the prompt:
        # - Native tools (Qwen3 with proper chat template): role="tool"
        #   messages keyed by tool_call_id.
        # - Text-based (most local models): role="tool" gets dropped by the
        #   chat template, so we strip any existing tool messages and inject
        #   the results as a single user message instead.
        use_native_continue: bool = (
            self.prompt_provider is not None
            and self.prompt_provider.supports_native_tools
        )

        # Two parallel lists:
        # - `messages` is what we COMMIT to the conversation cache (must
        #   reflect the conversation as a future turn would understand it).
        # - `llm_messages` is what we send to the LLM for THIS call only;
        #   it can include transient overrides (e.g. "respond in plain text")
        #   that we don't want to poison the cache.
        if use_native_continue:
            for result in tool_results:
                output = result["output"]
                if not isinstance(output, str):
                    output = json.dumps(output)
                messages.append({
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "content": output,
                })
            llm_messages = list(messages)
        else:
            # Text-based: collect outputs, drop existing role="tool", inject
            # as user message. Mirrors _format_tool_result_text_mode.
            result_parts: List[str] = []
            for tr in tool_results:
                output = tr["output"]
                if not isinstance(output, str):
                    output = json.dumps(output)
                result_parts.append(output)
            tool_context = "\n".join(result_parts)

            messages = [m for m in messages if m.get("role") != "tool"]

            is_knowledge_query = self._is_knowledge_delegation(tool_context)
            if is_knowledge_query:
                messages.append({
                    "role": "user",
                    "content": "Answer the question from your own knowledge. Be brief and conversational.",
                })
            else:
                messages.append({
                    "role": "user",
                    "content": (
                        f"Here are the tool results. Craft a natural response using the "
                        f"ACTUAL values — never use placeholders. Be brief and conversational.\n\n"
                        f"{tool_context}"
                    ),
                })

            # Override the JSON-only constraint from the warmup system prompt
            # so the model emits plain text for the streaming TTS path. Add
            # to llm_messages ONLY — caching this would block tools on every
            # subsequent turn (broke "turn them back off" follow-ups).
            llm_messages = list(messages)
            llm_messages.append({
                "role": "system",
                "content": (
                    "Respond naturally in plain text. Do not use JSON format "
                    "or call any tools. Use the tool results above to answer."
                ),
            })

        # Adapter settings
        node_context = conversation_cache.get_node_context(conversation_id) or {}
        adapter_settings: Optional[Dict[str, Any]] = None
        adapter_hash = node_context.get("adapter_hash")
        if adapter_hash:
            adapter_settings = {"hash": adapter_hash, "enabled": True}

        logger.info(
            "🎙️ Streaming continue: %d tool results, starting LLM stream",
            len(tool_results),
        )

        _SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
        _THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
        _TOOL_CALL_TAG_RE = re.compile(r"</?tool_call>")

        async def _audio_generator():
            token_buffer = ""
            full_response = ""
            sentences_sent = 0

            try:
                async for event in self.llm_client.chat_completion_stream(
                    messages=llm_messages,
                    adapter_settings=adapter_settings,
                    max_tokens=512,
                ):
                    if event.get("done"):
                        break
                    delta = event.get("delta", "")
                    if not delta:
                        continue
                    token_buffer += delta
                    full_response += delta

                    token_buffer = _THINK_BLOCK_RE.sub("", token_buffer)
                    if "<think>" in token_buffer:
                        continue

                    parts = _SENTENCE_BOUNDARY.split(token_buffer)
                    if len(parts) >= 2:
                        complete_parts = parts[:-1]
                        token_buffer = parts[-1]
                        for sentence in complete_parts:
                            sentence = clean_for_tts(sentence.strip())
                            if not sentence:
                                continue
                            sentences_sent += 1
                            try:
                                audio_iter, _meta = await tts_client.speak_stream(sentence)
                                async for chunk in audio_iter:
                                    yield chunk
                            except Exception as e:
                                logger.warning(
                                    "TTS error sentence %d: %s", sentences_sent, e,
                                )

                # Flush trailing partial
                remaining_text = clean_for_tts(
                    _TOOL_CALL_TAG_RE.sub(
                        "", _THINK_BLOCK_RE.sub("", token_buffer)
                    ).strip()
                )
                if remaining_text:
                    sentences_sent += 1
                    try:
                        audio_iter, _meta = await tts_client.speak_stream(remaining_text)
                        async for chunk in audio_iter:
                            yield chunk
                    except Exception as e:
                        logger.warning("TTS error final sentence: %s", e)

            except Exception as e:
                logger.error("Streaming continue iter error: %s", e)
                return

            # Commit the conversation (tool results + final assistant message)
            # to the cache, with scaffolding scrubbed.
            clean_response = _TOOL_CALL_TAG_RE.sub(
                "", _THINK_BLOCK_RE.sub("", full_response)
            ).strip()
            if clean_response:
                messages.append({"role": "assistant", "content": clean_response})
                conversation_cache.update_messages(conversation_id, messages)

            logger.info(
                "✅ Streaming continue complete: %d sentences, %d chars (T+%dms)",
                sentences_sent, len(full_response), (_time.time() - _t0) * 1000,
            )

        return _audio_generator()

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
        logger.info(f"🔄 Continuing conversation {conversation_id[:8]} with {len(tool_results)} tool results")

        # Get conversation state
        messages = conversation_cache.get_messages(conversation_id)
        tools = conversation_cache.get_tools(conversation_id)

        if not messages:
            raise ValueError(f"Conversation {conversation_id} not found or expired")

        # Add tool result messages to conversation history
        for result in tool_results:
            output = result["output"]
            if not isinstance(output, str):
                output = json.dumps(output)

            messages.append({
                "role": "tool",
                "tool_call_id": result["tool_call_id"],
                "content": output,
            })

        use_native_continue: bool = (
            self.prompt_provider is not None
            and self.prompt_provider.supports_native_tools
        )

        if use_native_continue:
            # Native tool calling: model understands role="tool" messages
            last_user_utterance: Optional[str] = None
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    last_user_utterance = msg.get("content")
                    break

            engine = ToolExecutionEngine(self.llm_client, prompt_provider=self.prompt_provider)
            result = await engine.execute(
                conversation_id=conversation_id,
                messages=messages,
                tools=tools or [],
                user_utterance=last_user_utterance,
                max_iterations=10,
            )
        else:
            # Text-based tool calling: local models loop on tool calls because
            # role="tool" messages are dropped/ignored by the chat template.
            # Bypass the tool loop with a single formatting LLM call.
            result = await self._format_tool_result_text_mode(
                conversation_id, messages, tool_results
            )

        # Update cache
        conversation_cache.update_messages(conversation_id, messages)

        return result

    async def _format_tool_result_text_mode(
        self,
        conversation_id: str,
        messages: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Format tool results for text-based models via a single LLM call.

        Text-based models (supports_native_tools=False) can't process role="tool"
        messages — the chat template either drops them or the model re-calls tools
        endlessly.  Instead, inject the tool results as a user message into the
        existing conversation so the KV prefix cache is preserved.
        """
        # Collect tool result outputs
        result_parts: List[str] = []
        for tr in tool_results:
            output = tr["output"]
            if not isinstance(output, str):
                output = json.dumps(output)
            result_parts.append(output)

        tool_context: str = "\n".join(result_parts)

        is_knowledge_query: bool = self._is_knowledge_delegation(tool_context)

        # Strip any role="tool" messages that the text-based model can't handle
        messages[:] = [m for m in messages if m.get("role") != "tool"]

        # Inject tool results as a user message into the EXISTING conversation
        # so the KV prefix cache (system prompt + tools + prior turns) is reused.
        if is_knowledge_query:
            messages.append({
                "role": "user",
                "content": "Answer the question from your own knowledge. Be brief and conversational.",
            })
        else:
            messages.append({
                "role": "user",
                "content": (
                    f"Here are the tool results. Craft a natural response using the "
                    f"ACTUAL values — never use placeholders. Be brief and conversational.\n\n"
                    f"{tool_context}"
                ),
            })

        # Use the existing conversation's LLM call path
        adapter_settings = conversation_cache.get_adapter_settings(conversation_id) if hasattr(conversation_cache, 'get_adapter_settings') else None

        response = await self.llm_client.chat_completion(
            messages=messages,
            conversation_id=conversation_id,
            include_date_context=True,
            adapter_settings=adapter_settings,
            max_tokens=256,
        )

        # Defensive content extraction — llm-proxy can return either
        #   {"choices": [{"message": {"content": "..."}}]}  (OpenAI-ish)
        #   {"choices": [{"message": "..."}]}               (plain string)
        #   {"content": "..."}                               (flat fallback)
        # and we must not crash on any of them, since this path runs on
        # every text-based tool-result formatting (i.e. every Qwen3 turn).
        content: str = ""
        try:
            choice = response["choices"][0]
            msg = choice.get("message") if isinstance(choice, dict) else None
            if isinstance(msg, dict):
                content = msg.get("content", "") or ""
            elif isinstance(msg, str):
                content = msg
            else:
                content = str(choice.get("content", "") if isinstance(choice, dict) else "") or ""
        except (KeyError, IndexError, TypeError, AttributeError):
            content = str(response.get("content", "") if isinstance(response, dict) else "")

        # Clean up tool_call tags that text-based models emit
        content = re.sub(r"</?tool_call>", "", content).strip()

        # Provider-specific scrub (Qwen3 <think> blocks etc.), then
        # universal TTS-safe scrub (emojis). Without this, tool-result
        # responses from text-based models bypass the tool_execution_engine
        # sanitize pass and leak scaffolding into TTS.
        if self.prompt_provider:
            content = self.prompt_provider.sanitize_text(content)
        content = clean_for_tts(content)

        # Update cache
        if content:
            messages.append({"role": "assistant", "content": content})

        return {
            "stop_reason": "complete",
            "assistant_message": content,
        }

    async def _format_tool_result_text_mode_UNUSED(
        self,
        conversation_id: str,
        messages: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """DEPRECATED: Original implementation that created a separate formatting
        conversation, which evicted llama.cpp's KV prefix cache on every tool call.
        Kept for reference.
        """
        user_utterance: str = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_utterance = msg.get("content", "")
                break

        result_parts: List[str] = []
        for tr in tool_results:
            output = tr["output"]
            if not isinstance(output, str):
                output = json.dumps(output)
            result_parts.append(output)

        tool_context: str = "\n".join(result_parts)
        is_knowledge_query: bool = self._is_knowledge_delegation(tool_context)

        if is_knowledge_query:
            formatting_messages: List[Dict[str, Any]] = [
                {
                    "role": "system",
                    "content": (
                        "You are Jarvis, a voice assistant. "
                        "Answer the user's question from your own knowledge. "
                        "Be brief and conversational."
                    ),
                },
                {"role": "user", "content": user_utterance},
            ]
        else:
            formatting_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are Jarvis, a voice assistant. "
                        "The user asked a question and a tool was executed to get the answer. "
                        "The tool result data is below. Craft a natural, spoken response using "
                        "the ACTUAL values from the data. Never use placeholders like "
                        "'[temperature]' — use the real numbers and text from the result. "
                        "Be brief and conversational."
                    ),
                },
                {"role": "user", "content": user_utterance},
                {"role": "assistant", "content": f"Tool result:\n{tool_context}"},
                {"role": "user", "content": "Now give me a brief spoken response using those exact values."},
            ]

        # Use a fresh conversation_id so the MLX backend does NOT reuse the
        # cached KV state (which contains the original system prompt + tools).
        format_conv_id: str = f"fmt-{uuid.uuid4().hex[:12]}"

        response = await self.llm_client.chat_completion(
            messages=formatting_messages,
            conversation_id=format_conv_id,
            include_date_context=True,
            max_tokens=256,
        )

        raw_content: str = response["choices"][0]["message"].get("content", "")

        # Let prompt provider parse (handles <tool_call> tag extraction)
        assistant_message: str = raw_content
        if self.prompt_provider is not None:
            transformed = self.prompt_provider.parse_response(raw_content)
            if transformed is not None:
                try:
                    parsed = json.loads(transformed)
                    # Standard response format: {"message": "...", "tool_calls": [...]}
                    msg = parsed.get("message", "")
                    if msg:
                        assistant_message = msg
                    # Tool call format: look for answer in arguments
                    elif "arguments" in parsed:
                        args = parsed.get("arguments", {})
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                args = {}
                        if isinstance(args, dict):
                            for key in ("message", "answer", "response"):
                                if key in args and isinstance(args[key], str):
                                    assistant_message = args[key]
                                    break
                except json.JSONDecodeError:
                    pass

        # Strip any residual <tool_call> tags the model may have emitted
        assistant_message = re.sub(r"</?tool_call>", "", assistant_message).strip()

        # If the model still produced garbled output, extract from tool results
        if not assistant_message or assistant_message.startswith("{"):
            assistant_message = self._extract_from_tool_results(tool_context)

        logger.info(
            "Text-mode tool result formatted: %d chars (from %d tool results)",
            len(assistant_message),
            len(tool_results),
        )

        # For text-based models, role="tool" messages are invisible (dropped by
        # the chat template) and the raw assistant tool-call JSON is noise.
        # Replace the trailing [assistant(tool_calls), tool(results)] sequence
        # with a single assistant message that embeds the data so follow-up
        # questions can reference specific values from the tool results.
        self._replace_tool_exchange_in_history(
            messages, tool_context, assistant_message
        )

        return {
            "stop_reason": "complete",
            "assistant_message": assistant_message,
        }

    async def stream_final_response(
        self,
        conversation_id: str,
        tool_results: List[Dict[str, Any]],
    ):
        """Stream the final LLM response token-by-token after tool execution.

        Same logic as _format_tool_result_text_mode but uses chat_completion_stream()
        to yield tokens as they arrive. For text-based models only.

        Yields:
            str tokens as they arrive from the LLM
        """
        messages = conversation_cache.get_messages(conversation_id)
        if not messages:
            raise ValueError(f"Conversation {conversation_id} not found or expired")

        # Add tool results to messages (same as continue_conversation)
        for result in tool_results:
            output = result["output"]
            if not isinstance(output, str):
                output = json.dumps(output)
            messages.append({
                "role": "tool",
                "tool_call_id": result["tool_call_id"],
                "content": output,
            })

        # Find user utterance
        user_utterance: str = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_utterance = msg.get("content", "")
                break

        # Collect tool result outputs
        result_parts: List[str] = []
        for tr in tool_results:
            output = tr["output"]
            if not isinstance(output, str):
                output = json.dumps(output)
            result_parts.append(output)
        tool_context: str = "\n".join(result_parts)

        is_knowledge_query: bool = self._is_knowledge_delegation(tool_context)

        # Check if the client supports rich/markdown responses
        node_ctx = conversation_cache.get_node_context(conversation_id)
        rich = node_ctx.get("rich_response", False) if node_ctx else False
        style_hint = (
            "Use markdown formatting (bold, lists, headers) when it improves readability."
            if rich else "Keep it brief and spoken-friendly."
        )

        if is_knowledge_query:
            formatting_messages: List[Dict[str, Any]] = [
                {
                    "role": "system",
                    "content": (
                        "You are Jarvis, a voice assistant. "
                        "Answer the user's question from your own knowledge. "
                        f"Be conversational. {style_hint}"
                    ),
                },
                {"role": "user", "content": user_utterance},
            ]
        else:
            formatting_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are Jarvis, a voice assistant. "
                        "The user asked a question and a tool was executed to get the answer. "
                        "The tool result data is below. Craft a natural response using "
                        "the ACTUAL values from the data. Never use placeholders like "
                        f"'[temperature]' — use the real numbers and text from the result. "
                        f"Be conversational. {style_hint}"
                    ),
                },
                {"role": "user", "content": user_utterance},
                {"role": "assistant", "content": f"Tool result:\n{tool_context}"},
                {"role": "user", "content": "Now give me a response using those exact values."},
            ]

        format_conv_id: str = f"fmt-{uuid.uuid4().hex[:12]}"

        # Buffer tokens so we can strip complete <think>...</think> spans
        # (Qwen3 etc.) before they reach TTS. Emits whenever the buffer is
        # "safe" (no unclosed <think>); holds otherwise. Also runs the
        # universal TTS-safe scrub (emojis).
        _THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
        token_buffer = ""
        full_text = ""
        async for event in self.llm_client.chat_completion_stream(
            messages=formatting_messages,
            include_date_context=True,
            max_tokens=256,
        ):
            if event.get("done"):
                break
            delta = event.get("delta", "")
            if not delta:
                continue
            token_buffer += delta
            token_buffer = _THINK_BLOCK_RE.sub("", token_buffer)
            if "<think>" in token_buffer:
                continue  # wait for </think> before emitting anything
            cleaned = clean_for_tts(re.sub(r"</?tool_call>", "", token_buffer))
            if cleaned:
                full_text += cleaned
                yield cleaned
            token_buffer = ""

        # Flush any trailing buffer (handles final chunk case where all
        # tokens arrived before the loop had a chance to emit).
        if token_buffer:
            remaining = clean_for_tts(
                re.sub(r"</?tool_call>", "", _THINK_BLOCK_RE.sub("", token_buffer))
            )
            if remaining:
                full_text += remaining
                yield remaining

        # Update conversation history with the formatted response
        self._replace_tool_exchange_in_history(
            messages, tool_context, full_text
        )
        conversation_cache.update_messages(conversation_id, messages)

        logger.info(
            "Streamed final response: %d chars (from %d tool results)",
            len(full_text),
            len(tool_results),
        )

    @staticmethod
    def _replace_tool_exchange_in_history(
        messages: List[Dict[str, Any]],
        tool_context: str,
        formatted_response: str,
    ) -> None:
        """Replace tool-exchange messages with a text-friendly assistant message.

        Finds the trailing [assistant(tool_calls), tool(result), ...] sequence
        and replaces it with a single assistant message that includes both the
        condensed data (for follow-up context) and the spoken response.
        """
        # Walk backwards to find the start of the tool exchange
        cut_start: int = len(messages)
        for i in range(len(messages) - 1, -1, -1):
            role = messages[i].get("role")
            if role == "tool":
                cut_start = i
            elif role == "assistant" and messages[i].get("tool_calls"):
                cut_start = i
                break
            else:
                break

        # Condense tool data to avoid bloating the context window
        max_data_chars = 1500
        condensed = tool_context[:max_data_chars]
        if len(tool_context) > max_data_chars:
            condensed += "\n..."

        # Replace the tool exchange with a clean assistant message
        del messages[cut_start:]
        messages.append({
            "role": "assistant",
            "content": (
                f"[Tool data: {condensed}]\n\n{formatted_response}"
            ),
        })

    @staticmethod
    def _is_knowledge_delegation(tool_context: str) -> bool:
        """Detect delegation tools that echo the query instead of providing data.

        Server-side ``answer_question`` returns ``{"query": "..."}``.
        Client-side legacy returns ``{"context": {"query": "..."}}``.
        The model should answer these from its own knowledge rather than trying
        to format a nearly-empty tool result.
        """
        try:
            parsed = json.loads(tool_context)
            if isinstance(parsed, dict):
                # Direct format: {"query": "..."} (server-side tool)
                keys = set(parsed.keys())
                if keys == {"query"} or keys == {"question"}:
                    return True
                # Wrapped format: {"context": {"query": "..."}} (legacy client)
                ctx = parsed.get("context", {})
                if isinstance(ctx, dict):
                    ctx_keys = set(ctx.keys())
                    return ctx_keys == {"query"} or ctx_keys == {"question"}
        except (json.JSONDecodeError, TypeError):
            pass
        return False

    @staticmethod
    def _extract_from_tool_results(tool_context: str) -> str:
        """Extract human-readable content from raw tool result JSON."""
        try:
            parsed = json.loads(tool_context)
            if isinstance(parsed, dict):
                ctx = parsed.get("context", parsed)
                if isinstance(ctx, dict):
                    # Direct result value (e.g., calculator "result": 62)
                    for key in ("result", "answer", "response", "message"):
                        if key in ctx:
                            return str(ctx[key])
                    # Fall back to key-value summary
                    parts = [f"{k}: {v}" for k, v in ctx.items()
                             if v is not None and k not in ("success", "error")]
                    if parts:
                        return ", ".join(parts)
        except (json.JSONDecodeError, TypeError):
            pass
        return tool_context

    def _apply_tool_filtering(
        self,
        voice_command: str,
        tools: Optional[List[Dict[str, Any]]],
        available_commands: List[Dict[str, Any]],
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Apply keyword-based tool filtering.

        Returns filtered tools but does NOT rebuild the system prompt,
        preserving the warmup KV cache.
        """
        if not tools:
            return tools

        filtered_tools = filter_tools_for_utterance(voice_command, tools, available_commands)
        if filtered_tools is not tools:
            filtered_names = [get_tool_name(t) for t in filtered_tools if get_tool_name(t)]
            logger.info("🧹 Filtered tools to %d (KV cache preserved): %s", len(filtered_tools), filtered_names)

        return filtered_tools

    def _apply_tool_routing_with_cache(
        self,
        voice_command: str,
        tools: List[Dict[str, Any]],
        conversation_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Apply tool routing classifier and cache the decision.

        Returns:
            Router decision dict or None
        """
        # Check prompt provider first, then model
        use_classifier = (
            self.prompt_provider.use_tool_classifier
            if self.prompt_provider is not None
            else self.model.use_tool_classifier
        )
        if not use_classifier:
            name = self.prompt_provider.name if self.prompt_provider else self.model.name
            logger.info("Tool classifier disabled for %s", name)
            conversation_cache.set_router_decision(conversation_id, None)
            return None

        try:
            min_conf = float(os.getenv("JARVIS_TOOL_CLASSIFIER_MIN_CONFIDENCE", "0.6"))
        except ValueError:
            min_conf = 0.6

        classifier = get_shared_tool_classifier()
        router_decision = apply_tool_routing(
            voice_command=voice_command,
            tools=tools,
            use_classifier=True,
            route_tool_fn=lambda vc, t: route_tool(vc, t, classifier),
            min_confidence=min_conf,
        )

        conversation_cache.set_router_decision(conversation_id, router_decision)

        if router_decision:
            logger.info(
                "🧭 Router predicted tool=%s score=%.3f use_hint=%s",
                router_decision["tool_name"],
                router_decision["score"],
                router_decision["used"],
            )

        return router_decision

    def _apply_high_confidence_pruning(
        self,
        tools: Optional[List[Dict[str, Any]]],
        router_decision: Optional[Dict[str, Any]],
        server_tool_names: Set[str],
        messages: Optional[List[Dict[str, Any]]] = None,
        conversation_id: Optional[str] = None,
        available_commands: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Prune tools based on high-confidence router decision.

        When pruning fires, rebuilds the system prompt with only the pruned
        tools. This trades the warmup KV cache for a much smaller prompt
        (~500 tokens instead of ~3000), which is a net win because prefill
        on the smaller prompt is faster than reusing a cached large prompt.
        """
        if not router_decision or not tools:
            return tools

        # Determine pruning confidence threshold
        small_mode = get_settings_service().get_bool("model.small_model_mode", True)
        try:
            default_conf = 0.8 if small_mode else 0.85
            prune_conf = float(os.getenv("JARVIS_TOOL_ROUTER_FILTER_MIN_CONFIDENCE", str(default_conf)))
        except ValueError:
            prune_conf = 0.8 if small_mode else 0.85

        pruned_tools = prune_tools_by_router_decision(
            tools=tools,
            router_decision=router_decision,
            server_tool_names=server_tool_names,
            prune_confidence=prune_conf,
        )

        if len(pruned_tools) < len(tools):
            predicted_tool = router_decision.get("tool_name")
            logger.info(
                "🧹 Pruned tools from %d to %d (predicted=%s)",
                len(tools), len(pruned_tools), predicted_tool,
            )

            # Rebuild system prompt with only the pruned tools so the LLM
            # processes far fewer tokens. This invalidates the warmup KV
            # cache but the smaller prompt more than compensates.
            if messages and conversation_id:
                self._rebuild_system_prompt_for_pruned_tools(
                    messages, conversation_id, pruned_tools, available_commands,
                )

        return pruned_tools

    # Keyword patterns that indicate the user wants real-time web info.
    _SEARCH_PATTERNS: list[re.Pattern[str]] = [
        re.compile(r"\b(search|look up|google|find out|search for)\b", re.I),
        re.compile(r"\b(latest|current|recent|today'?s|right now|happening)\b", re.I),
        re.compile(r"\bwho is the (current|new|present)\b", re.I),
        re.compile(r"\bwho (is|are|was) (president|ceo|prime minister|leader|king|queen|governor|mayor)\b", re.I),
        re.compile(r"\b(stock price|share price|market cap)\b", re.I),
        re.compile(r"\bhow much (is|does|are|do)\b.*\b(cost|worth)\b", re.I),
        re.compile(r"\bwhat('?s| is) (new|happening|going on)\b", re.I),
        re.compile(r"\b(did .+ win|who won|final score)\b", re.I),
    ]

    async def _maybe_quick_search(self, utterance: str) -> str | None:
        """Run quick_search if the utterance matches search-intent keywords.

        Returns formatted search results as a system message string, or None
        if no search was triggered.
        """
        if not any(p.search(utterance) for p in self._SEARCH_PATTERNS):
            return None

        tool = tool_registry.get_tool("quick_search")
        if not tool:
            return None

        logger.info("🔍 Keyword-triggered quick_search for: %r", utterance)
        result = tool.execute(query=utterance)

        if "error" in result:
            logger.warning("Quick search failed: %s", result.get("message"))
            return None

        sources = result.get("sources", [])
        if not sources:
            return None

        parts = [f"[Web search results for: {utterance}]"]
        for i, src in enumerate(sources, 1):
            title = src.get("title", f"Source {i}")
            url = src.get("url", "")
            content = src.get("content", "")[:3000]
            parts.append(f"\n## Source {i}: {title}\nURL: {url}\n{content}")
        parts.append(
            "\nUse these web sources to answer the user's question. "
            "Cite sources when relevant."
        )

        elapsed = result.get("elapsed_seconds", "?")
        logger.info("Quick search completed in %ss, %d sources", elapsed, len(sources))

        return "\n".join(parts)

    def _rebuild_system_prompt_for_pruned_tools(
        self,
        messages: List[Dict[str, Any]],
        conversation_id: str,
        pruned_tools: List[Dict[str, Any]],
        available_commands: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Replace the system prompt in messages[0] with one built from pruned tools.

        This dramatically reduces input token count (e.g. 3000 → 500 tokens)
        at the cost of invalidating the warmup KV cache.
        """
        node_context = conversation_cache.get_node_context(conversation_id)
        timezone = conversation_cache.get_timezone(conversation_id)

        if not node_context:
            logger.warning("Cannot rebuild system prompt: no cached node_context")
            return

        # Filter available_commands to only include pruned tool names
        pruned_names = {get_tool_name(t) for t in pruned_tools if get_tool_name(t)}
        filtered_commands = None
        if available_commands:
            filtered_commands = [
                cmd for cmd in available_commands
                if cmd.get("command_name") in pruned_names
            ]

        available_command_flags = build_available_command_flags(
            filtered_commands or []
        )

        new_prompt = self._get_system_prompt(
            node_context, timezone, pruned_tools, available_command_flags
        )

        old_len = len(messages[0]["content"]) if messages else 0
        messages[0] = {"role": "system", "content": new_prompt}
        logger.info(
            "📝 Rebuilt system prompt: %d → %d chars (~%d → ~%d tokens)",
            old_len, len(new_prompt), old_len // 4, len(new_prompt) // 4,
        )

    def _get_system_prompt(
        self,
        node_context: Dict[str, Any],
        timezone: Optional[str],
        tools: List[Dict[str, Any]],
        available_command_flags: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Build system prompt using the best available source.

        Dispatch order:
        1. New-style IJarvisPromptProvider.build_system_prompt (if set)
        2. Legacy model._build_system_prompt (duck-typed)
        3. Minimal fallback

        Returns:
            System prompt string.
        """
        if self.prompt_provider is not None:
            return self.prompt_provider.build_system_prompt(
                node_context, timezone, tools, available_command_flags
            )
        if hasattr(self.model, "_build_system_prompt"):
            return self.model._build_system_prompt(  # type: ignore[attr-defined]
                node_context, timezone, tools, available_command_flags
            )
        return "You are a helpful voice assistant."

    @staticmethod
    def _get_memory_settings(node_context: Dict[str, Any] | None) -> tuple[bool, bool]:
        """Check memory settings for the current context.

        Returns:
            Tuple of (memory_enabled, recall_enabled)
        """
        try:
            from app.services.settings_service import get_settings_service

            settings = get_settings_service()
            household_id = node_context.get("household_id") if node_context else None
            user_id = node_context.get("speaker_user_id") if node_context else None

            kwargs: Dict[str, Any] = {}
            if household_id:
                kwargs["household_id"] = str(household_id)

            memory_enabled = settings.get("memory.enabled", **kwargs)
            recall_enabled = settings.get("memory.recall_enabled", **kwargs)

            # Settings returns the raw value — may be string "true"/"false" or bool
            def to_bool(val: Any) -> bool:
                if isinstance(val, bool):
                    return val
                if isinstance(val, str):
                    return val.lower() in ("true", "1", "yes")
                return bool(val)

            return to_bool(memory_enabled), to_bool(recall_enabled)
        except Exception as e:
            logger.warning(f"⚠️ Failed to check memory settings, defaulting to enabled: {e}")
            return True, True

    async def _reload_memories_for_speaker(
        self,
        conversation_id: str,
        speaker_user_id: int,
        node_context: Dict[str, Any] | None,
        messages: list,
    ) -> None:
        """Reload user memories after a speaker mismatch from parallel warmup.

        When warmup ran during recording, it used the last-known speaker's
        memories.  If STT identified a different speaker, we swap the
        memories in the system prompt and update the conversation cache so
        the LLM sees the correct user context.
        """
        if node_context is None:
            return

        _memory_enabled, _ = self._get_memory_settings(node_context)
        if not _memory_enabled:
            return

        household_id = node_context.get("household_id")
        if not household_id:
            return

        # Update node_context with actual speaker
        node_context["speaker_user_id"] = speaker_user_id

        try:
            from app.db import get_session_local
            from app.services.memory_service import MemoryService

            pinned_max_chars = 500
            try:
                from app.services.settings_service import get_settings_service
                settings = get_settings_service()
                val = settings.get("memory.pinned_max_chars")
                if val is not None:
                    pinned_max_chars = int(val)
            except Exception:
                pass

            SessionLocal = get_session_local()
            db = SessionLocal()
            try:
                svc = MemoryService(db)
                memories_text = svc.get_memories_for_prompt(
                    speaker_user_id, household_id, max_chars=pinned_max_chars
                )
                node_context["user_memories"] = memories_text or ""
                logger.info(
                    f"🔄 Reloaded {len(memories_text or '')} chars of memories for speaker {speaker_user_id}"
                )
            finally:
                db.close()
        except Exception as e:
            logger.warning(f"⚠️ Failed to reload memories for speaker {speaker_user_id}: {e}")
            return

        # Rebuild system prompt with correct memories and replace in cache
        tools = conversation_cache.get_tools(conversation_id)
        timezone = node_context.get("timezone")
        available_command_flags = ""
        available_commands = conversation_cache.get_available_commands(conversation_id) or []
        if available_commands:
            from app.core.voice_command_helpers import build_available_command_flags
            available_command_flags = build_available_command_flags(
                [cmd for cmd in available_commands if cmd.get("command_name")]
            )

        new_system_prompt = self._get_system_prompt(
            node_context, timezone, tools, available_command_flags
        )

        # Replace the system message in the conversation
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = new_system_prompt
        else:
            messages.insert(0, {"role": "system", "content": new_system_prompt})

        # Update cache
        conversation_cache.set(
            conversation_id=conversation_id,
            messages=messages,
            available_commands=available_commands,
            timezone=timezone,
            tools=tools,
            node_context=node_context,
        )


