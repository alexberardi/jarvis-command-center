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

from app.core.conversation_cache import conversation_cache, trim_history_to_max_turns
from app.core.direction_hint import build_direction_hint
from app.core.affect_hint import build_affect_hint
from app.core.turn_context import build_turn_hint, should_double_check_sentinel
from app.core.wake_verification import resolve_wake_verification
from app.core.profile_match import build_profile_match_hint
from app.core.exchange_complete import apply_to_result as apply_exchange_complete
from app.core.errors import ConversationPreconditionError
from app.core.not_for_me import contains_sentinel
from app.core.prompt_providers.shared.core_rules import (
    AMBIENT_CONTEXT_PREFIX,
    EXCHANGE_COMPLETE_INSTRUCTION,
    NOT_FOR_ME_INSTRUCTION,
    RECENTLY_SHOWN_PREFIX,
    build_ambient_context_block,
    build_characterization_section,
    render_referenced_items_block,
    render_signal_block,
)
from app.core.transcript_filter import is_stt_noise
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
from app.core.utils.think_block_stripper import (
    DEFAULT_THINK_DELIMITERS,
    ThinkBlockStripper,
)
from app.core.voice_command_helpers import (
    get_tool_name,
    build_available_command_flags,
    apply_tool_routing,)
from app.core.warmup_service import warmup_service

logger = logging.getLogger("uvicorn")

# Fallback for conversation.max_turns when the settings service is unreachable.
# Must match the definition default in app/services/settings_definitions.py.
_DEFAULT_MAX_HISTORY_TURNS: int = 10


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


# Generic completion-filler phrases a small model emits when it gives up on a
# tool turn (instead of calling/correcting a tool). These are never a useful
# spoken response, so the terminal-filler guard in
# ``process_voice_command_with_tools`` rewrites them to a clarification.
# Compared case-insensitively after stripping trailing whitespace/punctuation.
# Deliberately narrow — excludes "done"/"okay" (sanctioned terse confirmations)
# to avoid suppressing legitimate replies.
_GENERIC_FILLER_RESPONSES: frozenset[str] = frozenset({
    "task completed",
    "task complete",
    "task is complete",
    "task done",
    "task finished",
    "completed",
    "all tasks completed",
})


def _is_transient_system_block(msg: Dict[str, Any]) -> bool:
    """True if ``msg`` is a PER-TURN ephemeral system block that is re-derived
    every turn — the speaker block (name + memories), the router hint, or the
    plain-text streaming override. These must be stripped before re-injection
    so they don't accumulate across multi-turn conversations and bloat the
    prompt toward the model's context limit (there must be exactly one of each
    per turn, not N after N turns).

    Matched by content prefix:
      - "You are speaking with …" / "User Profile - If user asks …"  (speaker)
      - "Router hint:"                                                (router)
      - "Respond naturally in plain text"                            (stream override)
      - "RECENTLY SHOWN …"                                           (referenced items)
      - "<ambient_context> …"                                        (situational bundle)

    Does NOT match the cached system prompt (messages[0]) — including the
    static "User Profile are different…" rule baked inside it — nor genuine
    user/assistant conversation history.
    """
    if not isinstance(msg, dict) or msg.get("role") != "system":
        return False
    content = msg.get("content") or ""
    return (
        content.startswith("You are speaking with ")
        or content.startswith("User Profile - If user asks")
        or content.startswith("Router hint:")
        or content.startswith("Respond naturally in plain text")
        or content.startswith(RECENTLY_SHOWN_PREFIX)
        or content.startswith(AMBIENT_CONTEXT_PREFIX)
    )


def _stash_referenced_items(conversation_id: str, tool_results: List[Dict[str, Any]]) -> None:
    """Remember any ``referenceable_items`` a tool result carried this turn.

    Surfacing commands (e.g. "what emails do I have?") return items the node
    forwards inside each tool result's ``output``. We stash them per
    conversation so the next turn's prompt can re-inject a numbered "recently
    shown" block, letting the model resolve "those"/"#3" to a ref_id and call
    act_on_items. Only stashes when items are present — a non-surfacing turn
    (e.g. mark_read) leaves the prior list intact.
    """
    items: List[Dict[str, Any]] = []
    for tr in tool_results or []:
        output = tr.get("output") if isinstance(tr, dict) else None
        if isinstance(output, str):
            try:
                output = json.loads(output)
            except (json.JSONDecodeError, TypeError):
                output = None
        if isinstance(output, dict):
            ri = output.get("referenceable_items")
            if isinstance(ri, list) and ri:
                items.extend(ri)
    if items:
        conversation_cache.set_referenced_items(conversation_id, items)


def _append_referenced_items_block(messages: List[Dict[str, Any]], conversation_id: str) -> None:
    """Append the per-turn RECENTLY SHOWN transient block, if any items are stashed.

    Mirrors the speaker-block pattern: a trailing ``role=system`` message after
    the cached prefix, stripped+rebuilt every turn via _is_transient_system_block.
    No-op when nothing has been surfaced.
    """
    block = render_referenced_items_block(conversation_cache.get_referenced_items(conversation_id))
    if block:
        messages.append({"role": "system", "content": block})


def _apply_characterization_swap(messages: List[Dict[str, Any]], conversation_id: str) -> None:
    """Reconcile ``messages[0]``'s characterization tail against the stashed base.

    The warmup builds ``messages[0]`` once (for the predicted speaker) and stashes
    the byte-stable base — everything up to the ``<person_view>`` tail — in
    ``node_context["_system_prompt_base"]``. Each turn we recompute the desired
    tail from ``node_context["characterization"]`` and swap ``messages[0]`` to it:

      * prediction correct → desired == current → **no-op** (full KV-cache hit),
      * no characterization → tail cleared back to the bare base,
      * characterization present → base + one ``<person_view>`` section.

    No-op when nothing was stashed (injection disabled / not warmed). Always
    REPLACES the ``messages[0]`` dict rather than mutating it — the tool-stream
    path shares that dict with the conversation cache, so an in-place edit would
    corrupt the cached prefix. Hermetic: reads only the cached node_context, never
    the DB or the model.
    """
    if not messages:
        return
    node_context = conversation_cache.get_node_context(conversation_id)
    if not node_context:
        return
    base = node_context.get("_system_prompt_base")
    if not base:
        return
    section = build_characterization_section(node_context.get("characterization", ""))
    desired = f"{base}\n{section}" if section else base
    if messages[0].get("content") == desired:
        return
    messages[0] = {**messages[0], "content": desired}


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
    ) -> None:
        """
        Initialize a tool-based conversation with warmup.

        Sets up the conversation cache with system prompt, tools, and available commands.
        ALWAYS performs a warmup inference call to reduce first-response latency — the
        warmup is not skippable. Skipping it left the prompt prefix uncached, so every
        voice turn paid a cold prefill (~1.4s).

        Args:
            conversation_id: Unique conversation identifier
            node_context: Context about the node (room, user, etc.)
            timezone: User's timezone
            client_tools: Tools provided by the client
            available_commands: Command definitions for examples/antipatterns
            adapter_settings: Optional adapter configuration
        """
        logger.info(f"🔥 Warming up tool-based conversation {conversation_id[:8]}...")

        # Check memory settings (used for both tool gating and memory loading)
        _memory_enabled, _recall_enabled = self._get_memory_settings(node_context)
        # Web search (outbound-web tools) gate — default OFF, fail-closed.
        _web_search_enabled = self._get_web_search_enabled(node_context)
        logger.info(
            "[web_search] gate household=%s enabled=%s (quick_search/deep_research %s)",
            (node_context or {}).get("household_id"),
            _web_search_enabled,
            "OFFERED" if _web_search_enabled else "WITHHELD",
        )

        # Household speaking-voice persona (VOICE/tone only, walled off from tools
        # + safety). Resolved once here so it rides the cached prefix — byte-stable
        # across speakers, like room/style. Injected in TWO places: the top
        # <personality> block (build_context_header) and the end-of-prompt reminder
        # (_get_system_prompt) for recency. Setting it before prompt-build + cache
        # keeps every turn's rebuild identical.
        if node_context is not None:
            _persona = self._get_household_persona(node_context)
            node_context["household_persona"] = _persona
            logger.info(
                "[persona] household=%s voice=%s (%d chars)",
                node_context.get("household_id"),
                "SET" if _persona else "none",
                len(_persona),
            )

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
            _safe_tool_names: list[str] = ["answer_question"]
            # make_phone_call is ALWAYS offered — even when the household
            # gate is off. Withholding it makes call-shaped utterances
            # improvise with unrelated tools (observed 2026-07-19:
            # hallucinated device control for "Call Tony's Pizzeria"); the
            # tool's execute() speaks an honest refusal when disabled.
            _safe_tool_names.append("make_phone_call")
            # run_errand: the voice-inline errand launcher. A safe one-shot — it
            # fires a fire-and-forget plan draft and returns a spoken ack; the
            # errand only runs after the user taps Run on the plan card.
            _safe_tool_names.append("run_errand")
            # schedule_errand: the same launcher, but for a FUTURE time — it only
            # writes a schedule row and returns a spoken ack. Even safer than
            # run_errand (nothing is planned or shown until the scheduled time, and
            # it still requires a Run tap then). Must be offered alongside run_errand
            # or the model can't pick it for "schedule an errand for tonight…".
            _safe_tool_names.append("schedule_errand")
            # list_scheduled_errands: read-only — lists the household's scheduled +
            # recurring errands and posts a management card with per-errand Cancel
            # buttons. Must be offered or the model improvises "you have no errands
            # scheduled" for "what errands do I have scheduled?" instead of checking.
            _safe_tool_names.append("list_scheduled_errands")
            # Outbound-web tools (quick_search live lookups, deep_research) are
            # gated on the per-household web_search.enabled setting (default
            # OFF). When disabled they never enter the tool list — and because
            # the prompt builder only pulls included_system_prompt_text from
            # tools that made the list, their "you MUST search the web" prompt
            # guidance drops out automatically (absent, not just ignored).
            if _web_search_enabled:
                _safe_tool_names.extend(["deep_research", "quick_search"])
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

        # Load the synthesized characterization for the (predicted) speaker so the
        # prompt build can append its <person_view> tail. Gated behind the same
        # per-household injection flag (default OFF) so this DB read stays entirely
        # off the hot path unless the household opted in. The per-turn
        # _apply_characterization_swap reconciles this against the confirmed
        # speaker without a re-read.
        if (
            speaker_user_id
            and household_id
            and self._get_characterization_injection_enabled(node_context)
        ):
            try:
                from app.db import get_session_local
                from app.services.characterization_service import CharacterizationService

                SessionLocal = get_session_local()
                db = SessionLocal()
                try:
                    row = CharacterizationService(db).get(speaker_user_id, household_id)
                    rendered = (row.rendered or "").strip() if row else ""
                    if rendered:
                        node_context["characterization"] = rendered
                        logger.info(
                            "🧭 Loaded characterization for speaker %s (v%s, %d chars)",
                            speaker_user_id, getattr(row, "version", "?"), len(rendered),
                        )
                finally:
                    db.close()
            except Exception as e:
                logger.warning(f"⚠️ Failed to load characterization: {e}")

        # Assemble the household's ambient situational snapshot (time + weather +
        # today's calendar) ONCE at warmup and freeze it onto node_context. It is NOT
        # put into the warmup system prompt (messages[0]) — that must stay byte-stable
        # across conversations for the llama.cpp prefix cache. Instead the real turn
        # re-wraps this cached text into a TRAILING system message (see
        # _process_voice_command / build_ambient_context_block). Opt-in per household
        # (default off → node_context unchanged for anyone).
        if household_id and _memory_enabled and self._get_ambient_context_enabled(household_id):
            ambient = self._assemble_ambient_bundle(household_id, timezone)
            if ambient:
                node_context["ambient_context"] = ambient
                logger.info("🌤️ Ambient context snapshot assembled (%d chars)", len(ambient))

        # Fetch date keys for prompt providers that need them.
        # We INTENTIONALLY trim to a small high-frequency subset before
        # injecting into the prompt — the full list is ~60 keys (~250 prompt
        # tokens). The backend resolver still accepts the full list, so any
        # less-common key the model emits ("next_friday" etc.) still
        # resolves; the trim just stops sending all 60 as a hint every call.
        try:
            date_keys = await self.llm_client.get_date_keys()
            if date_keys:
                # Send all date keys — the model can't emit keys it doesn't know about
                node_context["date_keys"] = date_keys
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

        # Seed the recently-shown items the node carried over in node_context, so
        # a follow-up on this fresh conversation ("mark those as read") gets the
        # RECENTLY SHOWN block. The LLM-path same-conversation follow-up is seeded
        # separately from the tool result; this covers pre-route / re-wake starts.
        _recent_items = (node_context or {}).get("recently_shown_items")
        if _recent_items:
            conversation_cache.set_referenced_items(conversation_id, _recent_items)

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
        # Warmup inference ALWAYS runs — it is never skippable. (A client-supplied
        # skip_warmup_inference flag used to gate this; a node sending it as True
        # disabled the pre-warm and made every voice turn pay a cold prefill.)
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

    @staticmethod
    def _rewrite_terminal_filler(result: Dict[str, Any]) -> Dict[str, Any]:
        """Terminal-filler guard: a completed tool turn must never degrade into
        a bare generic acknowledgment like "Task completed." When the engine's
        final prose is known filler (e.g. the model gave up after an
        invalid-param retry instead of emitting a corrected tool call), rewrite
        it to a clarification so the user retries rather than hearing
        meaningless filler. Non-prose outcomes (not_for_me, tool_calls,
        validation, error) are left untouched.
        """
        am_norm = (result.get("assistant_message") or "").strip().lower().rstrip(" .!")
        if (
            result.get("stop_reason") not in (
                "not_for_me", "tool_calls", "validation_required",
                "error", "server_tool_complete",
            )
            and am_norm in _GENERIC_FILLER_RESPONSES
        ):
            logger.warning(
                "🛟 Terminal-filler guard: replacing generic filler %r with a clarification",
                result.get("assistant_message"),
            )
            result["assistant_message"] = (
                "Sorry, I didn't quite catch that — could you say it again?"
            )
        return result

    async def process_voice_command_with_tools(
        self,
        voice_command: str,
        conversation_id: str,
        speaker_user_id: int | None = None,
        pre_wake_speech_seconds: float | None = None,
        affect: dict | None = None,
        turn_context: dict | None = None,
    ) -> Dict[str, Any]:
        """
        Process a voice command using tool-based architecture.

        Args:
            voice_command: The user's voice command text
            conversation_id: Conversation ID
            speaker_user_id: Actual speaker from STT (for mismatch detection
                            when warmup used a cached/predicted speaker ID)
            pre_wake_speech_seconds: Optional pre-wake VAD signal from the node
                            (seconds of speech detected in the fixed window
                            before wake fired). Surfaced to the LLM as a
                            direction hint for the not_for_me decision.

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

        # Pre-LLM STT-noise filter. Whisper occasionally emits bracketed
        # action notation (``*sniff*``, ``[laughter]``) for non-speech; if
        # we hand that to the LLM it hallucinates context (e.g. "I smell
        # something burning" off ``*sniff*``). Short-circuit to not_for_me
        # before any inference work — saves an LLM round-trip and forces
        # the node to push its wake-suppression gate forward.
        if is_stt_noise(voice_command):
            logger.info(
                "🚫 not_for_me_prefilter | "
                "conversation_id=%s speaker_user_id=%s "
                "pre_wake_speech_secs=%.2f transcript=%r",
                conversation_id,
                speaker_user_id,
                pre_wake_speech_seconds if pre_wake_speech_seconds is not None else -1.0,
                voice_command,
            )
            return {
                "stop_reason": "not_for_me",
                "assistant_message": "",
            }

        # Wake-clip verification gate. The media proxy transcribed the leading
        # seconds of the speaker_audio (the wake snapshot) in the background;
        # resolve that verdict here with a short fail-open wait. An unverified
        # wake (clip contains nothing wake-word-shaped) is the acoustic
        # fingerprint of an openWakeWord misfire — the one false-wake class
        # confidence/VAD can't catch (prod 2026-08-15: two 0.95-score misfires
        # in a quiet room; one marked a medication off overheard family talk).
        # enforce → silent not_for_me like the noise gate above; bias → the
        # verdict rides turn_context into the wake hint + disables the /think
        # sentinel rescue.
        wake_verification = await resolve_wake_verification(
            conversation_id, turn_context,
        )
        if wake_verification is not None and not wake_verification.get("verified"):
            logger.info(
                "🚫 wake_unverified | mode=%s conversation_id=%s "
                "clip_transcript=%r command=%r",
                wake_verification.get("mode"),
                conversation_id,
                wake_verification.get("transcript"),
                voice_command,
            )
            if wake_verification.get("mode") == "enforce":
                return {
                    "stop_reason": "not_for_me",
                    "assistant_message": "",
                }
            if turn_context is not None:
                turn_context["wake_verified"] = False

        # Get conversation state from cache
        with timing.measure("cache_lookups") if timing else nullcontext():
            messages = conversation_cache.get_messages(conversation_id)
            tools = conversation_cache.get_tools(conversation_id)
            available_commands = conversation_cache.get_available_commands(conversation_id) or []

        # NOTE: the speaker (name + memories) is no longer baked into the
        # cached system prompt. It is injected per-turn as a trailing system
        # message below (see _build_turn_speaker_message), so the warmup prefix
        # stays speaker-agnostic and a speaker change can never invalidate the
        # cache or clobber the cached timezone (the old mismatch-rebuild bug).
        logger.debug(f"⏱️  [T+{(time.time()-_t0)*1000:.0f}ms] Cache lookups done")

        # Propagate model.include_thinking to the prompt provider so
        # user_message_suffix returns /think or /no_think accordingly.
        self._sync_include_thinking(conversation_id)

        if not messages:
            raise ConversationPreconditionError(
                f"Conversation {conversation_id} not found or expired"
            )

        # Get server tool names for filtering
        server_tool_names = get_server_tool_names(tools)

        # Apply keyword-based tool filtering (if enabled)
        with timing.measure("tool_filtering") if timing else nullcontext():
            tools = self._apply_tool_filtering(
                voice_command, tools, available_commands
            )
        logger.debug(f"⏱️  [T+{(time.time()-_t0)*1000:.0f}ms] Tool filtering done")

        # Apply tool routing classifier
        _router_t0 = time.time()
        with timing.measure("tool_routing") if timing else nullcontext():
            router_decision = self._apply_tool_routing_with_cache(
                voice_command, tools or [], conversation_id
            )
        logger.debug(f"⏱️  [T+{(time.time()-_t0)*1000:.0f}ms] Tool routing done (router took {(time.time()-_router_t0)*1000:.0f}ms)")

        # Tool pruning removed 2026-07-21: rebuilding the system prompt with a
        # pruned tool set overwrote messages[0], so the command prompt no longer
        # matched the warmed one and cold-prefilled the whole ~6k-token prompt
        # (~1.3s) every turn — to save ~200 tokens. Keeping the full tool set
        # means every voice request reuses the warmup KV prefix. The router's
        # prediction is still handed to the model via the hint block below.

        # Inject the per-turn speaker block (name + memories) as a trailing
        # system message AFTER the cached/pruned prefix — never into messages[0].
        # Resolved fresh from the actual turn speaker; replaces the old
        # mismatch-rebuild path (which clobbered the prefix + timezone).
        # Strip any prior per-turn transient system blocks (speaker, router
        # hint, stream override) first — exactly ONE of each per turn, so they
        # never accumulate across a multi-turn conversation and bloat the
        # prompt toward the model's context limit.
        messages[:] = [m for m in messages if not _is_transient_system_block(m)]
        # Sliding-window history trim (conversation.max_turns): drop the OLDEST
        # user/assistant exchanges — atomically, tool calls + results ride with
        # their turn — so the prompt can't grow unbounded within the cache TTL.
        # Front-drop on the turn boundary keeps the retained tail byte-stable
        # after the system prefix, so the KV prefix cache re-prefills only from
        # the trim boundary, never the whole prompt.
        trim_history_to_max_turns(
            messages, self._get_max_history_turns(conversation_id)
        )
        # Reconcile messages[0]'s characterization tail for the confirmed speaker —
        # a no-op (full KV-cache hit) when the warmup prediction was already right.
        _apply_characterization_swap(messages, conversation_id)
        _turn_node_ctx = conversation_cache.get_node_context(conversation_id)
        _speaker_msg = await self._build_turn_speaker_message(
            conversation_id, speaker_user_id, _turn_node_ctx,
        )
        if _speaker_msg:
            messages.append({"role": "system", "content": _speaker_msg})

        # Ambient situational block (time/weather/calendar) as a TRAILING per-turn
        # system message AFTER the cached prefix — never in messages[0]. The bundle
        # was assembled + frozen onto node_context at warmup; here we only re-wrap the
        # cached text. Keeping it out of the warmup prefix preserves the llama.cpp
        # prefix cache across conversations (ambient is situational, so it differs
        # between conversations; at the top of the prefix it cold-prefilled the whole
        # prompt — prod TTFS 3s→5s regression, 2026-08-06). Stripped + rebuilt each
        # turn via _is_transient_system_block so it never accumulates.
        _ambient_msg = build_ambient_context_block((_turn_node_ctx or {}).get("ambient_context", ""))
        if _ambient_msg:
            messages.append({"role": "system", "content": _ambient_msg})

        # Per-turn "recently shown" block (mirrors the speaker block): lets the
        # model resolve "those"/"#3"/"the one from abc" to a ref_id for act_on_items.
        _append_referenced_items_block(messages, conversation_id)

        # Add router hint if decision was used
        if router_decision and router_decision.get("used"):
            hint_tool = router_decision.get("tool_name")
            messages.append({
                "role": "system",
                "content": f"Router hint: likely tool is '{hint_tool}'. Use it if it matches intent; otherwise choose the best tool."
            })


        # Inject agent-supplied context (calendar, news, weather, etc.)
        # relevant to the user's utterance via vector similarity search.
        # Appended AFTER the user message so recency bias in attention
        # keeps it top-of-mind when the model generates its response.
        # Gated behind model.advanced_context — when disabled, skip the
        # vector search and embedding query to save latency.
        agent_context: str | None = None
        if self._get_advanced_context_enabled(conversation_id):
            with timing.measure("agent_context") if timing else nullcontext():
                agent_context = await self._get_agent_context(voice_command, conversation_id)

        # Add user message (with optional provider suffix, e.g. /no_think for Qwen3)
        suffix: str = (
            self.prompt_provider.user_message_suffix
            if self.prompt_provider else ""
        )
        direction_hint: str | None = build_direction_hint(
            pre_wake_speech_seconds,
            wake_confidence=(turn_context or {}).get("wake_confidence"),
            turn_source=(turn_context or {}).get("source"),
        )
        if direction_hint:
            logger.info(
                "🎯 Direction hint applied | pre_wake_speech_seconds=%.2f | hint=%s",
                pre_wake_speech_seconds or 0.0,
                direction_hint,
            )
        user_content: str = voice_command
        if direction_hint:
            user_content = f"{user_content}\n\n{direction_hint}"
        affect_hint: str | None = build_affect_hint(affect)
        if affect_hint:
            logger.info("🎭 Affect hint applied | hint=%s", affect_hint)
            user_content = f"{user_content}\n\n{affect_hint}"
        turn_hint: str | None = build_turn_hint(
            (turn_context or {}).get("source"),
            wake_confidence=(turn_context or {}).get("wake_confidence"),
            follow_up_iteration=(turn_context or {}).get("follow_up_iteration"),
            pre_wake_speech_seconds=pre_wake_speech_seconds,
            wake_verified=(turn_context or {}).get("wake_verified"),
        )
        if turn_hint:
            logger.info("🧭 Turn hint applied | hint=%s", turn_hint)
            user_content = f"{user_content}\n\n{turn_hint}"
        profile_hint: str | None = build_profile_match_hint(voice_command, _speaker_msg)
        if profile_hint:
            logger.info("📌 Profile match applied | hint=%s", profile_hint)
            user_content = f"{user_content}\n\n{profile_hint}"
        if agent_context:
            user_content = f"{user_content}\n\n{agent_context}"
        if suffix:
            user_content = f"{user_content}\n{suffix}"
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
                agent_context_chars=len(agent_context) if agent_context else 0,
                sentinel_double_check=should_double_check_sentinel(
                    (turn_context or {}).get("source"),
                    wake_confidence=(turn_context or {}).get("wake_confidence"),
                    follow_up_iteration=(turn_context or {}).get("follow_up_iteration"),
                    pre_wake_speech_seconds=pre_wake_speech_seconds,
                    wake_verified=(turn_context or {}).get("wake_verified"),
                ),
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

        # If the LLM emitted the <not_for_me/> sentinel, the user wasn't
        # addressing Jarvis (overheard conversation, TV audio, etc.). Convert
        # to a silent abort so the node skips TTS and exits the follow-up
        # loop immediately rather than reading the LLM's polite refusal.
        if contains_sentinel(result.get("assistant_message")):
            raw_msg = result.get("assistant_message") or ""
            # Structured fields after the stable ``not_for_me_sentinel`` token
            # so we can chart this in Loki without parsing prose: rate over
            # time, per prompt provider, per direction-hint state. Without
            # this the only way to count not_for_me events was to manually
            # grep the prose log line.
            provider_name = (
                self.prompt_provider.__class__.__name__
                if self.prompt_provider is not None else "none"
            )
            pre_wake = (
                pre_wake_speech_seconds
                if pre_wake_speech_seconds is not None else -1.0
            )
            hint_state = "active" if direction_hint else "absent"
            turn_source = (turn_context or {}).get("source") or (
                "wake_inferred" if pre_wake_speech_seconds is not None else "unknown"
            )
            logger.info(
                "🚫 not_for_me_sentinel | "
                "conversation_id=%s speaker_user_id=%s prompt_provider=%s "
                "pre_wake_speech_secs=%.2f direction_hint=%s turn_source=%s "
                "transcript=%r raw_assistant=%r",
                conversation_id,
                speaker_user_id,
                provider_name,
                pre_wake,
                hint_state,
                turn_source,
                voice_command,
                raw_msg[:160],
            )
            result = {
                "stop_reason": "not_for_me",
                "assistant_message": "",
            }

        # Update cache and return
        conversation_cache.update_messages(conversation_id, messages)
        if result.get("stop_reason") == "error":
            logger.error("❌ Tool loop returned stop_reason=error: %s", result)

        # Strip the <exchange_complete/> marker BEFORE the filler rewrite so
        # the prose check sees the clean reply; sets end_of_exchange for the
        # node to skip its follow-up window.
        result = apply_exchange_complete(result)
        if result.get("end_of_exchange"):
            logger.info("🏁 exchange_complete | conversation_id=%s", conversation_id)

        result = self._rewrite_terminal_filler(result)

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
        pre_wake_speech_seconds: float | None = None,
        affect: dict | None = None,
        turn_context: dict | None = None,
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

        # Pre-LLM STT-noise filter — same gate the blocking path runs. Bail
        # out before opening any LLM stream so the blocking path (which the
        # caller falls through to when we return None) can emit the
        # ``not_for_me`` response without the streaming pre-buffer eating
        # tokens for nothing.
        if is_stt_noise(voice_command):
            logger.debug(
                "Streaming path: STT-noise transcript, deferring to blocking path"
            )
            return None

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

        # quick_search is only streaming-eligible when web search is enabled
        # for the household. When disabled, fall through to the blocking path
        # (where the tool is absent from the whitelist and the keyword
        # pre-exec is gated) so a current-events query never routes into a
        # search-shaped path that can't actually search.
        if predicted_tool == "quick_search" and not self._get_web_search_enabled(
            conversation_cache.get_node_context(conversation_id)
        ):
            logger.debug(
                "quick_search predicted but web_search disabled — blocking path",
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

        # Inject the per-turn speaker block (name + memories) as a trailing
        # system message — the cached prefix stays speaker-agnostic, so a
        # speaker change can't invalidate it or clobber the cached timezone.
        # Strip any prior per-turn transient system blocks (speaker, router
        # hint, stream override) first — exactly ONE of each per turn, so they
        # never accumulate across a multi-turn conversation and bloat the
        # prompt toward the model's context limit.
        messages[:] = [m for m in messages if not _is_transient_system_block(m)]
        # Sliding-window history trim (conversation.max_turns): drop the OLDEST
        # user/assistant exchanges — atomically, tool calls + results ride with
        # their turn — so the prompt can't grow unbounded within the cache TTL.
        # Front-drop on the turn boundary keeps the retained tail byte-stable
        # after the system prefix, so the KV prefix cache re-prefills only from
        # the trim boundary, never the whole prompt.
        trim_history_to_max_turns(
            messages, self._get_max_history_turns(conversation_id)
        )
        # Reconcile messages[0]'s characterization tail for the confirmed speaker —
        # a no-op (full KV-cache hit) when the warmup prediction was already right.
        _apply_characterization_swap(messages, conversation_id)
        _turn_node_ctx = conversation_cache.get_node_context(conversation_id)
        _speaker_msg = await self._build_turn_speaker_message(
            conversation_id, speaker_user_id, _turn_node_ctx,
        )
        if _speaker_msg:
            messages.append({"role": "system", "content": _speaker_msg})

        # Ambient situational block (time/weather/calendar) as a TRAILING per-turn
        # system message AFTER the cached prefix — never in messages[0]. The bundle
        # was assembled + frozen onto node_context at warmup; here we only re-wrap the
        # cached text. Keeping it out of the warmup prefix preserves the llama.cpp
        # prefix cache across conversations (ambient is situational, so it differs
        # between conversations; at the top of the prefix it cold-prefilled the whole
        # prompt — prod TTFS 3s→5s regression, 2026-08-06). Stripped + rebuilt each
        # turn via _is_transient_system_block so it never accumulates.
        _ambient_msg = build_ambient_context_block((_turn_node_ctx or {}).get("ambient_context", ""))
        if _ambient_msg:
            messages.append({"role": "system", "content": _ambient_msg})

        # Per-turn "recently shown" block (mirrors the speaker block): lets the
        # model resolve "those"/"#3"/"the one from abc" to a ref_id for act_on_items.
        _append_referenced_items_block(messages, conversation_id)

        # Sync advanced thinking setting so suffix adapts
        self._sync_include_thinking(conversation_id)

        # Add user message
        suffix: str = (
            self.prompt_provider.user_message_suffix
            if self.prompt_provider else ""
        )
        direction_hint: str | None = build_direction_hint(
            pre_wake_speech_seconds,
            wake_confidence=(turn_context or {}).get("wake_confidence"),
            turn_source=(turn_context or {}).get("source"),
        )
        if direction_hint:
            logger.info(
                "🎯 Direction hint applied (stream path) | pre_wake_speech_seconds=%.2f",
                pre_wake_speech_seconds or 0.0,
            )
        user_content: str = voice_command
        if direction_hint:
            user_content = f"{user_content}\n\n{direction_hint}"
        affect_hint: str | None = build_affect_hint(affect)
        if affect_hint:
            logger.info("🎭 Affect hint applied (stream path) | hint=%s", affect_hint)
            user_content = f"{user_content}\n\n{affect_hint}"
        turn_hint: str | None = build_turn_hint(
            (turn_context or {}).get("source"),
            wake_confidence=(turn_context or {}).get("wake_confidence"),
            follow_up_iteration=(turn_context or {}).get("follow_up_iteration"),
            pre_wake_speech_seconds=pre_wake_speech_seconds,
            wake_verified=(turn_context or {}).get("wake_verified"),
        )
        if turn_hint:
            logger.info("🧭 Turn hint applied (stream path) | hint=%s", turn_hint)
            user_content = f"{user_content}\n\n{turn_hint}"
        profile_hint: str | None = build_profile_match_hint(voice_command, _speaker_msg)
        if profile_hint:
            logger.info("📌 Profile match applied (stream path) | hint=%s", profile_hint)
            user_content = f"{user_content}\n\n{profile_hint}"
        if suffix:
            user_content = f"{user_content}\n{suffix}"
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
        # Delimiters come from the active provider so non-standard markers
        # (e.g. Llama 3.3 thinking's [[[thinking start]]]) just work.
        think_pair = (
            self.prompt_provider.think_delimiters
            if self.prompt_provider
            else DEFAULT_THINK_DELIMITERS
        )
        think_stripper = ThinkBlockStripper.from_pair(think_pair)

        # Open the LLM stream up-front so we can peek for the <not_for_me/>
        # sentinel before committing tokens to TTS. process_voice_command_with_tools
        # catches the sentinel after the blocking call, but the fast path
        # has no such gate — without this pre-check, an ambient utterance
        # the router mis-classified as conversational would get its sentinel
        # TTS'd token-by-token. On detection we return None; the caller
        # falls through to the blocking path which repeats the (cheap)
        # LLM call and converts to a 202 JSON not_for_me.
        llm_stream = self.llm_client.chat_completion_stream(
            messages=messages,
            adapter_settings=adapter_settings,
            max_tokens=512,
        )
        pre_buffer_events: list = []
        pre_buffer_text = ""
        # 80 chars > sentinel length plus think-block padding.
        SENTINEL_PRECHECK_CHARS = 80
        try:
            async for event in llm_stream:
                pre_buffer_events.append(event)
                if event.get("done"):
                    break
                delta = event.get("delta", "")
                if not delta:
                    continue
                pre_buffer_text += delta
                scan_text = think_stripper.strip_complete_blocks(pre_buffer_text)
                if contains_sentinel(scan_text):
                    logger.info(
                        "🚫 Streaming path: <not_for_me/> in pre-buffer — "
                        "falling through to blocking path (buf=%r)",
                        pre_buffer_text[:120],
                    )
                    return None
                # Partial sentinel still being emitted — wait for closing `>`.
                if "<not_for_me" in scan_text.lower():
                    continue
                # Wait through an open think block — can't decide yet.
                if think_stripper.has_open_block(pre_buffer_text):
                    continue
                if len(scan_text) >= SENTINEL_PRECHECK_CHARS:
                    break
        except Exception as e:
            logger.error("Streaming pre-buffer LLM error: %s", e)
            return None

        async def _chain_events(buffered, remaining):
            for ev in buffered:
                yield ev
            async for ev in remaining:
                yield ev

        async def _audio_generator():
            token_buffer = ""
            full_response = ""
            sentences_sent = 0

            try:
                async for event in _chain_events(pre_buffer_events, llm_stream):
                    if event.get("done"):
                        break

                    delta = event.get("delta", "")
                    if not delta:
                        continue

                    token_buffer += delta
                    full_response += delta

                    # Strip any complete think spans from the buffer.
                    # Incomplete opens are left intact for the next
                    # iteration when the end marker arrives.
                    token_buffer = think_stripper.strip_complete_blocks(token_buffer)

                    # If there's an unclosed think block in the buffer,
                    # pause sentence emission until it closes. Otherwise
                    # anything we flush now would speak raw scaffolding.
                    if think_stripper.has_open_block(token_buffer):
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
                remaining_text = clean_for_tts(
                    think_stripper.strip_complete_blocks(token_buffer).strip()
                )
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
            clean_response = think_stripper.strip_complete_blocks(full_response).strip()
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
        pre_wake_speech_seconds: float | None = None,
        affect: dict | None = None,
        turn_context: dict | None = None,
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

        # STT-noise pre-filter — same gate the other two voice paths run.
        # Falls through to the blocking path which converts to not_for_me.
        if is_stt_noise(voice_command):
            logger.debug(
                "Tool-streaming path: STT-noise transcript, deferring to blocking path"
            )
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

        # Work on a copy so fall-back doesn't pollute the cache.
        messages = list(cached_messages)

        # Inject the per-turn speaker block (name + memories) as a trailing
        # system message into the working copy — the cached prefix stays
        # speaker-agnostic, so a speaker change can't invalidate it or clobber
        # the cached timezone (the old mismatch-rebuild bug).
        # Strip any prior per-turn transient system blocks (speaker, router
        # hint, stream override) first — exactly ONE of each per turn, so they
        # never accumulate across a multi-turn conversation and bloat the
        # prompt toward the model's context limit.
        messages[:] = [m for m in messages if not _is_transient_system_block(m)]
        # Sliding-window history trim (conversation.max_turns): drop the OLDEST
        # user/assistant exchanges — atomically, tool calls + results ride with
        # their turn — so the prompt can't grow unbounded within the cache TTL.
        # Front-drop on the turn boundary keeps the retained tail byte-stable
        # after the system prefix, so the KV prefix cache re-prefills only from
        # the trim boundary, never the whole prompt.
        trim_history_to_max_turns(
            messages, self._get_max_history_turns(conversation_id)
        )
        # Reconcile messages[0]'s characterization tail for the confirmed speaker —
        # a no-op (full KV-cache hit) when the warmup prediction was already right.
        _apply_characterization_swap(messages, conversation_id)
        _turn_node_ctx = conversation_cache.get_node_context(conversation_id)
        _speaker_msg = await self._build_turn_speaker_message(
            conversation_id, speaker_user_id, _turn_node_ctx,
        )
        if _speaker_msg:
            messages.append({"role": "system", "content": _speaker_msg})

        # Ambient situational block (time/weather/calendar) as a TRAILING per-turn
        # system message AFTER the cached prefix — never in messages[0]. The bundle
        # was assembled + frozen onto node_context at warmup; here we only re-wrap the
        # cached text. Keeping it out of the warmup prefix preserves the llama.cpp
        # prefix cache across conversations (ambient is situational, so it differs
        # between conversations; at the top of the prefix it cold-prefilled the whole
        # prompt — prod TTFS 3s→5s regression, 2026-08-06). Stripped + rebuilt each
        # turn via _is_transient_system_block so it never accumulates.
        _ambient_msg = build_ambient_context_block((_turn_node_ctx or {}).get("ambient_context", ""))
        if _ambient_msg:
            messages.append({"role": "system", "content": _ambient_msg})

        # Per-turn "recently shown" block (mirrors the speaker block): lets the
        # model resolve "those"/"#3"/"the one from abc" to a ref_id for act_on_items.
        _append_referenced_items_block(messages, conversation_id)

        # Sync advanced thinking setting so suffix adapts
        self._sync_include_thinking(conversation_id)

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
        direction_hint: str | None = build_direction_hint(
            pre_wake_speech_seconds,
            wake_confidence=(turn_context or {}).get("wake_confidence"),
            turn_source=(turn_context or {}).get("source"),
        )
        if direction_hint:
            logger.info(
                "🎯 Direction hint applied (tool-stream path) | pre_wake_speech_seconds=%.2f",
                pre_wake_speech_seconds or 0.0,
            )
        user_content = voice_command
        if direction_hint:
            user_content = f"{user_content}\n\n{direction_hint}"
        affect_hint: str | None = build_affect_hint(affect)
        if affect_hint:
            logger.info("🎭 Affect hint applied (tool-stream path) | hint=%s", affect_hint)
            user_content = f"{user_content}\n\n{affect_hint}"
        turn_hint: str | None = build_turn_hint(
            (turn_context or {}).get("source"),
            wake_confidence=(turn_context or {}).get("wake_confidence"),
            follow_up_iteration=(turn_context or {}).get("follow_up_iteration"),
            pre_wake_speech_seconds=pre_wake_speech_seconds,
            wake_verified=(turn_context or {}).get("wake_verified"),
        )
        if turn_hint:
            logger.info("🧭 Turn hint applied (tool-stream path) | hint=%s", turn_hint)
            user_content = f"{user_content}\n\n{turn_hint}"
        profile_hint: str | None = build_profile_match_hint(voice_command, _speaker_msg)
        if profile_hint:
            logger.info("📌 Profile match applied (tool-stream path) | hint=%s", profile_hint)
            user_content = f"{user_content}\n\n{profile_hint}"
        if suffix:
            user_content = f"{user_content}\n{suffix}"
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
        # matches (think-block strip, sentence detection, etc.). Think
        # delimiters come from the active provider.
        _SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
        _TOOL_CALL_TAG_RE = re.compile(r"</?tool_call>")
        think_pair = (
            self.prompt_provider.think_delimiters
            if self.prompt_provider
            else DEFAULT_THINK_DELIMITERS
        )
        think_stripper = ThinkBlockStripper.from_pair(think_pair)

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

                    token_buffer = think_stripper.strip_complete_blocks(token_buffer)
                    if think_stripper.has_open_block(token_buffer):
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
                    think_stripper.strip_complete_blocks(token_buffer).strip()
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
            clean_response = think_stripper.strip_complete_blocks(full_response).strip()
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

        # Remember any items a tool surfaced this turn (parity with the blocking
        # continue) so a later turn can re-inject the "recently shown" list.
        _stash_referenced_items(conversation_id, tool_results)

        cached_messages = conversation_cache.get_messages(conversation_id)
        if not cached_messages:
            logger.info("Continue-stream skip: no cached messages for %s", conversation_id[:8])
            return None

        # Work on a copy so fall-back doesn't pollute the cache.
        messages = list(cached_messages)

        # Native-tools providers (Qwen3 with a proper chat template) keep
        # role="tool" messages keyed by tool_call_id; text-based providers drop
        # them. Computed up front because the fast-path below only applies to
        # text-based providers (it commits a text-mode history shape).
        use_native_continue: bool = (
            self.prompt_provider is not None
            and self.prompt_provider.supports_native_tools
        )

        # ── Fast-path ──────────────────────────────────────────────────────
        # If every tool result already carries a clean spoken message/response
        # string, speak it directly and skip the formatting LLM call entirely.
        # This mirrors the blocking text-mode path (_format_tool_result_text_
        # mode) so the voice path never re-renders a perfectly good answer into
        # generic filler ("Task completed.") — and it shaves the ~2-3s
        # formatting call off the turn. Excluded:
        #   • knowledge-delegation results (answer_question etc.) — they carry
        #     no message and MUST still hit the LLM.
        # Native-tools providers ARE included (previously excluded): they skip
        # the formatting LLM too, but commit the NATIVE history shape below
        # (preserve the role="tool" messages keyed by tool_call_id) so a
        # follow-up turn's history stays valid. Excluding them regressed the
        # move to native (Qwen3.5-9B): every tool command paid a ~0.8s
        # formatting call that produced no usable prose -> robotic fallback.
        fast_context = "\n".join(
            out if isinstance(out, str) else json.dumps(out)
            for out in (tr.get("output", {}) for tr in tool_results)
        )
        if (
            tool_results
            and not self._is_knowledge_delegation(fast_context)
        ):
            fast_parts: List[str] = []
            for tr in tool_results:
                output = tr.get("output", {})
                if isinstance(output, str):
                    try:
                        output = json.loads(output)
                    except (json.JSONDecodeError, TypeError):
                        output = {}
                if isinstance(output, dict):
                    msg = output.get("message") or output.get("response")
                    if isinstance(msg, str) and msg.strip():
                        fast_parts.append(msg.strip())
            spoken = (
                clean_for_tts(" ".join(fast_parts))
                if fast_parts and len(fast_parts) == len(tool_results)
                else ""
            )
            # Only fast-path when there is actually something to speak. If
            # clean_for_tts wiped the message to empty (emoji/markdown-only),
            # fall through to the LLM path (which has its own tool-result
            # fallback) rather than streaming a 0-byte 200 the node would
            # treat as a successful but silent turn.
            if spoken:
                # Commit the spoken turn to cache so follow-up turns see a
                # coherent history (mirrors the LLM branch's commit below).
                # Native-tools providers MUST keep the role="tool" messages
                # (keyed by tool_call_id, appended after the assistant tool_call
                # already in `messages`) or the next turn's history is invalid;
                # text-based providers drop them (text-mode shape). This mirrors
                # the native vs text commit in the LLM path below.
                if use_native_continue:
                    committed = list(messages)
                    for result in tool_results:
                        _out = result["output"]
                        if not isinstance(_out, str):
                            _out = json.dumps(_out)
                        committed.append({
                            "role": "tool",
                            "tool_call_id": result["tool_call_id"],
                            "content": _out,
                        })
                else:
                    committed = [m for m in messages if m.get("role") != "tool"]
                committed.append({"role": "assistant", "content": spoken})
                conversation_cache.update_messages(conversation_id, committed)
                logger.info(
                    "🚀 Streaming continue fast-path: speaking tool message "
                    "directly (%d chars), skipping formatting LLM call",
                    len(spoken),
                )
                _FAST_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

                async def _fast_audio_generator():
                    for sentence in _FAST_SENTENCE_BOUNDARY.split(spoken):
                        sentence = sentence.strip()
                        if not sentence:
                            continue
                        try:
                            audio_iter, _meta = await tts_client.speak_stream(sentence)
                            async for chunk in audio_iter:
                                yield chunk
                        except Exception as e:
                            logger.warning("TTS error (fast-path): %s", e)

                return _fast_audio_generator()

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
                        f"ACTUAL values — never use placeholders. If the results are "
                        f"empty or contain no items, say so plainly (e.g. \"You have "
                        f"nothing on your calendar today\"). Never reply with a bare "
                        f"acknowledgment like \"Task completed\", \"Done\", or \"Okay\" — "
                        f"always state the actual result. Be brief and conversational.\n\n"
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
                    "or call any tools. Use the tool results above to answer. "
                    "Never reply with a generic acknowledgment like \"Task "
                    "completed\" or \"Done\" — always state the actual result, "
                    "and if there is nothing to report, say so plainly."
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
        _TOOL_CALL_TAG_RE = re.compile(r"</?tool_call>")
        think_pair = (
            self.prompt_provider.think_delimiters
            if self.prompt_provider
            else DEFAULT_THINK_DELIMITERS
        )
        think_stripper = ThinkBlockStripper.from_pair(think_pair)

        def _looks_unspeakable(text: str) -> bool:
            """True if `text` is empty, raw JSON (a bare tool-call re-emit),
            or leftover think-residue — i.e. must NOT be sent to TTS."""
            t = text.strip()
            if not t:
                return True
            if "<think" in t.lower():
                return True
            return t.startswith("{") or t.startswith("[")

        def _tool_result_fallback() -> str:
            """Spoken text derived from the tool results when the formatting
            LLM fails to produce usable prose (mirrors blocking path 1586-1613)."""
            parts: List[str] = []
            for tr in tool_results:
                out = tr.get("output", {})
                if isinstance(out, str):
                    try:
                        out = json.loads(out)
                    except (json.JSONDecodeError, TypeError):
                        out = {}
                if isinstance(out, dict):
                    m = out.get("message") or out.get("response")
                    if m:
                        parts.append(str(m))
                    elif out.get("success") is False:
                        parts.append(out.get("error", "The command failed."))
            return clean_for_tts(" ".join(parts)).strip()

        async def _audio_generator():
            token_buffer = ""
            full_response = ""
            sentences_sent = 0
            substituted: Optional[str] = None

            try:
                async for event in self.llm_client.chat_completion_stream(
                    messages=llm_messages,
                    adapter_settings=adapter_settings,
                    max_tokens=512,
                    temperature=0.7,
                ):
                    if event.get("done"):
                        break
                    delta = event.get("delta", "")
                    if not delta:
                        continue
                    token_buffer += delta
                    full_response += delta

                    token_buffer = think_stripper.strip_complete_blocks(token_buffer)
                    if think_stripper.has_open_block(token_buffer):
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

                # Flush trailing partial — but never speak raw JSON / think
                # residue (a model that re-emits a bare tool-call instead of
                # prose). Suppress it; the substitution below handles it.
                remaining_text = clean_for_tts(
                    _TOOL_CALL_TAG_RE.sub(
                        "", think_stripper.strip_complete_blocks(token_buffer)
                    ).strip()
                )
                if remaining_text and not _looks_unspeakable(remaining_text):
                    sentences_sent += 1
                    try:
                        audio_iter, _meta = await tts_client.speak_stream(remaining_text)
                        async for chunk in audio_iter:
                            yield chunk
                    except Exception as e:
                        logger.warning("TTS error final sentence: %s", e)

                # Guard: if nothing usable was spoken (model emitted only JSON /
                # think-residue / empty), substitute a tool-derived message so
                # the user never hears raw JSON, silence, or "Task completed."
                if sentences_sent == 0:
                    substituted = _tool_result_fallback()
                    if substituted:
                        logger.info(
                            "Streaming continue produced no usable prose — "
                            "substituting tool-result fallback: %s", substituted,
                        )
                        for sentence in _SENTENCE_BOUNDARY.split(substituted):
                            sentence = sentence.strip()
                            if not sentence:
                                continue
                            try:
                                audio_iter, _meta = await tts_client.speak_stream(sentence)
                                async for chunk in audio_iter:
                                    yield chunk
                            except Exception as e:
                                logger.warning("TTS error (fallback): %s", e)
                    else:
                        logger.warning(
                            "Streaming continue produced no usable prose and no "
                            "tool-result fallback available — staying silent",
                        )

            except Exception as e:
                logger.error("Streaming continue iter error: %s", e)
                return

            # Commit the conversation (tool results + final assistant message)
            # to the cache, with scaffolding scrubbed. Prefer the substituted
            # text; never commit raw JSON / think-residue to the cache.
            clean_response = _TOOL_CALL_TAG_RE.sub(
                "", think_stripper.strip_complete_blocks(full_response)
            ).strip()
            if substituted:
                clean_response = substituted
            elif _looks_unspeakable(clean_response):
                clean_response = ""
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
            raise ConversationPreconditionError(
                f"Conversation {conversation_id} not found or expired"
            )

        # Remember any items a tool surfaced this turn so the next turn can
        # re-inject them ("mark those as read" / "send me #3").
        _stash_referenced_items(conversation_id, tool_results)

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

        # Final replies after tool rounds ("Timer set!") are the most common
        # terminal replies — same marker handling as the initial turn.
        result = apply_exchange_complete(result)
        if result.get("end_of_exchange"):
            logger.info("🏁 exchange_complete (continue) | conversation_id=%s", conversation_id)

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

        # Fast path: if all tool results have a clear message/response field,
        # skip the expensive LLM formatting call (~3s) and return directly.
        # Most commands return {"success": true, "message": "..."} which
        # doesn't need LLM rephrasing.
        fast_parts: list[str] = []
        for tr in tool_results:
            output = tr.get("output", {})
            if isinstance(output, str):
                try:
                    output = json.loads(output)
                except (json.JSONDecodeError, TypeError):
                    output = {}
            if isinstance(output, dict):
                msg = output.get("message") or output.get("response")
                if isinstance(msg, str) and msg.strip():
                    fast_parts.append(msg.strip())
        if fast_parts and len(fast_parts) == len(tool_results):
            content = " ".join(fast_parts)
            logger.info("Fast-path formatting: skipping LLM call, using tool message directly")
            if content:
                messages[:] = [m for m in messages if m.get("role") != "tool"]
                messages.append({"role": "assistant", "content": content})
            return {
                "stop_reason": "complete",
                "assistant_message": content,
            }

        is_knowledge_query: bool = self._is_knowledge_delegation(tool_context)

        # Pull the user's most recent question out of the conversation so the
        # formatting prompt can name it explicitly. Without this, the model
        # tends to narrate every field in the tool result instead of answering
        # the actual question (e.g. "Is it going to rain?" → wall of current
        # temp/humidity/wind).
        voice_command: Optional[str] = None
        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                voice_command = content.strip()
                break
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text = (part.get("text") or "").strip()
                        if text:
                            voice_command = text
                            break
                if voice_command:
                    break

        # Strip any role="tool" messages that the text-based model can't handle
        messages[:] = [m for m in messages if m.get("role") != "tool"]

        # Disable thinking on formatting calls — just rephrase tool results.
        # Saves ~140 tokens and ~2s per formatting response.
        node_ctx = conversation_cache.get_node_context(conversation_id) or {}
        think_suffix: str = "\n/no_think"
        max_tokens: int = 256

        question_preamble: str = (
            f'The user asked: "{voice_command}"\n\n' if voice_command else ""
        )

        # Inject tool results as a user message into the EXISTING conversation
        # so the KV prefix cache (system prompt + tools + prior turns) is reused.
        if is_knowledge_query:
            messages.append({
                "role": "user",
                "content": (
                    f"{question_preamble}"
                    f"Answer the question from your own knowledge. Be brief "
                    f"and conversational.{think_suffix}"
                ),
            })
        else:
            messages.append({
                "role": "user",
                "content": (
                    f"{question_preamble}"
                    f"Tool results below. Answer the user's question DIRECTLY "
                    f"using only the fields relevant to what they asked — don't "
                    f"list fields they didn't ask about. Use ACTUAL values, "
                    f"never placeholders. Be brief and conversational.\n\n"
                    f"{tool_context}{think_suffix}"
                ),
            })

        # Use the existing conversation's LLM call path
        adapter_settings = conversation_cache.get_adapter_settings(conversation_id) if hasattr(conversation_cache, 'get_adapter_settings') else None

        response = await self.llm_client.chat_completion(
            messages=messages,
            conversation_id=conversation_id,
            include_date_context=True,
            adapter_settings=adapter_settings,
            max_tokens=max_tokens,
            temperature=0.7,
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

        # Clean up model scaffolding: tool_call tags, think blocks, etc.
        content = re.sub(r"</?tool_call>", "", content).strip()

        # Extract <think> content before stripping for optional client use
        _think_matches = re.findall(r"<think>(.*?)</think>", content, re.DOTALL)
        reasoning = "\n\n".join(m.strip() for m in _think_matches if m.strip()) or None

        # Provider-specific scrub (Qwen3 <think> blocks etc.), then
        # universal TTS-safe scrub (emojis).
        if self.prompt_provider:
            content = self.prompt_provider.sanitize_text(content)
        content = clean_for_tts(content)

        # AFTER all scrubbing: if the model emitted a tool call instead of
        # prose, the content is now bare JSON. Build a fallback from the
        # actual tool results rather than showing raw JSON to the user.
        if content.startswith("{") and content.endswith("}"):
            try:
                parsed_tc = json.loads(content)
                if isinstance(parsed_tc, dict) and "name" in parsed_tc:
                    fallback_parts: list[str] = []
                    for tr in tool_results:
                        output = tr.get("output", {})
                        if isinstance(output, str):
                            try:
                                output = json.loads(output)
                            except (json.JSONDecodeError, TypeError):
                                pass
                        if isinstance(output, dict):
                            msg = output.get("message") or output.get("response")
                            if msg:
                                fallback_parts.append(str(msg))
                            elif output.get("success") is False:
                                fallback_parts.append(output.get("error", "The command failed."))
                    content = " ".join(fallback_parts) if fallback_parts else "Done."
                    logger.info(
                        "Formatting call emitted tool call instead of prose — "
                        "using tool result fallback: %s", content,
                    )
            except (json.JSONDecodeError, TypeError):
                pass

        # Update cache
        if content:
            messages.append({"role": "assistant", "content": content})

        result: Dict[str, Any] = {
            "stop_reason": "complete",
            "assistant_message": content,
        }
        if reasoning:
            result["reasoning"] = reasoning
        return result

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

        # Buffer tokens so we can strip complete think spans before they
        # reach TTS. Emits whenever the buffer is "safe" (no unclosed
        # think block); holds otherwise. Also runs the universal TTS-safe
        # scrub (emojis). Think delimiters come from the active provider.
        think_pair = (
            self.prompt_provider.think_delimiters
            if self.prompt_provider
            else DEFAULT_THINK_DELIMITERS
        )
        think_stripper = ThinkBlockStripper.from_pair(think_pair)
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
            token_buffer = think_stripper.strip_complete_blocks(token_buffer)
            if think_stripper.has_open_block(token_buffer):
                continue  # wait for end marker before emitting anything
            cleaned = clean_for_tts(re.sub(r"</?tool_call>", "", token_buffer))
            if cleaned:
                full_text += cleaned
                yield cleaned
            token_buffer = ""

        # Flush any trailing buffer (handles final chunk case where all
        # tokens arrived before the loop had a chance to emit).
        if token_buffer:
            remaining = clean_for_tts(
                re.sub(r"</?tool_call>", "", think_stripper.strip_complete_blocks(token_buffer))
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

    def _sync_include_thinking(self, conversation_id: str) -> bool:
        """Read model.include_thinking and propagate to the provider.

        Sets ``self.prompt_provider.include_thinking`` so user_message_suffix returns
        /think or /no_think. DECOUPLED from model.advanced_context (proactive context
        injection), which is read separately by ``_get_advanced_context_enabled`` — a
        household can have fast context injection without paying for chain-of-thought.
        """
        enabled = False
        try:
            node_ctx = conversation_cache.get_node_context(conversation_id)
            hh_id = node_ctx.get("household_id") if node_ctx else None
            if hh_id:
                settings = get_settings_service()
                val = settings.get("model.include_thinking", household_id=str(hh_id))
                if val is not None:
                    enabled = str(val).lower() in ("true", "1")
        except Exception:
            pass  # default to False
        if self.prompt_provider:
            self.prompt_provider.include_thinking = enabled
        return enabled

    def _get_max_history_turns(self, conversation_id: str) -> int:
        """conversation.max_turns — sliding-window size for conversation history
        (per household). Each turn is one user/assistant exchange (including any
        tool calls/results it made); oldest turns are dropped first at the start
        of every turn so the prompt can't grow unbounded within the cache TTL."""
        try:
            node_ctx = conversation_cache.get_node_context(conversation_id)
            hh_id = node_ctx.get("household_id") if node_ctx else None
            val = get_settings_service().get(
                "conversation.max_turns",
                default=_DEFAULT_MAX_HISTORY_TURNS,
                household_id=str(hh_id) if hh_id else None,
            )
            return int(val)
        except Exception:
            return _DEFAULT_MAX_HISTORY_TURNS

    def _get_advanced_context_enabled(self, conversation_id: str) -> bool:
        """model.advanced_context — whether the background agents' weather/calendar/
        news/reminder context is injected into this turn (per household)."""
        try:
            node_ctx = conversation_cache.get_node_context(conversation_id)
            hh_id = node_ctx.get("household_id") if node_ctx else None
            if hh_id:
                val = get_settings_service().get(
                    "model.advanced_context", household_id=str(hh_id)
                )
                return val is not None and str(val).lower() in ("true", "1")
        except Exception:
            pass
        return False

    async def _get_agent_context(
        self,
        utterance: str,
        conversation_id: str,
    ) -> str | None:
        """Retrieve agent-injected context relevant to the user's utterance.

        Searches household-wide memories (user_id IS NULL) injected by
        background agents (calendar, news, weather).  Returns a formatted
        "Current context" block for prompt injection, or None if nothing
        relevant was found.

        Non-fatal: all errors are caught and logged so agent context
        never blocks voice command processing.
        """
        node_context = conversation_cache.get_node_context(conversation_id)
        if not node_context:
            return None

        household_id = node_context.get("household_id")
        if not household_id:
            return None

        # Check if agent context is enabled
        try:
            settings = get_settings_service()
            enabled = settings.get(
                "memory.agent_context_enabled",
                household_id=str(household_id),
            )
            if enabled is not None and str(enabled).lower() in ("false", "0"):
                return None

            max_results = int(
                settings.get("memory.agent_context_max_results", household_id=str(household_id)) or 5
            )
            max_chars = int(
                settings.get("memory.agent_context_max_chars", household_id=str(household_id)) or 500
            )
            similarity_threshold = float(
                settings.get("memory.agent_context_similarity_threshold", household_id=str(household_id)) or 0.25
            )
        except Exception:
            max_results = 5
            max_chars = 500
            similarity_threshold = 0.25

        try:
            from app.db import get_session_local
            from app.services.agent_context_service import AgentContextService

            SessionLocal = get_session_local()
            db = SessionLocal()
            try:
                svc = AgentContextService(db)
                result = svc.get_relevant_context(
                    household_id=household_id,
                    query=utterance,
                    max_results=max_results,
                    max_chars=max_chars,
                    similarity_threshold=similarity_threshold,
                )
                if result:
                    logger.info(
                        "📋 Injecting %d chars of agent context for household %s",
                        len(result), household_id,
                    )
                return result or None
            finally:
                db.close()

        except Exception as e:
            logger.warning("Failed to get agent context: %s", e)
            return None

    def _get_ambient_context_enabled(self, household_id: str) -> bool:
        """Opt-in gate for the cached ambient situational snapshot (default OFF).

        Read PER-HOUSEHOLD (matches memory.agent_context_enabled) — a global read
        would ignore a household that enabled it in admin.
        """
        try:
            from app.services.settings_service import get_settings_service
            val = get_settings_service().get(
                "ambient_context.enabled", household_id=str(household_id)
            )
            return val is True or str(val).lower() in ("true", "1", "yes")
        except Exception:
            return False

    def _assemble_ambient_bundle(self, household_id: str, timezone: Optional[str]) -> str:
        """Snapshot the household's situational context (current time + weather + today's
        calendar) as a byte-stable string for the cached prefix.

        Household-scoped (weather/calendar are household-wide agent memories), so it's
        speaker-agnostic and safe in the shared prefix. The clock is quantized to 15 min
        and rendered ABSOLUTE — no live/relative value may survive into the string, or the
        per-turn prompt would differ and break the llama.cpp prefix cache. Best-effort: a
        missing source drops its line; any failure returns "" (fail-open, prefix stays
        byte-identical to the pre-feature header).
        """
        try:
            from datetime import datetime
            try:
                from zoneinfo import ZoneInfo
                now = datetime.now(ZoneInfo(timezone or "UTC"))
            except Exception:
                now = datetime.utcnow()
            clock = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)
            lines = [f"As of {clock.strftime('%-I:%M %p')}, {clock.strftime('%A, %b %-d')}."]

            from app.db import get_session_local
            from app.models import UserMemory, Signal

            # Latest non-expired household (user_id IS NULL) memory per category, labeled.
            # Weather + calendar are fed by the node's agents today; reminders arrive once
            # reminder_agent injects a "reminder" memory (see slice-2 plumbing).
            categories = (("weather", "Weather"), ("calendar", "Today"), ("reminder", "Reminders"))
            SessionLocal = get_session_local()
            db = SessionLocal()
            try:
                utcnow = datetime.utcnow()
                for category, label in categories:
                    base = db.query(UserMemory).filter(
                        UserMemory.user_id.is_(None),
                        UserMemory.household_id == household_id,
                        UserMemory.category == category,
                        UserMemory.is_active == True,  # noqa: E712
                        (UserMemory.expires_at.is_(None)) | (UserMemory.expires_at > utcnow),
                    )
                    row = None
                    if category == "weather":
                        # The weather agent injects BOTH a "Current weather ..." row and a
                        # "Tomorrow's forecast ..." row. The forecast usually carries the
                        # newer updated_at and would win a plain recency sort — making a
                        # "how's today looking" reply describe TOMORROW. Prefer the
                        # current-conditions row when one exists.
                        row = (
                            base.filter(UserMemory.content.ilike("current weather%"))
                            .order_by(UserMemory.updated_at.desc())
                            .first()
                        )
                    if row is None:
                        row = base.order_by(UserMemory.updated_at.desc()).first()
                    if row and row.content and row.content.strip():
                        content = row.content.strip()
                        lines.append(content if content.lower().startswith(label.lower())
                                     else f"{label}: {content}")

                # Signal Bus: append every live household Signal (presence, device
                # state, external producers) as its summary line. Rides this
                # trailing ambient snapshot — never the cached prefix.
                sig_rows = db.query(Signal).filter(
                    Signal.household_id == household_id,
                    Signal.is_active == True,  # noqa: E712
                    (Signal.expires_at.is_(None)) | (Signal.expires_at > utcnow),
                ).all()
                sig_text = render_signal_block(sig_rows)
                if sig_text:
                    lines.append(sig_text)
            finally:
                db.close()

            return "\n".join(lines)
        except Exception as e:  # noqa: BLE001 — fail-open, never break warmup
            logger.warning("Failed to assemble ambient bundle: %s", e)
            return ""

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

        The false-wake gating instruction is appended here so it lives in
        one place and applies uniformly across providers — placing it at
        the END of the system prompt also gives it strong recency right
        before the user message lands.

        Returns:
            System prompt string.
        """
        if self.prompt_provider is not None:
            base = self.prompt_provider.build_system_prompt(
                node_context, timezone, tools, available_command_flags
            )
        elif hasattr(self.model, "_build_system_prompt"):
            base = self.model._build_system_prompt(  # type: ignore[attr-defined]
                node_context, timezone, tools, available_command_flags
            )
        else:
            base = "You are a helpful voice assistant."
        prompt = f"{base.rstrip()}\n\n{NOT_FOR_ME_INSTRUCTION}\n\n{EXCHANGE_COMPLETE_INSTRUCTION}\n"

        # Reinforce the household VOICE at the very end for recency. The top-of-
        # prompt <personality> block is buried under 90%+ terse tool-calling +
        # not-for-me text, so a small model loses the voice by generation time.
        # This restates it as the last instruction before the user turn (stays in
        # the cached prefix — per-household constant; byte-identical when unset).
        if node_context is not None:
            from app.core.prompt_providers.shared.core_rules import build_personality_reminder
            _reminder = build_personality_reminder(node_context.get("household_persona", ""))
            if _reminder:
                prompt = f"{prompt}\n{_reminder}\n"

        # Characterization tail (Slice 3) — Jarvis's evolving "view of the person",
        # appended LAST for recency. Gated per-household (default OFF). When enabled
        # we stash the byte-stable base (everything above) so the per-turn
        # _apply_characterization_swap can reconcile the tail for the confirmed
        # speaker without disturbing the cached prefix on a correct prediction.
        # When disabled the prompt is byte-identical to the pre-characterization
        # path and nothing is stashed.
        if node_context is not None and self._get_characterization_injection_enabled(node_context):
            node_context["_system_prompt_base"] = prompt
            _char_section = build_characterization_section(
                node_context.get("characterization", "")
            )
            if _char_section:
                prompt = f"{prompt}\n{_char_section}"
        return prompt

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

    @staticmethod
    def _get_web_search_enabled(node_context: Dict[str, Any] | None) -> bool:
        """Whether web search is enabled for the current household.

        Gates the outbound-web tools (quick_search live lookups and
        deep_research). Default OFF.

        FAILS CLOSED — deliberately the opposite of ``_get_memory_settings``
        (which fails open). Web search performs outbound internet requests, so
        a settings outage, a missing key, or a typo must never silently enable
        egress for a household that opted out. Any error → disabled.
        """
        try:
            from app.services.settings_service import get_settings_service

            settings = get_settings_service()
            household_id = node_context.get("household_id") if node_context else None

            kwargs: Dict[str, Any] = {}
            if household_id:
                kwargs["household_id"] = str(household_id)

            enabled = settings.get("web_search.enabled", **kwargs)

            # Settings returns the raw value — may be string "true"/"false" or bool
            if isinstance(enabled, bool):
                return enabled
            if isinstance(enabled, str):
                return enabled.lower() in ("true", "1", "yes")
            return bool(enabled)
        except Exception as e:
            logger.warning(
                f"⚠️ Failed to check web_search setting, defaulting to DISABLED: {e}"
            )
            return False

    @staticmethod
    def _get_household_persona(node_context: Dict[str, Any] | None) -> str:
        """Resolve the household speaking-voice persona for the cached prompt.

        Returns the household's persona text, or the warm default when no override
        row exists (the SettingDefinition default flows through). An empty stored
        value is HONORED — a household that cleared the box gets no
        ``<personality>`` block, dropping back to the flat identity line.

        Unlike ``_get_web_search_enabled`` this FAILS SAFE, not closed: the
        persona is voice/tone only, with no egress or safety consequence, so on a
        settings error we fall back to the warm ``DEFAULT_PERSONA`` rather than
        silently stripping Jarvis down to the flat "function calling assistant".
        """
        from app.services.persona_presets import DEFAULT_PERSONA

        try:
            from app.services.settings_service import get_settings_service

            settings = get_settings_service()
            household_id = node_context.get("household_id") if node_context else None

            kwargs: Dict[str, Any] = {}
            if household_id:
                kwargs["household_id"] = str(household_id)

            value = settings.get("persona.household_prompt", **kwargs)
            if value is None:
                return ""
            return str(value).strip()
        except Exception as e:
            logger.warning(
                f"⚠️ Failed to load persona.household_prompt, using default voice: {e}"
            )
            return DEFAULT_PERSONA

    @staticmethod
    def _get_characterization_injection_enabled(node_context: Dict[str, Any] | None) -> bool:
        """Whether the synthesized characterization may be injected into the prompt.

        Gates ONLY the Slice-3 prompt tail (the ``<person_view>`` block). Background
        synthesis is governed separately by ``characterization.synthesis_enabled``,
        so a household can build a characterization for inspection long before it
        ever touches a live prompt.

        Default OFF. Fails CLOSED — a settings outage, missing key, or typo leaves
        the prompt byte-identical to the no-characterization path (no surprise
        injection, no cache churn), matching ``_get_web_search_enabled``.
        """
        try:
            from app.services.settings_service import get_settings_service

            settings = get_settings_service()
            household_id = node_context.get("household_id") if node_context else None

            kwargs: Dict[str, Any] = {}
            if household_id:
                kwargs["household_id"] = str(household_id)

            enabled = settings.get("characterization.injection_enabled", **kwargs)

            if isinstance(enabled, bool):
                return enabled
            if isinstance(enabled, str):
                return enabled.lower() in ("true", "1", "yes")
            return bool(enabled)
        except Exception as e:
            logger.warning(
                f"⚠️ Failed to check characterization.injection_enabled, "
                f"defaulting to DISABLED: {e}"
            )
            return False

    async def _build_turn_speaker_message(
        self,
        conversation_id: str,
        speaker_user_id: int | None,
        node_context: Dict[str, Any] | None,
    ) -> Optional[str]:
        """Build the per-turn speaker block (name + memories) injected as a
        trailing system message AFTER the cached prefix.

        Replaces the old speaker-mismatch rebuild. The warmup prefix is now
        speaker-agnostic, so instead of rewriting messages[0] (which clobbered
        the prefix AND the cached timezone), we resolve the actual turn speaker
        and emit a small trailing block. ``node_context`` is the LIVE cached
        dict, so updating it here also keeps transcript attribution correct for
        a speaker that differs from the warmup guess. Returns None when there
        is nothing speaker-specific to say (unknown speaker / no memories).
        """
        from app.core.prompt_providers.shared.core_rules import build_speaker_block

        ctx = node_context or {}
        warmup_speaker = ctx.get("speaker_user_id")
        effective = speaker_user_id if speaker_user_id is not None else warmup_speaker

        # Turn speaker differs from the warmup guess → re-resolve name +
        # memories for the real speaker and update the cached node_context.
        # No prompt/cache rebuild (that was the bug).
        if effective is not None and effective != warmup_speaker:
            await self._resolve_speaker_into_context(effective, ctx)

        name = ctx.get("speaker_name") or "default"
        memories = ctx.get("user_memories", "") or ""
        block = build_speaker_block(name, memories)

        # Observability (B7): make the identity decision measurable. `source` tells us
        # whether the memory context is grounded in THIS turn's confident STT id or a
        # fallback (session warmup / 30s stickiness) — the fallback is the cross-user
        # misattribution vector. One line/turn lets us quantify how often it fires
        # before we gate writes on it.
        decision_source = (
            "stt" if speaker_user_id is not None
            else "session" if warmup_speaker is not None
            else "none"
        )
        logger.info(
            "🪪 identity-decision turn_speaker=%s warmup_speaker=%s effective=%s "
            "source=%s has_memories=%s",
            speaker_user_id, warmup_speaker, effective, decision_source, bool(memories),
        )

        logger.info(
            "[TURN cid=%s] [SPEAKER] resolved_user_id=%s source=%s name=%r "
            "memories_chars=%d block=%s",
            conversation_id,
            effective,
            "stt" if speaker_user_id is not None else "warmup-cached",
            name,
            len(memories),
            "yes" if block else "none",
        )

        # Signal Bus (Phase 2): a confident STT speaker at a node is a free
        # presence observation → write a presence.seen Signal. ONLY the confident
        # STT id (speaker_user_id), never the sticky session fallback. Gated on
        # the ambient opt-in (same switch that renders it), best-effort — never
        # blocks or breaks the turn.
        if speaker_user_id is not None:
            _household_id = ctx.get("household_id")
            if _household_id and self._get_ambient_context_enabled(str(_household_id)):
                try:
                    from app.db import get_session_local
                    from app.services.signal_service import record_voice_presence

                    _pdb = get_session_local()()
                    try:
                        record_voice_presence(
                            _pdb,
                            household_id=str(_household_id),
                            user_id=speaker_user_id,
                            node_id=ctx.get("node_id"),
                            speaker_name=ctx.get("speaker_name"),
                        )
                    finally:
                        _pdb.close()
                except Exception as e:  # noqa: BLE001 — presence is best-effort
                    logger.debug("voice presence emit skipped: %s", e)

        return block or None

    async def _resolve_speaker_into_context(
        self,
        speaker_user_id: int,
        node_context: Dict[str, Any],
    ) -> None:
        """Resolve display name + load memories for ``speaker_user_id`` into the
        (live cached) ``node_context``. Called when the turn speaker differs
        from the warmup guess. Does NOT rebuild the cached system prompt — that
        rewrite was the source of the prefix/timezone clobber.
        """
        node_context["speaker_user_id"] = speaker_user_id

        # Resolve display name (independent of the memory setting).
        try:
            from app.core import service_config
            from app.core.utils.speaker_resolver import resolve_speaker_name
            auth_url = service_config.get_auth_url()
            speaker_name = await resolve_speaker_name(auth_url, speaker_user_id)
            if speaker_name:
                node_context["speaker_name"] = speaker_name
        except Exception as e:
            logger.warning("⚠️ Failed to resolve speaker name for turn: %s", e)

        # Load memories for the new speaker — only when memory is enabled.
        # Always overwrite user_memories so we never carry the previous
        # speaker's memories into this turn's block.
        _memory_enabled, _ = self._get_memory_settings(node_context)
        household_id = node_context.get("household_id")
        if not _memory_enabled or not household_id:
            node_context["user_memories"] = ""
            return
        try:
            from app.db import get_session_local
            from app.services.memory_service import MemoryService

            pinned_max_chars = 500
            try:
                from app.services.settings_service import get_settings_service
                val = get_settings_service().get("memory.pinned_max_chars")
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
                    "🔄 Loaded %d chars of memories for turn speaker %s",
                    len(memories_text or ""), speaker_user_id,
                )
            finally:
                db.close()
        except Exception as e:
            logger.warning(
                "⚠️ Failed to load memories for turn speaker %s: %s",
                speaker_user_id, e,
            )
            node_context["user_memories"] = ""


