"""Wiring for the media-aware wake hints (self-playback context).

The node sends two additive fields with voice turns — ``self_playback``
(its own speaker was playing media when the wake fired) and
``self_playback_kind`` ("music"). They enter through
``VoiceCommandRequest``, ride ``turn_context`` alongside the existing wake
evidence, and feed the direction/turn hints and the wake-verification
clip-plausibility gate. Old nodes never send them → None → behavior
byte-identical to before the fields existed.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.conversation_handler import ConversationHandler
from app.core.direction_hint import is_media_self_playback
from app.request_models.voice_command_request import VoiceCommandRequest


class TestRequestModelDefaults:
    """Old-node compatibility: the new fields are additive and optional."""

    def test_old_node_payload_parses_with_none_defaults(self):
        req = VoiceCommandRequest(
            voice_command="what time is it", conversation_id="conv-1"
        )
        assert req.self_playback is None
        assert req.self_playback_kind is None

    def test_new_node_payload_carries_the_fields(self):
        req = VoiceCommandRequest(
            voice_command="skip this song",
            conversation_id="conv-1",
            self_playback=True,
            self_playback_kind="music",
        )
        assert req.self_playback is True
        assert req.self_playback_kind == "music"


class TestIsMediaSelfPlayback:
    """Single predicate every consumer shares (direction hint, turn hint,
    clip-plausibility gate) — one definition of "the node's own music"."""

    def test_flag_with_music_kind(self):
        assert is_media_self_playback(True, "music") is True

    def test_flag_without_kind_counts_as_music(self):
        # Only "music" exists today; a node build that omits the kind must
        # not silently lose the treatment.
        assert is_media_self_playback(True, None) is True

    def test_future_kind_does_not_inherit(self):
        assert is_media_self_playback(True, "podcast") is False

    def test_absent_or_false_flag(self):
        assert is_media_self_playback(None, None) is False
        assert is_media_self_playback(False, "music") is False


class TestBlockingPathPassthrough:
    """process_voice_command_with_tools hands the self-playback fields from
    turn_context to both hint builders — the persistence contract."""

    @pytest.mark.asyncio
    async def test_hint_builders_receive_self_playback(self):
        handler = ConversationHandler(
            model=MagicMock(), llm_client=MagicMock(), prompt_provider=None
        )
        engine = MagicMock()
        engine.execute = AsyncMock(
            return_value={"stop_reason": "complete", "assistant_message": "ok"}
        )
        direction_hint = MagicMock(return_value=None)
        turn_hint = MagicMock(return_value=None)
        with patch(
            "app.core.conversation_handler.conversation_cache"
        ) as mock_cache, patch(
            "app.core.conversation_handler.ToolExecutionEngine",
            return_value=engine,
        ), patch(
            "app.core.conversation_handler.build_direction_hint", direction_hint
        ), patch(
            "app.core.conversation_handler.build_turn_hint", turn_hint
        ), patch.object(
            handler, "_resolve_wake_verified_into_context",
            new=AsyncMock(return_value=False),
        ), patch.object(
            handler, "_apply_tool_filtering", return_value=[]
        ), patch.object(
            handler, "_apply_tool_routing_with_cache", return_value=None
        ), patch.object(
            handler, "_sync_include_thinking"
        ), patch.object(
            handler, "_build_turn_speaker_message",
            new=AsyncMock(return_value=None),
        ), patch.object(
            handler, "_get_advanced_context_enabled", return_value=False
        ), patch.object(
            handler, "_get_max_history_turns", return_value=10
        ):
            mock_cache.get_messages.return_value = [
                {"role": "system", "content": "x"}
            ]
            mock_cache.get_tools.return_value = []
            mock_cache.get_available_commands.return_value = []
            mock_cache.get_node_context.return_value = {}
            mock_cache.get_referenced_items.return_value = []

            await handler.process_voice_command_with_tools(
                voice_command="skip this song",
                conversation_id="conv-1",
                pre_wake_speech_seconds=0.0,
                turn_context={
                    "source": "wake",
                    "wake_confidence": 0.95,
                    "follow_up_iteration": None,
                    "self_playback": True,
                    "self_playback_kind": "music",
                },
            )

        assert direction_hint.call_args.kwargs["self_playback"] is True
        assert direction_hint.call_args.kwargs["self_playback_kind"] == "music"
        assert turn_hint.call_args.kwargs["self_playback"] is True
        assert turn_hint.call_args.kwargs["self_playback_kind"] == "music"

    @pytest.mark.asyncio
    async def test_old_node_turn_context_passes_none(self):
        handler = ConversationHandler(
            model=MagicMock(), llm_client=MagicMock(), prompt_provider=None
        )
        engine = MagicMock()
        engine.execute = AsyncMock(
            return_value={"stop_reason": "complete", "assistant_message": "ok"}
        )
        direction_hint = MagicMock(return_value=None)
        turn_hint = MagicMock(return_value=None)
        with patch(
            "app.core.conversation_handler.conversation_cache"
        ) as mock_cache, patch(
            "app.core.conversation_handler.ToolExecutionEngine",
            return_value=engine,
        ), patch(
            "app.core.conversation_handler.build_direction_hint", direction_hint
        ), patch(
            "app.core.conversation_handler.build_turn_hint", turn_hint
        ), patch.object(
            handler, "_resolve_wake_verified_into_context",
            new=AsyncMock(return_value=False),
        ), patch.object(
            handler, "_apply_tool_filtering", return_value=[]
        ), patch.object(
            handler, "_apply_tool_routing_with_cache", return_value=None
        ), patch.object(
            handler, "_sync_include_thinking"
        ), patch.object(
            handler, "_build_turn_speaker_message",
            new=AsyncMock(return_value=None),
        ), patch.object(
            handler, "_get_advanced_context_enabled", return_value=False
        ), patch.object(
            handler, "_get_max_history_turns", return_value=10
        ):
            mock_cache.get_messages.return_value = [
                {"role": "system", "content": "x"}
            ]
            mock_cache.get_tools.return_value = []
            mock_cache.get_available_commands.return_value = []
            mock_cache.get_node_context.return_value = {}
            mock_cache.get_referenced_items.return_value = []

            # Old node: turn_context without the new keys at all.
            await handler.process_voice_command_with_tools(
                voice_command="what time is it",
                conversation_id="conv-1",
                pre_wake_speech_seconds=0.0,
                turn_context={"source": "wake", "wake_confidence": 0.95},
            )

        assert direction_hint.call_args.kwargs["self_playback"] is None
        assert direction_hint.call_args.kwargs["self_playback_kind"] is None
        assert turn_hint.call_args.kwargs["self_playback"] is None
        assert turn_hint.call_args.kwargs["self_playback_kind"] is None
