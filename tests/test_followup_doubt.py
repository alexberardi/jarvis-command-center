"""Verdict propagation onto follow-up turns + the doubted-conversation
round cap — the 2026-08-15 kitchen-runaway defense.

The failure: a false wake gets ANSWERED (ambient speech that survives the
wake-turn hints — a bare "Okay." from a recognized speaker in a quiet room
scored 0.988) → the node opens its follow-up window → the family's
continuing conversation is captured as follow_up turns → CC answers those
too, because follow-up turns got the engaged-conversation posture and have
no wake clip to re-verify → the window re-opens after each answer. The
#111 sentinel-is-terminal fix only ends the loop when the model actually
EMITS the sentinel; in runaway mode it never does.

The defense, fail-open throughout (a REAL engaged user must not fight
suppression):

* Only an ``unverified`` WAKE verdict marks the conversation as doubted;
  ``verified`` / ``clip_unreliable`` / absent keep follow-ups
  byte-identical to today.
* Doubted follow-ups get a caution hint (suspected-misfire origin), not
  the engaged posture — but device-command / mid-music music-control
  shapes keep the directed posture (guard stays senior).
* After ``voice.followup_doubt_max_rounds`` answered rounds a doubted
  conversation's hint gains a wrap-up LEAN (answer briefly + close, or
  sentinel) — never a hard cut. Verified conversations are exempt.
* The /think sentinel rescue stays available on doubted follow-ups, and a
  sentinel on one still terminates cleanly (silent not_for_me).
"""

import asyncio
import inspect
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.conversation_cache import ConversationCache
from app.core.conversation_handler import ConversationHandler
from app.core.turn_context import (
    DOUBTED_WAKE_VERDICT,
    build_turn_hint,
    should_double_check_sentinel,
)
from app.core.wake_verification import (
    FOLLOWUP_DOUBT_MAX_ROUNDS_DEFAULT,
    get_followup_doubt_max_rounds,
)


def _followup_hint(**kwargs) -> str:
    return build_turn_hint("follow_up", follow_up_iteration=1, **kwargs)


class TestDoubtedFollowUpHint:
    """Rule 1: verdict propagation selects the caution posture."""

    def test_unverified_verdict_selects_caution_posture(self):
        hint = _followup_hint(conversation_wake_verdict="unverified")
        assert hint is not None
        assert hint.startswith("[turn context:")
        assert hint.endswith("]")
        assert "misfire" in hint.lower()
        assert "<not_for_me/>" in hint
        # The load-bearing framing: if it reads like people talking to
        # each other, sentinel.
        assert "talking to each other" in hint.lower()

    def test_caution_posture_is_not_the_engaged_posture(self):
        doubted = _followup_hint(conversation_wake_verdict="unverified")
        normal = _followup_hint()
        assert doubted != normal
        # The engaged framing ("clearly continues your exchange") is gone.
        assert "continues your exchange" not in doubted

    def test_caution_posture_keeps_answer_and_tool_path_open(self):
        # Fail-open: a real engaged user must still be answered — and a
        # tool-needing continuation must still be allowed to run the tool.
        hint = _followup_hint(conversation_wake_verdict="unverified")
        assert "answer" in hint.lower()
        assert "run the tool" in hint.lower()

    def test_verified_verdict_is_byte_identical_to_today(self):
        assert _followup_hint(conversation_wake_verdict="verified") == _followup_hint()

    def test_clip_unreliable_is_no_signal_not_doubt(self):
        assert (
            _followup_hint(conversation_wake_verdict="clip_unreliable")
            == _followup_hint()
        )

    def test_absent_verdict_is_byte_identical_to_today(self):
        assert _followup_hint(conversation_wake_verdict=None) == _followup_hint()

    def test_doubt_ignored_on_wake_turns(self):
        # The verdict propagation is follow-up-only; wake turns have their
        # own per-turn verdict (wake_verified).
        assert build_turn_hint(
            "wake", wake_confidence=0.95, conversation_wake_verdict="unverified"
        ) == build_turn_hint("wake", wake_confidence=0.95)

    def test_doubt_ignored_on_chat_turns(self):
        assert build_turn_hint(
            "chat", conversation_wake_verdict="unverified"
        ) == build_turn_hint("chat")

    def test_caution_hint_still_carries_iteration_number(self):
        hint = build_turn_hint(
            "follow_up",
            follow_up_iteration=2,
            conversation_wake_verdict="unverified",
        )
        assert "iteration 2" in hint

    def test_doubted_sentinel_value_constant(self):
        assert DOUBTED_WAKE_VERDICT == "unverified"


class TestGuardSeniority:
    """Music-control / imperative-command shapes keep the directed posture
    even in a doubted conversation — acoustic-side doubt about the ORIGINAL
    wake must never talk the model out of an imperative command now."""

    def test_device_command_keeps_directed_posture(self):
        hint = _followup_hint(
            conversation_wake_verdict="unverified",
            transcript="Turn on the living room lights.",
        )
        assert "misfire" not in hint.lower()
        assert hint == _followup_hint(
            transcript="Turn on the living room lights."
        )

    def test_music_control_during_self_playback_keeps_directed_posture(self):
        hint = _followup_hint(
            conversation_wake_verdict="unverified",
            transcript="Skip this song.",
            self_playback=True,
            self_playback_kind="music",
        )
        assert "misfire" not in hint.lower()

    def test_music_control_without_media_keeps_caution(self):
        # Same seniority shape as the wake hint: the music-control guard
        # only exists during self-playback.
        hint = _followup_hint(
            conversation_wake_verdict="unverified", transcript="Pause."
        )
        assert "misfire" in hint.lower()

    def test_conversational_fragment_keeps_caution(self):
        hint = _followup_hint(
            conversation_wake_verdict="unverified",
            transcript="Yeah, and then she said we should just drive up.",
        )
        assert "misfire" in hint.lower()


class TestDoubtRoundCap:
    """Rule 2: after N answered rounds a doubted conversation leans toward
    closing — a lean, never a hard cut. Verified conversations are exempt."""

    def test_under_cap_no_wrap_up(self):
        hint = _followup_hint(
            conversation_wake_verdict="unverified",
            doubt_round=1,
            doubt_max_rounds=2,
        )
        assert "wrap up" not in hint.lower()
        assert "<exchange_complete/>" not in hint

    def test_at_cap_gains_wrap_up_lean(self):
        hint = _followup_hint(
            conversation_wake_verdict="unverified",
            doubt_round=2,
            doubt_max_rounds=2,
        )
        assert "wrap up" in hint.lower()
        # Both exits offered: brief answer + close, or sentinel.
        assert "<exchange_complete/>" in hint
        assert "<not_for_me/>" in hint
        assert "briefly" in hint.lower()

    def test_over_cap_gains_wrap_up_lean(self):
        hint = _followup_hint(
            conversation_wake_verdict="unverified",
            doubt_round=5,
            doubt_max_rounds=2,
        )
        assert "wrap up" in hint.lower()

    def test_wrap_up_is_a_lean_not_a_hard_cut(self):
        # The hint must still allow answering — never instruct silence
        # unconditionally.
        hint = _followup_hint(
            conversation_wake_verdict="unverified",
            doubt_round=3,
            doubt_max_rounds=2,
        )
        assert "answer briefly" in hint.lower()

    def test_verified_conversation_exempt_from_cap(self):
        # No cap for verified wakes — an engaged user can go as many
        # rounds as they like.
        assert _followup_hint(
            conversation_wake_verdict="verified",
            doubt_round=50,
            doubt_max_rounds=2,
        ) == _followup_hint()

    def test_missing_round_info_never_escalates(self):
        # Doubted but no round data (e.g. cache miss) → caution hint
        # without the wrap-up lean. Missing signal must not escalate.
        hint = _followup_hint(conversation_wake_verdict="unverified")
        assert "wrap up" not in hint.lower()
        hint = _followup_hint(
            conversation_wake_verdict="unverified", doubt_round=5
        )
        assert "wrap up" not in hint.lower()

    def test_guard_seniority_beats_the_cap(self):
        # Even past the cap, an imperative device command keeps the
        # directed posture entirely.
        hint = _followup_hint(
            conversation_wake_verdict="unverified",
            doubt_round=5,
            doubt_max_rounds=2,
            transcript="Turn off the kitchen lights.",
        )
        assert "wrap up" not in hint.lower()
        assert "misfire" not in hint.lower()


class TestDoubtedFollowUpKeepsSentinelRescue:
    """Rule 3: the /think rescue stays available on doubted follow-ups —
    the gate never consults the verdict (fail-open)."""

    def test_gate_signature_has_no_verdict_params(self):
        params = inspect.signature(should_double_check_sentinel).parameters
        assert "conversation_wake_verdict" not in params
        assert "wake_verified" not in params
        assert "doubt_round" not in params

    def test_early_follow_up_qualifies_regardless_of_doubt(self):
        assert should_double_check_sentinel("follow_up", None, 1, None) is True


class TestResolveFollowupDoubtIntoContext:
    """Handler helper: stored WAKE verdict → turn_context, follow-ups only."""

    def _run(self, verdict, turn_context, answered_rounds=0, max_rounds=2):
        handler = ConversationHandler(
            model=MagicMock(), llm_client=MagicMock(), prompt_provider=None
        )
        with patch(
            "app.core.conversation_handler.conversation_cache"
        ) as mock_cache, patch(
            "app.core.conversation_handler.get_followup_doubt_max_rounds",
            return_value=max_rounds,
        ) as mock_max:
            mock_cache.get_wake_verification.return_value = verdict
            mock_cache.get_answered_rounds.return_value = answered_rounds
            mock_cache.get_node_context.return_value = {
                "household_id": "hh-1",
                "node_id": "node-1",
            }
            handler._resolve_followup_doubt_into_context("conv-1", turn_context)
        return turn_context, mock_max

    def test_unverified_verdict_marks_conversation_doubted(self):
        ctx, _ = self._run(
            {"verdict": "unverified", "verified": False},
            {"source": "follow_up", "follow_up_iteration": 1},
            answered_rounds=2,
            max_rounds=2,
        )
        assert ctx["conversation_wake_verdict"] == "unverified"
        assert ctx["doubt_round"] == 2
        assert ctx["doubt_max_rounds"] == 2

    def test_verified_verdict_is_a_no_op(self):
        ctx, _ = self._run(
            {"verdict": "verified", "verified": True}, {"source": "follow_up"}
        )
        assert "conversation_wake_verdict" not in ctx
        assert "doubt_round" not in ctx

    def test_clip_unreliable_is_a_no_op(self):
        ctx, _ = self._run(
            {"verdict": "clip_unreliable", "verified": False},
            {"source": "follow_up"},
        )
        assert "conversation_wake_verdict" not in ctx

    def test_absent_verdict_is_a_no_op(self):
        # Old nodes / verification off / verdict expired: byte-identical
        # behavior to today.
        ctx, _ = self._run(None, {"source": "follow_up"})
        assert "conversation_wake_verdict" not in ctx

    def test_wake_turns_are_untouched(self):
        ctx, _ = self._run(
            {"verdict": "unverified", "verified": False}, {"source": "wake"}
        )
        assert "conversation_wake_verdict" not in ctx

    def test_old_node_without_turn_source_is_untouched(self):
        ctx, _ = self._run(
            {"verdict": "unverified", "verified": False}, {}
        )
        assert "conversation_wake_verdict" not in ctx
        # And a None turn_context must not crash.
        handler = ConversationHandler(
            model=MagicMock(), llm_client=MagicMock(), prompt_provider=None
        )
        with patch("app.core.conversation_handler.conversation_cache"):
            handler._resolve_followup_doubt_into_context("conv-1", None)

    def test_max_rounds_scoped_to_household_and_node(self):
        _, mock_max = self._run(
            {"verdict": "unverified", "verified": False},
            {"source": "follow_up"},
        )
        mock_max.assert_called_once_with("hh-1", "node-1")


class TestAnsweredRoundCounter:
    """ConversationCache round accounting."""

    def test_increment_and_get(self):
        cache = ConversationCache(ttl_minutes=10)
        cache.set("conv-1", [], [], tools=[])
        assert cache.get_answered_rounds("conv-1") == 0
        assert cache.increment_answered_rounds("conv-1") == 1
        assert cache.increment_answered_rounds("conv-1") == 2
        assert cache.get_answered_rounds("conv-1") == 2

    def test_missing_conversation_counts_zero(self):
        cache = ConversationCache(ttl_minutes=10)
        assert cache.increment_answered_rounds("nope") == 0
        assert cache.get_answered_rounds("nope") == 0


def _blocking_path_mocks(mock_cache, verdict=None, rounds=0):
    """Common cache stubbing for full blocking-path runs."""
    mock_cache.get_messages.return_value = [{"role": "system", "content": "x"}]
    mock_cache.get_tools.return_value = []
    mock_cache.get_available_commands.return_value = []
    mock_cache.get_node_context.return_value = {"node_id": "node-1"}
    mock_cache.get_referenced_items.return_value = []
    mock_cache.get_wake_verification.return_value = verdict
    mock_cache.get_answered_rounds.return_value = rounds
    mock_cache.increment_answered_rounds.return_value = rounds + 1


def _run_blocking(voice_command, turn_context, engine_result, verdict=None,
                  rounds=0, turn_hint_mock=None):
    handler = ConversationHandler(
        model=MagicMock(), llm_client=MagicMock(), prompt_provider=None
    )
    engine = MagicMock()
    engine.execute = AsyncMock(return_value=dict(engine_result))
    patches = [
        patch("app.core.conversation_handler.conversation_cache"),
        patch("app.core.conversation_handler.ToolExecutionEngine", return_value=engine),
        patch(
            "app.core.conversation_handler.get_followup_doubt_max_rounds",
            return_value=2,
        ),
        patch.object(
            handler, "_resolve_wake_verified_into_context",
            new=AsyncMock(return_value=False),
        ),
        patch.object(handler, "_apply_tool_filtering", return_value=[]),
        patch.object(handler, "_apply_tool_routing_with_cache", return_value=None),
        patch.object(handler, "_sync_include_thinking"),
        patch.object(
            handler, "_build_turn_speaker_message", new=AsyncMock(return_value=None)
        ),
        patch.object(handler, "_get_advanced_context_enabled", return_value=False),
        patch.object(handler, "_get_max_history_turns", return_value=10),
    ]
    if turn_hint_mock is not None:
        patches.append(
            patch("app.core.conversation_handler.build_turn_hint", turn_hint_mock)
        )
    with patches[0] as mock_cache:
        _blocking_path_mocks(mock_cache, verdict, rounds)
        with ExitStack() as stack:
            for p in patches[1:]:
                stack.enter_context(p)
            result = asyncio.run(
                handler.process_voice_command_with_tools(
                    voice_command=voice_command,
                    conversation_id="conv-1",
                    turn_context=turn_context,
                )
            )
    return result, mock_cache


class TestBlockingPathWiring:
    """End-to-end through process_voice_command_with_tools."""

    def test_doubted_followup_feeds_the_turn_hint(self):
        turn_hint = MagicMock(return_value=None)
        _run_blocking(
            "no I told him we'd go Saturday",
            {"source": "follow_up", "follow_up_iteration": 1},
            {"stop_reason": "complete", "assistant_message": "ok"},
            verdict={"verdict": "unverified", "verified": False},
            rounds=1,
            turn_hint_mock=turn_hint,
        )
        kwargs = turn_hint.call_args.kwargs
        assert kwargs["conversation_wake_verdict"] == "unverified"
        assert kwargs["doubt_round"] == 1
        assert kwargs["doubt_max_rounds"] == 2

    def test_missing_verdict_passes_no_doubt(self):
        # Rule 5: old nodes / missing verdict → the hint builder sees
        # exactly what it saw before this change.
        turn_hint = MagicMock(return_value=None)
        _run_blocking(
            "what about tomorrow",
            {"source": "follow_up", "follow_up_iteration": 1},
            {"stop_reason": "complete", "assistant_message": "ok"},
            verdict=None,
            turn_hint_mock=turn_hint,
        )
        kwargs = turn_hint.call_args.kwargs
        assert kwargs["conversation_wake_verdict"] is None
        assert kwargs["doubt_round"] is None
        assert kwargs["doubt_max_rounds"] is None

    def test_answered_round_is_counted(self):
        _, mock_cache = _run_blocking(
            "what about tomorrow",
            {"source": "follow_up", "follow_up_iteration": 1},
            {"stop_reason": "complete", "assistant_message": "ok"},
        )
        mock_cache.increment_answered_rounds.assert_called_once_with("conv-1")

    def test_sentinel_round_is_not_counted(self):
        result, mock_cache = _run_blocking(
            "yeah and then she said",
            {"source": "follow_up", "follow_up_iteration": 1},
            {"stop_reason": "complete", "assistant_message": "<not_for_me/>"},
            verdict={"verdict": "unverified", "verified": False},
        )
        assert result["stop_reason"] == "not_for_me"
        mock_cache.increment_answered_rounds.assert_not_called()

    def test_sentinel_on_doubted_followup_terminates_cleanly(self):
        # Rule 3 (CC side): the sentinel comes back unsuppressed as a
        # silent not_for_me — the node's #111 terminal handling does the
        # rest. Nothing about the doubt machinery may rewrite it.
        result, _ = _run_blocking(
            "so anyway he took the 101",
            {"source": "follow_up", "follow_up_iteration": 2},
            {"stop_reason": "complete", "assistant_message": "<not_for_me/>"},
            verdict={"verdict": "unverified", "verified": False},
            rounds=2,
        )
        assert result == {"stop_reason": "not_for_me", "assistant_message": ""}


class TestMaxRoundsSetting:
    """voice.followup_doubt_max_rounds — declared, defaulted, fail-safe."""

    def test_setting_is_declared_with_default(self):
        from app.services.settings_definitions import SETTINGS_DEFINITIONS

        matches = [
            d for d in SETTINGS_DEFINITIONS
            if d.key == "voice.followup_doubt_max_rounds"
        ]
        assert len(matches) == 1
        assert matches[0].default == FOLLOWUP_DOUBT_MAX_ROUNDS_DEFAULT
        assert matches[0].value_type == "int"

    def _get_with(self, raw):
        service = MagicMock()
        service.get.return_value = raw
        with patch(
            "app.services.settings_service.get_settings_service",
            return_value=service,
        ):
            return get_followup_doubt_max_rounds("hh-1", "node-1")

    def test_reads_the_setting(self):
        assert self._get_with(3) == 3
        assert self._get_with("4") == 4

    def test_garbage_falls_back_to_default(self):
        assert self._get_with("lots") == FOLLOWUP_DOUBT_MAX_ROUNDS_DEFAULT
        assert self._get_with(None) == FOLLOWUP_DOUBT_MAX_ROUNDS_DEFAULT

    def test_non_positive_falls_back_to_default(self):
        assert self._get_with(0) == FOLLOWUP_DOUBT_MAX_ROUNDS_DEFAULT
        assert self._get_with(-3) == FOLLOWUP_DOUBT_MAX_ROUNDS_DEFAULT

    def test_unreachable_settings_service_fails_safe(self):
        with patch(
            "app.services.settings_service.get_settings_service",
            side_effect=RuntimeError("down"),
        ):
            assert (
                get_followup_doubt_max_rounds()
                == FOLLOWUP_DOUBT_MAX_ROUNDS_DEFAULT
            )
