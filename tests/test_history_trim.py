"""
Tests for conversation-history trimming (conversation.max_turns).

The sliding-window trim (app.core.conversation_cache.trim_history_to_max_turns)
keeps the system prefix plus only the most recent N user/assistant exchanges.
Turns are dropped atomically — an assistant tool_call, its role="tool" results,
and text-mode tool-result user injections ride with their turn — and always
from the FRONT (oldest first) so the retained tail stays byte-stable for the
KV prefix cache.
"""
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from app.core.conversation_cache import trim_history_to_max_turns


def _system() -> Dict[str, Any]:
    return {"role": "system", "content": "SYSTEM PROMPT"}


def _simple_turn(i: int) -> List[Dict[str, Any]]:
    """A plain user/assistant exchange (no tools)."""
    return [
        {"role": "user", "content": f"question {i}"},
        {"role": "assistant", "content": f"answer {i}"},
    ]


def _native_tool_turn(i: int) -> List[Dict[str, Any]]:
    """A native-tools exchange: assistant tool_call + role='tool' results."""
    return [
        {"role": "user", "content": f"do thing {i}"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {"name": "control_device", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": f"call_{i}", "content": '{"success": true}'},
        {"role": "assistant", "content": f"Done with thing {i}."},
    ]


def _text_mode_tool_turn(i: int) -> List[Dict[str, Any]]:
    """A text-mode exchange: tool results injected as a follow-up user message."""
    return [
        {"role": "user", "content": f"what's the weather {i}"},
        {
            "role": "assistant",
            "content": '{"tool": "get_weather"}',
            "tool_calls": [
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{}"},
                }
            ],
        },
        {"role": "user", "content": "Here are the tool results. 72F and sunny."},
        {"role": "assistant", "content": f"It's 72 and sunny ({i})."},
    ]


class TestTrimKeepsSystemPlusLastN:
    """Trim keeps the system message plus only the most recent N pairs."""

    def test_keeps_system_message_plus_last_n_pairs(self):
        messages: List[Dict[str, Any]] = [_system()]
        for i in range(15):
            messages.extend(_simple_turn(i))
        expected_tail = [dict(m) for m in messages[-20:]]  # last 10 exchanges

        dropped = trim_history_to_max_turns(messages, 10)

        assert dropped == 10  # 5 turns x 2 messages
        assert messages[0] == _system()
        assert messages[1:] == expected_tail
        assert messages[1] == {"role": "user", "content": "question 5"}
        assert len(messages) == 1 + 10 * 2

    def test_oldest_turns_dropped_first_and_tail_untouched(self):
        """Front-drop only: retained messages are the SAME objects in the same
        order (byte-stable tail — the KV prefix-cache requirement)."""
        messages: List[Dict[str, Any]] = [_system()]
        for i in range(6):
            messages.extend(_simple_turn(i))
        originals = messages[:]  # capture object references

        trim_history_to_max_turns(messages, 2)

        assert messages[0] is originals[0]
        kept_tail = originals[-4:]  # last 2 exchanges
        assert len(messages) == 1 + len(kept_tail)
        assert all(kept is orig for kept, orig in zip(messages[1:], kept_tail))


class TestTrimAtomicToolTurns:
    """Tool-call exchanges are kept or dropped atomically."""

    def test_native_tool_turn_dropped_atomically(self):
        messages: List[Dict[str, Any]] = [_system()]
        messages.extend(_native_tool_turn(0))  # oldest — should drop whole
        for i in range(3):
            messages.extend(_simple_turn(i))

        trim_history_to_max_turns(messages, 3)

        roles = [m["role"] for m in messages]
        assert "tool" not in roles
        assert not any(m.get("tool_calls") for m in messages)
        assert messages[1] == {"role": "user", "content": "question 0"}

    def test_native_tool_turn_kept_atomically(self):
        messages: List[Dict[str, Any]] = [_system()]
        for i in range(3):
            messages.extend(_simple_turn(i))
        messages.extend(_native_tool_turn(9))  # newest — must survive intact

        trim_history_to_max_turns(messages, 2)

        # The retained window is [simple turn 2, native tool turn 9] — with the
        # tool_call immediately followed by its role="tool" result.
        assert messages[1] == {"role": "user", "content": "question 2"}
        tool_call_idx = next(
            i for i, m in enumerate(messages) if m.get("tool_calls")
        )
        assert messages[tool_call_idx + 1]["role"] == "tool"
        assert (
            messages[tool_call_idx + 1]["tool_call_id"]
            == messages[tool_call_idx]["tool_calls"][0]["id"]
        )

    def test_trim_never_orphans_tool_results(self):
        """After a trim across all-tool-turn history, the window starts at a
        user message — never at a role='tool' result or mid-exchange."""
        messages: List[Dict[str, Any]] = [_system()]
        for i in range(5):
            messages.extend(_native_tool_turn(i))

        trim_history_to_max_turns(messages, 2)

        assert messages[1]["role"] == "user"
        assert len([m for m in messages if m["role"] == "tool"]) == 2
        # Every tool result still directly follows its assistant tool_call.
        for i, m in enumerate(messages):
            if m["role"] == "tool":
                assert messages[i - 1].get("tool_calls")

    def test_text_mode_injection_counts_as_one_turn(self):
        """A text-mode tool-result user injection attaches to its exchange —
        it neither starts a new turn nor gets split from its utterance."""
        messages: List[Dict[str, Any]] = [_system()]
        messages.extend(_text_mode_tool_turn(0))
        for i in range(2):
            messages.extend(_simple_turn(i))

        # 3 logical turns total — a window of 3 keeps everything.
        assert trim_history_to_max_turns(messages, 3) == 0

        # A window of 2 drops the text-mode exchange WHOLE (all 4 messages).
        trim_history_to_max_turns(messages, 2)
        assert messages[1] == {"role": "user", "content": "question 0"}
        assert not any("tool results" in str(m.get("content", "")) for m in messages)


class TestTrimNoOpCases:
    """No trim when at/under the limit or when trimming is disabled."""

    def test_no_trim_when_under_limit(self):
        messages: List[Dict[str, Any]] = [_system()]
        for i in range(3):
            messages.extend(_simple_turn(i))
        before = [dict(m) for m in messages]

        assert trim_history_to_max_turns(messages, 10) == 0
        assert messages == before

    def test_no_trim_when_exactly_at_limit(self):
        messages: List[Dict[str, Any]] = [_system()]
        for i in range(10):
            messages.extend(_simple_turn(i))

        assert trim_history_to_max_turns(messages, 10) == 0
        assert len(messages) == 1 + 10 * 2

    def test_nonpositive_max_turns_disables_trimming(self):
        messages: List[Dict[str, Any]] = [_system()]
        for i in range(30):
            messages.extend(_simple_turn(i))

        assert trim_history_to_max_turns(messages, 0) == 0
        assert trim_history_to_max_turns(messages, -1) == 0
        assert len(messages) == 1 + 30 * 2

    def test_system_only_history_untouched(self):
        messages: List[Dict[str, Any]] = [_system()]
        assert trim_history_to_max_turns(messages, 1) == 0
        assert messages == [_system()]


class TestMaxTurnsFromSettings:
    """The window size N comes from the conversation.max_turns setting."""

    def _handler(self):
        from app.core.conversation_handler import ConversationHandler

        return ConversationHandler(model=MagicMock(), llm_client=MagicMock())

    def test_helper_reads_setting_scoped_to_household(self):
        handler = self._handler()
        mock_settings = MagicMock()
        mock_settings.get.return_value = 3

        with patch(
            "app.core.conversation_handler.get_settings_service",
            return_value=mock_settings,
        ), patch("app.core.conversation_handler.conversation_cache") as mock_cache:
            mock_cache.get_node_context.return_value = {"household_id": "hh-1"}

            assert handler._get_max_history_turns("conv-1") == 3
            mock_settings.get.assert_called_once_with(
                "conversation.max_turns", default=10, household_id="hh-1"
            )

    def test_helper_falls_back_to_default_on_settings_error(self):
        handler = self._handler()

        with patch(
            "app.core.conversation_handler.get_settings_service",
            side_effect=RuntimeError("settings down"),
        ), patch("app.core.conversation_handler.conversation_cache") as mock_cache:
            mock_cache.get_node_context.return_value = None

            assert handler._get_max_history_turns("conv-1") == 10

    def test_trim_uses_value_from_settings(self):
        """End-to-end: N from settings drives the window size."""
        handler = self._handler()
        mock_settings = MagicMock()
        mock_settings.get.return_value = 2

        with patch(
            "app.core.conversation_handler.get_settings_service",
            return_value=mock_settings,
        ), patch("app.core.conversation_handler.conversation_cache") as mock_cache:
            mock_cache.get_node_context.return_value = {"household_id": "hh-1"}

            messages: List[Dict[str, Any]] = [_system()]
            for i in range(5):
                messages.extend(_simple_turn(i))

            trim_history_to_max_turns(
                messages, handler._get_max_history_turns("conv-1")
            )

        assert len(messages) == 1 + 2 * 2
        assert messages[1] == {"role": "user", "content": "question 3"}

    def test_definition_default_is_10(self):
        from app.services.settings_definitions import SETTINGS_DEFINITIONS

        definition = next(
            d for d in SETTINGS_DEFINITIONS if d.key == "conversation.max_turns"
        )
        assert definition.default == 10
