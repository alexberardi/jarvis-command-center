"""Tests for the referenceable-items (act_on_items) memory in the command-center.

Covers:
- render_referenced_items_block (core_rules): the per-turn RECENTLY SHOWN block,
- ConversationCache.set/get_referenced_items: the per-conversation stash,
- conversation_handler._stash_referenced_items: ingest items off tool results,
- _is_transient_system_block: the block is stripped+rebuilt each turn,
- _append_referenced_items_block: injected only when something was surfaced.
"""

from unittest.mock import MagicMock, patch

from app.core.conversation_cache import ConversationCache
from app.core.prompt_providers.shared.core_rules import (
    RECENTLY_SHOWN_PREFIX,
    render_referenced_items_block,
)


_ITEMS = [
    {"ref_id": "eml_1", "label": "from ABC — 'Invoice #42'", "attrs": {"sender": "abc@x.com"}, "actions": ["mark_read", "draft_reply"]},
    {"ref_id": "eml_2", "label": "from Dana — 'Lunch?'", "attrs": {}, "actions": ["mark_read"]},
]


# ── render_referenced_items_block ────────────────────────────────────────────


def test_render_empty_returns_blank():
    assert render_referenced_items_block([]) == ""
    assert render_referenced_items_block(None) == ""


def test_render_numbers_items_with_ref_ids_and_actions():
    block = render_referenced_items_block(_ITEMS)
    assert block.startswith(RECENTLY_SHOWN_PREFIX)
    assert "1 [eml_1] from ABC — 'Invoice #42'" in block
    assert "2 [eml_2] from Dana — 'Lunch?'" in block
    # Actions are the union, sorted.
    assert "Valid actions: draft_reply, mark_read." in block
    assert "act_on_items" in block


def test_render_truncates_beyond_cap_with_note():
    many = [{"ref_id": f"r{i}", "label": f"item {i}", "attrs": {}, "actions": ["mark_read"]} for i in range(12)]
    block = render_referenced_items_block(many)
    assert "1 [r0] item 0" in block
    assert "8 [r7] item 7" in block
    assert "[r8]" not in block  # capped at 8
    assert "+4 more not shown" in block


# ── ConversationCache.set/get_referenced_items ───────────────────────────────


def test_cache_new_conversation_has_empty_referenced_items():
    cache = ConversationCache()
    cache.set("c1", messages=[], available_commands=[])
    assert cache.get_referenced_items("c1") == []


def test_cache_set_and_get_round_trip():
    cache = ConversationCache()
    cache.set("c1", messages=[], available_commands=[])
    cache.set_referenced_items("c1", _ITEMS)
    assert cache.get_referenced_items("c1") == _ITEMS


def test_cache_get_missing_conversation_is_empty():
    cache = ConversationCache()
    assert cache.get_referenced_items("nope") == []


def test_cache_set_on_missing_conversation_is_noop():
    cache = ConversationCache()
    cache.set_referenced_items("ghost", _ITEMS)  # must not raise
    assert cache.get_referenced_items("ghost") == []


def test_cache_latest_surfacing_wins():
    cache = ConversationCache()
    cache.set("c1", messages=[], available_commands=[])
    cache.set_referenced_items("c1", _ITEMS)
    newer = [{"ref_id": "art_1", "label": "headline", "attrs": {}, "actions": ["send_full_article"]}]
    cache.set_referenced_items("c1", newer)
    assert cache.get_referenced_items("c1") == newer


# ── conversation_handler helpers ─────────────────────────────────────────────


def test_is_transient_matches_recently_shown_block():
    from app.core.conversation_handler import _is_transient_system_block

    assert _is_transient_system_block({"role": "system", "content": f"{RECENTLY_SHOWN_PREFIX} (act on these...)"}) is True
    # Sanity: an ordinary system prompt is not stripped.
    assert _is_transient_system_block({"role": "system", "content": "You are Jarvis."}) is False
    assert _is_transient_system_block({"role": "user", "content": RECENTLY_SHOWN_PREFIX}) is False


def test_stash_reads_items_off_tool_output_dict():
    from app.core import conversation_handler as ch

    tool_results = [{"tool_call_id": "tc1", "output": {"success": True, "message": "You have 2.", "referenceable_items": _ITEMS}}]
    with patch.object(ch, "conversation_cache") as mock_cache:
        ch._stash_referenced_items("c1", tool_results)
        mock_cache.set_referenced_items.assert_called_once_with("c1", _ITEMS)


def test_stash_parses_json_string_output():
    import json
    from app.core import conversation_handler as ch

    tool_results = [{"tool_call_id": "tc1", "output": json.dumps({"referenceable_items": _ITEMS})}]
    with patch.object(ch, "conversation_cache") as mock_cache:
        ch._stash_referenced_items("c1", tool_results)
        mock_cache.set_referenced_items.assert_called_once_with("c1", _ITEMS)


def test_stash_noop_when_no_items_present():
    from app.core import conversation_handler as ch

    tool_results = [{"tool_call_id": "tc1", "output": {"success": True, "message": "Marked 2 as read."}}]
    with patch.object(ch, "conversation_cache") as mock_cache:
        ch._stash_referenced_items("c1", tool_results)
        mock_cache.set_referenced_items.assert_not_called()


def test_append_block_injects_when_items_stashed():
    from app.core import conversation_handler as ch

    messages: list = []
    with patch.object(ch, "conversation_cache") as mock_cache:
        mock_cache.get_referenced_items.return_value = _ITEMS
        ch._append_referenced_items_block(messages, "c1")
    assert len(messages) == 1
    assert messages[0]["role"] == "system"
    assert messages[0]["content"].startswith(RECENTLY_SHOWN_PREFIX)


def test_append_block_noop_when_nothing_stashed():
    from app.core import conversation_handler as ch

    messages: list = []
    with patch.object(ch, "conversation_cache") as mock_cache:
        mock_cache.get_referenced_items.return_value = []
        ch._append_referenced_items_block(messages, "c1")
    assert messages == []
