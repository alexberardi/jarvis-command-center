"""Tests for latency-safe characterization injection (Slice 3).

Covers the renderer (tone-shaping framing, empty fallback) and the messages[0]
swap helper (replace-not-mutate, byte-identical base, no-op on a correct
prediction). No DB or model needed.
"""

import types

from app.core.prompt_providers.shared.core_rules import (
    EXCHANGE_COMPLETE_INSTRUCTION,
    NOT_FOR_ME_INSTRUCTION,
    build_characterization_section,
)
from app.core import conversation_handler as ch


class TestBuildCharacterizationSection:
    def test_empty_returns_empty(self):
        assert build_characterization_section("") == ""
        assert build_characterization_section("   ") == ""
        assert build_characterization_section(None) == ""

    def test_contains_rendered_and_shaping_framing(self):
        out = build_characterization_section("Alex is a builder in Brick, NJ.")
        assert "Alex is a builder in Brick, NJ." in out
        assert "shape your tone" in out       # tone-shaping intent
        assert "draw on it naturally" in out   # may USE it (fixes "who is Leo")
        assert "don't recite it back" in out   # but not as a recited list


class TestApplyCharacterizationSwap:
    def _msgs(self, content="SYSTEM_BASE"):
        return [
            {"role": "system", "content": content},
            {"role": "user", "content": "hi"},
        ]

    def test_noop_when_no_base_stashed(self, monkeypatch):
        # Injection disabled / not warmed → no _system_prompt_base → untouched.
        monkeypatch.setattr(ch.conversation_cache, "get_node_context", lambda cid: {})
        msgs = self._msgs()
        first = msgs[0]
        ch._apply_characterization_swap(msgs, "c1")
        assert msgs[0] is first

    def test_appends_section_on_mispredict(self, monkeypatch):
        base = "BASE\n"
        ctx = {"_system_prompt_base": base, "characterization": "Likes brevity."}
        monkeypatch.setattr(ch.conversation_cache, "get_node_context", lambda cid: ctx)
        msgs = self._msgs(content=base)  # e.g. predicted speaker had no characterization
        ch._apply_characterization_swap(msgs, "c1")
        assert msgs[0]["content"].startswith(base)  # leading base byte-identical
        assert "Likes brevity." in msgs[0]["content"]
        assert msgs[1]["content"] == "hi"           # only messages[0] touched

    def test_clears_tail_when_confirmed_has_no_char(self, monkeypatch):
        base = "BASE\n"
        section = build_characterization_section("stale predicted view")
        ctx = {"_system_prompt_base": base, "characterization": ""}
        monkeypatch.setattr(ch.conversation_cache, "get_node_context", lambda cid: ctx)
        msgs = self._msgs(content=f"{base}\n{section}")
        ch._apply_characterization_swap(msgs, "c1")
        assert msgs[0]["content"] == base           # tail removed, base intact

    def test_noop_when_already_correct(self, monkeypatch):
        base = "BASE\n"
        rendered = "Steady view."
        desired = f"{base}\n{build_characterization_section(rendered)}"
        ctx = {"_system_prompt_base": base, "characterization": rendered}
        monkeypatch.setattr(ch.conversation_cache, "get_node_context", lambda cid: ctx)
        msgs = self._msgs(content=desired)
        first = msgs[0]
        ch._apply_characterization_swap(msgs, "c1")
        assert msgs[0] is first                      # unchanged object → full cache HIT

    def test_replaces_dict_never_mutates(self, monkeypatch):
        # The tool-stream path shares messages[0] with the cache — must not mutate it.
        base = "BASE\n"
        ctx = {"_system_prompt_base": base, "characterization": "New view."}
        monkeypatch.setattr(ch.conversation_cache, "get_node_context", lambda cid: ctx)
        original = {"role": "system", "content": base}
        msgs = [original, {"role": "user", "content": "hi"}]
        ch._apply_characterization_swap(msgs, "c1")
        assert msgs[0] is not original               # replaced
        assert original["content"] == base           # original dict untouched


class TestGetSystemPromptInjection:
    """The warmup-side prompt build: disabled = byte-identical; enabled = tail + stash."""

    def _handler(self, enabled):
        class _Provider:
            def build_system_prompt(self, nc, tz, tools, flags):
                return "PROVIDER_BASE"

        return types.SimpleNamespace(
            prompt_provider=_Provider(),
            model=object(),
            _get_characterization_injection_enabled=lambda nc: enabled,
        )

    def _base_full(self):
        return f"PROVIDER_BASE\n\n{NOT_FOR_ME_INSTRUCTION}\n\n{EXCHANGE_COMPLETE_INSTRUCTION}\n"

    def test_disabled_is_byte_identical(self):
        nc = {"characterization": "Loves jazz."}
        out = ch.ConversationHandler._get_system_prompt(self._handler(False), nc, None, [], None)
        assert out == self._base_full()
        assert "_system_prompt_base" not in nc   # nothing stashed when disabled
        assert "Loves jazz." not in out

    def test_enabled_appends_section_and_stashes_base(self):
        nc = {"characterization": "Loves jazz."}
        out = ch.ConversationHandler._get_system_prompt(self._handler(True), nc, None, [], None)
        assert out.startswith(self._base_full())
        assert "Loves jazz." in out
        assert nc["_system_prompt_base"] == self._base_full()

    def test_enabled_no_char_is_byte_identical_but_stashes_base(self):
        nc = {}
        out = ch.ConversationHandler._get_system_prompt(self._handler(True), nc, None, [], None)
        assert out == self._base_full()              # byte-identical prompt
        assert nc["_system_prompt_base"] == self._base_full()  # base stashed for later swap
