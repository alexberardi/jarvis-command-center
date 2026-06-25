"""Tests for the ``web_search.enabled`` household gate.

Web search (the ``quick_search`` live lookups and the ``deep_research`` tool)
makes OUTBOUND internet requests and is gated on a per-household setting that
defaults OFF. This feature is a *privacy claim*: a local-only household must
never egress. So every independent reach path that could trigger an outbound
request is asserted closed when the setting is disabled — and, critically, the
gate FAILS CLOSED on any settings error (the opposite of the memory gate).

Reach paths covered:
  1. ConversationHandler._get_web_search_enabled — the fail-closed primitive.
  2. ConversationHandler._maybe_quick_search — the keyword pre-exec path that
     runs the tool DIRECTLY, bypassing the LLM tool-call decision. The most
     dangerous path; gating only the prompt whitelist would leave it open.
  3. quick_search_tool.execute — defense-in-depth re-check at execution time.
  4. deep_research_tool.execute — same, for the background research tool.

Hermetic: no network, no DB. Settings + cache + registry are all stubbed.
"""
import pytest
from unittest.mock import MagicMock, patch

SETTINGS = "app.services.settings_service.get_settings_service"


def _settings_returning(value):
    """get_settings_service() stub whose .get(...) always returns ``value``."""
    svc = MagicMock()
    svc.get.return_value = value
    return MagicMock(return_value=svc)


def _settings_raising():
    """get_settings_service() stub that raises — exercises fail-closed."""
    return MagicMock(side_effect=RuntimeError("settings service down"))


def _handler():
    from app.core.conversation_handler import ConversationHandler
    return ConversationHandler(model=MagicMock(), llm_client=MagicMock())


class TestWebSearchEnabledHelper:
    """The fail-closed primitive every other gate is built on."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            (True, True), ("true", True), ("True", True), ("1", True), ("yes", True),
            (False, False), ("false", False), ("0", False), ("no", False),
            (None, False), ("", False),
        ],
    )
    def test_coerces_setting_value(self, raw, expected):
        with patch(SETTINGS, _settings_returning(raw)):
            assert _handler()._get_web_search_enabled({"household_id": "hh-1"}) is expected

    def test_fails_closed_on_settings_error(self):
        # A settings outage / typo'd key must DISABLE web search, never enable it.
        with patch(SETTINGS, _settings_raising()):
            assert _handler()._get_web_search_enabled({"household_id": "hh-1"}) is False

    def test_passes_household_id_to_settings(self):
        svc = MagicMock()
        svc.get.return_value = True
        with patch(SETTINGS, MagicMock(return_value=svc)):
            _handler()._get_web_search_enabled({"household_id": "hh-xyz"})
            svc.get.assert_called_once_with("web_search.enabled", household_id="hh-xyz")

    def test_none_context_fails_closed(self):
        svc = MagicMock()
        svc.get.return_value = False
        with patch(SETTINGS, MagicMock(return_value=svc)):
            assert _handler()._get_web_search_enabled(None) is False


class TestMaybeQuickSearchGate:
    """The keyword pre-exec path — the make-or-break reach path.

    It executes quick_search directly on a regex match, bypassing the prompt
    whitelist entirely. If this isn't gated, a disabled household still fires a
    live search + page scrape on any current-events-shaped utterance.
    """

    SEARCH_UTTERANCE = "who is the current president"

    @pytest.mark.asyncio
    async def test_disabled_does_not_search(self):
        tool = MagicMock()
        with patch("app.core.conversation_handler.conversation_cache") as cache, \
             patch("app.core.conversation_handler.tool_registry") as registry, \
             patch(SETTINGS, _settings_returning(False)):
            cache.get_node_context.return_value = {"household_id": "hh-1"}
            registry.get_tool.return_value = tool

            result = await _handler()._maybe_quick_search(self.SEARCH_UTTERANCE, "conv-1")

            assert result is None
            # Gated BEFORE fetching/executing the tool — no egress.
            registry.get_tool.assert_not_called()
            tool.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_disabled_fails_closed_on_settings_error(self):
        with patch("app.core.conversation_handler.conversation_cache") as cache, \
             patch("app.core.conversation_handler.tool_registry") as registry, \
             patch(SETTINGS, _settings_raising()):
            cache.get_node_context.return_value = {"household_id": "hh-1"}

            result = await _handler()._maybe_quick_search("who won last night", "conv-1")

            assert result is None
            registry.get_tool.assert_not_called()

    @pytest.mark.asyncio
    async def test_enabled_searches_and_threads_conversation_id(self):
        tool = MagicMock()
        tool.execute.return_value = {
            "sources": [{"title": "T", "url": "https://x", "content": "c"}],
            "elapsed_seconds": 1,
        }
        with patch("app.core.conversation_handler.conversation_cache") as cache, \
             patch("app.core.conversation_handler.tool_registry") as registry, \
             patch(SETTINGS, _settings_returning(True)):
            cache.get_node_context.return_value = {"household_id": "hh-1"}
            registry.get_tool.return_value = tool

            result = await _handler()._maybe_quick_search("what's the latest news", "conv-1")

            assert result is not None
            # conversation_id MUST be threaded through so the execute-time
            # defense-in-depth re-check can resolve the household.
            tool.execute.assert_called_once_with(
                query="what's the latest news", conversation_id="conv-1"
            )

    @pytest.mark.asyncio
    async def test_enabled_non_search_utterance_does_not_search(self):
        tool = MagicMock()
        with patch("app.core.conversation_handler.conversation_cache") as cache, \
             patch("app.core.conversation_handler.tool_registry") as registry, \
             patch(SETTINGS, _settings_returning(True)):
            cache.get_node_context.return_value = {"household_id": "hh-1"}
            registry.get_tool.return_value = tool

            result = await _handler()._maybe_quick_search("turn off the kitchen light", "conv-1")

            assert result is None
            tool.execute.assert_not_called()


class TestQuickSearchExecuteGuard:
    """Defense-in-depth: a hallucinated tool call must not egress either."""

    def _tool(self):
        from app.core.tools.quick_search_tool import QuickSearchTool
        return QuickSearchTool()

    def test_blocked_when_disabled(self):
        with patch("app.core.conversation_cache.conversation_cache") as cache, \
             patch(SETTINGS, _settings_returning(False)):
            cache.get_node_context.return_value = {"household_id": "hh-1"}
            res = self._tool().execute(query="who is x", conversation_id="conv-1")
            assert res.get("error") == "web_search_disabled"

    def test_blocked_when_no_conversation_id(self):
        # No conversation_id → cannot resolve household → fail closed.
        res = self._tool().execute(query="who is x")
        assert res.get("error") == "web_search_disabled"

    def test_blocked_fail_closed_on_settings_error(self):
        with patch("app.core.conversation_cache.conversation_cache") as cache, \
             patch(SETTINGS, _settings_raising()):
            cache.get_node_context.return_value = {"household_id": "hh-1"}
            res = self._tool().execute(query="who is x", conversation_id="conv-1")
            assert res.get("error") == "web_search_disabled"


class TestDeepResearchExecuteGuard:
    """deep_research shares the gate (user chose 'all outbound web')."""

    def _tool(self):
        from app.core.tools.deep_research_tool import DeepResearchTool
        return DeepResearchTool()

    def test_blocked_when_disabled(self):
        with patch("app.core.tools.deep_research_tool.conversation_cache") as cache, \
             patch(SETTINGS, _settings_returning(False)):
            cache.get_node_context.return_value = {
                "household_id": "hh-1", "speaker_user_id": 1,
            }
            res = self._tool().execute(query="research keyboards", conversation_id="conv-1")
            assert res.get("error") == "web_search_disabled"

    def test_blocked_fail_closed_on_settings_error(self):
        with patch("app.core.tools.deep_research_tool.conversation_cache") as cache, \
             patch(SETTINGS, _settings_raising()):
            cache.get_node_context.return_value = {
                "household_id": "hh-1", "speaker_user_id": 1,
            }
            res = self._tool().execute(query="research keyboards", conversation_id="conv-1")
            assert res.get("error") == "web_search_disabled"
