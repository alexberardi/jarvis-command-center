"""The proactive reasoner's enqueue side: it runs on the BACKGROUND model (never
the live voice slot), is OFF by default, and coalesces edges via a debounce.
"""
import asyncio
from unittest.mock import AsyncMock, patch

import app.services.situation_matcher_service as sm

_ACTIONS = [{"command": "add_event", "callback": "create_event",
             "params": [], "card_title": "Add?"}]
_BUNDLE = [{"source_key": "cal@x", "data": {"title": "D"}}]


def test_enqueue_uses_background_model():
    async def go():
        with patch.object(sm, "_proactive_enabled", return_value=True), \
             patch.object(sm, "_gather_bundle", return_value=_BUNDLE), \
             patch("app.services.capability_registry.list_proposable_actions",
                   new=AsyncMock(return_value=_ACTIONS)), \
             patch("app.core.utils.rest_client.post",
                   new=AsyncMock(return_value={})) as post_mock:
            await sm.run_match_batch("hh-1", node_id="node-7")
        assert post_mock.await_count == 1
        payload = post_mock.call_args.kwargs["json_data"]
        assert payload["request"]["model"] == "background"     # off the live slot
        assert payload["job_type"] == "chat"
        assert payload["callback"]["url"].endswith("/situation-matcher/callback")
        assert "bundle" in payload["metadata"]
    asyncio.run(go())


def test_enqueue_off_by_default():
    async def go():
        with patch.object(sm, "_proactive_enabled", return_value=False), \
             patch("app.core.utils.rest_client.post", new=AsyncMock()) as post_mock:
            await sm.run_match_batch("hh-1", node_id="node-7")
        post_mock.assert_not_awaited()                          # dormant unless enabled
    asyncio.run(go())


def test_no_bundle_no_enqueue():
    async def go():
        with patch.object(sm, "_proactive_enabled", return_value=True), \
             patch.object(sm, "_gather_bundle", return_value=[]), \
             patch("app.core.utils.rest_client.post", new=AsyncMock()) as post_mock:
            await sm.run_match_batch("hh-1", node_id="node-7")
        post_mock.assert_not_awaited()
    asyncio.run(go())


def test_debounce_coalesces_edges():
    async def go():
        with patch.object(sm, "DEBOUNCE_SECONDS", 0.02), \
             patch.object(sm, "run_match_batch", new=AsyncMock()) as rmb:
            sm.signal_situation_edge("hh-1")
            sm.signal_situation_edge("hh-1")
            sm.signal_situation_edge("hh-1")
            await asyncio.sleep(0.15)          # only the last debounce survives
            rmb.assert_awaited_once()
    asyncio.run(go())


def test_gather_bundle_excludes_bridge_handled_kinds():
    """appt.upcoming + leave_by.suggested are handled DETERMINISTICALLY by the
    SignalReactionBridge, so the proactive matcher must never see them (else it
    would propose a second leave-by card on top of the bridge's)."""
    import json
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    rows = [
        SimpleNamespace(kind="presence.seen", source_key="presence:1",
                        facts=json.dumps({"user": "A"}), summary="A is home"),
        SimpleNamespace(kind="appt.upcoming", source_key="appt:1:e1",
                        facts=json.dumps({"title": "D"}), summary="upcoming"),
        SimpleNamespace(kind="leave_by.suggested", source_key="leaveby:e1",
                        facts=json.dumps({"title": "D"}), summary="leave by"),
    ]
    session = MagicMock()
    session.query.return_value.filter.return_value.all.return_value = rows

    with patch("app.db.get_session_local", return_value=lambda: session):
        bundle = sm._gather_bundle("hh-1")

    assert {b["kind"] for b in bundle} == {"presence.seen"}
