"""Plan-time context queries + the phone-call availability envelope.

The governing rule (phone-calls PRD, cross-agent-context): context enters
at PLAN time and degrades in every direction. A missing node, a missing
calendar command, a dead broker, or a malformed answer must all land the
user on the fill-me-in placeholder — never a blocked plan, never invented
times.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from app.services import context_provider_client as cpc
from app.services.context_provider_client import ContextAnswer, query_context
from app.services.phone_call_service import (
    _format_availability,
    apply_availability_envelope,
    is_scheduling_goal,
)


HH = "hh-ctx"


class _FakeNode:
    def __init__(self, node_id: str, online: bool = True, last_seen=None):
        self.node_id = node_id
        self.household_id = HH
        self.is_active = True
        self._online = online
        self.last_seen = last_seen

    def is_online(self) -> bool:
        return self._online


def _patch_nodes(nodes):
    return patch.object(cpc, "_candidate_nodes", return_value=nodes)


def _mqtt_returning(*payloads):
    """Fake MQTT client whose request_response returns each payload in turn."""
    client = MagicMock()
    encoded = [None if p is None else json.dumps(p) for p in payloads]
    client.request_response.side_effect = encoded
    return client


def _patch_mqtt(client):
    return patch("app.node_settings.get_mqtt_client", return_value=client)


# ── query_context ──────────────────────────────────────────────────────────


class TestQueryContext:
    @pytest.mark.asyncio
    async def test_returns_first_successful_answer(self):
        client = _mqtt_returning(
            {"ok": True, "data": {"free": ["Thu 2-5pm"]}, "command_name": "calendar"}
        )
        with _patch_nodes([_FakeNode("n1")]), _patch_mqtt(client):
            answer = await query_context(HH, "availability", {"start": "a", "end": "b"})

        assert answer.ok
        assert answer.data["free"] == ["Thu 2-5pm"]
        assert answer.node_id == "n1"
        assert answer.command_name == "calendar"

        topic = client.request_response.call_args.args[0]
        payload = json.loads(client.request_response.call_args.args[2])
        assert topic == "jarvis/nodes/n1/context/query"
        assert payload["operation"] == "availability"
        assert "correlation_id" in payload

    @pytest.mark.asyncio
    async def test_no_provider_falls_through_to_next_node(self):
        """A node without the calendar command isn't a failure — keep looking."""
        client = _mqtt_returning(
            {"ok": False, "code": "no_provider", "error": "no provider"},
            {"ok": True, "data": {"free": ["Fri 9-11am"]}, "command_name": "calendar"},
        )
        with _patch_nodes([_FakeNode("n1"), _FakeNode("n2")]), _patch_mqtt(client):
            answer = await query_context(HH, "availability")

        assert answer.ok and answer.node_id == "n2"
        assert client.request_response.call_count == 2

    @pytest.mark.asyncio
    async def test_timeout_degrades(self):
        with _patch_nodes([_FakeNode("n1")]), _patch_mqtt(_mqtt_returning(None)):
            answer = await query_context(HH, "availability")
        assert not answer.ok and "did not respond" in answer.error

    @pytest.mark.asyncio
    async def test_no_nodes_degrades(self):
        with _patch_nodes([]), _patch_mqtt(_mqtt_returning()):
            answer = await query_context(HH, "availability")
        assert not answer.ok and "no nodes" in answer.error

    @pytest.mark.asyncio
    async def test_mqtt_unavailable_degrades(self):
        with _patch_nodes([_FakeNode("n1")]), _patch_mqtt(None):
            answer = await query_context(HH, "availability")
        assert not answer.ok

    @pytest.mark.asyncio
    async def test_malformed_answer_degrades(self):
        client = MagicMock()
        client.request_response.return_value = "{not json"
        with _patch_nodes([_FakeNode("n1")]), _patch_mqtt(client):
            answer = await query_context(HH, "availability")
        assert not answer.ok

    @pytest.mark.asyncio
    async def test_real_node_error_stops_and_reports(self):
        """A provider that exists but failed is reported, not retried blindly."""
        client = _mqtt_returning({"ok": False, "error": "iCloud unreachable"})
        with _patch_nodes([_FakeNode("n1"), _FakeNode("n2")]), _patch_mqtt(client):
            answer = await query_context(HH, "availability")
        assert not answer.ok and "iCloud unreachable" in answer.error
        assert client.request_response.call_count == 1


# ── Envelope formatting ────────────────────────────────────────────────────


class TestFormatAvailability:
    def test_free_and_busy_rendered(self):
        out = _format_availability(
            {"free": ["Thu 2-5pm", "Fri after 4"], "busy": ["Thu 3:30 soccer"]}
        )
        assert "Acceptable times: Thu 2-5pm; Fri after 4" in out
        assert "Do not book: Thu 3:30 soccer" in out

    def test_empty_answer_is_none(self):
        assert _format_availability({}) is None
        assert _format_availability({"free": [], "busy": []}) is None

    def test_busy_only_still_warns(self):
        out = _format_availability({"busy": ["all day Mon"]})
        assert "edit me" in out and "Do not book: all day Mon" in out


# ── The plan-time wiring ───────────────────────────────────────────────────


class TestApplyAvailabilityEnvelope:
    @pytest.mark.asyncio
    async def test_non_scheduling_goal_untouched(self):
        details = "Order a large pepperoni for pickup."
        out = await apply_availability_envelope(
            household_id=HH, goal="order a pizza", details=details
        )
        assert out == details

    @pytest.mark.asyncio
    async def test_real_availability_replaces_placeholder(self):
        answer = ContextAnswer(
            ok=True, data={"free": ["Thu 2-5pm"], "busy": ["Thu 3:30 soccer"]}
        )
        details = (
            "Book a haircut.\n"
            "Acceptable times: (fill in your availability before calling)"
        )
        with patch.object(cpc, "query_context", return_value=answer):
            with patch(
                "app.services.context_provider_client.query_context",
                return_value=answer,
            ):
                out = await apply_availability_envelope(
                    household_id=HH, goal="book a haircut", details=details
                )
        assert "fill in your availability" not in out
        assert "Acceptable times: Thu 2-5pm" in out
        assert "Do not book: Thu 3:30 soccer" in out

    @pytest.mark.asyncio
    async def test_unavailable_calendar_says_so(self):
        answer = ContextAnswer.unavailable("node did not respond")
        with patch(
            "app.services.context_provider_client.query_context", return_value=answer
        ):
            out = await apply_availability_envelope(
                household_id=HH,
                goal="book an appointment",
                details="Book an appointment.",
            )
        assert "calendar unavailable" in out
        assert "fill in your availability" in out

    @pytest.mark.asyncio
    async def test_user_supplied_times_are_not_overwritten(self):
        details = "Book a table.\nAcceptable times: Fri 6-8pm"
        called = MagicMock()
        with patch(
            "app.services.context_provider_client.query_context", side_effect=called
        ):
            out = await apply_availability_envelope(
                household_id=HH, goal="book a table", details=details
            )
        assert out == details
        called.assert_not_called()

    @pytest.mark.asyncio
    async def test_lookup_exception_falls_back_to_placeholder(self):
        with patch(
            "app.services.context_provider_client.query_context",
            side_effect=RuntimeError("boom"),
        ):
            out = await apply_availability_envelope(
                household_id=HH, goal="schedule a visit", details="Schedule a visit."
            )
        assert "Acceptable times:" in out and "fill in" in out


class TestSchedulingDetection:
    @pytest.mark.parametrize(
        "goal",
        ["book a haircut", "schedule an appointment", "make a reservation", "reserve a table"],
    )
    def test_scheduling_goals_detected(self, goal):
        assert is_scheduling_goal(goal)

    def test_order_is_not_scheduling(self):
        assert not is_scheduling_goal("order a large pizza for pickup")
