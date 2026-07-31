"""Errand executor — per-step server-vs-node dispatch (Errand Runner §2.5-2.6).

The executor routes each plan step to its plane by ``tool_registry.has_tool`` (the
SAME discriminator the conversation loop uses): node command → the node over MQTT
(``dispatch_node_command``); server tool → in-process (``tool_executor.execute_tool``)
with a synthesized ``conversation_cache`` context so make_phone_call & co. resolve
household/speaker exactly like a live turn. All mocked — no MQTT/registry/LLM.
"""

import asyncio
import json
from unittest.mock import AsyncMock, patch

from app.core.conversation_cache import conversation_cache
from app.services import errand_executor


def _run(steps, **kw):
    # Keep the completion message deterministic + offline: skip the LLM compose
    # and use the plain per-step join (the fallback). The compose itself is tested
    # separately in test_compose_errand_message_uses_llm_and_falls_back.
    with patch("app.services.errand_executor._compose_errand_message",
               new=AsyncMock(side_effect=lambda goal, results, fallback: fallback)):
        return asyncio.run(errand_executor.execute_errand("hh-1", "node-1", steps, **kw))


def test_node_steps_dispatched_and_aggregated():
    steps = [
        {"command": "get_weather", "args": {"city": "Boston"}, "label": "Weather"},
        {"command": "set_reminder", "args": {"text": "umbrella"}, "label": "Reminder"},
    ]
    dispatch = AsyncMock(side_effect=[
        {"success": True, "message": "Sunny."},
        {"success": True, "message": "Reminder set."},
    ])
    with patch("app.core.tool_registry.tool_registry.has_tool", return_value=False), \
         patch("app.services.node_command_service.dispatch_node_command", new=dispatch):
        result = _run(steps, user_id=7, goal="weather + reminder")
    assert result["status"] == "success" and result["passed"] == 2 and result["failed"] == 0
    # each step dispatched to the node as a single ad-hoc command, with its args
    assert dispatch.await_count == 2
    first = dispatch.await_args_list[0]
    assert first.args[0] == "node-1" and first.args[1] == "get_weather"
    assert first.args[2] == {"city": "Boston"}  # non-date args pass through untouched
    assert first.kwargs["user_id"] == 7
    assert "Sunny." in result["message"] and "Reminder set." in result["message"]


def test_node_step_resolves_relative_date_keys():
    """resolved_datetimes keywords ('today') must become ISO datetimes BEFORE node
    dispatch — the fix for the live 'Dates not found in forecast' failure (the
    conversation loop resolves them; the errand executor must too)."""
    captured: dict = {}

    async def fake_dispatch(node_id, command, args, **kw):
        captured["args"] = args
        return {"success": True, "message": "sunny"}

    with patch("app.core.tool_registry.tool_registry.has_tool", return_value=False), \
         patch("app.services.node_command_service.dispatch_node_command", new=fake_dispatch):
        _run([{"command": "get_weather", "args": {"resolved_datetimes": ["today"]}, "label": "W"}])
    sent = captured["args"]["resolved_datetimes"]
    assert sent and sent[0] != "today"  # resolved, not the raw keyword
    assert "T" in sent[0] and sent[0][:4].isdigit()  # looks like an ISO datetime


def test_server_step_runs_in_process_with_synthesized_context():
    """A SYNC server tool resolves household/speaker from the synthetic conversation
    entry the executor seeds (a card tap has no live conversation), and the entry is
    cleaned up afterwards. (make_phone_call is handled separately now — it's a
    DEFERRED step that suspends; see the suspend/resume tests below.)"""
    captured: dict = {}

    def fake_execute(name, args, conversation_id=None, user_utterance=None):
        ctx = conversation_cache.get_node_context(conversation_id)
        captured["ctx"] = dict(ctx) if ctx else None
        captured["conv"] = conversation_id
        return json.dumps({"status": "ok", "message": "searched"})

    steps = [{"command": "quick_search", "args": {"query": "drills"}, "label": "Search"}]
    with patch("app.core.tool_registry.tool_registry.has_tool", return_value=True), \
         patch("app.core.tool_executor.tool_executor.execute_tool", side_effect=fake_execute) as ex:
        result = _run(steps, user_id=7, goal="research drills")

    assert result["status"] == "success" and "searched" in result["message"]
    ex.assert_called_once()
    assert ex.call_args.args[0] == "quick_search"
    assert ex.call_args.kwargs["user_utterance"] == "research drills"
    # resolved household + speaker from the synthetic context
    assert captured["ctx"]["household_id"] == "hh-1"
    assert captured["ctx"]["speaker_user_id"] == 7
    assert captured["ctx"]["node_id"] == "node-1"
    # synthetic entry cleaned up (no leak in the shared cache)
    assert conversation_cache.get_node_context(captured["conv"]) is None


def test_mixed_node_and_server_steps_preserve_order():
    steps = [
        {"command": "get_weather", "args": {}, "label": "W"},          # node
        {"command": "quick_search", "args": {}, "label": "Search"},    # sync server tool
        {"command": "set_reminder", "args": {}, "label": "R"},         # node
    ]
    dispatch = AsyncMock(return_value={"success": True, "message": "ok"})
    with patch("app.core.tool_registry.tool_registry.has_tool",
               side_effect=lambda name: name == "quick_search"), \
         patch("app.services.node_command_service.dispatch_node_command", new=dispatch), \
         patch("app.core.tool_executor.tool_executor.execute_tool",
               return_value=json.dumps({"status": "ok", "message": "searched"})):
        result = _run(steps)
    assert result["passed"] == 3
    # only the 2 node steps went to the node, in order; the server step ran in-process
    assert [c.args[1] for c in dispatch.await_args_list] == ["get_weather", "set_reminder"]
    assert [r["label"] for r in result["results"]] == ["W", "Search", "R"]  # overall order preserved


def test_failed_node_step_yields_partial():
    steps = [{"command": "a", "args": {}, "label": "A"}, {"command": "b", "args": {}, "label": "B"}]
    dispatch = AsyncMock(side_effect=[
        {"success": True, "message": "ok"},
        {"success": False, "error": "device offline"},
    ])
    with patch("app.core.tool_registry.tool_registry.has_tool", return_value=False), \
         patch("app.services.node_command_service.dispatch_node_command", new=dispatch):
        result = _run(steps)
    assert result["status"] == "partial" and result["passed"] == 1 and result["failed"] == 1
    assert "device offline" in result["message"]


def test_all_failed_yields_failed():
    dispatch = AsyncMock(return_value={"success": False, "error": "boom"})
    with patch("app.core.tool_registry.tool_registry.has_tool", return_value=False), \
         patch("app.services.node_command_service.dispatch_node_command", new=dispatch):
        result = _run([{"command": "a", "args": {}, "label": "A"}])
    assert result["status"] == "failed" and result["failed"] == 1


def test_node_timeout_is_a_failed_step():
    dispatch = AsyncMock(return_value={
        "success": False, "error": "the node didn't respond in time", "timeout": True,
    })
    with patch("app.core.tool_registry.tool_registry.has_tool", return_value=False), \
         patch("app.services.node_command_service.dispatch_node_command", new=dispatch):
        result = _run([{"command": "a", "args": {}, "label": "A"}])
    assert result["status"] == "failed"


def test_server_tool_error_is_a_failed_step():
    # a refused server tool returns {error, message}: a failed step whose error
    # text is the human message.
    with patch("app.core.tool_registry.tool_registry.has_tool", return_value=True), \
         patch("app.core.tool_executor.tool_executor.execute_tool",
               return_value=json.dumps({"error": "web_search_disabled", "message": "not set up"})):
        result = _run([{"command": "quick_search", "args": {}, "label": "Search"}])
    assert result["status"] == "failed"
    assert result["results"][0]["error"] == "not set up"


def test_data_returning_server_tool_summarized_not_lost():
    """quick_search/recall return DATA with no 'message' key — the step must not
    render as a bare 'done', and the payload must be preserved on the outcome."""
    with patch("app.core.tool_registry.tool_registry.has_tool", return_value=True), \
         patch("app.core.tool_executor.tool_executor.execute_tool",
               return_value=json.dumps({"query": "drills", "sources": [{"title": "a"}, {"title": "b"}]})):
        result = _run([{"command": "quick_search", "args": {}, "label": "Search"}])
    assert result["status"] == "success"
    r = result["results"][0]
    assert r["message"] == "2 sources"  # summarized, not a bare "done"
    assert r["data"]["query"] == "drills"  # full payload preserved, not discarded


def test_step_crash_is_caught_and_fail_fast_stops():
    """A crashing step doesn't raise out — it becomes a failed outcome — and
    FAIL-FAST then stops (the dependent downstream step never runs)."""
    steps = [{"command": "boom", "args": {}, "label": "Boom"},
             {"command": "ok", "args": {}, "label": "OK"}]
    dispatch = AsyncMock(side_effect=[RuntimeError("kaboom"), {"success": True, "message": "fine"}])
    with patch("app.core.tool_registry.tool_registry.has_tool", return_value=False), \
         patch("app.services.node_command_service.dispatch_node_command", new=dispatch):
        result = _run(steps)
    assert dispatch.await_count == 1  # fail-fast: the OK step was never dispatched
    assert result["status"] == "failed" and result["passed"] == 0 and result["failed"] == 1
    assert result["results"][0]["success"] is False and "kaboom" in result["results"][0]["error"]


def test_phone_step_suspends_with_session_link():
    """make_phone_call is a DEFERRED step: the executor awaits create_call_plan
    (linked to the errand) and returns Suspended — NOT fire-and-forget, NOT the
    next call. The prior (weather) result is carried; the 2nd call isn't placed."""
    create = AsyncMock(return_value="sess-123")
    steps = [
        {"command": "get_weather", "args": {}, "label": "W"},
        {"command": "make_phone_call", "args": {"business": "CVS", "goal": "refill"}, "label": "Call CVS"},
        {"command": "make_phone_call", "args": {"business": "Dr", "goal": "appt"}, "label": "Call Dr"},
    ]
    dispatch = AsyncMock(return_value={"success": True, "message": "sunny"})
    with patch("app.core.tool_registry.tool_registry.has_tool", side_effect=lambda n: n == "make_phone_call"), \
         patch("app.services.node_command_service.dispatch_node_command", new=dispatch), \
         patch("app.services.phone_call_service.create_call_plan", new=create):
        result = _run(steps, user_id=7, goal="g", errand_id="pl_x")
    assert isinstance(result, errand_executor.Suspended)
    assert result.cursor == 2 and result.session_id == "sess-123"  # resume AFTER the 1st call
    assert [r["label"] for r in result.results] == ["W"]  # weather carried; 2nd call not placed
    create.assert_awaited_once()
    assert create.await_args.kwargs["errand_id"] == "pl_x"
    assert create.await_args.kwargs["errand_step"] == 1
    assert create.await_args.kwargs["business"] == "CVS"


def test_phone_step_without_errand_link_fails_not_hangs():
    create = AsyncMock(return_value="sess-1")
    with patch("app.core.tool_registry.tool_registry.has_tool", side_effect=lambda n: n == "make_phone_call"), \
         patch("app.services.phone_call_service.create_call_plan", new=create):
        result = _run([{"command": "make_phone_call", "args": {"business": "X", "goal": "y"}, "label": "Call"}])
    assert not isinstance(result, errand_executor.Suspended)
    assert result["status"] == "failed"  # no errand_id → can't resume → failed step
    create.assert_not_awaited()


def test_phone_step_draft_failure_fails_fast():
    """create_call_plan returns None (couldn't draft) → the step fails and the errand
    fails-fast: it doesn't suspend and doesn't run the downstream step."""
    create = AsyncMock(return_value=None)
    steps = [{"command": "make_phone_call", "args": {"business": "X", "goal": "y"}, "label": "Call"},
             {"command": "get_weather", "args": {}, "label": "W"}]
    dispatch = AsyncMock(return_value={"success": True})
    with patch("app.core.tool_registry.tool_registry.has_tool", side_effect=lambda n: n == "make_phone_call"), \
         patch("app.services.node_command_service.dispatch_node_command", new=dispatch), \
         patch("app.services.phone_call_service.create_call_plan", new=create):
        result = _run(steps, errand_id="pl_x")
    assert not isinstance(result, errand_executor.Suspended)
    assert result["status"] == "failed"
    dispatch.assert_not_awaited()  # fail-fast: weather never ran


def test_resume_continues_from_cursor_with_prior_results():
    """Resuming after a successful call: run from start_index, carrying prior
    results, and complete (no more deferred steps)."""
    steps = [
        {"command": "make_phone_call", "args": {"business": "CVS", "goal": "refill"}, "label": "Call CVS"},
        {"command": "get_weather", "args": {}, "label": "W"},
    ]
    prior = [{"command": "make_phone_call", "label": "Call CVS", "success": True,
              "message": "Called CVS.", "error": None, "data": None}]
    dispatch = AsyncMock(return_value={"success": True, "message": "sunny"})
    with patch("app.core.tool_registry.tool_registry.has_tool", side_effect=lambda n: n == "make_phone_call"), \
         patch("app.services.node_command_service.dispatch_node_command", new=dispatch):
        result = _run(steps, errand_id="pl_x", start_index=1, prior_results=prior)
    assert not isinstance(result, errand_executor.Suspended)
    assert result["status"] == "success" and result["passed"] == 2  # the carried call + the weather
    assert [r["label"] for r in result["results"]] == ["Call CVS", "W"]
    dispatch.assert_awaited_once()  # only the weather (step 1) ran on resume


def test_compose_errand_message_uses_llm_and_falls_back():
    """The completion message is LLM-composed from the step data (node commands
    return structured context_data, not prose); any failure → the plain fallback."""
    results = [{"command": "get_weather", "label": "Weather", "success": True,
                "message": None, "data": {"forecast": "sunny, high 78"}}]
    resp = {"choices": [{"message": {"content": "It'll be sunny today with a high of 78."}}]}
    with patch("app.core.llm_proxy_client.LLMProxyClient") as LC:
        LC.return_value.chat_completion = AsyncMock(return_value=resp)
        out = asyncio.run(errand_executor._compose_errand_message("weather", results, "fallback"))
    assert "sunny" in out.lower()
    # the prompt carried the step's structured data for the model to phrase
    sent = LC.return_value.chat_completion.call_args.kwargs["messages"][0]["content"]
    assert "forecast=sunny, high 78" in sent

    with patch("app.core.llm_proxy_client.LLMProxyClient") as LC:
        LC.return_value.chat_completion = AsyncMock(side_effect=RuntimeError("proxy down"))
        out = asyncio.run(errand_executor._compose_errand_message("weather", results, "fallback"))
    assert out == "fallback"  # composition never blocks the terminal card


def test_wait_for_step_suspends_the_run_on_a_timer():
    """A wait_for step suspends the run on the TIMER wakeup source (signal_kind
    'timer' + a wake_at), through the same executor loop a phone step uses — the
    prior weather step is carried; nothing after the wait runs until the timer fires."""
    steps = [
        {"command": "get_weather", "args": {}, "label": "W"},
        {"command": "wait_for", "args": {"delay_seconds": 3600}, "label": "Wait 1h"},
        {"command": "set_reminder", "args": {}, "label": "R"},
    ]
    dispatch = AsyncMock(return_value={"success": True, "message": "sunny"})
    with patch("app.core.tool_registry.tool_registry.has_tool", return_value=False), \
         patch("app.services.node_command_service.dispatch_node_command", new=dispatch):
        result = _run(steps, errand_id="pl_x", goal="g")
    assert isinstance(result, errand_executor.Suspend)
    assert result.signal_kind == "timer" and result.cursor == 2 and result.wake_at is not None
    assert [r["label"] for r in result.results] == ["W"]  # weather carried; reminder not run
    assert dispatch.await_count == 1  # only the weather node step ran


def test_blank_command_steps_are_skipped():
    dispatch = AsyncMock(return_value={"success": True, "message": "ok"})
    with patch("app.core.tool_registry.tool_registry.has_tool", return_value=False), \
         patch("app.services.node_command_service.dispatch_node_command", new=dispatch):
        result = _run([{"command": "", "args": {}, "label": "blank"},
                       {"command": "a", "args": {}, "label": "A"}])
    assert dispatch.await_count == 1 and result["passed"] == 1


def test_context_cleaned_up_even_when_a_step_raises():
    """The synthetic cache entry must be removed in `finally`, even on a crash."""

    async def _boom(node_id, command, args, **kw):
        raise RuntimeError("mqtt exploded")

    before = conversation_cache.size()
    with patch("app.core.tool_registry.tool_registry.has_tool", return_value=False), \
         patch("app.services.node_command_service.dispatch_node_command", new=_boom):
        _run([{"command": "a", "args": {}, "label": "A"}])
    # no leaked entries: size returns to baseline
    assert conversation_cache.size() == before
