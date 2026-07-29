"""Errand Runner POC chunk 3 — create_errand_plan orchestration + POST /errands.

Covers the load-bearing bits of the plan → transient-routine → draft → plan-card
flow (prds/errand-runner.md §2-§3), all mocked (no DB/MQTT/HTTP/LLM):

- the CRITICAL arg-shape inversion: Routine.steps must be MOBILE-native (args as
  [{key,value}] pairs) so the node-pull _flatten_args round-trips it, while
  ErrandPlan.steps stays NODE-native (args-as-object). Two columns, two shapes.
- the plan card is a SERVER-plane interactive card (interactive_elements with
  target:"server" → server_callback_registry), NOT node-plane.
- the endpoint's node-auth identity extraction + planner-ValueError → 422 mapping.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app.api.routines import _flatten_args
from app.deps import get_db, verify_api_key
from app.main import app
from app.models import ErrandPlan
from app.services import errand_service
from app.services.server_callback_registry import (
    ServerCallbackContext,
    registered_server_callbacks,
)


# ── helpers ──────────────────────────────────────────────────────────────────


def _fake_llm(summary: str, steps: list[dict]):
    """An llm_client whose chat_completion returns a planner-shaped response."""
    content = json.dumps({"summary": summary, "steps": steps})
    client = MagicMock()
    client.chat_completion = AsyncMock(
        return_value={"choices": [{"message": {"content": content}}]}
    )
    return client


def _mock_db():
    """A MagicMock Session whose refresh() assigns a PK, as a real flush would."""
    db = MagicMock()

    def _refresh(obj):
        if getattr(obj, "id", None) is None:
            obj.id = "pl_testid"

    db.refresh.side_effect = _refresh
    return db


def _node_ctx(household_id="hh-1", node_id="node-1"):
    ctx = MagicMock()
    ctx.household_id = household_id
    ctx.node = MagicMock()
    ctx.node.node_id = node_id
    return ctx


# ── the critical arg-shape inversion ─────────────────────────────────────────


def test_node_args_to_mobile_round_trips_through_flatten_args():
    for original in (
        {"resolved_datetimes": ["today"], "category": "sports"},  # list + scalar
        {"text": "bring an umbrella", "when": "tomorrow at 9am"},  # scalars
        {"nested": {"a": 1}},  # dict
        {},  # empty
    ):
        pairs = errand_service._node_args_to_mobile(original)
        assert all(isinstance(p["value"], str) for p in pairs)  # every value a string
        assert _flatten_args(pairs) == original  # survives the node-pull boundary


# ── the plan card is a server-plane interactive card ─────────────────────────


def test_plan_card_metadata_is_server_plane_with_errand_callbacks():
    row = ErrandPlan(
        id="pl_abc", household_id="hh-1", goal="check the weather", summary="Do the thing",
        routine_slug="errand_do_the_thing", revision=2, state="draft",
    )
    steps = [{"command": "get_weather", "args": {}, "label": "Weather"}]
    md = errand_service.build_plan_card_metadata(row, steps, "hh-1")

    # household_id must be present — mobile copies it into the /callbacks body.
    assert md["household_id"] == "hh-1"
    assert md["plan_id"] == "pl_abc"
    assert md["steps"] == steps  # structured steps for render
    # editor_schema must NOT be set (>2 disables buttons in the mobile client)
    assert "editor_schema" not in md

    els = md["interactive_elements"]
    assert [e["callback"] for e in els] == [
        "approve_errand_plan", "replan_errand_plan", "discard_errand_plan",
    ]
    for e in els:
        assert e["command"] == "errand" and e["target"] == "server"

    by_cb = {e["callback"]: e for e in els}
    # Run/Cancel carry only identity (no "goal" → unaffected by an unsaved edit)
    assert by_cb["approve_errand_plan"]["data"] == {"plan_id": "pl_abc", "revision": 2}
    assert by_cb["discard_errand_plan"]["data"] == {"plan_id": "pl_abc", "revision": 2}
    # Update SEEDS goal so the mobile merges the edited text into it
    assert by_cb["replan_errand_plan"]["data"] == {
        "plan_id": "pl_abc", "revision": 2, "goal": "check the weather",
    }
    # the editable goal field renders (generic mobile editor)
    ef = md["editable_fields"]
    assert ef == [{"label": "Goal", "initial": "check the weather",
                   "data_key": "goal", "input_type": "text", "required": True}]


# ── create_errand_plan orchestration ─────────────────────────────────────────


def _run_create(db, llm):
    with patch("app.api.routines._unique_slug", return_value="errand_test"), \
         patch("app.api.routines.publish_routines_sync") as nudge, \
         patch.object(errand_service, "_resolve_node_menu", new=AsyncMock(return_value=None)), \
         patch.object(errand_service, "post_inbox_item_sync", return_value="inbox-1") as card:
        row = asyncio.run(
            errand_service.create_errand_plan(
                db, "hh-1", "node-1", "check weather and remind me", user_id=7,
                llm_client=llm,
            )
        )
    return row, nudge, card


def test_create_errand_plan_persists_both_step_shapes_and_nudges():
    db = _mock_db()
    llm = _fake_llm(
        "Weather then reminder",
        [
            {"command": "get_weather", "args": {"resolved_datetimes": ["today"]}, "label": "Weather"},
            {"command": "set_reminder", "args": {"text": "umbrella", "when": "9am"}, "label": "Reminder"},
        ],
    )
    row, nudge, card = _run_create(db, llm)

    added = [c.args[0] for c in db.add.call_args_list]
    routine = next(a for a in added if a.__class__.__name__ == "Routine")
    plan = next(a for a in added if isinstance(a, ErrandPlan))

    # Routine.steps is MOBILE-native: args are [{key,value}] pairs and round-trip.
    r_steps = json.loads(routine.steps)
    assert r_steps[0]["command"] == "get_weather"
    assert r_steps[0]["args"] == [{"key": "resolved_datetimes", "value": '["today"]'}]
    assert _flatten_args(r_steps[0]["args"]) == {"resolved_datetimes": ["today"]}
    assert routine.enabled is True and json.loads(routine.trigger_phrases) == []

    # ErrandPlan.steps is NODE-native: args-as-object (the opposite shape).
    p_steps = json.loads(plan.steps)
    assert p_steps[0]["args"] == {"resolved_datetimes": ["today"]}
    assert plan.state == "draft" and plan.routine_slug == "errand_test"
    assert plan.user_id == 7 and plan.node_id == "node-1"

    nudge.assert_called_once_with("hh-1", db)  # node pre-pull nudge fired
    # card pushed to the initiating user with the plan_id assigned on refresh
    assert card.call_args.kwargs["user_id"] == 7
    assert card.call_args.kwargs["target_type"] == "user"
    assert card.call_args.kwargs["metadata"]["plan_id"] == row.id == "pl_testid"


def test_create_errand_plan_commits_draft_before_card_is_posted():
    """The draft must be committed even if the card push later fails."""
    db = _mock_db()
    llm = _fake_llm("W", [{"command": "get_weather", "args": {}, "label": "W"}])
    with patch("app.api.routines._unique_slug", return_value="errand_test"), \
         patch("app.api.routines.publish_routines_sync"), \
         patch.object(errand_service, "_resolve_node_menu", new=AsyncMock(return_value=None)), \
         patch.object(errand_service, "post_inbox_item_sync", side_effect=RuntimeError("boom")):
        row = asyncio.run(
            errand_service.create_errand_plan(db, "hh-1", "node-1", "weather", llm_client=llm)
        )
    db.commit.assert_called_once()  # committed once, before the (failing) card
    assert row.state == "draft"  # errand survives a dead card


# ── POST /errands endpoint ───────────────────────────────────────────────────


def test_errands_endpoint_happy_path():
    fake_row = SimpleNamespace(
        id="pl_xyz", state="draft", summary="Weather", routine_slug="errand_weather",
        steps='[{"command": "get_weather", "args": {}, "label": "W"}]',
    )
    app.dependency_overrides[get_db] = lambda: MagicMock()
    app.dependency_overrides[verify_api_key] = lambda: _node_ctx(node_id="node-9")
    try:
        with patch(
            "app.api.errands.create_errand_plan",
            new=AsyncMock(return_value=fake_row),
        ) as create:
            resp = TestClient(app).post(
                "/api/v0/errands", json={"goal": "what's the weather", "user_id": 7}
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["errand_plan_id"] == "pl_xyz"
        assert body["state"] == "draft"
        assert body["steps"][0]["command"] == "get_weather"
        # identity comes from the validated node row, not the payload
        assert create.call_args.args[1] == "hh-1"  # household_id
        assert create.call_args.args[2] == "node-9"  # node_id
        assert create.call_args.kwargs["user_id"] == 7
    finally:
        app.dependency_overrides.clear()


def test_errands_endpoint_empty_goal_422():
    app.dependency_overrides[get_db] = lambda: MagicMock()
    app.dependency_overrides[verify_api_key] = lambda: _node_ctx()
    try:
        resp = TestClient(app).post("/api/v0/errands", json={"goal": "   "})
        assert resp.status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_errands_endpoint_maps_planner_valueerror_to_422():
    app.dependency_overrides[get_db] = lambda: MagicMock()
    app.dependency_overrides[verify_api_key] = lambda: _node_ctx()
    try:
        with patch(
            "app.api.errands.create_errand_plan",
            new=AsyncMock(side_effect=ValueError("no usable steps")),
        ):
            resp = TestClient(app).post("/api/v0/errands", json={"goal": "do a barrel roll"})
        assert resp.status_code == 422
        assert "Couldn't plan that" in resp.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_errands_endpoint_403_when_node_has_no_household():
    ctx = MagicMock()
    ctx.household_id = ""
    ctx.node = None
    app.dependency_overrides[get_db] = lambda: MagicMock()
    app.dependency_overrides[verify_api_key] = lambda: ctx
    try:
        resp = TestClient(app).post("/api/v0/errands", json={"goal": "weather"})
        assert resp.status_code == 403
    finally:
        app.dependency_overrides.clear()


def test_errands_endpoint_node_override_404_if_not_in_household():
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = None  # node not found
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[verify_api_key] = lambda: _node_ctx(node_id="node-1")
    try:
        with patch("app.api.errands.create_errand_plan", new=AsyncMock()) as create:
            resp = TestClient(app).post(
                "/api/v0/errands", json={"goal": "weather", "node_id": "ghost-node"}
            )
        assert resp.status_code == 404
        create.assert_not_called()
    finally:
        app.dependency_overrides.clear()


def test_errands_endpoint_node_override_targets_validated_node():
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = SimpleNamespace(node_id="dev-node")
    fake_row = SimpleNamespace(id="pl_a", state="draft", summary="s", routine_slug="e_a", steps="[]")
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[verify_api_key] = lambda: _node_ctx(node_id="node-1")
    try:
        with patch("app.api.errands.create_errand_plan", new=AsyncMock(return_value=fake_row)) as create:
            resp = TestClient(app).post(
                "/api/v0/errands", json={"goal": "weather", "node_id": "dev-node"}
            )
        assert resp.status_code == 200
        assert create.call_args.args[2] == "dev-node"  # override honored, household-validated
    finally:
        app.dependency_overrides.clear()


# ── chunk 4: Run / Cancel server callbacks ───────────────────────────────────


def _plan_row(state="draft", revision=1):
    return ErrandPlan(
        id="pl_x", household_id="hh-1", state=state, revision=revision,
        routine_slug="errand_x", node_id="node-1", user_id=7, goal="g", summary="Do it",
    )


def _ctx(data):
    return ServerCallbackContext(job_id="j1", household_id="hh-1", user_id=7, data=data)


def _handler_db(first_results):
    db = MagicMock()
    q = db.query.return_value.filter.return_value
    q.first.side_effect = list(first_results)
    q.delete.return_value = 1
    return db


def test_approve_runs_errand_and_reuses_completion_card():
    row = _plan_row()
    routine = SimpleNamespace(slug="errand_x")
    db = _handler_db([row, routine])
    with patch("app.db.get_session_local", return_value=lambda: db), \
         patch("app.api.routines.execute_routine_on_node",
               new=AsyncMock(return_value={"status": "success"})) as ex, \
         patch("app.api.routines.publish_routines_sync"):
        res = asyncio.run(errand_service._handle_approve_errand_plan(
            _ctx({"plan_id": "pl_x", "revision": 1})))
    assert res.success is True
    assert res.context_data is None  # no second card — execute posts the completion card
    ex.assert_awaited_once()
    args, kwargs = ex.call_args
    assert args[1] == "hh-1" and args[2] is routine and args[3] == "node-1" and args[4] == "errand"
    assert kwargs["notify_on_complete"] is True and kwargs["notify_user_id"] == 7
    assert row.state == "done"
    db.query.return_value.filter.return_value.delete.assert_called()  # transient routine cleaned up
    db.close.assert_called_once()


def test_approve_reflects_nonsuccess_status():
    row = _plan_row()
    db = _handler_db([row, SimpleNamespace(slug="errand_x")])
    with patch("app.db.get_session_local", return_value=lambda: db), \
         patch("app.api.routines.execute_routine_on_node",
               new=AsyncMock(return_value={"status": "timeout"})), \
         patch("app.api.routines.publish_routines_sync"):
        asyncio.run(errand_service._handle_approve_errand_plan(
            _ctx({"plan_id": "pl_x", "revision": 1})))
    assert row.state == "timeout"


def test_approve_marks_failed_when_execute_raises():
    row = _plan_row()
    db = _handler_db([row, SimpleNamespace(slug="errand_x")])
    with patch("app.db.get_session_local", return_value=lambda: db), \
         patch("app.api.routines.execute_routine_on_node",
               new=AsyncMock(side_effect=RuntimeError("mqtt down"))), \
         patch("app.api.routines.publish_routines_sync"):
        res = asyncio.run(errand_service._handle_approve_errand_plan(
            _ctx({"plan_id": "pl_x", "revision": 1})))
    assert res.success is False
    assert row.state == "failed"  # never stuck on "running"
    db.query.return_value.filter.return_value.delete.assert_called()  # routine cleaned up even on failure
    db.close.assert_called_once()


def test_approve_rejects_stale_revision():
    row = _plan_row(revision=1)
    db = _handler_db([row])
    with patch("app.db.get_session_local", return_value=lambda: db), \
         patch("app.api.routines.execute_routine_on_node", new=AsyncMock()) as ex:
        res = asyncio.run(errand_service._handle_approve_errand_plan(
            _ctx({"plan_id": "pl_x", "revision": 2})))  # stale card
    assert res.success is False and "updated" in res.error
    ex.assert_not_awaited()
    assert row.state == "draft"  # untouched


def test_approve_noop_when_already_terminal():
    row = _plan_row(state="done")
    db = _handler_db([row])
    with patch("app.db.get_session_local", return_value=lambda: db), \
         patch("app.api.routines.execute_routine_on_node", new=AsyncMock()) as ex:
        res = asyncio.run(errand_service._handle_approve_errand_plan(
            _ctx({"plan_id": "pl_x", "revision": 1})))
    assert res.success is True and res.context_data["inbox"]  # friendly no-op card
    ex.assert_not_awaited()


def test_approve_fails_when_transient_routine_missing():
    row = _plan_row()
    db = _handler_db([row, None])  # ErrandPlan found, Routine gone
    with patch("app.db.get_session_local", return_value=lambda: db), \
         patch("app.api.routines.execute_routine_on_node", new=AsyncMock()) as ex:
        res = asyncio.run(errand_service._handle_approve_errand_plan(
            _ctx({"plan_id": "pl_x", "revision": 1})))
    assert res.success is False
    assert row.state == "failed"
    ex.assert_not_awaited()


def test_discard_cancels_draft_and_deletes_routine():
    row = _plan_row()
    db = _handler_db([row])
    with patch("app.db.get_session_local", return_value=lambda: db), \
         patch("app.api.routines.publish_routines_sync") as nudge:
        res = errand_service._handle_discard_errand_plan(_ctx({"plan_id": "pl_x"}))
    assert res.success is True
    assert row.state == "cancelled"
    db.query.return_value.filter.return_value.delete.assert_called()
    nudge.assert_called_once()


def test_register_errand_callbacks_registers_all_pairs():
    errand_service.register_errand_callbacks()
    pairs = registered_server_callbacks()
    assert ("errand", "approve_errand_plan") in pairs
    assert ("errand", "replan_errand_plan") in pairs
    assert ("errand", "discard_errand_plan") in pairs


# ── edit / re-plan handler (edit goal → re-plan → refreshed card) ────────────


def test_replan_updates_row_and_routine_and_bumps_revision():
    row = _plan_row(state="draft", revision=1)  # goal="g", summary="Do it"
    routine = SimpleNamespace(slug="errand_x", steps="[]", name="old")
    db = _handler_db([row, routine])
    plan = SimpleNamespace(
        summary="Sports news",
        routine_steps=lambda: [{"command": "get_news", "args": {"category": "sports"}, "label": "News"}],
    )
    with patch("app.db.get_session_local", return_value=lambda: db), \
         patch("app.core.llm_proxy_client.LLMProxyClient", return_value=MagicMock()), \
         patch.object(errand_service, "_resolve_node_menu", new=AsyncMock(return_value=None)), \
         patch.object(errand_service, "plan_errand", new=AsyncMock(return_value=plan)), \
         patch("app.api.routines.publish_routines_sync") as nudge, \
         patch.object(errand_service, "post_inbox_item_sync") as card:
        res = asyncio.run(errand_service._handle_replan_errand_plan(
            _ctx({"plan_id": "pl_x", "revision": 1, "goal": "get me the sports news"})))
    assert res.success is True
    assert row.goal == "get me the sports news"
    assert row.summary == "Sports news"
    assert row.revision == 2  # bumped → old card's Run is now stale
    assert json.loads(row.steps)[0]["command"] == "get_news"  # node-native args-as-object
    # transient routine updated IN PLACE with mobile-native (args-as-pairs) steps
    assert json.loads(routine.steps)[0]["args"] == [{"key": "category", "value": "sports"}]
    nudge.assert_called_once()
    card.assert_called_once()  # refreshed plan card re-posted
    md = card.call_args.kwargs["metadata"]
    assert md["revision"] == 2 and md["editable_fields"][0]["initial"] == "get me the sports news"


def test_replan_rejects_empty_goal():
    db = MagicMock()
    with patch("app.db.get_session_local", return_value=lambda: db):
        res = asyncio.run(errand_service._handle_replan_errand_plan(
            _ctx({"plan_id": "pl_x", "revision": 1, "goal": "   "})))
    assert res.success is False and "empty" in res.error.lower()


def test_replan_rejects_stale_revision():
    row = _plan_row(revision=3)
    db = _handler_db([row])
    with patch("app.db.get_session_local", return_value=lambda: db), \
         patch.object(errand_service, "plan_errand", new=AsyncMock()) as pe:
        res = asyncio.run(errand_service._handle_replan_errand_plan(
            _ctx({"plan_id": "pl_x", "revision": 1, "goal": "new goal"})))
    assert res.success is False and "updated" in res.error
    pe.assert_not_awaited()
    assert row.revision == 3  # untouched


def test_replan_error_when_planner_fails():
    row = _plan_row(revision=1)
    db = _handler_db([row])
    with patch("app.db.get_session_local", return_value=lambda: db), \
         patch("app.core.llm_proxy_client.LLMProxyClient", return_value=MagicMock()), \
         patch.object(errand_service, "_resolve_node_menu", new=AsyncMock(return_value=None)), \
         patch.object(errand_service, "plan_errand",
                      new=AsyncMock(side_effect=ValueError("no usable steps"))):
        res = asyncio.run(errand_service._handle_replan_errand_plan(
            _ctx({"plan_id": "pl_x", "revision": 1, "goal": "gibberish"})))
    assert res.success is False and "couldn't" in res.error.lower()
    assert row.revision == 1 and row.state == "draft"  # unchanged on failure


def test_replan_friendly_error_on_infra_failure():
    """A non-ValueError (LLM proxy down) → friendly retryable message, not a raw
    exception, and the old plan is left intact."""
    row = _plan_row(revision=1)
    db = _handler_db([row])
    with patch("app.db.get_session_local", return_value=lambda: db), \
         patch("app.core.llm_proxy_client.LLMProxyClient", return_value=MagicMock()), \
         patch.object(errand_service, "_resolve_node_menu", new=AsyncMock(return_value=None)), \
         patch.object(errand_service, "plan_errand",
                      new=AsyncMock(side_effect=RuntimeError("proxy unreachable"))):
        res = asyncio.run(errand_service._handle_replan_errand_plan(
            _ctx({"plan_id": "pl_x", "revision": 1, "goal": "news"})))
    assert res.success is False and "snag" in res.error.lower()
    assert row.revision == 1 and row.state == "draft"  # untouched


# ── dynamic planner menu: plan over the node's REAL commands ─────────────────


def test_resolve_node_menu_fetches_and_builds():
    report = {"available_commands": [
        {"command_name": "get_weather", "description": "w", "parameters": []},
        {"command_name": "chat", "description": "c"},  # denied
    ]}
    with patch("app.api.node_tools._request_tools_from_node", new=AsyncMock(return_value=report)):
        menu = asyncio.run(errand_service._resolve_node_menu("node-1"))
    assert [c["command"] for c in menu] == ["get_weather"]  # filtered to a real usable command


def test_resolve_node_menu_none_on_fetch_failure():
    with patch("app.api.node_tools._request_tools_from_node",
               new=AsyncMock(side_effect=RuntimeError("mqtt timeout"))):
        assert asyncio.run(errand_service._resolve_node_menu("node-1")) is None  # → default menu


def test_resolve_node_menu_none_when_node_silent():
    with patch("app.api.node_tools._request_tools_from_node", new=AsyncMock(return_value=None)):
        assert asyncio.run(errand_service._resolve_node_menu("node-1")) is None


def test_create_errand_plan_uses_passed_menu_without_fetching():
    """The voice path passes a menu built from the live conversation — create must
    plan over it and NOT do the MQTT fetch."""
    db = _mock_db()
    # the planned step's command is ONLY in the custom menu (not COMMAND_MENU)
    llm = _fake_llm("Custom", [{"command": "brew_coffee", "args": {"strength": "strong"}, "label": "Coffee"}])
    custom_menu = [{"command": "brew_coffee", "description": "Make coffee", "args": {"strength": "how strong"}}]
    with patch("app.api.routines._unique_slug", return_value="errand_test"), \
         patch("app.api.routines.publish_routines_sync"), \
         patch.object(errand_service, "_resolve_node_menu", new=AsyncMock()) as resolve, \
         patch.object(errand_service, "post_inbox_item_sync"):
        row = asyncio.run(errand_service.create_errand_plan(
            db, "hh-1", "node-1", "make me a strong coffee", user_id=1,
            menu=custom_menu, llm_client=llm))
    resolve.assert_not_awaited()  # menu supplied → no MQTT round-trip
    assert json.loads(row.steps)[0]["command"] == "brew_coffee"  # planned over the node's menu


# ── chunk 6: detached draft (for the run_errand voice tool) ──────────────────


def test_draft_detached_opens_own_session_and_drafts():
    db = MagicMock()
    with patch("app.db.get_session_local", return_value=lambda: db), \
         patch.object(errand_service, "create_errand_plan", new=AsyncMock()) as create:
        asyncio.run(errand_service.draft_errand_plan_detached("hh-1", "node-1", "weather", 7))
    create.assert_awaited_once_with(db, "hh-1", "node-1", "weather", user_id=7, menu=None)
    db.close.assert_called_once()  # own session, always closed


def test_draft_detached_posts_couldnt_plan_card_on_valueerror():
    db = MagicMock()
    with patch("app.db.get_session_local", return_value=lambda: db), \
         patch.object(errand_service, "create_errand_plan",
                      new=AsyncMock(side_effect=ValueError("no usable steps"))), \
         patch.object(errand_service, "post_inbox_item_sync") as card:
        asyncio.run(errand_service.draft_errand_plan_detached("hh-1", "node-1", "gibberish", 7))
    card.assert_called_once()  # user isn't left hanging after the spoken ack
    assert card.call_args.kwargs["household_id"] == "hh-1"
    assert card.call_args.kwargs["user_id"] == 7
    db.close.assert_called_once()


def test_draft_detached_posts_card_on_infra_failure():
    """A non-ValueError (LLM proxy down, DB error) must ALSO surface a card —
    never vanish after the spoken ack."""
    db = MagicMock()
    with patch("app.db.get_session_local", return_value=lambda: db), \
         patch.object(errand_service, "create_errand_plan",
                      new=AsyncMock(side_effect=RuntimeError("planner proxy unreachable"))), \
         patch.object(errand_service, "post_inbox_item_sync") as card:
        asyncio.run(errand_service.draft_errand_plan_detached("hh-1", "node-1", "weather", 7))
    card.assert_called_once()  # infra failure still posts a card
    assert "snag" in card.call_args.kwargs["summary"].lower()
    db.close.assert_called_once()
