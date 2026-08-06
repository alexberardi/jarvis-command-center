"""Errand Runner — create_errand_plan orchestration + per-step execution + POST /errands.

Covers the load-bearing bits of the plan → draft → plan-card → per-step-dispatch
flow (prds/errand-runner.md §2-§3), all mocked (no DB/MQTT/HTTP/LLM):

- ErrandPlan.steps is NODE-native (args-as-object) and IS the plan — there is no
  transient routine; the per-step executor dispatches each step to its plane
  (node command over MQTT / server tool in-process).
- the plan card is a SERVER-plane interactive card (interactive_elements with
  target:"server" → server_callback_registry), NOT node-plane.
- the endpoint's node-auth identity extraction + planner-ValueError → 422 mapping.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app.deps import get_db, verify_api_key
from app.main import app
from app.models import ErrandPlan, Workflow
from app.services import errand_service
from app.services.server_callback_registry import (
    ServerCallbackContext,
    registered_server_callbacks,
)


# ── helpers ──────────────────────────────────────────────────────────────────


class _FakeQuery:
    """A model-scoped query over ``_FakeDB``. Filter criteria are ignored (tests
    hold one row per model); ``first``/``all``/``update`` resolve by model."""

    def __init__(self, db: "_FakeDB", model):
        self._db = db
        self._model = model

    def filter(self, *a, **k):
        return self

    def filter_by(self, **k):
        return self

    def order_by(self, *a, **k):
        return self

    def first(self):
        rows = self._db.rows(self._model)
        return rows[0] if rows else None

    def all(self):
        return list(self._db.rows(self._model))

    def update(self, values):
        rows = self._db.rows(self._model)
        for r in rows:
            for key, val in values.items():
                setattr(r, key, val)
        return len(rows)


class _FakeDB:
    """A tiny in-memory DB that dispatches queries BY MODEL — the run/instance split
    means the durable path touches two tables (ErrandPlan draft + Workflow run), so
    a single-``.first()`` MagicMock no longer suffices. ``flush`` assigns a Workflow
    id (mimicking the column default) so the run can be loaded back by id."""

    def __init__(self, rows=None):
        self._store: dict[str, list] = {}
        for r in rows or []:
            self._add(r)
        self.closed = False

    @staticmethod
    def _key(model_or_obj):
        m = model_or_obj if isinstance(model_or_obj, type) else type(model_or_obj)
        return getattr(m, "__name__", str(m))

    def rows(self, model):
        return self._store.get(self._key(model), [])

    def _add(self, obj):
        self._store.setdefault(self._key(obj), []).append(obj)

    def query(self, model):
        return _FakeQuery(self, model)

    def add(self, obj):
        self._add(obj)

    def flush(self):
        for lst in self._store.values():
            for r in lst:
                if getattr(r, "id", None) is None:
                    r.id = f"wf_{json.dumps(id(r))}"

    def commit(self):
        pass

    def rollback(self):
        pass

    def refresh(self, obj):
        pass

    def close(self):
        self.closed = True


def _wf_row(state="running", cursor=0, steps=None, results="[]", household_id="hh-1",
            user_id=7, node_id="node-1", goal="g", title="Do it", wf_id="wf_1"):
    return Workflow(
        id=wf_id, kind="errand", household_id=household_id, user_id=user_id,
        node_id=node_id, goal=goal, title=title,
        steps=json.dumps(steps or [{"command": "get_weather", "args": {}, "label": "W"}]),
        cursor=cursor, results_json=results, state=state,
    )


def _sqlite_sessionmaker():
    """A real in-memory SQLite sessionmaker (shared connection) so the durable-path
    tests exercise the ACTUAL SQL WHERE clauses — the atomic claim guard, the sweep
    correlations — which the criteria-ignoring _FakeDB can't."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.models import Base

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


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
        "approve_errand_plan", "refine_errand_plan", "discard_errand_plan",
    ]
    for e in els:
        assert e["command"] == "errand" and e["target"] == "server"

    by_cb = {e["callback"]: e for e in els}
    # Run/Cancel carry only identity (no "instruction" → unaffected by an unsent edit)
    assert by_cb["approve_errand_plan"]["data"] == {"plan_id": "pl_abc", "revision": 2}
    assert by_cb["discard_errand_plan"]["data"] == {"plan_id": "pl_abc", "revision": 2}
    # Revise SEEDS an empty instruction so the mobile merges the typed change into it
    assert by_cb["refine_errand_plan"]["data"] == {
        "plan_id": "pl_abc", "revision": 2, "instruction": "",
    }
    # the "tell it what to change" field renders (generic mobile editor)
    ef = md["editable_fields"]
    assert ef == [{"label": "Tell me what to change", "initial": "",
                   "data_key": "instruction", "input_type": "text", "required": True}]


# ── plan card: in-place update vs post (so Revise refreshes, not duplicates) ──

_STEPS = [{"command": "get_weather", "args": {}, "label": "W"}]


def test_post_card_updates_in_place_when_id_exists():
    row = _plan_row()
    row.inbox_item_id = "card-123"
    with patch.object(errand_service, "update_inbox_item_sync", return_value=True) as upd, \
         patch.object(errand_service, "post_inbox_item_sync") as post:
        cid = errand_service._post_errand_plan_card(row, _STEPS, "hh-1", 7)
    upd.assert_called_once()
    assert upd.call_args.kwargs["item_id"] == "card-123"
    post.assert_not_called()  # updated in place, NOT duplicated
    assert cid == "card-123"


def test_post_card_falls_back_to_post_when_update_fails():
    row = _plan_row()
    row.inbox_item_id = "gone-card"  # e.g. the user deleted it → 404
    with patch.object(errand_service, "update_inbox_item_sync", return_value=False), \
         patch.object(errand_service, "post_inbox_item_sync", return_value="new-card") as post:
        cid = errand_service._post_errand_plan_card(row, _STEPS, "hh-1", 7)
    post.assert_called_once()
    assert cid == "new-card"


def test_post_card_posts_new_when_no_id():
    row = _plan_row()  # inbox_item_id is None on a fresh plan
    with patch.object(errand_service, "update_inbox_item_sync") as upd, \
         patch.object(errand_service, "post_inbox_item_sync", return_value="fresh") as post:
        cid = errand_service._post_errand_plan_card(row, _STEPS, "hh-1", 7)
    upd.assert_not_called()
    post.assert_called_once()
    assert cid == "fresh"


# ── create_errand_plan orchestration ─────────────────────────────────────────


def _run_create(db, llm):
    with patch.object(errand_service, "_resolve_node_menu", new=AsyncMock(return_value=None)), \
         patch.object(errand_service, "post_inbox_item_sync", return_value="inbox-1") as card:
        row = asyncio.run(
            errand_service.create_errand_plan(
                db, "hh-1", "node-1", "check weather and remind me", user_id=7,
                llm_client=llm,
            )
        )
    return row, card


def test_create_errand_plan_persists_draft_and_posts_card():
    db = _mock_db()
    llm = _fake_llm(
        "Weather then reminder",
        [
            {"command": "get_weather", "args": {"resolved_datetimes": ["today"]}, "label": "Weather"},
            {"command": "set_reminder", "args": {"text": "umbrella", "when": "9am"}, "label": "Reminder"},
        ],
    )
    row, card = _run_create(db, llm)

    added = [c.args[0] for c in db.add.call_args_list]
    # No transient Routine any more — only the ErrandPlan draft is persisted.
    assert all(a.__class__.__name__ != "Routine" for a in added)
    plan = next(a for a in added if isinstance(a, ErrandPlan))

    # ErrandPlan.steps is NODE-native (args-as-object) — the single execution source.
    p_steps = json.loads(plan.steps)
    assert p_steps[0]["command"] == "get_weather"
    assert p_steps[0]["args"] == {"resolved_datetimes": ["today"]}
    assert plan.state == "draft"
    assert plan.user_id == 7 and plan.node_id == "node-1"

    # card pushed to the initiating user with the plan_id assigned on refresh
    assert card.call_args.kwargs["user_id"] == 7
    assert card.call_args.kwargs["target_type"] == "user"
    assert card.call_args.kwargs["metadata"]["plan_id"] == row.id == "pl_testid"


def test_create_errand_plan_commits_draft_before_card_is_posted():
    """The draft must be committed even if the card push later fails."""
    db = _mock_db()
    llm = _fake_llm("W", [{"command": "get_weather", "args": {}, "label": "W"}])
    with patch.object(errand_service, "_resolve_node_menu", new=AsyncMock(return_value=None)), \
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


def _ok_result(status="success", message="Sunny."):
    return {"status": status, "message": message, "passed": 1, "failed": 0, "results": []}


def test_approve_spawns_workflow_run_and_posts_completion_card():
    """Run tap → a Workflow RUN is spawned from the draft (run/instance split), the
    draft is 'launched' + linked, and the engine drives the run to a completion card."""
    row = _plan_row()
    row.steps = json.dumps([{"command": "get_weather", "args": {}, "label": "W"}])
    db = _FakeDB([row])
    with patch("app.db.get_session_local", return_value=lambda: db), \
         patch("app.services.errand_executor.execute_errand",
               new=AsyncMock(return_value=_ok_result())) as ex, \
         patch.object(errand_service, "post_inbox_item_sync") as card:
        res = asyncio.run(errand_service._handle_approve_errand_plan(
            _ctx({"plan_id": "pl_x", "revision": 1})))
    assert res.success is True
    # draft → launched, linked to the spawned run
    assert row.state == "launched" and row.workflow_id
    wf = db.rows(Workflow)[0]
    assert wf.id == row.workflow_id and wf.kind == "errand"
    # the engine ran the plan on the run
    ex.assert_awaited_once()
    args, kwargs = ex.call_args
    assert args[0] == "hh-1" and args[1] == "node-1"  # household, node
    assert args[2] == [{"command": "get_weather", "args": {}, "label": "W"}]  # steps threaded through
    assert kwargs["user_id"] == 7 and kwargs["errand_id"] == wf.id  # linked to the RUN, not the draft
    assert wf.state == "done"  # run state lives on the workflow now
    card.assert_called_once()  # exactly one completion card
    assert card.call_args.kwargs["category"] == "errand"
    assert card.call_args.kwargs["user_id"] == 7


def test_approve_reflects_partial_status_on_the_run():
    row = _plan_row()
    row.steps = json.dumps([{"command": "get_weather", "args": {}, "label": "W"}])
    db = _FakeDB([row])
    with patch("app.db.get_session_local", return_value=lambda: db), \
         patch("app.services.errand_executor.execute_errand",
               new=AsyncMock(return_value=_ok_result("partial", "one step failed"))), \
         patch.object(errand_service, "post_inbox_item_sync"):
        asyncio.run(errand_service._handle_approve_errand_plan(
            _ctx({"plan_id": "pl_x", "revision": 1})))
    assert row.state == "launched"          # the draft is launched
    assert db.rows(Workflow)[0].state == "partial"  # honest non-success terminal on the run


def test_approve_marks_run_failed_and_cards_when_execute_raises():
    row = _plan_row()
    row.steps = json.dumps([{"command": "get_weather", "args": {}, "label": "W"}])
    db = _FakeDB([row])
    with patch("app.db.get_session_local", return_value=lambda: db), \
         patch("app.services.errand_executor.execute_errand",
               new=AsyncMock(side_effect=RuntimeError("boom"))), \
         patch.object(errand_service, "post_inbox_item_sync") as card:
        res = asyncio.run(errand_service._handle_approve_errand_plan(
            _ctx({"plan_id": "pl_x", "revision": 1})))
    # the tap SUCCEEDED in launching the errand; the run's failure is on its own card
    assert res.success is True
    assert row.state == "launched"
    assert db.rows(Workflow)[0].state == "failed"  # run never stuck on "running"
    card.assert_called_once()  # anti-vanishing: a failure card even on a raised error


def test_approve_rejects_stale_revision():
    row = _plan_row(revision=1)
    db = _handler_db([row])
    with patch("app.db.get_session_local", return_value=lambda: db), \
         patch("app.services.errand_executor.execute_errand", new=AsyncMock()) as ex:
        res = asyncio.run(errand_service._handle_approve_errand_plan(
            _ctx({"plan_id": "pl_x", "revision": 2})))  # stale card
    assert res.success is False and "updated" in res.error
    ex.assert_not_awaited()
    assert row.state == "draft"  # untouched


def test_approve_noop_when_already_terminal():
    row = _plan_row(state="done")
    db = _handler_db([row])
    with patch("app.db.get_session_local", return_value=lambda: db), \
         patch("app.services.errand_executor.execute_errand", new=AsyncMock()) as ex:
        res = asyncio.run(errand_service._handle_approve_errand_plan(
            _ctx({"plan_id": "pl_x", "revision": 1})))
    assert res.success is True and res.context_data["inbox"]  # friendly no-op card
    ex.assert_not_awaited()


def test_approve_fails_when_no_steps():
    row = _plan_row()
    row.steps = "[]"  # nothing to run
    db = _handler_db([row])
    with patch("app.db.get_session_local", return_value=lambda: db), \
         patch("app.services.errand_executor.execute_errand", new=AsyncMock()) as ex:
        res = asyncio.run(errand_service._handle_approve_errand_plan(
            _ctx({"plan_id": "pl_x", "revision": 1})))
    assert res.success is False and row.state == "failed"
    ex.assert_not_awaited()


def test_discard_cancels_draft():
    row = _plan_row()
    db = _handler_db([row])
    with patch("app.db.get_session_local", return_value=lambda: db):
        res = errand_service._handle_discard_errand_plan(_ctx({"plan_id": "pl_x"}))
    assert res.success is True
    assert row.state == "cancelled"
    assert res.context_data["inbox"]["title"].startswith("🗑️")


def test_register_errand_callbacks_registers_all_pairs():
    errand_service.register_errand_callbacks()
    pairs = registered_server_callbacks()
    assert ("errand", "approve_errand_plan") in pairs
    assert ("errand", "refine_errand_plan") in pairs
    assert ("errand", "replan_errand_plan") in pairs
    assert ("errand", "discard_errand_plan") in pairs


# ── refine handler: "tell it what to change" → revised plan ──────────────────


def test_refine_updates_plan_from_instruction_and_keeps_goal():
    row = _plan_row(state="draft", revision=1)  # goal="g", summary="Do it"
    row.steps = json.dumps([{"command": "get_news", "args": {}, "label": "News"}])
    db = _handler_db([row])
    plan = SimpleNamespace(
        summary="Weather instead",
        routine_steps=lambda: [{"command": "get_weather", "args": {"resolved_datetimes": ["today"]}, "label": "W"}],
    )
    with patch("app.db.get_session_local", return_value=lambda: db), \
         patch("app.core.llm_proxy_client.LLMProxyClient", return_value=MagicMock()), \
         patch.object(errand_service, "_resolve_node_menu", new=AsyncMock(return_value=None)), \
         patch.object(errand_service, "refine_errand_plan", new=AsyncMock(return_value=plan)) as refine, \
         patch.object(errand_service, "post_inbox_item_sync") as card:
        res = asyncio.run(errand_service._handle_refine_errand_plan(
            _ctx({"plan_id": "pl_x", "revision": 1, "instruction": "give me the weather, not the news"})))
    assert res.success is True
    assert row.summary == "Weather instead" and row.revision == 2
    assert row.goal == "g"  # a refine does NOT rewrite the goal
    # ErrandPlan.steps updated in place, NODE-native (args-as-object) — the exec source.
    assert json.loads(row.steps)[0]["command"] == "get_weather"
    assert json.loads(row.steps)[0]["args"] == {"resolved_datetimes": ["today"]}
    # refine got the CURRENT plan (summary + steps) + the instruction
    args = refine.call_args.args
    assert args[0] == "Do it" and args[1] == [{"command": "get_news", "args": {}, "label": "News"}]
    assert args[2] == "give me the weather, not the news"
    card.assert_called_once()  # refreshed plan card re-posted (no routine, no nudge)


def test_refine_rejects_empty_instruction():
    db = MagicMock()
    with patch("app.db.get_session_local", return_value=lambda: db):
        res = asyncio.run(errand_service._handle_refine_errand_plan(
            _ctx({"plan_id": "pl_x", "revision": 1, "instruction": "   "})))
    assert res.success is False and "what to change" in res.error.lower()


def test_refine_rejects_stale_revision():
    row = _plan_row(revision=3)
    db = _handler_db([row])
    with patch("app.db.get_session_local", return_value=lambda: db), \
         patch.object(errand_service, "refine_errand_plan", new=AsyncMock()) as refine:
        res = asyncio.run(errand_service._handle_refine_errand_plan(
            _ctx({"plan_id": "pl_x", "revision": 1, "instruction": "change it"})))
    assert res.success is False and "updated" in res.error
    refine.assert_not_awaited()
    assert row.revision == 3


def test_refine_friendly_error_on_infra_failure():
    row = _plan_row(revision=1)
    row.steps = "[]"
    db = _handler_db([row])
    with patch("app.db.get_session_local", return_value=lambda: db), \
         patch("app.core.llm_proxy_client.LLMProxyClient", return_value=MagicMock()), \
         patch.object(errand_service, "_resolve_node_menu", new=AsyncMock(return_value=None)), \
         patch.object(errand_service, "refine_errand_plan",
                      new=AsyncMock(side_effect=RuntimeError("proxy down"))):
        res = asyncio.run(errand_service._handle_refine_errand_plan(
            _ctx({"plan_id": "pl_x", "revision": 1, "instruction": "call instead"})))
    assert res.success is False and "snag" in res.error.lower()
    assert row.revision == 1  # untouched on failure


# ── edit / re-plan handler (edit goal → re-plan → refreshed card) ────────────


def test_replan_updates_row_and_bumps_revision():
    row = _plan_row(state="draft", revision=1)  # goal="g", summary="Do it"
    db = _handler_db([row])
    plan = SimpleNamespace(
        summary="Sports news",
        routine_steps=lambda: [{"command": "get_news", "args": {"category": "sports"}, "label": "News"}],
    )
    with patch("app.db.get_session_local", return_value=lambda: db), \
         patch("app.core.llm_proxy_client.LLMProxyClient", return_value=MagicMock()), \
         patch.object(errand_service, "_resolve_node_menu", new=AsyncMock(return_value=None)), \
         patch.object(errand_service, "plan_errand", new=AsyncMock(return_value=plan)), \
         patch.object(errand_service, "post_inbox_item_sync") as card:
        res = asyncio.run(errand_service._handle_replan_errand_plan(
            _ctx({"plan_id": "pl_x", "revision": 1, "goal": "get me the sports news"})))
    assert res.success is True
    assert row.goal == "get me the sports news"
    assert row.summary == "Sports news"
    assert row.revision == 2  # bumped → old card's Run is now stale
    # ErrandPlan.steps updated in place, node-native (args-as-object).
    assert json.loads(row.steps)[0]["command"] == "get_news"
    assert json.loads(row.steps)[0]["args"] == {"category": "sports"}
    card.assert_called_once()  # refreshed plan card re-posted
    assert card.call_args.kwargs["metadata"]["revision"] == 2  # carries the bumped revision


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


def test_resolve_node_menu_unions_node_and_server():
    report = {"available_commands": [
        {"command_name": "get_weather", "description": "w", "parameters": []},
        {"command_name": "chat", "description": "c"},  # denied
    ]}
    with patch("app.api.node_tools._request_tools_from_node", new=AsyncMock(return_value=report)), \
         patch("app.services.errand_planner.build_server_tool_menu",
               return_value=[{"command": "make_phone_call", "description": "call", "args": {}}]):
        menu = asyncio.run(errand_service._resolve_node_menu("node-1", "hh-1", 7))
    cmds = {c["command"] for c in menu}
    assert cmds == {"get_weather", "make_phone_call"}  # node command ∪ server tool, chat denied


def test_resolve_node_menu_server_only_when_node_offline():
    # Node fetch fails, but server tools run in CC — they must still be offered so a
    # phone-call errand plans even when the node is unreachable.
    with patch("app.api.node_tools._request_tools_from_node",
               new=AsyncMock(side_effect=RuntimeError("mqtt timeout"))), \
         patch("app.services.errand_planner.build_server_tool_menu",
               return_value=[{"command": "make_phone_call", "description": "call", "args": {}}]):
        menu = asyncio.run(errand_service._resolve_node_menu("node-1", "hh-1", 7))
    assert [c["command"] for c in menu] == ["make_phone_call"]


def test_resolve_node_menu_none_when_nothing_available():
    with patch("app.api.node_tools._request_tools_from_node", new=AsyncMock(return_value=None)), \
         patch("app.services.errand_planner.build_server_tool_menu", return_value=[]):
        assert asyncio.run(errand_service._resolve_node_menu("node-1", "hh-1", None)) is None


def test_create_errand_plan_uses_passed_menu_without_fetching():
    """The voice path passes a menu built from the live conversation — create must
    plan over it and NOT do the MQTT fetch."""
    db = _mock_db()
    # the planned step's command is ONLY in the custom menu (not COMMAND_MENU)
    llm = _fake_llm("Custom", [{"command": "brew_coffee", "args": {"strength": "strong"}, "label": "Coffee"}])
    custom_menu = [{"command": "brew_coffee", "description": "Make coffee", "args": {"strength": "how strong"}}]
    with patch.object(errand_service, "_resolve_node_menu", new=AsyncMock()) as resolve, \
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


# ── wait-and-decide: suspend on a phone step, resume on the call's outcome ────


def _fake_call_session(state="done", errand_id="wf_1", step=0, goal_achieved=None):
    # errand_id holds the WORKFLOW run id after the run/instance split. goal_achieved
    # gates the fail-fast: a 'done' call only continues the errand if it's True.
    return SimpleNamespace(
        id="sess-1", errand_id=errand_id, errand_step=step, state=state,
        contact_name="CVS", error_message=None, household_id="hh-1", user_id=7,
        goal_achieved=goal_achieved, outcome_json=None,
    )


def test_approve_suspends_the_run_on_a_phone_step():
    """A phone step SUSPENDS the RUN (Workflow state=waiting), no completion card —
    the phone confirm/outcome cards are the live feedback. The draft is launched."""
    from app.services.errand_executor import Suspended

    row = _plan_row()
    row.steps = json.dumps([{"command": "make_phone_call",
                             "args": {"business": "CVS", "goal": "refill"}, "label": "Call"}])
    db = _FakeDB([row])
    with patch("app.db.get_session_local", return_value=lambda: db), \
         patch("app.services.errand_executor.execute_errand",
               new=AsyncMock(return_value=Suspended(cursor=1, session_id="sess-1", results=[]))), \
         patch.object(errand_service, "post_inbox_item_sync") as card:
        res = asyncio.run(errand_service._handle_approve_errand_plan(
            _ctx({"plan_id": "pl_x", "revision": 1})))
    assert res.success is True
    assert row.state == "launched"
    wf = db.rows(Workflow)[0]
    assert wf.state == "waiting" and wf.cursor == 1  # the RUN carries wait state
    card.assert_not_called()  # no completion card while waiting on the call


def test_resume_continues_from_cursor_on_call_done():
    wf = _wf_row(state="waiting", cursor=1, steps=[
        {"command": "make_phone_call", "args": {}, "label": "Call"},
        {"command": "get_weather", "args": {}, "label": "W"}])
    db = _FakeDB([wf])
    with patch("app.db.get_session_local", return_value=lambda: db), \
         patch("app.services.errand_executor.execute_errand",
               new=AsyncMock(return_value={"status": "success", "message": "all done",
                                           "passed": 2, "failed": 0, "results": []})) as ex, \
         patch.object(errand_service, "post_inbox_item_sync") as card:
        asyncio.run(errand_service.resume_errand_after_call(
            _fake_call_session(state="done", errand_id=wf.id, step=0, goal_achieved=True)))
    ex.assert_awaited_once()
    assert ex.await_args.kwargs["start_index"] == 1  # resume from the cursor
    assert wf.state == "done"
    card.assert_called_once()  # completion card on final complete


def test_resume_fail_fast_on_call_failed():
    wf = _wf_row(state="waiting", cursor=1, steps=[
        {"command": "make_phone_call", "args": {}, "label": "Call CVS"},
        {"command": "make_phone_call", "args": {}, "label": "Call Dr"}])
    db = _FakeDB([wf])
    with patch("app.db.get_session_local", return_value=lambda: db), \
         patch("app.services.errand_executor.execute_errand", new=AsyncMock()) as ex, \
         patch("app.services.errand_executor.aggregate_and_compose",
               new=AsyncMock(return_value={"status": "failed", "message": "the pharmacy call didn't go through",
                                           "passed": 0, "failed": 1, "results": []})), \
         patch.object(errand_service, "post_inbox_item_sync") as card:
        asyncio.run(errand_service.resume_errand_after_call(
            _fake_call_session(state="failed", errand_id=wf.id, step=0)))
    ex.assert_not_awaited()  # fail-fast: the 2nd call is NEVER placed
    card.assert_called_once()  # honest completion card
    assert wf.state in ("failed", "partial")


def test_resume_fail_fast_when_call_connected_but_goal_not_achieved():
    """Regression (live 2026-07-30): the pharmacy call reached 'done' but its goal
    was NOT achieved, yet the errand still dialed Total Patient Care. A 'done' call
    STATE ≠ goal achieved — the next call must only be placed on an explicit success,
    so goal_achieved=False (and None) fail-fasts."""
    wf = _wf_row(state="waiting", cursor=1, steps=[
        {"command": "make_phone_call", "args": {}, "label": "Call pharmacy"},
        {"command": "make_phone_call", "args": {}, "label": "Call Total Patient Care"}])
    db = _FakeDB([wf])
    with patch("app.db.get_session_local", return_value=lambda: db), \
         patch("app.services.errand_executor.execute_errand", new=AsyncMock()) as ex, \
         patch("app.services.errand_executor.aggregate_and_compose",
               new=AsyncMock(return_value={"status": "failed",
                                           "message": "reached the pharmacy but the goal wasn't achieved",
                                           "passed": 0, "failed": 1, "results": []})), \
         patch.object(errand_service, "post_inbox_item_sync") as card:
        asyncio.run(errand_service.resume_errand_after_call(
            _fake_call_session(state="done", errand_id=wf.id, step=0, goal_achieved=False)))
    ex.assert_not_awaited()  # THE FIX: the 2nd call (Total Patient Care) is NOT placed
    card.assert_called_once()
    assert wf.state in ("failed", "partial")


def test_resume_noop_when_run_not_waiting():
    wf = _wf_row(state="done")  # already terminal
    db = _FakeDB([wf])
    with patch("app.db.get_session_local", return_value=lambda: db), \
         patch("app.services.errand_executor.execute_errand", new=AsyncMock()) as ex:
        asyncio.run(errand_service.resume_errand_after_call(
            _fake_call_session(state="done", errand_id=wf.id, step=0)))
    ex.assert_not_awaited()  # not waiting → nothing to resume


def test_resume_ignores_unlinked_or_nonterminal_session():
    with patch("app.db.get_session_local", return_value=lambda: MagicMock()), \
         patch("app.services.errand_executor.execute_errand", new=AsyncMock()) as ex:
        asyncio.run(errand_service.resume_errand_after_call(_fake_call_session(errand_id=None)))
        asyncio.run(errand_service.resume_errand_after_call(_fake_call_session(state="dialing")))
    ex.assert_not_awaited()


# ── the TIMER wakeup (wait_for) sweep — real SQLite so the WHERE filter is real ──


def test_timer_sweep_fires_only_due_timer_waits():
    """resume_due_timer_workflows resumes ONLY runs waiting on a timer whose wake_at
    has passed — not future timers, not phone waits. Uses a real in-memory DB so the
    SQL filter (and the real _WorkflowRunStore) are genuinely exercised."""
    from datetime import datetime, timedelta

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.models import Base

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    now = datetime.utcnow()
    two_step = json.dumps([{"command": "wait_for", "label": "Wait"},
                           {"command": "get_weather", "args": {}, "label": "W"}])
    seed = Session()
    seed.add_all([
        Workflow(id="wf_due", kind="errand", household_id="hh", goal="g", steps=two_step,
                 state="waiting", waiting_on="timer", wake_at=now - timedelta(seconds=10), cursor=1),
        Workflow(id="wf_future", kind="errand", household_id="hh", goal="g", steps="[]",
                 state="waiting", waiting_on="timer", wake_at=now + timedelta(hours=1), cursor=1),
        Workflow(id="wf_phone", kind="errand", household_id="hh", goal="g", steps="[]",
                 state="waiting", waiting_on="phone_call", wake_at=None, cursor=1),
    ])
    seed.commit()
    seed.close()

    with patch("app.db.get_session_local", return_value=Session), \
         patch("app.services.errand_executor.execute_errand",
               new=AsyncMock(return_value={"status": "success", "message": "done",
                                           "passed": 2, "failed": 0, "results": []})) as ex, \
         patch.object(errand_service, "post_inbox_item_sync"):
        fired = asyncio.run(errand_service.resume_due_timer_workflows())

    assert fired == 1                              # only the due timer wait
    ex.assert_awaited_once()
    assert ex.await_args.kwargs["start_index"] == 1  # resumed from the wait_for's cursor
    check = Session()
    try:
        assert check.query(Workflow).filter(Workflow.id == "wf_due").first().state == "done"
        assert check.query(Workflow).filter(Workflow.id == "wf_future").first().state == "waiting"
        assert check.query(Workflow).filter(Workflow.id == "wf_phone").first().state == "waiting"
    finally:
        check.close()


# ── discard on a mid-call (waiting) errand: actually cancel, don't falsely report ──


def test_discard_waiting_run_cancels_and_declines_call():
    """Cancel on a LAUNCHED errand whose RUN is mid-call (waiting) genuinely cancels
    the Workflow AND declines the pending call."""
    plan = _plan_row(state="launched")
    plan.workflow_id = "wf_1"
    wf = _wf_row(state="waiting", wf_id="wf_1")
    db = _FakeDB([plan, wf])
    with patch("app.db.get_session_local", return_value=lambda: db), \
         patch.object(errand_service, "_decline_pending_workflow_calls", return_value=True) as decline:
        res = errand_service._handle_discard_errand_plan(_ctx({"plan_id": "pl_x"}))
    assert res.success is True
    assert plan.state == "cancelled" and wf.state == "cancelled"  # both cancelled, not left running
    decline.assert_called_once()  # the pending call is declined so it won't dial
    assert decline.call_args.args[1] == "wf_1"  # declines by the RUN id
    assert "cancelled" in res.context_data["inbox"]["title"].lower()


def test_discard_finished_run_is_honest_noop():
    plan = _plan_row(state="launched")
    plan.workflow_id = "wf_1"
    wf = _wf_row(state="done", wf_id="wf_1")
    db = _FakeDB([plan, wf])
    with patch("app.db.get_session_local", return_value=lambda: db):
        res = errand_service._handle_discard_errand_plan(_ctx({"plan_id": "pl_x"}))
    assert res.success is True
    assert plan.state == "launched"  # untouched — the run already finished
    assert "already" in res.context_data["inbox"]["summary"].lower()  # honest, not "discarded"
    assert "done" in res.context_data["inbox"]["summary"].lower()
    assert wf.state == "done"  # the run is untouched


# ── durable-path coverage the adversarial review flagged (real SQLite = real WHERE) ──


def test_workflow_store_claim_is_atomic():
    """The atomic waiting→running claim (the SOLE serialization between the phone
    hook and the sweep). Two stores both load a 'waiting' run; the first claim wins
    (WHERE state='waiting' matches), the second loses (matches nothing). This
    genuinely exercises the WHERE guard, not just the pre-check."""
    Session = _sqlite_sessionmaker()
    seed = Session()
    seed.add(_wf_row(state="waiting", cursor=1, wf_id="wf_c"))
    seed.commit()
    seed.close()

    store1 = errand_service._WorkflowRunStore(Session())
    store2 = errand_service._WorkflowRunStore(Session())
    try:
        assert store1.load("wf_c").state == "waiting"
        assert store2.load("wf_c").state == "waiting"   # both observe 'waiting' first
        assert store1.claim_running("wf_c") is True      # first flips waiting→running
        assert store2.claim_running("wf_c") is False     # second: WHERE state='waiting' → 0 rows
    finally:
        store1.close()
        store2.close()
    check = Session()
    try:
        assert check.query(Workflow).filter(Workflow.id == "wf_c").first().state == "running"
    finally:
        check.close()


def test_phone_sweep_resumes_run_whose_call_is_terminal():
    """resume_waiting_errands job 1: a run 'waiting' whose linked call reached a
    terminal state (here 'declined', which skips the immediate hook) is resumed and
    fail-fasts with a completion card. Exercises the real PhoneCallSession correlation
    (errand_id==workflow.id AND errand_step==cursor-1)."""
    from datetime import datetime

    from app.models import PhoneCallSession

    Session = _sqlite_sessionmaker()
    seed = Session()
    seed.add(_wf_row(state="waiting", cursor=1, wf_id="wf_p", steps=[
        {"command": "make_phone_call", "args": {}, "label": "Call"},
        {"command": "make_phone_call", "args": {}, "label": "Call 2"}]))
    seed.add(PhoneCallSession(id="s1", household_id="hh-1", user_id=7, goal="g",
                             errand_id="wf_p", errand_step=0, state="declined",
                             created_at=datetime.utcnow()))
    seed.commit()
    seed.close()

    with patch("app.db.get_session_local", return_value=Session), \
         patch("app.services.errand_executor.execute_errand", new=AsyncMock()) as ex, \
         patch("app.services.errand_executor.aggregate_and_compose",
               new=AsyncMock(return_value={"status": "failed", "message": "the call was declined",
                                           "passed": 0, "failed": 1, "results": []})), \
         patch.object(errand_service, "post_inbox_item_sync") as card:
        resumed = asyncio.run(errand_service.resume_waiting_errands())

    assert resumed == 1
    ex.assert_not_awaited()          # fail-fast: the 2nd call is never placed
    card.assert_called_once()        # honest completion card
    check = Session()
    try:
        assert check.query(Workflow).filter(Workflow.id == "wf_p").first().state in ("failed", "partial")
    finally:
        check.close()


def test_phone_sweep_times_out_stuck_waiting_run():
    """resume_waiting_errands job 2: a run 'waiting' on a call that never terminates
    (confirmed-but-never-dialed) past the 60-min deadline is timed out (synthesized
    failure → fail-fast) so it never hangs forever."""
    from datetime import datetime, timedelta

    from app.models import PhoneCallSession

    Session = _sqlite_sessionmaker()
    seed = Session()
    seed.add(_wf_row(state="waiting", cursor=1, wf_id="wf_t", steps=[
        {"command": "make_phone_call", "args": {}, "label": "Call"}]))
    seed.add(PhoneCallSession(id="s2", household_id="hh-1", user_id=7, goal="g",
                             errand_id="wf_t", errand_step=0, state="confirmed",
                             created_at=datetime.utcnow() - timedelta(minutes=61)))
    seed.commit()
    seed.close()

    with patch("app.db.get_session_local", return_value=Session), \
         patch("app.services.errand_executor.aggregate_and_compose",
               new=AsyncMock(return_value={"status": "failed", "message": "the call wasn't completed in time",
                                           "passed": 0, "failed": 1, "results": []})), \
         patch.object(errand_service, "post_inbox_item_sync") as card:
        resumed = asyncio.run(errand_service.resume_waiting_errands())

    assert resumed == 1
    card.assert_called_once()
    check = Session()
    try:
        assert check.query(Workflow).filter(Workflow.id == "wf_t").first().state in ("failed", "partial")
    finally:
        check.close()


def test_startup_recovery_fails_orphaned_running_only():
    """recover_orphaned_running_workflows lands orphaned 'running' runs terminal with
    an honest card (kind-agnostic: covers timer, initial-run, and phone orphans), and
    leaves 'waiting' runs (parked on a signal) untouched."""
    Session = _sqlite_sessionmaker()
    seed = Session()
    # orphaned running, one step already succeeded → 'partial'
    seed.add(_wf_row(state="running", cursor=1, wf_id="wf_run",
                     results=json.dumps([{"command": "get_weather", "label": "W", "success": True}])))
    # a parked wait — must NOT be touched
    seed.add(_wf_row(state="waiting", cursor=1, wf_id="wf_wait"))
    seed.commit()
    seed.close()

    with patch("app.db.get_session_local", return_value=Session), \
         patch.object(errand_service, "post_inbox_item_sync") as card:
        n = asyncio.run(errand_service.recover_orphaned_running_workflows())

    assert n == 1
    card.assert_called_once()  # exactly one honest 'interrupted' card
    check = Session()
    try:
        assert check.query(Workflow).filter(Workflow.id == "wf_run").first().state == "partial"
        assert check.query(Workflow).filter(Workflow.id == "wf_wait").first().state == "waiting"  # untouched
    finally:
        check.close()


# ── Chunk 5: mid-run checkpoint resolution (cursor-aware re-plan + widen decision) ──

from app.services.errand_planner import ErrandPlan as _PlanObj  # noqa: E402
from app.services.errand_planner import PlanStep as _PlanStep  # noqa: E402


def _checkpoint_wf(Session, add_command="set_reminder"):
    """Seed a WAITING run parked on a request_replan checkpoint (step 1), after a
    get_weather step (step 0) that already ran. Returns the sessionmaker."""
    seed = Session()
    seed.add(_wf_row(
        state="waiting", cursor=2, wf_id="wf_cp",
        steps=[
            {"command": "get_weather", "args": {}, "label": "W", "is_risky": False},
            {"command": "request_replan", "args": {"reason": "if rain, remind"},
             "label": "Decide", "is_risky": False},
        ],
        results=json.dumps([{"label": "W", "success": True, "message": "Rain tomorrow 90%"}]),
    ))
    seed.commit()
    seed.close()
    return Session


def test_replan_resolution_widens_posts_a_card_and_persists_add_steps():
    Session = _checkpoint_wf(_sqlite_sessionmaker())

    async def fake_replan(*a, **k):  # a NEW command (set_reminder) → widens the envelope
        return _PlanObj(summary="s", steps=[
            _PlanStep("set_reminder", {"text": "umbrella", "when": "7am"}, "Remind")])

    with patch("app.db.get_session_local", return_value=Session), \
         patch("app.core.llm_proxy_client.LLMProxyClient", MagicMock()), \
         patch.object(errand_service, "_resolve_node_menu", new=AsyncMock(return_value=[])), \
         patch.object(errand_service, "replan_from_progress", new=fake_replan), \
         patch.object(errand_service, "_post_replan_card") as post_card, \
         patch.object(errand_service, "_auto_continue_replan", new=AsyncMock()) as auto:
        asyncio.run(errand_service._resolve_and_decide_replan("wf_cp", 1, "if rain, remind", []))

    post_card.assert_called_once()   # widening → surface a delta card
    auto.assert_not_called()         # ...and DON'T silently continue
    check = Session()
    try:
        steps = json.loads(check.query(Workflow).filter(Workflow.id == "wf_cp").first().steps)
        # the resolved continuation is persisted on the checkpoint step for the tap to splice
        assert steps[1]["args"]["add_steps"][0]["command"] == "set_reminder"
    finally:
        check.close()


def test_replan_resolution_in_envelope_auto_continues_no_card():
    Session = _checkpoint_wf(_sqlite_sessionmaker())

    async def fake_replan(*a, **k):  # get_weather is already in the approved plan → no widen
        return _PlanObj(summary="s", steps=[
            _PlanStep("get_weather", {}, "Check again", is_risky=False)])

    with patch("app.db.get_session_local", return_value=Session), \
         patch("app.core.llm_proxy_client.LLMProxyClient", MagicMock()), \
         patch.object(errand_service, "_resolve_node_menu", new=AsyncMock(return_value=[])), \
         patch.object(errand_service, "replan_from_progress", new=fake_replan), \
         patch.object(errand_service, "_post_replan_card") as post_card, \
         patch.object(errand_service, "_auto_continue_replan", new=AsyncMock()) as auto:
        asyncio.run(errand_service._resolve_and_decide_replan("wf_cp", 1, "if rain, remind", []))

    auto.assert_awaited_once()       # in-envelope → resume without a tap
    post_card.assert_not_called()


def test_replan_resolution_empty_continuation_auto_continues():
    Session = _checkpoint_wf(_sqlite_sessionmaker())

    async def fake_replan(*a, **k):  # planner decided nothing more is needed
        return _PlanObj(summary="done", steps=[])

    with patch("app.db.get_session_local", return_value=Session), \
         patch("app.core.llm_proxy_client.LLMProxyClient", MagicMock()), \
         patch.object(errand_service, "_resolve_node_menu", new=AsyncMock(return_value=[])), \
         patch.object(errand_service, "replan_from_progress", new=fake_replan), \
         patch.object(errand_service, "_post_replan_card") as post_card, \
         patch.object(errand_service, "_auto_continue_replan", new=AsyncMock()) as auto:
        asyncio.run(errand_service._resolve_and_decide_replan("wf_cp", 1, "if rain, remind", []))

    auto.assert_awaited_once()       # nothing to add → just proceed past the checkpoint
    post_card.assert_not_called()


def test_replan_resolution_planner_failure_still_continues():
    Session = _checkpoint_wf(_sqlite_sessionmaker())

    async def boom(*a, **k):
        raise ValueError("planner down")

    with patch("app.db.get_session_local", return_value=Session), \
         patch("app.core.llm_proxy_client.LLMProxyClient", MagicMock()), \
         patch.object(errand_service, "_resolve_node_menu", new=AsyncMock(return_value=[])), \
         patch.object(errand_service, "replan_from_progress", new=boom), \
         patch.object(errand_service, "_post_replan_card") as post_card, \
         patch.object(errand_service, "_auto_continue_replan", new=AsyncMock()) as auto:
        asyncio.run(errand_service._resolve_and_decide_replan("wf_cp", 1, "if rain, remind", []))

    auto.assert_awaited_once()       # a failed re-plan must not leave the errand parked
    post_card.assert_not_called()


def test_static_add_steps_skip_the_planner():
    Session = _checkpoint_wf(_sqlite_sessionmaker())
    static = [{"command": "make_phone_call", "args": {"business": "Vet"},
               "label": "Call vet", "is_risky": True}]

    async def should_not_run(*a, **k):
        raise AssertionError("planner should be skipped when static add_steps are given")

    with patch("app.db.get_session_local", return_value=Session), \
         patch("app.core.llm_proxy_client.LLMProxyClient", MagicMock()), \
         patch.object(errand_service, "_resolve_node_menu", new=AsyncMock(return_value=[])), \
         patch.object(errand_service, "replan_from_progress", new=should_not_run), \
         patch.object(errand_service, "_post_replan_card") as post_card, \
         patch.object(errand_service, "_auto_continue_replan", new=AsyncMock()):
        asyncio.run(errand_service._resolve_and_decide_replan("wf_cp", 1, "reason", static))

    post_card.assert_called_once()  # a risky pre-baked call widens → card


def test_needs_a_tap_schedules_the_resolver():
    run = MagicMock()
    run.id = "wf_cp"
    captured = {}

    async def fake_resolve(wf_id, idx, reason, static):
        captured["args"] = (wf_id, idx, reason, static)

    async def drive():
        with patch.object(errand_service, "_resolve_and_decide_replan", new=fake_resolve):
            errand_service._handle_replan_needs_a_tap(
                run, {"step_index": 1, "reason": "r", "add_steps": []})
            await asyncio.sleep(0)  # let the scheduled task run

    asyncio.run(drive())
    assert captured["args"] == ("wf_cp", 1, "r", [])
