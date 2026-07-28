"""Errand voice fast-follow (CC side): the run-background endpoint + the
detached background task that fires execute_routine_on_node with notify.

See prds/errand-runner.md §8 and the node's test_routine_background.py.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app.api import routines
from app.deps import get_db, verify_api_key
from app.main import app


def _routine(id="rt-1", slug="mb", name="MB"):
    # NOTE: a real object, not MagicMock — MagicMock treats name= specially.
    return SimpleNamespace(id=id, slug=slug, name=name)


def _node_ctx(household_id="hh-1", node_id="node-1"):
    ctx = MagicMock()
    ctx.household_id = household_id
    ctx.node = MagicMock()
    ctx.node.node_id = node_id
    return ctx


def _db_returning(routine):
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = routine
    return db


# ── the detached background task ─────────────────────────────────────────────


def test_background_task_runs_execute_with_notify_and_targeting():
    routine = _routine()
    db = _db_returning(routine)
    with patch("app.db.get_session_local", return_value=lambda: db), patch(
        "app.api.routines.execute_routine_on_node",
        new=AsyncMock(return_value={"status": "success"}),
    ) as ex:
        asyncio.run(routines._run_routine_background_task("hh-1", "rt-1", "node-1", 7))
    ex.assert_awaited_once()
    args, kwargs = ex.call_args
    assert args[1] == "hh-1" and args[3] == "node-1" and args[4] == "background"
    assert kwargs["notify_on_complete"] is True
    assert kwargs["notify_user_id"] == 7
    db.close.assert_called_once()


def test_background_task_posts_failure_card_when_run_raises():
    routine = _routine()
    db = _db_returning(routine)
    with patch("app.db.get_session_local", return_value=lambda: db), patch(
        "app.api.routines.execute_routine_on_node",
        new=AsyncMock(side_effect=RuntimeError("boom")),
    ), patch("app.api.routines._notify_routine_complete") as notify:
        asyncio.run(routines._run_routine_background_task("hh-1", "rt-1", "node-1", 7))
    notify.assert_called_once()
    args = notify.call_args.args
    assert args[1] == "hh-1"
    assert args[2]["status"] == "failed"
    assert args[3] == 7  # targeted at the speaker
    db.close.assert_called_once()


# ── the endpoint ─────────────────────────────────────────────────────────────


def test_run_background_endpoint_dispatches_and_acks():
    routine = _routine(slug="morning_briefing", name="Morning briefing")
    app.dependency_overrides[get_db] = lambda: _db_returning(routine)
    app.dependency_overrides[verify_api_key] = lambda: _node_ctx(node_id="node-9")
    try:
        with patch("app.api.routines._run_routine_background_task") as task_fn, patch(
            "app.api.routines.asyncio.create_task"
        ) as create_task:
            resp = TestClient(app).post(
                "/api/v0/routines/run-background",
                json={"routine_slug": "morning_briefing", "speaker_user_id": 7},
            )
        assert resp.status_code == 200
        assert resp.json() == {"status": "dispatched", "routine": "Morning briefing"}
        create_task.assert_called_once()
        task_fn.assert_called_once_with("hh-1", "rt-1", "node-9", 7)
    finally:
        app.dependency_overrides.clear()


def test_run_background_endpoint_404_for_unknown_routine():
    app.dependency_overrides[get_db] = lambda: _db_returning(None)
    app.dependency_overrides[verify_api_key] = lambda: _node_ctx()
    try:
        resp = TestClient(app).post(
            "/api/v0/routines/run-background", json={"routine_slug": "nope"}
        )
        assert resp.status_code == 404
    finally:
        app.dependency_overrides.clear()
