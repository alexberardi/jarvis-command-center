"""Tests for the routines API.

  Mobile CRUD (JWT, household-scoped via verify_provisioning_auth):
    GET/POST/PATCH/DELETE /api/v0/households/{hh}/routines[/{id}]
    POST .../{id}/run-now
  Node pull (X-API-Key):
    GET /api/v0/nodes/{node_id}/routines
"""

import json
import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.context_providers.node_context_provider import NodeContextProvider
from app.deps import get_db, verify_api_key
from app.main import app
from app.models import Node, Routine
from app.provisioning import ProvisioningAuthContext, verify_provisioning_auth

TEST_USER_ID = 7
TEST_HOUSEHOLD = "hh-routines-test"


def _create_node(db, *, household_id: str = TEST_HOUSEHOLD, active: bool = True) -> Node:
    uid = str(uuid.uuid4())[:8]
    node = Node(
        node_id=f"test-rt-node-{uid}",
        api_key=f"test-rt-key-{uid}",
        room="kitchen",
        user="test-user",
        household_id=household_id,
        is_active=active,
    )
    db.add(node)
    db.commit()
    db.refresh(node)
    return node


@pytest.fixture
def mobile_client(test_db):
    def override_get_db():
        yield test_db

    def override_auth():
        return ProvisioningAuthContext(auth_type="jwt", user_id=TEST_USER_ID)

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[verify_provisioning_auth] = override_auth
    with patch("app.api.routines.require_household_access", return_value=None):
        try:
            with TestClient(app) as client:
                yield client
        finally:
            app.dependency_overrides.clear()


def _node_client(test_db, node: Node) -> TestClient:
    def override_get_db():
        yield test_db

    def override_verify_api_key():
        return NodeContextProvider(node, household_id=node.household_id, household_member_ids=[TEST_USER_ID])

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[verify_api_key] = override_verify_api_key
    return TestClient(app)


def _routine_body(name="Good Morning", **over):
    body = {
        "name": name,
        "trigger_phrases": ["good morning", "morning"],
        "steps": [
            {
                "command": "get_weather",
                "args": [
                    {"key": "resolved_datetimes", "value": "[\"today\"]"},
                    {"key": "city", "value": "Boston"},
                ],
                "label": "weather",
            }
        ],
        "response_instruction": "Cheerful briefing.",
        "response_length": "short",
        "schedule": None,
    }
    body.update(over)
    return body


# ── Mobile CRUD ──────────────────────────────────────────────────────────────


class TestCreate:
    def test_creates_and_slugifies(self, mobile_client, test_db):
        with patch("app.node_settings.get_mqtt_client", return_value=None):
            resp = mobile_client.post(f"/api/v0/households/{TEST_HOUSEHOLD}/routines", json=_routine_body())
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["slug"] == "good_morning"
        assert body["name"] == "Good Morning"
        row = test_db.query(Routine).filter(Routine.id == body["id"]).one()
        assert row.household_id == TEST_HOUSEHOLD
        assert json.loads(row.steps)[0]["command"] == "get_weather"

    def test_duplicate_name_gets_unique_slug(self, mobile_client):
        with patch("app.node_settings.get_mqtt_client", return_value=None):
            a = mobile_client.post(f"/api/v0/households/{TEST_HOUSEHOLD}/routines", json=_routine_body("Night"))
            b = mobile_client.post(f"/api/v0/households/{TEST_HOUSEHOLD}/routines", json=_routine_body("Night"))
        assert a.json()["slug"] == "night"
        assert b.json()["slug"] == "night_2"

    def test_invalid_response_length_422(self, mobile_client):
        with patch("app.node_settings.get_mqtt_client", return_value=None):
            resp = mobile_client.post(
                f"/api/v0/households/{TEST_HOUSEHOLD}/routines",
                json=_routine_body(response_length="epic"),
            )
        assert resp.status_code == 422


class TestListGetPatchDelete:
    def test_list_and_get(self, mobile_client):
        with patch("app.node_settings.get_mqtt_client", return_value=None):
            created = mobile_client.post(f"/api/v0/households/{TEST_HOUSEHOLD}/routines", json=_routine_body("List Me")).json()
        lst = mobile_client.get(f"/api/v0/households/{TEST_HOUSEHOLD}/routines").json()
        assert any(r["id"] == created["id"] for r in lst["routines"])
        one = mobile_client.get(f"/api/v0/households/{TEST_HOUSEHOLD}/routines/{created['id']}").json()
        assert one["name"] == "List Me"

    def test_patch_updates_and_clears_schedule(self, mobile_client, test_db):
        node_id = "test-rt-target"
        with patch("app.node_settings.get_mqtt_client", return_value=None):
            created = mobile_client.post(
                f"/api/v0/households/{TEST_HOUSEHOLD}/routines",
                json=_routine_body("Sched", schedule={
                    "type": "interval", "interval_seconds": 3600, "timezone": "UTC",
                    "target_node_id": node_id, "enabled": True,
                }),
            ).json()
            assert created["schedule"]["interval_seconds"] == 3600
            # clear the schedule (explicit null)
            patched = mobile_client.patch(
                f"/api/v0/households/{TEST_HOUSEHOLD}/routines/{created['id']}",
                json={"schedule": None, "response_length": "medium"},
            ).json()
        assert patched["schedule"] is None
        assert patched["response_length"] == "medium"

    def test_delete(self, mobile_client, test_db):
        with patch("app.node_settings.get_mqtt_client", return_value=None):
            created = mobile_client.post(f"/api/v0/households/{TEST_HOUSEHOLD}/routines", json=_routine_body("Del")).json()
            resp = mobile_client.delete(f"/api/v0/households/{TEST_HOUSEHOLD}/routines/{created['id']}")
        assert resp.status_code == 204
        assert test_db.query(Routine).filter(Routine.id == created["id"]).first() is None


class TestNudge:
    def test_mutation_nudges_each_active_node(self, mobile_client, test_db):
        _create_node(test_db)
        _create_node(test_db)
        _create_node(test_db, active=False)  # inactive — must be skipped
        mqtt = MagicMock()
        with patch("app.node_settings.get_mqtt_client", return_value=mqtt):
            mobile_client.post(f"/api/v0/households/{TEST_HOUSEHOLD}/routines", json=_routine_body("Nudge"))
        topics = [c.args[0] for c in mqtt.publish.call_args_list]
        assert len(topics) == 2  # only the two active nodes
        assert all(t.endswith("/routines/sync") for t in topics)


# ── Node pull (the #1 footgun: arg flatten + array preservation) ──────────────


class TestNodePull:
    def test_flattens_args_and_preserves_arrays(self, mobile_client, test_db):
        with patch("app.node_settings.get_mqtt_client", return_value=None):
            mobile_client.post(f"/api/v0/households/{TEST_HOUSEHOLD}/routines", json=_routine_body("Pull Me"))
        node = _create_node(test_db)
        resp = _node_client(test_db, node).get(f"/api/v0/nodes/{node.node_id}/routines")
        assert resp.status_code == 200, resp.text
        routines = resp.json()["routines"]
        assert "pull_me" in routines
        step = routines["pull_me"]["steps"][0]
        # args flattened to an object, array value preserved (NOT stringified),
        # scalar string left as a string.
        assert step["args"] == {"resolved_datetimes": ["today"], "city": "Boston"}
        assert routines["pull_me"]["response_length"] == "short"

    def test_only_enabled_and_scoped_to_household(self, mobile_client, test_db):
        with patch("app.node_settings.get_mqtt_client", return_value=None):
            created = mobile_client.post(f"/api/v0/households/{TEST_HOUSEHOLD}/routines", json=_routine_body("Enabled One")).json()
            mobile_client.patch(
                f"/api/v0/households/{TEST_HOUSEHOLD}/routines/{created['id']}",
                json={"enabled": False},
            )
        node = _create_node(test_db)
        routines = _node_client(test_db, node).get(f"/api/v0/nodes/{node.node_id}/routines").json()["routines"]
        assert "enabled_one" not in routines  # disabled → not pulled

    def test_node_mismatch_403(self, mobile_client, test_db):
        node = _create_node(test_db)
        resp = _node_client(test_db, node).get("/api/v0/nodes/some-other-node/routines")
        assert resp.status_code == 403
