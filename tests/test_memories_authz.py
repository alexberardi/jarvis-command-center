"""Cross-household authorization on the legacy /api/v0/memories router.

Round-2 finding: unlike /api/v0/mobile/memories (which scopes by household +
ownership), the legacy router enforced nothing — list/create trusted a
client-supplied household_id, and get/update/delete used a bare integer PK
with no household filter. Any user JWT could read, tamper with, delete, or
INJECT memories (which are fed into another household's LLM prompt) across
every household.

The fix enforces household membership via require_household_access (the JWT
caller must belong to the claimed / the memory's household; admin-key bypasses).
verify_household_role is stubbed so member vs non-member is deterministic.
"""
from datetime import datetime
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import app
from app.models import UserMemory
from app.deps import get_db
from app.provisioning import verify_provisioning_auth, ProvisioningAuthContext


@pytest.fixture
def jwt_client(test_db):
    def override_get_db():
        try:
            yield test_db
        finally:
            pass

    def override_prov():
        return ProvisioningAuthContext(auth_type="jwt", user_id=777)

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[verify_provisioning_auth] = override_prov
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.clear()


def _make_memory(db, household_id: str = "hh-foreign") -> UserMemory:
    now = datetime.utcnow()
    mem = UserMemory(
        user_id=1, household_id=household_id, category="general",
        content="secret", source="ui", is_active=True, is_pinned=False,
        created_at=now, updated_at=now,
    )
    db.add(mem)
    db.commit()
    db.refresh(mem)
    return mem


class TestMemoriesHouseholdAuth:
    @patch("app.provisioning.verify_household_role",
           side_effect=HTTPException(status_code=403, detail="not a member"))
    def test_list_blocked_for_foreign_household(self, mock_role, jwt_client, test_db):
        resp = jwt_client.get(
            "/api/v0/memories", params={"user_id": 1, "household_id": "hh-foreign"},
        )
        assert resp.status_code == 403

    @patch("app.provisioning.verify_household_role",
           side_effect=HTTPException(status_code=403, detail="not a member"))
    def test_create_blocked_for_foreign_household(self, mock_role, jwt_client, test_db):
        resp = jwt_client.post(
            "/api/v0/memories",
            params={"user_id": 1, "household_id": "hh-foreign"},
            json={"content": "injected instruction"},
        )
        assert resp.status_code == 403
        assert test_db.query(UserMemory).count() == 0

    @patch("app.provisioning.verify_household_role",
           side_effect=HTTPException(status_code=403, detail="not a member"))
    def test_get_blocked_for_foreign_household(self, mock_role, jwt_client, test_db):
        mem = _make_memory(test_db)
        resp = jwt_client.get(f"/api/v0/memories/{mem.id}")
        assert resp.status_code == 403

    @patch("app.provisioning.verify_household_role",
           side_effect=HTTPException(status_code=403, detail="not a member"))
    def test_update_blocked_for_foreign_household(self, mock_role, jwt_client, test_db):
        mem = _make_memory(test_db)
        resp = jwt_client.put(
            f"/api/v0/memories/{mem.id}", json={"content": "hacked"},
        )
        assert resp.status_code == 403
        test_db.refresh(mem)
        assert mem.content == "secret"

    @patch("app.provisioning.verify_household_role",
           side_effect=HTTPException(status_code=403, detail="not a member"))
    def test_delete_blocked_for_foreign_household(self, mock_role, jwt_client, test_db):
        mem = _make_memory(test_db)
        resp = jwt_client.delete(f"/api/v0/memories/{mem.id}")
        assert resp.status_code == 403
        test_db.refresh(mem)
        assert mem.is_active is True

    @patch("app.provisioning.verify_household_role", return_value=None)
    def test_list_allowed_for_member(self, mock_role, jwt_client, test_db):
        resp = jwt_client.get(
            "/api/v0/memories", params={"user_id": 1, "household_id": "hh-owner"},
        )
        assert resp.status_code == 200
