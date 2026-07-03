"""Admin trace endpoints must require admin auth.

Round-2 finding: the CC admin trace router had no auth dependency (the mobile
trace router does), so any client reaching CC could GET /api/v0/admin/traces
and read every household's raw voice/chat transcripts. The fix gates the whole
admin_router behind verify_admin_key (the X-Api-Key admin token the admin
dashboard already sends).
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.deps import get_db, verify_admin_key


@pytest.fixture
def client_no_auth(test_db):
    def override_get_db():
        try:
            yield test_db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def client_admin(test_db):
    def override_get_db():
        try:
            yield test_db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[verify_admin_key] = lambda: None
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.clear()


class TestAdminTracesAuth:
    _BAD = {"X-Api-Key": "definitely-not-the-admin-key"}

    def test_list_traces_rejects_bad_key(self, client_no_auth):
        """Wrong admin key is 401 — before the fix this returned 200 with every
        household's transcripts."""
        resp = client_no_auth.get("/api/v0/admin/traces", headers=self._BAD)
        assert resp.status_code == 401

    def test_get_trace_rejects_bad_key(self, client_no_auth):
        resp = client_no_auth.get("/api/v0/admin/traces/some-id", headers=self._BAD)
        assert resp.status_code == 401

    def test_list_traces_rejects_missing_key(self, client_no_auth):
        """No admin token at all is also rejected (CC maps the missing-header
        validation error to 400)."""
        resp = client_no_auth.get("/api/v0/admin/traces")
        assert resp.status_code in (400, 401, 403, 422)

    def test_list_traces_allows_admin(self, client_admin):
        resp = client_admin.get("/api/v0/admin/traces")
        assert resp.status_code == 200
