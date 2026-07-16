"""Camera HLS proxy must be household-scoped and media-only.

Round-2 finding: GET /api/v0/cameras/stream/{stream_name}/{path} was the one
camera endpoint without require_household_access — it gated only on a GLOBAL
_active_streams dict, so any authenticated user could proxy another household's
live feed, and the attacker-controlled path could reach go2rtc /api/config
(disclosing every household's Nest OAuth credentials).

Fix: track the stream's owning household and enforce it, and restrict the
proxied path to HLS media files.
"""
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import app
from app.deps import get_db
from app.provisioning import verify_provisioning_auth, ProvisioningAuthContext
from app.api import cameras


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
        cameras._active_streams.clear()
        cameras._stream_households.clear()


class TestCameraStreamAuth:
    @patch("app.provisioning.verify_household_role",
           side_effect=HTTPException(status_code=403, detail="not a member"))
    def test_stream_blocked_for_foreign_household(self, mock_role, jwt_client):
        cameras._active_streams["dev1"] = "stream-A"
        cameras._stream_households["stream-A"] = "hh-A"

        resp = jwt_client.get("/api/v0/cameras/stream/stream-A/stream.m3u8")

        assert resp.status_code == 403

    @patch("app.provisioning.verify_household_role", return_value=None)
    def test_config_path_rejected_even_for_member(self, mock_role, jwt_client):
        """A member can't pivot the proxy to go2rtc /api/config (Nest creds)."""
        cameras._active_streams["dev1"] = "stream-A"
        cameras._stream_households["stream-A"] = "hh-A"

        resp = jwt_client.get("/api/v0/cameras/stream/stream-A/config")

        assert resp.status_code == 404

    def test_unknown_stream_404(self, jwt_client):
        resp = jwt_client.get("/api/v0/cameras/stream/nonexistent/stream.m3u8")
        assert resp.status_code == 404
