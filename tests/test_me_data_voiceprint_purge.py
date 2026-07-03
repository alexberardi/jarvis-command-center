"""Account deletion also purges the user's biometric voiceprints in whisper,
best-effort — a whisper failure must never block the deletion.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.deps import AuthenticatedUser, get_db, verify_user_jwt


@pytest.fixture
def jwt_client(test_db):
    def override_get_db():
        yield test_db

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[verify_user_jwt] = lambda: AuthenticatedUser(user_id=555)
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.clear()


def test_delete_my_data_purges_voiceprints(jwt_client):
    mock_client = MagicMock()
    mock_client.delete_all_voice_profiles = AsyncMock(return_value={"status": "deleted"})
    with patch("app.core.clients.whisper_client.WhisperClient", return_value=mock_client):
        resp = jwt_client.delete("/api/v0/me/data")

    assert resp.status_code == 204
    mock_client.delete_all_voice_profiles.assert_awaited_once_with(555)


def test_delete_my_data_survives_whisper_failure(jwt_client):
    mock_client = MagicMock()
    mock_client.delete_all_voice_profiles = AsyncMock(side_effect=RuntimeError("whisper down"))
    with patch("app.core.clients.whisper_client.WhisperClient", return_value=mock_client):
        resp = jwt_client.delete("/api/v0/me/data")

    # The account deletion is not blocked by the whisper side channel failing.
    assert resp.status_code == 204
