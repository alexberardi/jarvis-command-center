"""Auth tests for the async-LLM-job callback endpoints (FAIL-CLOSED).

Covers the queue-callback endpoints that llm-proxy POSTs back to:
  POST /api/v0/adapters/jobs/callback
  POST /api/v0/deep-research/callback
  POST /api/v0/memory-extraction/callback

These are distinct from the interactive CallbackJob endpoints in test_callbacks.py.
They share ``_verify_callback_auth`` and set node adapter_hash / write inbox content /
fan out household pushes, so the auth must be fail-closed.

We build ``TestClient(app)`` WITHOUT the lifespan context manager so ``startup_event``
(which needs full infra) does not run — the routes are mounted at import time and the
auth check needs no app state. ``app.main`` is imported lazily inside the fixture to match
conftest's pattern (a top-level import fails in CI without jarvis_auth_client).
"""

import pytest

TOKEN = "test-callback-token"

ENDPOINTS = [
    "/api/v0/adapters/jobs/callback",
    "/api/v0/deep-research/callback",
    "/api/v0/memory-extraction/callback",
]

# A payload that completes all three handlers WITHOUT touching the DB or external services:
# status != "succeeded"/"failed" short-circuits the adapter handler, and the deep-research /
# memory-extraction handlers swallow any downstream error in try/except and still return 200.
INERT_PAYLOAD = {"job_id": "test-job", "status": "unknown", "metadata": {}}


@pytest.fixture
def client():
    from app.main import app
    from fastapi.testclient import TestClient

    # No `with` → no lifespan/startup. Routes are mounted at import.
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Start each test from a known-unset auth posture."""
    monkeypatch.delenv("JARVIS_ADAPTER_CALLBACK_TOKEN", raising=False)
    monkeypatch.delenv("JARVIS_ALLOW_INSECURE_CALLBACKS", raising=False)


@pytest.mark.parametrize("endpoint", ENDPOINTS)
class TestAsyncJobCallbackAuth:
    def test_token_unset_is_rejected_503_fail_closed(self, client, endpoint):
        # The whole point of the hardening: an unset token is NOT an open door.
        resp = client.post(endpoint, json=INERT_PAYLOAD)
        assert resp.status_code == 503, resp.text

    def test_token_set_missing_bearer_rejected_401(self, client, endpoint, monkeypatch):
        monkeypatch.setenv("JARVIS_ADAPTER_CALLBACK_TOKEN", TOKEN)
        resp = client.post(endpoint, json=INERT_PAYLOAD)
        assert resp.status_code == 401, resp.text

    def test_token_set_wrong_bearer_rejected_401(self, client, endpoint, monkeypatch):
        monkeypatch.setenv("JARVIS_ADAPTER_CALLBACK_TOKEN", TOKEN)
        resp = client.post(
            endpoint, json=INERT_PAYLOAD, headers={"Authorization": "Bearer wrong"}
        )
        assert resp.status_code == 401, resp.text

    def test_token_set_correct_bearer_passes_auth(self, client, endpoint, monkeypatch):
        monkeypatch.setenv("JARVIS_ADAPTER_CALLBACK_TOKEN", TOKEN)
        resp = client.post(
            endpoint, json=INERT_PAYLOAD, headers={"Authorization": f"Bearer {TOKEN}"}
        )
        # Auth passed → handler ran to its inert return (NOT 401/503).
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"status": "ok"}

    def test_insecure_flag_allows_open_when_token_unset(self, client, endpoint, monkeypatch):
        monkeypatch.setenv("JARVIS_ALLOW_INSECURE_CALLBACKS", "1")
        resp = client.post(endpoint, json=INERT_PAYLOAD)
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"status": "ok"}

    def test_insecure_flag_does_not_bypass_a_set_token(self, client, endpoint, monkeypatch):
        # If a token IS configured, the insecure opt-out must not weaken the bearer requirement.
        monkeypatch.setenv("JARVIS_ADAPTER_CALLBACK_TOKEN", TOKEN)
        monkeypatch.setenv("JARVIS_ALLOW_INSECURE_CALLBACKS", "1")
        resp = client.post(endpoint, json=INERT_PAYLOAD)  # no bearer presented
        assert resp.status_code == 401, resp.text
