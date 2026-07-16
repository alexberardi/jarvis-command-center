"""HTTP-status contract tests for the blocking voice command endpoints (#51).

The blocking command endpoints (`/voice/command`, `/voice/command/continue`) must
surface *whole-request precondition failures* as HTTP **422**, not swallow them
into a `200` + errors body via the broad `except Exception` handler. Legitimate
per-command / partial-batch failures keep their intentional `200` shape, and
genuine internal errors must NOT be mis-mapped to 422.

Validated entirely by command-center's own fast lane (no cross-repo round-trip).
Fixture pattern modeled on tests/test_device_list_endpoints.py:30-57.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app
from app.deps import verify_api_key, get_model_service


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def model_service_mock():
    """A model-service double whose async methods can be driven per-test."""
    svc = MagicMock()
    svc.process_voice_command_with_tools = AsyncMock()
    svc.continue_conversation_with_tool_results = AsyncMock()
    return svc


@pytest.fixture
def voice_client(model_service_mock):
    """TestClient with auth + model-service overrides for the blocking endpoints."""

    def override_api_key():
        return MagicMock()

    def override_model_service():
        return model_service_mock

    app.dependency_overrides[verify_api_key] = override_api_key
    app.dependency_overrides[get_model_service] = override_model_service

    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()


def _voice_body(conversation_id: str = "conv-test-001") -> dict:
    return {"voice_command": "turn on the lights", "conversation_id": conversation_id}


def _continue_body(conversation_id: str = "conv-test-001") -> dict:
    return {
        "conversation_id": conversation_id,
        "tool_results": [{"tool_call_id": "tc-1", "output": "ok"}],
    }


# =============================================================================
# Happy path / the fix
# =============================================================================


def test_voice_command_uninitialized_conversation_returns_422(voice_client):
    """The inline precondition (tools is None) must reach the client as 422,
    not be swallowed into a 200 VoiceCommandResponse with commands[].errors."""
    with patch.object(main_module.conversation_cache, "get_tools", return_value=None):
        resp = voice_client.post("/api/v0/voice/command", json=_voice_body())

    assert resp.status_code == 422
    assert resp.status_code != 200
    body = resp.json()
    # FastAPI error/detail shape — NOT a VoiceCommandResponse with commands[0].errors.
    assert "detail" in body
    assert "commands" not in body


def test_voice_command_expired_conversation_returns_422(voice_client, model_service_mock):
    """A ConversationPreconditionError raised from the handler maps to 422
    instead of the broad `except Exception` converting it to a 200 body."""
    from app.core.errors import ConversationPreconditionError

    model_service_mock.process_voice_command_with_tools.side_effect = (
        ConversationPreconditionError("Conversation conv-test-001 not found or expired")
    )
    with patch.object(
        main_module.conversation_cache, "get_tools", return_value=[{"name": "noop"}]
    ):
        resp = voice_client.post("/api/v0/voice/command", json=_voice_body())

    assert resp.status_code == 422
    assert "not found or expired" in str(resp.json().get("detail", ""))


def test_voice_command_continue_unknown_conversation_returns_422(
    voice_client, model_service_mock
):
    """Same contract on the second blocking endpoint /voice/command/continue."""
    from app.core.errors import ConversationPreconditionError

    model_service_mock.continue_conversation_with_tool_results.side_effect = (
        ConversationPreconditionError("Conversation conv-test-001 not found or expired")
    )
    resp = voice_client.post("/api/v0/voice/command/continue", json=_continue_body())

    assert resp.status_code == 422
    assert "not found or expired" in str(resp.json().get("detail", ""))


# =============================================================================
# Edge / typed-exception unit
# =============================================================================


def test_conversation_precondition_error_subclasses_valueerror():
    """The new typed exception must remain a ValueError subclass so existing
    `except ValueError` callers/tests don't regress."""
    from app.core.errors import ConversationPreconditionError

    assert issubclass(ConversationPreconditionError, ValueError)


# =============================================================================
# Regression (must NOT change)
# =============================================================================


def test_voice_command_partial_batch_still_returns_200(voice_client, model_service_mock):
    """A legitimate per-command failure keeps the intentional 200 +
    commands[].success=False shape; the fix must not touch this path."""
    model_service_mock.process_voice_command_with_tools.return_value = {
        "stop_reason": "error",
        "error": "tool execution failed",
    }
    with patch.object(
        main_module.conversation_cache, "get_tools", return_value=[{"name": "noop"}]
    ):
        resp = voice_client.post("/api/v0/voice/command", json=_voice_body())

    assert resp.status_code == 200
    assert resp.json()["commands"][0]["success"] is False


def test_voice_command_internal_error_not_remapped_to_422(voice_client, model_service_mock):
    """A genuine internal error (a plain Exception, NOT a precondition) must not
    be mis-mapped to 422 — the new precondition/HTTPException handlers sit ahead
    of the broad `except Exception` without widening it."""
    model_service_mock.process_voice_command_with_tools.side_effect = RuntimeError("boom")
    with patch.object(
        main_module.conversation_cache, "get_tools", return_value=[{"name": "noop"}]
    ):
        resp = voice_client.post("/api/v0/voice/command", json=_voice_body())

    assert resp.status_code != 422
