from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from app.admin import _register_node_with_auth


def _mock_httpx_client(response_status: int = 201, response_json: dict | None = None):
    """Build a context-manager-compatible mock for httpx.Client()."""
    response = MagicMock()
    response.status_code = response_status
    response.json.return_value = response_json or {"node_key": "fake-key"}
    response.text = ""

    client = MagicMock()
    client.post.return_value = response

    ctx = MagicMock()
    ctx.__enter__.return_value = client
    ctx.__exit__.return_value = False
    return ctx, client


@patch("app.admin.JARVIS_APP_KEY", "test-key")
@patch("app.admin.httpx.Client")
def test_register_node_grants_jarvis_logs(client_factory):
    ctx, client = _mock_httpx_client()
    client_factory.return_value = ctx

    _register_node_with_auth(
        node_id="test-node-id",
        household_id="test-household-id",
        name="kitchen",
    )

    sent_body = client.post.call_args.kwargs["json"]
    assert "jarvis-logs" in sent_body["services"], (
        "Node registration must grant jarvis-logs so the node can ship logs to "
        "the centralized log service. Without this grant the node hits 403 and "
        "logs go to console only."
    )


@patch("app.admin.JARVIS_APP_KEY", "")
def test_register_node_without_app_key_raises():
    with pytest.raises(HTTPException) as exc:
        _register_node_with_auth(
            node_id="test-node-id",
            household_id="test-household-id",
            name="kitchen",
        )
    assert exc.value.status_code == 500
