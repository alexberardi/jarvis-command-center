"""Tests for ``voice_command`` plumbing through the MQTT ``tool_call`` payload.

Mobile chat used to drop the user's original phrase before forwarding tool
calls to the node. Commands like spotify rely on the raw phrase to detect
playlist intent ("play my X playlist") and ended up falling through to
catalog search, picking a track instead of a playlist. We now include
``voice_command`` in the MQTT details dict so the node can plumb it into
``RequestInformation``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api.mobile_chat import _route_tool_call_to_node


@pytest.mark.asyncio
async def test_route_tool_call_includes_voice_command_in_details() -> None:
    tool_call = {
        "id": "tc-abc",
        "function": {"name": "spotify", "arguments": {"action": "play", "query": "jungle night"}},
    }

    # _route_tool_call_to_node now delegates to node_command_service.dispatch_node_command,
    # which fetches the service and awaits the result file there.
    with patch("app.services.node_command_service.get_node_command_service") as svc_factory, \
         patch("app.services.node_command_service._await_result_file", new=AsyncMock(return_value=None)):
        svc = MagicMock()
        svc_factory.return_value = svc

        await _route_tool_call_to_node(
            "node-1", tool_call, user_id=42,
            voice_command="play my jungle night playlist",
        )

        svc.publish_command_with_id.assert_called_once()
        # publish_command_with_id(node_id, command, details, request_id)
        command = svc.publish_command_with_id.call_args.args[1]
        details = svc.publish_command_with_id.call_args.args[2]
        assert command == "tool_call"
        assert details["command_name"] == "spotify"
        assert details["voice_command"] == "play my jungle night playlist"
        assert details["user_id"] == 42


@pytest.mark.asyncio
async def test_route_tool_call_omits_voice_command_when_unset() -> None:
    """Routine builder and other callers without a user phrase shouldn't
    inject an empty voice_command into the payload — the node falls back to
    its placeholder.
    """
    tool_call = {
        "id": "tc-xyz",
        "function": {"name": "get_weather", "arguments": {}},
    }

    with patch("app.services.node_command_service.get_node_command_service") as svc_factory, \
         patch("app.services.node_command_service._await_result_file", new=AsyncMock(return_value=None)):
        svc = MagicMock()
        svc_factory.return_value = svc

        await _route_tool_call_to_node("node-1", tool_call)

        details = svc.publish_command_with_id.call_args.args[2]
        assert "voice_command" not in details
