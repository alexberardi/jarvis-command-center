"""Shared single-command node dispatch: ``dispatch_node_command``.

The ``tool_call`` MQTT verb + result-file await, extracted so mobile chat AND the
errand executor drive node commands through one path. All mocked — no real MQTT.
"""

import asyncio
import json
import os
from unittest.mock import MagicMock, patch

from app.services import node_command_service as ncs


def test_dispatch_publishes_tool_call_and_returns_output():
    written: dict = {}

    def fake_publish(node_id, command, details, request_id):
        written["node_id"] = node_id
        written["command"] = command
        written["details"] = details
        # simulate the node POSTing its result to /device-control-results/{id}
        os.makedirs(ncs._RESULT_DIR, exist_ok=True)
        with open(os.path.join(ncs._RESULT_DIR, f"{request_id}.json"), "w") as f:
            json.dump({"output": {"success": True, "message": "done", "temp": 72}}, f)
        return request_id

    svc = MagicMock()
    svc.publish_command_with_id.side_effect = fake_publish
    with patch("app.services.node_command_service.get_node_command_service", return_value=svc):
        output = asyncio.run(ncs.dispatch_node_command(
            "node-1", "get_weather", {"day": "today"}, user_id=7, voice_command="weather"))

    assert output == {"success": True, "message": "done", "temp": 72}
    assert written["node_id"] == "node-1" and written["command"] == "tool_call"
    d = written["details"]
    assert d["command_name"] == "get_weather" and d["arguments"] == {"day": "today"}
    assert d["trusted"] is True and d["user_id"] == 7 and d["voice_command"] == "weather"
    assert d["reply_request_id"] == d["tool_call_id"] or d["tool_call_id"]  # both present


def test_dispatch_timeout_returns_failure_dict():
    svc = MagicMock()  # publish is a no-op → no result file ever appears
    with patch("app.services.node_command_service.get_node_command_service", return_value=svc):
        output = asyncio.run(ncs.dispatch_node_command("node-1", "x", {}, timeout=0.2))
    assert output["success"] is False and output["timeout"] is True


def test_dispatch_publish_failure_returns_failure_dict():
    svc = MagicMock()
    svc.publish_command_with_id.side_effect = RuntimeError("mqtt down")
    with patch("app.services.node_command_service.get_node_command_service", return_value=svc):
        output = asyncio.run(ncs.dispatch_node_command("node-1", "x", {}))
    assert output["success"] is False and "dispatch" in output["error"]
