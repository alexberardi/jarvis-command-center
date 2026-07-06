"""Camera streaming: CC registers the node-provided go2rtc source verbatim.

After the node-driven refactor, command-center owns NO Nest/go2rtc source format.
It asks the node (over MQTT, passing device identity) for a `stream_source`
string and registers that string with go2rtc blindly.
"""
import asyncio
import json
import os
from unittest.mock import patch
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.deps import get_db
from app.models import Device, Node
from app.provisioning import verify_provisioning_auth, ProvisioningAuthContext
from app.api import cameras

HH = "hh-cam-1"
FIXED_UUID = UUID("11111111-1111-1111-1111-111111111111")


class _FakeResp:
    def __init__(self, status_code=200, text=""):
        self.status_code = status_code
        self.text = text


class _FakeGo2rtc:
    """Captures the go2rtc PUT so we can assert CC registered the node's src."""

    puts: list[dict] = []

    def __init__(self, *a, **k): ...
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False

    async def put(self, url, params=None):
        _FakeGo2rtc.puts.append(params or {})
        return _FakeResp(200)

    async def get(self, url, params=None):
        return _FakeResp(200, "")


@pytest.fixture
def cam_client(test_db):
    device = Device(
        id="dev-cam-1", household_id=HH, entity_id="nest_doorbell",
        name="Nest Doorbell", domain="camera", protocol="nest",
        cloud_id="enterprises/p/devices/XYZ", is_active=True,
    )
    test_db.add(device)
    test_db.commit()

    def override_get_db():
        yield test_db

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[verify_provisioning_auth] = lambda: ProvisioningAuthContext(
        auth_type="jwt", user_id=777,
    )
    _FakeGo2rtc.puts = []
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.clear()
        cameras._active_streams.clear()
        cameras._stream_households.clear()


@patch("app.provisioning.verify_household_role", return_value=None)
def test_registers_node_stream_source_verbatim(mock_role, cam_client, monkeypatch):
    """CC registers exactly the src the node returns — it does not build a nest: URL."""
    async def fake_fetch(household_id, device, db):
        return {"stream_source": "nest:?client_id=cid&device_id=XYZ"}

    monkeypatch.setattr(cameras, "_fetch_credentials_from_node", fake_fetch)
    monkeypatch.setattr(cameras.httpx, "AsyncClient", _FakeGo2rtc)

    resp = cam_client.post(f"/api/v0/households/{HH}/cameras/dev-cam-1/stream", json={})

    assert resp.status_code == 200, resp.text
    assert len(_FakeGo2rtc.puts) == 1
    assert _FakeGo2rtc.puts[0]["src"] == "nest:?client_id=cid&device_id=XYZ"
    assert _FakeGo2rtc.puts[0]["name"] == "cam_nest_doorbell"
    assert resp.json()["hls_url"].endswith("/cam_nest_doorbell/stream.m3u8")


def test_fetch_publishes_device_identity_over_mqtt(test_db, monkeypatch):
    """The MQTT request to the node carries protocol + full cloud_id + entity_id + domain."""
    test_db.add(Node(
        node_id="node-cam", api_key="k", room="kitchen",
        household_id=HH, is_active=True,
    ))
    test_db.commit()

    published: dict[str, str] = {}

    class _Mqtt:
        def publish(self, topic, payload):
            published["topic"] = topic
            published["payload"] = payload

    class _Settings:
        def get(self, *a, **k):
            return ""  # no primary_node_id → falls back to household node

    monkeypatch.setattr("app.services.settings_service.get_settings_service", lambda: _Settings())
    monkeypatch.setattr("app.node_settings.get_mqtt_client", lambda: _Mqtt())
    monkeypatch.setattr(cameras, "uuid4", lambda: FIXED_UUID)

    # Pre-write the node's response so the poll returns immediately.
    result_file = os.path.join(cameras._CREDS_DIR, f"creds-{FIXED_UUID}.json")
    with open(result_file, "w") as f:
        json.dump({"stream_source": "nest:?ok"}, f)

    device = Device(
        household_id=HH, entity_id="nest_doorbell", name="Nest Doorbell",
        domain="camera", protocol="nest", cloud_id="enterprises/p/devices/XYZ",
    )
    try:
        result = asyncio.run(cameras._fetch_credentials_from_node(HH, device, test_db))
    finally:
        if os.path.exists(result_file):
            os.remove(result_file)

    assert result == {"stream_source": "nest:?ok"}
    payload = json.loads(published["payload"])
    assert payload["protocol"] == "nest"
    assert payload["cloud_id"] == "enterprises/p/devices/XYZ"  # full, unstripped
    assert payload["entity_id"] == "nest_doorbell"
    assert payload["domain"] == "camera"
