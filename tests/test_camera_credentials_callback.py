"""The camera-credentials callback must require node auth and a UUID request_id.

Hardening: POST /api/v0/camera-credentials/{request_id} was unauthenticated and
wrote the request body to `creds-{request_id}.json`. That let ANY caller (a) poison
the credentials a node's camera-stream fetch is waiting on, and (b) path-traverse
the filename via a crafted request_id into an arbitrary-file write. Fix: require
node auth (X-API-Key) and validate request_id is the UUID CC issued.
"""
import json
import os
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.deps import verify_api_key
from app.api import cameras


@pytest.fixture
def node_client():
    # Stand in for a validated node (verify_api_key normally round-trips to auth).
    app.dependency_overrides[verify_api_key] = lambda: MagicMock()
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.clear()


def test_callback_requires_node_auth():
    """Without the node X-API-Key header the callback is rejected (was open).

    A valid UUID is sent so this can't be a request_id-validation 400 — the
    rejection is the missing required X-API-Key header (CC maps the 422 to 400).
    """
    with TestClient(app) as c:
        resp = c.post(f"/api/v0/camera-credentials/{uuid4()}", json={"user": "x"})
    assert resp.status_code != 200
    assert resp.status_code in (400, 401, 422)


def test_callback_rejects_non_uuid_request_id(node_client):
    """A non-UUID request_id (any path-traversal payload is non-UUID) → 400."""
    resp = node_client.post(
        "/api/v0/camera-credentials/not-a-uuid", json={"user": "x"}
    )
    assert resp.status_code == 400


def test_callback_accepts_uuid_and_writes_creds(node_client):
    rid = str(uuid4())
    written = os.path.join(cameras._CREDS_DIR, f"creds-{rid}.json")
    try:
        resp = node_client.post(
            f"/api/v0/camera-credentials/{rid}", json={"user": "u", "pass": "p"}
        )
        assert resp.status_code == 200
        assert os.path.exists(written)
        with open(written) as f:
            assert json.load(f) == {"user": "u", "pass": "p"}
    finally:
        if os.path.exists(written):
            os.remove(written)
