"""Cross-household authorization on node-scoped endpoints.

Round-2 finding: the mobile/user-triggered node endpoints authenticate the
caller but never bind them to the *target node's* household, so any user JWT
from any household can act on another household's node (install code, unlock
a door, repoint the node). The fix resolves the household from the node row
and calls the existing household check (require_household_access for
provisioning-auth routes, verify_household_role for user-JWT routes).

These tests stub the deep household check (verify_household_role) so we control
member vs non-member deterministically without a live jarvis-auth:
  - non-member  -> verify_household_role raises 403  -> endpoint must 403
  - member      -> verify_household_role returns None -> endpoint proceeds
  - admin key   -> require_household_access bypasses  -> endpoint proceeds
"""
import uuid
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import app
from app.models import Node, PackageInstallRequest
from app.deps import get_db, verify_api_key, verify_user_jwt, AuthenticatedUser
from app.provisioning import verify_provisioning_auth, ProvisioningAuthContext


def _make_node(db, household_id: str = "hh-owner") -> Node:
    uid = str(uuid.uuid4())[:8]
    node = Node(
        node_id=f"test-node-{uid}",
        api_key=f"test-key-{uid}",
        room="office",
        user="owner",
        household_id=household_id,
    )
    db.add(node)
    db.commit()
    db.refresh(node)
    return node


@pytest.fixture
def jwt_client(test_db):
    """Client authenticated as a JWT user (auth_type='jwt').

    Whether that user is a household member is decided by patching
    app.provisioning.verify_household_role in each test.
    """
    def override_get_db():
        try:
            yield test_db
        finally:
            pass

    def override_prov():
        return ProvisioningAuthContext(auth_type="jwt", user_id=777)

    def override_node():
        return MagicMock()

    def override_user_jwt():
        return AuthenticatedUser(user_id=777, email="u@example.com", is_superuser=False)

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[verify_provisioning_auth] = override_prov
    app.dependency_overrides[verify_user_jwt] = override_user_jwt
    app.dependency_overrides[verify_api_key] = override_node
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.clear()


# =============================================================================
# package-install / uninstall / revert — the cross-household RCE class
# =============================================================================


class TestPackageInstallHouseholdAuth:
    _INSTALL_BODY = {"command_name": "weather", "github_repo_url": "https://github.com/x/y"}

    @patch("app.provisioning.verify_household_role",
           side_effect=HTTPException(status_code=403, detail="not a member"))
    @patch("app.api.package_install._publish_package_install_mqtt")
    def test_install_blocked_for_foreign_household(self, mock_mqtt, mock_role, jwt_client, test_db):
        """A JWT user who is NOT a member of the node's household is refused —
        no install request is created and no MQTT command is published."""
        node = _make_node(test_db)

        resp = jwt_client.post(
            f"/api/v0/nodes/{node.node_id}/package-install", json=self._INSTALL_BODY,
        )

        assert resp.status_code == 403
        mock_mqtt.assert_not_called()
        assert test_db.query(PackageInstallRequest).count() == 0

    @patch("app.provisioning.verify_household_role", return_value=None)
    @patch("app.api.package_install._publish_package_install_mqtt")
    def test_install_allowed_for_member(self, mock_mqtt, mock_role, jwt_client, test_db):
        """A JWT user who IS a member proceeds normally (guards against over-blocking)."""
        node = _make_node(test_db)

        resp = jwt_client.post(
            f"/api/v0/nodes/{node.node_id}/package-install", json=self._INSTALL_BODY,
        )

        assert resp.status_code == 201
        mock_mqtt.assert_called_once()

    @patch("app.provisioning.verify_household_role",
           side_effect=HTTPException(status_code=403, detail="not a member"))
    @patch("app.api.package_install._publish_package_uninstall_mqtt")
    def test_uninstall_blocked_for_foreign_household(self, mock_mqtt, mock_role, jwt_client, test_db):
        node = _make_node(test_db)

        resp = jwt_client.post(
            f"/api/v0/nodes/{node.node_id}/package-uninstall",
            json={"command_name": "weather", "component_type": "command"},
        )

        assert resp.status_code == 403
        mock_mqtt.assert_not_called()

    @patch("app.provisioning.verify_household_role",
           side_effect=HTTPException(status_code=403, detail="not a member"))
    @patch("app.api.package_install._publish_package_revert_mqtt")
    def test_revert_blocked_for_foreign_household(self, mock_mqtt, mock_role, jwt_client, test_db):
        node = _make_node(test_db)

        resp = jwt_client.post(
            f"/api/v0/nodes/{node.node_id}/package-revert",
            json={"command_name": "weather"},
        )

        assert resp.status_code == 403
        mock_mqtt.assert_not_called()

    @patch("app.provisioning.verify_household_role",
           side_effect=HTTPException(status_code=403, detail="not a member"))
    def test_poll_blocked_for_foreign_household(self, mock_role, jwt_client, test_db):
        """Polling another household's install request must not leak its status."""
        node = _make_node(test_db)
        now = datetime.utcnow()
        req = PackageInstallRequest(
            id=str(uuid.uuid4()), node_id=node.node_id, household_id=node.household_id,
            command_name="weather", github_repo_url="https://github.com/x/y", status="pending",
            created_at=now, expires_at=now + timedelta(minutes=5),
        )
        test_db.add(req)
        test_db.commit()

        resp = jwt_client.get(
            f"/api/v0/nodes/{node.node_id}/package-install/{req.id}",
        )

        assert resp.status_code == 403


# =============================================================================
# node_commands — the action forwarder (door unlock) + node-config repoint
# =============================================================================


class TestNodeCommandsHouseholdAuth:
    @patch("app.api.node_commands.get_node_command_service")
    @patch("app.api.node_commands.verify_household_role",
           side_effect=HTTPException(status_code=403, detail="not a member"))
    def test_action_blocked_for_foreign_household(self, mock_role, mock_svc, jwt_client, test_db):
        """Inbox action forwarder (unlock-door / control-device) must reject
        a JWT user who isn't a member of the target node's household."""
        node = _make_node(test_db)

        resp = jwt_client.post(
            f"/api/v0/nodes/{node.node_id}/actions",
            json={"command_name": "control_device", "action_name": "unlock"},
        )

        assert resp.status_code == 403
        mock_svc.return_value.publish_command.assert_not_called()

    @patch("app.api.node_commands.get_node_command_service")
    @patch("app.api.node_commands.verify_household_role",
           side_effect=HTTPException(status_code=403, detail="not a member"))
    def test_node_config_blocked_for_foreign_household(self, mock_role, mock_svc, jwt_client, test_db):
        node = _make_node(test_db)

        resp = jwt_client.post(
            f"/api/v0/nodes/{node.node_id}/node-config",
            json={"settings": {"volume_percent": 50}},
        )

        assert resp.status_code == 403
        mock_svc.return_value.publish_command.assert_not_called()


# =============================================================================
# node_updates — forced version change / cross-node task access
# =============================================================================


class TestNodeUpdatesHouseholdAuth:
    @patch("app.api.node_updates.verify_household_role",
           side_effect=HTTPException(status_code=403, detail="not a member"))
    def test_update_blocked_for_foreign_household(self, mock_role, jwt_client, test_db):
        node = _make_node(test_db)

        resp = jwt_client.post(f"/api/v0/nodes/{node.node_id}/update", json={})

        assert resp.status_code == 403


# =============================================================================
# bluetooth + test-install — provisioning-auth requests
# =============================================================================


class TestBluetoothHouseholdAuth:
    @patch("app.provisioning.verify_household_role",
           side_effect=HTTPException(status_code=403, detail="not a member"))
    def test_scan_blocked_for_foreign_household(self, mock_role, jwt_client, test_db):
        node = _make_node(test_db)

        resp = jwt_client.post(
            f"/api/v0/nodes/{node.node_id}/bluetooth-scan/request", json={},
        )

        assert resp.status_code == 403


class TestTestInstallHouseholdAuth:
    @patch("app.provisioning.verify_household_role",
           side_effect=HTTPException(status_code=403, detail="not a member"))
    def test_test_install_blocked_for_foreign_household(self, mock_role, jwt_client, test_db):
        node = _make_node(test_db)

        resp = jwt_client.post(
            f"/api/v0/nodes/{node.node_id}/test-install", json={"share_code": "abc123"},
        )

        assert resp.status_code == 403
