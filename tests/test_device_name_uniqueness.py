"""Device name uniqueness:
  - PATCH rename to a name used by another device -> 409
  - import auto-suffixes duplicate names so no two devices share a name
"""

import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.deps import get_db
from app.main import app
from app.models import Device
from app.provisioning import ProvisioningAuthContext, verify_provisioning_auth

HH = "hh-devname-test"


def _device(db, name, entity_id):
    d = Device(
        id=str(uuid.uuid4()), household_id=HH, entity_id=entity_id,
        name=name, domain="light", source="test", is_controllable=True, is_active=True,
    )
    db.add(d)
    db.commit()
    db.refresh(d)
    return d


@pytest.fixture
def client(test_db):
    def override_get_db():
        yield test_db

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[verify_provisioning_auth] = lambda: ProvisioningAuthContext(auth_type="admin_key")
    with patch("app.api.smart_home._broadcast_device_cache_invalidate", return_value=None):
        try:
            with TestClient(app) as c:
                yield c
        finally:
            app.dependency_overrides.clear()


def test_rename_to_existing_name_409(client, test_db):
    a = _device(test_db, "ZW6HD", "zw6hd_2")
    _device(test_db, "Office Light", "light.office")
    # rename "Office Light" -> "ZW6HD" (already taken by a) -> 409
    target = test_db.query(Device).filter(Device.entity_id == "light.office").first()
    resp = client.patch(f"/api/v0/households/{HH}/devices/{target.id}", json={"name": "ZW6HD"})
    assert resp.status_code == 409, resp.text
    # case-insensitive
    resp2 = client.patch(f"/api/v0/households/{HH}/devices/{target.id}", json={"name": "zw6hd"})
    assert resp2.status_code == 409

    # renaming a device to its OWN name is fine
    resp3 = client.patch(f"/api/v0/households/{HH}/devices/{a.id}", json={"name": "ZW6HD"})
    assert resp3.status_code == 200


def test_reimport_fixes_preexisting_duplicates(client, test_db):
    # Two devices that predate the uniqueness check, both named "ZW6HD".
    _device(test_db, "ZW6HD", "zw6hd_2")
    _device(test_db, "ZW6HD", "zw6hd_3")
    # Re-running discovery import (matched by entity_id) resolves the duplicate.
    body = {"devices": [
        {"entity_id": "zw6hd_2", "name": "ZW6HD", "domain": "light", "source": "home_assistant"},
        {"entity_id": "zw6hd_3", "name": "ZW6HD", "domain": "light", "source": "home_assistant"},
    ]}
    resp = client.post(f"/api/v0/households/{HH}/devices/import", json=body)
    assert resp.status_code == 201, resp.text
    names = sorted(n for (n,) in test_db.query(Device.name).filter(Device.household_id == HH).all())
    assert names == ["ZW6HD", "ZW6HD (2)"]


def test_reimport_does_not_churn_unique_names(client, test_db):
    _device(test_db, "Office Light", "light.office")
    _device(test_db, "Kitchen Light", "light.kitchen")
    body = {"devices": [
        {"entity_id": "light.office", "name": "Office Light", "domain": "light", "source": "home_assistant"},
    ]}
    resp = client.post(f"/api/v0/households/{HH}/devices/import", json=body)
    assert resp.status_code == 201, resp.text
    office = test_db.query(Device).filter(Device.entity_id == "light.office").first()
    assert office.name == "Office Light"  # no collision → no rename


def test_import_auto_suffixes_duplicate_names(client, test_db):
    body = {"devices": [
        {"entity_id": "zw6hd_2", "name": "ZW6HD", "domain": "light", "source": "home_assistant"},
        {"entity_id": "zw6hd_3", "name": "ZW6HD", "domain": "light", "source": "home_assistant"},
        {"entity_id": "zw6hd_4", "name": "ZW6HD", "domain": "light", "source": "home_assistant"},
    ]}
    resp = client.post(f"/api/v0/households/{HH}/devices/import", json=body)
    assert resp.status_code == 201, resp.text
    names = sorted(n for (n,) in test_db.query(Device.name).filter(Device.household_id == HH).all())
    # three distinct names, no collisions
    assert names == ["ZW6HD", "ZW6HD (2)", "ZW6HD (3)"]
