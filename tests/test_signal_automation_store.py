"""signal_automation_store — the shared read/write of the signals.automations
setting, and the execution gate get_enabled_instruction."""
import json

import app.services.signal_automation_store as store


class _FakeSettings:
    def __init__(self, value=None):
        self.value = value
        self.saved = None

    def get(self, key, household_id=None):
        return self.value

    def set(self, key, value, household_id=None):
        self.saved = value
        return True


def _patch(monkeypatch, fake):
    import app.services.settings_service as ss

    monkeypatch.setattr(ss, "get_settings_service", lambda: fake)


def test_load_rules_parses_json(monkeypatch):
    _patch(monkeypatch, _FakeSettings(json.dumps({"presence.left": {"instruction": "Lock", "enabled": True}})))
    assert store.load_rules("hh-1")["presence.left"]["instruction"] == "Lock"


def test_load_rules_reads_empty_and_garbage_as_no_rules(monkeypatch):
    _patch(monkeypatch, _FakeSettings(None))
    assert store.load_rules("hh-1") == {}
    _patch(monkeypatch, _FakeSettings("not json{"))
    assert store.load_rules("hh-1") == {}
    _patch(monkeypatch, _FakeSettings("[1,2,3]"))  # valid JSON, not an object
    assert store.load_rules("hh-1") == {}


def test_save_rules_encodes_and_writes(monkeypatch):
    fake = _FakeSettings("{}")
    _patch(monkeypatch, fake)
    assert store.save_rules("hh-1", {"presence.seen": {"instruction": "Lights", "enabled": False}}) is True
    assert json.loads(fake.saved)["presence.seen"]["instruction"] == "Lights"


def test_normalize_delivery():
    assert store.normalize_delivery("automatic") == "automatic"
    assert store.normalize_delivery("notification") == "notification"
    assert store.normalize_delivery("garbage") == "notification"  # safe default
    assert store.normalize_delivery(None) == "notification"


def test_get_enabled_rule_returns_instruction_and_delivery(monkeypatch):
    _patch(
        monkeypatch,
        _FakeSettings(
            json.dumps(
                {"presence.left": {"instruction": "Lock", "enabled": True, "delivery": "automatic"}}
            )
        ),
    )
    assert store.get_enabled_rule("hh-1", "presence.left") == {
        "instruction": "Lock",
        "delivery": "automatic",
    }


def test_get_enabled_rule_defaults_delivery_to_notification(monkeypatch):
    # A legacy rule with no delivery reads as the safe "notification".
    _patch(monkeypatch, _FakeSettings(json.dumps({"presence.left": {"instruction": "Lock", "enabled": True}})))
    assert store.get_enabled_rule("hh-1", "presence.left")["delivery"] == "notification"


def test_get_enabled_rule_none_when_disabled_blank_or_missing(monkeypatch):
    _patch(monkeypatch, _FakeSettings(json.dumps({"presence.left": {"instruction": "Lock", "enabled": False}})))
    assert store.get_enabled_rule("hh-1", "presence.left") is None
    _patch(monkeypatch, _FakeSettings(json.dumps({"presence.left": {"instruction": "   ", "enabled": True}})))
    assert store.get_enabled_rule("hh-1", "presence.left") is None
    _patch(monkeypatch, _FakeSettings("{}"))
    assert store.get_enabled_rule("hh-1", "presence.left") is None
