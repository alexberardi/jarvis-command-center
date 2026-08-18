"""signal_catalog — the declared source of truth for authorable signal kinds.

Pins that the catalog matches the kinds real producers emit today, that the
CC-internal ``leave_by.suggested`` is NOT authorable, and the lookup helpers.
"""
from app.services import signal_catalog


def test_catalog_lists_the_three_real_producible_kinds():
    kinds = {s.kind for s in signal_catalog.catalog()}
    assert kinds == {"presence.left", "presence.seen", "appt.upcoming"}


def test_catalog_returns_a_copy_not_the_backing_list():
    a = signal_catalog.catalog()
    a.append("mutation")
    assert len(signal_catalog.catalog()) == 3


def test_every_entry_has_the_ui_fields():
    for s in signal_catalog.catalog():
        assert s.kind and s.label and s.description and s.example and s.source
        assert isinstance(s.facts, dict) and s.facts  # non-empty fact map


def test_leave_by_suggested_is_not_authorable():
    # It's a reaction's own output, never something a user attaches a rule to.
    assert signal_catalog.is_authorable("leave_by.suggested") is False
    assert signal_catalog.get_kind("leave_by.suggested") is None


def test_get_kind_and_is_authorable():
    assert signal_catalog.is_authorable("presence.left") is True
    assert signal_catalog.get_kind("presence.left").label == "I leave home"
    assert signal_catalog.is_authorable("nope.nonexistent") is False
    assert signal_catalog.get_kind("nope.nonexistent") is None
