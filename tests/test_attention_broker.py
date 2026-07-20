"""Gate-behavior tests for the attention broker (prds/attention-broker.md).

Settings are stubbed (module-level get_settings_service monkeypatch) so each
gate is tested in isolation; rows go through the real Postgres test harness
(test_db fixture) because dedup/budget gates are SQL queries.

The invariants pinned here are the broker's laws: demote-only (never promote),
never drop below the journal floor, safety-class bypass, fail-safe truncation.
"""

import uuid

import pytest

from app.models import AttentionConsent, AttentionDelivery, AttentionEvent, AttentionSourceTier
from app.services import attention_broker
from app.services.attention_broker import fallback_dedupe_key, mark_outcome, record_and_route
from app.services.attention_journal import compose_journal_card


class StubSettings:
    """Dict-backed stand-in for the settings service cascade."""

    DEFAULTS = {
        "attention.enabled": True,
        "attention.daily_push_budget": 8,
        "attention.daily_inbox_budget": 30,
        "attention.source_daily_cap": 4,
        "attention.dedupe_window_hours": 24,
        "attention.quiet_hours": "",  # off unless a test sets it
        "attention.timezone": "UTC",
        "attention.safety_categories": "medication,reminder,security,safety",
    }

    def __init__(self, **overrides):
        self.values = {**self.DEFAULTS, **overrides}

    def get(self, key, household_id=None, **kwargs):
        return self.values.get(key)


@pytest.fixture
def settings_stub(monkeypatch):
    def _install(**overrides):
        stub = StubSettings(**overrides)
        monkeypatch.setattr(attention_broker, "get_settings_service", lambda: stub)
        return stub

    return _install


def _household():
    return f"h-attn-{uuid.uuid4().hex[:8]}"


def _route(db, household_id, **kwargs):
    defaults = dict(
        source="news",
        category="news",
        title=f"Headline {uuid.uuid4().hex[:6]}",
        summary="s",
        requested_rung="push",
    )
    defaults.update(kwargs)
    return record_and_route(db, household_id=household_id, **defaults)


class TestHappyPath:
    def test_push_delivers_at_push(self, test_db, settings_stub):
        settings_stub()
        decision = _route(test_db, _household())
        assert decision.deliver is True
        assert decision.rung == "push"
        assert decision.withheld_by is None
        gates = [g["gate"] for g in decision.gate_trail]
        assert "consent" in gates and "budget" in gates and "context" in gates

    def test_event_and_delivery_rows_persist(self, test_db, settings_stub):
        settings_stub()
        household = _household()
        decision = _route(test_db, household)
        event = test_db.query(AttentionEvent).filter_by(id=decision.event_id).one()
        delivery = test_db.query(AttentionDelivery).filter_by(id=decision.delivery_id).one()
        assert event.household_id == household
        assert delivery.rung == "push"

    def test_requested_inbox_is_never_promoted(self, test_db, settings_stub):
        # Demote-only law: budgets being wide open must not raise the rung.
        settings_stub()
        decision = _route(test_db, _household(), requested_rung="inbox")
        assert decision.rung == "inbox"

    def test_mark_outcome_stamps_delivery(self, test_db, settings_stub):
        settings_stub()
        decision = _route(test_db, _household())
        mark_outcome(test_db, decision.delivery_id, outcome="delivered", inbox_item_id="ib-1")
        delivery = test_db.query(AttentionDelivery).filter_by(id=decision.delivery_id).one()
        assert delivery.outcome == "delivered"
        assert delivery.inbox_item_id == "ib-1"


class TestDedup:
    def test_same_key_within_window_is_duplicate(self, test_db, settings_stub):
        settings_stub()
        household = _household()
        first = _route(test_db, household, dedupe_key="storm:1")
        second = _route(test_db, household, dedupe_key="storm:1")
        assert first.deliver is True
        assert second.deliver is False
        assert second.rung == "journal"
        assert second.withheld_by == "dedupe"
        delivery = test_db.query(AttentionDelivery).filter_by(id=second.delivery_id).one()
        assert delivery.outcome == "duplicate"

    def test_distinct_keys_both_deliver(self, test_db, settings_stub):
        settings_stub()
        household = _household()
        assert _route(test_db, household, dedupe_key="a").deliver is True
        assert _route(test_db, household, dedupe_key="b").deliver is True

    def test_legacy_title_fallback_dedupes(self, test_db, settings_stub):
        # No dedupe_key: normalized-title hash is the identity, so the exact
        # re-emit (the news agent's uuid-per-run bug) is caught.
        settings_stub()
        household = _household()
        first = _route(test_db, household, title="Same  Headline ")
        second = _route(test_db, household, title="same headline")
        assert first.deliver is True
        assert second.withheld_by == "dedupe"

    def test_fallback_key_is_stable_across_whitespace_and_case(self):
        assert fallback_dedupe_key(" Same  Headline ") == fallback_dedupe_key("same headline")

    def test_same_key_different_household_is_not_duplicate(self, test_db, settings_stub):
        settings_stub()
        assert _route(test_db, _household(), dedupe_key="k").deliver is True
        assert _route(test_db, _household(), dedupe_key="k").deliver is True


class TestBudgets:
    def test_push_budget_exhausted_demotes_to_inbox(self, test_db, settings_stub):
        settings_stub(**{"attention.daily_push_budget": 0})
        decision = _route(test_db, _household())
        assert decision.deliver is True
        assert decision.rung == "inbox"

    def test_all_budgets_exhausted_lands_in_journal_not_dropped(self, test_db, settings_stub):
        settings_stub(**{"attention.daily_push_budget": 0, "attention.daily_inbox_budget": 0})
        decision = _route(test_db, _household())
        assert decision.deliver is False
        assert decision.rung == "journal"
        assert decision.withheld_by == "budget"
        # Never a silent drop: the journal row exists.
        assert test_db.query(AttentionDelivery).filter_by(id=decision.delivery_id).count() == 1

    def test_source_cap_withholds(self, test_db, settings_stub):
        settings_stub(**{"attention.source_daily_cap": 1})
        household = _household()
        assert _route(test_db, household, dedupe_key="a").deliver is True
        second = _route(test_db, household, dedupe_key="b")
        assert second.deliver is False
        assert second.withheld_by == "budget"

    def test_source_cap_is_per_source(self, test_db, settings_stub):
        settings_stub(**{"attention.source_daily_cap": 1})
        household = _household()
        assert _route(test_db, household, source="news", dedupe_key="a").deliver is True
        assert _route(test_db, household, source="weather", category="weather", dedupe_key="b").deliver is True


class TestQuietHours:
    def test_quiet_hours_demote_push_to_inbox(self, test_db, settings_stub):
        settings_stub(**{"attention.quiet_hours": "00:00-23:59"})
        decision = _route(test_db, _household())
        assert decision.rung == "inbox"

    def test_malformed_quiet_hours_never_silence(self, test_db, settings_stub):
        settings_stub(**{"attention.quiet_hours": "garbage"})
        decision = _route(test_db, _household())
        assert decision.rung == "push"


class TestSafetyClass:
    def test_safety_category_bypasses_budgets_and_quiet_hours(self, test_db, settings_stub):
        settings_stub(**{
            "attention.daily_push_budget": 0,
            "attention.quiet_hours": "00:00-23:59",
        })
        decision = _route(test_db, _household(), source="reminder", category="medication", dedupe_key="med-1")
        assert decision.deliver is True
        assert decision.rung == "push"
        assert decision.gate_trail[0]["gate"] == "safety_class"

    def test_safety_category_still_dedupes_exact_key(self, test_db, settings_stub):
        settings_stub()
        household = _household()
        assert _route(test_db, household, category="medication", dedupe_key="dose-1").deliver is True
        second = _route(test_db, household, category="medication", dedupe_key="dose-1")
        assert second.withheld_by == "dedupe"

    def test_safety_recurring_same_title_no_key_both_deliver(self, test_db, settings_stub):
        # THE KEPPRA REGRESSION (2026-07-19): a twice-daily medication reminder
        # reuses its title. Without an explicit key the title-hash fallback
        # MUST NOT dedup the second dose, or the evening reminder is silently
        # suppressed as a "duplicate" of the morning one. Both must deliver.
        settings_stub()
        household = _household()
        morning = _route(test_db, household, source="medication", category="medication", title="Time for Leo kepra")
        evening = _route(test_db, household, source="medication", category="medication", title="Time for Leo kepra")
        assert morning.deliver is True
        assert evening.deliver is True
        assert evening.withheld_by is None
        assert evening.gate_trail[0]["gate"] == "safety_class"

    def test_safety_explicit_key_dedups_but_different_keys_both_fire(self, test_db, settings_stub):
        # Correct long-term shape: a per-dose key lets morning/evening differ
        # (both fire) while an exact re-emit of the SAME dose dedups.
        settings_stub()
        household = _household()
        assert _route(test_db, household, category="medication", dedupe_key="2026-07-19:am").deliver is True
        assert _route(test_db, household, category="medication", dedupe_key="2026-07-19:pm").deliver is True
        dupe = _route(test_db, household, category="medication", dedupe_key="2026-07-19:pm")
        assert dupe.withheld_by == "dedupe"


class TestForce:
    def test_force_bypasses_dedup(self, test_db, settings_stub):
        settings_stub()
        household = _household()
        assert _route(test_db, household, title="same", force=True).deliver is True
        second = _route(test_db, household, title="same", force=True)
        assert second.deliver is True
        assert second.gate_trail[0]["gate"] == "force"

    def test_force_bypasses_budget_and_quiet_hours(self, test_db, settings_stub):
        settings_stub(**{
            "attention.daily_push_budget": 0,
            "attention.daily_inbox_budget": 0,
            "attention.quiet_hours": "00:00-23:59",
        })
        decision = _route(test_db, _household(), title="urgent", force=True)
        assert decision.deliver is True
        assert decision.rung == "push"

    def test_force_bypasses_consent_never(self, test_db, settings_stub):
        settings_stub()
        household = _household()
        test_db.add(AttentionConsent(household_id=household, source="news", category="news", max_rung="never"))
        test_db.commit()
        decision = _route(test_db, household, force=True)
        assert decision.deliver is True


class TestConsentAndTiers:
    def test_consent_never_withholds(self, test_db, settings_stub):
        settings_stub()
        household = _household()
        test_db.add(AttentionConsent(household_id=household, source="news", category="news", max_rung="never"))
        test_db.commit()
        decision = _route(test_db, household)
        assert decision.deliver is False
        assert decision.withheld_by == "consent"

    def test_consent_inbox_ceiling_demotes_push(self, test_db, settings_stub):
        settings_stub()
        household = _household()
        test_db.add(AttentionConsent(household_id=household, source="news", category="news", max_rung="inbox"))
        test_db.commit()
        decision = _route(test_db, household)
        assert decision.rung == "inbox"

    def test_tier_zero_withholds_with_reason_in_trail(self, test_db, settings_stub):
        settings_stub()
        household = _household()
        test_db.add(AttentionSourceTier(household_id=household, source="news", category="news", tier=0, state_reason="muted by user"))
        test_db.commit()
        decision = _route(test_db, household)
        assert decision.deliver is False
        assert decision.withheld_by == "tier"
        tier_gate = next(g for g in decision.gate_trail if g["gate"] == "tier")
        assert "muted by user" in tier_gate["detail"]

    def test_tier_one_caps_at_inbox(self, test_db, settings_stub):
        settings_stub()
        household = _household()
        test_db.add(AttentionSourceTier(household_id=household, source="news", category="news", tier=1))
        test_db.commit()
        assert _route(test_db, household).rung == "inbox"


class TestInputHardening:
    def test_oversize_title_truncated_to_notifications_cap(self, test_db, settings_stub):
        # notifications' title column is String(500) and 5xxes on overflow.
        settings_stub()
        decision = _route(test_db, _household(), title="x" * 900)
        event = test_db.query(AttentionEvent).filter_by(id=decision.event_id).one()
        assert len(event.title) == 500

    def test_blank_title_and_category_get_defaults(self, test_db, settings_stub):
        settings_stub()
        decision = _route(test_db, _household(), title="   ", category="  ")
        event = test_db.query(AttentionEvent).filter_by(id=decision.event_id).one()
        assert event.title == "(untitled)"
        assert event.category == "general"


class TestJournalCard:
    def test_compose_counts_delivered_and_withheld(self, test_db, settings_stub):
        settings_stub(**{"attention.daily_push_budget": 1, "attention.daily_inbox_budget": 0})
        household = _household()
        _route(test_db, household, dedupe_key="a")   # delivered (push 1/1)
        _route(test_db, household, dedupe_key="b")   # push exhausted -> inbox 0 -> journal
        composed = compose_journal_card(test_db, household)
        assert composed is not None
        summary, body = composed
        assert "Delivered 1" in summary and "withheld 1" in summary
        assert "**Delivered (1)**" in body
        assert "**Withheld (1)**" in body
        assert "budget" in body

    def test_no_activity_returns_none(self, test_db, settings_stub):
        settings_stub()
        assert compose_journal_card(test_db, _household()) is None
