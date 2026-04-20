"""Tests for the Phase 5 AdapterScheduler."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.models import Base
from app.services.adapter_registry import AdapterRegistry
from app.services.adapter_scheduler import AdapterScheduler, extract_proposal_metrics
from app.services.training_orchestrator import OrchestratorResult


@pytest.fixture()
def db():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def registry(db):
    return AdapterRegistry(db)


# --- helpers ---


def _success(adapter_hash: str, verdict: str = "PASS", pass_rate: float = 91.7, rows: int = 123):
    return OrchestratorResult(
        status="success",
        adapter_hash=adapter_hash,
        job_id="job-1",
        training_seconds=42.0,
        dataset_rows=rows,
        by_source={"organic": rows},
        eval={"trailer": {"verdict": verdict, "pass_rate": pass_rate}},
    )


def _failed(status: str, error: str = "boom"):
    return OrchestratorResult(
        status=status,
        adapter_hash=None,
        job_id="job-1",
        training_seconds=None,
        dataset_rows=0,
        by_source={},
        error=error,
    )


# --- tests ---


def test_skips_when_not_enough_examples(registry, db):
    sched = AdapterScheduler(
        registry=registry,
        count_new_examples=lambda hid, since: 10,  # below threshold
        list_active_households=lambda: ["h1"],
        orchestrate_for_household=lambda hid: _success("a" * 32),
        min_new_examples=50,
        min_hours_between_runs=0,
    )
    tick = sched.run_tick()
    db.commit()

    assert tick.considered == ["h1"]
    assert "h1" in tick.skipped
    assert "not_enough_examples" in tick.skipped["h1"]
    assert tick.deployed == []
    assert registry.get_active("h1") is None


def test_skips_when_in_cooldown(registry, db):
    now = datetime(2026, 4, 19, 12, 0, 0)
    state = registry.ensure_state("h1")
    state.last_trained_at = now - timedelta(hours=2)
    db.commit()

    sched = AdapterScheduler(
        registry=registry,
        count_new_examples=lambda hid, since: 500,
        list_active_households=lambda: ["h1"],
        orchestrate_for_household=lambda hid: _success("a" * 32),
        min_new_examples=50,
        min_hours_between_runs=6,
        now_fn=lambda: now,
    )
    tick = sched.run_tick()

    assert "h1" in tick.skipped
    assert "cooldown" in tick.skipped["h1"]
    assert tick.deployed == []


def test_skips_when_already_training(registry, db):
    registry.try_acquire_training_lock("h1")
    db.commit()

    orch_called = []

    sched = AdapterScheduler(
        registry=registry,
        count_new_examples=lambda hid, since: 500,
        list_active_households=lambda: ["h1"],
        orchestrate_for_household=lambda hid: (
            orch_called.append(hid) or _success("a" * 32)
        ),
        min_new_examples=50,
        min_hours_between_runs=0,
    )
    tick = sched.run_tick()

    assert tick.skipped == {"h1": "already_training"}
    assert orch_called == []


def test_deploys_on_eval_pass(registry, db):
    sched = AdapterScheduler(
        registry=registry,
        count_new_examples=lambda hid, since: 500,
        list_active_households=lambda: ["h1"],
        orchestrate_for_household=lambda hid: _success("c" * 32, verdict="PASS", pass_rate=93.4, rows=220),
        min_new_examples=50,
        min_hours_between_runs=0,
        auto_approve=True,
    )
    tick = sched.run_tick()
    db.commit()

    assert tick.deployed == ["h1"]
    assert tick.eval_failed == []
    active = registry.get_active("h1")
    assert active is not None
    assert active.adapter_hash == "c" * 32
    assert active.pass_rate == 93.4
    assert active.trained_on_examples == 220

    state = registry.get_state("h1")
    assert state.is_training is False
    assert state.last_trained_at is not None
    assert state.last_example_count == 220


def test_does_not_deploy_on_eval_fail(registry, db):
    sched = AdapterScheduler(
        registry=registry,
        count_new_examples=lambda hid, since: 500,
        list_active_households=lambda: ["h1"],
        orchestrate_for_household=lambda hid: _success("d" * 32, verdict="FAIL", pass_rate=70.0),
        min_new_examples=50,
        min_hours_between_runs=0,
    )
    tick = sched.run_tick()
    db.commit()

    assert tick.deployed == []
    assert tick.eval_failed == ["h1"]
    assert registry.get_active("h1") is None

    state = registry.get_state("h1")
    assert state.is_training is False
    assert state.last_trained_at is not None


def test_does_not_deploy_on_missing_eval_trailer(registry, db):
    """If the trailer is absent, treat it as an eval failure (conservative)."""
    def orch(hid):
        r = _success("e" * 32)
        r.eval = {}  # no trailer
        return r

    sched = AdapterScheduler(
        registry=registry,
        count_new_examples=lambda hid, since: 500,
        list_active_households=lambda: ["h1"],
        orchestrate_for_household=orch,
        min_new_examples=50,
        min_hours_between_runs=0,
    )
    tick = sched.run_tick()
    db.commit()

    assert tick.deployed == []
    assert tick.eval_failed == ["h1"]


def test_records_errored_on_training_failure(registry, db):
    sched = AdapterScheduler(
        registry=registry,
        count_new_examples=lambda hid, since: 500,
        list_active_households=lambda: ["h1"],
        orchestrate_for_household=lambda hid: _failed("training_failed", "timeout"),
        min_new_examples=50,
        min_hours_between_runs=0,
    )
    tick = sched.run_tick()
    db.commit()

    assert tick.deployed == []
    assert tick.errored == {"h1": "timeout"}
    assert registry.get_active("h1") is None
    state = registry.get_state("h1")
    assert state.is_training is False


def test_releases_lock_on_orchestrator_exception(registry, db):
    def explode(hid):
        raise RuntimeError("network died")

    sched = AdapterScheduler(
        registry=registry,
        count_new_examples=lambda hid, since: 500,
        list_active_households=lambda: ["h1"],
        orchestrate_for_household=explode,
        min_new_examples=50,
        min_hours_between_runs=0,
    )
    tick = sched.run_tick()
    db.commit()

    assert tick.errored and "network died" in next(iter(tick.errored.values()))
    state = registry.get_state("h1")
    assert state.is_training is False  # lock released despite exception


def test_preserves_prior_adapter_on_eval_fail(registry, db):
    """An adapter that regresses must NOT replace the currently-deployed one."""
    registry.deploy("h1", "old" * 11 + "o", pass_rate=0.85, trained_on_examples=100)
    db.commit()

    sched = AdapterScheduler(
        registry=registry,
        count_new_examples=lambda hid, since: 500,
        list_active_households=lambda: ["h1"],
        orchestrate_for_household=lambda hid: _success("new" * 11 + "n", verdict="FAIL", pass_rate=60.0),
        min_new_examples=50,
        min_hours_between_runs=0,
    )
    sched.run_tick()
    db.commit()

    active = registry.get_active("h1")
    assert active.adapter_hash == "old" * 11 + "o"
    assert active.pass_rate == 0.85


def test_independent_households(registry, db):
    sched = AdapterScheduler(
        registry=registry,
        count_new_examples=lambda hid, since: 500 if hid == "h1" else 10,
        list_active_households=lambda: ["h1", "h2"],
        orchestrate_for_household=lambda hid: _success("a" * 32, rows=100),
        min_new_examples=50,
        min_hours_between_runs=0,
        auto_approve=True,
    )
    tick = sched.run_tick()
    db.commit()

    assert tick.deployed == ["h1"]
    assert "h2" in tick.skipped
    assert registry.get_active("h1") is not None
    assert registry.get_active("h2") is None


# ===== Phase 7.2 propose-instead-of-deploy default =====


def test_proposes_on_eval_pass_by_default(registry, db):
    """auto_approve=False (default) → PASS fires propose callback, no deploy."""
    calls: list[tuple[str, str]] = []

    def propose(hid, result):
        calls.append((hid, result.adapter_hash))
        return True

    sched = AdapterScheduler(
        registry=registry,
        count_new_examples=lambda hid, since: 500,
        list_active_households=lambda: ["h1"],
        orchestrate_for_household=lambda hid: _success("c" * 32, verdict="PASS"),
        min_new_examples=50,
        min_hours_between_runs=0,
        propose_for_household=propose,
    )
    tick = sched.run_tick()
    db.commit()

    assert tick.deployed == []
    assert tick.proposed == ["h1"]
    assert calls == [("h1", "c" * 32)]
    # Adapter was NOT deployed
    assert registry.get_active("h1") is None
    state = registry.get_state("h1")
    assert state.is_training is False
    assert state.last_trained_at is not None


def test_errors_when_no_propose_fn_wired(registry, db):
    """auto_approve=False + no propose_fn → scheduler must NOT silently drop PASS."""
    sched = AdapterScheduler(
        registry=registry,
        count_new_examples=lambda hid, since: 500,
        list_active_households=lambda: ["h1"],
        orchestrate_for_household=lambda hid: _success("c" * 32, verdict="PASS"),
        min_new_examples=50,
        min_hours_between_runs=0,
    )
    tick = sched.run_tick()
    db.commit()

    assert tick.deployed == []
    assert tick.proposed == []
    assert "h1" in tick.errored
    assert registry.get_active("h1") is None
    # Lock still released
    state = registry.get_state("h1")
    assert state.is_training is False


def test_propose_exception_is_errored_and_releases_lock(registry, db):
    def propose(hid, result):
        raise RuntimeError("inbox service down")

    sched = AdapterScheduler(
        registry=registry,
        count_new_examples=lambda hid, since: 500,
        list_active_households=lambda: ["h1"],
        orchestrate_for_household=lambda hid: _success("c" * 32, verdict="PASS"),
        min_new_examples=50,
        min_hours_between_runs=0,
        propose_for_household=propose,
    )
    tick = sched.run_tick()
    db.commit()

    assert tick.proposed == []
    assert "h1" in tick.errored
    state = registry.get_state("h1")
    assert state.is_training is False


def test_propose_eval_fail_does_not_call_propose(registry, db):
    """An eval-fail outcome must NOT invoke the propose callback."""
    calls: list[str] = []

    def propose(hid, result):
        calls.append(hid)
        return True

    sched = AdapterScheduler(
        registry=registry,
        count_new_examples=lambda hid, since: 500,
        list_active_households=lambda: ["h1"],
        orchestrate_for_household=lambda hid: _success("c" * 32, verdict="FAIL"),
        min_new_examples=50,
        min_hours_between_runs=0,
        propose_for_household=propose,
    )
    tick = sched.run_tick()
    db.commit()

    assert tick.eval_failed == ["h1"]
    assert tick.proposed == []
    assert calls == []


def test_propose_returning_false_is_errored(registry, db):
    """A propose callback that reports failure keeps the adapter unpromoted."""
    def propose(hid, result):
        return False

    sched = AdapterScheduler(
        registry=registry,
        count_new_examples=lambda hid, since: 500,
        list_active_households=lambda: ["h1"],
        orchestrate_for_household=lambda hid: _success("c" * 32, verdict="PASS"),
        min_new_examples=50,
        min_hours_between_runs=0,
        propose_for_household=propose,
    )
    tick = sched.run_tick()
    db.commit()

    assert tick.proposed == []
    assert "h1" in tick.errored


# ===== Phase 7.2b extract_proposal_metrics trailer decoding =====


def test_extract_metrics_prefers_with_baseline_fields():
    trailer = {
        "verdict": "PASS",
        "pass_rate": 86.4,             # adapter run (legacy key)
        "pass_rate_before": 78.8,      # --with-baseline fields
        "pass_rate_after": 86.4,
        "latency_before_s": 5.88,
        "latency_after_s": 4.18,
        "per_command_delta": {
            "calculate": {
                "before": {"success_rate": 87.5, "passed": 7, "total": 8},
                "after": {"success_rate": 100.0, "passed": 8, "total": 8},
                "delta_pp": 12.5,
            }
        },
        "threshold": 78.8,
    }
    m = extract_proposal_metrics(trailer)
    assert m.pass_rate_before == 78.8
    assert m.pass_rate_after == 86.4
    assert m.latency_before_s == 5.88
    assert m.latency_after_s == 4.18
    assert m.per_command_delta is not None
    assert m.per_command_delta["calculate"]["delta_pp"] == 12.5


def test_extract_metrics_falls_back_to_threshold_without_baseline():
    """Legacy trailer (no --with-baseline) still yields usable before/after."""
    trailer = {
        "verdict": "PASS",
        "pass_rate": 81.0,   # adapter side
        "threshold": 73.0,   # literal baseline stands in as before
        "tolerance": 1.0,
    }
    m = extract_proposal_metrics(trailer)
    assert m.pass_rate_before == 73.0
    assert m.pass_rate_after == 81.0
    assert m.latency_before_s is None
    assert m.latency_after_s is None
    assert m.per_command_delta is None


def test_extract_metrics_handles_empty_trailer():
    m = extract_proposal_metrics({})
    assert m.pass_rate_before is None
    assert m.pass_rate_after is None
    assert m.latency_before_s is None
    assert m.latency_after_s is None
    assert m.per_command_delta is None


def test_extract_metrics_coerces_string_numerics():
    """Trailer comes from JSON — just in case a number shows up as a string."""
    trailer = {
        "pass_rate_before": "78.8",
        "pass_rate_after": "86.4",
        "latency_before_s": "5.88",
        "latency_after_s": "4.18",
    }
    m = extract_proposal_metrics(trailer)
    assert m.pass_rate_before == 78.8
    assert m.pass_rate_after == 86.4


def test_extract_metrics_drops_unparsable_values():
    trailer = {
        "pass_rate_before": "not-a-number",
        "pass_rate_after": None,
    }
    m = extract_proposal_metrics(trailer)
    assert m.pass_rate_before is None
    assert m.pass_rate_after is None


def test_cooldown_zero_disables_rate_limiter(registry, db):
    now = datetime(2026, 4, 19, 12, 0, 0)
    state = registry.ensure_state("h1")
    state.last_trained_at = now - timedelta(minutes=1)  # very recent
    db.commit()

    sched = AdapterScheduler(
        registry=registry,
        count_new_examples=lambda hid, since: 500,
        list_active_households=lambda: ["h1"],
        orchestrate_for_household=lambda hid: _success("a" * 32),
        min_new_examples=50,
        min_hours_between_runs=0,   # rate limiter off
        now_fn=lambda: now,
        auto_approve=True,
    )
    tick = sched.run_tick()
    db.commit()

    assert tick.deployed == ["h1"]
