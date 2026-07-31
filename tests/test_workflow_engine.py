"""Workflow engine — the durable, resumable, event-driven job engine (errand-runner
PRD §2.5-2.6), tested BLACK-BOX against a real in-memory store rather than ORM mocks.

Errands are the engine's first consumer; these tests exercise the engine's own
contract — the handler registry (SEAM 2), the signal primitive (SEAM 1,
``deliver_signal``), the progress channel (SEAM 3), and the run/finish/suspend
orchestration (``run_workflow`` + ``_drive_run``) — with a ``FakeStore`` and a fake
consumer, so a bug in the engine can't hide behind a mock that agrees with it.
"""

import asyncio

import pytest

from app.services import workflow_engine as we
from app.services.workflow_engine import (
    NodeCommandHandler,
    PhoneCallHandler,
    ServerToolHandler,
    Suspend,
    WorkflowRun,
    resolve_handler,
    terminal_state,
)

FAKE_KIND = "faketest"


class FakeStore(we.WorkflowStore):
    """A dict-backed WorkflowStore over mutable ``WorkflowRun`` objects. ``claim_running``
    honours the atomic ``waiting→running`` guard so idempotency is really tested."""

    def __init__(self, runs):
        self.runs = {r.id: r for r in runs}
        self.closed = False
        self.claims = 0

    def load(self, workflow_id):
        return self.runs.get(workflow_id)

    def claim_running(self, workflow_id):
        run = self.runs.get(workflow_id)
        if run is not None and run.state == "waiting":
            run.state = "running"
            self.claims += 1
            return True
        return False

    def save_waiting(self, run, cursor, results, *, signal_kind="phone_call", wake_at=None):
        stored = self.runs[run.id]
        stored.state = "waiting"
        stored.cursor = cursor
        stored.results = results
        stored.waiting_on = signal_kind
        stored.wake_at = wake_at

    def save_terminal(self, run, state, error):
        stored = self.runs[run.id]
        stored.state = state

    def close(self):
        self.closed = True


def _run(state="running", cursor=0, results=None, steps=None):
    return WorkflowRun(
        id="wf_1", kind=FAKE_KIND, household_id="hh-1", node_id="node-1", user_id=7,
        goal="do the thing", title="Thing",
        steps=steps if steps is not None else [{"command": "x", "args": {}, "label": "X"}],
        cursor=cursor, results=list(results or []), state=state,
    )


@pytest.fixture
def engine(monkeypatch):
    """Install a FakeStore + a fake ``faketest`` consumer, capturing every progress
    event. Restored automatically (store via monkeypatch; consumer popped)."""
    calls = {"run": [], "finalize": []}
    events = []

    async def run_fn(household_id, node_id, steps, *, user_id, goal, errand_id,
                     start_index, prior_results):
        calls["run"].append({
            "household_id": household_id, "node_id": node_id, "errand_id": errand_id,
            "start_index": start_index, "prior_results": list(prior_results),
        })
        behaviour = calls.get("_run_returns")
        if behaviour is not None:
            r = behaviour.pop(0)
            if isinstance(r, Exception):
                raise r
            # A returned Suspend carries the accumulated prior_results forward, so a
            # re-suspend's persisted results include the earlier step outcomes.
            if isinstance(r, Suspend) and not r.results:
                r.results = list(prior_results)
            return r
        return {"status": "success", "message": "ok", "passed": 1, "failed": 0, "results": []}

    async def finalize_fn(goal, results):
        calls["finalize"].append({"goal": goal, "results": list(results)})
        return {"status": "failed", "message": "call failed", "passed": 0, "failed": 1,
                "results": list(results)}

    class Emitter(we.ProgressEmitter):
        def emit(self, run, event):
            events.append((run.id, event.type, event.result))

    store_holder = {}

    def factory():
        return store_holder["store"]

    monkeypatch.setattr(we, "_STORE_FACTORY", factory)
    we.register_consumer(FAKE_KIND, we.WorkflowConsumer(run_fn=run_fn, finalize_fn=finalize_fn,
                                                        progress=Emitter()))
    try:
        yield {"calls": calls, "events": events, "store_holder": store_holder}
    finally:
        we._CONSUMERS.pop(FAKE_KIND, None)


# ── SEAM 2: the handler registry ──────────────────────────────────────────────


def test_resolve_handler_precedence(monkeypatch):
    """phone_call (deferred) is resolved BEFORE server_tool even though make_phone_call
    IS a server tool; anything else falls through to the node handler."""
    monkeypatch.setattr("app.core.tool_registry.tool_registry.has_tool",
                        lambda name: name in ("make_phone_call", "quick_search"))
    assert isinstance(resolve_handler("make_phone_call"), PhoneCallHandler)  # deferred wins
    assert isinstance(resolve_handler("quick_search"), ServerToolHandler)
    assert isinstance(resolve_handler("get_weather"), NodeCommandHandler)   # fallback


def _call(state="done", contact_name="CVS", goal_achieved=None, error_message=None, outcome_json=None):
    return type("S", (), {"state": state, "contact_name": contact_name,
                          "goal_achieved": goal_achieved, "error_message": error_message,
                          "outcome_json": outcome_json})()


def test_phone_handler_interpret_signal():
    h = PhoneCallHandler()
    steps = [{"command": "make_phone_call", "label": "Call CVS"}]

    # 'done' AND the goal explicitly achieved → success (continue).
    out = h.interpret_signal(_call("done", goal_achieved=True), steps, 0)
    assert out["success"] is True and out["label"] == "Call CVS" and "CVS" in out["message"]

    # 'done' but the goal NOT achieved → fail-fast (this is the pharmacy case).
    out = h.interpret_signal(_call("done", goal_achieved=False), steps, 0)
    assert out["success"] is False and "wasn't achieved" in out["error"]

    # 'done' but the result is UNCONFIRMED (goal_achieved None) → fail-fast too — a
    # stranger-facing call only chains the next call on an explicit success.
    out = h.interpret_signal(_call("done", goal_achieved=None), steps, 0)
    assert out["success"] is False and "couldn't confirm" in out["error"]

    # goal_achieved read from outcome_json (the real ORM/snapshot shape).
    out = h.interpret_signal(_call("done", goal_achieved=None,
                                   outcome_json='{"goal_achieved": true}'), steps, 0)
    assert out["success"] is True

    assert h.interpret_signal(_call("failed"), steps, 0)["success"] is False
    assert "didn't go through" in h.interpret_signal(_call("failed"), steps, 0)["error"]
    assert "declined" in h.interpret_signal(_call("declined", contact_name=None), steps, 0)["error"]


def test_phone_step_result_carries_wrapup_summary():
    """A later step needs to know what an earlier call established — the step outcome
    now carries the gateway's wrapup summary + confirmed facts."""
    h = PhoneCallHandler()
    steps = [{"command": "make_phone_call", "label": "Call the pharmacy"}]
    oj = ('{"goal_achieved": true, "summary": "Authorized the refill.", '
          '"facts": ["acetaminophen", "authorized"]}')
    out = h.interpret_signal(_call("done", goal_achieved=None, outcome_json=oj), steps, 0)
    assert out["success"] is True
    assert "Authorized the refill." in out["summary"]
    assert "acetaminophen" in out["summary"]  # confirmed facts folded in
    # A failed call still summarizes what happened, for the later step's context.
    out = h.interpret_signal(
        _call("done", goal_achieved=False,
              outcome_json='{"goal_achieved": false, "summary": "Could not confirm the medication."}'),
        steps, 0)
    assert out["success"] is False and "Could not confirm the medication." in out["summary"]


def test_prior_steps_context_summarizes_earlier_steps():
    results = [
        {"label": "Call the pharmacy", "summary": "Refill authorized for acetaminophen.",
         "success": True},
        {"label": "Look up hours", "message": "Open until 9pm.", "success": True},
        {"label": "Blank step", "success": True},  # no detail → skipped
    ]
    ctx = we._prior_steps_context(results)
    assert "Earlier in this errand" in ctx
    assert "Call the pharmacy: Refill authorized for acetaminophen." in ctx
    assert "Look up hours: Open until 9pm." in ctx
    assert "Blank step" not in ctx
    assert we._prior_steps_context([]) == "" and we._prior_steps_context(None) == ""


def test_interpret_gateway_outcome_contract():
    """CONTRACT PIN (errand ↔ phone gateway). The gateway wrapup now emits a STRUCTURED
    ``goal_achieved`` (dial_worker._assess); this pins how the errand reads it:
      - goal_achieved=true  → the call succeeded → CONTINUE (chain the next step).
      - goal_achieved=false → not accomplished → fail-fast.
      - ABSENT (a legacy outcome, or the wrapup model couldn't be assessed → None) →
        UNCONFIRMED → fail-fast (defensive; don't chain on an ambiguous result).
    Live bug 2026-07-30: a call the pharmacist AUTHORIZED reported as errand-failed
    because the old outcome had no goal_achieved and this handler fail-fasted on absent.
    """
    import json as _json

    h = PhoneCallHandler()
    steps = [{"command": "make_phone_call", "label": "Call the pharmacy"}]

    ok = type("S", (), {"state": "done", "contact_name": "CVS", "error_message": None,
                        "outcome_json": _json.dumps(
                            {"summary": "Authorization granted.", "goal_achieved": True,
                             "facts": ["authorized"], "turns": 5})})()
    assert h.interpret_signal(ok, steps, 0)["success"] is True

    no = type("S", (), {"state": "done", "contact_name": "CVS", "error_message": None,
                        "outcome_json": _json.dumps(
                            {"summary": "Still pending.", "goal_achieved": False, "turns": 5})})()
    assert h.interpret_signal(no, steps, 0)["success"] is False

    # Legacy / unassessed outcome with NO goal_achieved key → unconfirmed → fail-fast.
    legacy = type("S", (), {"state": "done", "contact_name": "CVS", "error_message": None,
                            "outcome_json": _json.dumps(
                                {"summary": "Called the pharmacy.", "facts": [], "turns": 4})})()
    out = h.interpret_signal(legacy, steps, 0)
    assert out["success"] is False and "couldn't confirm" in out["error"]


def test_terminal_state_mapping():
    assert terminal_state({"status": "success"}) == "done"
    assert terminal_state({"status": "partial"}) == "partial"
    assert terminal_state({"status": "failed"}) == "failed"
    assert terminal_state({}) == "failed"


# ── run_workflow: the Run-tap entrypoint ──────────────────────────────────────


def test_run_workflow_completes_and_reports(engine):
    store = FakeStore([_run(state="running")])
    engine["store_holder"]["store"] = store
    ok = asyncio.run(we.run_workflow("wf_1"))
    assert ok is True
    assert store.runs["wf_1"].state == "done"        # terminal persisted
    assert len(engine["events"]) == 1                 # exactly one progress event
    assert engine["events"][0][0] == "wf_1" and engine["events"][0][1] == "complete"
    assert engine["calls"]["run"][0]["start_index"] == 0
    assert store.closed is True                       # store closed


def test_run_workflow_suspends_on_deferred_step(engine):
    engine["calls"]["_run_returns"] = [Suspend(cursor=1, signal_key="sess-1")]
    store = FakeStore([_run(state="running")])
    engine["store_holder"]["store"] = store
    asyncio.run(we.run_workflow("wf_1"))
    assert store.runs["wf_1"].state == "waiting" and store.runs["wf_1"].cursor == 1
    assert engine["events"] == []                     # NO completion card while waiting


def test_run_workflow_never_vanishes_on_crash(engine):
    engine["calls"]["_run_returns"] = [RuntimeError("boom")]
    store = FakeStore([_run(state="running")])
    engine["store_holder"]["store"] = store
    ok = asyncio.run(we.run_workflow("wf_1"))
    assert ok is True
    assert store.runs["wf_1"].state == "failed"       # terminal, not stuck running
    assert engine["events"][0][1] == "complete"       # an honest failure card


def test_run_workflow_missing_run_is_noop(engine):
    engine["store_holder"]["store"] = FakeStore([])
    assert asyncio.run(we.run_workflow("nope")) is False


# ── SEAM 1: deliver_signal (resume) ───────────────────────────────────────────


def test_deliver_signal_resumes_from_cursor_on_success(engine):
    run = _run(state="waiting", cursor=1,
               steps=[{"command": "make_phone_call", "label": "Call"},
                      {"command": "get_weather", "label": "W"}])
    store = FakeStore([run])
    engine["store_holder"]["store"] = store
    done = _call("done", goal_achieved=True)

    ok = asyncio.run(we.deliver_signal("wf_1", 0, "phone_call", done))
    assert ok is True
    assert store.claims == 1                            # atomically claimed once
    call = engine["calls"]["run"][0]
    assert call["start_index"] == 1                     # resumed from the cursor
    assert call["prior_results"][0]["success"] is True  # the call outcome carried in
    assert store.runs["wf_1"].state == "done"
    assert engine["events"][0][1] == "complete"


def test_deliver_signal_fail_fast_on_call_failure(engine):
    run = _run(state="waiting", cursor=1,
               steps=[{"command": "make_phone_call", "label": "Call CVS"},
                      {"command": "make_phone_call", "label": "Call Dr"}])
    store = FakeStore([run])
    engine["store_holder"]["store"] = store
    failed = type("S", (), {"state": "failed", "contact_name": "CVS", "error_message": None})()

    ok = asyncio.run(we.deliver_signal("wf_1", 0, "phone_call", failed))
    assert ok is True
    assert engine["calls"]["run"] == []                 # fail-fast: run_fn NEVER called (no 2nd call)
    assert engine["calls"]["finalize"][0]["results"][0]["success"] is False
    assert store.runs["wf_1"].state == "failed"
    assert engine["events"][0][1] == "complete"


def test_deliver_signal_suspends_again_on_next_deferred_step(engine):
    engine["calls"]["_run_returns"] = [Suspend(cursor=2, signal_key="sess-2")]
    run = _run(state="waiting", cursor=1,
               steps=[{"command": "make_phone_call", "label": "1"},
                      {"command": "make_phone_call", "label": "2"}])
    store = FakeStore([run])
    engine["store_holder"]["store"] = store
    done = _call("done", goal_achieved=True)

    asyncio.run(we.deliver_signal("wf_1", 0, "phone_call", done))
    assert store.runs["wf_1"].state == "waiting" and store.runs["wf_1"].cursor == 2
    assert engine["events"] == []                       # still waiting → no completion card
    # the first call's outcome accumulated into the re-suspended run's results
    assert store.runs["wf_1"].results[0]["command"] == "make_phone_call"
    assert store.runs["wf_1"].waiting_on == "phone_call"


def test_deliver_signal_noop_when_not_waiting(engine):
    store = FakeStore([_run(state="done", cursor=1)])
    engine["store_holder"]["store"] = store
    done = _call("done", goal_achieved=True)
    assert asyncio.run(we.deliver_signal("wf_1", 0, "phone_call", done)) is False
    assert engine["calls"]["run"] == [] and store.claims == 0


def test_deliver_signal_noop_on_cursor_mismatch(engine):
    store = FakeStore([_run(state="waiting", cursor=5)])   # cursor != step+1
    engine["store_holder"]["store"] = store
    done = _call("done", goal_achieved=True)
    assert asyncio.run(we.deliver_signal("wf_1", 0, "phone_call", done)) is False
    assert store.claims == 0


def test_deliver_signal_is_idempotent(engine):
    """A second delivery for the same step (immediate hook + safety sweep both fire)
    is a no-op — the first claim flipped waiting→running, so the second can't claim."""
    run = _run(state="waiting", cursor=1,
               steps=[{"command": "make_phone_call", "label": "Call"},
                      {"command": "get_weather", "label": "W"}])
    store = FakeStore([run])
    engine["store_holder"]["store"] = store
    done = _call("done", goal_achieved=True)

    first = asyncio.run(we.deliver_signal("wf_1", 0, "phone_call", done))
    second = asyncio.run(we.deliver_signal("wf_1", 0, "phone_call", done))
    assert first is True and second is False
    assert store.claims == 1                             # only ONE trigger drove the run
    assert len(engine["calls"]["run"]) == 1


def test_deliver_signal_no_store_is_safe(monkeypatch):
    monkeypatch.setattr(we, "_STORE_FACTORY", None)
    done = _call("done", goal_achieved=True)
    assert asyncio.run(we.deliver_signal("wf_1", 0, "phone_call", done)) is False


# ── SEAM 2 + the TIMER wakeup source (wait_for) ───────────────────────────────


def _ctx(workflow_id="wf_1"):
    return we.WorkflowContext(household_id="hh-1", node_id="node-1", user_id=7,
                              goal="g", workflow_id=workflow_id)


def test_wait_for_resolves_to_the_timer_handler(monkeypatch):
    monkeypatch.setattr("app.core.tool_registry.tool_registry.has_tool", lambda name: False)
    h = resolve_handler("wait_for")
    assert isinstance(h, we.WaitForHandler) and h.kind == "deferred" and h.signal_kind == "timer"


def test_wait_for_suspends_on_a_future_timer():
    h = we.WaitForHandler()
    ret = asyncio.run(h.run(_ctx(), "wait_for", {"delay_seconds": 300}, "Wait 5m", 1))
    assert isinstance(ret, Suspend)
    assert ret.signal_kind == "timer" and ret.cursor == 2 and ret.wake_at is not None
    assert ret.signal_key  # the wake instant, ISO


def test_wait_for_in_the_past_runs_through_without_suspending():
    h = we.WaitForHandler()
    ret = asyncio.run(h.run(_ctx(), "wait_for", {"until": "2000-01-01T00:00:00"}, "Wait", 0))
    assert not isinstance(ret, Suspend)
    assert ret["success"] is True  # already over → no suspend


def test_wait_for_bad_args_is_a_failed_step():
    h = we.WaitForHandler()
    ret = asyncio.run(h.run(_ctx(), "wait_for", {"until": "not a date"}, "Wait", 0))
    assert not isinstance(ret, Suspend) and ret["success"] is False


def test_wait_for_without_workflow_link_fails():
    h = we.WaitForHandler()
    ret = asyncio.run(h.run(_ctx(workflow_id=None), "wait_for", {"delay_seconds": 60}, "Wait", 0))
    assert not isinstance(ret, Suspend) and ret["success"] is False


def test_timer_interpret_signal_is_always_success():
    h = we.WaitForHandler()
    out = h.interpret_signal(type("S", (), {"state": "fired"})(),
                             [{"command": "wait_for", "label": "Wait 5m"}], 0)
    assert out["success"] is True and out["label"] == "Wait 5m"


def test_deliver_signal_timer_resumes_the_run(engine):
    """A timer wakeup resumes a waiting run through the SAME deliver_signal primitive
    the phone outcome uses — the engine's second signal source (proves generality)."""
    run = _run(state="waiting", cursor=1,
               steps=[{"command": "wait_for", "label": "Wait"},
                      {"command": "get_weather", "label": "W"}])
    store = FakeStore([run])
    engine["store_holder"]["store"] = store

    ok = asyncio.run(we.deliver_signal("wf_1", 0, "timer", type("S", (), {"state": "fired"})()))
    assert ok is True and store.claims == 1
    call = engine["calls"]["run"][0]
    assert call["start_index"] == 1  # resumed from the cursor
    assert call["prior_results"][0]["command"] == "wait_for"  # the wait outcome carried in
    assert store.runs["wf_1"].state == "done"
    assert engine["events"][0][1] == "complete"
