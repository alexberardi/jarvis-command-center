"""Unit tests for the Signal-Bus proactive-precision harness (``evals.signal_precision``).

Two layers, both offline (no live model):
  * scoring — pure classify/aggregate/gate logic (the measurement definitions).
  * harness plumbing — run_scenario/run_corpus drive the REAL ``match_situation``
    with an injected synthetic fetch + a scripted mock LLM, so we prove the harness
    reduces matcher output to a prediction and scores it, without a model.
"""
import asyncio
import json
from unittest.mock import AsyncMock

from evals.signal_precision import harness as H
from evals.signal_precision import scoring as S
from evals.signal_precision.scoring import Outcome


# ── scoring: classify ─────────────────────────────────────────────────────────
def test_classify_true_negative():
    assert S.classify(None, None) is Outcome.TRUE_NEGATIVE


def test_classify_false_positive():
    # expected nothing, but the model proposed something → a nag
    assert S.classify(None, ("add_event", "create_event")) is Outcome.FALSE_POSITIVE


def test_classify_true_positive():
    assert S.classify(("add_event", "create_event"), ("add_event", "create_event")) is Outcome.TRUE_POSITIVE


def test_classify_miss():
    assert S.classify(("add_event", "create_event"), None) is Outcome.MISS


def test_classify_wrong_action():
    # fired, but not the expected (command, action)
    assert S.classify(("add_event", "create_event"), ("play_music", "play")) is Outcome.WRONG
    assert S.classify(("add_event", "create_event"), ("add_event", "update_event")) is Outcome.WRONG


# ── scoring: ERROR (model/infra failure must not masquerade as a clean quiet) ─
def test_aggregate_excludes_errors_from_negatives_and_positives():
    # An empty/errored model response is Outcome.ERROR: it is neither a correct
    # quiet nor a real miss, and must not inflate quiet_accuracy.
    m = S.aggregate([Outcome.ERROR, Outcome.ERROR, Outcome.TRUE_NEGATIVE, Outcome.TRUE_POSITIVE])
    assert m.errors == 2
    assert m.negatives == 1          # the one real TN, NOT the 2 errors
    assert m.positives == 1
    assert m.false_positive_rate == 0.0


def test_gate_fails_when_any_scenario_errored_even_if_rates_look_clean():
    # 8 quiet + 1 hit + 1 error → fp 0, recall 1... but the error means the run is
    # untrustworthy, so the gate must still fail.
    m = S.aggregate([Outcome.TRUE_NEGATIVE] * 8 + [Outcome.TRUE_POSITIVE, Outcome.ERROR])
    g = S.evaluate_gate(m, S.Thresholds(max_false_positive_rate=0.05, min_recall=0.8))
    assert g.passed is False
    assert any("error" in r.lower() and "✗" in r for r in g.reasons)


# ── scoring: aggregate ────────────────────────────────────────────────────────
def test_aggregate_rates():
    outcomes = [
        Outcome.TRUE_NEGATIVE, Outcome.TRUE_NEGATIVE, Outcome.TRUE_NEGATIVE, Outcome.FALSE_POSITIVE,  # 4 negatives, 1 FP
        Outcome.TRUE_POSITIVE, Outcome.TRUE_POSITIVE, Outcome.MISS, Outcome.WRONG,                    # 4 positives, 2 TP
    ]
    m = S.aggregate(outcomes)
    assert m.total == 8
    assert m.negatives == 4 and m.positives == 4
    # negatives: 1 FP of 4 → 0.25 FP-rate, 0.75 quiet accuracy
    assert m.false_positive_rate == 0.25
    assert m.quiet_accuracy == 0.75
    # positives: 2 TP of 4 → recall 0.5; 1 miss; 1 wrong
    assert m.recall == 0.5
    assert m.miss_rate == 0.25
    assert m.wrong_rate == 0.25
    # fires = TP + FP + WRONG = 2 + 1 + 1 = 4; precision = 2/4 = 0.5
    assert m.fires == 4
    assert m.precision == 0.5


def test_aggregate_all_quiet_no_positives_is_vacuously_clean():
    m = S.aggregate([Outcome.TRUE_NEGATIVE, Outcome.TRUE_NEGATIVE])
    assert m.false_positive_rate == 0.0
    assert m.recall == 1.0        # no positives to miss → vacuously perfect
    assert m.precision == 1.0     # never fired → no bad fire


# ── scoring: gate ─────────────────────────────────────────────────────────────
def test_gate_passes_when_under_thresholds():
    m = S.aggregate([Outcome.TRUE_NEGATIVE] * 19 + [Outcome.TRUE_POSITIVE])
    g = S.evaluate_gate(m, S.Thresholds(max_false_positive_rate=0.05, min_recall=0.8))
    assert g.passed is True
    assert any("false_positive_rate" in r for r in g.reasons)


def test_gate_fails_on_high_false_positive_rate():
    # 2 FP of 4 negatives = 0.5 > 0.05 → fail (the nag ceiling is what matters)
    m = S.aggregate([Outcome.FALSE_POSITIVE, Outcome.FALSE_POSITIVE, Outcome.TRUE_NEGATIVE,
                     Outcome.TRUE_NEGATIVE, Outcome.TRUE_POSITIVE])
    g = S.evaluate_gate(m, S.Thresholds(max_false_positive_rate=0.05, min_recall=0.8))
    assert g.passed is False
    assert any("false_positive_rate" in r and "✗" in r for r in g.reasons)


def test_gate_fails_on_low_recall():
    m = S.aggregate([Outcome.TRUE_NEGATIVE] * 5 + [Outcome.MISS, Outcome.MISS, Outcome.TRUE_POSITIVE])
    g = S.evaluate_gate(m, S.Thresholds(max_false_positive_rate=0.05, min_recall=0.8))
    assert g.passed is False
    assert any("recall" in r and "✗" in r for r in g.reasons)


# ── harness plumbing (real match_situation, scripted LLM) ─────────────────────
def _report_add_event():
    """A node tool report advertising add_event.create_event as proposable."""
    return {"available_commands": [
        {"command_name": "add_event", "listening_signal_types": ["appt.detected"],
         "proposable_actions": [
            {"callback": "create_event",
             "params": [{"name": "title", "required": True},
                        {"name": "start", "required": True},
                        {"name": "idempotency_key", "required": True}],
             "idempotency_param": "idempotency_key",
             "card_title": "Add to your calendar?"}]}]}


def _scripted_llm(matches):
    c = AsyncMock()
    c.chat_completion = AsyncMock(
        return_value={"choices": [{"message": {"content": json.dumps({"matches": matches})}}]})
    return c


def test_run_scenario_reduces_match_to_prediction():
    sc = H.Scenario(
        name="appt", expect=("add_event", "create_event"),
        bundle=[{"source_key": "a1", "kind": "appt.detected",
                 "data": {"title": "Dentist", "start": "2026-08-13T09:00:00"}}],
        available_commands=_report_add_event()["available_commands"])
    llm = _scripted_llm([{"command": "add_event", "action": "create_event",
                          "args": {"title": "Dentist", "start": "2026-08-13T09:00:00"}, "sources": [0]}])
    predicted = asyncio.run(H.run_scenario(sc, llm))
    assert predicted == ("add_event", "create_event")


def test_run_scenario_none_when_matcher_returns_empty():
    sc = H.Scenario(
        name="quiet", expect=None,
        bundle=[{"source_key": "p1", "kind": "presence.seen", "data": {"summary": "Alex is home"}}],
        available_commands=_report_add_event()["available_commands"])
    predicted = asyncio.run(H.run_scenario(sc, _scripted_llm([])))
    assert predicted is None


def test_run_corpus_scores_each_scenario():
    positive = H.Scenario(
        name="pos", expect=("add_event", "create_event"),
        bundle=[{"source_key": "a1", "kind": "appt.detected",
                 "data": {"title": "Dentist", "start": "2026-08-13T09:00:00"}}],
        available_commands=_report_add_event()["available_commands"])
    negative = H.Scenario(
        name="neg", expect=None,
        bundle=[{"source_key": "p1", "kind": "presence.seen", "data": {"summary": "Alex is home"}}],
        available_commands=_report_add_event()["available_commands"])

    # A single scripted client that always proposes the appt match: the positive
    # scores TRUE_POSITIVE, the negative scores FALSE_POSITIVE (it nagged).
    llm = _scripted_llm([{"command": "add_event", "action": "create_event",
                          "args": {"title": "Dentist", "start": "2026-08-13T09:00:00"}, "sources": [0]}])
    runs = asyncio.run(H.run_corpus([positive, negative], llm))
    by_name = {r.scenario.name: r.outcome for r in runs}
    assert by_name["pos"] is Outcome.TRUE_POSITIVE
    assert by_name["neg"] is Outcome.FALSE_POSITIVE


def _empty_content_llm():
    """A client that returns a well-formed response whose content is EMPTY — the
    Gemma/degraded-model failure signature (finish=stop, content='')."""
    c = AsyncMock()
    c.chat_completion = AsyncMock(return_value={"choices": [{"message": {"content": ""}}]})
    return c


def _raising_llm():
    c = AsyncMock()
    c.chat_completion = AsyncMock(side_effect=RuntimeError("500 Internal Server Error"))
    return c


def test_run_corpus_marks_empty_response_as_error_not_quiet():
    # A NEGATIVE scenario + an empty-returning model. Naively this collapses to
    # None and scores as a correct quiet — the trap. It must be Outcome.ERROR.
    negative = H.Scenario(
        name="neg", expect=None,
        bundle=[{"source_key": "p1", "kind": "presence.seen", "data": {"summary": "home"}}],
        available_commands=_report_add_event()["available_commands"])
    runs = asyncio.run(H.run_corpus([negative], _empty_content_llm()))
    assert runs[0].outcome is Outcome.ERROR
    assert runs[0].error


def test_run_corpus_marks_llm_exception_as_error():
    positive = H.Scenario(
        name="pos", expect=("add_event", "create_event"),
        bundle=[{"source_key": "a1", "kind": "appt.detected", "data": {"title": "x", "start": "y"}}],
        available_commands=_report_add_event()["available_commands"])
    runs = asyncio.run(H.run_corpus([positive], _raising_llm()))
    assert runs[0].outcome is Outcome.ERROR
    assert "500" in (runs[0].error or "")


def test_background_model_client_pins_the_background_slot():
    inner = AsyncMock()
    inner.chat_completion = AsyncMock(return_value={"choices": [{"message": {"content": "{}"}}]})
    client = H.BackgroundModelClient(inner=inner)
    asyncio.run(client.chat_completion(messages=[{"role": "user", "content": "hi"}],
                                       temperature=0, max_tokens=10))
    # _run_planner never passes a model; the wrapper must inject the background slot.
    assert inner.chat_completion.call_args.kwargs["model"] == "background"
    assert inner.chat_completion.call_args.kwargs["max_tokens"] == 10
