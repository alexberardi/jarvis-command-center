"""Scoring for the proactive-precision harness — pure, no I/O, CI-safe.

A prediction is the single action the matcher would propose for a bundle:
``(command, action)`` or ``None`` (propose nothing). Each scenario carries a
labeled ``expected`` prediction, and we classify (expected, predicted) into one
of five outcomes, then aggregate into the two rates the gate cares about:

  * ``false_positive_rate`` — of the NEGATIVE scenarios (nothing should fire),
    how often did the model propose anyway. This is the nag rate; it is THE
    reason the proactive reasoner is off by default, so it's the primary gate.
  * ``recall`` — of the POSITIVE scenarios, how often did it propose the right
    action. Guards against tightening the model into uselessness.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# A concrete proposal the matcher would surface, or None for "propose nothing".
Prediction = tuple[str, str] | None


class Outcome(str, Enum):
    TRUE_NEGATIVE = "true_negative"    # expected None, predicted None      → correctly quiet
    FALSE_POSITIVE = "false_positive"  # expected None, predicted an action → a NAG (the risk)
    TRUE_POSITIVE = "true_positive"    # expected A, predicted A            → right proposal
    MISS = "miss"                      # expected A, predicted None         → missed a real one
    WRONG = "wrong"                    # expected A, predicted B != A       → wrong proposal
    ERROR = "error"                    # model returned empty / errored     → NOT a decision at all

    # ERROR is the anti-fooling guard: an empty or errored model response collapses
    # to "propose nothing", which would otherwise score as a correct quiet — so a
    # dead model would post a perfect nag rate. It is set by the harness (never by
    # ``classify``, which only sees expected vs. predicted), excluded from every
    # rate denominator, and forces the gate to fail (the run is untrustworthy).


def classify(expected: Prediction, predicted: Prediction) -> Outcome:
    """Map a labeled ``expected`` and the matcher's ``predicted`` to an Outcome."""
    if expected is None:
        return Outcome.TRUE_NEGATIVE if predicted is None else Outcome.FALSE_POSITIVE
    if predicted is None:
        return Outcome.MISS
    return Outcome.TRUE_POSITIVE if predicted == expected else Outcome.WRONG


def _rate(numerator: int, denominator: int, *, empty: float) -> float:
    """numerator/denominator, or ``empty`` when the denominator is 0 (vacuous set)."""
    return numerator / denominator if denominator else empty


@dataclass(frozen=True)
class Metrics:
    total: int
    counts: dict[Outcome, int]

    # scenarios where the model failed to produce a usable response (empty/errored)
    errors: int                  # excluded from every rate below; forces a gate fail

    # negative set (nothing should fire) — the nag measurement
    negatives: int
    false_positive_rate: float   # FP / negatives  (0.0 when no negatives)
    quiet_accuracy: float        # TN / negatives = 1 - false_positive_rate

    # positive set (a specific action should fire) — the usefulness measurement
    positives: int
    recall: float                # TP / positives  (1.0 when no positives — vacuous)
    miss_rate: float             # MISS / positives
    wrong_rate: float            # WRONG / positives

    # everything the model proposed
    fires: int                   # TP + FP + WRONG
    precision: float             # TP / fires      (1.0 when it never fired — vacuous)


def aggregate(outcomes: list[Outcome]) -> Metrics:
    """Roll a list of per-scenario outcomes into rates."""
    counts = {o: 0 for o in Outcome}
    for o in outcomes:
        counts[o] += 1

    tn, fp = counts[Outcome.TRUE_NEGATIVE], counts[Outcome.FALSE_POSITIVE]
    tp, miss, wrong = counts[Outcome.TRUE_POSITIVE], counts[Outcome.MISS], counts[Outcome.WRONG]

    # ERROR is neither a negative nor a positive — the model never got to decide.
    negatives = tn + fp
    positives = tp + miss + wrong
    fires = tp + fp + wrong

    return Metrics(
        total=len(outcomes),
        counts=counts,
        errors=counts[Outcome.ERROR],
        negatives=negatives,
        false_positive_rate=_rate(fp, negatives, empty=0.0),
        quiet_accuracy=_rate(tn, negatives, empty=1.0),
        positives=positives,
        recall=_rate(tp, positives, empty=1.0),
        miss_rate=_rate(miss, positives, empty=0.0),
        wrong_rate=_rate(wrong, positives, empty=0.0),
        fires=fires,
        precision=_rate(tp, fires, empty=1.0),
    )


@dataclass(frozen=True)
class Thresholds:
    """The bar proactive proposals must clear on the corpus before graduating from
    off-by-default. FP-rate is the load-bearing one (nag ceiling)."""
    max_false_positive_rate: float = 0.05
    min_recall: float = 0.80


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reasons: list[str]   # one human-readable line per check, always ✓/✗ annotated


def evaluate_gate(metrics: Metrics, thresholds: Thresholds) -> GateResult:
    """Three checks must hold: NO errored scenarios (the run must be trustworthy),
    FP-rate at/under the ceiling, AND recall at/above the floor. Reasons list every
    check (passing and failing) so the report is legible."""
    no_errors = metrics.errors == 0
    fp_ok = metrics.false_positive_rate <= thresholds.max_false_positive_rate
    recall_ok = metrics.recall >= thresholds.min_recall
    reasons = [
        f"errors {metrics.errors} "
        + ("✓" if no_errors else "✗ (model/infra failure — results not trustworthy, rerun)"),
        f"false_positive_rate {metrics.false_positive_rate:.2f} "
        f"{'<=' if fp_ok else '>'} {thresholds.max_false_positive_rate:.2f} "
        f"{'✓' if fp_ok else '✗'}",
        f"recall {metrics.recall:.2f} "
        f"{'>=' if recall_ok else '<'} {thresholds.min_recall:.2f} "
        f"{'✓' if recall_ok else '✗'}",
    ]
    return GateResult(passed=no_errors and fp_ok and recall_ok, reasons=reasons)
