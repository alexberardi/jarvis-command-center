"""Signal-Bus proactive-proposal precision harness.

The proactive reasoner (``situation_matcher_service``) is OFF by default because
the risk with a small background model is *over-proposing* — nagging the user with
cards for situations that don't warrant one. This harness measures exactly that:
it replays a labeled corpus of Signal bundles through the real ``match_situation``
matcher on the background model and scores how often it correctly stays quiet
(the negatives) vs. proposes the right action (the positives).

It is the explicit GATE the PRD calls for: proactive proposals should not graduate
from off-by-default until the model clears a false-positive-rate ceiling on this
corpus.

Entry point: ``python -m evals.signal_precision`` (see ``__main__``).
"""

from evals.signal_precision.scoring import (
    GateResult,
    Metrics,
    Outcome,
    Thresholds,
    aggregate,
    classify,
    evaluate_gate,
)

__all__ = [
    "GateResult",
    "Metrics",
    "Outcome",
    "Thresholds",
    "aggregate",
    "classify",
    "evaluate_gate",
]
