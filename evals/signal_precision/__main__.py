"""CLI: replay the precision corpus and report the nag/recall gate.

    python -m evals.signal_precision                 # live, vs the background model
    python -m evals.signal_precision --mock           # offline plumbing check (proposes nothing)
    python -m evals.signal_precision --json out.json   # also write a machine-readable baseline
    python -m evals.signal_precision --only appt       # filter scenarios by name substring

Exit code is 0 when the gate passes, 1 when it fails — so this can guard a
"turn proactive on" change in CI. ``--mock`` always exits 0 (it measures nothing).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from evals.signal_precision.harness import BackgroundModelClient, ScenarioRun, run_corpus
from evals.signal_precision.scenarios import CORPUS
from evals.signal_precision.scoring import Metrics, Outcome, Thresholds, aggregate, evaluate_gate

_SYMBOL = {
    Outcome.TRUE_POSITIVE: "✓ hit",
    Outcome.TRUE_NEGATIVE: "✓ quiet",
    Outcome.FALSE_POSITIVE: "✗ NAG",
    Outcome.MISS: "✗ miss",
    Outcome.WRONG: "✗ wrong",
    Outcome.ERROR: "⚠ ERROR",
}


class _AlwaysQuietClient:
    """Offline stand-in that proposes nothing — for a deterministic plumbing check
    (every negative scores quiet, every positive scores a miss)."""

    async def chat_completion(self, **_kwargs: Any) -> dict[str, Any]:
        return {"choices": [{"message": {"content": json.dumps({"matches": []})}}]}


def _fmt(pred: tuple[str, str] | None) -> str:
    return f"{pred[0]}.{pred[1]}" if pred else "—"


def _render_table(runs: list[ScenarioRun]) -> str:
    # Floor each column at its header label so nothing overflows when every cell is short.
    name_w = max(8, *(len(r.scenario.name) for r in runs))
    exp_w = max(8, *(len(_fmt(r.scenario.expect)) for r in runs))
    pred_w = max(9, *(len(_fmt(r.predicted)) for r in runs))
    lines = [
        f"  {'scenario':<{name_w}}  {'expected':<{exp_w}}  {'predicted':<{pred_w}}  outcome",
        f"  {'-' * name_w}  {'-' * exp_w}  {'-' * pred_w}  -------",
    ]
    for r in runs:
        line = (f"  {r.scenario.name:<{name_w}}  {_fmt(r.scenario.expect):<{exp_w}}  "
                f"{_fmt(r.predicted):<{pred_w}}  {_SYMBOL[r.outcome]}")
        if r.error:
            line += f"   [error: {r.error}]"
        lines.append(line)
    return "\n".join(lines)


def _render_summary(m: Metrics, thresholds: Thresholds) -> str:
    c = m.counts
    lines = [
        f"Scenarios: {m.total}  (positives {m.positives}, negatives {m.negatives}"
        + (f", ERRORED {m.errors}" if m.errors else "") + ")",
    ]
    if m.errors:
        lines.append(
            f"⚠  {m.errors} scenario(s) got no usable model response — "
            "the model/infra is broken; rates below are on the surviving subset and NOT trustworthy.")
    lines += [
        f"False-positive rate (nag):  {m.false_positive_rate:.2f}  "
        f"({c[Outcome.FALSE_POSITIVE]}/{m.negatives} negatives fired)   "
        f"[ceiling {thresholds.max_false_positive_rate:.2f}]",
        f"Recall (right proposal):    {m.recall:.2f}  "
        f"({c[Outcome.TRUE_POSITIVE]}/{m.positives} positives)          "
        f"[floor {thresholds.min_recall:.2f}]",
        f"Precision (of fires):       {m.precision:.2f}  "
        f"({c[Outcome.TRUE_POSITIVE]}/{m.fires} fired correctly)",
        f"Misses: {c[Outcome.MISS]}   Wrong-action: {c[Outcome.WRONG]}",
    ]
    return "\n".join(lines)


def _metrics_json(m: Metrics) -> dict[str, Any]:
    return {
        "total": m.total,
        "positives": m.positives,
        "negatives": m.negatives,
        "false_positive_rate": m.false_positive_rate,
        "quiet_accuracy": m.quiet_accuracy,
        "recall": m.recall,
        "precision": m.precision,
        "miss_rate": m.miss_rate,
        "wrong_rate": m.wrong_rate,
        "fires": m.fires,
        "counts": {o.value: n for o, n in m.counts.items()},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m evals.signal_precision",
                                 description="Signal-Bus proactive-proposal precision gate.")
    ap.add_argument("--max-fp", type=float, default=0.05, help="false-positive (nag) ceiling")
    ap.add_argument("--min-recall", type=float, default=0.80, help="recall floor")
    ap.add_argument("--model", default="background", help="LLM proxy model slot to measure")
    ap.add_argument("--only", default=None, help="only run scenarios whose name contains this")
    ap.add_argument("--concurrency", type=int, default=1, help="in-flight scenarios (default serial)")
    ap.add_argument("--mock", action="store_true", help="offline plumbing check (proposes nothing)")
    ap.add_argument("--json", dest="json_path", default=None, help="write metrics + runs to this path")
    args = ap.parse_args(argv)

    scenarios = [s for s in CORPUS if not args.only or args.only in s.name]
    if not scenarios:
        print(f"No scenarios match --only {args.only!r}", file=sys.stderr)
        return 2

    thresholds = Thresholds(max_false_positive_rate=args.max_fp, min_recall=args.min_recall)
    client = _AlwaysQuietClient() if args.mock else BackgroundModelClient(model=args.model)

    banner = "MOCK (offline plumbing check — proposes nothing)" if args.mock else \
        f"LIVE vs model slot '{args.model}'"
    print(f"\nSignal-Bus proactive precision — {banner}")
    print(f"Running {len(scenarios)} scenarios...\n")

    runs = asyncio.run(run_corpus(scenarios, client, concurrency=args.concurrency))
    metrics = aggregate([r.outcome for r in runs])
    gate = evaluate_gate(metrics, thresholds)

    print(_render_table(runs))
    print()
    print(_render_summary(metrics, thresholds))
    print("\nGate: " + ("PASS ✓" if gate.passed else "FAIL ✗"))
    for reason in gate.reasons:
        print(f"  - {reason}")

    if args.json_path:
        payload = {
            "metrics": _metrics_json(metrics),
            "gate": {"passed": gate.passed, "reasons": gate.reasons,
                     "thresholds": {"max_false_positive_rate": thresholds.max_false_positive_rate,
                                    "min_recall": thresholds.min_recall}},
            "runs": [{"name": r.scenario.name, "expected": _fmt(r.scenario.expect),
                      "predicted": _fmt(r.predicted), "outcome": r.outcome.value,
                      "error": r.error} for r in runs],
        }
        with open(args.json_path, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nWrote {args.json_path}")

    if args.mock:
        print("\n(mock mode measures nothing — exit 0)")
        return 0
    return 0 if gate.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
