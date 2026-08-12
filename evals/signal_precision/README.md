# Signal-Bus proactive-proposal precision harness

The proactive Signal reasoner (`app/services/situation_matcher_service.py`) is
**off by default**. The reason isn't that it doesn't work — it's that the failure
mode of a small background model on this task is *over-proposing*: nagging the user
with tap-to-confirm cards for situations that don't warrant one. Off-by-default is
the safe posture until we can **measure** that nag rate and show it's low enough.

This harness is that measurement — the explicit gate the Signal-Bus PRD calls for.
It replays a labeled corpus of Signal bundles through the **real** `match_situation`
matcher (same prompt, same anti-hallucination drop, same param validation) on the
**background** model slot, and scores:

| metric | what it means | why it's the gate |
|---|---|---|
| **false-positive rate** | of the NEGATIVE scenarios (nothing should fire), how often the model proposed anyway | this is the **nag rate** — the whole reason proactive is off |
| **recall** | of the POSITIVE scenarios, how often it proposed the right `(command, action)` | guards against tightening the model into uselessness |
| precision | of everything it proposed, how much was correct | context |

The corpus (`scenarios.py`) is deliberately **weighted toward negatives** — presence,
weather, media, diagnostics, ambient noise — because that's where the nagging shows
up. Positives span every `listening_signal_types` kind so recall isn't measured on
one command alone.

## Run it

Live, against the background model (needs command-center's env — app creds + LLM-proxy
discovery — so run from the CC venv or inside the container):

```bash
# from the repo root, using the CC virtualenv
JARVIS_LLM_PROXY_API_URL=http://localhost:7704 .venv/bin/python -m evals.signal_precision

# or inside the running container (copy the package in first, it's not bind-mounted)
docker cp evals <cc-container>:/app/evals
docker exec <cc-container> python -m evals.signal_precision
```

Useful flags:

```
--mock             offline plumbing check (proposes nothing; exits 0, measures nothing)
--only appt        only scenarios whose name contains this substring
--max-fp 0.05      false-positive (nag) ceiling  [default 0.05]
--min-recall 0.80  recall floor                  [default 0.80]
--model background LLM-proxy model slot to measure
--json out.json    also write a machine-readable baseline
--concurrency N    in-flight scenarios (default serial — the background slot is one model)
```

Exit code is **0 when the gate passes, 1 when it fails**, so a "turn proactive on"
change can be guarded by this in CI. `--mock` always exits 0.

## Interpreting a run

- **FP-rate over the ceiling** → the model nags. Do NOT turn proactive on. Look at
  which negatives fired (the `✗ NAG` rows) — usually a signal kind the prompt should
  more clearly treat as ambient, or a menu command that's too eager.
- **Recall under the floor** → the model misses real proposals (`✗ miss`) or picks the
  wrong action (`✗ wrong`). Check whether the right command was even in the menu and
  whether its `listening_signal_types` segmentation is steering the pick.

## Layout

| file | role |
|---|---|
| `scoring.py` | pure classify/aggregate/gate — the measurement definitions (CI-tested) |
| `harness.py` | drives real `match_situation` with a synthetic node fetch + background-model client |
| `scenarios.py` | the labeled corpus |
| `__main__.py` | CLI: run, render table + summary, gate, exit code |

Unit tests: `tests/test_signal_precision_harness.py` (scoring + plumbing, no live model).
