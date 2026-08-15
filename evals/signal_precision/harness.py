"""Harness plumbing — drive the REAL ``match_situation`` over a labeled corpus.

The only thing this adds over the shipped matcher is measurement scaffolding:
  * a synthetic ``fetch`` so a scenario's advertised commands stand in for a live
    node (no MQTT round-trip, no online node required),
  * a client wrapper that pins ``model="background"`` so we measure the SAME slot
    the proactive reasoner enqueues to (not the live voice slot), and
  * reduction of the matcher's ranked output to a single ``(command, action)``
    prediction (or ``None``), which the scoring layer classifies.

Everything downstream of the prompt — the prompt text, the anti-hallucination
drop, param validation — is the production code path, so a passing corpus means
the real reasoner behaves this way.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from app.services.proposal_matcher import match_situation

from evals.signal_precision.scoring import Outcome, Prediction, classify


@dataclass
class Scenario:
    """One labeled test case.

    ``bundle`` is the list of live Signals (``{source_key, kind, data}``).
    ``available_commands`` is a node tool report's ``available_commands`` list —
    each ``{command_name, listening_signal_types, proposable_actions:[...]}`` —
    i.e. the menu the matcher plans over. ``expect`` is the single proposal that
    SHOULD result, or ``None`` for "propose nothing" (a negative scenario).
    """
    name: str
    bundle: list[dict[str, Any]]
    available_commands: list[dict[str, Any]]
    expect: Prediction
    note: str = ""


@dataclass
class ScenarioRun:
    scenario: Scenario
    predicted: Prediction
    outcome: Outcome
    error: str | None = None


def _synthetic_fetch(
    available_commands: list[dict[str, Any]]
) -> Callable[[str], Awaitable[dict[str, Any]]]:
    """A ``capability_registry`` fetch that returns the scenario's advertised
    commands, standing in for a live node's ``report_tools`` answer."""

    async def fetch(_node_id: str) -> dict[str, Any]:
        return {"available_commands": available_commands}

    return fetch


class _RecordingClient:
    """Per-scenario wrapper that flags whether the model actually produced a usable
    response. ``match_situation`` swallows planner failures into an empty match list,
    so from its return value alone we can't tell a genuine "propose nothing" from an
    empty/errored response. This captures that distinction: ``failure`` is set when
    the completion raised or came back with empty content."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.failure: str | None = None

    async def chat_completion(self, **kwargs: Any) -> Any:
        try:
            resp = await self._inner.chat_completion(**kwargs)
        except Exception as exc:  # noqa: BLE001 — recorded, then re-raised for match_situation
            self.failure = f"{type(exc).__name__}: {exc}"
            raise
        from app.services.errand_planner import _extract_content

        if not (_extract_content(resp) or "").strip():
            self.failure = "empty model response"
        return resp


class BackgroundModelClient:
    """Wrap ``LLMProxyClient`` to pin ``model="background"``.

    ``_run_planner`` (which ``match_situation`` calls) never passes a model, so a
    bare client would hit the LIVE slot. The proactive reasoner enqueues with
    ``model="background"``; measuring anything else would be dishonest, so we
    inject it here.
    """

    def __init__(self, inner: Any = None, model: str = "background") -> None:
        if inner is None:
            from app.core.llm_proxy_client import LLMProxyClient

            inner = LLMProxyClient()
        self._inner = inner
        self._model = model

    async def chat_completion(self, **kwargs: Any) -> Any:
        kwargs.setdefault("model", self._model)
        return await self._inner.chat_completion(**kwargs)


async def run_scenario(scenario: Scenario, llm_client: Any) -> Prediction:
    """Run one scenario through ``match_situation`` and reduce to a single
    prediction: the top-ranked proposal's ``(command, action)``, or ``None`` when
    the matcher proposes nothing (the correct outcome for a negative scenario)."""
    matches = await match_situation(
        bundle=scenario.bundle,
        node_id="harness-node",
        llm_client=llm_client,
        fetch=_synthetic_fetch(scenario.available_commands),
    )
    if not matches:
        return None
    top = matches[0]
    return (top["command"], top["action"])


async def run_corpus(
    scenarios: list[Scenario], llm_client: Any, *, concurrency: int = 1
) -> list[ScenarioRun]:
    """Run every scenario and classify it. Serial by default — the background slot
    is a single model, so parallel requests just contend. ``concurrency`` >1 caps
    in-flight scenarios for when the backend can genuinely take them."""
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _one(sc: Scenario) -> ScenarioRun:
        async with sem:
            rec = _RecordingClient(llm_client)
            try:
                predicted = await run_scenario(sc, rec)
            except Exception as exc:  # noqa: BLE001 — belt-and-suspenders; usually swallowed
                predicted = None
                if rec.failure is None:
                    rec.failure = f"{type(exc).__name__}: {exc}"
            if rec.failure is not None:
                # An empty/errored response is NOT a decision — never let it score as a
                # correct quiet (which would let a dead model fake a perfect nag rate).
                return ScenarioRun(sc, None, Outcome.ERROR, rec.failure)
            return ScenarioRun(sc, predicted, classify(sc.expect, predicted), None)

    if concurrency <= 1:
        return [await _one(sc) for sc in scenarios]
    return list(await asyncio.gather(*(_one(sc) for sc in scenarios)))
