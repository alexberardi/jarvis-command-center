"""Training orchestrator (Phase 4).

Pulls together Phase 3 + Phase 3.5, submits a training job to llm-proxy,
polls until complete, and invokes the eval gate. Returns a clean result
dict the CLI layer can render or Phase 5's scheduler can consume.

This module is DB-free and network-aware — it talks to llm-proxy via the
enqueue + status endpoints documented in
`jarvis-llm-proxy-api/prds/adapter-training-queue-contract.md`.
"""

from __future__ import annotations

import logging
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import httpx

from app.services.training_data_extractor import ExtractedRow

logger = logging.getLogger("uvicorn")


# --------------------------------------------------------------------------
# Result shapes
# --------------------------------------------------------------------------


@dataclass
class OrchestratorResult:
    status: str  # "success" | "training_failed" | "eval_failed" | "enqueue_failed"
    adapter_hash: str | None
    job_id: str | None
    training_seconds: float | None
    dataset_rows: int
    by_source: dict[str, int]
    eval: dict[str, Any] | None = None  # populated when eval runs
    error: str | None = None
    timings: dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------
# dataset_ref shaping
# --------------------------------------------------------------------------


def build_dataset_ref(rows: list[dict]) -> dict:
    """Group dataset_ref-shaped rows by command_name into the contract shape."""
    by_cmd: dict[str, list[dict]] = {}
    for r in rows:
        name = r["expected_tool_call"]["name"]
        by_cmd.setdefault(name, []).append(r)
    return {
        "format": "inline-json",
        "data": {
            "commands": [
                {"command_name": name, "examples": exs}
                for name, exs in by_cmd.items()
            ],
        },
    }


# --------------------------------------------------------------------------
# llm-proxy client helpers
# --------------------------------------------------------------------------


def enqueue_training(
    *,
    llm_proxy_url: str,
    dataset_ref: dict,
    base_model_id: str,
    node_id: str,
    params: dict,
    app_id: str,
    app_key: str,
    callback_url: str = "",
    ttl_seconds: int = 7200,
) -> str:
    """POST to llm-proxy /internal/queue/enqueue. Returns job_id."""
    job_id = str(uuid.uuid4())
    body = {
        "job_id": job_id,
        "job_type": "adapter_train",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "idempotency_key": job_id,
        "ttl_seconds": ttl_seconds,
        "request": {
            "node_id": node_id,
            "base_model_id": base_model_id,
            "dataset_ref": dataset_ref,
            "params": params,
        },
        "callback": {"url": callback_url, "auth_type": "internal"},
    }
    headers = {
        "X-Jarvis-App-Id": app_id,
        "X-Jarvis-App-Key": app_key,
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=30) as client:
        r = client.post(
            f"{llm_proxy_url.rstrip('/')}/internal/queue/enqueue",
            json=body, headers=headers,
        )
        r.raise_for_status()
        resp = r.json()
    logger.info("orchestrator: enqueued job %s (deduped=%s)", job_id, resp.get("deduped"))
    return job_id


def poll_status(
    *,
    llm_proxy_url: str,
    job_id: str,
    app_id: str,
    app_key: str,
    interval_seconds: float = 5.0,
    timeout_seconds: float = 7200.0,
) -> dict:
    """Poll /v1/training/status/{job_id} until COMPLETE or FAILED."""
    deadline = time.time() + timeout_seconds
    headers = {"X-Jarvis-App-Id": app_id, "X-Jarvis-App-Key": app_key}
    last_status: str | None = None
    with httpx.Client(timeout=15) as client:
        while time.time() < deadline:
            try:
                r = client.get(
                    f"{llm_proxy_url.rstrip('/')}/v1/training/status/{job_id}",
                    headers=headers,
                )
                if r.status_code == 200:
                    data = r.json()
                    status = data.get("status")
                    if status != last_status:
                        logger.info("orchestrator: job %s status=%s progress=%s",
                                    job_id, status, data.get("progress_pct"))
                        last_status = status
                    if status in ("COMPLETE", "FAILED"):
                        return data
            except (httpx.HTTPError, ValueError) as e:
                logger.warning("orchestrator: poll error: %s", e)
            time.sleep(interval_seconds)
    raise TimeoutError(f"Job {job_id} did not finish in {timeout_seconds}s")


# --------------------------------------------------------------------------
# Eval invocation — defaults to docker exec against the jarvis-node container
# --------------------------------------------------------------------------


def invoke_eval_via_docker(
    *,
    adapter_hash: str,
    node_container: str = "jarvis-node",
    eval_script_path: str = "/app/eval_gate.py",
    output_path: str = "/app/eval_artifacts/orchestrator_eval.json",
    threshold: float | None = None,
    tolerance: float = 1.0,
    timeout_seconds: int = 1800,
    with_baseline: bool = True,
) -> dict:
    """Run eval_gate.py inside the node container via docker exec. Returns
    a parsed dict with the verdict + key metrics. Raises on hard errors.

    The eval_gate CLI prints a single-line verdict followed by a JSON
    trailer we parse here.

    `with_baseline=True` (Phase 7 default) doubles the eval cost but produces
    a trailer that carries `pass_rate_before/after`, `latency_before_s/after_s`
    and `per_command_delta` — the fields the mobile proposal preview reads.
    """
    cmd = [
        "docker", "exec", node_container,
        "python", eval_script_path,
        "--adapter-hash", adapter_hash,
        "--output", output_path,
        "--tolerance", str(tolerance),
    ]
    if with_baseline:
        cmd.append("--with-baseline")
    if threshold is not None:
        cmd += ["--threshold", str(threshold)]

    logger.info("orchestrator: invoking eval via docker exec: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)
    stdout, stderr = proc.stdout, proc.stderr

    # eval_gate.py emits a final-line JSON trailer — parse it back
    import json as _json
    trailer: dict | None = None
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                trailer = _json.loads(line)
                break
            except _json.JSONDecodeError:
                continue

    return {
        "exit_code": proc.returncode,
        "trailer": trailer,
        "stdout_tail": "\n".join(stdout.splitlines()[-10:]),
        "stderr_tail": "\n".join(stderr.splitlines()[-10:]),
    }


# --------------------------------------------------------------------------
# The orchestrator itself
# --------------------------------------------------------------------------


OrganicExtractorFn = Callable[[], list[ExtractedRow]]
SyntheticExpanderFn = Callable[[], list[ExtractedRow]]
FormatterFn = Callable[[ExtractedRow], dict]


def orchestrate_training(
    *,
    organic_source: OrganicExtractorFn | None,
    synthetic_source: SyntheticExpanderFn | None,
    formatter: FormatterFn,
    llm_proxy_url: str,
    base_model_id: str,
    node_id: str,
    app_id: str,
    app_key: str,
    training_params: dict,
    # Eval (optional — caller can skip by passing None)
    eval_runner: Callable[[str], dict] | None = None,
    enqueue_timeout_seconds: float = 7200.0,
) -> OrchestratorResult:
    """Run the full extract → train → eval loop.

    Pluggable: organic/synthetic/formatter/eval are all callables. The CLI
    wires up defaults; tests can inject fakes.
    """
    timings: dict[str, float] = {}

    # 1. Gather rows
    t0 = time.time()
    organic_rows: list[ExtractedRow] = organic_source() if organic_source else []
    timings["extract"] = time.time() - t0

    t0 = time.time()
    synth_rows: list[ExtractedRow] = synthetic_source() if synthetic_source else []
    timings["expand"] = time.time() - t0

    all_rows = [*organic_rows, *synth_rows]
    by_source: dict[str, int] = {}
    for r in all_rows:
        by_source[r.source] = by_source.get(r.source, 0) + 1

    if not all_rows:
        return OrchestratorResult(
            status="enqueue_failed",
            adapter_hash=None, job_id=None, training_seconds=None,
            dataset_rows=0, by_source={},
            error="no rows from either source", timings=timings,
        )

    logger.info("orchestrator: %d rows total (by source: %s)", len(all_rows), by_source)

    # 2. Format + shape into dataset_ref
    formatted = [formatter(r) for r in all_rows]
    dataset_ref = build_dataset_ref(formatted)

    # 3. Enqueue
    try:
        t0 = time.time()
        job_id = enqueue_training(
            llm_proxy_url=llm_proxy_url,
            dataset_ref=dataset_ref,
            base_model_id=base_model_id,
            node_id=node_id,
            params=training_params,
            app_id=app_id, app_key=app_key,
        )
        timings["enqueue"] = time.time() - t0
    except Exception as e:
        return OrchestratorResult(
            status="enqueue_failed",
            adapter_hash=None, job_id=None, training_seconds=None,
            dataset_rows=len(all_rows), by_source=by_source,
            error=f"enqueue error: {e}", timings=timings,
        )

    # 4. Poll
    try:
        t0 = time.time()
        status = poll_status(
            llm_proxy_url=llm_proxy_url,
            job_id=job_id,
            app_id=app_id, app_key=app_key,
            timeout_seconds=enqueue_timeout_seconds,
        )
        timings["train_poll"] = time.time() - t0
    except TimeoutError as e:
        return OrchestratorResult(
            status="training_failed",
            adapter_hash=None, job_id=job_id, training_seconds=None,
            dataset_rows=len(all_rows), by_source=by_source,
            error=str(e), timings=timings,
        )

    if status.get("status") != "COMPLETE":
        return OrchestratorResult(
            status="training_failed",
            adapter_hash=None, job_id=job_id, training_seconds=None,
            dataset_rows=len(all_rows), by_source=by_source,
            error=f"training {status.get('status')}: {status.get('error_message')}",
            timings=timings,
        )

    adapter_hash = status.get("dataset_hash")
    started = status.get("started_at")
    completed = status.get("completed_at")
    train_seconds: float | None = None
    if started and completed:
        try:
            ts0 = datetime.fromisoformat(started.replace("Z", "+00:00"))
            ts1 = datetime.fromisoformat(completed.replace("Z", "+00:00"))
            train_seconds = (ts1 - ts0).total_seconds()
        except (ValueError, TypeError):
            pass

    logger.info(
        "orchestrator: training COMPLETE  adapter_hash=%s  train_seconds=%s",
        adapter_hash, train_seconds,
    )

    # 5. Eval (optional)
    eval_result: dict | None = None
    if eval_runner and adapter_hash:
        try:
            t0 = time.time()
            eval_result = eval_runner(adapter_hash)
            timings["eval"] = time.time() - t0
        except Exception as e:
            return OrchestratorResult(
                status="eval_failed",
                adapter_hash=adapter_hash, job_id=job_id,
                training_seconds=train_seconds,
                dataset_rows=len(all_rows), by_source=by_source,
                eval=None, error=f"eval error: {e}", timings=timings,
            )

    return OrchestratorResult(
        status="success",
        adapter_hash=adapter_hash,
        job_id=job_id,
        training_seconds=train_seconds,
        dataset_rows=len(all_rows),
        by_source=by_source,
        eval=eval_result,
        timings=timings,
    )
