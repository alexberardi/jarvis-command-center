"""Adapter scheduler — Phase 5.

Decides whether to fire a new training run for a household, invokes the Phase 4
orchestrator when the signal is strong enough, and deploys the result on eval
PASS. Everything network- or DB-adjacent is passed in as a callable so the
scheduler can be unit-tested with zero I/O.

The periodic tick (in main.py) calls `run_tick(...)` every N seconds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Optional

from app.services.adapter_registry import (
    TRIGGER_SCHEDULER,
    AdapterDeployment,
    AdapterRegistry,
)
from app.services.training_orchestrator import OrchestratorResult

logger = logging.getLogger("uvicorn")


# ---------- typed callables wired by main.py, faked by tests ----------

CountNewExamplesFn = Callable[[str, Optional[datetime]], int]
"""(household_id, since) → number of usable positive examples since `since`."""

ListActiveHouseholdsFn = Callable[[], list[str]]
"""Return household IDs with training_enabled status."""

OrchestratorFn = Callable[[str], OrchestratorResult]
"""(household_id) → OrchestratorResult (extract → train → eval for one household)."""

ProposeFn = Callable[[str, OrchestratorResult], bool]
"""(household_id, orchestrator_result) → True on successful proposal create.

Called on eval PASS when auto_approve is False (Phase 7 default). The
scheduler delegates the write; the production implementation creates a row
in `adapter_proposals` and posts an inbox item + push notification.
"""


# ---------- result shapes ----------


@dataclass(frozen=True)
class TickResult:
    """One scheduler tick's decisions, per household."""

    considered: list[str]
    skipped: dict[str, str]          # household_id → reason
    acquired: list[str]              # locked + started
    deployed: list[str]              # auto_approve=true  → immediate deploy on PASS
    proposed: list[str]              # auto_approve=false → PASS → proposal written
    eval_failed: list[str]           # ran but eval regressed → no action
    errored: dict[str, str]          # household_id → error


# ---------- the scheduler ----------


class AdapterScheduler:
    """Decides *whether* to train and *whether* to deploy.

    Policy knobs:
      - min_new_examples: don't retrain until we have at least this many new
        clean positive examples since last cutoff.
      - min_hours_between_runs: rate limiter so a flurry of new traces
        doesn't produce back-to-back training jobs.
    """

    def __init__(
        self,
        *,
        registry: AdapterRegistry,
        count_new_examples: CountNewExamplesFn,
        list_active_households: ListActiveHouseholdsFn,
        orchestrate_for_household: OrchestratorFn,
        min_new_examples: int = 50,
        min_hours_between_runs: int = 6,
        now_fn: Callable[[], datetime] = datetime.utcnow,
        auto_approve: bool = False,
        propose_for_household: Optional[ProposeFn] = None,
    ):
        self.registry = registry
        self._count_new_examples = count_new_examples
        self._list_active_households = list_active_households
        self._orchestrate = orchestrate_for_household
        self.min_new_examples = max(1, int(min_new_examples))
        self.min_hours_between_runs = max(0, int(min_hours_between_runs))
        self._now = now_fn
        self.auto_approve = bool(auto_approve)
        self._propose = propose_for_household

    # --- main entry point ---

    def run_tick(self) -> TickResult:
        considered: list[str] = []
        skipped: dict[str, str] = {}
        acquired: list[str] = []
        deployed: list[str] = []
        proposed: list[str] = []
        eval_failed: list[str] = []
        errored: dict[str, str] = {}

        households = list(self._list_active_households())
        considered.extend(households)

        for hid in households:
            reason = self._reason_to_skip(hid)
            if reason is not None:
                skipped[hid] = reason
                continue

            if not self.registry.try_acquire_training_lock(hid):
                skipped[hid] = "already_training"
                continue

            acquired.append(hid)
            try:
                result = self._orchestrate(hid)
            except Exception as e:
                logger.exception("scheduler: orchestrator raised for %s", hid)
                errored[hid] = f"orchestrator raised: {e}"
                self.registry.release_training_lock(hid)
                continue

            outcome = self._handle_outcome(hid, result)
            if outcome == "deployed":
                deployed.append(hid)
            elif outcome == "proposed":
                proposed.append(hid)
            elif outcome == "eval_failed":
                eval_failed.append(hid)
            elif outcome == "errored":
                errored[hid] = result.error or result.status

        return TickResult(
            considered=considered,
            skipped=skipped,
            acquired=acquired,
            deployed=deployed,
            proposed=proposed,
            eval_failed=eval_failed,
            errored=errored,
        )

    # --- policy ---

    def _reason_to_skip(self, household_id: str) -> Optional[str]:
        state = self.registry.ensure_state(household_id)
        if state.is_training:
            return "already_training"

        now = self._now()
        if state.last_trained_at and self.min_hours_between_runs:
            elapsed = now - state.last_trained_at
            if elapsed < timedelta(hours=self.min_hours_between_runs):
                return (
                    f"cooldown (trained {int(elapsed.total_seconds() // 60)} min ago, "
                    f"need {self.min_hours_between_runs}h)"
                )

        new_examples = self._count_new_examples(household_id, state.last_cutoff_at)
        if new_examples < self.min_new_examples:
            return f"not_enough_examples ({new_examples} < {self.min_new_examples})"
        return None

    # --- post-training handling ---

    def _handle_outcome(
        self, household_id: str, result: OrchestratorResult
    ) -> str:
        """Return one of: deployed, proposed, eval_failed, errored. Always
        releases the lock.

        PASS criteria:
          - result.status == "success"
          - eval trailer exists AND its verdict is PASS

        On PASS:
          - auto_approve=True  → deploy directly (Phase 5 behavior, now opt-in)
          - auto_approve=False → write an adapter_proposal and post to inbox
                                 (Phase 7 default)
        """
        now = self._now()

        if result.status != "success" or not result.adapter_hash:
            logger.warning(
                "scheduler: orchestrator status=%s for %s (error=%s)",
                result.status, household_id, result.error,
            )
            self.registry.release_training_lock(
                household_id,
                last_trained_at=now,
                last_cutoff_at=now,
                last_example_count=result.dataset_rows,
            )
            return "errored"

        eval_trailer = (result.eval or {}).get("trailer") or {}
        verdict = str(eval_trailer.get("verdict") or "").upper()
        pass_rate = eval_trailer.get("pass_rate")

        if verdict != "PASS":
            logger.warning(
                "scheduler: eval %s for %s — NOT acting on adapter %s",
                verdict or "MISSING", household_id, result.adapter_hash[:12],
            )
            self.registry.release_training_lock(
                household_id,
                last_trained_at=now,
                last_cutoff_at=now,
                last_example_count=result.dataset_rows,
            )
            return "eval_failed"

        try:
            pass_rate_f = float(pass_rate) if pass_rate is not None else None
        except (TypeError, ValueError):
            pass_rate_f = None

        if self.auto_approve:
            dep: AdapterDeployment = self.registry.deploy(
                household_id=household_id,
                adapter_hash=result.adapter_hash,
                pass_rate=pass_rate_f,
                trained_on_examples=result.dataset_rows,
                trigger=TRIGGER_SCHEDULER,
            )
            self.registry.release_training_lock(
                household_id,
                last_trained_at=now,
                last_cutoff_at=now,
                last_example_count=result.dataset_rows,
            )
            logger.info(
                "scheduler: deployed adapter %s for household %s (pass_rate=%s, auto_approve)",
                dep.adapter_hash[:12], household_id, pass_rate_f,
            )
            return "deployed"

        # Propose flow — Phase 7 default
        if self._propose is None:
            logger.error(
                "scheduler: auto_approve=False but no propose_for_household wired — "
                "eval PASS for %s ignored", household_id,
            )
            self.registry.release_training_lock(
                household_id,
                last_trained_at=now,
                last_cutoff_at=now,
                last_example_count=result.dataset_rows,
            )
            return "errored"

        try:
            proposed = bool(self._propose(household_id, result))
        except Exception:
            logger.exception(
                "scheduler: propose callback raised for %s", household_id,
            )
            self.registry.release_training_lock(
                household_id,
                last_trained_at=now,
                last_cutoff_at=now,
                last_example_count=result.dataset_rows,
            )
            return "errored"

        self.registry.release_training_lock(
            household_id,
            last_trained_at=now,
            last_cutoff_at=now,
            last_example_count=result.dataset_rows,
        )
        if proposed:
            logger.info(
                "scheduler: proposed adapter %s for household %s (pass_rate=%s)",
                result.adapter_hash[:12], household_id, pass_rate_f,
            )
            return "proposed"

        logger.warning(
            "scheduler: propose callback returned False for %s — treating as errored",
            household_id,
        )
        return "errored"


# --------------------------------------------------------------------------
# Production entry point — wires real dependencies from settings + DB.
# Called from the periodic task in app/main.py.
# --------------------------------------------------------------------------


def run_scheduled_training(settings_service) -> TickResult:
    """Build + run one scheduler tick using real services.

    Synchronous because the orchestrator uses blocking httpx + subprocess.
    The async periodic task in app/main.py calls this via asyncio.to_thread
    so the event loop stays free.

    Graceful degradation: if any of the required settings are missing (base
    model id, commands spec path, app-to-app auth key), logs a warning and
    returns an empty TickResult. This keeps the periodic task safe to run
    even before the operator has finished configuration.
    """
    import json
    import os
    import random
    from pathlib import Path

    from app.db import get_session_local
    from app.models import ConversationTranscript
    from app.services.training_data_extractor import (
        TrainingDataExtractor,
        to_dataset_ref_row,
    )
    from app.services.training_orchestrator import (
        invoke_eval_via_docker,
        orchestrate_training,
    )

    empty = TickResult(
        considered=[],
        skipped={},
        acquired=[],
        deployed=[],
        proposed=[],
        eval_failed=[],
        errored={},
    )

    base_model_id = (settings_service.get("adapter.base_model_id") or "").strip()
    spec_path = (settings_service.get("adapter.commands_spec_path") or "").strip()
    llm_proxy_url = (
        (settings_service.get("adapter.llm_proxy_url") or "").strip()
        or os.getenv("JARVIS_LLM_PROXY_API_URL", "http://localhost:7704")
    )
    eval_container = (settings_service.get("adapter.eval_container") or "jarvis-node").strip()
    app_id = os.getenv("JARVIS_APP_ID") or "jarvis-command-center"
    app_key = os.getenv("JARVIS_APP_KEY") or ""

    if not base_model_id:
        logger.warning("adapter scheduler: adapter.base_model_id not set — skipping")
        return empty
    if not spec_path or not Path(spec_path).is_file():
        logger.warning("adapter scheduler: adapter.commands_spec_path missing or not a file — skipping")
        return empty
    if not app_key:
        logger.warning("adapter scheduler: JARVIS_APP_KEY env var required for llm-proxy enqueue — skipping")
        return empty

    try:
        spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("adapter scheduler: commands spec unreadable (%s) — skipping", e)
        return empty

    from app.services.prompt_variant_builder import build_prompt_variants
    variants = build_prompt_variants(spec)
    variant_names = list(variants.keys())

    min_examples = int(settings_service.get("adapter.auto_train_min_examples") or 50)
    min_hours = int(settings_service.get("adapter.auto_train_min_interval_hours") or 6)

    SessionLocal = get_session_local()
    db = SessionLocal()
    try:
        from app.services.adapter_registry import AdapterRegistry
        registry = AdapterRegistry(db)

        def list_active_households() -> list[str]:
            """Households with at least one transcript — i.e., where training *could* help."""
            rows = (
                db.query(ConversationTranscript.household_id)
                .filter(ConversationTranscript.household_id.isnot(None))
                .distinct()
                .all()
            )
            return [r[0] for r in rows if r[0]]

        def count_new_examples(household_id: str, since: datetime | None) -> int:
            extractor = TrainingDataExtractor(db)
            rows = extractor.extract(since=since, household_id=household_id)
            return len(rows)

        rng = random.Random(42)

        def orchestrate_for(household_id: str):
            """Run extract → train → eval for one household."""
            def organic_source():
                state = registry.get_state(household_id)
                since = state.last_cutoff_at if state else None
                extractor = TrainingDataExtractor(db)
                return extractor.extract(since=since, household_id=household_id)

            def formatter(row):
                variant = rng.choice(variant_names)
                return to_dataset_ref_row(row, system_prompt=variants[variant])

            def eval_runner(adapter_hash: str):
                return invoke_eval_via_docker(
                    adapter_hash=adapter_hash,
                    node_container=eval_container,
                )

            return orchestrate_training(
                organic_source=organic_source,
                synthetic_source=None,
                formatter=formatter,
                llm_proxy_url=llm_proxy_url,
                base_model_id=base_model_id,
                node_id=f"scheduler-{household_id}",
                app_id=app_id,
                app_key=app_key,
                training_params={
                    "epochs": 1,
                    "batch_size": 1,
                    "lora_r": 16,
                    "lora_alpha": 32,
                    "max_seq_len": 768,
                    "learning_rate": 2e-4,
                },
                eval_runner=eval_runner,
            )

        # Propose vs auto-deploy (Phase 7 default is propose).
        auto_approve = bool(settings_service.get("adapter.auto_approve_deployments") or False)
        expiry_days = int(settings_service.get("adapter.proposal_expiry_days") or 7)

        from app.services.adapter_proposal_service import AdapterProposalService
        from app.services.inbox_notification_service import (
            push_adapter_proposal_to_inbox,
        )

        def propose_for(household_id: str, result) -> bool:
            trailer = (result.eval or {}).get("trailer") or {}
            metrics = extract_proposal_metrics(trailer)
            current_interface = str(settings_service.get("llm.interface") or "") or None

            proposal_svc = AdapterProposalService(db, registry)
            view = proposal_svc.create(
                household_id=household_id,
                adapter_hash=result.adapter_hash,
                provider_name_before=current_interface,
                provider_name_after=current_interface,
                pass_rate_before=metrics.pass_rate_before,
                pass_rate_after=metrics.pass_rate_after,
                latency_before_s=metrics.latency_before_s,
                latency_after_s=metrics.latency_after_s,
                per_command_delta=metrics.per_command_delta,
                trained_on_examples=result.dataset_rows,
                expiry_days=expiry_days,
            )
            db.flush()

            inbox_id = push_adapter_proposal_to_inbox(
                household_id=household_id,
                proposal_id=view.id,
                adapter_hash=view.adapter_hash,
                pass_rate_before=metrics.pass_rate_before,
                pass_rate_after=metrics.pass_rate_after,
                latency_before_s=metrics.latency_before_s,
                latency_after_s=metrics.latency_after_s,
                per_command_delta=metrics.per_command_delta,
                trained_on_examples=view.trained_on_examples,
                provider_name_before=current_interface,
                provider_name_after=current_interface,
                expires_at=view.expires_at.isoformat(),
            )
            if inbox_id:
                proposal_svc.attach_inbox_item(view.id, inbox_id)
            return True

        scheduler = AdapterScheduler(
            registry=registry,
            count_new_examples=count_new_examples,
            list_active_households=list_active_households,
            orchestrate_for_household=orchestrate_for,
            min_new_examples=min_examples,
            min_hours_between_runs=min_hours,
            auto_approve=auto_approve,
            propose_for_household=propose_for,
        )
        tick = scheduler.run_tick()
        db.commit()
        return tick
    finally:
        db.close()


def _to_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class ProposalMetrics:
    """Metrics ready to write to an AdapterProposal row.

    Extracted from the eval trailer in one place so the scheduler wiring and
    tests share the same decoding logic. When eval_gate was invoked with
    --with-baseline the trailer carries the full before/after breakdown;
    otherwise we fall back to the stored threshold for the baseline rate
    (and leave latency + per-command fields None).
    """

    pass_rate_before: float | None
    pass_rate_after: float | None
    latency_before_s: float | None
    latency_after_s: float | None
    per_command_delta: dict | None


def extract_proposal_metrics(trailer: dict) -> ProposalMetrics:
    pass_rate_before = _to_float(
        trailer.get("pass_rate_before")
        if "pass_rate_before" in trailer
        else trailer.get("threshold")
    )
    pass_rate_after = _to_float(
        trailer.get("pass_rate_after")
        if "pass_rate_after" in trailer
        else trailer.get("pass_rate")
    )
    return ProposalMetrics(
        pass_rate_before=pass_rate_before,
        pass_rate_after=pass_rate_after,
        latency_before_s=_to_float(trailer.get("latency_before_s")),
        latency_after_s=_to_float(trailer.get("latency_after_s")),
        per_command_delta=trailer.get("per_command_delta") or None,
    )
