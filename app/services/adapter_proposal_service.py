"""Adapter proposal service — Phase 7.1.

Read/write layer over the `adapter_proposals` table, plus the
deploy/dismiss/revert transitions the mobile app drives.

Transaction model mirrors `adapter_registry.py`: methods flush but do not
commit; the caller owns the transaction. This lets one API handler run
registry.deploy() + proposal.apply() + settings.set("llm.interface", ...)
atomically.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.models import AdapterProposal
from app.services.adapter_registry import (
    TRIGGER_MANUAL,
    TRIGGER_ROLLBACK,
    AdapterDeployment,
    AdapterRegistry,
)

logger = logging.getLogger("uvicorn")

# Structured event emitter for Phase 7.4 dashboards. Kept as a module
# singleton so importing this file doesn't blow up in environments where
# jarvis-log-client is unavailable (tests, tools, etc.) — we lazily create
# on first use and fall back to a no-op.
_event_logger = None
_event_logger_init_failed = False


def _get_event_logger():
    """Return a JarvisLogger, or None when remote logging isn't configured.

    Skips init in environments without JARVIS_APP_KEY (tests, CLI tools) so
    the background flush thread doesn't spin up and spam console errors.
    """
    global _event_logger, _event_logger_init_failed
    if _event_logger is not None or _event_logger_init_failed:
        return _event_logger
    import os
    if not os.getenv("JARVIS_APP_KEY"):
        _event_logger_init_failed = True
        return None
    try:
        from jarvis_log_client import JarvisLogger  # type: ignore
        _event_logger = JarvisLogger(service="jarvis-command-center")
    except Exception as e:
        _event_logger_init_failed = True
        logger.debug("adapter_event: JarvisLogger init failed: %s", e)
    return _event_logger


def _emit_event(event_type: str, **context) -> None:
    """Fire a structured event that Grafana Loki dashboards can aggregate.

    Uses `event_type` as a first-class context field so LogQL selectors read
    `{service="jarvis-command-center"} | json | event_type="adapter_..."`.
    """
    emitter = _get_event_logger()
    if emitter is None:
        return
    try:
        emitter.info(event_type, event_type=event_type, **context)
    except Exception as e:
        # Never let observability errors leak into the user-facing flow.
        logger.debug("adapter_event emit failed (%s): %s", event_type, e)


# Valid values for adapter_proposals.status
STATUS_PENDING = "pending"
STATUS_APPLIED = "applied"
STATUS_DISMISSED = "dismissed"
STATUS_EXPIRED = "expired"
STATUS_SUPERSEDED = "superseded"

_DEFAULT_EXPIRY_DAYS = 7


# --------------------------------------------------------------------------
# Exceptions — raised to API handlers as specific HTTP error mappings
# --------------------------------------------------------------------------


class ProposalNotFoundError(LookupError):
    """Proposal id does not exist or does not belong to the household."""


class ProposalNotPendingError(RuntimeError):
    """Proposal was already applied, dismissed, expired, or superseded."""


class ProposalExpiredError(RuntimeError):
    """Proposal passed its expires_at cutoff."""


class NoActiveAdapterError(RuntimeError):
    """Revert requested but no active adapter for the household."""


class HashMismatchError(RuntimeError):
    """Revert requested for an adapter_hash that isn't currently active."""


# --------------------------------------------------------------------------
# DTO — immutable view returned to callers
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ProposalView:
    id: str
    household_id: str
    adapter_hash: str
    provider_name_before: Optional[str]
    provider_name_after: Optional[str]
    pass_rate_before: Optional[float]
    pass_rate_after: Optional[float]
    latency_before_s: Optional[float]
    latency_after_s: Optional[float]
    per_command_delta: Optional[dict[str, Any]]
    trained_on_examples: Optional[int]
    status: str
    inbox_item_id: Optional[str]
    created_at: datetime
    expires_at: datetime
    decided_at: Optional[datetime]


def _to_view(row: AdapterProposal) -> ProposalView:
    per_cmd: Optional[dict[str, Any]] = None
    if row.per_command_delta_json:
        try:
            per_cmd = json.loads(row.per_command_delta_json)
        except (json.JSONDecodeError, TypeError, ValueError):
            per_cmd = None
    return ProposalView(
        id=row.id,
        household_id=row.household_id,
        adapter_hash=row.adapter_hash,
        provider_name_before=row.provider_name_before,
        provider_name_after=row.provider_name_after,
        pass_rate_before=row.pass_rate_before,
        pass_rate_after=row.pass_rate_after,
        latency_before_s=row.latency_before_s,
        latency_after_s=row.latency_after_s,
        per_command_delta=per_cmd,
        trained_on_examples=row.trained_on_examples,
        status=row.status,
        inbox_item_id=row.inbox_item_id,
        created_at=row.created_at,
        expires_at=row.expires_at,
        decided_at=row.decided_at,
    )


# --------------------------------------------------------------------------
# Service
# --------------------------------------------------------------------------


class AdapterProposalService:
    def __init__(self, db: Session, registry: Optional[AdapterRegistry] = None):
        self.db = db
        self.registry = registry or AdapterRegistry(db)

    # ---------- create / read ----------

    def create(
        self,
        *,
        household_id: str,
        adapter_hash: str,
        provider_name_before: Optional[str] = None,
        provider_name_after: Optional[str] = None,
        pass_rate_before: Optional[float] = None,
        pass_rate_after: Optional[float] = None,
        latency_before_s: Optional[float] = None,
        latency_after_s: Optional[float] = None,
        per_command_delta: Optional[dict[str, Any]] = None,
        trained_on_examples: Optional[int] = None,
        expiry_days: int = _DEFAULT_EXPIRY_DAYS,
        now: Optional[datetime] = None,
    ) -> ProposalView:
        """Insert a pending proposal. Supersedes any prior pending proposal
        for the same household (only one outstanding proposal at a time —
        the newest is always the one the user sees).
        """
        now = now or datetime.utcnow()
        self._supersede_prior_pending(household_id, now=now)

        row = AdapterProposal(
            id=str(uuid.uuid4()),
            household_id=household_id,
            adapter_hash=adapter_hash,
            provider_name_before=provider_name_before,
            provider_name_after=provider_name_after,
            pass_rate_before=pass_rate_before,
            pass_rate_after=pass_rate_after,
            latency_before_s=latency_before_s,
            latency_after_s=latency_after_s,
            per_command_delta_json=(
                json.dumps(per_command_delta, separators=(",", ":"))
                if per_command_delta is not None
                else None
            ),
            trained_on_examples=trained_on_examples,
            status=STATUS_PENDING,
            inbox_item_id=None,
            created_at=now,
            expires_at=now + timedelta(days=max(1, int(expiry_days))),
            decided_at=None,
        )
        self.db.add(row)
        self.db.flush()

        logger.info(
            "adapter_proposal: created id=%s household=%s hash=%s",
            row.id, household_id, adapter_hash[:12],
        )
        _emit_event(
            "adapter_proposal_created",
            proposal_id=row.id,
            household_id=household_id,
            adapter_hash=adapter_hash,
            provider_name_before=provider_name_before,
            provider_name_after=provider_name_after,
            pass_rate_before=pass_rate_before,
            pass_rate_after=pass_rate_after,
            latency_before_s=latency_before_s,
            latency_after_s=latency_after_s,
            trained_on_examples=trained_on_examples,
            expires_at=row.expires_at.isoformat(),
        )
        return _to_view(row)

    def get(self, proposal_id: str) -> Optional[ProposalView]:
        row = (
            self.db.query(AdapterProposal)
            .filter(AdapterProposal.id == proposal_id)
            .first()
        )
        return _to_view(row) if row else None

    def list_for_household(
        self,
        household_id: str,
        *,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> list[ProposalView]:
        q = (
            self.db.query(AdapterProposal)
            .filter(AdapterProposal.household_id == household_id)
        )
        if status is not None:
            q = q.filter(AdapterProposal.status == status)
        rows = q.order_by(AdapterProposal.created_at.desc()).limit(limit).all()
        return [_to_view(r) for r in rows]

    def attach_inbox_item(self, proposal_id: str, inbox_item_id: str) -> None:
        """Back-fill the inbox_item_id once the item has been posted."""
        row = (
            self.db.query(AdapterProposal)
            .filter(AdapterProposal.id == proposal_id)
            .first()
        )
        if row is None:
            return
        row.inbox_item_id = inbox_item_id
        self.db.flush()

    # ---------- transitions ----------

    def apply(
        self,
        proposal_id: str,
        household_id: str,
        *,
        now: Optional[datetime] = None,
    ) -> tuple[ProposalView, AdapterDeployment]:
        """Deploy the proposed adapter and mark the proposal applied.

        Caller is responsible for flipping llm.interface to the paired
        provider_name_after *within the same DB transaction* so that a
        commit-or-rollback keeps the two in sync.
        """
        now = now or datetime.utcnow()
        row = self._load_for_household(proposal_id, household_id)

        if row.status != STATUS_PENDING:
            raise ProposalNotPendingError(
                f"proposal {proposal_id} is {row.status}, not pending"
            )
        if row.expires_at <= now:
            row.status = STATUS_EXPIRED
            row.decided_at = now
            self.db.flush()
            raise ProposalExpiredError(
                f"proposal {proposal_id} expired at {row.expires_at.isoformat()}"
            )

        dep = self.registry.deploy(
            household_id=household_id,
            adapter_hash=row.adapter_hash,
            pass_rate=row.pass_rate_after,
            trained_on_examples=row.trained_on_examples,
            trigger=TRIGGER_MANUAL,
        )
        row.status = STATUS_APPLIED
        row.decided_at = now
        self.db.flush()

        time_to_decision_s = (now - row.created_at).total_seconds()
        logger.info(
            "adapter_proposal: applied id=%s household=%s hash=%s",
            row.id, household_id, row.adapter_hash[:12],
        )
        _emit_event(
            "adapter_proposal_applied",
            proposal_id=row.id,
            household_id=household_id,
            adapter_hash=row.adapter_hash,
            pass_rate_before=row.pass_rate_before,
            pass_rate_after=row.pass_rate_after,
            latency_before_s=row.latency_before_s,
            latency_after_s=row.latency_after_s,
            time_to_decision_s=time_to_decision_s,
            trained_on_examples=row.trained_on_examples,
        )
        return _to_view(row), dep

    def dismiss(
        self,
        proposal_id: str,
        household_id: str,
        *,
        now: Optional[datetime] = None,
    ) -> ProposalView:
        """Mark the proposal dismissed. No adapter state changes."""
        now = now or datetime.utcnow()
        row = self._load_for_household(proposal_id, household_id)

        if row.status != STATUS_PENDING:
            raise ProposalNotPendingError(
                f"proposal {proposal_id} is {row.status}, not pending"
            )

        row.status = STATUS_DISMISSED
        row.decided_at = now
        self.db.flush()

        time_to_decision_s = (now - row.created_at).total_seconds()
        logger.info(
            "adapter_proposal: dismissed id=%s household=%s",
            row.id, household_id,
        )
        _emit_event(
            "adapter_proposal_dismissed",
            proposal_id=row.id,
            household_id=household_id,
            adapter_hash=row.adapter_hash,
            time_to_decision_s=time_to_decision_s,
        )
        return _to_view(row)

    def revert(
        self,
        adapter_hash: str,
        household_id: str,
        *,
        now: Optional[datetime] = None,
    ) -> tuple[Optional[AdapterDeployment], Optional[str]]:
        """Roll back from the currently-active adapter.

        The adapter_hash in the URL must match the currently-active one — the
        user is reverting *this specific deployment*, not an arbitrary history
        entry. Returns (restored_deployment, provider_name_to_restore).
        The caller flips llm.interface to the returned provider name.

        Raises NoActiveAdapterError / HashMismatchError when the request
        doesn't match DB state (so the handler can return a useful 4xx).
        """
        now = now or datetime.utcnow()
        active = self.registry.get_active(household_id)
        if active is None:
            raise NoActiveAdapterError(
                f"no active adapter for household {household_id}"
            )
        if active.adapter_hash != adapter_hash:
            raise HashMismatchError(
                f"active adapter is {active.adapter_hash[:12]}, not {adapter_hash[:12]}"
            )

        # Find the proposal that deployed this adapter to recover the paired
        # provider_name_before. If the adapter pre-dates Phase 7 (no
        # proposal), provider_name_to_restore comes back None and the
        # caller leaves llm.interface untouched.
        proposal_row = (
            self.db.query(AdapterProposal)
            .filter(AdapterProposal.household_id == household_id)
            .filter(AdapterProposal.adapter_hash == adapter_hash)
            .filter(AdapterProposal.status == STATUS_APPLIED)
            .order_by(AdapterProposal.decided_at.desc().nullslast())
            .first()
        )
        provider_name_to_restore = (
            proposal_row.provider_name_before if proposal_row else None
        )

        restored = self.registry.rollback(household_id)

        # Measure how long the adapter was live before the user pulled it.
        lifetime_s: float | None = None
        if proposal_row and proposal_row.decided_at:
            lifetime_s = (now - proposal_row.decided_at).total_seconds()

        logger.info(
            "adapter_proposal: reverted household=%s from_hash=%s to_hash=%s "
            "to_provider=%s",
            household_id,
            adapter_hash[:12],
            restored.adapter_hash[:12] if restored else "<none>",
            provider_name_to_restore,
        )
        _emit_event(
            "adapter_deployment_reverted",
            household_id=household_id,
            from_adapter_hash=adapter_hash,
            to_adapter_hash=restored.adapter_hash if restored else None,
            to_provider_name=provider_name_to_restore,
            lifetime_s=lifetime_s,
        )
        _ = TRIGGER_ROLLBACK  # silence unused import; registry.rollback tags it
        return restored, provider_name_to_restore

    def expire_stale(self, *, now: Optional[datetime] = None) -> int:
        """Janitor — flip pending → expired for anything past its cutoff.

        Returns the number of rows touched. Safe to call repeatedly; a
        background task in main.py runs it periodically.
        """
        now = now or datetime.utcnow()
        stale = (
            self.db.query(AdapterProposal)
            .filter(AdapterProposal.status == STATUS_PENDING)
            .filter(AdapterProposal.expires_at <= now)
            .all()
        )
        for row in stale:
            row.status = STATUS_EXPIRED
            row.decided_at = now
            _emit_event(
                "adapter_proposal_expired",
                proposal_id=row.id,
                household_id=row.household_id,
                adapter_hash=row.adapter_hash,
                age_s=(now - row.created_at).total_seconds(),
            )
        self.db.flush()
        if stale:
            logger.info("adapter_proposal: expired %d stale proposals", len(stale))
        return len(stale)

    # ---------- internals ----------

    def _load_for_household(
        self, proposal_id: str, household_id: str
    ) -> AdapterProposal:
        row = (
            self.db.query(AdapterProposal)
            .filter(AdapterProposal.id == proposal_id)
            .first()
        )
        if row is None or row.household_id != household_id:
            raise ProposalNotFoundError(
                f"proposal {proposal_id} not found for household {household_id}"
            )
        return row

    def _supersede_prior_pending(
        self, household_id: str, *, now: datetime
    ) -> None:
        prior = (
            self.db.query(AdapterProposal)
            .filter(AdapterProposal.household_id == household_id)
            .filter(AdapterProposal.status == STATUS_PENDING)
            .all()
        )
        for row in prior:
            row.status = STATUS_SUPERSEDED
            row.decided_at = now
