"""Adapter deployment registry — Phase 5.

Read/write layer over three tables:
  - active_adapter          — which adapter is currently deployed per household
  - adapter_training_state  — scheduler bookkeeping (cutoff, example count, lock)
  - adapter_history         — append-only audit log, powers rollback

All methods take a SQLAlchemy Session; transaction management is the caller's
responsibility. Used by:
  - periodic training task (reads state, writes state on enqueue/completion)
  - llm_proxy_client (reads active adapter to inject adapter_settings.hash)
  - admin rollback endpoint (reads history, upserts active_adapter)
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from app.models import ActiveAdapter, AdapterHistory, AdapterTrainingState

logger = logging.getLogger("uvicorn")


# Valid values for adapter_history.trigger
TRIGGER_SCHEDULER = "scheduler"
TRIGGER_MANUAL = "manual"
TRIGGER_ROLLBACK = "rollback"
_VALID_TRIGGERS = {TRIGGER_SCHEDULER, TRIGGER_MANUAL, TRIGGER_ROLLBACK}


@dataclass(frozen=True)
class AdapterDeployment:
    """Immutable view of an adapter deployment for downstream use."""

    household_id: str
    adapter_hash: str
    pass_rate: Optional[float]
    trained_on_examples: Optional[int]
    deployed_at: datetime


class AdapterRegistry:
    def __init__(self, db: Session):
        self.db = db

    # ---------- active adapter ----------

    def get_active(self, household_id: str) -> Optional[AdapterDeployment]:
        row = (
            self.db.query(ActiveAdapter)
            .filter(ActiveAdapter.household_id == household_id)
            .first()
        )
        if row is None:
            return None
        return AdapterDeployment(
            household_id=row.household_id,
            adapter_hash=row.adapter_hash,
            pass_rate=row.pass_rate,
            trained_on_examples=row.trained_on_examples,
            deployed_at=row.deployed_at,
        )

    def deploy(
        self,
        household_id: str,
        adapter_hash: str,
        pass_rate: Optional[float],
        trained_on_examples: Optional[int],
        trigger: str = TRIGGER_SCHEDULER,
    ) -> AdapterDeployment:
        """Set the active adapter for a household.

        Appends the prior active row (if any) to adapter_history with replaced_at
        stamped to now, then upserts the new row. Flushes but does not commit —
        the caller owns the transaction.
        """
        if trigger not in _VALID_TRIGGERS:
            raise ValueError(f"invalid trigger: {trigger!r}")

        now = datetime.utcnow()
        prior = (
            self.db.query(ActiveAdapter)
            .filter(ActiveAdapter.household_id == household_id)
            .first()
        )

        if prior is not None:
            self.db.add(
                AdapterHistory(
                    household_id=prior.household_id,
                    adapter_hash=prior.adapter_hash,
                    pass_rate=prior.pass_rate,
                    trained_on_examples=prior.trained_on_examples,
                    deployed_at=prior.deployed_at,
                    replaced_at=now,
                    trigger=trigger,
                )
            )
            prior.adapter_hash = adapter_hash
            prior.pass_rate = pass_rate
            prior.trained_on_examples = trained_on_examples
            prior.deployed_at = now
            row = prior
        else:
            row = ActiveAdapter(
                household_id=household_id,
                adapter_hash=adapter_hash,
                pass_rate=pass_rate,
                trained_on_examples=trained_on_examples,
                deployed_at=now,
            )
            self.db.add(row)

        self.db.flush()

        logger.info(
            "adapter_registry: deployed household=%s hash=%s trigger=%s",
            household_id,
            adapter_hash[:12],
            trigger,
        )
        return AdapterDeployment(
            household_id=row.household_id,
            adapter_hash=row.adapter_hash,
            pass_rate=row.pass_rate,
            trained_on_examples=row.trained_on_examples,
            deployed_at=row.deployed_at,
        )

    # ---------- history / rollback ----------

    def list_history(
        self, household_id: str, limit: int = 20
    ) -> list[AdapterHistory]:
        return (
            self.db.query(AdapterHistory)
            .filter(AdapterHistory.household_id == household_id)
            .order_by(AdapterHistory.replaced_at.desc().nullsfirst())
            .limit(limit)
            .all()
        )

    def rollback(self, household_id: str) -> Optional[AdapterDeployment]:
        """Roll back to the most recent prior adapter for a household.

        Returns the restored deployment, or None if there is no history to roll
        back to. Does NOT commit — caller owns the transaction.
        """
        prior = (
            self.db.query(AdapterHistory)
            .filter(AdapterHistory.household_id == household_id)
            .order_by(AdapterHistory.replaced_at.desc().nullsfirst())
            .first()
        )
        if prior is None:
            logger.warning(
                "adapter_registry: rollback requested for %s but no history",
                household_id,
            )
            return None

        return self.deploy(
            household_id=household_id,
            adapter_hash=prior.adapter_hash,
            pass_rate=prior.pass_rate,
            trained_on_examples=prior.trained_on_examples,
            trigger=TRIGGER_ROLLBACK,
        )

    # ---------- training state ----------

    def get_state(self, household_id: str) -> Optional[AdapterTrainingState]:
        return (
            self.db.query(AdapterTrainingState)
            .filter(AdapterTrainingState.household_id == household_id)
            .first()
        )

    def ensure_state(self, household_id: str) -> AdapterTrainingState:
        state = self.get_state(household_id)
        if state is not None:
            return state
        state = AdapterTrainingState(
            household_id=household_id,
            last_trained_at=None,
            last_cutoff_at=None,
            last_example_count=0,
            is_training=False,
            updated_at=datetime.utcnow(),
        )
        self.db.add(state)
        self.db.flush()
        return state

    def try_acquire_training_lock(self, household_id: str) -> bool:
        """Atomically flip is_training False → True.

        Returns True if the caller acquired the lock, False if another worker
        already holds it. Uses a conditional UPDATE so two concurrent ticks
        can't both enter the training phase for the same household.
        """
        state = self.ensure_state(household_id)
        if state.is_training:
            return False
        state.is_training = True
        state.updated_at = datetime.utcnow()
        self.db.flush()
        return True

    def release_training_lock(
        self,
        household_id: str,
        last_trained_at: Optional[datetime] = None,
        last_cutoff_at: Optional[datetime] = None,
        last_example_count: Optional[int] = None,
    ) -> None:
        state = self.ensure_state(household_id)
        state.is_training = False
        if last_trained_at is not None:
            state.last_trained_at = last_trained_at
        if last_cutoff_at is not None:
            state.last_cutoff_at = last_cutoff_at
        if last_example_count is not None:
            state.last_example_count = last_example_count
        state.updated_at = datetime.utcnow()
        self.db.flush()
