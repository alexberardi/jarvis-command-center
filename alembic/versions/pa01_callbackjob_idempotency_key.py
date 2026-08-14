"""add callback_jobs.idempotency_key for proposable-action dedup

Revision ID: pa01idempotency
Revises: l4h5wfrev008
Create Date: 2026-08-08

Adds a nullable, indexed idempotency_key to callback_jobs. The proposable-action
dispatcher stamps a stable key (e.g. "appt:<message_id>") on the node-plane job
it creates, and short-circuits a second job for the same (household_id,
idempotency_key) that already completed — so a double-tap / re-post never runs
the target command's callback twice. NULL for all ordinary callback jobs.
"""
from alembic import op
import sqlalchemy as sa


revision = "pa01idempotency"
down_revision = "l4h5wfrev008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "callback_jobs",
        sa.Column("idempotency_key", sa.String(length=255), nullable=True),
    )
    op.create_index(
        "ix_callback_jobs_idempotency_key",
        "callback_jobs",
        ["idempotency_key"],
    )


def downgrade() -> None:
    op.drop_index("ix_callback_jobs_idempotency_key", table_name="callback_jobs")
    op.drop_column("callback_jobs", "idempotency_key")
