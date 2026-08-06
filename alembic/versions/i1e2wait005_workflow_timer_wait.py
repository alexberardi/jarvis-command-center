"""Workflow timer wakeup: waiting_on + wake_at for the wait_for step

The workflow engine's SECOND wakeup source (errand-runner PRD §2.5, the timer). A
``wait_for`` control-flow step SUSPENDS the run on a TIMER instead of a phone call,
so the run row must record what it's waiting on and when to wake:

- ``waiting_on`` — the signal kind the run is suspended on (``phone_call`` | ``timer``),
  so the resume sweep knows which wakeup source to check (a phone session vs the clock).
- ``wake_at`` — for a timer wait, the UTC instant to resume; persisted so the wait
  survives a restart (the whole point of a durable engine). Indexed for the due-timer
  sweep query.

Revision ID: i1e2wait005
Revises: i0d1wkfl004
Create Date: 2026-07-30 00:00:00.000000
"""

import sqlalchemy as sa
from alembic import op

revision = "i1e2wait005"
down_revision = "i0d1wkfl004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("workflows", sa.Column("waiting_on", sa.String(20), nullable=True))
    op.add_column("workflows", sa.Column("wake_at", sa.DateTime(), nullable=True))
    op.create_index("ix_workflows_wake_at", "workflows", ["wake_at"])


def downgrade() -> None:
    op.drop_index("ix_workflows_wake_at", table_name="workflows")
    op.drop_column("workflows", "wake_at")
    op.drop_column("workflows", "waiting_on")
