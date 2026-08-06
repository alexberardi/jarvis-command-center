"""Workflow.revision + inbox_item_id — mid-run replan card support

A running errand can pause on a ``request_replan`` step (the new ``approval``
wait source), post a delta card, and resume over an amended step list. Two
columns support that on the RUN row (the draft ``ErrandPlan`` already carries
its own revision/inbox_item_id; the run needs its own once launched):
- ``revision``: bumps each time a replan splices new steps in; the delta card
  carries the revision it was posted for, so a tap on a stale card is rejected.
- ``inbox_item_id``: the run's live card, so an update posts IN PLACE.

Revision ID: l4h5wfrev008
Revises: k3g4ctxthink007
Create Date: 2026-08-04 00:00:00.000000
"""

import sqlalchemy as sa
from alembic import op

revision = "l4h5wfrev008"
down_revision = "k3g4ctxthink007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # server_default backfills existing rows to revision 1; the ORM default
    # (1) governs new rows.
    op.add_column(
        "workflows",
        sa.Column("revision", sa.Integer(), nullable=False, server_default="1"),
    )
    op.add_column(
        "workflows",
        sa.Column("inbox_item_id", sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("workflows", "inbox_item_id")
    op.drop_column("workflows", "revision")
