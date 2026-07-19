"""callback_jobs.node_id nullable — server-plane (node-less) callbacks

CC server tools (deep-research follow-ups, phone-call confirm/escalation
cards) produce interactive elements with no owning node. Their CallbackJob
rows carry node_id NULL and are executed in-process by CC via the
server-callback registry instead of over MQTT.

Revision ID: b2c3cbsrv001
Revises: a1b2attn0001
"""

import sqlalchemy as sa
from alembic import op

revision = "b2c3cbsrv001"
down_revision = "a1b2attn0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "callback_jobs",
        "node_id",
        existing_type=sa.String(),
        nullable=True,
    )


def downgrade() -> None:
    # Server-plane rows (node_id NULL) cannot survive a NOT NULL constraint —
    # remove them before restoring it.
    op.execute("DELETE FROM callback_jobs WHERE node_id IS NULL")
    op.alter_column(
        "callback_jobs",
        "node_id",
        existing_type=sa.String(),
        nullable=False,
    )
