"""add routines and routine_executions tables

Server-owned, per-household routines (the source of truth). Mobile does CRUD
here; nodes pull the household set (pull-on-nudge) and execute on-node.
routine_executions is an optional audit trail for run-now + scheduled runs.

Revision ID: d9c8b7a6e5f4
Revises: f2g3h4i5j6k7
Create Date: 2026-06-14 21:30:00.000000
"""

from alembic import op
import sqlalchemy as sa

revision = "d9c8b7a6e5f4"
down_revision = "f2g3h4i5j6k7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "routines",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("household_id", sa.String(255), nullable=False, index=True),
        sa.Column("slug", sa.String(255), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("trigger_phrases", sa.Text, nullable=False, server_default="[]"),
        sa.Column("steps", sa.Text, nullable=False, server_default="[]"),
        sa.Column("response_instruction", sa.Text, nullable=False, server_default=""),
        sa.Column("response_length", sa.String(16), nullable=False, server_default="short"),
        sa.Column("schedule", sa.Text, nullable=True),
        sa.Column("enabled", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("household_id", "slug", name="uq_routine_household_slug"),
    )

    op.create_table(
        "routine_executions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("routine_id", sa.String(36), sa.ForeignKey("routines.id", ondelete="CASCADE"), nullable=True, index=True),
        sa.Column("household_id", sa.String(255), nullable=False, index=True),
        sa.Column("node_id", sa.String, nullable=True),
        sa.Column("trigger", sa.String(16), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("passed", sa.Integer, nullable=False, server_default="0"),
        sa.Column("failed", sa.Integer, nullable=False, server_default="0"),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column("started_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("finished_at", sa.DateTime, nullable=True),
    )


def downgrade() -> None:
    op.drop_table("routine_executions")
    op.drop_table("routines")
