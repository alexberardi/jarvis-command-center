"""Schedules: a durable trigger that fires the plan->card loop on a clock

A ``schedule`` is a TIME TRIGGER, not a plan or a run. At ``next_fire_at`` a sweep
re-plans the schedule's ``intent`` against fresh context and posts a plan card the
user must approve (design: re-plan + re-confirm each run — a schedule never acts
autonomously). One-shot for now (``recurrence`` NULL); recurrence re-arms
``next_fire_at`` after each fire. ``next_fire_at`` indexed for the due sweep.

Revision ID: j2f3sched006
Revises: i1e2wait005
Create Date: 2026-07-31 00:00:00.000000
"""

import sqlalchemy as sa
from alembic import op

revision = "j2f3sched006"
down_revision = "i1e2wait005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "schedules",
        sa.Column("id", sa.String(40), primary_key=True),
        sa.Column("household_id", sa.String(255), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("node_id", sa.String(), nullable=True),
        sa.Column("intent", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("timezone", sa.String(64), nullable=False, server_default="UTC"),
        sa.Column("next_fire_at", sa.DateTime(), nullable=False),
        sa.Column("recurrence", sa.Text(), nullable=True),
        sa.Column("state", sa.String(16), nullable=False, server_default="active"),
        sa.Column("last_fired_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_schedules_household_id", "schedules", ["household_id"])
    op.create_index("ix_schedules_next_fire_at", "schedules", ["next_fire_at"])
    op.create_index("ix_schedules_state", "schedules", ["state"])


def downgrade() -> None:
    op.drop_index("ix_schedules_state", table_name="schedules")
    op.drop_index("ix_schedules_next_fire_at", table_name="schedules")
    op.drop_index("ix_schedules_household_id", table_name="schedules")
    op.drop_table("schedules")
