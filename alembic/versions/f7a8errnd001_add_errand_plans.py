"""errand_plans — Errand Runner POC draft-plan store (errand-runner PRD §2-§3)

A drafted errand plan is persisted here as a DRAFT before anything runs; the
plan card is the trust checkpoint. CC owns this row exclusively (the node holds
no errand state). Lifecycle mirrors the phone-call state machine; `steps` is the
routine-shaped step list the executor (execute_routine_on_node) consumes.

Revision ID: f7a8errnd001
Revises: m3n4o5p6q7r8
Create Date: 2026-07-28 12:00:00.000000
"""

import sqlalchemy as sa
from alembic import op

revision = "f7a8errnd001"
down_revision = "m3n4o5p6q7r8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "errand_plans",
        sa.Column("id", sa.String(40), primary_key=True),
        sa.Column("household_id", sa.String(255), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("node_id", sa.String(), nullable=True),
        sa.Column("goal", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("steps", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("routine_slug", sa.String(255), nullable=True),
        sa.Column("state", sa.String(16), nullable=False, server_default="draft"),
        sa.Column("revision", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("confirmed_at", sa.DateTime(), nullable=True),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_errand_plans_household_id", "errand_plans", ["household_id"])
    op.create_index("ix_errand_plans_state", "errand_plans", ["state"])


def downgrade() -> None:
    op.drop_index("ix_errand_plans_state", table_name="errand_plans")
    op.drop_index("ix_errand_plans_household_id", table_name="errand_plans")
    op.drop_table("errand_plans")
