"""add Phase 5 adapter deployment tables

Revision ID: s9n0o1p2q3r4
Revises: r8m9n0o1p2q3
Create Date: 2026-04-19

Three new tables that back the auto-deploy loop:

• active_adapter          — which adapter is currently in effect per household
• adapter_training_state  — per-household scheduler state (last_trained_at,
                             threshold counters, training-in-flight flag)
• adapter_history         — append-only audit log, supports rollback
"""
from alembic import op
import sqlalchemy as sa

revision = "s9n0o1p2q3r4"
down_revision = "r8m9n0o1p2q3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "active_adapter",
        sa.Column("household_id", sa.String(255), primary_key=True),
        sa.Column("adapter_hash", sa.String(128), nullable=False),
        sa.Column("pass_rate", sa.Float(), nullable=True),
        sa.Column("trained_on_examples", sa.Integer(), nullable=True),
        sa.Column("deployed_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "adapter_training_state",
        sa.Column("household_id", sa.String(255), primary_key=True),
        sa.Column("last_trained_at", sa.DateTime(), nullable=True),
        sa.Column("last_cutoff_at", sa.DateTime(), nullable=True),
        sa.Column("last_example_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "is_training",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
            index=True,
        ),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "adapter_history",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("household_id", sa.String(255), nullable=False, index=True),
        sa.Column("adapter_hash", sa.String(128), nullable=False),
        sa.Column("pass_rate", sa.Float(), nullable=True),
        sa.Column("trained_on_examples", sa.Integer(), nullable=True),
        sa.Column("deployed_at", sa.DateTime(), nullable=False),
        sa.Column("replaced_at", sa.DateTime(), nullable=True),
        sa.Column(
            "trigger",
            sa.String(32),
            nullable=False,
            server_default="scheduler",
        ),  # "scheduler" | "manual" | "rollback"
    )


def downgrade() -> None:
    op.drop_table("adapter_history")
    op.drop_table("adapter_training_state")
    op.drop_table("active_adapter")
