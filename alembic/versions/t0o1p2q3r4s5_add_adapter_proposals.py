"""add adapter proposals table (Phase 7.1)

Revision ID: t0o1p2q3r4s5
Revises: s9n0o1p2q3r4
Create Date: 2026-04-20

One table: adapter_proposals. Backs the user-approved adapter deploy flow —
scheduler writes a row + posts an inbox item, apply/dismiss/revert endpoints
mutate status and drive AdapterRegistry deploy/rollback.
"""
from alembic import op
import sqlalchemy as sa


revision = "t0o1p2q3r4s5"
down_revision = "s9n0o1p2q3r4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "adapter_proposals",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("household_id", sa.String(255), nullable=False, index=True),
        sa.Column("adapter_hash", sa.String(128), nullable=False),
        sa.Column("provider_name_before", sa.String(255), nullable=True),
        sa.Column("provider_name_after", sa.String(255), nullable=True),
        sa.Column("pass_rate_before", sa.Float(), nullable=True),
        sa.Column("pass_rate_after", sa.Float(), nullable=True),
        sa.Column("latency_before_s", sa.Float(), nullable=True),
        sa.Column("latency_after_s", sa.Float(), nullable=True),
        sa.Column("per_command_delta_json", sa.Text(), nullable=True),
        sa.Column("trained_on_examples", sa.Integer(), nullable=True),
        # pending | applied | dismissed | expired | superseded
        sa.Column(
            "status",
            sa.String(20),
            nullable=False,
            server_default="pending",
            index=True,
        ),
        sa.Column("inbox_item_id", sa.String(36), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("decided_at", sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("adapter_proposals")
