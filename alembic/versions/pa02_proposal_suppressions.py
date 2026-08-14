"""add proposal_suppressions (the 'never suggest this again' blocklist)

Revision ID: pa02suppressions
Revises: pa01idempotency
Create Date: 2026-08-09

Per-user "never suggest this again" entries for proposable actions. Written by
the suppress handler on a card tap; read by the detector agent (deterministic
source_key skip + descriptor injected into its extraction prompt) and by the
mobile management screen.
"""
from alembic import op
import sqlalchemy as sa


revision = "pa02suppressions"
down_revision = "pa01idempotency"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "proposal_suppressions",
        sa.Column("id", sa.String(length=40), primary_key=True),
        sa.Column("household_id", sa.String(length=255), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("command", sa.String(length=128), nullable=False),
        sa.Column("source_key", sa.String(length=512), nullable=True),
        sa.Column("descriptor", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_proposal_suppressions_household_id", "proposal_suppressions", ["household_id"])
    op.create_index("ix_proposal_suppressions_user_id", "proposal_suppressions", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_proposal_suppressions_user_id", table_name="proposal_suppressions")
    op.drop_index("ix_proposal_suppressions_household_id", table_name="proposal_suppressions")
    op.drop_table("proposal_suppressions")
