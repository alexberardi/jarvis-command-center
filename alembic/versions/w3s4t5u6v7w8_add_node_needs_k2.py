"""add needs_k2 column to nodes

Revision ID: w3s4t5u6v7w8
Revises: v2q3r4s5t6u7
Create Date: 2026-04-30

Reported by the node in its heartbeat. True when the node has no K2
secrets-encryption key. Mobile uses this to show the settings gear so
the user can pair K2 — without it, container nodes (which never go
through AP provisioning) have no way to add K2.

Defaults to True so the gear is shown until proven otherwise; existing
nodes that already have K2 will flip it to False on their next heartbeat.
"""
from alembic import op
import sqlalchemy as sa


revision = "w3s4t5u6v7w8"
down_revision = "v2q3r4s5t6u7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "nodes",
        sa.Column("needs_k2", sa.Boolean(), nullable=False, server_default=sa.true()),
    )


def downgrade() -> None:
    op.drop_column("nodes", "needs_k2")
