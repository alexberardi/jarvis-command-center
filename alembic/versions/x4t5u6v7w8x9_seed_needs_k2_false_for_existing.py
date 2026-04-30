"""seed needs_k2=false for nodes that pre-date the column

Revision ID: x4t5u6v7w8x9
Revises: w3s4t5u6v7w8
Create Date: 2026-04-30

Nodes registered before this column existed were provisioned via the
AP-mode flow that delivers K2, so they already have it. The schema
migration defaults the column to True (correct for new rows so the
settings gear shows until proven otherwise), but every paired-on-the-
old-flow node is incorrectly flagged as needing K2 until its next
heartbeat. Reset them all to False here; nodes that genuinely need K2
will report so on their next heartbeat.
"""
from alembic import op


revision = "x4t5u6v7w8x9"
down_revision = "w3s4t5u6v7w8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("UPDATE nodes SET needs_k2 = false")


def downgrade() -> None:
    pass
