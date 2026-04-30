"""add protocols column to nodes

Revision ID: v2q3r4s5t6u7
Revises: u1p2q3r4s5t6
Create Date: 2026-04-29

Stores a JSON list of protocol names installed on each node (populated
from heartbeat). Used by the CC to route device control commands to a
node that has the required protocol adapter.
"""
from alembic import op
import sqlalchemy as sa


revision = "v2q3r4s5t6u7"
down_revision = "u1p2q3r4s5t6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("nodes", sa.Column("protocols", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("nodes", "protocols")
