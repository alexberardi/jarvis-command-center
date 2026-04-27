"""add is_active flag to nodes

Revision ID: u1p2q3r4s5t6
Revises: t0o1p2q3r4s5
Create Date: 2026-04-27

Replaces the all-or-nothing DELETE-node flow with a soft-delete flag.
After a successful factory reset the node row stays for audit + to keep
node_tasks foreign keys valid; we just flip is_active=false and filter
list queries. New nodes default to is_active=true.
"""
from alembic import op
import sqlalchemy as sa


revision = "u1p2q3r4s5t6"
down_revision = "t0o1p2q3r4s5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "nodes",
        sa.Column(
            "is_active",
            sa.Boolean(),
            nullable=False,
            server_default=sa.true(),
        ),
    )
    # Drop the server_default once the column exists so future inserts
    # explicitly set the value via the model default.
    op.alter_column("nodes", "is_active", server_default=None)


def downgrade() -> None:
    op.drop_column("nodes", "is_active")
