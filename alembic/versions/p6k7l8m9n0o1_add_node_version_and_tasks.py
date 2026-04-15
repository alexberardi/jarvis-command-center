"""add node version fields and node_tasks table

Revision ID: p6k7l8m9n0o1
Revises: o5j6k7l8m9n0
Create Date: 2026-04-15

Adds columns to `nodes` so the mobile app can show per-node version info
(last_seen_version, install_mode, is_busy) populated from the heartbeat
payload. Adds `node_tasks` for tracking long-running per-node operations
— update is the first consumer.
"""
from alembic import op
import sqlalchemy as sa


revision = "p6k7l8m9n0o1"
down_revision = "o5j6k7l8m9n0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("nodes", sa.Column("last_seen_version", sa.String(64), nullable=True))
    op.add_column("nodes", sa.Column("install_mode", sa.String(16), nullable=True))
    op.add_column("nodes", sa.Column("is_busy", sa.Boolean(), nullable=False, server_default=sa.false()))
    op.add_column("nodes", sa.Column("git_sha", sa.String(40), nullable=True))

    op.create_table(
        "node_tasks",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("node_id", sa.String, sa.ForeignKey("nodes.node_id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("target_version", sa.String(64), nullable=True),
        sa.Column("state", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("finished_at", sa.DateTime, nullable=True),
    )
    op.create_index("idx_node_tasks_node_state", "node_tasks", ["node_id", "state"])


def downgrade() -> None:
    op.drop_index("idx_node_tasks_node_state", table_name="node_tasks")
    op.drop_table("node_tasks")
    op.drop_column("nodes", "git_sha")
    op.drop_column("nodes", "is_busy")
    op.drop_column("nodes", "install_mode")
    op.drop_column("nodes", "last_seen_version")
