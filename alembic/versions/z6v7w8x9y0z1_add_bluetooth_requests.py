"""add bluetooth_scan_requests and bluetooth_pair_requests tables

Revision ID: z6v7w8x9y0z1
Revises: y5u6v7w8x9y0
Create Date: 2026-05-04

Supports mobile-driven Bluetooth scan/pair flow via MQTT request/poll pattern.
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "z6v7w8x9y0z1"
down_revision = "y5u6v7w8x9y0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "bluetooth_scan_requests",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("node_id", sa.String, sa.ForeignKey("nodes.node_id", ondelete="CASCADE"), nullable=False),
        sa.Column("household_id", sa.String(255), nullable=False, index=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("role", sa.String(20), nullable=False, server_default="source"),
        sa.Column("results_json", sa.Text, nullable=True),
        sa.Column("device_count", sa.Integer, nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("source", sa.String(20), nullable=False, server_default="mobile"),
        sa.Column("user_id", sa.Integer, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("expires_at", sa.DateTime, nullable=False),
        sa.Column("completed_at", sa.DateTime, nullable=True),
    )

    op.create_table(
        "bluetooth_pair_requests",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("node_id", sa.String, sa.ForeignKey("nodes.node_id", ondelete="CASCADE"), nullable=False),
        sa.Column("household_id", sa.String(255), nullable=False, index=True),
        sa.Column("mac_address", sa.String(17), nullable=False),
        sa.Column("role", sa.String(20), nullable=False, server_default="source"),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("device_name", sa.String(255), nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("expires_at", sa.DateTime, nullable=False),
        sa.Column("completed_at", sa.DateTime, nullable=True),
    )


def downgrade() -> None:
    op.drop_table("bluetooth_pair_requests")
    op.drop_table("bluetooth_scan_requests")
