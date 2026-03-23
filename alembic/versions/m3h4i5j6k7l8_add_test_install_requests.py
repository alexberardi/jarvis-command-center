"""Add test_install_requests table for Forge share code test installs.

Revision ID: m3h4i5j6k7l8
Revises: l2g3h4i5j6k7
Create Date: 2026-03-23
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "m3h4i5j6k7l8"
down_revision: Union[str, None] = "l2g3h4i5j6k7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "test_install_requests",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("node_id", sa.String(), sa.ForeignKey("nodes.node_id", ondelete="CASCADE"), nullable=False),
        sa.Column("household_id", sa.String(255), nullable=False, index=True),
        sa.Column("share_code", sa.String(6), nullable=False),
        sa.Column("package_name", sa.String(255), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("results_json", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("test_install_requests")
