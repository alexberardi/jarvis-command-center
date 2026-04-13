"""Add prompt_provider_install_requests table for async prompt provider installs.

Revision ID: n4i5j6k7l8m9
Revises: m3h4i5j6k7l8
Create Date: 2026-04-10
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "n4i5j6k7l8m9"
down_revision: Union[str, None] = "m3h4i5j6k7l8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "prompt_provider_install_requests",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("household_id", sa.String(255), nullable=False, index=True),
        sa.Column("package_name", sa.String(255), nullable=True),
        sa.Column("github_repo_url", sa.Text(), nullable=False),
        sa.Column("git_tag", sa.String(100), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("results_json", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("prompt_provider_install_requests")
