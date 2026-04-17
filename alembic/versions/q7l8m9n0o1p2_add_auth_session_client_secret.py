"""add client_secret_enc to auth_sessions

Revision ID: q7l8m9n0o1p2
Revises: p6k7l8m9n0o1
Create Date: 2026-04-17

Adds encrypted client_secret column to auth_sessions for Web Application
OAuth flows (e.g., Nest camera support). Null for PKCE-only flows.
"""
from alembic import op
import sqlalchemy as sa


revision = "q7l8m9n0o1p2"
down_revision = "p6k7l8m9n0o1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("auth_sessions", sa.Column("client_secret_enc", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("auth_sessions", "client_secret_enc")
