"""add user_id to auth_sessions

The user who initiates an OAuth flow from mobile is the owner of the
resulting tokens. Nullable: admin-key flows and pre-upgrade sessions have
no user, and nodes treat a missing user_id as the legacy (integration
scope) behavior.

Revision ID: f2g3h4i5j6k7
Revises: e1f2g3h4i5j6
Create Date: 2026-06-12 12:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

revision = "f2g3h4i5j6k7"
down_revision = "e1f2g3h4i5j6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("auth_sessions", sa.Column("user_id", sa.Integer, nullable=True))


def downgrade() -> None:
    op.drop_column("auth_sessions", "user_id")
