"""add expires_at to config_pushes

Revision ID: c2d3e4f5a6b7
Revises: b1c2d3e4f5a6
Create Date: 2026-02-14 12:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

revision = 'c2d3e4f5a6b7'
down_revision = 'b1c2d3e4f5a6'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('config_pushes', sa.Column('expires_at', sa.DateTime, nullable=True))


def downgrade() -> None:
    op.drop_column('config_pushes', 'expires_at')
