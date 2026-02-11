"""add provisioning_tokens table

Revision ID: a1b2c3d4e5f6
Revises: f6a7b8c9d0e1
Create Date: 2026-02-10 12:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

revision = 'a1b2c3d4e5f6'
down_revision = 'f6a7b8c9d0e1'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'provisioning_tokens',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('token_hash', sa.String(64), nullable=False, unique=True),
        sa.Column('node_id', sa.String(36), nullable=False, index=True),
        sa.Column('household_id', sa.String(255), nullable=False),
        sa.Column('room', sa.String(255), nullable=True),
        sa.Column('name', sa.String(255), nullable=True),
        sa.Column('created_by_user_id', sa.Integer, nullable=True),
        sa.Column('expires_at', sa.DateTime, nullable=False),
        sa.Column('created_at', sa.DateTime, nullable=False),
        sa.Column('consumed_at', sa.DateTime, nullable=True),
    )


def downgrade() -> None:
    op.drop_table('provisioning_tokens')
