"""Add user_memories table

Revision ID: d8017a84f38b
Revises: d3e4f5a6b7c8
Create Date: 2026-02-26 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd8017a84f38b'
down_revision: Union[str, None] = 'd3e4f5a6b7c8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'user_memories',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('household_id', sa.String(255), nullable=False),
        sa.Column('category', sa.String(100), nullable=False, server_default='general'),
        sa.Column('key', sa.String(255), nullable=True),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('source', sa.String(50), nullable=False, server_default='voice'),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default=sa.text('true')),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_user_memories_lookup', 'user_memories', ['user_id', 'household_id', 'is_active'])
    op.create_index('ix_user_memories_category', 'user_memories', ['category'])


def downgrade() -> None:
    op.drop_index('ix_user_memories_category', table_name='user_memories')
    op.drop_index('ix_user_memories_lookup', table_name='user_memories')
    op.drop_table('user_memories')
