"""add settings table for scoped configuration

Revision ID: d4e5f6a7b8c9
Revises: 0f1097258abc
Create Date: 2026-02-03 21:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd4e5f6a7b8c9'
down_revision: Union[str, None] = '0f1097258abc'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create settings table with cascade lookup: Node > Household > Default
    op.create_table('settings',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('key', sa.String(), nullable=False),
        sa.Column('value', sa.String(), nullable=False),
        sa.Column('household_id', sa.String(), nullable=True),
        sa.Column('node_id', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('key', 'household_id', 'node_id', name='uq_setting_scope')
    )

    # Create index for faster lookups by key
    op.create_index('ix_settings_key', 'settings', ['key'], unique=False)

    # Create index for household lookups
    op.create_index('ix_settings_household_id', 'settings', ['household_id'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_settings_household_id', table_name='settings')
    op.drop_index('ix_settings_key', table_name='settings')
    op.drop_table('settings')
