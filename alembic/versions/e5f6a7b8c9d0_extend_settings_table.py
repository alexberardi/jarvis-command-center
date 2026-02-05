"""Extend settings table with jarvis-settings-client fields

Add fields required by jarvis-settings-client:
- value_type: Type of the setting value (string, int, float, bool, json)
- category: Category for grouping settings
- description: Human-readable description
- requires_reload: Whether change needs service restart
- is_secret: Whether value should be masked in API responses
- env_fallback: Environment variable to use as fallback
- user_id: User-level scoping

Revision ID: e5f6a7b8c9d0
Revises: 0f1097258abc
Create Date: 2026-02-05 15:20:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e5f6a7b8c9d0'
down_revision: Union[str, None] = '0f1097258abc'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add new columns to settings table
    op.add_column('settings', sa.Column('value_type', sa.String(50), nullable=True))
    op.add_column('settings', sa.Column('category', sa.String(100), nullable=True))
    op.add_column('settings', sa.Column('description', sa.Text(), nullable=True))
    op.add_column('settings', sa.Column('requires_reload', sa.Boolean(), nullable=True, server_default='false'))
    op.add_column('settings', sa.Column('is_secret', sa.Boolean(), nullable=True, server_default='false'))
    op.add_column('settings', sa.Column('env_fallback', sa.String(255), nullable=True))
    op.add_column('settings', sa.Column('user_id', sa.Integer(), nullable=True))

    # Set defaults for existing rows
    op.execute("UPDATE settings SET value_type = 'string' WHERE value_type IS NULL")
    op.execute("UPDATE settings SET category = 'general' WHERE category IS NULL")

    # Make value_type and category non-nullable after setting defaults
    op.alter_column('settings', 'value_type', nullable=False, server_default=None)
    op.alter_column('settings', 'category', nullable=False, server_default=None)

    # Make value nullable (can be NULL if using env fallback)
    op.alter_column('settings', 'value', nullable=True)

    # Create indexes on new columns
    op.create_index(op.f('ix_settings_category'), 'settings', ['category'])
    op.create_index(op.f('ix_settings_user_id'), 'settings', ['user_id'])

    # Drop old unique constraint and create new one including user_id
    op.drop_constraint('uq_setting_scope', 'settings', type_='unique')
    op.create_unique_constraint('uq_setting_scope', 'settings', ['key', 'household_id', 'node_id', 'user_id'])


def downgrade() -> None:
    # Drop new unique constraint and restore old one
    op.drop_constraint('uq_setting_scope', 'settings', type_='unique')
    op.create_unique_constraint('uq_setting_scope', 'settings', ['key', 'household_id', 'node_id'])

    # Drop indexes
    op.drop_index(op.f('ix_settings_user_id'), table_name='settings')
    op.drop_index(op.f('ix_settings_category'), table_name='settings')

    # Make value non-nullable again
    op.alter_column('settings', 'value', nullable=False)

    # Drop new columns
    op.drop_column('settings', 'user_id')
    op.drop_column('settings', 'env_fallback')
    op.drop_column('settings', 'is_secret')
    op.drop_column('settings', 'requires_reload')
    op.drop_column('settings', 'description')
    op.drop_column('settings', 'category')
    op.drop_column('settings', 'value_type')
