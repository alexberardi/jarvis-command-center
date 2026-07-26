"""add person_characterizations table

Revision ID: m3n4o5p6q7r8
Revises: d4e5phone002
Create Date: 2026-07-23 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'm3n4o5p6q7r8'
down_revision = 'd4e5phone002'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'person_characterizations',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('household_id', sa.String(255), nullable=False),
        sa.Column('body', sa.Text(), nullable=False),
        sa.Column('rendered', sa.Text(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('version', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('last_transcript_at', sa.DateTime(), nullable=True),
        sa.Column('model', sa.String(100), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id', 'household_id', name='uq_person_characterization_user_household'),
    )
    op.create_index(
        'ix_person_characterizations_user_household',
        'person_characterizations',
        ['user_id', 'household_id'],
    )


def downgrade() -> None:
    op.drop_index('ix_person_characterizations_user_household', table_name='person_characterizations')
    op.drop_table('person_characterizations')
