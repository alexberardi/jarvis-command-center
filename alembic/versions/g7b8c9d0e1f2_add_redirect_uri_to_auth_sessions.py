"""Add redirect_uri to auth_sessions

Store the redirect URI used when creating an OAuth session so it can be
replayed during the code exchange (required by OAuth2 spec and by Google).

Revision ID: g7b8c9d0e1f2
Revises: e9f0a1b2c3d4
Create Date: 2026-03-15 16:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'g7b8c9d0e1f2'
down_revision: Union[str, None] = 'e9f0a1b2c3d4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('auth_sessions', sa.Column('redirect_uri', sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column('auth_sessions', 'redirect_uri')
