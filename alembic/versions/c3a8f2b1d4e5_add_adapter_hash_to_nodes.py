"""Add adapter_hash column to nodes table

Revision ID: c3a8f2b1d4e5
Revises: 82feef6dbb91
Create Date: 2026-01-22

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c3a8f2b1d4e5'
down_revision: Union[str, None] = '82feef6dbb91'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add adapter_hash column for per-node LoRA adapter selection
    op.add_column('nodes', sa.Column('adapter_hash', sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column('nodes', 'adapter_hash')
