"""Add parent_room_id to rooms table for hierarchical room structure

Enables parent-child room relationships (e.g., Upstairs → Bedroom 1 → Bathroom).
ondelete=SET NULL so deleting a parent orphans children to root level.

Revision ID: j0e1f2g3h4i5
Revises: i9d0e1f2g3h4
Create Date: 2026-03-17 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'j0e1f2g3h4i5'
down_revision: Union[str, None] = 'i9d0e1f2g3h4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'rooms',
        sa.Column('parent_room_id', sa.String(36), sa.ForeignKey('rooms.id', ondelete='SET NULL'), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('rooms', 'parent_room_id')
