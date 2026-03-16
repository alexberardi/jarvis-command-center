"""Add direct device control columns to devices table

Add protocol, local_ip, and mac_address columns to support direct WiFi
device control without Home Assistant.

Revision ID: h8c9d0e1f2g3
Revises: g7b8c9d0e1f2
Create Date: 2026-03-16 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'h8c9d0e1f2g3'
down_revision: Union[str, None] = 'g7b8c9d0e1f2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('devices', sa.Column('protocol', sa.String(50), nullable=True))
    op.add_column('devices', sa.Column('local_ip', sa.String(45), nullable=True))
    op.add_column('devices', sa.Column('mac_address', sa.String(17), nullable=True))


def downgrade() -> None:
    op.drop_column('devices', 'mac_address')
    op.drop_column('devices', 'local_ip')
    op.drop_column('devices', 'protocol')
