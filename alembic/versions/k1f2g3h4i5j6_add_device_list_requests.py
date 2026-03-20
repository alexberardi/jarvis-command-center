"""Add device_list_requests table

Revision ID: k1f2g3h4i5j6
Revises: j0e1f2g3h4i5
Create Date: 2026-03-17 12:00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'k1f2g3h4i5j6'
down_revision: Union[str, None] = 'j0e1f2g3h4i5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'device_list_requests',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('node_id', sa.String(), sa.ForeignKey('nodes.node_id', ondelete='CASCADE'), nullable=False),
        sa.Column('household_id', sa.String(255), nullable=False, index=True),
        sa.Column('status', sa.String(20), nullable=False, server_default='pending'),
        sa.Column('manager_name', sa.String(100), nullable=True),
        sa.Column('can_edit_devices', sa.Boolean(), nullable=True),
        sa.Column('results_json', sa.Text(), nullable=True),
        sa.Column('device_count', sa.Integer(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table('device_list_requests')
