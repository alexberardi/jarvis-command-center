"""Add cloud_id to devices and device_scan_requests table

cloud_id supports cloud-only devices (Govee, Nest, Schlage) that have no
MAC or LAN presence. device_scan_requests enables the user-driven scan flow:
mobile → CC → MQTT → node → CC → mobile.

Revision ID: i9d0e1f2g3h4
Revises: h8c9d0e1f2g3
Create Date: 2026-03-16 18:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'i9d0e1f2g3h4'
down_revision: Union[str, None] = 'h8c9d0e1f2g3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add cloud_id to devices table
    op.add_column('devices', sa.Column('cloud_id', sa.String(255), nullable=True))

    # Create device_scan_requests table
    op.create_table(
        'device_scan_requests',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('node_id', sa.String(), sa.ForeignKey('nodes.node_id', ondelete='CASCADE'), nullable=False),
        sa.Column('household_id', sa.String(255), nullable=False, index=True),
        sa.Column('status', sa.String(20), nullable=False, server_default='pending'),
        sa.Column('results_json', sa.Text(), nullable=True),
        sa.Column('device_count', sa.Integer(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table('device_scan_requests')
    op.drop_column('devices', 'cloud_id')
