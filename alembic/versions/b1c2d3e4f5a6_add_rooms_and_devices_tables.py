"""add rooms and devices tables

Revision ID: b1c2d3e4f5a6
Revises: a1b2c3d4e5f6
Create Date: 2026-02-14 12:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

revision = 'b1c2d3e4f5a6'
down_revision = 'a1b2c3d4e5f6'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'rooms',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('household_id', sa.String(255), nullable=False, index=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('normalized_name', sa.String(255), nullable=False),
        sa.Column('icon', sa.String(50), nullable=True),
        sa.Column('ha_area_id', sa.String(255), nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=True),
        sa.Column('updated_at', sa.DateTime, nullable=True),
        sa.UniqueConstraint('household_id', 'normalized_name', name='uq_room_household_name'),
    )

    op.create_table(
        'devices',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('household_id', sa.String(255), nullable=False, index=True),
        sa.Column('room_id', sa.String(36), sa.ForeignKey('rooms.id', ondelete='SET NULL'), nullable=True),
        sa.Column('entity_id', sa.String(255), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('domain', sa.String(50), nullable=False),
        sa.Column('device_class', sa.String(100), nullable=True),
        sa.Column('manufacturer', sa.String(255), nullable=True),
        sa.Column('model', sa.String(255), nullable=True),
        sa.Column('source', sa.String(50), nullable=True, server_default='home_assistant'),
        sa.Column('ha_device_id', sa.String(255), nullable=True),
        sa.Column('is_controllable', sa.Boolean, nullable=True, server_default='true'),
        sa.Column('is_active', sa.Boolean, nullable=True, server_default='true'),
        sa.Column('created_at', sa.DateTime, nullable=True),
        sa.Column('updated_at', sa.DateTime, nullable=True),
        sa.UniqueConstraint('household_id', 'entity_id', name='uq_device_household_entity'),
    )

    # Add room_id and household_id to nodes table
    op.add_column('nodes', sa.Column('room_id', sa.String(36), nullable=True))
    op.add_column('nodes', sa.Column('household_id', sa.String(255), nullable=True, index=True))
    op.create_foreign_key(
        'fk_nodes_room_id',
        'nodes', 'rooms',
        ['room_id'], ['id'],
        ondelete='SET NULL',
    )

    # Config push table for encrypted config relay (phone -> CC -> MQTT -> node)
    op.create_table(
        'config_pushes',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('node_id', sa.String, sa.ForeignKey('nodes.node_id', ondelete='CASCADE'), nullable=False),
        sa.Column('config_type', sa.String(50), nullable=False),
        sa.Column('ciphertext', sa.Text, nullable=False),
        sa.Column('nonce', sa.Text, nullable=False),
        sa.Column('tag', sa.Text, nullable=False),
        sa.Column('status', sa.String(20), nullable=False, server_default='pending'),
        sa.Column('created_at', sa.DateTime, nullable=False),
        sa.Column('consumed_at', sa.DateTime, nullable=True),
    )


def downgrade() -> None:
    op.drop_table('config_pushes')
    op.drop_constraint('fk_nodes_room_id', 'nodes', type_='foreignkey')
    op.drop_column('nodes', 'household_id')
    op.drop_column('nodes', 'room_id')
    op.drop_table('devices')
    op.drop_table('rooms')
