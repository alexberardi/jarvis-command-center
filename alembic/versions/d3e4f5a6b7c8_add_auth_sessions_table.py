"""add auth_sessions table

Revision ID: d3e4f5a6b7c8
Revises: c2d3e4f5a6b7
Create Date: 2026-02-15 12:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

revision = 'd3e4f5a6b7c8'
down_revision = 'c2d3e4f5a6b7'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'auth_sessions',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('provider', sa.String(100), nullable=False),
        sa.Column('node_id', sa.String, sa.ForeignKey('nodes.node_id', ondelete='CASCADE'), nullable=False),
        sa.Column('status', sa.String(20), nullable=False, server_default='pending'),
        sa.Column('state', sa.String(64), nullable=False, unique=True),
        sa.Column('code_verifier', sa.String(128), nullable=True),
        sa.Column('provider_base_url', sa.String(500), nullable=True),
        sa.Column('authorize_url', sa.Text, nullable=True),
        sa.Column('exchange_url', sa.Text, nullable=True),
        sa.Column('client_id', sa.String(255), nullable=False),
        sa.Column('access_token_enc', sa.Text, nullable=True),
        sa.Column('refresh_token_enc', sa.Text, nullable=True),
        sa.Column('token_data_enc', sa.Text, nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False),
        sa.Column('expires_at', sa.DateTime, nullable=False),
        sa.Column('completed_at', sa.DateTime, nullable=True),
    )
    op.create_index('ix_auth_sessions_provider', 'auth_sessions', ['provider'])
    op.create_index('ix_auth_sessions_node_id', 'auth_sessions', ['node_id'])
    op.create_index('ix_auth_sessions_status', 'auth_sessions', ['status'])


def downgrade() -> None:
    op.drop_index('ix_auth_sessions_status')
    op.drop_index('ix_auth_sessions_node_id')
    op.drop_index('ix_auth_sessions_provider')
    op.drop_table('auth_sessions')
