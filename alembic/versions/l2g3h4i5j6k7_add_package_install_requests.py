"""add package_install_requests table

Revision ID: l2g3h4i5j6k7
Revises: k1f2g3h4i5j6
Create Date: 2026-03-18 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'l2g3h4i5j6k7'
down_revision = 'k1f2g3h4i5j6'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'package_install_requests',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('node_id', sa.String(), sa.ForeignKey('nodes.node_id', ondelete='CASCADE'), nullable=False),
        sa.Column('household_id', sa.String(255), nullable=False, index=True),
        sa.Column('command_name', sa.String(255), nullable=False),
        sa.Column('github_repo_url', sa.Text(), nullable=False),
        sa.Column('git_tag', sa.String(100), nullable=True),
        sa.Column('status', sa.String(20), nullable=False, server_default='pending'),
        sa.Column('results_json', sa.Text(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table('package_install_requests')
