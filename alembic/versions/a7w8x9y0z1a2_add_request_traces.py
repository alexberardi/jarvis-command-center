"""add request_traces table for trace visualizer

Revision ID: a7w8x9y0z1a2
Revises: z6v7w8x9y0z1
Create Date: 2026-05-05

Stores per-request distributed traces with timing spans for the
admin trace page and mobile trace waterfall.
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "a7w8x9y0z1a2"
down_revision = "z6v7w8x9y0z1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "request_traces",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("conversation_id", sa.String(255), nullable=False, index=True),
        sa.Column("request_type", sa.String(20), nullable=False),
        sa.Column("source", sa.String(20), nullable=False),
        sa.Column("node_id", sa.String, sa.ForeignKey("nodes.node_id", ondelete="SET NULL"), nullable=True),
        sa.Column("household_id", sa.String(255), nullable=True, index=True),
        sa.Column("user_command", sa.Text, nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="ok"),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("total_duration_ms", sa.Float, nullable=False),
        sa.Column("spans_json", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_request_traces_created_at", "request_traces", ["created_at"])


def downgrade() -> None:
    op.drop_table("request_traces")
