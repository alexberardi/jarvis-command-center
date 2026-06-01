"""add callback_jobs table

Revision ID: c9y0z1a2b3c4
Revises: b8x9y0z1a2b3
Create Date: 2026-05-31

Supports the interactive-notification callback flow: mobile tap on a rich
inbox item -> CC creates a job -> MQTT signals the node -> node fetches
the opaque payload over HTTPS and dispatches to a command's @callback method.
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "c9y0z1a2b3c4"
down_revision = "b8x9y0z1a2b3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "callback_jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "node_id",
            sa.String,
            sa.ForeignKey("nodes.node_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("household_id", sa.String(255), nullable=False, index=True),
        sa.Column("user_id", sa.Integer, nullable=True),
        sa.Column("command_name", sa.String(128), nullable=False),
        sa.Column("callback_name", sa.String(128), nullable=False),
        sa.Column("data_json", sa.Text, nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("result_context_data_json", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("expires_at", sa.DateTime, nullable=False),
        sa.Column("completed_at", sa.DateTime, nullable=True),
    )
    op.create_index(
        "ix_callback_jobs_node_status",
        "callback_jobs",
        ["node_id", "status"],
    )


def downgrade() -> None:
    op.drop_index("ix_callback_jobs_node_status", table_name="callback_jobs")
    op.drop_table("callback_jobs")
