"""add conversation_transcripts table

Revision ID: o5j6k7l8m9n0
Revises: n4i5j6k7l8m9
Create Date: 2026-04-11

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "o5j6k7l8m9n0"
down_revision = "n4i5j6k7l8m9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "conversation_transcripts",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.Integer(), nullable=False, index=True),
        sa.Column("household_id", sa.String(255), nullable=False),
        sa.Column("conversation_id", sa.String(255), nullable=False),
        sa.Column("user_message", sa.Text(), nullable=False),
        sa.Column("assistant_message", sa.Text(), nullable=True),
        sa.Column("tool_calls_json", sa.Text(), nullable=True),
        sa.Column("is_processed", sa.Boolean(), nullable=False, server_default="false", index=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now(), index=True),
        sa.Column("processed_at", sa.DateTime(), nullable=True),
        sa.Column("extraction_job_id", sa.String(36), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("conversation_transcripts")
