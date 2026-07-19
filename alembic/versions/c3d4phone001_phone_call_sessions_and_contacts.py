"""phone_call_sessions + phone_contacts — phone-calls P1 data model

Session rows are written BEFORE dialing (PRD: no session ever vanishes).
CC owns both tables exclusively; the gateway reports over HTTP.

Revision ID: c3d4phone001
Revises: b2c3cbsrv001
"""

import sqlalchemy as sa
from alembic import op

revision = "c3d4phone001"
down_revision = "b2c3cbsrv001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "phone_contacts",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("household_id", sa.String(255), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("normalized_name", sa.String(255), nullable=False),
        sa.Column("number", sa.String(32), nullable=False),
        sa.Column("address", sa.Text(), nullable=True),
        sa.Column("source", sa.String(16), nullable=False, server_default="manual"),
        sa.Column("line_type", sa.String(16), nullable=True),
        sa.Column("do_not_call", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("overlay_json", sa.Text(), nullable=True),
        sa.Column("verified_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint(
            "household_id", "normalized_name", name="uq_phone_contact_household_name"
        ),
    )
    op.create_index(
        "ix_phone_contacts_household_id", "phone_contacts", ["household_id"]
    )

    op.create_table(
        "phone_call_sessions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("household_id", sa.String(255), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("confirmed_by", sa.Integer(), nullable=True),
        sa.Column(
            "contact_id",
            sa.String(36),
            sa.ForeignKey("phone_contacts.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("contact_name", sa.String(255), nullable=True),
        sa.Column("goal", sa.Text(), nullable=False),
        sa.Column("details", sa.Text(), nullable=True),
        sa.Column("constraints", sa.Text(), nullable=True),
        sa.Column("resolved_number", sa.String(32), nullable=True),
        sa.Column("dialed_number", sa.String(32), nullable=True),
        sa.Column("number_edited", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("line_type", sa.String(16), nullable=True),
        sa.Column("state", sa.String(16), nullable=False, server_default="draft"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("transcript_json", sa.Text(), nullable=True),
        sa.Column("outcome_json", sa.Text(), nullable=True),
        sa.Column("audio_object_key", sa.String(512), nullable=True),
        sa.Column("worker_url", sa.String(512), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(), nullable=True),
        sa.Column("twilio_call_sid", sa.String(64), nullable=True),
        sa.Column("duration_seconds", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("confirmed_at", sa.DateTime(), nullable=True),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
        sa.Column("ended_at", sa.DateTime(), nullable=True),
    )
    op.create_index(
        "ix_phone_call_sessions_household_id", "phone_call_sessions", ["household_id"]
    )
    op.create_index("ix_phone_call_sessions_state", "phone_call_sessions", ["state"])


def downgrade() -> None:
    op.drop_index("ix_phone_call_sessions_state", table_name="phone_call_sessions")
    op.drop_index(
        "ix_phone_call_sessions_household_id", table_name="phone_call_sessions"
    )
    op.drop_table("phone_call_sessions")
    op.drop_index("ix_phone_contacts_household_id", table_name="phone_contacts")
    op.drop_table("phone_contacts")
