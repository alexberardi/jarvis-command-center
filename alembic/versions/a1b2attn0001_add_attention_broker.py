"""add attention broker tables

The deterministic notification-governance layer (prds/attention-broker.md):
attention_events (what sources raised), attention_deliveries (rung + gate
trail per event), attention_source_tiers (earned speaking rights, phase 2),
attention_consents (user-granted ceilings, phase 3), attention_feedback
(useful/mute verbs, phase 2). All five ship in one migration so later phases
are settings/code-only.

Revision ID: a1b2attn0001
Revises: d9c8b7a6e5f4
Create Date: 2026-07-18 21:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

revision = "a1b2attn0001"
down_revision = "d9c8b7a6e5f4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "attention_events",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("household_id", sa.String(255), nullable=False, index=True),
        sa.Column("source", sa.String(100), nullable=False),
        sa.Column("category", sa.String(50), nullable=False),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("summary", sa.Text, nullable=False, server_default=""),
        sa.Column("dedupe_key", sa.String(255), nullable=True, index=True),
        sa.Column("target_user_id", sa.Integer, nullable=True),
        sa.Column("origin_node_id", sa.String(255), nullable=True),
        sa.Column("payload_json", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "attention_deliveries",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("event_id", sa.String(36), sa.ForeignKey("attention_events.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("household_id", sa.String(255), nullable=False, index=True),
        sa.Column("rung", sa.String(20), nullable=False),
        sa.Column("gate_trail_json", sa.Text, nullable=False, server_default="[]"),
        sa.Column("withheld_by", sa.String(50), nullable=True),
        sa.Column("inbox_item_id", sa.String(36), nullable=True),
        sa.Column("request_id", sa.String(36), nullable=True, index=True),
        sa.Column("outcome", sa.String(30), nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now(), index=True),
    )

    op.create_table(
        "attention_source_tiers",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("household_id", sa.String(255), nullable=False, index=True),
        sa.Column("source", sa.String(100), nullable=False),
        sa.Column("category", sa.String(50), nullable=False),
        sa.Column("tier", sa.Integer, nullable=False, server_default="1"),
        sa.Column("score", sa.Float, nullable=False, server_default="0"),
        sa.Column("state_reason", sa.String(255), nullable=True),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("household_id", "source", "category", name="uq_attention_tier_scope"),
    )

    op.create_table(
        "attention_consents",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("household_id", sa.String(255), nullable=False, index=True),
        sa.Column("source", sa.String(100), nullable=False),
        sa.Column("category", sa.String(50), nullable=False),
        sa.Column("max_rung", sa.String(20), nullable=False, server_default="push"),
        sa.Column("granted_by_user_id", sa.Integer, nullable=True),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("household_id", "source", "category", name="uq_attention_consent_scope"),
    )

    op.create_table(
        "attention_feedback",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("delivery_id", sa.String(36), sa.ForeignKey("attention_deliveries.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("user_id", sa.Integer, nullable=True),
        sa.Column("verb", sa.String(20), nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("attention_feedback")
    op.drop_table("attention_consents")
    op.drop_table("attention_source_tiers")
    op.drop_table("attention_deliveries")
    op.drop_table("attention_events")
