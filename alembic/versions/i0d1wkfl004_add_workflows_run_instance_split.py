"""Workflow run/instance split: a durable ``workflows`` RUN table + errand_plans link

The Errand Runner's durable executor is a general workflow engine (errand-runner
PRD §2.5-2.6). This splits the RUN/INSTANCE off the DRAFT/DEFINITION:

- NEW ``workflows`` table — one row per RUN. The engine (workflow_engine) owns its
  lifecycle (running → waiting → done|partial|failed|timeout|cancelled), suspending
  on a deferred step (a phone call) and resuming on a signal. Consumer-agnostic
  (``kind``) so scheduling (one definition → many runs) and future workflow kinds
  drop in without re-modeling. Carries the execution state that used to live on
  ``errand_plans`` (steps/cursor/results_json/state) plus goal/title for the summary.
- ``errand_plans`` gains ``workflow_id`` — the DRAFT links to the RUN created on Run.
  ``errand_plans.cursor``/``results_json`` (added in h9c0errnd003) become vestigial;
  left in place (not dropped) — the Workflow owns run state now. ``errand_plans.state``
  gains a ``launched`` value (free String — no schema change) once its run is created.

The phone→errand link (``phone_call_sessions.errand_id``, h9c0errnd003) now holds the
WORKFLOW run id — the executor links a call to the run, and the resume sweep looks it
up by the workflow id. Same column, run-scoped value; no schema change needed.

Revision ID: i0d1wkfl004
Revises: h9c0errnd003
Create Date: 2026-07-30 00:00:00.000000
"""

import sqlalchemy as sa
from alembic import op

revision = "i0d1wkfl004"
down_revision = "h9c0errnd003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "workflows",
        sa.Column("id", sa.String(40), primary_key=True),
        sa.Column("kind", sa.String(40), nullable=False),
        sa.Column("household_id", sa.String(255), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("node_id", sa.String(), nullable=True),
        sa.Column("goal", sa.Text(), nullable=False, server_default=""),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("steps", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("cursor", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("results_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("state", sa.String(16), nullable=False, server_default="running"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_workflows_kind", "workflows", ["kind"])
    op.create_index("ix_workflows_household_id", "workflows", ["household_id"])
    op.create_index("ix_workflows_state", "workflows", ["state"])

    op.add_column("errand_plans", sa.Column("workflow_id", sa.String(40), nullable=True))
    op.create_index("ix_errand_plans_workflow_id", "errand_plans", ["workflow_id"])


def downgrade() -> None:
    op.drop_index("ix_errand_plans_workflow_id", table_name="errand_plans")
    op.drop_column("errand_plans", "workflow_id")
    op.drop_index("ix_workflows_state", table_name="workflows")
    op.drop_index("ix_workflows_household_id", table_name="workflows")
    op.drop_index("ix_workflows_kind", table_name="workflows")
    op.drop_table("workflows")
