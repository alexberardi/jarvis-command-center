"""phone_call_sessions.contact_address — carry a resolved address to auto-save

Web-search resolution can turn up the business's street address alongside
its number. The phonebook row that auto-save writes when a call succeeds
has an ``address`` column, but the plan→outcome hop had nowhere to keep it,
so the address was being discovered and then thrown away.

Revision ID: d4e5phone002
Revises: c3d4phone001
"""

import sqlalchemy as sa
from alembic import op

revision = "d4e5phone002"
down_revision = "c3d4phone001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "phone_call_sessions",
        sa.Column("contact_address", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("phone_call_sessions", "contact_address")
