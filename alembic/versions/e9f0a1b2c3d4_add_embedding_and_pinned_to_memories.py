"""Add embedding and is_pinned to user_memories

Revision ID: e9f0a1b2c3d4
Revises: d8017a84f38b
Create Date: 2026-02-27 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e9f0a1b2c3d4'
down_revision: Union[str, None] = 'd8017a84f38b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Safety net: ensure pgvector extension exists (initdb may not re-run on existing data dirs)
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Add is_pinned column
    op.add_column(
        'user_memories',
        sa.Column('is_pinned', sa.Boolean(), nullable=False, server_default=sa.text('false')),
    )

    # Add embedding column (384-dim vector for all-MiniLM-L6-v2)
    op.execute("ALTER TABLE user_memories ADD COLUMN embedding vector(384)")

    # HNSW index for fast cosine similarity search on active memories with embeddings
    op.execute(
        "CREATE INDEX ix_user_memories_embedding_hnsw "
        "ON user_memories USING hnsw (embedding vector_cosine_ops) "
        "WHERE is_active = true AND embedding IS NOT NULL"
    )

    # Partial index for fast pinned memory lookups
    op.create_index(
        'ix_user_memories_pinned',
        'user_memories',
        ['user_id', 'household_id'],
        postgresql_where=sa.text("is_active = true AND is_pinned = true"),
    )


def downgrade() -> None:
    op.drop_index('ix_user_memories_pinned', table_name='user_memories')
    op.execute("DROP INDEX IF EXISTS ix_user_memories_embedding_hnsw")
    op.execute("ALTER TABLE user_memories DROP COLUMN IF EXISTS embedding")
    op.drop_column('user_memories', 'is_pinned')
