"""Guard the postgres image pin in the compose files.

roadmap#46: command-center's own compose files were left on
`pgvector/pgvector:pg15` while the installer standardized on `pg16`. Postgres
majors are not data-compatible, so a mixed setup fails hard with
`FATAL: database files are incompatible with server`. These tests lock both
compose files (and the operator migration doc) to the pg16 target so the
mismatch cannot silently regress.
"""
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
PROD_COMPOSE = REPO_ROOT / "docker-compose.prod.yaml"
DEV_COMPOSE = REPO_ROOT / "docker-compose.dev.yaml"
MIGRATION_DOC = REPO_ROOT / "docs" / "postgres-pg15-to-pg16-migration.md"


def _postgres_image(compose_path: Path) -> str:
    data = yaml.safe_load(compose_path.read_text())
    return data["services"]["postgres"]["image"]


def test_prod_compose_postgres_pinned_pg16():
    """The prod compose's postgres service pins pg16."""
    assert _postgres_image(PROD_COMPOSE) == "pgvector/pgvector:pg16"


def test_dev_compose_postgres_pinned_pg16():
    """The dev compose's standalone-profile postgres service pins pg16."""
    assert _postgres_image(DEV_COMPOSE) == "pgvector/pgvector:pg16"


@pytest.mark.parametrize("compose_path", [PROD_COMPOSE, DEV_COMPOSE])
def test_no_pg15_reference_remains_in_compose_files(compose_path):
    """No stray pg15 reference survives in either compose file."""
    text = compose_path.read_text()
    assert "pgvector/pgvector:pg15" not in text
    assert ":pg15" not in text


@pytest.mark.parametrize("compose_path", [PROD_COMPOSE, DEV_COMPOSE])
def test_pgvector_repository_unchanged(compose_path):
    """The bump only moves the tag; the image stays pgvector-flavored so the
    vector extension survives (guards against an accidental postgres:16 swap)."""
    assert _postgres_image(compose_path).startswith("pgvector/pgvector:")


def test_migration_doc_present():
    """The operator pg15->pg16 migration doc exists and documents a real path."""
    assert MIGRATION_DOC.exists(), f"missing migration doc: {MIGRATION_DOC}"
    text = MIGRATION_DOC.read_text()
    assert text.strip(), "migration doc is empty"
    assert "pg_dump" in text or "pg_restore" in text or "pg_upgrade" in text
