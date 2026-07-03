"""verify_admin_key compares the admin token in constant time + stays fail-closed."""
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from app import deps

TOKEN = "a" * 40


def test_correct_key_passes():
    with patch.object(deps, "ADMIN_API_KEY", TOKEN):
        assert deps.verify_admin_key(TOKEN) is None


def test_wrong_key_rejected():
    with patch.object(deps, "ADMIN_API_KEY", TOKEN):
        with pytest.raises(HTTPException) as exc:
            deps.verify_admin_key("b" * 40)
    assert exc.value.status_code == 401


def test_unset_admin_key_fails_closed():
    # An unset ADMIN_API_KEY must reject everything, not accept an empty string.
    with patch.object(deps, "ADMIN_API_KEY", None):
        with pytest.raises(HTTPException) as exc:
            deps.verify_admin_key("anything")
    assert exc.value.status_code == 401

    with patch.object(deps, "ADMIN_API_KEY", ""):
        with pytest.raises(HTTPException):
            deps.verify_admin_key("")
