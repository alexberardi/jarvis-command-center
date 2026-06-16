"""Self-scoped account-data endpoints (user JWT).

Mounted at ``/api/v0/me``. The ``user_id`` is taken from the validated Bearer
token (``sub``), never from the URL — a caller can only ever purge their own data.

Part of the cross-service account-deletion contract: jarvis-auth orchestrates a
``DELETE /auth/me`` and fans out to ``DELETE /api/v0/me/data`` here (and to the
notifications service) before deleting the user locally.
"""

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.orm import Session

from app.deps import AuthenticatedUser, get_db, verify_user_jwt
from app.services.user_data_service import purge_user_data

router = APIRouter(prefix="/me", tags=["me"])


@router.delete("/data", status_code=status.HTTP_204_NO_CONTENT)
def delete_my_data(
    user: AuthenticatedUser = Depends(verify_user_jwt),
    db: Session = Depends(get_db),
) -> Response:
    """Purge every command-center row keyed to the authenticated user.

    Deletes the caller's memories, conversation transcripts, user-scoped settings,
    and OAuth auth-sessions. Idempotent — a user with no rows still returns 204.
    Other users' rows and household-/node-scoped rows are never touched.
    """
    purge_user_data(db, user.user_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
