"""Agent utility endpoints — stateless data fetching for node-side agents.

These endpoints provide heavy-lifting (RSS parsing, calendar fetching) that
Pi Zero nodes offload to the command center. Isolated in their own router
for easy future extraction to a standalone service.

Also provides admin-only trigger endpoints to force background agent
refreshes on-demand (useful for testing / dev).
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.deps import get_db, verify_api_key, verify_household_role
from app.provisioning import ProvisioningAuthContext, verify_provisioning_auth
from app.services.agent_service import fetch_news_headlines

router = APIRouter(prefix="/agents", tags=["agents"])


class NewsRequest(BaseModel):
    category: str = "general"
    count: int = 5


class NewsResponse(BaseModel):
    articles: list[dict]
    count: int


@router.post("/news", response_model=NewsResponse)
def get_news_for_agent(
    body: NewsRequest,
    _node=Depends(verify_api_key),
) -> NewsResponse:
    """Fetch RSS headlines for a node-side agent."""
    articles = fetch_news_headlines(category=body.category, count=body.count)
    return NewsResponse(articles=articles, count=len(articles))


# ---------------------------------------------------------------------------
# Admin-only: trigger background agent refresh on-demand
# ---------------------------------------------------------------------------


class AgentRefreshRequest(BaseModel):
    household_id: str
    categories: list[str] | None = None  # news only
    count: int = 5                       # news only


class AgentRefreshResponse(BaseModel):
    agent: str
    household_id: str
    memories_injected: int


@router.post("/news/refresh", response_model=AgentRefreshResponse)
def trigger_news_refresh(
    body: AgentRefreshRequest,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> AgentRefreshResponse:
    """Force a news agent refresh for a household.

    Auth: admin API key (any household) or JWT (must belong to household).
    """
    _enforce_household_access(auth, body.household_id)

    from app.services.background_agents import refresh_news_context

    injected = refresh_news_context(
        db=db,
        household_id=body.household_id,
        categories=body.categories or ["general"],
        headlines_per_category=body.count,
    )
    return AgentRefreshResponse(
        agent="news",
        household_id=body.household_id,
        memories_injected=injected,
    )


@router.post("/calendar/refresh", response_model=AgentRefreshResponse)
def trigger_calendar_refresh(
    body: AgentRefreshRequest,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> AgentRefreshResponse:
    """Force a calendar agent refresh for a household.

    Auth: admin API key (any household) or JWT (must belong to household).
    """
    _enforce_household_access(auth, body.household_id)

    from app.services.background_agents import refresh_calendar_context

    injected = refresh_calendar_context(
        db=db,
        household_id=body.household_id,
    )
    return AgentRefreshResponse(
        agent="calendar",
        household_id=body.household_id,
        memories_injected=injected,
    )


def _enforce_household_access(
    auth: ProvisioningAuthContext,
    household_id: str,
) -> None:
    """Verify the caller has access to the specified household.

    Admin key callers are trusted for any household.
    JWT callers must belong to the household (checked via jarvis-auth).
    """
    if auth.auth_type == "admin_key":
        return  # admin can access any household

    if auth.user_id is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=401, detail="Could not determine user identity")

    verify_household_role(
        user_id=auth.user_id,
        household_id=household_id,
        required_role="member",
    )
