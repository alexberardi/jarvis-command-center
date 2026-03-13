"""Agent utility endpoints — stateless data fetching for node-side agents.

These endpoints provide heavy-lifting (RSS parsing, calendar fetching) that
Pi Zero nodes offload to the command center. Isolated in their own router
for easy future extraction to a standalone service.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.deps import verify_api_key
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
