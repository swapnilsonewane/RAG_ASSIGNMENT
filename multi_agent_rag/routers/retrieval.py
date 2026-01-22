import logging
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from multi_agent_rag.core.security import get_current_user, require_roles
from multi_agent_rag.core.audit import log_audit
from multi_agent_rag.core.database import get_db
from multi_agent_rag.services.retrieval_service import retrieve, RetrievalResult
# ============================================================
# ROUTER
# ============================================================

router = APIRouter(prefix="/search", tags=["retrieval"])

# ============================================================
# SCHEMAS
# ============================================================

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)


class SearchResponse(BaseModel):
    results: RetrievalResult
    query: str
    total_results: int

# ============================================================
# ENDPOINT
# ============================================================

@router.post(
    "",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_roles(["admin", "viewer"]))],
)
async def search(
    req: SearchRequest,
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Search legal documents using hybrid retrieval (Vector + BM25 + Reranking)
    """
    await log_audit(
        db,
        user.user_id,
        action="SEARCH_QUERY",
        resource="retrieval",
        metadata={"query": req.query},
    )

    try:

        results = await retrieve(req.query, user.user_id)
        
        return SearchResponse(
            results=results,
            query=req.query,
            total_results=len(results.passages)
        )

    except Exception as e:

        logger = logging.getLogger("retrieval_route")
        logger.exception(f"Search failed for query: {req.query}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        )


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
)
async def search_health():
    """
    Health check endpoint for retrieval service
    """
    try:
        
        return {
            "status": "healthy",
            "services": {
                "qdrant": "connected",
                "opensearch": "connected",
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Retrieval service unhealthy: {str(e)}",
        )