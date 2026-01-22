import asyncio
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from multi_agent_rag.core.security import get_current_user, require_roles
from multi_agent_rag.core.database import get_db
from multi_agent_rag.core.audit import log_audit
from multi_agent_rag.services.graph import app_graph, GraphState
from multi_agent_rag.core.metrics import REQ_TOTAL, REQ_FAILURE

# ==================================================
# ROUTER
# ==================================================

router = APIRouter(prefix="/qa", tags=["qa"])

# ==================================================
# SCHEMAS
# ==================================================

class LegalSupportResponse(BaseModel):
    point: str
    citation: str


class QAResponse(BaseModel):
    answer: str
    support: List[LegalSupportResponse]
    verdict: str
    feedback: Optional[str] = None
    cached: bool = False
    trace: Optional[List[str]] = None


# ==================================================
# ENDPOINT
# ==================================================

@router.post(
    "",
    response_model=QAResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_roles(["admin", "viewer"]))],
)
async def qa(
    query: str = Query(..., min_length=3, max_length=500),
    debug: bool = Query(False),
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        REQ_TOTAL.inc()

        initial_state: GraphState = {
            "query": query,
            "user_id": user.user_id,
            "answer": None,
            "verdict": "",
            "feedback": "",
            "retries": 0,
            "trace": [],
        }

        final_state = await app_graph.ainvoke(initial_state)

        if not final_state.get("answer"):
            REQ_FAILURE.inc()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No answer produced",
            )

        answer_obj = final_state["answer"]

        await log_audit(
            db,
            user.user_id,
            action="QA_QUERY",
            resource="qa",
            metadata={
                "query": query,
                "verdict": final_state.get("verdict", ""),
                "cached": "cache_hit" in final_state.get("trace", []),
            },
        )

        return QAResponse(
            answer=answer_obj.answer,
            support=[
                LegalSupportResponse(point=s.point, citation=s.citation)
                for s in answer_obj.support
            ],
            verdict=final_state.get("verdict", "unknown"),
            feedback=final_state.get("feedback") if debug else None,
            cached="cache_hit" in final_state.get("trace", []),
            trace=final_state.get("trace") if debug else None,
        )

    except asyncio.TimeoutError:
        REQ_FAILURE.inc()
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="QA processing timed out",
        )

    except HTTPException:
        raise

    except Exception as e:
        REQ_FAILURE.inc()     
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"QA processing failed: {str(e)}" if debug else "QA processing failed",
        )