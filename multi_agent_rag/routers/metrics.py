from fastapi import APIRouter, Depends, Response, status
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from multi_agent_rag.core.security import require_roles

# ==================================================
# ROUTER
# ==================================================

router = APIRouter(prefix="/metrics", tags=["metrics"])

# ============================================================
# ENDPOINTS
# ============================================================
@router.get(
    "",
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_roles(["admin"]))],
)
def metrics():
    data = generate_latest()
    return Response(
        content=data,
        media_type=CONTENT_TYPE_LATEST,
    )
