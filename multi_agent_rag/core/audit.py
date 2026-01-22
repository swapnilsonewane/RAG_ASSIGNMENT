import uuid
import json
import logging
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession

from multi_agent_rag.models.audit import AuditLog


logger = logging.getLogger("audit")

MAX_METADATA_BYTES = 8 * 1024  


def _sanitize_metadata(metadata: Optional[dict]) -> dict:
    if not metadata:
        return {}

    try:
        raw = json.dumps(metadata)
        if len(raw.encode()) <= MAX_METADATA_BYTES:
            return metadata
    except Exception:
        pass

    return {"_truncated": True}


async def log_audit(
    db: AsyncSession,
    user_id: str,
    action: str,
    resource: str,
    metadata: Optional[dict] = None,
) -> None:
    try:
        log = AuditLog(
            id=str(uuid.uuid4()),
            user_id=user_id,
            action=action,
            resource=resource,
            metadata=_sanitize_metadata(metadata),
        )

        db.add(log)
        await db.flush()

    except Exception:
        logger.exception("Audit logging failed")
        await db.rollback()
