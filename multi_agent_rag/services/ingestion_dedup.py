import uuid
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from multi_agent_rag.models.document import IngestedDocument


async def check_and_register_document(
    *,
    db: AsyncSession,
    user_id: str,
    file_hash: str,
    filename: str,
) -> tuple[bool, str | None]:
    """
    Deduplication rules:

    - BLOCK if status == COMPLETED
    - BLOCK if status == IN_PROGRESS
    - ALLOW retry if status == FAILED
    """

    result = await db.execute(
        select(IngestedDocument).where(
            IngestedDocument.user_id == user_id,
            IngestedDocument.file_hash == file_hash,
        )
    )
    existing = result.scalar_one_or_none()

    # --------------------------------------------------
    # Existing document logic
    # --------------------------------------------------
    if existing:
        if existing.status == "COMPLETED":
            return True, existing.id

        if existing.status == "IN_PROGRESS":
            return True, existing.id

        if existing.status == "FAILED":

            existing.status = "IN_PROGRESS"
            await db.commit()
            return False, existing.id

        return True, existing.id

    # --------------------------------------------------
    # First-time registration
    # --------------------------------------------------
    try:
        doc = IngestedDocument(
            id=str(uuid.uuid4()),
            user_id=user_id,
            file_hash=file_hash,
            filename=filename,
            status="IN_PROGRESS",
        )

        db.add(doc)
        await db.commit()
        await db.refresh(doc)

        return False, doc.id

    except IntegrityError:
        # --------------------------------------------------
        # Race condition protection
        # --------------------------------------------------
        await db.rollback()

        result = await db.execute(
            select(IngestedDocument).where(
                IngestedDocument.user_id == user_id,
                IngestedDocument.file_hash == file_hash,
            )
        )
        existing = result.scalar_one_or_none()

        if not existing:
            return True, None

        if existing.status == "FAILED":
            existing.status = "IN_PROGRESS"
            await db.commit()
            return False, existing.id

        return True, existing.id
