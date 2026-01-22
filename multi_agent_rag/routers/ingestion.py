import os
import uuid
import logging
import hashlib
from typing import Optional, List

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from multi_agent_rag.core.security import get_current_user, require_roles
from multi_agent_rag.core.audit import log_audit
from multi_agent_rag.core.database import get_db
from multi_agent_rag.services.ingestion_service import LegalIngestionService
from multi_agent_rag.services.ingestion_dedup import check_and_register_document

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = os.getenv("DATA_DIR", "./data")
os.makedirs(DATA_DIR, exist_ok=True)

logger = logging.getLogger("ingestion_api")


def hash_file_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

# ==================================================
# ROUTER
# ==================================================

router = APIRouter(prefix="/ingest", tags=["ingestion"])


# ============================================================
# SCHEMAS
# ============================================================

class IngestResponse(BaseModel):
    job_id: str
    status: str


class BulkIngestResponse(BaseModel):
    batch_id: str
    total_files: int
    jobs: List[IngestResponse]
    skipped: List[dict]
    status: str


class JobStatus(BaseModel):
    job_id: str
    status: str
    documents_processed: int = 0
    chunks_indexed: int = 0
    error: Optional[str] = None
    updated_at: str


class BatchStatus(BaseModel):
    batch_id: str
    total_files: int
    completed: int
    failed: int
    running: int
    jobs: List[JobStatus]


# ============================================================
# SERVICE SINGLETON
# ============================================================

_ingestion_service: Optional[LegalIngestionService] = None


def get_ingestion_service() -> LegalIngestionService:
    global _ingestion_service
    if _ingestion_service is None:
        _ingestion_service = LegalIngestionService()
    return _ingestion_service


# ============================================================
# ENDPOINTS
# ============================================================

@router.post(
    "/pdf",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_roles(["admin", "ingestor"]))],
)
async def ingest_pdf(
    file: UploadFile = File(...),
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    service: LegalIngestionService = Depends(get_ingestion_service),
):
    """Upload and ingest a single PDF file"""

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files allowed",
        )

    if file.content_type not in {
        "application/pdf",
        "application/octet-stream",
    }:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid content type",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file",
        )

    file_hash = hash_file_bytes(file_bytes)

    # --------------------------------------------------
    # Dedup / registration
    # --------------------------------------------------
    is_duplicate, document_id = await check_and_register_document(
        db=db,
        user_id=user.user_id,
        file_hash=file_hash,
        filename=file.filename,
    )

    if is_duplicate:
        await log_audit(
            db,
            user.user_id,
            action="INGEST_DUPLICATE_BLOCKED",
            resource="ingestion",
            metadata={
                "filename": file.filename,
                "file_hash": file_hash,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This document has already been ingested or is currently processing",
        )

    temp_path = os.path.join(DATA_DIR, f"{uuid.uuid4()}.pdf")

    try:
        with open(temp_path, "wb") as f:
            f.write(file_bytes)

        job_id = await service.create_job(
            pdf_path=temp_path,
            user_id=user.user_id,
            document_id=document_id,
        )

        await log_audit(
            db,
            user.user_id,
            action="INGEST_PDF",
            resource="ingestion",
            metadata={
                "filename": file.filename,
                "job_id": job_id,
                "document_id": document_id,
            },
        )

        return IngestResponse(job_id=job_id, status="queued")

    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)

        logger.exception("Ingestion failed to start")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start ingestion job",
        )


@router.post(
    "/bulk",
    response_model=BulkIngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_roles(["admin", "ingestor"]))],
)
async def ingest_bulk(
    files: List[UploadFile] = File(...),
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    service: LegalIngestionService = Depends(get_ingestion_service),
):
    """
    Upload and ingest multiple PDF files at once.
    
    This endpoint accepts multiple files and processes them in parallel.
    Each file is validated, deduplicated, and queued for ingestion.
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided",
        )
    
    batch_id = str(uuid.uuid4())
    jobs = []
    skipped = []
    
    logger.info(f"Starting bulk ingestion batch {batch_id} with {len(files)} files")
    
    for file in files:
        try:

            if not file.filename or not file.filename.lower().endswith(".pdf"):
                skipped.append({
                    "filename": file.filename or "unknown",
                    "reason": "Not a PDF file",
                })
                continue
            
            if file.content_type not in {"application/pdf", "application/octet-stream"}:
                skipped.append({
                    "filename": file.filename,
                    "reason": "Invalid content type",
                })
                continue

            file_bytes = await file.read()
            if not file_bytes:
                skipped.append({
                    "filename": file.filename,
                    "reason": "Empty file",
                })
                continue
            
            file_hash = hash_file_bytes(file_bytes)
            
            is_duplicate, document_id = await check_and_register_document(
                db=db,
                user_id=user.user_id,
                file_hash=file_hash,
                filename=file.filename,
            )
            
            if is_duplicate:
                skipped.append({
                    "filename": file.filename,
                    "reason": "Duplicate - already ingested",
                    "document_id": document_id,
                })
                continue

            temp_path = os.path.join(DATA_DIR, f"{uuid.uuid4()}.pdf")
            
            with open(temp_path, "wb") as f:
                f.write(file_bytes)

            job_id = await service.create_job(
                pdf_path=temp_path,
                user_id=user.user_id,
                document_id=document_id,
            )

            await service.redis.hset(job_id, "batch_id", batch_id)
            
            jobs.append(IngestResponse(
                job_id=job_id,
                status="queued",
            ))
            
            logger.info(f"Queued {file.filename} with job_id {job_id}")
            
        except Exception as e:
            logger.exception(f"Failed to process file {file.filename}")
            skipped.append({
                "filename": file.filename,
                "reason": f"Processing error: {str(e)[:100]}",
            })

    await service.redis.hset(f"batch:{batch_id}", mapping={
        "batch_id": batch_id,
        "user_id": user.user_id,
        "total_files": len(files),
        "successful": len(jobs),
        "skipped": len(skipped),
        "created_at": str(uuid.uuid1().time),
    })

    if jobs:
        job_ids = [job.job_id for job in jobs]
        await service.redis.sadd(f"batch:{batch_id}:jobs", *job_ids)

    await log_audit(
        db,
        user.user_id,
        action="INGEST_BULK",
        resource="ingestion",
        metadata={
            "batch_id": batch_id,
            "total_files": len(files),
            "successful": len(jobs),
            "skipped": len(skipped),
        },
    )
    
    logger.info(
        f"Bulk ingestion batch {batch_id}: "
        f"{len(jobs)} queued, {len(skipped)} skipped"
    )
    
    return BulkIngestResponse(
        batch_id=batch_id,
        total_files=len(files),
        jobs=jobs,
        skipped=skipped,
        status="processing",
    )

@router.get(
    "/batch/{batch_id}",
    response_model=BatchStatus,
    dependencies=[Depends(require_roles(["admin", "viewer", "ingestor"]))],
)
async def get_batch_status(
    batch_id: str,
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    service: LegalIngestionService = Depends(get_ingestion_service),
):
    """Get the status of a bulk ingestion batch"""

    batch_data = await service.redis.hgetall(f"batch:{batch_id}")
    
    if not batch_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Batch not found",
        )

    if batch_data.get("user_id") != user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )

    job_ids = await service.redis.smembers(f"batch:{batch_id}:jobs")

    jobs = []
    completed = 0
    failed = 0
    running = 0
    
    for job_id in job_ids:
        job_data = await service.redis.hgetall(job_id)
        
        if job_data:
            job_status = job_data.get("status", "unknown")
            
            if job_status == "completed":
                completed += 1
            elif job_status == "failed":
                failed += 1
            elif job_status == "running":
                running += 1
            
            jobs.append(JobStatus(
                job_id=job_id,
                status=job_status,
                documents_processed=int(job_data.get("documents_processed", 0)),
                chunks_indexed=int(job_data.get("chunks_indexed", 0)),
                error=job_data.get("error") or None,
                updated_at=job_data.get("updated_at", ""),
            ))
    
    await log_audit(
        db,
        user.user_id,
        action="BATCH_STATUS_CHECK",
        resource="ingestion",
        metadata={"batch_id": batch_id},
    )
    
    return BatchStatus(
        batch_id=batch_id,
        total_files=int(batch_data.get("total_files", len(jobs))),
        completed=completed,
        failed=failed,
        running=running,
        jobs=jobs,
    )


@router.get(
    "/status/{job_id}",
    response_model=JobStatus,
    dependencies=[Depends(require_roles(["admin", "viewer", "ingestor"]))],
)
async def get_job_status(
    job_id: str,
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    service: LegalIngestionService = Depends(get_ingestion_service),
):
    """Get the status of a single ingestion job"""
    data = await service.redis.hgetall(job_id)

    if not data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )

    if data.get("user_id") != user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )

    await log_audit(
        db,
        user.user_id,
        action="INGEST_STATUS_CHECK",
        resource="ingestion",
        metadata={"job_id": job_id},
    )

    return JobStatus(
        job_id=job_id,
        status=data.get("status", "unknown"),
        documents_processed=int(data.get("documents_processed", 0)),
        chunks_indexed=int(data.get("chunks_indexed", 0)),
        error=data.get("error") or None,
        updated_at=data.get("updated_at", ""),
    )