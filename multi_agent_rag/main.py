import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from multi_agent_rag.routers import auth, ingestion, retrieval, qa, metrics,evaluation
from multi_agent_rag.core.database import close_db,engine, Base
from multi_agent_rag.services.ingestion_service import LegalIngestionService

logger = logging.getLogger("app")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Legal AI Platform")
    import multi_agent_rag.models.user

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created")

    ingestion_service = LegalIngestionService()
    app.state.ingestion_service = ingestion_service

    try:
        yield
    finally:
        await ingestion_service.close()
        await close_db()

ENV = os.getenv("ENV", "local")

app = FastAPI(
    title="Legal AI Platform",
    lifespan=lifespan,
    docs_url="/docs" if ENV != "production" else None,
    redoc_url="/redoc" if ENV != "production" else None,
    openapi_url="/openapi.json",
)

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )

app.include_router(auth.router)
app.include_router(ingestion.router)
app.include_router(retrieval.router)
app.include_router(qa.router)
app.include_router(metrics.router)
app.include_router(evaluation.router)