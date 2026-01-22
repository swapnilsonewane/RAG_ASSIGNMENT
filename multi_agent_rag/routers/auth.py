import os
import uuid
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from passlib.context import CryptContext

from multi_agent_rag.core.database import get_db
from multi_agent_rag.core.security import create_access_token
from multi_agent_rag.models.user import User


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

ALLOWED_ROLES = {"viewer", "ingestor", "admin"}

ENABLE_TEST_ROLE_OVERRIDE = (
    os.getenv("ENABLE_TEST_ROLE_OVERRIDE", "false").lower() == "true"
)
ALLOWED_TEST_ROLES = {"viewer", "ingestor", "admin"}

if ENABLE_TEST_ROLE_OVERRIDE:
    logger.warning("⚠️ TEST ROLE OVERRIDE ENABLED — DO NOT USE IN PRODUCTION")

# ============================================================
# SCHEMAS
# ============================================================
from enum import Enum

class UserRole(str, Enum):
    admin = "admin"
    viewer = "viewer"
    ingestor = "ingestor"

class RegisterRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=255)
    password: str = Field(..., min_length=8, max_length=128)
    role: UserRole


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=255)
    password: str
    test_role: str | None = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

# ============================================================
# ROUTES
# ============================================================

@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(
    req: RegisterRequest,
    db: AsyncSession = Depends(get_db),
):
    email = req.email.lower().strip()
    role = req.role.lower().strip()

    if role not in ALLOWED_ROLES:
        raise HTTPException(status_code=400, detail="Invalid role")

    user = User(
        user_id=str(uuid.uuid4()),
        email=email,
        role=role,
        hashed_password=pwd_context.hash(req.password),
    )

    try:
        db.add(user)
        await db.commit()
        return {"status": "created", "role": role}

    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=409, detail="Email already registered")

    except Exception:
        await db.rollback()
        logger.exception("User registration failed")
        raise HTTPException(status_code=500, detail="Registration failed")


@router.post("/login", response_model=TokenResponse)
async def login(
    req: LoginRequest,
    db: AsyncSession = Depends(get_db),
):
    email = req.email.lower().strip()

    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()

    if not user or not pwd_context.verify(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    role = user.role

    if ENABLE_TEST_ROLE_OVERRIDE and req.test_role:
        if req.test_role not in ALLOWED_TEST_ROLES:
            raise HTTPException(status_code=400, detail="Invalid test role")
        role = req.test_role

    token = create_access_token(sub=user.user_id, role=role)
    return TokenResponse(access_token=token)
