import os
from datetime import datetime, timedelta, timezone
from typing import Iterable

from jose import jwt, JWTError, ExpiredSignatureError
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from dataclasses import dataclass

# ============================================================
# JWT CONFIG
# ============================================================

SECRET_KEY = os.getenv("JWT_SECRET")
if not SECRET_KEY:
    raise RuntimeError("JWT_SECRET must be set via container environment")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(
    os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30")
)

TOKEN_TYPE = "access"

# ============================================================
# BEARER SCHEME
# ============================================================

bearer_scheme = HTTPBearer(auto_error=False)

@dataclass(frozen=True)
class CurrentUser:
    user_id: str
    role: str

# ============================================================
# TOKEN CREATION
# ============================================================

def create_access_token(*, sub: str, role: str) -> str:
    now = datetime.now(tz=timezone.utc)

    payload = {
        "sub": sub,
        "role": role,
        "type": TOKEN_TYPE,
        "iat": int(now.timestamp()),
        "nbf": int(now.timestamp()),
        "exp": int(
            (now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)).timestamp()
        ),
    }

    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

# ============================================================
# CURRENT USER
# ============================================================

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> CurrentUser:
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing authorization token")

    try:
        payload = jwt.decode(
            credentials.credentials,
            SECRET_KEY,
            algorithms=[ALGORITHM],
            options={"require": ["exp", "sub", "iat", "nbf"]},
        )
    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    if payload.get("type") != TOKEN_TYPE:
        raise HTTPException(status_code=401, detail="Invalid token type")

    user_id = payload.get("sub")
    role = payload.get("role")

    if not user_id or not role:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    return CurrentUser(user_id=user_id, role=role)

# ============================================================
# ROLE ENFORCEMENT
# ============================================================

def require_roles(roles: Iterable[str]):
    allowed = set(roles)

    async def checker(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if user.role not in allowed:
            raise HTTPException(status_code=403, detail="Forbidden")
        return user

    return checker
