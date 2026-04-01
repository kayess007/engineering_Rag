"""
Phase 10 — JWT Authentication
Usage:
  - POST /auth/token  with form fields username + password  →  returns access_token
  - Protected routes use: token: str = Depends(require_auth)

Configure credentials and secret via environment variables:
  RAG_USERNAME   (default: admin)
  RAG_PASSWORD   (default: changeme)
  RAG_JWT_SECRET (default: insecure-dev-secret — MUST be overridden in production)
  RAG_JWT_EXPIRE_MINUTES (default: 60)
"""

import logging
import os
from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import jwt

logger = logging.getLogger("rag.auth")

_ALGORITHM = "HS256"

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


def _secret() -> str:
    return os.getenv("RAG_JWT_SECRET", "insecure-dev-secret")


def _expire_minutes() -> int:
    return int(os.getenv("RAG_JWT_EXPIRE_MINUTES", "60"))


def _valid_credentials(username: str, password: str) -> bool:
    expected_user = os.getenv("RAG_USERNAME", "admin")
    expected_pass = os.getenv("RAG_PASSWORD", "changeme")
    return username == expected_user and password == expected_pass


def create_access_token(username: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=_expire_minutes())
    payload = {"sub": username, "exp": expire}
    return jwt.encode(payload, _secret(), algorithm=_ALGORITHM)


def require_auth(token: str = Depends(oauth2_scheme)) -> str:
    """FastAPI dependency — raises 401 if token is missing or invalid."""
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, _secret(), algorithms=[_ALGORITHM])
        username: str = payload.get("sub", "")
        if not username:
            raise credentials_exc
        return username
    except jwt.ExpiredSignatureError:
        logger.warning("Expired JWT token")
        raise credentials_exc
    except jwt.InvalidTokenError:
        logger.warning("Invalid JWT token")
        raise credentials_exc


def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Route handler for POST /auth/token."""
    if not _valid_credentials(form_data.username, form_data.password):
        logger.warning("Failed login attempt", extra={"username": form_data.username})
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(form_data.username)
    logger.info("Login successful", extra={"username": form_data.username})
    return {"access_token": token, "token_type": "bearer"}
