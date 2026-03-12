"""Auth endpoints: register and login."""

import logging

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.auth.database import get_db
from src.auth.jwt import create_token, hash_password, verify_password

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


# ── Pydantic models ────────────────────────────────────────────

class AuthRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6, max_length=128)


class AuthResponse(BaseModel):
    token: str
    user_id: int
    username: str


# ── Endpoints ──────────────────────────────────────────────────

@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def register(body: AuthRequest):
    """Create a new user account and return a JWT."""
    db = await get_db()
    try:
        existing = await db.execute(
            "SELECT id FROM users WHERE username = ?", (body.username,)
        )
        if await existing.fetchone():
            raise HTTPException(status_code=409, detail="Username already taken.")

        hashed = hash_password(body.password)
        cursor = await db.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (body.username, hashed),
        )
        await db.commit()
        user_id = cursor.lastrowid
    finally:
        await db.close()

    token = create_token(user_id, body.username)
    logger.info("Registered user %s (id=%s)", body.username, user_id)
    return AuthResponse(token=token, user_id=user_id, username=body.username)


@router.post("/login", response_model=AuthResponse)
async def login(body: AuthRequest):
    """Authenticate and return a JWT."""
    db = await get_db()
    try:
        row = await (
            await db.execute(
                "SELECT id, username, password FROM users WHERE username = ?",
                (body.username,),
            )
        ).fetchone()
    finally:
        await db.close()

    if not row or not verify_password(body.password, row["password"]):
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    token = create_token(row["id"], row["username"])
    logger.info("Login: user %s (id=%s)", row["username"], row["id"])
    return AuthResponse(token=token, user_id=row["id"], username=row["username"])
