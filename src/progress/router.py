"""Progress tracking endpoints."""

import logging
from collections import defaultdict
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from pydantic import BaseModel, Field

from src.auth.database import get_db
from src.auth.jwt import decode_token

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/progress", tags=["progress"])
bearer = HTTPBearer()


# ── Auth dependency ────────────────────────────────────────────

async def current_user(
    creds: HTTPAuthorizationCredentials = Security(bearer),
) -> dict:
    try:
        return decode_token(creds.credentials)
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")


# ── Pydantic models ────────────────────────────────────────────

class ProgressIn(BaseModel):
    user_id: int
    rule_name: str = Field(..., min_length=1, max_length=200)
    score: float = Field(..., ge=0.0, le=1.0)
    type: str = Field(..., pattern="^(chat|quiz)$")


class ProgressEntry(BaseModel):
    id: int
    rule_name: str
    score: float
    type: str
    created_at: str


class WeakRule(BaseModel):
    rule_name: str
    avg_score: float
    attempts: int


class ProgressResponse(BaseModel):
    user_id: int
    username: str
    history: list[ProgressEntry]
    weak_rules: list[WeakRule]
    total_sessions: int


# ── Endpoints ──────────────────────────────────────────────────

@router.post("", status_code=201)
async def save_progress(
    body: ProgressIn,
    user: dict = Depends(current_user),
) -> Any:
    """Save a learning event (chat interaction or quiz answer) for a user."""
    # Token must belong to the user_id being written
    if str(body.user_id) != user["sub"]:
        raise HTTPException(status_code=403, detail="Cannot write progress for another user.")

    db = await get_db()
    try:
        await db.execute(
            "INSERT INTO progress (user_id, rule_name, score, type) VALUES (?, ?, ?, ?)",
            (body.user_id, body.rule_name, body.score, body.type),
        )
        await db.commit()
    finally:
        await db.close()

    return {"status": "saved"}


@router.get("/{user_id}", response_model=ProgressResponse)
async def get_progress(
    user_id: int,
    user: dict = Depends(current_user),
) -> Any:
    """Return a user's full learning history and computed weak rules."""
    if str(user_id) != user["sub"]:
        raise HTTPException(status_code=403, detail="Cannot read another user's progress.")

    db = await get_db()
    try:
        rows = await (
            await db.execute(
                """SELECT id, rule_name, score, type, created_at
                   FROM progress WHERE user_id = ?
                   ORDER BY created_at DESC""",
                (user_id,),
            )
        ).fetchall()
        user_row = await (
            await db.execute("SELECT username FROM users WHERE id = ?", (user_id,))
        ).fetchone()
    finally:
        await db.close()

    if not user_row:
        raise HTTPException(status_code=404, detail="User not found.")

    history = [ProgressEntry(**dict(r)) for r in rows]

    # Aggregate per rule: avg score + attempt count
    rule_stats: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        rule_stats[r["rule_name"]].append(r["score"])

    weak_rules = sorted(
        [
            WeakRule(
                rule_name=rule,
                avg_score=round(sum(scores) / len(scores), 3),
                attempts=len(scores),
            )
            for rule, scores in rule_stats.items()
        ],
        key=lambda w: w.avg_score,  # lowest average score first
    )

    return ProgressResponse(
        user_id=user_id,
        username=user_row["username"],
        history=history,
        weak_rules=weak_rules,
        total_sessions=len(rows),
    )
