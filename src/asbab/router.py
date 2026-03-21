"""Asbab al-Nuzul (\u0623\u0633\u0628\u0627\u0628 \u0627\u0644\u0646\u0632\u0648\u0644) API endpoints.

Provides read access to revelation-context entries stored in the
``asbab_nuzul`` and ``asbab_sources`` tables of the main SQLite DB.

Admin endpoints for seeding data live in src/api/main.py:
  POST /api/v1/admin/ingest-asbab
"""

import logging

import aiosqlite
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from src.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/asbab", tags=["Asbab al-Nuzul"])


# ------------------------------------------------------------------
# Pydantic models
# ------------------------------------------------------------------

class AsbabEntry(BaseModel):
    id: int
    verse_key: str
    sabab_text: str
    source_name: Optional[str] = None
    author_arabic: Optional[str] = None
    isnad: Optional[str] = None
    authenticity_grade: Optional[str] = None
    revelation_period: Optional[str] = None


class AsbabResponse(BaseModel):
    verse_key: str
    surah_id: int
    ayah_number: int
    entries: list[AsbabEntry]
    total: int


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@router.get("/verse/{verse_key}", response_model=AsbabResponse)
async def get_verse_asbab(verse_key: str):
    """Get all asbab al-nuzul for a specific verse (format: surah:ayah)."""
    try:
        surah_str, ayah_str = verse_key.split(":", 1)
        surah_id = int(surah_str)
        ayah_number = int(ayah_str)
    except (ValueError, AttributeError):
        raise HTTPException(
            status_code=400,
            detail="Invalid verse key. Use format surah:ayah (e.g. 2:255)",
        )

    async with aiosqlite.connect(settings.db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT a.id, a.verse_key, a.sabab_text, a.isnad,
                   a.authenticity_grade, a.grading_scholar,
                   a.revelation_period,
                   s.name_arabic AS source_name,
                   s.author_arabic
            FROM   asbab_nuzul  a
            LEFT JOIN asbab_sources s ON a.source_id = s.id
            WHERE  a.verse_key = ?
            ORDER  BY a.id
            """,
            (verse_key,),
        ) as cursor:
            rows = await cursor.fetchall()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No asbab al-nuzul found for verse {verse_key}.",
        )

    entries = [
        AsbabEntry(
            id=row["id"],
            verse_key=row["verse_key"],
            sabab_text=row["sabab_text"],
            source_name=row["source_name"],
            author_arabic=row["author_arabic"],
            isnad=row["isnad"],
            authenticity_grade=row["authenticity_grade"],
            revelation_period=row["revelation_period"],
        )
        for row in rows
    ]

    return AsbabResponse(
        verse_key=verse_key,
        surah_id=surah_id,
        ayah_number=ayah_number,
        entries=entries,
        total=len(entries),
    )


@router.get("/surah/{surah_id}")
async def get_surah_asbab_summary(surah_id: int):
    """Get count of asbab al-nuzul entries per ayah for an entire surah."""
    async with aiosqlite.connect(settings.db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT verse_key, ayah_number, COUNT(*) AS asbab_count
            FROM   asbab_nuzul
            WHERE  surah_id = ?
            GROUP  BY verse_key, ayah_number
            ORDER  BY ayah_number
            """,
            (surah_id,),
        ) as cursor:
            rows = await cursor.fetchall()

    total = sum(row["asbab_count"] for row in rows)
    return {
        "surah_id": surah_id,
        "verses_with_asbab": [dict(row) for row in rows],
        "total_entries": total,
    }


@router.get("/sources")
async def get_asbab_sources():
    """List all asbab al-nuzul source books seeded in the database."""
    async with aiosqlite.connect(settings.db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM asbab_sources ORDER BY id") as cursor:
            rows = await cursor.fetchall()
    return [dict(row) for row in rows]
