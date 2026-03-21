"""\u0627\u0644\u0642\u0631\u0627\u0621\u0627\u062a \u0627\u0644\u0639\u0634\u0631 \u2014 Ten Qiraat readings API endpoints.

Provides read access to qiraat variant texts stored in the
``qiraat_variants``, ``qurra``, and ``riwayat`` tables.

Admin endpoints for seeding metadata live in src/api/main.py:
  POST /api/v1/admin/ingest-qiraat
"""

import logging

import aiosqlite
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from src.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/qiraat", tags=["Qiraat"])


# ------------------------------------------------------------------
# Pydantic models
# ------------------------------------------------------------------

class QiraatVariant(BaseModel):
    riwaya_code: str
    riwaya_name: str
    verse_text: str
    difference_note: Optional[str] = None


class QiraatVerseResponse(BaseModel):
    verse_key: str
    surah_id: int
    ayah_number: int
    readings: list[QiraatVariant]
    total_readings: int


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@router.get("/verse/{verse_key}", response_model=QiraatVerseResponse)
async def get_verse_readings(verse_key: str):
    """Get all available qiraat readings for a verse (format: surah:ayah)."""
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
            SELECT riwaya_code, riwaya_name, verse_text, difference_note
            FROM   qiraat_variants
            WHERE  verse_key = ?
            ORDER  BY id
            """,
            (verse_key,),
        ) as cursor:
            rows = await cursor.fetchall()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No qiraat variants recorded for verse {verse_key} yet.",
        )

    readings = [QiraatVariant(**dict(row)) for row in rows]

    return QiraatVerseResponse(
        verse_key=verse_key,
        surah_id=surah_id,
        ayah_number=ayah_number,
        readings=readings,
        total_readings=len(readings),
    )


@router.get("/surah/{surah_id}")
async def get_surah_qiraat_summary(surah_id: int):
    """Get all verses in a surah that have recorded qiraat differences."""
    async with aiosqlite.connect(settings.db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT verse_key,
                   ayah_number,
                   COUNT(DISTINCT riwaya_code)  AS riwayat_count,
                   GROUP_CONCAT(DISTINCT riwaya_name) AS riwayat_names
            FROM   qiraat_variants
            WHERE  surah_id = ?
            GROUP  BY verse_key, ayah_number
            ORDER  BY ayah_number
            """,
            (surah_id,),
        ) as cursor:
            rows = await cursor.fetchall()

    return {
        "surah_id": surah_id,
        "verses_with_differences": [dict(row) for row in rows],
        "total": len(rows),
    }


@router.get("/riwayat")
async def get_all_riwayat():
    """Get all available riwayat (transmission chains) seeded in the DB."""
    async with aiosqlite.connect(settings.db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM riwayat ORDER BY id") as cursor:
            rows = await cursor.fetchall()
    return [dict(row) for row in rows]
