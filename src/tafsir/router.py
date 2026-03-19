"""Tafsir endpoints.

POST /api/v1/tafsir/search
  Accepts an ayah reference (``surah:verse``) **and/or** a free-text query.

  Default lookup order (no OpenAI calls required):
  1. Reference lookup  — exact SQLite query against tafaseer.db.
                         Works immediately after POST /api/v1/admin/ingest-tafsir.
  2. Keyword search    — SQLite LIKE over the ``nass`` column.
                         Default for free-text queries; no embeddings needed.

  Opt-in semantic mode (requires prior ChromaDB ingestion):
  3. Semantic search   — ChromaDB cosine-similarity.
                         Only used when ``semantic=true`` is explicitly passed.

  Returns commentary from Ibn Kathir (ابن كثير) and Al-Tabari (الطبري).

GET /api/v1/tafsir/schema  (admin key required)
  Returns the DB schema, table row counts, and TafseerName mapping.
"""

import logging
import re
from typing import Any

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field, model_validator

from src.config import settings
from src.tafsir.database import (
    ensure_database,
    get_schema_info,
    get_tafsir_for_ayah,
    search_tafsir_text,
)
from src.tafsir.store import TafsirStore

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/tafsir", tags=["Tafsir"])

_store: TafsirStore | None = None
_REF_RE = re.compile(r"^(\d{1,3}):(\d{1,3})$")

_TASHKEEL_RE = re.compile(
    r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]'
)


def strip_tashkeel(text: str) -> str:
    """Remove Arabic diacritical marks (tashkeel/harakat) from text."""
    return _TASHKEEL_RE.sub('', text)


def _get_store() -> TafsirStore:
    global _store
    if _store is None:
        _store = TafsirStore()
    return _store


# ------------------------------------------------------------------
# Pydantic models
# ------------------------------------------------------------------

class TafsirSearchRequest(BaseModel):
    reference: str | None = Field(
        default=None,
        description="Ayah reference in 'surah:verse' format, e.g. '2:255'.",
        examples=["2:255"],
    )
    query: str | None = Field(
        default=None,
        min_length=1,
        max_length=500,
        description="Free-text Arabic or English search query.",
        examples=["ما تفسير آية الكرسي"],
    )
    semantic: bool = Field(
        default=False,
        description=(
            "Set to true to use ChromaDB semantic search for free-text queries. "
            "Requires POST /api/v1/admin/ingest-tafsir-semantic to have been run. "
            "When false (default) SQLite LIKE is used — no OpenAI calls needed."
        ),
    )
    language: str = Field(
        default="en",
        pattern="^(en|ar)$",
        description="Response language hint ('en' or 'ar'). Currently informational.",
    )
    top_k: int = Field(default=5, ge=1, le=20)

    @model_validator(mode="after")
    def _require_at_least_one(self) -> "TafsirSearchRequest":
        if not self.reference and not self.query:
            raise ValueError("Provide at least one of 'reference' or 'query'.")
        return self


class TafsirResult(BaseModel):
    source: str         # "Ibn Kathir"
    source_ar: str      # "ابن كثير"
    reference: str      # "2:255"
    surah: int
    ayah: int
    text: str
    relevance: float    # 1.0 for exact reference lookup; 0–1 for semantic/keyword


class TafsirSearchResponse(BaseModel):
    reference: str | None       # echo of a resolved valid reference
    query: str | None           # echo of the free-text query
    mode: str                   # "reference" | "semantic" | "keyword"
    results: list[TafsirResult]


# ------------------------------------------------------------------
# Endpoint
# ------------------------------------------------------------------

@router.post(
    "/search",
    response_model=TafsirSearchResponse,
    summary="Search Tafsir (Ibn Kathir & Al-Tabari)",
)
async def search_tafsir(request: TafsirSearchRequest) -> Any:
    """Search Quranic commentary from Ibn Kathir and Al-Tabari.

    ### Lookup modes

    | Inputs | ``semantic`` | Mode | Requirements |
    |---|---|---|---|
    | ``reference`` | any | **reference** | SQLite DB downloaded |
    | ``query`` | ``false`` (default) | **keyword** | SQLite DB downloaded |
    | ``query`` | ``true`` | **semantic** | ChromaDB index built |

    ### Setup

    - Run ``POST /api/v1/admin/ingest-tafsir`` once to download the SQLite DB
      (~36 MB). Reference and keyword search work immediately after.
    - Run ``POST /api/v1/admin/ingest-tafsir-semantic`` to build the optional
      ChromaDB index and unlock ``semantic=true``.
    """
    # Ensure the SQLite database is available (lazy download on first call)
    try:
        import asyncio as _asyncio
        loop = _asyncio.get_running_loop()
        await loop.run_in_executor(None, ensure_database)
    except Exception as exc:
        logger.exception("Tafseer database download failed")
        raise HTTPException(
            status_code=503,
            detail=f"Could not load tafseer database: {exc}",
        ) from exc

    results: list[TafsirResult] = []
    resolved_ref: str | None = None
    mode: str = "reference"

    # ----------------------------------------------------------------
    # Mode 1 — exact reference lookup (always SQL, no embeddings)
    # ----------------------------------------------------------------
    if request.reference:
        m = _REF_RE.match(request.reference.strip())
        if not m:
            raise HTTPException(
                status_code=422,
                detail="Invalid reference format. Use 'surah:verse', e.g. '2:255'.",
            )
        surah, ayah = int(m.group(1)), int(m.group(2))
        if not (1 <= surah <= 114):
            raise HTTPException(
                status_code=422,
                detail="Surah number must be between 1 and 114.",
            )
        if ayah < 1:
            raise HTTPException(
                status_code=422,
                detail="Ayah number must be at least 1.",
            )

        resolved_ref = f"{surah}:{ayah}"
        entries = get_tafsir_for_ayah(surah, ayah)

        if not entries:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No tafsir found for {resolved_ref}. "
                    "The database may still be downloading, or the ayah "
                    "number exceeds the surah length."
                ),
            )

        for entry in entries:
            results.append(TafsirResult(
                source=entry["name_en"],
                source_ar=entry["name_ar"],
                reference=resolved_ref,
                surah=surah,
                ayah=ayah,
                text=entry["text"],
                relevance=1.0,
            ))
        mode = "reference"

    # ----------------------------------------------------------------
    # Mode 2 — free-text query (when reference produced no results)
    # ----------------------------------------------------------------
    if request.query and not results:
        if request.semantic:
            # --- Opt-in: ChromaDB semantic search ---
            store = _get_store()
            if not store.is_populated():
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Semantic search index is not built. "
                        "Run POST /api/v1/admin/ingest-tafsir-semantic first, "
                        "or set semantic=false to use keyword search."
                    ),
                )
            mode = "semantic"
            hits = store.search(request.query, top_k=request.top_k)
            for hit in hits:
                results.append(TafsirResult(
                    source=    hit["source"],
                    source_ar= hit["source_ar"],
                    reference= hit["reference"],
                    surah=     hit["surah"],
                    ayah=      hit["ayah"],
                    text=      hit["text"],
                    relevance= hit["relevance"],
                ))
        else:
            # --- Default: SQLite LIKE keyword search (no OpenAI calls) ---
            mode = "keyword"
            clean_query = strip_tashkeel(request.query)
            rows = search_tafsir_text(clean_query, limit=request.top_k)
            for row in rows:
                results.append(TafsirResult(
                    source=    row["name_en"],
                    source_ar= row["name_ar"],
                    reference= row["reference"],
                    surah=     row["surah"],
                    ayah=      row["ayah"],
                    text=      row["text"],
                    relevance= 0.5,   # keyword match — no numeric relevance score
                ))

        if not results:
            hint = (
                "Try a different query, or use semantic=true after running "
                "POST /api/v1/admin/ingest-tafsir-semantic."
                if not request.semantic
                else "Try a different query or a reference lookup."
            )
            raise HTTPException(
                status_code=404,
                detail=f"No tafsir results found for the given query. {hint}",
            )

    return TafsirSearchResponse(
        reference=resolved_ref,
        query=request.query,
        mode=mode,
        results=results,
    )


# ------------------------------------------------------------------
# Debug: schema inspection (admin only)
# ------------------------------------------------------------------

@router.get(
    "/schema",
    summary="Inspect tafaseer.db schema (admin only)",
    tags=["Tafsir"],
)
async def tafsir_schema(
    x_admin_key: str = Header(..., alias="X-Admin-Key"),
) -> Any:
    """Return the full schema of ``tafaseer.db``: table DDL, column names, row
    counts, and the TafseerName ID ↔ book mapping.

    Useful for verifying the database was downloaded and extracted correctly.
    Requires the ``X-Admin-Key`` header.
    """
    if x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=403, detail="Invalid admin key.")
    try:
        ensure_database()
        return get_schema_info()
    except Exception as exc:
        logger.exception("Schema inspection failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
