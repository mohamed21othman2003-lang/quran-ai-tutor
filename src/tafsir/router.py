"""Tafsir search endpoint.

POST /api/v1/tafsir/search
  Accepts an ayah reference (``surah:verse``) **or** a free-text query, or both.

  Lookup modes (evaluated in order):
  1. Reference lookup  — exact SQLite query against the downloaded tafseer DBs.
                         Always available once the DB is downloaded (~36 MB, lazy).
  2. Semantic search   — ChromaDB cosine-similarity search.
                         Requires prior ingestion via POST /api/v1/admin/ingest-tafsir.
  3. Keyword fallback  — SQLite LIKE search used only when semantic store is empty
                         and a free-text query was given.

  Returns commentary from Ibn Kathir (ابن كثير) and Al-Tabari (الطبري).
"""

import logging
import re
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

from src.tafsir.database import (
    TAFSIR_SOURCES,
    ensure_databases,
    get_tafsir_for_ayah,
    search_tafsir_text,
)
from src.tafsir.store import TafsirStore

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/tafsir", tags=["Tafsir"])

_store: TafsirStore | None = None
_REF_RE = re.compile(r"^(\d{1,3}):(\d{1,3})$")

# Arabic names keyed by the English name returned by database.py
_AR_NAMES: dict[str, str] = {
    meta["name_en"]: meta["name_ar"] for meta in TAFSIR_SOURCES.values()
}


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

    | Inputs supplied | Mode used | Notes |
    |---|---|---|
    | ``reference`` only | **Reference** | Exact SQLite lookup — fast, always available |
    | ``query`` only (semantic store populated) | **Semantic** | ChromaDB cosine search |
    | ``query`` only (semantic store empty) | **Keyword** | SQLite LIKE fallback |
    | Both ``reference`` + ``query`` | **Reference** | Query is ignored when a valid reference resolves results |

    ### One-time setup

    - **Reference mode** works immediately; the tafseer databases (~36 MB) are
      downloaded automatically on the first request.
    - **Semantic mode** requires running ``POST /api/v1/admin/ingest-tafsir``
      once to build the ChromaDB index (~12 000 embeddings, a few minutes).
    """
    # Ensure the SQLite databases are available (lazy download on first call)
    try:
        ensure_databases()
    except Exception as exc:
        logger.exception("Tafseer database download failed")
        raise HTTPException(
            status_code=503,
            detail=f"Could not load tafseer databases: {exc}",
        ) from exc

    results: list[TafsirResult] = []
    resolved_ref: str | None = None
    mode: str = "reference"

    # ----------------------------------------------------------------
    # Mode 1 — exact reference lookup
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
        tafsir_map = get_tafsir_for_ayah(surah, ayah)

        if not tafsir_map:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No tafsir found for {resolved_ref}. "
                    "The tafseer databases may still be downloading, or the ayah "
                    "number exceeds the surah length."
                ),
            )

        for _stem, entry in tafsir_map.items():
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
    # Mode 2 — semantic or keyword search (when no reference resolved results)
    # ----------------------------------------------------------------
    if request.query and not results:
        store = _get_store()

        if store.is_populated():
            # Semantic search via ChromaDB
            mode = "semantic"
            hits = store.search(request.query, top_k=request.top_k)
            for hit in hits:
                results.append(TafsirResult(
                    source=hit["source"],
                    source_ar=hit["source_ar"],
                    reference=hit["reference"],
                    surah=hit["surah"],
                    ayah=hit["ayah"],
                    text=hit["text"],
                    relevance=hit["relevance"],
                ))
        else:
            # Keyword fallback via SQLite LIKE — always available without ingestion
            mode = "keyword"
            logger.info(
                "Tafsir semantic store not populated; falling back to LIKE search "
                "for query: %.80s", request.query,
            )
            rows = search_tafsir_text(request.query, limit=request.top_k)
            for row in rows:
                results.append(TafsirResult(
                    source=row["name_en"],
                    source_ar=row["name_ar"],
                    reference=row["reference"],
                    surah=row["surah"],
                    ayah=row["ayah"],
                    text=row["text"],
                    relevance=0.5,  # keyword match has no relevance score
                ))

        if not results:
            raise HTTPException(
                status_code=404,
                detail=(
                    "No tafsir results found for the given query. "
                    "Try a different query, or run POST /api/v1/admin/ingest-tafsir "
                    "to enable semantic search."
                ),
            )

    return TafsirSearchResponse(
        reference=resolved_ref,
        query=request.query,
        mode=mode,
        results=results,
    )
