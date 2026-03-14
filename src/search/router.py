"""Smart Ayah Search — semantic search over the Quran verses collection."""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.rag.quran_verifier import QuranVerifier

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/search", tags=["Search"])

_verifier: QuranVerifier | None = None


def _get_verifier() -> QuranVerifier:
    global _verifier
    if _verifier is None:
        _verifier = QuranVerifier()
    return _verifier


# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------

class AyahSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    language: str = Field(default="en", pattern="^(en|ar)$")
    top_k: int = Field(default=5, ge=1, le=20)


class AyahResult(BaseModel):
    reference: str       # "2:255"
    surah_num: int
    ayah_num: int
    surah_name: str
    text: str
    relevance: float     # 0–1 (converted from cosine distance)


class AyahSearchResponse(BaseModel):
    query: str
    results: list[AyahResult]


# ------------------------------------------------------------------
# Endpoint
# ------------------------------------------------------------------

@router.post("/ayah", response_model=AyahSearchResponse, summary="Semantic Ayah search")
async def search_ayah(request: AyahSearchRequest) -> Any:
    """Find the most semantically relevant Quran ayahs for a free-text query.

    Searches the ``quran_verses`` ChromaDB collection using cosine similarity.
    Works with queries in Arabic or English — the embedding model is multilingual.
    """
    verifier = _get_verifier()
    if not verifier.is_populated():
        raise HTTPException(
            status_code=503,
            detail="Quran verse collection is empty. Run POST /api/v1/admin/ingest-quran first.",
        )
    try:
        store = verifier._get_store()
        raw = store.similarity_search_with_score(request.query, k=request.top_k)
    except Exception as exc:
        logger.exception("Ayah search failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    results: list[AyahResult] = []
    for doc, distance in raw:
        # Cosine distance ∈ [0, 2]; convert to relevance ∈ [0, 1]
        relevance = round(max(0.0, 1.0 - distance / 2.0), 3)
        meta = doc.metadata
        results.append(AyahResult(
            reference=meta.get("reference", ""),
            surah_num=int(meta.get("chapter", 0)),
            ayah_num=int(meta.get("verse", 0)),
            surah_name=meta.get("surah_name", ""),
            text=doc.page_content,
            relevance=relevance,
        ))

    return AyahSearchResponse(query=request.query, results=results)
