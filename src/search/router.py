"""Smart Ayah Search — semantic search over the Quran verses collection.

Query expansion via GPT-4o-mini translates conceptual queries into Quranic
vocabulary before embedding, dramatically improving recall for Arabic and
English inputs.
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.config import settings
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
# Query expansion
# ------------------------------------------------------------------

async def _expand_query(query: str, language: str) -> str:
    """Translate a conceptual query into Quranic Arabic vocabulary via GPT.

    Arabic embeddings score significantly lower than English ones for the same
    semantic content.  Expanding the query to actual Quranic phrasing before
    embedding closes that gap and drastically improves recall.
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=200,
        openai_api_key=settings.openai_api_key,
    )
    lang_note = "Arabic" if language == "ar" else "English"
    prompt = f"""You are a Quran scholar. A user is searching for Quran verses.

User query: "{query}"
Query language: {lang_note}

Rewrite this query as a short Arabic Quranic phrase that would appear \
directly in the Quran text. Focus on the root words and Quranic vocabulary.
Return ONLY the Arabic search phrase, nothing else. No explanation.
Examples:
- "patience in hardship" \u2192 "\u0627\u0635\u0628\u0631\u0648\u0627 \u0625\u0646 \u0627\u0644\u0644\u0647 \u0645\u0639 \u0627\u0644\u0635\u0627\u0628\u0631\u064a\u0646"
- "\u0627\u0644\u0635\u0628\u0631 \u0641\u064a \u0627\u0644\u0634\u062f\u0629" \u2192 "\u0625\u0646\u0645\u0627 \u064a\u0648\u0641\u0649 \u0627\u0644\u0635\u0627\u0628\u0631\u0648\u0646 \u0623\u062c\u0631\u0647\u0645 \u0628\u063a\u064a\u0631 \u062d\u0633\u0627\u0628"
- "mercy and forgiveness" \u2192 "\u0625\u0646 \u0627\u0644\u0644\u0647 \u063a\u0641\u0648\u0631 \u0631\u062d\u064a\u0645"
- "\u0622\u064a\u0627\u062a \u0627\u0644\u0631\u0632\u0642 \u0648\u0627\u0644\u0645\u0627\u0644" \u2192 "\u0648\u0645\u0627 \u0645\u0646 \u062f\u0627\u0628\u0629 \u0641\u064a \u0627\u0644\u0623\u0631\u0636 \u0625\u0644\u0627 \u0639\u0644\u0649 \u0627\u0644\u0644\u0647 \u0631\u0632\u0642\u0647\u0627"
"""
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        expanded = response.content.strip()
        return expanded if expanded else query
    except Exception:
        # Fall back to the original query if GPT is unavailable
        return query


# ------------------------------------------------------------------
# Endpoint
# ------------------------------------------------------------------

@router.post("/ayah", response_model=AyahSearchResponse,
             summary="Semantic Ayah search")
async def search_ayah(request: AyahSearchRequest) -> Any:
    """Find the most semantically relevant Quran ayahs for a free-text query.

    Searches the ``quran_verses`` ChromaDB collection using cosine similarity.
    The query is first expanded to Quranic vocabulary by GPT-4o-mini to
    improve recall for both Arabic and English inputs.
    """
    verifier = _get_verifier()
    if not verifier.is_populated():
        raise HTTPException(
            status_code=503,
            detail=(
                "Quran verse collection is empty. "
                "Run POST /api/v1/admin/ingest-quran first."
            ),
        )
    try:
        # Expand to Quranic vocabulary before embedding for better recall
        expanded_query = await _expand_query(request.query, request.language)
        logger.info(
            "Ayah search query expanded: '%s' \u2192 '%s'",
            request.query, expanded_query,
        )

        store = verifier._get_store()
        # Fetch more candidates than needed so the relevance filter has room
        raw = store.similarity_search_with_score(
            expanded_query, k=request.top_k * 3
        )
    except Exception as exc:
        logger.exception("Ayah search failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    results: list[AyahResult] = []
    for doc, distance in raw:
        # Cosine distance ∈ [0, 2]; convert to relevance ∈ [0, 1].
        # Threshold is 0.20 (not 0.40) because Arabic embeddings naturally
        # score lower than English for the same semantic match.
        relevance = round(max(0.0, 1.0 - distance / 2.0), 3)
        if relevance < 0.20:
            continue
        meta = doc.metadata
        results.append(AyahResult(
            reference=meta.get("reference", ""),
            surah_num=int(meta.get("chapter", 0)),
            ayah_num=int(meta.get("verse", 0)),
            surah_name=meta.get("surah_name", ""),
            text=doc.page_content,
            relevance=relevance,
        ))
        if len(results) >= request.top_k:
            break

    return AyahSearchResponse(query=request.query, results=results)
