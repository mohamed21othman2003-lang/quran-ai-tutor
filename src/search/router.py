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

async def _expand_query(query: str, _language: str) -> str:
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
    prompt = f"""You are a Quran scholar helping with semantic search.

User query: "{query}"

Your task: rewrite this query into Arabic words/phrases that would
most likely appear in the ACTUAL TEXT of the Quran, to maximize
semantic search accuracy.

Rules:
1. If the query is about a Quranic event or battle (غزوة بدر, هجرة,
   فتح مكة etc.) → output the Surah name and key Arabic words from
   that context (e.g. for بدر → "يوم التقى الجمعان سورة الأنفال")
2. If the query is about a concept (الصبر, الرزق, الرحمة) → output
   Quranic vocabulary for that concept
3. If the query is a direct Arabic Quranic phrase → return it as-is
4. If the query is in English → translate to Quranic Arabic vocabulary
5. NEVER output abstract or theological expansions unrelated to the
   original query intent

Return ONLY the Arabic search phrase. No explanation. No punctuation
beyond what appears in the Quran.

Examples:
- "غزوة بدر" → "يوم التقى الجمعان الأنفال إذ يريكموهم الله"
- "battle of badr" → "يوم التقى الجمعان الأنفال إذ يريكموهم الله"
- "الصبر في الشدة" → "اصبروا إن الله مع الصابرين"
- "mercy forgiveness" → "إن الله غفور رحيم وسع كل شيء رحمة"
- "آيات الرزق" → "وما من دابة في الأرض إلا على الله رزقها"
- "يوم القيامة" → "يوم القيامة"
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
