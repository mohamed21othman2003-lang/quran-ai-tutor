"""Tajweed Auto-Detect — identify Tajweed rules in arbitrary Arabic text using GPT-4o + RAG."""

import json
import logging
from typing import Any

from fastapi import HTTPException
from fastapi.routing import APIRouter
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.config import settings
from src.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/tajweed", tags=["Tajweed"])

_llm: ChatOpenAI | None = None
_rag: RAGPipeline | None = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.2,
            max_tokens=1024,
            openai_api_key=settings.openai_api_key,
        )
    return _llm


def _get_rag() -> RAGPipeline:
    global _rag
    if _rag is None:
        _rag = RAGPipeline()
    return _rag


# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------

class DetectRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    language: str = Field(default="en", pattern="^(en|ar)$")


class DetectedRule(BaseModel):
    rule_name: str
    arabic_name: str
    explanation: str
    location: str      # word or phrase in the text where the rule applies


class DetectResponse(BaseModel):
    text: str
    rules: list[DetectedRule]
    summary: str


# ------------------------------------------------------------------
# Prompt
# ------------------------------------------------------------------

_DETECT_PROMPT = """\
You are an expert Tajweed teacher. Analyze the Arabic text below and identify every \
Tajweed rule that applies to it.

Arabic text:
{text}

Tajweed reference context:
{context}

Instructions:
1. Identify ALL Tajweed rules present (Idgham, Ikhfa, Iqlab, Izhar, Madd variants, \
   Qalqala, Ghunna, Waqf rules, etc.).
2. For each rule give:
   - rule_name: English name
   - arabic_name: Arabic name
   - explanation: 1-2 sentence explanation of the rule as it applies here
   - location: the specific word(s) or letter(s) in the text where it occurs
3. Respond in {language}.
4. Return a JSON object with exactly these keys:
   - "rules": array of rule objects (may be empty if no rules detected)
   - "summary": one sentence overall summary of the Tajweed characteristics

Return only valid JSON, no markdown fences."""


# ------------------------------------------------------------------
# Endpoint
# ------------------------------------------------------------------

@router.post("/detect", response_model=DetectResponse, summary="Detect Tajweed rules in text")
async def detect_rules(request: DetectRequest) -> Any:
    """Identify all Tajweed rules present in the provided Arabic text.

    Retrieves relevant Tajweed rule documentation via RAG, then asks GPT-4o to
    analyse the text and return a structured list of applicable rules.
    """
    rag = _get_rag()
    if not rag.is_populated():
        raise HTTPException(
            status_code=503,
            detail="Knowledge base is empty. Run POST /api/v1/admin/ingest first.",
        )

    # Retrieve Tajweed rule context relevant to this text
    docs = rag.retrieve(request.text)
    context = "\n\n".join(d.page_content for d in docs) if docs else "No specific context found."
    lang_label = "Arabic" if request.language == "ar" else "English"

    prompt = _DETECT_PROMPT.format(
        text=request.text,
        context=context,
        language=lang_label,
    )

    try:
        llm = _get_llm()
        from langchain_core.messages import HumanMessage
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
    except Exception as exc:
        logger.exception("LLM call failed in /tajweed/detect")
        raise HTTPException(status_code=500, detail=f"AI error: {exc}") from exc

    try:
        data = json.loads(raw)
        rules = [DetectedRule(**r) for r in data.get("rules", [])]
        return DetectResponse(
            text=request.text,
            rules=rules,
            summary=data.get("summary", ""),
        )
    except (json.JSONDecodeError, Exception) as exc:
        logger.warning("GPT-4o returned non-JSON for detect: %.300s", raw)
        # Return the raw summary as fallback
        return DetectResponse(
            text=request.text,
            rules=[],
            summary=raw[:500],
        )
