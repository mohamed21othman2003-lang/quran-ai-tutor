"""Tajweed tutor agent: GPT-4o + RAG retrieval via LangChain LCEL."""

import json
import logging
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from src.config import settings
from src.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Prompts
# ------------------------------------------------------------------

_QA_SYSTEM = """You are an expert Tajweed teacher with deep knowledge of Quranic recitation rules.
Your role is to help students understand and practice Tajweed correctly.

Rules:
1. Answer ONLY from the provided context. Do not fabricate rules or references.
2. Detect the language of the user's question and respond in the same language \
(Arabic or English).
3. If the answer is not found in the context, say so clearly — do not guess.
4. For Arabic responses, use clear Modern Standard Arabic (فصحى).
5. Keep answers concise and pedagogically structured.

Context:
{context}"""

_QUIZ_SYSTEM = """You are an expert Tajweed teacher. Using ONLY the context below, \
generate a multiple-choice quiz question about the Tajweed rule: {rule}.

Context:
{context}

Return a JSON object with exactly these keys:
- "question": the quiz question (string)
- "options": list of 4 answer strings
- "correct_index": zero-based index of the correct option (integer)
- "explanation": brief explanation of the correct answer (string)

Respond in {language}. Return only valid JSON, no markdown fences."""

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _QA_SYSTEM),
    ("human", "{question}"),
])

QUIZ_PROMPT = ChatPromptTemplate.from_messages([
    ("human", _QUIZ_SYSTEM),
])

_EMPTY_KB_MSG = {
    "en": (
        "The knowledge base is currently empty. "
        "Please run `python -m src.rag.pipeline` to ingest the Tajweed knowledge files, "
        "then restart the server."
    ),
    "ar": (
        "قاعدة البيانات المعرفية فارغة حالياً. "
        "يرجى تشغيل `python -m src.rag.pipeline` لاستيعاب ملفات قواعد التجويد، "
        "ثم إعادة تشغيل الخادم."
    ),
}


# ------------------------------------------------------------------
# Agent
# ------------------------------------------------------------------

class TutorAgent:
    """AI Tajweed tutor backed by GPT-4o and a ChromaDB retriever."""

    def __init__(self, rag: RAGPipeline) -> None:
        self.rag = rag
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.3,
            max_tokens=1024,
            openai_api_key=settings.openai_api_key,
        )

    def _format_docs(self, docs) -> str:
        return "\n\n".join(d.page_content for d in docs)

    def _build_qa_chain(self):
        retriever = self.rag.get_vector_store().as_retriever(
            search_kwargs={"k": settings.top_k}
        )
        chain = (
            {
                "context": retriever | self._format_docs,
                "question": RunnablePassthrough(),
            }
            | QA_PROMPT
            | self.llm
            | StrOutputParser()
        )
        return chain, retriever

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    async def answer(self, question: str, language: str = "en") -> dict[str, Any]:
        """Run the QA chain and return answer + source chunks."""
        if not self.rag.is_populated():
            return {
                "answer": _EMPTY_KB_MSG.get(language, _EMPTY_KB_MSG["en"]),
                "sources": [],
            }

        chain, retriever = self._build_qa_chain()
        docs = await retriever.ainvoke(question)
        sources = [doc.page_content for doc in docs]
        answer_text = await chain.ainvoke(question)
        return {"answer": answer_text, "sources": sources}

    async def generate_quiz(self, rule: str, language: str = "en") -> dict[str, Any]:
        """Generate a multiple-choice quiz for the given Tajweed rule."""
        if not self.rag.is_populated():
            return {
                "question": _EMPTY_KB_MSG.get(language, _EMPTY_KB_MSG["en"]),
                "options": [],
                "correct_index": 0,
                "explanation": "",
            }

        docs = self.rag.retrieve(rule)
        context = "\n\n".join(d.page_content for d in docs)
        lang_label = "Arabic" if language == "ar" else "English"

        chain = QUIZ_PROMPT | self.llm | StrOutputParser()
        raw = await chain.ainvoke(
            {"context": context, "rule": rule, "language": lang_label}
        )

        try:
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            logger.warning("GPT-4o returned non-JSON for quiz; raw: %.200s", raw)
            return {
                "question": raw,
                "options": [],
                "correct_index": 0,
                "explanation": "",
            }
