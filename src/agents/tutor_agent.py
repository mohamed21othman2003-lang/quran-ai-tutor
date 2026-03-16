"""Tajweed tutor agent: GPT-4o + RAG retrieval via LangChain LCEL.

Provides three async public methods:
  - answer()          — Tajweed Q&A grounded in the knowledge base
  - generate_quiz()   — Multiple-choice quiz for a given Tajweed rule
  - answer_tafsir()   — Quranic commentary Q&A using TafsirStore context
"""

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

_TAFSIR_QA_SYSTEM = """You are a knowledgeable Islamic scholar specialising in Quranic exegesis (tafsir).
Your role is to explain the meaning and commentary of Quran verses clearly and accurately.

Rules:
1. Base your answer ONLY on the tafsir context provided below.
2. Identify the source scholar when quoting (Ibn Kathir or Al-Tabari).
3. Respond in {language}.
4. If the context does not address the question, say so clearly — do not fabricate interpretations.
5. Use respectful, scholarly language appropriate for Islamic scholarship.

Tafsir context:
{context}"""

TAFSIR_QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _TAFSIR_QA_SYSTEM),
    ("human", "{question}"),
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

_EMPTY_TAFSIR_MSG = {
    "en": (
        "The tafsir index is not available. "
        "Run POST /api/v1/admin/ingest-tafsir to build the semantic index, "
        "or use POST /api/v1/tafsir/search with a 'reference' for direct lookup."
    ),
    "ar": (
        "فهرس التفسير غير متاح حالياً. "
        "قم بتشغيل POST /api/v1/admin/ingest-tafsir لبناء الفهرس الدلالي، "
        "أو استخدم POST /api/v1/tafsir/search مع حقل 'reference' للبحث المباشر."
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

    async def answer_tafsir(
        self,
        question: str,
        language: str = "en",
        top_k: int = 4,
    ) -> dict[str, Any]:
        """Answer a tafsir question grounded in Ibn Kathir and Al-Tabari context.

        Retrieves the most relevant tafsir chunks from the ``tafsir_knowledge``
        ChromaDB collection, then asks GPT-4o to synthesise a scholarly answer.

        Returns the same ``{"answer": str, "sources": list[str]}`` shape as
        ``answer()`` so callers can treat both methods uniformly.
        """
        # Import lazily to avoid a hard startup dependency on TafsirStore
        from src.tafsir.store import TafsirStore

        store = TafsirStore()
        if not store.is_populated():
            return {
                "answer": _EMPTY_TAFSIR_MSG.get(language, _EMPTY_TAFSIR_MSG["en"]),
                "sources": [],
            }

        docs = store.retrieve_for_agent(question, top_k=top_k)
        context = "\n\n".join(
            f"[{doc.metadata.get('source', '')} — {doc.metadata.get('reference', '')}]\n"
            f"{doc.page_content}"
            for doc in docs
        )
        sources = [
            f"{doc.metadata.get('source', '')} ({doc.metadata.get('reference', '')})"
            for doc in docs
        ]

        lang_label = "Arabic" if language == "ar" else "English"
        chain = TAFSIR_QA_PROMPT | self.llm | StrOutputParser()
        answer_text = await chain.ainvoke({
            "context":  context,
            "question": question,
            "language": lang_label,
        })
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
