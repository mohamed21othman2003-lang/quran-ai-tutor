"""ChromaDB-backed semantic search layer for tafseer content.

Follows the same patterns as ``src/rag/pipeline.py`` and
``src/rag/quran_verifier.py``:
  - Lazy-loaded ``Chroma`` store via ``_get_store()``
  - ``is_populated()`` guard before any search
  - ``build_collection()`` called from the admin endpoint only

Collection: ``tafsir_knowledge``
Metadata keys per document: source, source_ar, surah, ayah, reference

Note: building the collection makes ~12 000 embedding API calls
(6 236 verses × 2 books) — run the admin endpoint once and the results
are persisted to ``data/chroma_db``.
"""

import logging
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from src.config import settings
from src.tafsir.database import iter_all_tafsir

logger = logging.getLogger(__name__)

COLLECTION_NAME = "tafsir_knowledge"

# Tafsir commentary is verbose; chunk to stay within embedding token limits
# while preserving enough context for meaningful similarity search.
_CHUNK_CHARS = 800
_CHUNK_OVERLAP = 80


# ------------------------------------------------------------------
# Text splitting
# ------------------------------------------------------------------

def _split_text(text: str) -> list[str]:
    """Split *text* into overlapping character-level chunks.

    Tries to break on whitespace so words are not cut in the middle.
    Single-chunk texts are returned as-is.
    """
    if len(text) <= _CHUNK_CHARS:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + _CHUNK_CHARS
        if end >= len(text):
            chunks.append(text[start:])
            break
        # Prefer splitting at the last whitespace before `end`
        boundary = text.rfind(" ", start, end)
        if boundary <= start:
            boundary = end
        chunks.append(text[start:boundary])
        start = max(0, boundary - _CHUNK_OVERLAP)

    return [c.strip() for c in chunks if c.strip()]


# ------------------------------------------------------------------
# TafsirStore
# ------------------------------------------------------------------

class TafsirStore:
    """Semantic search over embedded tafseer content via ChromaDB.

    Mirrors the interface of ``QuranVerifier`` for consistency.
    """

    def __init__(self) -> None:
        self._embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=settings.openai_api_key,
        )
        self._store: Chroma | None = None
        Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)

    def _get_store(self) -> Chroma:
        if self._store is None:
            self._store = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=self._embeddings,
                persist_directory=str(Path(settings.chroma_persist_dir)),
            )
        return self._store

    def is_populated(self) -> bool:
        """Return True if the collection contains at least one document."""
        try:
            return self._get_store()._collection.count() > 0
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def build_collection(self) -> int:
        """Embed all tafsir content into the ``tafsir_knowledge`` ChromaDB collection.

        Iterates both Ibn Kathir and Al-Tabari via ``iter_all_tafsir()``,
        chunks long texts, and upserts in batches of 500 to respect the
        OpenAI embedding rate limit.

        Returns the total number of document chunks embedded.
        """
        docs: list[Document] = []
        for row in iter_all_tafsir():
            for chunk in _split_text(row["text"]):
                docs.append(Document(
                    page_content=chunk,
                    metadata={
                        "source":      row["name_en"],
                        "source_ar":   row["name_ar"],
                        "tafseer_id":  row["tafseer_id"],
                        "surah":       row["surah"],
                        "ayah":        row["ayah"],
                        "reference":   row["reference"],
                    },
                ))

        if not docs:
            logger.warning("No tafsir documents found to embed — is the database downloaded?")
            return 0

        logger.info(
            "Embedding %d tafsir chunks into '%s' collection …",
            len(docs), COLLECTION_NAME,
        )

        persist_dir = str(Path(settings.chroma_persist_dir))
        BATCH = 500
        store: Chroma | None = None
        for i in range(0, len(docs), BATCH):
            batch = docs[i : i + BATCH]
            if store is None:
                store = Chroma.from_documents(
                    documents=batch,
                    embedding=self._embeddings,
                    collection_name=COLLECTION_NAME,
                    persist_directory=persist_dir,
                )
            else:
                store.add_documents(batch)
            logger.info(
                "  … embedded %d / %d tafsir chunks",
                min(i + BATCH, len(docs)), len(docs),
            )

        self._store = store
        logger.info("Tafsir collection built: %d chunks.", len(docs))
        return len(docs)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return the top-k most semantically relevant tafsir chunks for *query*.

        Returns an empty list when the collection is not populated or a search
        error occurs — callers must check ``is_populated()`` before calling if
        they want to surface a 503 to the user.

        Each result dict has keys:
          source, source_ar, reference, surah, ayah, text, relevance (0–1)
        """
        if not self.is_populated():
            return []
        try:
            raw = self._get_store().similarity_search_with_score(query, k=top_k)
        except Exception:
            logger.exception("Tafsir semantic search failed for query: %.80s", query)
            return []

        results: list[dict[str, Any]] = []
        for doc, distance in raw:
            # Cosine distance ∈ [0, 2]; map to relevance ∈ [0, 1]
            relevance = round(max(0.0, 1.0 - distance / 2.0), 3)
            meta = doc.metadata
            results.append({
                "source":    meta.get("source", ""),
                "source_ar": meta.get("source_ar", ""),
                "reference": meta.get("reference", ""),
                "surah":     int(meta.get("surah", 0)),
                "ayah":      int(meta.get("ayah", 0)),
                "text":      doc.page_content,
                "relevance": relevance,
            })
        return results

    def retrieve_for_agent(self, query: str, top_k: int = 3) -> list[Document]:
        """Return top-k ``Document`` objects for use inside a LangChain chain.

        Called by ``TutorAgent.answer_tafsir()`` so the existing LCEL
        chain plumbing can be reused unchanged.
        """
        if not self.is_populated():
            return []
        try:
            return self._get_store().similarity_search(query, k=top_k)
        except Exception:
            logger.exception("Tafsir retrieval failed for query: %.80s", query)
            return []
