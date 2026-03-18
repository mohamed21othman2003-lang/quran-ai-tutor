"""FAISS-backed semantic search layer for tafseer content.

Replaces the earlier ChromaDB implementation which suffered from persistent
"no such table: tenants/acquire_write" errors on Railway when creating a
fresh database after a wipe.

FAISS advantages over ChromaDB for this use-case:
  - No SQLite / WAL / migration issues — just two files (index.faiss + index.pkl)
  - Clean overwrite: delete files, write new ones, done.
  - Fast in-memory search; index loaded once per process.

Index is stored under ``settings.tafsir_chroma_dir`` (reusing the same env var
so no config change is needed):
  <tafsir_chroma_dir>/
    index.faiss   — FAISS vector index
    index.pkl     — serialised docstore + id_to_docstore_id mapping

Collection: ``tafsir_knowledge``
Metadata keys per document: source, source_ar, tafseer_id, surah, ayah, reference

Ingestion scope: Ibn Kathir (ID=2) + Al-Tabari (ID=1) only = ~12 472 rows.
"""

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from src.config import settings
from src.tafsir.database import iter_all_tafsir

logger = logging.getLogger(__name__)

# --- Rate-limit-safe ingestion tunables ---
EMBED_BATCH = 200          # documents sent to OpenAI per embedding call
BATCH_SLEEP_S = 2          # seconds to pause between every batch
RATE_LIMIT_SLEEP_S = 60    # seconds to back off after an HTTP 429
PROGRESS_LOG_EVERY = 1000  # emit a progress line every N documents embedded

# Tafsir commentary is verbose; chunk to stay within embedding token limits
# while preserving enough context for meaningful similarity search.
_CHUNK_CHARS = 800
_CHUNK_OVERLAP = 80


# ------------------------------------------------------------------
# Text splitting
# ------------------------------------------------------------------

def _split_text(text: str) -> list[str]:
    """Split *text* into overlapping character-level chunks."""
    if len(text) <= _CHUNK_CHARS:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + _CHUNK_CHARS
        if end >= len(text):
            chunks.append(text[start:])
            break
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
    """FAISS-backed semantic search over embedded tafseer content."""

    _INDEX_NAME = "index"   # FAISS saves <name>.faiss + <name>.pkl

    def __init__(self) -> None:
        # Use 256-dim embeddings — 6× smaller than the default 1536 dims.
        self._embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            dimensions=256,
            openai_api_key=settings.openai_api_key,
        )
        self._index_dir = os.environ.get("TAFSIR_CHROMA_DIR", settings.tafsir_chroma_dir)
        os.makedirs(self._index_dir, exist_ok=True)
        self._store = None   # lazy-loaded FAISS VectorStore

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _index_path(self) -> Path:
        return Path(self._index_dir)

    def _faiss_file(self) -> Path:
        return self._index_path() / f"{self._INDEX_NAME}.faiss"

    def _pkl_file(self) -> Path:
        return self._index_path() / f"{self._INDEX_NAME}.pkl"

    def _get_store(self):
        """Return the loaded FAISS store (loads from disk on first call)."""
        if self._store is None:
            from langchain_community.vectorstores import FAISS
            self._store = FAISS.load_local(
                folder_path=self._index_dir,
                embeddings=self._embeddings,
                index_name=self._INDEX_NAME,
                allow_dangerous_deserialization=True,
            )
        return self._store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_populated(self) -> bool:
        """Return True if the FAISS index files exist on disk."""
        return self._faiss_file().exists() and self._pkl_file().exists()

    def reset_collection(self) -> None:
        """Delete the FAISS index files and reset the in-memory store."""
        self._store = None
        for f in [self._faiss_file(), self._pkl_file()]:
            try:
                if f.exists():
                    f.unlink()
                    logger.info("Deleted %s", f)
            except Exception as exc:
                logger.warning("Could not delete %s: %s", f, exc)
        logger.info("Tafsir FAISS index reset.")

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def build_collection(self) -> int:
        """Embed Ibn Kathir + Al-Tabari tafsir into a FAISS index.

        Strategy
        --------
        1. If index files already exist → skip (already populated).
        2. Stream all rows from ``iter_all_tafsir()``, split into 800-char chunks.
        3. Embed in batches of ``EMBED_BATCH`` (200) to stay within OpenAI TPM limits.
        4. Sleep ``BATCH_SLEEP_S`` between batches; back off 60 s on HTTP 429.
        5. After the last batch, save the FAISS index to disk.

        Returns the total number of chunks embedded.
        """
        if self.is_populated():
            # Load to get count
            try:
                store = self._get_store()
                count = store.index.ntotal
                logger.info("FAISS index already populated (%d vectors) — skipping.", count)
                return count
            except Exception as exc:
                logger.warning("Existing FAISS index unreadable (%s) — rebuilding.", exc)
                self._store = None
                self.reset_collection()

        # --- Build full chunk list ---
        docs: list[Document] = []
        for row in iter_all_tafsir():
            for chunk in _split_text(row["text"]):
                docs.append(Document(
                    page_content=chunk,
                    metadata={
                        "source":     row["name_en"],
                        "source_ar":  row["name_ar"],
                        "tafseer_id": row["tafseer_id"],
                        "surah":      row["surah"],
                        "ayah":       row["ayah"],
                        "reference":  row["reference"],
                    },
                ))

        if not docs:
            logger.warning("No tafsir documents found — is tafaseer.db downloaded?")
            return 0

        total = len(docs)
        total_batches = (total + EMBED_BATCH - 1) // EMBED_BATCH
        estimated_min = (total_batches * (BATCH_SLEEP_S + 1)) / 60
        logger.info(
            "Tafsir FAISS ingest: %d chunks, %d batches of %d (~%.0f min estimated).",
            total, total_batches, EMBED_BATCH, estimated_min,
        )

        from langchain_community.vectorstores import FAISS

        faiss_store = None
        embedded = 0

        for i in range(0, total, EMBED_BATCH):
            batch = docs[i : i + EMBED_BATCH]
            batch_num = i // EMBED_BATCH + 1

            for attempt in range(1, 3):
                try:
                    if faiss_store is None:
                        faiss_store = FAISS.from_documents(
                            documents=batch,
                            embedding=self._embeddings,
                        )
                    else:
                        faiss_store.add_documents(batch)
                    break
                except Exception as exc:
                    err = str(exc).lower()
                    is_rate_limit = (
                        "rate" in err or "429" in err or "ratelimit" in err
                        or "too many" in err
                    )
                    if attempt == 1 and is_rate_limit:
                        logger.warning(
                            "Rate limit on batch %d/%d — sleeping %ds …",
                            batch_num, total_batches, RATE_LIMIT_SLEEP_S,
                        )
                        time.sleep(RATE_LIMIT_SLEEP_S)
                    else:
                        logger.exception(
                            "Embedding failed on batch %d/%d (attempt %d): %s",
                            batch_num, total_batches, attempt, exc,
                        )
                        raise

            embedded += len(batch)

            prev_m = (embedded - len(batch)) // PROGRESS_LOG_EVERY
            curr_m = embedded // PROGRESS_LOG_EVERY
            if curr_m > prev_m or embedded >= total:
                logger.info(
                    "FAISS ingest progress: %d / %d (%.1f%%) — batch %d / %d",
                    embedded, total, embedded / total * 100, batch_num, total_batches,
                )

            if i + EMBED_BATCH < total:
                time.sleep(BATCH_SLEEP_S)

        # --- Persist to disk ---
        if faiss_store is not None:
            faiss_store.save_local(
                folder_path=self._index_dir,
                index_name=self._INDEX_NAME,
            )
            self._store = faiss_store
            logger.info("FAISS index saved: %d vectors → %s", embedded, self._index_dir)

        return embedded

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return top-k most semantically relevant tafsir chunks for *query*."""
        if not self.is_populated():
            return []
        try:
            store = self._get_store()
            raw = store.similarity_search_with_score(query, k=top_k)
        except Exception:
            logger.exception("FAISS tafsir search failed for query: %.80s", query)
            return []

        results: list[dict[str, Any]] = []
        for doc, distance in raw:
            # L2 distance → relevance in [0, 1]: smaller distance = more relevant
            relevance = round(max(0.0, 1.0 - distance / 4.0), 3)
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
        """Return top-k Documents for use inside a LangChain chain."""
        if not self.is_populated():
            return []
        try:
            return self._get_store().similarity_search(query, k=top_k)
        except Exception:
            logger.exception("FAISS tafsir retrieval failed for query: %.80s", query)
            return []


# ------------------------------------------------------------------
# Module-level singleton
# ------------------------------------------------------------------

_tafsir_store: TafsirStore | None = None


def get_tafsir_store() -> TafsirStore:
    """Return the shared ``TafsirStore`` singleton (created on first call)."""
    global _tafsir_store
    if _tafsir_store is None:
        _tafsir_store = TafsirStore()
    return _tafsir_store
