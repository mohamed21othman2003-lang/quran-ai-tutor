"""ChromaDB-backed semantic search layer for tafseer content.

Follows the same patterns as ``src/rag/pipeline.py`` and
``src/rag/quran_verifier.py``:
  - Lazy-loaded ``Chroma`` store via ``_get_store()``
  - ``is_populated()`` guard before any search
  - ``build_collection()`` called from the admin endpoint only

Collection: ``tafsir_knowledge``
Metadata keys per document: source, source_ar, tafseer_id, surah, ayah, reference

Ingestion scope: Ibn Kathir (ID=2) + Al-Tabari (ID=1) only = ~12 472 rows.
Because each row can be tens of thousands of chars, the total chunk count
is much higher. At EMBED_BATCH=100 with a 2-second inter-batch pause and a
60-second back-off on HTTP 429, the ingestion runs safely within OpenAI's
1M TPM rate limit.
"""

import logging
import os
import shutil
import time
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

# --- Rate-limit-safe ingestion tunables ---
EMBED_BATCH = 100          # documents sent to OpenAI per embedding call
BATCH_SLEEP_S = 2          # seconds to pause between every batch
RATE_LIMIT_SLEEP_S = 60    # seconds to back off after an HTTP 429
PROGRESS_LOG_EVERY = 500   # emit a progress line every N documents embedded


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
        # Use 256-dim embeddings (text-embedding-3-small supports dimensional
        # reduction) — 6× smaller vectors vs the default 1536 dims, keeping
        # disk usage well under Railway's Volume limit.
        self._embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            dimensions=256,
            openai_api_key=settings.openai_api_key,
        )
        self._store: Chroma | None = None
        # Use a dedicated persistent directory so the tafsir index survives
        # across Railway deployments independently of the main RAG store.
        self._chroma_dir = os.environ.get("TAFSIR_CHROMA_DIR", settings.tafsir_chroma_dir)
        os.makedirs(self._chroma_dir, exist_ok=True)
        os.chmod(self._chroma_dir, 0o777)

    def _get_store(self) -> Chroma:
        if self._store is None:
            self._store = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=self._embeddings,
                persist_directory=self._chroma_dir,
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

    def _wipe_dir_contents(self) -> None:
        """Delete all ChromaDB files inside the chroma directory.

        Deletes known ChromaDB artefacts (SQLite, WAL, SHM, UUID subdirs)
        without removing the directory itself — Railway may rely on the
        directory being present as a Volume sub-path.

        A short sleep after deletion lets the filesystem fully sync before
        ChromaDB creates a fresh database.
        """
        chroma_path = Path(self._chroma_dir)
        if not chroma_path.exists():
            chroma_path.mkdir(parents=True, exist_ok=True)
            os.chmod(self._chroma_dir, 0o777)
            return

        removed = 0
        for item in list(chroma_path.iterdir()):
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
                removed += 1
            except Exception as exc:
                logger.warning("Could not remove %s: %s", item, exc)

        os.chmod(self._chroma_dir, 0o777)
        logger.info("Wiped %d items from %s", removed, self._chroma_dir)
        time.sleep(1)   # allow filesystem to settle before ChromaDB reinitialises

    def build_collection(self) -> int:
        """Embed Ibn Kathir + Al-Tabari tafsir into the ``tafsir_knowledge`` collection.

        Strategy
        --------
        1. Stream all rows from ``iter_all_tafsir()`` (IDs 1 + 2, ~12 472 rows),
           split each into 800-char chunks, and collect them into a flat list.
        2. Send chunks to OpenAI in small batches of ``EMBED_BATCH`` (100) so we
           stay well below the 1M-token-per-minute rate limit.
        3. Sleep ``BATCH_SLEEP_S`` (2 s) between every batch.
        4. On HTTP 429 / RateLimitError, sleep ``RATE_LIMIT_SLEEP_S`` (60 s)
           and retry the same batch once before re-raising.
        5. Log a progress line every ``PROGRESS_LOG_EVERY`` (500) documents.

        Returns the total number of chunks successfully embedded.
        """
        # --- Skip if collection already contains documents (healthy state) ---
        # If opening fails (compaction / WAL / disk-full error), wipe and rebuild.
        try:
            store_ok = self._get_store()
            count = store_ok._collection.count()
            if count > 0:
                logger.info(
                    "Tafsir collection already populated (%d chunks) — skipping ingestion.",
                    count,
                )
                return count
            # Empty collection — wipe any stale files before rebuilding
            logger.info("Tafsir collection exists but is empty — wiping and rebuilding.")
            self._store = None
            self._wipe_dir_contents()
        except Exception as exc:
            logger.warning(
                "Existing tafsir collection unreadable (%s) — wiping and rebuilding.", exc
            )
            self._store = None
            self._wipe_dir_contents()

        # --- Build full chunk list up-front so we know the total for logging ---
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
            logger.warning(
                "No tafsir documents found to embed — is tafaseer.db downloaded?"
            )
            return 0

        total = len(docs)
        total_batches = (total + EMBED_BATCH - 1) // EMBED_BATCH
        estimated_min = (total_batches * (BATCH_SLEEP_S + 1)) / 60
        logger.info(
            "Tafsir ingestion starting: %d chunks across %d batches of %d "
            "(~%.0f min estimated, excluding any rate-limit back-offs).",
            total, total_batches, EMBED_BATCH, estimated_min,
        )

        persist_dir = self._chroma_dir
        store: Chroma | None = None
        embedded = 0

        for i in range(0, total, EMBED_BATCH):
            batch = docs[i : i + EMBED_BATCH]
            batch_num = i // EMBED_BATCH + 1

            # --- Embed with one retry on rate-limit ---
            for attempt in range(1, 3):   # attempts 1 and 2
                try:
                    if store is None:
                        store = Chroma.from_documents(
                            documents=batch,
                            embedding=self._embeddings,
                            collection_name=COLLECTION_NAME,
                            persist_directory=persist_dir,
                        )
                    else:
                        store.add_documents(batch)
                    break  # success

                except Exception as exc:
                    err = str(exc).lower()
                    is_rate_limit = (
                        "rate" in err or "429" in err or "ratelimit" in err
                        or "too many" in err
                    )
                    if attempt == 1 and is_rate_limit:
                        logger.warning(
                            "Rate limit hit on batch %d/%d — sleeping %d s then retrying …",
                            batch_num, total_batches, RATE_LIMIT_SLEEP_S,
                        )
                        time.sleep(RATE_LIMIT_SLEEP_S)
                        # loop continues to attempt 2
                    else:
                        logger.exception(
                            "Embedding failed on batch %d/%d (attempt %d): %s",
                            batch_num, total_batches, attempt, exc,
                        )
                        raise

            embedded += len(batch)

            # --- Progress log every PROGRESS_LOG_EVERY documents ---
            prev_milestone = (embedded - len(batch)) // PROGRESS_LOG_EVERY
            curr_milestone = embedded // PROGRESS_LOG_EVERY
            if curr_milestone > prev_milestone or embedded >= total:
                logger.info(
                    "Tafsir ingestion progress: %d / %d chunks (%.1f%%) — "
                    "batch %d / %d",
                    embedded, total, embedded / total * 100,
                    batch_num, total_batches,
                )

            # --- Pause between batches (skip after the last one) ---
            if i + EMBED_BATCH < total:
                time.sleep(BATCH_SLEEP_S)

        self._store = store
        logger.info(
            "Tafsir collection complete: %d / %d chunks embedded into '%s'.",
            embedded, total, COLLECTION_NAME,
        )
        return embedded

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

    def reset_collection(self) -> None:
        """Wipe the ChromaDB persistence directory contents and reset in-memory state.

        Keeps the directory itself (Railway Volume mount point must not be deleted).
        After calling this, run ``build_collection()`` to rebuild the index.
        """
        self._store = None
        self._wipe_dir_contents()


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
