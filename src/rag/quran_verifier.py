"""Quran verse verifier — looks up Arabic text against the quran_verses ChromaDB collection.

Used by the voice router to:
1. Find the closest matching ayah to a transcription.
2. Detect if a recitation contradicts the canonical Quran text.
"""

import logging
import re
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from src.config import settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "quran_verses"
# Cosine distance threshold: 0 = identical, 2 = opposite.
# Matches below this distance are considered valid Quran text.
_MATCH_DISTANCE_THRESHOLD = 0.65

_DIACRITICS = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]"
)


def _normalize(text: str) -> str:
    text = _DIACRITICS.sub("", text)
    return " ".join(text.split()).strip()


class QuranVerifier:
    """Similarity-based Quran text verifier backed by the quran_verses ChromaDB collection."""

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
        try:
            return self._get_store()._collection.count() > 0
        except Exception:
            return False

    def find_closest_ayah(self, text: str) -> dict[str, Any] | None:
        """Return the closest Quran ayah to *text*, or None if no good match.

        Returns a dict with keys:
          chapter, verse, surah_name, reference, canonical_text, distance
        """
        if not self.is_populated():
            return None
        try:
            results = self._get_store().similarity_search_with_score(text, k=1)
            if not results:
                return None
            doc, distance = results[0]
            if distance > _MATCH_DISTANCE_THRESHOLD:
                logger.debug(
                    "No Quran match for text (distance %.3f > threshold %.3f)",
                    distance, _MATCH_DISTANCE_THRESHOLD,
                )
                return None
            return {
                "chapter":        doc.metadata.get("chapter"),
                "verse":          doc.metadata.get("verse"),
                "surah_name":     doc.metadata.get("surah_name", ""),
                "reference":      doc.metadata.get("reference", ""),
                "canonical_text": doc.page_content,
                "distance":       round(float(distance), 4),
            }
        except Exception:
            logger.exception("Quran similarity search failed.")
            return None

    def verify(self, transcribed: str, expected: str) -> dict[str, Any]:
        """Verify *transcribed* text against the expected Quran ayah.

        Returns a dict with:
          - quran_found       bool  — True if expected_text matched a known ayah
          - quran_reference   str   — e.g. "2:255" (empty if not found)
          - quran_surah       str   — Arabic surah name (empty if not found)
          - quran_canonical   str   — exact Quran text (empty if not found)
          - quran_mismatch    bool  — True when transcription contradicts canonical text
          - mismatch_detail   str   — human-readable explanation (empty if no mismatch)
        """
        result: dict[str, Any] = {
            "quran_found":     False,
            "quran_reference": "",
            "quran_surah":     "",
            "quran_canonical": "",
            "quran_mismatch":  False,
            "mismatch_detail": "",
        }

        if not expected.strip():
            return result

        # Look up the expected text to get the canonical form
        match = self.find_closest_ayah(expected)
        if not match:
            return result

        result["quran_found"]     = True
        result["quran_reference"] = match["reference"]
        result["quran_surah"]     = match["surah_name"]
        result["quran_canonical"] = match["canonical_text"]

        # Compare normalised transcription against normalised canonical text
        norm_transcribed = _normalize(transcribed)
        norm_canonical   = _normalize(match["canonical_text"])

        if norm_transcribed and norm_canonical and norm_transcribed != norm_canonical:
            result["quran_mismatch"] = True
            result["mismatch_detail"] = (
                f"Transcription does not match the canonical text for "
                f"Surah {match['surah_name']} ({match['reference']}). "
                f"Expected: {match['canonical_text']}"
            )

        return result
