"""Tafseer SQLite database layer.

Downloads ``tafaseer.zip`` from:
  https://github.com/Mr-DDDAlKilanny/tafseer-sqlite-db

The zip contains 8 individual SQLite files (one per tafseer book).
Each file uses the schema:
  CREATE TABLE verses (id INTEGER, sura INTEGER, ayah INTEGER, text TEXT)

This module exposes two public functions used by the router and the ChromaDB
ingest pipeline:
  - ensure_databases()         — download + extract on first call (lazy)
  - get_tafsir_for_ayah()      — exact reference lookup (surah, ayah)
  - search_tafsir_text()       — SQLite LIKE fallback for free-text queries
  - iter_all_tafsir()          — full-table iterator used during ChromaDB ingestion
"""

import io
import logging
import sqlite3
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterator

from src.config import settings

logger = logging.getLogger(__name__)

ZIP_URL = (
    "https://raw.githubusercontent.com/"
    "Mr-DDDAlKilanny/tafseer-sqlite-db/master/tafaseer.zip"
)

# Only these two books are surfaced via the API; others stay on disk but unused.
TAFSIR_SOURCES: dict[str, dict[str, str]] = {
    "tabary":  {"name_en": "Al-Tabari",  "name_ar": "الطبري"},
    "katheer": {"name_en": "Ibn Kathir", "name_ar": "ابن كثير"},
}

# Additional filename stems that might appear inside the zip for each book.
# The first match wins; keys must match TAFSIR_SOURCES keys.
_STEM_ALIASES: dict[str, list[str]] = {
    "tabary":  ["tabary", "tabari", "al-tabari", "altabari", "1"],
    "katheer": ["katheer", "kathir", "ibnkathir", "ibn-kathir", "ibn_kathir", "2"],
}


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _tafsir_dir() -> Path:
    return Path(settings.tafsir_db_dir)


def _identify_db_stem(zip_member: str) -> str | None:
    """Map a zip-member filename to a TAFSIR_SOURCES key, or None if not recognised."""
    name = Path(zip_member).stem.lower()
    for stem, aliases in _STEM_ALIASES.items():
        if any(alias in name for alias in aliases):
            return stem
    return None


def _db_path(stem: str) -> Path:
    return _tafsir_dir() / f"{stem}.db"


def _open(stem: str) -> sqlite3.Connection:
    path = _db_path(stem)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


# ------------------------------------------------------------------
# Download & extraction
# ------------------------------------------------------------------

def ensure_databases() -> None:
    """Download and extract tafseer databases on first call.

    Subsequent calls are a no-op when both ``tabary.db`` and ``katheer.db``
    already exist in ``settings.tafsir_db_dir``.
    """
    db_dir = _tafsir_dir()
    db_dir.mkdir(parents=True, exist_ok=True)

    needed = set(TAFSIR_SOURCES.keys())
    present = {p.stem for p in db_dir.glob("*.db")}
    if needed.issubset(present):
        logger.debug("Tafseer DBs already present at %s", db_dir)
        return

    logger.info("Downloading tafseer ZIP (~36 MB) from %s …", ZIP_URL)
    with urllib.request.urlopen(ZIP_URL, timeout=180) as resp:  # noqa: S310
        raw = resp.read()
    logger.info("Download complete. Extracting …")

    extracted: list[str] = []
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        for member in zf.namelist():
            if not member.endswith(".db"):
                continue
            stem = _identify_db_stem(member)
            if stem is None:
                logger.debug("Skipping unrecognised DB in zip: %s", member)
                continue
            target = _db_path(stem)
            target.write_bytes(zf.read(member))
            extracted.append(f"{member} → {target.name}")
            logger.info("Extracted %s → %s", member, target.name)

    if not extracted:
        raise RuntimeError(
            "Could not identify tabary.db / katheer.db inside the downloaded zip. "
            "File listing: " + str(
                [m for m in zipfile.ZipFile(io.BytesIO(raw)).namelist() if m.endswith(".db")]
            )
        )
    logger.info("Tafseer databases ready. Extracted: %s", extracted)


# ------------------------------------------------------------------
# Query helpers
# ------------------------------------------------------------------

def get_tafsir_for_ayah(surah: int, ayah: int) -> dict[str, dict]:
    """Return tafsir texts for a specific ayah from both Ibn Kathir and Al-Tabari.

    Returns a dict keyed by TAFSIR_SOURCES stem::

        {
          "tabary":  {"name_en": "Al-Tabari", "name_ar": "الطبري",  "text": "..."},
          "katheer": {"name_en": "Ibn Kathir","name_ar": "ابن كثير","text": "..."},
        }

    Missing entries (DB absent or ayah not found) are omitted from the result.
    """
    result: dict[str, dict] = {}
    for stem, meta in TAFSIR_SOURCES.items():
        if not _db_path(stem).exists():
            logger.warning("Tafseer DB missing: %s", _db_path(stem))
            continue
        try:
            conn = _open(stem)
            try:
                row = conn.execute(
                    "SELECT text FROM verses WHERE sura = ? AND ayah = ?",
                    (surah, ayah),
                ).fetchone()
            finally:
                conn.close()

            if row and row["text"]:
                result[stem] = {**meta, "text": row["text"].strip()}
        except sqlite3.OperationalError as exc:
            logger.error("Query failed for %s (%d:%d): %s", stem, surah, ayah, exc)

    return result


def search_tafsir_text(query: str, limit: int = 10) -> list[dict]:
    """Keyword search across both tafseer databases using SQLite LIKE.

    Used as a fallback when the ChromaDB semantic index has not been built yet.
    Returns a list of dicts with keys: stem, name_en, name_ar, surah, ayah,
    reference, text.
    """
    results: list[dict] = []
    pattern = f"%{query}%"

    for stem, meta in TAFSIR_SOURCES.items():
        if not _db_path(stem).exists():
            continue
        try:
            conn = _open(stem)
            try:
                rows = conn.execute(
                    "SELECT sura, ayah, text FROM verses WHERE text LIKE ? LIMIT ?",
                    (pattern, limit),
                ).fetchall()
            finally:
                conn.close()

            for row in rows:
                results.append({
                    "stem":      stem,
                    "name_en":   meta["name_en"],
                    "name_ar":   meta["name_ar"],
                    "surah":     row["sura"],
                    "ayah":      row["ayah"],
                    "reference": f"{row['sura']}:{row['ayah']}",
                    "text":      (row["text"] or "").strip(),
                })
        except sqlite3.OperationalError as exc:
            logger.error("LIKE search failed for %s: %s", stem, exc)

    return results


def iter_all_tafsir() -> Iterator[dict]:
    """Yield every row from both tafseer databases in (sura, ayah) order.

    Yields dicts with keys: stem, name_en, name_ar, surah, ayah, reference, text.
    Used by TafsirStore.build_collection() to embed content into ChromaDB.
    """
    for stem, meta in TAFSIR_SOURCES.items():
        if not _db_path(stem).exists():
            logger.warning("Skipping missing DB during iteration: %s", _db_path(stem))
            continue
        conn = _open(stem)
        try:
            rows = conn.execute(
                "SELECT sura, ayah, text FROM verses ORDER BY sura, ayah"
            ).fetchall()
        except sqlite3.OperationalError as exc:
            logger.error("iter_all_tafsir failed for %s: %s", stem, exc)
            conn.close()
            continue

        conn.close()
        for row in rows:
            text = (row["text"] or "").strip()
            if text:
                yield {
                    "stem":      stem,
                    "name_en":   meta["name_en"],
                    "name_ar":   meta["name_ar"],
                    "surah":     row["sura"],
                    "ayah":      row["ayah"],
                    "reference": f"{row['sura']}:{row['ayah']}",
                    "text":      text,
                }
