"""Tafseer SQLite database layer.

Downloads ``tafaseer.zip`` from:
  https://github.com/Mr-DDDAlKilanny/tafseer-sqlite-db

The zip contains a **single** file: ``tafaseer.db`` (~146 MB uncompressed).

Actual schema (verified against the real database):

    CREATE TABLE TafseerName (
        ID    INTEGER PRIMARY KEY AUTOINCREMENT,
        Name  TEXT NOT NULL,   -- Arabic name  e.g. "الطبري"
        NameE TEXT             -- Latin  name  e.g. "tabary"
    );

    CREATE TABLE Tafseer (
        tafseer INTEGER,       -- foreign key → TafseerName.ID
        sura    INTEGER,
        ayah    INTEGER,
        nass    TEXT NOT NULL, -- commentary text in Arabic
        PRIMARY KEY (tafseer, sura, ayah)
    );

Tafseer IDs surfaced by this module:
    1 = Al-Tabari  (الطبري)
    2 = Ibn Kathir (ابن كثير)
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

DB_FILENAME = "tafaseer.db"   # sole file inside the zip

# Tafseer IDs that the public API surfaces (verified from TafseerName table)
TAFSIR_SOURCES: dict[int, dict[str, str]] = {
    1: {"name_en": "Al-Tabari",  "name_ar": "الطبري"},
    2: {"name_en": "Ibn Kathir", "name_ar": "ابن كثير"},
}


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _db_path() -> Path:
    return Path(settings.tafsir_db_dir) / DB_FILENAME


def _open_db() -> sqlite3.Connection:
    """Return an open, row-factory-enabled connection to tafaseer.db."""
    conn = sqlite3.connect(str(_db_path()))
    conn.row_factory = sqlite3.Row
    return conn


# ------------------------------------------------------------------
# Download & extraction
# ------------------------------------------------------------------

def ensure_database() -> None:
    """Download and extract ``tafaseer.db`` on first call.

    Subsequent calls are a no-op when the database file already exists
    in ``settings.tafsir_db_dir``.
    """
    target = _db_path()
    if target.exists():
        logger.debug("tafaseer.db already present at %s", target)
        return

    target.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading tafaseer.zip (~36 MB compressed) from %s …", ZIP_URL)
    with urllib.request.urlopen(ZIP_URL, timeout=180) as resp:  # noqa: S310
        raw = resp.read()
    logger.info("Download complete (%d bytes). Extracting …", len(raw))

    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        # Locate any .db member — robust against subdirectory nesting
        db_members = [m for m in zf.namelist() if m.endswith(".db")]
        if not db_members:
            raise RuntimeError(
                f"No .db file found inside tafaseer.zip. "
                f"Contents: {zf.namelist()}"
            )
        # Prefer the exact known filename; otherwise take the first .db
        chosen = next((m for m in db_members if Path(m).name == DB_FILENAME), db_members[0])
        target.write_bytes(zf.read(chosen))
        logger.info("Extracted %r → %s (%d bytes)", chosen, target, target.stat().st_size)


# ------------------------------------------------------------------
# Schema introspection (used by debug endpoint)
# ------------------------------------------------------------------

def get_schema_info() -> dict:
    """Return a dict describing every table's DDL, column names, and row count.

    Used by the admin ``/api/v1/tafsir/schema`` debug endpoint.
    """
    ensure_database()
    conn = _open_db()
    try:
        tables = conn.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type='table' ORDER BY name"
        ).fetchall()

        result: dict = {"db_path": str(_db_path()), "tables": {}}
        for tbl in tables:
            name = tbl["name"]
            cols = [
                {"name": c["name"], "type": c["type"]}
                for c in conn.execute(f"PRAGMA table_info([{name}])").fetchall()
            ]
            count = conn.execute(f"SELECT COUNT(*) FROM [{name}]").fetchone()[0]
            result["tables"][name] = {
                "ddl":     tbl["sql"],
                "columns": cols,
                "rows":    count,
            }

        # Also pull TafseerName so the caller sees the ID ↔ book mapping
        tafseer_names = [
            {"id": r["ID"], "name_ar": r["Name"], "name_en": r["NameE"]}
            for r in conn.execute(
                "SELECT ID, Name, NameE FROM TafseerName ORDER BY ID"
            ).fetchall()
        ]
        result["tafseer_names"] = tafseer_names
        return result
    finally:
        conn.close()


# ------------------------------------------------------------------
# Query helpers
# ------------------------------------------------------------------

def get_tafsir_for_ayah(surah: int, ayah: int) -> list[dict]:
    """Return tafsir commentary for a specific ayah from Ibn Kathir and Al-Tabari.

    Returns a list of dicts (one per source found), each with keys:
        tafseer_id, name_en, name_ar, surah, ayah, reference, text

    Missing entries (ayah not in DB) are omitted from the result.
    """
    ids = tuple(TAFSIR_SOURCES.keys())          # (1, 2)
    placeholders = ",".join("?" * len(ids))

    conn = _open_db()
    try:
        rows = conn.execute(
            f"SELECT tafseer, sura, ayah, nass "
            f"FROM Tafseer "
            f"WHERE tafseer IN ({placeholders}) AND sura = ? AND ayah = ? "
            f"ORDER BY tafseer",
            (*ids, surah, ayah),
        ).fetchall()
    finally:
        conn.close()

    result: list[dict] = []
    for row in rows:
        tid = row["tafseer"]
        meta = TAFSIR_SOURCES.get(tid, {"name_en": f"Tafseer {tid}", "name_ar": ""})
        nass = (row["nass"] or "").strip()
        if nass:
            result.append({
                "tafseer_id": tid,
                "name_en":   meta["name_en"],
                "name_ar":   meta["name_ar"],
                "surah":     row["sura"],
                "ayah":      row["ayah"],
                "reference": f"{row['sura']}:{row['ayah']}",
                "text":      nass,
            })
    return result


def search_tafsir_text(query: str, limit: int = 10) -> list[dict]:
    """Full-text search over Ibn Kathir and Al-Tabari using SQLite LIKE.

    Used as a fallback when the ChromaDB semantic index has not been built.
    Returns a list of dicts with the same keys as ``get_tafsir_for_ayah()``.
    """
    ids = tuple(TAFSIR_SOURCES.keys())
    placeholders = ",".join("?" * len(ids))
    pattern = f"%{query}%"

    conn = _open_db()
    try:
        rows = conn.execute(
            f"SELECT tafseer, sura, ayah, nass "
            f"FROM Tafseer "
            f"WHERE tafseer IN ({placeholders}) AND nass LIKE ? "
            f"ORDER BY tafseer, sura, ayah "
            f"LIMIT ?",
            (*ids, pattern, limit),
        ).fetchall()
    finally:
        conn.close()

    result: list[dict] = []
    for row in rows:
        tid = row["tafseer"]
        meta = TAFSIR_SOURCES.get(tid, {"name_en": f"Tafseer {tid}", "name_ar": ""})
        nass = (row["nass"] or "").strip()
        if nass:
            result.append({
                "tafseer_id": tid,
                "name_en":   meta["name_en"],
                "name_ar":   meta["name_ar"],
                "surah":     row["sura"],
                "ayah":      row["ayah"],
                "reference": f"{row['sura']}:{row['ayah']}",
                "text":      nass,
            })
    return result


def iter_all_tafsir() -> Iterator[dict]:
    """Yield every row for Ibn Kathir and Al-Tabari in (sura, ayah) order.

    Yields dicts with the same keys as ``get_tafsir_for_ayah()``.
    Used by ``TafsirStore.build_collection()`` to embed content into ChromaDB.
    """
    ids = tuple(TAFSIR_SOURCES.keys())
    placeholders = ",".join("?" * len(ids))

    conn = _open_db()
    try:
        rows = conn.execute(
            f"SELECT tafseer, sura, ayah, nass "
            f"FROM Tafseer "
            f"WHERE tafseer IN ({placeholders}) "
            f"ORDER BY tafseer, sura, ayah",
            ids,
        ).fetchall()
    finally:
        conn.close()

    for row in rows:
        tid = row["tafseer"]
        meta = TAFSIR_SOURCES.get(tid, {"name_en": f"Tafseer {tid}", "name_ar": ""})
        nass = (row["nass"] or "").strip()
        if nass:
            yield {
                "tafseer_id": tid,
                "name_en":   meta["name_en"],
                "name_ar":   meta["name_ar"],
                "surah":     row["sura"],
                "ayah":      row["ayah"],
                "reference": f"{row['sura']}:{row['ayah']}",
                "text":      nass,
            }
