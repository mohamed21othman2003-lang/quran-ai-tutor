"""SQLite database setup via aiosqlite."""

import aiosqlite

from src.config import settings


async def get_db() -> aiosqlite.Connection:
    """Open a database connection with row_factory set."""
    db = await aiosqlite.connect(settings.db_path)
    db.row_factory = aiosqlite.Row
    return db


async def init_db() -> None:
    """Create tables if they don't exist."""
    async with aiosqlite.connect(settings.db_path) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                username  TEXT    NOT NULL UNIQUE,
                password  TEXT    NOT NULL,
                created_at TEXT   DEFAULT (datetime('now'))
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS progress (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    INTEGER NOT NULL REFERENCES users(id),
                rule_name  TEXT    NOT NULL,
                score      REAL    NOT NULL CHECK(score BETWEEN 0 AND 1),
                type       TEXT    NOT NULL CHECK(type IN ('chat', 'quiz')),
                created_at TEXT    DEFAULT (datetime('now'))
            )
        """)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_progress_user ON progress(user_id)")

        # ── Asbab al-Nuzul tables ────────────────────────────────────────────
        await db.execute("""
            CREATE TABLE IF NOT EXISTS asbab_sources (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                name_arabic      TEXT    NOT NULL,
                name_english     TEXT,
                author_arabic    TEXT    NOT NULL,
                death_year_hijri INTEGER,
                description      TEXT
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS asbab_nuzul (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                verse_key          TEXT    NOT NULL,
                surah_id           INTEGER NOT NULL,
                ayah_number        INTEGER NOT NULL,
                sabab_text         TEXT    NOT NULL,
                source_id          INTEGER REFERENCES asbab_sources(id),
                isnad              TEXT,
                authenticity_grade TEXT,
                grading_scholar    TEXT,
                revelation_period  TEXT,
                created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_asbab_verse ON asbab_nuzul(verse_key)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_asbab_surah ON asbab_nuzul(surah_id)")

        # ── Qiraat tables ────────────────────────────────────────────────────
        await db.execute("""
            CREATE TABLE IF NOT EXISTS qurra (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                name_arabic      TEXT    NOT NULL,
                name_english     TEXT,
                death_year_hijri INTEGER,
                city             TEXT,
                rank_order       INTEGER
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS riwayat (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                qari_id      INTEGER REFERENCES qurra(id),
                code         TEXT    UNIQUE NOT NULL,
                name_arabic  TEXT    NOT NULL,
                name_english TEXT,
                description  TEXT
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS qiraat_variants (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                verse_key       TEXT    NOT NULL,
                surah_id        INTEGER NOT NULL,
                ayah_number     INTEGER NOT NULL,
                riwaya_code     TEXT    NOT NULL,
                riwaya_name     TEXT    NOT NULL,
                verse_text      TEXT    NOT NULL,
                difference_note TEXT,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_qiraat_verse ON qiraat_variants(verse_key)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_qiraat_surah ON qiraat_variants(surah_id)")

        await db.commit()
