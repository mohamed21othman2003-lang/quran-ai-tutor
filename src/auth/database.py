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
        await db.commit()
