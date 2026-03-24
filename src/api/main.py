"""FastAPI application — Quran AI Tutor MVP.

Endpoints
---------
POST /api/v1/chat                  — Tajweed Q&A (RAG + GPT-4o)
GET  /api/v1/rules                 — List all Tajweed rules in the knowledge base
POST /api/v1/quiz                  — Generate a single multiple-choice quiz question
POST /api/v1/quiz/batch            — Generate N unique quiz questions (no duplicates)
POST /api/v1/auth/register         — Create account, returns JWT
POST /api/v1/auth/login            — Authenticate, returns JWT
POST /api/v1/progress              — Save a learning event (auth required)
GET  /api/v1/progress/{uid}        — Get user history + weak rules (auth required)
POST /api/v1/admin/ingest          — Re-ingest knowledge base into ChromaDB (admin key required)
POST /api/v1/admin/ingest-quran    — Ingest full Quran into quran_verses collection
POST /api/v1/admin/ingest-tafsir          — Download tafaseer.db (SQLite only, no embedding)
POST /api/v1/admin/ingest-tafsir-semantic — Build ChromaDB semantic index (opt-in, slow)
POST /api/v1/voice/check           — Transcribe recitation + compare with expected ayah
POST /api/v1/tafsir/search         — Search Ibn Kathir & Al-Tabari commentary
POST /api/v1/tafsir/ask            — AI-synthesised tafsir Q&A via GPT-4o
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel, Field

from src.agents.tutor_agent import TutorAgent
from src.auth.database import init_db
from src.auth.router import router as auth_router
from src.config import settings
from src.progress.router import router as progress_router
from src.rag.pipeline import RAGPipeline
from src.search.router import router as search_router
from src.tajweed.router import router as tajweed_router
from src.tafsir.router import router as tafsir_router
from src.voice.router import router as voice_router
from src.asbab.router import router as asbab_router
from src.qiraat.router import router as qiraat_router

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

logging.basicConfig(level=settings.log_level.upper())
logger = logging.getLogger(__name__)
limiter = Limiter(key_func=get_remote_address)

# ------------------------------------------------------------------
# App state (shared across requests)
# ------------------------------------------------------------------

rag: RAGPipeline | None = None
agent: TutorAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag, agent
    logger.info("Starting up — initialising database…")
    await init_db()
    logger.info("Starting up — loading RAG pipeline…")
    try:
        rag = RAGPipeline()
        agent = TutorAgent(rag)
        if rag.is_populated():
            logger.info("Ready. Vector store is populated.")
        else:
            logger.warning(
                "Ready — but vector store is EMPTY. "
                "Run `python -m src.rag.pipeline` to ingest knowledge files. "
                "Chat and quiz endpoints will return an informational message until then."
            )
    except Exception:
        logger.exception(
            "Failed to initialise RAG pipeline — server will start without AI features. "
            "Check OPENAI_API_KEY and ChromaDB configuration."
        )
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Quran AI Tutor",
    description="AI-powered Tajweed teaching assistant (Arabic + English).",
    version="0.1.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ------------------------------------------------------------------
# CORS
# ------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://web-production-fa50b.up.railway.app",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Frontend
# ------------------------------------------------------------------

FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend"

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/health", include_in_schema=False)
async def health_check():
    return {
        "status": "ok",
        "rag_populated": rag.is_populated() if rag else False,
        "agent_ready": agent is not None,
    }


@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse(str(FRONTEND_DIR / "index.html"))

# ------------------------------------------------------------------
# Routers
# ------------------------------------------------------------------

app.include_router(auth_router)
app.include_router(progress_router)
app.include_router(voice_router)
app.include_router(search_router)
app.include_router(tajweed_router)
app.include_router(tafsir_router)
app.include_router(asbab_router)
app.include_router(qiraat_router)

# ------------------------------------------------------------------
# Pydantic models
# ------------------------------------------------------------------


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, examples=["What is Idgham?"])
    language: str = Field(default="en", pattern="^(en|ar)$")


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


class RulesResponse(BaseModel):
    rules: list[str]


class QuizRequest(BaseModel):
    rule: str = Field(..., min_length=1, max_length=200, examples=["Ikhfa"])
    language: str = Field(default="en", pattern="^(en|ar)$")


class QuizResponse(BaseModel):
    question: str
    options: list[str]
    correct_index: int
    explanation: str


class IngestResponse(BaseModel):
    message: str
    chunks_indexed: int
    rules_found: int


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@app.post("/api/v1/chat", response_model=ChatResponse, summary="Tajweed Q&A")
async def chat(request: ChatRequest) -> Any:
    """Answer a Tajweed question using RAG retrieval + GPT-4o.

    The model responds in the language of the question (Arabic or English).
    Answers are grounded in the retrieved knowledge base only.
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialised.")
    try:
        result = await agent.answer(request.question, language=request.language)
    except Exception as exc:
        logger.exception("Error in /chat")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ChatResponse(**result)


@app.get("/api/v1/rules", response_model=RulesResponse, summary="List Tajweed rules")
async def list_rules() -> Any:
    """Return all unique Tajweed rule names stored in the knowledge base.

    Rule names are read from the ``rule_name`` metadata field of each
    ChromaDB document.
    """
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialised.")
    try:
        rules = rag.get_all_rule_names()
    except Exception as exc:
        logger.exception("Error in /rules")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return RulesResponse(rules=rules)


@app.post("/api/v1/quiz", response_model=QuizResponse, summary="Generate a quiz")
async def generate_quiz(request: QuizRequest) -> Any:
    """Generate a single multiple-choice quiz question for the specified Tajweed rule.

    The quiz question, options, and explanation are produced by GPT-4o using
    only context retrieved from the knowledge base.
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialised.")
    try:
        quiz = await agent.generate_quiz(rule=request.rule, language=request.language)
    except Exception as exc:
        logger.exception("Error in /quiz")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return QuizResponse(**quiz)


class QuizBatchRequest(BaseModel):
    rule: str = Field(..., min_length=1, max_length=200,
                      examples=["الإخفاء الحقيقي"])
    count: int = Field(default=3, ge=1, le=10)
    language: str = Field(default="ar", pattern="^(en|ar)$")


@app.post(
    "/api/v1/quiz/batch",
    summary="Generate multiple unique quiz questions in one call",
)
async def generate_quiz_batch(request: QuizBatchRequest) -> Any:
    """Generate *count* unique quiz questions in a single GPT call.

    Eliminates the duplicate-question problem caused by firing N parallel
    requests with the same topic.  The model is instructed to produce
    completely different questions testing different sub-aspects of the topic.
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialised.")
    try:
        questions = await agent.generate_quiz_batch(
            rule=request.rule,
            count=request.count,
            language=request.language,
        )
    except Exception as exc:
        logger.exception("Error in /quiz/batch")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"questions": questions, "count": len(questions)}


@app.post(
    "/api/v1/admin/ingest",
    response_model=IngestResponse,
    summary="Re-ingest knowledge base",
    tags=["Admin"],
)
async def ingest(x_admin_key: str = Header(..., alias="X-Admin-Key")) -> Any:
    """Trigger a full re-ingestion of the knowledge base into ChromaDB.

    Requires the ``X-Admin-Key`` header to match the ``ADMIN_API_KEY`` env var.
    Ingestion runs in a thread pool so it does not block the event loop.
    """
    if x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=403, detail="Invalid admin key.")
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialised.")
    try:
        logger.info("Admin triggered re-ingestion.")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, rag.ingest)
        rules = rag.get_all_rule_names()
        chunks = rag.get_vector_store()._collection.count()
        logger.info("Re-ingestion complete: %d chunks, %d rules.", chunks, len(rules))
        return IngestResponse(
            message="Ingestion complete.",
            chunks_indexed=chunks,
            rules_found=len(rules),
        )
    except Exception as exc:
        logger.exception("Error in /admin/ingest")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


class QuranIngestResponse(BaseModel):
    message: str
    ayahs_indexed: int
    markdown_files: int


@app.post(
    "/api/v1/admin/ingest-quran",
    response_model=QuranIngestResponse,
    summary="Ingest full Quran into quran_verses collection",
    tags=["Admin"],
)
async def ingest_quran(x_admin_key: str = Header(..., alias="X-Admin-Key")) -> Any:
    """Download quran_full.json (if absent), write per-Surah markdown files, and
    build the quran_verses ChromaDB collection used by the voice verifier.

    This is a one-time setup step (~6 236 embeddings, takes a few minutes).
    """
    if x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=403, detail="Invalid admin key.")
    try:
        from src.rag.ingest_quran import (
            build_quran_collection,
            create_markdown_files,
            ensure_quran_json,
        )
        loop = asyncio.get_running_loop()
        quran_data = await loop.run_in_executor(None, ensure_quran_json)
        md_count = await loop.run_in_executor(None, create_markdown_files, quran_data)
        ayah_count = await loop.run_in_executor(None, build_quran_collection, quran_data)
        return QuranIngestResponse(
            message="Quran ingestion complete.",
            ayahs_indexed=ayah_count,
            markdown_files=md_count,
        )
    except Exception as exc:
        logger.exception("Error in /admin/ingest-quran")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ------------------------------------------------------------------
# Admin: ingest tafsir
# ------------------------------------------------------------------

class TafsirIngestResponse(BaseModel):
    message: str
    db_size_mb: float
    chunks_indexed: int   # 0 unless semantic embedding is explicitly triggered


# Shared state for the long-running tafsir semantic ingest background task.
# Keys: "status" ("idle"|"running"|"done"|"error"), "chunks", "error"
_tafsir_ingest_state: dict = {"status": "idle", "chunks": 0, "error": ""}


@app.post(
    "/api/v1/admin/ingest-tafsir",
    response_model=TafsirIngestResponse,
    summary="Download tafaseer.db (SQLite only — no ChromaDB embedding)",
    tags=["Admin"],
)
async def ingest_tafsir(x_admin_key: str = Header(..., alias="X-Admin-Key")) -> Any:
    """Download and extract ``tafaseer.db`` (~36 MB compressed / ~146 MB on disk).

    **This endpoint only downloads the SQLite database.**
    ChromaDB embedding is intentionally skipped to avoid OpenAI rate limits.

    After this call completes:
    - ``POST /api/v1/tafsir/search`` with a ``reference`` field works immediately
      (exact SQLite lookup, no embeddings needed).
    - Free-text ``query`` searches fall back to SQLite ``LIKE``.
    - To enable semantic (cosine) search, call
      ``POST /api/v1/admin/ingest-tafsir-semantic`` instead
      (separate endpoint, triggers the full ChromaDB embedding pipeline).

    Requires the ``X-Admin-Key`` header.
    """
    if x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=403, detail="Invalid admin key.")
    try:
        from pathlib import Path as _Path

        from src.tafsir.database import ensure_database, _db_path

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, ensure_database)

        size_mb = round(_db_path().stat().st_size / 1_048_576, 1)
        return TafsirIngestResponse(
            message="tafaseer.db downloaded and ready. SQL reference lookup is active.",
            db_size_mb=size_mb,
            chunks_indexed=0,
        )
    except Exception as exc:
        logger.exception("Error in /admin/ingest-tafsir")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _run_tafsir_semantic_ingest() -> None:
    """Background worker for tafsir semantic ingestion.

    Runs in a thread-pool executor so the HTTP response is returned immediately
    (202 Accepted) and Railway's ~5-minute request timeout is not a concern.
    Updates ``_tafsir_ingest_state`` so callers can poll
    ``GET /api/v1/admin/tafsir-ingest-status``.
    """
    global _tafsir_ingest_state
    _tafsir_ingest_state = {"status": "running", "chunks": 0, "error": ""}
    try:
        from src.tafsir.database import ensure_database
        from src.tafsir.store import get_tafsir_store

        ensure_database()
        store = get_tafsir_store()   # use singleton — avoids concurrent-init race
        chunks = store.build_collection()
        _tafsir_ingest_state = {"status": "done", "chunks": chunks, "error": ""}
        logger.info("Background tafsir ingest complete: %d chunks.", chunks)
    except Exception as exc:
        import traceback as _tb
        full_tb = _tb.format_exc()
        logger.exception("Background tafsir ingest failed")
        _tafsir_ingest_state = {
            "status": "error", "chunks": 0,
            "error": str(exc),
            "traceback": full_tb[-2000:],  # last 2000 chars of traceback
        }


@app.post(
    "/api/v1/admin/ingest-tafsir-semantic",
    summary="Build ChromaDB semantic index for tafsir (async — returns 202 immediately)",
    tags=["Admin"],
    status_code=202,
)
async def ingest_tafsir_semantic(
    background_tasks: BackgroundTasks,
    x_admin_key: str = Header(..., alias="X-Admin-Key"),
) -> Any:
    """Start background embedding of Ibn Kathir + Al-Tabari commentary.

    Returns **202 Accepted** immediately.  Poll
    ``GET /api/v1/admin/tafsir-ingest-status`` to track progress.

    **Prerequisites**: ``POST /api/v1/admin/ingest-tafsir`` must be run first.
    Ingestion is skipped automatically if the collection is already populated.
    """
    if x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=403, detail="Invalid admin key.")

    if _tafsir_ingest_state.get("status") == "running":
        return JSONResponse(
            status_code=202,
            content={"message": "Ingest already running. Poll /admin/tafsir-ingest-status."},
        )

    background_tasks.add_task(_run_tafsir_semantic_ingest)
    return JSONResponse(
        status_code=202,
        content={
            "message": "Tafsir semantic ingest started in background. "
                       "Poll GET /api/v1/admin/tafsir-ingest-status for progress.",
        },
    )


@app.get(
    "/api/v1/admin/tafsir-ingest-status",
    summary="Poll the tafsir semantic ingest background job status",
    tags=["Admin"],
)
async def tafsir_ingest_status(
    x_admin_key: str = Header(..., alias="X-Admin-Key"),
) -> Any:
    """Return the current status of the tafsir semantic ingest job.

    Possible ``status`` values: ``idle``, ``running``, ``done``, ``error``.
    """
    if x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=403, detail="Invalid admin key.")
    return _tafsir_ingest_state


@app.delete(
    "/api/v1/admin/reset-tafsir-index",
    summary="Delete the tafsir ChromaDB collection (fixes compaction errors)",
    tags=["Admin"],
)
async def reset_tafsir_index(
    x_admin_key: str = Header(..., alias="X-Admin-Key"),
) -> Any:
    """Delete the ``tafsir_knowledge`` ChromaDB collection and reset the
    in-memory handle.

    Use this when ``ingest-tafsir-semantic`` fails with a compaction or
    metadata-segment error.  After resetting, call
    ``POST /api/v1/admin/ingest-tafsir-semantic`` to rebuild the index.

    Requires the ``X-Admin-Key`` header.
    """
    if x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=403, detail="Invalid admin key.")
    try:
        from src.tafsir.store import get_tafsir_store
        loop = asyncio.get_running_loop()
        store = get_tafsir_store()
        await loop.run_in_executor(None, store.reset_collection)
        return {
            "message": (
                "Tafsir index reset successfully. "
                "Run POST /api/v1/admin/ingest-tafsir-semantic to rebuild."
            )
        }
    except Exception as exc:
        logger.exception("Error in /admin/reset-tafsir-index")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/v1/admin/tafsir-books", tags=["Admin"])
async def get_tafsir_books(x_admin_key: str = Header(..., alias="X-Admin-Key")):
    """Return all book names and IDs from the TafseerName table in tafaseer.db."""
    if x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=403, detail="Invalid admin key.")
    import aiosqlite as _aiosqlite
    import os as _os
    db_path = _os.path.join(settings.tafsir_db_dir, "tafaseer.db")
    async with _aiosqlite.connect(db_path) as db:
        async with db.execute("SELECT ID, Name, NameE FROM TafseerName ORDER BY ID") as cur:
            rows = await cur.fetchall()
    return [{"id": r[0], "name_ar": r[1], "name_en": r[2]} for r in rows]


@app.get(
    "/api/v1/admin/disk-usage",
    summary="Show disk usage of key data directories",
    tags=["Admin"],
)
async def disk_usage(x_admin_key: str = Header(..., alias="X-Admin-Key")) -> Any:
    """Return sizes (MB) of the data directories and overall disk stats."""
    if x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=403, detail="Invalid admin key.")
    import shutil as _shutil
    from pathlib import Path as _Path

    def dir_size_mb(p: str) -> float:
        path = _Path(p)
        if not path.exists():
            return 0.0
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return round(total / 1024 / 1024, 2)

    disk = _shutil.disk_usage("/")
    # Check Railway volume at /data/ separately (different filesystem from root)
    try:
        vol = _shutil.disk_usage("/data")
        vol_total_gb  = round(vol.total / 1024**3, 2)
        vol_used_gb   = round(vol.used  / 1024**3, 2)
        vol_free_gb   = round(vol.free  / 1024**3, 2)
    except Exception:
        vol_total_gb = vol_used_gb = vol_free_gb = None

    # List top-level entries in /data/ with sizes
    data_entries: dict = {}
    data_path = _Path("/data")
    if data_path.exists():
        for child in sorted(data_path.iterdir()):
            try:
                if child.is_file():
                    data_entries[child.name] = round(child.stat().st_size / 1024**2, 2)
                elif child.is_dir():
                    total = sum(f.stat().st_size for f in child.rglob("*") if f.is_file())
                    data_entries[child.name + "/"] = round(total / 1024**2, 2)
            except Exception:
                data_entries[child.name] = "error"

    return {
        "root_disk_total_gb": round(disk.total / 1024**3, 2),
        "root_disk_used_gb":  round(disk.used  / 1024**3, 2),
        "root_disk_free_gb":  round(disk.free  / 1024**3, 2),
        "volume_total_gb":    vol_total_gb,
        "volume_used_gb":     vol_used_gb,
        "volume_free_gb":     vol_free_gb,
        "data_entries_mb":    data_entries,
        "tafsir_chroma_mb": dir_size_mb(settings.tafsir_chroma_dir),
        "tafsir_db_mb":     dir_size_mb(settings.tafsir_db_dir),
        "chroma_db_mb":     dir_size_mb(settings.chroma_persist_dir),
        "tafsir_chroma_dir": settings.tafsir_chroma_dir,
        "tafsir_db_dir":     settings.tafsir_db_dir,
    }


# ------------------------------------------------------------------
# Tafsir ask — AI-synthesised answer via TutorAgent
# ------------------------------------------------------------------

class TafsirAskRequest(BaseModel):
    question: str = Field(
        ..., min_length=1, max_length=1000,
        examples=["What does Ibn Kathir say about Ayat Al-Kursi?"],
    )
    language: str = Field(default="en", pattern="^(en|ar)$")


class TafsirAskResponse(BaseModel):
    answer: str
    sources: list[str]


@app.post(
    "/api/v1/tafsir/ask",
    response_model=TafsirAskResponse,
    summary="AI-synthesised tafsir Q&A",
    tags=["Tafsir"],
)
async def tafsir_ask(request: TafsirAskRequest) -> Any:
    """Answer a question about Quranic interpretation using GPT-4o grounded in
    Ibn Kathir and Al-Tabari commentary retrieved from the tafsir semantic index.

    Requires ``POST /api/v1/admin/ingest-tafsir`` to be run first.
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialised.")
    try:
        result = await agent.answer_tafsir(
            request.question, language=request.language
        )
    except Exception as exc:
        logger.exception("Error in /tafsir/ask")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return TafsirAskResponse(**result)


# ------------------------------------------------------------------
# Admin — Asbab al-Nuzul seed
# ------------------------------------------------------------------

@app.post("/api/v1/admin/ingest-asbab", tags=["Admin"])
async def ingest_asbab_data(x_admin_key: str = Header(..., alias="X-Admin-Key")):
    """Seed the ``asbab_sources`` metadata table with the two canonical books.

    The actual asbab entries (asbab_nuzul rows) must be imported separately
    via a data pipeline — this endpoint only ensures the source-book rows exist.
    """
    if x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=403, detail="Invalid admin key.")

    import aiosqlite as _aiosqlite

    sources = [
        (1, "\u0644\u0628\u0627\u0628 \u0627\u0644\u0646\u0642\u0648\u0644 \u0641\u064a \u0623\u0633\u0628\u0627\u0628 \u0627\u0644\u0646\u0632\u0648\u0644",
         "Asbab al-Nuzul",
         "\u0627\u0644\u0625\u0645\u0627\u0645 \u0627\u0644\u0633\u064a\u0648\u0637\u064a",
         911,
         "\u0627\u0644\u0643\u062a\u0627\u0628 \u0627\u0644\u0623\u0634\u0647\u0631 \u0641\u064a \u0623\u0633\u0628\u0627\u0628 \u0627\u0644\u0646\u0632\u0648\u0644"),
        (2, "\u0623\u0633\u0628\u0627\u0628 \u0627\u0644\u0646\u0632\u0648\u0644",
         "Asbab al-Nuzul - Al-Wahidi",
         "\u0627\u0644\u0625\u0645\u0627\u0645 \u0627\u0644\u0648\u0627\u062d\u062f\u064a",
         468,
         "\u0645\u0646 \u0623\u0648\u0627\u0626\u0644 \u0627\u0644\u0645\u0624\u0644\u0641\u0627\u062a \u0641\u064a \u0623\u0633\u0628\u0627\u0628 \u0627\u0644\u0646\u0632\u0648\u0644"),
    ]

    async with _aiosqlite.connect(settings.db_path) as db:
        await db.executemany(
            """
            INSERT OR IGNORE INTO asbab_sources
                (id, name_arabic, name_english, author_arabic, death_year_hijri, description)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            sources,
        )
        await db.commit()

        async with db.execute("SELECT COUNT(*) FROM asbab_nuzul") as cur:
            count = (await cur.fetchone())[0]

    return {
        "message": "Asbab sources initialised.",
        "sources_seeded": len(sources),
        "asbab_entries_in_db": count,
        "note": (
            "Source book metadata is ready. "
            "Populate asbab_nuzul entries via a data import script."
        ),
    }


# ------------------------------------------------------------------
# Admin — Qiraat metadata seed
# ------------------------------------------------------------------

@app.post("/api/v1/admin/ingest-qiraat", tags=["Admin"])
async def ingest_qiraat_data(x_admin_key: str = Header(..., alias="X-Admin-Key")):
    """Seed ``qurra`` and ``riwayat`` tables with the ten canonical readers
    and eight primary transmission chains.

    Variant texts (qiraat_variants rows) require a separate data import.
    """
    if x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=403, detail="Invalid admin key.")

    import aiosqlite as _aiosqlite

    qurra_data = [
        (1,  "\u0646\u0627\u0641\u0639 \u0627\u0644\u0645\u062f\u0646\u064a",          "Nafi al-Madani",     169, "\u0627\u0644\u0645\u062f\u064a\u0646\u0629 \u0627\u0644\u0645\u0646\u0648\u0631\u0629", 1),
        (2,  "\u0627\u0628\u0646 \u0643\u062b\u064a\u0631 \u0627\u0644\u0645\u0643\u064a",        "Ibn Kathir al-Makki",120, "\u0645\u0643\u0629 \u0627\u0644\u0645\u0643\u0631\u0645\u0629",   2),
        (3,  "\u0623\u0628\u0648 \u0639\u0645\u0631\u0648 \u0627\u0644\u0628\u0635\u0631\u064a",       "Abu Amr al-Basri",   154, "\u0627\u0644\u0628\u0635\u0631\u0629",         3),
        (4,  "\u0627\u0628\u0646 \u0639\u0627\u0645\u0631 \u0627\u0644\u0634\u0627\u0645\u064a",       "Ibn Amir al-Shami",  118, "\u0627\u0644\u0634\u0627\u0645",            4),
        (5,  "\u0639\u0627\u0635\u0645 \u0627\u0644\u0643\u0648\u0641\u064a",         "Asim al-Kufi",       127, "\u0627\u0644\u0643\u0648\u0641\u0629",           5),
        (6,  "\u062d\u0645\u0632\u0629 \u0627\u0644\u0643\u0648\u0641\u064a",         "Hamza al-Kufi",      156, "\u0627\u0644\u0643\u0648\u0641\u0629",           6),
        (7,  "\u0627\u0644\u0643\u0633\u0627\u0626\u064a \u0627\u0644\u0643\u0648\u0641\u064a",       "Al-Kisai al-Kufi",   189, "\u0627\u0644\u0643\u0648\u0641\u0629",           7),
        (8,  "\u0623\u0628\u0648 \u062c\u0639\u0641\u0631 \u0627\u0644\u0645\u062f\u0646\u064a",      "Abu Jafar al-Madani",130, "\u0627\u0644\u0645\u062f\u064a\u0646\u0629 \u0627\u0644\u0645\u0646\u0648\u0631\u0629", 8),
        (9,  "\u064a\u0639\u0642\u0648\u0628 \u0627\u0644\u062d\u0636\u0631\u0645\u064a",      "Yaqub al-Hadrami",   205, "\u0627\u0644\u0628\u0635\u0631\u0629",         9),
        (10, "\u062e\u0644\u0641 \u0627\u0644\u0628\u0632\u0627\u0631",           "Khalaf al-Bazzar",   229, "\u0628\u063a\u062f\u0627\u062f",           10),
    ]

    riwayat_data = [
        (1, 5, "hafs",   "\u062d\u0641\u0635 \u0639\u0646 \u0639\u0627\u0635\u0645",       "Hafs an Asim",
         "\u0627\u0644\u0631\u0648\u0627\u064a\u0629 \u0627\u0644\u0623\u0643\u062b\u0631 \u0627\u0646\u062a\u0634\u0627\u0631\u0627\u064b \u0641\u064a \u0627\u0644\u0639\u0627\u0644\u0645"),
        (2, 1, "warsh",  "\u0648\u0631\u0634 \u0639\u0646 \u0646\u0627\u0641\u0639",        "Warsh an Nafi",
         "\u0634\u0627\u0626\u0639\u0629 \u0641\u064a \u0634\u0645\u0627\u0644 \u0623\u0641\u0631\u064a\u0642\u064a\u0627"),
        (3, 1, "qaloon", "\u0642\u0627\u0644\u0648\u0646 \u0639\u0646 \u0646\u0627\u0641\u0639",      "Qaloon an Nafi",
         "\u0634\u0627\u0626\u0639\u0629 \u0641\u064a \u0644\u064a\u0628\u064a\u0627 \u0648\u062a\u0648\u0646\u0633"),
        (4, 5, "shouba", "\u0634\u0639\u0628\u0629 \u0639\u0646 \u0639\u0627\u0635\u0645",      "Shu'ba an Asim",
         "\u0631\u0648\u0627\u064a\u0629 \u0639\u0646 \u0639\u0627\u0635\u0645 \u0627\u0644\u0643\u0648\u0641\u064a"),
        (5, 3, "doori",  "\u0627\u0644\u062f\u0648\u0631\u064a \u0639\u0646 \u0623\u0628\u064a \u0639\u0645\u0631\u0648", "Ad-Duri",
         "\u0631\u0648\u0627\u064a\u0629 \u0639\u0646 \u0623\u0628\u064a \u0639\u0645\u0631\u0648 \u0627\u0644\u0628\u0635\u0631\u064a"),
        (6, 3, "soosi",  "\u0627\u0644\u0633\u0648\u0633\u064a \u0639\u0646 \u0623\u0628\u064a \u0639\u0645\u0631\u0648",  "As-Sousi",
         "\u0631\u0648\u0627\u064a\u0629 \u0639\u0646 \u0623\u0628\u064a \u0639\u0645\u0631\u0648 \u0627\u0644\u0628\u0635\u0631\u064a"),
        (7, 2, "bazzi",  "\u0627\u0644\u0628\u0632\u064a \u0639\u0646 \u0627\u0628\u0646 \u0643\u062b\u064a\u0631",  "Al-Bazzi",
         "\u0631\u0648\u0627\u064a\u0629 \u0639\u0646 \u0627\u0628\u0646 \u0643\u062b\u064a\u0631 \u0627\u0644\u0645\u0643\u064a"),
        (8, 2, "qumbul", "\u0642\u0646\u0628\u0644 \u0639\u0646 \u0627\u0628\u0646 \u0643\u062b\u064a\u0631",   "Qunbul",
         "\u0631\u0648\u0627\u064a\u0629 \u0639\u0646 \u0627\u0628\u0646 \u0643\u062b\u064a\u0631 \u0627\u0644\u0645\u0643\u064a"),
    ]

    async with _aiosqlite.connect(settings.db_path) as db:
        await db.executemany(
            """
            INSERT OR IGNORE INTO qurra
                (id, name_arabic, name_english, death_year_hijri, city, rank_order)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            qurra_data,
        )
        await db.executemany(
            """
            INSERT OR IGNORE INTO riwayat
                (id, qari_id, code, name_arabic, name_english, description)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            riwayat_data,
        )
        await db.commit()

        async with db.execute("SELECT COUNT(*) FROM qiraat_variants") as cur:
            variants_count = (await cur.fetchone())[0]

    return {
        "message": "Qiraat metadata seeded successfully.",
        "qurra_seeded": len(qurra_data),
        "riwayat_seeded": len(riwayat_data),
        "qiraat_variants_in_db": variants_count,
        "note": (
            "Reader and riwaya tables are ready. "
            "Populate qiraat_variants via a data import script."
        ),
    }
