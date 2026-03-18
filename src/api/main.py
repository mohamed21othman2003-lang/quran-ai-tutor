"""FastAPI application — Quran AI Tutor MVP.

Endpoints
---------
POST /api/v1/chat                  — Tajweed Q&A (RAG + GPT-4o)
GET  /api/v1/rules                 — List all Tajweed rules in the knowledge base
POST /api/v1/quiz                  — Generate a multiple-choice quiz for a given rule
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
    """Generate a multiple-choice quiz question for the specified Tajweed rule.

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
        from src.tafsir.store import TafsirStore

        ensure_database()
        store = TafsirStore()
        chunks = store.build_collection()
        _tafsir_ingest_state = {"status": "done", "chunks": chunks, "error": ""}
        logger.info("Background tafsir ingest complete: %d chunks.", chunks)
    except Exception as exc:
        logger.exception("Background tafsir ingest failed")
        _tafsir_ingest_state = {"status": "error", "chunks": 0, "error": str(exc)}


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

    loop = asyncio.get_running_loop()
    background_tasks.add_task(
        loop.run_in_executor, None, _run_tafsir_semantic_ingest
    )
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
