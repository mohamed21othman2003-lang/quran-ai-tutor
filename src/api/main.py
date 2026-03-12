"""FastAPI application — Quran AI Tutor MVP.

Endpoints
---------
POST /api/v1/chat              — Tajweed Q&A (RAG + GPT-4o)
GET  /api/v1/rules             — List all Tajweed rules in the knowledge base
POST /api/v1/quiz              — Generate a multiple-choice quiz for a given rule
POST /api/v1/auth/register     — Create account, returns JWT
POST /api/v1/auth/login        — Authenticate, returns JWT
POST /api/v1/progress          — Save a learning event (auth required)
GET  /api/v1/progress/{uid}    — Get user history + weak rules (auth required)
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.agents.tutor_agent import TutorAgent
from src.auth.database import init_db
from src.auth.router import router as auth_router
from src.config import settings
from src.progress.router import router as progress_router
from src.rag.pipeline import RAGPipeline

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

logging.basicConfig(level=settings.log_level.upper())
logger = logging.getLogger(__name__)

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

# ------------------------------------------------------------------
# CORS
# ------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Frontend
# ------------------------------------------------------------------

FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend"

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse(str(FRONTEND_DIR / "index.html"))

# ------------------------------------------------------------------
# Routers
# ------------------------------------------------------------------

app.include_router(auth_router)
app.include_router(progress_router)

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
