# Quran AI Tutor — Claude Code Instructions

## Project Overview
FastAPI backend for an AI-powered Tajweed tutor. Uses GPT-4o for reasoning, ChromaDB for vector storage, LangChain for the RAG pipeline, and SQLite for user auth and progress tracking.

## Stack
- **API**: FastAPI (async)
- **LLM**: GPT-4o via OpenAI SDK
- **Embeddings**: text-embedding-3-small
- **Vector DB**: ChromaDB (local persistent)
- **Orchestration**: LangChain
- **Auth**: JWT (python-jose) + bcrypt password hashing
- **Database**: SQLite via aiosqlite
- **Config**: python-dotenv, Pydantic Settings
- **Languages**: Arabic + English (respond in user's language)

## Endpoints

### Tajweed AI
| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| POST | /api/v1/chat | No | Tajweed Q&A via RAG + GPT-4o |
| GET | /api/v1/rules | No | List all Tajweed rules from knowledge base |
| POST | /api/v1/quiz | No | Generate a quiz question on a given rule |

### Auth
| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| POST | /api/v1/auth/register | No | Create account → returns JWT token |
| POST | /api/v1/auth/login | No | Authenticate → returns JWT token |

### Progress
| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| POST | /api/v1/progress | Bearer JWT | Save learning event (chat/quiz + score) |
| GET | /api/v1/progress/{user_id} | Bearer JWT | Get history and weak rules |

### Admin
| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| POST | /api/v1/admin/ingest | X-Admin-Key header | Re-ingest knowledge base into ChromaDB |

## Project Structure
```
quran-ai-mvp/
├── CLAUDE.md
├── .env.example
├── requirements.txt
├── data/
│   ├── knowledge/        # Tajweed source documents (.txt / .md)
│   ├── chroma_db/        # ChromaDB vector store (auto-created)
│   └── quran_tutor.db    # SQLite database (auto-created on startup)
├── docs/
│   ├── PRD.md
│   └── SPEC.md
├── frontend/
│   └── index.html        # Single-file UI (Arabic/English, served at GET /)
└── src/
    ├── config.py          # Pydantic Settings (all env vars)
    ├── api/
    │   └── main.py        # FastAPI app, CORS, lifespan, all routers
    ├── rag/
    │   └── pipeline.py    # RAG pipeline (load → chunk → embed → store → retrieve)
    ├── agents/
    │   └── tutor_agent.py # LangChain LCEL agent with Tajweed system prompt
    ├── auth/
    │   ├── database.py    # aiosqlite connection + table init
    │   ├── jwt.py         # bcrypt hashing + JWT encode/decode
    │   └── router.py      # /register and /login endpoints
    ├── progress/
    │   └── router.py      # /progress POST + GET endpoints
    └── voice/             # Reserved for future TTS/STT
```

## Coding Conventions
- All FastAPI route functions must be `async def`
- Use Pydantic v2 models for all request/response bodies
- Secrets loaded exclusively from `.env` via `pydantic-settings`
- Never hard-code API keys or JWT secrets
- All docstrings in English; user-facing text supports Arabic + English
- Progress scores are floats in [0.0, 1.0]

## Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Run dev server
uvicorn src.api.main:app --reload --port 8000

# Populate vector store (run once, or after adding knowledge files)
python -m src.rag.pipeline
```

## Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| OPENAI_API_KEY | — | Required. OpenAI API key |
| CHROMA_PERSIST_DIR | data/chroma_db | ChromaDB storage path |
| KNOWLEDGE_DIR | data/knowledge | Tajweed document directory |
| TOP_K | 3 | Chunks retrieved per query |
| LOG_LEVEL | INFO | Logging verbosity |
| JWT_SECRET | change-me | Secret key for JWT signing |
| JWT_EXPIRE_MINUTES | 1440 | Token lifetime (default 24h) |
| DB_PATH | data/quran_tutor.db | SQLite file path |
| ADMIN_API_KEY | change-me-admin-key | Key for POST /api/v1/admin/ingest |
