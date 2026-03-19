# Technical Spec: Quran AI Tutor

## Architecture
Client (SPA) → FastAPI → Routers → OpenAI APIs
                              ↓
                    ChromaDB (RAG + Quran)
                    SQLite (Users + Tafsir)
                    FAISS (Tafsir semantic)

## API Endpoints

### Auth
POST   /api/v1/auth/register   — register new user
POST   /api/v1/auth/login      — login, returns JWT token
GET    /api/v1/auth/me         — current user info

### Core Features
POST   /api/v1/chat            — Tajweed Q&A via RAG + GPT-4o
GET    /api/v1/rules           — list all Tajweed rules
POST   /api/v1/quiz            — generate AI quiz question (MCQ)
POST   /api/v1/voice/check     — check recitation word-by-word
POST   /api/v1/voice/memorization — identify ayah + score memorization
POST   /api/v1/tafsir/search   — search Ibn Kathir + Tabari
POST   /api/v1/search/ayah     — semantic search in 6,236 ayahs
POST   /api/v1/tajweed/detect  — detect Tajweed rules in Arabic text
GET    /api/v1/progress        — user progress and stats
POST   /api/v1/progress        — save session result

### Admin (X-Admin-Key required)
POST   /api/v1/admin/ingest                  — build Tajweed RAG
POST   /api/v1/admin/ingest-quran            — index Quran ayahs
POST   /api/v1/admin/ingest-tafsir           — download tafaseer.db
POST   /api/v1/admin/ingest-tafsir-semantic  — build FAISS index (async)
GET    /api/v1/admin/tafsir-ingest-status    — poll ingest progress
DELETE /api/v1/admin/reset-tafsir-index      — wipe FAISS index
GET    /api/v1/admin/disk-usage              — check Railway Volume usage

## RAG Pipeline
1. Load Tajweed knowledge base (Markdown files in data/knowledge/)
2. Chunk → Embed with text-embedding-3-small (dimensions=256)
3. Store in ChromaDB at /data/chroma_db/
4. On query: retrieve top-3 relevant chunks → inject in GPT-4o prompt

## Voice Pipeline
1. Browser records audio (WebM via MediaRecorder API)
2. ffmpeg converts WebM → WAV 16kHz mono
3. whisper-1 API transcribes Arabic recitation
4. Compare transcription vs expected ayah word-by-word
5. QuranVerifier validates against 6,236 indexed ayahs in ChromaDB

## Tafsir Search
- Mode: keyword (default) — SQLite FTS on tafaseer.db
- Tashkeel stripped from query before search
- Returns: Ibn Kathir + Al-Tabari results side-by-side

## Agent System Prompt
- Role: Expert Tajweed teacher
- Language: Respond in same language as query (AR/EN)
- Grounding: Only answer from retrieved RAG context
- Format: Clear, structured, with examples from Quran

## Data Model

TajweedRule:
  - id: str
  - name_ar: str
  - name_en: str
  - category: enum (noon_sakinah | meem_sakinah | madd | qalqala | waqf)
  - description: str
  - examples: List[str]

User:
  - id: int
  - username: str
  - hashed_password: str
  - created_at: datetime

ProgressSession:
  - user_id: int
  - rule_name: str
  - score: float (0.0–1.0)
  - session_type: enum (quiz | voice | chat)
  - created_at: datetime
