# Quran AI Tutor — Project Instructions for Claude Code

## Project Overview
AI-powered Quran learning platform with Tajweed correction, voice recitation check,
memorization testing, tafsir search, and AI quiz.
Production URL: https://web-production-fa50b.up.railway.app

---

## Tech Stack
- Backend: Python 3.11 + FastAPI + Uvicorn
- AI: GPT-4o (chat/eval/quiz) + whisper-1 (STT via API) + text-embedding-3-small (256 dims)
- Vector DB: ChromaDB (persistent) — two collections: tajweed RAG + quran verses
- Tafsir DB: SQLite (tafaseer.db) — Ibn Kathir + Al-Tabari, 12,472 rows
- Auth: JWT (python-jose) + bcrypt
- Deployment: Railway Cloud — Nixpacks + Persistent Volume at /data/

---

## Project Structure

quran-ai-mvp/
├── src/
│   ├── api/main.py           # FastAPI app + all admin endpoints
│   ├── voice/router.py       # Whisper STT + voice check + memorization
│   ├── tafsir/
│   │   ├── router.py         # Tafsir search endpoint
│   │   ├── store.py          # FAISS index (tafsir semantic search)
│   │   └── database.py       # SQLite tafaseer.db queries
│   ├── rag/
│   │   ├── pipeline.py       # Tajweed RAG (ChromaDB)
│   │   └── quran_verifier.py # Quran verses ChromaDB collection
│   ├── search/router.py      # Semantic ayah search
│   ├── tajweed/router.py     # Tajweed auto-detect
│   ├── auth/                 # JWT + user SQLite DB
│   ├── progress/router.py    # User progress tracking
│   └── config.py             # Settings via pydantic-settings
├── frontend/index.html       # Single-file SPA — all JS/CSS inline, no framework
├── data/                     # Railway Volume mount point — NEVER delete
└── requirements.txt

---

## CRITICAL: Railway Volume & Paths

The Railway Volume is mounted at /data/ (absolute path).
All persistent data MUST use absolute paths under /data/.

| Data                        | Absolute Path          |
|-----------------------------|------------------------|
| SQLite DB (users/progress)  | /data/quran_tutor.db   |
| ChromaDB (RAG + Quran)      | /data/chroma_db/       |
| Tafsir SQLite               | /data/tafsir/          |
| Tafsir FAISS index          | /data/tafsir_faiss/    |

NEVER use relative paths like "data/chroma_db" — they point to ephemeral
container storage and get wiped on every deploy.

Volume limit: 500MB (currently ~421MB used: 283MB ChromaDB + 138MB tafaseer.db).
The tafsir FAISS index needs ~50MB extra — upgrade Volume to 2GB before building it.

---

## CRITICAL: AI Model Rules

| Task                  | Model                    | Notes                                      |
|-----------------------|--------------------------|--------------------------------------------|
| Voice transcription   | whisper-1 (OpenAI API)   | NEVER use local transformers model         |
| Chat + RAG answers    | gpt-4o                   | temperature=0.3, max_tokens=1024           |
| Voice evaluation      | gpt-4o                   | temperature=0.2                            |
| Quiz generation       | gpt-4o                   | Switch to gpt-4o-mini at scale (90% cheaper) |
| All embeddings        | text-embedding-3-small   | Always use dimensions=256                  |

Local Whisper model (tarteel-ai/whisper-base-ar-quran) is PERMANENTLY DISABLED.
It fails on Railway due to torch < 2.6 + missing safetensors format.
Transcription is handled by _transcribe_with_openai() in src/voice/router.py.
Do NOT reintroduce the local model or transformers pipeline under any circumstances.

---

## Admin Endpoints

All require header: X-Admin-Key: <value of ADMIN_API_KEY env var>
Run these after first deploy — most persist on Volume and survive future deploys.

# Build Tajweed RAG — 89 chunks, 25 rules (~30 seconds)
POST /api/v1/admin/ingest

# Index 6,236 Quran ayahs into ChromaDB (~2 minutes)
POST /api/v1/admin/ingest-quran

# Download tafaseer.db — Ibn Kathir + Tabari (~1 minute)
POST /api/v1/admin/ingest-tafsir

# Build FAISS semantic index for tafsir — runs in background, returns 202 immediately
POST /api/v1/admin/ingest-tafsir-semantic

# Poll tafsir ingest progress
GET /api/v1/admin/tafsir-ingest-status

# Reset corrupted tafsir FAISS index (wipes /data/tafsir_faiss/ contents)
DELETE /api/v1/admin/reset-tafsir-index

# Check Volume disk usage
GET /api/v1/admin/disk-usage

After every new Railway deploy:
- Run ingest (#1) only if tajweed knowledge files in data/knowledge/ changed.
- Runs #2 and #3 persist on Volume and do NOT need to be re-run.

---

## Known Issues & Gotchas

### ChromaDB compaction error ("Failed to apply logs to the metadata segment")
Cause: stale WAL/segment files from a previous failed write.
Fix:
  1. DELETE /api/v1/admin/reset-tafsir-index
  2. POST /api/v1/admin/ingest-tafsir-semantic

### Tafsir semantic ingest always times out at 5 minutes (Railway HTTP limit)
The endpoint returns 202 Accepted immediately and runs in a background thread.
Use GET /api/v1/admin/tafsir-ingest-status to poll until status = "done".
If status stays "idle" forever after POST: the background task is broken.
Check src/api/main.py — the fix is background_tasks.add_task(_run_tafsir_semantic_ingest)
passing the function DIRECTLY, not via loop.run_in_executor().

### TAFSIR_CHROMA_DIR environment variable
This Railway dashboard variable used to override config.py and pointed to an
ephemeral path. It is now intentionally ignored in src/tafsir/store.py.
Do NOT re-add TAFSIR_CHROMA_DIR to Railway environment variables.

### Arabic strings appearing as garbled characters (Ø symbols)
Cause 1 (backend): Arabic string literals stored with wrong encoding.
Fix: Use Unicode escapes (\uXXXX) for all Arabic strings in Python files.
See _generate_memorization_tips() in src/voice/router.py for the pattern.

Cause 2 (frontend): Using innerHTML to insert Arabic API responses.
Fix: Always use element.textContent = text for Arabic strings from the API.

### applyI18n null reference error on page load
Cause: A DOM element referenced in applyI18n() was removed from the HTML
but the JS function still tries to set its textContent.
Fix: Wrap every line in applyI18n() with a null check:
  const el = $('element-id'); if (el) el.textContent = t('key');

### Volume full (500MB limit)
Current usage: ~421MB. Only 79MB free.
Check: GET /api/v1/admin/disk-usage (shows /data/ volume stats separately from root fs).
To free space: the tafsir FAISS index at /data/tafsir_faiss/ can be deleted safely
and rebuilt with POST /api/v1/admin/ingest-tafsir-semantic.

---

## Cost Reference

| Operation                        | Approximate Cost              |
|----------------------------------|-------------------------------|
| Voice check (1 request)          | ~$0.006 Whisper + ~$0.0001 embedding |
| Quiz (10 questions, GPT-4o)      | ~$0.015                       |
| Chat message (GPT-4o)            | ~$0.002–0.005                 |
| Tafsir FAISS full rebuild        | ~$2–5 (13,900 embeddings)     |
| Monthly at <50 users             | < $10                         |
| Monthly at >1000 users           | > $200 — switch to gpt-4o-mini |

---

## Phase 3 Features

### New Features Being Added
- القراءات العشر (Ten Qiraat readings) — new module: src/qiraat/
- أسباب النزول (Asbab al-Nuzul) — new module: src/asbab/
- Expanded Tafsir — from 2 books to 5 books (add: Al-Baghawi ID=3, Al-Qurtubi ID=5, Al-Sadi ID=7)

### New Database Tables (added to quran_tutor.db)
- qurra: Ten Quranic readers metadata
- riwayat: Eight canonical transmission chains
- qiraat_variants: Reading differences per verse
- asbab_sources: Books/sources for revelation contexts
- asbab_nuzul: Revelation context entries per verse

### New Routes
- /api/v1/qiraat/verse/{surah}:{ayah} — get all readings for a verse
- /api/v1/qiraat/surah/{surah_id} — get all differences in a surah
- /api/v1/asbab/verse/{surah}:{ayah} — get asbab for a verse
- /api/v1/asbab/surah/{surah_id} — list surahs with asbab count

### Data Sources
- Qiraat data: scraped from nquran.com (see uloom-quran scripts for reference)
- Asbab data: from tafseer-sqlite-db (same DB already downloaded at /data/tafsir/tafaseer.db)
- Both are populated via new admin endpoints after deploy

---

## Coding Standards
- Always use async/await in FastAPI endpoints
- Always use Pydantic models for request and response schemas
- Never hardcode API keys — always use settings from src/config.py
- All user-facing responses must support both Arabic and English
- For Arabic text in HTML: always set dir="rtl" on the element
- For Arabic text in docx generation: always set rightToLeft=True and bidirectional=True
- Rate limiting is configured via slowapi — respect 10 req/min for voice endpoints
- When adding new admin endpoints, always check X-Admin-Key against settings.admin_api_key
