# PRD: Quran AI Tutor Platform

## Problem
Millions of Muslims want to learn Quran Tajweed but lack access to qualified teachers.
Existing apps are limited — no personalized AI tutoring with voice feedback in Arabic.

## Solution
A full-stack AI platform that teaches Tajweed, evaluates recitation, tests memorization,
provides tafsir search, and quizzes learners — all powered by GPT-4o and Whisper.

## Target Users
- Arabic-speaking Muslims learning or improving Quran recitation (primary)
- Non-Arabic speakers learning Quran (secondary)
- Tajweed teachers wanting digital assessment tools

## Features (All Complete — Phase 1)

| Feature                  | Priority | Status   |
|--------------------------|----------|----------|
| Tajweed Q&A (RAG chat)   | P0       | ✅ Done  |
| Rules listing            | P0       | ✅ Done  |
| Quiz generation (AI)     | P0       | ✅ Done  |
| Voice recitation check   | P1       | ✅ Done  |
| Memorization testing     | P1       | ✅ Done  |
| Tajweed auto-detect      | P1       | ✅ Done  |
| Tafsir search            | P1       | ✅ Done  |
| Semantic ayah search     | P1       | ✅ Done  |
| User auth + progress     | P1       | ✅ Done  |
| Bilingual AR/EN UI       | P1       | ✅ Done  |

## Phase 2 (Planned)
- Admin dashboard for content and user management
- n8n workflow automation integration
- GPT-4o mini migration for cost reduction at scale
- Mobile app (React Native or Flutter)

## Success Metrics
- Voice transcription accuracy > 90% (Whisper API)
- Response time < 5 seconds for all endpoints
- Supports 6,236 indexed Quran ayahs
- Supports 89 Tajweed knowledge chunks across 25 rules
- Tafsir coverage: Ibn Kathir + Al-Tabari (12,472 rows)

## Production URL
https://web-production-fa50b.up.railway.app
