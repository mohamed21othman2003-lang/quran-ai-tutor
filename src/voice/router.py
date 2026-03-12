"""Voice correction endpoint — Quran recitation transcription + Tajweed error detection.

Uses tarteel-ai/whisper-base-ar-quran (fine-tuned Whisper for Quranic Arabic).
Model is downloaded on first request (~290 MB) and cached for subsequent calls.
"""

import difflib
import io
import logging
import re
from typing import Any

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/voice", tags=["Voice"])

# Lazy singleton — loaded on first request so startup stays fast
_asr_pipeline: Any = None
_ASR_MODEL = "tarteel-ai/whisper-base-ar-quran"

# Arabic diacritics (harakat) — stripped before comparison so missing
# tashkeel in user input doesn't count as an error
_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]")


# ------------------------------------------------------------------
# Response models
# ------------------------------------------------------------------

class RecitationError(BaseModel):
    type: str          # "substitution" | "missing_word" | "extra_word"
    expected: str      # what was expected (empty for extra_word)
    got: str           # what was recited (empty for missing_word)


class VoiceCheckResponse(BaseModel):
    transcribed_text: str
    expected_text: str
    match_score: float          # 0.0 – 1.0
    is_correct: bool            # True if match_score >= 0.85
    errors: list[RecitationError]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _get_pipeline():
    """Return the ASR pipeline, initialising it on first call."""
    global _asr_pipeline
    if _asr_pipeline is None:
        logger.info("Loading ASR model %s (first request — may take a moment)…", _ASR_MODEL)
        from transformers import pipeline as hf_pipeline
        _asr_pipeline = hf_pipeline(
            "automatic-speech-recognition",
            model=_ASR_MODEL,
            device=-1,  # CPU
        )
        logger.info("ASR model loaded.")
    return _asr_pipeline


def _load_audio(file_bytes: bytes) -> np.ndarray:
    """Decode audio bytes to a 16 kHz mono float32 numpy array.

    Supports WAV, FLAC, OGG natively via soundfile.
    MP3/M4A/WebM require ffmpeg to be installed (provided via nixpacks.toml).
    """
    import librosa
    audio, _ = librosa.load(io.BytesIO(file_bytes), sr=16000, mono=True)
    return audio.astype(np.float32)


def _normalize(text: str) -> str:
    """Strip diacritics, collapse whitespace, lowercase."""
    text = _DIACRITICS.sub("", text)
    return " ".join(text.split()).strip()


def _compare(transcribed: str, expected: str) -> tuple[float, list[RecitationError]]:
    """Word-level diff between transcribed and expected Arabic text.

    Returns (similarity_score, list_of_errors).
    Comparison is diacritic-insensitive.
    """
    norm_t = _normalize(transcribed).split()
    norm_e = _normalize(expected).split()

    matcher = difflib.SequenceMatcher(None, norm_e, norm_t, autojunk=False)
    score = round(matcher.ratio(), 3)

    errors: list[RecitationError] = []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            continue
        elif op == "replace":
            errors.append(RecitationError(
                type="substitution",
                expected=" ".join(norm_e[i1:i2]),
                got=" ".join(norm_t[j1:j2]),
            ))
        elif op == "delete":
            errors.append(RecitationError(
                type="missing_word",
                expected=" ".join(norm_e[i1:i2]),
                got="",
            ))
        elif op == "insert":
            errors.append(RecitationError(
                type="extra_word",
                expected="",
                got=" ".join(norm_t[j1:j2]),
            ))

    return score, errors


# ------------------------------------------------------------------
# Endpoint
# ------------------------------------------------------------------

@router.post("/check", response_model=VoiceCheckResponse, summary="Check Quran recitation")
async def voice_check(
    audio: UploadFile = File(..., description="Audio file (WAV, MP3, FLAC, OGG, WebM)"),
    expected_text: str = Form(
        default="",
        description="Expected Arabic ayah text. If omitted, only transcription is returned.",
    ),
) -> VoiceCheckResponse:
    """Transcribe a Quran recitation and compare it with the expected ayah.

    - Accepts any common audio format (WAV, MP3, FLAC, OGG, WebM).
    - Transcribes using **tarteel-ai/whisper-base-ar-quran** (fine-tuned for Quranic Arabic).
    - Compares word-by-word with `expected_text` (diacritic-insensitive).
    - Returns a match score and a list of substitution / missing / extra word errors.
    """
    # --- Read upload ---
    file_bytes = await audio.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Audio file is empty.")
    if len(file_bytes) > 25 * 1024 * 1024:  # 25 MB cap
        raise HTTPException(status_code=413, detail="Audio file exceeds 25 MB limit.")

    # --- Decode audio ---
    try:
        audio_array = _load_audio(file_bytes)
    except Exception as exc:
        logger.exception("Audio decode failed")
        raise HTTPException(
            status_code=422,
            detail=f"Could not decode audio file: {exc}",
        ) from exc

    # --- Transcribe ---
    try:
        asr = _get_pipeline()
        result = asr({"raw": audio_array, "sampling_rate": 16000})
        transcribed = result["text"].strip()
    except Exception as exc:
        logger.exception("ASR inference failed")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc

    # --- Compare ---
    if expected_text.strip():
        score, errors = _compare(transcribed, expected_text)
    else:
        # No expected text — return transcription only, score = 1.0
        score, errors = 1.0, []

    return VoiceCheckResponse(
        transcribed_text=transcribed,
        expected_text=expected_text,
        match_score=score,
        is_correct=score >= 0.85,
        errors=errors,
    )
