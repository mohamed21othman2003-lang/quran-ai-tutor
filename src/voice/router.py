"""Voice correction endpoint — Quran recitation transcription + Tajweed error detection.

Uses tarteel-ai/whisper-base-ar-quran (fine-tuned Whisper for Quranic Arabic).
Model is downloaded on first request (~290 MB) and cached for subsequent calls.
"""

import difflib
import io
import json
import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.config import settings
from src.rag.quran_verifier import QuranVerifier

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/voice", tags=["Voice"])

# Lazy singletons — loaded on first request so startup stays fast
_asr_pipeline: Any = None
_ASR_MODEL = "tarteel-ai/whisper-base-ar-quran"
_quran_verifier: QuranVerifier | None = None


def _get_verifier() -> QuranVerifier:
    global _quran_verifier
    if _quran_verifier is None:
        _quran_verifier = QuranVerifier()
    return _quran_verifier


_report_llm: ChatOpenAI | None = None


def _get_report_llm() -> ChatOpenAI:
    global _report_llm
    if _report_llm is None:
        _report_llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.3,
            max_tokens=1500,
            openai_api_key=settings.openai_api_key,
        )
    return _report_llm

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
    match_score: float              # 0.0 – 1.0
    is_correct: bool                # True if match_score >= 0.85
    errors: list[RecitationError]
    # Quran verification fields
    quran_found: bool               # True if expected_text matched a known ayah
    quran_reference: str            # e.g. "2:255"
    quran_surah: str                # Arabic surah name
    quran_canonical: str            # exact Quran text from DB
    quran_mismatch: bool            # True if transcription contradicts canonical text
    mismatch_detail: str            # explanation when quran_mismatch is True


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


def _ffmpeg_exe() -> str:
    """Return path to ffmpeg binary — prefers system ffmpeg, falls back to imageio-ffmpeg bundle."""
    import shutil
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise RuntimeError(
            "ffmpeg not found. Install ffmpeg system-wide or add imageio-ffmpeg to requirements."
        )


def _to_wav(file_bytes: bytes, original_filename: str = "audio") -> bytes:
    """Convert any audio format to 16 kHz mono WAV using ffmpeg.

    Uses two temp files (input + output) so ffmpeg can seek the container.
    Returns the raw WAV bytes.
    """
    suffix = Path(original_filename).suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as src:
        src.write(file_bytes)
        src_path = src.name

    dst_path = src_path + ".wav"
    try:
        result = subprocess.run(
            [
                _ffmpeg_exe(),
                "-y",            # overwrite output without asking
                "-i", src_path,  # input file
                "-ar", "16000",  # resample to 16 kHz
                "-ac", "1",      # mono
                "-f", "wav",     # force WAV container
                dst_path,
            ],
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode(errors="replace"))
        return Path(dst_path).read_bytes()
    finally:
        Path(src_path).unlink(missing_ok=True)
        Path(dst_path).unlink(missing_ok=True)


def _load_audio(file_bytes: bytes, filename: str = "audio") -> np.ndarray:
    """Decode audio bytes to a 16 kHz mono float32 numpy array.

    For WAV/FLAC files soundfile reads directly.
    For WebM/MP3/OGG/M4A, ffmpeg converts to WAV first, then soundfile reads.
    """
    import librosa

    # Detect WebM/Matroska by magic bytes (0x1A 0x45 0xDF 0xA3)
    # Also convert MP3 (ID3 / 0xFF 0xFB), MP4/M4A (ftyp box), OGG (OggS)
    # and any format librosa can't open directly.
    _NEEDS_FFMPEG = {b"\x1a\x45\xdf\xa3", b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"}
    magic = file_bytes[:4]
    needs_convert = (
        any(file_bytes.startswith(sig) for sig in _NEEDS_FFMPEG)
        or magic[4:8] == b"ftyp"           # MP4/M4A
        or file_bytes[:4] == b"OggS"       # OGG
        or filename.lower().endswith((".webm", ".mp3", ".m4a", ".ogg", ".opus"))
    )

    if needs_convert:
        logger.debug("Converting %s via ffmpeg before loading", filename)
        try:
            wav_bytes = _to_wav(file_bytes, filename)
        except FileNotFoundError:
            raise RuntimeError(
                "ffmpeg not found. Install ffmpeg to support WebM/MP3/OGG audio."
            )
        audio, _ = librosa.load(io.BytesIO(wav_bytes), sr=16000, mono=True)
    else:
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

    # --- Validate expected_text against Quran DB BEFORE running ASR ---
    # This gives instant 422 rejection without loading the heavy ASR model.
    quran_info: dict = {
        "quran_found": False, "quran_reference": "", "quran_surah": "",
        "quran_canonical": "", "quran_mismatch": False, "mismatch_detail": "",
    }
    canonical_match: dict | None = None  # populated if expected_text is a valid ayah

    if expected_text.strip():
        verifier = _get_verifier()
        if verifier.is_populated():
            canonical_match = verifier.find_closest_ayah(expected_text)
            if canonical_match is None:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "code": "AYAH_NOT_FOUND",
                        "en": "The provided text was not found in the Quran verse database. "
                              "Please enter a valid ayah.",
                        "ar": "لم يُعثر على النص المُدخَل في قاعدة بيانات الآيات القرآنية. "
                              "يُرجى إدخال آية قرآنية صحيحة.",
                    },
                )
        else:
            logger.warning(
                "Quran verifier not populated — comparing vs user input. "
                "Run `python -m src.rag.ingest_quran` to enable canonical comparison."
            )

    # --- Decode audio ---
    try:
        audio_array = _load_audio(file_bytes, filename=audio.filename or "audio.webm")
    except Exception as exc:
        logger.exception("Audio decode failed")
        raise HTTPException(
            status_code=422,
            detail=f"Could not decode audio file: {exc}",
        ) from exc

    # --- Reject silent / empty audio (Whisper crashes on all-zero input) ---
    MIN_DURATION_S = 0.5   # at least half a second
    MIN_RMS = 1e-4          # non-trivial energy (avoids recording noise floor)
    if len(audio_array) < int(MIN_DURATION_S * 16000):
        raise HTTPException(status_code=422, detail="Audio is too short (minimum 0.5 s).")
    rms = float(np.sqrt(np.mean(audio_array ** 2)))
    if rms < MIN_RMS:
        raise HTTPException(status_code=422, detail="No speech detected in the audio. Please record your recitation and try again.")

    # --- Transcribe ---
    try:
        asr = _get_pipeline()
        result = asr({"raw": audio_array, "sampling_rate": 16000})
        transcribed = result["text"].strip()
    except IndexError as exc:
        # Whisper model crashes on near-silent / too-short audio (index out of bounds)
        logger.warning("ASR IndexError — likely silent audio: %s", exc)
        raise HTTPException(
            status_code=422,
            detail="No speech detected in the audio. Please record your recitation and try again.",
        ) from exc
    except Exception as exc:
        logger.exception("ASR inference failed")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc

    # --- Build canonical comparison target ---
    compare_target: str = expected_text  # fallback when verifier not populated

    if canonical_match is not None:
        canonical = canonical_match["canonical_text"]
        compare_target = canonical

        # Mismatch: diacritic-insensitive diff of transcription vs Mushaf
        norm_t = _normalize(transcribed)
        norm_c = _normalize(canonical)
        mismatch = bool(norm_t and norm_c and norm_t != norm_c)

        quran_info = {
            "quran_found":     True,
            "quran_reference": canonical_match["reference"],
            "quran_surah":     canonical_match["surah_name"],
            "quran_canonical": canonical,
            "quran_mismatch":  mismatch,
            "mismatch_detail": (
                f"Transcription does not match canonical text for "
                f"Surah {canonical_match['surah_name']} ({canonical_match['reference']}). "
                f"Canonical: {canonical}"
            ) if mismatch else "",
        }

    # --- Score against canonical (or user input as fallback) ---
    if compare_target.strip():
        score, errors = _compare(transcribed, compare_target)
    else:
        score, errors = 1.0, []

    # Return canonical text as expected_text so the frontend always shows real Mushaf
    display_expected = quran_info.get("quran_canonical") or expected_text

    return VoiceCheckResponse(
        transcribed_text=transcribed,
        expected_text=display_expected,
        match_score=score,
        is_correct=score >= 0.85,
        errors=errors,
        **quran_info,
    )


# ------------------------------------------------------------------
# Recitation Report
# ------------------------------------------------------------------

_REPORT_PROMPT = """\
You are an expert Tajweed teacher. Generate a detailed, encouraging recitation \
improvement report for a student.

Student recited: {transcribed}
Expected ayah:   {expected}
Match score:     {score_pct}%
Errors detected: {errors}

Instructions:
- Analyse the student's performance honestly but encouragingly.
- Explain each error type and why it matters for correct recitation.
- Give 3–5 specific, actionable improvement tips.
- Suggest a practice routine.
- Assign a letter grade: A (≥90%), B (≥75%), C (≥60%), D (≥40%), F (<40%).
- Respond in {language}.

Return a JSON object with exactly these keys:
  "grade":     letter grade string (A/B/C/D/F)
  "overall":   paragraph — overall assessment of the recitation
  "error_analysis": paragraph — analysis of specific errors
  "tips":      array of 3–5 actionable tip strings
  "practice":  paragraph — suggested practice routine

Return only valid JSON, no markdown fences."""


class ErrorItem(BaseModel):
    type: str
    expected: str
    got: str


class ReportRequest(BaseModel):
    transcribed: str = Field(..., min_length=1)
    expected: str = Field(default="")
    errors: list[ErrorItem] = Field(default_factory=list)
    score: float = Field(..., ge=0.0, le=1.0)
    language: str = Field(default="en", pattern="^(en|ar)$")


class ReportResponse(BaseModel):
    grade: str
    overall: str
    error_analysis: str
    tips: list[str]
    practice: str


@router.post("/report", response_model=ReportResponse, summary="Generate recitation report")
async def recitation_report(request: ReportRequest) -> Any:
    """Generate a full pedagogical recitation report using GPT-4o.

    Takes the transcription, expected ayah, detected errors, and match score,
    and returns a structured report with a grade, error analysis, improvement
    tips, and a practice routine.
    """
    errors_str = (
        "\n".join(
            f"  - {e.type}: expected '{e.expected}', got '{e.got}'"
            for e in request.errors
        )
        if request.errors
        else "  None detected."
    )
    lang_label = "Arabic" if request.language == "ar" else "English"
    prompt = _REPORT_PROMPT.format(
        transcribed=request.transcribed or "(no transcription)",
        expected=request.expected or "(no expected text provided)",
        score_pct=round(request.score * 100),
        errors=errors_str,
        language=lang_label,
    )

    try:
        llm = _get_report_llm()
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
    except Exception as exc:
        logger.exception("LLM call failed in /voice/report")
        raise HTTPException(status_code=500, detail=f"AI error: {exc}") from exc

    try:
        data = json.loads(raw)
        return ReportResponse(
            grade=data.get("grade", "—"),
            overall=data.get("overall", ""),
            error_analysis=data.get("error_analysis", ""),
            tips=data.get("tips", []),
            practice=data.get("practice", ""),
        )
    except json.JSONDecodeError:
        logger.warning("GPT-4o returned non-JSON for report; raw: %.300s", raw)
        return ReportResponse(
            grade="—",
            overall=raw[:800],
            error_analysis="",
            tips=[],
            practice="",
        )
