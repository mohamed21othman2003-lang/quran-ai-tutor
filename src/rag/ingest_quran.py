"""Quran ingestion script.

Reads data/knowledge/quran_full.json, creates one Markdown file per Surah in
data/knowledge/quran/, and builds a dedicated ChromaDB collection (quran_verses)
for fast ayah lookup by the voice verification layer.

Usage:
    python -m src.rag.ingest_quran
"""

import json
import logging
import urllib.request
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from src.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Surah name mapping (Arabic) — 114 surahs
# ------------------------------------------------------------------

SURAH_NAMES: dict[int, str] = {
    1: "الفاتحة", 2: "البقرة", 3: "آل عمران", 4: "النساء", 5: "المائدة",
    6: "الأنعام", 7: "الأعراف", 8: "الأنفال", 9: "التوبة", 10: "يونس",
    11: "هود", 12: "يوسف", 13: "الرعد", 14: "إبراهيم", 15: "الحجر",
    16: "النحل", 17: "الإسراء", 18: "الكهف", 19: "مريم", 20: "طه",
    21: "الأنبياء", 22: "الحج", 23: "المؤمنون", 24: "النور", 25: "الفرقان",
    26: "الشعراء", 27: "النمل", 28: "القصص", 29: "العنكبوت", 30: "الروم",
    31: "لقمان", 32: "السجدة", 33: "الأحزاب", 34: "سبأ", 35: "فاطر",
    36: "يس", 37: "الصافات", 38: "ص", 39: "الزمر", 40: "غافر",
    41: "فصلت", 42: "الشورى", 43: "الزخرف", 44: "الدخان", 45: "الجاثية",
    46: "الأحقاف", 47: "محمد", 48: "الفتح", 49: "الحجرات", 50: "ق",
    51: "الذاريات", 52: "الطور", 53: "النجم", 54: "القمر", 55: "الرحمن",
    56: "الواقعة", 57: "الحديد", 58: "المجادلة", 59: "الحشر", 60: "الممتحنة",
    61: "الصف", 62: "الجمعة", 63: "المنافقون", 64: "التغابن", 65: "الطلاق",
    66: "التحريم", 67: "الملك", 68: "القلم", 69: "الحاقة", 70: "المعارج",
    71: "نوح", 72: "الجن", 73: "المزمل", 74: "المدثر", 75: "القيامة",
    76: "الإنسان", 77: "المرسلات", 78: "النبأ", 79: "النازعات", 80: "عبس",
    81: "التكوير", 82: "الانفطار", 83: "المطففين", 84: "الانشقاق", 85: "البروج",
    86: "الطارق", 87: "الأعلى", 88: "الغاشية", 89: "الفجر", 90: "البلد",
    91: "الشمس", 92: "الليل", 93: "الضحى", 94: "الشرح", 95: "التين",
    96: "العلق", 97: "القدر", 98: "البينة", 99: "الزلزلة", 100: "العاديات",
    101: "القارعة", 102: "التكاثر", 103: "العصر", 104: "الهمزة", 105: "الفيل",
    106: "قريش", 107: "الماعون", 108: "الكوثر", 109: "الكافرون", 110: "النصر",
    111: "المسد", 112: "الإخلاص", 113: "الفلق", 114: "الناس",
}

QURAN_JSON_URL = "https://raw.githubusercontent.com/risan/quran-json/master/data/quran.json"
QURAN_JSON_PATH = Path("data/knowledge/quran_full.json")
QURAN_DIR = Path("data/knowledge/quran")
COLLECTION_NAME = "quran_verses"
AYAH_GROUP_SIZE = 5   # ayahs per markdown chunk


# ------------------------------------------------------------------
# Step 1 — Download Quran JSON if missing
# ------------------------------------------------------------------

def ensure_quran_json() -> dict:
    if not QURAN_JSON_PATH.exists():
        logger.info("Downloading Quran JSON from %s …", QURAN_JSON_URL)
        QURAN_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(QURAN_JSON_URL, QURAN_JSON_PATH)
        logger.info("Saved to %s", QURAN_JSON_PATH)
    with open(QURAN_JSON_PATH, encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------
# Step 2 — Create Markdown files (one per Surah, chunked by 5 ayahs)
# ------------------------------------------------------------------

def create_markdown_files(quran_data: dict) -> int:
    QURAN_DIR.mkdir(parents=True, exist_ok=True)
    files_written = 0

    for surah_key in sorted(quran_data.keys(), key=int):
        surah_num = int(surah_key)
        verses = quran_data[surah_key]
        name = SURAH_NAMES.get(surah_num, f"سورة {surah_num}")

        lines = [
            f"rule_name: سورة {name}",
            f"",
            f"# سورة {name} (Surah {surah_num})",
            f"",
        ]

        # Group verses in blocks of AYAH_GROUP_SIZE
        for i in range(0, len(verses), AYAH_GROUP_SIZE):
            group = verses[i : i + AYAH_GROUP_SIZE]
            start = group[0]["verse"]
            end = group[-1]["verse"]
            lines.append(f"## آيات {start}–{end}")
            lines.append("")
            for v in group:
                lines.append(f"سورة {name} - آية {v['verse']}: {v['text']}")
            lines.append("")

        out_path = QURAN_DIR / f"surah_{surah_num:03d}.md"
        out_path.write_text("\n".join(lines), encoding="utf-8")
        files_written += 1

    logger.info("Wrote %d Surah markdown files to %s", files_written, QURAN_DIR)
    return files_written


# ------------------------------------------------------------------
# Step 3 — Build dedicated ChromaDB collection (individual ayahs)
# ------------------------------------------------------------------

def build_quran_collection(quran_data: dict) -> int:
    """Embed every individual ayah into the quran_verses ChromaDB collection.

    Uses individual ayahs (not groups) so similarity search can pinpoint
    the exact verse during voice verification.
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=settings.openai_api_key,
    )
    persist_dir = str(Path(settings.chroma_persist_dir))
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    # Collect all ayahs as Documents
    docs: list[Document] = []
    for surah_key in sorted(quran_data.keys(), key=int):
        surah_num = int(surah_key)
        name = SURAH_NAMES.get(surah_num, f"سورة {surah_num}")
        for v in quran_data[surah_key]:
            docs.append(Document(
                page_content=v["text"],
                metadata={
                    "chapter": surah_num,
                    "verse": v["verse"],
                    "surah_name": name,
                    "reference": f"{surah_num}:{v['verse']}",
                },
            ))

    logger.info("Embedding %d ayahs into '%s' collection …", len(docs), COLLECTION_NAME)

    # Build in batches of 500 to avoid hitting embedding rate limits
    BATCH = 500
    store: Chroma | None = None
    for i in range(0, len(docs), BATCH):
        batch = docs[i : i + BATCH]
        if store is None:
            store = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                collection_name=COLLECTION_NAME,
                persist_directory=persist_dir,
            )
        else:
            store.add_documents(batch)
        logger.info("  … embedded %d / %d ayahs", min(i + BATCH, len(docs)), len(docs))

    logger.info("quran_verses collection built with %d ayahs.", len(docs))
    return len(docs)


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    quran_data = ensure_quran_json()
    total_verses = sum(len(v) for v in quran_data.values())
    logger.info("Loaded %d surahs / %d ayahs.", len(quran_data), total_verses)

    create_markdown_files(quran_data)
    build_quran_collection(quran_data)
    logger.info("Done.")
