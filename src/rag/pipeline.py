"""RAG pipeline: load Tajweed knowledge → chunk → embed → store in ChromaDB → retrieve."""

import logging
import re
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "tajweed_knowledge"


class RAGPipeline:
    """Manages the full RAG lifecycle: ingest, store, and retrieve."""

    def __init__(self) -> None:
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=settings.openai_api_key,
        )
        self.vector_store: Chroma | None = None
        # Ensure the persist directory exists so ChromaDB never fails on a missing path
        Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def load_documents(self) -> List[Document]:
        """Load all .txt and .md files from the knowledge directory."""
        knowledge_path = Path(settings.knowledge_dir)
        if not knowledge_path.exists():
            raise FileNotFoundError(f"Knowledge directory not found: {knowledge_path}")

        # quran/ and quran_full.json belong to the Quran verses collection, not
        # the Tajweed knowledge base — exclude them to avoid polluting tajweed_knowledge.
        _EXCLUDE = ["quran/**", "quran_full.json"]

        docs: List[Document] = []
        for pattern in ("**/*.md", "**/*.txt"):
            loader = DirectoryLoader(
                str(knowledge_path),
                glob=pattern,
                exclude=_EXCLUDE,
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"},
                show_progress=True,
            )
            docs.extend(loader.load())

        if not docs:
            raise ValueError(
                f"No .md or .txt files found in {knowledge_path}. "
                "Add knowledge files before ingesting."
            )
        logger.info("Loaded %d documents from %s", len(docs), knowledge_path)
        return docs

    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        """Split documents into overlapping chunks and inject rule_name metadata."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        chunks = splitter.split_documents(docs)

        # Carry the last seen rule_name forward into each chunk's metadata
        current_rule: str | None = None
        for chunk in chunks:
            match = re.search(r"^rule_name:\s*(.+)$", chunk.page_content, re.MULTILINE)
            if match:
                current_rule = match.group(1).strip()
            if current_rule:
                chunk.metadata["rule_name"] = current_rule

        logger.info("Split into %d chunks", len(chunks))
        return chunks

    def build_vector_store(self, chunks: List[Document]) -> Chroma:
        """Embed chunks and persist to ChromaDB."""
        persist_dir = str(Path(settings.chroma_persist_dir))
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=persist_dir,
        )
        logger.info("Vector store built and persisted to %s", persist_dir)
        return vector_store

    def ingest(self) -> None:
        """Full ingestion pipeline: load → chunk → embed → store."""
        # Delete existing collection to avoid duplicate chunks on re-ingest
        try:
            existing = self.get_vector_store()
            existing._client.delete_collection(COLLECTION_NAME)
            self.vector_store = None
            logger.info("Deleted existing vector store collection before re-ingest.")
        except Exception:
            logger.warning("Could not delete existing collection — proceeding anyway.")
        docs = self.load_documents()
        chunks = self.chunk_documents(docs)
        self.vector_store = self.build_vector_store(chunks)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_vector_store(self) -> Chroma:
        """Load existing ChromaDB vector store (lazy, cached)."""
        if self.vector_store is None:
            self.vector_store = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=str(Path(settings.chroma_persist_dir)),
            )
        return self.vector_store

    def is_populated(self) -> bool:
        """Return True if the vector store contains at least one document."""
        try:
            store = self.get_vector_store()
            return store._collection.count() > 0
        except Exception:
            return False

    def retrieve(self, query: str) -> List[Document]:
        """Return top-K most relevant chunks for *query*.

        Returns an empty list (instead of raising) when the store is empty
        or an error occurs, so callers can handle the no-context case gracefully.
        """
        if not self.is_populated():
            logger.warning("Vector store is empty — skipping retrieval for query: %.80s", query)
            return []
        try:
            store = self.get_vector_store()
            results = store.similarity_search(query, k=settings.top_k)
            logger.debug("Retrieved %d chunks for query: %.80s", len(results), query)
            return results
        except Exception:
            logger.exception("Retrieval failed for query: %.80s", query)
            return []

    def get_all_rule_names(self) -> List[str]:
        """Return unique rule names from ChromaDB.

        Returns an empty list when the store is empty or unavailable.
        """
        if not self.is_populated():
            logger.warning("Vector store is empty — no rule names available.")
            return []
        try:
            store = self.get_vector_store()
            collection = store._collection
            items = collection.get(include=["metadatas", "documents"])

            seen: set = set()
            names: List[str] = []

            for meta, text in zip(
                items.get("metadatas", []), items.get("documents", [])
            ):
                rule = (meta or {}).get("rule_name")
                if not rule and text:
                    match = re.search(r"^rule_name:\s*(.+)$", text, re.MULTILINE)
                    if match:
                        rule = match.group(1).strip()
                if rule and rule not in seen:
                    names.append(rule)
                    seen.add(rule)

            return sorted(names)
        except Exception:
            logger.exception("Failed to read rule names from vector store.")
            return []


# ------------------------------------------------------------------
# CLI entry point: python -m src.rag.pipeline
# ------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline = RAGPipeline()
    pipeline.ingest()
    print("Ingestion complete.")
