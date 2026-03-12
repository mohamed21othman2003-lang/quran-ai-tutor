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
            model="text-embedding-3-small",
            openai_api_key=settings.openai_api_key,
        )
        self.vector_store: Chroma | None = None

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def load_documents(self) -> List[Document]:
        """Load all .txt and .md files from the knowledge directory."""
        knowledge_path = Path(settings.knowledge_dir)
        if not knowledge_path.exists():
            raise FileNotFoundError(f"Knowledge directory not found: {knowledge_path}")

        docs: List[Document] = []
        for pattern in ("**/*.md", "**/*.txt"):
            loader = DirectoryLoader(
                str(knowledge_path),
                glob=pattern,
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

    def retrieve(self, query: str) -> List[Document]:
        """Return top-K most relevant chunks for *query*."""
        store = self.get_vector_store()
        results = store.similarity_search(query, k=settings.top_k)
        logger.debug("Retrieved %d chunks for query: %.80s", len(results), query)
        return results

    def get_all_rule_names(self) -> List[str]:
        """Return unique rule names from ChromaDB.

        Reads the ``rule_name`` metadata field when present (set on re-ingestion),
        and falls back to parsing ``rule_name:`` lines from chunk text so that
        existing data ingested without metadata still works.
        """
        store = self.get_vector_store()
        collection = store._collection
        items = collection.get(include=["metadatas", "documents"])

        seen: set = set()
        names: List[str] = []

        for meta, text in zip(
            items.get("metadatas", []), items.get("documents", [])
        ):
            # Prefer explicit metadata
            rule = (meta or {}).get("rule_name")

            # Fall back: parse from chunk text
            if not rule and text:
                match = re.search(r"^rule_name:\s*(.+)$", text, re.MULTILINE)
                if match:
                    rule = match.group(1).strip()

            if rule and rule not in seen:
                names.append(rule)
                seen.add(rule)

        return sorted(names)


# ------------------------------------------------------------------
# CLI entry point: python -m src.rag.pipeline
# ------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline = RAGPipeline()
    pipeline.ingest()
    print("Ingestion complete.")
