from dataclasses import dataclass, field
from pathlib import Path

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# BGE models retrieve better when queries use this prefix.
_BGE_QUERY_PREFIX = "Represent this sentence: "


@dataclass
class RetrievedChunk:
    text: str
    score: float
    source_file: str
    page_no: int
    chunk_type: str
    chunk_id: str
    parent_id: str
    parent_text: str = ""     # populated if parent context is fetched
    section_heading: str = ""


@dataclass
class RetrievalResult:
    query: str
    chunks: list[RetrievedChunk] = field(default_factory=list)

    def format_context(self, max_chars: int = settings.CONTEXT_MAX_CHARS) -> str:
        """Build a numbered context block for the LLM prompt."""
        parts = []
        total = 0
        for i, c in enumerate(self.chunks, 1):
            # Prefer parent text (more context) when available
            body = (c.parent_text or c.text).strip()
            src = Path(c.source_file).name
            heading = f" — {c.section_heading}" if c.section_heading else ""
            header = f"[{i}] {src}, page {c.page_no}{heading}"
            block = f"{header}\n{body}"
            # Always include the first chunk; stop before adding subsequent ones if over budget
            if parts and total + len(block) > max_chars:
                break
            parts.append(block)
            total += len(block)
        return "\n\n---\n\n".join(parts)

    def format_citations(self) -> str:
        """Compact citation list, e.g. [1] GOOG-10-K-2025.pdf p.34"""
        lines = []
        for i, c in enumerate(self.chunks, 1):
            src = Path(c.source_file).name
            lines.append(f"  [{i}] {src}  p.{c.page_no}  (score={c.score:.3f})")
        return "\n".join(lines)


class Retriever:
    _embed_model = None   # lazy singleton — shared across calls

    def __init__(self):
        from vector_store.chroma_store import ChromaStore
        self.store = ChromaStore()

    @property
    def embed_model(self):
        if Retriever._embed_model is None:
            from embedding.models.local_model import LocalEmbeddingModel
            Retriever._embed_model = LocalEmbeddingModel()
        return Retriever._embed_model

    def retrieve(self, query: str, top_k: int = settings.RETRIEVAL_TOP_K) -> RetrievalResult:
        """Embed the query, search child chunks, expand with parent context."""
        prefixed = _BGE_QUERY_PREFIX + query
        vec, _ = self.embed_model.embed([prefixed])

        raw = self.store.query(vec[0], top_k=top_k, filters={"chunk_type": "child"})

        chunks = []
        parent_ids = list({
            r["metadata"].get("parent_id", "")
            for r in raw
            if r["metadata"].get("parent_id")
        })

        # Batch-fetch all parent texts in one go
        parent_texts: dict[str, str] = {}
        if parent_ids:
            try:
                resp = self.store.collection.get(ids=parent_ids, include=["documents"])
                parent_texts = dict(zip(resp["ids"], resp["documents"]))
            except Exception as exc:
                logger.warning(f"Parent context fetch failed: {exc}")

        for r in raw:
            meta = r["metadata"]
            pid = meta.get("parent_id", "")
            chunks.append(RetrievedChunk(
                text=r["text"],
                score=r["score"],
                source_file=meta.get("source_path", ""),
                page_no=int(meta.get("page_no", 0)),
                chunk_type=meta.get("chunk_type", "child"),
                chunk_id=meta.get("content_hash", ""),
                parent_id=pid,
                parent_text=parent_texts.get(pid, ""),
                section_heading=meta.get("section_heading", ""),
            ))

        logger.info(f"Retrieved {len(chunks)} chunks for: {query!r}")
        return RetrievalResult(query=query, chunks=chunks)
