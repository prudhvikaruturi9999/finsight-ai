import hashlib
from datetime import datetime, timezone

import tiktoken

from config import settings
from enrichment.metadata_schema import ParsedDocument, Chunk, ChunkMetadata
from utils.logger import get_logger

logger = get_logger(__name__)

_enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _split_by_tokens(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Token-aware text splitter preserving sentence boundaries where possible."""
    tokens = _enc.encode(text)
    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(_enc.decode(chunk_tokens))
        if end == len(tokens):
            break
        start = end - overlap

    return [c for c in chunks if c.strip()]


def _make_chunk_id(source_path: str, chunk_type: str, idx: int) -> str:
    raw = f"{source_path}::{chunk_type}::{idx}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


class HierarchicalChunker:
    def __init__(self):
        self.parent_size = settings.PARENT_CHUNK_SIZE
        self.child_size = settings.CHILD_CHUNK_SIZE
        self.child_overlap = int(self.child_size * settings.CHUNK_OVERLAP_PCT)
        self.parent_overlap = int(self.parent_size * settings.CHUNK_OVERLAP_PCT)

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        now = datetime.now(timezone.utc).isoformat()
        full_text = "\n\n".join(p.text for p in doc.pages if p.text.strip())

        if not full_text.strip():
            logger.warning(f"No text extracted from {doc.source_path}")
            return []

        parent_texts = _split_by_tokens(full_text, self.parent_size, self.parent_overlap)
        chunks: list[Chunk] = []

        for parent_idx, parent_text in enumerate(parent_texts):
            page_no, heading = self._locate(parent_text, doc)
            parent_id = _make_chunk_id(doc.source_path, "parent", parent_idx)

            chunks.append(Chunk(
                text=parent_text,
                metadata=ChunkMetadata(
                    chunk_id=parent_id,
                    source_path=doc.source_path,
                    doc_type=doc.doc_type,
                    page_no=page_no,
                    ingested_at=now,
                    section_heading=heading,
                    chunk_type="parent",
                    content_hash="",
                    embedding_model=settings.EMBEDDING_MODEL,
                    parent_id=None,
                    token_count=_count_tokens(parent_text),
                    char_count=len(parent_text),
                ),
            ))

            for child_idx, child_text in enumerate(
                _split_by_tokens(parent_text, self.child_size, self.child_overlap)
            ):
                child_id = _make_chunk_id(
                    doc.source_path, "child", parent_idx * 10_000 + child_idx
                )
                chunks.append(Chunk(
                    text=child_text,
                    metadata=ChunkMetadata(
                        chunk_id=child_id,
                        source_path=doc.source_path,
                        doc_type=doc.doc_type,
                        page_no=page_no,
                        ingested_at=now,
                        section_heading=heading,
                        chunk_type="child",
                        content_hash="",
                        embedding_model=settings.EMBEDDING_MODEL,
                        parent_id=parent_id,
                        token_count=_count_tokens(child_text),
                        char_count=len(child_text),
                    ),
                ))

        parents = sum(1 for c in chunks if c.metadata.chunk_type == "parent")
        children = sum(1 for c in chunks if c.metadata.chunk_type == "child")
        logger.info(f"{doc.source_path}: {parents} parent chunks, {children} child chunks")
        return chunks

    def _locate(self, chunk_text: str, doc: ParsedDocument) -> tuple[int, str]:
        """Return (page_no, section_heading) for the page most likely containing this chunk."""
        probe = chunk_text[:100]
        for page in doc.pages:
            if probe[:40] in page.text:
                return page.page_no, page.section_heading
        return doc.pages[0].page_no if doc.pages else 1, ""
