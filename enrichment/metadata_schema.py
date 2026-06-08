from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ChunkMetadata:
    chunk_id: str
    source_path: str
    doc_type: str
    page_no: int
    ingested_at: str
    section_heading: str
    chunk_type: str          # "parent" or "child"
    content_hash: str
    embedding_model: str
    parent_id: Optional[str] = None
    char_count: int = 0
    token_count: int = 0


@dataclass
class ParsedPage:
    page_no: int
    text: str
    tables: list = field(default_factory=list)
    section_heading: str = ""


@dataclass
class ParsedDocument:
    source_path: str
    doc_type: str
    pages: list = field(default_factory=list)   # list[ParsedPage]
    title: str = ""
    total_pages: int = 0


@dataclass
class Chunk:
    text: str
    metadata: ChunkMetadata
