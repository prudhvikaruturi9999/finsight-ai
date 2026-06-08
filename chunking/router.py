from pathlib import Path

from enrichment.metadata_schema import ParsedDocument, Chunk
from chunking.hierarchical import HierarchicalChunker
from utils.logger import get_logger

logger = get_logger(__name__)

_hierarchical = HierarchicalChunker()


def chunk_document(doc: ParsedDocument) -> list[Chunk]:
    """Route to the correct chunking strategy based on doc_type."""
    strategy = _get_strategy(doc.doc_type)
    logger.info(f"Chunking {Path(doc.source_path).name} with strategy: {strategy}")

    if strategy == "hierarchical":
        return _hierarchical.chunk(doc)

    # Future strategies (excel row-level, audio utterance, etc.) go here
    logger.warning(f"Unknown chunking strategy '{strategy}'; defaulting to hierarchical")
    return _hierarchical.chunk(doc)


def _get_strategy(doc_type: str) -> str:
    return {
        "pdf": "hierarchical",
        "docx": "hierarchical",
        "excel": "row_level",
        "audio": "utterance_level",
        "api": "recursive_character",
    }.get(doc_type, "hierarchical")
