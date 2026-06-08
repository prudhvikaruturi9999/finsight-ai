from abc import ABC, abstractmethod
from enrichment.metadata_schema import Chunk


class BaseVectorStore(ABC):
    @abstractmethod
    def upsert(self, chunks_with_embeddings: list[tuple[Chunk, list[float]]]) -> int:
        """Upsert chunks. Returns number of chunks written."""
        ...

    @abstractmethod
    def query(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[dict]:
        """Return top-k chunks as dicts with keys: text, metadata, score."""
        ...

    @abstractmethod
    def delete_by_source(self, source_path: str) -> int:
        """Delete all chunks for a source document. Returns count deleted."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Total number of chunks in the store."""
        ...
