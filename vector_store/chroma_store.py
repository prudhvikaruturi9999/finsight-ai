from pathlib import Path

import chromadb

from config import settings
from enrichment.metadata_schema import Chunk
from vector_store.base_store import BaseVectorStore
from utils.logger import get_logger

logger = get_logger(__name__)


class ChromaStore(BaseVectorStore):
    def __init__(self, collection_name: str = "finsight_chunks"):
        Path(settings.CHROMA_PATH).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB ready — collection='{collection_name}' "
            f"existing_chunks={self.collection.count()}"
        )

    def upsert(self, chunks_with_embeddings: list[tuple[Chunk, list[float]]]) -> int:
        if not chunks_with_embeddings:
            return 0

        ids, embeddings, documents, metadatas = [], [], [], []

        for chunk, embedding in chunks_with_embeddings:
            ids.append(chunk.metadata.chunk_id)
            embeddings.append(embedding)
            documents.append(chunk.text)
            metadatas.append({
                "source_path":     chunk.metadata.source_path,
                "doc_type":        chunk.metadata.doc_type,
                "page_no":         chunk.metadata.page_no,
                "ingested_at":     chunk.metadata.ingested_at,
                "section_heading": chunk.metadata.section_heading,
                "chunk_type":      chunk.metadata.chunk_type,
                "content_hash":    chunk.metadata.content_hash,
                "embedding_model": chunk.metadata.embedding_model,
                "parent_id":       chunk.metadata.parent_id or "",
                "token_count":     chunk.metadata.token_count,
            })

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info(f"Upserted {len(ids)} chunks — total in store: {self.collection.count()}")
        return len(ids)

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[dict]:
        kwargs: dict = dict(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        if filters:
            kwargs["where"] = filters

        results = self.collection.query(**kwargs)
        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({"text": doc, "metadata": meta, "score": 1.0 - dist})
        return output

    def delete_by_source(self, source_path: str) -> int:
        results = self.collection.get(
            where={"source_path": source_path}, include=[]
        )
        ids = results["ids"]
        if ids:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} chunks for: {source_path}")
        return len(ids)

    def count(self) -> int:
        return self.collection.count()
