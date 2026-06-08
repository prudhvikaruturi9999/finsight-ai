from config import settings
from embedding.hash_cache import HashCache
from embedding.cost_tracker import CostTracker
from enrichment.metadata_schema import Chunk
from utils.logger import get_logger

logger = get_logger(__name__)


def _load_model():
    if settings.EMBEDDING_PROVIDER == "local":
        from embedding.models.local_model import LocalEmbeddingModel
        return LocalEmbeddingModel()
    from embedding.models.openai_model import OpenAIEmbeddingModel
    return OpenAIEmbeddingModel()


class Embedder:
    def __init__(self):
        self.cache = HashCache(settings.HASH_CACHE_PATH)
        self.model = _load_model()
        self.cost_tracker = CostTracker()
        logger.info(f"Embedder ready — provider={settings.EMBEDDING_PROVIDER}")

    def embed_chunks(self, chunks: list[Chunk]) -> list[tuple[Chunk, list[float]]]:
        """Embed chunks, skipping any whose content hash is already in the cache.

        Returns list of (chunk, embedding) for newly embedded chunks only.
        Chunks already in the cache are assumed to exist in the vector store.
        """
        new_chunks: list[Chunk] = []

        for chunk in chunks:
            content_hash = HashCache.compute(chunk.text)
            chunk.metadata.content_hash = content_hash

            if self.cache.has(content_hash):
                self.cost_tracker.record_cache_hit()
            else:
                new_chunks.append(chunk)

        skipped = len(chunks) - len(new_chunks)
        logger.info(
            f"Chunks: {len(chunks)} total | {skipped} cached (skip) | "
            f"{len(new_chunks)} to embed"
        )

        results: list[tuple[Chunk, list[float]]] = []
        batch_size = settings.EMBEDDING_BATCH_SIZE

        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i : i + batch_size]
            texts = [c.text for c in batch]
            batch_num = i // batch_size + 1
            total_batches = (len(new_chunks) + batch_size - 1) // batch_size

            logger.info(f"Embedding batch {batch_num}/{total_batches}: {len(texts)} chunks")
            embeddings, tokens = self.model.embed(texts)
            model_key = settings.LOCAL_EMBEDDING_MODEL if settings.EMBEDDING_PROVIDER == "local" else settings.EMBEDDING_MODEL
            self.cost_tracker.record_api_call(tokens, model_key)

            for chunk, embedding in zip(batch, embeddings):
                self.cache.add(chunk.metadata.content_hash, chunk.metadata.chunk_id)
                results.append((chunk, embedding))

        logger.info(f"Embedding complete — {self.cost_tracker.summary()}")
        return results
