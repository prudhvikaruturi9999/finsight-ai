from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class LocalEmbeddingModel:
    """Sentence-transformers embedding model — runs locally, no API cost.

    Default: BAAI/bge-small-en-v1.5 (384 dims, ~130 MB, strong retrieval quality).
    Set LOCAL_EMBEDDING_MODEL in .env to swap models without code changes.
    """

    def __init__(self):
        from sentence_transformers import SentenceTransformer

        model_name = settings.LOCAL_EMBEDDING_MODEL
        logger.info(f"Loading local model: {model_name}  (first run downloads ~130 MB)")
        self.model = SentenceTransformer(model_name)
        dims = self.model.get_embedding_dimension()
        logger.info(f"Model ready — dims={dims}")

    def embed(self, texts: list[str]) -> tuple[list[list[float]], int]:
        """Embed a batch of texts.

        Returns (embeddings, approx_token_count).
        Embeddings are L2-normalised (required for cosine similarity).
        """
        vectors = self.model.encode(
            texts,
            batch_size=min(32, len(texts)),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        # Approximate token count: ~4 chars per token (for cost-tracker display)
        approx_tokens = sum(len(t) // 4 for t in texts)
        return vectors.tolist(), approx_tokens
