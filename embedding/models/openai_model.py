from openai import OpenAI
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class OpenAIEmbeddingModel:
    def __init__(self):
        if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY.startswith("sk-..."):
            raise ValueError(
                "OPENAI_API_KEY is not set. Update your .env file with a real API key."
            )
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.EMBEDDING_MODEL
        logger.info(f"OpenAI embedding model: {self.model}")

    def embed(self, texts: list[str]) -> tuple[list[list[float]], int]:
        """Embed a batch of texts.

        Returns (embeddings, total_tokens_used).
        """
        response = self.client.embeddings.create(model=self.model, input=texts)
        embeddings = [item.embedding for item in response.data]
        total_tokens = response.usage.total_tokens
        return embeddings, total_tokens
