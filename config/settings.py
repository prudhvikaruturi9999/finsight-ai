import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# "local" uses sentence-transformers (free, CPU); "openai" uses OpenAI API
EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "local")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LOCAL_EMBEDDING_MODEL: str = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIMENSIONS: int = int(os.getenv("EMBEDDING_DIMENSIONS", "384"))

CHROMA_PATH: str = os.getenv("CHROMA_PATH", "./data/chroma")
HASH_CACHE_PATH: str = os.getenv("HASH_CACHE_PATH", "./data/hash_cache.db")

MAX_FILE_SIZE_MB: float = float(os.getenv("MAX_FILE_SIZE_MB", "100"))
EMBEDDING_DAILY_COST_CAP_USD: float = float(os.getenv("EMBEDDING_DAILY_COST_CAP_USD", "10.0"))

PARENT_CHUNK_SIZE: int = int(os.getenv("PARENT_CHUNK_SIZE", "1500"))
CHILD_CHUNK_SIZE: int = int(os.getenv("CHILD_CHUNK_SIZE", "300"))
CHUNK_OVERLAP_PCT: float = float(os.getenv("CHUNK_OVERLAP_PCT", "0.15"))

EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))

# LLM settings
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "anthropic")   # "anthropic" or "openai"
LLM_MODEL: str = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))

# Retrieval settings
RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "10"))
# ~24 000 chars ≈ 6 000 tokens — fits comfortably in Haiku's 200K context
CONTEXT_MAX_CHARS: int = int(os.getenv("CONTEXT_MAX_CHARS", "24000"))

# Cost per token (USD) for OpenAI embedding models
EMBEDDING_COST_PER_TOKEN: dict[str, float] = {
    "text-embedding-3-small": 0.02 / 1_000_000,
    "text-embedding-3-large": 0.13 / 1_000_000,
    "text-embedding-ada-002": 0.10 / 1_000_000,
}
