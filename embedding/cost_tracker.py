from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class CostTracker:
    def __init__(self):
        self.total_tokens: int = 0
        self.total_cost_usd: float = 0.0
        self.api_calls: int = 0
        self.cache_hits: int = 0

    def record_api_call(self, tokens: int, model: str) -> None:
        cost_per_token = settings.EMBEDDING_COST_PER_TOKEN.get(model, 0.0)
        cost = tokens * cost_per_token
        self.total_tokens += tokens
        self.total_cost_usd += cost
        self.api_calls += 1

        # Cost cap only applies to paid API calls
        if cost_per_token > 0 and self.total_cost_usd > settings.EMBEDDING_DAILY_COST_CAP_USD:
            raise RuntimeError(
                f"Daily embedding cost cap ${settings.EMBEDDING_DAILY_COST_CAP_USD:.2f} exceeded. "
                f"Spent so far: ${self.total_cost_usd:.4f}"
            )

    def record_cache_hit(self) -> None:
        self.cache_hits += 1

    def summary(self) -> str:
        saved = self.cache_hits / max(1, self.cache_hits + self.api_calls) * 100
        return (
            f"tokens={self.total_tokens:,}  cost=${self.total_cost_usd:.4f}  "
            f"api_calls={self.api_calls}  cache_hits={self.cache_hits}  "
            f"cache_save_rate={saved:.0f}%"
        )
