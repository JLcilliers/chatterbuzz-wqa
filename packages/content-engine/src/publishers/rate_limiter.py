"""Rate limiter for CMS publishing — 100/week limit, 2s delay between posts."""

import time
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class PublishRateLimiter:
    """Enforces publishing rate limits per client."""

    def __init__(self, supabase_client, max_per_week: int = 100, delay_seconds: float = 2.0):
        self.db = supabase_client
        self.max_per_week = max_per_week
        self.delay_seconds = delay_seconds
        self._last_publish_time = 0.0

    def get_published_this_week(self, client_id: str) -> int:
        """Count items published in the current week."""
        week_start = (datetime.utcnow() - timedelta(days=datetime.utcnow().weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        result = (
            self.db.table("content_queue")
            .select("id", count="exact")
            .eq("client_id", client_id)
            .eq("status", "published")
            .gte("published_at", week_start.isoformat())
            .execute()
        )
        return result.count or 0

    def can_publish(self, client_id: str) -> tuple[bool, str]:
        """Check if publishing is allowed."""
        count = self.get_published_this_week(client_id)
        if count >= self.max_per_week:
            return False, f"Weekly limit reached: {count}/{self.max_per_week}"
        return True, f"{self.max_per_week - count} publishes remaining this week"

    def wait_for_delay(self):
        """Enforce minimum delay between publishes."""
        elapsed = time.time() - self._last_publish_time
        if elapsed < self.delay_seconds:
            wait = self.delay_seconds - elapsed
            logger.debug(f"Rate limiting: waiting {wait:.1f}s")
            time.sleep(wait)
        self._last_publish_time = time.time()
