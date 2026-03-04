"""Content queue manager — manages the content pipeline from generation to publishing."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ContentQueueManager:
    """Manages the content generation queue via Supabase."""

    def __init__(self, supabase_client):
        self.db = supabase_client

    def add_to_queue(
        self,
        client_id: str,
        title: str,
        content_type: str,
        content_body: Optional[str] = None,
        meta_title: Optional[str] = None,
        meta_description: Optional[str] = None,
        schema_markup: Optional[dict] = None,
        quality_score: Optional[int] = None,
        keyword_cluster_id: Optional[str] = None,
        status: str = "draft",
    ) -> dict:
        """Add a content item to the queue."""
        record = {
            "client_id": client_id,
            "title": title,
            "content_type": content_type,
            "content_body": content_body,
            "meta_title": meta_title,
            "meta_description": meta_description,
            "schema_markup": schema_markup,
            "quality_score": quality_score,
            "keyword_cluster_id": keyword_cluster_id,
            "status": status,
        }
        result = self.db.table("content_queue").insert(record).execute()
        logger.info(f"Added to queue: {title} (status={status})")
        return result.data[0] if result.data else {}

    def get_pending_review(self, client_id: str) -> list[dict]:
        """Get content items pending review."""
        result = (
            self.db.table("content_queue")
            .select("*")
            .eq("client_id", client_id)
            .eq("status", "review")
            .order("created_at", desc=False)
            .execute()
        )
        return result.data or []

    def get_approved(self, client_id: str, limit: int = 100) -> list[dict]:
        """Get approved content ready for publishing."""
        result = (
            self.db.table("content_queue")
            .select("*")
            .eq("client_id", client_id)
            .eq("status", "approved")
            .order("created_at", desc=False)
            .limit(limit)
            .execute()
        )
        return result.data or []

    def approve(self, item_id: str) -> dict:
        """Approve a content item."""
        result = (
            self.db.table("content_queue")
            .update({"status": "approved"})
            .eq("id", item_id)
            .execute()
        )
        return result.data[0] if result.data else {}

    def reject(self, item_id: str) -> dict:
        """Reject a content item."""
        result = (
            self.db.table("content_queue")
            .update({"status": "rejected"})
            .eq("id", item_id)
            .execute()
        )
        return result.data[0] if result.data else {}

    def mark_published(self, item_id: str, cms_post_id: str) -> dict:
        """Mark a content item as published."""
        from datetime import datetime

        result = (
            self.db.table("content_queue")
            .update({
                "status": "published",
                "cms_post_id": cms_post_id,
                "published_at": datetime.utcnow().isoformat(),
            })
            .eq("id", item_id)
            .execute()
        )
        return result.data[0] if result.data else {}

    def get_stats(self, client_id: str) -> dict:
        """Get queue statistics for a client."""
        statuses = ["draft", "review", "approved", "published", "rejected"]
        stats = {}
        for status in statuses:
            result = (
                self.db.table("content_queue")
                .select("id", count="exact")
                .eq("client_id", client_id)
                .eq("status", status)
                .execute()
            )
            stats[status] = result.count or 0
        return stats
