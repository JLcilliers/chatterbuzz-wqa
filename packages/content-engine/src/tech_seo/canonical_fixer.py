"""Canonical URL fixer via WordPress REST API (Yoast/RankMath)."""

import logging

logger = logging.getLogger(__name__)


class CanonicalFixer:
    """Fix canonical URLs via CMS REST APIs."""

    def __init__(self, wp_publisher=None):
        self.wp = wp_publisher

    def fix_canonical(self, post_id: int, canonical_url: str) -> dict:
        """Set the correct canonical URL on a WordPress post/page."""
        if not self.wp:
            raise ValueError("WordPress publisher not configured")
        result = self.wp.set_canonical(post_id, canonical_url)
        logger.info(f"Fixed canonical for post {post_id}: {canonical_url}")
        return result

    def bulk_fix_canonicals(self, fixes: list[dict]) -> list[dict]:
        """Fix multiple canonical URLs. Each dict needs 'post_id' and 'canonical_url'."""
        results = []
        for fix in fixes:
            try:
                result = self.fix_canonical(fix["post_id"], fix["canonical_url"])
                results.append({"status": "success", **fix})
            except Exception as e:
                logger.error(f"Failed to fix canonical for post {fix['post_id']}: {e}")
                results.append({"status": "error", **fix, "error": str(e)})
        return results
