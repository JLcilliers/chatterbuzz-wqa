"""Redirect manager — WordPress Redirection plugin API + Webflow redirects."""

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class RedirectManager:
    """Manages 301 redirects. Never auto-deletes existing redirects."""

    def __init__(self, wp_publisher=None, webflow_publisher=None):
        self.wp = wp_publisher
        self.webflow = webflow_publisher

    def add_wp_redirect(self, source_url: str, target_url: str, match_type: str = "url") -> dict:
        """Add a 301 redirect via WordPress Redirection plugin REST API."""
        if not self.wp:
            raise ValueError("WordPress publisher not configured")

        payload = {
            "url": source_url,
            "action_data": {"url": target_url},
            "action_type": "url",
            "action_code": 301,
            "match_type": match_type,
            "group_id": 1,
        }
        result = self.wp._request("POST", "/../redirection/v1/redirect", json=payload)
        logger.info(f"Added WP redirect: {source_url} -> {target_url}")
        return result

    def add_webflow_redirect(self, source_path: str, target_path: str) -> dict:
        """Add a 301 redirect via Webflow API."""
        if not self.webflow:
            raise ValueError("Webflow publisher not configured")

        result = self.webflow._request(
            "POST",
            f"/sites/{self.webflow.site_id}/redirects",
            json={
                "sourcePath": source_path,
                "targetPath": target_path,
                "statusCode": 301,
            },
        )
        logger.info(f"Added Webflow redirect: {source_path} -> {target_path}")
        return result

    def bulk_add_redirects(
        self,
        redirects: list[dict],
        platform: str = "wordpress",
    ) -> list[dict]:
        """Add multiple redirects. Each dict needs 'source' and 'target' keys."""
        results = []
        for redirect in redirects:
            try:
                if platform == "wordpress":
                    result = self.add_wp_redirect(redirect["source"], redirect["target"])
                elif platform == "webflow":
                    result = self.add_webflow_redirect(redirect["source"], redirect["target"])
                else:
                    result = {"error": f"Unknown platform: {platform}"}
                results.append({"status": "success", **redirect, "result": result})
            except Exception as e:
                logger.error(f"Failed to add redirect {redirect['source']}: {e}")
                results.append({"status": "error", **redirect, "error": str(e)})
        return results
