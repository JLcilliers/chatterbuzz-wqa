"""WordPress CMS publisher via REST API v2."""

import base64
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class WordPressPublisher:
    """Publish content to WordPress via REST API v2 with Application Passwords."""

    def __init__(self, base_url: str, username: str, app_password: str):
        self.base_url = base_url.rstrip("/")
        self.auth = base64.b64encode(f"{username}:{app_password}".encode()).decode()
        self.headers = {
            "Authorization": f"Basic {self.auth}",
            "Content-Type": "application/json",
        }

    def _request(self, method: str, endpoint: str, **kwargs) -> dict:
        """Make an authenticated request to WP REST API."""
        url = f"{self.base_url}{endpoint}"
        with httpx.Client(timeout=30) as client:
            response = client.request(method, url, headers=self.headers, **kwargs)
            response.raise_for_status()
            return response.json()

    def create_post(
        self,
        title: str,
        content: str,
        status: str = "draft",
        meta_title: Optional[str] = None,
        meta_description: Optional[str] = None,
        categories: Optional[list[int]] = None,
        tags: Optional[list[int]] = None,
        slug: Optional[str] = None,
    ) -> dict:
        """Create a new post."""
        payload: dict = {
            "title": title,
            "content": content,
            "status": status,
        }
        if slug:
            payload["slug"] = slug
        if categories:
            payload["categories"] = categories
        if tags:
            payload["tags"] = tags

        # Yoast/RankMath SEO meta via REST
        meta: dict = {}
        if meta_title:
            meta["_yoast_wpseo_title"] = meta_title
            meta["rank_math_title"] = meta_title
        if meta_description:
            meta["_yoast_wpseo_metadesc"] = meta_description
            meta["rank_math_description"] = meta_description
        if meta:
            payload["meta"] = meta

        result = self._request("POST", "/posts", json=payload)
        logger.info(f"Created WP post: {result.get('id')} — {title}")
        return result

    def create_page(
        self,
        title: str,
        content: str,
        status: str = "draft",
        meta_title: Optional[str] = None,
        meta_description: Optional[str] = None,
        parent: Optional[int] = None,
        slug: Optional[str] = None,
    ) -> dict:
        """Create a new page."""
        payload: dict = {
            "title": title,
            "content": content,
            "status": status,
        }
        if slug:
            payload["slug"] = slug
        if parent:
            payload["parent"] = parent

        meta: dict = {}
        if meta_title:
            meta["_yoast_wpseo_title"] = meta_title
            meta["rank_math_title"] = meta_title
        if meta_description:
            meta["_yoast_wpseo_metadesc"] = meta_description
            meta["rank_math_description"] = meta_description
        if meta:
            payload["meta"] = meta

        result = self._request("POST", "/pages", json=payload)
        logger.info(f"Created WP page: {result.get('id')} — {title}")
        return result

    def update_post_meta(self, post_id: int, meta: dict) -> dict:
        """Update meta fields on an existing post."""
        return self._request("POST", f"/posts/{post_id}", json={"meta": meta})

    def set_canonical(self, post_id: int, canonical_url: str) -> dict:
        """Set canonical URL via Yoast/RankMath meta."""
        return self.update_post_meta(post_id, {
            "_yoast_wpseo_canonical": canonical_url,
            "rank_math_canonical_url": canonical_url,
        })

    def set_schema_markup(self, post_id: int, schema_json: str) -> dict:
        """Set JSON-LD schema via RankMath meta."""
        return self.update_post_meta(post_id, {
            "rank_math_schema_Article": schema_json,
        })
