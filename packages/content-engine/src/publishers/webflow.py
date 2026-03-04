"""Webflow CMS publisher via API v2."""

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class WebflowPublisher:
    """Publish content to Webflow CMS via API v2."""

    API_BASE = "https://api.webflow.com/v2"

    def __init__(self, api_token: str, site_id: str, collection_id: str):
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
            "accept": "application/json",
        }
        self.site_id = site_id
        self.collection_id = collection_id

    def _request(self, method: str, endpoint: str, **kwargs) -> dict:
        """Make an authenticated request to Webflow API."""
        url = f"{self.API_BASE}{endpoint}"
        with httpx.Client(timeout=30) as client:
            response = client.request(method, url, headers=self.headers, **kwargs)
            response.raise_for_status()
            return response.json()

    def create_item(
        self,
        name: str,
        slug: str,
        content_html: str,
        meta_title: Optional[str] = None,
        meta_description: Optional[str] = None,
        is_draft: bool = True,
    ) -> dict:
        """Create a new CMS collection item."""
        field_data: dict = {
            "name": name,
            "slug": slug,
            "_archived": False,
            "_draft": is_draft,
            "post-body": content_html,
        }
        if meta_title:
            field_data["meta-title"] = meta_title
        if meta_description:
            field_data["meta-description"] = meta_description

        payload = {"fieldData": field_data}
        result = self._request(
            "POST",
            f"/collections/{self.collection_id}/items",
            json=payload,
        )
        logger.info(f"Created Webflow item: {result.get('id')} — {name}")
        return result

    def update_item(self, item_id: str, field_data: dict) -> dict:
        """Update an existing CMS item."""
        return self._request(
            "PATCH",
            f"/collections/{self.collection_id}/items/{item_id}",
            json={"fieldData": field_data},
        )

    def publish_items(self, item_ids: list[str]) -> dict:
        """Publish specific items to the live site."""
        return self._request(
            "POST",
            f"/sites/{self.site_id}/publish",
            json={"domains": ["all"]},
        )

    def list_items(self, limit: int = 100, offset: int = 0) -> dict:
        """List CMS collection items."""
        return self._request(
            "GET",
            f"/collections/{self.collection_id}/items?limit={limit}&offset={offset}",
        )
