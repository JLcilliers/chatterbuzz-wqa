"""Ping search engines after publishing to request crawl."""

import logging
import httpx

logger = logging.getLogger(__name__)

PING_URLS = {
    "google": "https://www.google.com/ping?sitemap={sitemap_url}",
    "bing": "https://www.bing.com/ping?sitemap={sitemap_url}",
}


def ping_search_engines(sitemap_url: str) -> dict[str, bool]:
    """Ping Google and Bing with the sitemap URL after publishing."""
    results = {}
    with httpx.Client(timeout=10) as client:
        for engine, url_template in PING_URLS.items():
            try:
                url = url_template.format(sitemap_url=sitemap_url)
                response = client.get(url)
                success = response.status_code == 200
                results[engine] = success
                if success:
                    logger.info(f"Pinged {engine} successfully")
                else:
                    logger.warning(f"Ping {engine} returned {response.status_code}")
            except httpx.HTTPError as e:
                logger.error(f"Failed to ping {engine}: {e}")
                results[engine] = False
    return results
