"""
Index Monitor Worker
====================
Checks Google index status for client URLs via the GSC URL Inspection API,
detects de-indexation events, and identifies crawl-budget issues for pages
that remain unindexed long after publication.

Results are written to the ``index_status`` table in Supabase.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Optional

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-URL inspection
# ---------------------------------------------------------------------------

def check_index_status(url: str, credentials: dict) -> dict:
    """Call the GSC URL Inspection API for a single URL.

    Args:
        url: Fully-qualified URL to inspect.
        credentials: Dict with ``token``, ``refresh_token``, ``client_id``,
            ``client_secret``, and ``token_uri`` keys used to build
            :class:`google.oauth2.credentials.Credentials`.

    Returns:
        Dict with keys: url, is_indexed, last_crawled, coverage_state, verdict.
    """
    creds = Credentials(
        token=credentials.get("token"),
        refresh_token=credentials.get("refresh_token"),
        client_id=credentials.get("client_id"),
        client_secret=credentials.get("client_secret"),
        token_uri=credentials.get("token_uri", "https://oauth2.googleapis.com/token"),
    )

    service = build("searchconsole", "v1", credentials=creds)

    request_body = {
        "inspectionUrl": url,
        "siteUrl": credentials.get("site_url", url),
    }

    try:
        response = service.urlInspection().index().inspect(body=request_body).execute()
        result = response.get("inspectionResult", {})
        index_status_result = result.get("indexStatusResult", {})

        verdict = index_status_result.get("verdict", "VERDICT_UNSPECIFIED")
        coverage_state = index_status_result.get("coverageState", "")
        last_crawled = index_status_result.get("lastCrawlTime", "")
        is_indexed = verdict == "PASS"

        return {
            "url": url,
            "is_indexed": is_indexed,
            "last_crawled": last_crawled,
            "coverage_state": coverage_state,
            "verdict": verdict,
        }

    except Exception as e:
        logger.error(f"[Index Monitor] Inspection failed for {url}: {e}")
        return {
            "url": url,
            "is_indexed": False,
            "last_crawled": None,
            "coverage_state": "ERROR",
            "verdict": f"ERROR: {e}",
        }


# ---------------------------------------------------------------------------
# Batch inspection
# ---------------------------------------------------------------------------

def batch_check_index_status(
    urls: list[str],
    credentials: dict,
    supabase_client=None,
    client_id: str = "",
) -> list[dict]:
    """Check index status for a list of URLs, one at a time with a 1 s delay.

    If *supabase_client* is provided the results are upserted into the
    ``index_status`` table keyed on ``(client_id, url)``.

    Args:
        urls: URLs to inspect.
        credentials: Credential dict forwarded to :func:`check_index_status`.
        supabase_client: Optional Supabase client for persistence.
        client_id: Identifier for the client / project.

    Returns:
        List of result dicts (one per URL).
    """
    results: list[dict] = []
    now_iso = datetime.utcnow().isoformat()

    for idx, url in enumerate(urls):
        logger.info(f"[Index Monitor] Checking {idx + 1}/{len(urls)}: {url}")
        result = check_index_status(url, credentials)
        results.append(result)

        # Persist to Supabase
        if supabase_client and client_id:
            record = {
                "client_id": client_id,
                "url": result["url"],
                "is_indexed": result["is_indexed"],
                "last_crawled": result["last_crawled"],
                "coverage_state": result["coverage_state"],
                "verdict": result["verdict"],
                "checked_at": now_iso,
            }
            try:
                supabase_client.table("index_status").upsert(
                    record,
                    on_conflict="client_id,url",
                ).execute()
            except Exception as e:
                logger.error(f"[Index Monitor] Supabase write failed for {url}: {e}")

        # Rate-limit: 1 s between requests
        if idx < len(urls) - 1:
            time.sleep(1)

    logger.info(
        f"[Index Monitor] Batch complete: {len(results)} URLs checked for "
        f"client={client_id}"
    )
    return results


# ---------------------------------------------------------------------------
# De-indexation detection
# ---------------------------------------------------------------------------

def detect_deindexation(client_id: str, supabase_client) -> list[dict]:
    """Compare current vs previous index checks and find newly de-indexed URLs.

    A URL is considered *de-indexed* when its most recent check shows
    ``is_indexed = False`` but the immediately preceding check showed
    ``is_indexed = True``.

    Args:
        client_id: Client / project identifier.
        supabase_client: Initialised Supabase client.

    Returns:
        List of dicts with keys: url, previous_checked_at, current_checked_at,
        coverage_state, verdict.
    """
    deindexed: list[dict] = []

    try:
        # Fetch all index_status rows for the client ordered by checked_at desc
        response = (
            supabase_client.table("index_status")
            .select("url, is_indexed, checked_at, coverage_state, verdict")
            .eq("client_id", client_id)
            .order("checked_at", desc=True)
            .execute()
        )

        rows = response.data or []
        if not rows:
            return deindexed

        # Group rows by URL, keeping chronological order (newest first)
        url_history: dict[str, list[dict]] = {}
        for row in rows:
            url_history.setdefault(row["url"], []).append(row)

        for url, history in url_history.items():
            if len(history) < 2:
                continue

            current = history[0]   # most recent
            previous = history[1]  # prior check

            if not current["is_indexed"] and previous["is_indexed"]:
                deindexed.append({
                    "url": url,
                    "previous_checked_at": previous["checked_at"],
                    "current_checked_at": current["checked_at"],
                    "coverage_state": current["coverage_state"],
                    "verdict": current["verdict"],
                })

        if deindexed:
            logger.warning(
                f"[Index Monitor] Detected {len(deindexed)} de-indexed URLs "
                f"for client={client_id}"
            )

    except Exception as e:
        logger.error(f"[Index Monitor] De-indexation detection failed: {e}")

    return deindexed


# ---------------------------------------------------------------------------
# Crawl-budget issue detection
# ---------------------------------------------------------------------------

def detect_crawl_budget_issues(client_id: str, supabase_client) -> list[dict]:
    """Find published pages older than 30 days that are still not indexed.

    Cross-references the ``index_status`` table (latest check) with
    ``pages`` or ``gsc_data`` to determine page age.

    Args:
        client_id: Client / project identifier.
        supabase_client: Initialised Supabase client.

    Returns:
        List of dicts with keys: url, coverage_state, verdict, page_age_days,
        last_crawled.
    """
    issues: list[dict] = []
    cutoff_date = (datetime.utcnow() - timedelta(days=30)).isoformat()

    try:
        # Get latest index status for each URL (most recent check)
        idx_response = (
            supabase_client.table("index_status")
            .select("url, is_indexed, coverage_state, verdict, last_crawled, checked_at")
            .eq("client_id", client_id)
            .eq("is_indexed", False)
            .order("checked_at", desc=True)
            .execute()
        )

        not_indexed_rows = idx_response.data or []
        if not not_indexed_rows:
            return issues

        # De-duplicate to latest check per URL
        seen_urls: set[str] = set()
        latest_rows: list[dict] = []
        for row in not_indexed_rows:
            if row["url"] not in seen_urls:
                seen_urls.add(row["url"])
                latest_rows.append(row)

        # Get published page dates from the pages table
        pages_response = (
            supabase_client.table("pages")
            .select("url, published_at")
            .eq("client_id", client_id)
            .lt("published_at", cutoff_date)
            .execute()
        )

        old_pages = {p["url"]: p["published_at"] for p in (pages_response.data or [])}

        for row in latest_rows:
            url = row["url"]
            if url in old_pages:
                published_at = datetime.fromisoformat(
                    old_pages[url].replace("Z", "+00:00").replace("+00:00", "")
                )
                age_days = (datetime.utcnow() - published_at).days

                issues.append({
                    "url": url,
                    "coverage_state": row["coverage_state"],
                    "verdict": row["verdict"],
                    "page_age_days": age_days,
                    "last_crawled": row["last_crawled"],
                })

        if issues:
            logger.warning(
                f"[Index Monitor] Found {len(issues)} crawl-budget issues "
                f"(published >30 days, still not indexed) for client={client_id}"
            )

    except Exception as e:
        logger.error(f"[Index Monitor] Crawl-budget detection failed: {e}")

    return issues
