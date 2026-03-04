"""
Phase 7 — Bulk Schema Generation

Generates JSON-LD markup for individual pages or entire DataFrames of pages,
with optional persistence to a Supabase ``schema_markup`` table.
"""

import json
import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from .templates import SCHEMA_TEMPLATES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-page generation
# ---------------------------------------------------------------------------

def generate_schema_for_page(
    url: str,
    page_type: str,
    page_data: dict,
    business_data: dict,
) -> dict:
    """Pick the right template for *page_type*, merge business-level and
    page-level data, and return a complete JSON-LD dict.

    Parameters
    ----------
    url : str
        Canonical URL of the page.
    page_type : str
        Key into ``SCHEMA_TEMPLATES`` (e.g. ``"article"``, ``"faq"``).
    page_data : dict
        Page-specific fields (headline, questions, steps, etc.).
    business_data : dict
        Shared business-level fields (name, logo, phone, address, etc.).

    Returns
    -------
    dict
        ``{"url": ..., "page_type": ..., "schema_json": <JSON-LD dict>}``
        or ``{"url": ..., "page_type": ..., "error": <message>}`` on failure.
    """
    page_type_key = page_type.strip().lower().replace(" ", "_").replace("-", "_")

    template_fn = SCHEMA_TEMPLATES.get(page_type_key)
    if template_fn is None:
        logger.warning("No template for page_type=%r (url=%s)", page_type, url)
        return {
            "url": url,
            "page_type": page_type,
            "error": f"Unknown page type: {page_type}",
        }

    # Merge: page-level data wins over business-level defaults.
    merged: Dict[str, Any] = {**business_data, **page_data}
    merged.setdefault("url", url)

    # For article / blog types, propagate business info to publisher fields.
    if page_type_key in ("article", "blog"):
        merged.setdefault("publisher_name", business_data.get("name", ""))
        merged.setdefault("publisher_logo", business_data.get("logo", ""))

    # For service type, propagate provider info.
    if page_type_key == "service":
        merged.setdefault("provider_name", business_data.get("name", ""))
        merged.setdefault("provider_url", business_data.get("url", ""))

    try:
        schema_json = template_fn(merged)
    except Exception as exc:
        logger.error("Template error for %s (%s): %s", url, page_type, exc)
        return {
            "url": url,
            "page_type": page_type,
            "error": str(exc),
        }

    return {
        "url": url,
        "page_type": page_type,
        "schema_json": schema_json,
    }


# ---------------------------------------------------------------------------
# Bulk generation
# ---------------------------------------------------------------------------

def bulk_generate(
    pages_df: pd.DataFrame,
    business_data: dict,
    supabase_client: Optional[Any] = None,
) -> List[dict]:
    """Generate JSON-LD schemas for every row in *pages_df*.

    Expected DataFrame columns
    --------------------------
    - ``url`` (required)
    - ``page_type`` (required) — must match a key in ``SCHEMA_TEMPLATES``
    - ``page_data`` (optional) — dict or JSON string of page-specific fields

    Any additional columns are treated as extra page-level fields that will be
    merged into the template data.

    Parameters
    ----------
    pages_df : pd.DataFrame
        One row per page.
    business_data : dict
        Shared business-level data.
    supabase_client : optional
        If provided, upsert results to the ``schema_markup`` table.

    Returns
    -------
    list[dict]
        List of result dicts from ``generate_schema_for_page``.
    """
    if pages_df.empty:
        logger.info("bulk_generate called with empty DataFrame — nothing to do.")
        return []

    required_cols = {"url", "page_type"}
    missing = required_cols - set(pages_df.columns)
    if missing:
        raise ValueError(f"pages_df is missing required columns: {missing}")

    results: List[dict] = []

    for idx, row in pages_df.iterrows():
        url = str(row["url"]).strip()
        page_type = str(row["page_type"]).strip()

        # Build page_data from explicit column or remaining columns.
        if "page_data" in row.index and pd.notna(row.get("page_data")):
            raw = row["page_data"]
            if isinstance(raw, str):
                try:
                    page_data = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON in page_data for row %s", idx)
                    page_data = {}
            elif isinstance(raw, dict):
                page_data = raw
            else:
                page_data = {}
        else:
            page_data = {}

        # Merge any extra columns as fallback fields.
        extra_cols = set(row.index) - {"url", "page_type", "page_data"}
        for col in extra_cols:
            val = row[col]
            if pd.notna(val):
                page_data.setdefault(col, val)

        result = generate_schema_for_page(url, page_type, page_data, business_data)
        results.append(result)

    logger.info(
        "bulk_generate complete: %d pages processed, %d errors.",
        len(results),
        sum(1 for r in results if "error" in r),
    )

    # Optional Supabase persistence.
    if supabase_client is not None:
        _persist_to_supabase(supabase_client, results)

    return results


# ---------------------------------------------------------------------------
# Supabase persistence helper
# ---------------------------------------------------------------------------

def _persist_to_supabase(client: Any, results: List[dict]) -> None:
    """Upsert successful schema results to the ``schema_markup`` table."""
    rows_to_upsert = []
    for r in results:
        if "error" in r:
            continue
        rows_to_upsert.append({
            "url": r["url"],
            "page_type": r["page_type"],
            "schema_json": json.dumps(r["schema_json"]),
        })

    if not rows_to_upsert:
        logger.info("No valid schemas to persist.")
        return

    try:
        client.table("schema_markup").upsert(
            rows_to_upsert, on_conflict="url"
        ).execute()
        logger.info("Persisted %d schemas to schema_markup.", len(rows_to_upsert))
    except Exception as exc:
        logger.error("Supabase upsert failed: %s", exc)
