"""
GSC Pattern Mining — extracts keyword patterns and content gaps from
Google Search Console query data.
"""

import logging
from collections import Counter
from typing import List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would shall should may might can could of in to for on with "
    "at by from as into about between through after before above below "
    "and or but not no nor so yet both either neither each every all "
    "any few more most other some such".split()
)

GENERIC_PATH_TOKENS = frozenset({"", "/", "/index", "/home", "/default"})


def _tokenize(text: str) -> List[str]:
    """Lower-case split, strip non-alpha, drop stop words."""
    tokens = []
    for tok in str(text).lower().split():
        cleaned = "".join(ch for ch in tok if ch.isalnum())
        if cleaned and cleaned not in _STOP_WORDS:
            tokens.append(cleaned)
    return tokens


def _ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Return all n-grams from a token list."""
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def mine_keyword_patterns(gsc_df: pd.DataFrame) -> pd.DataFrame:
    """Tokenise GSC queries, find common 2- and 3-word n-grams, and group
    by pattern.

    Parameters
    ----------
    gsc_df : pd.DataFrame
        Must contain columns: query, impressions, clicks, position, ctr.

    Returns
    -------
    pd.DataFrame
        Columns: pattern, frequency, avg_position, avg_ctr,
        total_impressions, total_clicks.
    """
    required = {"query", "impressions", "clicks", "position", "ctr"}
    missing = required - set(gsc_df.columns)
    if missing:
        raise ValueError(f"gsc_df missing required columns: {missing}")

    if gsc_df.empty:
        logger.warning("mine_keyword_patterns received an empty DataFrame.")
        return pd.DataFrame(
            columns=[
                "pattern", "frequency", "avg_position", "avg_ctr",
                "total_impressions", "total_clicks",
            ]
        )

    # Collect n-gram → list of row indices mapping
    pattern_rows: dict[tuple, list[int]] = {}

    for idx, row in gsc_df.iterrows():
        tokens = _tokenize(row["query"])
        seen: set[tuple] = set()
        for n in (2, 3):
            for gram in _ngrams(tokens, n):
                if gram not in seen:
                    seen.add(gram)
                    pattern_rows.setdefault(gram, []).append(idx)

    # Keep only patterns appearing 3+ times
    min_freq = 3
    records = []
    for gram, indices in pattern_rows.items():
        if len(indices) < min_freq:
            continue
        subset = gsc_df.loc[indices]
        records.append(
            {
                "pattern": " ".join(gram),
                "frequency": len(indices),
                "avg_position": round(float(subset["position"].mean()), 2),
                "avg_ctr": round(float(subset["ctr"].mean()), 4),
                "total_impressions": int(subset["impressions"].sum()),
                "total_clicks": int(subset["clicks"].sum()),
            }
        )

    result = pd.DataFrame(records)
    if not result.empty:
        result.sort_values("total_impressions", ascending=False, inplace=True)
        result.reset_index(drop=True, inplace=True)

    logger.info("Mined %d keyword patterns from %d queries.", len(result), len(gsc_df))
    return result


def find_content_gaps(
    gsc_df: pd.DataFrame,
    crawl_df: pd.DataFrame,
    impression_threshold: int = 50,
) -> pd.DataFrame:
    """Identify queries with impressions but no dedicated page.

    A query is considered a *gap* when its ranking URL is the homepage or
    another generic page (path is ``/``, ``/index``, etc.) while the query
    receives at least *impression_threshold* impressions.

    Parameters
    ----------
    gsc_df : pd.DataFrame
        Must contain: query, page, impressions, clicks, position, ctr.
    crawl_df : pd.DataFrame
        Must contain: url (canonical URLs from the crawl).
    impression_threshold : int
        Minimum impressions for a query to qualify as a gap.

    Returns
    -------
    pd.DataFrame
        Columns: query, page, impressions, clicks, position, ctr,
        gap_reason.
    """
    required_gsc = {"query", "page", "impressions", "clicks", "position", "ctr"}
    missing_gsc = required_gsc - set(gsc_df.columns)
    if missing_gsc:
        raise ValueError(f"gsc_df missing required columns: {missing_gsc}")

    if "url" not in crawl_df.columns:
        raise ValueError("crawl_df must contain a 'url' column.")

    if gsc_df.empty:
        logger.warning("find_content_gaps received an empty GSC DataFrame.")
        return pd.DataFrame(
            columns=[
                "query", "page", "impressions", "clicks",
                "position", "ctr", "gap_reason",
            ]
        )

    # Build set of known dedicated URLs from the crawl
    crawl_urls = set(crawl_df["url"].dropna().str.strip().str.rstrip("/"))

    # Filter to queries with sufficient impressions
    df = gsc_df[gsc_df["impressions"] >= impression_threshold].copy()

    gaps = []
    for _, row in df.iterrows():
        page = str(row["page"]).strip().rstrip("/")
        reason = None

        # Check 1: ranking page is generic / homepage
        try:
            from urllib.parse import urlparse

            path = urlparse(page).path.rstrip("/")
        except Exception:
            path = ""

        if path in GENERIC_PATH_TOKENS:
            reason = "ranking_on_homepage"

        # Check 2: ranking URL not in crawl (orphan / redirect target)
        if reason is None and page not in crawl_urls:
            reason = "ranking_url_not_in_crawl"

        if reason:
            gaps.append(
                {
                    "query": row["query"],
                    "page": row["page"],
                    "impressions": int(row["impressions"]),
                    "clicks": int(row["clicks"]),
                    "position": round(float(row["position"]), 2),
                    "ctr": round(float(row["ctr"]), 4),
                    "gap_reason": reason,
                }
            )

    result = pd.DataFrame(gaps)
    if not result.empty:
        result.sort_values("impressions", ascending=False, inplace=True)
        result.reset_index(drop=True, inplace=True)

    logger.info(
        "Found %d content gaps from %d qualifying queries.",
        len(result),
        len(df),
    )
    return result
