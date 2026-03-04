"""
Keyword Clusterer — groups keywords by token overlap, assigns search
intent and page type, and scores business relevance.
"""

import logging
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intent signals
# ---------------------------------------------------------------------------

_INTENT_SIGNALS = {
    "transactional": {
        "buy", "purchase", "order", "price", "pricing", "cost", "cheap",
        "deal", "discount", "coupon", "hire", "book", "schedule",
        "quote", "estimate", "shop", "store",
    },
    "commercial": {
        "best", "top", "review", "reviews", "compare", "comparison",
        "vs", "versus", "alternative", "alternatives", "recommended",
        "rated", "pros", "cons",
    },
    "navigational": {
        "login", "signin", "sign-in", "dashboard", "account", "portal",
        "app", "website", "official", "contact",
    },
    # Everything else falls through to informational
}

_PAGE_TYPE_MAP = {
    "transactional": "service_page",
    "commercial": "comparison_page",
    "navigational": "landing_page",
    "informational": "blog_post",
}

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


def _token_set(text: str) -> set[str]:
    """Return a set of meaningful tokens for a keyword string."""
    return {
        tok
        for tok in str(text).lower().split()
        if tok not in _STOP_WORDS and len(tok) > 1
    }


def _classify_intent(keyword: str) -> str:
    """Return the search intent label for a keyword."""
    tokens = set(str(keyword).lower().split())
    for intent, signals in _INTENT_SIGNALS.items():
        if tokens & signals:
            return intent
    return "informational"


def _token_overlap(set_a: set[str], set_b: set[str]) -> float:
    """Jaccard-style overlap: |intersection| / |smaller set|."""
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / min(len(set_a), len(set_b))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def cluster_keywords(
    keywords_df: pd.DataFrame,
    overlap_threshold: float = 0.5,
    supabase_client=None,
) -> pd.DataFrame:
    """Cluster keywords by token overlap and assign intent + page type.

    Parameters
    ----------
    keywords_df : pd.DataFrame
        Must contain a ``keyword`` column. Optional columns such as
        ``intent``, ``estimated_volume_tier``, and ``parent_topic`` will
        be preserved if present.
    overlap_threshold : float
        Minimum token-overlap ratio to merge two keywords into the same
        cluster (default 0.5 = 50 %).
    supabase_client : optional
        If provided, upserts results into the ``keyword_clusters`` table.

    Returns
    -------
    pd.DataFrame
        Original columns plus: cluster_id, cluster_label, intent,
        page_type.
    """
    if "keyword" not in keywords_df.columns:
        raise ValueError("keywords_df must contain a 'keyword' column.")

    if keywords_df.empty:
        logger.warning("cluster_keywords received an empty DataFrame.")
        return keywords_df.assign(
            cluster_id=pd.Series(dtype="int"),
            cluster_label=pd.Series(dtype="str"),
            intent=pd.Series(dtype="str"),
            page_type=pd.Series(dtype="str"),
        )

    keywords = keywords_df["keyword"].astype(str).tolist()
    token_sets = [_token_set(kw) for kw in keywords]

    # Greedy single-pass clustering
    cluster_ids = [-1] * len(keywords)
    cluster_centroids: list[set[str]] = []  # representative token set
    next_cluster = 0

    for i, ts in enumerate(token_sets):
        best_cluster = -1
        best_overlap = 0.0

        for cid, centroid in enumerate(cluster_centroids):
            ov = _token_overlap(ts, centroid)
            if ov >= overlap_threshold and ov > best_overlap:
                best_overlap = ov
                best_cluster = cid

        if best_cluster >= 0:
            cluster_ids[i] = best_cluster
            # Grow centroid
            cluster_centroids[best_cluster] = (
                cluster_centroids[best_cluster] | ts
            )
        else:
            cluster_ids[i] = next_cluster
            cluster_centroids.append(ts.copy())
            next_cluster += 1

    # Build cluster labels from most-common tokens in each cluster
    cluster_labels: dict[int, str] = {}
    for cid in range(next_cluster):
        member_indices = [j for j, c in enumerate(cluster_ids) if c == cid]
        all_tokens: list[str] = []
        for j in member_indices:
            all_tokens.extend(token_sets[j])
        common = Counter(all_tokens).most_common(3)
        cluster_labels[cid] = " ".join(tok for tok, _ in common)

    # Assign intent per keyword; then majority-vote per cluster
    kw_intents = [_classify_intent(kw) for kw in keywords]

    cluster_intent: dict[int, str] = {}
    for cid in range(next_cluster):
        member_intents = [
            kw_intents[j] for j, c in enumerate(cluster_ids) if c == cid
        ]
        cluster_intent[cid] = Counter(member_intents).most_common(1)[0][0]

    result = keywords_df.copy()
    result["cluster_id"] = cluster_ids
    result["cluster_label"] = [cluster_labels[c] for c in cluster_ids]
    result["intent"] = [cluster_intent[c] for c in cluster_ids]
    result["page_type"] = result["intent"].map(_PAGE_TYPE_MAP).fillna("blog_post")

    logger.info(
        "Clustered %d keywords into %d clusters (threshold=%.2f).",
        len(keywords),
        next_cluster,
        overlap_threshold,
    )

    # Persist to Supabase if client provided
    if supabase_client is not None:
        _write_to_supabase(supabase_client, result)

    return result


def score_business_relevance(
    clusters_df: pd.DataFrame,
    business_rules: dict,
) -> pd.DataFrame:
    """Score each cluster 0-100 based on business relevance.

    Parameters
    ----------
    clusters_df : pd.DataFrame
        Output of :func:`cluster_keywords`.
    business_rules : dict
        Configuration with optional keys:

        - ``priority_keywords`` : list[str] — keywords / fragments that
          boost relevance.
        - ``priority_intents`` : list[str] — intents that are most
          valuable (e.g. ["transactional", "commercial"]).
        - ``location`` : str — geographic qualifier; clusters whose
          label contains this string get a boost.
        - ``business_type`` : str — industry descriptor; matching
          cluster labels get a boost.

    Returns
    -------
    pd.DataFrame
        Same as input plus a ``relevance_score`` column (0-100).
    """
    if clusters_df.empty:
        return clusters_df.assign(relevance_score=pd.Series(dtype="float"))

    priority_kws = {
        kw.lower() for kw in business_rules.get("priority_keywords", [])
    }
    priority_intents = {
        i.lower() for i in business_rules.get("priority_intents", [])
    }
    location = str(business_rules.get("location", "")).lower()
    biz_type = str(business_rules.get("business_type", "")).lower()

    scores: list[float] = []

    for _, row in clusters_df.iterrows():
        score = 0.0
        kw = str(row.get("keyword", "")).lower()
        label = str(row.get("cluster_label", "")).lower()
        intent = str(row.get("intent", "")).lower()

        # +30 if keyword matches a priority keyword fragment
        for pkw in priority_kws:
            if pkw in kw:
                score += 30
                break

        # +25 if intent is a priority intent
        if intent in priority_intents:
            score += 25

        # +20 if cluster label contains business type tokens
        if biz_type and biz_type in label:
            score += 20

        # +15 if keyword or label contains location
        if location and (location in kw or location in label):
            score += 15

        # +10 baseline for having a cluster assignment
        score += 10

        scores.append(min(score, 100.0))

    result = clusters_df.copy()
    result["relevance_score"] = scores
    result.sort_values("relevance_score", ascending=False, inplace=True)
    result.reset_index(drop=True, inplace=True)

    logger.info(
        "Scored %d rows. Mean relevance = %.1f.",
        len(result),
        np.mean(scores),
    )
    return result


# ---------------------------------------------------------------------------
# Supabase persistence
# ---------------------------------------------------------------------------


def _write_to_supabase(client, df: pd.DataFrame) -> None:
    """Upsert cluster rows into the ``keyword_clusters`` table."""
    try:
        records = df[
            ["keyword", "cluster_id", "cluster_label", "intent", "page_type"]
        ].to_dict(orient="records")

        if not records:
            return

        client.table("keyword_clusters").upsert(
            records, on_conflict="keyword"
        ).execute()
        logger.info("Upserted %d rows to keyword_clusters.", len(records))
    except Exception as exc:
        logger.error("Supabase upsert failed: %s", exc)
