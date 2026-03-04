"""Internal link injector — max 3 per page, topical relevance via token overlap."""

import re
import logging
from collections import Counter
from typing import Optional

logger = logging.getLogger(__name__)

MAX_LINKS_PER_PAGE = 3
MIN_RELEVANCE_SCORE = 0.3


def tokenize(text: str) -> set[str]:
    """Extract meaningful tokens from text."""
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    stopwords = {"the", "and", "for", "are", "but", "not", "you", "all", "can", "has",
                 "her", "was", "one", "our", "out", "his", "had", "how", "its", "may",
                 "who", "did", "get", "let", "say", "she", "too", "use", "this", "that",
                 "with", "have", "from", "they", "been", "will", "more", "when", "what"}
    return {w for w in words if w not in stopwords}


def compute_relevance(source_tokens: set[str], target_tokens: set[str]) -> float:
    """Compute token overlap relevance between two pages."""
    if not source_tokens or not target_tokens:
        return 0.0
    overlap = source_tokens & target_tokens
    return len(overlap) / min(len(source_tokens), len(target_tokens))


def find_link_opportunities(
    source_url: str,
    source_content: str,
    candidate_pages: list[dict],
    max_links: int = MAX_LINKS_PER_PAGE,
    min_relevance: float = MIN_RELEVANCE_SCORE,
) -> list[dict]:
    """Find the best internal link opportunities for a source page.

    Args:
        source_url: URL of the page being linked from
        source_content: HTML/text content of the source page
        candidate_pages: List of dicts with 'url', 'title', 'content' keys
        max_links: Maximum number of links to inject (default 3)
        min_relevance: Minimum relevance score threshold

    Returns:
        List of dicts with: target_url, target_title, anchor_text, relevance_score
    """
    source_tokens = tokenize(source_content)
    scored = []

    for page in candidate_pages:
        if page["url"] == source_url:
            continue
        target_tokens = tokenize(page.get("content", "") + " " + page.get("title", ""))
        relevance = compute_relevance(source_tokens, target_tokens)
        if relevance >= min_relevance:
            scored.append({
                "target_url": page["url"],
                "target_title": page.get("title", ""),
                "anchor_text": page.get("title", page["url"]),
                "relevance_score": round(relevance, 3),
            })

    # Sort by relevance descending, take top N
    scored.sort(key=lambda x: x["relevance_score"], reverse=True)
    return scored[:max_links]


def inject_links_into_html(
    html: str,
    links: list[dict],
) -> str:
    """Inject internal links into HTML content at natural positions.

    Finds the first occurrence of anchor text in a paragraph and wraps it in <a>.
    """
    for link in links:
        anchor = link["anchor_text"]
        url = link["target_url"]

        # Find anchor text in content (case-insensitive, whole word)
        pattern = re.compile(re.escape(anchor), re.IGNORECASE)
        match = pattern.search(html)
        if match:
            matched_text = match.group(0)
            replacement = f'<a href="{url}" title="{anchor}">{matched_text}</a>'
            html = html[:match.start()] + replacement + html[match.end():]

    return html
