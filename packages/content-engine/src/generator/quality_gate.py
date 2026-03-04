"""Content quality gate — 100-point scoring system."""

import re
import logging
from typing import Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class QualityScore(BaseModel):
    """Quality assessment result."""
    total: int  # 0-100
    relevance: int  # 0-25
    depth: int  # 0-25
    readability: int  # 0-25
    seo: int  # 0-25
    passed: bool
    issues: list[str]


def score_content(
    content_html: str,
    meta_title: str,
    meta_description: str,
    target_keyword: str,
    min_word_count: int = 800,
    threshold: int = 70,
) -> QualityScore:
    """Score content on a 100-point scale across 4 dimensions.

    - Relevance (25pts): keyword presence, topic alignment
    - Depth (25pts): word count, heading structure, sections
    - Readability (25pts): sentence length, paragraph structure
    - SEO (25pts): meta tags, keyword placement, internal links
    """
    issues = []
    text = re.sub(r"<[^>]+>", " ", content_html)
    words = text.split()
    word_count = len(words)
    keyword_lower = target_keyword.lower()
    text_lower = text.lower()

    # --- Relevance (0-25) ---
    relevance = 0
    keyword_count = text_lower.count(keyword_lower)
    keyword_density = keyword_count / max(word_count, 1) * 100

    if keyword_count >= 3:
        relevance += 10
    elif keyword_count >= 1:
        relevance += 5
    else:
        issues.append("Target keyword not found in content")

    if 0.5 <= keyword_density <= 2.5:
        relevance += 10
    elif keyword_density > 0:
        relevance += 5
    else:
        issues.append("Keyword density too low")

    # Keyword in first 100 words
    first_100 = " ".join(words[:100]).lower()
    if keyword_lower in first_100:
        relevance += 5
    else:
        issues.append("Keyword not in first 100 words")

    # --- Depth (0-25) ---
    depth = 0
    if word_count >= min_word_count:
        depth += 10
    elif word_count >= min_word_count * 0.7:
        depth += 5
    else:
        issues.append(f"Content too short: {word_count} words (min {min_word_count})")

    headings = re.findall(r"<h[2-4][^>]*>", content_html, re.IGNORECASE)
    if len(headings) >= 3:
        depth += 8
    elif len(headings) >= 1:
        depth += 4
    else:
        issues.append("No subheadings found")

    # Lists and structured content
    has_lists = bool(re.search(r"<[ou]l[^>]*>", content_html, re.IGNORECASE))
    if has_lists:
        depth += 4

    # FAQ section
    has_faq = bool(re.search(r"faq|frequently asked|common questions", text_lower))
    if has_faq:
        depth += 3

    # --- Readability (0-25) ---
    readability = 0
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)

    if 10 <= avg_sentence_length <= 20:
        readability += 10
    elif avg_sentence_length < 25:
        readability += 5
    else:
        issues.append(f"Average sentence length too high: {avg_sentence_length:.0f} words")

    paragraphs = re.split(r"\n\n|<p[^>]*>|</p>", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    if len(paragraphs) >= 5:
        readability += 8
    elif len(paragraphs) >= 3:
        readability += 4

    # Short paragraphs (under 100 words each)
    long_paragraphs = sum(1 for p in paragraphs if len(p.split()) > 100)
    if long_paragraphs == 0:
        readability += 7
    elif long_paragraphs <= 2:
        readability += 3
    else:
        issues.append("Some paragraphs are too long")

    # --- SEO (0-25) ---
    seo = 0

    # Meta title
    if meta_title and 30 <= len(meta_title) <= 60:
        seo += 5
    elif meta_title:
        seo += 2
        issues.append(f"Meta title length: {len(meta_title)} chars (ideal: 50-60)")
    else:
        issues.append("Missing meta title")

    # Meta description
    if meta_description and 120 <= len(meta_description) <= 160:
        seo += 5
    elif meta_description:
        seo += 2
        issues.append(f"Meta description length: {len(meta_description)} chars (ideal: 150-160)")
    else:
        issues.append("Missing meta description")

    # Keyword in meta title
    if keyword_lower in (meta_title or "").lower():
        seo += 5
    else:
        issues.append("Target keyword not in meta title")

    # Keyword in meta description
    if keyword_lower in (meta_description or "").lower():
        seo += 3

    # Internal links
    internal_link_count = len(re.findall(r"\[INTERNAL_LINK:", content_html))
    if internal_link_count >= 2:
        seo += 4
    elif internal_link_count >= 1:
        seo += 2
    else:
        issues.append("No internal linking opportunities marked")

    # H1/Title has keyword
    h1_match = re.search(r"<h1[^>]*>(.*?)</h1>", content_html, re.IGNORECASE)
    if h1_match and keyword_lower in h1_match.group(1).lower():
        seo += 3

    total = relevance + depth + readability + seo
    passed = total >= threshold

    return QualityScore(
        total=min(total, 100),
        relevance=min(relevance, 25),
        depth=min(depth, 25),
        readability=min(readability, 25),
        seo=min(seo, 25),
        passed=passed,
        issues=issues,
    )
