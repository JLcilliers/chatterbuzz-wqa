"""Google Business Profile review auto-responder."""

import json
import logging
import os
from typing import Optional

import anthropic
import httpx

logger = logging.getLogger(__name__)

GBP_API_BASE = "https://mybusiness.googleapis.com/v4"


def classify_review(star_rating: int) -> str:
    """Classify review as positive (4-5) or negative (1-3)."""
    return "positive" if star_rating >= 4 else "negative"


def generate_review_response(
    reviewer_name: str,
    star_rating: int,
    review_text: str,
    business_name: str,
    brand_voice: dict,
) -> Optional[str]:
    """Generate an AI response to a review.

    Only auto-responds to positive reviews (4-5 stars).
    Negative reviews (1-3 stars) return None — flagged for human review.
    """
    if star_rating < 4:
        logger.info(f"Negative review from {reviewer_name} ({star_rating} stars) — flagging for human review")
        return None

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)
    tone = brand_voice.get("tone", "professional")

    prompt = f"""Write a brief, genuine response to this positive Google review.

Business: {business_name}
Reviewer: {reviewer_name}
Rating: {star_rating}/5 stars
Review: {review_text}

Requirements:
- Tone: {tone}
- Thank the reviewer by name
- Reference something specific from their review
- Keep it under 300 characters
- Sound genuine, not templated
- Do NOT use emojis

Return only the response text, no JSON or formatting."""

    response = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def reply_to_review(
    account_id: str,
    location_id: str,
    review_id: str,
    reply_text: str,
    access_token: str,
) -> dict:
    """Post a reply to a GBP review via API."""
    endpoint = f"{GBP_API_BASE}/accounts/{account_id}/locations/{location_id}/reviews/{review_id}/reply"

    with httpx.Client(timeout=30) as client:
        response = client.put(
            endpoint,
            json={"comment": reply_text},
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()
        logger.info(f"Replied to review {review_id}")
        return response.json()


def get_reviews(
    account_id: str,
    location_id: str,
    access_token: str,
    page_size: int = 50,
) -> list[dict]:
    """Fetch recent reviews from GBP API."""
    endpoint = f"{GBP_API_BASE}/accounts/{account_id}/locations/{location_id}/reviews"

    with httpx.Client(timeout=30) as client:
        response = client.get(
            endpoint,
            params={"pageSize": page_size},
            headers={"Authorization": f"Bearer {access_token}"},
        )
        response.raise_for_status()
        data = response.json()
        return data.get("reviews", [])
