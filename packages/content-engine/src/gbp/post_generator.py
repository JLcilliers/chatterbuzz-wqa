"""Google Business Profile post generator using Claude AI."""

import json
import logging
import os
from typing import Optional

import anthropic
import httpx

logger = logging.getLogger(__name__)

GBP_API_BASE = "https://mybusiness.googleapis.com/v4"


def generate_gbp_post(
    business_name: str,
    industry: str,
    topic: str,
    brand_voice: dict,
    post_type: str = "STANDARD",
) -> dict:
    """Generate a GBP post using Claude AI.

    Args:
        business_name: Name of the business
        industry: Business industry
        topic: Post topic/theme
        brand_voice: Brand voice configuration
        post_type: STANDARD, EVENT, or OFFER

    Returns:
        Dict with: summary (1500 char max), call_to_action_type, url
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)
    tone = brand_voice.get("tone", "professional")

    prompt = f"""Write a Google Business Profile post for {business_name} ({industry}).

Topic: {topic}
Tone: {tone}
Post type: {post_type}

Requirements:
- Maximum 1500 characters for the summary
- Engaging and action-oriented
- Include a clear call to action
- Local relevance where possible

Return JSON: {{"summary": "...", "call_to_action_type": "LEARN_MORE|BOOK|ORDER|SHOP|SIGN_UP|CALL", "url": "/suggested-landing-page"}}"""

    response = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    return json.loads(text)


def publish_gbp_post(
    account_id: str,
    location_id: str,
    summary: str,
    call_to_action_type: str = "LEARN_MORE",
    url: str = "",
    access_token: str = "",
) -> dict:
    """Publish a post to Google Business Profile via API."""
    endpoint = f"{GBP_API_BASE}/accounts/{account_id}/locations/{location_id}/localPosts"

    payload: dict = {
        "languageCode": "en",
        "summary": summary[:1500],
        "topicType": "STANDARD",
    }

    if call_to_action_type and url:
        payload["callToAction"] = {
            "actionType": call_to_action_type,
            "url": url,
        }

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    with httpx.Client(timeout=30) as client:
        response = client.post(endpoint, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        logger.info(f"Published GBP post: {result.get('name', 'unknown')}")
        return result
