"""AI content generation engine using Claude Sonnet."""

import json
import logging
import os
from typing import Optional

import anthropic
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ContentOutput(BaseModel):
    """Structured output from content generation."""
    title: str
    meta_title: str = Field(max_length=60)
    meta_description: str = Field(max_length=160)
    content_html: str
    suggested_schema_type: str = ""
    internal_links: list[dict] = Field(default_factory=list)
    word_count: int = 0


def generate_content(
    system_prompt: str,
    user_prompt: str,
    model: str = "claude-sonnet-4-5-20250514",
    max_tokens: int = 4096,
) -> Optional[ContentOutput]:
    """Generate a single piece of content using Claude."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set")
        return None

    client = anthropic.Anthropic(api_key=api_key)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        text = response.content[0].text

        # Parse JSON from response
        # Handle markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        data = json.loads(text)
        return ContentOutput(**data)

    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logger.error(f"Failed to parse content response: {e}")
        return None
    except anthropic.APIError as e:
        logger.error(f"Anthropic API error: {e}")
        return None


def batch_generate(
    tasks: list[dict],
    system_prompt: str,
    max_per_run: int = 50,
    model: str = "claude-sonnet-4-5-20250514",
) -> list[Optional[ContentOutput]]:
    """Generate content in batches. Each task dict needs a 'user_prompt' key.

    Args:
        tasks: List of dicts with 'user_prompt' and optional metadata
        system_prompt: Shared system prompt for all generations
        max_per_run: Maximum items to process (default 50)
        model: Claude model to use

    Returns:
        List of ContentOutput objects (None for failures)
    """
    results = []
    batch = tasks[:max_per_run]

    for i, task in enumerate(batch):
        logger.info(f"Generating content {i + 1}/{len(batch)}: {task.get('title', 'untitled')}")
        result = generate_content(
            system_prompt=system_prompt,
            user_prompt=task["user_prompt"],
            model=model,
        )
        results.append(result)

    success_count = sum(1 for r in results if r is not None)
    logger.info(f"Batch complete: {success_count}/{len(batch)} successful")
    return results
