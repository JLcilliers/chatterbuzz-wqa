"""
AI Keyword Expander — uses Claude Sonnet to generate semantically related
keyword suggestions from a set of seed keywords.
"""

import json
import logging
import os
from typing import List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an expert SEO keyword researcher. Given a list of seed keywords, "
    "an industry, and optionally a geographic location, generate expanded keyword "
    "suggestions. For EACH suggestion return a JSON object with exactly these keys:\n"
    "  keyword        — the suggested keyword phrase\n"
    "  intent         — one of: informational, transactional, commercial, navigational\n"
    "  estimated_volume_tier — one of: high, medium, low\n"
    "  parent_topic   — a short parent topic that groups related keywords\n\n"
    "Return ONLY a JSON array of objects. No markdown fences, no commentary."
)


def _build_user_prompt(
    seeds: list[str], industry: str, location: str
) -> str:
    location_part = f" in {location}" if location else ""
    seed_list = "\n".join(f"- {s}" for s in seeds)
    return (
        f"Industry: {industry}{location_part}\n\n"
        f"Seed keywords:\n{seed_list}\n\n"
        "Generate 5-10 expanded keyword suggestions per seed keyword. "
        "Return the full list as a single JSON array."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def expand_keywords(
    seed_keywords: list[str],
    industry: str,
    location: str = "",
    batch_size: int = 20,
) -> list[dict]:
    """Send batches of seed keywords to Claude Sonnet for expansion.

    Parameters
    ----------
    seed_keywords : list[str]
        Starting keywords to expand from.
    industry : str
        The client's industry / vertical (e.g. "plumbing", "SaaS").
    location : str, optional
        Geographic qualifier (e.g. "Richmond, VA").
    batch_size : int
        Maximum number of seeds to send per API call.

    Returns
    -------
    list[dict]
        Each dict contains: keyword, intent, estimated_volume_tier,
        parent_topic.
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "The anthropic package is required. Install it with: "
            "pip install anthropic"
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable is not set."
        )

    client = anthropic.Anthropic(api_key=api_key)
    all_results: list[dict] = []
    seen_keywords: set[str] = set()

    # Process in batches
    batches = [
        seed_keywords[i : i + batch_size]
        for i in range(0, len(seed_keywords), batch_size)
    ]

    for batch_idx, batch in enumerate(batches, start=1):
        logger.info(
            "Expanding batch %d/%d (%d seeds).",
            batch_idx,
            len(batches),
            len(batch),
        )

        user_prompt = _build_user_prompt(batch, industry, location)

        try:
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )

            raw_text = message.content[0].text.strip()

            # Handle possible markdown code fences
            if raw_text.startswith("```"):
                raw_text = raw_text.split("\n", 1)[-1]
                if raw_text.endswith("```"):
                    raw_text = raw_text[:-3].strip()

            suggestions = json.loads(raw_text)

            if not isinstance(suggestions, list):
                logger.warning("Batch %d: expected list, got %s", batch_idx, type(suggestions))
                continue

            for item in suggestions:
                if not isinstance(item, dict):
                    continue
                kw = str(item.get("keyword", "")).strip().lower()
                if not kw or kw in seen_keywords:
                    continue
                seen_keywords.add(kw)
                all_results.append(
                    {
                        "keyword": kw,
                        "intent": str(item.get("intent", "informational")).lower(),
                        "estimated_volume_tier": str(
                            item.get("estimated_volume_tier", "low")
                        ).lower(),
                        "parent_topic": str(item.get("parent_topic", "")).strip(),
                    }
                )

        except json.JSONDecodeError as exc:
            logger.error("Batch %d: JSON parse error — %s", batch_idx, exc)
        except Exception as exc:
            logger.error("Batch %d: API call failed — %s", batch_idx, exc)

    logger.info(
        "AI expansion complete. %d unique keywords from %d seeds.",
        len(all_results),
        len(seed_keywords),
    )
    return all_results
