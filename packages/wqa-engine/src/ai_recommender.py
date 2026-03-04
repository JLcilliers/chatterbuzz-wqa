"""
AI-powered strategic recommendations for Website Quality Audit (WQA) URLs.

Uses the Anthropic Claude API to generate SEO recommendations.
"""

import logging
import os
from typing import List

import anthropic
import pandas as pd

logger = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


async def generate_ai_recommendations(df: pd.DataFrame) -> List[str]:
    """
    Generate AI-powered strategic recommendations for each URL using Claude Haiku.

    Processes URLs in batches of 20, caps at 500 URLs (ranked by sessions + impressions).
    Returns a list of recommendation strings aligned to the DataFrame index.
    Falls back to empty strings on any error.
    """
    if not ANTHROPIC_API_KEY:
        logger.warning("AI recommendations skipped: Anthropic not available or API key not set")
        return [''] * len(df)

    try:
        client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize Anthropic client: {e}")
        return [''] * len(df)

    recommendations = [''] * len(df)

    # Rank URLs by importance (sessions + impressions) and cap at 500
    df_work = df.copy()
    df_work['_importance'] = (
        pd.to_numeric(df_work.get('sessions', pd.Series(dtype=float)), errors='coerce').fillna(0) +
        pd.to_numeric(df_work.get('impressions', pd.Series(dtype=float)), errors='coerce').fillna(0)
    )
    top_indices = df_work.nlargest(500, '_importance').index.tolist()

    BATCH_SIZE = 20
    batches = [top_indices[i:i + BATCH_SIZE] for i in range(0, len(top_indices), BATCH_SIZE)]

    for batch_indices in batches:
        try:
            # Build context for this batch
            url_entries = []
            for idx_num, idx in enumerate(batch_indices):
                row = df.iloc[idx]
                url_entries.append(
                    f"[{idx_num}] URL: {row.get('url', '')}\n"
                    f"    Title: {row.get('page_title', '')}\n"
                    f"    Meta: {str(row.get('meta_description', ''))[:150]}\n"
                    f"    Sessions: {row.get('sessions', 0)} | Impressions: {row.get('impressions', 0)} | "
                    f"CTR: {row.get('ctr', '')} | Position: {row.get('avg_position', '')}\n"
                    f"    Word Count: {row.get('word_count', 0)} | Status: {row.get('status_code', '')}\n"
                    f"    Tech Actions: {row.get('technical_actions', '')} | "
                    f"Content Actions: {row.get('content_actions', '')}"
                )

            prompt = (
                "You are an expert SEO strategist. For each URL below, provide a 1-2 sentence "
                "strategic recommendation that goes beyond the existing rule-based actions. "
                "Focus on content strategy, user intent, competitive positioning, or conversion optimization.\n\n"
                "Format your response EXACTLY as:\n"
                "[0] Your recommendation here\n"
                "[1] Your recommendation here\n"
                "...and so on for each URL.\n\n"
                + "\n\n".join(url_entries)
            )

            response = await client.messages.create(
                model="claude-haiku-4-20250414",
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse response
            response_text = response.content[0].text
            for line in response_text.strip().split('\n'):
                line = line.strip()
                if line.startswith('[') and ']' in line:
                    try:
                        bracket_end = line.index(']')
                        local_idx = int(line[1:bracket_end])
                        rec_text = line[bracket_end + 1:].strip()
                        if rec_text.startswith('-'):
                            rec_text = rec_text[1:].strip()
                        if 0 <= local_idx < len(batch_indices):
                            recommendations[batch_indices[local_idx]] = rec_text
                    except (ValueError, IndexError):
                        continue

        except Exception as e:
            logger.warning(f"AI recommendation batch failed: {e}")
            continue

    filled = sum(1 for r in recommendations if r)
    logger.info(f"AI recommendations generated: {filled}/{len(df)} URLs")
    return recommendations
