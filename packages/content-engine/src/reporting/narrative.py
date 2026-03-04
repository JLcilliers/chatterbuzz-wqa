"""Executive summary narrative generation using Claude AI."""

import json
import logging
import os
from typing import Optional

import anthropic

logger = logging.getLogger(__name__)


def generate_executive_summary(
    metrics: dict,
    anomalies: list[dict],
    client_name: str,
    report_month: str,
) -> str:
    """Generate an AI-powered executive summary for the monthly report.

    Args:
        metrics: Full metrics snapshot from DataCollector
        anomalies: List of detected anomalies
        client_name: Client business name
        report_month: YYYY-MM format

    Returns:
        Markdown-formatted executive summary
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set, generating basic summary")
        return _generate_basic_summary(metrics, anomalies, client_name, report_month)

    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""Generate a concise executive summary for {client_name}'s monthly SEO report ({report_month}).

Metrics data:
{json.dumps(metrics, indent=2, default=str)}

Anomalies detected:
{json.dumps(anomalies, indent=2, default=str) if anomalies else "None"}

Requirements:
- 3-5 paragraphs max
- Lead with the most important finding (positive or negative)
- Use specific numbers, not vague language
- Include MoM comparisons where data is available
- End with 2-3 recommended next steps
- Professional tone, suitable for a client-facing report
- Use Markdown formatting with bold for key numbers"""

    response = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text.strip()


def _generate_basic_summary(metrics: dict, anomalies: list[dict], client_name: str, report_month: str) -> str:
    """Fallback summary without AI."""
    traffic = metrics.get("organic_traffic", {})
    search = metrics.get("search_performance", {})

    lines = [
        f"## {client_name} — SEO Report ({report_month})",
        "",
        f"**Organic Traffic:** {traffic.get('total_sessions', 0):,} sessions with {traffic.get('engagement_rate', 0)}% engagement rate.",
        f"**Search Performance:** {search.get('total_clicks', 0):,} clicks from {search.get('total_impressions', 0):,} impressions (CTR: {search.get('avg_ctr', 0)}%).",
    ]

    if anomalies:
        lines.append("")
        lines.append(f"**Alerts:** {len(anomalies)} anomalies detected requiring attention.")
        for a in anomalies[:3]:
            lines.append(f"- {a.get('detail', 'Unknown anomaly')}")

    return "\n".join(lines)
