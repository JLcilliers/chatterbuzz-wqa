"""Dynamic system prompts built from business rules and brand voice."""

from typing import Optional


def build_system_prompt(
    business_name: str,
    industry: str,
    business_type: str,
    brand_voice: dict,
    content_type: str,
    target_location: str = "",
) -> str:
    """Build a system prompt for content generation from business context."""
    tone = brand_voice.get("tone", "professional")
    style = brand_voice.get("style", "informative")
    avoid = brand_voice.get("avoid", [])

    avoid_section = ""
    if avoid:
        avoid_section = f"\n\nIMPORTANT: Never use or reference: {', '.join(avoid)}."

    location_section = ""
    if target_location:
        location_section = f"\nTarget location context: {target_location}. Include local relevance naturally."

    return f"""You are an expert SEO content writer for {business_name}, a {business_type} in the {industry} industry.

Writing style:
- Tone: {tone}
- Style: {style}
- Content type: {content_type}
{location_section}

Guidelines:
- Write comprehensive, well-structured content with proper heading hierarchy (H2, H3)
- Include relevant internal linking opportunities marked as [INTERNAL_LINK: anchor text -> /suggested-path]
- Optimize for featured snippets where appropriate
- Include a compelling meta title (50-60 chars) and meta description (150-160 chars)
- Use natural keyword placement without stuffing
- Include FAQ section where relevant (for FAQPage schema)
- Target 800+ words for landing pages, 1200+ words for blog posts
{avoid_section}

Output format:
Return a JSON object with keys: title, meta_title, meta_description, content_html, suggested_schema_type, internal_links, word_count"""


def build_rewrite_prompt(
    original_content: str,
    target_keyword: str,
    issues: list[str],
    brand_voice: dict,
) -> str:
    """Build a prompt for content rewriting/optimization."""
    tone = brand_voice.get("tone", "professional")
    issues_text = "\n".join(f"- {issue}" for issue in issues)

    return f"""Rewrite and optimize the following content.

Target keyword: {target_keyword}
Tone: {tone}

Issues to fix:
{issues_text}

Original content:
{original_content}

Requirements:
- Maintain the core message but improve SEO optimization
- Fix all listed issues
- Ensure the target keyword appears naturally in title, H1, first paragraph, and throughout
- Improve readability and structure
- Add FAQ section if appropriate

Output format:
Return a JSON object with keys: title, meta_title, meta_description, content_html, word_count, changes_made"""
