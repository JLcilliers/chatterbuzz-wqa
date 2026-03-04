"""
Rule-based action assignment and priority scoring for Website Quality Audit (WQA).

Contains functions to assign technical/content actions and priority levels to URLs.
"""

from typing import List, Tuple


def assign_actions(row, low_traffic_threshold=5, thin_content_threshold=1000,
                   high_rank_max_position=20.0, low_ctr_threshold=0.05) -> Tuple[List[str], List[str]]:
    """
    Assign Technical Actions and Content Actions to a URL.

    Technical Actions: Infrastructure/SEO technical changes (redirects, schema, sitemaps, internal links)
    Content Actions: Content quality changes (rewrite, refresh, update metadata) - NOT redirects
    """
    technical_actions = []
    content_actions = []

    status_code = int(row.get('status_code', 0))
    indexable = bool(row.get('indexable', True))
    canonical_url = str(row.get('canonical_url', '')).strip()
    url = str(row.get('url', '')).strip().lower()
    inlinks = int(row.get('inlinks', 0))
    crawl_depth = int(row.get('crawl_depth', 0))
    in_sitemap = bool(row.get('in_sitemap', False))
    page_title = str(row.get('page_title', '')).strip()
    meta_description = str(row.get('meta_description', '')).strip()
    word_count = int(row.get('word_count', 0))
    sessions = float(row.get('sessions', 0))
    conversions = float(row.get('conversions', 0))
    avg_position = float(row.get('avg_position', 0))
    ctr = float(row.get('ctr', 0))
    impressions = float(row.get('impressions', 0))
    referring_domains = int(row.get('referring_domains', 0))
    backlinks = int(row.get('backlinks', 0))
    page_type = str(row.get('page_type', 'Other'))

    # Track if page should be removed/redirected (affects content action assignment)
    marked_for_removal = False

    # =========================================================================
    # TECHNICAL ACTIONS - Infrastructure/SEO technical changes
    # =========================================================================

    # Status code fixes (redirects are TECHNICAL, not content)
    if status_code == 302:
        technical_actions.append('Convert to 301')

    if status_code == 404:
        if backlinks > 0 or referring_domains > 0:
            technical_actions.append('301 Redirect (has backlinks)')
            marked_for_removal = True
        elif inlinks > 0:
            technical_actions.append('Remove Internal Links')
            marked_for_removal = True
        else:
            marked_for_removal = True

    if status_code in [301, 308]:
        technical_actions.append('Review Redirect Chain')
        marked_for_removal = True

    # Low-value pages that should be redirected (TECHNICAL decision)
    if (status_code == 200 and sessions <= low_traffic_threshold and conversions == 0):
        if backlinks > 0 or referring_domains > 0:
            # Has link equity - redirect to preserve it
            technical_actions.append('301 Redirect (low value, has links)')
            marked_for_removal = True
        elif sessions == 0 and impressions == 0 and word_count < 300:
            # Zero-value thin content - delete
            technical_actions.append('Delete (404)')
            marked_for_removal = True

    # Sitemap management
    if not indexable and in_sitemap:
        technical_actions.append('Remove from Sitemap')
    if indexable and status_code == 200 and sessions > 0 and not in_sitemap:
        technical_actions.append('Add to Sitemap')

    # Canonicalization
    if canonical_url and canonical_url.lower() != url:
        technical_actions.append('Review Canonical')

    # Internal linking
    if status_code == 200 and not marked_for_removal:
        if sessions > 0 and inlinks == 0:
            technical_actions.append('Add Internal Links')
        important_pages = ['Home Page', 'Landing Page', 'Local Lander', 'Lead Generation']
        if crawl_depth >= 4 and page_type in important_pages:
            if 'Add Internal Links' not in technical_actions:
                technical_actions.append('Improve Page Depth')

    # Schema markup recommendations
    if status_code == 200 and not marked_for_removal:
        if page_type in ['Blog Post']:
            technical_actions.append('Add Schema: Article')
        elif page_type == 'Local Lander':
            technical_actions.append('Add Schema: LocalBusiness')
        elif page_type == 'Home Page':
            technical_actions.append('Add Schema: Organization')
        elif page_type == 'Resource / Guide':
            technical_actions.append('Add Schema: HowTo/FAQ')
        elif page_type == 'Lead Generation':
            technical_actions.append('Add Schema: ContactPage')

    # =========================================================================
    # CONTENT ACTIONS - Content quality improvements (only for live pages)
    # =========================================================================

    if status_code == 200 and not marked_for_removal:
        # Thin content needs rewrite
        if (word_count < thin_content_threshold and
            (sessions > 0 or avg_position > 0 or impressions > 0)):
            content_actions.append('Rewrite (Thin Content)')

        # Ranking content that could be improved
        elif word_count >= thin_content_threshold:
            # Content refresh for pages ranking but could do better
            if (sessions > 0 and 2 < avg_position <= high_rank_max_position):
                content_actions.append('Refresh')

            # Link building target for pages with no backlinks
            if (sessions > 0 and 3 <= avg_position <= 20 and
                referring_domains == 0 and backlinks == 0):
                content_actions.append('Target w/ Links')

        # Metadata improvements
        if not meta_description:
            content_actions.append('Add Meta Description')
        elif (avg_position > 0 and avg_position <= high_rank_max_position and
              ctr < low_ctr_threshold and impressions > 100):
            content_actions.append('Improve Meta Description')

        if not page_title:
            content_actions.append('Add Page Title')
        elif (avg_position > 0 and avg_position <= high_rank_max_position and
              ctr < low_ctr_threshold and impressions > 100):
            content_actions.append('Improve Page Title')

    # Default if no content actions needed
    if not content_actions and status_code == 200 and not marked_for_removal:
        content_actions.append('Leave As Is')

    return technical_actions, content_actions


def assign_priority(row) -> str:
    """Assign priority based on page importance and required actions."""
    status_code = int(row.get('status_code', 0))
    page_type = str(row.get('page_type', 'Other'))
    technical_actions = str(row.get('technical_actions', ''))
    content_actions = str(row.get('content_actions', ''))

    # High-priority page types
    important_pages = ['Home Page', 'Landing Page', 'Local Lander', 'Lead Generation']

    # Error status codes are always high priority
    if status_code in [404, 302, 500, 502, 503]:
        return 'High'

    # Important pages needing content work are high priority
    if page_type in important_pages:
        if 'Rewrite' in content_actions or 'Refresh' in content_actions:
            return 'High'
        if '301 Redirect' in technical_actions or 'Delete' in technical_actions:
            return 'High'  # Something's wrong with an important page

    # Redirect/delete decisions for less important pages
    if '301 Redirect' in technical_actions or 'Delete' in technical_actions:
        return 'Medium'

    # Any pages needing content work
    if 'Rewrite' in content_actions or 'Refresh' in content_actions:
        return 'Medium'

    return 'Low'
