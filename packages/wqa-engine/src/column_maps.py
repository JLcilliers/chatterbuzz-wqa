"""
Column mapping dictionaries and provider protocol/registry definitions.

Defines standard column name mappings for crawl, GA, GSC, backlink, and keyword
data sources, along with abstract provider interfaces for future API integrations.
"""

import pandas as pd
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable, List, Dict


# =============================================================================
# COLUMN MAPS
# =============================================================================

CRAWL_COLUMN_MAP = {
    'url': ['address', 'url', 'page_url', 'page', 'page url', 'full url'],
    'status_code': ['status code', 'status_code', 'status', 'http_status', 'response_code', 'http status code'],
    'indexable': ['indexability', 'indexable', 'is_indexable', 'index_status', 'index status'],
    'meta_robots': ['meta robots 1', 'meta_robots', 'meta robots', 'robots', 'robots directive'],
    'canonical_url': ['canonical link element 1', 'canonical_url', 'canonical', 'canonical_link',
                      'canonical_tag', 'canonical url', 'canonicals'],
    'content_type': ['content type', 'content_type', 'mime type', 'type'],
    'inlinks': ['unique inlinks', 'inlinks', 'internal_inlinks', 'inlinks_count', 'internal inlinks',
                'internal links in', 'links in'],
    'outlinks': ['unique outlinks', 'outlinks', 'internal_outlinks', 'outlinks_count', 'internal outlinks',
                 'internal links out', 'links out'],
    'crawl_depth': ['crawl depth', 'crawl_depth', 'depth', 'click_depth', 'level', 'click depth',
                    'page depth', 'distance'],
    'in_sitemap': ['indexability status', 'in xml sitemap', 'in_sitemap', 'sitemap', 'in_xml_sitemap',
                   'sitemap_status', 'xml sitemap'],
    'page_title': ['title 1', 'page_title', 'title', 'meta_title', 'page title', 'seo title'],
    'meta_description': ['meta description 1', 'meta_description', 'description', 'meta description'],
    'h1': ['h1-1', 'h1', 'h1 1', 'heading 1', 'first h1'],
    'word_count': ['word count', 'word_count', 'words', 'content_word_count', 'text_word_count',
                   'text content', 'body word count'],
    'last_modified': ['last modified', 'last_modified', 'lastmod', 'modified', 'date modified',
                      'last-modified', 'modified date'],
}

GA_COLUMN_MAP = {
    'url': ['url', 'page_path', 'page', 'landing_page', 'page_location'],
    'sessions': ['sessions', 'session_count', 'visits'],
    'sessions_prev': ['sessions_prev', 'sessions_previous', 'previous_sessions', 'sessions_yoy',
                      'sessions_last_year', 'prior_sessions'],
    'conversions': ['conversions', 'goal_completions', 'conversion_count', 'key_events', 'keyEvents'],
    'bounce_rate': ['bounce_rate', 'bounceRate', 'bounce rate'],
    'avg_session_duration': ['avg_session_duration', 'averageSessionDuration', 'average session duration',
                             'session_duration', 'avgSessionDuration'],
    'ecom_revenue': ['ecom_revenue', 'revenue', 'transactionRevenue', 'purchaseRevenue', 'totalRevenue',
                     'ecommerce_revenue', 'total_revenue'],
}

GSC_COLUMN_MAP = {
    'url': ['url', 'page', 'top_pages', 'landing_page'],
    'avg_position': ['avg_position', 'position', 'average_position', 'avg_pos'],
    'ctr': ['ctr', 'click_through_rate', 'clickthrough_rate'],
    'clicks': ['clicks', 'click_count'],
    'impressions': ['impressions', 'impression_count'],
    'primary_keyword': ['primary_keyword', 'query', 'top_query', 'keyword'],
}

# Backlink Data Column Mapping - supports Ahrefs, Semrush, Moz, Majestic, etc.
# Supports both per-URL aggregated exports AND per-backlink raw exports (like SEMRush)
BACKLINK_COLUMN_MAP = {
    # Target URL (the page receiving the backlink)
    'url': ['url', 'target url', 'target_url', 'page', 'target', 'page url', 'target page'],
    # Source URL (the page providing the backlink) - used for per-backlink formats
    'source_url': ['source url', 'source_url', 'source', 'referring url', 'referring_url',
                   'from url', 'from_url', 'linking page'],
    # Pre-aggregated referring domains count
    'referring_domains': ['referring domains', 'referring_domains', 'ref domains', 'ref_domains',
                          'domains', 'rd', 'dofollow referring domains', 'root domains'],
    # Pre-aggregated backlinks count
    'backlinks': ['backlinks', 'backlink_count', 'external_links', 'links', 'total backlinks',
                  'dofollow backlinks', 'external backlinks', 'inbound links'],
    # Authority score (source page authority for per-backlink, target page for aggregated)
    'authority': ['ur', 'url rating', 'url_rating', 'authority score', 'authority_score',
                  'page authority', 'page_authority', 'pa', 'trust flow', 'citation flow',
                  'domain rating', 'dr', 'as', 'page ascore', 'ascore'],
    # Anchor text
    'anchor_texts': ['anchor_texts', 'anchors', 'anchor_text', 'anchor', 'top anchor'],
    # Nofollow indicator (for per-backlink formats)
    'nofollow': ['nofollow', 'no_follow', 'is_nofollow', 'rel_nofollow'],
}

# Keyword Tracking Data Column Mapping - supports SEMrush Position Tracking, Ahrefs Rank Tracker, etc.
KEYWORD_COLUMN_MAP = {
    # Target URL that ranks for the keyword
    'url': ['url', 'landing page', 'landing_page', 'page', 'page url', 'ranking url', 'target url'],
    # The keyword/query
    'keyword': ['keyword', 'query', 'search term', 'keyphrase', 'key phrase', 'target keyword'],
    # Search volume for the keyword
    'volume': ['volume', 'search volume', 'search_volume', 'monthly volume', 'avg. volume',
               'average volume', 'monthly searches'],
    # Current ranking position
    'position': ['position', 'rank', 'ranking', 'current position', 'current rank',
                 'google position', 'serp position'],
    # Previous ranking position (for tracking changes)
    'prev_position': ['previous position', 'prev_position', 'previous rank', 'last position',
                      'position change'],
    # Keyword difficulty
    'difficulty': ['difficulty', 'kd', 'keyword difficulty', 'kw difficulty', 'competition'],
    # CPC value
    'cpc': ['cpc', 'cost per click', 'avg cpc', 'avg. cpc'],
    # Intent classification
    'intent': ['intent', 'search intent', 'keyword intent'],
}


# =============================================================================
# API PROVIDER INTERFACES (Future Extension Points)
# =============================================================================
# These abstract base classes define the interface for future API integrations.
# Implement a provider by subclassing and registering with the provider registry.

@runtime_checkable
class CrawlProvider(Protocol):
    """Interface for crawl data providers (JetOctopus, custom crawlers, etc.)"""
    provider_name: str

    async def fetch_crawl_data(self, site_url: str, **options) -> pd.DataFrame:
        """Fetch crawl data for a site. Returns DataFrame with CRAWL_COLUMN_MAP fields."""
        ...

    def get_required_credentials(self) -> List[str]:
        """Return list of required credential keys (e.g., ['api_key', 'project_id'])"""
        ...


@runtime_checkable
class BacklinkProvider(Protocol):
    """Interface for backlink data providers (Ahrefs, Semrush, Moz, etc.)"""
    provider_name: str

    async def fetch_backlink_data(self, site_url: str, **options) -> pd.DataFrame:
        """Fetch backlink data for a site. Returns DataFrame with BACKLINK_COLUMN_MAP fields."""
        ...

    def get_required_credentials(self) -> List[str]:
        """Return list of required credential keys (e.g., ['api_key'])"""
        ...


# Provider registries (populated when providers are implemented)
CRAWL_PROVIDERS: Dict[str, type] = {}
BACKLINK_PROVIDERS: Dict[str, type] = {}

def register_crawl_provider(name: str):
    """Decorator to register a crawl provider"""
    def decorator(cls):
        CRAWL_PROVIDERS[name] = cls
        return cls
    return decorator

def register_backlink_provider(name: str):
    """Decorator to register a backlink provider"""
    def decorator(cls):
        BACKLINK_PROVIDERS[name] = cls
        return cls
    return decorator
