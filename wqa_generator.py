#!/usr/bin/env python3
"""
Website Quality Audit (WQA) Generator

A Python tool that automatically generates Website Quality Audit reports for SEO agencies.
Ingests multiple CSV exports (crawl, GA4, GSC, backlinks), joins them into a single
URL-level dataset, applies rule-based SEO logic, and exports comprehensive Excel reports.

Usage:
    python wqa_generator.py --crawl crawl.csv --output wqa_report.xlsx
    python wqa_generator.py --crawl crawl.csv --ga ga.csv --gsc gsc.csv --backlinks backlinks.csv --output wqa_report.xlsx

Author: Generated for SEO Agency WQA automation
Python Version: 3.11+
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# COLUMN MAPPING CONFIGURATION
# =============================================================================
# Adapt these mappings if your CSV exports use different column names

CRAWL_COLUMN_MAP = {
    'url': ['url', 'address', 'page_url', 'page'],
    'status_code': ['status_code', 'status', 'http_status', 'response_code'],
    'indexable': ['indexable', 'indexability', 'is_indexable', 'index_status'],
    'canonical_url': ['canonical_url', 'canonical', 'canonical_link', 'canonical_tag'],
    'inlinks': ['inlinks', 'internal_inlinks', 'inlinks_count', 'unique_inlinks'],
    'outlinks': ['outlinks', 'internal_outlinks', 'outlinks_count', 'unique_outlinks'],
    'crawl_depth': ['crawl_depth', 'depth', 'click_depth', 'level'],
    'in_sitemap': ['in_sitemap', 'sitemap', 'in_xml_sitemap', 'sitemap_status'],
    'page_title': ['page_title', 'title', 'title_1', 'meta_title'],
    'meta_description': ['meta_description', 'description', 'meta_description_1'],
    'word_count': ['word_count', 'words', 'content_word_count', 'text_word_count'],
}

GA_COLUMN_MAP = {
    'url': ['url', 'page_path', 'page', 'landing_page', 'page_location'],
    'sessions': ['sessions', 'session_count', 'visits'],
    'conversions': ['conversions', 'goal_completions', 'conversion_count', 'key_events'],
}

GSC_COLUMN_MAP = {
    'url': ['url', 'page', 'top_pages', 'landing_page'],
    'avg_position': ['avg_position', 'position', 'average_position', 'avg_pos'],
    'ctr': ['ctr', 'click_through_rate', 'clickthrough_rate'],
    'clicks': ['clicks', 'click_count'],
    'impressions': ['impressions', 'impression_count'],
    'primary_keyword': ['primary_keyword', 'query', 'top_query', 'keyword'],
}

BACKLINK_COLUMN_MAP = {
    'url': ['url', 'target_url', 'page', 'target'],
    'referring_domains': ['referring_domains', 'ref_domains', 'domains', 'rd'],
    'backlinks': ['backlinks', 'backlink_count', 'external_links', 'links'],
    'anchor_texts': ['anchor_texts', 'anchors', 'anchor_text'],
}


# =============================================================================
# CLI ARGUMENT PARSING
# =============================================================================

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the WQA generator.

    Returns:
        argparse.Namespace: Parsed arguments with all configuration options.
    """
    parser = argparse.ArgumentParser(
        description='Website Quality Audit (WQA) Generator - Generate comprehensive SEO audit reports',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python wqa_generator.py --crawl crawl.csv --output report.xlsx
  python wqa_generator.py --crawl crawl.csv --ga ga.csv --gsc gsc.csv --backlinks backlinks.csv --output report.xlsx
  python wqa_generator.py --crawl crawl.csv --output report.xlsx --low-traffic-threshold 10 --thin-content-threshold 500
        '''
    )

    # Required arguments
    parser.add_argument(
        '--crawl',
        type=str,
        required=True,
        help='Path to crawl CSV file (required)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output Excel file (required)'
    )

    # Optional data source arguments
    parser.add_argument(
        '--ga',
        type=str,
        default=None,
        help='Path to GA4 CSV file (optional)'
    )
    parser.add_argument(
        '--gsc',
        type=str,
        default=None,
        help='Path to GSC CSV file (optional)'
    )
    parser.add_argument(
        '--backlinks',
        type=str,
        default=None,
        help='Path to backlinks CSV file (optional)'
    )

    # Threshold arguments
    parser.add_argument(
        '--low-traffic-threshold',
        type=int,
        default=5,
        help='Sessions threshold for low traffic classification (default: 5)'
    )
    parser.add_argument(
        '--thin-content-threshold',
        type=int,
        default=1000,
        help='Word count threshold for thin content (default: 1000)'
    )
    parser.add_argument(
        '--high-rank-max-position',
        type=float,
        default=20.0,
        help='Maximum position to consider for ranking optimization (default: 20)'
    )
    parser.add_argument(
        '--low-ctr-threshold',
        type=float,
        default=0.05,
        help='CTR threshold for metadata optimization (default: 0.05 = 5%%)'
    )

    return parser.parse_args()


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def find_column(df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
    """
    Find the first matching column name from a list of possibilities.

    Args:
        df: DataFrame to search in
        possible_names: List of possible column names to look for

    Returns:
        The first matching column name found, or None if no match
    """
    df_columns_lower = {col.lower().strip(): col for col in df.columns}
    for name in possible_names:
        if name.lower() in df_columns_lower:
            return df_columns_lower[name.lower()]
    return None


def map_columns(df: pd.DataFrame, column_map: Dict[str, List[str]], source_name: str) -> pd.DataFrame:
    """
    Map DataFrame columns to standardized names using a mapping dictionary.

    Args:
        df: DataFrame with original column names
        column_map: Dictionary mapping standard names to possible source names
        source_name: Name of the data source for logging

    Returns:
        DataFrame with standardized column names
    """
    result = pd.DataFrame()

    for standard_name, possible_names in column_map.items():
        found_col = find_column(df, possible_names)
        if found_col:
            result[standard_name] = df[found_col]
        else:
            logger.warning(f"[{source_name}] Column '{standard_name}' not found. Tried: {possible_names}")
            result[standard_name] = None

    return result


def load_crawl_data(path: str) -> pd.DataFrame:
    """
    Load and normalize crawl data from CSV.

    Args:
        path: Path to the crawl CSV file

    Returns:
        DataFrame with standardized crawl data columns

    Raises:
        FileNotFoundError: If the crawl file doesn't exist
    """
    logger.info(f"Loading crawl data from: {path}")

    if not Path(path).exists():
        raise FileNotFoundError(f"Crawl file not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    logger.info(f"Loaded {len(df)} rows from crawl data")

    # Map columns to standard names
    df_mapped = map_columns(df, CRAWL_COLUMN_MAP, 'Crawl')

    # Normalize indexable column to boolean
    if 'indexable' in df_mapped.columns and df_mapped['indexable'] is not None:
        df_mapped['indexable'] = df_mapped['indexable'].apply(normalize_indexable)

    # Normalize in_sitemap column to boolean
    if 'in_sitemap' in df_mapped.columns and df_mapped['in_sitemap'] is not None:
        df_mapped['in_sitemap'] = df_mapped['in_sitemap'].apply(normalize_boolean)

    # Ensure numeric columns
    numeric_cols = ['status_code', 'inlinks', 'outlinks', 'crawl_depth', 'word_count']
    for col in numeric_cols:
        if col in df_mapped.columns:
            df_mapped[col] = pd.to_numeric(df_mapped[col], errors='coerce').fillna(0).astype(int)

    return df_mapped


def load_ga_data(path: Optional[str]) -> Optional[pd.DataFrame]:
    """
    Load and normalize GA4 data from CSV.

    Args:
        path: Path to the GA4 CSV file, or None if not provided

    Returns:
        DataFrame with standardized GA data columns, or None if path is None
    """
    if path is None:
        logger.info("GA4 data not provided - skipping")
        return None

    if not Path(path).exists():
        logger.warning(f"GA4 file not found: {path} - skipping")
        return None

    logger.info(f"Loading GA4 data from: {path}")
    df = pd.read_csv(path, low_memory=False)
    logger.info(f"Loaded {len(df)} rows from GA4 data")

    df_mapped = map_columns(df, GA_COLUMN_MAP, 'GA4')

    # Ensure numeric columns
    for col in ['sessions', 'conversions']:
        if col in df_mapped.columns:
            df_mapped[col] = pd.to_numeric(df_mapped[col], errors='coerce').fillna(0)

    return df_mapped


def load_gsc_data(path: Optional[str]) -> Optional[pd.DataFrame]:
    """
    Load and normalize GSC data from CSV.

    Args:
        path: Path to the GSC CSV file, or None if not provided

    Returns:
        DataFrame with standardized GSC data columns, or None if path is None
    """
    if path is None:
        logger.info("GSC data not provided - skipping")
        return None

    if not Path(path).exists():
        logger.warning(f"GSC file not found: {path} - skipping")
        return None

    logger.info(f"Loading GSC data from: {path}")
    df = pd.read_csv(path, low_memory=False)
    logger.info(f"Loaded {len(df)} rows from GSC data")

    df_mapped = map_columns(df, GSC_COLUMN_MAP, 'GSC')

    # Ensure numeric columns
    for col in ['avg_position', 'ctr', 'clicks', 'impressions']:
        if col in df_mapped.columns:
            df_mapped[col] = pd.to_numeric(df_mapped[col], errors='coerce').fillna(0)

    return df_mapped


def load_backlink_data(path: Optional[str]) -> Optional[pd.DataFrame]:
    """
    Load and normalize backlink data from CSV.

    Args:
        path: Path to the backlinks CSV file, or None if not provided

    Returns:
        DataFrame with standardized backlink data columns, or None if path is None
    """
    if path is None:
        logger.info("Backlink data not provided - skipping")
        return None

    if not Path(path).exists():
        logger.warning(f"Backlink file not found: {path} - skipping")
        return None

    logger.info(f"Loading backlink data from: {path}")
    df = pd.read_csv(path, low_memory=False)
    logger.info(f"Loaded {len(df)} rows from backlink data")

    df_mapped = map_columns(df, BACKLINK_COLUMN_MAP, 'Backlinks')

    # Ensure numeric columns
    for col in ['referring_domains', 'backlinks']:
        if col in df_mapped.columns:
            df_mapped[col] = pd.to_numeric(df_mapped[col], errors='coerce').fillna(0)

    return df_mapped


# =============================================================================
# DATA NORMALIZATION HELPERS
# =============================================================================

def normalize_indexable(value) -> bool:
    """
    Normalize indexable field to boolean.

    Args:
        value: Raw value from CSV (could be bool, string, or other)

    Returns:
        Boolean indicating if the page is indexable
    """
    if pd.isna(value):
        return True  # Assume indexable if not specified

    if isinstance(value, bool):
        return value

    str_val = str(value).lower().strip()

    # Check for indexable indicators
    if str_val in ['true', '1', 'yes', 'index', 'indexable']:
        return True

    # Check for non-indexable indicators
    if str_val in ['false', '0', 'no', 'noindex', 'non-indexable', 'not indexable']:
        return False

    return True  # Default to indexable


def normalize_boolean(value) -> bool:
    """
    Normalize boolean-like field to boolean.

    Args:
        value: Raw value from CSV

    Returns:
        Boolean value
    """
    if pd.isna(value):
        return False

    if isinstance(value, bool):
        return value

    str_val = str(value).lower().strip()
    return str_val in ['true', '1', 'yes', 'y']


def normalize_url(url: str) -> str:
    """
    Normalize URL for consistent matching.

    Args:
        url: Raw URL string

    Returns:
        Normalized URL string
    """
    if pd.isna(url) or not url:
        return ''

    url = str(url).strip().lower()

    # Remove trailing slash for consistency (except for root)
    if url.endswith('/') and len(url) > 1:
        parsed = urlparse(url)
        if parsed.path != '/':
            url = url.rstrip('/')

    return url


# =============================================================================
# DATASET MERGING
# =============================================================================

def merge_datasets(
    crawl_df: pd.DataFrame,
    ga_df: Optional[pd.DataFrame],
    gsc_df: Optional[pd.DataFrame],
    backlink_df: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """
    Merge all datasets into a single URL-level DataFrame.

    Uses the crawl data as the base universe and left joins other data sources.

    Args:
        crawl_df: Crawl data (required, forms the base)
        ga_df: GA4 data (optional)
        gsc_df: GSC data (optional)
        backlink_df: Backlink data (optional)

    Returns:
        Merged DataFrame with all columns
    """
    logger.info("Merging datasets...")

    # Start with crawl data as base
    df = crawl_df.copy()

    # Normalize URLs in base dataset
    df['url'] = df['url'].apply(normalize_url)
    df = df[df['url'] != '']  # Remove empty URLs

    # Merge GA4 data
    if ga_df is not None and 'url' in ga_df.columns:
        ga_df = ga_df.copy()
        ga_df['url'] = ga_df['url'].apply(normalize_url)
        ga_df = ga_df[ga_df['url'] != '']

        # Aggregate GA data by URL (in case of duplicates)
        ga_agg = ga_df.groupby('url').agg({
            'sessions': 'sum',
            'conversions': 'sum'
        }).reset_index()

        df = df.merge(ga_agg, on='url', how='left', suffixes=('', '_ga'))
        logger.info(f"Merged GA4 data - {len(ga_agg)} unique URLs")

    # Merge GSC data
    if gsc_df is not None and 'url' in gsc_df.columns:
        gsc_df = gsc_df.copy()
        gsc_df['url'] = gsc_df['url'].apply(normalize_url)
        gsc_df = gsc_df[gsc_df['url'] != '']

        # Aggregate GSC data by URL
        gsc_agg_cols = {}
        if 'avg_position' in gsc_df.columns:
            gsc_agg_cols['avg_position'] = 'mean'
        if 'ctr' in gsc_df.columns:
            gsc_agg_cols['ctr'] = 'mean'
        if 'clicks' in gsc_df.columns:
            gsc_agg_cols['clicks'] = 'sum'
        if 'impressions' in gsc_df.columns:
            gsc_agg_cols['impressions'] = 'sum'
        if 'primary_keyword' in gsc_df.columns:
            gsc_agg_cols['primary_keyword'] = 'first'

        if gsc_agg_cols:
            gsc_agg = gsc_df.groupby('url').agg(gsc_agg_cols).reset_index()
            df = df.merge(gsc_agg, on='url', how='left', suffixes=('', '_gsc'))
            logger.info(f"Merged GSC data - {len(gsc_agg)} unique URLs")

    # Merge backlink data
    if backlink_df is not None and 'url' in backlink_df.columns:
        backlink_df = backlink_df.copy()
        backlink_df['url'] = backlink_df['url'].apply(normalize_url)
        backlink_df = backlink_df[backlink_df['url'] != '']

        # Aggregate backlink data by URL
        bl_agg_cols = {}
        if 'referring_domains' in backlink_df.columns:
            bl_agg_cols['referring_domains'] = 'sum'
        if 'backlinks' in backlink_df.columns:
            bl_agg_cols['backlinks'] = 'sum'

        if bl_agg_cols:
            bl_agg = backlink_df.groupby('url').agg(bl_agg_cols).reset_index()
            df = df.merge(bl_agg, on='url', how='left', suffixes=('', '_bl'))
            logger.info(f"Merged backlink data - {len(bl_agg)} unique URLs")

    # Ensure all expected columns exist with defaults
    expected_columns = {
        'url': '',
        'status_code': 0,
        'indexable': True,
        'canonical_url': '',
        'inlinks': 0,
        'outlinks': 0,
        'crawl_depth': 0,
        'in_sitemap': False,
        'page_title': '',
        'meta_description': '',
        'word_count': 0,
        'sessions': 0,
        'conversions': 0,
        'avg_position': 0.0,
        'ctr': 0.0,
        'clicks': 0,
        'impressions': 0,
        'referring_domains': 0,
        'backlinks': 0,
        'primary_keyword': '',
    }

    for col, default in expected_columns.items():
        if col not in df.columns:
            df[col] = default
        else:
            # Fill NaN values with defaults
            if isinstance(default, str):
                df[col] = df[col].fillna(default).astype(str)
            elif isinstance(default, bool):
                df[col] = df[col].fillna(default).astype(bool)
            elif isinstance(default, int):
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default).astype(int)
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default)

    logger.info(f"Final merged dataset: {len(df)} URLs with {len(df.columns)} columns")
    return df


# =============================================================================
# PAGE TYPE CLASSIFICATION
# =============================================================================

def classify_page_type(url: str) -> str:
    """
    Classify the page type based on URL patterns.

    Args:
        url: The URL to classify

    Returns:
        Page type string: 'Home', 'Blog', 'Local Lander', 'Service', or 'Other'
    """
    if not url or pd.isna(url):
        return 'Other'

    url_lower = str(url).lower()

    # Parse URL to get path
    try:
        parsed = urlparse(url_lower)
        path = parsed.path
    except Exception:
        path = url_lower

    # Home page detection
    if path in ['/', ''] or path.rstrip('/') == '':
        return 'Home'

    # Blog/News detection
    if any(pattern in path for pattern in ['/blog/', '/blog', '/news/', '/news', '/articles/', '/posts/']):
        return 'Blog'

    # Local lander detection
    if any(pattern in path for pattern in ['/location/', '/locations/', '/city/', '/cities/',
                                            '/service-area/', '/service-areas/', '/area/', '/areas/',
                                            '/near-me/', '/local/']):
        return 'Local Lander'

    # Service page detection
    if any(pattern in path for pattern in ['/service/', '/services/', '/solutions/', '/products/',
                                            '/what-we-do/', '/offerings/']):
        return 'Service'

    return 'Other'


# =============================================================================
# RULES ENGINE - ACTION ASSIGNMENT
# =============================================================================

def assign_actions(
    row: pd.Series,
    low_traffic_threshold: int = 5,
    thin_content_threshold: int = 1000,
    high_rank_max_position: float = 20.0,
    low_ctr_threshold: float = 0.05
) -> Tuple[List[str], List[str]]:
    """
    Apply rule-based SEO logic to assign technical and content actions.

    Args:
        row: A single row from the URL-level DataFrame
        low_traffic_threshold: Sessions threshold for low traffic
        thin_content_threshold: Word count threshold for thin content
        high_rank_max_position: Maximum position for ranking optimization
        low_ctr_threshold: CTR threshold for metadata optimization

    Returns:
        Tuple of (technical_actions, content_actions) lists
    """
    technical_actions: List[str] = []
    content_actions: List[str] = []

    # Extract row values with safe defaults
    status_code = int(row.get('status_code', 0))
    indexable = bool(row.get('indexable', True))
    canonical_url = str(row.get('canonical_url', '')).strip()
    url = str(row.get('url', '')).strip().lower()
    inlinks = int(row.get('inlinks', 0))
    outlinks = int(row.get('outlinks', 0))
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

    # =========================================================================
    # TECHNICAL ACTIONS
    # =========================================================================

    # 6.1 Status code & redirect logic
    if status_code == 302:
        technical_actions.append('301 Redirect')

    if status_code == 404:
        if backlinks > 0 or referring_domains > 0:
            technical_actions.append('301 Redirect')
        elif backlinks == 0 and inlinks > 0:
            technical_actions.append('Remove Internal Links')

    if status_code in [301, 308]:
        technical_actions.append('Review Redirect')

    # 6.2 Indexability, sitemap, and robots
    if not indexable and in_sitemap:
        technical_actions.append('Remove from Sitemap')

    if indexable and status_code == 200 and sessions > 0 and not in_sitemap:
        technical_actions.append('Add to Sitemap')

    # 6.3 Canonical tags
    if canonical_url and canonical_url.lower() != url:
        technical_actions.append('Canonicalize')

    # 6.4 Internal links & crawl depth
    if sessions > 0 and inlinks == 0:
        technical_actions.append('Add Internal Links')

    if crawl_depth >= 4 and page_type in ['Home', 'Service', 'Local Lander']:
        if 'Add Internal Links' not in technical_actions:
            technical_actions.append('Add Internal Links')

    # 6.5 Schema hints
    if page_type == 'Blog':
        technical_actions.append('Add Schema: Article')
    elif page_type == 'Local Lander':
        technical_actions.append('Add Schema: LocalBusiness')
    elif page_type == 'Home':
        technical_actions.append('Add Schema: Organization')

    # =========================================================================
    # CONTENT ACTIONS
    # =========================================================================

    # 7.1 Deletion / redirect candidates
    # Hard delete (404) candidate
    if (sessions <= low_traffic_threshold and
        conversions == 0 and
        backlinks == 0 and
        referring_domains == 0 and
        status_code == 200):  # Only for live pages
        content_actions.append('Delete (404)')

    # Redirect candidate (content-level consolidation)
    elif (sessions <= low_traffic_threshold and
          conversions == 0 and
          (backlinks > 0 or referring_domains > 0) and
          status_code == 200):
        content_actions.append('301 Redirect')

    # 7.2 Thin vs substantial content
    # Rewrite
    if (word_count < thin_content_threshold and
        (sessions > 0 or avg_position > 0) and
        status_code == 200 and
        'Delete (404)' not in content_actions and
        '301 Redirect' not in content_actions):
        content_actions.append('Rewrite')

    # Refresh
    if (word_count >= thin_content_threshold and
        sessions > 0 and
        2 < avg_position <= high_rank_max_position and
        status_code == 200 and
        'Delete (404)' not in content_actions and
        '301 Redirect' not in content_actions):
        content_actions.append('Refresh')

    # Target w/ Links
    if (word_count >= thin_content_threshold and
        sessions > 0 and
        3 <= avg_position <= 20 and
        referring_domains == 0 and
        status_code == 200 and
        'Delete (404)' not in content_actions and
        '301 Redirect' not in content_actions):
        content_actions.append('Target w/ Links')

    # 7.3 Metadata optimizations
    # Update Meta Description
    if status_code == 200:
        if (not meta_description or
            (avg_position <= high_rank_max_position and
             ctr < low_ctr_threshold and
             impressions > 100)):
            content_actions.append('Update Meta Description')

    # Update Page Title
    if status_code == 200:
        if (not page_title or
            (avg_position <= high_rank_max_position and
             ctr < low_ctr_threshold and
             impressions > 100)):
            content_actions.append('Update Page Title')

    # 7.4 Default / No-action
    if not content_actions:
        content_actions.append('Leave As Is')

    return technical_actions, content_actions


# =============================================================================
# PRIORITY ASSIGNMENT
# =============================================================================

def assign_priority(row: pd.Series) -> str:
    """
    Assign priority level to a URL based on status, page type, and actions.

    Args:
        row: A single row from the URL-level DataFrame

    Returns:
        Priority string: 'High', 'Medium', or 'Low'
    """
    status_code = int(row.get('status_code', 0))
    page_type = str(row.get('page_type', 'Other'))
    content_actions = str(row.get('content_actions', ''))

    # High priority conditions
    if status_code in [404, 302]:
        return 'High'

    if page_type in ['Home', 'Service', 'Local Lander']:
        if 'Rewrite' in content_actions or 'Refresh' in content_actions:
            return 'High'

    # Medium priority conditions
    if 'Delete (404)' in content_actions or '301 Redirect' in content_actions:
        if page_type not in ['Home', 'Service', 'Local Lander']:
            return 'Medium'

    return 'Low'


# =============================================================================
# SUMMARY GENERATION
# =============================================================================

def generate_summary(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Generate summary statistics from the processed DataFrame.

    Args:
        df: Processed URL-level DataFrame with actions and priorities

    Returns:
        Dictionary containing summary DataFrames
    """
    logger.info("Generating summary statistics...")

    summaries = {}

    # Count technical actions
    tech_actions_flat = []
    for actions in df['technical_actions']:
        if actions:
            for action in str(actions).split(', '):
                action = action.strip()
                if action:
                    tech_actions_flat.append(action)

    tech_counts = pd.Series(tech_actions_flat).value_counts().reset_index()
    tech_counts.columns = ['Technical Action', 'Count']
    summaries['technical_actions'] = tech_counts

    # Count content actions
    content_actions_flat = []
    for actions in df['content_actions']:
        if actions:
            for action in str(actions).split(', '):
                action = action.strip()
                if action:
                    content_actions_flat.append(action)

    content_counts = pd.Series(content_actions_flat).value_counts().reset_index()
    content_counts.columns = ['Content Action', 'Count']
    summaries['content_actions'] = content_counts

    # Count by priority
    priority_counts = df['priority'].value_counts().reset_index()
    priority_counts.columns = ['Priority', 'Count']
    # Ensure proper order
    priority_order = ['High', 'Medium', 'Low']
    priority_counts['sort_order'] = priority_counts['Priority'].apply(
        lambda x: priority_order.index(x) if x in priority_order else 99
    )
    priority_counts = priority_counts.sort_values('sort_order').drop('sort_order', axis=1)
    summaries['priority'] = priority_counts

    # Count by page type
    page_type_counts = df['page_type'].value_counts().reset_index()
    page_type_counts.columns = ['Page Type', 'Count']
    summaries['page_type'] = page_type_counts

    # Count by status code
    status_counts = df['status_code'].value_counts().reset_index()
    status_counts.columns = ['Status Code', 'Count']
    status_counts = status_counts.sort_values('Status Code')
    summaries['status_code'] = status_counts

    logger.info(f"Generated {len(summaries)} summary tables")
    return summaries


# =============================================================================
# EXCEL OUTPUT
# =============================================================================

def write_excel(df: pd.DataFrame, summaries: Dict[str, pd.DataFrame], output_path: str) -> None:
    """
    Write the processed data and summaries to an Excel file.

    Creates three sheets:
    - Aggregation: Full dataset with all columns
    - Actions: Filtered view with key columns, sorted by priority
    - Summary: Statistical summaries

    Args:
        df: Processed URL-level DataFrame
        summaries: Dictionary of summary DataFrames
        output_path: Path to output Excel file
    """
    logger.info(f"Writing Excel report to: {output_path}")

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: Aggregation (full data)
        df.to_excel(writer, sheet_name='Aggregation', index=False)
        logger.info(f"Wrote Aggregation sheet: {len(df)} rows")

        # Sheet 2: Actions (filtered and sorted)
        action_columns = [
            'url', 'page_type', 'status_code', 'sessions', 'avg_position',
            'referring_domains', 'backlinks', 'word_count', 'inlinks',
            'technical_actions', 'content_actions', 'priority'
        ]

        # Only include columns that exist
        action_columns = [col for col in action_columns if col in df.columns]

        df_actions = df[action_columns].copy()

        # Sort by priority (High > Medium > Low), then by sessions descending
        priority_map = {'High': 0, 'Medium': 1, 'Low': 2}
        df_actions['priority_sort'] = df_actions['priority'].map(priority_map)
        df_actions = df_actions.sort_values(
            ['priority_sort', 'sessions'],
            ascending=[True, False]
        ).drop('priority_sort', axis=1)

        df_actions.to_excel(writer, sheet_name='Actions', index=False)
        logger.info(f"Wrote Actions sheet: {len(df_actions)} rows")

        # Sheet 3: Summary
        current_row = 0

        # Write each summary table with headers
        summary_order = [
            ('Priority Distribution', 'priority'),
            ('Technical Actions', 'technical_actions'),
            ('Content Actions', 'content_actions'),
            ('Page Types', 'page_type'),
            ('Status Codes', 'status_code'),
        ]

        for title, key in summary_order:
            if key in summaries:
                # Write title
                title_df = pd.DataFrame([[title]], columns=[''])
                title_df.to_excel(
                    writer,
                    sheet_name='Summary',
                    index=False,
                    header=False,
                    startrow=current_row
                )
                current_row += 1

                # Write data
                summaries[key].to_excel(
                    writer,
                    sheet_name='Summary',
                    index=False,
                    startrow=current_row
                )
                current_row += len(summaries[key]) + 3  # Add spacing

        logger.info("Wrote Summary sheet")

    logger.info(f"Excel report saved successfully: {output_path}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main() -> int:
    """
    Main entry point for the WQA generator.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Parse arguments
        args = parse_args()

        logger.info("=" * 60)
        logger.info("Website Quality Audit (WQA) Generator")
        logger.info("=" * 60)
        logger.info(f"Crawl file: {args.crawl}")
        logger.info(f"GA4 file: {args.ga or 'Not provided'}")
        logger.info(f"GSC file: {args.gsc or 'Not provided'}")
        logger.info(f"Backlinks file: {args.backlinks or 'Not provided'}")
        logger.info(f"Output file: {args.output}")
        logger.info("-" * 60)
        logger.info("Thresholds:")
        logger.info(f"  Low traffic: {args.low_traffic_threshold} sessions")
        logger.info(f"  Thin content: {args.thin_content_threshold} words")
        logger.info(f"  High rank max position: {args.high_rank_max_position}")
        logger.info(f"  Low CTR: {args.low_ctr_threshold * 100}%")
        logger.info("-" * 60)

        # Load data
        crawl_df = load_crawl_data(args.crawl)
        ga_df = load_ga_data(args.ga)
        gsc_df = load_gsc_data(args.gsc)
        backlink_df = load_backlink_data(args.backlinks)

        # Merge datasets
        df = merge_datasets(crawl_df, ga_df, gsc_df, backlink_df)

        # Classify page types
        logger.info("Classifying page types...")
        df['page_type'] = df['url'].apply(classify_page_type)

        page_type_dist = df['page_type'].value_counts()
        for ptype, count in page_type_dist.items():
            logger.info(f"  {ptype}: {count} URLs")

        # Assign actions
        logger.info("Applying rules engine to assign actions...")
        actions_result = df.apply(
            lambda row: assign_actions(
                row,
                low_traffic_threshold=args.low_traffic_threshold,
                thin_content_threshold=args.thin_content_threshold,
                high_rank_max_position=args.high_rank_max_position,
                low_ctr_threshold=args.low_ctr_threshold
            ),
            axis=1
        )

        # Unpack results
        df['technical_actions'] = actions_result.apply(lambda x: ', '.join(x[0]) if x[0] else '')
        df['content_actions'] = actions_result.apply(lambda x: ', '.join(x[1]) if x[1] else '')

        # Assign priority
        logger.info("Assigning priorities...")
        df['priority'] = df.apply(assign_priority, axis=1)

        priority_dist = df['priority'].value_counts()
        for priority, count in priority_dist.items():
            logger.info(f"  {priority}: {count} URLs")

        # Generate summary
        summaries = generate_summary(df)

        # Write output
        write_excel(df, summaries, args.output)

        logger.info("=" * 60)
        logger.info("WQA GENERATION COMPLETE")
        logger.info(f"Total URLs processed: {len(df)}")
        logger.info(f"Report saved to: {args.output}")
        logger.info("=" * 60)

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error generating WQA report: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
