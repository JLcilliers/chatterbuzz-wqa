"""
Dataset merging functions for the WQA Engine.

Merges crawl, GA4, GSC, and backlink DataFrames into a single
unified DataFrame with all expected columns filled and typed.
"""

import logging
from typing import Optional

import pandas as pd

from .normalization import (
    normalize_url,
    normalize_page_path,
    url_to_page_path,
)

logger = logging.getLogger(__name__)


# =============================================================================
# MERGING & PROCESSING
# =============================================================================

def merge_datasets(crawl_df, ga_df, gsc_df, backlink_df) -> pd.DataFrame:
    df = crawl_df.copy()
    df['url'] = df['url'].apply(normalize_url)
    df = df[df['url'] != '']

    # Create page_path column for GA4 joining
    df['page_path'] = df['url'].apply(url_to_page_path)

    # GA4 data joins on page_path (not full URL)
    if ga_df is not None and 'page_path' in ga_df.columns:
        ga_df = ga_df.copy()
        # Normalize page_path in GA data (should already be normalized from fetch)
        ga_df['page_path'] = ga_df['page_path'].apply(normalize_page_path)

        # Build aggregation dict dynamically based on available columns
        ga_agg_cols = {}
        if 'sessions' in ga_df.columns:
            ga_agg_cols['sessions'] = 'sum'
        if 'conversions' in ga_df.columns:
            ga_agg_cols['conversions'] = 'sum'
        if 'bounce_rate' in ga_df.columns:
            ga_agg_cols['bounce_rate'] = 'mean'
        if 'avg_session_duration' in ga_df.columns:
            ga_agg_cols['avg_session_duration'] = 'mean'
        if 'ecom_revenue' in ga_df.columns:
            ga_agg_cols['ecom_revenue'] = 'sum'
        if 'sessions_prev' in ga_df.columns:
            ga_agg_cols['sessions_prev'] = 'sum'
        if ga_agg_cols:
            ga_agg = ga_df.groupby('page_path').agg(ga_agg_cols).reset_index()
            df = df.merge(ga_agg, on='page_path', how='left', suffixes=('', '_ga'))
            logger.info(f"Merged GA4 data: {len(ga_agg)} paths, {df['sessions'].notna().sum()} matches")

    # Also support old-style GA data with 'url' column (from CSV uploads)
    elif ga_df is not None and 'url' in ga_df.columns:
        ga_df = ga_df.copy()
        ga_df['url'] = ga_df['url'].apply(normalize_url)
        ga_df = ga_df[ga_df['url'] != '']
        # Build aggregation dict dynamically based on available columns
        ga_agg_cols = {}
        if 'sessions' in ga_df.columns:
            ga_agg_cols['sessions'] = 'sum'
        if 'conversions' in ga_df.columns:
            ga_agg_cols['conversions'] = 'sum'
        if 'bounce_rate' in ga_df.columns:
            ga_agg_cols['bounce_rate'] = 'mean'
        if 'avg_session_duration' in ga_df.columns:
            ga_agg_cols['avg_session_duration'] = 'mean'
        if 'ecom_revenue' in ga_df.columns:
            ga_agg_cols['ecom_revenue'] = 'sum'
        if 'sessions_prev' in ga_df.columns:
            ga_agg_cols['sessions_prev'] = 'sum'
        if ga_agg_cols:
            ga_agg = ga_df.groupby('url').agg(ga_agg_cols).reset_index()
            df = df.merge(ga_agg, on='url', how='left', suffixes=('', '_ga'))

    if gsc_df is not None and 'url' in gsc_df.columns:
        gsc_df = gsc_df.copy()
        gsc_df['url'] = gsc_df['url'].apply(normalize_url)
        gsc_df = gsc_df[gsc_df['url'] != '']
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

    if backlink_df is not None and 'url' in backlink_df.columns:
        backlink_df = backlink_df.copy()
        backlink_df['url'] = backlink_df['url'].apply(normalize_url)
        backlink_df = backlink_df[backlink_df['url'] != '']
        bl_agg_cols = {}
        if 'referring_domains' in backlink_df.columns:
            bl_agg_cols['referring_domains'] = 'sum'
        if 'backlinks' in backlink_df.columns:
            bl_agg_cols['backlinks'] = 'sum'
        if 'dofollow_links' in backlink_df.columns:
            bl_agg_cols['dofollow_links'] = 'sum'
        if 'authority' in backlink_df.columns:
            bl_agg_cols['authority'] = 'max'  # Take highest authority score if multiple entries
        if bl_agg_cols:
            bl_agg = backlink_df.groupby('url').agg(bl_agg_cols).reset_index()
            df = df.merge(bl_agg, on='url', how='left', suffixes=('', '_bl'))

    expected_columns = {
        # Crawl data columns
        'url': '', 'status_code': 0, 'indexable': True, 'canonical_url': '',
        'meta_robots': '', 'content_type': '', 'h1': '',
        'inlinks': 0, 'outlinks': 0, 'crawl_depth': 0, 'in_sitemap': False,
        'page_title': '', 'meta_description': '', 'word_count': 0,
        'last_modified': '',
        # GA data columns
        'sessions': 0, 'conversions': 0, 'bounce_rate': 0.0, 'avg_session_duration': 0.0,
        'ecom_revenue': 0.0, 'sessions_prev': 0,
        # GSC data columns
        'avg_position': 0.0, 'ctr': 0.0, 'clicks': 0, 'impressions': 0, 'primary_keyword': '',
        # Backlink data columns
        'referring_domains': 0, 'backlinks': 0, 'dofollow_links': 0, 'authority': 0,
    }

    for col, default in expected_columns.items():
        if col not in df.columns:
            df[col] = default
        else:
            if isinstance(default, str):
                df[col] = df[col].fillna(default).astype(str)
            elif isinstance(default, bool):
                df[col] = df[col].fillna(default).astype(bool)
            elif isinstance(default, int):
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default).astype(int)
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default)

    return df
