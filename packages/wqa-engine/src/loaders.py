"""
Data loading functions for the WQA Engine.

Handles reading and parsing of crawl, GA4, GSC, keyword, and backlink
data files into normalised pandas DataFrames.
"""

import io
import logging
from typing import Dict, List, Optional
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from .column_maps import (
    CRAWL_COLUMN_MAP,
    GA_COLUMN_MAP,
    GSC_COLUMN_MAP,
    KEYWORD_COLUMN_MAP,
    BACKLINK_COLUMN_MAP,
)
from .normalization import (
    map_columns,
    normalize_indexable,
    normalize_boolean,
    normalize_url,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATA LOADING
# =============================================================================

def read_file_to_dataframe(file_content: bytes, filename: str = "") -> pd.DataFrame:
    """Read file content into DataFrame, auto-detecting CSV or Excel format"""
    filename_lower = filename.lower() if filename else ""

    if filename_lower.endswith('.xlsx') or filename_lower.endswith('.xls'):
        return pd.read_excel(io.BytesIO(file_content))
    else:
        # Try CSV with different encodings, fall back to Excel if all fail
        encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        for encoding in encodings_to_try:
            try:
                return pd.read_csv(io.BytesIO(file_content), low_memory=False, encoding=encoding)
            except UnicodeDecodeError:
                continue
            except Exception as e:
                # If it's not an encoding error, try Excel
                break
        # Try Excel as last resort
        try:
            return pd.read_excel(io.BytesIO(file_content))
        except Exception:
            raise ValueError(f"Could not read file as CSV or Excel: {filename}")


def load_crawl_data(file_content: bytes, filename: str = "") -> pd.DataFrame:
    df = read_file_to_dataframe(file_content, filename)
    df_mapped = map_columns(df, CRAWL_COLUMN_MAP, 'Crawl')

    if 'indexable' in df_mapped.columns and df_mapped['indexable'] is not None:
        df_mapped['indexable'] = df_mapped['indexable'].apply(normalize_indexable)
    if 'in_sitemap' in df_mapped.columns and df_mapped['in_sitemap'] is not None:
        df_mapped['in_sitemap'] = df_mapped['in_sitemap'].apply(normalize_boolean)

    numeric_cols = ['status_code', 'inlinks', 'outlinks', 'crawl_depth', 'word_count']
    for col in numeric_cols:
        if col in df_mapped.columns:
            df_mapped[col] = pd.to_numeric(df_mapped[col], errors='coerce').fillna(0).astype(int)

    return df_mapped


def load_ga_data(file_content: Optional[bytes], filename: str = "") -> Optional[pd.DataFrame]:
    if file_content is None:
        return None
    df = read_file_to_dataframe(file_content, filename)
    df_mapped = map_columns(df, GA_COLUMN_MAP, 'GA4')
    # Convert numeric columns
    numeric_cols = ['sessions', 'sessions_prev', 'conversions', 'bounce_rate', 'avg_session_duration', 'ecom_revenue']
    for col in numeric_cols:
        if col in df_mapped.columns:
            df_mapped[col] = pd.to_numeric(df_mapped[col], errors='coerce').fillna(0)
    return df_mapped


def load_gsc_data(file_content: Optional[bytes], filename: str = "") -> Optional[pd.DataFrame]:
    if file_content is None:
        return None
    df = read_file_to_dataframe(file_content, filename)
    df_mapped = map_columns(df, GSC_COLUMN_MAP, 'GSC')
    for col in ['avg_position', 'ctr', 'clicks', 'impressions']:
        if col in df_mapped.columns:
            df_mapped[col] = pd.to_numeric(df_mapped[col], errors='coerce').fillna(0)
    return df_mapped


def load_keyword_data(file_content: Optional[bytes], filename: str = "") -> Optional[pd.DataFrame]:
    """Load keyword tracking data and aggregate by URL to get best keyword per page.

    Returns DataFrame with columns: url, main_kw, main_kw_volume, main_kw_ranking, best_kw, best_kw_ranking
    - main_kw: Keyword with highest search volume for the URL
    - best_kw: Keyword with best (lowest) ranking position for the URL
    """
    if file_content is None:
        return None

    df = read_file_to_dataframe(file_content, filename)
    df_mapped = map_columns(df, KEYWORD_COLUMN_MAP, 'Keywords')

    # Check required columns
    if 'url' not in df_mapped.columns or 'keyword' not in df_mapped.columns:
        logger.warning("Keyword data missing required columns (url, keyword)")
        return None

    # Normalize URLs
    df_mapped['url'] = df_mapped['url'].apply(normalize_url)
    df_mapped = df_mapped[df_mapped['url'] != '']

    # Convert numeric columns
    for col in ['volume', 'position', 'difficulty', 'cpc']:
        if col in df_mapped.columns:
            df_mapped[col] = pd.to_numeric(df_mapped[col], errors='coerce').fillna(0)

    # Ensure position is valid (greater than 0)
    if 'position' in df_mapped.columns:
        df_mapped.loc[df_mapped['position'] <= 0, 'position'] = 999

    result_rows = []

    for url, group in df_mapped.groupby('url'):
        row_data = {'url': url}

        # Main KW: keyword with highest volume
        if 'volume' in group.columns:
            main_kw_row = group.loc[group['volume'].idxmax()]
            row_data['main_kw'] = main_kw_row.get('keyword', '')
            row_data['main_kw_volume'] = main_kw_row.get('volume', 0)
            row_data['main_kw_ranking'] = main_kw_row.get('position', 0) if main_kw_row.get('position', 0) < 999 else 0
        else:
            # If no volume, use first keyword
            first_row = group.iloc[0]
            row_data['main_kw'] = first_row.get('keyword', '')
            row_data['main_kw_volume'] = 0
            row_data['main_kw_ranking'] = first_row.get('position', 0) if first_row.get('position', 0) < 999 else 0

        # Best KW: keyword with best (lowest) ranking position
        if 'position' in group.columns:
            valid_positions = group[group['position'] < 999]
            if not valid_positions.empty:
                best_kw_row = valid_positions.loc[valid_positions['position'].idxmin()]
                row_data['best_kw'] = best_kw_row.get('keyword', '')
                row_data['best_kw_ranking'] = best_kw_row.get('position', 0)
            else:
                row_data['best_kw'] = ''
                row_data['best_kw_ranking'] = 0
        else:
            row_data['best_kw'] = ''
            row_data['best_kw_ranking'] = 0

        result_rows.append(row_data)

    result_df = pd.DataFrame(result_rows)
    logger.info(f"Loaded keyword data: {len(df_mapped)} keywords across {len(result_df)} URLs")
    return result_df


def extract_domain(url: str) -> str:
    """Extract domain from URL for referring domain counting"""
    if pd.isna(url) or not url:
        return ''
    try:
        parsed = urlparse(str(url))
        domain = parsed.netloc or parsed.path.split('/')[0]
        # Remove www. prefix for consistent domain counting
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain.lower()
    except:
        return ''


def load_backlink_data(file_content: Optional[bytes], filename: str = "") -> Optional[pd.DataFrame]:
    """Load and process backlink data, supporting both aggregated and per-backlink formats.

    Detects format automatically:
    - If 'source_url' column exists: per-backlink format (SEMRush style) - aggregates by target URL
    - Otherwise: pre-aggregated format (Ahrefs style) - uses data as-is
    """
    if file_content is None:
        return None

    df = read_file_to_dataframe(file_content, filename)
    df_mapped = map_columns(df, BACKLINK_COLUMN_MAP, 'Backlinks')

    # Check if this is a per-backlink format (has source_url column)
    # This indicates SEMRush-style export where each row is one backlink
    if 'source_url' in df_mapped.columns and 'url' in df_mapped.columns:
        logger.info("Detected per-backlink format (SEMRush style) - aggregating by target URL")

        # Normalize target URLs
        df_mapped['url'] = df_mapped['url'].apply(normalize_url)
        df_mapped = df_mapped[df_mapped['url'] != '']

        # Extract source domains for referring domain counting
        df_mapped['source_domain'] = df_mapped['source_url'].apply(extract_domain)

        # Determine if link is dofollow (not nofollow)
        if 'nofollow' in df_mapped.columns:
            df_mapped['is_dofollow'] = ~df_mapped['nofollow'].fillna(False).astype(bool)
        else:
            df_mapped['is_dofollow'] = True

        # Aggregate by target URL
        aggregated = df_mapped.groupby('url').agg(
            backlinks=('url', 'count'),  # Count total backlinks
            referring_domains=('source_domain', 'nunique'),  # Count unique referring domains
            dofollow_links=('is_dofollow', 'sum'),  # Count dofollow links
            authority=('authority', 'max') if 'authority' in df_mapped.columns else ('url', lambda x: 0),  # Max authority score
        ).reset_index()

        # Ensure numeric types
        for col in ['backlinks', 'referring_domains', 'dofollow_links', 'authority']:
            if col in aggregated.columns:
                aggregated[col] = pd.to_numeric(aggregated[col], errors='coerce').fillna(0).astype(int)

        logger.info(f"Aggregated {len(df_mapped)} backlinks into {len(aggregated)} URLs")
        return aggregated

    else:
        # Pre-aggregated format - use as-is
        logger.info("Detected pre-aggregated backlink format")
        for col in ['referring_domains', 'backlinks', 'authority']:
            if col in df_mapped.columns:
                df_mapped[col] = pd.to_numeric(df_mapped[col], errors='coerce').fillna(0)
        return df_mapped
