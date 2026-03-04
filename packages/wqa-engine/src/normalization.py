"""
Normalization and helper functions for data processing.

Provides column finding/mapping utilities and URL/value normalization functions
used across the WQA engine pipeline.
"""

import pandas as pd
from typing import List, Dict, Optional
from urllib.parse import urlparse


def find_column(df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
    df_columns_lower = {col.lower().strip(): col for col in df.columns}
    for name in possible_names:
        if name.lower() in df_columns_lower:
            return df_columns_lower[name.lower()]
    return None


def map_columns(df: pd.DataFrame, column_map: Dict[str, List[str]], source_name: str) -> pd.DataFrame:
    result = pd.DataFrame()
    for standard_name, possible_names in column_map.items():
        found_col = find_column(df, possible_names)
        if found_col:
            result[standard_name] = df[found_col]
        else:
            result[standard_name] = None
    return result


def normalize_indexable(value) -> bool:
    if pd.isna(value):
        return True
    if isinstance(value, bool):
        return value
    str_val = str(value).lower().strip()
    if str_val in ['true', '1', 'yes', 'index', 'indexable']:
        return True
    if str_val in ['false', '0', 'no', 'noindex', 'non-indexable', 'not indexable']:
        return False
    return True


def normalize_boolean(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    str_val = str(value).lower().strip()
    return str_val in ['true', '1', 'yes', 'y']


def normalize_url(url: str) -> str:
    if pd.isna(url) or not url:
        return ''
    url = str(url).strip().lower()
    if url.endswith('/') and len(url) > 1:
        parsed = urlparse(url)
        if parsed.path != '/':
            url = url.rstrip('/')
    return url


def url_to_page_path(url: str) -> str:
    """Extract normalized page path from a full URL for GA4 join.

    Examples:
        https://example.com/states/tennessee/ -> /states/tennessee
        https://example.com/ -> /
        /states/tennessee/ -> /states/tennessee
    """
    if pd.isna(url) or not url:
        return '/'
    url = str(url).strip()

    try:
        # If it's already just a path (starts with /), use it directly
        if url.startswith('/'):
            path = url
        else:
            parsed = urlparse(url)
            path = parsed.path or '/'

        # Normalize: lowercase, strip trailing slash (except for root)
        path = path.lower()
        if path != '/' and path.endswith('/'):
            path = path.rstrip('/')

        # Ensure it starts with /
        if not path.startswith('/'):
            path = '/' + path

        return path if path else '/'
    except Exception:
        return '/'


def normalize_page_path(path: str) -> str:
    """Normalize a GA4 pagePath value for joining.

    GA4 returns pagePath like '/states/tennessee/' - we normalize it to match url_to_page_path output.
    """
    if pd.isna(path) or not path:
        return '/'
    path = str(path).strip().lower()

    # Ensure starts with /
    if not path.startswith('/'):
        path = '/' + path

    # Strip trailing slash (except for root)
    if path != '/' and path.endswith('/'):
        path = path.rstrip('/')

    return path if path else '/'
