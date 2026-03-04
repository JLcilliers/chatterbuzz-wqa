"""
Backlink Data Pipeline Worker
==============================
Worker for backlink data ingestion: accepts file bytes (CSV/Excel upload)
or a pre-parsed API response dict, normalizes with BACKLINK_COLUMN_MAP,
and writes to the backlink_data table in Supabase.

Ported from api/index.py load_backlink_data (lines 662-715).
"""

import io
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union
from urllib.parse import urlparse

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column mapping (mirrored from monolith)
# ---------------------------------------------------------------------------

BACKLINK_COLUMN_MAP: Dict[str, List[str]] = {
    # Target URL (the page receiving the backlink)
    'url': ['url', 'target url', 'target_url', 'page', 'target', 'page url', 'target page'],
    # Source URL (the page providing the backlink) - used for per-backlink formats
    'source_url': [
        'source url', 'source_url', 'source', 'referring url', 'referring_url',
        'from url', 'from_url', 'linking page',
    ],
    # Pre-aggregated referring domains count
    'referring_domains': [
        'referring domains', 'referring_domains', 'ref domains', 'ref_domains',
        'domains', 'rd', 'dofollow referring domains', 'root domains',
    ],
    # Pre-aggregated backlinks count
    'backlinks': [
        'backlinks', 'backlink_count', 'external_links', 'links', 'total backlinks',
        'dofollow backlinks', 'external backlinks', 'inbound links',
    ],
    # Authority score
    'authority': [
        'ur', 'url rating', 'url_rating', 'authority score', 'authority_score',
        'page authority', 'page_authority', 'pa', 'trust flow', 'citation flow',
        'domain rating', 'dr', 'as', 'page ascore', 'ascore',
    ],
    # Anchor text
    'anchor_texts': ['anchor_texts', 'anchors', 'anchor_text', 'anchor', 'top anchor'],
    # Nofollow indicator (for per-backlink formats)
    'nofollow': ['nofollow', 'no_follow', 'is_nofollow', 'rel_nofollow'],
}


# ---------------------------------------------------------------------------
# Helpers (mirrored from monolith)
# ---------------------------------------------------------------------------

def _find_column(df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
    df_columns_lower = {col.lower().strip(): col for col in df.columns}
    for name in possible_names:
        if name.lower() in df_columns_lower:
            return df_columns_lower[name.lower()]
    return None


def _map_columns(df: pd.DataFrame, column_map: Dict[str, List[str]], source_name: str) -> pd.DataFrame:
    result = pd.DataFrame()
    for standard_name, possible_names in column_map.items():
        found_col = _find_column(df, possible_names)
        if found_col:
            result[standard_name] = df[found_col]
        else:
            result[standard_name] = None
    return result


def _normalize_url(url: str) -> str:
    if pd.isna(url) or not url:
        return ''
    return str(url).strip().lower()


def _extract_domain(url: str) -> str:
    """Extract domain from URL for referring domain counting."""
    if pd.isna(url) or not url:
        return ''
    try:
        parsed = urlparse(str(url))
        domain = parsed.netloc or parsed.path.split('/')[0]
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain.lower()
    except Exception:
        return ''


def _read_file_to_dataframe(file_content: bytes, filename: str = "") -> pd.DataFrame:
    """Read file content into DataFrame, auto-detecting CSV or Excel format."""
    filename_lower = filename.lower() if filename else ""
    if filename_lower.endswith('.xlsx') or filename_lower.endswith('.xls'):
        return pd.read_excel(io.BytesIO(file_content))
    else:
        encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        for encoding in encodings_to_try:
            try:
                return pd.read_csv(io.BytesIO(file_content), low_memory=False, encoding=encoding)
            except UnicodeDecodeError:
                continue
            except Exception:
                break
        try:
            return pd.read_excel(io.BytesIO(file_content))
        except Exception:
            raise ValueError(f"Could not read file as CSV or Excel: {filename}")


# ---------------------------------------------------------------------------
# Core backlink loading (ported identically from monolith)
# ---------------------------------------------------------------------------

def load_backlink_data(file_content: bytes, filename: str = "") -> Optional[pd.DataFrame]:
    """Load and process backlink data, supporting both aggregated and per-backlink formats.

    Detects format automatically:
    - If 'source_url' column exists: per-backlink format (SEMRush style) -> aggregates by target URL
    - Otherwise: pre-aggregated format (Ahrefs style) -> uses data as-is
    """
    df = _read_file_to_dataframe(file_content, filename)
    df_mapped = _map_columns(df, BACKLINK_COLUMN_MAP, 'Backlinks')

    # Check if this is a per-backlink format (has source_url column)
    if 'source_url' in df_mapped.columns and 'url' in df_mapped.columns:
        logger.info("Detected per-backlink format (SEMRush style) - aggregating by target URL")

        # Normalize target URLs
        df_mapped['url'] = df_mapped['url'].apply(_normalize_url)
        df_mapped = df_mapped[df_mapped['url'] != '']

        # Extract source domains for referring domain counting
        df_mapped['source_domain'] = df_mapped['source_url'].apply(_extract_domain)

        # Determine if link is dofollow (not nofollow)
        if 'nofollow' in df_mapped.columns:
            df_mapped['is_dofollow'] = ~df_mapped['nofollow'].fillna(False).astype(bool)
        else:
            df_mapped['is_dofollow'] = True

        # Aggregate by target URL
        agg_spec = {
            'backlinks': ('url', 'count'),
            'referring_domains': ('source_domain', 'nunique'),
            'dofollow_links': ('is_dofollow', 'sum'),
        }
        if 'authority' in df_mapped.columns:
            agg_spec['authority'] = ('authority', 'max')

        aggregated = df_mapped.groupby('url').agg(**agg_spec).reset_index()

        if 'authority' not in aggregated.columns:
            aggregated['authority'] = 0

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


def load_backlink_data_from_api(api_response: List[dict]) -> pd.DataFrame:
    """Normalize an API response (list of dicts) into the standard backlink DataFrame.

    This path is for providers that return pre-structured JSON (e.g. future Ahrefs
    or Moz API integrations).
    """
    if not api_response:
        return pd.DataFrame(columns=['url', 'referring_domains', 'backlinks', 'authority'])

    df = pd.DataFrame(api_response)
    df_mapped = _map_columns(df, BACKLINK_COLUMN_MAP, 'Backlinks API')

    for col in ['referring_domains', 'backlinks', 'authority']:
        if col in df_mapped.columns:
            df_mapped[col] = pd.to_numeric(df_mapped[col], errors='coerce').fillna(0)

    return df_mapped


# ---------------------------------------------------------------------------
# Public worker entry point
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    reraise=True,
)
def _supabase_batch_insert(supabase_client, table: str, records: list, batch_size: int = 500):
    """Insert records into Supabase in batches with retry."""
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        supabase_client.table(table).insert(batch).execute()


def run_backlink_worker(
    supabase_client,
    client_id: str,
    file_content: Optional[bytes] = None,
    filename: str = "",
    api_response: Optional[List[dict]] = None,
) -> dict:
    """Pipeline worker entry point: ingest backlink data, write to Supabase backlink_data table.

    Accepts EITHER file_content (CSV/Excel bytes) OR api_response (list of dicts).
    Exactly one must be provided.

    Args:
        supabase_client: Initialised Supabase client.
        client_id: Unique identifier for the client / project.
        file_content: Raw bytes of backlink export file (CSV or Excel). Mutually exclusive with api_response.
        filename: Original filename (used for format detection when file_content is provided).
        api_response: Pre-parsed list of dicts from a backlink API. Mutually exclusive with file_content.

    Returns:
        dict with keys: status, rows_written, error (if any).
    """
    logger.info(f"[Backlink Worker] Starting for client={client_id}")

    if file_content is None and api_response is None:
        return {'status': 'error', 'rows_written': 0, 'error': 'No input provided: supply file_content or api_response'}

    try:
        if file_content is not None:
            df = load_backlink_data(file_content, filename)
        else:
            df = load_backlink_data_from_api(api_response)

        if df is None or df.empty:
            logger.warning(f"[Backlink Worker] No data parsed for client={client_id}")
            return {'status': 'empty', 'rows_written': 0, 'error': None}

        now_iso = datetime.utcnow().isoformat()
        records = []
        for _, row in df.iterrows():
            record = {
                'client_id': client_id,
                'fetched_at': now_iso,
            }
            for col in df.columns:
                val = row[col]
                if pd.isna(val):
                    record[col] = None
                elif isinstance(val, (bool,)):
                    record[col] = val
                elif isinstance(val, (int,)):
                    record[col] = int(val)
                elif isinstance(val, (float,)):
                    record[col] = float(val)
                else:
                    record[col] = str(val)
            records.append(record)

        # Delete previous backlink data for this client, then insert
        supabase_client.table('backlink_data').delete().eq('client_id', client_id).execute()
        _supabase_batch_insert(supabase_client, 'backlink_data', records)

        logger.info(f"[Backlink Worker] Wrote {len(records)} rows for client={client_id}")
        return {'status': 'success', 'rows_written': len(records), 'error': None}

    except Exception as e:
        logger.error(f"[Backlink Worker] Failed for client={client_id}: {e}")
        return {'status': 'error', 'rows_written': 0, 'error': str(e)}
