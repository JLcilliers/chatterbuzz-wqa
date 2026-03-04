"""
Crawl Data Pipeline Worker
===========================
Worker for CSV/Excel upload mode: accepts file bytes, parses with column
mapping logic (mirrored from monolith), and writes to crawl_data + crawl_runs
tables in Supabase.

Ported from api/index.py load_crawl_data / read_file_to_dataframe helpers.
"""

import io
import logging
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column mapping (mirrored from monolith for self-containment)
# ---------------------------------------------------------------------------

CRAWL_COLUMN_MAP: Dict[str, List[str]] = {
    'url': ['address', 'url', 'page_url', 'page', 'page url', 'full url'],
    'status_code': ['status code', 'status_code', 'status', 'http_status', 'response_code', 'http status code'],
    'indexable': ['indexability', 'indexable', 'is_indexable', 'index_status', 'index status'],
    'meta_robots': ['meta robots 1', 'meta_robots', 'meta robots', 'robots', 'robots directive'],
    'canonical_url': [
        'canonical link element 1', 'canonical_url', 'canonical', 'canonical_link',
        'canonical_tag', 'canonical url', 'canonicals',
    ],
    'content_type': ['content type', 'content_type', 'mime type', 'type'],
    'inlinks': [
        'unique inlinks', 'inlinks', 'internal_inlinks', 'inlinks_count', 'internal inlinks',
        'internal links in', 'links in',
    ],
    'outlinks': [
        'unique outlinks', 'outlinks', 'internal_outlinks', 'outlinks_count', 'internal outlinks',
        'internal links out', 'links out',
    ],
    'crawl_depth': [
        'crawl depth', 'crawl_depth', 'depth', 'click_depth', 'level', 'click depth',
        'page depth', 'distance',
    ],
    'in_sitemap': [
        'indexability status', 'in xml sitemap', 'in_sitemap', 'sitemap', 'in_xml_sitemap',
        'sitemap_status', 'xml sitemap',
    ],
    'page_title': ['title 1', 'page_title', 'title', 'meta_title', 'page title', 'seo title'],
    'meta_description': ['meta description 1', 'meta_description', 'description', 'meta description'],
    'h1': ['h1-1', 'h1', 'h1 1', 'heading 1', 'first h1'],
    'word_count': [
        'word count', 'word_count', 'words', 'content_word_count', 'text_word_count',
        'text content', 'body word count',
    ],
    'last_modified': [
        'last modified', 'last_modified', 'lastmod', 'modified', 'date modified',
        'last-modified', 'modified date',
    ],
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


def _normalize_indexable(value) -> bool:
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


def _normalize_boolean(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    str_val = str(value).lower().strip()
    return str_val in ['true', '1', 'yes', 'y']


def _read_file_to_dataframe(file_content: bytes, filename: str = "") -> pd.DataFrame:
    """Read file content into DataFrame, auto-detecting CSV or Excel format."""
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
            except Exception:
                break
        # Try Excel as last resort
        try:
            return pd.read_excel(io.BytesIO(file_content))
        except Exception:
            raise ValueError(f"Could not read file as CSV or Excel: {filename}")


# ---------------------------------------------------------------------------
# Core crawl-data loading (ported identically from monolith)
# ---------------------------------------------------------------------------

def load_crawl_data(file_content: bytes, filename: str = "") -> pd.DataFrame:
    """Parse crawl export file and return normalised DataFrame."""
    df = _read_file_to_dataframe(file_content, filename)
    df_mapped = _map_columns(df, CRAWL_COLUMN_MAP, 'Crawl')

    if 'indexable' in df_mapped.columns and df_mapped['indexable'] is not None:
        df_mapped['indexable'] = df_mapped['indexable'].apply(_normalize_indexable)
    if 'in_sitemap' in df_mapped.columns and df_mapped['in_sitemap'] is not None:
        df_mapped['in_sitemap'] = df_mapped['in_sitemap'].apply(_normalize_boolean)

    numeric_cols = ['status_code', 'inlinks', 'outlinks', 'crawl_depth', 'word_count']
    for col in numeric_cols:
        if col in df_mapped.columns:
            df_mapped[col] = pd.to_numeric(df_mapped[col], errors='coerce').fillna(0).astype(int)

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


def run_crawl_worker(
    supabase_client,
    client_id: str,
    file_content: bytes,
    filename: str = "",
    crawl_source: str = "csv_upload",
) -> dict:
    """Pipeline worker entry point: parse crawl file, write to crawl_data + crawl_runs tables.

    Args:
        supabase_client: Initialised Supabase client.
        client_id: Unique identifier for the client / project.
        file_content: Raw bytes of the crawl export file (CSV or Excel).
        filename: Original filename (used for format detection).
        crawl_source: Label for the data source (default 'csv_upload').

    Returns:
        dict with keys: status, rows_written, crawl_run_id, error (if any).
    """
    logger.info(f"[Crawl Worker] Starting for client={client_id}, file={filename}, source={crawl_source}")

    try:
        df = load_crawl_data(file_content, filename)

        if df.empty:
            logger.warning(f"[Crawl Worker] Parsed file is empty for client={client_id}")
            return {'status': 'empty', 'rows_written': 0, 'crawl_run_id': None, 'error': None}

        now_iso = datetime.utcnow().isoformat()

        # --- Create crawl_runs record ---
        crawl_run = {
            'client_id': client_id,
            'source': crawl_source,
            'filename': filename,
            'total_urls': len(df),
            'started_at': now_iso,
            'completed_at': now_iso,
            'status': 'completed',
        }
        run_result = supabase_client.table('crawl_runs').insert(crawl_run).execute()
        crawl_run_id = run_result.data[0]['id'] if run_result.data else None

        # --- Prepare crawl_data records ---
        records = []
        for _, row in df.iterrows():
            record = {
                'client_id': client_id,
                'crawl_run_id': crawl_run_id,
                'fetched_at': now_iso,
            }
            # Copy all mapped columns
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

        # Delete previous crawl data for this client, then insert
        supabase_client.table('crawl_data').delete().eq('client_id', client_id).execute()
        _supabase_batch_insert(supabase_client, 'crawl_data', records)

        logger.info(f"[Crawl Worker] Wrote {len(records)} rows, crawl_run_id={crawl_run_id}")
        return {
            'status': 'success',
            'rows_written': len(records),
            'crawl_run_id': crawl_run_id,
            'error': None,
        }

    except Exception as e:
        logger.error(f"[Crawl Worker] Failed for client={client_id}: {e}")
        return {'status': 'error', 'rows_written': 0, 'crawl_run_id': None, 'error': str(e)}
