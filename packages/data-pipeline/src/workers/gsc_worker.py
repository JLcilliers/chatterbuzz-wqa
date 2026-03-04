"""
GSC Data Pipeline Worker
========================
Standalone workers that fetch Google Search Console page-level and
query-level data, then write results to the gsc_data table in Supabase.

Ported from api/index.py fetch_gsc_report (lines 3503-3573) and
fetch_gsc_query_data (lines 3576-3644).
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GSC_PAGE_EMPTY_COLUMNS = ['url', 'clicks', 'impressions', 'ctr', 'avg_position', 'primary_keyword']
GSC_QUERY_EMPTY_COLUMNS = ['query', 'url', 'clicks', 'impressions', 'ctr', 'avg_position']


# ---------------------------------------------------------------------------
# GSC page-level fetch (ported identically from monolith)
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    reraise=True,
)
def _fetch_gsc_page_data(service, site_url: str, start_date: datetime, end_date: datetime) -> list:
    """Fetch page-level GSC performance data with retry."""
    request_body = {
        'startDate': start_date.strftime('%Y-%m-%d'),
        'endDate': end_date.strftime('%Y-%m-%d'),
        'dimensions': ['page'],
        'rowLimit': 25000,
    }

    response = service.searchanalytics().query(
        siteUrl=site_url,
        body=request_body,
    ).execute()

    rows = []
    for row in response.get('rows', []):
        rows.append({
            'url': row['keys'][0] if row.get('keys') else '',
            'clicks': row.get('clicks', 0),
            'impressions': row.get('impressions', 0),
            'ctr': row.get('ctr', 0),
            'avg_position': row.get('position', 0),
        })

    return rows


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    reraise=True,
)
def _fetch_gsc_page_query_keywords(service, site_url: str, start_date: datetime, end_date: datetime) -> dict:
    """Fetch query+page data to determine primary keyword per page, with retry."""
    query_request = {
        'startDate': start_date.strftime('%Y-%m-%d'),
        'endDate': end_date.strftime('%Y-%m-%d'),
        'dimensions': ['page', 'query'],
        'rowLimit': 25000,
    }

    query_response = service.searchanalytics().query(
        siteUrl=site_url,
        body=query_request,
    ).execute()

    # Group by page and get top query by impressions
    page_keywords: dict = {}
    for row in query_response.get('rows', []):
        page = row['keys'][0] if row.get('keys') else ''
        query = row['keys'][1] if len(row.get('keys', [])) > 1 else ''
        impressions = row.get('impressions', 0)

        if page not in page_keywords or impressions > page_keywords[page]['impressions']:
            page_keywords[page] = {'query': query, 'impressions': impressions}

    return page_keywords


def fetch_gsc_report(
    credentials: Credentials,
    site_url: str,
    days: int = 90,
) -> pd.DataFrame:
    """Fetch GSC performance data and return as DataFrame matching CSV format.

    Returns DataFrame with columns: url, clicks, impressions, ctr, avg_position, primary_keyword
    """
    try:
        service = build('searchconsole', 'v1', credentials=credentials)

        end_date = datetime.now() - timedelta(days=3)  # GSC data has ~3 day lag
        start_date = end_date - timedelta(days=days)

        rows = _fetch_gsc_page_data(service, site_url, start_date, end_date)

        # Enrich with primary_keyword from query+page data
        try:
            page_keywords = _fetch_gsc_page_query_keywords(service, site_url, start_date, end_date)
            for row in rows:
                if row['url'] in page_keywords:
                    row['primary_keyword'] = page_keywords[row['url']]['query']
                else:
                    row['primary_keyword'] = ''
        except Exception as e:
            logger.warning(f"Could not fetch query data: {e}")
            for row in rows:
                row['primary_keyword'] = ''

        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=GSC_PAGE_EMPTY_COLUMNS)

    except Exception as e:
        logger.error(f"Error fetching GSC report: {e}")
        return pd.DataFrame(columns=GSC_PAGE_EMPTY_COLUMNS)


# ---------------------------------------------------------------------------
# GSC query-level fetch (ported identically from monolith)
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    reraise=True,
)
def _fetch_gsc_query_page_data(service, site_url: str, start_date: datetime, end_date: datetime) -> list:
    """Fetch query+page level rows from GSC with retry."""
    request_body = {
        'startDate': start_date.strftime('%Y-%m-%d'),
        'endDate': end_date.strftime('%Y-%m-%d'),
        'dimensions': ['query', 'page'],  # Critical: query first, then page
        'rowLimit': 25000,
    }

    logger.info(f"[GSC QUERY] Fetching query+page data for {site_url} ({(end_date - start_date).days} days)")
    response = service.searchanalytics().query(
        siteUrl=site_url,
        body=request_body,
    ).execute()

    rows = []
    for row in response.get('rows', []):
        keys = row.get('keys', [])
        if len(keys) >= 2:
            rows.append({
                'query': keys[0],
                'url': keys[1],
                'clicks': row.get('clicks', 0),
                'impressions': row.get('impressions', 0),
                'ctr': row.get('ctr', 0),
                'avg_position': row.get('position', 0),
            })

    return rows


def fetch_gsc_query_data(
    credentials: Credentials,
    site_url: str,
    days: int = 90,
) -> pd.DataFrame:
    """Fetch GSC query-level data for topic clustering and content analysis.

    Returns query+page level data needed for:
    - New Content Opportunities (topic clustering)
    - Redirect & Merge Plan (cannibalization detection)
    - Merge Playbooks (consolidation recommendations)

    Args:
        credentials: Google OAuth credentials.
        site_url: GSC property URL (must match exactly, including protocol).
        days: Number of days to fetch (default 90).

    Returns:
        DataFrame with columns: query, url, clicks, impressions, ctr, avg_position
    """
    empty_df = pd.DataFrame(columns=GSC_QUERY_EMPTY_COLUMNS)

    try:
        service = build('searchconsole', 'v1', credentials=credentials)

        end_date = datetime.now() - timedelta(days=3)  # GSC data has ~3 day lag
        start_date = end_date - timedelta(days=days)

        rows = _fetch_gsc_query_page_data(service, site_url, start_date, end_date)

        if rows:
            df = pd.DataFrame(rows)
            unique_queries = df['query'].nunique()
            unique_pages = df['url'].nunique()
            total_impressions = df['impressions'].sum()
            logger.info(
                f"[GSC QUERY] Fetched {len(df)} query+page rows: "
                f"{unique_queries} unique queries, {unique_pages} unique pages, "
                f"{total_impressions:,.0f} total impressions"
            )
            return df
        else:
            logger.warning(
                f"[GSC QUERY] No query+page data returned for {site_url} "
                f"(date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})"
            )
            return empty_df

    except Exception as e:
        logger.error(f"[GSC QUERY] Error fetching query data: {e}")
        return empty_df


# ---------------------------------------------------------------------------
# Public worker entry point
# ---------------------------------------------------------------------------

def run_gsc_worker(
    supabase_client,
    client_id: str,
    credentials: Credentials,
    site_url: str,
    days: int = 90,
) -> dict:
    """Pipeline worker entry point: fetch GSC page + query data, write to Supabase gsc_data table.

    Args:
        supabase_client: Initialised Supabase client.
        client_id: Unique identifier for the client / project.
        credentials: Google OAuth2 Credentials object.
        site_url: GSC property URL.
        days: Look-back window in days (default 90).

    Returns:
        dict with keys: status, rows_written, query_rows_written, error (if any).
    """
    logger.info(f"[GSC Worker] Starting for client={client_id}, site={site_url}, days={days}")

    page_rows_written = 0
    query_rows_written = 0

    try:
        # --- Page-level data ---
        page_df = fetch_gsc_report(credentials, site_url, days)

        if not page_df.empty:
            now_iso = datetime.utcnow().isoformat()
            page_records = []
            for _, row in page_df.iterrows():
                page_records.append({
                    'client_id': client_id,
                    'url': row['url'],
                    'clicks': int(row['clicks']),
                    'impressions': int(row['impressions']),
                    'ctr': float(row['ctr']),
                    'avg_position': float(row['avg_position']),
                    'primary_keyword': row.get('primary_keyword', ''),
                    'data_type': 'page',
                    'fetched_at': now_iso,
                })

            # Clear existing page-level rows, then insert
            supabase_client.table('gsc_data').delete().eq(
                'client_id', client_id
            ).eq('data_type', 'page').execute()

            batch_size = 500
            for i in range(0, len(page_records), batch_size):
                batch = page_records[i:i + batch_size]
                supabase_client.table('gsc_data').insert(batch).execute()

            page_rows_written = len(page_records)
            logger.info(f"[GSC Worker] Wrote {page_rows_written} page rows for client={client_id}")
        else:
            logger.warning(f"[GSC Worker] No page-level data for client={client_id}")

        # --- Query-level data ---
        query_df = fetch_gsc_query_data(credentials, site_url, days)

        if not query_df.empty:
            now_iso = datetime.utcnow().isoformat()
            query_records = []
            for _, row in query_df.iterrows():
                query_records.append({
                    'client_id': client_id,
                    'query': row['query'],
                    'url': row['url'],
                    'clicks': int(row['clicks']),
                    'impressions': int(row['impressions']),
                    'ctr': float(row['ctr']),
                    'avg_position': float(row['avg_position']),
                    'data_type': 'query',
                    'fetched_at': now_iso,
                })

            # Clear existing query-level rows, then insert
            supabase_client.table('gsc_data').delete().eq(
                'client_id', client_id
            ).eq('data_type', 'query').execute()

            batch_size = 500
            for i in range(0, len(query_records), batch_size):
                batch = query_records[i:i + batch_size]
                supabase_client.table('gsc_data').insert(batch).execute()

            query_rows_written = len(query_records)
            logger.info(f"[GSC Worker] Wrote {query_rows_written} query rows for client={client_id}")
        else:
            logger.warning(f"[GSC Worker] No query-level data for client={client_id}")

        status = 'success' if (page_rows_written + query_rows_written) > 0 else 'empty'
        return {
            'status': status,
            'rows_written': page_rows_written,
            'query_rows_written': query_rows_written,
            'error': None,
        }

    except Exception as e:
        logger.error(f"[GSC Worker] Failed for client={client_id}: {e}")
        return {
            'status': 'error',
            'rows_written': page_rows_written,
            'query_rows_written': query_rows_written,
            'error': str(e),
        }
