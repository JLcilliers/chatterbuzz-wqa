"""
GA4 Data Pipeline Worker
========================
Standalone worker that fetches GA4 page-level analytics data and writes
results to the ga4_data table in Supabase.

Ported from api/index.py fetch_ga4_report (lines 3353-3484).
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
# Helpers (mirrored from monolith to keep workers self-contained)
# ---------------------------------------------------------------------------

def normalize_page_path(path: str) -> str:
    """Normalize a GA4 pagePath value for joining.

    GA4 returns pagePath like '/states/tennessee/' - we normalize it
    to match url_to_page_path output.
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


# ---------------------------------------------------------------------------
# GA4 fetch logic (ported identically from monolith)
# ---------------------------------------------------------------------------

GA4_EMPTY_COLUMNS = [
    'page_path', 'sessions', 'conversions', 'bounce_rate',
    'avg_session_duration', 'ecom_revenue', 'sessions_prev',
]


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    reraise=True,
)
def _fetch_ga4_current_period(
    service, property_id: str, start_date: datetime, end_date: datetime
) -> dict:
    """Fetch current-period GA4 page metrics with retry."""
    current_request = {
        'dateRanges': [{
            'startDate': start_date.strftime('%Y-%m-%d'),
            'endDate': end_date.strftime('%Y-%m-%d'),
        }],
        'dimensions': [{'name': 'pagePath'}],
        'metrics': [
            {'name': 'sessions'},
            {'name': 'keyEvents'},          # GA4's conversion metric
            {'name': 'bounceRate'},
            {'name': 'averageSessionDuration'},
            {'name': 'purchaseRevenue'},     # E-commerce revenue
        ],
        'limit': 25000,
    }

    response = service.properties().runReport(
        property=f'properties/{property_id}',
        body=current_request,
    ).execute()

    # Build current period data map: page_path -> metrics
    current_data = {}
    for row in response.get('rows', []):
        raw_path = row['dimensionValues'][0]['value'] if row.get('dimensionValues') else '/'
        page_path = normalize_page_path(raw_path)
        metrics = row.get('metricValues', [])

        sessions = int(float(metrics[0]['value'])) if len(metrics) > 0 else 0
        conversions = int(float(metrics[1]['value'])) if len(metrics) > 1 else 0
        bounce_rate = float(metrics[2]['value']) if len(metrics) > 2 else 0.0
        avg_duration = float(metrics[3]['value']) if len(metrics) > 3 else 0.0
        revenue = float(metrics[4]['value']) if len(metrics) > 4 else 0.0

        # Aggregate if same path appears multiple times
        if page_path in current_data:
            current_data[page_path]['sessions'] += sessions
            current_data[page_path]['conversions'] += conversions
            current_data[page_path]['ecom_revenue'] += revenue
            # For rates, we'd need weighted average - keep simple for now
        else:
            current_data[page_path] = {
                'sessions': sessions,
                'conversions': conversions,
                'bounce_rate': bounce_rate,
                'avg_session_duration': avg_duration,
                'ecom_revenue': revenue,
            }

    return current_data


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    reraise=True,
)
def _fetch_ga4_previous_period(
    service, property_id: str, prev_start_date: datetime, prev_end_date: datetime
) -> dict:
    """Fetch previous-period GA4 sessions with retry."""
    prev_request = {
        'dateRanges': [{
            'startDate': prev_start_date.strftime('%Y-%m-%d'),
            'endDate': prev_end_date.strftime('%Y-%m-%d'),
        }],
        'dimensions': [{'name': 'pagePath'}],
        'metrics': [{'name': 'sessions'}],
        'limit': 25000,
    }

    response = service.properties().runReport(
        property=f'properties/{property_id}',
        body=prev_request,
    ).execute()

    prev_data = {}
    for row in response.get('rows', []):
        raw_path = row['dimensionValues'][0]['value'] if row.get('dimensionValues') else '/'
        page_path = normalize_page_path(raw_path)
        metrics = row.get('metricValues', [])
        sessions_prev = int(float(metrics[0]['value'])) if len(metrics) > 0 else 0

        if page_path in prev_data:
            prev_data[page_path] += sessions_prev
        else:
            prev_data[page_path] = sessions_prev

    return prev_data


def _combine_ga4_periods(current_data: dict, prev_data: dict) -> pd.DataFrame:
    """Merge current and previous period data into a single DataFrame."""
    rows = []
    all_paths = set(current_data.keys()) | set(prev_data.keys())

    for page_path in all_paths:
        current = current_data.get(page_path, {})
        rows.append({
            'page_path': page_path,
            'sessions': current.get('sessions', 0),
            'conversions': current.get('conversions', 0),
            'bounce_rate': current.get('bounce_rate', 0.0),
            'avg_session_duration': current.get('avg_session_duration', 0.0),
            'ecom_revenue': current.get('ecom_revenue', 0.0),
            'sessions_prev': prev_data.get(page_path, 0),
        })

    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(columns=GA4_EMPTY_COLUMNS)


# ---------------------------------------------------------------------------
# Public worker entry point
# ---------------------------------------------------------------------------

def fetch_ga4_report(
    credentials: Credentials,
    property_id: str,
    days: int = 90,
) -> pd.DataFrame:
    """Fetch GA4 page data with current and previous period for YoY comparison.

    Returns DataFrame with columns:
    - page_path: normalized path for joining with crawl data
    - sessions, conversions, bounce_rate, avg_session_duration, ecom_revenue: current period
    - sessions_prev: previous period sessions for YoY comparison
    """
    try:
        service = build('analyticsdata', 'v1beta', credentials=credentials)

        # Current period: last N days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Previous period: same duration, one year ago
        prev_end_date = end_date - timedelta(days=365)
        prev_start_date = prev_end_date - timedelta(days=days)

        # === CURRENT PERIOD ===
        current_data = _fetch_ga4_current_period(service, property_id, start_date, end_date)
        logger.info(f"GA4 current period: {len(current_data)} unique page paths")

        # === PREVIOUS PERIOD (for YoY comparison) ===
        prev_data: dict = {}
        try:
            prev_data = _fetch_ga4_previous_period(
                service, property_id, prev_start_date, prev_end_date,
            )
            logger.info(f"GA4 previous period: {len(prev_data)} unique page paths")
        except Exception as prev_e:
            logger.warning(f"Could not fetch previous period GA4 data: {prev_e}")

        # === COMBINE INTO DATAFRAME ===
        result_df = _combine_ga4_periods(current_data, prev_data)
        logger.info(f"GA4 combined result: {len(result_df)} rows")
        return result_df

    except Exception as e:
        logger.error(f"Error fetching GA4 report: {e}")
        return pd.DataFrame(columns=GA4_EMPTY_COLUMNS)


def run_ga4_worker(
    supabase_client,
    client_id: str,
    credentials: Credentials,
    property_id: str,
    days: int = 90,
) -> dict:
    """Pipeline worker entry point: fetch GA4 data and write to Supabase ga4_data table.

    Args:
        supabase_client: Initialised Supabase client.
        client_id: Unique identifier for the client / project.
        credentials: Google OAuth2 Credentials object.
        property_id: GA4 property ID (numeric string).
        days: Look-back window in days (default 90).

    Returns:
        dict with keys: status, rows_written, error (if any).
    """
    logger.info(f"[GA4 Worker] Starting for client={client_id}, property={property_id}, days={days}")

    try:
        df = fetch_ga4_report(credentials, property_id, days)

        if df.empty:
            logger.warning(f"[GA4 Worker] No data returned for client={client_id}")
            return {'status': 'empty', 'rows_written': 0, 'error': None}

        # Prepare records for upsert
        now_iso = datetime.utcnow().isoformat()
        records = []
        for _, row in df.iterrows():
            records.append({
                'client_id': client_id,
                'page_path': row['page_path'],
                'sessions': int(row['sessions']),
                'conversions': int(row['conversions']),
                'bounce_rate': float(row['bounce_rate']),
                'avg_session_duration': float(row['avg_session_duration']),
                'ecom_revenue': float(row['ecom_revenue']),
                'sessions_prev': int(row['sessions_prev']),
                'fetched_at': now_iso,
            })

        # Delete existing rows for this client, then insert fresh data
        supabase_client.table('ga4_data').delete().eq('client_id', client_id).execute()
        # Insert in batches of 500
        batch_size = 500
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            supabase_client.table('ga4_data').insert(batch).execute()

        logger.info(f"[GA4 Worker] Wrote {len(records)} rows for client={client_id}")
        return {'status': 'success', 'rows_written': len(records), 'error': None}

    except Exception as e:
        logger.error(f"[GA4 Worker] Failed for client={client_id}: {e}")
        return {'status': 'error', 'rows_written': 0, 'error': str(e)}
