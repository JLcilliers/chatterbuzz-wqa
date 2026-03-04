"""
Google Business Profile (GBP) Data Pipeline Worker
====================================================
Stub worker for Google Business Profile API ingestion.

The actual GBP API integration will be implemented later. This module
defines the public interface so that the orchestrator and other pipeline
components can reference it now.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GBP data schema (expected columns once API is wired up)
# ---------------------------------------------------------------------------

GBP_COLUMNS = [
    'location_name',
    'location_id',
    'address',
    'phone',
    'website_url',
    'primary_category',
    'additional_categories',
    'total_reviews',
    'average_rating',
    'search_views',
    'map_views',
    'website_clicks',
    'direction_requests',
    'phone_calls',
    'photo_views',
    'posts_count',
]


# ---------------------------------------------------------------------------
# Stub fetch function (to be replaced with real GBP API calls)
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    reraise=True,
)
def fetch_gbp_data(
    credentials: Any,
    account_id: str,
    location_ids: Optional[List[str]] = None,
    days: int = 90,
) -> pd.DataFrame:
    """Fetch Google Business Profile performance data.

    This is a stub -- the real implementation will call the GBP API
    (Business Profile Performance API / My Business API) once details
    are finalized.

    Args:
        credentials: Google OAuth2 credentials with GBP scopes.
        account_id: GBP account identifier.
        location_ids: Optional list of specific location IDs to fetch.
            If None, fetches all locations under the account.
        days: Look-back window in days for performance metrics.

    Returns:
        DataFrame with columns matching GBP_COLUMNS.
    """
    # TODO: Implement GBP API calls:
    #   1. List locations (accounts.locations.list)
    #   2. For each location, fetch insights/performance metrics
    #   3. Fetch review summary
    #   4. Normalize into DataFrame
    logger.warning("[GBP Worker] fetch_gbp_data is a stub -- returning empty DataFrame")
    return pd.DataFrame(columns=GBP_COLUMNS)


def fetch_gbp_reviews(
    credentials: Any,
    account_id: str,
    location_id: str,
    max_reviews: int = 500,
) -> pd.DataFrame:
    """Fetch reviews for a specific GBP location.

    Stub -- to be implemented with the GBP Reviews API.

    Args:
        credentials: Google OAuth2 credentials with GBP scopes.
        account_id: GBP account identifier.
        location_id: Specific location ID.
        max_reviews: Maximum number of reviews to retrieve.

    Returns:
        DataFrame with columns: review_id, reviewer_name, rating, comment,
        create_time, update_time, reply_comment, reply_time.
    """
    review_columns = [
        'review_id', 'reviewer_name', 'rating', 'comment',
        'create_time', 'update_time', 'reply_comment', 'reply_time',
    ]
    logger.warning("[GBP Worker] fetch_gbp_reviews is a stub -- returning empty DataFrame")
    return pd.DataFrame(columns=review_columns)


# ---------------------------------------------------------------------------
# Public worker entry point
# ---------------------------------------------------------------------------

def run_gbp_worker(
    supabase_client,
    client_id: str,
    credentials: Any,
    account_id: str,
    location_ids: Optional[List[str]] = None,
    days: int = 90,
) -> dict:
    """Pipeline worker entry point: fetch GBP data and write to Supabase gbp_data table.

    Args:
        supabase_client: Initialised Supabase client.
        client_id: Unique identifier for the client / project.
        credentials: Google OAuth2 credentials with GBP scopes.
        account_id: GBP account identifier.
        location_ids: Optional list of specific location IDs.
        days: Look-back window in days (default 90).

    Returns:
        dict with keys: status, rows_written, error (if any).
    """
    logger.info(f"[GBP Worker] Starting for client={client_id}, account={account_id}")

    try:
        df = fetch_gbp_data(credentials, account_id, location_ids, days)

        if df.empty:
            logger.warning(f"[GBP Worker] No data returned for client={client_id} (stub)")
            return {'status': 'empty', 'rows_written': 0, 'error': None}

        now_iso = datetime.utcnow().isoformat()
        records = []
        for _, row in df.iterrows():
            record = {'client_id': client_id, 'fetched_at': now_iso}
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

        # Delete previous GBP data for this client, then insert
        supabase_client.table('gbp_data').delete().eq('client_id', client_id).execute()

        batch_size = 500
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            supabase_client.table('gbp_data').insert(batch).execute()

        logger.info(f"[GBP Worker] Wrote {len(records)} rows for client={client_id}")
        return {'status': 'success', 'rows_written': len(records), 'error': None}

    except Exception as e:
        logger.error(f"[GBP Worker] Failed for client={client_id}: {e}")
        return {'status': 'error', 'rows_written': 0, 'error': str(e)}
