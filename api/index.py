"""
Website Quality Audit (WQA) Generator - FastAPI Web Application

A web-based version of the WQA tool that can be deployed on Vercel.
Allows users to upload CSV files and receive Excel audit reports.
Supports Google OAuth for GA4 and GSC data integration.
"""

import io
import os
import json
import logging
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, urlencode, quote

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request, Response, Cookie
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

# Google OAuth imports
try:
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request as GoogleRequest
    from googleapiclient.discovery import build
    import httpx
    GOOGLE_OAUTH_AVAILABLE = True
except ImportError:
    GOOGLE_OAUTH_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="WQA Generator",
    description="Website Quality Audit Generator for SEO Agencies",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# GOOGLE OAUTH CONFIGURATION
# =============================================================================

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = os.environ.get("GOOGLE_REDIRECT_URI", "")
SECRET_KEY = os.environ.get("SECRET_KEY", secrets.token_hex(32))

# OAuth scopes needed for GA4 and GSC
GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/analytics.readonly",
    "https://www.googleapis.com/auth/webmasters.readonly",
    "openid",
    "email",
    "profile"
]

# In-memory token storage (for serverless - tokens stored in encrypted cookies)
# In production, you'd want to use a database
token_storage: Dict[str, dict] = {}

def encrypt_token(token_data: dict) -> str:
    """Simple token encoding for cookie storage"""
    from itsdangerous import URLSafeTimedSerializer
    serializer = URLSafeTimedSerializer(SECRET_KEY)
    return serializer.dumps(token_data)

def decrypt_token(token_string: str) -> Optional[dict]:
    """Decode token from cookie"""
    from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
    serializer = URLSafeTimedSerializer(SECRET_KEY)
    try:
        return serializer.loads(token_string, max_age=86400)  # 24 hour expiry
    except (BadSignature, SignatureExpired):
        return None

def get_google_auth_url(state: str) -> str:
    """Generate Google OAuth URL"""
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": " ".join(GOOGLE_SCOPES),
        "access_type": "offline",
        "prompt": "consent",
        "state": state
    }
    return f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"

async def exchange_code_for_tokens(code: str) -> dict:
    """Exchange authorization code for tokens"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": GOOGLE_REDIRECT_URI
            }
        )
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Token exchange failed: {response.text}")
        return response.json()

async def refresh_access_token(refresh_token: str) -> dict:
    """Refresh an expired access token"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token"
            }
        )
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Token refresh failed")
        return response.json()

def get_credentials_from_tokens(tokens: dict) -> Credentials:
    """Create Google credentials from token dict"""
    return Credentials(
        token=tokens.get("access_token"),
        refresh_token=tokens.get("refresh_token"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        scopes=GOOGLE_SCOPES
    )

# =============================================================================
# COLUMN MAPPING CONFIGURATION
# =============================================================================

# Crawl Data Column Mapping - supports Screaming Frog, Sitebulb, JetOctopus, OnCrawl, etc.
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
    'conversions': ['conversions', 'goal_completions', 'conversion_count', 'key_events'],
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
BACKLINK_COLUMN_MAP = {
    'url': ['url', 'target url', 'target_url', 'page', 'target', 'page url', 'target page'],
    'referring_domains': ['referring domains', 'referring_domains', 'ref domains', 'ref_domains',
                          'domains', 'rd', 'dofollow referring domains', 'root domains'],
    'backlinks': ['backlinks', 'backlink_count', 'external_links', 'links', 'total backlinks',
                  'dofollow backlinks', 'external backlinks', 'inbound links'],
    'authority': ['ur', 'url rating', 'url_rating', 'authority score', 'authority_score',
                  'page authority', 'page_authority', 'pa', 'trust flow', 'citation flow',
                  'domain rating', 'dr', 'as'],
    'anchor_texts': ['anchor_texts', 'anchors', 'anchor_text', 'anchor', 'top anchor'],
}


# =============================================================================
# API PROVIDER INTERFACES (Future Extension Points)
# =============================================================================
# These abstract base classes define the interface for future API integrations.
# Implement a provider by subclassing and registering with the provider registry.

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

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


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

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


def classify_page_type(url: str) -> str:
    if not url or pd.isna(url):
        return 'Other'
    url_lower = str(url).lower()
    try:
        parsed = urlparse(url_lower)
        path = parsed.path
    except Exception:
        path = url_lower

    if path in ['/', ''] or path.rstrip('/') == '':
        return 'Home'
    if any(pattern in path for pattern in ['/blog/', '/blog', '/news/', '/news', '/articles/', '/posts/']):
        return 'Blog'
    if any(pattern in path for pattern in ['/location/', '/locations/', '/city/', '/cities/',
                                            '/service-area/', '/service-areas/', '/area/', '/areas/',
                                            '/near-me/', '/local/']):
        return 'Local Lander'
    if any(pattern in path for pattern in ['/service/', '/services/', '/solutions/', '/products/',
                                            '/what-we-do/', '/offerings/']):
        return 'Service'
    return 'Other'


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
    numeric_cols = ['sessions', 'conversions', 'bounce_rate', 'avg_session_duration', 'ecom_revenue']
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


def load_backlink_data(file_content: Optional[bytes], filename: str = "") -> Optional[pd.DataFrame]:
    if file_content is None:
        return None
    df = read_file_to_dataframe(file_content, filename)
    df_mapped = map_columns(df, BACKLINK_COLUMN_MAP, 'Backlinks')
    for col in ['referring_domains', 'backlinks', 'authority']:
        if col in df_mapped.columns:
            df_mapped[col] = pd.to_numeric(df_mapped[col], errors='coerce').fillna(0)
    return df_mapped


# =============================================================================
# MERGING & PROCESSING
# =============================================================================

def merge_datasets(crawl_df, ga_df, gsc_df, backlink_df) -> pd.DataFrame:
    df = crawl_df.copy()
    df['url'] = df['url'].apply(normalize_url)
    df = df[df['url'] != '']

    if ga_df is not None and 'url' in ga_df.columns:
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
        'referring_domains': 0, 'backlinks': 0, 'authority': 0,
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


def assign_actions(row, low_traffic_threshold=5, thin_content_threshold=1000,
                   high_rank_max_position=20.0, low_ctr_threshold=0.05) -> Tuple[List[str], List[str]]:
    technical_actions = []
    content_actions = []

    status_code = int(row.get('status_code', 0))
    indexable = bool(row.get('indexable', True))
    canonical_url = str(row.get('canonical_url', '')).strip()
    url = str(row.get('url', '')).strip().lower()
    inlinks = int(row.get('inlinks', 0))
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

    # Technical actions
    if status_code == 302:
        technical_actions.append('301 Redirect')
    if status_code == 404:
        if backlinks > 0 or referring_domains > 0:
            technical_actions.append('301 Redirect')
        elif backlinks == 0 and inlinks > 0:
            technical_actions.append('Remove Internal Links')
    if status_code in [301, 308]:
        technical_actions.append('Review Redirect')
    if not indexable and in_sitemap:
        technical_actions.append('Remove from Sitemap')
    if indexable and status_code == 200 and sessions > 0 and not in_sitemap:
        technical_actions.append('Add to Sitemap')
    if canonical_url and canonical_url.lower() != url:
        technical_actions.append('Canonicalize')
    if sessions > 0 and inlinks == 0:
        technical_actions.append('Add Internal Links')
    if crawl_depth >= 4 and page_type in ['Home', 'Service', 'Local Lander']:
        if 'Add Internal Links' not in technical_actions:
            technical_actions.append('Add Internal Links')

    if page_type == 'Blog':
        technical_actions.append('Add Schema: Article')
    elif page_type == 'Local Lander':
        technical_actions.append('Add Schema: LocalBusiness')
    elif page_type == 'Home':
        technical_actions.append('Add Schema: Organization')

    # Content actions
    if (sessions <= low_traffic_threshold and conversions == 0 and
        backlinks == 0 and referring_domains == 0 and status_code == 200):
        content_actions.append('Delete (404)')
    elif (sessions <= low_traffic_threshold and conversions == 0 and
          (backlinks > 0 or referring_domains > 0) and status_code == 200):
        content_actions.append('301 Redirect')

    if (word_count < thin_content_threshold and (sessions > 0 or avg_position > 0) and
        status_code == 200 and 'Delete (404)' not in content_actions and
        '301 Redirect' not in content_actions):
        content_actions.append('Rewrite')

    if (word_count >= thin_content_threshold and sessions > 0 and
        2 < avg_position <= high_rank_max_position and status_code == 200 and
        'Delete (404)' not in content_actions and '301 Redirect' not in content_actions):
        content_actions.append('Refresh')

    if (word_count >= thin_content_threshold and sessions > 0 and
        3 <= avg_position <= 20 and referring_domains == 0 and status_code == 200 and
        'Delete (404)' not in content_actions and '301 Redirect' not in content_actions):
        content_actions.append('Target w/ Links')

    if status_code == 200:
        if (not meta_description or
            (avg_position <= high_rank_max_position and ctr < low_ctr_threshold and impressions > 100)):
            content_actions.append('Update Meta Description')
        if (not page_title or
            (avg_position <= high_rank_max_position and ctr < low_ctr_threshold and impressions > 100)):
            content_actions.append('Update Page Title')

    if not content_actions:
        content_actions.append('Leave As Is')

    return technical_actions, content_actions


def assign_priority(row) -> str:
    status_code = int(row.get('status_code', 0))
    page_type = str(row.get('page_type', 'Other'))
    content_actions = str(row.get('content_actions', ''))

    if status_code in [404, 302]:
        return 'High'
    if page_type in ['Home', 'Service', 'Local Lander']:
        if 'Rewrite' in content_actions or 'Refresh' in content_actions:
            return 'High'
    if 'Delete (404)' in content_actions or '301 Redirect' in content_actions:
        if page_type not in ['Home', 'Service', 'Local Lander']:
            return 'Medium'
    return 'Low'


def generate_summary(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    summaries = {}

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

    priority_counts = df['priority'].value_counts().reset_index()
    priority_counts.columns = ['Priority', 'Count']
    priority_order = ['High', 'Medium', 'Low']
    priority_counts['sort_order'] = priority_counts['Priority'].apply(
        lambda x: priority_order.index(x) if x in priority_order else 99
    )
    priority_counts = priority_counts.sort_values('sort_order').drop('sort_order', axis=1)
    summaries['priority'] = priority_counts

    page_type_counts = df['page_type'].value_counts().reset_index()
    page_type_counts.columns = ['Page Type', 'Count']
    summaries['page_type'] = page_type_counts

    status_counts = df['status_code'].value_counts().reset_index()
    status_counts.columns = ['Status Code', 'Count']
    status_counts = status_counts.sort_values('Status Code')
    summaries['status_code'] = status_counts

    return summaries


def extract_page_path(url: str) -> str:
    """Extract page path from URL (no protocol, no domain, no querystring)"""
    if pd.isna(url) or not url:
        return '/'
    try:
        parsed = urlparse(str(url))
        path = parsed.path if parsed.path else '/'
        # Ensure path starts with /
        if not path.startswith('/'):
            path = '/' + path
        return path
    except:
        return '/'


def format_indexability(row) -> Tuple[str, str]:
    """Return (Index/Noindex label, Indexation Status explanation)"""
    indexable = row.get('indexable', True)
    meta_robots = str(row.get('meta_robots', '')).lower()
    status_code = int(row.get('status_code', 200))
    canonical = str(row.get('canonical_url', '')).strip()
    url = str(row.get('url', '')).strip().lower()

    # Determine index/noindex
    if status_code >= 400:
        index_label = 'Non-Indexable'
        status_label = f'{status_code} Error'
    elif 'noindex' in meta_robots:
        index_label = 'Non-Indexable'
        status_label = 'Noindex'
    elif not indexable:
        index_label = 'Non-Indexable'
        if canonical and canonical.lower() != url:
            status_label = 'Canonicalised'
        else:
            status_label = 'Non-Indexable'
    else:
        index_label = 'Indexable'
        status_label = 'Indexable'

    return index_label, status_label


def create_excel_report(df: pd.DataFrame, summaries: Dict[str, pd.DataFrame]) -> bytes:
    """Create WQA Excel report with proper 3-header-row structure"""
    output = io.BytesIO()

    # Define the 38-column structure
    # Row 1: Section headings (sparse)
    section_headers = {
        0: '', 1: 'ANALYSIS', 2: '', 3: '', 4: '', 5: '',
        6: 'KEYWORD PERFORMANCE', 7: '', 8: '', 9: '', 10: '', 11: '',
        12: 'TRAFFIC PERFORMANCE', 13: '', 14: '', 15: '',
        16: 'ENGAGEMENT', 17: '',
        18: 'CONVERSIONS', 19: '', 20: '', 21: '',
        22: 'ON PAGE', 23: '', 24: '', 25: '', 26: '', 27: '', 28: '', 29: '', 30: '',
        31: 'TECHNICAL', 32: '', 33: '', 34: '', 35: '', 36: '', 37: ''
    }

    # Row 2: Source labels
    source_labels = {
        0: 'Formula', 1: 'SF', 2: 'Manual', 3: 'Manual', 4: 'Manual', 5: 'Manual',
        6: 'SEMrush', 7: 'SEMrush', 8: 'SEMrush', 9: 'SEMrush', 10: 'SEMrush', 11: 'SEMrush',
        12: 'GSC', 13: 'GA', 14: 'GA', 15: 'GA',
        16: 'GA', 17: 'GA',
        18: 'GA', 19: 'GA', 20: 'GA', 21: 'GA',
        22: 'SF', 23: 'SF', 24: 'GSC', 25: 'SF', 26: 'SF', 27: 'SF', 28: 'SF', 29: 'SF', 30: 'Ahrefs',
        31: 'SF', 32: 'SF', 33: 'SF', 34: 'SF', 35: 'SF', 36: 'Formula', 37: 'SF'
    }

    # Row 3: Column names
    column_names = [
        'page-path', 'URL', 'Category', 'Technical Action', 'Content Action', 'Final URL',
        'Main KW', 'Volume', 'Ranking', '"Best" KW', 'Volume', 'Ranking',
        'Impressions', 'Sessions', '% Change Sessions', 'Losing Traffic?',
        'Bounce rate (%)', 'Average session duration',
        'Conversions (All Goals)', 'Conversion Rate (%)', 'Ecom Revenue Generated', 'Ecom Conversion Rate',
        'Type', 'Current Title', 'SERP CTR', 'Meta', 'H1', 'Word Count', 'Inlinks', 'Outlinks', 'DOFOLLOW Links',
        'Canonical Link Element', 'Status Code', 'Index / Noindex', 'Indexation Status', 'Page Depth', 'In Sitemap?', 'Last Modified'
    ]

    # Build data rows
    data_rows = []
    for _, row in df.iterrows():
        # Get canonical URL or regular URL
        canonical = str(row.get('canonical_url', '')).strip()
        url = str(row.get('url', '')).strip()
        display_url = canonical if canonical and canonical != '' else url

        # Extract page path
        page_path = extract_page_path(display_url)

        # Get indexability info
        index_label, indexation_status = format_indexability(row)

        # Calculate conversion rate
        sessions = float(row.get('sessions', 0))
        conversions = float(row.get('conversions', 0))
        conv_rate = (conversions / sessions) if sessions > 0 else ''

        # Calculate % change sessions and losing traffic
        sessions_prev = float(row.get('sessions_prev', 0))
        if sessions_prev > 0:
            pct_change = (sessions - sessions_prev) / sessions_prev
            losing_traffic = 'Yes' if pct_change <= -0.1 else 'No'
        else:
            pct_change = ''
            losing_traffic = 'No YoY Data'

        # Format in_sitemap
        in_sitemap = 'Yes' if row.get('in_sitemap', False) else 'No'

        # Build row data matching column order
        data_row = [
            page_path,                                          # 0: page-path
            display_url,                                        # 1: URL
            row.get('page_type', 'Other'),                      # 2: Category
            row.get('technical_actions', ''),                   # 3: Technical Action
            row.get('content_actions', ''),                     # 4: Content Action
            row.get('final_url', ''),                           # 5: Final URL
            row.get('main_kw', ''),                             # 6: Main KW
            row.get('main_kw_volume', ''),                      # 7: Volume (Main KW)
            row.get('main_kw_ranking', ''),                     # 8: Ranking (Main KW)
            row.get('primary_keyword', ''),                     # 9: "Best" KW (from GSC)
            row.get('best_kw_volume', ''),                      # 10: Volume (Best KW)
            row.get('avg_position', ''),                        # 11: Ranking (Best KW) - using GSC position
            row.get('impressions', 0),                          # 12: Impressions
            sessions if sessions > 0 else '',                   # 13: Sessions
            pct_change,                                         # 14: % Change Sessions
            losing_traffic,                                     # 15: Losing Traffic?
            row.get('bounce_rate', ''),                         # 16: Bounce rate (%)
            row.get('avg_session_duration', ''),                # 17: Average session duration
            conversions if conversions > 0 else '',             # 18: Conversions (All Goals)
            conv_rate,                                          # 19: Conversion Rate (%)
            row.get('ecom_revenue', ''),                        # 20: Ecom Revenue Generated
            row.get('ecom_conv_rate', ''),                      # 21: Ecom Conversion Rate
            row.get('content_type', ''),                        # 22: Type
            row.get('page_title', ''),                          # 23: Current Title
            row.get('ctr', ''),                                 # 24: SERP CTR
            row.get('meta_description', ''),                    # 25: Meta
            row.get('h1', ''),                                  # 26: H1
            row.get('word_count', 0),                           # 27: Word Count
            row.get('inlinks', 0),                              # 28: Inlinks
            row.get('outlinks', 0),                             # 29: Outlinks
            row.get('backlinks', 0),                            # 30: DOFOLLOW Links
            row.get('canonical_url', ''),                       # 31: Canonical Link Element
            row.get('status_code', ''),                         # 32: Status Code
            index_label,                                        # 33: Index / Noindex
            indexation_status,                                  # 34: Indexation Status
            row.get('crawl_depth', ''),                         # 35: Page Depth
            in_sitemap,                                         # 36: In Sitemap?
            row.get('last_modified', ''),                       # 37: Last Modified
        ]
        data_rows.append(data_row)

    # Create DataFrame with proper structure
    wqa_df = pd.DataFrame(data_rows, columns=column_names)

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Write WQA sheet with 3 header rows
        workbook = writer.book
        worksheet = workbook.create_sheet('WQA', 0)

        # Row 1: Section headers
        for col_idx, header in section_headers.items():
            worksheet.cell(row=1, column=col_idx + 1, value=header)

        # Row 2: Source labels
        for col_idx, source in source_labels.items():
            worksheet.cell(row=2, column=col_idx + 1, value=source)

        # Row 3: Column names
        for col_idx, col_name in enumerate(column_names):
            worksheet.cell(row=3, column=col_idx + 1, value=col_name)

        # Row 4+: Data rows
        for row_idx, data_row in enumerate(data_rows):
            for col_idx, value in enumerate(data_row):
                worksheet.cell(row=row_idx + 4, column=col_idx + 1, value=value)

        # Apply some basic formatting
        from openpyxl.styles import Font, PatternFill, Alignment

        # Header row styling
        header_fill = PatternFill(start_color='4472C4', end_color='4472C4', fill_type='solid')
        header_font = Font(color='FFFFFF', bold=True)

        for col_idx in range(len(column_names)):
            # Section headers (Row 1) - only style cells with content
            cell1 = worksheet.cell(row=1, column=col_idx + 1)
            if cell1.value:
                cell1.fill = PatternFill(start_color='2F5496', end_color='2F5496', fill_type='solid')
                cell1.font = Font(color='FFFFFF', bold=True, size=12)

            # Source labels (Row 2)
            cell2 = worksheet.cell(row=2, column=col_idx + 1)
            cell2.fill = PatternFill(start_color='D6DCE5', end_color='D6DCE5', fill_type='solid')
            cell2.font = Font(italic=True, size=9)

            # Column names (Row 3)
            cell3 = worksheet.cell(row=3, column=col_idx + 1)
            cell3.fill = header_fill
            cell3.font = header_font

        # Freeze panes at row 4 (after headers)
        worksheet.freeze_panes = 'A4'

        # Auto-adjust column widths (basic)
        for col_idx, col_name in enumerate(column_names):
            col_letter = worksheet.cell(row=1, column=col_idx + 1).column_letter
            # Set reasonable default widths based on column type
            if col_name in ['URL', 'Canonical Link Element', 'Meta', 'Current Title']:
                worksheet.column_dimensions[col_letter].width = 50
            elif col_name in ['page-path', 'Final URL', 'H1']:
                worksheet.column_dimensions[col_letter].width = 35
            elif col_name in ['Technical Action', 'Content Action', 'Category', 'Indexation Status']:
                worksheet.column_dimensions[col_letter].width = 20
            else:
                worksheet.column_dimensions[col_letter].width = 15

        # Also write the raw aggregation data to a separate sheet for reference
        df.to_excel(writer, sheet_name='Raw Data', index=False)

        # Write summary sheet
        current_row = 0
        summary_order = [
            ('Priority Distribution', 'priority'),
            ('Technical Actions', 'technical_actions'),
            ('Content Actions', 'content_actions'),
            ('Page Types', 'page_type'),
            ('Status Codes', 'status_code'),
        ]
        for title, key in summary_order:
            if key in summaries:
                title_df = pd.DataFrame([[title]], columns=[''])
                title_df.to_excel(writer, sheet_name='Summary', index=False,
                                  header=False, startrow=current_row)
                current_row += 1
                summaries[key].to_excel(writer, sheet_name='Summary', index=False,
                                        startrow=current_row)
                current_row += len(summaries[key]) + 3

    output.seek(0)
    return output.getvalue()


# =============================================================================
# GOOGLE API INTEGRATION
# =============================================================================

async def fetch_ga4_properties(credentials: Credentials) -> List[dict]:
    """Fetch list of GA4 properties the user has access to"""
    try:
        service = build('analyticsadmin', 'v1beta', credentials=credentials)
        accounts = service.accountSummaries().list().execute()

        properties = []
        for account in accounts.get('accountSummaries', []):
            for prop in account.get('propertySummaries', []):
                properties.append({
                    'id': prop.get('property', '').replace('properties/', ''),
                    'name': prop.get('displayName', 'Unknown'),
                    'account': account.get('displayName', 'Unknown')
                })
        return properties
    except Exception as e:
        logger.error(f"Error fetching GA4 properties: {e}")
        return []

async def fetch_ga4_report(credentials: Credentials, property_id: str, days: int = 30) -> pd.DataFrame:
    """Fetch GA4 page data and return as DataFrame matching CSV format"""
    try:
        service = build('analyticsdata', 'v1beta', credentials=credentials)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        request_body = {
            'dateRanges': [{
                'startDate': start_date.strftime('%Y-%m-%d'),
                'endDate': end_date.strftime('%Y-%m-%d')
            }],
            'dimensions': [{'name': 'pagePath'}],
            'metrics': [
                {'name': 'sessions'},
                {'name': 'keyEvents'}  # This is GA4's conversion metric
            ],
            'limit': 25000
        }

        response = service.properties().runReport(
            property=f'properties/{property_id}',
            body=request_body
        ).execute()

        rows = []
        for row in response.get('rows', []):
            page_path = row['dimensionValues'][0]['value'] if row.get('dimensionValues') else ''
            sessions = int(row['metricValues'][0]['value']) if row.get('metricValues') else 0
            conversions = int(row['metricValues'][1]['value']) if len(row.get('metricValues', [])) > 1 else 0

            rows.append({
                'url': page_path,
                'sessions': sessions,
                'conversions': conversions
            })

        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=['url', 'sessions', 'conversions'])
    except Exception as e:
        logger.error(f"Error fetching GA4 report: {e}")
        return pd.DataFrame(columns=['url', 'sessions', 'conversions'])

async def fetch_gsc_sites(credentials: Credentials) -> List[dict]:
    """Fetch list of verified GSC sites"""
    try:
        service = build('searchconsole', 'v1', credentials=credentials)
        sites = service.sites().list().execute()

        return [
            {
                'url': site.get('siteUrl', ''),
                'permission': site.get('permissionLevel', 'Unknown')
            }
            for site in sites.get('siteEntry', [])
        ]
    except Exception as e:
        logger.error(f"Error fetching GSC sites: {e}")
        return []

async def fetch_gsc_report(credentials: Credentials, site_url: str, days: int = 90) -> pd.DataFrame:
    """Fetch GSC performance data and return as DataFrame matching CSV format"""
    try:
        service = build('searchconsole', 'v1', credentials=credentials)

        end_date = datetime.now() - timedelta(days=3)  # GSC data has ~3 day lag
        start_date = end_date - timedelta(days=days)

        request_body = {
            'startDate': start_date.strftime('%Y-%m-%d'),
            'endDate': end_date.strftime('%Y-%m-%d'),
            'dimensions': ['page'],
            'rowLimit': 25000
        }

        response = service.searchanalytics().query(
            siteUrl=site_url,
            body=request_body
        ).execute()

        rows = []
        for row in response.get('rows', []):
            rows.append({
                'url': row['keys'][0] if row.get('keys') else '',
                'clicks': row.get('clicks', 0),
                'impressions': row.get('impressions', 0),
                'ctr': row.get('ctr', 0),
                'avg_position': row.get('position', 0)
            })

        # Also fetch top query per page for primary_keyword
        query_request = {
            'startDate': start_date.strftime('%Y-%m-%d'),
            'endDate': end_date.strftime('%Y-%m-%d'),
            'dimensions': ['page', 'query'],
            'rowLimit': 25000
        }

        try:
            query_response = service.searchanalytics().query(
                siteUrl=site_url,
                body=query_request
            ).execute()

            # Group by page and get top query by impressions
            page_keywords = {}
            for row in query_response.get('rows', []):
                page = row['keys'][0] if row.get('keys') else ''
                query = row['keys'][1] if len(row.get('keys', [])) > 1 else ''
                impressions = row.get('impressions', 0)

                if page not in page_keywords or impressions > page_keywords[page]['impressions']:
                    page_keywords[page] = {'query': query, 'impressions': impressions}

            # Add primary_keyword to rows
            for row in rows:
                if row['url'] in page_keywords:
                    row['primary_keyword'] = page_keywords[row['url']]['query']
                else:
                    row['primary_keyword'] = ''
        except Exception as e:
            logger.warning(f"Could not fetch query data: {e}")
            for row in rows:
                row['primary_keyword'] = ''

        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=['url', 'clicks', 'impressions', 'ctr', 'avg_position', 'primary_keyword']
        )
    except Exception as e:
        logger.error(f"Error fetching GSC report: {e}")
        return pd.DataFrame(columns=['url', 'clicks', 'impressions', 'ctr', 'avg_position', 'primary_keyword'])


# =============================================================================
# OAUTH ENDPOINTS
# =============================================================================

@app.get("/api/auth/google")
async def google_auth_start(response: Response):
    """Start Google OAuth flow"""
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise HTTPException(
            status_code=503,
            detail="Google OAuth not configured. Set GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, and GOOGLE_REDIRECT_URI environment variables."
        )

    state = secrets.token_urlsafe(32)
    auth_url = get_google_auth_url(state)

    # Store state in cookie for verification
    redirect = RedirectResponse(url=auth_url, status_code=302)
    redirect.set_cookie(key="oauth_state", value=state, httponly=True, max_age=600, samesite="lax")
    return redirect

@app.get("/api/auth/google/callback")
async def google_auth_callback(
    request: Request,
    code: str = None,
    state: str = None,
    error: str = None,
    oauth_state: Optional[str] = Cookie(None)
):
    """Handle Google OAuth callback"""
    if error:
        return RedirectResponse(url=f"/?error={error}", status_code=302)

    if not code:
        return RedirectResponse(url="/?error=no_code", status_code=302)

    # Verify state
    if state != oauth_state:
        return RedirectResponse(url="/?error=invalid_state", status_code=302)

    try:
        # Exchange code for tokens
        tokens = await exchange_code_for_tokens(code)

        # Encrypt tokens for cookie storage
        encrypted = encrypt_token(tokens)

        # Redirect to app with success
        redirect = RedirectResponse(url="/?connected=google", status_code=302)
        redirect.set_cookie(
            key="google_tokens",
            value=encrypted,
            httponly=True,
            max_age=86400,  # 24 hours
            samesite="lax",
            secure=True
        )
        redirect.delete_cookie(key="oauth_state")
        return redirect
    except Exception as e:
        logger.error(f"OAuth callback error: {e}")
        return RedirectResponse(url=f"/?error=auth_failed", status_code=302)

@app.get("/api/auth/status")
async def auth_status(google_tokens: Optional[str] = Cookie(None)):
    """Check if user is authenticated with Google"""
    if not google_tokens:
        return {"connected": False}

    tokens = decrypt_token(google_tokens)
    if not tokens:
        return {"connected": False}

    return {"connected": True, "has_refresh": "refresh_token" in tokens}

@app.post("/api/auth/logout")
async def logout(response: Response):
    """Clear Google authentication"""
    response = JSONResponse({"success": True})
    response.delete_cookie(key="google_tokens")
    return response

@app.get("/api/ga4/properties")
async def list_ga4_properties(google_tokens: Optional[str] = Cookie(None)):
    """List GA4 properties available to the user"""
    if not google_tokens:
        raise HTTPException(status_code=401, detail="Not authenticated with Google")

    tokens = decrypt_token(google_tokens)
    if not tokens:
        raise HTTPException(status_code=401, detail="Invalid or expired session")

    credentials = get_credentials_from_tokens(tokens)
    properties = await fetch_ga4_properties(credentials)
    return {"properties": properties}

@app.get("/api/gsc/sites")
async def list_gsc_sites(google_tokens: Optional[str] = Cookie(None)):
    """List GSC sites available to the user"""
    if not google_tokens:
        raise HTTPException(status_code=401, detail="Not authenticated with Google")

    tokens = decrypt_token(google_tokens)
    if not tokens:
        raise HTTPException(status_code=401, detail="Invalid or expired session")

    credentials = get_credentials_from_tokens(tokens)
    sites = await fetch_gsc_sites(credentials)
    return {"sites": sites}

@app.post("/api/ga4/report")
async def get_ga4_report(
    property_id: str = Form(...),
    days: int = Form(30),
    google_tokens: Optional[str] = Cookie(None)
):
    """Fetch GA4 report data"""
    if not google_tokens:
        raise HTTPException(status_code=401, detail="Not authenticated with Google")

    tokens = decrypt_token(google_tokens)
    if not tokens:
        raise HTTPException(status_code=401, detail="Invalid or expired session")

    credentials = get_credentials_from_tokens(tokens)
    df = await fetch_ga4_report(credentials, property_id, days)
    return {"rows": df.to_dict('records'), "count": len(df)}

@app.post("/api/gsc/report")
async def get_gsc_report(
    site_url: str = Form(...),
    days: int = Form(90),
    google_tokens: Optional[str] = Cookie(None)
):
    """Fetch GSC report data"""
    if not google_tokens:
        raise HTTPException(status_code=401, detail="Not authenticated with Google")

    tokens = decrypt_token(google_tokens)
    if not tokens:
        raise HTTPException(status_code=401, detail="Invalid or expired session")

    credentials = get_credentials_from_tokens(tokens)
    df = await fetch_gsc_report(credentials, site_url, days)
    return {"rows": df.to_dict('records'), "count": len(df)}


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WQA Generator - Website Quality Audit</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 40px;
        }
        h1 { color: #333; margin-bottom: 10px; font-size: 2rem; }
        .subtitle { color: #666; margin-bottom: 30px; font-size: 1.1rem; }
        .form-section { margin-bottom: 25px; }
        .form-section h3 {
            color: #444; margin-bottom: 10px; font-size: 1rem;
            display: flex; align-items: center; gap: 8px;
        }
        .required { color: #e74c3c; }
        .optional { color: #27ae60; font-size: 0.85rem; }
        .file-input-wrapper { position: relative; overflow: hidden; display: inline-block; width: 100%; }
        .file-input-wrapper input[type=file] {
            font-size: 100px; position: absolute; left: 0; top: 0;
            opacity: 0; cursor: pointer; width: 100%; height: 100%;
        }
        .file-input-btn {
            background: #f8f9fa; border: 2px dashed #ddd; border-radius: 8px;
            padding: 20px; text-align: center; transition: all 0.3s; cursor: pointer;
        }
        .file-input-btn:hover { border-color: #667eea; background: #f0f0ff; }
        .file-input-btn.has-file { border-color: #27ae60; background: #e8f8f0; }
        .file-input-btn.disabled { opacity: 0.5; cursor: not-allowed; }
        .file-name { margin-top: 8px; font-size: 0.9rem; color: #27ae60; }
        .threshold-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin-top: 20px; }
        .threshold-item label { display: block; font-size: 0.85rem; color: #666; margin-bottom: 5px; }
        .threshold-item input, .threshold-item select {
            width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 6px; font-size: 1rem;
        }
        .submit-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; border: none; padding: 15px 40px; font-size: 1.1rem;
            border-radius: 8px; cursor: pointer; width: 100%; margin-top: 20px;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .submit-btn:hover { transform: translateY(-2px); box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4); }
        .submit-btn:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
        .status { margin-top: 20px; padding: 15px; border-radius: 8px; display: none; }
        .status.loading { display: block; background: #fff3cd; color: #856404; }
        .status.success { display: block; background: #d4edda; color: #155724; }
        .status.error { display: block; background: #f8d7da; color: #721c24; }
        .info-box {
            background: #e8f4fd; border-left: 4px solid #667eea;
            padding: 15px; margin-bottom: 25px; border-radius: 0 8px 8px 0;
        }
        .info-box p { color: #444; font-size: 0.9rem; line-height: 1.5; }
        .helper-text {
            margin-top: 10px; padding: 12px 15px; background: #f8f9fa;
            border-radius: 6px; font-size: 0.85rem; color: #555; line-height: 1.6;
        }
        .helper-text strong { color: #444; }

        /* Google OAuth styles */
        .google-section {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border: 2px solid #dee2e6; border-radius: 12px;
            padding: 25px; margin-bottom: 30px;
        }
        .google-section h2 { color: #333; font-size: 1.3rem; margin-bottom: 15px; display: flex; align-items: center; gap: 10px; }
        .google-btn {
            display: inline-flex; align-items: center; gap: 10px;
            background: white; border: 2px solid #4285f4; color: #4285f4;
            padding: 12px 24px; border-radius: 8px; font-size: 1rem;
            cursor: pointer; transition: all 0.3s; font-weight: 500;
        }
        .google-btn:hover { background: #4285f4; color: white; }
        .google-btn.connected { background: #34a853; border-color: #34a853; color: white; }
        .google-btn svg { width: 20px; height: 20px; }
        .google-status { margin-top: 15px; font-size: 0.9rem; }
        .google-status.connected { color: #34a853; }
        .google-status.not-connected { color: #666; }
        .logout-btn {
            background: none; border: none; color: #dc3545;
            cursor: pointer; font-size: 0.85rem; margin-left: 15px;
            text-decoration: underline;
        }

        /* Data source toggle */
        .data-source-toggle {
            display: flex; gap: 10px; margin-bottom: 15px;
            background: #f1f3f4; padding: 5px; border-radius: 8px;
        }
        .toggle-btn {
            flex: 1; padding: 10px 15px; border: none; background: transparent;
            border-radius: 6px; cursor: pointer; font-size: 0.9rem;
            transition: all 0.2s; color: #666;
        }
        .toggle-btn.active { background: white; color: #667eea; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .toggle-btn:hover:not(.active) { background: rgba(255,255,255,0.5); }

        /* Property/site selectors */
        .api-selector { margin-top: 15px; }
        .api-selector select {
            width: 100%; padding: 12px; border: 2px solid #ddd;
            border-radius: 8px; font-size: 1rem; background: white;
        }
        .api-selector select:focus { border-color: #667eea; outline: none; }
        .api-selector.has-selection select { border-color: #27ae60; }
        .api-selector label { display: block; font-size: 0.85rem; color: #666; margin-bottom: 8px; }
        .loading-indicator { color: #666; font-style: italic; padding: 10px 0; }

        .hidden { display: none !important; }

        @media (max-width: 600px) {
            .threshold-grid { grid-template-columns: 1fr; }
            .container { padding: 20px; }
            .data-source-toggle { flex-direction: column; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>WQA Generator</h1>
        <p class="subtitle">Website Quality Audit Report Generator for SEO Agencies</p>

        <div class="info-box">
            <p><strong>How it works:</strong> Upload your crawl data (required) and add GA4/GSC data either by connecting your Google account or uploading CSVs. The tool merges all data sources, applies SEO rules, and generates a comprehensive Excel report with prioritized actions.</p>
        </div>

        <!-- Google OAuth Section -->
        <div class="google-section">
            <h2>
                <svg viewBox="0 0 24 24" width="24" height="24"><path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/><path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/><path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/><path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/></svg>
                Connect Google Data
            </h2>
            <p style="color: #666; margin-bottom: 15px; font-size: 0.9rem;">Connect your Google account to automatically pull GA4 and Search Console data instead of uploading CSVs.</p>

            <div id="googleAuthSection">
                <button class="google-btn" id="googleConnectBtn" onclick="window.location.href='/api/auth/google'">
                    <svg viewBox="0 0 24 24"><path fill="currentColor" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/><path fill="currentColor" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/></svg>
                    Connect with Google
                </button>
                <div class="google-status not-connected" id="googleStatus">Not connected</div>
            </div>

            <!-- Property/Site selectors (shown when connected) -->
            <div id="googleSelectors" class="hidden">
                <div class="api-selector" id="ga4SelectorWrapper">
                    <label>GA4 Property</label>
                    <select id="ga4PropertySelect">
                        <option value="">-- Select GA4 Property --</option>
                    </select>
                </div>
                <div class="api-selector" id="gscSelectorWrapper" style="margin-top: 15px;">
                    <label>Search Console Site</label>
                    <select id="gscSiteSelect">
                        <option value="">-- Select GSC Site --</option>
                    </select>
                </div>
            </div>
        </div>

        <form id="wqaForm" enctype="multipart/form-data">
            <div class="form-section">
                <h3><span class="required">*</span> Crawl Data (Required)</h3>
                <div class="file-input-wrapper">
                    <div class="file-input-btn" id="crawlBtn">
                        <div>Click or drag to upload crawl CSV (e.g., Screaming Frog / Sitebulb "All URLs" export)</div>
                        <div class="file-name" id="crawlFileName"></div>
                    </div>
                    <input type="file" name="crawl_file" id="crawlFile" accept=".csv" required>
                </div>
                <div class="helper-text">
                    Upload a full-site crawl report with one row per URL. Recommended: Screaming Frog "Internal → Export → All" or Sitebulb "All URLs" table export.<br>
                    <strong>Required columns (or equivalents):</strong> URL, Status Code, Indexability/Meta Robots, Canonical URL, Word Count, Title, Meta Description, H1, Crawl Depth, Inlinks, Sitemap status
                </div>
            </div>

            <!-- GA4 Section with toggle -->
            <div class="form-section" id="ga4Section">
                <h3>GA4 Data <span class="optional">(Optional)</span></h3>
                <div class="data-source-toggle" id="ga4Toggle">
                    <button type="button" class="toggle-btn active" data-source="csv" onclick="toggleDataSource('ga4', 'csv')">Upload CSV</button>
                    <button type="button" class="toggle-btn" data-source="api" onclick="toggleDataSource('ga4', 'api')" id="ga4ApiToggle" disabled>Use Google API</button>
                </div>
                <div id="ga4CsvUpload">
                    <div class="file-input-wrapper">
                        <div class="file-input-btn" id="gaBtn">
                            <div>Click or drag to upload GA4 CSV</div>
                            <div class="file-name" id="gaFileName"></div>
                        </div>
                        <input type="file" name="ga_file" id="gaFile" accept=".csv">
                    </div>
                </div>
                <div id="ga4ApiInfo" class="hidden" style="padding: 15px; background: #e8f8f0; border-radius: 8px; color: #155724;">
                    <strong>Using Google API</strong> - Data will be pulled from the selected GA4 property above.
                </div>
            </div>

            <!-- GSC Section with toggle -->
            <div class="form-section" id="gscSection">
                <h3>GSC Data <span class="optional">(Optional)</span></h3>
                <div class="data-source-toggle" id="gscToggle">
                    <button type="button" class="toggle-btn active" data-source="csv" onclick="toggleDataSource('gsc', 'csv')">Upload CSV</button>
                    <button type="button" class="toggle-btn" data-source="api" onclick="toggleDataSource('gsc', 'api')" id="gscApiToggle" disabled>Use Google API</button>
                </div>
                <div id="gscCsvUpload">
                    <div class="file-input-wrapper">
                        <div class="file-input-btn" id="gscBtn">
                            <div>Click or drag to upload GSC CSV</div>
                            <div class="file-name" id="gscFileName"></div>
                        </div>
                        <input type="file" name="gsc_file" id="gscFile" accept=".csv">
                    </div>
                </div>
                <div id="gscApiInfo" class="hidden" style="padding: 15px; background: #e8f8f0; border-radius: 8px; color: #155724;">
                    <strong>Using Google API</strong> - Data will be pulled from the selected GSC site above.
                </div>
            </div>

            <div class="form-section">
                <h3>Backlink Data <span class="optional">(Optional)</span></h3>
                <div class="file-input-wrapper">
                    <div class="file-input-btn" id="backlinkBtn">
                        <div>Click or drag to upload backlinks CSV (e.g., Ahrefs / Semrush / Moz per-URL export)</div>
                        <div class="file-name" id="backlinkFileName"></div>
                    </div>
                    <input type="file" name="backlink_file" id="backlinkFile" accept=".csv">
                </div>
                <div class="helper-text">
                    Upload per-URL backlink metrics from your link tool (Ahrefs, Semrush, Moz, Majestic).<br>
                    Recommended: "Best by links" / "Indexed pages" export.<br>
                    <strong>Required columns (or equivalents):</strong> URL, Referring Domains, Backlinks, Page Authority (UR/PA/AS)
                </div>
            </div>

            <div class="form-section">
                <h3>Thresholds</h3>
                <div class="threshold-grid">
                    <div class="threshold-item">
                        <label>Low Traffic (sessions)</label>
                        <input type="number" name="low_traffic_threshold" value="5" min="0">
                    </div>
                    <div class="threshold-item">
                        <label>Thin Content (words)</label>
                        <input type="number" name="thin_content_threshold" value="1000" min="0">
                    </div>
                    <div class="threshold-item">
                        <label>High Rank Max Position</label>
                        <input type="number" name="high_rank_max_position" value="20" min="1" step="0.1">
                    </div>
                    <div class="threshold-item">
                        <label>Low CTR Threshold</label>
                        <input type="number" name="low_ctr_threshold" value="0.05" min="0" max="1" step="0.01">
                    </div>
                </div>
            </div>

            <!-- Hidden fields for API data source flags -->
            <input type="hidden" name="use_ga4_api" id="useGa4Api" value="false">
            <input type="hidden" name="ga4_property_id" id="ga4PropertyId" value="">
            <input type="hidden" name="use_gsc_api" id="useGscApi" value="false">
            <input type="hidden" name="gsc_site_url" id="gscSiteUrl" value="">

            <button type="submit" class="submit-btn" id="submitBtn">Generate WQA Report</button>
        </form>

        <div id="status" class="status"></div>
    </div>

    <script>
        let isGoogleConnected = false;
        let ga4DataSource = 'csv';
        let gscDataSource = 'csv';

        // Check auth status on page load
        async function checkAuthStatus() {
            try {
                const response = await fetch('/api/auth/status');
                const data = await response.json();

                isGoogleConnected = data.connected;
                updateGoogleUI();

                if (isGoogleConnected) {
                    await loadGoogleData();
                }
            } catch (error) {
                console.error('Error checking auth status:', error);
            }
        }

        function updateGoogleUI() {
            const connectBtn = document.getElementById('googleConnectBtn');
            const status = document.getElementById('googleStatus');
            const selectors = document.getElementById('googleSelectors');
            const ga4ApiToggle = document.getElementById('ga4ApiToggle');
            const gscApiToggle = document.getElementById('gscApiToggle');

            if (isGoogleConnected) {
                connectBtn.classList.add('connected');
                connectBtn.innerHTML = '<svg viewBox="0 0 24 24" width="20" height="20"><path fill="currentColor" d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/></svg> Connected';
                status.className = 'google-status connected';
                status.innerHTML = 'Connected to Google <button class="logout-btn" onclick="logout()">Disconnect</button>';
                selectors.classList.remove('hidden');
                ga4ApiToggle.disabled = false;
                gscApiToggle.disabled = false;
            } else {
                connectBtn.classList.remove('connected');
                connectBtn.innerHTML = '<svg viewBox="0 0 24 24" width="20" height="20"><path fill="currentColor" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/></svg> Connect with Google';
                status.className = 'google-status not-connected';
                status.textContent = 'Not connected';
                selectors.classList.add('hidden');
                ga4ApiToggle.disabled = true;
                gscApiToggle.disabled = true;
            }
        }

        async function loadGoogleData() {
            // Load GA4 properties
            try {
                const ga4Response = await fetch('/api/ga4/properties');
                if (ga4Response.ok) {
                    const ga4Data = await ga4Response.json();
                    const select = document.getElementById('ga4PropertySelect');
                    select.innerHTML = '<option value="">-- Select GA4 Property --</option>';
                    ga4Data.properties.forEach(prop => {
                        select.innerHTML += `<option value="${prop.id}">${prop.name} (${prop.account})</option>`;
                    });
                }
            } catch (error) {
                console.error('Error loading GA4 properties:', error);
            }

            // Load GSC sites
            try {
                const gscResponse = await fetch('/api/gsc/sites');
                if (gscResponse.ok) {
                    const gscData = await gscResponse.json();
                    const select = document.getElementById('gscSiteSelect');
                    select.innerHTML = '<option value="">-- Select GSC Site --</option>';
                    gscData.sites.forEach(site => {
                        select.innerHTML += `<option value="${site.url}">${site.url}</option>`;
                    });
                }
            } catch (error) {
                console.error('Error loading GSC sites:', error);
            }
        }

        async function logout() {
            try {
                await fetch('/api/auth/logout', { method: 'POST' });
                isGoogleConnected = false;
                toggleDataSource('ga4', 'csv');
                toggleDataSource('gsc', 'csv');
                updateGoogleUI();
            } catch (error) {
                console.error('Error logging out:', error);
            }
        }

        function toggleDataSource(type, source) {
            const toggleBtns = document.querySelectorAll(`#${type}Toggle .toggle-btn`);
            toggleBtns.forEach(btn => {
                btn.classList.toggle('active', btn.dataset.source === source);
            });

            const csvUpload = document.getElementById(`${type}CsvUpload`);
            const apiInfo = document.getElementById(`${type}ApiInfo`);
            const useApiField = document.getElementById(`use${type.charAt(0).toUpperCase() + type.slice(1)}Api`);

            if (source === 'csv') {
                csvUpload.classList.remove('hidden');
                apiInfo.classList.add('hidden');
                useApiField.value = 'false';
            } else {
                csvUpload.classList.add('hidden');
                apiInfo.classList.remove('hidden');
                useApiField.value = 'true';
            }

            if (type === 'ga4') ga4DataSource = source;
            if (type === 'gsc') gscDataSource = source;
        }

        // File input handlers
        function setupFileInput(inputId, btnId, nameId) {
            const input = document.getElementById(inputId);
            const btn = document.getElementById(btnId);
            const nameEl = document.getElementById(nameId);

            input.addEventListener('change', function() {
                if (this.files.length > 0) {
                    nameEl.textContent = this.files[0].name;
                    btn.classList.add('has-file');
                } else {
                    nameEl.textContent = '';
                    btn.classList.remove('has-file');
                }
            });
        }

        setupFileInput('crawlFile', 'crawlBtn', 'crawlFileName');
        setupFileInput('gaFile', 'gaBtn', 'gaFileName');
        setupFileInput('gscFile', 'gscBtn', 'gscFileName');
        setupFileInput('backlinkFile', 'backlinkBtn', 'backlinkFileName');

        // Update hidden fields when selectors change
        document.getElementById('ga4PropertySelect').addEventListener('change', function() {
            document.getElementById('ga4PropertyId').value = this.value;
        });
        document.getElementById('gscSiteSelect').addEventListener('change', function() {
            document.getElementById('gscSiteUrl').value = this.value;
        });

        // Form submission
        document.getElementById('wqaForm').addEventListener('submit', async function(e) {
            e.preventDefault();

            const submitBtn = document.getElementById('submitBtn');
            const status = document.getElementById('status');

            submitBtn.disabled = true;
            submitBtn.textContent = 'Generating Report...';
            status.className = 'status loading';
            status.textContent = 'Processing your data. This may take a moment...';

            const formData = new FormData(this);

            try {
                const response = await fetch('/api/generate', {
                    method: 'POST',
                    body: formData
                });

                if (response.ok) {
                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'wqa_report.xlsx';
                    document.body.appendChild(a);
                    a.click();
                    window.URL.revokeObjectURL(url);
                    a.remove();

                    status.className = 'status success';
                    status.textContent = 'Report generated successfully! Download should start automatically.';
                } else {
                    const error = await response.json();
                    throw new Error(error.detail || 'Failed to generate report');
                }
            } catch (error) {
                status.className = 'status error';
                status.textContent = 'Error: ' + error.message;
            } finally {
                submitBtn.disabled = false;
                submitBtn.textContent = 'Generate WQA Report';
            }
        });

        // Check for connection status on URL params
        const urlParams = new URLSearchParams(window.location.search);
        if (urlParams.get('connected') === 'google') {
            // Clear the URL param
            window.history.replaceState({}, document.title, window.location.pathname);
        }
        if (urlParams.get('error')) {
            const status = document.getElementById('status');
            status.className = 'status error';
            status.textContent = 'Google authentication failed: ' + urlParams.get('error');
            window.history.replaceState({}, document.title, window.location.pathname);
        }

        // Initialize
        checkAuthStatus();
    </script>
</body>
</html>
"""


@app.post("/api/generate")
async def generate_report(
    request: Request,
    crawl_file: UploadFile = File(...),
    ga_file: Optional[UploadFile] = File(None),
    gsc_file: Optional[UploadFile] = File(None),
    backlink_file: Optional[UploadFile] = File(None),
    low_traffic_threshold: int = Form(5),
    thin_content_threshold: int = Form(1000),
    high_rank_max_position: float = Form(20.0),
    low_ctr_threshold: float = Form(0.05),
    use_ga4_api: str = Form("false"),
    ga4_property_id: str = Form(""),
    use_gsc_api: str = Form("false"),
    gsc_site_url: str = Form(""),
    google_tokens: Optional[str] = Cookie(None)
):
    """Generate WQA report from uploaded CSV/Excel files or Google API data"""
    try:
        # Read crawl file (required)
        crawl_content = await crawl_file.read()
        crawl_df = load_crawl_data(crawl_content, crawl_file.filename or "")
        logger.info(f"Loaded crawl data: {len(crawl_df)} rows")

        # Get GA4 data - either from API or file
        ga_df = None
        if use_ga4_api == "true" and ga4_property_id and google_tokens:
            tokens = decrypt_token(google_tokens)
            if tokens:
                credentials = get_credentials_from_tokens(tokens)
                ga_df = await fetch_ga4_report(credentials, ga4_property_id)
                logger.info(f"Loaded GA4 data from API: {len(ga_df)} rows")
        elif ga_file and ga_file.filename:
            ga_content = await ga_file.read()
            if ga_content:
                ga_df = load_ga_data(ga_content, ga_file.filename or "")
                logger.info(f"Loaded GA4 data from file: {len(ga_df)} rows")

        # Get GSC data - either from API or file
        gsc_df = None
        if use_gsc_api == "true" and gsc_site_url and google_tokens:
            tokens = decrypt_token(google_tokens)
            if tokens:
                credentials = get_credentials_from_tokens(tokens)
                gsc_df = await fetch_gsc_report(credentials, gsc_site_url)
                logger.info(f"Loaded GSC data from API: {len(gsc_df)} rows")
        elif gsc_file and gsc_file.filename:
            gsc_content = await gsc_file.read()
            if gsc_content:
                gsc_df = load_gsc_data(gsc_content, gsc_file.filename or "")
                logger.info(f"Loaded GSC data from file: {len(gsc_df)} rows")

        backlink_df = None
        if backlink_file and backlink_file.filename:
            backlink_content = await backlink_file.read()
            if backlink_content:
                backlink_df = load_backlink_data(backlink_content, backlink_file.filename or "")
                logger.info(f"Loaded backlink data: {len(backlink_df)} rows")

        # Merge datasets
        df = merge_datasets(crawl_df, ga_df, gsc_df, backlink_df)
        logger.info(f"Merged data: {len(df)} rows")

        # Classify page types
        df['page_type'] = df['url'].apply(classify_page_type)

        # Assign actions
        actions_result = df.apply(
            lambda row: assign_actions(
                row,
                low_traffic_threshold=low_traffic_threshold,
                thin_content_threshold=thin_content_threshold,
                high_rank_max_position=high_rank_max_position,
                low_ctr_threshold=low_ctr_threshold
            ),
            axis=1,
            result_type='expand'
        )
        df['technical_actions'] = actions_result[0].apply(lambda x: ', '.join(x) if x else '')
        df['content_actions'] = actions_result[1].apply(lambda x: ', '.join(x) if x else '')

        # Assign priority
        df['priority'] = df.apply(assign_priority, axis=1)

        # Generate summary
        summaries = generate_summary(df)

        # Create Excel report
        excel_bytes = create_excel_report(df, summaries)

        return StreamingResponse(
            io.BytesIO(excel_bytes),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=wqa_report.xlsx"}
        )

    except Exception as e:
        logger.error(f"Error generating report: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "WQA Generator"}
