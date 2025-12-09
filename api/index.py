"""
Website Quality Audit (WQA) Generator - FastAPI Web Application

A web-based version of the WQA tool that can be deployed on Vercel.
Allows users to upload CSV files and receive Excel audit reports.
"""

import io
import logging
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="WQA Generator",
    description="Website Quality Audit Generator for SEO Agencies",
    version="1.0.0"
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
# COLUMN MAPPING CONFIGURATION
# =============================================================================

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

def load_crawl_data(file_content: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_content), low_memory=False)
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


def load_ga_data(file_content: Optional[bytes]) -> Optional[pd.DataFrame]:
    if file_content is None:
        return None
    df = pd.read_csv(io.BytesIO(file_content), low_memory=False)
    df_mapped = map_columns(df, GA_COLUMN_MAP, 'GA4')
    for col in ['sessions', 'conversions']:
        if col in df_mapped.columns:
            df_mapped[col] = pd.to_numeric(df_mapped[col], errors='coerce').fillna(0)
    return df_mapped


def load_gsc_data(file_content: Optional[bytes]) -> Optional[pd.DataFrame]:
    if file_content is None:
        return None
    df = pd.read_csv(io.BytesIO(file_content), low_memory=False)
    df_mapped = map_columns(df, GSC_COLUMN_MAP, 'GSC')
    for col in ['avg_position', 'ctr', 'clicks', 'impressions']:
        if col in df_mapped.columns:
            df_mapped[col] = pd.to_numeric(df_mapped[col], errors='coerce').fillna(0)
    return df_mapped


def load_backlink_data(file_content: Optional[bytes]) -> Optional[pd.DataFrame]:
    if file_content is None:
        return None
    df = pd.read_csv(io.BytesIO(file_content), low_memory=False)
    df_mapped = map_columns(df, BACKLINK_COLUMN_MAP, 'Backlinks')
    for col in ['referring_domains', 'backlinks']:
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
        ga_agg = ga_df.groupby('url').agg({'sessions': 'sum', 'conversions': 'sum'}).reset_index()
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
        if bl_agg_cols:
            bl_agg = backlink_df.groupby('url').agg(bl_agg_cols).reset_index()
            df = df.merge(bl_agg, on='url', how='left', suffixes=('', '_bl'))

    expected_columns = {
        'url': '', 'status_code': 0, 'indexable': True, 'canonical_url': '',
        'inlinks': 0, 'outlinks': 0, 'crawl_depth': 0, 'in_sitemap': False,
        'page_title': '', 'meta_description': '', 'word_count': 0,
        'sessions': 0, 'conversions': 0, 'avg_position': 0.0, 'ctr': 0.0,
        'clicks': 0, 'impressions': 0, 'referring_domains': 0, 'backlinks': 0,
        'primary_keyword': '',
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


def create_excel_report(df: pd.DataFrame, summaries: Dict[str, pd.DataFrame]) -> bytes:
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Aggregation', index=False)

        action_columns = [
            'url', 'page_type', 'status_code', 'sessions', 'avg_position',
            'referring_domains', 'backlinks', 'word_count', 'inlinks',
            'technical_actions', 'content_actions', 'priority'
        ]
        action_columns = [col for col in action_columns if col in df.columns]
        df_actions = df[action_columns].copy()
        priority_map = {'High': 0, 'Medium': 1, 'Low': 2}
        df_actions['priority_sort'] = df_actions['priority'].map(priority_map)
        df_actions = df_actions.sort_values(
            ['priority_sort', 'sessions'], ascending=[True, False]
        ).drop('priority_sort', axis=1)
        df_actions.to_excel(writer, sheet_name='Actions', index=False)

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
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 40px;
        }
        h1 {
            color: #333;
            margin-bottom: 10px;
            font-size: 2rem;
        }
        .subtitle {
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1rem;
        }
        .form-section {
            margin-bottom: 25px;
        }
        .form-section h3 {
            color: #444;
            margin-bottom: 10px;
            font-size: 1rem;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .required { color: #e74c3c; }
        .optional { color: #27ae60; font-size: 0.85rem; }
        .file-input-wrapper {
            position: relative;
            overflow: hidden;
            display: inline-block;
            width: 100%;
        }
        .file-input-wrapper input[type=file] {
            font-size: 100px;
            position: absolute;
            left: 0;
            top: 0;
            opacity: 0;
            cursor: pointer;
            width: 100%;
            height: 100%;
        }
        .file-input-btn {
            background: #f8f9fa;
            border: 2px dashed #ddd;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            transition: all 0.3s;
            cursor: pointer;
        }
        .file-input-btn:hover {
            border-color: #667eea;
            background: #f0f0ff;
        }
        .file-input-btn.has-file {
            border-color: #27ae60;
            background: #e8f8f0;
        }
        .file-name {
            margin-top: 8px;
            font-size: 0.9rem;
            color: #27ae60;
        }
        .threshold-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin-top: 20px;
        }
        .threshold-item label {
            display: block;
            font-size: 0.85rem;
            color: #666;
            margin-bottom: 5px;
        }
        .threshold-item input {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-size: 1rem;
        }
        .submit-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            font-size: 1.1rem;
            border-radius: 8px;
            cursor: pointer;
            width: 100%;
            margin-top: 20px;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .submit-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }
        .submit-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        .status {
            margin-top: 20px;
            padding: 15px;
            border-radius: 8px;
            display: none;
        }
        .status.loading {
            display: block;
            background: #fff3cd;
            color: #856404;
        }
        .status.success {
            display: block;
            background: #d4edda;
            color: #155724;
        }
        .status.error {
            display: block;
            background: #f8d7da;
            color: #721c24;
        }
        .info-box {
            background: #e8f4fd;
            border-left: 4px solid #667eea;
            padding: 15px;
            margin-bottom: 25px;
            border-radius: 0 8px 8px 0;
        }
        .info-box p {
            color: #444;
            font-size: 0.9rem;
            line-height: 1.5;
        }
        @media (max-width: 600px) {
            .threshold-grid { grid-template-columns: 1fr; }
            .container { padding: 20px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>WQA Generator</h1>
        <p class="subtitle">Website Quality Audit Report Generator for SEO Agencies</p>

        <div class="info-box">
            <p><strong>How it works:</strong> Upload your crawl data (required) and optionally add GA4, GSC, and backlink data. The tool will merge all data sources, apply SEO rules, and generate a comprehensive Excel report with prioritized actions.</p>
        </div>

        <form id="wqaForm" enctype="multipart/form-data">
            <div class="form-section">
                <h3><span class="required">*</span> Crawl Data (Required)</h3>
                <div class="file-input-wrapper">
                    <div class="file-input-btn" id="crawlBtn">
                        <div>Click or drag to upload crawl CSV</div>
                        <div class="file-name" id="crawlFileName"></div>
                    </div>
                    <input type="file" name="crawl_file" id="crawlFile" accept=".csv" required>
                </div>
            </div>

            <div class="form-section">
                <h3>GA4 Data <span class="optional">(Optional)</span></h3>
                <div class="file-input-wrapper">
                    <div class="file-input-btn" id="gaBtn">
                        <div>Click or drag to upload GA4 CSV</div>
                        <div class="file-name" id="gaFileName"></div>
                    </div>
                    <input type="file" name="ga_file" id="gaFile" accept=".csv">
                </div>
            </div>

            <div class="form-section">
                <h3>GSC Data <span class="optional">(Optional)</span></h3>
                <div class="file-input-wrapper">
                    <div class="file-input-btn" id="gscBtn">
                        <div>Click or drag to upload GSC CSV</div>
                        <div class="file-name" id="gscFileName"></div>
                    </div>
                    <input type="file" name="gsc_file" id="gscFile" accept=".csv">
                </div>
            </div>

            <div class="form-section">
                <h3>Backlink Data <span class="optional">(Optional)</span></h3>
                <div class="file-input-wrapper">
                    <div class="file-input-btn" id="backlinkBtn">
                        <div>Click or drag to upload backlinks CSV</div>
                        <div class="file-name" id="backlinkFileName"></div>
                    </div>
                    <input type="file" name="backlink_file" id="backlinkFile" accept=".csv">
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

            <button type="submit" class="submit-btn" id="submitBtn">Generate WQA Report</button>
        </form>

        <div id="status" class="status"></div>
    </div>

    <script>
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
    </script>
</body>
</html>
"""


@app.post("/api/generate")
async def generate_report(
    crawl_file: UploadFile = File(...),
    ga_file: Optional[UploadFile] = File(None),
    gsc_file: Optional[UploadFile] = File(None),
    backlink_file: Optional[UploadFile] = File(None),
    low_traffic_threshold: int = Form(5),
    thin_content_threshold: int = Form(1000),
    high_rank_max_position: float = Form(20.0),
    low_ctr_threshold: float = Form(0.05)
):
    """Generate WQA report from uploaded CSV files"""
    try:
        # Read crawl file (required)
        crawl_content = await crawl_file.read()
        crawl_df = load_crawl_data(crawl_content)
        logger.info(f"Loaded crawl data: {len(crawl_df)} rows")

        # Read optional files
        ga_df = None
        if ga_file and ga_file.filename:
            ga_content = await ga_file.read()
            if ga_content:
                ga_df = load_ga_data(ga_content)
                logger.info(f"Loaded GA4 data: {len(ga_df)} rows")

        gsc_df = None
        if gsc_file and gsc_file.filename:
            gsc_content = await gsc_file.read()
            if gsc_content:
                gsc_df = load_gsc_data(gsc_content)
                logger.info(f"Loaded GSC data: {len(gsc_df)} rows")

        backlink_df = None
        if backlink_file and backlink_file.filename:
            backlink_content = await backlink_file.read()
            if backlink_content:
                backlink_df = load_backlink_data(backlink_content)
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
            axis=1
        )
        df['technical_actions'] = actions_result.apply(lambda x: ', '.join(x[0]) if x[0] else '')
        df['content_actions'] = actions_result.apply(lambda x: ', '.join(x[1]) if x[1] else '')

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
