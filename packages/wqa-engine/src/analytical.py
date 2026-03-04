"""
Analytical sheet builders for the WQA Engine.

Ported from api/index.py (lines 1528-2755). Each function takes a DataFrame
and returns a DataFrame or dict. Logic is kept identical to the API version.
"""

import re
import math
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# BUSINESS RELEVANCE & FEASIBILITY FILTERING CONSTANTS
# =============================================================================
# These constants filter and score topics to ensure only realistic, sensible,
# on-brand opportunities are surfaced. A topic must pass all three gates:
# 1. Search Demand Gate (handled by clustering)
# 2. SEO Feasibility Gate (handled by action recommendation)
# 3. Business Relevance Gate (this module)

# Exclusion terms - topics containing these are automatically excluded
NAVIGATIONAL_TERMS_API = {
    'login', 'signin', 'sign in', 'signup', 'sign up', 'logout', 'log out',
    'register', 'forgot password', 'reset password', 'my account', 'dashboard',
    'admin', 'portal', 'intranet', 'employee', 'staff only'
}

POLICY_LEGAL_TERMS_API = {
    'privacy policy', 'terms of service', 'terms and conditions', 'cookie policy',
    'gdpr', 'disclaimer', 'legal notice', 'refund policy', 'return policy',
    'shipping policy', 'cancellation policy', 'accessibility statement'
}

JUNK_ARTIFACT_TERMS_API = {
    'utm_', 'utm=', '?p=', '?s=', 'index.php', 'page=', 'amp', '&amp',
    'session', 'sessionid', 'jsessionid', 'phpsessid', 'cfid', 'cftoken',
    'www.', 'http://', 'https://', '.html', '.php', '.aspx', '.jsp'
}

AUTHORITY_REQUIRED_TERMS_API = {
    # Medical - requires licensed practitioner
    'diagnosis', 'treatment plan', 'medical advice', 'prescription', 'dosage',
    'symptoms of', 'cure for', 'medication for', 'drug interaction',
    # Legal - requires licensed attorney
    'legal advice', 'attorney', 'lawsuit', 'sue for', 'liability for',
    'contract law', 'legal rights', 'file a claim',
    # Financial - requires licensed advisor
    'investment advice', 'tax advice', 'financial planning', 'retirement planning',
    'stock picks', 'crypto investment', 'guaranteed returns',
    # Government - requires official authority
    'official form', 'government application', 'visa application', 'passport renewal'
}

# Intent classification modifiers
TRANSACTIONAL_MODIFIERS_API = {
    'pricing', 'price', 'cost', 'quote', 'buy', 'purchase', 'order', 'hire',
    'service', 'services', 'provider', 'providers', 'company', 'companies',
    'contractor', 'contractors', 'professional', 'professionals', 'near me',
    'in my area', 'local', 'get', 'schedule', 'book', 'appointment'
}

COMMERCIAL_MODIFIERS_API = {
    'best', 'top', 'review', 'reviews', 'compare', 'comparison', 'vs', 'versus',
    'alternative', 'alternatives', 'software', 'tool', 'tools', 'solution',
    'solutions', 'platform', 'app', 'rated', 'recommended', 'affordable'
}

INFORMATIONAL_MODIFIERS_API = {
    'how', 'what', 'why', 'when', 'where', 'guide', 'tutorial', 'tips',
    'steps', 'process', 'explained', 'meaning', 'definition', 'example',
    'examples', 'learn', 'understand', 'basics', 'introduction', 'overview'
}


# =============================================================================
# CONTENT TO OPTIMIZE
# =============================================================================

def build_content_to_optimize_api(df: pd.DataFrame) -> pd.DataFrame:
    """Build Content to Optimize sheet - API version."""
    mask = (df['status_code'] == 200) & (df['indexable'] == True)
    candidates = df[mask].copy()

    if len(candidates) == 0:
        return pd.DataFrame(columns=[
            'URL', 'Page Type', 'Avg Position', 'Impressions', 'Clicks', 'CTR (%)',
            'Word Count', 'Has Meta Description', 'Sessions', 'Optimization Signals', 'Priority Score'
        ])

    optimization_signals = []
    priority_scores = []

    for _, row in candidates.iterrows():
        signals = []
        score = 0

        if row['impressions'] >= 100 and row['ctr'] < 0.05:
            signals.append('Low CTR despite impressions')
            score += 3
        elif row['impressions'] >= 50 and row['ctr'] < 0.03:
            signals.append('Very low CTR')
            score += 4

        if 5 <= row['avg_position'] <= 10:
            signals.append('Position 5-10 (near page 1)')
            score += 5
        elif 11 <= row['avg_position'] <= 20:
            signals.append('Position 11-20 (striking distance)')
            score += 3

        meta_desc = str(row.get('meta_description', '')).strip()
        if not meta_desc:
            signals.append('Missing meta description')
            score += 2
        elif len(meta_desc) < 70:
            signals.append('Short meta description')
            score += 1

        if row['word_count'] < 500:
            signals.append('Very thin content (<500 words)')
            score += 3
        elif row['word_count'] < 1000:
            signals.append('Thin content (<1000 words)')
            score += 2

        if row['sessions'] > 0 and row['avg_position'] > 5:
            signals.append('Has traffic, room to improve')
            score += 1

        if row['impressions'] >= 500:
            signals.append('High search demand')
            score += 2
        elif row['impressions'] >= 100:
            signals.append('Moderate search demand')
            score += 1

        optimization_signals.append('; '.join(signals) if signals else 'No specific signals')
        priority_scores.append(score)

    candidates['Optimization Signals'] = optimization_signals
    candidates['Priority Score'] = priority_scores
    candidates = candidates[candidates['Priority Score'] > 0]

    if len(candidates) == 0:
        return pd.DataFrame(columns=[
            'URL', 'Page Type', 'Avg Position', 'Impressions', 'Clicks', 'CTR (%)',
            'Word Count', 'Has Meta Description', 'Sessions', 'Optimization Signals', 'Priority Score'
        ])

    # Sort by priority score descending and cap at top 20
    candidates = candidates.sort_values('Priority Score', ascending=False).head(20)

    return pd.DataFrame({
        'URL': candidates['url'],
        'Page Type': candidates['page_type'],
        'Avg Position': candidates['avg_position'].round(1),
        'Impressions': candidates['impressions'].astype(int),
        'Clicks': candidates['clicks'].astype(int),
        'CTR (%)': (candidates['ctr'] * 100).round(2),
        'Word Count': candidates['word_count'].astype(int),
        'Has Meta Description': candidates['meta_description'].apply(lambda x: 'Yes' if str(x).strip() else 'No'),
        'Sessions': candidates['sessions'].astype(int),
        'Optimization Signals': candidates['Optimization Signals'],
        'Priority Score': candidates['Priority Score']
    })


# =============================================================================
# THIN CONTENT OPPORTUNITIES
# =============================================================================

def build_thin_content_api(df: pd.DataFrame, thin_threshold: int = 1000) -> pd.DataFrame:
    """Build Thin Content Opportunities sheet - API version."""
    mask = (df['status_code'] == 200) & (df['indexable'] == True) & (df['word_count'] < thin_threshold)
    candidates = df[mask].copy()

    if len(candidates) == 0:
        return pd.DataFrame(columns=[
            'URL', 'Page Type', 'Word Count', 'Content Gap', 'Sessions',
            'Impressions', 'Avg Position', 'Referring Domains', 'Opportunity Type', 'Value Score'
        ])

    opportunity_types = []
    value_scores = []

    for _, row in candidates.iterrows():
        opp_type = []
        score = 0

        if row['sessions'] >= 10:
            opp_type.append('Has significant traffic')
            score += 5
        elif row['sessions'] > 0:
            opp_type.append('Has some traffic')
            score += 2

        if row['impressions'] >= 100:
            opp_type.append('High search visibility')
            score += 4
        elif row['impressions'] > 0:
            opp_type.append('Some search visibility')
            score += 1

        if 0 < row['avg_position'] <= 20:
            opp_type.append('Already ranking (pos <= 20)')
            score += 3
        elif 20 < row['avg_position'] <= 50:
            opp_type.append('Has rankings (pos 20-50)')
            score += 1

        if row['referring_domains'] >= 5:
            opp_type.append('Strong backlink profile')
            score += 4
        elif row['referring_domains'] > 0:
            opp_type.append('Has backlinks')
            score += 2

        if row['page_type'] in ['Service', 'Local Lander', 'Home']:
            opp_type.append(f'Strategic page ({row["page_type"]})')
            score += 3

        if row['word_count'] < 300:
            opp_type.append('Severely thin (<300 words)')
        elif row['word_count'] < 500:
            opp_type.append('Very thin (<500 words)')

        opportunity_types.append('; '.join(opp_type) if opp_type else 'Low priority')
        value_scores.append(score)

    candidates['Opportunity Type'] = opportunity_types
    candidates['Value Score'] = value_scores
    candidates = candidates[candidates['Value Score'] > 0]

    if len(candidates) == 0:
        return pd.DataFrame(columns=[
            'URL', 'Page Type', 'Word Count', 'Content Gap', 'Sessions',
            'Impressions', 'Avg Position', 'Referring Domains', 'Opportunity Type', 'Value Score'
        ])

    candidates = candidates.sort_values('Value Score', ascending=False)
    content_gap = thin_threshold - candidates['word_count']

    return pd.DataFrame({
        'URL': candidates['url'],
        'Page Type': candidates['page_type'],
        'Word Count': candidates['word_count'].astype(int),
        'Content Gap': content_gap.astype(int),
        'Sessions': candidates['sessions'].astype(int),
        'Impressions': candidates['impressions'].astype(int),
        'Avg Position': candidates['avg_position'].round(1),
        'Referring Domains': candidates['referring_domains'].astype(int),
        'Opportunity Type': candidates['Opportunity Type'],
        'Value Score': candidates['Value Score']
    })


# =============================================================================
# BUSINESS RELEVANCE SCORING
# =============================================================================

def calculate_business_relevance_score_api(
    topic_tokens: set,
    primary_query: str,
    site_context: dict = None
) -> tuple:
    """
    Calculate Business Relevance Score (0-100) for a topic cluster - API version.
    """
    if site_context is None:
        site_context = {}

    query_lower = primary_query.lower()
    notes = []
    exclusion_reason = ""

    # HARD EXCLUSION CHECKS
    for term in NAVIGATIONAL_TERMS_API:
        if term in query_lower:
            return (0, "", True, f"Navigational intent: '{term}'")

    for term in POLICY_LEGAL_TERMS_API:
        if term in query_lower:
            return (0, "", True, f"Policy/legal page: '{term}'")

    for term in JUNK_ARTIFACT_TERMS_API:
        if term in query_lower:
            return (0, "", True, f"Junk/artifact term: '{term}'")

    for term in AUTHORITY_REQUIRED_TERMS_API:
        if term in query_lower:
            return (0, "", True, f"Requires professional authority: '{term}'")

    clean_query = ''.join(c for c in query_lower if c.isalnum() or c.isspace())
    if len(clean_query.strip()) < 5:
        return (0, "", True, "Query too short/malformed")

    if clean_query.replace(' ', '').isdigit():
        return (0, "", True, "Query is only numbers")

    # COMPONENT 1: Site Theme Alignment (40%)
    theme_score = 0
    max_theme_score = 40

    existing_urls = set(str(u).lower() for u in site_context.get('existing_urls', []))
    existing_titles = set(str(t).lower() for t in site_context.get('existing_titles', []))
    top_queries = set(str(q).lower() for q in site_context.get('top_queries', []))
    site_categories = set(str(c).lower() for c in site_context.get('site_categories', []))

    all_site_text = ' '.join(existing_urls) + ' ' + ' '.join(existing_titles) + ' ' + ' '.join(top_queries)
    site_tokens = set(all_site_text.split())

    if topic_tokens and site_tokens:
        overlap = len(topic_tokens & site_tokens)
        overlap_ratio = overlap / len(topic_tokens) if topic_tokens else 0

        if overlap_ratio >= 0.6:
            theme_score = max_theme_score
        elif overlap_ratio >= 0.4:
            theme_score = int(max_theme_score * 0.75)
        elif overlap_ratio >= 0.2:
            theme_score = int(max_theme_score * 0.5)
        elif overlap_ratio > 0:
            theme_score = int(max_theme_score * 0.25)
        else:
            theme_score = 0
            notes.append("No site theme overlap")

    for category in site_categories:
        if category in query_lower:
            theme_score = min(max_theme_score, theme_score + 10)
            break

    # COMPONENT 2: Commercial/Strategic Intent (30%)
    intent_score = 0
    max_intent_score = 30

    transactional_matches = sum(1 for term in TRANSACTIONAL_MODIFIERS_API if term in query_lower)
    commercial_matches = sum(1 for term in COMMERCIAL_MODIFIERS_API if term in query_lower)
    informational_matches = sum(1 for term in INFORMATIONAL_MODIFIERS_API if term in query_lower)

    if transactional_matches > 0:
        intent_score = max_intent_score
    elif commercial_matches > 0:
        intent_score = int(max_intent_score * 0.7)
    elif informational_matches > 0:
        if theme_score >= max_theme_score * 0.5:
            intent_score = int(max_intent_score * 0.4)
        else:
            intent_score = int(max_intent_score * 0.15)
            notes.append("Informational intent with weak commercial tie-in")
    else:
        intent_score = int(max_intent_score * 0.3)

    # COMPONENT 3: Content Feasibility (20%)
    feasibility_score = 20
    questionable_expertise = {
        'diy', 'homemade', 'home remedy', 'self-diagnose', 'without professional',
        'instead of doctor', 'instead of lawyer', 'free legal', 'free medical'
    }

    for term in questionable_expertise:
        if term in query_lower:
            feasibility_score = int(feasibility_score * 0.5)
            notes.append(f"Questionable expertise required: '{term}'")
            break

    # COMPONENT 4: Brand Safety / Nonsense Filter (10%)
    safety_score = 10

    if len(query_lower.split()) > 8:
        safety_score = int(safety_score * 0.7)
        notes.append("Query unusually long")

    word_list = query_lower.split()
    if len(word_list) > len(set(word_list)) + 2:
        safety_score = int(safety_score * 0.5)
        notes.append("Repeated words in query")

    # CALCULATE FINAL SCORE
    total_score = theme_score + intent_score + feasibility_score + safety_score
    total_score = max(0, min(100, total_score))

    if total_score < 50:
        exclusion_reason = f"Low business relevance score ({total_score}/100)"
        return (total_score, '; '.join(notes) if notes else '', True, exclusion_reason)

    return (total_score, '; '.join(notes) if notes else '', False, '')


# =============================================================================
# SITE CONTEXT BUILDER
# =============================================================================

def build_site_context_from_df_api(df: pd.DataFrame, gsc_df: pd.DataFrame = None) -> dict:
    """
    Build site context dictionary from crawl and GSC data - API version.
    """
    context = {
        'existing_urls': [],
        'existing_titles': [],
        'top_queries': [],
        'site_categories': [],
        'brand_terms': []
    }

    if df is not None and len(df) > 0:
        if 'url' in df.columns:
            context['existing_urls'] = df['url'].dropna().tolist()

        if 'page_title' in df.columns:
            context['existing_titles'] = df['page_title'].dropna().tolist()

        if 'page_type' in df.columns:
            categories = df['page_type'].dropna().unique().tolist()
            for cat in categories:
                context['site_categories'].extend(str(cat).lower().split())
            context['site_categories'] = list(set(context['site_categories']))

    if gsc_df is not None and len(gsc_df) > 0:
        query_col = 'query' if 'query' in gsc_df.columns else 'primary_keyword'
        if query_col in gsc_df.columns and 'impressions' in gsc_df.columns:
            top_queries_df = gsc_df.nlargest(100, 'impressions')
            context['top_queries'] = top_queries_df[query_col].dropna().tolist()
        elif query_col in gsc_df.columns:
            context['top_queries'] = gsc_df[query_col].dropna().head(100).tolist()

    if context['existing_urls']:
        try:
            sample_url = context['existing_urls'][0]
            parsed = urlparse(sample_url)
            domain_parts = parsed.netloc.replace('www.', '').split('.')
            if domain_parts:
                context['brand_terms'] = [domain_parts[0]]
        except:
            pass

    return context


# =============================================================================
# BUSINESS RELEVANCE FILTERING
# =============================================================================

def filter_topics_by_business_relevance_api(
    topics_df: pd.DataFrame,
    site_context: dict = None,
    min_score: int = 50,
    max_topics: int = 15
) -> tuple:
    """
    Filter and score topics by business relevance - API version.
    """
    if topics_df is None or len(topics_df) == 0:
        return topics_df, []

    if site_context is None:
        site_context = {}

    relevance_scores = []
    feasibility_notes = []
    excluded_topics = []

    for _, row in topics_df.iterrows():
        primary_query = str(row.get('Primary Keyword', '')).strip()
        secondary_keywords = str(row.get('Secondary Keywords', '')).strip()

        all_keywords = primary_query + ' ' + secondary_keywords
        tokens = set(all_keywords.lower().split())

        score, notes, excluded, reason = calculate_business_relevance_score_api(
            topic_tokens=tokens,
            primary_query=primary_query,
            site_context=site_context
        )

        if excluded:
            excluded_topics.append({
                'Topic': row.get('Suggested Topic', primary_query),
                'Primary Keyword': primary_query,
                'Score': score,
                'Reason': reason
            })
            relevance_scores.append(0)
            feasibility_notes.append(reason)
        else:
            relevance_scores.append(score)
            feasibility_notes.append(notes if notes else '')

    topics_df = topics_df.copy()
    topics_df['Business Relevance Score'] = relevance_scores
    topics_df['Feasibility Notes'] = feasibility_notes

    mask = topics_df['Business Relevance Score'] >= min_score
    filtered_df = topics_df[mask].copy()

    if len(filtered_df) > 0:
        filtered_df = filtered_df.sort_values(
            ['Priority Score', 'Business Relevance Score'],
            ascending=[False, False]
        )
        filtered_df = filtered_df.head(max_topics)

    logger.info(f"[BUSINESS RELEVANCE] Topics before filtering: {len(topics_df)}")
    logger.info(f"[BUSINESS RELEVANCE] Topics after filtering: {len(filtered_df)}")
    logger.info(f"[BUSINESS RELEVANCE] Topics excluded: {len(excluded_topics)}")

    if excluded_topics:
        exclusion_reasons = {}
        for ex in excluded_topics:
            reason = ex['Reason'].split(':')[0] if ':' in ex['Reason'] else ex['Reason']
            exclusion_reasons[reason] = exclusion_reasons.get(reason, 0) + 1
        logger.info(f"[BUSINESS RELEVANCE] Exclusion breakdown: {exclusion_reasons}")

    return filtered_df, excluded_topics


# =============================================================================
# QUERY CLUSTERING INTO TOPICS
# =============================================================================

def cluster_queries_into_topics_api(queries_df: pd.DataFrame, max_topics: int = 20, thin_urls: set = None, url_word_counts: dict = None) -> pd.DataFrame:
    """
    Cluster similar queries into topic groups for new content opportunities.
    API version - uses word-overlap algorithm with Content Action Recommendations.

    Implements Create/Expand/Consolidate decision logic:
    - Create new page: No dominant page, content gap exists
    - Expand existing page: One page ranks but needs depth/optimization
    - Consolidate pages: Multiple pages cannibalize each other
    """
    if queries_df is None or len(queries_df) == 0:
        return pd.DataFrame()

    if thin_urls is None:
        thin_urls = set()

    if url_word_counts is None:
        url_word_counts = {}

    def tokenize(query):
        query = str(query).lower().strip()
        stopwords = {'a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'with', 'as', 'by', 'from', 'how', 'when', 'where', 'why', 'if', 'then', 'so', 'than', 'too', 'very', 'just', 'only', 'also', 'even', 'more', 'most', 'other', 'into', 'over', 'after', 'before', 'between', 'through', 'during', 'under', 'around', 'about', 'near'}
        words = re.findall(r'\b[a-z]{2,}\b', query)
        return [w for w in words if w not in stopwords]

    query_data = []
    for _, row in queries_df.iterrows():
        query = str(row.get('query', row.get('primary_keyword', ''))).strip()
        if not query or query == 'nan':
            continue
        tokens = tokenize(query)
        if tokens:
            query_data.append({
                'query': query, 'tokens': set(tokens),
                'impressions': int(row.get('impressions', 0)),
                'clicks': int(row.get('clicks', 0)),
                'avg_position': float(row.get('avg_position', 0)),
                'url': str(row.get('url', ''))
            })

    if not query_data:
        return pd.DataFrame()

    query_data.sort(key=lambda x: x['impressions'], reverse=True)

    clusters = []
    used_queries = set()

    for seed in query_data:
        if seed['query'] in used_queries:
            continue

        cluster = {
            'primary_query': seed['query'], 'queries': [seed],
            'all_tokens': seed['tokens'].copy(),
            'total_impressions': seed['impressions'], 'total_clicks': seed['clicks'],
            'positions': [seed['avg_position']] if seed['avg_position'] > 0 else [],
            'urls': defaultdict(lambda: {'impressions': 0, 'positions': [], 'clicks': 0})
        }
        # Track URL-level metrics for cannibalization detection
        if seed['url']:
            cluster['urls'][seed['url']]['impressions'] += seed['impressions']
            cluster['urls'][seed['url']]['clicks'] += seed['clicks']
            if seed['avg_position'] > 0:
                cluster['urls'][seed['url']]['positions'].append(seed['avg_position'])
        used_queries.add(seed['query'])

        for candidate in query_data:
            if candidate['query'] in used_queries:
                continue
            overlap = len(seed['tokens'] & candidate['tokens'])
            min_len = min(len(seed['tokens']), len(candidate['tokens']))
            if min_len > 0 and overlap / min_len >= 0.5:
                cluster['queries'].append(candidate)
                cluster['all_tokens'].update(candidate['tokens'])
                cluster['total_impressions'] += candidate['impressions']
                cluster['total_clicks'] += candidate['clicks']
                if candidate['avg_position'] > 0:
                    cluster['positions'].append(candidate['avg_position'])
                # Track URL-level metrics
                if candidate['url']:
                    cluster['urls'][candidate['url']]['impressions'] += candidate['impressions']
                    cluster['urls'][candidate['url']]['clicks'] += candidate['clicks']
                    if candidate['avg_position'] > 0:
                        cluster['urls'][candidate['url']]['positions'].append(candidate['avg_position'])
                used_queries.add(candidate['query'])

        clusters.append(cluster)

    topic_opportunities = []
    for cluster in clusters:
        if len(cluster['queries']) == 1 and cluster['total_impressions'] < 100:
            continue

        if cluster['positions']:
            weighted_pos = sum(q['avg_position'] * q['impressions'] for q in cluster['queries'] if q['avg_position'] > 0)
            total_weight = sum(q['impressions'] for q in cluster['queries'] if q['avg_position'] > 0)
            avg_position = weighted_pos / total_weight if total_weight > 0 else 0
        else:
            avg_position = 0

        # =====================================================================
        # URL-LEVEL METRICS ANALYSIS
        # =====================================================================
        url_metrics = cluster['urls']
        num_urls = len(url_metrics)

        # Calculate per-URL metrics with CTR
        url_analysis = []
        for url, metrics in url_metrics.items():
            url_impressions = metrics['impressions']
            url_clicks = metrics['clicks']
            url_positions = metrics['positions']
            url_avg_pos = sum(url_positions) / len(url_positions) if url_positions else 0
            url_ctr = (url_clicks / url_impressions * 100) if url_impressions > 0 else 0
            url_share = url_impressions / cluster['total_impressions'] if cluster['total_impressions'] > 0 else 0
            url_word_count = url_word_counts.get(url, 0)
            is_thin = url in thin_urls or url_word_count < 1000

            url_analysis.append({
                'url': url, 'impressions': url_impressions, 'clicks': url_clicks,
                'avg_position': url_avg_pos, 'ctr': url_ctr, 'share': url_share,
                'word_count': url_word_count, 'is_thin': is_thin
            })

        # Sort by impressions (strongest first)
        url_analysis.sort(key=lambda x: x['impressions'], reverse=True)

        # Identify dominant page (if any)
        dominant_url = None
        dominant_data = None
        if url_analysis:
            top_url = url_analysis[0]
            if top_url['share'] >= 0.60 and top_url['avg_position'] <= 15:
                if not top_url['is_thin']:
                    dominant_url = top_url['url']
                    dominant_data = top_url

        # =====================================================================
        # CONTENT ACTION RECOMMENDATION LOGIC
        # =====================================================================
        recommended_action = ""
        primary_url = ""
        secondary_urls = ""
        reasoning = ""

        # Track clusters with dominant pages for debugging (but DO NOT exclude them)
        if dominant_url and dominant_data:
            logger.debug(f"[CLUSTER DEBUG] Topic '{cluster['primary_query']}' has dominant page {dominant_url} "
                        f"({dominant_data['share']:.0%} impressions at position {dominant_data['avg_position']:.1f}) - evaluating for action")
            # NOTE: Previously this would `continue` and exclude the cluster.
            # Now we flow through to assign an appropriate action (usually Expand existing page)

        # DECISION LOGIC
        if num_urls == 0:
            recommended_action = "Create new page"
            primary_url = ""
            secondary_urls = ""
            reasoning = f"Search demand exists ({cluster['total_impressions']:,} impressions) but no landing page currently serves these queries. This is a true content gap."

        elif num_urls == 1:
            single_url = url_analysis[0]

            if single_url['avg_position'] > 20:
                recommended_action = "Create new page"
                primary_url = ""
                secondary_urls = ""
                reasoning = f"Existing page ({single_url['url'][:60]}...) ranks poorly at position {single_url['avg_position']:.0f}. A dedicated, well-optimized page would better serve this topic."

            elif single_url['is_thin']:
                recommended_action = "Expand existing page"
                primary_url = single_url['url']
                secondary_urls = ""
                reasoning = f"Page exists but has thin content ({single_url['word_count']} words). Expanding with comprehensive content at position {single_url['avg_position']:.0f} can capture more traffic."

            elif single_url['ctr'] < 2.0 and single_url['impressions'] >= 100:
                recommended_action = "Expand existing page"
                primary_url = single_url['url']
                secondary_urls = ""
                reasoning = f"Page ranks at position {single_url['avg_position']:.0f} with high impressions ({single_url['impressions']:,}) but very low CTR ({single_url['ctr']:.1f}%). Improve content depth and meta data to boost clicks."

            elif single_url['avg_position'] > 10:
                recommended_action = "Expand existing page"
                primary_url = single_url['url']
                secondary_urls = ""
                reasoning = f"Page ranks at position {single_url['avg_position']:.0f} but not in top 10. Expanding content depth can improve rankings and capture more traffic."

            else:
                recommended_action = "Expand existing page"
                primary_url = single_url['url']
                secondary_urls = ""
                reasoning = f"Page ranks at position {single_url['avg_position']:.0f} but doesn't fully capture the topic. Expanding with related subtopics can increase topical authority."

        elif num_urls == 2:
            url1, url2 = url_analysis[0], url_analysis[1]

            if url1['share'] < 0.65 and url2['share'] >= 0.10:
                recommended_action = "Consolidate pages"
                primary_url = url1['url']
                secondary_urls = url2['url']
                reasoning = f"Two pages are competing for the same keywords, diluting rankings. {url1['url'][:50]}... has {url1['share']:.0%} impressions at position {url1['avg_position']:.0f}; {url2['url'][:50]}... has {url2['share']:.0%}. Consolidate into one authoritative page."
            else:
                recommended_action = "Expand existing page"
                primary_url = url1['url']
                secondary_urls = ""
                reasoning = f"One page ({url1['url'][:50]}...) receives most impressions ({url1['share']:.0%}) at position {url1['avg_position']:.0f}. Expand this page to fully own the topic."

        else:
            # 3+ URLs - Consolidation needed
            top_urls = url_analysis[:3]
            meaningful_urls = [u for u in top_urls if u['share'] >= 0.08]

            if len(meaningful_urls) >= 2:
                recommended_action = "Consolidate pages"
                primary_url = meaningful_urls[0]['url']
                secondary_urls = ', '.join([u['url'] for u in meaningful_urls[1:]])
                url_summary = '; '.join([f"{u['url'][:40]}... ({u['share']:.0%})" for u in meaningful_urls])
                reasoning = f"Traffic is fragmented across {num_urls} pages causing keyword cannibalization. Top competing pages: {url_summary}. Consolidate into one authoritative hub."
            elif url_analysis[0]['avg_position'] > 20:
                recommended_action = "Create new page"
                primary_url = ""
                secondary_urls = ""
                reasoning = f"Traffic is split across {num_urls} weak pages (best position: {url_analysis[0]['avg_position']:.0f}). Creating a dedicated, comprehensive page would better serve this topic than consolidating poor performers."
            else:
                recommended_action = "Expand existing page"
                primary_url = url_analysis[0]['url']
                secondary_urls = ""
                reasoning = f"Multiple pages exist but {url_analysis[0]['url'][:50]}... leads with {url_analysis[0]['share']:.0%} impressions. Expand this page to consolidate authority."

        # Generate suggested topic and page type
        primary_query = cluster['primary_query']
        common_tokens = sorted(cluster['all_tokens'], key=lambda t: sum(1 for q in cluster['queries'] if t in q['tokens']), reverse=True)
        suggested_topic = ' '.join(common_tokens[:4]).title() if common_tokens else primary_query.title()

        query_lower = primary_query.lower()
        intent_value = 0.5
        if any(word in query_lower for word in ['how to', 'guide', 'tutorial', 'tips', 'steps']):
            page_type = 'Guide / How-To'
            intent_value = 0.7
        elif any(word in query_lower for word in ['best', 'top', 'review', 'compare', 'vs']):
            page_type = 'Comparison / Review'
            intent_value = 0.9
        elif any(word in query_lower for word in ['what is', 'meaning', 'definition', 'explain']):
            page_type = 'Educational / Explainer'
            intent_value = 0.6
        elif any(word in query_lower for word in ['near me', 'in ', 'local']):
            page_type = 'Local Landing Page'
            intent_value = 1.0
        elif any(word in query_lower for word in ['cost', 'price', 'pricing', 'quote', 'estimate']):
            page_type = 'Service / Pricing Page'
            intent_value = 1.0
        elif any(word in query_lower for word in ['buy', 'order', 'purchase', 'shop']):
            page_type = 'Product / Service Page'
            intent_value = 1.0
        else:
            page_type = 'Blog Post / Article'
            intent_value = 0.5

        secondary = [q['query'] for q in cluster['queries'] if q['query'] != primary_query][:8]

        # =====================================================================
        # PRIORITY SCORE CALCULATION
        # 40% Impressions, 25% Action urgency, 20% Position weakness, 15% Intent value
        # =====================================================================
        impressions_score = min(10, math.log10(max(1, cluster['total_impressions'])) * 2.5)

        # Action urgency score (25% weight)
        if recommended_action == "Create new page":
            action_urgency_score = 10 if num_urls == 0 else 7
        elif recommended_action == "Consolidate pages":
            action_urgency_score = 9  # High urgency - cannibalization hurts rankings
        else:  # Expand
            action_urgency_score = 5  # Lower urgency - existing page works somewhat

        # Position weakness score (20% weight)
        if avg_position == 0 or num_urls == 0:
            position_score = 10
        elif avg_position > 30:
            position_score = 9
        elif avg_position > 20:
            position_score = 7
        elif avg_position > 15:
            position_score = 5
        elif avg_position > 10:
            position_score = 3
        else:
            position_score = 1

        intent_score = intent_value * 10

        priority_score = round(
            (impressions_score * 0.40) +
            (action_urgency_score * 0.25) +
            (position_score * 0.20) +
            (intent_score * 0.15),
            1
        )

        topic_opportunities.append({
            'Suggested Topic': suggested_topic,
            'Primary Keyword': primary_query,
            'Secondary Keywords': ', '.join(secondary) if secondary else '',
            'Total Impressions': cluster['total_impressions'],
            'Avg Position': round(avg_position, 1) if avg_position > 0 else 'N/A',
            'Recommended Action': recommended_action,
            'Primary URL': primary_url,
            'Secondary URLs': secondary_urls,
            'Reasoning': reasoning,
            'Suggested Page Type': page_type,
            'Priority Score': priority_score
        })

    if not topic_opportunities:
        return pd.DataFrame()

    result = pd.DataFrame(topic_opportunities)
    return result.sort_values('Priority Score', ascending=False).head(max_topics)


# =============================================================================
# NEW CONTENT OPPORTUNITIES
# =============================================================================

def build_new_content_opportunities_api(df: pd.DataFrame, gsc_df: Optional[pd.DataFrame] = None, thin_threshold: int = 1000) -> pd.DataFrame:
    """
    Build New Content Opportunities sheet - API version.

    Returns TOPICS (not URLs) derived from query-level GSC data.
    This sheet answers: "What should we write that doesn't exist on the site yet?"

    Includes cannibalization detection to exclude topics where a strong page already exists.
    """
    # Define output columns (includes Business Relevance columns)
    empty_columns = [
        'Suggested Topic', 'Primary Keyword', 'Secondary Keywords', 'Total Impressions',
        'Avg Position', 'Recommended Action', 'Primary URL', 'Secondary URLs', 'Reasoning',
        'Suggested Page Type', 'Priority Score', 'Business Relevance Score', 'Feasibility Notes'
    ]

    if gsc_df is None or len(gsc_df) == 0:
        return pd.DataFrame(columns=empty_columns)

    # Build set of thin content URLs and url_word_counts dict for Content Action logic
    thin_urls = set()
    url_word_counts = {}
    if df is not None and len(df) > 0:
        word_count_col = None
        for col in ['word_count', 'Word Count', 'wordcount']:
            if col in df.columns:
                word_count_col = col
                break

        url_col = None
        for col in ['url', 'URL', 'Address']:
            if col in df.columns:
                url_col = col
                break

        if word_count_col and url_col:
            df_copy = df.copy()
            df_copy[word_count_col] = pd.to_numeric(df_copy[word_count_col], errors='coerce').fillna(0)
            thin_urls = set(df_copy[df_copy[word_count_col] < thin_threshold][url_col].tolist())
            url_word_counts = dict(zip(df_copy[url_col], df_copy[word_count_col]))
            logger.debug(f"Identified {len(thin_urls)} thin content URLs and {len(url_word_counts)} URL word counts")

    gsc_copy = gsc_df.copy()

    # Map column names if needed
    query_col = None
    for col in ['query', 'primary_keyword', 'Query', 'Keyword', 'Top queries']:
        if col in gsc_copy.columns:
            query_col = col
            break

    if query_col is None:
        if 'primary_keyword' in gsc_copy.columns:
            gsc_copy['query'] = gsc_copy['primary_keyword']
        else:
            logger.warning("GSC data missing query column - cannot build topic-based opportunities")
            return pd.DataFrame(columns=empty_columns)
    elif query_col != 'query':
        gsc_copy['query'] = gsc_copy[query_col]

    # Ensure URL column exists for cannibalization detection
    url_col = None
    for col in ['url', 'URL', 'page', 'Page', 'landing_page']:
        if col in gsc_copy.columns:
            url_col = col
            break

    if url_col and url_col != 'url':
        gsc_copy['url'] = gsc_copy[url_col]
    elif url_col is None:
        gsc_copy['url'] = ''  # No URL data available

    for col in ['impressions', 'clicks', 'avg_position']:
        if col not in gsc_copy.columns:
            gsc_copy[col] = 0
        gsc_copy[col] = pd.to_numeric(gsc_copy[col], errors='coerce').fillna(0)

    qualifying_queries = gsc_copy[gsc_copy['impressions'] >= 10].copy()

    if len(qualifying_queries) == 0:
        return pd.DataFrame(columns=empty_columns)

    result = cluster_queries_into_topics_api(qualifying_queries, max_topics=20, thin_urls=thin_urls, url_word_counts=url_word_counts)

    if len(result) == 0:
        return pd.DataFrame(columns=empty_columns)

    # =========================================================================
    # BUSINESS RELEVANCE FILTERING (Gate 3)
    # =========================================================================
    # Build site context from crawl and GSC data
    site_context = build_site_context_from_df_api(df, gsc_df)

    # Apply business relevance filtering
    filtered_result, excluded_topics = filter_topics_by_business_relevance_api(
        topics_df=result,
        site_context=site_context,
        min_score=50,  # Minimum score threshold
        max_topics=15  # Cap to 15 topics max
    )

    # Log excluded topics for debugging
    if excluded_topics:
        logger.debug(f"[NEW CONTENT] Excluded {len(excluded_topics)} topics by business relevance filter")
        for ex in excluded_topics[:5]:  # Log first 5
            logger.debug(f"  - Excluded: {ex.get('Primary Keyword', 'N/A')} | Reason: {ex.get('Reason', 'N/A')}")

    # Ensure all expected columns exist
    for col in empty_columns:
        if col not in filtered_result.columns:
            filtered_result[col] = ''

    # Reorder columns to match expected output
    return filtered_result[empty_columns]


# =============================================================================
# REDIRECT & MERGE PLAN
# =============================================================================

def build_redirect_merge_plan_api(new_content_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the 'Redirect & Merge Plan' sheet from consolidation recommendations - API version.

    Generates a 301 redirect mapping for every topic where the Recommended Action
    is "Consolidate pages". Each Secondary URL gets its own row mapping to the
    Primary URL.

    Args:
        new_content_df: DataFrame from build_new_content_opportunities_api
            containing Recommended Action, Primary URL, Secondary URLs columns

    Returns:
        DataFrame with columns: Topic, Recommended Action, Primary URL,
        Secondary URL, Redirect Type, Reason
    """
    empty_columns = [
        'Topic', 'Recommended Action', 'Primary URL', 'Secondary URL',
        'Redirect Type', 'Reason'
    ]

    if new_content_df is None or len(new_content_df) == 0:
        return pd.DataFrame(columns=empty_columns)

    # Filter for consolidation actions only
    consolidations = new_content_df[
        new_content_df['Recommended Action'] == 'Consolidate pages'
    ].copy()

    if len(consolidations) == 0:
        return pd.DataFrame(columns=empty_columns)

    redirect_rows = []

    for _, row in consolidations.iterrows():
        topic = row.get('Suggested Topic', '')
        primary_url = row.get('Primary URL', '')
        secondary_urls_str = row.get('Secondary URLs', '')

        # Skip if no primary URL or secondary URLs
        if not primary_url or not secondary_urls_str:
            continue

        # Parse secondary URLs (comma-separated)
        secondary_urls = [
            url.strip() for url in secondary_urls_str.split(',')
            if url.strip() and url.strip() != primary_url
        ]

        # Skip if no valid secondary URLs after filtering
        if not secondary_urls:
            continue

        # Generate redirect entries for each secondary URL
        for secondary_url in secondary_urls:
            # Generate redirect-specific reasoning
            reason = (
                f"This page competes for the same keywords as the primary page "
                f"({primary_url[:50]}...) but has weaker rankings and engagement. "
                f"Redirecting consolidates authority and prevents keyword cannibalization."
            )

            redirect_rows.append({
                'Topic': topic,
                'Recommended Action': 'Consolidate pages',
                'Primary URL': primary_url,
                'Secondary URL': secondary_url,
                'Redirect Type': '301',
                'Reason': reason
            })

    if not redirect_rows:
        return pd.DataFrame(columns=empty_columns)

    result_df = pd.DataFrame(redirect_rows)

    # Remove any duplicate redirects (same Secondary URL)
    result_df = result_df.drop_duplicates(subset=['Secondary URL'], keep='first')

    # Ensure no self-redirects (Primary URL == Secondary URL)
    result_df = result_df[result_df['Primary URL'] != result_df['Secondary URL']]

    logger.info(f"Generated {len(result_df)} redirect entries from {len(consolidations)} consolidation topics")

    return result_df


# =============================================================================
# MERGE PLAYBOOKS
# =============================================================================

def build_merge_playbooks_api(new_content_df: pd.DataFrame, gsc_df: Optional[pd.DataFrame] = None, url_word_counts: Optional[dict] = None) -> pd.DataFrame:
    """
    Build the 'Merge Playbooks' sheet with content-level merge instructions - API version.

    For each "Consolidate pages" action, generates detailed content recommendations
    on what to keep from the Primary URL and what to move from Secondary URLs.

    Args:
        new_content_df: DataFrame from build_new_content_opportunities_api
            containing Recommended Action, Primary URL, Secondary URLs, etc.
        gsc_df: Raw GSC query-level data for analyzing per-URL query coverage
        url_word_counts: Dict mapping URL -> word count (optional)

    Returns:
        DataFrame with columns: Topic, Primary URL, Secondary URL, Keep This Content,
        Move These Sections, Retire This Page, Reasoning
    """
    empty_columns = [
        'Topic', 'Primary URL', 'Secondary URL', 'Keep This Content',
        'Move These Sections', 'Retire This Page', 'Reasoning'
    ]

    if new_content_df is None or len(new_content_df) == 0:
        return pd.DataFrame(columns=empty_columns)

    # Filter for consolidation actions only
    consolidations = new_content_df[
        new_content_df['Recommended Action'] == 'Consolidate pages'
    ].copy()

    if len(consolidations) == 0:
        return pd.DataFrame(columns=empty_columns)

    if url_word_counts is None:
        url_word_counts = {}

    # Build URL -> queries mapping from GSC data if available
    url_queries = {}  # url -> list of {query, impressions, clicks, position}
    if gsc_df is not None and len(gsc_df) > 0:
        gsc_copy = gsc_df.copy()

        # Find query column
        query_col = None
        for col in ['query', 'primary_keyword', 'Query', 'Keyword', 'Top queries']:
            if col in gsc_copy.columns:
                query_col = col
                break

        # Find URL column
        url_col = None
        for col in ['url', 'URL', 'page', 'Page', 'landing_page']:
            if col in gsc_copy.columns:
                url_col = col
                break

        if query_col and url_col:
            for _, row in gsc_copy.iterrows():
                url = str(row.get(url_col, '')).strip()
                query = str(row.get(query_col, '')).strip()
                if url and query and query != 'nan':
                    if url not in url_queries:
                        url_queries[url] = []
                    url_queries[url].append({
                        'query': query,
                        'impressions': int(row.get('impressions', 0)),
                        'clicks': int(row.get('clicks', 0)),
                        'position': float(row.get('avg_position', row.get('position', 0)))
                    })

    playbook_rows = []

    for _, row in consolidations.iterrows():
        topic = row.get('Suggested Topic', '')
        primary_url = row.get('Primary URL', '')
        secondary_urls_str = row.get('Secondary URLs', '')

        # Skip if no primary URL or secondary URLs
        if not primary_url or not secondary_urls_str:
            continue

        # Parse secondary URLs (comma-separated)
        secondary_urls = [
            url.strip() for url in secondary_urls_str.split(',')
            if url.strip() and url.strip() != primary_url
        ]

        if not secondary_urls:
            continue

        # Get Primary URL's queries
        primary_queries = url_queries.get(primary_url, [])
        primary_query_set = {q['query'].lower() for q in primary_queries}
        primary_word_count = url_word_counts.get(primary_url, 0)

        # Calculate total impressions for Primary URL
        primary_total_impressions = sum(q['impressions'] for q in primary_queries)

        # Build "Keep This Content" for Primary URL
        keep_content_bullets = []

        if primary_queries:
            # Group by position ranges
            top_position_queries = [q for q in primary_queries if q['position'] > 0 and q['position'] <= 5]
            strong_position_queries = [q for q in primary_queries if q['position'] > 5 and q['position'] <= 10]

            # Check for queries ranking 1-5
            if top_position_queries:
                top_query_examples = sorted(top_position_queries, key=lambda x: x['impressions'], reverse=True)[:3]
                query_names = ', '.join([f"'{q['query']}'" for q in top_query_examples])
                keep_content_bullets.append(f"Top-ranking content (positions 1-5): {query_names}")

            # Check for high-impression query clusters
            high_impression_queries = [q for q in primary_queries if q['impressions'] >= 50]
            if high_impression_queries:
                total_high_imp = sum(q['impressions'] for q in high_impression_queries)
                if primary_total_impressions > 0:
                    share = total_high_imp / primary_total_impressions
                    if share >= 0.20:
                        keep_content_bullets.append(f"High-traffic sections ({share:.0%} of impressions)")

            # Check for strong position queries (6-10)
            if strong_position_queries:
                keep_content_bullets.append(f"Content ranking positions 6-10 ({len(strong_position_queries)} queries)")

        # Add word count context
        if primary_word_count >= 1000:
            keep_content_bullets.append(f"Comprehensive content ({primary_word_count:,} words)")
        elif primary_word_count > 0:
            keep_content_bullets.append(f"Existing content ({primary_word_count:,} words)")

        if not keep_content_bullets:
            keep_content_bullets.append("Primary page structure and core content")

        keep_content = '\n'.join([f"• {b}" for b in keep_content_bullets])

        # Generate one row per secondary URL
        for secondary_url in secondary_urls:
            # Get Secondary URL's queries
            secondary_queries = url_queries.get(secondary_url, [])
            secondary_word_count = url_word_counts.get(secondary_url, 0)

            # Build "Move These Sections" for Secondary URL
            move_sections_bullets = []

            if secondary_queries:
                # Find unique queries not covered by Primary
                unique_queries = [
                    q for q in secondary_queries
                    if q['query'].lower() not in primary_query_set
                ]

                # Find queries where Secondary ranks better (>3 positions better)
                better_ranking_queries = []
                for sq in secondary_queries:
                    sq_query_lower = sq['query'].lower()
                    for pq in primary_queries:
                        if pq['query'].lower() == sq_query_lower:
                            if pq['position'] > 0 and sq['position'] > 0:
                                if pq['position'] - sq['position'] > 3:
                                    better_ranking_queries.append(sq)
                            break

                # Add unique query coverage
                if unique_queries:
                    unique_examples = sorted(unique_queries, key=lambda x: x['impressions'], reverse=True)[:3]
                    unique_names = ', '.join([f"'{q['query']}'" for q in unique_examples])
                    move_sections_bullets.append(f"Unique content not on Primary: {unique_names}")

                # Add better-ranking content
                if better_ranking_queries:
                    better_examples = sorted(better_ranking_queries, key=lambda x: x['impressions'], reverse=True)[:2]
                    better_names = ', '.join([f"'{q['query']}'" for q in better_examples])
                    move_sections_bullets.append(f"Content ranking better than Primary: {better_names}")

                # Check for high-value queries on Secondary
                high_value_secondary = [q for q in secondary_queries if q['impressions'] >= 30]
                if high_value_secondary and not move_sections_bullets:
                    hv_examples = sorted(high_value_secondary, key=lambda x: x['impressions'], reverse=True)[:2]
                    hv_names = ', '.join([f"'{q['query']}'" for q in hv_examples])
                    move_sections_bullets.append(f"High-impression queries: {hv_names}")

            # Check if Secondary has more content depth
            if secondary_word_count > primary_word_count and secondary_word_count > 500:
                diff = secondary_word_count - primary_word_count
                move_sections_bullets.append(f"Additional content depth ({diff:,} more words) - review for valuable sections")

            if not move_sections_bullets:
                move_sections_bullets.append("Review page for any unique angles or examples to preserve")

            move_sections = '\n'.join([f"• {b}" for b in move_sections_bullets])

            # Generate reasoning for this specific merge
            reasoning = (
                f"Merging '{secondary_url}' into '{primary_url}' eliminates keyword cannibalization "
                f"for the '{topic}' topic. Primary URL has stronger overall authority; "
                f"Secondary URL's unique content should be integrated before redirect."
            )

            playbook_rows.append({
                'Topic': topic,
                'Primary URL': primary_url,
                'Secondary URL': secondary_url,
                'Keep This Content': keep_content,
                'Move These Sections': move_sections,
                'Retire This Page': 'Yes',
                'Reasoning': reasoning
            })

    if not playbook_rows:
        return pd.DataFrame(columns=empty_columns)

    result_df = pd.DataFrame(playbook_rows)

    logger.info(f"Generated {len(result_df)} merge playbook entries from {len(consolidations)} consolidation topics")

    return result_df


# =============================================================================
# ORCHESTRATOR — build all analytical sheets in one call
# =============================================================================

def build_all_analytical_sheets(
    df: pd.DataFrame,
    gsc_df: Optional[pd.DataFrame] = None,
    thin_threshold: int = 1000,
) -> Dict[str, pd.DataFrame]:
    """Build all analytical insight sheets from a merged WQA DataFrame.

    This is the entry point used by supabase_adapter.run_full_audit().

    Args:
        df: Merged WQA DataFrame (crawl + GA4 + GSC + backlinks).
        gsc_df: Optional raw GSC query-level DataFrame for topic clustering.
        thin_threshold: Word count threshold for thin content detection.

    Returns:
        Dict mapping sheet name -> DataFrame, suitable for passing to
        create_excel_report() as the ``summaries`` parameter.
    """
    sheets: Dict[str, pd.DataFrame] = {}

    try:
        sheets['Content to Optimize'] = build_content_to_optimize_api(df)
    except Exception as e:
        logger.error(f"Failed to build Content to Optimize: {e}")

    try:
        sheets['Thin Content'] = build_thin_content_api(df, thin_threshold=thin_threshold)
    except Exception as e:
        logger.error(f"Failed to build Thin Content: {e}")

    try:
        new_content = build_new_content_opportunities_api(df, gsc_df=gsc_df, thin_threshold=thin_threshold)
        sheets['New Content Opportunities'] = new_content

        # Redirect & Merge Plan depends on New Content Opportunities
        if not new_content.empty:
            try:
                sheets['Redirect & Merge Plan'] = build_redirect_merge_plan_api(new_content)
            except Exception as e:
                logger.error(f"Failed to build Redirect & Merge Plan: {e}")

            # Merge Playbooks also depends on New Content Opportunities
            try:
                url_word_counts = {}
                if 'word_count' in df.columns and 'url' in df.columns:
                    url_word_counts = dict(zip(df['url'], df['word_count'].fillna(0).astype(int)))
                sheets['Merge Playbooks'] = build_merge_playbooks_api(
                    new_content, gsc_df=gsc_df, url_word_counts=url_word_counts,
                )
            except Exception as e:
                logger.error(f"Failed to build Merge Playbooks: {e}")

    except Exception as e:
        logger.error(f"Failed to build New Content Opportunities: {e}")

    logger.info(f"Built {len(sheets)} analytical sheets: {list(sheets.keys())}")
    return sheets
