"""
Data quality validation for Website Quality Audit (WQA) reports.

Contains the DataQualityReport class and validation/reporting functions.
"""

import logging
from typing import Dict, List, Optional

import pandas as pd

from .normalization import normalize_url, normalize_page_path

logger = logging.getLogger(__name__)


class DataQualityReport:
    """Container for data quality validation results."""

    def __init__(self):
        self.missing_columns: Dict[str, List[str]] = {}
        self.empty_columns: List[str] = []
        self.sparse_columns: Dict[str, float] = {}
        self.url_match_stats: Dict[str, Dict] = {}
        self.data_source_coverage: Dict[str, int] = {}
        self.issues: List[Dict] = []
        self.warnings: List[str] = []

    def add_issue(self, category: str, severity: str, description: str,
                  affected_count: int = 0, examples: List[str] = None):
        """Add a specific data quality issue."""
        self.issues.append({
            'category': category,
            'severity': severity,
            'description': description,
            'affected_count': affected_count,
            'examples': examples or []
        })

    def to_dataframe(self) -> pd.DataFrame:
        """Convert issues to a DataFrame for Excel export."""
        if not self.issues:
            return pd.DataFrame(columns=['Category', 'Severity', 'Description', 'Affected Count', 'Examples'])

        rows = []
        for issue in self.issues:
            rows.append({
                'Category': issue['category'],
                'Severity': issue['severity'],
                'Description': issue['description'],
                'Affected Count': issue['affected_count'],
                'Examples': '; '.join(str(e) for e in issue['examples'][:5]) if issue['examples'] else ''
            })

        return pd.DataFrame(rows)


def validate_data_quality(
    merged_df: pd.DataFrame,
    crawl_df: pd.DataFrame,
    ga_df: Optional[pd.DataFrame],
    gsc_df: Optional[pd.DataFrame],
    backlink_df: Optional[pd.DataFrame]
) -> DataQualityReport:
    """
    Validate merged data quality and identify missing/problematic data.
    """
    report = DataQualityReport()
    total_urls = len(merged_df)

    logger.info("=" * 60)
    logger.info("DATA QUALITY VALIDATION")
    logger.info("=" * 60)

    crawl_urls = set(merged_df['url'].dropna().unique())

    # GA4 URL matching
    if ga_df is not None:
        if 'page_path' in ga_df.columns:
            # API uses page_path for GA joining
            ga_paths = set(ga_df['page_path'].apply(normalize_page_path).dropna().unique())
            ga_paths = ga_paths - {''}
            crawl_paths = set(merged_df['page_path'].dropna().unique()) if 'page_path' in merged_df.columns else set()
            matched_ga = crawl_paths & ga_paths
            unmatched_ga = ga_paths - crawl_paths
            total_ga = len(ga_paths)
        elif 'url' in ga_df.columns:
            ga_urls = set(ga_df['url'].apply(normalize_url).dropna().unique())
            ga_urls = ga_urls - {''}
            matched_ga = crawl_urls & ga_urls
            unmatched_ga = ga_urls - crawl_urls
            total_ga = len(ga_urls)
        else:
            matched_ga = set()
            unmatched_ga = set()
            total_ga = 0

        report.data_source_coverage['GA4'] = len(matched_ga)
        report.url_match_stats['GA4'] = {
            'total_source_urls': total_ga,
            'matched': len(matched_ga),
            'unmatched': len(unmatched_ga),
            'match_rate': len(matched_ga) / total_ga * 100 if total_ga else 0
        }

        logger.info(f"GA4: {len(matched_ga)}/{total_ga} URLs matched ({report.url_match_stats['GA4']['match_rate']:.1f}%)")

        if report.url_match_stats['GA4']['match_rate'] < 50 and total_ga > 0:
            report.add_issue(
                category='URL Matching',
                severity='High',
                description=f'GA4 URL match rate is low ({report.url_match_stats["GA4"]["match_rate"]:.1f}%). Check URL format differences.',
                affected_count=len(unmatched_ga),
                examples=list(unmatched_ga)[:10]
            )
    else:
        report.data_source_coverage['GA4'] = 0
        report.warnings.append('GA4 data not provided - sessions/conversions will be 0')

    # GSC URL matching
    if gsc_df is not None and 'url' in gsc_df.columns:
        gsc_urls = set(gsc_df['url'].apply(normalize_url).dropna().unique())
        gsc_urls = gsc_urls - {''}
        matched_gsc = crawl_urls & gsc_urls
        unmatched_gsc = gsc_urls - crawl_urls

        report.data_source_coverage['GSC'] = len(matched_gsc)
        report.url_match_stats['GSC'] = {
            'total_source_urls': len(gsc_urls),
            'matched': len(matched_gsc),
            'unmatched': len(unmatched_gsc),
            'match_rate': len(matched_gsc) / len(gsc_urls) * 100 if gsc_urls else 0
        }

        logger.info(f"GSC: {len(matched_gsc)}/{len(gsc_urls)} URLs matched ({report.url_match_stats['GSC']['match_rate']:.1f}%)")

        if report.url_match_stats['GSC']['match_rate'] < 50:
            report.add_issue(
                category='URL Matching',
                severity='High',
                description=f'GSC URL match rate is low ({report.url_match_stats["GSC"]["match_rate"]:.1f}%). Check URL format differences.',
                affected_count=len(unmatched_gsc),
                examples=list(unmatched_gsc)[:10]
            )
    else:
        report.data_source_coverage['GSC'] = 0
        report.warnings.append('GSC data not provided - rankings/CTR will be 0')

    # Backlink URL matching
    if backlink_df is not None and 'url' in backlink_df.columns:
        bl_urls = set(backlink_df['url'].apply(normalize_url).dropna().unique())
        bl_urls = bl_urls - {''}
        matched_bl = crawl_urls & bl_urls
        unmatched_bl = bl_urls - crawl_urls

        report.data_source_coverage['Backlinks'] = len(matched_bl)
        report.url_match_stats['Backlinks'] = {
            'total_source_urls': len(bl_urls),
            'matched': len(matched_bl),
            'unmatched': len(unmatched_bl),
            'match_rate': len(matched_bl) / len(bl_urls) * 100 if bl_urls else 0
        }

        logger.info(f"Backlinks: {len(matched_bl)}/{len(bl_urls)} URLs matched ({report.url_match_stats['Backlinks']['match_rate']:.1f}%)")

        if report.url_match_stats['Backlinks']['match_rate'] < 50:
            report.add_issue(
                category='URL Matching',
                severity='High',
                description=f'Backlink URL match rate is low ({report.url_match_stats["Backlinks"]["match_rate"]:.1f}%). Check URL format differences.',
                affected_count=len(unmatched_bl),
                examples=list(unmatched_bl)[:10]
            )
    else:
        report.data_source_coverage['Backlinks'] = 0
        report.warnings.append('Backlink data not provided - referring_domains/backlinks will be 0')

    # Check for empty/sparse columns
    critical_columns = {
        'sessions': ('GA4', 0),
        'conversions': ('GA4', 0),
        'avg_position': ('GSC', 0.0),
        'ctr': ('GSC', 0.0),
        'clicks': ('GSC', 0),
        'impressions': ('GSC', 0),
        'primary_keyword': ('GSC', ''),
        'referring_domains': ('Backlinks', 0),
        'backlinks': ('Backlinks', 0),
        'word_count': ('Crawl', 0),
        'page_title': ('Crawl', ''),
        'meta_description': ('Crawl', ''),
    }

    for col, (source, default) in critical_columns.items():
        if col not in merged_df.columns:
            report.empty_columns.append(col)
            continue

        if isinstance(default, str):
            empty_mask = (merged_df[col].isna()) | (merged_df[col] == '') | (merged_df[col] == default)
        else:
            empty_mask = (merged_df[col].isna()) | (merged_df[col] == default)

        empty_count = empty_mask.sum()
        empty_pct = empty_count / total_urls * 100 if total_urls > 0 else 0
        report.sparse_columns[col] = empty_pct

        if empty_pct == 100:
            report.empty_columns.append(col)
            report.add_issue(
                category='Empty Column',
                severity='High',
                description=f'Column "{col}" has no data (100% empty/default). Source: {source}',
                affected_count=empty_count
            )
        elif empty_pct >= 90:
            report.add_issue(
                category='Sparse Data',
                severity='High',
                description=f'Column "{col}" is {empty_pct:.1f}% empty/default. Source: {source}',
                affected_count=empty_count
            )

    # Check data inconsistencies
    if 'sessions' in merged_df.columns and 'avg_position' in merged_df.columns:
        traffic_no_rank = merged_df[(merged_df['sessions'] > 0) & (merged_df['avg_position'] == 0)]
        if len(traffic_no_rank) > 0:
            report.add_issue(
                category='Data Inconsistency',
                severity='Medium',
                description=f'URLs with traffic but no GSC ranking data.',
                affected_count=len(traffic_no_rank),
                examples=list(traffic_no_rank['url'].head(10))
            )

    high_issues = sum(1 for i in report.issues if i['severity'] == 'High')
    medium_issues = sum(1 for i in report.issues if i['severity'] == 'Medium')

    logger.info(f"Data Quality: {len(report.issues)} issues ({high_issues} High, {medium_issues} Medium)")
    logger.info("=" * 60)

    return report


def generate_data_quality_sheets(report: DataQualityReport) -> Dict[str, pd.DataFrame]:
    """Generate DataFrames for data quality reporting in Excel."""
    sheets = {}

    sheets['issues'] = report.to_dataframe()

    coverage_data = []
    for source, count in report.data_source_coverage.items():
        stats = report.url_match_stats.get(source, {})
        coverage_data.append({
            'Data Source': source,
            'URLs Matched': count,
            'Total Source URLs': stats.get('total_source_urls', 'N/A'),
            'Match Rate (%)': f"{stats.get('match_rate', 0):.1f}" if stats else 'N/A',
            'Unmatched URLs': stats.get('unmatched', 0)
        })
    sheets['coverage'] = pd.DataFrame(coverage_data)

    sparsity_data = []
    for col, pct in sorted(report.sparse_columns.items(), key=lambda x: -x[1]):
        sparsity_data.append({
            'Column': col,
            'Empty/Default (%)': f"{pct:.1f}",
            'Status': 'Critical' if pct >= 90 else ('Warning' if pct >= 70 else 'OK')
        })
    sheets['column_quality'] = pd.DataFrame(sparsity_data)

    return sheets
