"""WQA Supabase Adapter — loads data from Supabase, writes results + uploads Excel."""

import io
import os
import logging
from typing import Optional
from datetime import datetime

import pandas as pd
from supabase import create_client, Client

logger = logging.getLogger(__name__)


class WQASupabaseEngine:
    """Orchestrates WQA analysis using Supabase as data source and output target."""

    def __init__(self, supabase_url: str = "", supabase_key: str = ""):
        self.url = supabase_url or os.environ.get("NEXT_PUBLIC_SUPABASE_URL", "")
        self.key = supabase_key or os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
        self.client: Client = create_client(self.url, self.key)

    def load_crawl_data(self, client_id: str, crawl_run_id: Optional[str] = None) -> pd.DataFrame:
        """Load crawl data from Supabase for a client."""
        query = self.client.table("crawl_data").select("*").eq("client_id", client_id)
        if crawl_run_id:
            query = query.eq("crawl_run_id", crawl_run_id)
        else:
            # Get latest crawl run
            latest_run = (
                self.client.table("crawl_runs")
                .select("id")
                .eq("client_id", client_id)
                .eq("status", "completed")
                .order("completed_at", desc=True)
                .limit(1)
                .execute()
            )
            if latest_run.data:
                query = query.eq("crawl_run_id", latest_run.data[0]["id"])

        result = query.execute()
        if not result.data:
            logger.warning(f"No crawl data found for client {client_id}")
            return pd.DataFrame()
        return pd.DataFrame(result.data)

    def load_ga4_data(self, client_id: str) -> pd.DataFrame:
        """Load latest GA4 data from Supabase."""
        result = (
            self.client.table("ga4_data")
            .select("*")
            .eq("client_id", client_id)
            .order("created_at", desc=True)
            .execute()
        )
        if not result.data:
            return pd.DataFrame()
        return pd.DataFrame(result.data)

    def load_gsc_data(self, client_id: str) -> pd.DataFrame:
        """Load latest GSC data from Supabase."""
        result = (
            self.client.table("gsc_data")
            .select("*")
            .eq("client_id", client_id)
            .order("created_at", desc=True)
            .execute()
        )
        if not result.data:
            return pd.DataFrame()
        return pd.DataFrame(result.data)

    def load_backlink_data(self, client_id: str) -> pd.DataFrame:
        """Load backlink data from Supabase."""
        result = (
            self.client.table("backlink_data")
            .select("*")
            .eq("client_id", client_id)
            .execute()
        )
        if not result.data:
            return pd.DataFrame()
        return pd.DataFrame(result.data)

    def load_business_rules(self, client_id: str) -> dict:
        """Load business rules for a client."""
        result = (
            self.client.table("business_rules")
            .select("*")
            .eq("client_id", client_id)
            .execute()
        )
        rules = {}
        for row in result.data or []:
            rules[row["rule_type"]] = row["rule_config"]
        return rules

    def save_wqa_results(self, client_id: str, results_df: pd.DataFrame) -> int:
        """Save WQA results to Supabase. Returns count of rows saved."""
        if results_df.empty:
            return 0

        records = []
        for _, row in results_df.iterrows():
            record = {
                "client_id": client_id,
                "url": str(row.get("url", "")),
                "page_type": str(row.get("page_type", "")),
                "action": str(row.get("action", "")),
                "priority": str(row.get("priority", "medium")).lower(),
                "status": "pending",
                "details": {
                    col: (None if pd.isna(row[col]) else row[col])
                    for col in results_df.columns
                    if col not in ("url", "page_type", "action", "priority")
                },
            }
            # Ensure priority is valid
            if record["priority"] not in ("critical", "high", "medium", "low"):
                record["priority"] = "medium"
            records.append(record)

        # Batch insert (Supabase handles up to 1000 at a time)
        batch_size = 500
        total = 0
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            self.client.table("wqa_results").insert(batch).execute()
            total += len(batch)

        logger.info(f"Saved {total} WQA results for client {client_id}")
        return total

    def upload_excel(self, client_id: str, excel_bytes: bytes, filename: Optional[str] = None) -> str:
        """Upload Excel report to Supabase Storage and return public URL."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"wqa_reports/{client_id}/{timestamp}_wqa_report.xlsx"

        self.client.storage.from_("reports").upload(
            filename,
            excel_bytes,
            {"content-type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"},
        )

        url = self.client.storage.from_("reports").get_public_url(filename)
        logger.info(f"Uploaded Excel report: {url}")
        return url

    def run_full_audit(self, client_id: str) -> dict:
        """Run a complete WQA audit for a client using Supabase data."""
        from .merger import merge_datasets
        from .rules import assign_actions, assign_priority
        from .quality import validate_data_quality
        from .excel_writer import create_excel_report

        logger.info(f"Starting WQA audit for client {client_id}")

        # Load data from Supabase
        crawl_df = self.load_crawl_data(client_id)
        ga4_df = self.load_ga4_data(client_id)
        gsc_df = self.load_gsc_data(client_id)
        backlink_df = self.load_backlink_data(client_id)

        if crawl_df.empty:
            return {"error": "No crawl data available", "client_id": client_id}

        # Run quality validation
        quality_report = validate_data_quality(crawl_df, ga4_df, gsc_df, backlink_df)

        # Merge datasets
        merged_df = merge_datasets(crawl_df, ga4_df, gsc_df, backlink_df)

        # Assign actions and priorities
        merged_df = assign_actions(merged_df)
        merged_df = assign_priority(merged_df)

        # Build summary stats for the Summary sheet
        summaries = self._build_summaries(merged_df)

        # Generate Excel report (analytical sheets are built internally by the writer)
        excel_bytes = create_excel_report(merged_df, summaries, quality_report, gsc_df=gsc_df)

        # Upload to Supabase Storage
        excel_url = self.upload_excel(client_id, excel_bytes)

        # Save results to DB
        results_count = self.save_wqa_results(client_id, merged_df)

        return {
            "client_id": client_id,
            "results_count": results_count,
            "excel_url": excel_url,
            "quality_report": quality_report.__dict__ if hasattr(quality_report, '__dict__') else quality_report,
        }

    @staticmethod
    def _build_summaries(df: pd.DataFrame) -> dict:
        """Build summary DataFrames for the Excel Summary sheet."""
        summaries = {}

        if 'priority' in df.columns:
            summaries['priority'] = df['priority'].value_counts().reset_index()
            summaries['priority'].columns = ['Priority', 'Count']

        if 'action' in df.columns:
            tech_actions = df[df['action'].str.contains('Redirect|Schema|Canonical|Sitemap', case=False, na=False)]
            if not tech_actions.empty:
                summaries['technical_actions'] = tech_actions['action'].value_counts().reset_index()
                summaries['technical_actions'].columns = ['Action', 'Count']

            content_actions = df[df['action'].str.contains('Rewrite|Optimize|Expand|Content', case=False, na=False)]
            if not content_actions.empty:
                summaries['content_actions'] = content_actions['action'].value_counts().reset_index()
                summaries['content_actions'].columns = ['Action', 'Count']

        if 'page_type' in df.columns:
            summaries['page_type'] = df['page_type'].value_counts().reset_index()
            summaries['page_type'].columns = ['Page Type', 'Count']

        if 'status_code' in df.columns:
            summaries['status_code'] = df['status_code'].value_counts().reset_index()
            summaries['status_code'].columns = ['Status Code', 'Count']

        return summaries
