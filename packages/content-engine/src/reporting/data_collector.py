"""Data collector — pull MoM/YoY snapshots from all tables."""

import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


class DataCollector:
    """Collects metrics snapshots for reporting."""

    def __init__(self, supabase_client):
        self.db = supabase_client

    def collect_metrics(self, client_id: str, report_month: Optional[str] = None) -> dict:
        """Collect a complete metrics snapshot for a client.

        Args:
            client_id: Client UUID
            report_month: YYYY-MM format (defaults to previous month)

        Returns:
            Dict with all metric categories
        """
        if not report_month:
            last_month = datetime.utcnow().replace(day=1) - timedelta(days=1)
            report_month = last_month.strftime("%Y-%m")

        year, month = report_month.split("-")
        month_start = f"{year}-{month}-01"
        month_end = (datetime(int(year), int(month), 1) + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        month_end_str = month_end.strftime("%Y-%m-%d")

        return {
            "report_month": report_month,
            "organic_traffic": self._get_traffic_metrics(client_id, month_start, month_end_str),
            "search_performance": self._get_search_metrics(client_id, month_start, month_end_str),
            "content_metrics": self._get_content_metrics(client_id),
            "technical_health": self._get_technical_metrics(client_id),
            "indexation": self._get_index_metrics(client_id),
            "pipeline_activity": self._get_pipeline_metrics(client_id, month_start, month_end_str),
        }

    def _get_traffic_metrics(self, client_id: str, start: str, end: str) -> dict:
        """Get GA4 traffic metrics for the period."""
        result = self.db.table("ga4_data").select("sessions, engaged_sessions, conversions, revenue") \
            .eq("client_id", client_id) \
            .gte("date_range_start", start) \
            .lte("date_range_end", end) \
            .execute()

        data = result.data or []
        total_sessions = sum(r.get("sessions", 0) or 0 for r in data)
        total_engaged = sum(r.get("engaged_sessions", 0) or 0 for r in data)
        total_conversions = sum(r.get("conversions", 0) or 0 for r in data)
        total_revenue = sum(r.get("revenue", 0) or 0 for r in data)

        return {
            "total_sessions": total_sessions,
            "engaged_sessions": total_engaged,
            "engagement_rate": round(total_engaged / max(total_sessions, 1) * 100, 1),
            "conversions": total_conversions,
            "revenue": round(total_revenue, 2),
            "pages_measured": len(data),
        }

    def _get_search_metrics(self, client_id: str, start: str, end: str) -> dict:
        """Get GSC search performance metrics."""
        result = self.db.table("gsc_data").select("clicks, impressions, ctr, position") \
            .eq("client_id", client_id) \
            .gte("date_range_start", start) \
            .lte("date_range_end", end) \
            .execute()

        data = result.data or []
        total_clicks = sum(r.get("clicks", 0) or 0 for r in data)
        total_impressions = sum(r.get("impressions", 0) or 0 for r in data)
        positions = [r["position"] for r in data if r.get("position")]

        return {
            "total_clicks": total_clicks,
            "total_impressions": total_impressions,
            "avg_ctr": round(total_clicks / max(total_impressions, 1) * 100, 2),
            "avg_position": round(sum(positions) / max(len(positions), 1), 1) if positions else None,
            "queries_measured": len(data),
        }

    def _get_content_metrics(self, client_id: str) -> dict:
        """Get content queue metrics."""
        statuses = {}
        for status in ["draft", "review", "approved", "published", "rejected"]:
            result = self.db.table("content_queue").select("id", count="exact") \
                .eq("client_id", client_id).eq("status", status).execute()
            statuses[status] = result.count or 0

        return statuses

    def _get_technical_metrics(self, client_id: str) -> dict:
        """Get WQA technical health metrics."""
        result = self.db.table("wqa_results").select("priority, status") \
            .eq("client_id", client_id).execute()

        data = result.data or []
        by_priority = {}
        by_status = {}
        for r in data:
            p = r.get("priority", "medium")
            s = r.get("status", "pending")
            by_priority[p] = by_priority.get(p, 0) + 1
            by_status[s] = by_status.get(s, 0) + 1

        return {"by_priority": by_priority, "by_status": by_status, "total_issues": len(data)}

    def _get_index_metrics(self, client_id: str) -> dict:
        """Get indexation status metrics."""
        indexed = self.db.table("index_status").select("id", count="exact") \
            .eq("client_id", client_id).eq("is_indexed", True).execute()
        not_indexed = self.db.table("index_status").select("id", count="exact") \
            .eq("client_id", client_id).eq("is_indexed", False).execute()

        return {
            "indexed": indexed.count or 0,
            "not_indexed": not_indexed.count or 0,
        }

    def _get_pipeline_metrics(self, client_id: str, start: str, end: str) -> dict:
        """Get pipeline execution metrics for the period."""
        result = self.db.table("pipeline_runs").select("status, pipeline_type") \
            .eq("client_id", client_id) \
            .gte("started_at", start) \
            .lte("started_at", end + "T23:59:59") \
            .execute()

        data = result.data or []
        return {
            "total_runs": len(data),
            "successful": sum(1 for r in data if r.get("status") == "completed"),
            "failed": sum(1 for r in data if r.get("status") == "failed"),
        }
