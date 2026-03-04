"""Anomaly detection — traffic cliffs, CTR drops, de-indexation spikes, ranking losses."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Detects anomalies across SEO metrics and creates alerts."""

    def __init__(self, supabase_client):
        self.db = supabase_client

    def detect_all(self, client_id: str, current_metrics: dict, previous_metrics: Optional[dict] = None) -> list[dict]:
        """Run all anomaly detections. Returns list of alert dicts."""
        alerts = []

        if previous_metrics:
            alerts.extend(self._check_traffic_cliff(client_id, current_metrics, previous_metrics))
            alerts.extend(self._check_ctr_drop(client_id, current_metrics, previous_metrics))
            alerts.extend(self._check_ranking_loss(client_id, current_metrics, previous_metrics))

        alerts.extend(self._check_deindexation_spike(client_id))

        return alerts

    def _check_traffic_cliff(self, client_id: str, current: dict, previous: dict) -> list[dict]:
        """Detect >20% traffic drop month over month."""
        alerts = []
        curr_sessions = current.get("organic_traffic", {}).get("total_sessions", 0)
        prev_sessions = previous.get("organic_traffic", {}).get("total_sessions", 0)

        if prev_sessions > 0:
            change_pct = ((curr_sessions - prev_sessions) / prev_sessions) * 100
            if change_pct < -20:
                alerts.append({
                    "client_id": client_id,
                    "alert_type": "traffic_cliff",
                    "severity": "critical" if change_pct < -40 else "high",
                    "detail": f"Organic sessions dropped {abs(change_pct):.1f}% MoM ({prev_sessions:,} -> {curr_sessions:,})",
                    "metric_current": curr_sessions,
                    "metric_previous": prev_sessions,
                    "change_pct": round(change_pct, 1),
                })
        return alerts

    def _check_ctr_drop(self, client_id: str, current: dict, previous: dict) -> list[dict]:
        """Detect >30% CTR drop."""
        alerts = []
        curr_ctr = current.get("search_performance", {}).get("avg_ctr", 0)
        prev_ctr = previous.get("search_performance", {}).get("avg_ctr", 0)

        if prev_ctr > 0:
            change_pct = ((curr_ctr - prev_ctr) / prev_ctr) * 100
            if change_pct < -30:
                alerts.append({
                    "client_id": client_id,
                    "alert_type": "ctr_drop",
                    "severity": "high",
                    "detail": f"Average CTR dropped {abs(change_pct):.1f}% MoM ({prev_ctr:.2f}% -> {curr_ctr:.2f}%)",
                    "metric_current": curr_ctr,
                    "metric_previous": prev_ctr,
                    "change_pct": round(change_pct, 1),
                })
        return alerts

    def _check_deindexation_spike(self, client_id: str) -> list[dict]:
        """Detect >5 pages de-indexed."""
        alerts = []
        result = self.db.table("index_status").select("url") \
            .eq("client_id", client_id).eq("is_indexed", False).execute()

        deindexed_count = len(result.data or [])
        if deindexed_count > 5:
            alerts.append({
                "client_id": client_id,
                "alert_type": "deindexation_spike",
                "severity": "critical",
                "detail": f"{deindexed_count} pages are currently de-indexed",
                "metric_current": deindexed_count,
            })
        return alerts

    def _check_ranking_loss(self, client_id: str, current: dict, previous: dict) -> list[dict]:
        """Detect average position loss >10 positions."""
        alerts = []
        curr_pos = current.get("search_performance", {}).get("avg_position")
        prev_pos = previous.get("search_performance", {}).get("avg_position")

        if curr_pos and prev_pos:
            loss = curr_pos - prev_pos  # Higher position = worse
            if loss > 10:
                alerts.append({
                    "client_id": client_id,
                    "alert_type": "ranking_loss",
                    "severity": "high",
                    "detail": f"Average position worsened by {loss:.1f} positions ({prev_pos:.1f} -> {curr_pos:.1f})",
                    "metric_current": curr_pos,
                    "metric_previous": prev_pos,
                    "change_positions": round(loss, 1),
                })
        return alerts
