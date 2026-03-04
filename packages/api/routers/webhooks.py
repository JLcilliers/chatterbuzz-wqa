"""Webhook router — n8n automation triggers."""

import os
import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks
from supabase import create_client

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Webhooks"])


def get_supabase():
    return create_client(
        os.environ.get("NEXT_PUBLIC_SUPABASE_URL", ""),
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY", ""),
    )


@router.post("/gsc-sync/{client_id}")
async def trigger_gsc_sync(client_id: str, background_tasks: BackgroundTasks):
    """Trigger daily GSC data sync (n8n: daily 02:00 UTC)."""
    from data_pipeline.orchestrator import run_worker
    db = get_supabase()
    background_tasks.add_task(run_worker, db, client_id, "gsc")
    return {"status": "triggered", "worker": "gsc", "client_id": client_id}


@router.post("/ga4-sync/{client_id}")
async def trigger_ga4_sync(client_id: str, background_tasks: BackgroundTasks):
    """Trigger weekly GA4+GBP+Backlinks sync (n8n: Sun 02:00)."""
    from data_pipeline.orchestrator import run_full_pipeline
    db = get_supabase()
    background_tasks.add_task(run_full_pipeline, db, client_id, workers=["ga4", "gbp", "backlinks"])
    return {"status": "triggered", "workers": ["ga4", "gbp", "backlinks"], "client_id": client_id}


@router.post("/full-pipeline/{client_id}")
async def trigger_full_pipeline(client_id: str, background_tasks: BackgroundTasks):
    """Trigger monthly full pipeline (n8n: 1st 01:00)."""
    from data_pipeline.orchestrator import run_full_pipeline
    db = get_supabase()
    background_tasks.add_task(run_full_pipeline, db, client_id)
    return {"status": "triggered", "pipeline": "full", "client_id": client_id}


@router.post("/gbp-posts/{client_id}")
async def trigger_gbp_posts(client_id: str, background_tasks: BackgroundTasks):
    """Trigger weekly GBP post generation (n8n: Mon 08:00)."""
    # Implementation will be wired when GBP module is ready
    return {"status": "triggered", "worker": "gbp_posts", "client_id": client_id}


@router.post("/review-replies/{client_id}")
async def trigger_review_replies(client_id: str, background_tasks: BackgroundTasks):
    """Trigger daily review auto-reply (n8n: daily 09:00)."""
    return {"status": "triggered", "worker": "review_replies", "client_id": client_id}


@router.post("/anomaly-detection/{client_id}")
async def trigger_anomaly_detection(client_id: str, background_tasks: BackgroundTasks):
    """Trigger weekly anomaly detection (n8n: Fri 10:00)."""
    from content_engine.reporting.data_collector import DataCollector
    from content_engine.reporting.anomaly_detector import AnomalyDetector
    db = get_supabase()
    collector = DataCollector(db)
    detector = AnomalyDetector(db)

    async def run():
        metrics = collector.collect_metrics(client_id)
        alerts = detector.detect_all(client_id, metrics)
        if alerts:
            logger.warning(f"Found {len(alerts)} anomalies for client {client_id}")
        return alerts

    background_tasks.add_task(run)
    return {"status": "triggered", "worker": "anomaly_detection", "client_id": client_id}


@router.post("/monthly-report/{client_id}")
async def trigger_monthly_report(client_id: str, background_tasks: BackgroundTasks):
    """Trigger monthly report generation (n8n: 28th)."""
    return {"status": "triggered", "worker": "monthly_report", "client_id": client_id}


@router.post("/content-batch/{client_id}")
async def trigger_content_batch(client_id: str, background_tasks: BackgroundTasks):
    """Trigger weekly content batch generation (n8n: Tue 06:00)."""
    return {"status": "triggered", "worker": "content_batch", "client_id": client_id}


@router.post("/cms-publish/{client_id}")
async def trigger_cms_publish(client_id: str, background_tasks: BackgroundTasks):
    """Trigger weekly CMS publishing (n8n: Wed 06:00)."""
    return {"status": "triggered", "worker": "cms_publish", "client_id": client_id}


@router.post("/index-monitor/{client_id}")
async def trigger_index_monitor(client_id: str, background_tasks: BackgroundTasks):
    """Trigger daily index monitoring (n8n: daily 12:00)."""
    return {"status": "triggered", "worker": "index_monitor", "client_id": client_id}
