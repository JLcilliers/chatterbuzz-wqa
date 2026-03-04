"""Pipeline router — trigger and monitor data ingestion pipelines."""

import os
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from supabase import create_client

router = APIRouter(tags=["Pipeline"])


def get_supabase():
    return create_client(
        os.environ.get("NEXT_PUBLIC_SUPABASE_URL", ""),
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY", ""),
    )


@router.post("/{client_id}/run")
async def run_pipeline(client_id: str, workers: str = "all"):
    """Trigger a full or partial pipeline run for a client."""
    from data_pipeline.orchestrator import run_full_pipeline

    db = get_supabase()
    worker_list = None if workers == "all" else workers.split(",")

    result = run_full_pipeline(db, client_id, workers=worker_list)
    return result


@router.post("/{client_id}/upload/crawl")
async def upload_crawl(
    client_id: str,
    file: UploadFile = File(...),
    source: str = Form("screaming_frog"),
):
    """Upload crawl data CSV/Excel for a client."""
    from data_pipeline.workers.crawl_worker import run_crawl_worker

    db = get_supabase()
    content = await file.read()
    result = run_crawl_worker(db, client_id, content, file.filename or "", source)
    return result


@router.post("/{client_id}/upload/backlinks")
async def upload_backlinks(client_id: str, file: UploadFile = File(...)):
    """Upload backlink data CSV/Excel for a client."""
    from data_pipeline.workers.backlink_worker import run_backlink_worker

    db = get_supabase()
    content = await file.read()
    result = run_backlink_worker(db, client_id, file_content=content, filename=file.filename or "")
    return result


@router.get("/{client_id}/status")
async def pipeline_status(client_id: str, limit: int = 10):
    """Get recent pipeline runs for a client."""
    db = get_supabase()
    result = db.table("pipeline_runs") \
        .select("*") \
        .eq("client_id", client_id) \
        .order("started_at", desc=True) \
        .limit(limit) \
        .execute()
    return result.data or []
