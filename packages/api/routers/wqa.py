"""WQA router — run audits and retrieve results."""

import os
from fastapi import APIRouter, HTTPException
from supabase import create_client

router = APIRouter(tags=["WQA"])


def get_supabase():
    return create_client(
        os.environ.get("NEXT_PUBLIC_SUPABASE_URL", ""),
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY", ""),
    )


@router.post("/{client_id}/run")
async def run_wqa(client_id: str):
    """Run a full WQA audit for a client."""
    from wqa_engine.supabase_adapter import WQASupabaseEngine

    engine = WQASupabaseEngine()
    result = engine.run_full_audit(client_id)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.get("/{client_id}/results")
async def get_results(client_id: str, status: str = "", priority: str = "", limit: int = 100):
    """Get WQA results for a client with optional filters."""
    db = get_supabase()
    query = db.table("wqa_results").select("*").eq("client_id", client_id)
    if status:
        query = query.eq("status", status)
    if priority:
        query = query.eq("priority", priority)
    result = query.order("created_at", desc=True).limit(limit).execute()
    return result.data or []


@router.patch("/{client_id}/results/{result_id}")
async def update_result(client_id: str, result_id: str, status: str):
    """Update the status of a WQA result."""
    db = get_supabase()
    result = db.table("wqa_results") \
        .update({"status": status}) \
        .eq("id", result_id) \
        .eq("client_id", client_id) \
        .execute()
    return result.data[0] if result.data else {}
