"""Content router — manage content queue, generation, and publishing."""

import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from supabase import create_client

router = APIRouter(tags=["Content"])


def get_supabase():
    return create_client(
        os.environ.get("NEXT_PUBLIC_SUPABASE_URL", ""),
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY", ""),
    )


class ContentGenerateRequest(BaseModel):
    keyword: str
    content_type: str = "blog"
    target_location: str = ""


@router.post("/{client_id}/generate")
async def generate_content(client_id: str, req: ContentGenerateRequest):
    """Generate AI content for a client."""
    from content_engine.generator.prompts import build_system_prompt
    from content_engine.generator.engine import generate_content as gen
    from content_engine.generator.quality_gate import score_content
    from content_engine.generator.queue import ContentQueueManager

    db = get_supabase()

    # Get client info
    client = db.table("clients").select("*").eq("id", client_id).single().execute()
    if not client.data:
        raise HTTPException(status_code=404, detail="Client not found")

    c = client.data
    prompt = build_system_prompt(
        business_name=c["name"],
        industry=c["industry"],
        business_type=c["business_type"],
        brand_voice=c.get("brand_voice", {}),
        content_type=req.content_type,
        target_location=req.target_location,
    )

    result = gen(system_prompt=prompt, user_prompt=f"Write about: {req.keyword}")
    if not result:
        raise HTTPException(status_code=500, detail="Content generation failed")

    # Quality gate
    score = score_content(
        content_html=result.content_html,
        meta_title=result.meta_title,
        meta_description=result.meta_description,
        target_keyword=req.keyword,
    )

    # Add to queue
    queue = ContentQueueManager(db)
    status = "review" if score.passed else "draft"
    item = queue.add_to_queue(
        client_id=client_id,
        title=result.title,
        content_type=req.content_type,
        content_body=result.content_html,
        meta_title=result.meta_title,
        meta_description=result.meta_description,
        quality_score=score.total,
        status=status,
    )

    return {"item": item, "quality_score": score.model_dump()}


@router.get("/{client_id}/queue")
async def get_queue(client_id: str, status: str = ""):
    """Get content queue items."""
    db = get_supabase()
    query = db.table("content_queue").select("*").eq("client_id", client_id)
    if status:
        query = query.eq("status", status)
    result = query.order("created_at", desc=True).execute()
    return result.data or []


@router.post("/{client_id}/queue/{item_id}/approve")
async def approve_content(client_id: str, item_id: str):
    """Approve a content item for publishing."""
    db = get_supabase()
    queue = ContentQueueManager(db)
    return queue.approve(item_id)


@router.post("/{client_id}/queue/{item_id}/reject")
async def reject_content(client_id: str, item_id: str):
    """Reject a content item."""
    db = get_supabase()
    queue = ContentQueueManager(db)
    return queue.reject(item_id)
