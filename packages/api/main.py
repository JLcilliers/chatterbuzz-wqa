"""FastAPI Gateway — consolidates all routers under one API."""

import os
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader

app = FastAPI(
    title="Chatterbuzz SEO Platform API",
    description="Unified API gateway for the SEO automation platform",
    version="0.1.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key auth
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    expected = os.environ.get("API_SECRET_KEY", "")
    if not api_key or api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key


@app.get("/health")
async def health():
    return {"status": "ok"}


# Rate limiting
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded

    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
except ImportError:
    pass

# Routers
from routers.pipeline import router as pipeline_router
from routers.wqa import router as wqa_router
from routers.content import router as content_router
from routers.webhooks import router as webhooks_router

app.include_router(pipeline_router, prefix="/pipeline", dependencies=[Depends(verify_api_key)])
app.include_router(wqa_router, prefix="/wqa", dependencies=[Depends(verify_api_key)])
app.include_router(content_router, prefix="/content", dependencies=[Depends(verify_api_key)])
app.include_router(webhooks_router, prefix="/webhooks", dependencies=[Depends(verify_api_key)])
