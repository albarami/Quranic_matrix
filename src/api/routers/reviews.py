"""
Reviews Router - /api/reviews/*

Phase 7.4: Scholar review workflow (placeholder for now)
Will be fully implemented in Phase 7.4.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/api/reviews", tags=["Reviews"])


@router.get("/status")
async def reviews_status():
    """Get reviews system status (Phase 7.4 placeholder)."""
    return {
        "status": "not_implemented",
        "message": "Scholar review workflow will be implemented in Phase 7.4"
    }
