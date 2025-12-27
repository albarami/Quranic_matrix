"""
Genome Router - /api/genome/*

Phase 7.3: Genome export endpoint (placeholder for now)
Will be fully implemented in Phase 7.3.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/api/genome", tags=["Genome"])


@router.get("/status")
async def genome_status():
    """Get genome export status (Phase 7.3 placeholder)."""
    return {
        "status": "not_implemented",
        "message": "Genome export will be implemented in Phase 7.3"
    }
