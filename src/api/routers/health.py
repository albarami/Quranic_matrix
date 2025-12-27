"""
Health Router - /health, /ready, /

Phase 7.1: Modular API structure
"""

from fastapi import APIRouter

router = APIRouter(tags=["Health"])

# API metadata
API_VERSION = "0.8.0"


@router.get("/")
async def health_check():
    """Health check endpoint."""
    from ..dependencies import get_all_spans
    
    try:
        spans = get_all_spans()
        loaded = len(spans) > 0
    except Exception:
        loaded = False
    
    return {
        "status": "ok",
        "version": API_VERSION,
        "dataset_loaded": loaded
    }


@router.get("/health")
async def health():
    """Alias for health check."""
    return await health_check()


@router.get("/ready")
async def ready():
    """Readiness check - verifies all components are loaded."""
    from ..dependencies import get_all_spans
    
    checks = {
        "dataset": False,
        "brain": False,
        "graph": False,
    }
    
    try:
        spans = get_all_spans()
        checks["dataset"] = len(spans) > 0
        
        from ..unified_brain import get_brain
        brain = get_brain(spans)
        checks["brain"] = brain is not None
        
        from ..unified_graph import get_unified_graph
        graph = get_unified_graph(spans)
        checks["graph"] = graph is not None
    except Exception:
        pass
    
    all_ready = all(checks.values())
    
    return {
        "ready": all_ready,
        "checks": checks,
        "version": API_VERSION
    }
