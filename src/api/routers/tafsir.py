"""
Tafsir Router - /tafsir/*, /api/tafsir/*

Phase 7.1: Modular API structure
Contains tafsir retrieval and comparison endpoints.
"""

from fastapi import APIRouter, HTTPException

from ..dependencies import get_all_spans
from ..unified_brain import get_brain, TAFSIR_SOURCES

router = APIRouter(tags=["Tafsir"])


@router.get("/tafsir/{surah}/{ayah}")
async def get_tafsir(surah: int, ayah: int):
    """Get tafsir for a specific ayah from all sources."""
    if surah < 1 or surah > 114:
        raise HTTPException(status_code=400, detail="Surah must be between 1 and 114")
    if ayah < 1:
        raise HTTPException(status_code=400, detail="Ayah must be positive")
    
    spans = get_all_spans()
    brain = get_brain(spans)
    
    tafsir_data = {}
    for source in TAFSIR_SOURCES:
        try:
            tafsir = brain.get_tafsir(surah, ayah, source)
            if tafsir:
                tafsir_data[source] = tafsir
        except Exception:
            continue
    
    if not tafsir_data:
        raise HTTPException(
            status_code=404,
            detail=f"No tafsir found for {surah}:{ayah}"
        )
    
    return {
        "surah": surah,
        "ayah": ayah,
        "sources": list(tafsir_data.keys()),
        "tafsir": tafsir_data
    }


@router.get("/tafsir/compare/{surah}/{ayah}")
async def compare_tafsir(surah: int, ayah: int):
    """Compare tafsir from all sources for a specific ayah."""
    if surah < 1 or surah > 114:
        raise HTTPException(status_code=400, detail="Surah must be between 1 and 114")
    if ayah < 1:
        raise HTTPException(status_code=400, detail="Ayah must be positive")
    
    spans = get_all_spans()
    brain = get_brain(spans)
    
    comparisons = []
    for source in TAFSIR_SOURCES:
        try:
            tafsir = brain.get_tafsir(surah, ayah, source)
            if tafsir:
                comparisons.append({
                    "source": source,
                    "text": tafsir.get("text", ""),
                    "behaviors_mentioned": tafsir.get("behaviors", []),
                    "word_count": len(tafsir.get("text", "").split())
                })
        except Exception:
            continue
    
    if not comparisons:
        raise HTTPException(
            status_code=404,
            detail=f"No tafsir found for {surah}:{ayah}"
        )
    
    return {
        "surah": surah,
        "ayah": ayah,
        "source_count": len(comparisons),
        "comparisons": comparisons
    }


@router.get("/api/brain/tafsir/{surah}/{ayah}")
async def brain_tafsir(surah: int, ayah: int):
    """Get tafsir via brain interface."""
    return await get_tafsir(surah, ayah)


@router.get("/api/brain/tafsir/behaviors/{behavior}")
async def tafsir_by_behavior(behavior: str):
    """Get tafsir passages that mention a specific behavior."""
    spans = get_all_spans()
    brain = get_brain(spans)
    
    results = brain.search_tafsir_by_behavior(behavior)
    
    return {
        "behavior": behavior,
        "total": len(results),
        "passages": results[:50]  # Limit to 50
    }


@router.get("/api/brain/tafsir/search")
async def search_tafsir(q: str, limit: int = 20):
    """Search tafsir text."""
    if not q or len(q) < 2:
        raise HTTPException(status_code=400, detail="Query must be at least 2 characters")
    
    spans = get_all_spans()
    brain = get_brain(spans)
    
    results = brain.search_tafsir(q, limit=limit)
    
    return {
        "query": q,
        "total": len(results),
        "results": results
    }
