"""
Quran Router - /api/quran/*, /spans/*, /surahs/*, /datasets/*

Phase 7.1: Modular API structure
Contains Quran verse and span endpoints.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query

from ..dependencies import (
    get_all_spans, get_dataset_meta, load_dataset, build_surah_summary
)
from ..models import (
    DatasetResponse, DatasetMetadata, SpansResponse, SurahsResponse
)

router = APIRouter(tags=["Quran"])


@router.get("/datasets/{tier}", response_model=DatasetResponse)
async def get_dataset(tier: str):
    """Get full dataset by tier (gold, silver, research)."""
    if tier not in ["gold", "silver", "research"]:
        raise HTTPException(status_code=400, detail="Invalid tier. Use: gold, silver, research")
    
    data = load_dataset(tier)
    
    metadata = data.get("metadata", {})
    if not metadata:
        metadata = {
            "tier": tier,
            "version": data.get("version", "1.0.0"),
            "exported_at": data.get("exported_at", "unknown"),
            "total_spans": len(data.get("spans", []))
        }
    
    return DatasetResponse(
        metadata=DatasetMetadata(**metadata),
        spans=data.get("spans", [])
    )


@router.get("/spans", response_model=SpansResponse)
async def search_spans(
    surah: Optional[int] = Query(None, ge=1, le=114, description="Filter by surah number"),
    agent: Optional[str] = Query(None, description="Filter by agent type (e.g., AGT_ALLAH)"),
    behavior: Optional[str] = Query(None, description="Filter by behavior form"),
    evaluation: Optional[str] = Query(None, description="Filter by evaluation"),
    deontic: Optional[str] = Query(None, description="Filter by deontic signal"),
    limit: int = Query(100, ge=1, le=1000, description="Max results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
):
    """Search spans with filters."""
    spans = get_all_spans()
    
    # Apply filters
    if surah:
        spans = [s for s in spans if s.get("reference", {}).get("surah") == surah]
    
    if agent:
        spans = [s for s in spans if s.get("agent", {}).get("type") == agent]
    
    if behavior:
        spans = [s for s in spans if s.get("behavior_form") == behavior]
    
    if evaluation:
        spans = [s for s in spans if s.get("normative", {}).get("evaluation") == evaluation]
    
    if deontic:
        spans = [s for s in spans if s.get("normative", {}).get("deontic_signal") == deontic]
    
    total = len(spans)
    spans = spans[offset:offset + limit]
    
    return SpansResponse(total=total, spans=spans)


@router.get("/spans/recent", response_model=SpansResponse)
async def get_recent_spans(limit: int = Query(10, ge=1, le=100)):
    """Get most recently added spans."""
    spans = get_all_spans()
    
    # Sort by annotation date if available
    spans_with_date = [s for s in spans if s.get("annotation_date")]
    spans_with_date.sort(key=lambda x: x.get("annotation_date", ""), reverse=True)
    
    recent = spans_with_date[:limit] if spans_with_date else spans[:limit]
    return SpansResponse(total=len(recent), spans=recent)


@router.get("/spans/{span_id}")
async def get_span(span_id: str):
    """Get a specific span by ID."""
    spans = get_all_spans()
    
    for span in spans:
        if span.get("id") == span_id:
            return span
    
    raise HTTPException(status_code=404, detail=f"Span {span_id} not found")


@router.get("/surahs", response_model=SurahsResponse)
async def list_surahs():
    """List all surahs with span counts."""
    spans = get_all_spans()
    summaries = build_surah_summary(spans)
    return SurahsResponse(surahs=summaries)


@router.get("/surahs/{surah_num}", response_model=SpansResponse)
async def get_surah_spans(surah_num: int):
    """Get all spans for a specific surah."""
    if surah_num < 1 or surah_num > 114:
        raise HTTPException(status_code=400, detail="Surah number must be between 1 and 114")
    
    spans = get_all_spans()
    surah_spans = [s for s in spans if s.get("reference", {}).get("surah") == surah_num]
    
    return SpansResponse(total=len(surah_spans), spans=surah_spans)


@router.get("/ayah/{surah}/{ayah}")
async def get_ayah(surah: int, ayah: int):
    """Get ayah with all annotations."""
    spans = get_all_spans()
    
    ayah_spans = [
        s for s in spans
        if s.get("reference", {}).get("surah") == surah
        and s.get("reference", {}).get("ayah") == ayah
    ]
    
    if not ayah_spans:
        raise HTTPException(status_code=404, detail=f"No annotations found for {surah}:{ayah}")
    
    # Get unique behaviors
    behaviors = list(set(s.get("behavior_form", "") for s in ayah_spans if s.get("behavior_form")))
    
    # Get unique agents
    agents = list(set(s.get("agent", {}).get("type", "") for s in ayah_spans if s.get("agent", {}).get("type")))
    
    return {
        "surah": surah,
        "ayah": ayah,
        "surah_name": ayah_spans[0].get("reference", {}).get("surah_name", ""),
        "text": ayah_spans[0].get("text", ""),
        "spans": ayah_spans,
        "behaviors": behaviors,
        "agents": agents,
        "span_count": len(ayah_spans)
    }
