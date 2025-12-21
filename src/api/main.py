"""
QBM REST API - Quranic Behavioral Matrix

FastAPI backend for accessing QBM annotation data.

Endpoints:
- GET / - Health check
- GET /datasets/{tier} - Full dataset (gold/silver/research)
- GET /spans - Search with filters
- GET /spans/{id} - Get specific span
- GET /surahs/{num} - Get spans by surah
- GET /stats - Dataset statistics
- GET /vocabularies - Controlled vocabularies

Usage:
    uvicorn src.api.main:app --reload
"""

import json
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from .models import (
    HealthResponse,
    DatasetResponse,
    DatasetMetadata,
    SpansResponse,
    SpanResponse,
    StatsResponse,
    VocabulariesResponse,
    Span,
)

# API metadata
API_VERSION = "0.7.0"
API_TITLE = "QBM API"
API_DESCRIPTION = """
Quranic Behavioral Matrix (QBM) REST API.

Access 6,236 annotated ayat covering the complete Quran with behavioral,
normative, and semantic annotations.

## Tiers
- **gold**: Fully reviewed, high confidence
- **silver**: Quality checked, single annotator  
- **research**: All annotations including drafts

## Filters
Use query parameters to filter spans by surah, agent type, behavior form,
evaluation, and deontic signal.
"""

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Data paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"
EXPORTS_DIR = DATA_DIR / "exports"
VOCAB_DIR = DATA_DIR / "vocab"

# Cache for loaded data
_dataset_cache = {}


def load_dataset(tier: str = "silver") -> dict:
    """Load dataset from exports directory."""
    if tier in _dataset_cache:
        return _dataset_cache[tier]
    
    # Find latest export for tier
    pattern = f"qbm_{tier}_*.json"
    files = sorted(EXPORTS_DIR.glob(pattern), reverse=True)
    
    if not files:
        raise HTTPException(status_code=404, detail=f"No {tier} dataset found")
    
    with open(files[0], encoding="utf-8") as f:
        data = json.load(f)
    
    _dataset_cache[tier] = data
    return data


def get_all_spans(tier: str = "silver") -> List[dict]:
    """Get all spans from dataset."""
    data = load_dataset(tier)
    return data.get("spans", [])


@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        spans = get_all_spans()
        loaded = len(spans) > 0
    except Exception:
        loaded = False
    
    return HealthResponse(
        status="ok",
        version=API_VERSION,
        dataset_loaded=loaded
    )


@app.get("/datasets/{tier}", response_model=DatasetResponse)
async def get_dataset(tier: str):
    """Get full dataset by tier (gold, silver, research)."""
    if tier not in ["gold", "silver", "research"]:
        raise HTTPException(status_code=400, detail="Invalid tier. Use: gold, silver, research")
    
    data = load_dataset(tier)
    
    metadata = data.get("metadata", {})
    if not metadata:
        metadata = {
            "tier": tier,
            "version": API_VERSION,
            "exported_at": data.get("exported_at", "unknown"),
            "total_spans": len(data.get("spans", []))
        }
    
    return DatasetResponse(
        metadata=DatasetMetadata(**metadata),
        spans=data.get("spans", [])
    )


@app.get("/spans", response_model=SpansResponse)
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


@app.get("/spans/{span_id}")
async def get_span(span_id: str):
    """Get a specific span by ID."""
    spans = get_all_spans()
    
    for span in spans:
        if span.get("span_id") == span_id or span.get("id") == span_id:
            return {"span": span}
    
    raise HTTPException(status_code=404, detail=f"Span {span_id} not found")


@app.get("/surahs/{surah_num}", response_model=SpansResponse)
async def get_surah_spans(surah_num: int):
    """Get all spans for a surah."""
    if surah_num < 1 or surah_num > 114:
        raise HTTPException(status_code=400, detail="Surah must be 1-114")
    
    spans = get_all_spans()
    surah_spans = [s for s in spans if s.get("reference", {}).get("surah") == surah_num]
    
    return SpansResponse(total=len(surah_spans), spans=surah_spans)


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get dataset statistics."""
    spans = get_all_spans()
    
    surahs = set()
    ayat = set()
    agent_types = {}
    behavior_forms = {}
    evaluations = {}
    deontic_signals = {}
    
    for span in spans:
        ref = span.get("reference", {})
        if ref.get("surah"):
            surahs.add(ref["surah"])
            ayat.add(f"{ref['surah']}:{ref.get('ayah', 0)}")
        
        agent = span.get("agent", {}).get("type", "unknown")
        agent_types[agent] = agent_types.get(agent, 0) + 1
        
        bf = span.get("behavior_form", "unknown")
        behavior_forms[bf] = behavior_forms.get(bf, 0) + 1
        
        ev = span.get("normative", {}).get("evaluation", "unknown")
        evaluations[ev] = evaluations.get(ev, 0) + 1
        
        ds = span.get("normative", {}).get("deontic_signal", "unknown")
        deontic_signals[ds] = deontic_signals.get(ds, 0) + 1
    
    return StatsResponse(
        total_spans=len(spans),
        unique_surahs=len(surahs),
        unique_ayat=len(ayat),
        agent_types=agent_types,
        behavior_forms=behavior_forms,
        evaluations=evaluations,
        deontic_signals=deontic_signals
    )


@app.get("/vocabularies", response_model=VocabulariesResponse)
async def get_vocabularies():
    """Get controlled vocabularies."""
    return VocabulariesResponse(
        agent_types=[
            "AGT_ALLAH", "AGT_PROPHET", "AGT_BELIEVER", "AGT_DISBELIEVER",
            "AGT_HYPOCRITE", "AGT_HUMAN_GENERAL", "AGT_ANGEL", "AGT_JINN",
            "AGT_HISTORICAL_FIGURE", "AGT_WRONGDOER", "AGT_POLYTHEIST",
            "AGT_PEOPLE_BOOK", "AGT_OTHER"
        ],
        behavior_forms=[
            "physical_act", "speech_act", "inner_state", "trait_disposition",
            "relational_act", "omission", "mixed", "unknown"
        ],
        evaluations=["praise", "blame", "neutral"],
        deontic_signals=["amr", "nahy", "targhib", "tarhib", "khabar"],
        speech_modes=["command", "prohibition", "informative", "interrogative"],
        systemic=["SYS_GOD", "SYS_SOCIAL"]
    )
