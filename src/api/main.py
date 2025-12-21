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
- GET /tafsir/{surah}/{ayah} - Get tafsir for ayah
- GET /tafsir/compare/{surah}/{ayah} - Compare tafsir sources
- GET /ayah/{surah}/{ayah} - Get ayah with annotations
- POST /api/spans/search - Search spans (C1 frontend)

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
API_VERSION = "0.8.0"
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
            "version": data.get("version", "1.0.0"),
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


# =============================================================================
# TAFSIR ENDPOINTS (for C1 frontend integration)
# =============================================================================

TAFSIR_DIR = DATA_DIR / "tafsir"
_tafsir_cache = {}


def load_tafsir(source: str) -> dict:
    """Load tafsir data from JSON file."""
    if source in _tafsir_cache:
        return _tafsir_cache[source]
    
    filepath = TAFSIR_DIR / f"{source}.json"
    if not filepath.exists():
        # Try with .jsonl extension
        filepath = TAFSIR_DIR / f"{source}.jsonl"
        if not filepath.exists():
            return {}
    
    try:
        if filepath.suffix == ".jsonl":
            # Load JSONL format
            data = {"ayat": {}}
            with open(filepath, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        key = f"{entry.get('surah', 0)}:{entry.get('ayah', 0)}"
                        data["ayat"][key] = entry
            _tafsir_cache[source] = data
        else:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
            _tafsir_cache[source] = data
        return _tafsir_cache[source]
    except Exception:
        return {}


@app.get("/tafsir/{surah}/{ayah}")
async def get_tafsir(
    surah: int,
    ayah: int,
    sources: List[str] = Query(default=["ibn_kathir"]),
):
    """Get tafsir for a specific ayah from one or more sources."""
    if surah < 1 or surah > 114:
        raise HTTPException(status_code=400, detail="Surah must be 1-114")
    
    result = {
        "surah": surah,
        "ayah": ayah,
        "reference": f"{surah}:{ayah}",
        "tafsir": {}
    }
    
    # Get ayah text from annotations
    spans = get_all_spans()
    for span in spans:
        ref = span.get("reference", {})
        if ref.get("surah") == surah and ref.get("ayah") == ayah:
            result["ayah_text"] = span.get("text_ar", "")
            break
    
    # Get tafsir from each source
    for source in sources:
        tafsir_data = load_tafsir(source)
        ayat = tafsir_data.get("ayat", {})
        key = f"{surah}:{ayah}"
        
        if key in ayat:
            entry = ayat[key]
            result["tafsir"][source] = {
                "source_name": source.replace("_", " ").title(),
                "text": entry.get("text", entry.get("tafsir", "")),
                "word_count": len(entry.get("text", entry.get("tafsir", "")).split())
            }
    
    return result


@app.get("/tafsir/compare/{surah}/{ayah}")
async def compare_tafsir(surah: int, ayah: int):
    """Compare tafsir from all available sources for an ayah."""
    if surah < 1 or surah > 114:
        raise HTTPException(status_code=400, detail="Surah must be 1-114")
    
    # Available sources
    available_sources = []
    for f in TAFSIR_DIR.glob("*.json"):
        available_sources.append(f.stem)
    for f in TAFSIR_DIR.glob("*.jsonl"):
        available_sources.append(f.stem)
    
    if not available_sources:
        available_sources = ["ibn_kathir"]
    
    # Get tafsir from all sources
    result = {
        "surah": surah,
        "ayah": ayah,
        "reference": f"{surah}:{ayah}",
        "tafsir": {},
        "comparison": {
            "sources_count": 0,
            "sources_with_content": [],
            "word_counts": {}
        }
    }
    
    for source in available_sources:
        tafsir_data = load_tafsir(source)
        ayat = tafsir_data.get("ayat", {})
        key = f"{surah}:{ayah}"
        
        if key in ayat:
            entry = ayat[key]
            text = entry.get("text", entry.get("tafsir", ""))
            result["tafsir"][source] = {
                "source_name": source.replace("_", " ").title(),
                "text": text,
                "word_count": len(text.split())
            }
            result["comparison"]["sources_with_content"].append(source)
            result["comparison"]["word_counts"][source] = len(text.split())
    
    result["comparison"]["sources_count"] = len(result["comparison"]["sources_with_content"])
    
    return result


@app.get("/ayah/{surah}/{ayah}")
async def get_ayah(
    surah: int,
    ayah: int,
    include_annotations: bool = Query(default=True),
):
    """Get ayah text and metadata."""
    if surah < 1 or surah > 114:
        raise HTTPException(status_code=400, detail="Surah must be 1-114")
    
    result = {
        "surah": surah,
        "ayah": ayah,
        "reference": f"{surah}:{ayah}",
        "text_ar": "",
        "annotations": []
    }
    
    # Find ayah in spans
    spans = get_all_spans()
    for span in spans:
        ref = span.get("reference", {})
        if ref.get("surah") == surah and ref.get("ayah") == ayah:
            if not result["text_ar"]:
                result["text_ar"] = span.get("text_ar", "")
            if include_annotations:
                result["annotations"].append({
                    "id": span.get("span_id", span.get("id")),
                    "agent_type": span.get("agent", {}).get("type"),
                    "behavior_form": span.get("behavior_form"),
                    "evaluation": span.get("normative", {}).get("evaluation"),
                    "deontic_signal": span.get("normative", {}).get("deontic_signal"),
                })
    
    return result


@app.post("/api/spans/search")
async def search_spans_post(
    behavior_concept: Optional[str] = None,
    surah: Optional[int] = None,
    agent_type: Optional[str] = None,
    organ: Optional[str] = None,
    text_search: Optional[str] = None,
    limit: int = 20,
):
    """Search spans (POST endpoint for C1 tools)."""
    spans = get_all_spans()
    
    if surah:
        spans = [s for s in spans if s.get("reference", {}).get("surah") == surah]
    
    if agent_type:
        spans = [s for s in spans if s.get("agent", {}).get("type") == agent_type]
    
    if behavior_concept:
        spans = [s for s in spans if behavior_concept.lower() in s.get("behavior_form", "").lower()]
    
    if text_search:
        spans = [s for s in spans if text_search in s.get("text_ar", "")]
    
    return {"total": len(spans), "spans": spans[:limit]}
