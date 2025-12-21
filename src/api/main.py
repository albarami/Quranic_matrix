#!/usr/bin/env python3
"""
QBM REST API - Quranic Behavior Matrix Dataset Access

FastAPI backend for accessing the QBM dataset with filtering,
search, and statistics endpoints.
"""

from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
import json
from pathlib import Path as FilePath
from datetime import datetime
import re

from .models import (
    SpanAnnotation, DatasetResponse, DatasetMeta,
    SearchQuery, StatsResponse, HealthResponse,
    Reference, Agent, Behavior, Normative, Axes, Evidence
)

app = FastAPI(
    title="Quranic Behavior Matrix API",
    description="""
    REST API for accessing the Quranic Human-Behavior Classification Matrix (QBM) dataset.

    ## Dataset Tiers
    - **Gold**: Fully validated, reviewer-approved annotations
    - **Silver**: High-confidence annotations meeting IAA threshold
    - **Research**: All annotations including disputed cases

    ## Features
    - Query by surah, agent type, behavior form, evaluation
    - Full-text search in Arabic
    - Dataset statistics and distributions
    - Pagination support
    """,
    version="1.0.0",
    contact={
        "name": "QBM Project",
        "url": "https://github.com/quranic-behavior-matrix"
    },
    license_info={
        "name": "Research Use License",
        "url": "https://github.com/quranic-behavior-matrix/LICENSE"
    }
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Data paths
DATA_PATH = FilePath(__file__).parent.parent.parent / "data" / "exports"
_dataset_cache = {}


def load_dataset(tier: str = "silver") -> dict:
    """Load dataset from exports, with caching."""
    if tier in _dataset_cache:
        return _dataset_cache[tier]

    files = list(DATA_PATH.glob(f"qbm_{tier}_*.json"))
    if not files:
        return {"meta": {}, "spans": []}

    latest = sorted(files)[-1]
    with open(latest, encoding="utf-8") as f:
        data = json.load(f)

    _dataset_cache[tier] = data
    return data


def normalize_span(raw: dict) -> dict:
    """Normalize raw span data to API schema."""
    # Handle reference
    ref = raw.get("reference", {})
    if isinstance(ref, str):
        # Parse "1:1" or "1:1:1" format
        parts = ref.split(":")
        ref = {
            "surah": int(parts[0]) if len(parts) > 0 else 1,
            "ayah": int(parts[1]) if len(parts) > 1 else 1,
            "span": int(parts[2]) if len(parts) > 2 else None
        }

    # Build normalized structure
    return {
        "span_id": raw.get("span_id", raw.get("id", "")),
        "reference": ref,
        "raw_text_ar": raw.get("raw_text_ar", raw.get("text_ar", "")),
        "translation_en": raw.get("translation_en"),
        "agent": {
            "type": raw.get("agent", {}).get("type", raw.get("agent_type", "")),
            "label_en": raw.get("agent", {}).get("label_en"),
            "label_ar": raw.get("agent", {}).get("label_ar")
        },
        "behavior": {
            "form": raw.get("behavior_form", raw.get("behavior", {}).get("form", "")),
            "action_class": raw.get("action_class", raw.get("action", {}).get("class"))
        },
        "normative": {
            "speech_mode": raw.get("speech_mode", raw.get("normative", {}).get("speech_mode", "")),
            "evaluation": raw.get("evaluation", raw.get("normative", {}).get("evaluation", "")),
            "deontic_signal": raw.get("deontic_signal", raw.get("normative", {}).get("deontic_signal", "")),
            "textual_eval": raw.get("textual_eval", raw.get("action", {}).get("textual_eval"))
        },
        "axes": {
            "situational": raw.get("axes_situational", raw.get("axes", {}).get("situational", ""))
        } if raw.get("axes_situational") or raw.get("axes") else None,
        "evidence": {
            "support_type": raw.get("evidence_support_type", raw.get("evidence", {}).get("support_type", "")),
            "tafsir_consulted": raw.get("tafsir_consulted")
        } if raw.get("evidence_support_type") or raw.get("evidence") else None,
        "metadata": raw.get("metadata")
    }


@app.get("/", response_model=HealthResponse, tags=["Health"])
async def root():
    """Health check and API info."""
    data = load_dataset("silver")
    spans = data.get("spans", [])
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        dataset_loaded=len(spans) > 0,
        spans_available=len(spans)
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Detailed health check."""
    return await root()


@app.get("/datasets/{tier}", response_model=DatasetResponse, tags=["Datasets"])
async def get_dataset(
    tier: str = Path(..., description="Dataset tier", enum=["gold", "silver", "research"]),
    limit: int = Query(100, ge=1, le=1000, description="Maximum spans to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
):
    """
    Get full dataset by tier.

    Returns paginated list of span annotations with metadata.
    """
    data = load_dataset(tier)
    if not data.get("spans"):
        raise HTTPException(status_code=404, detail=f"Dataset tier '{tier}' not found")

    spans = data["spans"][offset:offset + limit]
    normalized = [normalize_span(s) for s in spans]

    meta = data.get("meta", {})
    return DatasetResponse(
        meta=DatasetMeta(
            tier=tier,
            version=meta.get("version", "1.0.0"),
            exported_at=datetime.fromisoformat(meta.get("exported_at", datetime.now().isoformat())),
            total_spans=len(data["spans"]),
            total_ayat=meta.get("total_ayat", len(set(f"{s.get('reference', {}).get('surah', 0)}:{s.get('reference', {}).get('ayah', 0)}" for s in data["spans"]))),
            total_surahs=meta.get("total_surahs", len(set(s.get("reference", {}).get("surah", 0) for s in data["spans"]))),
            avg_iaa_kappa=meta.get("avg_iaa_kappa")
        ),
        spans=normalized
    )


@app.get("/spans", tags=["Spans"])
async def search_spans(
    tier: str = Query("silver", description="Dataset tier", enum=["gold", "silver", "research"]),
    surah: Optional[int] = Query(None, ge=1, le=114, description="Filter by surah number"),
    agent_type: Optional[str] = Query(None, description="Filter by agent type (e.g., AGT_BELIEVER)"),
    behavior_form: Optional[str] = Query(None, description="Filter by behavior form"),
    evaluation: Optional[str] = Query(None, description="Filter by evaluation"),
    deontic_signal: Optional[str] = Query(None, description="Filter by deontic signal"),
    text_search: Optional[str] = Query(None, description="Search in Arabic text"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """
    Search and filter spans.

    Supports filtering by surah, agent type, behavior form, evaluation,
    deontic signal, and full-text search in Arabic.
    """
    data = load_dataset(tier)
    spans = data.get("spans", [])

    # Apply filters
    filtered = []
    for span in spans:
        # Surah filter
        if surah is not None:
            ref = span.get("reference", {})
            span_surah = ref.get("surah") if isinstance(ref, dict) else int(ref.split(":")[0])
            if span_surah != surah:
                continue

        # Agent type filter
        if agent_type:
            span_agent = span.get("agent", {}).get("type", span.get("agent_type", ""))
            if agent_type.upper() not in span_agent.upper():
                continue

        # Behavior form filter
        if behavior_form:
            span_form = span.get("behavior_form", span.get("behavior", {}).get("form", ""))
            if behavior_form.lower() not in span_form.lower():
                continue

        # Evaluation filter
        if evaluation:
            span_eval = span.get("evaluation", span.get("normative", {}).get("evaluation", ""))
            if evaluation.lower() not in span_eval.lower():
                continue

        # Deontic signal filter
        if deontic_signal:
            span_deontic = span.get("deontic_signal", span.get("normative", {}).get("deontic_signal", ""))
            if deontic_signal.lower() not in span_deontic.lower():
                continue

        # Text search
        if text_search:
            text = span.get("raw_text_ar", span.get("text_ar", ""))
            if text_search not in text:
                continue

        filtered.append(span)

    # Paginate
    paginated = filtered[offset:offset + limit]
    normalized = [normalize_span(s) for s in paginated]

    return {
        "total": len(filtered),
        "limit": limit,
        "offset": offset,
        "spans": normalized
    }


@app.get("/spans/{span_id}", tags=["Spans"])
async def get_span(
    span_id: str = Path(..., description="Span ID (e.g., QBM_00001)"),
    tier: str = Query("silver", description="Dataset tier")
):
    """Get a specific span by ID."""
    data = load_dataset(tier)

    for span in data.get("spans", []):
        sid = span.get("span_id", span.get("id", ""))
        if sid == span_id or sid.endswith(span_id):
            return normalize_span(span)

    raise HTTPException(status_code=404, detail=f"Span '{span_id}' not found")


@app.get("/surahs/{surah_num}", tags=["Surahs"])
async def get_surah_spans(
    surah_num: int = Path(..., ge=1, le=114, description="Surah number"),
    tier: str = Query("silver", description="Dataset tier"),
    limit: int = Query(500, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """Get all spans from a specific surah."""
    return await search_spans(tier=tier, surah=surah_num, limit=limit, offset=offset)


@app.get("/stats", response_model=StatsResponse, tags=["Statistics"])
async def get_stats(
    tier: str = Query("silver", description="Dataset tier")
):
    """
    Get dataset statistics.

    Returns total counts and distributions for key fields.
    """
    data = load_dataset(tier)
    spans = data.get("spans", [])

    if not spans:
        raise HTTPException(status_code=404, detail=f"No data for tier '{tier}'")

    # Calculate distributions
    agent_dist = {}
    behavior_dist = {}
    eval_dist = {}
    deontic_dist = {}
    surahs = set()
    ayat = set()

    for span in spans:
        # Agent
        agent = span.get("agent", {}).get("type", span.get("agent_type", "UNKNOWN"))
        agent_dist[agent] = agent_dist.get(agent, 0) + 1

        # Behavior
        behavior = span.get("behavior_form", span.get("behavior", {}).get("form", "UNKNOWN"))
        behavior_dist[behavior] = behavior_dist.get(behavior, 0) + 1

        # Evaluation
        evaluation = span.get("evaluation", span.get("normative", {}).get("evaluation", "UNKNOWN"))
        eval_dist[evaluation] = eval_dist.get(evaluation, 0) + 1

        # Deontic
        deontic = span.get("deontic_signal", span.get("normative", {}).get("deontic_signal", "UNKNOWN"))
        deontic_dist[deontic] = deontic_dist.get(deontic, 0) + 1

        # Coverage
        ref = span.get("reference", {})
        if isinstance(ref, dict):
            surahs.add(ref.get("surah"))
            ayat.add(f"{ref.get('surah')}:{ref.get('ayah')}")
        elif isinstance(ref, str):
            parts = ref.split(":")
            if len(parts) >= 2:
                surahs.add(int(parts[0]))
                ayat.add(f"{parts[0]}:{parts[1]}")

    meta = data.get("meta", {})

    return StatsResponse(
        total_spans=len(spans),
        total_ayat=len(ayat),
        surahs_covered=len(surahs),
        agent_distribution=dict(sorted(agent_dist.items(), key=lambda x: -x[1])),
        behavior_distribution=dict(sorted(behavior_dist.items(), key=lambda x: -x[1])),
        evaluation_distribution=dict(sorted(eval_dist.items(), key=lambda x: -x[1])),
        deontic_distribution=dict(sorted(deontic_dist.items(), key=lambda x: -x[1])),
        avg_iaa_kappa=meta.get("avg_iaa_kappa")
    )


@app.get("/vocabularies", tags=["Reference"])
async def get_vocabularies():
    """Get controlled vocabularies for all fields."""
    vocab_path = FilePath(__file__).parent.parent.parent / "config" / "controlled_vocabularies_v1.json"

    if not vocab_path.exists():
        # Return basic vocabularies
        return {
            "agent_types": [
                "AGT_ALLAH", "AGT_BELIEVER", "AGT_DISBELIEVER", "AGT_HYPOCRITE",
                "AGT_HUMAN_GENERAL", "AGT_PROPHET", "AGT_PROPHET_MUHAMMAD",
                "AGT_MESSENGER", "AGT_ANGEL", "AGT_SATAN", "AGT_JINN"
            ],
            "behavior_forms": [
                "inner_state", "speech_act", "physical_act", "relational_act",
                "trait_disposition", "mixed"
            ],
            "speech_modes": ["command", "prohibition", "informative", "narrative"],
            "evaluations": ["praise", "blame", "promise", "warning", "neutral"],
            "deontic_signals": ["amr", "nahy", "targhib", "tarhib", "khabar"]
        }

    with open(vocab_path, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
