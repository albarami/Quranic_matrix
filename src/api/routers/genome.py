"""
Genome Router - /api/genome/*

Phase 7.3: Genome export endpoint (Q25 productization)

Exports the complete Quranic Behavioral Genome artifact:
- All 73 behaviors from canonical_entities.json
- All agents, organs, heart states, consequences from canonical registry
- All relationships from semantic_graph_v2.json (typed, evidence-backed)
- Provenance for every edge (chunk_id, verse_key, source, char_start, char_end, quote)
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from ..dependencies import DATA_DIR

router = APIRouter(prefix="/api/genome", tags=["Genome"])

# Genome version - increment when schema changes
GENOME_VERSION = "2.0.0"

# Paths to canonical data sources
CANONICAL_ENTITIES_PATH = Path("vocab/canonical_entities.json")
SEMANTIC_GRAPH_PATH = DATA_DIR / "graph" / "semantic_graph_v2.json"

# Cache for loaded data
_canonical_entities_cache: Optional[Dict] = None
_semantic_graph_cache: Optional[Dict] = None


def load_canonical_entities() -> Dict:
    """Load canonical entities from vocab/canonical_entities.json."""
    global _canonical_entities_cache
    if _canonical_entities_cache is not None:
        return _canonical_entities_cache
    
    try:
        with open(CANONICAL_ENTITIES_PATH, 'r', encoding='utf-8') as f:
            _canonical_entities_cache = json.load(f)
        return _canonical_entities_cache
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail=f"Canonical entities file not found: {CANONICAL_ENTITIES_PATH}"
        )


def load_semantic_graph() -> Dict:
    """Load semantic graph from data/graph/semantic_graph_v2.json."""
    global _semantic_graph_cache
    if _semantic_graph_cache is not None:
        return _semantic_graph_cache
    
    try:
        with open(SEMANTIC_GRAPH_PATH, 'r', encoding='utf-8') as f:
            _semantic_graph_cache = json.load(f)
        return _semantic_graph_cache
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail=f"Semantic graph file not found: {SEMANTIC_GRAPH_PATH}"
        )


def build_q25_genome_artifact(mode: str = "full") -> dict:
    """
    Build the Q25 Quranic Behavioral Genome artifact from canonical sources.
    
    Sources:
    - vocab/canonical_entities.json: 73 behaviors, 14 agents, 26 organs, 12 heart states, 16 consequences
    - data/graph/semantic_graph_v2.json: Evidence-backed semantic edges with provenance
    
    Args:
        mode: "full" includes all evidence, "light" includes metadata only
    
    Returns:
        Complete genome artifact with all canonical entities and semantic relationships
    """
    canonical = load_canonical_entities()
    graph = load_semantic_graph()
    
    # Extract canonical entities
    behaviors = canonical.get("behaviors", [])
    agents = canonical.get("agents", [])
    organs = canonical.get("organs", [])
    heart_states = canonical.get("heart_states", [])
    consequences = canonical.get("consequences", [])
    axes_11 = canonical.get("axes_11", {})
    entity_types = canonical.get("entity_types", {})
    
    # Extract semantic graph data
    graph_nodes = graph.get("nodes", [])
    graph_edges = graph.get("edges", [])
    
    # Build edges with provenance
    edges_with_provenance = []
    for edge in graph_edges:
        edge_entry = {
            "source": edge.get("source"),
            "target": edge.get("target"),
            "edge_type": edge.get("edge_type"),
            "confidence": edge.get("confidence"),
            "evidence_count": edge.get("evidence_count", 0),
            "sources_count": edge.get("sources_count", 0),
            "cue_strength": edge.get("cue_strength"),
            "validation": edge.get("validation", {})
        }
        
        if mode == "full":
            # Include full evidence with provenance
            evidence_list = edge.get("evidence", [])
            edge_entry["evidence"] = [
                {
                    "source": ev.get("source"),
                    "verse_key": ev.get("verse_key"),
                    "chunk_id": ev.get("chunk_id"),
                    "char_start": ev.get("char_start"),
                    "char_end": ev.get("char_end"),
                    "quote": ev.get("quote", "")[:300],  # Limit quote length
                    "cue_phrase": ev.get("cue_phrase"),
                    "endpoints_validated": ev.get("endpoints_validated", [])
                }
                for ev in evidence_list[:5]  # Limit to 5 evidence items per edge
            ]
        
        edges_with_provenance.append(edge_entry)
    
    # Calculate statistics from canonical registry
    stats = {
        "canonical_behaviors": len(behaviors),
        "canonical_agents": len(agents),
        "canonical_organs": len(organs),
        "canonical_heart_states": len(heart_states),
        "canonical_consequences": len(consequences),
        "graph_nodes": len(graph_nodes),
        "semantic_edges": len(graph_edges),
        "axes_count": len(axes_11)
    }
    
    # Build canonical payload for checksum
    canonical_payload = {
        "behaviors": [b["id"] for b in behaviors],
        "agents": [a["id"] for a in agents],
        "organs": [o["id"] for o in organs],
        "heart_states": [h["id"] for h in heart_states],
        "consequences": [c["id"] for c in consequences],
        "edge_count": len(graph_edges)
    }
    checksum_input = json.dumps(canonical_payload, sort_keys=True)
    checksum = hashlib.sha256(checksum_input.encode()).hexdigest()
    
    artifact = {
        "version": GENOME_VERSION,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "mode": mode,
        "checksum": checksum,
        "source_versions": {
            "canonical_entities": canonical.get("version", "unknown"),
            "semantic_graph": graph.get("version", "unknown"),
            "canonical_entities_frozen": canonical.get("frozen_date", "unknown")
        },
        "statistics": stats,
        "entity_types": entity_types,
        "behaviors": behaviors,
        "agents": agents,
        "organs": organs,
        "heart_states": heart_states,
        "consequences": consequences,
        "axes_11": axes_11 if mode == "full" else {"count": len(axes_11)},
        "semantic_edges": edges_with_provenance if mode == "full" else [],
        "edge_types": graph.get("allowed_edge_types", []),
        "causal_edge_types": graph.get("causal_edge_types", [])
    }
    
    return artifact


@router.get("/status")
async def genome_status():
    """Get genome export status and statistics from canonical sources."""
    canonical = load_canonical_entities()
    graph = load_semantic_graph()
    
    entity_types = canonical.get("entity_types", {})
    
    return {
        "status": "ready",
        "version": GENOME_VERSION,
        "source_versions": {
            "canonical_entities": canonical.get("version", "unknown"),
            "semantic_graph": graph.get("version", "unknown")
        },
        "statistics": {
            "canonical_behaviors": entity_types.get("BEHAVIOR", {}).get("count", 0),
            "canonical_agents": entity_types.get("AGENT", {}).get("count", 0),
            "canonical_organs": entity_types.get("ORGAN", {}).get("count", 0),
            "canonical_heart_states": entity_types.get("HEART_STATE", {}).get("count", 0),
            "canonical_consequences": entity_types.get("CONSEQUENCE", {}).get("count", 0),
            "semantic_edges": len(graph.get("edges", []))
        },
        "endpoints": {
            "full_export": "/api/genome/export?mode=full",
            "light_export": "/api/genome/export?mode=light",
            "behaviors_only": "/api/genome/behaviors",
            "agents_only": "/api/genome/agents",
            "relationships": "/api/genome/relationships"
        }
    }


@router.get("/export")
async def export_genome(
    mode: str = Query("full", description="Export mode: 'full' (with evidence) or 'light' (metadata only)")
):
    """
    Export the complete Q25 Quranic Behavioral Genome artifact.
    
    Q25 productization: Returns versioned, reproducible artifact with:
    - All 73 canonical behaviors from vocab/canonical_entities.json
    - All 14 agents, 26 organs, 12 heart states, 16 consequences
    - All semantic edges from data/graph/semantic_graph_v2.json with provenance
    - Full checksum for reproducibility verification
    
    Modes:
    - full: Includes all evidence with provenance (chunk_id, verse_key, char_start, char_end, quote)
    - light: Metadata only (counts, entity IDs, no evidence payloads)
    """
    if mode not in ["full", "light"]:
        raise HTTPException(status_code=400, detail="mode must be 'full' or 'light'")
    
    artifact = build_q25_genome_artifact(mode=mode)
    return artifact


@router.get("/behaviors")
async def get_behaviors(
    category: Optional[str] = Query(None, description="Filter by category (speech, financial, emotional, etc.)"),
    limit: int = Query(100, ge=1, le=100, description="Max behaviors to return")
):
    """Get all 73 canonical behaviors from the registry."""
    canonical = load_canonical_entities()
    behaviors = canonical.get("behaviors", [])
    
    if category:
        behaviors = [b for b in behaviors if b.get("category") == category]
    
    # Get unique categories
    categories = list(set(b.get("category", "unknown") for b in canonical.get("behaviors", [])))
    
    return {
        "total": len(behaviors),
        "categories": sorted(categories),
        "behaviors": behaviors[:limit]
    }


@router.get("/agents")
async def get_agents():
    """Get all 14 canonical agent types from the registry."""
    canonical = load_canonical_entities()
    agents = canonical.get("agents", [])
    
    return {
        "total": len(agents),
        "agents": agents
    }


@router.get("/organs")
async def get_organs():
    """Get all 26 canonical organs from the registry."""
    canonical = load_canonical_entities()
    organs = canonical.get("organs", [])
    
    return {
        "total": len(organs),
        "organs": organs
    }


@router.get("/heart-states")
async def get_heart_states():
    """Get all 12 canonical heart states from the registry."""
    canonical = load_canonical_entities()
    heart_states = canonical.get("heart_states", [])
    
    return {
        "total": len(heart_states),
        "heart_states": heart_states
    }


@router.get("/consequences")
async def get_consequences():
    """Get all 16 canonical consequences from the registry."""
    canonical = load_canonical_entities()
    consequences = canonical.get("consequences", [])
    
    return {
        "total": len(consequences),
        "consequences": consequences
    }


@router.get("/relationships")
async def get_relationships(
    edge_type: Optional[str] = Query(None, description="Filter by edge type (CAUSES, LEADS_TO, PREVENTS, etc.)"),
    source: Optional[str] = Query(None, description="Filter by source node ID"),
    target: Optional[str] = Query(None, description="Filter by target node ID"),
    limit: int = Query(100, ge=1, le=1000, description="Max relationships to return")
):
    """
    Get semantic relationships with evidence from the graph.
    
    Edge types: CAUSES, LEADS_TO, PREVENTS, STRENGTHENS, OPPOSITE_OF, COMPLEMENTS, CONDITIONAL_ON
    """
    graph = load_semantic_graph()
    edges = graph.get("edges", [])
    
    # Apply filters
    if edge_type:
        edges = [e for e in edges if e.get("edge_type") == edge_type]
    if source:
        edges = [e for e in edges if e.get("source") == source]
    if target:
        edges = [e for e in edges if e.get("target") == target]
    
    # Build response with limited evidence
    result = []
    for edge in edges[:limit]:
        entry = {
            "source": edge.get("source"),
            "target": edge.get("target"),
            "edge_type": edge.get("edge_type"),
            "confidence": edge.get("confidence"),
            "evidence_count": edge.get("evidence_count", 0),
            "sources_count": edge.get("sources_count", 0),
            "cue_strength": edge.get("cue_strength"),
            "validation": edge.get("validation", {}),
            "sample_evidence": []
        }
        
        # Include first 2 evidence items as samples
        evidence_list = edge.get("evidence", [])[:2]
        for ev in evidence_list:
            entry["sample_evidence"].append({
                "source": ev.get("source"),
                "verse_key": ev.get("verse_key"),
                "chunk_id": ev.get("chunk_id"),
                "quote": ev.get("quote", "")[:200]
            })
        
        result.append(entry)
    
    return {
        "total_matching": len(edges),
        "returned": len(result),
        "edge_types": graph.get("allowed_edge_types", []),
        "relationships": result
    }
