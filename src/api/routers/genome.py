"""
Genome Router - /api/genome/*

Phase 7.3: Genome export endpoint (Q25 productization)

Exports the complete Quranic Behavioral Genome artifact:
- All behaviors (73+)
- All agents, organs, heart states, consequences
- All relationships (typed, evidence-backed)
- Provenance for every edge and mapping
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from ..dependencies import get_all_spans, DATA_DIR

router = APIRouter(prefix="/api/genome", tags=["Genome"])

# Genome version - increment when schema changes
GENOME_VERSION = "1.0.0"


def build_genome_artifact(spans: list) -> dict:
    """
    Build the complete Quranic Behavioral Genome artifact.
    
    Returns a structured artifact with:
    - All behaviors with verse evidence
    - All agents (AGT_*) with mappings
    - All organs (ORG_*) with mappings
    - All heart states with mappings
    - All relationships with provenance
    """
    # Initialize collections
    behaviors = {}
    agents = {}
    organs = {}
    heart_states = {}
    consequences = {}
    relationships = []
    
    # Process all spans
    for span in spans:
        # Extract behavior
        behavior = span.get("behavior_form", "")
        if behavior:
            if behavior not in behaviors:
                behaviors[behavior] = {
                    "name": behavior,
                    "verses": [],
                    "evaluations": set(),
                    "agents": set(),
                    "count": 0
                }
            
            ref = span.get("reference", {})
            verse_ref = f"{ref.get('surah', '?')}:{ref.get('ayah', '?')}"
            
            behaviors[behavior]["verses"].append({
                "surah": ref.get("surah"),
                "ayah": ref.get("ayah"),
                "surah_name": ref.get("surah_name", ""),
                "text": span.get("text", "")[:200],  # Limit text length
                "offset": span.get("offset", {})
            })
            behaviors[behavior]["count"] += 1
            
            # Track evaluation
            evaluation = span.get("normative", {}).get("evaluation", "")
            if evaluation:
                behaviors[behavior]["evaluations"].add(evaluation)
        
        # Extract agent
        agent_data = span.get("agent", {})
        agent_type = agent_data.get("type", "")
        if agent_type:
            if agent_type not in agents:
                agents[agent_type] = {
                    "type": agent_type,
                    "behaviors": set(),
                    "verses": [],
                    "count": 0
                }
            agents[agent_type]["behaviors"].add(behavior)
            agents[agent_type]["count"] += 1
            
            ref = span.get("reference", {})
            agents[agent_type]["verses"].append({
                "surah": ref.get("surah"),
                "ayah": ref.get("ayah")
            })
        
        # Extract organ
        organ = span.get("organ", "")
        if organ:
            if organ not in organs:
                organs[organ] = {
                    "name": organ,
                    "behaviors": set(),
                    "count": 0
                }
            organs[organ]["behaviors"].add(behavior)
            organs[organ]["count"] += 1
        
        # Extract heart state (from normative or context)
        heart_state = span.get("normative", {}).get("heart_state", "")
        if heart_state:
            if heart_state not in heart_states:
                heart_states[heart_state] = {
                    "name": heart_state,
                    "behaviors": set(),
                    "count": 0
                }
            heart_states[heart_state]["behaviors"].add(behavior)
            heart_states[heart_state]["count"] += 1
        
        # Extract consequence
        consequence = span.get("consequence", "")
        if consequence:
            if consequence not in consequences:
                consequences[consequence] = {
                    "name": consequence,
                    "behaviors": set(),
                    "count": 0
                }
            consequences[consequence]["behaviors"].add(behavior)
            consequences[consequence]["count"] += 1
        
        # Build relationships with evidence
        if behavior and agent_type:
            relationships.append({
                "type": "AGENT_PERFORMS",
                "source": agent_type,
                "target": behavior,
                "evidence": {
                    "surah": ref.get("surah"),
                    "ayah": ref.get("ayah"),
                    "offset": span.get("offset", {})
                }
            })
    
    # Convert sets to lists for JSON serialization
    for b in behaviors.values():
        b["evaluations"] = list(b["evaluations"])
        b["agents"] = list(b.get("agents", set()))
    
    for a in agents.values():
        a["behaviors"] = list(a["behaviors"])
        # Limit verses to first 10
        a["verses"] = a["verses"][:10]
    
    for o in organs.values():
        o["behaviors"] = list(o["behaviors"])
    
    for h in heart_states.values():
        h["behaviors"] = list(h["behaviors"])
    
    for c in consequences.values():
        c["behaviors"] = list(c["behaviors"])
    
    # Build artifact
    artifact = {
        "version": GENOME_VERSION,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "statistics": {
            "total_spans": len(spans),
            "unique_behaviors": len(behaviors),
            "unique_agents": len(agents),
            "unique_organs": len(organs),
            "unique_heart_states": len(heart_states),
            "unique_consequences": len(consequences),
            "total_relationships": len(relationships)
        },
        "behaviors": list(behaviors.values()),
        "agents": list(agents.values()),
        "organs": list(organs.values()),
        "heart_states": list(heart_states.values()),
        "consequences": list(consequences.values()),
        "relationships": relationships[:1000]  # Limit to first 1000 for API response
    }
    
    # Generate checksum for reproducibility
    content_str = json.dumps(artifact["statistics"], sort_keys=True)
    artifact["checksum"] = hashlib.sha256(content_str.encode()).hexdigest()[:16]
    
    return artifact


@router.get("/status")
async def genome_status():
    """Get genome export status and statistics."""
    spans = get_all_spans()
    
    # Quick stats without full build
    behaviors = set()
    agents = set()
    
    for span in spans:
        behavior = span.get("behavior_form", "")
        if behavior:
            behaviors.add(behavior)
        agent = span.get("agent", {}).get("type", "")
        if agent:
            agents.add(agent)
    
    return {
        "status": "ready",
        "version": GENOME_VERSION,
        "statistics": {
            "total_spans": len(spans),
            "unique_behaviors": len(behaviors),
            "unique_agents": len(agents)
        },
        "endpoints": {
            "full_export": "/api/genome/export",
            "behaviors_only": "/api/genome/behaviors",
            "agents_only": "/api/genome/agents",
            "relationships": "/api/genome/relationships"
        }
    }


@router.get("/export")
async def export_genome(
    format: str = Query("json", description="Export format: json or jsonl"),
    include_relationships: bool = Query(True, description="Include relationship edges")
):
    """
    Export the complete Quranic Behavioral Genome artifact.
    
    Q25 productization: Returns versioned, reproducible artifact with:
    - All behaviors with verse evidence
    - All agents, organs, heart states, consequences
    - All relationships with provenance
    """
    spans = get_all_spans()
    artifact = build_genome_artifact(spans)
    
    if not include_relationships:
        artifact["relationships"] = []
        artifact["statistics"]["total_relationships"] = 0
    
    return artifact


@router.get("/behaviors")
async def get_behaviors(
    limit: int = Query(100, ge=1, le=500, description="Max behaviors to return"),
    min_count: int = Query(1, ge=1, description="Minimum verse count")
):
    """Get all behaviors with verse counts."""
    spans = get_all_spans()
    
    behaviors = {}
    for span in spans:
        behavior = span.get("behavior_form", "")
        if behavior:
            if behavior not in behaviors:
                behaviors[behavior] = {
                    "name": behavior,
                    "count": 0,
                    "evaluations": set()
                }
            behaviors[behavior]["count"] += 1
            eval_type = span.get("normative", {}).get("evaluation", "")
            if eval_type:
                behaviors[behavior]["evaluations"].add(eval_type)
    
    # Filter and convert
    result = []
    for b in behaviors.values():
        if b["count"] >= min_count:
            result.append({
                "name": b["name"],
                "count": b["count"],
                "evaluations": list(b["evaluations"])
            })
    
    # Sort by count descending
    result.sort(key=lambda x: x["count"], reverse=True)
    
    return {
        "total": len(result),
        "behaviors": result[:limit]
    }


@router.get("/agents")
async def get_agents():
    """Get all agent types with behavior mappings."""
    spans = get_all_spans()
    
    agents = {}
    for span in spans:
        agent = span.get("agent", {}).get("type", "")
        behavior = span.get("behavior_form", "")
        if agent:
            if agent not in agents:
                agents[agent] = {
                    "type": agent,
                    "behaviors": set(),
                    "count": 0
                }
            agents[agent]["count"] += 1
            if behavior:
                agents[agent]["behaviors"].add(behavior)
    
    result = []
    for a in agents.values():
        result.append({
            "type": a["type"],
            "count": a["count"],
            "unique_behaviors": len(a["behaviors"]),
            "behaviors": list(a["behaviors"])[:20]  # Limit to 20
        })
    
    result.sort(key=lambda x: x["count"], reverse=True)
    
    return {
        "total": len(result),
        "agents": result
    }


@router.get("/relationships")
async def get_relationships(
    relationship_type: Optional[str] = Query(None, description="Filter by type"),
    limit: int = Query(100, ge=1, le=1000, description="Max relationships")
):
    """Get behavioral relationships with evidence."""
    spans = get_all_spans()
    
    relationships = []
    for span in spans:
        behavior = span.get("behavior_form", "")
        agent = span.get("agent", {}).get("type", "")
        ref = span.get("reference", {})
        
        if behavior and agent:
            rel = {
                "type": "AGENT_PERFORMS",
                "source": agent,
                "target": behavior,
                "evidence": {
                    "surah": ref.get("surah"),
                    "ayah": ref.get("ayah")
                }
            }
            
            if relationship_type is None or rel["type"] == relationship_type:
                relationships.append(rel)
    
    return {
        "total": len(relationships),
        "relationships": relationships[:limit]
    }
