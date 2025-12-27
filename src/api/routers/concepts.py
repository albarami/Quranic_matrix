"""
Concepts Router - /api/behavior/*, /api/analyze/*

Phase 7.1: Modular API structure
Contains behavior/concept analysis endpoints.
"""

from fastapi import APIRouter, HTTPException

from ..dependencies import get_all_spans
from ..unified_brain import get_brain
from ..unified_graph import get_unified_graph

router = APIRouter(tags=["Concepts"])


@router.get("/api/behavior/profile/{behavior}")
async def get_behavior_profile(behavior: str):
    """
    Complete Behavior Profile - ملف السلوك الشامل
    
    Returns EVERYTHING about a behavior systematically across all 11 dimensions.
    """
    import time
    start_time = time.time()
    
    spans = get_all_spans()
    brain = get_brain(spans)
    graph = get_unified_graph(spans)
    
    # 1. Get all verses for this behavior
    behavior_lower = behavior.lower().strip()
    matching_spans = []
    verses_set = set()
    
    for span in spans:
        behavior_form = span.get("behavior_form", "").lower()
        if behavior_lower in behavior_form or behavior_form in behavior_lower:
            matching_spans.append(span)
            ref = span.get("reference", {})
            if ref.get("surah") and ref.get("ayah"):
                verses_set.add((ref["surah"], ref["ayah"]))
    
    # 2. Build verse details with metadata
    verses = []
    for span in matching_spans:
        ref = span.get("reference", {})
        verses.append({
            "surah": ref.get("surah"),
            "ayah": ref.get("ayah"),
            "surah_name": ref.get("surah_name"),
            "text": span.get("text", ""),
            "agent": span.get("agent", {}).get("type"),
            "evaluation": span.get("normative", {}).get("evaluation"),
        })
    
    # 3. Get graph relationships
    relationships = {
        "causes": [],
        "effects": [],
        "opposites": [],
        "related": []
    }
    
    if graph:
        try:
            node_data = graph.get_node(behavior)
            if node_data:
                relationships["causes"] = node_data.get("causes", [])
                relationships["effects"] = node_data.get("effects", [])
                relationships["opposites"] = node_data.get("opposites", [])
                relationships["related"] = node_data.get("related", [])
        except Exception:
            pass
    
    # 4. Get tafsir mentions
    tafsir_mentions = []
    if brain:
        try:
            tafsir_results = brain.search_tafsir_by_behavior(behavior)
            tafsir_mentions = tafsir_results[:20]
        except Exception:
            pass
    
    processing_time = (time.time() - start_time) * 1000
    
    return {
        "behavior": behavior,
        "verse_count": len(verses_set),
        "span_count": len(matching_spans),
        "verses": verses[:50],  # Limit to 50
        "relationships": relationships,
        "tafsir_mentions": tafsir_mentions,
        "processing_time_ms": round(processing_time, 2)
    }


@router.get("/api/behavior/list")
async def list_behaviors():
    """List all unique behaviors in the dataset."""
    spans = get_all_spans()
    
    behaviors = {}
    for span in spans:
        behavior = span.get("behavior_form", "")
        if behavior:
            if behavior not in behaviors:
                behaviors[behavior] = {"count": 0, "evaluations": set()}
            behaviors[behavior]["count"] += 1
            eval_type = span.get("normative", {}).get("evaluation", "")
            if eval_type:
                behaviors[behavior]["evaluations"].add(eval_type)
    
    result = []
    for behavior, data in sorted(behaviors.items(), key=lambda x: x[1]["count"], reverse=True):
        result.append({
            "behavior": behavior,
            "count": data["count"],
            "evaluations": list(data["evaluations"])
        })
    
    return {
        "total": len(result),
        "behaviors": result
    }


@router.get("/api/analyze/behavior")
async def analyze_behavior_endpoint(behavior: str):
    """Analyze a specific behavior."""
    return await get_behavior_profile(behavior)


@router.get("/api/analyze/behavior/{behavior}")
async def analyze_behavior_by_path(behavior: str):
    """Analyze a specific behavior (path parameter version)."""
    return await get_behavior_profile(behavior)


@router.get("/api/analyze/statistics")
async def get_statistics():
    """Get dataset statistics."""
    spans = get_all_spans()
    
    # Count by various dimensions
    by_surah = {}
    by_agent = {}
    by_evaluation = {}
    by_behavior = {}
    
    for span in spans:
        # By surah
        surah = span.get("reference", {}).get("surah")
        if surah:
            by_surah[surah] = by_surah.get(surah, 0) + 1
        
        # By agent
        agent = span.get("agent", {}).get("type")
        if agent:
            by_agent[agent] = by_agent.get(agent, 0) + 1
        
        # By evaluation
        evaluation = span.get("normative", {}).get("evaluation")
        if evaluation:
            by_evaluation[evaluation] = by_evaluation.get(evaluation, 0) + 1
        
        # By behavior
        behavior = span.get("behavior_form")
        if behavior:
            by_behavior[behavior] = by_behavior.get(behavior, 0) + 1
    
    return {
        "total_spans": len(spans),
        "unique_surahs": len(by_surah),
        "unique_behaviors": len(by_behavior),
        "by_surah": dict(sorted(by_surah.items())),
        "by_agent": dict(sorted(by_agent.items(), key=lambda x: x[1], reverse=True)),
        "by_evaluation": by_evaluation,
        "top_behaviors": dict(sorted(by_behavior.items(), key=lambda x: x[1], reverse=True)[:20])
    }
