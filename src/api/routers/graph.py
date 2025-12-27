"""
Graph Router - /api/graph/*

Phase 7.1: Modular API structure
Contains graph traversal and relationship endpoints.
"""

from fastapi import APIRouter, HTTPException

from ..dependencies import get_all_spans
from ..unified_graph import get_unified_graph

router = APIRouter(prefix="/api/graph", tags=["Graph"])


@router.get("/status")
async def graph_status():
    """Get graph status and statistics."""
    spans = get_all_spans()
    graph = get_unified_graph(spans)
    
    if not graph:
        return {"status": "unavailable", "nodes": 0, "edges": 0}
    
    return {
        "status": "ready",
        "nodes": graph.node_count() if hasattr(graph, 'node_count') else 0,
        "edges": graph.edge_count() if hasattr(graph, 'edge_count') else 0,
        "types": graph.get_edge_types() if hasattr(graph, 'get_edge_types') else []
    }


@router.get("/verse/{surah}/{ayah}")
async def get_verse_graph(surah: int, ayah: int):
    """Get graph data for a specific verse."""
    if surah < 1 or surah > 114:
        raise HTTPException(status_code=400, detail="Surah must be between 1 and 114")
    
    spans = get_all_spans()
    graph = get_unified_graph(spans)
    
    if not graph:
        raise HTTPException(status_code=503, detail="Graph not available")
    
    verse_key = f"{surah}:{ayah}"
    
    try:
        node_data = graph.get_verse_node(verse_key)
        return {
            "verse": verse_key,
            "behaviors": node_data.get("behaviors", []) if node_data else [],
            "connections": node_data.get("connections", []) if node_data else []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/behavior/{behavior}")
async def get_behavior_graph(behavior: str):
    """Get graph data for a specific behavior."""
    spans = get_all_spans()
    graph = get_unified_graph(spans)
    
    if not graph:
        raise HTTPException(status_code=503, detail="Graph not available")
    
    try:
        node_data = graph.get_node(behavior)
        if not node_data:
            raise HTTPException(status_code=404, detail=f"Behavior '{behavior}' not found in graph")
        
        return {
            "behavior": behavior,
            "causes": node_data.get("causes", []),
            "effects": node_data.get("effects", []),
            "opposites": node_data.get("opposites", []),
            "related": node_data.get("related", []),
            "verse_count": node_data.get("verse_count", 0)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/traverse/{node_id}")
async def traverse_graph(node_id: str, depth: int = 2, direction: str = "both"):
    """Traverse graph from a starting node."""
    if depth < 1 or depth > 5:
        raise HTTPException(status_code=400, detail="Depth must be between 1 and 5")
    if direction not in ["in", "out", "both"]:
        raise HTTPException(status_code=400, detail="Direction must be 'in', 'out', or 'both'")
    
    spans = get_all_spans()
    graph = get_unified_graph(spans)
    
    if not graph:
        raise HTTPException(status_code=503, detail="Graph not available")
    
    try:
        result = graph.traverse(node_id, depth=depth, direction=direction)
        return {
            "start_node": node_id,
            "depth": depth,
            "direction": direction,
            "nodes": result.get("nodes", []),
            "edges": result.get("edges", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/path")
async def find_path(start: str, end: str, max_depth: int = 5):
    """Find path between two nodes."""
    if max_depth < 1 or max_depth > 10:
        raise HTTPException(status_code=400, detail="max_depth must be between 1 and 10")
    
    spans = get_all_spans()
    graph = get_unified_graph(spans)
    
    if not graph:
        raise HTTPException(status_code=503, detail="Graph not available")
    
    try:
        path = graph.find_path(start, end, max_depth=max_depth)
        return {
            "start": start,
            "end": end,
            "found": len(path) > 0,
            "path": path,
            "length": len(path)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
