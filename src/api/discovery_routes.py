"""
Discovery API Routes for QBM.

GPU-accelerated semantic search, pattern discovery, cross-references, and clustering.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

# Lazy loading to avoid GPU initialization on import
_search_engine = None
_pattern_discovery = None
_cross_reference = None
_thematic_clustering = None


def get_search_engine():
    """Lazy load semantic search engine."""
    global _search_engine
    if _search_engine is None:
        from ..ai.discovery import SemanticSearchEngine
        _search_engine = SemanticSearchEngine(use_reranker=False)
    return _search_engine


def get_pattern_discovery():
    """Lazy load pattern discovery."""
    global _pattern_discovery
    if _pattern_discovery is None:
        from ..ai.discovery import PatternDiscovery
        _pattern_discovery = PatternDiscovery()
    return _pattern_discovery


def get_cross_reference():
    """Lazy load cross-reference finder."""
    global _cross_reference
    if _cross_reference is None:
        from ..ai.discovery import CrossReferenceFinder
        _cross_reference = CrossReferenceFinder()
    return _cross_reference


def get_thematic_clustering():
    """Lazy load thematic clustering."""
    global _thematic_clustering
    if _thematic_clustering is None:
        from ..ai.discovery import ThematicClustering
        _thematic_clustering = ThematicClustering(n_clusters=15)
    return _thematic_clustering


# Pydantic models for request/response
class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query (Arabic or English)")
    top_k: int = Field(10, ge=1, le=100, description="Number of results")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional filters")


class SearchResult(BaseModel):
    rank: int
    score: float
    source: str
    surah: int
    ayah: int
    type: str
    text: str
    behavior: str


class SearchResponse(BaseModel):
    query: str
    total: int
    results: List[Dict[str, Any]]


class CooccurrenceResponse(BaseModel):
    total_pairs: int
    min_cooccurrence: int
    pairs: List[Dict[str, Any]]


class SurahThemesResponse(BaseModel):
    surah: int
    themes: List[Dict[str, Any]]


class CrossReferenceResponse(BaseModel):
    surah: int
    ayah: int
    semantic_refs: List[Dict[str, Any]]
    behavior_refs: List[Dict[str, Any]]


class ClusterResponse(BaseModel):
    n_clusters: int
    silhouette_score: float
    clusters: Dict[int, Dict[str, Any]]


class BehaviorNetworkResponse(BaseModel):
    node_count: int
    edge_count: int
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]


# Router
router = APIRouter(prefix="/discovery", tags=["Discovery"])


@router.post("/search", response_model=SearchResponse)
async def semantic_search(request: SearchRequest):
    """
    GPU-accelerated semantic search across all annotations.
    
    Searches 322,939 annotations using AraBERT embeddings.
    """
    try:
        engine = get_search_engine()
        results = engine.search(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters,
        )
        return SearchResponse(
            query=request.query,
            total=len(results),
            results=results,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search")
async def semantic_search_get(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(10, ge=1, le=100),
    source: Optional[str] = Query(None, description="Filter by source"),
    surah: Optional[int] = Query(None, ge=1, le=114),
    type: Optional[str] = Query(None, description="Filter by type"),
):
    """GET endpoint for semantic search."""
    try:
        engine = get_search_engine()
        filters = {}
        if source:
            filters["source"] = source
        if surah:
            filters["surah"] = surah
        if type:
            filters["type"] = type

        results = engine.search(
            query=q,
            top_k=top_k,
            filters=filters if filters else None,
        )
        return {
            "query": q,
            "total": len(results),
            "results": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/similar/{surah}/{ayah}")
async def find_similar_ayat(
    surah: int,
    ayah: int,
    top_k: int = Query(10, ge=1, le=50),
):
    """Find ayat semantically similar to a given ayah."""
    if surah < 1 or surah > 114:
        raise HTTPException(status_code=400, detail="Surah must be 1-114")

    try:
        finder = get_cross_reference()
        similar = finder.find_similar_ayat(surah, ayah, top_k=top_k)
        return {
            "surah": surah,
            "ayah": ayah,
            "similar_ayat": similar,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patterns/cooccurrence", response_model=CooccurrenceResponse)
async def get_cooccurring_behaviors(
    min_count: int = Query(10, ge=1, description="Minimum co-occurrence count"),
    same_ayah: bool = Query(True, description="Same ayah only"),
    limit: int = Query(100, ge=1, le=1000),
):
    """Find behaviors that frequently co-occur."""
    try:
        discovery = get_pattern_discovery()
        pairs = discovery.find_cooccurring_behaviors(
            min_cooccurrence=min_count,
            same_ayah=same_ayah,
        )
        return CooccurrenceResponse(
            total_pairs=len(pairs),
            min_cooccurrence=min_count,
            pairs=pairs[:limit],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patterns/surah/{surah_num}", response_model=SurahThemesResponse)
async def get_surah_themes(
    surah_num: int,
    top_n: int = Query(5, ge=1, le=20),
):
    """Get dominant behavioral themes for a surah."""
    if surah_num < 1 or surah_num > 114:
        raise HTTPException(status_code=400, detail="Surah must be 1-114")

    try:
        discovery = get_pattern_discovery()
        all_themes = discovery.find_surah_themes(top_n=top_n)
        themes = all_themes.get(surah_num, [])
        return SurahThemesResponse(
            surah=surah_num,
            themes=themes,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patterns/distribution")
async def get_behavior_distribution(
    behavior_only: bool = Query(False, description="Only type=behavior"),
    limit: int = Query(50, ge=1, le=500),
):
    """Get distribution of behaviors across annotations."""
    try:
        discovery = get_pattern_discovery()
        dist = discovery.get_behavior_distribution(behavior_only=behavior_only)
        items = [{"behavior": k, "count": v} for k, v in list(dist.items())[:limit]]
        return {
            "total_unique": len(dist),
            "behavior_only": behavior_only,
            "distribution": items,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/crossref/{surah}/{ayah}", response_model=CrossReferenceResponse)
async def get_cross_references(
    surah: int,
    ayah: int,
    top_k: int = Query(10, ge=1, le=50),
):
    """Find cross-references for an ayah (semantic and behavior-based)."""
    if surah < 1 or surah > 114:
        raise HTTPException(status_code=400, detail="Surah must be 1-114")

    try:
        finder = get_cross_reference()
        refs = finder.find_cross_references(surah, ayah, method="both", top_k=top_k)
        return CrossReferenceResponse(
            surah=surah,
            ayah=ayah,
            semantic_refs=refs.get("semantic", []),
            behavior_refs=refs.get("behavior", []),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/crossref/behavior/{behavior}")
async def get_ayat_by_behavior(
    behavior: str,
    include_similar: bool = Query(True),
    limit: int = Query(100, ge=1, le=500),
):
    """Find all ayat discussing a specific behavior."""
    try:
        finder = get_cross_reference()
        ayat = finder.find_ayat_by_behavior(behavior, include_similar=include_similar)
        return {
            "behavior": behavior,
            "total": len(ayat),
            "ayat": ayat[:limit],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/network/behaviors", response_model=BehaviorNetworkResponse)
async def get_behavior_network(
    limit_edges: int = Query(500, ge=1, le=5000),
):
    """Get behavior co-occurrence network for visualization."""
    try:
        finder = get_cross_reference()
        network = finder.get_behavior_network()
        return BehaviorNetworkResponse(
            node_count=network["stats"]["behavior_count"],
            edge_count=min(len(network["edges"]), limit_edges),
            nodes=network["nodes"],
            edges=network["edges"][:limit_edges],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cluster")
async def run_clustering(
    n_clusters: int = Query(15, ge=2, le=100),
):
    """Run thematic clustering on annotations."""
    try:
        clustering = get_thematic_clustering()
        result = clustering.cluster_kmeans(n_clusters=n_clusters)
        themes = clustering.get_cluster_themes(top_n=5)
        return {
            "n_clusters": n_clusters,
            "silhouette_score": result["silhouette_score"],
            "cluster_sizes": result["cluster_sizes"],
            "themes": themes,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cluster/{cluster_id}/samples")
async def get_cluster_samples(
    cluster_id: int,
    n_samples: int = Query(10, ge=1, le=50),
):
    """Get sample annotations from a cluster."""
    try:
        clustering = get_thematic_clustering()
        if clustering.cluster_labels is None:
            clustering.cluster_kmeans(n_clusters=15)
        
        samples = clustering.get_cluster_samples(
            cluster_id=cluster_id,
            n_samples=n_samples,
            method="centroid",
        )
        return {
            "cluster_id": cluster_id,
            "n_samples": len(samples),
            "samples": samples,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_discovery_stats():
    """Get discovery system statistics."""
    try:
        stats = {}
        
        try:
            engine = get_search_engine()
            stats["search"] = engine.get_stats()
        except Exception:
            stats["search"] = {"status": "not_loaded"}

        try:
            discovery = get_pattern_discovery()
            stats["patterns"] = {
                "total_annotations": len(discovery.annotations),
                "unique_behaviors": len(discovery.behavior_index),
                "unique_ayat": len(discovery.ayah_index),
            }
        except Exception:
            stats["patterns"] = {"status": "not_loaded"}

        try:
            finder = get_cross_reference()
            stats["crossref"] = finder.get_stats()
        except Exception:
            stats["crossref"] = {"status": "not_loaded"}

        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
