"""
Proof Router - /api/proof/*

Phase 7.1: Modular API structure
Phase 7.2: Pagination + summary modes for SURAH_REF / CONCEPT_REF / AYAH_REF

Contains the Full Power proof system endpoints.
"""

import time
from typing import Optional, Literal
from fastapi import APIRouter, HTTPException, Request, Query
from pydantic import BaseModel, field_validator

router = APIRouter(prefix="/api/proof", tags=["Proof"])

# =============================================================================
# Request/Response Models
# =============================================================================

class ProofQueryRequest(BaseModel):
    """Request model for proof queries with input validation."""
    question: str
    # Phase 7.2: Pagination and summary mode parameters
    mode: Literal["summary", "full"] = "summary"
    page: int = 1
    page_size: int = 20
    per_ayah: bool = True  # For SURAH_REF: return per-ayah breakdown
    max_chunks_per_source: int = 1  # In summary mode, limit chunks per tafsir source
    
    @field_validator('question')
    @classmethod
    def validate_question(cls, v):
        if not v or len(v.strip()) < 2:
            raise ValueError("Question must be at least 2 characters")
        if len(v) > 1000:
            raise ValueError("Question must be less than 1000 characters")
        # Basic sanitization - reject obvious injection attempts
        dangerous_patterns = ['<script', 'javascript:', 'onclick=', 'onerror=']
        v_lower = v.lower()
        for pattern in dangerous_patterns:
            if pattern in v_lower:
                raise ValueError("Invalid characters in question")
        return v
    
    @field_validator('page')
    @classmethod
    def validate_page(cls, v):
        if v < 1:
            raise ValueError("Page must be >= 1")
        return v
    
    @field_validator('page_size')
    @classmethod
    def validate_page_size(cls, v):
        if v < 1 or v > 100:
            raise ValueError("page_size must be between 1 and 100")
        return v


# =============================================================================
# Global State (lazy initialization)
# =============================================================================

_full_power_system = None


def get_full_power_system():
    """
    Lazy initialization of Full Power System with index.
    
    Phase 10.1f: Fail-fast if FullPower is requested but index is missing.
    Returns 503 with clear error instead of silently degrading.
    """
    global _full_power_system
    if _full_power_system is None:
        import os
        from pathlib import Path
        
        # Phase 10.1f: Check if FullPower is explicitly enabled
        fullpower_ready = os.getenv("QBM_FULLPOWER_READY", "0") == "1"
        index_path = Path("data/indexes/full_power_index.npy")
        
        # If index exists, we can proceed even without explicit flag
        if not index_path.exists() and not fullpower_ready:
            raise RuntimeError(
                "FullPower index not found and QBM_FULLPOWER_READY not set. "
                f"Expected index at: {index_path.absolute()}. "
                "Either build the index with 'python -m src.ml.full_power_system' "
                "or set QBM_FULLPOWER_READY=1 to build on first request."
            )
        
        from src.ml.full_power_system import FullPowerQBMSystem
        from src.ml.mandatory_proof_system import integrate_with_system
        _full_power_system = FullPowerQBMSystem()
        _full_power_system.build_index()
        _full_power_system.build_graph()
        _full_power_system = integrate_with_system(_full_power_system)
    return _full_power_system


# =============================================================================
# Helper Functions
# =============================================================================

def filter_tafsir(quotes):
    """Filter out placeholder tafsir entries"""
    if not quotes:
        return []
    filtered = []
    for q in quotes:
        text = q.get("text", "")
        # Skip placeholder/empty entries
        if not text or "لا يوجد تفسير" in text or len(text.strip()) < 10:
            continue
        filtered.append({
            "surah": q.get("surah", "?"),
            "ayah": q.get("ayah", "?"),
            "text": text,
            "score": q.get("score", 0)
        })
    return filtered


def filter_quran_verses(verses, question):
    """Filter Quran verses to only include relevant ones"""
    if not verses:
        return []
    
    import re
    mentioned_surahs = set()
    surah_names = {
        'الحجرات': 49, 'الإسراء': 17, 'البقرة': 2, 'آل عمران': 3,
        'النساء': 4, 'المائدة': 5, 'الأنعام': 6, 'الأعراف': 7,
        'الأنفال': 8, 'التوبة': 9, 'يونس': 10, 'هود': 11,
        'يوسف': 12, 'الرعد': 13, 'إبراهيم': 14, 'الحجر': 15,
        'النحل': 16, 'الكهف': 18, 'مريم': 19, 'طه': 20,
    }
    for name, num in surah_names.items():
        if name in question:
            mentioned_surahs.add(num)
    
    num_matches = re.findall(r'سورة\s*(\d+)|surah\s*(\d+)', question.lower())
    for match in num_matches:
        for m in match:
            if m:
                mentioned_surahs.add(int(m))
    
    filtered = []
    for v in verses:
        text = v.get("text", "")
        if not text or len(text.strip()) < 10:
            continue
        
        relevance = v.get("relevance", 0)
        
        # Include all verses - filtering is now handled upstream by deterministic logic
        filtered.append({
            "surah": v.get("surah"),
            "ayah": v.get("ayah"),
            "text": text,
            "relevance": relevance
        })
    
    # Sort by relevance descending
    filtered.sort(key=lambda x: x.get("relevance", 0), reverse=True)
    return filtered[:20]


# =============================================================================
# Phase 7.2: Pagination and Summary Mode Helpers
# =============================================================================

def sort_deterministic(items: list, sort_key: str = "verse") -> list:
    """
    Phase 7.2 Fix B: Ensure deterministic ordering with tie-breakers.
    
    Sort keys:
    - "verse": Sort by (-score, surah, ayah, source, chunk_id)
    - "surah": Sort by (surah, ayah, source, chunk_id)
    - "score": Sort by (-score, verse_key, chunk_id)
    
    All orderings use chunk_id as final tie-breaker for stability.
    """
    if not items:
        return items
    
    def get_sort_key(item):
        score = item.get("score", 0) if isinstance(item.get("score"), (int, float)) else 0
        surah = int(item.get("surah", 0)) if item.get("surah") else 0
        ayah = int(item.get("ayah", 0)) if item.get("ayah") else 0
        source = item.get("source", "") or ""
        chunk_id = item.get("chunk_id", "") or item.get("id", "") or ""
        verse_key = f"{surah:03d}:{ayah:03d}"
        
        if sort_key == "verse":
            # Primary: -score (descending), then canonical verse order, then source, then chunk_id
            return (-score, surah, ayah, source, chunk_id)
        elif sort_key == "surah":
            # Primary: canonical surah:ayah order, then source, then chunk_id
            return (surah, ayah, source, chunk_id)
        else:  # "score"
            # Primary: -score (descending), then verse_key, then chunk_id
            return (-score, verse_key, chunk_id)
    
    return sorted(items, key=get_sort_key)


def paginate_list(items: list, page: int, page_size: int, sort_key: str = "verse") -> dict:
    """
    Paginate a list of items with deterministic ordering.
    
    Phase 7.2 Fix B: Ensures stable ordering across pages using tie-breakers.
    
    Returns:
        dict with items, page, page_size, total_items, total_pages, has_next, has_prev
    """
    # Apply deterministic sorting before pagination
    sorted_items = sort_deterministic(items, sort_key)
    
    total_items = len(sorted_items)
    total_pages = (total_items + page_size - 1) // page_size if page_size > 0 else 1
    
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_items = sorted_items[start_idx:end_idx]
    
    return {
        "items": page_items,
        "page": page,
        "page_size": page_size,
        "total_items": total_items,
        "total_pages": total_pages,
        "has_next": page < total_pages,
        "has_prev": page > 1,
        "sort_key": sort_key
    }


def apply_summary_mode(proof_data: dict, mode: str, max_chunks_per_source: int, per_ayah: bool) -> dict:
    """
    Phase 7.2: Apply summary mode to proof data.
    
    Summary mode:
    - Limits tafsir chunks per source to max_chunks_per_source
    - Groups by ayah if per_ayah=True
    - Never truncates silently - always shows totals
    
    Full mode:
    - Returns all data (caller handles pagination)
    """
    if mode == "full":
        return proof_data
    
    # Summary mode: limit chunks per source
    tafsir_sources = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"]
    
    summarized = proof_data.copy()
    
    for source in tafsir_sources:
        if source in summarized:
            full_list = summarized[source]
            summarized[source] = full_list[:max_chunks_per_source]
            # Add metadata about truncation
            if len(full_list) > max_chunks_per_source:
                summarized[f"{source}_total"] = len(full_list)
                summarized[f"{source}_truncated"] = True
    
    return summarized


def build_surah_summary(quran_verses: list, tafsir_data: dict, per_ayah: bool) -> dict:
    """
    Phase 7.2: Build per-ayah summary for SURAH_REF queries.
    
    Returns:
        dict with ayat list, each containing verse + 1 chunk per tafsir source
    """
    if not per_ayah:
        return {"ayat": quran_verses, "grouped": False}
    
    # Group by ayah
    ayah_map = {}
    for verse in quran_verses:
        ayah_key = f"{verse.get('surah')}:{verse.get('ayah')}"
        if ayah_key not in ayah_map:
            ayah_map[ayah_key] = {
                "surah": verse.get("surah"),
                "ayah": verse.get("ayah"),
                "surah_name": verse.get("surah_name", ""),
                "text": verse.get("text", ""),
                "tafsir": {}
            }
    
    # Add tafsir for each ayah (1 chunk per source)
    tafsir_sources = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"]
    for source in tafsir_sources:
        source_chunks = tafsir_data.get(source, [])
        for chunk in source_chunks:
            ayah_key = f"{chunk.get('surah')}:{chunk.get('ayah')}"
            if ayah_key in ayah_map and source not in ayah_map[ayah_key]["tafsir"]:
                ayah_map[ayah_key]["tafsir"][source] = chunk.get("text", "")[:500]  # Limit text length in summary
    
    # Sort by ayah number
    sorted_ayat = sorted(ayah_map.values(), key=lambda x: (int(x.get("surah", 0)), int(x.get("ayah", 0))))
    
    return {
        "ayat": sorted_ayat,
        "grouped": True,
        "total_ayat": len(sorted_ayat)
    }


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/query")
async def proof_query(request: Request, request_body: ProofQueryRequest):
    """
    Run a query through the Full Power QBM System.
    Returns answer with mandatory 13-component proof structure.
    
    Phase 7.2: Supports pagination and summary modes:
    - mode=summary (default): Returns per-ayah × per-source (1 chunk each)
    - mode=full: Returns all deterministic chunks, paginated
    - page, page_size: Pagination controls
    - per_ayah: Group results by ayah (for SURAH_REF)
    - max_chunks_per_source: Limit chunks per tafsir source in summary mode
    
    This endpoint powers the /proof page in the frontend.
    """
    start_time = time.time()
    
    try:
        system = get_full_power_system()
        result = system.answer_with_full_proof(request_body.question)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Extract proof components from dataclass
        proof = result.get("proof")
        
        # Get query intent from debug info
        debug_info = result.get("debug", {})
        intent = debug_info.get("intent", "FREE_TEXT")
        
        # Phase 7.2: Build base proof data
        quran_verses = filter_quran_verses(proof.quran.verses if proof else [], request_body.question)
        
        tafsir_data = {
            "ibn_kathir": filter_tafsir(proof.ibn_kathir.quotes if proof else []),
            "tabari": filter_tafsir(proof.tabari.quotes if proof else []),
            "qurtubi": filter_tafsir(proof.qurtubi.quotes if proof else []),
            "saadi": filter_tafsir(proof.saadi.quotes if proof else []),
            "jalalayn": filter_tafsir(proof.jalalayn.quotes if proof else []),
        }
        
        # Phase 7.2: Apply mode-specific processing
        if intent == "SURAH_REF" and request_body.per_ayah:
            # SURAH_REF summary: per-ayah × per-source breakdown
            surah_summary = build_surah_summary(quran_verses, tafsir_data, request_body.per_ayah)
            
            if request_body.mode == "summary":
                # Paginate the ayat
                paginated = paginate_list(
                    surah_summary["ayat"],
                    request_body.page,
                    request_body.page_size
                )
                proof_output = {
                    "quran": paginated,
                    "mode": "surah_summary",
                    "intent": intent,
                }
                # Add limited tafsir in summary mode
                for source, chunks in tafsir_data.items():
                    proof_output[source] = chunks[:request_body.max_chunks_per_source]
                    if len(chunks) > request_body.max_chunks_per_source:
                        proof_output[f"{source}_total"] = len(chunks)
            else:
                # Full mode: all ayat paginated
                paginated = paginate_list(
                    surah_summary["ayat"],
                    request_body.page,
                    request_body.page_size
                )
                proof_output = {
                    "quran": paginated,
                    "mode": "surah_full",
                    "intent": intent,
                    **tafsir_data
                }
        
        elif intent == "CONCEPT_REF":
            # CONCEPT_REF: deterministic verse list + tafsir
            if request_body.mode == "summary":
                proof_output = {
                    "quran": quran_verses[:request_body.page_size],
                    "quran_total": len(quran_verses),
                    "mode": "concept_summary",
                    "intent": intent,
                }
                for source, chunks in tafsir_data.items():
                    proof_output[source] = chunks[:request_body.max_chunks_per_source]
                    if len(chunks) > request_body.max_chunks_per_source:
                        proof_output[f"{source}_total"] = len(chunks)
            else:
                paginated = paginate_list(quran_verses, request_body.page, request_body.page_size)
                proof_output = {
                    "quran": paginated,
                    "mode": "concept_full",
                    "intent": intent,
                    **tafsir_data
                }
        
        else:
            # FREE_TEXT or AYAH_REF: standard output
            proof_output = {
                "quran": quran_verses,
                "mode": request_body.mode,
                "intent": intent,
                **tafsir_data
            }
        
        # Add remaining proof components
        proof_output.update({
            "graph": {
                "nodes": proof.graph.nodes if proof else [],
                "edges": proof.graph.edges if proof else [],
                "paths": proof.graph.paths if proof else []
            },
            "embeddings": {
                "similarities": proof.embeddings.similarities if proof else [],
                "clusters": proof.embeddings.clusters if proof else [],
                "nearest_neighbors": proof.embeddings.nearest_neighbors if proof else []
            },
            "rag_retrieval": {
                "query": proof.rag.query if proof else "",
                "retrieved_docs": proof.rag.retrieved_docs[:10] if proof else [],
                "sources_breakdown": proof.rag.sources_breakdown if proof else {}
            },
            "taxonomy": {
                "behaviors": proof.taxonomy.behaviors if proof else [],
                "dimensions": proof.taxonomy.dimensions if proof else {}
            },
            "statistics": {
                "counts": proof.statistics.counts if proof else {},
                "percentages": proof.statistics.percentages if proof else {}
            }
        })
        
        return {
            "question": request_body.question,
            "answer": result.get("answer", ""),
            "proof": proof_output,
            "validation": result.get("validation", {}),
            "processing_time_ms": round(processing_time, 2),
            "pagination": {
                "mode": request_body.mode,
                "page": request_body.page,
                "page_size": request_body.page_size,
                "per_ayah": request_body.per_ayah,
                "max_chunks_per_source": request_body.max_chunks_per_source
            },
            "debug": {
                **debug_info,
                "graph_backend": getattr(system, 'graph_backend', 'unknown'),
                "graph_backend_reason": getattr(system, 'graph_backend_reason', ''),
            }
        }
        
    except RuntimeError as e:
        # Phase 10.1f: Fail-fast with 503 for missing index
        if "FullPower index not found" in str(e):
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "full_power_index_missing",
                    "reason": str(e),
                    "resolution": "Build index with 'python -m src.ml.full_power_system' or set QBM_FULLPOWER_READY=1"
                }
            )
        raise HTTPException(
            status_code=500,
            detail=f"QBM Full Power System error: {str(e)}"
        )
    except Exception as e:
        import traceback
        raise HTTPException(
            status_code=500,
            detail=f"QBM Full Power System error: {str(e)}\n{traceback.format_exc()}"
        )


@router.get("/status")
async def proof_system_status():
    """Check if Full Power System is initialized and ready"""
    global _full_power_system
    return {
        "initialized": _full_power_system is not None,
        "ready": _full_power_system is not None,
        "components": {
            "gpu_count": 8,
            "vector_index": "107,646 vectors",
            "graph": "736,302 behavioral relations",
            "tafsir_sources": 5,
            "behaviors": 46
        }
    }
