"""
Proof Response Schema Contract v2 (Phase 9.10E+)

Canonical response contract for QBM proof system.
Both full and proof_only backends MUST produce responses that validate against this schema.

Key invariants:
- proof.intent and proof.mode ALWAYS present
- proof.tafsir is nested dict (not spread at proof level)
- debug.component_fallbacks.tafsir (not debug.tafsir_fallbacks)
- 7 tafsir sources for structured intents (SURAH_REF, AYAH_REF)

This schema is the SINGLE SOURCE OF TRUTH for API contract validation.
"""

from typing import Dict, List, Any, Optional, Literal, Union
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


# =============================================================================
# TAFSIR EVIDENCE MODELS
# =============================================================================

class TafsirChunk(BaseModel):
    """A single tafsir chunk with provenance."""
    source: str = Field(..., description="Mufassir ID (e.g., ibn_kathir)")
    verse_key: str = Field(..., description="surah:ayah format")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    text: str = Field(..., min_length=1, description="Chunk text content")
    char_start: Optional[int] = Field(None, ge=0, description="Start offset")
    char_end: Optional[int] = Field(None, gt=0, description="End offset")
    surah: Optional[int] = None
    ayah: Optional[int] = None


# TafsirSourceData is just a list of chunks - use List[TafsirChunk] directly
# No need for a separate model in Pydantic v2


# =============================================================================
# QURAN EVIDENCE MODELS
# =============================================================================

class QuranVerse(BaseModel):
    """A single Quran verse."""
    surah: int = Field(..., ge=1, le=114)
    ayah: int = Field(..., ge=1)
    text: str = Field(..., min_length=1)
    verse_key: Optional[str] = None


# =============================================================================
# DEBUG SCHEMA MODELS (CANONICAL)
# =============================================================================

class ComponentFallbacks(BaseModel):
    """
    Component-level fallback tracking.
    
    CANONICAL STRUCTURE - both backends must produce this exact shape.
    """
    quran: bool = False
    graph: bool = False
    taxonomy: bool = False
    tafsir: Dict[str, bool] = Field(
        default_factory=dict,
        description="Per-source fallback status: {source_name: bool}"
    )


class ProofDebugV2(BaseModel):
    """
    Canonical debug schema for proof responses.
    
    INVARIANTS:
    - Both full and proof_only backends produce this exact structure
    - component_fallbacks.tafsir is the canonical location for tafsir fallbacks
    - NO tafsir_fallbacks at top level (legacy pattern)
    """
    # Core fields
    fallback_used: bool = False
    fallback_reasons: List[str] = Field(default_factory=list)
    retrieval_distribution: Dict[str, int] = Field(default_factory=dict)
    primary_path_latency_ms: int = 0
    index_source: str = Field(..., description="disk | runtime_build | json_chunked")
    
    # Intent tracking
    intent: str = Field(..., description="SURAH_REF | AYAH_REF | CONCEPT_REF | FREE_TEXT")
    retrieval_mode: str = Field(..., description="hybrid | stratified | rag_only | deterministic_chunked")
    
    # Source coverage
    sources_covered: List[str] = Field(default_factory=list)
    core_sources_count: int = Field(0, ge=0, le=7)
    
    # Component fallbacks (CANONICAL STRUCTURE)
    component_fallbacks: ComponentFallbacks = Field(default_factory=ComponentFallbacks)
    
    # Optional fields (proof_only specific)
    fullpower_used: Optional[bool] = None
    
    @field_validator('intent')
    @classmethod
    def validate_intent(cls, v):
        valid_intents = {"SURAH_REF", "AYAH_REF", "CONCEPT_REF", "FREE_TEXT"}
        if v not in valid_intents:
            raise ValueError(f"intent must be one of {valid_intents}, got {v}")
        return v


# =============================================================================
# PROOF PAYLOAD MODELS (CANONICAL)
# =============================================================================

class TafsirEvidence(BaseModel):
    """
    Tafsir evidence organized by source.
    
    Each source key maps to a list of chunks.
    """
    ibn_kathir: List[Dict[str, Any]] = Field(default_factory=list)
    tabari: List[Dict[str, Any]] = Field(default_factory=list)
    qurtubi: List[Dict[str, Any]] = Field(default_factory=list)
    saadi: List[Dict[str, Any]] = Field(default_factory=list)
    jalalayn: List[Dict[str, Any]] = Field(default_factory=list)
    baghawi: List[Dict[str, Any]] = Field(default_factory=list)
    muyassar: List[Dict[str, Any]] = Field(default_factory=list)
    
    class Config:
        extra = "allow"  # Allow additional sources


class ProofPayloadV2(BaseModel):
    """
    Canonical proof payload structure.
    
    INVARIANTS:
    - intent and mode ALWAYS present (this was the original bug)
    - tafsir is a nested dict (not spread at proof level)
    - quran is a list of verse objects
    """
    # REQUIRED - these were missing in the original bug
    intent: str = Field(..., description="Query intent (SURAH_REF, AYAH_REF, etc.)")
    mode: str = Field(..., description="Response mode (summary, full)")
    
    # Evidence sections
    quran: List[Dict[str, Any]] = Field(default_factory=list)
    tafsir: Dict[str, Any] = Field(
        default_factory=dict,
        description="Nested tafsir evidence: {source_name: [chunks]}"
    )
    
    # Optional sections
    graph: Optional[Dict[str, Any]] = None
    taxonomy: Optional[Dict[str, Any]] = None
    statistics: Optional[Dict[str, Any]] = None
    
    class Config:
        extra = "allow"  # Allow additional fields


# =============================================================================
# TOP-LEVEL RESPONSE MODEL
# =============================================================================

class ProofResponseV2(BaseModel):
    """
    Canonical proof response contract v2.
    
    Both full and proof_only backends MUST produce responses that validate
    against this schema. This is the SINGLE SOURCE OF TRUTH.
    
    INVARIANTS:
    1. proof.intent always present
    2. proof.mode always present
    3. proof.tafsir is nested dict
    4. debug.component_fallbacks.tafsir is canonical fallback location
    5. No tafsir_fallbacks at debug top level
    """
    question: str = Field(..., description="Original query text")
    answer: str = Field(..., description="Generated or placeholder answer")
    proof: ProofPayloadV2 = Field(..., description="Evidence payload")
    debug: ProofDebugV2 = Field(..., description="Debug/tracing info")
    processing_time_ms: float = Field(..., ge=0, description="Total processing time in ms")
    
    # Optional fields
    validation: Optional[Dict[str, Any]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "سورة الفاتحة",
                "answer": "Evidence retrieved for Surah Al-Fatiha",
                "proof": {
                    "intent": "SURAH_REF",
                    "mode": "summary",
                    "quran": [{"surah": 1, "ayah": 1, "text": "..."}],
                    "tafsir": {
                        "ibn_kathir": [{"chunk_id": "...", "text": "..."}],
                        "tabari": [],
                        "qurtubi": [],
                        "saadi": [],
                        "jalalayn": [],
                        "baghawi": [],
                        "muyassar": []
                    }
                },
                "debug": {
                    "intent": "SURAH_REF",
                    "retrieval_mode": "deterministic_chunked",
                    "fallback_used": False,
                    "fallback_reasons": [],
                    "retrieval_distribution": {"ibn_kathir": 7},
                    "sources_covered": ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn", "baghawi", "muyassar"],
                    "core_sources_count": 7,
                    "component_fallbacks": {
                        "quran": False,
                        "graph": False,
                        "taxonomy": False,
                        "tafsir": {"ibn_kathir": False, "tabari": False}
                    },
                    "index_source": "json_chunked",
                    "primary_path_latency_ms": 150,
                    "fullpower_used": False
                },
                "processing_time_ms": 200
            }
        }


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_response_against_contract(response: Dict[str, Any]) -> tuple[bool, List[str]]:
    """
    Validate a response dict against the canonical ProofResponseV2 contract.
    
    Returns:
        (is_valid, list_of_issues)
    """
    issues = []
    
    # Try to parse with Pydantic
    try:
        ProofResponseV2(**response)
        return True, []
    except Exception as e:
        issues.append(f"Schema validation failed: {str(e)}")
    
    # Manual checks for specific invariants
    
    # Invariant 1: proof.intent present
    if "proof" in response:
        proof = response["proof"]
        if "intent" not in proof:
            issues.append("INVARIANT VIOLATION: proof.intent missing")
        if "mode" not in proof:
            issues.append("INVARIANT VIOLATION: proof.mode missing")
        
        # Invariant 3: tafsir is nested dict
        if "tafsir" in proof:
            if not isinstance(proof["tafsir"], dict):
                issues.append("INVARIANT VIOLATION: proof.tafsir must be a dict")
        
        # Check for spread pattern (tafsir sources at proof level)
        tafsir_sources = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn", "baghawi", "muyassar"]
        for source in tafsir_sources:
            if source in proof and source not in ["quran", "tafsir", "intent", "mode", "graph", "taxonomy", "statistics"]:
                issues.append(f"INVARIANT VIOLATION: tafsir source '{source}' at proof level (use proof.tafsir.{source})")
    
    # Invariant 4: debug.component_fallbacks.tafsir is canonical
    if "debug" in response:
        debug = response["debug"]
        
        # Check for legacy tafsir_fallbacks at top level
        if "tafsir_fallbacks" in debug:
            issues.append("INVARIANT VIOLATION: debug.tafsir_fallbacks at top level (use debug.component_fallbacks.tafsir)")
        
        # Check component_fallbacks structure
        if "component_fallbacks" in debug:
            cf = debug["component_fallbacks"]
            if "tafsir" not in cf:
                issues.append("INVARIANT VIOLATION: debug.component_fallbacks.tafsir missing")
            elif not isinstance(cf["tafsir"], dict):
                issues.append("INVARIANT VIOLATION: debug.component_fallbacks.tafsir must be a dict")
    
    return len(issues) == 0, issues


# =============================================================================
# CANONICAL CONSTANTS
# =============================================================================

# 7 tafsir sources guaranteed for structured intents
CANONICAL_TAFSIR_SOURCES = [
    "ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn", "baghawi", "muyassar"
]

# Valid query intents
CANONICAL_INTENTS = ["SURAH_REF", "AYAH_REF", "CONCEPT_REF", "FREE_TEXT"]

# Valid retrieval modes
CANONICAL_RETRIEVAL_MODES = ["hybrid", "stratified", "rag_only", "deterministic_chunked"]

# Structured intents that get 7-source guarantee
STRUCTURED_INTENTS = ["SURAH_REF", "AYAH_REF"]
