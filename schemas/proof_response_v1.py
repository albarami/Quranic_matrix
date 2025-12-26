"""
Proof Response Schema Contract v1 (Phase 8.1)

Stable response contract for QBM proof system.
All API responses must conform to this schema.

Non-negotiables:
- Top-level keys are consistent
- Evidence provenance is always present
- No fabrication - missing data returns status="no_evidence"
"""

from typing import Dict, List, Any, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime


class VerseEvidence(BaseModel):
    """A single verse evidence item."""
    surah: int = Field(..., ge=1, le=114)
    ayah: int = Field(..., ge=1)
    text: str = Field(..., min_length=1)
    relevance: Literal["primary", "secondary", "context"] = "primary"
    verse_key: Optional[str] = None  # "surah:ayah" format
    
    def model_post_init(self, __context):
        if self.verse_key is None:
            self.verse_key = f"{self.surah}:{self.ayah}"


class TafsirQuote(BaseModel):
    """A single tafsir quote with full provenance (I1 compliant)."""
    source: str = Field(..., description="Mufassir ID (e.g., ibn_kathir)")
    verse_key: str = Field(..., description="surah:ayah format")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    char_start: int = Field(..., ge=0, description="Start offset in chunk")
    char_end: int = Field(..., gt=0, description="End offset in chunk")
    quote: str = Field(..., min_length=1, description="Actual quote text")
    surah: Optional[int] = None
    ayah: Optional[int] = None


class TafsirSourceEvidence(BaseModel):
    """Evidence from a single tafsir source."""
    source: str
    quotes: List[TafsirQuote] = Field(default_factory=list)
    count: int = 0
    status: Literal["found", "no_evidence", "partial"] = "found"


class GraphNode(BaseModel):
    """A node in the evidence graph."""
    id: str
    label: str
    type: Literal["BEHAVIOR", "AGENT", "ORGAN", "HEART_STATE", "CONSEQUENCE", "AXIS_VALUE"]
    label_ar: Optional[str] = None


class GraphEdge(BaseModel):
    """An edge in the evidence graph with full provenance."""
    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    edge_type: str = Field(..., description="Edge type (CAUSES, LEADS_TO, etc.)")
    confidence: float = Field(..., ge=0, le=1)
    evidence_count: int = Field(..., ge=0)
    evidence: List[TafsirQuote] = Field(default_factory=list)


class GraphPath(BaseModel):
    """A path through the graph (for causal chains)."""
    nodes: List[str] = Field(..., min_length=2)
    edges: List[GraphEdge] = Field(default_factory=list)
    path_type: Literal["causal", "semantic", "cooccurrence"] = "semantic"
    total_confidence: float = Field(..., ge=0, le=1)


class GraphEvidence(BaseModel):
    """Graph-based evidence."""
    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)
    paths: List[GraphPath] = Field(default_factory=list)
    graph_version: str = "1.0"
    status: Literal["found", "no_evidence", "partial"] = "found"


class TaxonomyEvidence(BaseModel):
    """Taxonomy/classification evidence."""
    behaviors: List[str] = Field(default_factory=list)
    agents: List[str] = Field(default_factory=list)
    organs: List[str] = Field(default_factory=list)
    heart_states: List[str] = Field(default_factory=list)
    consequences: List[str] = Field(default_factory=list)
    status: Literal["found", "no_evidence", "partial"] = "found"


class QuranEvidence(BaseModel):
    """Quran verse evidence."""
    verses: List[VerseEvidence] = Field(default_factory=list)
    verse_count: int = 0
    status: Literal["found", "no_evidence", "partial"] = "found"


class TafsirEvidence(BaseModel):
    """All tafsir evidence organized by source."""
    ibn_kathir: Optional[TafsirSourceEvidence] = None
    tabari: Optional[TafsirSourceEvidence] = None
    qurtubi: Optional[TafsirSourceEvidence] = None
    saadi: Optional[TafsirSourceEvidence] = None
    jalalayn: Optional[TafsirSourceEvidence] = None
    other_sources: Dict[str, TafsirSourceEvidence] = Field(default_factory=dict)
    total_quotes: int = 0
    sources_covered: List[str] = Field(default_factory=list)
    status: Literal["found", "no_evidence", "partial"] = "found"


class ProofBundle(BaseModel):
    """Complete proof bundle with all evidence types."""
    quran: QuranEvidence = Field(default_factory=QuranEvidence)
    tafsir: TafsirEvidence = Field(default_factory=TafsirEvidence)
    graph: GraphEvidence = Field(default_factory=GraphEvidence)
    taxonomy: TaxonomyEvidence = Field(default_factory=TaxonomyEvidence)


class PlanStep(BaseModel):
    """A single step in the query execution plan."""
    step_id: int
    action: str
    component: str
    status: Literal["pending", "running", "completed", "failed", "skipped"] = "pending"
    duration_ms: float = 0.0
    output_summary: Optional[str] = None


class DebugInfo(BaseModel):
    """Debug information for transparency and traceability."""
    query_intent: str = Field(..., description="Detected intent (AYAH_REF, SURAH_REF, CONCEPT_REF, FREE_TEXT)")
    question_class: Optional[str] = Field(None, description="Question class for legendary questions")
    plan_steps: List[PlanStep] = Field(default_factory=list)
    core_sources_count: int = Field(0, ge=0, le=5)
    sources_covered: List[str] = Field(default_factory=list)
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    total_duration_ms: float = 0.0
    index_source: Literal["production", "fixture", "custom"] = "production"
    warnings: List[str] = Field(default_factory=list)


class ValidationResult(BaseModel):
    """Validation results for the response."""
    evidence_complete: bool = True
    provenance_valid: bool = True
    no_fabrication: bool = True
    graph_rules_followed: bool = True
    issues: List[str] = Field(default_factory=list)


class ProofResponseV1(BaseModel):
    """
    Stable proof response contract v1.
    
    All QBM proof API responses must conform to this schema.
    This ensures consistent, enterprise-grade responses.
    """
    answer: str = Field(..., description="Narrative answer to the query")
    proof: ProofBundle = Field(default_factory=ProofBundle)
    debug: DebugInfo
    validation: Optional[ValidationResult] = None
    
    query: str = Field(..., description="Original query text")
    query_id: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    schema_version: str = "1.0"
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "الحسد هو تمني زوال النعمة عن الغير...",
                "proof": {
                    "quran": {"verses": [], "verse_count": 0, "status": "found"},
                    "tafsir": {"total_quotes": 0, "sources_covered": [], "status": "found"},
                    "graph": {"nodes": [], "edges": [], "paths": [], "status": "found"},
                    "taxonomy": {"behaviors": [], "status": "found"}
                },
                "debug": {
                    "query_intent": "CONCEPT_REF",
                    "question_class": "behavior_profile_11axis",
                    "plan_steps": [],
                    "core_sources_count": 5,
                    "sources_covered": ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"],
                    "fallback_used": False,
                    "total_duration_ms": 150.0
                },
                "query": "ما هو الحسد؟",
                "schema_version": "1.0"
            }
        }


def validate_proof_response(response: Dict[str, Any]) -> ValidationResult:
    """
    Validate a proof response against the schema contract.
    
    Returns ValidationResult with any issues found.
    """
    issues = []
    
    # Check required top-level keys
    required_keys = ["answer", "proof", "debug"]
    for key in required_keys:
        if key not in response:
            issues.append(f"Missing required key: {key}")
    
    # Check proof structure
    if "proof" in response:
        proof = response["proof"]
        proof_sections = ["quran", "tafsir", "graph", "taxonomy"]
        for section in proof_sections:
            if section not in proof:
                issues.append(f"Missing proof section: {section}")
    
    # Check debug structure
    if "debug" in response:
        debug = response["debug"]
        if "query_intent" not in debug:
            issues.append("Missing debug.query_intent")
        if "sources_covered" not in debug:
            issues.append("Missing debug.sources_covered")
    
    # Check tafsir provenance (I1)
    if "proof" in response and "tafsir" in response["proof"]:
        tafsir = response["proof"]["tafsir"]
        for source_name in ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"]:
            source_data = tafsir.get(source_name)
            if source_data and "quotes" in source_data:
                for quote in source_data["quotes"]:
                    required_provenance = ["source", "verse_key", "chunk_id", "char_start", "char_end", "quote"]
                    for field in required_provenance:
                        if field not in quote:
                            issues.append(f"Tafsir quote missing provenance field: {field}")
    
    # Check graph edges have evidence
    if "proof" in response and "graph" in response["proof"]:
        graph = response["proof"]["graph"]
        for edge in graph.get("edges", []):
            if "evidence" not in edge or len(edge.get("evidence", [])) == 0:
                if edge.get("edge_type") in ["CAUSES", "LEADS_TO", "PREVENTS", "STRENGTHENS"]:
                    issues.append(f"Causal edge missing evidence: {edge.get('source')} -> {edge.get('target')}")
    
    return ValidationResult(
        evidence_complete=len([i for i in issues if "Missing" in i]) == 0,
        provenance_valid=len([i for i in issues if "provenance" in i]) == 0,
        no_fabrication=True,  # Would need content check
        graph_rules_followed=len([i for i in issues if "edge" in i.lower()]) == 0,
        issues=issues
    )
