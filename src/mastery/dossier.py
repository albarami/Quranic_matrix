"""
Behavior Dossier Schema for QBM.

Defines the complete dossier structure for each of the 87 behaviors:
- Identity (ID, labels, roots, stems)
- Vocabulary mastery (synonyms, antonyms, variants)
- Evidence mastery (Quran verses, tafsir chunks)
- Relationship mastery (edges, causal paths)
- Bouzidani context matrix (5 axes + intention + recurrence)
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class VerseEvidence:
    """Quran verse evidence with relevance classification."""
    verse_key: str           # e.g., "2:153"
    text_ar: str             # Arabic verse text
    text_en: str             # English translation
    relevance: str           # direct, indirect, supporting
    confidence: float        # 0.0-1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TafsirChunk:
    """Tafsir chunk with full provenance."""
    source: str              # ibn_kathir, tabari, etc.
    chunk_id: str            # Unique chunk identifier
    verse_key: str           # Associated verse
    text_ar: str             # Arabic tafsir text
    char_start: int          # Character offset start
    char_end: int            # Character offset end
    relevance: str           # direct, indirect, supporting
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RelationshipEdge:
    """Relationship edge with evidence."""
    target_id: str           # Target behavior/entity ID
    target_label_ar: str     # Arabic label
    target_label_en: str     # English label
    edge_type: str           # CAUSES, LEADS_TO, PREVENTS, etc.
    direction: str           # outgoing, incoming
    evidence_count: int
    evidence_refs: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CausalPath:
    """Multi-hop causal path with evidence per edge."""
    path_id: str
    nodes: List[str]         # List of behavior IDs in path
    edges: List[Dict[str, Any]]  # Edge info with evidence
    total_hops: int
    total_evidence: int
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ContextAssertion:
    """
    Context assertion with citation - no invented assignments.
    
    Every non-unknown context value must cite its source:
    - verse: Quranic verse reference
    - tafsir: Tafsir chunk reference
    - rule: Deterministic rule ID
    - unknown: No evidence found
    """
    value: str                       # The context value or "unknown"
    citation_type: str               # "verse", "tafsir", "rule", "unknown"
    citation_ref: Optional[str] = None  # verse_key, chunk_id, or rule_id
    confidence: float = 0.0          # 0.0-1.0
    reason_if_unknown: Optional[str] = None  # Required if value is "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "value": self.value,
            "citation_type": self.citation_type,
            "confidence": self.confidence,
        }
        if self.citation_ref:
            result["citation_ref"] = self.citation_ref
        if self.reason_if_unknown:
            result["reason_if_unknown"] = self.reason_if_unknown
        return result
    
    @classmethod
    def unknown(cls, reason: str) -> "ContextAssertion":
        """Create an unknown context assertion with reason."""
        return cls(
            value="unknown",
            citation_type="unknown",
            citation_ref=None,
            confidence=0.0,
            reason_if_unknown=reason,
        )
    
    @classmethod
    def from_verse(cls, value: str, verse_key: str, confidence: float = 0.8) -> "ContextAssertion":
        """Create a context assertion from verse evidence."""
        return cls(
            value=value,
            citation_type="verse",
            citation_ref=verse_key,
            confidence=confidence,
        )
    
    @classmethod
    def from_rule(cls, value: str, rule_id: str, confidence: float = 1.0) -> "ContextAssertion":
        """Create a context assertion from deterministic rule."""
        return cls(
            value=value,
            citation_type="rule",
            citation_ref=rule_id,
            confidence=confidence,
        )


@dataclass
class BouzidaniContexts:
    """
    Bouzidani Five-Context Framework (مصفوفة بوزيداني للسياقات الخمسة).
    
    Every context assertion must cite its source - no invented assignments.
    """
    # Axis 1: Organic/Biological (التصنيف العضوي البيولوجي)
    organ_links: List[ContextAssertion] = field(default_factory=list)  # ORG_HEART, ORG_TONGUE, etc.
    internal_external: ContextAssertion = field(default_factory=lambda: ContextAssertion.unknown("no_evidence"))
    
    # Axis 2: Situational (التصنيف الموضعي)
    situational_context: ContextAssertion = field(default_factory=lambda: ContextAssertion.unknown("no_evidence"))
    
    # Axis 3: Systemic (التصنيف النسقي)
    systemic_context: ContextAssertion = field(default_factory=lambda: ContextAssertion.unknown("no_evidence"))
    
    # Axis 4: Spatial (التصنيف المكاني)
    spatial_context: ContextAssertion = field(default_factory=lambda: ContextAssertion.unknown("no_evidence"))
    
    # Axis 5: Temporal (التصنيف الزماني)
    temporal_context: ContextAssertion = field(default_factory=lambda: ContextAssertion.unknown("no_evidence"))
    
    # Additional dimensions
    intention_niyyah: ContextAssertion = field(default_factory=lambda: ContextAssertion.unknown("no_evidence"))
    recurrence_dawrah: ContextAssertion = field(default_factory=lambda: ContextAssertion.unknown("no_evidence"))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "organ_links": [o.to_dict() for o in self.organ_links],
            "internal_external": self.internal_external.to_dict(),
            "situational_context": self.situational_context.to_dict(),
            "systemic_context": self.systemic_context.to_dict(),
            "spatial_context": self.spatial_context.to_dict(),
            "temporal_context": self.temporal_context.to_dict(),
            "intention_niyyah": self.intention_niyyah.to_dict(),
            "recurrence_dawrah": self.recurrence_dawrah.to_dict(),
        }
    
    def get_unknown_fields(self) -> List[str]:
        """Get list of fields that are unknown."""
        unknown = []
        if self.internal_external.value == "unknown":
            unknown.append("internal_external")
        if self.situational_context.value == "unknown":
            unknown.append("situational_context")
        if self.systemic_context.value == "unknown":
            unknown.append("systemic_context")
        if self.spatial_context.value == "unknown":
            unknown.append("spatial_context")
        if self.temporal_context.value == "unknown":
            unknown.append("temporal_context")
        if self.intention_niyyah.value == "unknown":
            unknown.append("intention_niyyah")
        if self.recurrence_dawrah.value == "unknown":
            unknown.append("recurrence_dawrah")
        return unknown


@dataclass
class EvidenceStats:
    """Statistics about evidence coverage."""
    quran_verse_count: int = 0
    quran_direct_count: int = 0
    quran_indirect_count: int = 0
    tafsir_chunk_count: int = 0
    tafsir_sources_covered: List[str] = field(default_factory=list)
    relationship_count: int = 0
    causal_path_count: int = 0
    context_known_count: int = 0
    context_unknown_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BehaviorDossier:
    """
    Complete dossier for a single behavior.
    
    Contains all mastery information:
    - Identity and vocabulary
    - Evidence (Quran + tafsir)
    - Relationships and causal paths
    - Bouzidani context matrix
    """
    # Identity
    behavior_id: str           # BEH_EMO_PATIENCE
    label_ar: str              # الصبر
    label_en: str              # Patience
    category: str              # emotion, speech, social, etc.
    roots: List[str] = field(default_factory=list)  # صبر
    stems: List[str] = field(default_factory=list)  # صابر، صبور، اصطبر
    
    # Vocabulary Mastery
    synonyms_ar: List[str] = field(default_factory=list)
    synonyms_en: List[str] = field(default_factory=list)
    antonyms: List[str] = field(default_factory=list)
    morphological_variants: List[str] = field(default_factory=list)
    quranic_phrases: List[str] = field(default_factory=list)
    
    # Evidence Mastery
    quran_evidence: List[VerseEvidence] = field(default_factory=list)
    tafsir_evidence: Dict[str, List[TafsirChunk]] = field(default_factory=dict)
    
    # Relationship Mastery
    outgoing_edges: List[RelationshipEdge] = field(default_factory=list)
    incoming_edges: List[RelationshipEdge] = field(default_factory=list)
    causal_paths: List[CausalPath] = field(default_factory=list)
    
    # Bouzidani Context Matrix
    bouzidani_contexts: BouzidaniContexts = field(default_factory=BouzidaniContexts)
    
    # Metadata
    evidence_stats: EvidenceStats = field(default_factory=EvidenceStats)
    completeness_score: float = 0.0
    missing_fields: List[str] = field(default_factory=list)
    generated_at: str = ""
    schema_version: str = "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "behavior_id": self.behavior_id,
            "label_ar": self.label_ar,
            "label_en": self.label_en,
            "category": self.category,
            "roots": self.roots,
            "stems": self.stems,
            "synonyms_ar": self.synonyms_ar,
            "synonyms_en": self.synonyms_en,
            "antonyms": self.antonyms,
            "morphological_variants": self.morphological_variants,
            "quranic_phrases": self.quranic_phrases,
            "quran_evidence": [e.to_dict() for e in self.quran_evidence],
            "tafsir_evidence": {
                src: [c.to_dict() for c in chunks]
                for src, chunks in self.tafsir_evidence.items()
            },
            "outgoing_edges": [e.to_dict() for e in self.outgoing_edges],
            "incoming_edges": [e.to_dict() for e in self.incoming_edges],
            "causal_paths": [p.to_dict() for p in self.causal_paths],
            "bouzidani_contexts": self.bouzidani_contexts.to_dict(),
            "evidence_stats": self.evidence_stats.to_dict(),
            "completeness_score": self.completeness_score,
            "missing_fields": self.missing_fields,
            "generated_at": self.generated_at,
            "schema_version": self.schema_version,
        }
    
    def compute_hash(self) -> str:
        """Compute deterministic hash of dossier content."""
        # Exclude generated_at for stable hash
        content = self.to_dict()
        content.pop("generated_at", None)
        json_str = json.dumps(content, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()[:16]
    
    def compute_completeness(self) -> float:
        """Compute completeness score (0.0-1.0)."""
        scores = []
        
        # Identity (20%)
        identity_score = 1.0 if self.label_ar and self.label_en else 0.5
        scores.append(("identity", 0.2, identity_score))
        
        # Vocabulary (15%)
        vocab_items = len(self.synonyms_ar) + len(self.roots) + len(self.stems)
        vocab_score = min(1.0, vocab_items / 5)  # Target: 5 vocab items
        scores.append(("vocabulary", 0.15, vocab_score))
        
        # Quran evidence (25%)
        quran_score = min(1.0, len(self.quran_evidence) / 10)  # Target: 10 verses
        scores.append(("quran_evidence", 0.25, quran_score))
        
        # Tafsir evidence (15%)
        tafsir_count = sum(len(chunks) for chunks in self.tafsir_evidence.values())
        tafsir_score = min(1.0, tafsir_count / 20)  # Target: 20 chunks
        scores.append(("tafsir_evidence", 0.15, tafsir_score))
        
        # Relationships (15%)
        rel_count = len(self.outgoing_edges) + len(self.incoming_edges)
        rel_score = min(1.0, rel_count / 5)  # Target: 5 relationships
        scores.append(("relationships", 0.15, rel_score))
        
        # Bouzidani contexts (10%)
        unknown_count = len(self.bouzidani_contexts.get_unknown_fields())
        context_score = 1.0 - (unknown_count / 7)  # 7 context fields
        scores.append(("bouzidani_contexts", 0.10, max(0, context_score)))
        
        # Compute weighted average
        total = sum(weight * score for _, weight, score in scores)
        return round(total, 3)
    
    def update_stats(self):
        """Update evidence statistics."""
        self.evidence_stats.quran_verse_count = len(self.quran_evidence)
        self.evidence_stats.quran_direct_count = sum(
            1 for e in self.quran_evidence if e.relevance == "direct"
        )
        self.evidence_stats.quran_indirect_count = sum(
            1 for e in self.quran_evidence if e.relevance == "indirect"
        )
        self.evidence_stats.tafsir_chunk_count = sum(
            len(chunks) for chunks in self.tafsir_evidence.values()
        )
        self.evidence_stats.tafsir_sources_covered = list(self.tafsir_evidence.keys())
        self.evidence_stats.relationship_count = (
            len(self.outgoing_edges) + len(self.incoming_edges)
        )
        self.evidence_stats.causal_path_count = len(self.causal_paths)
        
        unknown_fields = self.bouzidani_contexts.get_unknown_fields()
        self.evidence_stats.context_unknown_count = len(unknown_fields)
        self.evidence_stats.context_known_count = 7 - len(unknown_fields)
        
        self.missing_fields = unknown_fields
        self.completeness_score = self.compute_completeness()
    
    def save(self, output_dir: Path):
        """Save dossier to JSON file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{self.behavior_id}.json"
        
        self.generated_at = datetime.utcnow().isoformat() + "Z"
        self.update_stats()
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        
        return output_path
    
    @classmethod
    def load(cls, path: Path) -> "BehaviorDossier":
        """Load dossier from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Reconstruct dataclass from dict
        dossier = cls(
            behavior_id=data["behavior_id"],
            label_ar=data["label_ar"],
            label_en=data["label_en"],
            category=data.get("category", ""),
            roots=data.get("roots", []),
            stems=data.get("stems", []),
            synonyms_ar=data.get("synonyms_ar", []),
            synonyms_en=data.get("synonyms_en", []),
            antonyms=data.get("antonyms", []),
            morphological_variants=data.get("morphological_variants", []),
            quranic_phrases=data.get("quranic_phrases", []),
        )
        
        # Reconstruct quran evidence
        for ev in data.get("quran_evidence", []):
            dossier.quran_evidence.append(VerseEvidence(**ev))
        
        # Reconstruct tafsir evidence
        for src, chunks in data.get("tafsir_evidence", {}).items():
            dossier.tafsir_evidence[src] = [TafsirChunk(**c) for c in chunks]
        
        # Reconstruct relationships
        for edge in data.get("outgoing_edges", []):
            dossier.outgoing_edges.append(RelationshipEdge(**edge))
        for edge in data.get("incoming_edges", []):
            dossier.incoming_edges.append(RelationshipEdge(**edge))
        
        # Reconstruct causal paths
        for path in data.get("causal_paths", []):
            dossier.causal_paths.append(CausalPath(**path))
        
        # Reconstruct Bouzidani contexts
        ctx_data = data.get("bouzidani_contexts", {})
        dossier.bouzidani_contexts = BouzidaniContexts(
            organ_links=[ContextAssertion(**o) for o in ctx_data.get("organ_links", [])],
            internal_external=ContextAssertion(**ctx_data.get("internal_external", {"value": "unknown", "citation_type": "unknown", "confidence": 0.0, "reason_if_unknown": "no_evidence"})),
            situational_context=ContextAssertion(**ctx_data.get("situational_context", {"value": "unknown", "citation_type": "unknown", "confidence": 0.0, "reason_if_unknown": "no_evidence"})),
            systemic_context=ContextAssertion(**ctx_data.get("systemic_context", {"value": "unknown", "citation_type": "unknown", "confidence": 0.0, "reason_if_unknown": "no_evidence"})),
            spatial_context=ContextAssertion(**ctx_data.get("spatial_context", {"value": "unknown", "citation_type": "unknown", "confidence": 0.0, "reason_if_unknown": "no_evidence"})),
            temporal_context=ContextAssertion(**ctx_data.get("temporal_context", {"value": "unknown", "citation_type": "unknown", "confidence": 0.0, "reason_if_unknown": "no_evidence"})),
            intention_niyyah=ContextAssertion(**ctx_data.get("intention_niyyah", {"value": "unknown", "citation_type": "unknown", "confidence": 0.0, "reason_if_unknown": "no_evidence"})),
            recurrence_dawrah=ContextAssertion(**ctx_data.get("recurrence_dawrah", {"value": "unknown", "citation_type": "unknown", "confidence": 0.0, "reason_if_unknown": "no_evidence"})),
        )
        
        # Reconstruct stats
        stats_data = data.get("evidence_stats", {})
        dossier.evidence_stats = EvidenceStats(**stats_data) if stats_data else EvidenceStats()
        
        dossier.completeness_score = data.get("completeness_score", 0.0)
        dossier.missing_fields = data.get("missing_fields", [])
        dossier.generated_at = data.get("generated_at", "")
        dossier.schema_version = data.get("schema_version", "1.0.0")
        
        return dossier
