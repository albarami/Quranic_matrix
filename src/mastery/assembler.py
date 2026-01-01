"""
Dossier Assembler for QBM.

Merges all sources into complete behavior dossiers:
- Canonical registry (vocab/canonical_entities.json)
- Normalized graph (src/graph/normalize.py)
- Concept index (data/concept_index_v3.json)
- Evidence index (data/evidence/evidence_index_v1.jsonl)
- Quran store (src/ml/quran_store.py)
"""

import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from dataclasses import asdict

from .dossier import (
    BehaviorDossier,
    VerseEvidence,
    TafsirChunk,
    RelationshipEdge,
    CausalPath,
    BouzidaniContexts,
    ContextAssertion,
    EvidenceStats,
)

logger = logging.getLogger(__name__)

# Paths
CANONICAL_ENTITIES_PATH = Path("vocab/canonical_entities.json")
CONCEPT_INDEX_PATH = Path("data/concept_index_v3.json")
EVIDENCE_INDEX_PATH = Path("data/evidence/evidence_index_v1.jsonl")
DOSSIER_OUTPUT_DIR = Path("artifacts/mastery/behaviors")
SUMMARY_OUTPUT_PATH = Path("artifacts/mastery/mastery_summary.json")
MANIFEST_OUTPUT_PATH = Path("artifacts/mastery/mastery_manifest.json")

# Organ mappings for Bouzidani context
ORGAN_KEYWORDS = {
    "ORG_HEART": ["قلب", "قلوب", "فؤاد", "صدر", "صدور"],
    "ORG_TONGUE": ["لسان", "ألسنة", "قول", "كلام", "نطق"],
    "ORG_EYE": ["عين", "أعين", "بصر", "أبصار", "نظر"],
    "ORG_EAR": ["أذن", "آذان", "سمع", "أسماع"],
    "ORG_HAND": ["يد", "أيدي", "أيد"],
    "ORG_FOOT": ["رجل", "أرجل", "قدم", "أقدام"],
}

# Internal/External classification rules
INTERNAL_BEHAVIORS = {
    "BEH_EMO_", "BEH_COG_", "BEH_SPI_", "BEH_INT_",
}
EXTERNAL_BEHAVIORS = {
    "BEH_PHY_", "BEH_SOC_", "BEH_SPEECH_", "BEH_FIN_",
}


class DossierAssembler:
    """Assembles complete dossiers from all sources."""
    
    def __init__(self):
        self.canonical_behaviors: Dict[str, Dict[str, Any]] = {}
        self.concept_index: Dict[str, Any] = {}
        self.evidence_index: Dict[str, List[str]] = {}
        self.normalized_graph = None
        self.quran_store = None
        
    def load_sources(self):
        """Load all source data."""
        # Load canonical behaviors
        if CANONICAL_ENTITIES_PATH.exists():
            with open(CANONICAL_ENTITIES_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.canonical_behaviors = {b["id"]: b for b in data.get("behaviors", [])}
            logger.info(f"Loaded {len(self.canonical_behaviors)} canonical behaviors")
        
        # Load concept index from LegendaryPlanner (has verse mappings)
        try:
            from src.ml.legendary_planner import get_legendary_planner
            planner = get_legendary_planner()
            planner.load()
            self.concept_index = planner.concept_index or {}
            logger.info(f"Loaded concept index with {len(self.concept_index)} entries from LegendaryPlanner")
        except Exception as e:
            logger.warning(f"Could not load concept index from LegendaryPlanner: {e}")
            # Fallback to file if exists
            if CONCEPT_INDEX_PATH.exists():
                with open(CONCEPT_INDEX_PATH, "r", encoding="utf-8") as f:
                    self.concept_index = json.load(f)
                logger.info(f"Loaded concept index from file with {len(self.concept_index)} entries")
        
        # Load evidence index
        if EVIDENCE_INDEX_PATH.exists():
            with open(EVIDENCE_INDEX_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        verse_key = entry.get("verse_key", "")
                        if verse_key:
                            if verse_key not in self.evidence_index:
                                self.evidence_index[verse_key] = []
                            self.evidence_index[verse_key].append(entry)
            logger.info(f"Loaded evidence index with {len(self.evidence_index)} verse keys")
        
        # Load normalized graph
        try:
            from src.graph.normalize import get_normalized_graph
            self.normalized_graph = get_normalized_graph()
            logger.info(f"Loaded normalized graph with {self.normalized_graph.metadata.behavior_count} behaviors")
        except Exception as e:
            logger.warning(f"Could not load normalized graph: {e}")
        
        # Load Quran store
        try:
            from src.ml.quran_store import QuranStore
            self.quran_store = QuranStore()
            self.quran_store.load()
            logger.info(f"Loaded Quran store with {self.quran_store.get_verse_count()} verses")
        except Exception as e:
            logger.warning(f"Could not load Quran store: {e}")
    
    def _get_verse_text(self, verse_key: str) -> tuple:
        """Get verse text (Arabic, English) from Quran store."""
        if self.quran_store:
            verse = self.quran_store.get_verse(verse_key)
            if verse:
                return verse.get("text_ar", ""), verse.get("text_en", "")
        return "", ""
    
    def _extract_quran_evidence(self, behavior_id: str) -> List[VerseEvidence]:
        """Extract Quran evidence for a behavior from concept_index."""
        evidence = []
        
        # Get from concept index (LegendaryPlanner format has 'verses' list)
        concept_data = self.concept_index.get(behavior_id, {})
        
        # The concept_index has 'verses' list with verse_key and text_uthmani
        verses = concept_data.get("verses", [])
        
        for verse_data in verses[:50]:  # Limit to 50 verses
            if isinstance(verse_data, dict):
                verse_key = verse_data.get("verse_key", "")
                text_ar = verse_data.get("text_uthmani", "")
                directness = verse_data.get("directness", "direct")
                
                # Get English translation from Quran store if available
                _, text_en = self._get_verse_text(verse_key)
                
                evidence.append(VerseEvidence(
                    verse_key=verse_key,
                    text_ar=text_ar,
                    text_en=text_en,
                    relevance=directness,
                    confidence=0.85 if directness == "direct" else 0.6,
                ))
        
        return evidence
    
    def _extract_tafsir_evidence(self, behavior_id: str, verse_keys: List[str]) -> Dict[str, List[TafsirChunk]]:
        """Extract tafsir evidence for a behavior."""
        tafsir_evidence: Dict[str, List[TafsirChunk]] = {}
        
        # First try concept_index tafsir_chunks (LegendaryPlanner format)
        concept_data = self.concept_index.get(behavior_id, {})
        tafsir_chunks = concept_data.get("tafsir_chunks", [])
        
        for chunk in tafsir_chunks[:30]:  # Limit to 30 chunks
            if isinstance(chunk, dict):
                source = chunk.get("source", "")
                if source:
                    if source not in tafsir_evidence:
                        tafsir_evidence[source] = []
                    
                    tafsir_evidence[source].append(TafsirChunk(
                        source=source,
                        chunk_id=chunk.get("chunk_id", ""),
                        verse_key=chunk.get("verse_key", ""),
                        text_ar=chunk.get("text", "")[:500],
                        char_start=chunk.get("char_start", 0),
                        char_end=chunk.get("char_end", 0),
                        relevance=chunk.get("relevance", "direct"),
                    ))
        
        # Fallback to evidence_index if no tafsir_chunks in concept_index
        if not tafsir_evidence:
            for verse_key in verse_keys[:20]:
                entries = self.evidence_index.get(verse_key, [])
                for entry in entries:
                    source = entry.get("source", "")
                    if source and source != "quran":
                        if source not in tafsir_evidence:
                            tafsir_evidence[source] = []
                        
                        tafsir_evidence[source].append(TafsirChunk(
                            source=source,
                            chunk_id=entry.get("chunk_id", f"{source}_{verse_key}"),
                            verse_key=verse_key,
                            text_ar=entry.get("text_clean", entry.get("text", ""))[:500],
                            char_start=entry.get("char_start", 0),
                            char_end=entry.get("char_end", 0),
                            relevance="direct",
                        ))
        
        return tafsir_evidence
    
    def _extract_relationships(self, behavior_id: str) -> tuple:
        """Extract relationship edges from normalized graph."""
        outgoing = []
        incoming = []
        
        if not self.normalized_graph:
            return outgoing, incoming
        
        # Build node lookup
        node_lookup = {n.id: n for n in self.normalized_graph.nodes}
        
        for edge in self.normalized_graph.edges:
            if edge.source == behavior_id:
                target_node = node_lookup.get(edge.target)
                if target_node:
                    outgoing.append(RelationshipEdge(
                        target_id=edge.target,
                        target_label_ar=target_node.label_ar,
                        target_label_en=target_node.label_en,
                        edge_type=edge.edge_type,
                        direction="outgoing",
                        evidence_count=edge.evidence_count,
                        evidence_refs=[ref.to_dict() for ref in edge.evidence_refs[:5]],
                        confidence=edge.confidence,
                    ))
            
            if edge.target == behavior_id:
                source_node = node_lookup.get(edge.source)
                if source_node:
                    incoming.append(RelationshipEdge(
                        target_id=edge.source,
                        target_label_ar=source_node.label_ar,
                        target_label_en=source_node.label_en,
                        edge_type=edge.edge_type,
                        direction="incoming",
                        evidence_count=edge.evidence_count,
                        evidence_refs=[ref.to_dict() for ref in edge.evidence_refs[:5]],
                        confidence=edge.confidence,
                    ))
        
        return outgoing, incoming
    
    def _infer_bouzidani_contexts(self, behavior_id: str, canonical_data: Dict[str, Any]) -> BouzidaniContexts:
        """Infer Bouzidani contexts from behavior data."""
        contexts = BouzidaniContexts()
        
        # Axis 1: Internal/External based on behavior category
        category = canonical_data.get("category", "")
        for prefix in INTERNAL_BEHAVIORS:
            if behavior_id.startswith(prefix):
                contexts.internal_external = ContextAssertion.from_rule(
                    "باطن", f"RULE_INTERNAL_{prefix}", 0.9
                )
                break
        for prefix in EXTERNAL_BEHAVIORS:
            if behavior_id.startswith(prefix):
                contexts.internal_external = ContextAssertion.from_rule(
                    "ظاهر", f"RULE_EXTERNAL_{prefix}", 0.9
                )
                break
        
        # Organ links based on category
        if "SPEECH" in behavior_id:
            contexts.organ_links.append(ContextAssertion.from_rule(
                "ORG_TONGUE", "RULE_SPEECH_TONGUE", 0.95
            ))
        if "EMO" in behavior_id or "SPI" in behavior_id:
            contexts.organ_links.append(ContextAssertion.from_rule(
                "ORG_HEART", "RULE_EMOTION_HEART", 0.9
            ))
        if "COG" in behavior_id:
            contexts.organ_links.append(ContextAssertion.from_rule(
                "ORG_HEART", "RULE_COGNITION_HEART", 0.85
            ))
        
        # Intention based on category
        if category in ["worship", "spiritual", "devotion"]:
            contexts.intention_niyyah = ContextAssertion.from_rule(
                "with_intention", "RULE_WORSHIP_INTENTION", 0.85
            )
        elif category in ["instinct", "reflex"]:
            contexts.intention_niyyah = ContextAssertion.from_rule(
                "instinctive", "RULE_INSTINCT_NO_INTENTION", 0.8
            )
        
        return contexts
    
    def build_dossier(self, behavior_id: str) -> BehaviorDossier:
        """Build complete dossier for a single behavior."""
        canonical_data = self.canonical_behaviors.get(behavior_id, {})
        
        # Create base dossier
        dossier = BehaviorDossier(
            behavior_id=behavior_id,
            label_ar=canonical_data.get("ar", ""),
            label_en=canonical_data.get("en", ""),
            category=canonical_data.get("category", ""),
            roots=canonical_data.get("roots", []),
            stems=canonical_data.get("stems", []),
            synonyms_ar=canonical_data.get("synonyms", []),
            synonyms_en=canonical_data.get("synonyms_en", []),
            antonyms=canonical_data.get("antonyms", []),
            morphological_variants=canonical_data.get("variants", []),
        )
        
        # Extract Quran evidence
        dossier.quran_evidence = self._extract_quran_evidence(behavior_id)
        
        # Extract tafsir evidence
        verse_keys = [e.verse_key for e in dossier.quran_evidence]
        dossier.tafsir_evidence = self._extract_tafsir_evidence(behavior_id, verse_keys)
        
        # Extract relationships
        dossier.outgoing_edges, dossier.incoming_edges = self._extract_relationships(behavior_id)
        
        # Infer Bouzidani contexts
        dossier.bouzidani_contexts = self._infer_bouzidani_contexts(behavior_id, canonical_data)
        
        # Update stats
        dossier.update_stats()
        
        return dossier
    
    def build_all_dossiers(self) -> List[BehaviorDossier]:
        """Build dossiers for all 87 canonical behaviors."""
        dossiers = []
        
        for behavior_id in sorted(self.canonical_behaviors.keys()):
            logger.info(f"Building dossier for {behavior_id}")
            dossier = self.build_dossier(behavior_id)
            dossiers.append(dossier)
        
        return dossiers
    
    def save_all_dossiers(self, dossiers: List[BehaviorDossier]) -> Dict[str, Any]:
        """Save all dossiers and generate summary/manifest."""
        DOSSIER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save individual dossiers
        dossier_hashes = {}
        for dossier in dossiers:
            path = dossier.save(DOSSIER_OUTPUT_DIR)
            dossier_hashes[dossier.behavior_id] = dossier.compute_hash()
            logger.info(f"Saved {dossier.behavior_id} -> {path}")
        
        # Generate summary
        summary = {
            "total_dossiers": len(dossiers),
            "schema_version": "1.0.0",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "coverage": {
                "behaviors_with_quran_evidence": sum(1 for d in dossiers if d.quran_evidence),
                "behaviors_with_tafsir_evidence": sum(1 for d in dossiers if d.tafsir_evidence),
                "behaviors_with_relationships": sum(1 for d in dossiers if d.outgoing_edges or d.incoming_edges),
                "avg_completeness_score": round(sum(d.completeness_score for d in dossiers) / len(dossiers), 3) if dossiers else 0,
            },
            "per_behavior": [
                {
                    "behavior_id": d.behavior_id,
                    "label_ar": d.label_ar,
                    "label_en": d.label_en,
                    "completeness_score": d.completeness_score,
                    "quran_verse_count": d.evidence_stats.quran_verse_count,
                    "tafsir_chunk_count": d.evidence_stats.tafsir_chunk_count,
                    "relationship_count": d.evidence_stats.relationship_count,
                    "missing_fields": d.missing_fields,
                }
                for d in dossiers
            ],
        }
        
        with open(SUMMARY_OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved summary -> {SUMMARY_OUTPUT_PATH}")
        
        # Generate manifest
        manifest_content = json.dumps(dossier_hashes, sort_keys=True)
        manifest_hash = hashlib.sha256(manifest_content.encode()).hexdigest()[:16]
        
        manifest = {
            "schema_version": "1.0.0",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "dossier_count": len(dossiers),
            "manifest_hash": manifest_hash,
            "dossier_hashes": dossier_hashes,
        }
        
        with open(MANIFEST_OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved manifest -> {MANIFEST_OUTPUT_PATH}")
        
        return summary


def build_all_dossiers() -> List[BehaviorDossier]:
    """Build all 87 behavior dossiers."""
    assembler = DossierAssembler()
    assembler.load_sources()
    dossiers = assembler.build_all_dossiers()
    assembler.save_all_dossiers(dossiers)
    return dossiers


def load_dossier(behavior_id: str) -> Optional[BehaviorDossier]:
    """Load a single dossier from disk."""
    path = DOSSIER_OUTPUT_DIR / f"{behavior_id}.json"
    if path.exists():
        return BehaviorDossier.load(path)
    return None


# Singleton cache
_dossier_cache: Dict[str, BehaviorDossier] = {}


def get_dossier(behavior_id: str, use_cache: bool = True) -> Optional[BehaviorDossier]:
    """Get a dossier, using cache if available."""
    if use_cache and behavior_id in _dossier_cache:
        return _dossier_cache[behavior_id]
    
    dossier = load_dossier(behavior_id)
    if dossier and use_cache:
        _dossier_cache[behavior_id] = dossier
    
    return dossier


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dossiers = build_all_dossiers()
    print(f"Built {len(dossiers)} dossiers")
