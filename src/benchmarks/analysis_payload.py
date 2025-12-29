"""
Phase 2: Deterministic Analysis Payload Builder

Builds structured payloads from proof data for deterministic answer generation.
All numbers and statistics come from computed data, never invented.

RULES:
- Wrap existing truth-layer assets (LegendaryPlanner, concept_index, semantic_graph)
- All numbers must be computable from payload
- Explicit "gaps" list when data is missing
- No fabrication, no generic defaults
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Data paths (same as LegendaryPlanner)
CANONICAL_ENTITIES_FILE = Path("vocab/canonical_entities.json")
CONCEPT_INDEX_FILE = Path("data/evidence/concept_index_v2.jsonl")
SEMANTIC_GRAPH_FILE = Path("data/graph/semantic_graph_v2.json")


@dataclass
class EntityInfo:
    """Extracted entity with ID and labels."""
    entity_id: str
    entity_type: str  # behavior, agent, organ, heart_state, consequence
    label_ar: str = ""
    label_en: str = ""
    total_mentions: int = 0
    verse_count: int = 0


@dataclass
class GraphOutput:
    """Computed graph outputs (paths, cycles, metrics)."""
    paths: List[Dict[str, Any]] = field(default_factory=list)
    cycles: List[Dict[str, Any]] = field(default_factory=list)
    centrality: Dict[str, Any] = field(default_factory=dict)
    causal_density: Dict[str, Any] = field(default_factory=dict)
    chain_distribution: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComputedTable:
    """A computed table with counts, percentages, rankings."""
    name: str
    columns: List[str] = field(default_factory=list)
    rows: List[Dict[str, Any]] = field(default_factory=list)
    totals: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceBundle:
    """Evidence bundle with provenance."""
    verse_key: str  # e.g., "2:255"
    surah: int
    ayah: int
    text: str = ""
    source: str = ""  # e.g., "ibn_kathir", "qurtubi"
    chunk_id: str = ""
    char_start: int = 0
    char_end: int = 0
    confidence: float = 1.0


@dataclass
class AnalysisPayload:
    """
    Complete deterministic analysis payload.

    All numbers and statistics in this payload are computed from data,
    never invented. LLM may only rephrase text, not add facts.
    """
    # Query info
    question: str
    question_class: str
    intent: str

    # Extracted entities (IDs + labels)
    entities: List[EntityInfo] = field(default_factory=list)

    # Computed graph outputs
    graph_output: GraphOutput = field(default_factory=GraphOutput)

    # Computed tables (counts, consensus %, rankings)
    tables: List[ComputedTable] = field(default_factory=list)

    # Evidence bundles with provenance
    quran_evidence: List[EvidenceBundle] = field(default_factory=list)
    tafsir_evidence: Dict[str, List[EvidenceBundle]] = field(default_factory=dict)

    # Explicit gaps (missing data)
    gaps: List[str] = field(default_factory=list)

    # All numbers that can be cited in the answer
    computed_numbers: Dict[str, Any] = field(default_factory=dict)

    # Derivation trace for validator gate
    derivations: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "question_class": self.question_class,
            "intent": self.intent,
            "entities": [asdict(e) for e in self.entities],
            "graph_output": asdict(self.graph_output),
            "tables": [asdict(t) for t in self.tables],
            "quran_evidence": [asdict(e) for e in self.quran_evidence],
            "tafsir_evidence": {
                source: [asdict(e) for e in bundles]
                for source, bundles in self.tafsir_evidence.items()
            },
            "gaps": self.gaps,
            "computed_numbers": self.computed_numbers,
            "derivations": self.derivations,
        }

    def get_all_numbers(self) -> Set[float]:
        """Extract all numbers from computed payload for validator gate."""
        numbers: Set[float] = set()

        # From computed_numbers
        for key, value in self.computed_numbers.items():
            if isinstance(value, (int, float)):
                numbers.add(float(value))
                if isinstance(value, float):
                    numbers.add(round(value * 100, 1))  # Add as percentage

        # From entities
        for entity in self.entities:
            numbers.add(float(entity.total_mentions))
            numbers.add(float(entity.verse_count))

        # From tables
        for table in self.tables:
            for key, value in table.totals.items():
                if isinstance(value, (int, float)):
                    numbers.add(float(value))
            for row in table.rows:
                for key, value in row.items():
                    if isinstance(value, (int, float)):
                        numbers.add(float(value))

        # From graph output
        if self.graph_output.centrality:
            for key, value in self.graph_output.centrality.items():
                if isinstance(value, (int, float)):
                    numbers.add(float(value))

        if self.graph_output.causal_density:
            if "total_causal_edges" in self.graph_output.causal_density:
                numbers.add(float(self.graph_output.causal_density["total_causal_edges"]))

        # Evidence counts
        numbers.add(float(len(self.quran_evidence)))
        for source, bundles in self.tafsir_evidence.items():
            numbers.add(float(len(bundles)))

        return numbers


class AnalysisPayloadBuilder:
    """
    Builds deterministic analysis payloads from proof data.

    Wraps LegendaryPlanner for graph traversal and evidence retrieval.
    All outputs are computed, never fabricated.
    """

    def __init__(self):
        self._planner = None
        self._canonical_entities = None
        self._concept_index = None
        self._loaded = False

    def _ensure_loaded(self):
        """Lazy load truth layer assets."""
        if self._loaded:
            return

        # Load canonical entities
        if CANONICAL_ENTITIES_FILE.exists():
            with open(CANONICAL_ENTITIES_FILE, "r", encoding="utf-8") as f:
                self._canonical_entities = json.load(f)
        else:
            self._canonical_entities = {}

        # Load concept index
        self._concept_index = {}
        if CONCEPT_INDEX_FILE.exists():
            with open(CONCEPT_INDEX_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line.strip())
                    concept_id = entry.get("concept_id", entry.get("term", ""))
                    if concept_id:
                        self._concept_index[concept_id] = entry

        self._loaded = True

    def _get_planner(self):
        """Get or create LegendaryPlanner instance."""
        if self._planner is None:
            from src.ml.legendary_planner import get_legendary_planner
            self._planner = get_legendary_planner()
        return self._planner

    def build_analysis_payload(
        self,
        question: str,
        question_class: str,
        proof: Dict[str, Any],
        debug: Dict[str, Any],
    ) -> AnalysisPayload:
        """
        Build deterministic analysis payload from proof and debug data.

        Args:
            question: The original question text
            question_class: The detected question class (e.g., "causal_chain")
            proof: The proof object from MandatoryProofSystem
            debug: The debug trace from MandatoryProofSystem

        Returns:
            AnalysisPayload with all computed data
        """
        self._ensure_loaded()

        payload = AnalysisPayload(
            question=question,
            question_class=question_class,
            intent=debug.get("intent", "FREE_TEXT"),
        )

        # 1. Extract entities from debug.entity_resolution or proof
        payload.entities = self._extract_entities(proof, debug)

        # 2. Build graph outputs
        payload.graph_output = self._build_graph_output(question_class, proof, debug)

        # 3. Build computed tables
        payload.tables = self._build_tables(question_class, proof, debug)

        # 4. Extract evidence bundles
        payload.quran_evidence, payload.tafsir_evidence = self._extract_evidence(proof)

        # 5. Identify gaps
        payload.gaps = self._identify_gaps(question_class, payload)

        # 6. Compute all numbers for validator gate
        payload.computed_numbers = self._compute_numbers(payload, proof, debug)

        # 7. Build derivations trace
        payload.derivations = {
            "entity_ids": [e.entity_id for e in payload.entities],
            "quran_verse_count": len(payload.quran_evidence),
            "tafsir_sources": list(payload.tafsir_evidence.keys()),
            "tafsir_chunk_counts": {
                src: len(chunks) for src, chunks in payload.tafsir_evidence.items()
            },
            "graph_paths_count": len(payload.graph_output.paths),
            "graph_cycles_count": len(payload.graph_output.cycles),
            "gaps": payload.gaps,
        }

        return payload

    def _extract_entities(
        self, proof: Dict[str, Any], debug: Dict[str, Any]
    ) -> List[EntityInfo]:
        """Extract entities from proof and debug data."""
        entities = []
        seen_ids = set()

        # From debug.entity_resolution (LegendaryPlanner)
        entity_resolution = debug.get("entity_resolution", {})
        for entity_data in entity_resolution.get("entities", []):
            entity_id = entity_data.get("entity_id", "")
            if entity_id and entity_id not in seen_ids:
                seen_ids.add(entity_id)

                # Get labels from canonical entities
                label_ar, label_en = self._get_entity_labels(entity_id)

                entities.append(EntityInfo(
                    entity_id=entity_id,
                    entity_type=entity_data.get("entity_type", "UNKNOWN"),
                    label_ar=label_ar,
                    label_en=label_en,
                    total_mentions=entity_data.get("total_mentions", 0),
                    verse_count=len(entity_data.get("verse_keys", [])),
                ))

        # From concept_lookups in debug
        for lookup in debug.get("concept_lookups", []):
            entity_id = lookup.get("entity_id", "")
            if entity_id and entity_id not in seen_ids:
                seen_ids.add(entity_id)
                label_ar, label_en = self._get_entity_labels(entity_id)
                entities.append(EntityInfo(
                    entity_id=entity_id,
                    entity_type="CONCEPT",
                    label_ar=label_ar,
                    label_en=label_en,
                    total_mentions=lookup.get("mentions", 0),
                ))

        return entities

    def _get_entity_labels(self, entity_id: str) -> Tuple[str, str]:
        """Get Arabic and English labels for an entity."""
        if not self._canonical_entities:
            return "", ""

        for section in ["behaviors", "agents", "organs", "heart_states", "consequences"]:
            for item in self._canonical_entities.get(section, []):
                if item.get("id") == entity_id:
                    return item.get("ar", ""), item.get("en", "")

        return "", ""

    def _build_graph_output(
        self, question_class: str, proof: Dict[str, Any], debug: Dict[str, Any]
    ) -> GraphOutput:
        """Build graph outputs from proof data."""
        output = GraphOutput()

        # Extract from debug.graph_traversals (LegendaryPlanner)
        for traversal in debug.get("graph_traversals", []):
            action = traversal.get("action", "")

            if action == "causal_paths":
                # Paths between specific entities
                paths_found = traversal.get("paths_found", 0)
                output.paths = proof.get("graph", {}).get("paths", [])

            elif action == "global_cycle_detection":
                # Entity-free cycle detection
                output.cycles = proof.get("graph", {}).get("cycles", [])

            elif action == "global_centrality":
                # Entity-free centrality computation
                output.centrality = {
                    "total_nodes": traversal.get("total_nodes", 0),
                    "total_edges": traversal.get("total_edges", 0),
                    "entity_free": traversal.get("entity_free", False),
                }

        # Extract from proof.graph
        proof_graph = proof.get("graph", {})
        if proof_graph:
            if "paths" in proof_graph and not output.paths:
                output.paths = proof_graph["paths"]
            if "validated_paths" in proof_graph:
                output.paths = proof_graph["validated_paths"]
            if "centrality" in proof_graph:
                output.centrality = proof_graph["centrality"]
            if "causal_density" in proof_graph:
                output.causal_density = proof_graph["causal_density"]
            if "chain_distribution" in proof_graph:
                output.chain_distribution = proof_graph["chain_distribution"]
            if "cycles" in proof_graph:
                output.cycles = proof_graph["cycles"]

        return output

    def _build_tables(
        self, question_class: str, proof: Dict[str, Any], debug: Dict[str, Any]
    ) -> List[ComputedTable]:
        """Build computed tables from proof data."""
        tables = []

        # Cross-tafsir stats table
        cross_tafsir = debug.get("cross_tafsir_stats", {})
        if cross_tafsir:
            source_dist = cross_tafsir.get("source_distribution", {})
            if source_dist:
                rows = [
                    {"source": src, "count": cnt}
                    for src, cnt in source_dist.items()
                ]
                tables.append(ComputedTable(
                    name="tafsir_source_distribution",
                    columns=["source", "count"],
                    rows=rows,
                    totals={
                        "sources_with_evidence": cross_tafsir.get("sources_count", 0),
                        "total_sources": cross_tafsir.get("total_sources", 7),
                        "agreement_ratio": cross_tafsir.get("agreement_ratio", 0),
                    },
                ))

        # Graph centrality table
        graph_output = self._build_graph_output(question_class, proof, debug)
        if graph_output.causal_density:
            top_outgoing = graph_output.causal_density.get("outgoing_top10", [])
            if top_outgoing:
                rows = [
                    {
                        "entity_id": item.get("id", ""),
                        "label_ar": item.get("label", {}).get("ar", ""),
                        "outgoing_count": item.get("count", 0),
                    }
                    for item in top_outgoing
                ]
                tables.append(ComputedTable(
                    name="causal_density_outgoing",
                    columns=["entity_id", "label_ar", "outgoing_count"],
                    rows=rows,
                    totals={
                        "total_causal_edges": graph_output.causal_density.get("total_causal_edges", 0),
                    },
                ))

        # Statistics from proof
        proof_stats = proof.get("statistics", {})
        if isinstance(proof_stats, dict):
            counts = proof_stats.get("counts", {})
            if counts:
                rows = [{"metric": k, "value": v} for k, v in counts.items()]
                tables.append(ComputedTable(
                    name="statistics_counts",
                    columns=["metric", "value"],
                    rows=rows,
                    totals=counts,
                ))

        return tables

    def _extract_evidence(
        self, proof: Dict[str, Any]
    ) -> Tuple[List[EvidenceBundle], Dict[str, List[EvidenceBundle]]]:
        """Extract evidence bundles from proof."""
        quran_evidence = []
        tafsir_evidence = {}

        # Quran verses
        quran_data = proof.get("quran", [])
        if isinstance(quran_data, list):
            for v in quran_data:
                if isinstance(v, dict):
                    try:
                        quran_evidence.append(EvidenceBundle(
                            verse_key=v.get("verse_key", f"{v.get('surah', '')}:{v.get('ayah', '')}"),
                            surah=int(v.get("surah", 0)),
                            ayah=int(v.get("ayah", 0)),
                            text=v.get("text", ""),
                            source="quran",
                            confidence=float(v.get("relevance", v.get("score", 1.0))),
                        ))
                    except (ValueError, TypeError):
                        pass

        # Tafsir chunks
        tafsir_data = proof.get("tafsir", {})
        if isinstance(tafsir_data, dict):
            for source, chunks in tafsir_data.items():
                if not isinstance(chunks, list):
                    continue
                tafsir_evidence[source] = []
                for chunk in chunks:
                    if isinstance(chunk, dict):
                        try:
                            tafsir_evidence[source].append(EvidenceBundle(
                                verse_key=chunk.get("verse_key", f"{chunk.get('surah', '')}:{chunk.get('ayah', '')}"),
                                surah=int(chunk.get("surah", 0)),
                                ayah=int(chunk.get("ayah", 0)),
                                text=chunk.get("text", ""),
                                source=source,
                                chunk_id=chunk.get("chunk_id", ""),
                                char_start=int(chunk.get("char_start", 0)),
                                char_end=int(chunk.get("char_end", 0)),
                                confidence=float(chunk.get("score", 1.0)),
                            ))
                        except (ValueError, TypeError):
                            pass

        return quran_evidence, tafsir_evidence

    def _identify_gaps(
        self, question_class: str, payload: AnalysisPayload
    ) -> List[str]:
        """Identify gaps in the payload data."""
        gaps = []

        # No entities resolved
        if not payload.entities:
            gaps.append("no_entities_resolved")

        # No Quran evidence
        if not payload.quran_evidence:
            gaps.append("no_quran_evidence")

        # Missing tafsir sources
        from src.ml.tafsir_constants import CANONICAL_TAFSIR_SOURCES
        for source in CANONICAL_TAFSIR_SOURCES:
            if source not in payload.tafsir_evidence or not payload.tafsir_evidence[source]:
                gaps.append(f"missing_tafsir_{source}")

        # Question class specific gaps
        if question_class in ("causal_chain", "GRAPH_CAUSAL"):
            if not payload.graph_output.paths:
                gaps.append("no_causal_paths")

        if question_class in ("reinforcement_loop", "REINFORCEMENT_LOOP"):
            if not payload.graph_output.cycles:
                gaps.append("no_cycles_found")

        if question_class in ("network_centrality", "GRAPH_METRICS"):
            if not payload.graph_output.centrality:
                gaps.append("no_centrality_computed")

        return gaps

    def _compute_numbers(
        self, payload: AnalysisPayload, proof: Dict[str, Any], debug: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute all numbers that can be cited in the answer."""
        numbers = {}

        # Evidence counts
        numbers["quran_verse_count"] = len(payload.quran_evidence)
        numbers["total_tafsir_chunks"] = sum(
            len(chunks) for chunks in payload.tafsir_evidence.values()
        )
        numbers["tafsir_sources_with_evidence"] = sum(
            1 for chunks in payload.tafsir_evidence.values() if chunks
        )
        numbers["total_tafsir_sources"] = 7

        # Entity counts
        numbers["entities_resolved"] = len(payload.entities)
        numbers["total_entity_mentions"] = sum(e.total_mentions for e in payload.entities)

        # Graph counts
        numbers["causal_paths_found"] = len(payload.graph_output.paths)
        numbers["cycles_found"] = len(payload.graph_output.cycles)

        # From cross_tafsir_stats
        cross_tafsir = debug.get("cross_tafsir_stats", {})
        if cross_tafsir:
            numbers["agreement_ratio"] = cross_tafsir.get("agreement_ratio", 0)
            numbers["consensus_percentage"] = round(
                cross_tafsir.get("agreement_ratio", 0) * 100, 1
            )

        # From causal density
        if payload.graph_output.causal_density:
            numbers["total_causal_edges"] = payload.graph_output.causal_density.get(
                "total_causal_edges", 0
            )

        # From chain distribution
        if payload.graph_output.chain_distribution:
            numbers["longest_chain_length"] = payload.graph_output.chain_distribution.get(
                "longest_chain_length", 0
            )
            numbers["average_chain_length"] = payload.graph_output.chain_distribution.get(
                "average_chain_length", 0
            )

        # Gap count
        numbers["gaps_count"] = len(payload.gaps)

        return numbers


# Module-level convenience function
_builder = None


def get_payload_builder() -> AnalysisPayloadBuilder:
    """Get or create the payload builder singleton."""
    global _builder
    if _builder is None:
        _builder = AnalysisPayloadBuilder()
    return _builder


def build_analysis_payload(
    question: str,
    question_class: str,
    proof: Dict[str, Any],
    debug: Dict[str, Any],
) -> AnalysisPayload:
    """
    Build deterministic analysis payload from proof data.

    This is the main entry point for Phase 2 payload building.

    Args:
        question: The original question text
        question_class: The detected question class
        proof: The proof object from MandatoryProofSystem
        debug: The debug trace from MandatoryProofSystem

    Returns:
        AnalysisPayload with all computed data
    """
    builder = get_payload_builder()
    return builder.build_analysis_payload(question, question_class, proof, debug)
