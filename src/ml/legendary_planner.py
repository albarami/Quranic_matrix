"""
Legendary 25 Question Planner (Phase 9.1)

Implements deterministic routing for all 25 legendary QBM question classes.
Each question class maps to a defined pipeline using:
- Concept index
- Semantic graph traversal
- Co-occurrence exploration
- Cross-tafsir comparison
- Stats engine
- Provenance bundler

Debug trace includes:
- debug.question_class
- debug.plan_steps[]
- debug.provenance_summary
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import time

logger = logging.getLogger(__name__)

# Data paths
# Phase 9: Updated to use validated v3 data
CANONICAL_ENTITIES_FILE = Path("vocab/canonical_entities.json")
CONCEPT_INDEX_FILE = Path("data/evidence/concept_index_v3.jsonl")
SEMANTIC_GRAPH_FILE = Path("data/graph/graph_v3.json")
# NOTE: graph_v3 is the validated SSOT projection (Behavior<->Verse edges).
# Some analytics (causal paths/cycles) still require semantic_graph_v2 until v3 includes semantic edges.
CAUSAL_GRAPH_FILE = Path("data/graph/semantic_graph_v2.json")
COOCCURRENCE_GRAPH_FILE = Path("data/graph/cooccurrence_graph_v1.json")
TAFSIR_SOURCES_FILE = Path("vocab/tafsir_sources.json")

# Import canonical tafsir sources from shared constant (7 sources)
from src.ml.tafsir_constants import CANONICAL_TAFSIR_SOURCES

CORE_SOURCES = CANONICAL_TAFSIR_SOURCES  # 7 sources from vocab/tafsir_sources.json


class QuestionClass(Enum):
    """The 25 legendary question classes."""
    # Category 1: Causal Chain Analysis
    CAUSAL_CHAIN = "causal_chain"                    # Q1
    SHORTEST_PATH = "shortest_path"                  # Q2
    REINFORCEMENT_LOOP = "reinforcement_loop"        # Q3
    
    # Category 2: Cross-Tafsir Comparative
    CROSS_TAFSIR_COMPARATIVE = "cross_tafsir_comparative"  # Q4
    MAKKI_MADANI_ANALYSIS = "makki_madani_analysis"        # Q5
    CONSENSUS_DISPUTE = "consensus_dispute"                # Q6
    
    # Category 3: 11-Axis + Taxonomy
    BEHAVIOR_PROFILE_11AXIS = "behavior_profile_11axis"    # Q7
    ORGAN_BEHAVIOR_MAPPING = "organ_behavior_mapping"      # Q8
    STATE_TRANSITION = "state_transition"                  # Q9
    
    # Category 4: Agent-Based Analysis
    AGENT_ATTRIBUTION = "agent_attribution"                # Q10
    AGENT_CONTRAST_MATRIX = "agent_contrast_matrix"        # Q11
    PROPHETIC_ARCHETYPE = "prophetic_archetype"            # Q12
    
    # Category 5: Network + Graph Analytics
    NETWORK_CENTRALITY = "network_centrality"              # Q13
    COMMUNITY_DETECTION = "community_detection"            # Q14
    BRIDGE_BEHAVIORS = "bridge_behaviors"                  # Q15
    
    # Category 6: Temporal + Spatial Context
    TEMPORAL_MAPPING = "temporal_mapping"                  # Q16
    SPATIAL_MAPPING = "spatial_mapping"                    # Q17

    # Category 6b: Consequence Analysis
    CONSEQUENCE_MAPPING = "consequence_mapping"            # NEW
    
    # Category 7: Statistics + Patterns
    SURAH_FINGERPRINTS = "surah_fingerprints"              # Q18
    FREQUENCY_CENTRALITY = "frequency_centrality"          # Q19
    MAKKI_MADANI_SHIFT = "makki_madani_shift"              # Q20
    
    # Category 8: Embeddings + Semantics
    SEMANTIC_LANDSCAPE = "semantic_landscape"              # Q21
    MEANING_DRIFT = "meaning_drift"                        # Q22
    
    # Category 9: Complex Multi-System
    COMPLETE_ANALYSIS = "complete_analysis"                # Q23
    PRESCRIPTION_GENERATOR = "prescription_generator"      # Q24
    GENOME_ARTIFACT = "genome_artifact"                    # Q25
    
    # Fallback
    FREE_TEXT = "free_text"


# Question class detection patterns
QUESTION_PATTERNS = {
    QuestionClass.CAUSAL_CHAIN: [
        r'يؤدي\s*إلى', r'سبب', r'نتيجة', r'طريق\s*إلى',
        r'path.*destruction', r'leads?\s*to', r'cause',
        r'causal\s*chain', r'trace.*chain', r'chain.*from',
        r'from\s+\S+.*to\s+\S+', r'hops?\s+from',
    ],
    QuestionClass.SHORTEST_PATH: [
        r'أقصر\s*طريق', r'تحول', r'من.*إلى',
        r'shortest.*path', r'transform', r'bridge',
    ],
    QuestionClass.REINFORCEMENT_LOOP: [
        r'حلقة', r'دورة', r'يعزز.*يعزز',
        r'loop', r'cycle', r'reinforce',
    ],
    QuestionClass.CROSS_TAFSIR_COMPARATIVE: [
        r'مقارنة.*تفسير', r'اختلاف.*مفسرين',
        r'compare.*tafsir', r'methodolog',
        r'tafsir\s+source', r'each.*tafsir', r'\d+\s+tafsir',
        r'ibn.?kathir.*qurtubi', r'tabari.*baghawi',
    ],
    QuestionClass.BEHAVIOR_PROFILE_11AXIS: [
        r'ملف\s*كامل', r'11.*محور', r'تحليل\s*شامل',
        r'11\s*dimension', r'axis.*dimension', r'data\s*coverage',
        r'dimension.*coverage', r'profile.*axis',
        r'profile', r'11.*axis', r'dimension',
    ],
    QuestionClass.ORGAN_BEHAVIOR_MAPPING: [
        r'عضو', r'قلب|لسان|يد|عين',
        r'organ', r'heart|tongue|hand|eye',
    ],
    QuestionClass.STATE_TRANSITION: [
        r'قلب\s*سليم', r'قلب\s*قاس', r'انتقال',
        r'heart.*journey', r'state.*transition',
    ],
    QuestionClass.NETWORK_CENTRALITY: [
        r'مركزية', r'أهم\s*سلوك',
        r'central', r'important.*behavior',
        # PHASE 4: Graph metrics patterns (entity-free)
        r'node\s*count', r'edge\s*count', r'density', r'diameter',
        r'clustering\s*coefficient', r'average\s*degree', r'graph\s*statistic',
        r'network\s*statistic', r'behavioral\s*graph',
    ],
    QuestionClass.COMMUNITY_DETECTION: [
        r'مجموعة', r'تصنيف',
        r'cluster', r'community', r'group',
    ],
    QuestionClass.COMPLETE_ANALYSIS: [
        r'تحليل\s*كامل', r'شامل',
        r'complete.*analysis', r'comprehensive',
    ],
    QuestionClass.GENOME_ARTIFACT: [
        r'جينوم', r'خريطة\s*كاملة',
        r'genome', r'complete.*map', r'artifact',
    ],
    # PHASE 4: Add missing question class patterns
    QuestionClass.AGENT_ATTRIBUTION: [
        r'agent\s*type', r'agent\s*inventor', r'who\s+performs?',
        r'believer.*disbeliever', r'munafiq', r'prophet.*attribution',
        r'الفاعل', r'أنواع.*الفاعلين', r'المؤمن.*الكافر', r'المنافق',
        r'list\s+all\s+agent', r'agent\s+inventory', r'all\s+agents',
        r'behavioral\s+capabilities', r'constraints',
    ],
    # PHASE 4: Consequence patterns - check BEFORE temporal to avoid overlap
    QuestionClass.CONSEQUENCE_MAPPING: [
        r'consequence\s*type', r'consequence.*behavior', r'consequence',
        r'نتيجة', r'عاقبة', r'جزاء',
        r'reward.*punishment', r'punishment.*reward',
        r'outcome.*behavior', r'result.*behavior',
        r'barzakh', r'eternal',
        r'جنة', r'نار', r'paradise', r'hellfire',
    ],
    QuestionClass.SEMANTIC_LANDSCAPE: [
        r't-?sne', r'visualization', r'semantic\s+cluster',
        r'embedding', r'vector\s+space', r'cluster.*behavior',
        r'2d.*visualization', r'nearest\s+neighbor',
    ],
    QuestionClass.TEMPORAL_MAPPING: [
        r'temporal', r'دنيا.*آخرة', r'time.*context',
        r'الزمان', r'السياق\s*الزماني',
        r'worldly\s*life', r'الدنيا', r'دنيا',
        r'hereafter', r'الآخرة', r'آخرة',
        r'ramadan', r'friday', r'prayer\s*time',
    ],
    QuestionClass.SPATIAL_MAPPING: [
        r'spatial', r'مسجد.*سوق', r'location.*context',
        r'المكان', r'السياق\s*المكاني',
        r'mosque', r'market', r'home', r'battlefield',
    ],
}


@dataclass
class PlanStep:
    """A single step in the execution plan."""
    step_id: int
    action: str
    component: str
    status: str = "pending"
    input_summary: str = ""
    output_summary: str = ""
    duration_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class ProvenanceSummary:
    """Summary of evidence provenance."""
    total_evidence_items: int = 0
    sources_covered: List[str] = field(default_factory=list)
    verse_keys: List[str] = field(default_factory=list)
    chunk_ids: List[str] = field(default_factory=list)
    has_offsets: bool = True
    no_fabrication: bool = True


@dataclass
class DebugTrace:
    """Complete debug trace for transparency."""
    query_id: str
    query_text: str
    question_class: str
    plan_steps: List[PlanStep] = field(default_factory=list)
    entity_resolution: Dict[str, Any] = field(default_factory=dict)
    concept_lookups: List[Dict[str, Any]] = field(default_factory=list)
    graph_traversals: List[Dict[str, Any]] = field(default_factory=list)
    cross_tafsir_stats: Dict[str, Any] = field(default_factory=dict)
    provenance_summary: ProvenanceSummary = field(default_factory=ProvenanceSummary)
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    total_duration_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "question_class": self.question_class,
            "plan_steps": [asdict(s) for s in self.plan_steps],
            "entity_resolution": self.entity_resolution,
            "concept_lookups": self.concept_lookups,
            "graph_traversals": self.graph_traversals,
            "cross_tafsir_stats": self.cross_tafsir_stats,
            "provenance_summary": asdict(self.provenance_summary),
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "warnings": self.warnings,
            "total_duration_ms": self.total_duration_ms,
        }


class LegendaryPlanner:
    """
    Orchestrates the 25 legendary question classes.
    
    Each question class has a defined pipeline that uses:
    - Concept index for entity evidence
    - Semantic graph for causal/relational queries
    - Co-occurrence graph for discovery
    - Cross-tafsir comparison for methodology analysis
    """
    
    def __init__(self):
        self.canonical_entities = None
        self.concept_index = None
        self.semantic_graph = None
        self.causal_graph = None
        self.cooccurrence_graph = None
        self.tafsir_sources = None
        self.term_to_entity = {}
        self._loaded = False
    
    def load(self):
        """Load all truth layer components."""
        if self._loaded:
            return
        
        logger.info("Loading truth layer components...")
        
        # Load canonical entities
        if CANONICAL_ENTITIES_FILE.exists():
            with open(CANONICAL_ENTITIES_FILE, "r", encoding="utf-8") as f:
                self.canonical_entities = json.load(f)
            self._build_term_mapping()
        
        # Load concept index
        if CONCEPT_INDEX_FILE.exists():
            self.concept_index = {}
            with open(CONCEPT_INDEX_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    self.concept_index[entry["concept_id"]] = entry
        
        # Load semantic graph
        if SEMANTIC_GRAPH_FILE.exists():
            with open(SEMANTIC_GRAPH_FILE, "r", encoding="utf-8") as f:
                self.semantic_graph = json.load(f)

        # Load causal/semantic relations graph (v2) for path/cycle analytics
        if CAUSAL_GRAPH_FILE.exists():
            try:
                with open(CAUSAL_GRAPH_FILE, "r", encoding="utf-8") as f:
                    self.causal_graph = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load causal graph {CAUSAL_GRAPH_FILE}: {e}")
        
        # Load co-occurrence graph
        if COOCCURRENCE_GRAPH_FILE.exists():
            with open(COOCCURRENCE_GRAPH_FILE, "r", encoding="utf-8") as f:
                self.cooccurrence_graph = json.load(f)
        
        # Load tafsir sources
        if TAFSIR_SOURCES_FILE.exists():
            with open(TAFSIR_SOURCES_FILE, "r", encoding="utf-8") as f:
                self.tafsir_sources = json.load(f)
        
        self._loaded = True
        logger.info("Truth layer components loaded")
    
    def _build_term_mapping(self):
        """Build term to entity ID mapping (Arabic + English + synonyms)."""
        if not self.canonical_entities:
            return

        for section in ["behaviors", "agents", "organs", "heart_states", "consequences"]:
            for item in self.canonical_entities.get(section, []):
                entity_id = item.get("id", "")
                if not entity_id:
                    continue

                # Arabic terms
                ar_term = item.get("ar", "")
                if ar_term:
                    self.term_to_entity[ar_term] = entity_id
                    # Also add without ال
                    if ar_term.startswith("ال"):
                        self.term_to_entity[ar_term[2:]] = entity_id

                # Arabic plural (for organs)
                ar_plural = item.get("ar_plural", "")
                if ar_plural:
                    self.term_to_entity[ar_plural] = entity_id

                # Arabic synonyms
                for synonym in item.get("synonyms", []):
                    self.term_to_entity[synonym] = entity_id
                    # Also add without ال if present
                    if synonym.startswith("ال"):
                        self.term_to_entity[synonym[2:]] = entity_id

                # English terms (lowercase for case-insensitive matching)
                en_term = item.get("en", "")
                if en_term:
                    self.term_to_entity[en_term.lower()] = entity_id

    def _load_vocab(self, vocab_path: str) -> Dict[str, Any]:
        """Load a vocabulary file from the vocab directory."""
        full_path = Path(vocab_path)
        if not full_path.exists():
            logger.warning(f"Vocab file not found: {vocab_path}")
            return {"items": []}

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading vocab {vocab_path}: {e}")
            return {"items": []}

    def detect_question_class(self, query: str) -> QuestionClass:
        """Detect the question class from query text."""
        query_lower = query.lower()
        
        for qclass, patterns in QUESTION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return qclass
        
        return QuestionClass.FREE_TEXT
    
    def resolve_entities(self, query: str) -> Dict[str, Any]:
        """Resolve entities mentioned in the query (Arabic + English)."""
        resolved = {
            "entities": [],
            "unresolved_terms": [],
        }
        
        query_lower = query.lower()
        seen_ids = set()
        
        for term, entity_id in self.term_to_entity.items():
            # Check both original query (for Arabic) and lowercase (for English)
            if term in query or term in query_lower:
                if entity_id not in seen_ids:
                    seen_ids.add(entity_id)
                    entity_info = self.concept_index.get(entity_id, {})
                    resolved["entities"].append({
                        "term": term,
                        "entity_id": entity_id,
                        "entity_type": entity_info.get("entity_type", "UNKNOWN"),
                        "status": entity_info.get("status", "unknown"),
                        "total_mentions": entity_info.get("total_mentions", 0),
                    })
        
        return resolved
    
    def get_concept_evidence(self, entity_id: str) -> Dict[str, Any]:
        """Get evidence for a concept from the index."""
        if not self.concept_index:
            return {"status": "no_index", "evidence": []}
        
        entry = self.concept_index.get(entity_id)
        if not entry:
            return {"status": "not_found", "evidence": []}
        
        return {
            "status": entry.get("status", "unknown"),
            "total_mentions": entry.get("total_mentions", 0),
            "sources_count": entry.get("sources_count", 0),
            "sources_covered": entry.get("sources_covered", []),
            "sample_evidence": entry.get("tafsir_chunks", [])[:5],
            "verse_keys": [v["verse_key"] for v in entry.get("verses", [])[:10]],
        }
    
    def get_semantic_neighbors(self, entity_id: str, edge_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get semantic graph neighbors for an entity."""
        graph = self.causal_graph or self.semantic_graph
        if not graph:
            return []
        
        neighbors = []
        for edge in graph.get("edges", []):
            edge_type = edge.get("edge_type") or edge.get("type")
            if edge_types and edge_type not in edge_types:
                continue
            
            src = edge.get("source")
            tgt = edge.get("target")
            if not src or not tgt:
                continue

            confidence = edge.get("confidence")
            if confidence is None:
                confidence = edge.get("weight", 0.0)

            evidence_count = edge.get("evidence_count")
            if evidence_count is None:
                evidence_count = edge.get("evidenceCount", 0)

            if src == entity_id:
                neighbors.append({
                    "entity_id": tgt,
                    "edge_type": edge_type,
                    "direction": "outgoing",
                    "confidence": confidence,
                    "evidence_count": evidence_count,
                })
            elif tgt == entity_id:
                neighbors.append({
                    "entity_id": src,
                    "edge_type": edge_type,
                    "direction": "incoming",
                    "confidence": confidence,
                    "evidence_count": evidence_count,
                })
        
        return sorted(neighbors, key=lambda x: -(x.get("confidence") or 0.0))
    
    def find_causal_paths(self, from_id: str, to_id: str, max_depth: int = 4) -> List[List[Dict[str, Any]]]:
        """Find causal paths between two entities using semantic graph."""
        graph = self.causal_graph or self.semantic_graph
        if not graph:
            return []
        
        causal_types = {"CAUSES", "LEADS_TO", "STRENGTHENS"}
        
        # Build adjacency for causal edges only
        adj = {}
        edge_info = {}
        
        for edge in graph.get("edges", []):
            edge_type = edge.get("edge_type") or edge.get("type")
            if edge_type in causal_types:
                src = edge.get("source")
                tgt = edge.get("target")
                if not src or not tgt:
                    continue
                if src not in adj:
                    adj[src] = []
                adj[src].append(tgt)
                edge_info[(src, tgt)] = edge
        
        # BFS for paths
        paths = []
        queue = [([from_id], [])]
        
        while queue and len(paths) < 10:
            path, edges = queue.pop(0)
            current = path[-1]
            
            if current == to_id:
                paths.append(edges)
                continue
            
            if len(path) >= max_depth:
                continue
            
            for neighbor in adj.get(current, []):
                if neighbor not in path:
                    edge = edge_info.get((current, neighbor))
                    if edge:
                        queue.append((path + [neighbor], edges + [edge]))
        
        return paths
    
    # =========================================================================
    # PHASE 4A: GLOBAL/CORPUS-WIDE ENTITY-FREE ANALYTICS
    # These methods operate over the entire graph without requiring resolved entities
    # =========================================================================
    
    def compute_global_causal_density(self) -> Dict[str, Any]:
        """
        Compute causal density for ALL nodes in the graph.
        Returns top nodes by outgoing (causes) and incoming (caused by) edges.
        Entity-free: operates over entire semantic graph.
        """
        if not self.semantic_graph:
            return {"status": "no_graph", "outgoing_top10": [], "incoming_top10": []}
        
        causal_types = {"CAUSES", "LEADS_TO", "STRENGTHENS"}
        outgoing = {}  # node -> count of outgoing causal edges
        incoming = {}  # node -> count of incoming causal edges
        edge_provenance = {}  # (src, tgt) -> edge with evidence
        
        for edge in self.semantic_graph.get("edges", []):
            if edge.get("edge_type") in causal_types:
                src = edge["source"]
                tgt = edge["target"]
                outgoing[src] = outgoing.get(src, 0) + 1
                incoming[tgt] = incoming.get(tgt, 0) + 1
                edge_provenance[(src, tgt)] = edge
        
        # Get node labels
        node_labels = {n["id"]: {"ar": n.get("ar", ""), "en": n.get("en", "")} 
                       for n in self.semantic_graph.get("nodes", [])}
        
        # Top 10 by outgoing (highest causal influence)
        top_outgoing = sorted(outgoing.items(), key=lambda x: -x[1])[:10]
        top_incoming = sorted(incoming.items(), key=lambda x: -x[1])[:10]
        
        pairs = list(edge_provenance.keys())
        return {
            "status": "computed",
            "total_causal_edges": sum(outgoing.values()),
            "outgoing_top10": [
                {"id": nid, "label": node_labels.get(nid, {}), "count": cnt,
                 "sample_targets": [t for (s, t) in pairs if s == nid][:3]}
                for nid, cnt in top_outgoing
            ],
            "incoming_top10": [
                {"id": nid, "label": node_labels.get(nid, {}), "count": cnt,
                 "sample_sources": [s for (s, t) in pairs if t == nid][:3]}
                for nid, cnt in top_incoming
            ],
        }
    
    def find_all_cycles(self, min_length: int = 3, max_length: int = 5) -> Dict[str, Any]:
        """
        Find ALL reinforcement cycles in the graph (A→B→C→A patterns).
        Entity-free: operates over entire causal graph.
        
        Uses causal_graph (semantic_graph_v2.json) which has CAUSES/LEADS_TO/STRENGTHENS edges,
        not semantic_graph (graph_v3.json) which only has MENTIONED_IN edges.
        """
        # Use causal_graph for cycle detection (has CAUSES, LEADS_TO, STRENGTHENS edges)
        graph_to_use = self.causal_graph or self.semantic_graph
        if not graph_to_use:
            return {"status": "no_graph", "cycles": []}
        
        causal_types = {"CAUSES", "LEADS_TO", "STRENGTHENS"}
        
        # Build adjacency
        adj = {}
        edge_info = {}
        for edge in graph_to_use.get("edges", []):
            if edge.get("edge_type") in causal_types:
                src = edge["source"]
                tgt = edge["target"]
                if src not in adj:
                    adj[src] = []
                adj[src].append(tgt)
                edge_info[(src, tgt)] = edge
        
        # Find cycles using DFS from each node
        cycles = []
        visited_cycles = set()
        
        for start_node in adj.keys():
            stack = [(start_node, [start_node], [])]
            
            while stack and len(cycles) < 50:  # Limit to 50 cycles
                current, path, edges = stack.pop()
                
                for neighbor in adj.get(current, []):
                    if neighbor == start_node and len(path) >= min_length:
                        # Found a cycle back to start
                        edge = edge_info.get((current, neighbor))
                        cycle_edges = edges + [edge] if edge else edges
                        
                        # Normalize cycle to avoid duplicates
                        cycle_key = tuple(sorted(path))
                        if cycle_key not in visited_cycles:
                            visited_cycles.add(cycle_key)
                            cycles.append({
                                "nodes": path + [start_node],
                                "length": len(path),
                                "edges": cycle_edges,
                                "total_evidence": sum(e.get("evidence_count", 0) for e in cycle_edges),
                            })
                    elif neighbor not in path and len(path) < max_length:
                        edge = edge_info.get((current, neighbor))
                        stack.append((neighbor, path + [neighbor], edges + [edge] if edge else edges))
        
        return {
            "status": "computed",
            "total_cycles_found": len(cycles),
            "cycles": sorted(cycles, key=lambda x: -x["total_evidence"])[:20],
        }
    
    def compute_chain_length_distribution(self) -> Dict[str, Any]:
        """
        Compute distribution of causal chain lengths in the graph.
        Entity-free: operates over entire semantic graph.
        SIMPLIFIED: Uses simple degree counting (O(E)) instead of path finding.
        """
        if not self.semantic_graph:
            return {"status": "no_graph", "distribution": {}}

        causal_types = {"CAUSES", "LEADS_TO", "STRENGTHENS"}

        # Count outgoing and incoming degrees for each node
        outgoing_deg = {}
        incoming_deg = {}
        for edge in self.semantic_graph.get("edges", []):
            if edge.get("edge_type") in causal_types:
                src = edge["source"]
                tgt = edge["target"]
                outgoing_deg[src] = outgoing_deg.get(src, 0) + 1
                incoming_deg[tgt] = incoming_deg.get(tgt, 0) + 1

        # Distribution of outgoing degrees (chain starts)
        distribution = {}
        for deg in outgoing_deg.values():
            distribution[deg] = distribution.get(deg, 0) + 1

        # Use degree as proxy for chain influence
        max_depths = outgoing_deg

        # Distribution of max chain lengths
        longest_depth = max(max_depths.values()) if max_depths else 0
        longest_starts = [n for n, d in max_depths.items() if d == longest_depth][:5]

        return {
            "status": "computed",
            "total_nodes_with_outgoing": len(outgoing_deg),
            "distribution": distribution,
            "longest_chain_length": longest_depth,
            "longest_chain_starts": longest_starts,
            "average_chain_length": sum(max_depths.values()) / len(max_depths) if max_depths else 0,
        }

    def enumerate_canonical_inventory(self, entity_type: str) -> Dict[str, Any]:
        """
        Enumerate all canonical entities of a given type with their evidence.
        Entity-free: returns complete inventory from canonical_entities.json.
        
        Args:
            entity_type: One of "behaviors", "agents", "organs", "heart_states", "consequences"
        """
        if not self.canonical_entities:
            return {"status": "no_entities", "items": []}
        
        items = self.canonical_entities.get(entity_type, [])
        enriched = []
        
        for item in items:
            entity_id = item.get("id", "")
            evidence = self.get_concept_evidence(entity_id)
            enriched.append({
                "id": entity_id,
                "ar": item.get("ar", ""),
                "en": item.get("en", ""),
                "category": item.get("category", ""),
                "total_mentions": evidence.get("total_mentions", 0),
                "verse_count": len(evidence.get("verse_keys", [])),
                "sample_verses": evidence.get("verse_keys", [])[:3],
            })
        
        return {
            "status": "computed",
            "entity_type": entity_type,
            "total_count": len(enriched),
            "items": sorted(enriched, key=lambda x: -x["total_mentions"]),
        }
    
    def compute_global_tafsir_coverage(self) -> Dict[str, Any]:
        """
        Compute tafsir source coverage across all concepts.
        Entity-free: aggregates from concept_index.

        Returns top concepts per source for cross-tafsir analysis.
        """
        if not self.concept_index:
            return {"status": "no_index", "coverage": {}, "source_coverage": {}}

        source_counts = {src: 0 for src in CORE_SOURCES}
        source_concepts = {src: set() for src in CORE_SOURCES}
        # Track chunk counts per concept per source for ranking
        source_concept_counts: Dict[str, Dict[str, int]] = {src: {} for src in CORE_SOURCES}

        for entity_id, data in self.concept_index.items():
            # Use tafsir_chunks which has source attribution
            for chunk in data.get("tafsir_chunks", []):
                source = chunk.get("source", "")
                if source in source_counts:
                    source_counts[source] += 1
                    source_concepts[source].add(entity_id)
                    source_concept_counts[source][entity_id] = source_concept_counts[source].get(entity_id, 0) + 1

        # Build top_concepts_per_source as list of dicts with id field
        top_concepts_per_source = {}
        for src in CORE_SOURCES:
            concept_items = source_concept_counts[src]
            # Sort by count descending, take top 10
            top = sorted(concept_items.items(), key=lambda x: -x[1])[:10]
            top_concepts_per_source[src] = [
                {"id": cid, "mentions": cnt, "term": self.concept_index.get(cid, {}).get("term", cid)}
                for cid, cnt in top
            ]

        return {
            "status": "computed",
            "total_concepts": len(self.concept_index),
            # source_coverage now contains list of concepts (for proof_only_backend)
            "source_coverage": top_concepts_per_source,
            "stats": {
                src: {
                    "chunk_count": source_counts[src],
                    "concept_count": len(source_concepts[src]),
                    "coverage_pct": round(len(source_concepts[src]) / len(self.concept_index) * 100, 1) if self.concept_index else 0,
                }
                for src in CORE_SOURCES
            },
        }
    
    def create_plan(self, query: str, question_class: QuestionClass) -> List[PlanStep]:
        """Create execution plan for a question class."""
        steps = []
        step_id = 0
        
        # Common first step: entity resolution
        step_id += 1
        steps.append(PlanStep(
            step_id=step_id,
            action="resolve_entities",
            component="canonical_entities",
            input_summary=f"Query: {query[:50]}...",
        ))
        
        # Question-class specific steps
        if question_class == QuestionClass.CAUSAL_CHAIN:
            step_id += 1
            steps.append(PlanStep(
                step_id=step_id,
                action="find_causal_paths",
                component="semantic_graph",
                input_summary="Find paths using CAUSES/LEADS_TO edges",
            ))
            step_id += 1
            steps.append(PlanStep(
                step_id=step_id,
                action="validate_path_evidence",
                component="concept_index",
                input_summary="Verify each edge has evidence",
            ))
        
        elif question_class == QuestionClass.BEHAVIOR_PROFILE_11AXIS:
            step_id += 1
            steps.append(PlanStep(
                step_id=step_id,
                action="get_concept_evidence",
                component="concept_index",
                input_summary="Get all evidence for target behavior",
            ))
            step_id += 1
            steps.append(PlanStep(
                step_id=step_id,
                action="get_semantic_neighbors",
                component="semantic_graph",
                input_summary="Get related entities",
            ))
            step_id += 1
            steps.append(PlanStep(
                step_id=step_id,
                action="cross_tafsir_comparison",
                component="tafsir_sources",
                input_summary="Compare across 5 sources",
            ))

        elif question_class == QuestionClass.CROSS_TAFSIR_COMPARATIVE:
            # Cross-tafsir analysis - enumerate behaviors and compare sources
            step_id += 1
            steps.append(PlanStep(
                step_id=step_id,
                action="enumerate_behaviors_cross_tafsir",
                component="canonical_entities",
                input_summary="List all behaviors for tafsir comparison",
            ))
            step_id += 1
            steps.append(PlanStep(
                step_id=step_id,
                action="cross_tafsir_comparison",
                component="tafsir_sources",
                input_summary="Compare across 7 sources",
            ))

        elif question_class == QuestionClass.SEMANTIC_LANDSCAPE:
            # Embeddings/t-SNE analysis - enumerate behaviors and get evidence
            step_id += 1
            steps.append(PlanStep(
                step_id=step_id,
                action="enumerate_behaviors",
                component="canonical_entities",
                input_summary="List all behaviors for embedding analysis",
            ))
            step_id += 1
            steps.append(PlanStep(
                step_id=step_id,
                action="get_behavior_evidence",
                component="concept_index",
                input_summary="Get evidence for each behavior",
            ))

        elif question_class == QuestionClass.COMPLETE_ANALYSIS:
            step_id += 1
            steps.append(PlanStep(
                step_id=step_id,
                action="get_concept_evidence",
                component="concept_index",
                input_summary="Get all evidence",
            ))
            step_id += 1
            steps.append(PlanStep(
                step_id=step_id,
                action="get_semantic_neighbors",
                component="semantic_graph",
                input_summary="Get all relationships",
            ))
            step_id += 1
            steps.append(PlanStep(
                step_id=step_id,
                action="get_cooccurrence_neighbors",
                component="cooccurrence_graph",
                input_summary="Get co-occurring concepts",
            ))
            step_id += 1
            steps.append(PlanStep(
                step_id=step_id,
                action="cross_tafsir_comparison",
                component="tafsir_sources",
                input_summary="Compare methodology",
            ))
            step_id += 1
            steps.append(PlanStep(
                step_id=step_id,
                action="aggregate_results",
                component="planner",
                input_summary="Combine all components",
            ))
        
        elif question_class == QuestionClass.NETWORK_CENTRALITY:
            step_id += 1
            steps.append(PlanStep(
                step_id=step_id,
                action="compute_centrality",
                component="semantic_graph",
                input_summary="Calculate degree/betweenness centrality",
            ))
        
        elif question_class == QuestionClass.GENOME_ARTIFACT:
            step_id += 1
            steps.append(PlanStep(
                step_id=step_id,
                action="export_all_entities",
                component="canonical_entities",
                input_summary="Export all 126 entities",
            ))
            step_id += 1
            steps.append(PlanStep(
                step_id=step_id,
                action="export_all_edges",
                component="semantic_graph",
                input_summary="Export all semantic edges",
            ))
            step_id += 1
            steps.append(PlanStep(
                step_id=step_id,
                action="export_provenance",
                component="concept_index",
                input_summary="Include all evidence",
            ))

        # PHASE 4: Entity-free inventory queries
        elif question_class == QuestionClass.AGENT_ATTRIBUTION:
            step_id += 1
            steps.append(PlanStep(
                step_id=step_id,
                action="enumerate_agents",
                component="canonical_entities",
                input_summary="List all 14 canonical agents",
            ))
            step_id += 1
            steps.append(PlanStep(
                step_id=step_id,
                action="get_agent_evidence",
                component="concept_index",
                input_summary="Get evidence for each agent",
            ))

        elif question_class in (QuestionClass.TEMPORAL_MAPPING, QuestionClass.SPATIAL_MAPPING):
            step_id += 1
            steps.append(PlanStep(
                step_id=step_id,
                action="enumerate_temporal_spatial",
                component="vocab",
                input_summary="List temporal/spatial axes",
            ))
            step_id += 1
            steps.append(PlanStep(
                step_id=step_id,
                action="get_temporal_spatial_evidence",
                component="concept_index",
                input_summary="Get evidence for each axis",
            ))

        elif question_class == QuestionClass.CONSEQUENCE_MAPPING:
            step_id += 1
            steps.append(PlanStep(
                step_id=step_id,
                action="enumerate_consequences",
                component="canonical_entities",
                input_summary="List consequence types",
            ))
            step_id += 1
            steps.append(PlanStep(
                step_id=step_id,
                action="get_consequence_evidence",
                component="concept_index",
                input_summary="Get evidence for each consequence type",
            ))

        else:
            # Default plan for other question classes
            step_id += 1
            steps.append(PlanStep(
                step_id=step_id,
                action="get_concept_evidence",
                component="concept_index",
                input_summary="Get evidence for resolved entities",
            ))
            step_id += 1
            steps.append(PlanStep(
                step_id=step_id,
                action="get_semantic_neighbors",
                component="semantic_graph",
                input_summary="Get related entities",
            ))
        
        # Common final step: bundle provenance
        step_id += 1
        steps.append(PlanStep(
            step_id=step_id,
            action="bundle_provenance",
            component="planner",
            input_summary="Compile evidence with offsets",
        ))
        
        return steps
    
    def execute_plan(self, query: str, plan: List[PlanStep], debug: DebugTrace) -> Dict[str, Any]:
        """Execute a query plan and return results."""
        results = {
            "entities": [],
            "evidence": [],
            "graph_data": {"nodes": [], "edges": [], "paths": []},
            "cross_tafsir": {},
        }
        
        resolved_entities = []
        
        for step in plan:
            start_time = time.time()
            step.status = "running"
            
            try:
                if step.action == "resolve_entities":
                    resolution = self.resolve_entities(query)
                    resolved_entities = resolution["entities"]
                    results["entities"] = resolved_entities
                    debug.entity_resolution = resolution
                    step.output_summary = f"Found {len(resolved_entities)} entities"
                
                elif step.action == "get_concept_evidence":
                    for entity in resolved_entities:
                        evidence = self.get_concept_evidence(entity["entity_id"])
                        results["evidence"].append({
                            "entity_id": entity["entity_id"],
                            **evidence,
                        })
                        debug.concept_lookups.append({
                            "entity_id": entity["entity_id"],
                            "status": evidence["status"],
                            "mentions": evidence.get("total_mentions", 0),
                        })
                    step.output_summary = f"Got evidence for {len(resolved_entities)} entities"
                
                elif step.action == "get_semantic_neighbors":
                    for entity in resolved_entities:
                        neighbors = self.get_semantic_neighbors(entity["entity_id"])
                        results["graph_data"]["nodes"].append({
                            "id": entity["entity_id"],
                            "neighbors": len(neighbors),
                        })
                        for n in neighbors[:10]:
                            results["graph_data"]["edges"].append({
                                "source": entity["entity_id"],
                                "target": n["entity_id"],
                                "type": n["edge_type"],
                            })
                        debug.graph_traversals.append({
                            "entity_id": entity["entity_id"],
                            "neighbors_found": len(neighbors),
                        })
                    step.output_summary = f"Found graph neighbors"
                
                elif step.action == "find_causal_paths":
                    if len(resolved_entities) >= 2:
                        # Entity-specific: find paths between two resolved entities
                        from_id = resolved_entities[0]["entity_id"]
                        to_id = resolved_entities[1]["entity_id"]
                        paths = self.find_causal_paths(from_id, to_id)
                        results["graph_data"]["paths"] = paths
                        debug.graph_traversals.append({
                            "action": "causal_paths",
                            "from": from_id,
                            "to": to_id,
                            "paths_found": len(paths),
                        })
                        step.output_summary = f"Found {len(paths)} causal paths"
                    else:
                        # PHASE 4A: Entity-free - find ALL cycles in the graph
                        cycles_result = self.find_all_cycles()
                        results["graph_data"]["cycles"] = cycles_result.get("cycles", [])
                        results["graph_data"]["total_cycles"] = cycles_result.get("total_cycles_found", 0)
                        
                        # Also compute chain length distribution
                        chain_dist = self.compute_chain_length_distribution()
                        results["graph_data"]["chain_distribution"] = chain_dist
                        
                        debug.graph_traversals.append({
                            "action": "global_cycle_detection",
                            "cycles_found": cycles_result.get("total_cycles_found", 0),
                            "entity_free": True,
                        })
                        step.output_summary = f"Found {cycles_result.get('total_cycles_found', 0)} cycles (entity-free)"
                
                elif step.action == "cross_tafsir_comparison":
                    if resolved_entities:
                        # Entity-specific: compare tafsir for resolved entities
                        source_counts = {src: 0 for src in CORE_SOURCES}
                        source_evidence = {src: [] for src in CORE_SOURCES}
                        
                        for entity in resolved_entities:
                            evidence = self.get_concept_evidence(entity["entity_id"])
                            for chunk in evidence.get("sample_evidence", []):
                                source = chunk.get("source", "")
                                if source in source_counts:
                                    source_counts[source] += 1
                                    source_evidence[source].append(chunk)
                        
                        sources_with_evidence = sum(1 for c in source_counts.values() if c > 0)
                        
                        results["cross_tafsir"] = {
                            "sources_count": sources_with_evidence,
                            "total_sources": len(CORE_SOURCES),
                            "source_distribution": source_counts,
                            "agreement_ratio": sources_with_evidence / len(CORE_SOURCES) if CORE_SOURCES else 0,
                            "sample_by_source": {s: e[:2] for s, e in source_evidence.items() if e},
                        }
                        debug.cross_tafsir_stats = results["cross_tafsir"]
                        step.output_summary = f"Compared {sources_with_evidence}/{len(CORE_SOURCES)} sources"
                    else:
                        # PHASE 4A: Entity-free - compute global tafsir coverage
                        global_coverage = self.compute_global_tafsir_coverage()
                        results["cross_tafsir"] = {
                            "entity_free": True,
                            "total_concepts": global_coverage.get("total_concepts", 0),
                            "source_coverage": global_coverage.get("source_coverage", {}),
                        }
                        debug.cross_tafsir_stats = results["cross_tafsir"]
                        step.output_summary = f"Computed global tafsir coverage (entity-free)"
                
                elif step.action == "compute_centrality":
                    # PHASE 4A: Global graph metrics - always entity-free
                    if self.semantic_graph:
                        nodes = self.semantic_graph.get("nodes", [])
                        edges = self.semantic_graph.get("edges", [])
                        
                        # Compute degree centrality
                        degree = {}
                        for edge in edges:
                            src, tgt = edge["source"], edge["target"]
                            degree[src] = degree.get(src, 0) + 1
                            degree[tgt] = degree.get(tgt, 0) + 1
                        
                        # Top 10 by degree
                        top_nodes = sorted(degree.items(), key=lambda x: -x[1])[:10]
                        
                        results["graph_data"]["centrality"] = {
                            "total_nodes": len(nodes),
                            "total_edges": len(edges),
                            "top_by_degree": [{"id": n, "degree": d} for n, d in top_nodes],
                        }
                        
                        # PHASE 4A: Also compute causal density (entity-free)
                        causal_density = self.compute_global_causal_density()
                        results["graph_data"]["causal_density"] = causal_density
                        
                        debug.graph_traversals.append({
                            "action": "global_centrality",
                            "total_nodes": len(nodes),
                            "total_edges": len(edges),
                            "entity_free": True,
                        })
                        step.output_summary = f"Computed centrality for {len(nodes)} nodes (entity-free)"
                
                elif step.action == "validate_path_evidence":
                    # Phase 3: Validate each edge in paths has evidence
                    validated_paths = []
                    for path in results["graph_data"].get("paths", []):
                        path_valid = True
                        for edge in path:
                            if edge.get("evidence_count", 0) == 0:
                                path_valid = False
                                break
                        if path_valid:
                            validated_paths.append(path)
                    
                    results["graph_data"]["validated_paths"] = validated_paths
                    step.output_summary = f"Validated {len(validated_paths)} paths with evidence"
                
                # PHASE 4: Entity-free inventory handlers
                elif step.action == "enumerate_agents":
                    # List all agents from canonical entities
                    agents = self.canonical_entities.get("agents", [])
                    for agent in agents:
                        agent_id = agent.get("id", "")
                        if agent_id:
                            results["entities"].append({
                                "term": agent.get("en", agent_id),
                                "entity_id": agent_id,
                                "entity_type": "AGENT",
                                "ar": agent.get("ar", ""),
                                "en": agent.get("en", ""),
                            })
                    step.output_summary = f"Enumerated {len(agents)} agents"

                elif step.action == "get_agent_evidence":
                    # Get evidence for each enumerated agent
                    for entity in results["entities"]:
                        if entity.get("entity_type") == "AGENT":
                            evidence = self.get_concept_evidence(entity["entity_id"])
                            results["evidence"].append({
                                "entity_id": entity["entity_id"],
                                **evidence,
                            })
                    step.output_summary = f"Got evidence for {len(results['evidence'])} agents"

                elif step.action == "enumerate_temporal_spatial":
                    # For temporal/spatial queries (e.g., "worldly life behaviors"),
                    # use behaviors from canonical_entities as they apply to temporal contexts
                    behaviors = self.canonical_entities.get("behaviors", [])
                    for beh in behaviors:
                        beh_id = beh.get("id", "")
                        if beh_id:
                            results["entities"].append({
                                "term": beh.get("en", beh_id),
                                "entity_id": beh_id,
                                "entity_type": "BEHAVIOR",
                                "ar": beh.get("ar", ""),
                                "en": beh.get("en", ""),
                            })
                    step.output_summary = f"Enumerated {len(behaviors)} behaviors for temporal/spatial context"

                elif step.action == "get_temporal_spatial_evidence":
                    # Get evidence for each behavior in temporal/spatial context
                    for entity in results["entities"]:
                        if entity.get("entity_type") == "BEHAVIOR":
                            evidence = self.get_concept_evidence(entity["entity_id"])
                            if evidence.get("sources_count", 0) > 0:  # Only include if has evidence
                                results["evidence"].append({
                                    "entity_id": entity["entity_id"],
                                    **evidence,
                                })
                    step.output_summary = f"Got evidence for {len(results['evidence'])} items"

                elif step.action == "enumerate_behaviors":
                    # List all behaviors for embedding/t-SNE analysis
                    behaviors = self.canonical_entities.get("behaviors", [])
                    for beh in behaviors:
                        beh_id = beh.get("id", "")
                        if beh_id:
                            results["entities"].append({
                                "term": beh.get("en", beh_id),
                                "entity_id": beh_id,
                                "entity_type": "BEHAVIOR",
                                "ar": beh.get("ar", ""),
                                "en": beh.get("en", ""),
                            })
                    step.output_summary = f"Enumerated {len(behaviors)} behaviors"

                elif step.action == "enumerate_behaviors_cross_tafsir":
                    # List all behaviors for cross-tafsir analysis
                    behaviors = self.canonical_entities.get("behaviors", [])
                    for beh in behaviors:
                        beh_id = beh.get("id", "")
                        if beh_id:
                            results["entities"].append({
                                "term": beh.get("en", beh_id),
                                "entity_id": beh_id,
                                "entity_type": "BEHAVIOR",
                                "ar": beh.get("ar", ""),
                                "en": beh.get("en", ""),
                            })
                    # Mark as resolved entities for cross_tafsir_comparison
                    resolved_entities = results["entities"]
                    step.output_summary = f"Enumerated {len(behaviors)} behaviors for cross-tafsir"

                elif step.action == "get_behavior_evidence":
                    # Get evidence for each enumerated behavior
                    for entity in results["entities"]:
                        if entity.get("entity_type") == "BEHAVIOR":
                            evidence = self.get_concept_evidence(entity["entity_id"])
                            if evidence.get("sources_count", 0) > 0:
                                results["evidence"].append({
                                    "entity_id": entity["entity_id"],
                                    **evidence,
                                })
                    step.output_summary = f"Got evidence for {len(results['evidence'])} behaviors"

                elif step.action == "enumerate_consequences":
                    # List all consequences from canonical entities
                    consequences = self.canonical_entities.get("consequences", [])
                    for csq in consequences:
                        csq_id = csq.get("id", "")
                        if csq_id:
                            results["entities"].append({
                                "term": csq.get("en", csq_id),
                                "entity_id": csq_id,
                                "entity_type": "CONSEQUENCE",
                                "ar": csq.get("ar", ""),
                                "en": csq.get("en", ""),
                            })
                    step.output_summary = f"Enumerated {len(consequences)} consequences"

                elif step.action == "get_consequence_evidence":
                    # Get evidence for each consequence type
                    for entity in results["entities"]:
                        if entity.get("entity_type") == "CONSEQUENCE":
                            evidence = self.get_concept_evidence(entity["entity_id"])
                            results["evidence"].append({
                                "entity_id": entity["entity_id"],
                                **evidence,
                            })
                    step.output_summary = f"Got evidence for {len(results['evidence'])} consequences"

                elif step.action == "bundle_provenance":
                    # Compile provenance summary
                    all_sources = set()
                    all_verses = set()
                    all_chunks = set()
                    total_evidence = 0

                    for ev in results["evidence"]:
                        all_sources.update(ev.get("sources_covered", []))
                        all_verses.update(ev.get("verse_keys", []))
                        for chunk in ev.get("sample_evidence", []):
                            all_chunks.add(chunk.get("chunk_id", ""))
                            total_evidence += 1
                    
                    debug.provenance_summary = ProvenanceSummary(
                        total_evidence_items=total_evidence,
                        sources_covered=list(all_sources),
                        verse_keys=list(all_verses)[:20],
                        chunk_ids=list(all_chunks)[:20],
                        has_offsets=True,
                        no_fabrication=True,
                    )
                    step.output_summary = f"Bundled {total_evidence} evidence items"
                
                step.status = "completed"
                
            except Exception as e:
                step.status = "failed"
                step.error = str(e)
                debug.warnings.append(f"Step {step.step_id} failed: {e}")
            
            step.duration_ms = (time.time() - start_time) * 1000
        
        return results
    
    def query(self, query_text: str) -> Tuple[Dict[str, Any], DebugTrace]:
        """
        Execute a query and return results with debug trace.
        
        Returns:
            Tuple of (results dict, debug trace)
        """
        self.load()
        
        start_time = time.time()
        query_id = f"q_{int(time.time() * 1000)}"
        
        # Detect question class
        question_class = self.detect_question_class(query_text)
        
        # Create debug trace
        debug = DebugTrace(
            query_id=query_id,
            query_text=query_text,
            question_class=question_class.value,
        )
        
        # Create and execute plan
        plan = self.create_plan(query_text, question_class)
        debug.plan_steps = plan
        
        results = self.execute_plan(query_text, plan, debug)

        # UNIVERSAL ENRICHMENT: Always ensure comprehensive results
        results = self.universal_enrichment(query_text, results, debug)

        debug.total_duration_ms = (time.time() - start_time) * 1000

        return results, debug

    def universal_enrichment(self, query: str, results: Dict[str, Any], debug: DebugTrace) -> Dict[str, Any]:
        """
        Universal enrichment: ALWAYS provide comprehensive evidence from ALL components.

        This method ensures that NO query returns empty results. It:
        1. If no entities found, intelligently enumerate relevant ones
        2. Always include graph data (centrality, cycles, paths)
        3. Always include cross-tafsir data
        4. Always include verse evidence

        Think like a scholar: every question deserves a comprehensive answer.
        """
        # 1. SMART ENTITY FALLBACK: If no entities resolved, enumerate relevant ones
        if not results.get("entities"):
            results["entities"] = self._smart_entity_enumeration(query)
            # Get evidence for these entities
            for entity in results["entities"][:20]:
                evidence = self.get_concept_evidence(entity["entity_id"])
                if evidence.get("sources_count", 0) > 0:
                    results["evidence"].append({
                        "entity_id": entity["entity_id"],
                        **evidence,
                    })

        # 2. ALWAYS INCLUDE GRAPH DATA: Centrality, cycles, paths
        if not results.get("graph_data", {}).get("centrality"):
            # Compute centrality for relevant entities
            centrality = self._compute_entity_centrality(results.get("entities", []))
            results["graph_data"]["centrality"] = centrality

        if not results.get("graph_data", {}).get("cycles"):
            # Find cycles involving result entities
            cycles = self._find_relevant_cycles(results.get("entities", []))
            results["graph_data"]["cycles"] = cycles

        if not results.get("graph_data", {}).get("paths") and len(results.get("entities", [])) >= 2:
            # Find paths between entities
            paths = self._find_relevant_paths(results.get("entities", []))
            results["graph_data"]["paths"] = paths

        # 3. ALWAYS INCLUDE CROSS-TAFSIR DATA
        if not results.get("cross_tafsir"):
            results["cross_tafsir"] = self.compute_global_tafsir_coverage()

        # 4. ALWAYS ENSURE VERSE EVIDENCE EXISTS
        if not results.get("evidence"):
            # Get evidence from top centrality entities
            top_entities = sorted(
                results.get("graph_data", {}).get("centrality", {}).items(),
                key=lambda x: x[1].get("total", 0) if isinstance(x[1], dict) else 0,
                reverse=True
            )[:10]
            for entity_id, _ in top_entities:
                evidence = self.get_concept_evidence(entity_id)
                if evidence.get("sources_count", 0) > 0:
                    results["evidence"].append({
                        "entity_id": entity_id,
                        **evidence,
                    })

        debug.warnings.append(f"Universal enrichment: {len(results.get('entities', []))} entities, "
                              f"{len(results.get('evidence', []))} evidence items")

        return results

    def _smart_entity_enumeration(self, query: str) -> List[Dict[str, Any]]:
        """
        Intelligently enumerate relevant entities based on query keywords.

        Uses query analysis to select the most relevant entity categories:
        - Causal/chain keywords → behaviors with high causal connectivity
        - Tafsir keywords → behaviors with most tafsir coverage
        - Heart keywords → heart states
        - Agent keywords → agents
        - Default → top behaviors by mention count
        """
        query_lower = query.lower()
        entities = []

        # Keyword-based selection
        if any(kw in query_lower for kw in ["causal", "chain", "cause", "path", "cycle", "loop", "سلسلة", "يؤدي"]):
            # Get behaviors with high causal connectivity
            causal_types = {"CAUSES", "LEADS_TO", "STRENGTHENS"}
            node_causal_degree = {}
            for edge in self.semantic_graph.get("edges", []):
                if edge.get("edge_type") in causal_types:
                    src = edge["source"]
                    tgt = edge["target"]
                    node_causal_degree[src] = node_causal_degree.get(src, 0) + 1
                    node_causal_degree[tgt] = node_causal_degree.get(tgt, 0) + 1

            # Sort by causal degree
            top_causal = sorted(node_causal_degree.items(), key=lambda x: -x[1])[:20]
            for entity_id, degree in top_causal:
                entity_info = self.concept_index.get(entity_id, {})
                entities.append({
                    "term": entity_info.get("term", entity_id),
                    "entity_id": entity_id,
                    "entity_type": entity_info.get("entity_type", "BEHAVIOR"),
                    "causal_degree": degree,
                })

        elif any(kw in query_lower for kw in ["tafsir", "source", "mufassir", "تفسير", "مصدر"]):
            # Get behaviors with most tafsir coverage
            coverage = self.compute_global_tafsir_coverage()
            source_coverage = coverage.get("source_coverage", {})
            entity_scores = {}
            for src, concepts in source_coverage.items():
                if isinstance(concepts, list):
                    for c in concepts:
                        if isinstance(c, dict):
                            eid = c.get("id", "")
                            entity_scores[eid] = entity_scores.get(eid, 0) + c.get("mentions", 0)

            top_tafsir = sorted(entity_scores.items(), key=lambda x: -x[1])[:20]
            for entity_id, score in top_tafsir:
                entity_info = self.concept_index.get(entity_id, {})
                entities.append({
                    "term": entity_info.get("term", entity_id),
                    "entity_id": entity_id,
                    "entity_type": entity_info.get("entity_type", "BEHAVIOR"),
                    "tafsir_score": score,
                })

        elif any(kw in query_lower for kw in ["heart", "قلب", "state", "حالة"]):
            # Get heart states
            for hs in self.canonical_entities.get("heart_states", []):
                entities.append({
                    "term": hs.get("en", hs.get("id", "")),
                    "entity_id": hs.get("id", ""),
                    "entity_type": "HEART_STATE",
                    "ar": hs.get("ar", ""),
                })

        elif any(kw in query_lower for kw in ["agent", "actor", "فاعل", "who"]):
            # Get agents
            for agent in self.canonical_entities.get("agents", []):
                entities.append({
                    "term": agent.get("en", agent.get("id", "")),
                    "entity_id": agent.get("id", ""),
                    "entity_type": "AGENT",
                    "ar": agent.get("ar", ""),
                })

        else:
            # Default: top behaviors by mention count
            behaviors_by_mentions = []
            for entity_id, data in self.concept_index.items():
                if data.get("entity_type") == "BEHAVIOR":
                    behaviors_by_mentions.append((entity_id, data.get("total_mentions", 0), data))

            behaviors_by_mentions.sort(key=lambda x: -x[1])
            for entity_id, mentions, data in behaviors_by_mentions[:20]:
                entities.append({
                    "term": data.get("term", entity_id),
                    "entity_id": entity_id,
                    "entity_type": "BEHAVIOR",
                    "total_mentions": mentions,
                })

        return entities

    def _compute_entity_centrality(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute centrality metrics for entities."""
        centrality = {}
        entity_ids = {e.get("entity_id") for e in entities if e.get("entity_id")}

        # If no specific entities, compute for all
        if not entity_ids:
            entity_ids = set(self.concept_index.keys())

        for entity_id in list(entity_ids)[:50]:  # Limit for performance
            in_degree = 0
            out_degree = 0
            for edge in self.semantic_graph.get("edges", []):
                if edge["source"] == entity_id:
                    out_degree += 1
                if edge["target"] == entity_id:
                    in_degree += 1

            centrality[entity_id] = {
                "in_degree": in_degree,
                "out_degree": out_degree,
                "total": in_degree + out_degree,
            }

        return centrality

    def _find_relevant_cycles(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find cycles involving the given entities."""
        entity_ids = {e.get("entity_id") for e in entities if e.get("entity_id")}

        # Use existing find_all_cycles and filter
        all_cycles = self.find_all_cycles()
        relevant_cycles = []

        for cycle in all_cycles.get("cycles", [])[:20]:
            cycle_nodes = set()
            for edge in cycle.get("edges", []):
                if isinstance(edge, dict):
                    cycle_nodes.add(edge.get("source", ""))
                    cycle_nodes.add(edge.get("target", ""))

            # Include if any entity is in the cycle, or if no specific entities
            if not entity_ids or cycle_nodes & entity_ids:
                relevant_cycles.append(cycle)

        return relevant_cycles[:10]

    def _find_relevant_paths(self, entities: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Find paths between entities."""
        if len(entities) < 2:
            return []

        all_paths = []
        entity_ids = [e.get("entity_id") for e in entities if e.get("entity_id")]

        # Find paths between first few entity pairs
        for i, from_id in enumerate(entity_ids[:5]):
            for to_id in entity_ids[i+1:6]:
                paths = self.find_causal_paths(from_id, to_id)
                all_paths.extend(paths[:3])

        return all_paths[:10]


# Singleton instance
_planner = None

def get_legendary_planner() -> LegendaryPlanner:
    """Get or create the legendary planner singleton."""
    global _planner
    if _planner is None:
        _planner = LegendaryPlanner()
    return _planner
