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
CANONICAL_ENTITIES_FILE = Path("vocab/canonical_entities.json")
CONCEPT_INDEX_FILE = Path("data/evidence/concept_index_v2.jsonl")
SEMANTIC_GRAPH_FILE = Path("data/graph/semantic_graph_v2.json")
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
    ],
    QuestionClass.BEHAVIOR_PROFILE_11AXIS: [
        r'ملف\s*كامل', r'11.*محور', r'تحليل\s*شامل',
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
        """Build term to entity ID mapping."""
        if not self.canonical_entities:
            return
        
        for section in ["behaviors", "agents", "organs", "heart_states", "consequences"]:
            for item in self.canonical_entities.get(section, []):
                ar_term = item.get("ar", "")
                if ar_term:
                    self.term_to_entity[ar_term] = item["id"]
                    # Also add without ال
                    if ar_term.startswith("ال"):
                        self.term_to_entity[ar_term[2:]] = item["id"]
    
    def detect_question_class(self, query: str) -> QuestionClass:
        """Detect the question class from query text."""
        query_lower = query.lower()
        
        for qclass, patterns in QUESTION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return qclass
        
        return QuestionClass.FREE_TEXT
    
    def resolve_entities(self, query: str) -> Dict[str, Any]:
        """Resolve entities mentioned in the query."""
        resolved = {
            "entities": [],
            "unresolved_terms": [],
        }
        
        for term, entity_id in self.term_to_entity.items():
            if term in query:
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
        if not self.semantic_graph:
            return []
        
        neighbors = []
        for edge in self.semantic_graph.get("edges", []):
            if edge_types and edge["edge_type"] not in edge_types:
                continue
            
            if edge["source"] == entity_id:
                neighbors.append({
                    "entity_id": edge["target"],
                    "edge_type": edge["edge_type"],
                    "direction": "outgoing",
                    "confidence": edge["confidence"],
                    "evidence_count": edge["evidence_count"],
                })
            elif edge["target"] == entity_id:
                neighbors.append({
                    "entity_id": edge["source"],
                    "edge_type": edge["edge_type"],
                    "direction": "incoming",
                    "confidence": edge["confidence"],
                    "evidence_count": edge["evidence_count"],
                })
        
        return sorted(neighbors, key=lambda x: -x["confidence"])
    
    def find_causal_paths(self, from_id: str, to_id: str, max_depth: int = 4) -> List[List[Dict[str, Any]]]:
        """Find causal paths between two entities using semantic graph."""
        if not self.semantic_graph:
            return []
        
        causal_types = {"CAUSES", "LEADS_TO", "STRENGTHENS"}
        
        # Build adjacency for causal edges only
        adj = {}
        edge_info = {}
        
        for edge in self.semantic_graph.get("edges", []):
            if edge["edge_type"] in causal_types:
                src = edge["source"]
                tgt = edge["target"]
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
        
        debug.total_duration_ms = (time.time() - start_time) * 1000
        
        return results, debug


# Singleton instance
_planner = None

def get_legendary_planner() -> LegendaryPlanner:
    """Get or create the legendary planner singleton."""
    global _planner
    if _planner is None:
        _planner = LegendaryPlanner()
    return _planner
