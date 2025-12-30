"""
Query Planner with Debug Traceability (Phase 7.1)

Orchestrates the Truth Layer components to answer complex queries:
- Canonical entity resolution
- Concept index lookup
- Graph traversal (co-occurrence + semantic)
- Cross-tafsir evidence aggregation
- Debug trace for full transparency
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries the planner can handle."""
    BEHAVIOR_PROFILE = "behavior_profile"      # Q7: 11-dimensional profile
    CAUSAL_CHAIN = "causal_chain"              # Q1: A → B with evidence
    CONCEPT_ANALYSIS = "concept_analysis"      # Q23: Complete analysis
    VERSE_LOOKUP = "verse_lookup"              # Direct verse reference
    SURAH_ANALYSIS = "surah_analysis"          # Surah-level analysis
    GRAPH_QUERY = "graph_query"                # Graph traversal query


@dataclass
class PlanStep:
    """A single step in the query execution plan."""
    step_id: int
    action: str
    component: str
    input_data: Dict[str, Any]
    status: str = "pending"
    output_data: Optional[Dict[str, Any]] = None
    duration_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class QueryPlan:
    """Complete execution plan for a query."""
    query_id: str
    query_text: str
    query_type: QueryType
    target_entity: Optional[str] = None
    steps: List[PlanStep] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    total_duration_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "query_type": self.query_type.value,
            "target_entity": self.target_entity,
            "steps": [asdict(s) for s in self.steps],
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "total_duration_ms": self.total_duration_ms,
        }


@dataclass
class DebugTrace:
    """Debug trace for query execution."""
    query_id: str
    plan: QueryPlan
    entity_resolution: Dict[str, Any] = field(default_factory=dict)
    concept_lookups: List[Dict[str, Any]] = field(default_factory=list)
    graph_traversals: List[Dict[str, Any]] = field(default_factory=list)
    evidence_sources: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "plan": self.plan.to_dict(),
            "entity_resolution": self.entity_resolution,
            "concept_lookups": self.concept_lookups,
            "graph_traversals": self.graph_traversals,
            "evidence_sources": self.evidence_sources,
            "warnings": self.warnings,
        }


class QueryPlanner:
    """
    Orchestrates Truth Layer components for complex queries.
    
    Components used:
    - vocab/canonical_entities.json: Entity resolution
    - data/evidence/concept_index_v1.jsonl: Concept evidence lookup
    - data/graph/cooccurrence_graph_v1.json: Statistical relationships
    - data/graph/semantic_graph_v1.json: Causal relationships
    - vocab/tafsir_sources.json: Source weighting
    """
    
    def __init__(self, base_path: Path = None):
        self.base_path = base_path or Path(".")
        self.canonical_entities = None
        self.concept_index = None
        self.cooccurrence_graph = None
        self.semantic_graph = None
        self.tafsir_sources = None
        self._term_to_entity = {}
        self._loaded = False
    
    def load(self) -> None:
        """Load all Truth Layer components."""
        if self._loaded:
            return
        
        logger.info("Loading Truth Layer components...")
        
        # Load canonical entities
        entities_path = self.base_path / "vocab" / "canonical_entities.json"
        if entities_path.exists():
            with open(entities_path, "r", encoding="utf-8") as f:
                self.canonical_entities = json.load(f)
            self._build_term_to_entity()
            logger.info(f"Loaded canonical entities: {len(self._term_to_entity)} terms")
        
        # Load concept index
        concept_path = self.base_path / "data" / "evidence" / "concept_index_v1.jsonl"
        if concept_path.exists():
            self.concept_index = {}
            with open(concept_path, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    self.concept_index[entry["concept_id"]] = entry
            logger.info(f"Loaded concept index: {len(self.concept_index)} concepts")
        
        # Load graphs
        cooc_path = self.base_path / "data" / "graph" / "cooccurrence_graph_v1.json"
        if cooc_path.exists():
            with open(cooc_path, "r", encoding="utf-8") as f:
                self.cooccurrence_graph = json.load(f)
            logger.info(f"Loaded co-occurrence graph: {self.cooccurrence_graph['edge_count']} edges")
        
        semantic_path = self.base_path / "data" / "graph" / "semantic_graph_v1.json"
        if semantic_path.exists():
            with open(semantic_path, "r", encoding="utf-8") as f:
                self.semantic_graph = json.load(f)
            logger.info(f"Loaded semantic graph: {self.semantic_graph['edge_count']} edges")
        
        # Load tafsir sources
        tafsir_path = self.base_path / "vocab" / "tafsir_sources.json"
        if tafsir_path.exists():
            with open(tafsir_path, "r", encoding="utf-8") as f:
                self.tafsir_sources = json.load(f)
            logger.info(f"Loaded tafsir sources: {len(self.tafsir_sources.get('core_sources', []))} sources")
        
        self._loaded = True
    
    def _build_term_to_entity(self) -> None:
        """Build term to entity mapping from canonical entities (includes synonyms)."""
        if not self.canonical_entities:
            return

        def add_term(term: str, entity_id: str):
            """Helper to add a term and its variants."""
            if term:
                self._term_to_entity[term] = entity_id
                if term.startswith("ال"):
                    self._term_to_entity[term[2:]] = entity_id

        for behavior in self.canonical_entities.get("behaviors", []):
            entity_id = behavior["id"]
            add_term(behavior.get("ar", ""), entity_id)
            add_term(behavior.get("en", "").lower(), entity_id)
            for syn in behavior.get("synonyms", []):
                add_term(syn, entity_id)

        for agent in self.canonical_entities.get("agents", []):
            entity_id = agent["id"]
            add_term(agent.get("ar", ""), entity_id)
            add_term(agent.get("ar_plural", ""), entity_id)

        for organ in self.canonical_entities.get("organs", []):
            entity_id = organ["id"]
            add_term(organ.get("ar", ""), entity_id)
            add_term(organ.get("ar_plural", ""), entity_id)
            for syn in organ.get("synonyms", []):
                add_term(syn, entity_id)

        for state in self.canonical_entities.get("heart_states", []):
            entity_id = state["id"]
            add_term(state.get("ar", ""), entity_id)

        for consequence in self.canonical_entities.get("consequences", []):
            entity_id = consequence["id"]
            add_term(consequence.get("ar", ""), entity_id)
    
    def resolve_entity(self, term: str) -> Optional[str]:
        """Resolve Arabic term to canonical entity ID."""
        return self._term_to_entity.get(term)
    
    def detect_query_type(self, query: str) -> QueryType:
        """Detect the type of query from the text."""
        query_lower = query.lower()
        
        # Check for behavior profile patterns
        if any(p in query for p in ["ملف", "تحليل شامل", "11", "أحد عشر"]):
            return QueryType.BEHAVIOR_PROFILE
        
        # Check for causal chain patterns
        if any(p in query for p in ["يؤدي", "سبب", "نتيجة", "→", "->"]):
            return QueryType.CAUSAL_CHAIN
        
        # Check for verse reference
        if any(p in query for p in ["آية", "سورة", ":"]):
            if "سورة" in query and "آية" not in query:
                return QueryType.SURAH_ANALYSIS
            return QueryType.VERSE_LOOKUP
        
        # Check for graph query patterns
        if any(p in query for p in ["علاقة", "ارتباط", "جيران"]):
            return QueryType.GRAPH_QUERY
        
        # Default to concept analysis
        return QueryType.CONCEPT_ANALYSIS
    
    def create_plan(self, query: str, query_id: str = None) -> QueryPlan:
        """Create an execution plan for the query."""
        self.load()
        
        query_id = query_id or f"q_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        query_type = self.detect_query_type(query)
        
        plan = QueryPlan(
            query_id=query_id,
            query_text=query,
            query_type=query_type,
        )
        
        # Step 1: Entity resolution
        plan.steps.append(PlanStep(
            step_id=1,
            action="resolve_entities",
            component="canonical_entities",
            input_data={"query": query},
        ))
        
        # Add type-specific steps
        if query_type == QueryType.BEHAVIOR_PROFILE:
            self._add_behavior_profile_steps(plan)
        elif query_type == QueryType.CAUSAL_CHAIN:
            self._add_causal_chain_steps(plan)
        elif query_type == QueryType.CONCEPT_ANALYSIS:
            self._add_concept_analysis_steps(plan)
        elif query_type == QueryType.GRAPH_QUERY:
            self._add_graph_query_steps(plan)
        
        # Final step: Aggregate results
        plan.steps.append(PlanStep(
            step_id=len(plan.steps) + 1,
            action="aggregate_results",
            component="query_planner",
            input_data={"query_type": query_type.value},
        ))
        
        return plan
    
    def _add_behavior_profile_steps(self, plan: QueryPlan) -> None:
        """Add steps for behavior profile query (Q7 style)."""
        plan.steps.extend([
            PlanStep(
                step_id=2,
                action="lookup_concept_evidence",
                component="concept_index",
                input_data={"include_offsets": True},
            ),
            PlanStep(
                step_id=3,
                action="get_cooccurrence_neighbors",
                component="cooccurrence_graph",
                input_data={"max_neighbors": 20},
            ),
            PlanStep(
                step_id=4,
                action="get_semantic_edges",
                component="semantic_graph",
                input_data={"edge_types": ["CAUSES", "LEADS_TO", "OPPOSITE_OF", "PREVENTS"]},
            ),
            PlanStep(
                step_id=5,
                action="compute_11_axes",
                component="axes_classifier",
                input_data={"axes_version": "bouzidani_v1"},
            ),
            PlanStep(
                step_id=6,
                action="aggregate_cross_tafsir",
                component="tafsir_sources",
                input_data={"min_sources": 2},
            ),
        ])
    
    def _add_causal_chain_steps(self, plan: QueryPlan) -> None:
        """Add steps for causal chain query (Q1 style)."""
        plan.steps.extend([
            PlanStep(
                step_id=2,
                action="find_semantic_path",
                component="semantic_graph",
                input_data={"edge_types": ["CAUSES", "LEADS_TO"], "max_hops": 3},
            ),
            PlanStep(
                step_id=3,
                action="validate_path_evidence",
                component="concept_index",
                input_data={"require_multi_source": True},
            ),
            PlanStep(
                step_id=4,
                action="compute_chain_confidence",
                component="tafsir_sources",
                input_data={"apply_methodology_weights": True},
            ),
        ])
    
    def _add_concept_analysis_steps(self, plan: QueryPlan) -> None:
        """Add steps for concept analysis query (Q23 style)."""
        plan.steps.extend([
            PlanStep(
                step_id=2,
                action="lookup_concept_evidence",
                component="concept_index",
                input_data={"include_offsets": True},
            ),
            PlanStep(
                step_id=3,
                action="get_all_graph_edges",
                component="semantic_graph",
                input_data={"include_cooccurrence": True},
            ),
            PlanStep(
                step_id=4,
                action="aggregate_cross_tafsir",
                component="tafsir_sources",
                input_data={"min_sources": 2},
            ),
        ])
    
    def _add_graph_query_steps(self, plan: QueryPlan) -> None:
        """Add steps for graph traversal query."""
        plan.steps.extend([
            PlanStep(
                step_id=2,
                action="get_cooccurrence_neighbors",
                component="cooccurrence_graph",
                input_data={"max_neighbors": 30},
            ),
            PlanStep(
                step_id=3,
                action="get_semantic_edges",
                component="semantic_graph",
                input_data={"edge_types": "all"},
            ),
        ])
    
    def execute_plan(self, plan: QueryPlan) -> DebugTrace:
        """Execute the query plan and return debug trace."""
        import time
        
        trace = DebugTrace(query_id=plan.query_id, plan=plan)
        start_time = time.time()
        
        for step in plan.steps:
            step_start = time.time()
            step.status = "running"
            
            try:
                if step.action == "resolve_entities":
                    result = self._execute_resolve_entities(step, trace)
                elif step.action == "lookup_concept_evidence":
                    result = self._execute_concept_lookup(step, trace)
                elif step.action == "get_cooccurrence_neighbors":
                    result = self._execute_cooccurrence_neighbors(step, trace)
                elif step.action == "get_semantic_edges":
                    result = self._execute_semantic_edges(step, trace)
                elif step.action == "find_semantic_path":
                    result = self._execute_semantic_path(step, trace)
                elif step.action == "aggregate_results":
                    result = self._execute_aggregate(step, trace)
                else:
                    result = {"status": "skipped", "reason": f"Unknown action: {step.action}"}
                
                step.output_data = result
                step.status = "completed"
                
            except Exception as e:
                step.status = "failed"
                step.error = str(e)
                trace.warnings.append(f"Step {step.step_id} failed: {e}")
            
            step.duration_ms = (time.time() - step_start) * 1000
        
        plan.completed_at = datetime.now().isoformat()
        plan.total_duration_ms = (time.time() - start_time) * 1000
        
        return trace
    
    def _execute_resolve_entities(self, step: PlanStep, trace: DebugTrace) -> Dict[str, Any]:
        """Execute entity resolution step."""
        query = step.input_data.get("query", "")
        resolved = {}
        
        for term, entity_id in self._term_to_entity.items():
            if term in query:
                resolved[term] = entity_id
        
        trace.entity_resolution = {
            "query": query,
            "resolved_terms": resolved,
            "count": len(resolved),
        }
        
        # Set target entity on plan
        if resolved:
            trace.plan.target_entity = list(resolved.values())[0]
        
        return resolved
    
    def _execute_concept_lookup(self, step: PlanStep, trace: DebugTrace) -> Dict[str, Any]:
        """Execute concept index lookup."""
        target = trace.plan.target_entity
        if not target or not self.concept_index:
            return {"status": "no_target", "evidence": []}
        
        concept = self.concept_index.get(target)
        if not concept:
            return {"status": "not_found", "concept_id": target}
        
        lookup_result = {
            "concept_id": target,
            "term": concept.get("term"),
            "total_mentions": concept.get("total_mentions", 0),
            "verse_count": len(concept.get("verses", [])),
            "per_source_stats": concept.get("per_source_stats", {}),
        }
        
        trace.concept_lookups.append(lookup_result)
        return lookup_result
    
    def _execute_cooccurrence_neighbors(self, step: PlanStep, trace: DebugTrace) -> Dict[str, Any]:
        """Get co-occurrence neighbors from graph."""
        target = trace.plan.target_entity
        if not target or not self.cooccurrence_graph:
            return {"status": "no_target", "neighbors": []}
        
        max_neighbors = step.input_data.get("max_neighbors", 20)
        neighbors = []
        
        for edge in self.cooccurrence_graph.get("edges", []):
            if edge["source"] == target:
                neighbors.append({
                    "node": edge["target"],
                    "count": edge["count"],
                    "pmi": edge["pmi"],
                })
            elif edge["target"] == target:
                neighbors.append({
                    "node": edge["source"],
                    "count": edge["count"],
                    "pmi": edge["pmi"],
                })
        
        neighbors.sort(key=lambda x: -x["count"])
        neighbors = neighbors[:max_neighbors]
        
        trace.graph_traversals.append({
            "graph_type": "cooccurrence",
            "target": target,
            "neighbor_count": len(neighbors),
        })
        
        return {"neighbors": neighbors, "count": len(neighbors)}
    
    def _execute_semantic_edges(self, step: PlanStep, trace: DebugTrace) -> Dict[str, Any]:
        """Get semantic edges from graph."""
        target = trace.plan.target_entity
        if not target or not self.semantic_graph:
            return {"status": "no_target", "edges": []}
        
        edge_types = step.input_data.get("edge_types", "all")
        edges = []
        
        for edge in self.semantic_graph.get("edges", []):
            if edge["source"] == target or edge["target"] == target:
                if edge_types == "all" or edge["edge_type"] in edge_types:
                    edges.append({
                        "source": edge["source"],
                        "target": edge["target"],
                        "edge_type": edge["edge_type"],
                        "confidence": edge["confidence"],
                        "evidence_count": edge["evidence_count"],
                    })
        
        trace.graph_traversals.append({
            "graph_type": "semantic",
            "target": target,
            "edge_count": len(edges),
            "edge_types": list(set(e["edge_type"] for e in edges)),
        })
        
        return {"edges": edges, "count": len(edges)}
    
    def _execute_semantic_path(self, step: PlanStep, trace: DebugTrace) -> Dict[str, Any]:
        """Find semantic path between entities."""
        # This would implement BFS/DFS path finding
        # For now, return placeholder
        return {"status": "not_implemented", "paths": []}
    
    def _execute_aggregate(self, step: PlanStep, trace: DebugTrace) -> Dict[str, Any]:
        """Aggregate all results."""
        return {
            "entity_resolved": trace.entity_resolution.get("count", 0) > 0,
            "concept_lookups": len(trace.concept_lookups),
            "graph_traversals": len(trace.graph_traversals),
            "evidence_sources": len(trace.evidence_sources),
            "warnings": len(trace.warnings),
        }
    
    def query(self, query_text: str) -> Dict[str, Any]:
        """Execute a query and return results with debug trace."""
        plan = self.create_plan(query_text)
        trace = self.execute_plan(plan)
        
        return {
            "query": query_text,
            "query_type": plan.query_type.value,
            "target_entity": plan.target_entity,
            "results": self._compile_results(trace),
            "debug_trace": trace.to_dict(),
        }
    
    def _compile_results(self, trace: DebugTrace) -> Dict[str, Any]:
        """Compile results from trace into structured output."""
        results = {
            "entity": trace.plan.target_entity,
            "concept_evidence": {},
            "graph_neighbors": [],
            "semantic_edges": [],
        }
        
        # Add concept evidence
        for lookup in trace.concept_lookups:
            results["concept_evidence"] = lookup
        
        # Add graph data from steps
        for step in trace.plan.steps:
            if step.output_data:
                if "neighbors" in step.output_data:
                    results["graph_neighbors"] = step.output_data["neighbors"]
                if "edges" in step.output_data:
                    results["semantic_edges"] = step.output_data["edges"]
        
        return results


def get_query_planner(base_path: Path = None) -> QueryPlanner:
    """Factory function to get a QueryPlanner instance."""
    planner = QueryPlanner(base_path)
    planner.load()
    return planner


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    planner = get_query_planner()
    
    # Test query
    result = planner.query("ما هو تحليل الحسد في القرآن؟")
    
    print("\n" + "="*60)
    print("Query Result")
    print("="*60)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
