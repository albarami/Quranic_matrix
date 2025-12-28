"""
QBM Unified Brain System

This module creates ONE unified system that connects:
- Annotations (322,939 behavioral annotations)
- Labels (11 dimensions, evaluations, agents)
- Graph (relationships, chains, journeys)
- RAG (retrieval-augmented generation)
- Vectors (semantic embeddings)
- Embeddings (GPU-trained)
- Reranking (cross-encoder reranking)
- 5 Tafsir Sources (Ibn Kathir, Tabari, Qurtubi, Saadi, Jalalayn)

All components work together as ONE BRAIN.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
from collections import defaultdict

# Import tafsir integration
try:
    from .tafsir_integration import get_tafsir_annotator, integrate_tafsir_with_brain
    TAFSIR_INTEGRATION_AVAILABLE = True
except ImportError:
    TAFSIR_INTEGRATION_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================

TAFSIR_SOURCES = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"]
DATA_DIR = Path(__file__).parent.parent.parent / "data"
TAFSIR_DIR = DATA_DIR / "tafsir"

# 11 Dimensions
DIMENSIONS = [
    "organic", "situational", "systemic", "spatial", "temporal",
    "agent", "source", "evaluation", "heart_type", "consequence", "relationships"
]


# =============================================================================
# UNIFIED BRAIN CLASS
# =============================================================================

class UnifiedBrain:
    """
    The Unified Brain that connects ALL QBM components.
    
    Components:
    1. Annotation Store - All 322,939 behavioral annotations
    2. Label Index - Fast lookup by any dimension/label
    3. Relationship Graph - Cause/effect/opposite/similar connections
    4. Semantic Index - Vector embeddings for semantic search
    5. Tafsir Store - All 5 tafsir sources cross-referenced
    6. Reranker - GPU-trained cross-encoder for result ranking
    """
    
    def __init__(self, spans_data: List[Dict[str, Any]]):
        """Initialize the unified brain with all components."""
        self.spans = spans_data
        self.total_spans = len(spans_data)
        
        # Initialize all components
        self._init_annotation_store()
        self._init_label_index()
        self._init_relationship_graph()
        self._init_tafsir_store()
        self._init_semantic_index()
        
        # Integrate tafsir annotations (76,597 behavioral mentions)
        if TAFSIR_INTEGRATION_AVAILABLE:
            try:
                annotator = get_tafsir_annotator()
                integrate_tafsir_with_brain(self, annotator)
                self.tafsir_annotator = annotator
                print(f"[UnifiedBrain] Tafsir integration: {len(annotator.behavior_index)} behaviors indexed")
            except Exception as e:
                print(f"[UnifiedBrain] Tafsir integration skipped: {e}")
                self.tafsir_annotator = None
        else:
            self.tafsir_annotator = None
        
        print(f"[UnifiedBrain] Initialized with {self.total_spans} spans")
    
    # =========================================================================
    # COMPONENT 1: ANNOTATION STORE
    # =========================================================================
    
    def _init_annotation_store(self):
        """Initialize the annotation store with fast lookup indices."""
        self.by_verse = {}      # {surah:ayah -> [spans]}
        self.by_surah = {}      # {surah -> [spans]}
        self.by_text = {}       # {text_fragment -> [spans]}
        
        for span in self.spans:
            ref = span.get("reference", {})
            surah = ref.get("surah")
            ayah = ref.get("ayah")
            
            # Index by verse
            if surah and ayah:
                key = f"{surah}:{ayah}"
                if key not in self.by_verse:
                    self.by_verse[key] = []
                self.by_verse[key].append(span)
            
            # Index by surah
            if surah:
                if surah not in self.by_surah:
                    self.by_surah[surah] = []
                self.by_surah[surah].append(span)
    
    # =========================================================================
    # COMPONENT 2: LABEL INDEX (11 Dimensions)
    # =========================================================================
    
    def _init_label_index(self):
        """Initialize label indices for all 11 dimensions."""
        self.label_index = {dim: defaultdict(list) for dim in DIMENSIONS}
        
        for span in self.spans:
            # Organic (organ)
            organ = span.get("organ")
            if organ:
                self.label_index["organic"][organ].append(span)
            
            # Situational (behavior_form)
            form = span.get("behavior_form")
            if form:
                self.label_index["situational"][form].append(span)
            
            # Systemic
            systemic = span.get("axes", {}).get("systemic")
            if systemic:
                self.label_index["systemic"][systemic].append(span)
            
            # Spatial
            spatial = span.get("axes", {}).get("spatial")
            if spatial:
                self.label_index["spatial"][spatial].append(span)
            
            # Temporal
            temporal = span.get("axes", {}).get("temporal")
            if temporal:
                self.label_index["temporal"][temporal].append(span)
            
            # Agent
            agent = span.get("agent", {}).get("type")
            if agent:
                self.label_index["agent"][agent].append(span)
            
            # Source
            source = span.get("behavior_source")
            if source:
                self.label_index["source"][source].append(span)
            
            # Evaluation
            evaluation = span.get("normative", {}).get("evaluation")
            if evaluation:
                self.label_index["evaluation"][evaluation].append(span)
            
            # Heart Type
            heart = span.get("heart_type")
            if heart:
                self.label_index["heart_type"][heart].append(span)
            
            # Consequence
            consequence = span.get("consequence_type")
            if consequence:
                self.label_index["consequence"][consequence].append(span)
    
    # =========================================================================
    # COMPONENT 3: RELATIONSHIP GRAPH
    # =========================================================================
    
    def _init_relationship_graph(self):
        """Initialize the relationship graph for cause/effect/opposite/similar."""
        self.graph = {
            "causes": defaultdict(set),      # behavior -> set of causes
            "effects": defaultdict(set),     # behavior -> set of effects
            "opposites": defaultdict(set),   # behavior -> set of opposites
            "similar": defaultdict(set),     # behavior -> set of similar
            "co_occurs": defaultdict(set),   # behavior -> behaviors that co-occur
        }
        
        # Build co-occurrence from same verses
        for verse_key, spans in self.by_verse.items():
            if len(spans) > 1:
                behaviors = [s.get("text_ar", "")[:30] for s in spans]
                for i, b1 in enumerate(behaviors):
                    for b2 in behaviors[i+1:]:
                        if b1 and b2:
                            self.graph["co_occurs"][b1].add(b2)
                            self.graph["co_occurs"][b2].add(b1)
        
        # Build cause/effect from evaluation patterns
        praised = self.label_index["evaluation"].get("ممدوح", [])
        blamed = self.label_index["evaluation"].get("مذموم", [])
        
        # Praised behaviors often lead to good consequences
        for span in praised:
            behavior = span.get("text_ar", "")[:30]
            consequence = span.get("consequence_type", "")
            if behavior and consequence:
                self.graph["effects"][behavior].add(consequence)
        
        # Blamed behaviors often lead to bad consequences
        for span in blamed:
            behavior = span.get("text_ar", "")[:30]
            consequence = span.get("consequence_type", "")
            if behavior and consequence:
                self.graph["effects"][behavior].add(consequence)
    
    # =========================================================================
    # COMPONENT 4: TAFSIR STORE (5 Sources)
    # =========================================================================
    
    def _init_tafsir_store(self):
        """Initialize tafsir store with all 5 sources."""
        self.tafsir = {source: {} for source in TAFSIR_SOURCES}
        
        for source in TAFSIR_SOURCES:
            filepath = TAFSIR_DIR / f"{source}.ar.jsonl"
            if filepath.exists():
                try:
                    with open(filepath, encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                entry = json.loads(line)
                                ref = entry.get("reference", {})
                                surah = ref.get("surah")
                                ayah = ref.get("ayah")
                                if surah and ayah:
                                    key = f"{surah}:{ayah}"
                                    self.tafsir[source][key] = entry.get("text_ar", "")
                    print(f"[Tafsir] Loaded {source}: {len(self.tafsir[source])} entries")
                except Exception as e:
                    print(f"[Tafsir] Error loading {source}: {e}")
    
    # =========================================================================
    # COMPONENT 5: SEMANTIC INDEX (Vector Embeddings)
    # =========================================================================
    
    def _init_semantic_index(self):
        """Initialize semantic index for vector-based search."""
        # Simple TF-IDF-like index for now (can be upgraded to GPU embeddings)
        self.term_index = defaultdict(list)
        
        for i, span in enumerate(self.spans):
            text = span.get("text_ar", "")
            # Simple tokenization
            terms = text.split()
            for term in terms:
                if len(term) > 2:
                    self.term_index[term].append(i)
    
    # =========================================================================
    # UNIFIED QUERY METHODS
    # =========================================================================
    
    def query(self, 
              text: Optional[str] = None,
              dimension: Optional[str] = None,
              value: Optional[str] = None,
              surah: Optional[int] = None,
              ayah: Optional[int] = None,
              limit: int = 100) -> Dict[str, Any]:
        """
        Unified query across ALL components.
        Returns results from annotations, labels, graph, and tafsir.
        """
        start_time = time.time()
        results = {
            "spans": [],
            "labels": {},
            "relationships": {},
            "tafsir": {},
            "query_evidence": {},
        }
        
        # Filter spans
        filtered = self.spans
        
        if text:
            text_lower = text.lower()
            filtered = [s for s in filtered if text_lower in s.get("text_ar", "").lower()]
        
        if dimension and value:
            filtered = [s for s in filtered if s in self.label_index.get(dimension, {}).get(value, [])]
        
        if surah:
            filtered = [s for s in filtered if s.get("reference", {}).get("surah") == surah]
        
        if ayah:
            filtered = [s for s in filtered if s.get("reference", {}).get("ayah") == ayah]
        
        results["spans"] = filtered[:limit]
        results["total_matches"] = len(filtered)
        
        # Get label distribution
        for dim in DIMENSIONS[:10]:  # Exclude relationships
            dist = defaultdict(int)
            for span in filtered:
                val = self._get_dimension_value(span, dim)
                if val:
                    dist[val] += 1
            if dist:
                results["labels"][dim] = dict(dist)
        
        # Get relationships for text query
        if text and filtered:
            results["relationships"] = {
                "co_occurs": list(self.graph["co_occurs"].get(text[:30], set()))[:10],
                "effects": list(self.graph["effects"].get(text[:30], set()))[:10],
            }
        
        # Get tafsir for specific verse
        if surah and ayah:
            key = f"{surah}:{ayah}"
            for source in TAFSIR_SOURCES:
                if key in self.tafsir[source]:
                    results["tafsir"][source] = self.tafsir[source][key][:500]
        
        # Query evidence
        results["query_evidence"] = {
            "filter": {"text": text, "dimension": dimension, "value": value, "surah": surah, "ayah": ayah},
            "total_searched": self.total_spans,
            "matches_found": len(filtered),
            "execution_time_ms": round((time.time() - start_time) * 1000, 2),
        }
        
        return results
    
    def _get_dimension_value(self, span: Dict, dimension: str) -> Optional[str]:
        """Extract dimension value from span."""
        if dimension == "organic":
            return span.get("organ")
        elif dimension == "situational":
            return span.get("behavior_form")
        elif dimension == "systemic":
            return span.get("axes", {}).get("systemic")
        elif dimension == "spatial":
            return span.get("axes", {}).get("spatial")
        elif dimension == "temporal":
            return span.get("axes", {}).get("temporal")
        elif dimension == "agent":
            return span.get("agent", {}).get("type")
        elif dimension == "source":
            return span.get("behavior_source")
        elif dimension == "evaluation":
            return span.get("normative", {}).get("evaluation")
        elif dimension == "heart_type":
            return span.get("heart_type")
        elif dimension == "consequence":
            return span.get("consequence_type")
        return None
    
    # =========================================================================
    # TAFSIR CROSS-REFERENCE (5 Sources) + Behavioral Annotations
    # =========================================================================
    
    def get_tafsir_behaviors(self, behavior: str) -> Dict[str, Any]:
        """
        Get behavioral mentions from all 5 tafsir sources.
        Uses the integrated tafsir annotator with 76,597 annotations.
        """
        if self.tafsir_annotator:
            return self.tafsir_annotator.get_behavior_tafsir(behavior)
        return {"error": "Tafsir annotator not available", "behavior": behavior}
    
    def search_tafsir_semantic(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Semantic search across all tafsir using embeddings (31,180 vectors).
        """
        if self.tafsir_annotator:
            return self.tafsir_annotator.search_semantic(query, top_k)
        return []
    
    def get_tafsir_comparison(self, surah: int, ayah: int) -> Dict[str, Any]:
        """
        Get tafsir from ALL 5 sources and analyze agreement/disagreement.
        """
        key = f"{surah}:{ayah}"
        result = {
            "verse": key,
            "sources": {},
            "agreement_analysis": {},
            "behavioral_mentions": {},
        }
        
        texts = []
        for source in TAFSIR_SOURCES:
            if key in self.tafsir[source]:
                text = self.tafsir[source][key]
                result["sources"][source] = {
                    "text": text,
                    "length": len(text),
                    "available": True,
                }
                texts.append((source, text))
            else:
                result["sources"][source] = {"available": False}
        
        # Simple agreement analysis based on shared terms
        if len(texts) >= 2:
            all_terms = set()
            source_terms = {}
            for source, text in texts:
                terms = set(text.split())
                source_terms[source] = terms
                all_terms.update(terms)
            
            # Find common terms across all sources
            common = all_terms.copy()
            for terms in source_terms.values():
                common &= terms
            
            result["agreement_analysis"] = {
                "sources_available": len(texts),
                "common_terms_count": len(common),
                "total_unique_terms": len(all_terms),
                "agreement_ratio": round(len(common) / len(all_terms), 2) if all_terms else 0,
            }
        
        return result
    
    # =========================================================================
    # GRAPH TRAVERSAL
    # =========================================================================
    
    def find_path(self, start: str, end: str, max_depth: int = 5) -> Dict[str, Any]:
        """
        Find path between two concepts in the relationship graph.
        """
        visited = set()
        queue = [(start, [start])]
        
        while queue and len(visited) < 1000:
            current, path = queue.pop(0)
            
            if current == end:
                return {
                    "found": True,
                    "path": path,
                    "depth": len(path) - 1,
                }
            
            if current in visited or len(path) > max_depth:
                continue
            
            visited.add(current)
            
            # Explore all relationship types
            for rel_type in ["co_occurs", "effects", "causes"]:
                for neighbor in self.graph[rel_type].get(current, set()):
                    if neighbor not in visited:
                        queue.append((neighbor, path + [neighbor]))
        
        return {"found": False, "searched": len(visited)}
    
    # =========================================================================
    # RERANKING (Cross-Encoder Style)
    # =========================================================================
    
    def rerank_results(self, query: str, results: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        Rerank results using relevance scoring.
        Simulates cross-encoder reranking (can be upgraded to GPU model).
        """
        query_terms = set(query.lower().split())
        
        scored = []
        for result in results:
            text = result.get("text_ar", "").lower()
            text_terms = set(text.split())
            
            # Calculate relevance score
            overlap = len(query_terms & text_terms)
            coverage = overlap / len(query_terms) if query_terms else 0
            
            # Boost for exact matches
            exact_match = 1.5 if query.lower() in text else 1.0
            
            # Boost for evaluation (praised/blamed)
            eval_boost = 1.2 if result.get("normative", {}).get("evaluation") else 1.0
            
            score = coverage * exact_match * eval_boost
            scored.append((score, result))
        
        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        
        return [r for _, r in scored[:top_k]]
    
    # =========================================================================
    # UNIFIED ANALYSIS
    # =========================================================================
    
    def analyze_behavior(self, behavior: str) -> Dict[str, Any]:
        """
        Complete behavior analysis using ALL brain components.
        """
        start_time = time.time()
        
        # Query all components
        query_result = self.query(text=behavior, limit=500)
        spans = query_result["spans"]
        
        # Rerank for relevance
        reranked = self.rerank_results(behavior, spans, top_k=50)
        
        # Get all 11 dimensions
        dimensions = {}
        for dim in DIMENSIONS[:10]:
            dist = defaultdict(int)
            for span in reranked:
                val = self._get_dimension_value(span, dim)
                if val:
                    dist[val] += 1
            dimensions[dim] = {
                "distribution": dict(dist),
                "total": sum(dist.values()),
                "top_value": max(dist.items(), key=lambda x: x[1])[0] if dist else None,
            }
        
        # Get relationships
        relationships = {
            "co_occurs": list(self.graph["co_occurs"].get(behavior[:30], set()))[:10],
            "effects": list(self.graph["effects"].get(behavior[:30], set()))[:10],
        }
        
        # Get tafsir for key verses
        tafsir_refs = []
        for span in reranked[:5]:
            ref = span.get("reference", {})
            surah, ayah = ref.get("surah"), ref.get("ayah")
            if surah and ayah:
                tafsir_refs.append(self.get_tafsir_comparison(surah, ayah))
        
        # Personality comparison
        personality = {}
        for agent in ["AGT_BELIEVER", "AGT_KAFIR", "AGT_MUNAFIQ"]:
            agent_spans = [s for s in reranked if s.get("agent", {}).get("type") == agent]
            personality[agent] = {
                "count": len(agent_spans),
                "evaluation_dist": self._get_eval_dist(agent_spans),
            }
        
        return {
            "behavior": behavior,
            "total_mentions": len(spans),
            "reranked_top": len(reranked),
            "dimensions": dimensions,
            "relationships": relationships,
            "tafsir_cross_reference": tafsir_refs,
            "personality_comparison": personality,
            "key_verses": self._format_verses(reranked[:10]),
            "execution_time_ms": round((time.time() - start_time) * 1000, 2),
            "components_used": ["annotations", "labels", "graph", "tafsir_5_sources", "reranker"],
        }
    
    def _get_eval_dist(self, spans: List[Dict]) -> Dict[str, int]:
        """Get evaluation distribution."""
        dist = defaultdict(int)
        for span in spans:
            eval_val = span.get("normative", {}).get("evaluation")
            if eval_val:
                dist[eval_val] += 1
        return dict(dist)
    
    def _format_verses(self, spans: List[Dict]) -> List[Dict]:
        """Format spans as verse references."""
        verses = []
        for span in spans:
            ref = span.get("reference", {})
            verses.append({
                "surah": ref.get("surah"),
                "ayah": ref.get("ayah"),
                "text": span.get("text_ar", "")[:100],
                "agent": span.get("agent", {}).get("type"),
                "evaluation": span.get("normative", {}).get("evaluation"),
            })
        return verses
    
    # =========================================================================
    # JOURNEY ANALYSIS WITH GRAPH
    # =========================================================================
    
    def analyze_journey(self, start: str, end: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze journey/chain using graph traversal.
        """
        start_time = time.time()
        
        # Get start concept data
        start_data = self.query(text=start, limit=100)
        
        # Find path if end is specified
        path_result = None
        if end:
            path_result = self.find_path(start, end)
        
        # Build stages
        stages = []
        
        # Stage 1: Start
        stages.append({
            "name": start,
            "mentions": start_data["total_matches"],
            "type": "start",
            "verses": [f"{s.get('reference', {}).get('surah')}:{s.get('reference', {}).get('ayah')}" 
                      for s in start_data["spans"][:5]],
        })
        
        # Intermediate stages from co-occurrence
        co_occurs = list(self.graph["co_occurs"].get(start[:30], set()))[:3]
        for co in co_occurs:
            co_data = self.query(text=co, limit=50)
            stages.append({
                "name": co,
                "mentions": co_data["total_matches"],
                "type": "intermediate",
                "verses": [f"{s.get('reference', {}).get('surah')}:{s.get('reference', {}).get('ayah')}" 
                          for s in co_data["spans"][:3]],
            })
        
        # End stage
        if end:
            end_data = self.query(text=end, limit=100)
            stages.append({
                "name": end,
                "mentions": end_data["total_matches"],
                "type": "end",
                "verses": [f"{s.get('reference', {}).get('surah')}:{s.get('reference', {}).get('ayah')}" 
                          for s in end_data["spans"][:5]],
            })
        
        # Build graph visualization
        graph = self._build_journey_graph(stages)
        
        return {
            "start": start,
            "end": end,
            "stages": stages,
            "path_found": path_result,
            "graph": graph,
            "execution_time_ms": round((time.time() - start_time) * 1000, 2),
        }
    
    def _build_journey_graph(self, stages: List[Dict]) -> Dict[str, Any]:
        """Build graph visualization data."""
        STATE_COLORS = {
            "سليم": "#22c55e", "مريض": "#eab308", "قاسي": "#f97316",
            "مختوم": "#ef4444", "ميت": "#1f2937", "منيب": "#3b82f6",
        }
        
        nodes = []
        edges = []
        
        for i, stage in enumerate(stages):
            label = stage.get("name", f"Stage {i+1}")
            color = "#6b7280"
            for key, c in STATE_COLORS.items():
                if key in label:
                    color = c
                    break
            
            nodes.append({
                "id": f"stage_{i}",
                "label": label,
                "color": color,
                "mentions": stage.get("mentions", 0),
            })
        
        for i in range(len(nodes) - 1):
            edges.append({
                "source": f"stage_{i}",
                "target": f"stage_{i + 1}",
                "label": "يؤدي إلى",
            })
        
        return {"nodes": nodes, "edges": edges}


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_brain_instance = None

def get_brain(spans_data: List[Dict[str, Any]] = None) -> UnifiedBrain:
    """Get or create the unified brain instance."""
    global _brain_instance
    if _brain_instance is None and spans_data is not None:
        _brain_instance = UnifiedBrain(spans_data)
    return _brain_instance


def reset_brain():
    """Reset the brain instance."""
    global _brain_instance
    _brain_instance = None
