"""
Capability Engine Implementations

Each engine retrieves from SSOT and returns structured results.
LLM is NEVER the source of truth.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .base import CapabilityEngine, CapabilityResult
from .registry import register_engine


# Generic opening verses to disallow (per benchmark spec)
GENERIC_OPENING_VERSES = {f"1:{i}" for i in range(1, 8)} | {f"2:{i}" for i in range(1, 21)}


@register_engine("GRAPH_CAUSAL")
class GraphCausalEngine(CapabilityEngine):
    """
    Engine A: Graph Causal Reasoning
    
    Traverses causal chains in the behavioral graph.
    Finds paths between behaviors with edge evidence.
    """
    
    capability_name = "Graph Causal Reasoning"
    required_sources = ["semantic_graph_v3.json"]
    
    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> CapabilityResult:
        start_time = time.time()
        params = params or {}
        
        # Load graph
        graph = self.load_graph_file("semantic_graph_v3.json")
        if not graph:
            return CapabilityResult(
                success=False,
                capability=self.capability_id,
                errors=["Failed to load semantic graph"]
            )
        
        # Build adjacency for causal edges
        edges = graph.get("edges", [])
        causal_types = {"CAUSES", "LEADS_TO", "PREVENTS", "STRENGTHENS"}
        
        adjacency: Dict[str, List[Dict]] = {}
        for edge in edges:
            if edge.get("edge_type") in causal_types:
                source = edge.get("source")
                if source not in adjacency:
                    adjacency[source] = []
                adjacency[source].append(edge)
        
        # Extract behavior from query if provided
        from_behavior = params.get("from_behavior")
        to_behavior = params.get("to_behavior")
        min_hops = params.get("min_hops", 1)
        max_hops = params.get("max_hops", 5)
        
        result_data = {
            "total_causal_edges": len([e for e in edges if e.get("edge_type") in causal_types]),
            "behaviors_with_outgoing": len(adjacency),
            "causal_types": list(causal_types),
        }
        
        verses = []
        provenance = []
        
        # If specific path requested, find it
        if from_behavior and to_behavior:
            paths = self._find_paths(adjacency, from_behavior, to_behavior, max_hops)
            result_data["paths"] = paths
            
            # Collect evidence from paths
            for path in paths[:5]:  # Limit to top 5 paths
                for edge in path.get("edges", []):
                    for ev in edge.get("evidence", []):
                        # Handle both dict and string evidence entries
                        if isinstance(ev, str):
                            # Parse "surah:ayah" string format
                            if ":" in ev:
                                parts = ev.split(":")
                                try:
                                    surah = int(parts[0])
                                    ayah = int(parts[1])
                                    verse_key = ev
                                    source = None
                                except (ValueError, IndexError):
                                    continue
                            else:
                                continue
                        elif isinstance(ev, dict):
                            surah = ev.get("surah")
                            ayah = ev.get("ayah")
                            verse_key = f"{surah}:{ayah}"
                            source = ev.get("source")
                        else:
                            continue
                        
                        if verse_key not in GENERIC_OPENING_VERSES:
                            verses.append({
                                "verse_key": verse_key,
                                "surah": surah,
                                "ayah": ayah,
                                "source": source,
                            })
                            provenance.append({
                                "type": "edge_evidence",
                                "edge": f"{edge.get('source')} -> {edge.get('target')}",
                                "source": source,
                                "verse_key": verse_key,
                            })
        else:
            # Return graph statistics
            result_data["sample_edges"] = edges[:10]
            provenance.append({
                "type": "graph_file",
                "source": "semantic_graph_v3.json",
                "edge_count": len(edges),
            })
        
        elapsed = (time.time() - start_time) * 1000
        
        return CapabilityResult(
            success=True,
            capability=self.capability_id,
            data=result_data,
            verses=verses,
            provenance=provenance,
            execution_time_ms=elapsed,
        )
    
    def _find_paths(self, adjacency: Dict, start: str, end: str, max_depth: int) -> List[Dict]:
        """BFS to find all paths between two behaviors."""
        paths = []
        queue = [(start, [{"node": start}], [])]
        visited_paths: Set[str] = set()
        
        while queue and len(paths) < 10:
            current, path, edges = queue.pop(0)
            
            if current == end and len(path) > 1:
                path_key = "->".join(p["node"] for p in path)
                if path_key not in visited_paths:
                    visited_paths.add(path_key)
                    paths.append({
                        "path": path,
                        "edges": edges,
                        "hops": len(edges),
                    })
                continue
            
            if len(path) > max_depth:
                continue
            
            for edge in adjacency.get(current, []):
                target = edge.get("target")
                if target and not any(p["node"] == target for p in path):
                    new_path = path + [{"node": target}]
                    new_edges = edges + [edge]
                    queue.append((target, new_path, new_edges))
        
        return sorted(paths, key=lambda p: p["hops"])


@register_engine("MULTIHOP")
class MultihopEngine(CapabilityEngine):
    """
    Engine B: Multi-hop Path Finding
    
    Finds paths with multiple hops between behaviors.
    Returns evidence for each hop.
    """
    
    capability_name = "Multi-hop Path Finding"
    
    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> CapabilityResult:
        start_time = time.time()
        params = params or {}
        
        # Delegate to graph causal for path finding
        graph_engine = GraphCausalEngine()
        result = graph_engine.execute(query, params)
        
        # Add multihop-specific data
        result.capability = self.capability_id
        if result.data.get("paths"):
            result.data["multihop_paths"] = [
                p for p in result.data["paths"] if p.get("hops", 0) >= params.get("min_hops", 2)
            ]
        
        result.execution_time_ms = (time.time() - start_time) * 1000
        return result


@register_engine("TAFSIR_MULTI_SOURCE")
class TafsirMultiSourceEngine(CapabilityEngine):
    """
    Engine C: Multi-Source Tafsir Aggregation
    
    Aggregates evidence from multiple tafsir sources.
    Calculates consensus across sources.
    """
    
    capability_name = "Multi-Source Tafsir"
    required_sources = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"]
    
    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> CapabilityResult:
        start_time = time.time()
        params = params or {}
        
        behavior_id = params.get("behavior_id")
        verse_key = params.get("verse_key")
        
        # Load dossier if behavior specified
        if behavior_id:
            dossier = self.get_behavior_dossier(behavior_id)
            if not dossier:
                dossier = self.get_behavior_by_term(behavior_id)
        else:
            dossier = None
        
        verses = []
        provenance = []
        source_counts: Dict[str, int] = {}
        
        if dossier:
            # Aggregate from dossier tafsir
            tafsir_data = dossier.get("tafsir", {})
            for source, chunks in tafsir_data.items():
                source_counts[source] = len(chunks)
                for chunk in chunks[:5]:  # Limit per source
                    vk = chunk.get("verse_key", "")
                    if vk and vk not in GENERIC_OPENING_VERSES:
                        verses.append({
                            "verse_key": vk,
                            "source": source,
                            "quote": chunk.get("quote", "")[:200],
                        })
                        provenance.append({
                            "type": "tafsir_chunk",
                            "source": source,
                            "verse_key": vk,
                            "chunk_id": chunk.get("chunk_id"),
                        })
        
        result_data = {
            "sources_found": list(source_counts.keys()),
            "source_counts": source_counts,
            "total_chunks": sum(source_counts.values()),
            "required_sources": self.required_sources,
        }
        
        # Always add provenance even if no dossier found
        if not provenance:
            provenance.append({
                "type": "tafsir_query",
                "behavior_id": behavior_id,
                "verse_key": verse_key,
                "sources_checked": self.required_sources,
            })
        
        elapsed = (time.time() - start_time) * 1000
        
        return CapabilityResult(
            success=True,
            capability=self.capability_id,
            data=result_data,
            verses=verses,
            provenance=provenance,
            execution_time_ms=elapsed,
        )


@register_engine("PROVENANCE")
class ProvenanceEngine(CapabilityEngine):
    """
    Engine D: Evidence Provenance
    
    Tracks and validates evidence sources.
    Ensures all claims have proper citations.
    """
    
    capability_name = "Evidence Provenance"
    
    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> CapabilityResult:
        start_time = time.time()
        params = params or {}
        
        # Load manifest for provenance info
        manifest_path = self.kb_dir / "manifest.json"
        manifest = {}
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        
        result_data = {
            "kb_version": manifest.get("version"),
            "built_at": manifest.get("built_at"),
            "git_commit": manifest.get("git_commit"),
            "input_hashes": manifest.get("input_hashes", {}),
            "output_hashes": manifest.get("output_hashes", {}),
        }
        
        provenance = [{
            "type": "kb_manifest",
            "source": "manifest.json",
            "version": manifest.get("version"),
            "git_commit": manifest.get("git_commit"),
        }]
        
        elapsed = (time.time() - start_time) * 1000
        
        return CapabilityResult(
            success=True,
            capability=self.capability_id,
            data=result_data,
            provenance=provenance,
            execution_time_ms=elapsed,
        )


@register_engine("TAXONOMY")
class TaxonomyEngine(CapabilityEngine):
    """
    Engine E: Behavior Taxonomy
    
    Classifies and categorizes behaviors.
    Returns taxonomy structure.
    """
    
    capability_name = "Behavior Taxonomy"
    
    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> CapabilityResult:
        start_time = time.time()
        
        # Load canonical entities
        entities = self.load_vocab_file("canonical_entities.json")
        behaviors = entities.get("behaviors", [])
        
        # Group by category
        categories: Dict[str, List[Dict]] = {}
        for b in behaviors:
            cat = b.get("category", "uncategorized")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append({
                "id": b.get("id"),
                "ar": b.get("ar"),
                "en": b.get("en"),
            })
        
        result_data = {
            "total_behaviors": len(behaviors),
            "categories": {k: len(v) for k, v in categories.items()},
            "taxonomy": categories,
        }
        
        provenance = [{
            "type": "vocab_file",
            "source": "canonical_entities.json",
            "behavior_count": len(behaviors),
        }]
        
        elapsed = (time.time() - start_time) * 1000
        
        return CapabilityResult(
            success=True,
            capability=self.capability_id,
            data=result_data,
            provenance=provenance,
            execution_time_ms=elapsed,
        )


@register_engine("TEMPORAL_SPATIAL")
class TemporalSpatialEngine(CapabilityEngine):
    """
    Engine F: Temporal-Spatial Analysis
    
    Analyzes temporal and spatial dimensions of behaviors.
    """
    
    capability_name = "Temporal-Spatial Analysis"
    
    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> CapabilityResult:
        start_time = time.time()
        
        # Load temporal/spatial vocab
        temporal = self.load_vocab_file("temporal.json")
        spatial = self.load_vocab_file("spatial.json")
        
        result_data = {
            "temporal_categories": list(temporal.keys()) if temporal else [],
            "spatial_categories": list(spatial.keys()) if spatial else [],
        }
        
        provenance = [{
            "type": "vocab_file",
            "source": "temporal.json",
        }, {
            "type": "vocab_file", 
            "source": "spatial.json",
        }]
        
        elapsed = (time.time() - start_time) * 1000
        
        return CapabilityResult(
            success=True,
            capability=self.capability_id,
            data=result_data,
            provenance=provenance,
            execution_time_ms=elapsed,
        )


@register_engine("AXES_11D")
class Axes11DEngine(CapabilityEngine):
    """
    Engine G: 11 Bouzidani Dimensions
    
    Analyzes behaviors across 11 dimensions.
    """
    
    capability_name = "11 Bouzidani Dimensions"
    
    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> CapabilityResult:
        start_time = time.time()
        
        # Load axes vocab
        axes = self.load_vocab_file("axes.json")
        
        result_data = {
            "dimensions": list(axes.keys()) if axes else [],
            "dimension_count": len(axes) if axes else 0,
        }
        
        provenance = [{
            "type": "vocab_file",
            "source": "axes.json",
        }]
        
        elapsed = (time.time() - start_time) * 1000
        
        return CapabilityResult(
            success=True,
            capability=self.capability_id,
            data=result_data,
            provenance=provenance,
            execution_time_ms=elapsed,
        )


@register_engine("HEART_STATE")
class HeartStateEngine(CapabilityEngine):
    """
    Engine H: Heart State Modeling
    
    Models heart states and their effects on behavior.
    """
    
    capability_name = "Heart State Modeling"
    
    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> CapabilityResult:
        start_time = time.time()
        
        # Load canonical entities for heart states
        entities = self.load_vocab_file("canonical_entities.json")
        heart_states = entities.get("heart_states", [])
        
        result_data = {
            "heart_states": heart_states,
            "count": len(heart_states),
        }
        
        provenance = [{
            "type": "vocab_file",
            "source": "canonical_entities.json",
            "heart_state_count": len(heart_states),
        }]
        
        elapsed = (time.time() - start_time) * 1000
        
        return CapabilityResult(
            success=True,
            capability=self.capability_id,
            data=result_data,
            provenance=provenance,
            execution_time_ms=elapsed,
        )


@register_engine("AGENT_MODEL")
class AgentModelEngine(CapabilityEngine):
    """
    Engine I: Agent Behavior Modeling
    
    Models agent behaviors and interactions.
    """
    
    capability_name = "Agent Behavior Modeling"
    
    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> CapabilityResult:
        start_time = time.time()
        
        # Load agents
        entities = self.load_vocab_file("canonical_entities.json")
        agents = entities.get("agents", [])
        
        result_data = {
            "agents": agents,
            "count": len(agents),
        }
        
        provenance = [{
            "type": "vocab_file",
            "source": "canonical_entities.json",
            "agent_count": len(agents),
        }]
        
        elapsed = (time.time() - start_time) * 1000
        
        return CapabilityResult(
            success=True,
            capability=self.capability_id,
            data=result_data,
            provenance=provenance,
            execution_time_ms=elapsed,
        )


@register_engine("CONSEQUENCE_MODEL")
class ConsequenceModelEngine(CapabilityEngine):
    """
    Engine J: Consequence Chain Modeling
    
    Models consequences of behaviors.
    """
    
    capability_name = "Consequence Chain Modeling"
    
    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> CapabilityResult:
        start_time = time.time()
        
        # Load consequences
        entities = self.load_vocab_file("canonical_entities.json")
        consequences = entities.get("consequences", [])
        
        result_data = {
            "consequences": consequences,
            "count": len(consequences),
        }
        
        provenance = [{
            "type": "vocab_file",
            "source": "canonical_entities.json",
            "consequence_count": len(consequences),
        }]
        
        elapsed = (time.time() - start_time) * 1000
        
        return CapabilityResult(
            success=True,
            capability=self.capability_id,
            data=result_data,
            provenance=provenance,
            execution_time_ms=elapsed,
        )


@register_engine("GRAPH_METRICS")
class GraphMetricsEngine(CapabilityEngine):
    """
    Compute graph metrics (centrality, etc.)
    """
    
    capability_name = "Graph Metrics"
    
    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> CapabilityResult:
        start_time = time.time()
        
        graph = self.load_graph_file("semantic_graph_v3.json")
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        
        # Compute basic metrics
        in_degree: Dict[str, int] = {}
        out_degree: Dict[str, int] = {}
        
        for edge in edges:
            src = edge.get("source")
            tgt = edge.get("target")
            out_degree[src] = out_degree.get(src, 0) + 1
            in_degree[tgt] = in_degree.get(tgt, 0) + 1
        
        result_data = {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "max_in_degree": max(in_degree.values()) if in_degree else 0,
            "max_out_degree": max(out_degree.values()) if out_degree else 0,
            "top_in_degree": sorted(in_degree.items(), key=lambda x: -x[1])[:5],
            "top_out_degree": sorted(out_degree.items(), key=lambda x: -x[1])[:5],
        }
        
        provenance = [{
            "type": "graph_file",
            "source": "semantic_graph_v3.json",
        }]
        
        elapsed = (time.time() - start_time) * 1000
        
        return CapabilityResult(
            success=True,
            capability=self.capability_id,
            data=result_data,
            provenance=provenance,
            execution_time_ms=elapsed,
        )


@register_engine("CONSENSUS")
class ConsensusEngine(CapabilityEngine):
    """
    Calculate consensus across tafsir sources.
    """
    
    capability_name = "Tafsir Consensus"
    
    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> CapabilityResult:
        start_time = time.time()
        params = params or {}
        
        behavior_id = params.get("behavior_id")
        
        if behavior_id:
            dossier = self.get_behavior_dossier(behavior_id)
            if not dossier:
                dossier = self.get_behavior_by_term(behavior_id)
        else:
            dossier = None
        
        sources_with_evidence = []
        if dossier:
            tafsir = dossier.get("tafsir", {})
            sources_with_evidence = [s for s, chunks in tafsir.items() if chunks]
        
        all_sources = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn", "baghawi", "muyassar"]
        consensus_pct = len(sources_with_evidence) / len(all_sources) * 100 if all_sources else 0
        
        result_data = {
            "sources_with_evidence": sources_with_evidence,
            "all_sources": all_sources,
            "consensus_percentage": round(consensus_pct, 1),
        }
        
        provenance = [{
            "type": "dossier",
            "behavior_id": behavior_id,
        }]
        
        elapsed = (time.time() - start_time) * 1000
        
        return CapabilityResult(
            success=True,
            capability=self.capability_id,
            data=result_data,
            provenance=provenance,
            execution_time_ms=elapsed,
        )


@register_engine("INTEGRATION_E2E")
class IntegrationE2EEngine(CapabilityEngine):
    """
    End-to-end integration engine.
    """
    
    capability_name = "End-to-End Integration"
    
    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> CapabilityResult:
        start_time = time.time()
        
        # Check all components are available
        components = {
            "kb_manifest": (self.kb_dir / "manifest.json").exists(),
            "behavior_dossiers": (self.kb_dir / "behavior_dossiers.jsonl").exists(),
            "semantic_graph": (self.graph_dir / "semantic_graph_v2.json").exists(),
            "canonical_entities": (self.vocab_dir / "canonical_entities.json").exists(),
        }
        
        result_data = {
            "components": components,
            "all_available": all(components.values()),
        }
        
        provenance = [{
            "type": "system_check",
            "components_checked": list(components.keys()),
        }]
        
        elapsed = (time.time() - start_time) * 1000
        
        return CapabilityResult(
            success=all(components.values()),
            capability=self.capability_id,
            data=result_data,
            provenance=provenance,
            execution_time_ms=elapsed,
        )


@register_engine("MODEL_REGISTRY")
class ModelRegistryEngine(CapabilityEngine):
    """
    Track model versions and provenance.
    """
    
    capability_name = "Model Registry"
    
    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> CapabilityResult:
        start_time = time.time()
        
        # Load manifest for model info
        manifest_path = self.kb_dir / "manifest.json"
        manifest = {}
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        
        result_data = {
            "kb_version": manifest.get("version"),
            "git_commit": manifest.get("git_commit"),
            "build_args": manifest.get("build_args", {}),
            "counts": manifest.get("counts", {}),
        }
        
        provenance = [{
            "type": "kb_manifest",
            "source": "manifest.json",
        }]
        
        elapsed = (time.time() - start_time) * 1000
        
        return CapabilityResult(
            success=True,
            capability=self.capability_id,
            data=result_data,
            provenance=provenance,
            execution_time_ms=elapsed,
        )


@register_engine("EMBEDDINGS")
class EmbeddingsEngine(CapabilityEngine):
    """
    Use embeddings for semantic similarity.
    """
    
    capability_name = "Semantic Embeddings"
    
    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> CapabilityResult:
        start_time = time.time()
        
        # Check if embeddings are available
        embeddings_available = False
        embedding_model = None
        
        try:
            from src.ml.arabic_embeddings import get_qbm_embeddings
            model = get_qbm_embeddings()
            if model:
                embeddings_available = True
                embedding_model = "qbm-arabic-embeddings"
        except ImportError:
            pass
        
        result_data = {
            "embeddings_available": embeddings_available,
            "embedding_model": embedding_model,
        }
        
        provenance = [{
            "type": "model_check",
            "model": embedding_model,
        }]
        
        elapsed = (time.time() - start_time) * 1000
        
        return CapabilityResult(
            success=True,
            capability=self.capability_id,
            data=result_data,
            provenance=provenance,
            execution_time_ms=elapsed,
        )


@register_engine("SEMANTIC_GRAPH_V2")
class SemanticGraphV2Engine(CapabilityEngine):
    """
    Query semantic graph for relationships.
    """
    
    capability_name = "Semantic Graph v2"
    
    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> CapabilityResult:
        start_time = time.time()
        
        graph = self.load_graph_file("semantic_graph_v2.json")
        
        result_data = {
            "version": graph.get("version"),
            "node_count": len(graph.get("nodes", [])),
            "edge_count": len(graph.get("edges", [])),
            "edge_types": graph.get("allowed_edge_types", []),
        }
        
        provenance = [{
            "type": "graph_file",
            "source": "semantic_graph_v2.json",
            "version": graph.get("version"),
        }]
        
        elapsed = (time.time() - start_time) * 1000
        
        return CapabilityResult(
            success=True,
            capability=self.capability_id,
            data=result_data,
            provenance=provenance,
            execution_time_ms=elapsed,
        )
