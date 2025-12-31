"""
QBM Function-Calling Tools for Azure OpenAI

Phase 6: Tool definitions and implementations for function-calling.
These tools wrap the LightweightProofBackend to provide structured data retrieval.

Tools:
1. resolve_entity - Resolve Arabic/English term to canonical entity ID
2. get_behavior_dossier - Get complete behavior profile with verses and tafsir
3. get_causal_paths - Find causal chains between behaviors
4. get_tafsir_comparison - Get multi-source tafsir for verses
5. get_graph_metrics - Get graph statistics and centrality
6. get_verse_evidence - Get verse text and evidence for a behavior
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the proof backend
try:
    from src.ml.proof_only_backend import LightweightProofBackend
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False
    logger.warning("LightweightProofBackend not available")


# =============================================================================
# TOOL DEFINITIONS (OpenAI Function Calling Format)
# =============================================================================

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "resolve_entity",
            "description": "Resolve an Arabic or English term to its canonical QBM entity ID. Use this to find the correct behavior ID before querying.",
            "parameters": {
                "type": "object",
                "properties": {
                    "term": {
                        "type": "string",
                        "description": "The Arabic or English term to resolve (e.g., 'الصبر', 'patience', 'صدق')"
                    },
                    "entity_type": {
                        "type": "string",
                        "enum": ["BEHAVIOR", "AGENT", "ORGAN", "HEART_STATE", "CONSEQUENCE"],
                        "description": "Type of entity to search for (default: BEHAVIOR)"
                    }
                },
                "required": ["term"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_behavior_dossier",
            "description": "Get the complete dossier for a behavior including verses, tafsir quotes, and relationships. Returns structured evidence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "behavior_id": {
                        "type": "string",
                        "description": "Canonical behavior ID (e.g., 'BEH_EMO_PATIENCE', 'BEH_SPI_FAITH')"
                    },
                    "include_tafsir": {
                        "type": "boolean",
                        "description": "Whether to include tafsir quotes (default: true)"
                    },
                    "max_verses": {
                        "type": "integer",
                        "description": "Maximum number of verses to return (default: 10)"
                    }
                },
                "required": ["behavior_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_causal_paths",
            "description": "Find causal paths between two behaviors in the semantic graph. Returns paths with edge types and evidence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_behavior": {
                        "type": "string",
                        "description": "Source behavior ID (e.g., 'BEH_COG_HEEDLESSNESS')"
                    },
                    "to_behavior": {
                        "type": "string",
                        "description": "Target behavior ID (e.g., 'BEH_SPI_DISBELIEF')"
                    },
                    "max_hops": {
                        "type": "integer",
                        "description": "Maximum path length (default: 5)"
                    }
                },
                "required": ["from_behavior", "to_behavior"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_tafsir_comparison",
            "description": "Get tafsir quotes from multiple sources for specific verses. Useful for scholarly comparison.",
            "parameters": {
                "type": "object",
                "properties": {
                    "verse_keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of verse keys (e.g., ['2:45', '2:153', '3:200'])"
                    },
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tafsir sources to include (default: all 7 sources)"
                    },
                    "max_per_source": {
                        "type": "integer",
                        "description": "Maximum chunks per source (default: 3)"
                    }
                },
                "required": ["verse_keys"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_graph_metrics",
            "description": "Get graph statistics including node counts, edge counts, and centrality metrics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "behavior_id": {
                        "type": "string",
                        "description": "Optional: Get metrics for a specific behavior"
                    },
                    "include_neighbors": {
                        "type": "boolean",
                        "description": "Include neighbor behaviors (default: false)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_verse_evidence",
            "description": "Get verse text and evidence linking a behavior to specific verses.",
            "parameters": {
                "type": "object",
                "properties": {
                    "behavior_id": {
                        "type": "string",
                        "description": "Behavior ID to get evidence for"
                    },
                    "verse_keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional: Specific verse keys to retrieve"
                    },
                    "max_verses": {
                        "type": "integer",
                        "description": "Maximum verses to return (default: 20)"
                    }
                },
                "required": ["behavior_id"]
            }
        }
    }
]


# =============================================================================
# TOOL IMPLEMENTATIONS
# =============================================================================

class QBMTools:
    """
    Implementation of QBM function-calling tools.
    
    Wraps LightweightProofBackend to provide structured data retrieval
    for Azure OpenAI function calling.
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize tools with data directory."""
        if BACKEND_AVAILABLE:
            self.backend = LightweightProofBackend(data_dir=data_dir)
        else:
            self.backend = None
            logger.warning("Backend not available - tools will return errors")
        
        self._canonical_entities = None
        self._behavior_term_map = None
    
    def _load_canonical_entities(self) -> Dict[str, Any]:
        """Load canonical entities for term resolution."""
        if self._canonical_entities is not None:
            return self._canonical_entities
        
        vocab_path = Path("vocab/canonical_entities.json")
        if vocab_path.exists():
            with open(vocab_path, "r", encoding="utf-8") as f:
                self._canonical_entities = json.load(f)
            
            # Build term map
            self._behavior_term_map = {}
            for beh in self._canonical_entities.get("behaviors", []):
                beh_id = beh.get("id")
                if beh.get("ar"):
                    self._behavior_term_map[beh["ar"]] = beh_id
                    # Also map without ال prefix
                    if beh["ar"].startswith("ال"):
                        self._behavior_term_map[beh["ar"][2:]] = beh_id
                if beh.get("en"):
                    self._behavior_term_map[beh["en"].lower()] = beh_id
                for syn in beh.get("synonyms", []):
                    if syn:
                        self._behavior_term_map[syn] = beh_id
        else:
            self._canonical_entities = {"behaviors": [], "agents": [], "organs": []}
            self._behavior_term_map = {}
        
        return self._canonical_entities
    
    def resolve_entity(self, term: str, entity_type: str = "BEHAVIOR") -> Dict[str, Any]:
        """
        Resolve a term to its canonical entity ID.
        
        Args:
            term: Arabic or English term to resolve
            entity_type: Type of entity (BEHAVIOR, AGENT, etc.)
            
        Returns:
            Dict with entity_id, term_ar, term_en, and match_type
        """
        self._load_canonical_entities()
        
        # Direct lookup
        if term in self._behavior_term_map:
            entity_id = self._behavior_term_map[term]
            entity = next(
                (b for b in self._canonical_entities.get("behaviors", []) 
                 if b.get("id") == entity_id), 
                None
            )
            if entity:
                return {
                    "success": True,
                    "entity_id": entity_id,
                    "term_ar": entity.get("ar"),
                    "term_en": entity.get("en"),
                    "category": entity.get("category"),
                    "match_type": "exact"
                }
        
        # Case-insensitive English lookup
        term_lower = term.lower()
        if term_lower in self._behavior_term_map:
            entity_id = self._behavior_term_map[term_lower]
            entity = next(
                (b for b in self._canonical_entities.get("behaviors", []) 
                 if b.get("id") == entity_id), 
                None
            )
            if entity:
                return {
                    "success": True,
                    "entity_id": entity_id,
                    "term_ar": entity.get("ar"),
                    "term_en": entity.get("en"),
                    "category": entity.get("category"),
                    "match_type": "case_insensitive"
                }
        
        # Partial match
        for beh in self._canonical_entities.get("behaviors", []):
            if term_lower in beh.get("en", "").lower():
                return {
                    "success": True,
                    "entity_id": beh.get("id"),
                    "term_ar": beh.get("ar"),
                    "term_en": beh.get("en"),
                    "category": beh.get("category"),
                    "match_type": "partial"
                }
        
        return {
            "success": False,
            "error": f"Could not resolve term: {term}",
            "suggestions": self._get_suggestions(term)
        }
    
    def _get_suggestions(self, term: str, max_suggestions: int = 5) -> List[str]:
        """Get suggestions for unresolved terms."""
        suggestions = []
        term_lower = term.lower()
        
        for beh in self._canonical_entities.get("behaviors", []):
            en = beh.get("en", "").lower()
            ar = beh.get("ar", "")
            if term_lower[:3] in en or (len(term) > 2 and term[:2] in ar):
                suggestions.append(f"{beh.get('id')}: {ar} ({beh.get('en')})")
                if len(suggestions) >= max_suggestions:
                    break
        
        return suggestions
    
    def get_behavior_dossier(
        self, 
        behavior_id: str, 
        include_tafsir: bool = True,
        max_verses: int = 10
    ) -> Dict[str, Any]:
        """
        Get complete dossier for a behavior.
        
        Args:
            behavior_id: Canonical behavior ID
            include_tafsir: Whether to include tafsir quotes
            max_verses: Maximum verses to return
            
        Returns:
            Dict with behavior info, verses, tafsir, and relationships
        """
        if not self.backend:
            return {"success": False, "error": "Backend not available"}
        
        # Load concept index
        concept_index = self.backend._load_concept_index()
        behavior_data = concept_index.get(behavior_id, {})
        
        if not behavior_data:
            return {
                "success": False,
                "error": f"Behavior not found: {behavior_id}"
            }
        
        # Get verses
        verses = behavior_data.get("verses", [])[:max_verses]
        verse_keys = [v.get("verse_key") for v in verses if v.get("verse_key")]
        
        result = {
            "success": True,
            "behavior_id": behavior_id,
            "term_ar": behavior_data.get("term"),
            "term_en": behavior_data.get("term_en"),
            "entity_type": behavior_data.get("entity_type"),
            "verse_count": len(behavior_data.get("verses", [])),
            "verses": verses,
            "provenance": {
                "source": "concept_index_v3.jsonl",
                "verse_keys": verse_keys
            }
        }
        
        # Add tafsir if requested
        if include_tafsir and verse_keys:
            tafsir = self.backend._get_tafsir_for_verses(verse_keys[:5], max_per_source=2)
            result["tafsir"] = tafsir
            result["tafsir_sources"] = [src for src, chunks in tafsir.items() if chunks]
        
        return result
    
    def get_causal_paths(
        self, 
        from_behavior: str, 
        to_behavior: str, 
        max_hops: int = 5
    ) -> Dict[str, Any]:
        """
        Find causal paths between two behaviors.
        
        Args:
            from_behavior: Source behavior ID
            to_behavior: Target behavior ID
            max_hops: Maximum path length
            
        Returns:
            Dict with paths, each containing nodes, edges, and evidence
        """
        if not self.backend:
            return {"success": False, "error": "Backend not available"}
        
        paths = self.backend._find_causal_paths(
            source_id=from_behavior,
            target_id=to_behavior,
            min_hops=1,
            max_hops=max_hops
        )
        
        if not paths:
            return {
                "success": True,
                "from_behavior": from_behavior,
                "to_behavior": to_behavior,
                "paths_found": 0,
                "paths": [],
                "message": "No causal paths found between these behaviors"
            }
        
        # Format paths with evidence
        formatted_paths = []
        for path in paths[:10]:  # Limit to 10 paths
            nodes = path.get("nodes", [])
            edges = path.get("edges", [])
            
            formatted_path = {
                "nodes": nodes,
                "hops": path.get("hops", len(nodes) - 1),
                "edge_types": [e.get("edge_type") for e in edges],
                "evidence": []
            }
            
            for edge in edges:
                ev = edge.get("evidence", {})
                if isinstance(ev, dict) and ev.get("description"):
                    formatted_path["evidence"].append({
                        "edge": f"{edge.get('source')} -> {edge.get('target')}",
                        "type": edge.get("edge_type"),
                        "description": ev.get("description")
                    })
            
            formatted_paths.append(formatted_path)
        
        return {
            "success": True,
            "from_behavior": from_behavior,
            "to_behavior": to_behavior,
            "paths_found": len(paths),
            "paths": formatted_paths,
            "provenance": {
                "source": "semantic_graph_v3.json",
                "query": f"{from_behavior} -> {to_behavior}"
            }
        }
    
    def get_tafsir_comparison(
        self, 
        verse_keys: List[str],
        sources: Optional[List[str]] = None,
        max_per_source: int = 3
    ) -> Dict[str, Any]:
        """
        Get tafsir quotes from multiple sources for comparison.
        
        Args:
            verse_keys: List of verse keys
            sources: Optional list of sources to include
            max_per_source: Maximum chunks per source
            
        Returns:
            Dict with tafsir quotes organized by source
        """
        if not self.backend:
            return {"success": False, "error": "Backend not available"}
        
        tafsir = self.backend._get_tafsir_for_verses(verse_keys, max_per_source=max_per_source)
        
        # Filter sources if specified
        if sources:
            tafsir = {k: v for k, v in tafsir.items() if k in sources}
        
        # Count chunks per source
        source_counts = {src: len(chunks) for src, chunks in tafsir.items()}
        
        return {
            "success": True,
            "verse_keys": verse_keys,
            "sources": list(tafsir.keys()),
            "source_counts": source_counts,
            "total_chunks": sum(source_counts.values()),
            "tafsir": tafsir,
            "provenance": {
                "source": "evidence_index_v2_chunked.jsonl",
                "verse_keys": verse_keys
            }
        }
    
    def get_graph_metrics(
        self, 
        behavior_id: Optional[str] = None,
        include_neighbors: bool = False
    ) -> Dict[str, Any]:
        """
        Get graph statistics and metrics.
        
        Args:
            behavior_id: Optional behavior to get specific metrics for
            include_neighbors: Whether to include neighbor behaviors
            
        Returns:
            Dict with graph statistics
        """
        if not self.backend:
            return {"success": False, "error": "Backend not available"}
        
        graph = self.backend._load_semantic_graph()
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        
        # Basic metrics
        beh_nodes = [n for n in nodes if n.get("type") == "BEHAVIOR"]
        causal_types = {"CAUSES", "LEADS_TO", "PREVENTS", "STRENGTHENS"}
        causal_edges = [e for e in edges if e.get("edge_type") in causal_types]
        
        result = {
            "success": True,
            "graph_version": graph.get("version"),
            "total_nodes": len(nodes),
            "behavior_nodes": len(beh_nodes),
            "total_edges": len(edges),
            "causal_edges": len(causal_edges),
            "edge_types": {}
        }
        
        # Count edge types
        from collections import Counter
        edge_type_counts = Counter(e.get("edge_type") for e in edges)
        result["edge_types"] = dict(edge_type_counts)
        
        # Behavior-specific metrics
        if behavior_id:
            out_edges = [e for e in edges if e.get("source") == behavior_id]
            in_edges = [e for e in edges if e.get("target") == behavior_id]
            
            result["behavior_metrics"] = {
                "behavior_id": behavior_id,
                "out_degree": len(out_edges),
                "in_degree": len(in_edges),
                "total_degree": len(out_edges) + len(in_edges)
            }
            
            if include_neighbors:
                neighbors = set()
                for e in out_edges:
                    neighbors.add(e.get("target"))
                for e in in_edges:
                    neighbors.add(e.get("source"))
                result["behavior_metrics"]["neighbors"] = list(neighbors)
        
        result["provenance"] = {
            "source": "semantic_graph_v3.json"
        }
        
        return result
    
    def get_verse_evidence(
        self, 
        behavior_id: str,
        verse_keys: Optional[List[str]] = None,
        max_verses: int = 20
    ) -> Dict[str, Any]:
        """
        Get verse text and evidence for a behavior.
        
        Args:
            behavior_id: Behavior ID
            verse_keys: Optional specific verses to retrieve
            max_verses: Maximum verses to return
            
        Returns:
            Dict with verse texts and evidence
        """
        if not self.backend:
            return {"success": False, "error": "Backend not available"}
        
        # Load concept index
        concept_index = self.backend._load_concept_index()
        behavior_data = concept_index.get(behavior_id, {})
        
        if not behavior_data:
            return {
                "success": False,
                "error": f"Behavior not found: {behavior_id}"
            }
        
        # Get verses
        all_verses = behavior_data.get("verses", [])
        
        if verse_keys:
            # Filter to specific verses
            verses = [v for v in all_verses if v.get("verse_key") in verse_keys]
        else:
            verses = all_verses[:max_verses]
        
        # Load verse texts
        quran_verses = self.backend._load_quran_verses()
        
        # Enrich with verse text
        enriched_verses = []
        for v in verses:
            vk = v.get("verse_key")
            enriched = {
                "verse_key": vk,
                "surah": v.get("surah"),
                "ayah": v.get("ayah"),
                "text_uthmani": quran_verses.get(vk, v.get("text_uthmani", "")),
                "directness": v.get("directness"),
                "evidence_type": v.get("evidence", [{}])[0].get("type") if v.get("evidence") else None
            }
            enriched_verses.append(enriched)
        
        return {
            "success": True,
            "behavior_id": behavior_id,
            "term_ar": behavior_data.get("term"),
            "total_verses": len(all_verses),
            "returned_verses": len(enriched_verses),
            "verses": enriched_verses,
            "provenance": {
                "source": "concept_index_v3.jsonl + quran_index",
                "behavior_id": behavior_id
            }
        }
    
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool by name with given arguments.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            
        Returns:
            Tool result
        """
        tool_map = {
            "resolve_entity": self.resolve_entity,
            "get_behavior_dossier": self.get_behavior_dossier,
            "get_causal_paths": self.get_causal_paths,
            "get_tafsir_comparison": self.get_tafsir_comparison,
            "get_graph_metrics": self.get_graph_metrics,
            "get_verse_evidence": self.get_verse_evidence,
        }
        
        if tool_name not in tool_map:
            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}",
                "available_tools": list(tool_map.keys())
            }
        
        try:
            return tool_map[tool_name](**arguments)
        except Exception as e:
            logger.error(f"Tool execution error: {tool_name} - {e}")
            return {
                "success": False,
                "error": str(e),
                "tool": tool_name,
                "arguments": arguments
            }
