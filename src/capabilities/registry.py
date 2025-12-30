"""
Capability Engine Registry

Maps capability IDs to engine implementations.
Provides factory functions for engine instantiation.
"""

from typing import Dict, List, Optional, Type
from .base import CapabilityEngine


# Capability to Section mapping (from 200-question benchmark)
SECTION_CAPABILITIES = {
    "A": ["GRAPH_CAUSAL", "MULTIHOP", "TAFSIR_MULTI_SOURCE", "PROVENANCE"],
    "B": ["GRAPH_CAUSAL", "MULTIHOP", "PROVENANCE"],
    "C": ["TAFSIR_MULTI_SOURCE", "CONSENSUS", "PROVENANCE"],
    "D": ["TAXONOMY", "PROVENANCE"],
    "E": ["TEMPORAL_SPATIAL", "AXES_11D", "PROVENANCE"],
    "F": ["HEART_STATE", "PROVENANCE"],
    "G": ["AGENT_MODEL", "PROVENANCE"],
    "H": ["CONSEQUENCE_MODEL", "GRAPH_CAUSAL", "PROVENANCE"],
    "I": ["EMBEDDINGS", "SEMANTIC_GRAPH_V2", "GRAPH_METRICS", "PROVENANCE"],
    "J": ["INTEGRATION_E2E", "MODEL_REGISTRY", "PROVENANCE"],
}

# All unique capabilities
ALL_CAPABILITIES = [
    "GRAPH_CAUSAL",
    "MULTIHOP", 
    "TAFSIR_MULTI_SOURCE",
    "PROVENANCE",
    "TAXONOMY",
    "TEMPORAL_SPATIAL",
    "AXES_11D",
    "HEART_STATE",
    "AGENT_MODEL",
    "CONSEQUENCE_MODEL",
    "EMBEDDINGS",
    "SEMANTIC_GRAPH_V2",
    "GRAPH_METRICS",
    "CONSENSUS",
    "INTEGRATION_E2E",
    "MODEL_REGISTRY",
]

# Engine registry - populated by engine modules
_ENGINE_REGISTRY: Dict[str, Type[CapabilityEngine]] = {}


def register_engine(capability_id: str):
    """Decorator to register an engine class."""
    def decorator(cls: Type[CapabilityEngine]):
        _ENGINE_REGISTRY[capability_id] = cls
        cls.capability_id = capability_id
        return cls
    return decorator


def get_engine(capability_id: str) -> Optional[CapabilityEngine]:
    """Get an instance of the engine for a capability."""
    if capability_id not in _ENGINE_REGISTRY:
        # Try to import the engine module
        _import_engines()
    
    engine_class = _ENGINE_REGISTRY.get(capability_id)
    if engine_class:
        return engine_class()
    return None


def list_engines() -> List[str]:
    """List all registered engine capability IDs."""
    _import_engines()
    return list(_ENGINE_REGISTRY.keys())


def get_engines_for_section(section: str) -> List[CapabilityEngine]:
    """Get all engines needed for a benchmark section."""
    capabilities = SECTION_CAPABILITIES.get(section.upper(), [])
    engines = []
    for cap in capabilities:
        engine = get_engine(cap)
        if engine:
            engines.append(engine)
    return engines


def _import_engines():
    """Import all engine modules to populate registry."""
    try:
        from . import engines
    except ImportError:
        pass


# Capability map for external use
CAPABILITY_MAP = {
    "GRAPH_CAUSAL": {
        "name": "Graph Causal Reasoning",
        "description": "Traverse causal chains in behavioral graph",
        "sections": ["A", "B", "H"],
    },
    "MULTIHOP": {
        "name": "Multi-hop Path Finding",
        "description": "Find paths with multiple hops between behaviors",
        "sections": ["A", "B"],
    },
    "TAFSIR_MULTI_SOURCE": {
        "name": "Multi-Source Tafsir",
        "description": "Aggregate evidence from multiple tafsir sources",
        "sections": ["A", "C"],
    },
    "PROVENANCE": {
        "name": "Evidence Provenance",
        "description": "Track and validate evidence sources",
        "sections": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
    },
    "TAXONOMY": {
        "name": "Behavior Taxonomy",
        "description": "Classify and categorize behaviors",
        "sections": ["D"],
    },
    "TEMPORAL_SPATIAL": {
        "name": "Temporal-Spatial Analysis",
        "description": "Analyze temporal and spatial dimensions",
        "sections": ["E"],
    },
    "AXES_11D": {
        "name": "11 Bouzidani Dimensions",
        "description": "Analyze behaviors across 11 dimensions",
        "sections": ["E"],
    },
    "HEART_STATE": {
        "name": "Heart State Modeling",
        "description": "Model heart states and their effects",
        "sections": ["F"],
    },
    "AGENT_MODEL": {
        "name": "Agent Behavior Modeling",
        "description": "Model agent behaviors and interactions",
        "sections": ["G"],
    },
    "CONSEQUENCE_MODEL": {
        "name": "Consequence Chain Modeling",
        "description": "Model consequences of behaviors",
        "sections": ["H"],
    },
    "EMBEDDINGS": {
        "name": "Semantic Embeddings",
        "description": "Use embeddings for semantic similarity",
        "sections": ["I"],
    },
    "SEMANTIC_GRAPH_V2": {
        "name": "Semantic Graph v2",
        "description": "Query semantic graph for relationships",
        "sections": ["I"],
    },
    "GRAPH_METRICS": {
        "name": "Graph Metrics",
        "description": "Compute graph metrics (centrality, etc.)",
        "sections": ["I"],
    },
    "CONSENSUS": {
        "name": "Tafsir Consensus",
        "description": "Calculate consensus across tafsir sources",
        "sections": ["C"],
    },
    "INTEGRATION_E2E": {
        "name": "End-to-End Integration",
        "description": "Full pipeline integration",
        "sections": ["J"],
    },
    "MODEL_REGISTRY": {
        "name": "Model Registry",
        "description": "Track model versions and provenance",
        "sections": ["J"],
    },
}
