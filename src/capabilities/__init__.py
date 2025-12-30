"""
Capability Engines for QBM Brain

This module provides 10 capability engines (A-J) mapped to the 200-question benchmark.
Each engine is a deterministic, tool-first implementation that retrieves from SSOT.

Engines:
- A: GRAPH_CAUSAL - Causal chain traversal and multi-hop reasoning
- B: MULTIHOP - Multi-hop path finding with evidence
- C: TAFSIR_MULTI_SOURCE - Multi-source tafsir aggregation
- D: PROVENANCE - Evidence provenance and citation
- E: TAXONOMY - Behavior taxonomy and classification
- F: TEMPORAL_SPATIAL - Temporal and spatial analysis
- G: HEART_STATE - Heart state modeling
- H: AGENT_MODEL - Agent behavior modeling
- I: CONSEQUENCE_MODEL - Consequence chain modeling
- J: INTEGRATION_E2E - End-to-end integration

All engines follow the principle: LLM is NEVER the source of truth.
Data comes from Postgres SSOT, graph files, and precomputed KB.
"""

from .base import CapabilityEngine, CapabilityResult
from .registry import get_engine, list_engines, CAPABILITY_MAP

__all__ = [
    "CapabilityEngine",
    "CapabilityResult", 
    "get_engine",
    "list_engines",
    "CAPABILITY_MAP",
]
