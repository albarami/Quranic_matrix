"""
Phase 3: Thin Wrapper Planners for QBM

These planners wrap LegendaryPlanner functionality for specific question classes.
They do NOT re-implement logic - they compose and extend existing methods.

Each planner returns a structured result that feeds into the analysis payload.
"""

from .causal_chain_planner import CausalChainPlanner
from .cross_tafsir_planner import CrossTafsirPlanner
from .profile_11d_planner import Profile11DPlanner
from .graph_metrics_planner import GraphMetricsPlanner
from .heart_state_planner import HeartStatePlanner
from .agent_planner import AgentPlanner
from .temporal_spatial_planner import TemporalSpatialPlanner
from .consequence_planner import ConsequencePlanner
from .embeddings_planner import EmbeddingsPlanner
from .integration_planner import IntegrationPlanner

__all__ = [
    "CausalChainPlanner",
    "CrossTafsirPlanner",
    "Profile11DPlanner",
    "GraphMetricsPlanner",
    "HeartStatePlanner",
    "AgentPlanner",
    "TemporalSpatialPlanner",
    "ConsequencePlanner",
    "EmbeddingsPlanner",
    "IntegrationPlanner",
]
