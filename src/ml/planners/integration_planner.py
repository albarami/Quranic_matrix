"""
Phase 3: INTEGRATION_E2E Planner (3.10)

Orchestrator planner that combines multiple specialized planners.
Provides cross-component consistency checks and conflict flagging.

REUSES:
- All other planners via LegendaryPlanner.query()
- All specialized planners (causal, tafsir, profile, etc.)

ADDS:
- Cross-component consistency checks
- Conflict flagging in debug trace
- Unified response aggregation
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class ComponentResult:
    """Result from a single component/planner."""
    component_name: str
    status: str  # "success", "partial", "failed", "skipped"
    data: Dict[str, Any]
    gaps: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ConsistencyCheck:
    """Result of a cross-component consistency check."""
    check_name: str
    components: List[str]
    passed: bool
    details: str
    conflicts: List[str] = field(default_factory=list)


@dataclass
class IntegrationResult:
    """Result of integrated E2E analysis."""
    query_type: str  # "full_analysis", "consistency_check", "conflict_report"
    entity_id: str
    component_results: List[ComponentResult]
    consistency_checks: List[ConsistencyCheck]
    total_components: int
    successful_components: int
    total_gaps: List[str]
    total_conflicts: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_type": self.query_type,
            "entity_id": self.entity_id,
            "component_results": [
                {
                    "component_name": c.component_name,
                    "status": c.status,
                    "gaps": c.gaps,
                    "warnings": c.warnings,
                }
                for c in self.component_results
            ],
            "consistency_checks": [
                {
                    "check_name": c.check_name,
                    "components": c.components,
                    "passed": c.passed,
                    "details": c.details,
                    "conflicts": c.conflicts,
                }
                for c in self.consistency_checks
            ],
            "total_components": self.total_components,
            "successful_components": self.successful_components,
            "total_gaps": self.total_gaps,
            "total_conflicts": self.total_conflicts,
        }


class IntegrationPlanner:
    """
    E2E Integration planner that orchestrates all specialized planners.

    Provides:
    - Full integrated analysis combining multiple planners
    - Cross-component consistency checks
    - Conflict detection and flagging
    """

    def __init__(self, legendary_planner=None):
        self._planner = legendary_planner
        self._specialized_planners = {}

    def _ensure_planner(self):
        if self._planner is None:
            from src.ml.legendary_planner import get_legendary_planner
            self._planner = get_legendary_planner()

    def _get_specialized_planner(self, planner_type: str):
        """Get or create a specialized planner."""
        if planner_type not in self._specialized_planners:
            if planner_type == "causal_chain":
                from src.ml.planners.causal_chain_planner import CausalChainPlanner
                self._specialized_planners[planner_type] = CausalChainPlanner(self._planner)
            elif planner_type == "cross_tafsir":
                from src.ml.planners.cross_tafsir_planner import CrossTafsirPlanner
                self._specialized_planners[planner_type] = CrossTafsirPlanner(self._planner)
            elif planner_type == "profile_11d":
                from src.ml.planners.profile_11d_planner import Profile11DPlanner
                self._specialized_planners[planner_type] = Profile11DPlanner(self._planner)
            elif planner_type == "graph_metrics":
                from src.ml.planners.graph_metrics_planner import GraphMetricsPlanner
                self._specialized_planners[planner_type] = GraphMetricsPlanner(self._planner)
            elif planner_type == "heart_state":
                from src.ml.planners.heart_state_planner import HeartStatePlanner
                self._specialized_planners[planner_type] = HeartStatePlanner(self._planner)
            elif planner_type == "agent":
                from src.ml.planners.agent_planner import AgentPlanner
                self._specialized_planners[planner_type] = AgentPlanner(self._planner)
            elif planner_type == "temporal_spatial":
                from src.ml.planners.temporal_spatial_planner import TemporalSpatialPlanner
                self._specialized_planners[planner_type] = TemporalSpatialPlanner(self._planner)
            elif planner_type == "consequence":
                from src.ml.planners.consequence_planner import ConsequencePlanner
                self._specialized_planners[planner_type] = ConsequencePlanner(self._planner)
            elif planner_type == "embeddings":
                from src.ml.planners.embeddings_planner import EmbeddingsPlanner
                self._specialized_planners[planner_type] = EmbeddingsPlanner(self._planner)

        return self._specialized_planners.get(planner_type)

    def run_full_analysis(self, entity_id: str) -> IntegrationResult:
        """
        Run full integrated analysis for an entity.

        Args:
            entity_id: The entity to analyze

        Returns:
            IntegrationResult with all component results and consistency checks
        """
        self._ensure_planner()

        component_results = []
        all_gaps = []
        all_conflicts = []

        # 1. Cross-tafsir analysis
        try:
            tafsir_planner = self._get_specialized_planner("cross_tafsir")
            tafsir_result = tafsir_planner.compare_tafsir_for_entity(entity_id)
            component_results.append(ComponentResult(
                component_name="cross_tafsir",
                status="success" if tafsir_result.sources_with_evidence > 0 else "partial",
                data=tafsir_result.to_dict(),
                gaps=tafsir_result.gaps,
            ))
            all_gaps.extend(tafsir_result.gaps)
        except Exception as e:
            component_results.append(ComponentResult(
                component_name="cross_tafsir",
                status="failed",
                data={},
                gaps=[f"cross_tafsir_error:{str(e)}"],
            ))
            all_gaps.append(f"cross_tafsir_error:{str(e)}")

        # 2. 11D Profile analysis
        try:
            profile_planner = self._get_specialized_planner("profile_11d")
            profile_result = profile_planner.profile_behavior(entity_id)
            component_results.append(ComponentResult(
                component_name="profile_11d",
                status="success" if profile_result.filled_dimensions > 0 else "partial",
                data=profile_result.to_dict(),
                gaps=profile_result.gaps,
            ))
            all_gaps.extend(profile_result.gaps)
        except Exception as e:
            component_results.append(ComponentResult(
                component_name="profile_11d",
                status="failed",
                data={},
                gaps=[f"profile_11d_error:{str(e)}"],
            ))
            all_gaps.append(f"profile_11d_error:{str(e)}")

        # 3. Graph metrics analysis
        try:
            graph_planner = self._get_specialized_planner("graph_metrics")
            node_metrics = graph_planner.get_node_metrics(entity_id)
            has_connections = node_metrics.get("total_degree", 0) > 0
            gaps_list = ["entity_not_in_graph"] if not has_connections else []
            component_results.append(ComponentResult(
                component_name="graph_metrics",
                status="success" if has_connections else "partial",
                data=node_metrics,
                gaps=gaps_list,
            ))
            all_gaps.extend(gaps_list)
        except Exception as e:
            component_results.append(ComponentResult(
                component_name="graph_metrics",
                status="failed",
                data={},
                gaps=[f"graph_metrics_error:{str(e)}"],
            ))
            all_gaps.append(f"graph_metrics_error:{str(e)}")

        # 4. Consequence analysis
        try:
            consequence_planner = self._get_specialized_planner("consequence")
            consequence_result = consequence_planner.get_behavior_consequences(entity_id)
            component_results.append(ComponentResult(
                component_name="consequence",
                status="success" if consequence_result.total_consequences > 0 else "partial",
                data=consequence_result.to_dict(),
                gaps=consequence_result.gaps,
            ))
            all_gaps.extend(consequence_result.gaps)
        except Exception as e:
            component_results.append(ComponentResult(
                component_name="consequence",
                status="failed",
                data={},
                gaps=[f"consequence_error:{str(e)}"],
            ))
            all_gaps.append(f"consequence_error:{str(e)}")

        # Run consistency checks
        consistency_checks = self._run_consistency_checks(component_results)

        # Extract conflicts
        for check in consistency_checks:
            if not check.passed:
                all_conflicts.extend(check.conflicts)

        successful = sum(1 for c in component_results if c.status == "success")

        return IntegrationResult(
            query_type="full_analysis",
            entity_id=entity_id,
            component_results=component_results,
            consistency_checks=consistency_checks,
            total_components=len(component_results),
            successful_components=successful,
            total_gaps=list(set(all_gaps)),  # Deduplicate
            total_conflicts=all_conflicts,
        )

    def _run_consistency_checks(
        self,
        component_results: List[ComponentResult]
    ) -> List[ConsistencyCheck]:
        """Run cross-component consistency checks."""
        checks = []

        # Get result data by component name
        results_by_name = {c.component_name: c for c in component_results}

        # Check 1: Tafsir evidence vs Profile evidence
        if "cross_tafsir" in results_by_name and "profile_11d" in results_by_name:
            tafsir_data = results_by_name["cross_tafsir"].data
            profile_data = results_by_name["profile_11d"].data

            tafsir_verses = set()
            for source in tafsir_data.get("sources", []):
                tafsir_verses.update(source.get("sample_verses", []))

            profile_verses = set()
            for dim in profile_data.get("dimensions", []):
                profile_verses.update(dim.get("sample_verses", []))

            overlap = tafsir_verses & profile_verses
            conflicts = []

            if tafsir_verses and profile_verses and not overlap:
                conflicts.append("No overlapping verses between tafsir and profile analysis")

            checks.append(ConsistencyCheck(
                check_name="tafsir_profile_verse_consistency",
                components=["cross_tafsir", "profile_11d"],
                passed=len(conflicts) == 0,
                details=f"Tafsir verses: {len(tafsir_verses)}, Profile verses: {len(profile_verses)}, Overlap: {len(overlap)}",
                conflicts=conflicts,
            ))

        # Check 2: Graph connectivity vs Consequence existence
        if "graph_metrics" in results_by_name and "consequence" in results_by_name:
            graph_data = results_by_name["graph_metrics"].data
            consequence_data = results_by_name["consequence"].data

            # get_node_metrics returns flat dict with out_degree directly
            has_outgoing = graph_data.get("out_degree", 0) > 0

            has_consequences = consequence_data.get("total_consequences", 0) > 0

            conflicts = []
            if has_outgoing and not has_consequences:
                conflicts.append("Entity has graph connections but no consequences found")

            checks.append(ConsistencyCheck(
                check_name="graph_consequence_consistency",
                components=["graph_metrics", "consequence"],
                passed=len(conflicts) == 0,
                details=f"Has outgoing edges: {has_outgoing}, Has consequences: {has_consequences}",
                conflicts=conflicts,
            ))

        return checks

    def run_consistency_report(self, entity_id: str) -> IntegrationResult:
        """
        Run only consistency checks for an entity.

        Args:
            entity_id: The entity to check

        Returns:
            IntegrationResult focused on consistency
        """
        full_result = self.run_full_analysis(entity_id)

        return IntegrationResult(
            query_type="consistency_check",
            entity_id=entity_id,
            component_results=[],  # Don't include full data
            consistency_checks=full_result.consistency_checks,
            total_components=full_result.total_components,
            successful_components=full_result.successful_components,
            total_gaps=full_result.total_gaps,
            total_conflicts=full_result.total_conflicts,
        )

    def get_conflict_report(self, entity_id: str) -> IntegrationResult:
        """
        Get only conflicts for an entity.

        Args:
            entity_id: The entity to check

        Returns:
            IntegrationResult with conflicts only
        """
        full_result = self.run_full_analysis(entity_id)

        # Filter to only failed checks
        failed_checks = [c for c in full_result.consistency_checks if not c.passed]

        return IntegrationResult(
            query_type="conflict_report",
            entity_id=entity_id,
            component_results=[],
            consistency_checks=failed_checks,
            total_components=full_result.total_components,
            successful_components=full_result.successful_components,
            total_gaps=full_result.total_gaps,
            total_conflicts=full_result.total_conflicts,
        )
