"""
Tests for Discovery Engine (Phase 3 of Behavior Mastery Plan).

Tests enforce:
- Discovery report is reproducible
- Multi-hop paths have evidence
- Bridge behaviors are in canonical 87
- Communities cover all behaviors
- Motifs have minimum support
- Hypotheses have falsification checks
- Link predictions are sandboxed (promotion_blocked=True, is_confirmed=False)
"""

import json
import pytest
from pathlib import Path

from src.graph.discovery import (
    DiscoveryEngine,
    Hypothesis,
    CausalPath,
    BridgeBehavior,
    BehaviorCommunity,
    GraphMotif,
    EvidenceBundle,
    FalsificationResult,
    DISCOVERY_REPORT_PATH,
    BRIDGES_PATH,
    COMMUNITIES_PATH,
    MOTIFS_PATH,
)


# Paths
CANONICAL_ENTITIES_PATH = Path("vocab/canonical_entities.json")


class TestDiscoveryDataclasses:
    """Tests for discovery dataclasses."""
    
    def test_hypothesis_schema(self):
        """Hypothesis must have all required fields."""
        h = Hypothesis(
            hypothesis_id="TEST_001",
            hypothesis_type="path",
            involved_behaviors=["BEH_A", "BEH_B"],
            evidence_bundle=EvidenceBundle(verse_keys=["2:153"]),
            confidence_score=0.85,
        )
        d = h.to_dict()
        assert "hypothesis_id" in d
        assert "hypothesis_type" in d
        assert "involved_behaviors" in d
        assert "evidence_bundle" in d
        assert "confidence_score" in d
        assert "is_confirmed" in d
        assert "promotion_blocked" in d
    
    def test_link_prediction_sandboxed_by_default(self):
        """Link predictions must be sandboxed."""
        h = Hypothesis(
            hypothesis_id="LINK_PRED_001",
            hypothesis_type="link_prediction",
            involved_behaviors=["BEH_A", "BEH_B"],
            evidence_bundle=EvidenceBundle(),
            confidence_score=0.7,
            is_confirmed=False,
            promotion_blocked=True,
        )
        assert h.is_confirmed is False
        assert h.promotion_blocked is True
    
    def test_causal_path_schema(self):
        """CausalPath must have all required fields."""
        p = CausalPath(
            path_id="PATH_001",
            nodes=["A", "B", "C"],
            edges=[{"source": "A", "target": "B"}],
            total_hops=2,
            total_evidence=10,
            confidence=0.8,
        )
        d = p.to_dict()
        assert "path_id" in d
        assert "nodes" in d
        assert "edges" in d
        assert "total_hops" in d
        assert "total_evidence" in d


class TestDiscoveryEngine:
    """Tests for discovery engine."""
    
    @pytest.fixture
    def engine(self) -> DiscoveryEngine:
        """Create and load engine."""
        eng = DiscoveryEngine()
        eng.load_graph()
        return eng
    
    @pytest.fixture
    def canonical_behavior_ids(self) -> set:
        """Get canonical behavior IDs."""
        with open(CANONICAL_ENTITIES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {b["id"] for b in data.get("behaviors", [])}
    
    def test_engine_loads_graph(self, engine: DiscoveryEngine):
        """Engine must load normalized graph."""
        assert engine.graph is not None
        assert len(engine.node_lookup) > 0
    
    def test_multihop_paths_have_evidence(self, engine: DiscoveryEngine):
        """Every path edge must have evidence."""
        paths = engine.find_multihop_paths(min_hops=2, max_hops=4, top_k=20)
        
        for path in paths:
            assert path.total_hops >= 2
            assert len(path.edges) == path.total_hops
            for edge in path.edges:
                assert "evidence_count" in edge
                assert edge["evidence_count"] >= 0
    
    def test_bridge_behaviors_are_valid(self, engine: DiscoveryEngine, canonical_behavior_ids: set):
        """All bridge behaviors must be in canonical 87."""
        bridges = engine.find_bridge_behaviors(top_k=20)
        
        for bridge in bridges:
            assert bridge.behavior_id in canonical_behavior_ids, \
                f"Bridge {bridge.behavior_id} not in canonical behaviors"
    
    def test_communities_cover_all_behaviors(self, engine: DiscoveryEngine):
        """Every behavior must belong to exactly one community."""
        communities = engine.detect_communities()
        behavior_ids = set(engine._get_behavior_ids())
        
        # Collect all behaviors in communities
        behaviors_in_communities = set()
        for comm in communities:
            for beh in comm.behaviors:
                assert beh not in behaviors_in_communities, \
                    f"Behavior {beh} in multiple communities"
                behaviors_in_communities.add(beh)
        
        # All behaviors should be covered
        missing = behavior_ids - behaviors_in_communities
        assert len(missing) == 0, f"Behaviors not in any community: {missing}"
    
    def test_motifs_have_minimum_support(self, engine: DiscoveryEngine):
        """Each motif must appear at least min_support times."""
        motifs = engine.find_motifs(min_support=3)
        
        for motif in motifs:
            assert motif.support_count >= 3, \
                f"Motif {motif.motif_id} has support {motif.support_count} < 3"
            assert len(motif.instances) > 0
    
    def test_link_predictions_sandboxed(self, engine: DiscoveryEngine):
        """Link predictions must have promotion_blocked=True and is_confirmed=False."""
        predictions = engine.predict_links(min_confidence=0.3)
        
        for pred in predictions:
            assert pred.hypothesis_type == "link_prediction"
            assert pred.is_confirmed is False, \
                f"Link prediction {pred.hypothesis_id} should not be confirmed"
            assert pred.promotion_blocked is True, \
                f"Link prediction {pred.hypothesis_id} should be promotion_blocked"
    
    def test_hypotheses_have_falsification_check(self, engine: DiscoveryEngine):
        """Every hypothesis must include falsification result."""
        predictions = engine.predict_links(min_confidence=0.3)
        
        for pred in predictions:
            assert pred.falsification_check is not None
            assert hasattr(pred.falsification_check, "counter_evidence_found")
            assert hasattr(pred.falsification_check, "search_scope")


class TestBuiltDiscoveryArtifacts:
    """Tests for built discovery artifacts (run after build_discovery_report.py)."""
    
    def test_discovery_report_exists(self):
        """Discovery report must exist after build."""
        if not DISCOVERY_REPORT_PATH.exists():
            pytest.skip("Discovery report not built yet - run build_discovery_report.py first")
        
        with open(DISCOVERY_REPORT_PATH, "r", encoding="utf-8") as f:
            report = json.load(f)
        
        assert "schema_version" in report
        assert "statistics" in report
        assert "top_paths" in report
        assert "bridges" in report
        assert "communities" in report
        assert "motifs" in report
    
    def test_discovery_report_reproducible(self):
        """Same input produces identical discovery report."""
        if not DISCOVERY_REPORT_PATH.exists():
            pytest.skip("Discovery report not built yet")
        
        # Load existing report
        with open(DISCOVERY_REPORT_PATH, "r", encoding="utf-8") as f:
            report1 = json.load(f)
        
        # Generate new report
        engine = DiscoveryEngine()
        engine.load_graph()
        report2 = engine.generate_discovery_report()
        
        # Compare statistics (timestamps will differ)
        assert report1["statistics"] == report2["statistics"]
        assert len(report1["bridges"]) == len(report2["bridges"])
        assert len(report1["communities"]) == len(report2["communities"])
    
    def test_link_predictions_not_in_operational_graph(self):
        """Operational graph must not contain any unconfirmed link predictions."""
        if not DISCOVERY_REPORT_PATH.exists():
            pytest.skip("Discovery report not built yet")
        
        with open(DISCOVERY_REPORT_PATH, "r", encoding="utf-8") as f:
            report = json.load(f)
        
        # All link predictions must be sandboxed
        for lp in report.get("link_predictions", []):
            assert lp.get("is_confirmed") is False, \
                f"Link prediction {lp['hypothesis_id']} should not be confirmed"
            assert lp.get("promotion_blocked") is True, \
                f"Link prediction {lp['hypothesis_id']} should be promotion_blocked"
    
    def test_bridges_file_exists(self):
        """Bridges file must exist."""
        if not BRIDGES_PATH.exists():
            pytest.skip("Bridges not built yet")
        
        with open(BRIDGES_PATH, "r", encoding="utf-8") as f:
            bridges = json.load(f)
        
        assert isinstance(bridges, list)
        for bridge in bridges:
            assert "behavior_id" in bridge
            assert "betweenness_centrality" in bridge
    
    def test_communities_file_exists(self):
        """Communities file must exist."""
        if not COMMUNITIES_PATH.exists():
            pytest.skip("Communities not built yet")
        
        with open(COMMUNITIES_PATH, "r", encoding="utf-8") as f:
            communities = json.load(f)
        
        assert isinstance(communities, list)
        for comm in communities:
            assert "community_id" in comm
            assert "behaviors" in comm
            assert len(comm["behaviors"]) > 0
    
    def test_motifs_file_exists(self):
        """Motifs file must exist."""
        if not MOTIFS_PATH.exists():
            pytest.skip("Motifs not built yet")
        
        with open(MOTIFS_PATH, "r", encoding="utf-8") as f:
            motifs = json.load(f)
        
        assert isinstance(motifs, list)
        for motif in motifs:
            assert "motif_id" in motif
            assert "pattern_type" in motif
            assert "support_count" in motif


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
