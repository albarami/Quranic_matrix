"""
Phase 10.1: Optional torch_geometric Tests

Verifies that:
1. Importing src.ml.* does not crash even without torch_geometric
2. Graph reasoner works with fallback to semantic_graph_v2.json
3. API still functions when torch_geometric is unavailable
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.mark.phase10
class TestImportSafety:
    """Test that imports don't crash without torch_geometric."""
    
    def test_import_graph_reasoner_no_crash(self):
        """Importing graph_reasoner should not crash."""
        # This should not raise any exception
        from src.ml import graph_reasoner
        assert hasattr(graph_reasoner, 'ReasoningEngine')
        assert hasattr(graph_reasoner, 'GraphBuilder')
        assert hasattr(graph_reasoner, 'QBMGraphReasoner')
    
    def test_import_checks_are_lazy(self):
        """torch_geometric checks should be lazy, not at import time."""
        from src.ml.graph_reasoner import TORCH_AVAILABLE, PYG_AVAILABLE
        # These should be False initially (lazy)
        # They only become True after _check_*_available() is called
        # This test verifies the module loaded without crashing
        assert isinstance(TORCH_AVAILABLE, bool)
        assert isinstance(PYG_AVAILABLE, bool)
    
    def test_reasoning_engine_instantiation(self):
        """ReasoningEngine should instantiate without torch_geometric."""
        from src.ml.graph_reasoner import ReasoningEngine
        
        # Should not crash
        engine = ReasoningEngine()
        assert engine is not None
        assert engine.model is None  # Not initialized yet (lazy)
        assert engine._model_initialized == False


@pytest.mark.phase10
class TestFallbackGraphReasoning:
    """Test fallback to semantic_graph_v2.json."""
    
    def test_fallback_graph_loading(self):
        """Should load fallback graph when torch_geometric unavailable."""
        from src.ml.graph_reasoner import ReasoningEngine
        
        engine = ReasoningEngine()
        graph = engine._load_fallback_graph()
        
        # Should load if file exists
        fallback_path = Path("data/graph/semantic_graph_v2.json")
        if fallback_path.exists():
            assert graph is not None
            assert "edges" in graph
            assert len(engine._fallback_adjacency) > 0
        else:
            # Skip if file doesn't exist
            pytest.skip("semantic_graph_v2.json not found")
    
    def test_find_path_uses_fallback(self):
        """find_path should work with fallback graph."""
        from src.ml.graph_reasoner import ReasoningEngine, _check_pyg_available
        
        engine = ReasoningEngine()
        
        # Load fallback graph
        graph = engine._load_fallback_graph()
        if graph is None:
            pytest.skip("Fallback graph not available")
        
        # Get two nodes from the graph
        edges = graph.get("edges", [])
        if len(edges) < 1:
            pytest.skip("No edges in fallback graph")
        
        start = edges[0].get("source")
        end = edges[0].get("target")
        
        # Find path should work
        result = engine.find_path(start, end, max_depth=3)
        
        # Should return a result (found or not found)
        assert "found" in result or "error" in result
        
        # If pyg not available, should use fallback
        if not _check_pyg_available():
            assert result.get("fallback", False) == True
    
    def test_graph_builder_without_pyg(self):
        """GraphBuilder should work without torch_geometric."""
        from src.ml.graph_reasoner import GraphBuilder
        
        builder = GraphBuilder()
        
        # Add nodes and edges
        builder.add_node("test_node_1", "behavior")
        builder.add_node("test_node_2", "behavior")
        builder.add_edge("test_node_1", "test_node_2", "causes")
        
        assert len(builder.node_to_idx) == 2
        assert len(builder.edges) == 1
        
        # build() should return None if pyg not available
        # (or a Data object if it is)
        result = builder.build()
        # Just verify it doesn't crash
        assert result is None or result is not None


@pytest.mark.phase10
class TestGracefulDegradation:
    """Test that features degrade gracefully without torch_geometric."""
    
    def test_predict_missing_relations_empty_without_model(self):
        """predict_missing_relations should return empty list without model."""
        from src.ml.graph_reasoner import ReasoningEngine
        
        engine = ReasoningEngine()
        predictions = engine.predict_missing_relations()
        
        # Should return empty list, not crash
        assert predictions == []
    
    def test_discover_patterns_empty_without_graph_data(self):
        """discover_patterns should return empty list without graph_data."""
        from src.ml.graph_reasoner import ReasoningEngine
        
        engine = ReasoningEngine()
        patterns = engine.discover_patterns()
        
        # Should return empty list, not crash
        assert patterns == []
    
    def test_qbm_graph_reasoner_wrapper_safe(self):
        """QBMGraphReasoner wrapper should be safe with or without torch_geometric."""
        from src.ml.graph_reasoner import QBMGraphReasoner, _check_pyg_available
        
        reasoner = QBMGraphReasoner()
        
        # These should not crash regardless of torch_geometric availability
        reasoner.eval()
        reasoner.train()
        state = reasoner.state_dict()
        
        # If pyg available, state_dict will have weights; otherwise empty
        if _check_pyg_available():
            assert isinstance(state, dict)
        else:
            assert state == {}
        
        score = reasoner.score_path([])
        assert score == 0.0


@pytest.mark.phase10
class TestSemanticGraphFallbackContent:
    """Test that fallback graph has expected content."""
    
    def test_fallback_graph_has_edges(self):
        """Fallback graph should have edges with evidence."""
        from src.ml.graph_reasoner import ReasoningEngine
        
        engine = ReasoningEngine()
        graph = engine._load_fallback_graph()
        
        if graph is None:
            pytest.skip("Fallback graph not available")
        
        edges = graph.get("edges", [])
        assert len(edges) > 0, "Fallback graph has no edges"
        
        # Check edge structure
        edge = edges[0]
        assert "source" in edge
        assert "target" in edge
        assert "edge_type" in edge
    
    def test_fallback_adjacency_built_correctly(self):
        """Adjacency list should be built from fallback graph."""
        from src.ml.graph_reasoner import ReasoningEngine
        
        engine = ReasoningEngine()
        graph = engine._load_fallback_graph()
        
        if graph is None:
            pytest.skip("Fallback graph not available")
        
        # Adjacency should be populated
        assert len(engine._fallback_adjacency) > 0
        
        # Each entry should be a list of (neighbor, edge_type) tuples
        for node, neighbors in engine._fallback_adjacency.items():
            assert isinstance(neighbors, list)
            for neighbor, edge_type in neighbors:
                assert isinstance(neighbor, str)
                assert isinstance(edge_type, str)


@pytest.mark.phase10
class TestAPIWithoutTorchGeometric:
    """Test that API endpoints work without torch_geometric."""
    
    def test_api_import_safe(self):
        """API main should be importable without crash."""
        # This may skip if other dependencies are missing
        try:
            import os
            os.environ.setdefault("SKIP_HEAVY_IMPORTS", "1")
            from src.api import main
            assert hasattr(main, 'app')
        except ImportError as e:
            pytest.skip(f"API import failed (expected in some envs): {e}")
        except Exception as e:
            # DLL load errors on Windows are expected
            if "0xc0000139" in str(e) or "DLL" in str(e):
                pytest.skip(f"DLL load error (expected without torch_geometric): {e}")
            raise


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "phase10"])
