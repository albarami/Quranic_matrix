"""
Test: Proof System Integration with Hybrid Evidence Retriever

Phase 5.5: Ensures hybrid retriever is PRIMARY tafsir retrieval path.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.mandatory_proof_system import ProofDebug, GraphEvidence, TaxonomyEvidence, MANDATORY_COMPONENTS

CORE_SOURCES = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"]


def test_no_synthetic_evidence_in_proof_debug():
    """ProofDebug must track when no evidence is found without fabrication."""
    debug = ProofDebug()
    
    # Initially no fallbacks
    assert debug.fallback_used == False
    assert debug.graph_fallback == False
    assert debug.taxonomy_fallback == False
    
    # Add fallback for no evidence
    debug.graph_fallback = True
    debug.add_fallback("graph: no graph nodes found - returning empty (no synthetic data)")
    
    # Should be tracked correctly
    assert debug.fallback_used == True
    assert "no synthetic data" in debug.fallback_reasons[0]


def test_proof_debug_has_retrieval_mode():
    """ProofDebug must include retrieval_mode field."""
    debug = ProofDebug()
    
    assert hasattr(debug, 'retrieval_mode')
    assert debug.retrieval_mode in ["hybrid", "stratified", "rag_only"]


def test_proof_debug_has_sources_covered():
    """ProofDebug must include sources_covered field."""
    debug = ProofDebug()
    
    assert hasattr(debug, 'sources_covered')
    assert hasattr(debug, 'core_sources_count')
    
    # Test setting sources
    debug.sources_covered = ["ibn_kathir", "tabari", "qurtubi"]
    debug.core_sources_count = 3
    
    assert len(debug.sources_covered) == 3
    assert debug.core_sources_count == 3


def test_proof_debug_to_dict_includes_new_fields():
    """ProofDebug.to_dict() must include retrieval_mode and sources_covered."""
    debug = ProofDebug()
    debug.retrieval_mode = "hybrid"
    debug.sources_covered = ["ibn_kathir", "jalalayn"]
    debug.core_sources_count = 2
    
    d = debug.to_dict()
    
    assert "retrieval_mode" in d
    assert "sources_covered" in d
    assert "core_sources_count" in d
    assert d["retrieval_mode"] == "hybrid"
    assert d["sources_covered"] == ["ibn_kathir", "jalalayn"]
    assert d["core_sources_count"] == 2


def test_graph_evidence_empty_is_valid():
    """Empty GraphEvidence should not contain synthetic data."""
    graph = GraphEvidence(nodes=[], edges=[], paths=[])
    
    assert graph.nodes == []
    assert graph.edges == []
    assert graph.paths == []
    
    # Proof output should reflect empty data
    proof_text = graph.to_proof()
    assert "إجمالي العقد:** 0" in proof_text
    assert "إجمالي الروابط:** 0" in proof_text


def test_taxonomy_evidence_empty_is_valid():
    """Empty TaxonomyEvidence should not contain synthetic data."""
    taxonomy = TaxonomyEvidence(behaviors=[], dimensions={})
    
    assert taxonomy.behaviors == []


def test_no_fabricated_patterns_in_source():
    """Source code must not contain hardcoded synthetic evidence patterns."""
    source_file = Path(__file__).parent.parent / "src" / "ml" / "mandatory_proof_system.py"
    
    with open(source_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    forbidden_patterns = [
        ('graph_paths = [["إيمان", "شكر", "صبر"]', "Fabricated paths"),
        ('detected_behaviors = [\'إيمان\', \'كفر\'', "Fabricated behaviors"),
        ('"from": "سلوك1"', "Fabricated edge"),
    ]
    
    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        
        for pattern, description in forbidden_patterns:
            if pattern in line:
                pytest.fail(f"Line {line_num}: Found forbidden pattern ({description})")


def test_fallback_used_false_on_standard_debug():
    """Standard ProofDebug should have fallback_used=False initially."""
    debug = ProofDebug()
    
    assert debug.fallback_used == False
    assert debug.fallback_reasons == []


class TestHybridRetrieverIntegration:
    """Tests that verify hybrid retriever is PRIMARY tafsir path."""
    
    def test_mandatory_proof_system_has_hybrid_retriever(self):
        """MandatoryProofSystem must initialize hybrid_retriever."""
        # Check the source code has hybrid retriever initialization
        source_file = Path(__file__).parent.parent / "src" / "ml" / "mandatory_proof_system.py"
        
        with open(source_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        assert "self.hybrid_retriever" in content, "hybrid_retriever not initialized"
        assert "get_hybrid_retriever" in content, "get_hybrid_retriever not imported"
    
    def test_answer_with_full_proof_uses_hybrid(self):
        """answer_with_full_proof must use hybrid_retriever as PRIMARY path."""
        source_file = Path(__file__).parent.parent / "src" / "ml" / "mandatory_proof_system.py"
        
        with open(source_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Must have hybrid retriever search call
        assert "hybrid_retriever.search" in content, "hybrid_retriever.search not called"
        
        # Must set retrieval_mode to hybrid
        assert 'retrieval_mode = "hybrid"' in content, "retrieval_mode not set to hybrid"
    
    def test_tafsir_results_include_chunk_id(self):
        """Tafsir results from hybrid retriever must include chunk_id."""
        source_file = Path(__file__).parent.parent / "src" / "ml" / "mandatory_proof_system.py"
        
        with open(source_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Must include chunk_id in tafsir results
        assert '"chunk_id": r.chunk_id' in content, "chunk_id not included in tafsir results"
    
    def test_stratified_is_fallback_only(self):
        """Stratified retriever must be fallback, not primary."""
        source_file = Path(__file__).parent.parent / "src" / "ml" / "mandatory_proof_system.py"
        
        with open(source_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Stratified should be in else block (fallback)
        assert "FALLBACK: Use stratified retriever" in content, "stratified not marked as fallback"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
