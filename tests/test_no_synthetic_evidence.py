"""
Test: Proof system must never fabricate evidence.

Step 0 Gate: This test MUST pass before proceeding to Step 1.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_no_synthetic_evidence_in_graph():
    """Graph evidence must not contain synthetic edges when no real evidence exists."""
    from src.ml.mandatory_proof_system import GraphEvidence
    
    # Empty graph evidence should have empty lists
    graph = GraphEvidence()
    assert graph.nodes == [], "Empty GraphEvidence should have empty nodes list"
    assert graph.edges == [], "Empty GraphEvidence should have empty edges list"
    assert graph.paths == [], "Empty GraphEvidence should have empty paths list"


def test_no_synthetic_evidence_in_taxonomy():
    """Taxonomy evidence must not contain synthetic behaviors when no real evidence exists."""
    from src.ml.mandatory_proof_system import TaxonomyEvidence
    
    # Empty taxonomy evidence should have empty behaviors list
    taxonomy = TaxonomyEvidence()
    assert taxonomy.behaviors == [], "Empty TaxonomyEvidence should have empty behaviors list"


def test_graph_evidence_no_fabricated_edges():
    """GraphEvidence.to_proof() must not output fabricated edge data."""
    from src.ml.mandatory_proof_system import GraphEvidence
    
    # Create empty graph evidence
    graph = GraphEvidence(nodes=[], edges=[], paths=[])
    
    # The proof output should reflect empty data
    proof_text = graph.to_proof()
    
    # Should not contain fabricated edge patterns
    assert "سلوك1" not in proof_text, "Proof should not contain fabricated 'سلوك1'"
    assert "سلوك_أ" not in proof_text, "Proof should not contain fabricated 'سلوك_أ'"
    assert "يسبب" not in proof_text or "إجمالي الروابط:** 0" in proof_text, \
        "Proof should not contain fabricated edge types unless count is 0"


def test_taxonomy_evidence_no_fabricated_behaviors():
    """TaxonomyEvidence must not contain placeholder behaviors."""
    from src.ml.mandatory_proof_system import TaxonomyEvidence
    
    # Create empty taxonomy evidence
    taxonomy = TaxonomyEvidence(behaviors=[], dimensions={})
    
    # Should have empty behaviors
    assert len(taxonomy.behaviors) == 0, "Empty taxonomy should have 0 behaviors"
    
    # The proof output should reflect empty data
    proof_text = taxonomy.to_proof()
    
    # Should not contain fabricated behavior codes like "BHV_001" with "?" values
    # (The old code had: {"name": "سلوك", "code": "BHV_001", "evaluation": "?", ...})
    assert "BHV_001" not in proof_text or taxonomy.behaviors, \
        "Proof should not contain fabricated BHV_001 placeholder"


def test_proof_debug_tracks_no_evidence():
    """ProofDebug must correctly track when no evidence is found."""
    from src.ml.mandatory_proof_system import ProofDebug
    
    debug = ProofDebug()
    
    # Initially no fallbacks
    assert debug.fallback_used == False
    assert debug.graph_fallback == False
    assert debug.taxonomy_fallback == False
    
    # Add fallback for no evidence
    debug.graph_fallback = True
    debug.add_fallback("graph: no graph nodes found for query - returning empty (no synthetic data)")
    
    # Should be tracked
    assert debug.fallback_used == True
    assert "no synthetic data" in debug.fallback_reasons[0]


def test_complete_proof_validates_empty_components():
    """CompleteProof.validate() must correctly report missing components."""
    from src.ml.mandatory_proof_system import CompleteProof, GraphEvidence, TaxonomyEvidence
    
    # Create proof with empty graph and taxonomy
    proof = CompleteProof()
    proof.graph = GraphEvidence(nodes=[], edges=[], paths=[])
    proof.taxonomy = TaxonomyEvidence(behaviors=[], dimensions={})
    
    validation = proof.validate()
    
    # Should report these as missing
    assert "graph_nodes" in validation["missing"], "Empty graph nodes should be reported as missing"
    assert "graph_edges" in validation["missing"], "Empty graph edges should be reported as missing"
    assert "graph_paths" in validation["missing"], "Empty graph paths should be reported as missing"
    assert "taxonomy" in validation["missing"], "Empty taxonomy should be reported as missing"


def test_no_hardcoded_synthetic_patterns_in_source():
    """Source code must not contain hardcoded synthetic evidence patterns in executable code."""
    source_file = Path(__file__).parent.parent / "src" / "ml" / "mandatory_proof_system.py"
    
    with open(source_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # These patterns indicate synthetic evidence fabrication (only check non-comment lines)
    forbidden_patterns = [
        ('graph_paths = [["إيمان", "شكر", "صبر"]', "Fabricated paths assignment"),
        ('graph_paths = [["سلوك_أ"', "Fabricated placeholder paths"),
        ('detected_behaviors = [\'إيمان\', \'كفر\'', "Fabricated behaviors for framework"),
        ('"from": "سلوك1"', "Fabricated edge source"),
        ('"to": "سلوك2"', "Fabricated edge target"),
        ('graph_edges = [\n                    {"from": "إيمان"', "Fabricated edges block"),
    ]
    
    for line_num, line in enumerate(lines, 1):
        # Skip comment lines
        stripped = line.strip()
        if stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
            continue
        
        for pattern, description in forbidden_patterns:
            if pattern in line:
                pytest.fail(f"Line {line_num}: Found forbidden pattern ({description}): {pattern}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
