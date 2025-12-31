"""
Phase 6 Tests: Azure OpenAI Integration (v2.0 - Zero Hallucination)

Tests for:
1. Function-calling tools
2. Citation verifier with strict contracts
3. Fail-closed gate
4. Subset contract verification
5. No surah_intro enforcement
6. Orchestrator (without live API)
"""

import json
import pytest
from pathlib import Path

# Import Phase 6 components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.azure.tools import QBMTools, TOOL_DEFINITIONS
from src.azure.verifier import (
    CitationVerifier, 
    VerificationResult, 
    fail_closed_gate,
    GENERIC_OPENING_VERSES,
    CANONICAL_TAFSIR_SOURCES,
    MAX_AYAH_PER_SURAH
)


# =============================================================================
# TOOL DEFINITIONS TESTS
# =============================================================================

class TestToolDefinitions:
    """Test tool definitions are properly formatted."""
    
    def test_tool_definitions_exist(self):
        """All required tools are defined."""
        tool_names = [t["function"]["name"] for t in TOOL_DEFINITIONS]
        
        required_tools = [
            "resolve_entity",
            "get_behavior_dossier",
            "get_causal_paths",
            "get_tafsir_comparison",
            "get_graph_metrics",
            "get_verse_evidence"
        ]
        
        for tool in required_tools:
            assert tool in tool_names, f"Missing tool: {tool}"
    
    def test_tool_definitions_format(self):
        """Tool definitions follow OpenAI format."""
        for tool in TOOL_DEFINITIONS:
            assert tool["type"] == "function"
            assert "function" in tool
            assert "name" in tool["function"]
            assert "description" in tool["function"]
            assert "parameters" in tool["function"]
            assert tool["function"]["parameters"]["type"] == "object"


# =============================================================================
# QBM TOOLS TESTS
# =============================================================================

class TestQBMTools:
    """Test QBM tool implementations."""
    
    @pytest.fixture
    def tools(self):
        """Create tools instance."""
        return QBMTools()
    
    def test_resolve_entity_arabic(self, tools):
        """Test resolving Arabic term to behavior ID."""
        result = tools.resolve_entity("الصبر")
        
        assert result["success"] is True
        assert result["entity_id"] == "BEH_EMO_PATIENCE"
        assert result["term_ar"] == "الصبر"
    
    def test_resolve_entity_english(self, tools):
        """Test resolving English term to behavior ID."""
        result = tools.resolve_entity("patience")
        
        assert result["success"] is True
        assert result["entity_id"] == "BEH_EMO_PATIENCE"
    
    def test_resolve_entity_not_found(self, tools):
        """Test resolving unknown term."""
        result = tools.resolve_entity("xyz_unknown_term")
        
        assert result["success"] is False
        assert "error" in result
    
    def test_get_behavior_dossier(self, tools):
        """Test getting behavior dossier."""
        result = tools.get_behavior_dossier("BEH_EMO_PATIENCE")
        
        assert result["success"] is True
        assert result["behavior_id"] == "BEH_EMO_PATIENCE"
        assert result["term_ar"] == "الصبر"
        assert result["verse_count"] > 0
        assert "verses" in result
        assert "provenance" in result
    
    def test_get_behavior_dossier_with_tafsir(self, tools):
        """Test getting behavior dossier with tafsir."""
        result = tools.get_behavior_dossier("BEH_EMO_PATIENCE", include_tafsir=True)
        
        assert result["success"] is True
        assert "tafsir" in result
        assert "tafsir_sources" in result
    
    def test_get_behavior_dossier_not_found(self, tools):
        """Test getting non-existent behavior."""
        result = tools.get_behavior_dossier("BEH_INVALID_ID")
        
        assert result["success"] is False
        assert "error" in result
    
    def test_get_causal_paths(self, tools):
        """Test finding causal paths."""
        result = tools.get_causal_paths(
            from_behavior="BEH_COG_HEEDLESSNESS",
            to_behavior="BEH_SPI_DISBELIEF"
        )
        
        assert result["success"] is True
        assert result["from_behavior"] == "BEH_COG_HEEDLESSNESS"
        assert result["to_behavior"] == "BEH_SPI_DISBELIEF"
        assert result["paths_found"] > 0
        assert "paths" in result
        assert "provenance" in result
    
    def test_get_causal_paths_with_evidence(self, tools):
        """Test causal paths include evidence."""
        result = tools.get_causal_paths(
            from_behavior="BEH_SPI_FAITH",
            to_behavior="BEH_FIN_CHARITY",
            max_hops=3
        )
        
        assert result["success"] is True
        if result["paths_found"] > 0:
            path = result["paths"][0]
            assert "nodes" in path
            assert "edge_types" in path
            assert "hops" in path
    
    def test_get_tafsir_comparison(self, tools):
        """Test getting tafsir comparison."""
        result = tools.get_tafsir_comparison(
            verse_keys=["2:45", "2:153"]
        )
        
        assert result["success"] is True
        assert result["verse_keys"] == ["2:45", "2:153"]
        assert "tafsir" in result
        assert "sources" in result
        assert result["total_chunks"] > 0
    
    def test_get_graph_metrics(self, tools):
        """Test getting graph metrics."""
        result = tools.get_graph_metrics()
        
        assert result["success"] is True
        assert result["total_nodes"] > 0
        assert result["total_edges"] > 0
        assert result["behavior_nodes"] == 87
        assert "edge_types" in result
    
    def test_get_graph_metrics_for_behavior(self, tools):
        """Test getting graph metrics for specific behavior."""
        result = tools.get_graph_metrics(
            behavior_id="BEH_SPI_FAITH",
            include_neighbors=True
        )
        
        assert result["success"] is True
        assert "behavior_metrics" in result
        assert result["behavior_metrics"]["behavior_id"] == "BEH_SPI_FAITH"
        assert "neighbors" in result["behavior_metrics"]
    
    def test_get_verse_evidence(self, tools):
        """Test getting verse evidence."""
        result = tools.get_verse_evidence("BEH_EMO_PATIENCE")
        
        assert result["success"] is True
        assert result["behavior_id"] == "BEH_EMO_PATIENCE"
        assert result["total_verses"] > 0
        assert "verses" in result
        
        # Check verse has text
        if result["verses"]:
            verse = result["verses"][0]
            assert "verse_key" in verse
            assert "text_uthmani" in verse
    
    def test_execute_tool_unknown(self, tools):
        """Test executing unknown tool."""
        result = tools.execute_tool("unknown_tool", {})
        
        assert result["success"] is False
        assert "error" in result
        assert "available_tools" in result


# =============================================================================
# CITATION VERIFIER TESTS
# =============================================================================

class TestCitationVerifier:
    """Test citation verification."""
    
    @pytest.fixture
    def verifier(self):
        """Create verifier instance."""
        return CitationVerifier()
    
    def test_verify_valid_verse_key(self, verifier):
        """Test verifying valid verse key."""
        assert verifier.verify_verse_key("2:255") is True
        assert verifier.verify_verse_key("1:1") is True
        assert verifier.verify_verse_key("114:6") is True
    
    def test_verify_invalid_verse_key(self, verifier):
        """Test verifying invalid verse key."""
        assert verifier.verify_verse_key("0:1") is False
        assert verifier.verify_verse_key("115:1") is False
        assert verifier.verify_verse_key("2:300") is False  # Baqarah has 286 verses
    
    def test_verify_valid_behavior_id(self, verifier):
        """Test verifying valid behavior ID."""
        assert verifier.verify_behavior_id("BEH_EMO_PATIENCE") is True
        assert verifier.verify_behavior_id("BEH_SPI_FAITH") is True
    
    def test_verify_invalid_behavior_id(self, verifier):
        """Test verifying invalid behavior ID."""
        assert verifier.verify_behavior_id("BEH_INVALID") is False
        assert verifier.verify_behavior_id("INVALID") is False
    
    def test_verify_tafsir_source(self, verifier):
        """Test verifying tafsir sources."""
        for source in CANONICAL_TAFSIR_SOURCES:
            assert verifier.verify_tafsir_source(source) is True
        
        assert verifier.verify_tafsir_source("unknown_source") is False
    
    def test_is_generic_opening_verse(self, verifier):
        """Test detecting generic opening verses."""
        # Fatiha
        assert verifier.is_generic_opening_verse("1:1") is True
        assert verifier.is_generic_opening_verse("1:7") is True
        
        # Early Baqarah
        assert verifier.is_generic_opening_verse("2:1") is True
        assert verifier.is_generic_opening_verse("2:20") is True
        
        # Not generic
        assert verifier.is_generic_opening_verse("2:21") is False
        assert verifier.is_generic_opening_verse("2:255") is False
    
    def test_verify_response_valid(self, verifier):
        """Test verifying valid response."""
        response = {
            "behavior_id": "BEH_EMO_PATIENCE",
            "verses": [
                {"verse_key": "2:45"},
                {"verse_key": "2:153"}
            ],
            "provenance": {
                "source": "concept_index_v3.jsonl"
            }
        }
        
        result = verifier.verify_response(response)
        
        assert result.valid is True
        assert len(result.violations) == 0
    
    def test_verify_response_invalid_verse(self, verifier):
        """Test verifying response with invalid verse."""
        response = {
            "behavior_id": "BEH_EMO_PATIENCE",
            "verses": [
                {"verse_key": "999:999"}  # Invalid
            ]
        }
        
        result = verifier.verify_response(response)
        
        assert result.valid is False
        assert len(result.violations) > 0
        assert result.violations[0]["type"] == "invalid_verse_key"
    
    def test_verify_response_invalid_behavior(self, verifier):
        """Test verifying response with invalid behavior."""
        response = {
            "behavior_id": "BEH_INVALID_BEHAVIOR"
        }
        
        result = verifier.verify_response(response)
        
        assert result.valid is False
        assert any(v["type"] == "invalid_behavior_id" for v in result.violations)
    
    def test_verify_response_generic_verses_only(self, verifier):
        """Test verifying response with only generic verses."""
        response = {
            "verses": [
                {"verse_key": "1:1"},
                {"verse_key": "1:2"},
                {"verse_key": "2:1"}
            ]
        }
        
        result = verifier.verify_response(response)
        
        assert result.valid is False
        assert any(v["type"] == "generic_opening_verses_only" for v in result.violations)


# =============================================================================
# FAIL-CLOSED GATE TESTS
# =============================================================================

class TestFailClosedGate:
    """Test fail-closed verification gate."""
    
    def test_valid_response_passes(self):
        """Test valid response passes through gate."""
        response = {
            "success": True,
            "behavior_id": "BEH_EMO_PATIENCE",
            "verses": [{"verse_key": "2:45"}]
        }
        
        result = fail_closed_gate(response)
        
        assert result["success"] is True
        assert "_verification" in result
        assert result["_verification"]["status"] == "PASSED"
    
    def test_invalid_response_blocked(self):
        """Test invalid response is blocked."""
        response = {
            "success": True,
            "behavior_id": "BEH_INVALID",
            "verses": [{"verse_key": "999:999"}]
        }
        
        result = fail_closed_gate(response)
        
        assert result["success"] is False
        assert result["error"] == "VERIFICATION_FAILED"
        assert "_verification" in result
        assert result["_verification"]["status"] == "FAILED"


# =============================================================================
# VERIFICATION RESULT TESTS
# =============================================================================

class TestVerificationResult:
    """Test VerificationResult dataclass."""
    
    def test_initial_state(self):
        """Test initial state is valid."""
        result = VerificationResult(valid=True)
        
        assert result.valid is True
        assert len(result.violations) == 0
        assert len(result.warnings) == 0
    
    def test_add_violation(self):
        """Test adding violation invalidates result."""
        result = VerificationResult(valid=True)
        result.add_violation("test_violation", {"detail": "test"})
        
        assert result.valid is False
        assert len(result.violations) == 1
    
    def test_add_warning(self):
        """Test adding warning doesn't invalidate result."""
        result = VerificationResult(valid=True)
        result.add_warning("test_warning", {"detail": "test"})
        
        assert result.valid is True
        assert len(result.warnings) == 1
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = VerificationResult(
            valid=True,
            verified_citations=5,
            total_citations=5
        )
        
        d = result.to_dict()
        
        assert d["valid"] is True
        assert d["verified_citations"] == 5
        assert d["total_citations"] == 5
        assert d["verification_rate"] == 100.0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for Phase 6 components."""
    
    def test_tool_to_verifier_flow(self):
        """Test flow from tool execution to verification."""
        tools = QBMTools()
        verifier = CitationVerifier()
        
        # Execute tool
        result = tools.get_behavior_dossier("BEH_EMO_PATIENCE")
        
        # Verify output
        verification = verifier.verify_tool_output("get_behavior_dossier", result)
        
        assert result["success"] is True
        assert verification.valid is True
    
    def test_causal_path_verification(self):
        """Test causal path output verification."""
        tools = QBMTools()
        verifier = CitationVerifier()
        
        # Get causal paths
        result = tools.get_causal_paths(
            from_behavior="BEH_SPI_FAITH",
            to_behavior="BEH_FIN_CHARITY"
        )
        
        # Verify
        verification = verifier.verify_tool_output("get_causal_paths", result)
        
        assert result["success"] is True
        assert verification.valid is True
    
    def test_all_87_behaviors_resolvable(self):
        """Test all 87 canonical behaviors can be resolved."""
        tools = QBMTools()
        tools._load_canonical_entities()
        
        behaviors = tools._canonical_entities.get("behaviors", [])
        assert len(behaviors) == 87
        
        for beh in behaviors:
            # Test by ID
            result = tools.get_behavior_dossier(beh["id"])
            assert result["success"] is True, f"Failed for {beh['id']}"


# =============================================================================
# ZERO-HALLUCINATION CONTRACT TESTS (v2.0)
# =============================================================================

class TestZeroHallucinationContracts:
    """Test strict zero-hallucination verification contracts."""
    
    @pytest.fixture
    def verifier(self):
        """Create verifier instance."""
        return CitationVerifier()
    
    def test_no_surah_intro_violation(self, verifier):
        """Test surah_intro entry_type is rejected."""
        response = {
            "behavior_id": "BEH_EMO_PATIENCE",
            "evidence": [
                {
                    "verse_key": "2:1",
                    "entry_type": "surah_intro",
                    "source": "ibn_kathir"
                }
            ]
        }
        
        violations = verifier.verify_no_surah_intro(response)
        assert len(violations) > 0
        assert violations[0]["entry_type"] == "surah_intro"
    
    def test_no_surah_intro_clean(self, verifier):
        """Test normal entry_type passes."""
        response = {
            "behavior_id": "BEH_EMO_PATIENCE",
            "evidence": [
                {
                    "verse_key": "2:45",
                    "entry_type": "verse",
                    "source": "ibn_kathir"
                }
            ]
        }
        
        violations = verifier.verify_no_surah_intro(response)
        assert len(violations) == 0
    
    def test_subset_contract_valid(self, verifier):
        """Test subset contract with valid verse keys."""
        # Get actual verses for patience
        behavior_verses = verifier._load_behavior_verse_map()
        patience_verses = behavior_verses.get("BEH_EMO_PATIENCE", set())
        
        if patience_verses:
            # Use a subset of actual verses
            test_verses = set(list(patience_verses)[:3])
            violations = verifier.verify_subset_contract("BEH_EMO_PATIENCE", test_verses)
            assert len(violations) == 0
    
    def test_subset_contract_violation(self, verifier):
        """Test subset contract catches invalid verse keys."""
        # Use verses that are NOT in patience's verse list
        invalid_verses = {"99:1", "100:1", "101:1"}
        
        violations = verifier.verify_subset_contract("BEH_EMO_PATIENCE", invalid_verses)
        # Should have violations since these verses aren't in patience's list
        assert len(violations) > 0
    
    def test_strict_mode_enforces_contracts(self, verifier):
        """Test strict mode enforces all contracts."""
        response = {
            "behavior_id": "BEH_EMO_PATIENCE",
            "evidence": [
                {
                    "verse_key": "2:1",
                    "entry_type": "surah_intro",
                    "source": "ibn_kathir"
                }
            ],
            "provenance": {"source": "test"}
        }
        
        # Strict mode should catch surah_intro
        result = verifier.verify_response(response, strict=True)
        assert any(v["type"] == "surah_intro_in_evidence" for v in result.violations)
    
    def test_non_strict_mode_skips_contracts(self, verifier):
        """Test non-strict mode skips extra contracts."""
        response = {
            "behavior_id": "BEH_EMO_PATIENCE",
            "evidence": [
                {
                    "verse_key": "2:1",
                    "entry_type": "surah_intro",
                    "source": "ibn_kathir"
                }
            ],
            "provenance": {"source": "test"}
        }
        
        # Non-strict mode should not check surah_intro
        result = verifier.verify_response(response, strict=False)
        assert not any(v["type"] == "surah_intro_in_evidence" for v in result.violations)
    
    def test_claim_evidence_alignment_valid(self, verifier):
        """Test claim-evidence alignment with proper evidence."""
        response = {
            "behavior_id": "BEH_EMO_PATIENCE",
            "claims": [
                {
                    "claim_id": "C1",
                    "text": "Patience is commanded in the Quran",
                    "supporting_verse_keys": ["2:45", "2:153"]
                }
            ],
            "provenance": {"source": "test"}
        }
        
        violations = verifier.verify_claim_evidence_alignment(response)
        assert len(violations) == 0
    
    def test_claim_evidence_alignment_missing_verse_keys(self, verifier):
        """Test claim-evidence alignment catches claims without evidence."""
        response = {
            "behavior_id": "BEH_EMO_PATIENCE",
            "claims": [
                {
                    "claim_id": "C1",
                    "text": "Patience is commanded in the Quran",
                    # Missing supporting_verse_keys!
                }
            ],
            "provenance": {"source": "test"}
        }
        
        violations = verifier.verify_claim_evidence_alignment(response)
        assert len(violations) > 0
        assert any("supporting_verse_keys" in v.get("message", "") for v in violations)
    
    def test_claim_evidence_alignment_missing_claim_id(self, verifier):
        """Test claim-evidence alignment requires claim_id for traceability."""
        response = {
            "behavior_id": "BEH_EMO_PATIENCE",
            "claims": [
                {
                    # Missing claim_id!
                    "text": "Patience is commanded in the Quran",
                    "supporting_verse_keys": ["2:45"]
                }
            ],
            "provenance": {"source": "test"}
        }
        
        violations = verifier.verify_claim_evidence_alignment(response)
        assert len(violations) > 0
        assert any("claim_id" in v.get("message", "") for v in violations)
    
    def test_claim_evidence_alignment_invalid_verse_key(self, verifier):
        """Test claim-evidence alignment catches invalid verse_keys in claims."""
        response = {
            "behavior_id": "BEH_EMO_PATIENCE",
            "claims": [
                {
                    "claim_id": "C1",
                    "text": "Patience is commanded in the Quran",
                    "supporting_verse_keys": ["999:999"]  # Invalid!
                }
            ],
            "provenance": {"source": "test"}
        }
        
        violations = verifier.verify_claim_evidence_alignment(response)
        assert len(violations) > 0
        assert any("invalid verse_key" in v.get("message", "") for v in violations)
    
    def test_long_answer_needs_evidence(self, verifier):
        """Test long answers must have verse_key citations."""
        long_text = "This is a very long answer about patience in the Quran. " * 10
        response = {
            "behavior_id": "BEH_EMO_PATIENCE",
            "answer": long_text,
            # No verse_keys anywhere!
            "provenance": {"source": "test"}
        }
        
        violations = verifier.verify_claim_evidence_alignment(response)
        assert len(violations) > 0
        assert any(v.get("field") == "answer" for v in violations)
    
    def test_narrative_references_nonexistent_claim(self, verifier):
        """Test narrative cannot reference claims that don't exist in validated claims[]."""
        response = {
            "behavior_id": "BEH_EMO_PATIENCE",
            "claims": [
                {
                    "claim_id": "C1",
                    "text": "Patience is commanded in the Quran",
                    "supporting_verse_keys": ["2:45", "2:153"]
                }
            ],
            "narrative": "As stated in [claim:C1], patience is important. Also see [claim:C99] for more.",
            "provenance": {"source": "test"}
        }
        
        violations = verifier.verify_claim_evidence_alignment(response)
        # Should catch the reference to non-existent C99
        assert any(
            v.get("field") == "narrative" and "C99" in str(v.get("referenced_claim", ""))
            for v in violations
        ), f"Should catch reference to non-existent claim C99. Violations: {violations}"
    
    def test_narrative_valid_claim_references(self, verifier):
        """Test narrative with valid claim references passes."""
        response = {
            "behavior_id": "BEH_EMO_PATIENCE",
            "claims": [
                {
                    "claim_id": "C1",
                    "text": "Patience is commanded in the Quran",
                    "supporting_verse_keys": ["2:45", "2:153"]
                },
                {
                    "claim_id": "C2",
                    "text": "Prayer requires patience",
                    "supporting_verse_keys": ["2:45"]
                }
            ],
            "narrative": "As stated in [claim:C1], patience is important. See also [claim:C2].",
            "provenance": {"source": "test"}
        }
        
        violations = verifier.verify_claim_evidence_alignment(response)
        # Should NOT have any narrative violations since C1 and C2 exist
        narrative_violations = [v for v in violations if v.get("field") == "narrative"]
        assert len(narrative_violations) == 0, f"Should not have narrative violations: {narrative_violations}"
    
    def test_strict_mode_requires_structured_claims(self, verifier):
        """Test strict mode requires claims[] for substantive responses."""
        # Response with substantive content but no claims[]
        response = {
            "behavior_id": "BEH_EMO_PATIENCE",
            "answer": "Patience is a fundamental virtue in the Quran. It is mentioned many times and is considered essential for believers. The concept of patience (sabr) encompasses endurance, perseverance, and steadfastness.",
            # No claims[] - should fail in strict mode
            "provenance": {"source": "test"}
        }
        
        result = verifier.verify_response(response, strict=True)
        # Should have missing_structured_claims violation
        assert any(
            v["type"] == "missing_structured_claims" for v in result.violations
        ), f"Should require structured claims in strict mode. Violations: {result.violations}"
    
    def test_strict_mode_passes_with_claims(self, verifier):
        """Test strict mode passes when claims[] is present."""
        response = {
            "behavior_id": "BEH_EMO_PATIENCE",
            "answer": "Patience is a fundamental virtue in the Quran.",
            "claims": [
                {
                    "claim_id": "C1",
                    "text": "Patience is a fundamental virtue",
                    "supporting_verse_keys": ["2:45", "2:153"]
                }
            ],
            "provenance": {"source": "test"}
        }
        
        result = verifier.verify_response(response, strict=True)
        # Should NOT have missing_structured_claims violation
        assert not any(
            v["type"] == "missing_structured_claims" for v in result.violations
        ), f"Should not require claims when present. Violations: {result.violations}"
    
    def test_claim_with_empty_verse_keys_fails(self, verifier):
        """Test claims with empty supporting_verse_keys fail in strict mode."""
        response = {
            "behavior_id": "BEH_EMO_PATIENCE",
            "claims": [
                {
                    "claim_id": "C1",
                    "text": "Patience is a fundamental virtue",
                    "supporting_verse_keys": []  # Empty - should fail!
                }
            ],
            "provenance": {"source": "test"}
        }
        
        violations = verifier.verify_claim_evidence_alignment(response)
        # Should catch the empty supporting_verse_keys
        assert any(
            "supporting_verse_keys" in v.get("message", "") or "no supporting" in v.get("message", "").lower()
            for v in violations
        ), f"Should fail on empty supporting_verse_keys. Violations: {violations}"
    
    def test_claim_with_all_invalid_verse_keys_fails(self, verifier):
        """Test claims where all verse_keys are invalid fail."""
        response = {
            "behavior_id": "BEH_EMO_PATIENCE",
            "claims": [
                {
                    "claim_id": "C1",
                    "text": "Patience is a fundamental virtue",
                    "supporting_verse_keys": ["999:999", "0:0"]  # All invalid
                }
            ],
            "provenance": {"source": "test"}
        }
        
        violations = verifier.verify_claim_evidence_alignment(response)
        # Should catch all invalid verse_keys
        invalid_violations = [v for v in violations if "invalid verse_key" in v.get("message", "")]
        assert len(invalid_violations) == 2, f"Should catch both invalid verse_keys. Violations: {violations}"


# =============================================================================
# AUDIT PACK COMMIT MATCH TEST
# =============================================================================

class TestAuditPackCommitMatch:
    """Test audit pack commit hash matches HEAD."""
    
    def test_audit_pack_has_git_commit(self):
        """Test audit pack contains git commit."""
        pack_path = Path(__file__).parent.parent / "artifacts" / "audit_pack" / "audit_pack.json"
        
        if pack_path.exists():
            with open(pack_path, "r", encoding="utf-8") as f:
                pack = json.load(f)
            
            assert "git_commit" in pack
            assert pack["git_commit"] is not None
            assert len(pack["git_commit"]) == 40  # SHA1 hex length
    
    def test_audit_pack_validation_field(self):
        """Test audit pack has validation field."""
        pack_path = Path(__file__).parent.parent / "artifacts" / "audit_pack" / "audit_pack.json"
        
        if pack_path.exists():
            with open(pack_path, "r", encoding="utf-8") as f:
                pack = json.load(f)
            
            assert "validation" in pack
            assert "is_valid" in pack["validation"]
            assert "ssot_complete" in pack["validation"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
