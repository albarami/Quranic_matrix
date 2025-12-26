"""
Test: Proof Response Schema Contract v1 (Phase 8.1)

Ensures all proof responses conform to the stable schema contract.
"""

import pytest
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "schemas"))

from schemas.proof_response_v1 import (
    ProofResponseV1,
    ProofBundle,
    DebugInfo,
    ValidationResult,
    TafsirQuote,
    GraphEdge,
    validate_proof_response,
)


@pytest.mark.unit
class TestSchemaStructure:
    """Tests for schema structure and required fields."""
    
    def test_proof_response_has_required_keys(self):
        """ProofResponseV1 must have answer, proof, debug."""
        response = ProofResponseV1(
            answer="Test answer",
            query="Test query",
            debug=DebugInfo(query_intent="FREE_TEXT")
        )
        
        assert hasattr(response, "answer")
        assert hasattr(response, "proof")
        assert hasattr(response, "debug")
    
    def test_proof_bundle_has_all_sections(self):
        """ProofBundle must have quran, tafsir, graph, taxonomy."""
        bundle = ProofBundle()
        
        assert hasattr(bundle, "quran")
        assert hasattr(bundle, "tafsir")
        assert hasattr(bundle, "graph")
        assert hasattr(bundle, "taxonomy")
    
    def test_debug_info_has_required_fields(self):
        """DebugInfo must have query_intent and sources_covered."""
        debug = DebugInfo(query_intent="CONCEPT_REF")
        
        assert hasattr(debug, "query_intent")
        assert hasattr(debug, "question_class")
        assert hasattr(debug, "plan_steps")
        assert hasattr(debug, "sources_covered")
        assert hasattr(debug, "fallback_used")
    
    def test_schema_version_present(self):
        """Response must include schema_version."""
        response = ProofResponseV1(
            answer="Test",
            query="Test",
            debug=DebugInfo(query_intent="FREE_TEXT")
        )
        
        assert response.schema_version == "1.0"


@pytest.mark.unit
class TestTafsirProvenance:
    """Tests for tafsir quote provenance (I1 compliance)."""
    
    def test_tafsir_quote_has_all_provenance_fields(self):
        """TafsirQuote must have source, verse_key, chunk_id, char_start, char_end, quote."""
        quote = TafsirQuote(
            source="ibn_kathir",
            verse_key="2:102",
            chunk_id="ibn_kathir_002_102_chunk00",
            char_start=0,
            char_end=100,
            quote="Test quote text"
        )
        
        assert quote.source == "ibn_kathir"
        assert quote.verse_key == "2:102"
        assert quote.chunk_id == "ibn_kathir_002_102_chunk00"
        assert quote.char_start == 0
        assert quote.char_end == 100
        assert quote.quote == "Test quote text"
    
    def test_tafsir_quote_char_end_greater_than_start(self):
        """char_end must be greater than char_start."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            TafsirQuote(
                source="ibn_kathir",
                verse_key="2:102",
                chunk_id="test",
                char_start=100,
                char_end=0,  # Invalid: must be > 0
                quote="Test"
            )
    
    def test_tafsir_quote_requires_non_empty_quote(self):
        """quote field must not be empty."""
        with pytest.raises(ValueError):
            TafsirQuote(
                source="ibn_kathir",
                verse_key="2:102",
                chunk_id="test",
                char_start=0,
                char_end=10,
                quote=""  # Invalid: empty
            )


@pytest.mark.unit
class TestGraphEvidence:
    """Tests for graph edge evidence requirements."""
    
    def test_graph_edge_has_evidence_field(self):
        """GraphEdge must have evidence list."""
        edge = GraphEdge(
            source="BEH_EMO_ENVY",
            target="CSQ_PUNISHMENT",
            edge_type="CAUSES",
            confidence=0.8,
            evidence_count=2,
            evidence=[]
        )
        
        assert hasattr(edge, "evidence")
        assert isinstance(edge.evidence, list)
    
    def test_graph_edge_confidence_bounds(self):
        """Confidence must be between 0 and 1."""
        edge = GraphEdge(
            source="A",
            target="B",
            edge_type="CAUSES",
            confidence=0.5,
            evidence_count=1
        )
        assert 0 <= edge.confidence <= 1
        
        with pytest.raises(ValueError):
            GraphEdge(
                source="A",
                target="B",
                edge_type="CAUSES",
                confidence=1.5,  # Invalid: > 1
                evidence_count=1
            )


@pytest.mark.unit
class TestValidation:
    """Tests for response validation."""
    
    def test_validate_complete_response(self):
        """Complete response should pass validation."""
        response = {
            "answer": "Test answer",
            "proof": {
                "quran": {"verses": [], "status": "found"},
                "tafsir": {"total_quotes": 0, "sources_covered": []},
                "graph": {"nodes": [], "edges": [], "paths": []},
                "taxonomy": {"behaviors": [], "status": "found"}
            },
            "debug": {
                "query_intent": "CONCEPT_REF",
                "sources_covered": ["ibn_kathir"]
            }
        }
        
        result = validate_proof_response(response)
        assert len(result.issues) == 0
    
    def test_validate_missing_answer(self):
        """Missing answer should fail validation."""
        response = {
            "proof": {},
            "debug": {"query_intent": "FREE_TEXT"}
        }
        
        result = validate_proof_response(response)
        assert "Missing required key: answer" in result.issues
    
    def test_validate_missing_proof(self):
        """Missing proof should fail validation."""
        response = {
            "answer": "Test",
            "debug": {"query_intent": "FREE_TEXT"}
        }
        
        result = validate_proof_response(response)
        assert "Missing required key: proof" in result.issues
    
    def test_validate_missing_debug(self):
        """Missing debug should fail validation."""
        response = {
            "answer": "Test",
            "proof": {}
        }
        
        result = validate_proof_response(response)
        assert "Missing required key: debug" in result.issues


@pytest.mark.unit
class TestNoFabrication:
    """Tests for no-fabrication guarantees."""
    
    def test_status_no_evidence_when_empty(self):
        """Empty evidence should have status='no_evidence'."""
        from schemas.proof_response_v1 import QuranEvidence, TaxonomyEvidence
        
        quran = QuranEvidence(verses=[], verse_count=0, status="no_evidence")
        assert quran.status == "no_evidence"
        
        taxonomy = TaxonomyEvidence(behaviors=[], status="no_evidence")
        assert taxonomy.status == "no_evidence"
    
    def test_validation_result_tracks_fabrication(self):
        """ValidationResult must track no_fabrication flag."""
        result = ValidationResult(
            evidence_complete=True,
            provenance_valid=True,
            no_fabrication=True,
            graph_rules_followed=True
        )
        
        assert result.no_fabrication == True


@pytest.mark.unit
class TestSerialization:
    """Tests for JSON serialization."""
    
    def test_response_serializes_to_json(self):
        """ProofResponseV1 must serialize to valid JSON."""
        response = ProofResponseV1(
            answer="Test answer",
            query="Test query",
            debug=DebugInfo(query_intent="FREE_TEXT")
        )
        
        json_str = response.model_dump_json()
        parsed = json.loads(json_str)
        
        assert "answer" in parsed
        assert "proof" in parsed
        assert "debug" in parsed
        assert "schema_version" in parsed
    
    def test_response_roundtrip(self):
        """Response should survive JSON roundtrip."""
        original = ProofResponseV1(
            answer="الحسد هو تمني زوال النعمة",
            query="ما هو الحسد؟",
            debug=DebugInfo(
                query_intent="CONCEPT_REF",
                question_class="behavior_profile_11axis",
                sources_covered=["ibn_kathir", "tabari"]
            )
        )
        
        json_str = original.model_dump_json()
        parsed = json.loads(json_str)
        restored = ProofResponseV1(**parsed)
        
        assert restored.answer == original.answer
        assert restored.debug.query_intent == original.debug.query_intent
        assert restored.debug.question_class == original.debug.question_class


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
