"""
Phase 2 Tests: Deterministic Answer Generation

Tests that:
1. Analysis payload builds correctly from proof data
2. Answer generator produces deterministic output
3. Validator gate catches fabricated numbers
4. Numbers in answer match payload numbers
"""

import pytest
from typing import Dict, Any, List

# Import the modules we're testing
from src.benchmarks.analysis_payload import (
    AnalysisPayload,
    AnalysisPayloadBuilder,
    EntityInfo,
    EvidenceBundle,
    GraphOutput,
    ComputedTable,
    build_analysis_payload,
)
from src.benchmarks.answer_generator import (
    generate_answer,
    validate_no_new_claims,
    generate_answer_with_llm_rewrite,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_proof() -> Dict[str, Any]:
    """Sample proof object from MandatoryProofSystem."""
    return {
        "quran": [
            {"verse_key": "2:255", "surah": 2, "ayah": 255, "text": "آية الكرسي", "relevance": 0.95},
            {"verse_key": "3:14", "surah": 3, "ayah": 14, "text": "زين للناس", "relevance": 0.88},
        ],
        "tafsir": {
            "ibn_kathir": [
                {"verse_key": "2:255", "surah": 2, "ayah": 255, "text": "تفسير ابن كثير", "chunk_id": "ik_2_255_1", "char_start": 0, "char_end": 100, "score": 0.9},
            ],
            "qurtubi": [
                {"verse_key": "2:255", "surah": 2, "ayah": 255, "text": "تفسير القرطبي", "chunk_id": "qt_2_255_1", "char_start": 0, "char_end": 120, "score": 0.85},
            ],
        },
        "graph": {
            "paths": [
                {"nodes": ["الكبر", "الغفلة", "الكفر"], "hops": 2, "edges": [
                    {"source": "الكبر", "target": "الغفلة", "type": "CAUSES", "evidence_count": 3},
                    {"source": "الغفلة", "target": "الكفر", "type": "LEADS_TO", "evidence_count": 5},
                ]},
            ],
            "centrality": {"total_nodes": 87, "total_edges": 156},
        },
        "statistics": {
            "counts": {"total_behaviors": 87, "total_edges": 156},
            "percentages": {"coverage": 0.85},
        },
    }


@pytest.fixture
def sample_debug() -> Dict[str, Any]:
    """Sample debug trace from MandatoryProofSystem."""
    return {
        "intent": "GRAPH_CAUSAL",
        "entity_resolution": {
            "entities": [
                {"entity_id": "BEH_KIBR", "entity_type": "behavior", "total_mentions": 45, "verse_keys": ["2:34", "4:173"]},
                {"entity_id": "BEH_KUFR", "entity_type": "behavior", "total_mentions": 120, "verse_keys": ["2:6", "2:7"]},
            ],
        },
        "concept_lookups": [
            {"entity_id": "BEH_KIBR", "status": "found", "mentions": 45},
        ],
        "cross_tafsir_stats": {
            "sources_count": 2,
            "total_sources": 7,
            "source_distribution": {"ibn_kathir": 5, "qurtubi": 3},
            "agreement_ratio": 0.286,
        },
        "derivations": {
            "quran_verse_count": 2,
            "tafsir_chunk_counts": {"ibn_kathir": 1, "qurtubi": 1},
            "payload_numbers": [2, 45, 120, 87, 156, 0.85],
        },
    }


# =============================================================================
# Test: Analysis Payload Builder
# =============================================================================

class TestAnalysisPayloadBuilder:
    """Tests for AnalysisPayloadBuilder."""

    def test_build_payload_from_proof(self, sample_proof, sample_debug):
        """Test that payload builds correctly from proof and debug."""
        payload = build_analysis_payload(
            question="ما هي سلسلة الكبر إلى الكفر؟",
            question_class="causal_chain",
            proof=sample_proof,
            debug=sample_debug,
        )

        assert isinstance(payload, AnalysisPayload)
        assert payload.question_class == "causal_chain"
        assert payload.intent == "GRAPH_CAUSAL"

    def test_entities_extracted(self, sample_proof, sample_debug):
        """Test that entities are extracted from debug.entity_resolution."""
        payload = build_analysis_payload(
            question="الكبر والكفر",
            question_class="causal_chain",
            proof=sample_proof,
            debug=sample_debug,
        )

        assert len(payload.entities) >= 1
        entity_ids = [e.entity_id for e in payload.entities]
        assert "BEH_KIBR" in entity_ids or "BEH_KUFR" in entity_ids

    def test_quran_evidence_extracted(self, sample_proof, sample_debug):
        """Test that Quran evidence bundles are extracted."""
        payload = build_analysis_payload(
            question="آية الكرسي",
            question_class="free_text",
            proof=sample_proof,
            debug=sample_debug,
        )

        assert len(payload.quran_evidence) == 2
        verse_keys = [e.verse_key for e in payload.quran_evidence]
        assert "2:255" in verse_keys

    def test_tafsir_evidence_extracted(self, sample_proof, sample_debug):
        """Test that tafsir evidence bundles are extracted."""
        payload = build_analysis_payload(
            question="تفسير",
            question_class="cross_tafsir_comparative",
            proof=sample_proof,
            debug=sample_debug,
        )

        assert "ibn_kathir" in payload.tafsir_evidence
        assert len(payload.tafsir_evidence["ibn_kathir"]) == 1
        assert payload.tafsir_evidence["ibn_kathir"][0].chunk_id == "ik_2_255_1"

    def test_computed_numbers_populated(self, sample_proof, sample_debug):
        """Test that computed_numbers is populated from payload data."""
        payload = build_analysis_payload(
            question="إحصائيات",
            question_class="free_text",
            proof=sample_proof,
            debug=sample_debug,
        )

        assert "quran_verse_count" in payload.computed_numbers
        assert payload.computed_numbers["quran_verse_count"] == 2
        assert "total_tafsir_chunks" in payload.computed_numbers

    def test_gaps_identified_for_missing_tafsir(self):
        """Test that gaps are identified when tafsir sources are missing."""
        minimal_proof = {
            "quran": [{"verse_key": "1:1", "surah": 1, "ayah": 1, "text": "بسم الله"}],
            "tafsir": {},  # No tafsir
        }
        minimal_debug = {"intent": "FREE_TEXT"}

        payload = build_analysis_payload(
            question="test",
            question_class="free_text",
            proof=minimal_proof,
            debug=minimal_debug,
        )

        # Should have gaps for all 7 missing tafsir sources
        tafsir_gaps = [g for g in payload.gaps if g.startswith("missing_tafsir_")]
        assert len(tafsir_gaps) == 7

    def test_graph_output_from_proof(self, sample_proof, sample_debug):
        """Test that graph output is extracted from proof."""
        payload = build_analysis_payload(
            question="سلسلة سببية",
            question_class="causal_chain",
            proof=sample_proof,
            debug=sample_debug,
        )

        assert len(payload.graph_output.paths) == 1
        assert payload.graph_output.centrality.get("total_nodes") == 87

    def test_get_all_numbers_includes_payload_numbers(self, sample_proof, sample_debug):
        """Test that get_all_numbers() returns all computed numbers."""
        payload = build_analysis_payload(
            question="test",
            question_class="free_text",
            proof=sample_proof,
            debug=sample_debug,
        )

        all_numbers = payload.get_all_numbers()
        assert 2.0 in all_numbers  # quran_verse_count
        assert 87.0 in all_numbers  # total_nodes


# =============================================================================
# Test: Answer Generator
# =============================================================================

class TestAnswerGenerator:
    """Tests for answer generation."""

    def test_generate_answer_produces_output(self, sample_proof, sample_debug):
        """Test that generate_answer produces non-empty output."""
        payload = build_analysis_payload(
            question="تحليل الكبر",
            question_class="behavior_profile_11axis",
            proof=sample_proof,
            debug=sample_debug,
        )

        answer = generate_answer(payload)

        assert answer
        assert len(answer) > 100
        assert "##" in answer  # Has headers

    def test_generate_answer_includes_statistics(self, sample_proof, sample_debug):
        """Test that generated answer includes statistics section."""
        payload = build_analysis_payload(
            question="إحصائيات",
            question_class="free_text",
            proof=sample_proof,
            debug=sample_debug,
        )

        answer = generate_answer(payload)

        assert "الإحصائيات" in answer or "المحسوبة" in answer

    def test_generate_answer_includes_evidence(self, sample_proof, sample_debug):
        """Test that generated answer includes evidence citations."""
        payload = build_analysis_payload(
            question="دليل",
            question_class="free_text",
            proof=sample_proof,
            debug=sample_debug,
        )

        answer = generate_answer(payload, include_evidence=True)

        assert "2:255" in answer or "الأدلة" in answer

    def test_generate_answer_without_evidence(self, sample_proof, sample_debug):
        """Test that evidence can be excluded from answer."""
        payload = build_analysis_payload(
            question="test",
            question_class="free_text",
            proof=sample_proof,
            debug=sample_debug,
        )

        answer = generate_answer(payload, include_evidence=False)

        # Should still have other sections
        assert "##" in answer


# =============================================================================
# Test: Validator Gate
# =============================================================================

class TestValidatorGate:
    """Tests for the validator gate."""

    def test_validate_accepts_payload_numbers(self, sample_proof, sample_debug):
        """Test that validator accepts numbers from payload."""
        payload = build_analysis_payload(
            question="test",
            question_class="free_text",
            proof=sample_proof,
            debug=sample_debug,
        )

        # Use numbers from the payload
        llm_output = "تم العثور على 2 آيات و 87 عقدة في الشبكة"

        is_valid, violations = validate_no_new_claims(payload, llm_output)

        assert is_valid
        assert len(violations) == 0

    def test_validate_rejects_fabricated_numbers(self, sample_proof, sample_debug):
        """Test that validator rejects fabricated numbers."""
        payload = build_analysis_payload(
            question="test",
            question_class="free_text",
            proof=sample_proof,
            debug=sample_debug,
        )

        # Use a number NOT in payload (999 is very unlikely to be in payload)
        llm_output = "تم العثور على 999 نتيجة مذهلة"

        is_valid, violations = validate_no_new_claims(payload, llm_output)

        assert not is_valid
        assert len(violations) > 0
        assert any("999" in v for v in violations)

    def test_validate_allows_common_numbers(self, sample_proof, sample_debug):
        """Test that validator allows common numbers (0-14, 100)."""
        payload = build_analysis_payload(
            question="test",
            question_class="free_text",
            proof=sample_proof,
            debug=sample_debug,
        )

        llm_output = "هناك 5 نقاط رئيسية و 10 أمثلة و 100% تأكيد"

        is_valid, violations = validate_no_new_claims(payload, llm_output)

        assert is_valid

    def test_validate_allows_verse_references(self, sample_proof, sample_debug):
        """Test that validator allows verse reference numbers."""
        payload = build_analysis_payload(
            question="test",
            question_class="free_text",
            proof=sample_proof,
            debug=sample_debug,
        )

        # Verse references should be allowed
        llm_output = "كما في سورة البقرة 2:255 والآية 34 من سورة آل عمران 3:34"

        is_valid, violations = validate_no_new_claims(payload, llm_output)

        assert is_valid

    def test_validate_empty_output(self, sample_proof, sample_debug):
        """Test that validator accepts empty output."""
        payload = build_analysis_payload(
            question="test",
            question_class="free_text",
            proof=sample_proof,
            debug=sample_debug,
        )

        is_valid, violations = validate_no_new_claims(payload, "")

        assert is_valid
        assert len(violations) == 0


# =============================================================================
# Test: LLM Rewrite with Validation
# =============================================================================

class TestLLMRewriteWithValidation:
    """Tests for LLM rewrite with validator gate."""

    def test_rewrite_without_llm(self, sample_proof, sample_debug):
        """Test that rewrite works without LLM (returns base answer)."""
        payload = build_analysis_payload(
            question="test",
            question_class="free_text",
            proof=sample_proof,
            debug=sample_debug,
        )

        answer, is_valid, violations = generate_answer_with_llm_rewrite(
            payload, llm_rewriter=None
        )

        assert answer
        assert is_valid
        assert len(violations) == 0

    def test_rewrite_with_valid_llm(self, sample_proof, sample_debug):
        """Test that valid LLM rewrite passes validation."""
        payload = build_analysis_payload(
            question="test",
            question_class="free_text",
            proof=sample_proof,
            debug=sample_debug,
        )

        # Mock LLM that returns valid output (uses payload numbers)
        def mock_llm(text: str) -> str:
            return "تم تحليل 2 آيات من القرآن الكريم بنجاح"

        answer, is_valid, violations = generate_answer_with_llm_rewrite(
            payload, llm_rewriter=mock_llm
        )

        assert is_valid
        assert "2" in answer

    def test_rewrite_with_invalid_llm_strict(self, sample_proof, sample_debug):
        """Test that invalid LLM rewrite is rejected in strict mode."""
        payload = build_analysis_payload(
            question="test",
            question_class="free_text",
            proof=sample_proof,
            debug=sample_debug,
        )

        # Mock LLM that fabricates numbers
        def mock_llm_bad(text: str) -> str:
            return "تم تحليل 9999 نتيجة مذهلة جداً"

        answer, is_valid, violations = generate_answer_with_llm_rewrite(
            payload, llm_rewriter=mock_llm_bad, strict_validation=True
        )

        # In strict mode, should fall back to base answer
        assert not is_valid
        assert len(violations) > 0
        # Answer should be the base (deterministic) answer, not the LLM output
        assert "9999" not in answer


# =============================================================================
# Test: Derivations Trace
# =============================================================================

class TestDerivationsTrace:
    """Tests for derivations trace in payload."""

    def test_derivations_include_entity_ids(self, sample_proof, sample_debug):
        """Test that derivations include entity IDs."""
        payload = build_analysis_payload(
            question="test",
            question_class="free_text",
            proof=sample_proof,
            debug=sample_debug,
        )

        assert "entity_ids" in payload.derivations
        assert isinstance(payload.derivations["entity_ids"], list)

    def test_derivations_include_verse_count(self, sample_proof, sample_debug):
        """Test that derivations include verse count."""
        payload = build_analysis_payload(
            question="test",
            question_class="free_text",
            proof=sample_proof,
            debug=sample_debug,
        )

        assert "quran_verse_count" in payload.derivations
        assert payload.derivations["quran_verse_count"] == 2

    def test_derivations_include_tafsir_counts(self, sample_proof, sample_debug):
        """Test that derivations include tafsir chunk counts."""
        payload = build_analysis_payload(
            question="test",
            question_class="free_text",
            proof=sample_proof,
            debug=sample_debug,
        )

        assert "tafsir_chunk_counts" in payload.derivations
        assert "ibn_kathir" in payload.derivations["tafsir_chunk_counts"]

    def test_derivations_include_gaps(self):
        """Test that derivations include gaps."""
        minimal_proof = {"quran": [], "tafsir": {}}
        minimal_debug = {"intent": "FREE_TEXT"}

        payload = build_analysis_payload(
            question="test",
            question_class="free_text",
            proof=minimal_proof,
            debug=minimal_debug,
        )

        assert "gaps" in payload.derivations
        assert len(payload.derivations["gaps"]) > 0


# =============================================================================
# Test: Payload Serialization
# =============================================================================

class TestPayloadSerialization:
    """Tests for payload serialization."""

    def test_to_dict_produces_valid_dict(self, sample_proof, sample_debug):
        """Test that to_dict() produces a valid dictionary."""
        payload = build_analysis_payload(
            question="test",
            question_class="free_text",
            proof=sample_proof,
            debug=sample_debug,
        )

        result = payload.to_dict()

        assert isinstance(result, dict)
        assert "question" in result
        assert "entities" in result
        assert "computed_numbers" in result

    def test_to_dict_is_json_serializable(self, sample_proof, sample_debug):
        """Test that to_dict() output is JSON serializable."""
        import json

        payload = build_analysis_payload(
            question="test",
            question_class="free_text",
            proof=sample_proof,
            debug=sample_debug,
        )

        result = payload.to_dict()

        # Should not raise
        json_str = json.dumps(result, ensure_ascii=False)
        assert json_str
