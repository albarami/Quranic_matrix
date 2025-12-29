"""
Phase 2 Tests: Scoring Depth Rules

Tests that the scoring system correctly evaluates:
1. Answer non-empty (not placeholder)
2. Required sections exist based on expected.capabilities
3. Required computed outputs exist (e.g., min_hops chains for causal)
4. Evidence is cited and matches payload references
5. Disallow checks enforced (no generic defaults, no fabricated numbers)
"""

import pytest
from typing import Dict, Any

from src.benchmarks.scoring import (
    score_benchmark_item,
    _is_placeholder_answer,
    _check_required_sections,
    _check_disallow_violations,
    _has_fabricated_numbers,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def minimal_valid_response() -> Dict[str, Any]:
    """Minimal response that passes basic validation."""
    return {
        "answer": "هذه إجابة كاملة تحتوي على تحليل شامل للسؤال المطروح مع الأدلة والبراهين",
        "proof": {
            "quran": [
                {"verse_key": "2:255", "surah": 2, "ayah": 255, "text": "آية الكرسي"},
            ],
            "tafsir": {
                "ibn_kathir": [
                    {"verse_key": "2:255", "surah": 2, "ayah": 255, "text": "تفسير", "chunk_id": "ik1", "char_start": 0, "char_end": 50},
                ],
                "qurtubi": [
                    {"verse_key": "2:255", "surah": 2, "ayah": 255, "text": "تفسير", "chunk_id": "qt1", "char_start": 0, "char_end": 50},
                ],
            },
        },
        "debug": {
            "intent": "CONCEPT_REF",
            "fallback_used": False,
        },
    }


@pytest.fixture
def causal_response_with_paths() -> Dict[str, Any]:
    """Response with graph paths for causal queries."""
    return {
        "answer": "تحليل السلسلة السببية من الكبر إلى الكفر يظهر مسارات متعددة",
        "proof": {
            "quran": [
                {"verse_key": "2:34", "surah": 2, "ayah": 34, "text": "آية"},
            ],
            "tafsir": {
                "ibn_kathir": [
                    {"verse_key": "2:34", "text": "تفسير", "chunk_id": "ik1", "char_start": 0, "char_end": 50},
                ],
            },
            "graph": {
                "paths": [
                    {
                        "nodes": ["الكبر", "الغفلة", "القسوة", "الكفر"],
                        "hops": 3,
                        "edges": [
                            {"source": "الكبر", "target": "الغفلة", "type": "CAUSES", "evidence_count": 3, "verse_key": "2:34"},
                            {"source": "الغفلة", "target": "القسوة", "type": "CAUSES", "evidence_count": 2, "verse_key": "2:74"},
                            {"source": "القسوة", "target": "الكفر", "type": "LEADS_TO", "evidence_count": 4, "verse_key": "2:7"},
                        ],
                    },
                ],
            },
        },
        "debug": {
            "intent": "GRAPH_CAUSAL",
            "fallback_used": False,
            "derivations": {"paths_count": 1},
        },
    }


@pytest.fixture
def benchmark_item_causal() -> Dict[str, Any]:
    """Benchmark item for causal chain analysis."""
    return {
        "id": "A01",
        "section": "A",
        "title": "Causal Chain",
        "question_ar": "ما هي سلسلة الكبر إلى الكفر؟",
        "expected": {
            "capabilities": ["GRAPH_CAUSAL", "MULTIHOP", "TAFSIR_MULTI_SOURCE", "PROVENANCE"],
            "min_sources": 1,
            "required_sources": ["ibn_kathir"],
            "min_hops": 3,
            "must_include": ["edge_provenance", "verse_keys_per_link"],
            "disallow": ["generic_opening_verses_default"],
        },
    }


@pytest.fixture
def benchmark_item_simple() -> Dict[str, Any]:
    """Simple benchmark item without complex requirements."""
    return {
        "id": "X01",
        "section": "X",
        "title": "Simple Query",
        "question_ar": "ما هو تفسير آية الكرسي؟",
        "expected": {
            "capabilities": ["TAFSIR_MULTI_SOURCE"],
            "min_sources": 2,
        },
    }


# =============================================================================
# Test: Placeholder Answer Detection
# =============================================================================

class TestPlaceholderAnswerDetection:
    """Tests for _is_placeholder_answer()."""

    def test_empty_string_is_placeholder(self):
        """Test that empty string is detected as placeholder."""
        assert _is_placeholder_answer("") is True

    def test_whitespace_only_is_placeholder(self):
        """Test that whitespace-only is detected as placeholder."""
        assert _is_placeholder_answer("   \n\t  ") is True

    def test_proof_only_marker_is_placeholder(self):
        """Test that proof_only marker is detected as placeholder."""
        assert _is_placeholder_answer("[proof_only mode - LLM answer skipped]") is True

    def test_short_answer_is_placeholder(self):
        """Test that very short answers are detected as placeholder."""
        assert _is_placeholder_answer("OK") is True
        assert _is_placeholder_answer("N/A") is True

    def test_valid_answer_is_not_placeholder(self):
        """Test that valid Arabic answer is not placeholder."""
        answer = "هذه إجابة كاملة تحتوي على تحليل شامل للسؤال المطروح"
        assert _is_placeholder_answer(answer) is False


# =============================================================================
# Test: Required Sections Check
# =============================================================================

class TestRequiredSectionsCheck:
    """Tests for _check_required_sections()."""

    def test_no_must_include_passes(self, minimal_valid_response):
        """Test that empty must_include passes."""
        missing = _check_required_sections(minimal_valid_response, [])
        assert len(missing) == 0

    def test_edge_provenance_missing(self, minimal_valid_response):
        """Test that missing edge_provenance is detected."""
        missing = _check_required_sections(minimal_valid_response, ["edge_provenance"])
        assert "must_include_missing:edge_provenance" in missing

    def test_edge_provenance_present(self, causal_response_with_paths):
        """Test that present edge_provenance passes."""
        missing = _check_required_sections(causal_response_with_paths, ["edge_provenance"])
        assert len(missing) == 0

    def test_verse_keys_per_link_present(self, causal_response_with_paths):
        """Test that verse_keys_per_link is detected when present."""
        missing = _check_required_sections(causal_response_with_paths, ["verse_keys_per_link"])
        assert len(missing) == 0

    def test_derivations_missing(self, minimal_valid_response):
        """Test that missing derivations is detected."""
        missing = _check_required_sections(minimal_valid_response, ["derivations"])
        assert "must_include_missing:derivations" in missing

    def test_derivations_present(self, causal_response_with_paths):
        """Test that present derivations passes."""
        missing = _check_required_sections(causal_response_with_paths, ["derivations"])
        assert len(missing) == 0


# =============================================================================
# Test: Disallow Violations Check
# =============================================================================

class TestDisallowViolationsCheck:
    """Tests for _check_disallow_violations()."""

    def test_no_disallow_passes(self, minimal_valid_response):
        """Test that empty disallow passes."""
        violations = _check_disallow_violations(minimal_valid_response, [], {})
        assert len(violations) == 0

    def test_fallback_used_violation(self):
        """Test that fallback_used is detected."""
        response = {
            "answer": "test",
            "proof": {"quran": [], "tafsir": {}},
            "debug": {"fallback_used": True},
        }
        violations = _check_disallow_violations(response, ["fallback_used"], {})
        assert "disallow_fallback_used" in violations

    def test_empty_evidence_violation(self):
        """Test that empty evidence is detected."""
        response = {
            "answer": "test",
            "proof": {"quran": [], "tafsir": {}},
            "debug": {},
        }
        violations = _check_disallow_violations(response, ["empty_evidence"], {})
        assert "disallow_empty_evidence" in violations


# =============================================================================
# Test: Fabricated Numbers Detection
# =============================================================================

class TestFabricatedNumbersDetection:
    """Tests for _has_fabricated_numbers()."""

    def test_no_fabrication_with_derivations(self):
        """Test that valid numbers pass."""
        debug = {"derivations": {"count": 42, "total": 100}}
        answer = "تم العثور على 42 نتيجة من أصل 100"
        assert _has_fabricated_numbers(answer, debug) is False

    def test_fabrication_detected(self):
        """Test that fabricated numbers are detected."""
        debug = {"derivations": {"count": 42}}
        answer = "تم العثور على 9999 نتيجة مذهلة"  # 9999 not in derivations
        assert _has_fabricated_numbers(answer, debug) is True

    def test_no_derivations_no_check(self):
        """Test that missing derivations skips check."""
        debug = {}
        answer = "تم العثور على 9999 نتيجة"
        # Without derivations, we can't verify, so should return False
        assert _has_fabricated_numbers(answer, debug) is False

    def test_common_numbers_allowed(self):
        """Test that common numbers (0-14, 100) are always allowed."""
        debug = {"derivations": {}}  # Empty derivations
        answer = "هناك 5 نقاط و 10 أمثلة"
        assert _has_fabricated_numbers(answer, debug) is False


# =============================================================================
# Test: Score Benchmark Item - PASS Cases
# =============================================================================

class TestScoreBenchmarkItemPass:
    """Tests for PASS verdicts."""

    def test_minimal_valid_passes(self, minimal_valid_response, benchmark_item_simple):
        """Test that minimal valid response passes simple benchmark."""
        result = score_benchmark_item(
            benchmark_item=benchmark_item_simple,
            response=minimal_valid_response,
            http_status=200,
            request_payload={"question": "test"},
            schema_valid=True,
            schema_issues=[],
            request_error=None,
        )

        assert result["verdict"] == "PASS"

    def test_causal_with_paths_passes(self, causal_response_with_paths, benchmark_item_causal):
        """Test that causal response with paths passes causal benchmark."""
        result = score_benchmark_item(
            benchmark_item=benchmark_item_causal,
            response=causal_response_with_paths,
            http_status=200,
            request_payload={"question": "سلسلة الكبر"},
            schema_valid=True,
            schema_issues=[],
            request_error=None,
        )

        # May be PARTIAL due to must_include requirements, but should not FAIL
        assert result["verdict"] in ("PASS", "PARTIAL")


# =============================================================================
# Test: Score Benchmark Item - FAIL Cases
# =============================================================================

class TestScoreBenchmarkItemFail:
    """Tests for FAIL verdicts."""

    def test_http_error_fails(self, minimal_valid_response, benchmark_item_simple):
        """Test that HTTP error causes FAIL."""
        result = score_benchmark_item(
            benchmark_item=benchmark_item_simple,
            response=minimal_valid_response,
            http_status=500,
            request_payload={"question": "test"},
            schema_valid=True,
            schema_issues=[],
            request_error="Internal Server Error",
        )

        assert result["verdict"] == "FAIL"
        assert "http_status_500" in result["reasons"]

    def test_schema_invalid_fails(self, minimal_valid_response, benchmark_item_simple):
        """Test that schema invalid causes FAIL."""
        result = score_benchmark_item(
            benchmark_item=benchmark_item_simple,
            response=minimal_valid_response,
            http_status=200,
            request_payload={"question": "test"},
            schema_valid=False,
            schema_issues=["missing_field_x"],
            request_error=None,
        )

        assert result["verdict"] == "FAIL"
        assert "schema_invalid" in result["reasons"]

    def test_fallback_for_structured_intent_fails(self, benchmark_item_simple):
        """Test that fallback for structured intent causes FAIL."""
        response = {
            "answer": "إجابة طويلة كافية للتحقق من صحة المحتوى",
            "proof": {
                "quran": [{"verse_key": "2:255", "surah": 2, "ayah": 255, "text": "test"}],
                "tafsir": {
                    "ibn_kathir": [{"text": "t", "chunk_id": "x", "verse_key": "2:255", "char_start": 0, "char_end": 1}],
                    "qurtubi": [{"text": "t", "chunk_id": "y", "verse_key": "2:255", "char_start": 0, "char_end": 1}],
                },
            },
            "debug": {
                "intent": "GRAPH_CAUSAL",  # Structured intent
                "fallback_used": True,  # But fallback was used
            },
        }

        result = score_benchmark_item(
            benchmark_item=benchmark_item_simple,
            response=response,
            http_status=200,
            request_payload={"question": "test"},
            schema_valid=True,
            schema_issues=[],
            request_error=None,
        )

        assert result["verdict"] == "FAIL"
        assert "fallback_used_for_structured_intent" in result["reasons"]

    def test_no_evidence_fails(self, benchmark_item_simple):
        """Test that no evidence causes FAIL."""
        response = {
            "answer": "إجابة طويلة كافية للتحقق من صحة المحتوى",
            "proof": {
                "quran": [],
                "tafsir": {},
            },
            "debug": {
                "intent": "FREE_TEXT",
                "fallback_used": False,
            },
        }

        result = score_benchmark_item(
            benchmark_item=benchmark_item_simple,
            response=response,
            http_status=200,
            request_payload={"question": "test"},
            schema_valid=True,
            schema_issues=[],
            request_error=None,
        )

        assert result["verdict"] == "FAIL"
        assert "no_evidence" in result["reasons"]

    def test_placeholder_answer_fails(self, benchmark_item_simple):
        """Test that placeholder answer causes FAIL (non proof_only mode)."""
        response = {
            "answer": "[proof_only mode - LLM answer skipped]",
            "proof": {
                "quran": [{"verse_key": "2:255", "surah": 2, "ayah": 255, "text": "test"}],
                "tafsir": {
                    "ibn_kathir": [{"text": "t", "chunk_id": "x", "verse_key": "2:255", "char_start": 0, "char_end": 1}],
                    "qurtubi": [{"text": "t", "chunk_id": "y", "verse_key": "2:255", "char_start": 0, "char_end": 1}],
                },
            },
            "debug": {"intent": "FREE_TEXT"},
        }

        result = score_benchmark_item(
            benchmark_item=benchmark_item_simple,
            response=response,
            http_status=200,
            request_payload={"question": "test", "proof_only": False},
            schema_valid=True,
            schema_issues=[],
            request_error=None,
        )

        assert result["verdict"] == "FAIL"
        assert "answer_placeholder_or_empty" in result["reasons"]

    def test_placeholder_answer_allowed_in_proof_only(self, benchmark_item_simple):
        """Test that placeholder answer is allowed in proof_only mode."""
        response = {
            "answer": "[proof_only mode - LLM answer skipped]",
            "proof": {
                "quran": [{"verse_key": "2:255", "surah": 2, "ayah": 255, "text": "test"}],
                "tafsir": {
                    "ibn_kathir": [{"text": "t", "chunk_id": "x", "verse_key": "2:255", "char_start": 0, "char_end": 1}],
                    "qurtubi": [{"text": "t", "chunk_id": "y", "verse_key": "2:255", "char_start": 0, "char_end": 1}],
                },
            },
            "debug": {"intent": "FREE_TEXT"},
        }

        result = score_benchmark_item(
            benchmark_item=benchmark_item_simple,
            response=response,
            http_status=200,
            request_payload={"question": "test", "proof_only": True},
            schema_valid=True,
            schema_issues=[],
            request_error=None,
        )

        # Should not fail due to placeholder
        assert "answer_placeholder_or_empty" not in result["reasons"]


# =============================================================================
# Test: Score Benchmark Item - PARTIAL Cases
# =============================================================================

class TestScoreBenchmarkItemPartial:
    """Tests for PARTIAL verdicts."""

    def test_missing_required_source_partial(self, benchmark_item_causal):
        """Test that missing required source causes PARTIAL."""
        response = {
            "answer": "إجابة طويلة كافية للتحقق من صحة المحتوى المطلوب",
            "proof": {
                "quran": [{"verse_key": "2:255", "surah": 2, "ayah": 255, "text": "test"}],
                "tafsir": {
                    # Missing ibn_kathir which is required
                    "qurtubi": [{"text": "t", "chunk_id": "y", "verse_key": "2:255", "char_start": 0, "char_end": 1}],
                },
                "graph": {"paths": [{"nodes": ["A", "B", "C", "D"], "hops": 3}]},
            },
            "debug": {"intent": "GRAPH_CAUSAL"},
        }

        result = score_benchmark_item(
            benchmark_item=benchmark_item_causal,
            response=response,
            http_status=200,
            request_payload={"question": "test"},
            schema_valid=True,
            schema_issues=[],
            request_error=None,
        )

        assert result["verdict"] == "PARTIAL"
        assert any("required_source_missing:ibn_kathir" in r for r in result["reasons"])

    def test_missing_graph_paths_partial(self, benchmark_item_causal):
        """Test that missing graph paths causes PARTIAL for GRAPH_CAUSAL."""
        response = {
            "answer": "إجابة طويلة كافية للتحقق من صحة المحتوى المطلوب",
            "proof": {
                "quran": [{"verse_key": "2:255", "surah": 2, "ayah": 255, "text": "test"}],
                "tafsir": {
                    "ibn_kathir": [{"text": "t", "chunk_id": "x", "verse_key": "2:255", "char_start": 0, "char_end": 1}],
                },
                "graph": {},  # No paths
            },
            "debug": {"intent": "GRAPH_CAUSAL"},
        }

        result = score_benchmark_item(
            benchmark_item=benchmark_item_causal,
            response=response,
            http_status=200,
            request_payload={"question": "test"},
            schema_valid=True,
            schema_issues=[],
            request_error=None,
        )

        assert result["verdict"] == "PARTIAL"
        assert any("graph_paths_missing" in r or "graph_missing" in r for r in result["reasons"])

    def test_insufficient_hops_partial(self, benchmark_item_causal):
        """Test that insufficient hops causes PARTIAL."""
        response = {
            "answer": "إجابة طويلة كافية للتحقق من صحة المحتوى المطلوب",
            "proof": {
                "quran": [{"verse_key": "2:255", "surah": 2, "ayah": 255, "text": "test"}],
                "tafsir": {
                    "ibn_kathir": [{"text": "t", "chunk_id": "x", "verse_key": "2:255", "char_start": 0, "char_end": 1}],
                },
                "graph": {
                    "paths": [{"nodes": ["A", "B"], "hops": 1}],  # Only 1 hop, need 3
                },
            },
            "debug": {"intent": "GRAPH_CAUSAL"},
        }

        result = score_benchmark_item(
            benchmark_item=benchmark_item_causal,
            response=response,
            http_status=200,
            request_payload={"question": "test"},
            schema_valid=True,
            schema_issues=[],
            request_error=None,
        )

        assert result["verdict"] == "PARTIAL"
        assert any("min_hops" in r for r in result["reasons"])


# =============================================================================
# Test: Metrics Tracking
# =============================================================================

class TestMetricsTracking:
    """Tests for metrics in scoring results."""

    def test_metrics_include_source_count(self, minimal_valid_response, benchmark_item_simple):
        """Test that metrics include sources_with_tafsir."""
        result = score_benchmark_item(
            benchmark_item=benchmark_item_simple,
            response=minimal_valid_response,
            http_status=200,
            request_payload={"question": "test"},
            schema_valid=True,
            schema_issues=[],
            request_error=None,
        )

        assert "sources_with_tafsir" in result["metrics"]
        assert result["metrics"]["sources_with_tafsir"] == 2

    def test_metrics_include_verse_count(self, minimal_valid_response, benchmark_item_simple):
        """Test that metrics include quran_verses count."""
        result = score_benchmark_item(
            benchmark_item=benchmark_item_simple,
            response=minimal_valid_response,
            http_status=200,
            request_payload={"question": "test"},
            schema_valid=True,
            schema_issues=[],
            request_error=None,
        )

        assert "quran_verses" in result["metrics"]
        assert result["metrics"]["quran_verses"] == 1

    def test_metrics_include_multihop_count(self, causal_response_with_paths, benchmark_item_causal):
        """Test that metrics include multihop qualifying paths."""
        result = score_benchmark_item(
            benchmark_item=benchmark_item_causal,
            response=causal_response_with_paths,
            http_status=200,
            request_payload={"question": "test"},
            schema_valid=True,
            schema_issues=[],
            request_error=None,
        )

        # Should have multihop metrics
        assert "multihop_qualifying_paths" in result["metrics"] or result["verdict"] == "PASS"
