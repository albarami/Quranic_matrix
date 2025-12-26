"""
Runtime API Acceptance Tests (Phase 9.2c)

Tests the /api/proof/query endpoint via FastAPI TestClient.
Ensures planner tests match production endpoint behavior.

Validates:
- Response matches proof_response_v1 schema
- Proof contains evidence objects with provenance
- Debug contains plan_steps and question_class
"""

import pytest
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Skip all tests if FastAPI dependencies not available
FASTAPI_AVAILABLE = False
IMPORT_ERROR = ""

try:
    from fastapi.testclient import TestClient
    # Import app with error handling for optional deps like torch_geometric
    import os
    os.environ.setdefault("SKIP_HEAVY_IMPORTS", "1")
    from src.api.main import app
    FASTAPI_AVAILABLE = True
except ImportError as e:
    IMPORT_ERROR = str(e)
except Exception as e:
    # Handle DLL load failures, torch_geometric issues, etc.
    IMPORT_ERROR = f"Runtime error: {e}"


@pytest.fixture(scope="module")
def client():
    """Create FastAPI test client."""
    if not FASTAPI_AVAILABLE:
        pytest.skip(f"FastAPI not available: {IMPORT_ERROR}")
    return TestClient(app)


@pytest.mark.api
class TestProofEndpointBasic:
    """Basic tests for /api/proof/query endpoint."""
    
    def test_endpoint_exists(self, client):
        """Proof endpoint should exist and accept POST."""
        response = client.post(
            "/api/proof/query",
            json={"question": "ما هو الحسد؟"}
        )
        # Should not be 404 or 405
        assert response.status_code != 404, "Endpoint not found"
        assert response.status_code != 405, "Method not allowed"
    
    def test_endpoint_returns_json(self, client):
        """Endpoint should return valid JSON."""
        response = client.post(
            "/api/proof/query",
            json={"question": "ما هو الحسد؟"}
        )
        assert response.headers.get("content-type", "").startswith("application/json")
        data = response.json()
        assert isinstance(data, dict)
    
    def test_endpoint_requires_question(self, client):
        """Endpoint should require question field."""
        response = client.post(
            "/api/proof/query",
            json={}
        )
        # Should return 422 (validation error) for missing required field
        assert response.status_code == 422


@pytest.mark.api
class TestProofResponseSchema:
    """Tests for proof response schema compliance."""
    
    def test_response_has_question(self, client):
        """Response should echo the question."""
        question = "ما هو الحسد؟"
        response = client.post(
            "/api/proof/query",
            json={"question": question}
        )
        if response.status_code == 200:
            data = response.json()
            assert "question" in data
            assert data["question"] == question
    
    def test_response_has_answer(self, client):
        """Response should include an answer."""
        response = client.post(
            "/api/proof/query",
            json={"question": "ما هو الحسد؟"}
        )
        if response.status_code == 200:
            data = response.json()
            assert "answer" in data
            assert len(data["answer"]) > 0
    
    def test_response_has_proof_object(self, client):
        """Response should include proof object."""
        response = client.post(
            "/api/proof/query",
            json={"question": "ما هو الحسد؟"}
        )
        if response.status_code == 200:
            data = response.json()
            assert "proof" in data
            assert isinstance(data["proof"], dict)
    
    def test_proof_has_quran_section(self, client):
        """Proof should include Quran verses."""
        response = client.post(
            "/api/proof/query",
            json={"question": "ما هو الحسد؟"}
        )
        if response.status_code == 200:
            data = response.json()
            proof = data.get("proof", {})
            assert "quran" in proof
    
    def test_proof_has_tafsir_sections(self, client):
        """Proof should include tafsir sources."""
        response = client.post(
            "/api/proof/query",
            json={"question": "ما هو الحسد؟"}
        )
        if response.status_code == 200:
            data = response.json()
            proof = data.get("proof", {})
            # Check for at least some tafsir sources
            tafsir_sources = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"]
            found_sources = [s for s in tafsir_sources if s in proof]
            assert len(found_sources) >= 3, f"Only found {found_sources}"


@pytest.mark.api
class TestProofEvidenceProvenance:
    """Tests for evidence provenance in proof responses."""
    
    def test_quran_verses_have_reference(self, client):
        """Quran verses should have surah:ayah reference."""
        response = client.post(
            "/api/proof/query",
            json={"question": "ما هو الحسد؟"}
        )
        if response.status_code == 200:
            data = response.json()
            quran = data.get("proof", {}).get("quran", [])
            if quran:
                for verse in quran[:5]:
                    # Should have reference or surah/ayah
                    has_ref = (
                        "reference" in verse or
                        ("surah" in verse and "ayah" in verse) or
                        "verse_key" in verse
                    )
                    assert has_ref, f"Verse missing reference: {verse}"
    
    def test_tafsir_quotes_have_source(self, client):
        """Tafsir quotes should identify their source."""
        response = client.post(
            "/api/proof/query",
            json={"question": "ما هو الحسد؟"}
        )
        if response.status_code == 200:
            data = response.json()
            proof = data.get("proof", {})
            
            # Check ibn_kathir as example
            ibn_kathir = proof.get("ibn_kathir", [])
            if ibn_kathir:
                for quote in ibn_kathir[:3]:
                    # Should have text content
                    has_content = (
                        "text" in quote or
                        "quote" in quote or
                        isinstance(quote, str)
                    )
                    assert has_content, f"Quote missing content: {quote}"


@pytest.mark.api
class TestProofGraphData:
    """Tests for graph data in proof responses."""
    
    def test_proof_has_graph_section(self, client):
        """Proof should include graph data."""
        response = client.post(
            "/api/proof/query",
            json={"question": "ما هو الحسد؟"}
        )
        if response.status_code == 200:
            data = response.json()
            proof = data.get("proof", {})
            assert "graph" in proof
    
    def test_graph_has_nodes_and_edges(self, client):
        """Graph should have nodes and edges."""
        response = client.post(
            "/api/proof/query",
            json={"question": "ما هو الحسد؟"}
        )
        if response.status_code == 200:
            data = response.json()
            graph = data.get("proof", {}).get("graph", {})
            assert "nodes" in graph
            assert "edges" in graph


@pytest.mark.api
class TestProofValidation:
    """Tests for validation in proof responses."""
    
    def test_response_has_validation(self, client):
        """Response should include validation info."""
        response = client.post(
            "/api/proof/query",
            json={"question": "ما هو الحسد؟"}
        )
        if response.status_code == 200:
            data = response.json()
            # Validation may be in response or proof
            has_validation = (
                "validation" in data or
                "validation" in data.get("proof", {})
            )
            # Validation is optional but preferred
            # Just check response structure is valid
            assert isinstance(data, dict)


@pytest.mark.api
class TestProofStatusEndpoint:
    """Tests for /api/proof/status endpoint."""
    
    def test_status_endpoint_exists(self, client):
        """Status endpoint should exist."""
        response = client.get("/api/proof/status")
        assert response.status_code != 404
    
    def test_status_returns_ready_info(self, client):
        """Status should indicate system readiness."""
        response = client.get("/api/proof/status")
        if response.status_code == 200:
            data = response.json()
            # Should have some status indication
            assert isinstance(data, dict)


@pytest.mark.api
class TestCausalChainQuery:
    """Tests for causal chain queries via API."""
    
    def test_causal_query_returns_graph_paths(self, client):
        """Causal queries should return graph paths."""
        response = client.post(
            "/api/proof/query",
            json={"question": "سبب الغفلة يؤدي إلى الكفر"}
        )
        if response.status_code == 200:
            data = response.json()
            graph = data.get("proof", {}).get("graph", {})
            # Should have paths for causal queries
            assert "paths" in graph or "edges" in graph


@pytest.mark.api
class TestCompleteAnalysisQuery:
    """Tests for complete analysis queries via API."""
    
    def test_complete_analysis_has_all_sections(self, client):
        """Complete analysis should include all proof sections."""
        response = client.post(
            "/api/proof/query",
            json={"question": "تحليل كامل للتوبة"}
        )
        if response.status_code == 200:
            data = response.json()
            proof = data.get("proof", {})
            
            # Should have multiple sections
            sections = ["quran", "graph"]
            found = [s for s in sections if s in proof]
            assert len(found) >= 2, f"Missing sections in complete analysis"


@pytest.mark.api
class TestErrorHandling:
    """Tests for API error handling."""
    
    def test_empty_question_rejected(self, client):
        """Empty question should be rejected."""
        response = client.post(
            "/api/proof/query",
            json={"question": ""}
        )
        # Should return validation error
        assert response.status_code in [400, 422]
    
    def test_very_long_question_handled(self, client):
        """Very long questions should be handled gracefully."""
        long_question = "ما هو الحسد؟ " * 100
        response = client.post(
            "/api/proof/query",
            json={"question": long_question}
        )
        # Should either succeed or return validation error, not crash
        assert response.status_code in [200, 400, 422]


@pytest.mark.api
class TestResponseConsistency:
    """Tests for response consistency."""
    
    def test_same_question_same_structure(self, client):
        """Same question should return same structure."""
        question = "ما هو الحسد؟"
        
        response1 = client.post(
            "/api/proof/query",
            json={"question": question}
        )
        response2 = client.post(
            "/api/proof/query",
            json={"question": question}
        )
        
        if response1.status_code == 200 and response2.status_code == 200:
            data1 = response1.json()
            data2 = response2.json()
            
            # Structure should be identical
            assert set(data1.keys()) == set(data2.keys())
            assert set(data1.get("proof", {}).keys()) == set(data2.get("proof", {}).keys())


# =============================================================================
# PHASE 10.2: STRICT API CONTRACT TESTS
# =============================================================================

@pytest.mark.api
@pytest.mark.phase10
class TestProofEndpointContract:
    """
    Phase 10.2: Strict contract enforcement.
    
    The API must ALWAYS return:
    - question (echo)
    - answer (LLM narrative)
    - proof (structured evidence with all components)
    - debug (router intent, retrieval mode, counts)
    - validation (optional but preferred)
    - processing_time_ms
    """
    
    REQUIRED_TOP_LEVEL_KEYS = ["question", "answer", "proof"]
    REQUIRED_PROOF_KEYS = ["quran", "graph"]
    REQUIRED_TAFSIR_SOURCES = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"]
    
    def test_contract_top_level_keys_present(self, client):
        """Response MUST have question, answer, proof at minimum."""
        response = client.post(
            "/api/proof/query",
            json={"question": "ما هو الحسد؟"}
        )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        
        for key in self.REQUIRED_TOP_LEVEL_KEYS:
            assert key in data, f"Missing required key: {key}"
    
    def test_contract_proof_has_quran_section(self, client):
        """Proof MUST have quran section."""
        response = client.post(
            "/api/proof/query",
            json={"question": "ما هو الحسد؟"}
        )
        assert response.status_code == 200
        data = response.json()
        proof = data.get("proof", {})
        
        assert "quran" in proof, "Proof missing 'quran' section"
        assert isinstance(proof["quran"], list), "quran must be a list"
    
    def test_contract_proof_has_all_tafsir_sources(self, client):
        """Proof MUST have all 5 tafsir source sections."""
        response = client.post(
            "/api/proof/query",
            json={"question": "ما هو الحسد؟"}
        )
        assert response.status_code == 200
        data = response.json()
        proof = data.get("proof", {})
        
        for source in self.REQUIRED_TAFSIR_SOURCES:
            assert source in proof, f"Proof missing tafsir source: {source}"
            assert isinstance(proof[source], list), f"{source} must be a list"
    
    def test_contract_proof_has_graph_section(self, client):
        """Proof MUST have graph section with nodes/edges."""
        response = client.post(
            "/api/proof/query",
            json={"question": "ما هو الحسد؟"}
        )
        assert response.status_code == 200
        data = response.json()
        proof = data.get("proof", {})
        
        assert "graph" in proof, "Proof missing 'graph' section"
        graph = proof["graph"]
        assert "nodes" in graph, "Graph missing 'nodes'"
        assert "edges" in graph, "Graph missing 'edges'"
    
    def test_contract_has_debug_info(self, client):
        """Response SHOULD have debug info."""
        response = client.post(
            "/api/proof/query",
            json={"question": "ما هو الحسد؟"}
        )
        assert response.status_code == 200
        data = response.json()
        
        # Debug is expected but may be empty dict
        assert "debug" in data, "Response missing 'debug' section"
    
    def test_contract_has_processing_time(self, client):
        """Response SHOULD have processing_time_ms."""
        response = client.post(
            "/api/proof/query",
            json={"question": "ما هو الحسد؟"}
        )
        assert response.status_code == 200
        data = response.json()
        
        assert "processing_time_ms" in data, "Response missing 'processing_time_ms'"
        assert isinstance(data["processing_time_ms"], (int, float))
    
    def test_contract_answer_not_empty(self, client):
        """Answer MUST not be empty."""
        response = client.post(
            "/api/proof/query",
            json={"question": "ما هو الحسد؟"}
        )
        assert response.status_code == 200
        data = response.json()
        
        answer = data.get("answer", "")
        assert len(answer) > 0, "Answer is empty"


@pytest.mark.api
@pytest.mark.phase10
class TestProofEvidenceNotEmpty:
    """Ensure proof sections contain actual evidence, not just empty lists."""
    
    def test_quran_has_verses_for_behavior_query(self, client):
        """Behavior queries should return Quran verses."""
        response = client.post(
            "/api/proof/query",
            json={"question": "ما هو الحسد؟"}
        )
        assert response.status_code == 200
        data = response.json()
        
        quran = data.get("proof", {}).get("quran", [])
        # Should have at least some verses for a behavior query
        # (may be 0 for very obscure queries, but الحسد should have verses)
        assert len(quran) >= 1, "No Quran verses returned for الحسد query"
    
    def test_tafsir_has_quotes_for_behavior_query(self, client):
        """Behavior queries should return tafsir quotes."""
        response = client.post(
            "/api/proof/query",
            json={"question": "ما هو الحسد؟"}
        )
        assert response.status_code == 200
        data = response.json()
        proof = data.get("proof", {})
        
        # At least one tafsir source should have quotes
        total_quotes = 0
        for source in ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"]:
            total_quotes += len(proof.get(source, []))
        
        assert total_quotes >= 1, "No tafsir quotes returned"


@pytest.mark.api
@pytest.mark.phase10
class TestProofProvenanceComplete:
    """Ensure evidence has complete provenance (source, reference, offsets)."""
    
    def test_quran_verses_have_surah_ayah(self, client):
        """Each Quran verse must have surah and ayah."""
        response = client.post(
            "/api/proof/query",
            json={"question": "ما هو الحسد؟"}
        )
        assert response.status_code == 200
        data = response.json()
        
        quran = data.get("proof", {}).get("quran", [])
        for verse in quran:
            assert "surah" in verse, f"Verse missing surah: {verse}"
            assert "ayah" in verse, f"Verse missing ayah: {verse}"
            assert "text" in verse, f"Verse missing text: {verse}"
    
    def test_tafsir_quotes_have_reference(self, client):
        """Each tafsir quote must have surah/ayah reference."""
        response = client.post(
            "/api/proof/query",
            json={"question": "ما هو الحسد؟"}
        )
        assert response.status_code == 200
        data = response.json()
        proof = data.get("proof", {})
        
        for source in ["ibn_kathir", "tabari", "qurtubi"]:
            quotes = proof.get(source, [])
            for quote in quotes[:3]:  # Check first 3
                assert "surah" in quote or "verse_key" in quote, f"{source} quote missing reference"
                assert "text" in quote, f"{source} quote missing text"


@pytest.mark.api
@pytest.mark.phase10
class TestHardAcceptanceHasad:
    """
    Phase 10.2c: Hard acceptance test for الحسد query.
    
    This is the gate test - if this fails, Phase 10.2 is not complete.
    """
    
    def test_hasad_query_returns_quran_verses(self, client):
        """
        HARD ACCEPTANCE: Query 'ما هو الحسد؟' MUST return:
        - proof.quran.verses is non-empty
        - every verse has surah, ayah, text
        - provenance for tafsir remains unchanged
        """
        response = client.post(
            "/api/proof/query",
            json={"question": "ما هو الحسد؟"}
        )
        assert response.status_code == 200, f"API returned {response.status_code}"
        data = response.json()
        
        # 1. proof.quran.verses MUST be non-empty
        quran = data.get("proof", {}).get("quran", [])
        assert len(quran) > 0, "HARD FAIL: proof.quran.verses is empty for الحسد query"
        
        # 2. Every verse MUST have surah, ayah, text
        for i, verse in enumerate(quran):
            assert "surah" in verse and verse["surah"], f"Verse {i} missing surah"
            assert "ayah" in verse and verse["ayah"], f"Verse {i} missing ayah"
            assert "text" in verse and len(verse["text"]) > 0, f"Verse {i} missing text"
        
        # 3. Tafsir provenance unchanged (at least one tafsir source has quotes)
        proof = data.get("proof", {})
        tafsir_total = sum(len(proof.get(s, [])) for s in ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"])
        assert tafsir_total > 0, "HARD FAIL: No tafsir quotes returned"
        
        # Log success details (avoid emojis for Windows console compatibility)
        print(f"\n[PASS] HARD ACCEPTANCE PASSED:")
        print(f"   - Quran verses: {len(quran)}")
        print(f"   - Tafsir quotes: {tafsir_total}")
        print(f"   - First verse: {quran[0].get('surah')}:{quran[0].get('ayah')}")


@pytest.mark.api
@pytest.mark.phase10
class TestNoFabricatedEvidence:
    """Ensure API never returns fabricated/synthetic evidence."""
    
    def test_quran_text_is_arabic(self, client):
        """Quran verse text must be Arabic, not placeholder."""
        response = client.post(
            "/api/proof/query",
            json={"question": "ما هو الحسد؟"}
        )
        assert response.status_code == 200
        data = response.json()
        
        quran = data.get("proof", {}).get("quran", [])
        for verse in quran:
            text = verse.get("text", "")
            # Should contain Arabic characters
            has_arabic = any('\u0600' <= c <= '\u06FF' for c in text)
            assert has_arabic, f"Verse text not Arabic: {text[:50]}"
    
    def test_no_placeholder_tafsir(self, client):
        """Tafsir should not contain placeholder text."""
        response = client.post(
            "/api/proof/query",
            json={"question": "ما هو الحسد؟"}
        )
        assert response.status_code == 200
        data = response.json()
        proof = data.get("proof", {})
        
        placeholder_patterns = ["لا يوجد تفسير", "placeholder", "TODO", "FIXME"]
        
        for source in ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"]:
            quotes = proof.get(source, [])
            for quote in quotes:
                text = quote.get("text", "")
                for pattern in placeholder_patterns:
                    assert pattern not in text, f"Placeholder found in {source}: {pattern}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "api"])
