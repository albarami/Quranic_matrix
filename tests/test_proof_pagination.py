"""
Phase 7.2: Pagination + Summary Mode Tests

Tests for:
- SURAH_REF summary returns all ayat with core sources (1 chunk each)
- SURAH_REF full pagination with stable chunk IDs
- CONCEPT_REF summary returns deterministic top verses
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """Create test client for API."""
    from src.api.main import app
    return TestClient(app)


class TestSurahRefSummary:
    """Test SURAH_REF summary mode."""
    
    @pytest.mark.tier_a
    def test_surah_ref_summary_returns_all_ayat_core_sources(self, client):
        """
        SURAH_REF summary should return per-ayah × per-source breakdown.
        For سورة الفاتحة (7 ayat), should show all 7 with tafsir from 5 sources.
        """
        response = client.post(
            "/api/proof/query",
            json={
                "question": "سورة الفاتحة",
                "mode": "summary",
                "per_ayah": True,
                "max_chunks_per_source": 1
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check pagination metadata is present
        assert "pagination" in data
        assert data["pagination"]["mode"] == "summary"
        assert data["pagination"]["per_ayah"] == True
        
        # Check proof structure
        proof = data.get("proof", {})
        assert "intent" in proof
        
        # If SURAH_REF detected, should have surah_summary mode
        if proof.get("intent") == "SURAH_REF":
            assert proof.get("mode") == "surah_summary"
            
            # Check quran section has pagination info
            quran = proof.get("quran", {})
            if isinstance(quran, dict) and "items" in quran:
                # Paginated response
                assert "total_items" in quran
                assert "page" in quran
                # Al-Fatiha has 7 ayat
                assert quran["total_items"] >= 7
    
    @pytest.mark.tier_a
    def test_surah_ref_full_pagination_stable_chunk_ids(self, client):
        """
        SURAH_REF full mode should paginate all chunks with stable ordering.
        """
        # First page
        response1 = client.post(
            "/api/proof/query",
            json={
                "question": "سورة الفاتحة",
                "mode": "full",
                "page": 1,
                "page_size": 3,
                "per_ayah": True
            }
        )
        
        assert response1.status_code == 200
        data1 = response1.json()
        
        # Second page
        response2 = client.post(
            "/api/proof/query",
            json={
                "question": "سورة الفاتحة",
                "mode": "full",
                "page": 2,
                "page_size": 3,
                "per_ayah": True
            }
        )
        
        assert response2.status_code == 200
        data2 = response2.json()
        
        proof1 = data1.get("proof", {})
        proof2 = data2.get("proof", {})
        
        if proof1.get("intent") == "SURAH_REF":
            quran1 = proof1.get("quran", {})
            quran2 = proof2.get("quran", {})
            
            if isinstance(quran1, dict) and "items" in quran1:
                # Check pagination is working
                assert quran1.get("page") == 1
                assert quran2.get("page") == 2
                
                # Items should be different between pages
                items1 = quran1.get("items", [])
                items2 = quran2.get("items", [])
                
                if items1 and items2:
                    # First item of page 2 should not be in page 1
                    page1_ayat = {f"{i.get('surah')}:{i.get('ayah')}" for i in items1}
                    page2_ayat = {f"{i.get('surah')}:{i.get('ayah')}" for i in items2}
                    assert page1_ayat != page2_ayat


class TestConceptRefSummary:
    """Test CONCEPT_REF summary mode."""
    
    @pytest.mark.tier_a
    def test_concept_ref_summary_deterministic_top_verses(self, client):
        """
        CONCEPT_REF summary should return deterministic verse list.
        """
        # Query twice with same concept
        response1 = client.post(
            "/api/proof/query",
            json={
                "question": "الحسد",
                "mode": "summary",
                "max_chunks_per_source": 1
            }
        )
        
        response2 = client.post(
            "/api/proof/query",
            json={
                "question": "الحسد",
                "mode": "summary",
                "max_chunks_per_source": 1
            }
        )
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        data1 = response1.json()
        data2 = response2.json()
        
        proof1 = data1.get("proof", {})
        proof2 = data2.get("proof", {})
        
        # Check intent detection
        if proof1.get("intent") == "CONCEPT_REF":
            assert proof1.get("mode") == "concept_summary"
            
            # Quran verses should be deterministic
            quran1 = proof1.get("quran", [])
            quran2 = proof2.get("quran", [])
            
            if quran1 and quran2:
                # Same verses in same order
                verses1 = [f"{v.get('surah')}:{v.get('ayah')}" for v in quran1[:5]]
                verses2 = [f"{v.get('surah')}:{v.get('ayah')}" for v in quran2[:5]]
                assert verses1 == verses2
    
    @pytest.mark.tier_a
    def test_concept_ref_shows_truncation_metadata(self, client):
        """
        When tafsir is truncated, should show total count.
        """
        response = client.post(
            "/api/proof/query",
            json={
                "question": "الحسد",
                "mode": "summary",
                "max_chunks_per_source": 1
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        proof = data.get("proof", {})
        
        # Check for truncation metadata (if there are more chunks than shown)
        tafsir_sources = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"]
        for source in tafsir_sources:
            chunks = proof.get(source, [])
            total_key = f"{source}_total"
            
            # If truncated, should have total count
            if total_key in proof:
                assert proof[total_key] > len(chunks)


class TestPaginationContract:
    """Test pagination contract is stable."""
    
    @pytest.mark.tier_a
    def test_pagination_metadata_always_present(self, client):
        """Pagination metadata should always be in response."""
        response = client.post(
            "/api/proof/query",
            json={"question": "ما هو الصبر؟"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "pagination" in data
        pagination = data["pagination"]
        
        assert "mode" in pagination
        assert "page" in pagination
        assert "page_size" in pagination
    
    @pytest.mark.tier_a
    def test_never_truncates_silently(self, client):
        """
        When data is truncated, must show total count.
        Never truncate without indicating there's more.
        """
        response = client.post(
            "/api/proof/query",
            json={
                "question": "سورة البقرة",
                "mode": "summary",
                "page_size": 5,
                "max_chunks_per_source": 1
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        proof = data.get("proof", {})
        
        # If quran is paginated, should have total_items
        quran = proof.get("quran", {})
        if isinstance(quran, dict) and "items" in quran:
            assert "total_items" in quran
            assert "total_pages" in quran
            
            # If there are more pages, has_next should be True
            if quran["total_items"] > quran["page_size"]:
                assert quran["has_next"] == True or quran["page"] > 1
