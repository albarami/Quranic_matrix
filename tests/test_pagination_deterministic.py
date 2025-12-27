"""
Phase 7.2 Fix B: Deterministic Ordering Tests

Tests for:
- Pagination ordering is deterministic across repeated calls
- Tie-breakers ensure no skip/duplicate across pages
- Sort keys work correctly (verse, surah, score)
"""

import pytest


class TestDeterministicOrdering:
    """Test that pagination ordering is deterministic."""
    
    def test_sort_deterministic_by_verse(self):
        """Sort by verse uses (-score, surah, ayah, source, chunk_id)."""
        from src.api.routers.proof import sort_deterministic
        
        items = [
            {"surah": 2, "ayah": 255, "score": 0.9, "source": "ibn_kathir", "chunk_id": "c1"},
            {"surah": 2, "ayah": 255, "score": 0.9, "source": "tabari", "chunk_id": "c2"},
            {"surah": 1, "ayah": 1, "score": 0.95, "source": "jalalayn", "chunk_id": "c3"},
            {"surah": 2, "ayah": 255, "score": 0.9, "source": "ibn_kathir", "chunk_id": "c0"},  # Same source, different chunk
        ]
        
        sorted_items = sort_deterministic(items, "verse")
        
        # First should be highest score (0.95)
        assert sorted_items[0]["chunk_id"] == "c3"
        
        # Same score items should be sorted by surah, ayah, source, chunk_id
        same_score = [i for i in sorted_items if i["score"] == 0.9]
        assert same_score[0]["chunk_id"] == "c0"  # ibn_kathir c0 before c1
        assert same_score[1]["chunk_id"] == "c1"  # ibn_kathir c1
        assert same_score[2]["chunk_id"] == "c2"  # tabari c2
    
    def test_sort_deterministic_by_surah(self):
        """Sort by surah uses (surah, ayah, source, chunk_id)."""
        from src.api.routers.proof import sort_deterministic
        
        items = [
            {"surah": 2, "ayah": 255, "source": "ibn_kathir", "chunk_id": "c1"},
            {"surah": 1, "ayah": 7, "source": "tabari", "chunk_id": "c2"},
            {"surah": 1, "ayah": 1, "source": "jalalayn", "chunk_id": "c3"},
        ]
        
        sorted_items = sort_deterministic(items, "surah")
        
        # Should be in canonical order: 1:1, 1:7, 2:255
        assert sorted_items[0]["chunk_id"] == "c3"  # 1:1
        assert sorted_items[1]["chunk_id"] == "c2"  # 1:7
        assert sorted_items[2]["chunk_id"] == "c1"  # 2:255
    
    def test_sort_deterministic_empty_list(self):
        """Empty list should return empty list."""
        from src.api.routers.proof import sort_deterministic
        
        assert sort_deterministic([], "verse") == []
    
    def test_paginate_list_deterministic(self):
        """Pagination should be deterministic across calls."""
        from src.api.routers.proof import paginate_list
        
        items = [
            {"surah": i, "ayah": 1, "score": 1.0 - i * 0.01, "chunk_id": f"c{i}"}
            for i in range(1, 21)
        ]
        
        # Get page 1 twice
        result1 = paginate_list(items, page=1, page_size=5)
        result2 = paginate_list(items, page=1, page_size=5)
        
        # Should be identical
        assert result1["items"] == result2["items"]
    
    def test_paginate_no_skip_no_duplicate(self):
        """Items should not skip or duplicate across pages."""
        from src.api.routers.proof import paginate_list
        
        items = [
            {"surah": i, "ayah": 1, "score": 1.0, "chunk_id": f"c{i}"}
            for i in range(1, 11)
        ]
        
        page1 = paginate_list(items, page=1, page_size=3)
        page2 = paginate_list(items, page=2, page_size=3)
        page3 = paginate_list(items, page=3, page_size=3)
        page4 = paginate_list(items, page=4, page_size=3)
        
        # Collect all chunk_ids
        all_ids = []
        for page in [page1, page2, page3, page4]:
            all_ids.extend([i["chunk_id"] for i in page["items"]])
        
        # Should have exactly 10 unique items, no duplicates
        assert len(all_ids) == 10
        assert len(set(all_ids)) == 10
    
    def test_paginate_metadata_correct(self):
        """Pagination metadata should be correct."""
        from src.api.routers.proof import paginate_list
        
        items = [{"surah": i, "ayah": 1, "chunk_id": f"c{i}"} for i in range(1, 26)]
        
        result = paginate_list(items, page=2, page_size=10)
        
        assert result["page"] == 2
        assert result["page_size"] == 10
        assert result["total_items"] == 25
        assert result["total_pages"] == 3
        assert result["has_next"] == True
        assert result["has_prev"] == True
        assert len(result["items"]) == 10


class TestSortKeyVariants:
    """Test different sort key variants."""
    
    def test_score_sort_key(self):
        """Score sort key uses (-score, verse_key, chunk_id)."""
        from src.api.routers.proof import sort_deterministic
        
        items = [
            {"surah": 2, "ayah": 1, "score": 0.5, "chunk_id": "c1"},
            {"surah": 1, "ayah": 1, "score": 0.9, "chunk_id": "c2"},
            {"surah": 3, "ayah": 1, "score": 0.7, "chunk_id": "c3"},
        ]
        
        sorted_items = sort_deterministic(items, "score")
        
        # Should be sorted by score descending
        assert sorted_items[0]["score"] == 0.9
        assert sorted_items[1]["score"] == 0.7
        assert sorted_items[2]["score"] == 0.5
    
    def test_missing_fields_handled(self):
        """Items with missing fields should not crash."""
        from src.api.routers.proof import sort_deterministic
        
        items = [
            {"surah": 1},  # Missing ayah, score, source, chunk_id
            {"ayah": 1},   # Missing surah
            {},            # Empty
        ]
        
        # Should not raise
        result = sort_deterministic(items, "verse")
        assert len(result) == 3
