"""
Test semantic similarity gold data for correct labeling.

Phase 5.1: Ensures opposites have LOW similarity and similar pairs have HIGH similarity.
"""

import json
import pytest
from pathlib import Path

GOLD_FILE = Path(__file__).parent.parent / "data" / "evaluation" / "semantic_similarity_gold_v3.jsonl"


def load_gold_data():
    """Load gold benchmark data."""
    records = []
    with open(GOLD_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


class TestGoldDataSemantics:
    """Test that gold data has correct semantic labels."""
    
    def test_gold_file_exists(self):
        """Gold file must exist."""
        assert GOLD_FILE.exists(), f"Gold file not found: {GOLD_FILE}"
    
    def test_opposites_have_low_similarity(self):
        """Opposites should have similarity <= 0.2 (they are NOT similar)."""
        records = load_gold_data()
        
        for record in records:
            if record["category"] == "opposite":
                assert record["similarity"] <= 0.2, (
                    f"Opposite pair should have LOW similarity: "
                    f"{record['text_a']} <-> {record['text_b']} = {record['similarity']}"
                )
    
    def test_similar_pairs_have_high_similarity(self):
        """Same concept and synonym pairs should have similarity >= 0.7."""
        records = load_gold_data()
        
        for record in records:
            if record["category"] in ("same_concept", "synonym"):
                assert record["similarity"] >= 0.7, (
                    f"Similar pair should have HIGH similarity: "
                    f"{record['text_a']} <-> {record['text_b']} = {record['similarity']}"
                )
    
    def test_unrelated_pairs_have_zero_similarity(self):
        """Unrelated pairs should have similarity = 0.0."""
        records = load_gold_data()
        
        for record in records:
            if record["category"] == "unrelated":
                assert record["similarity"] == 0.0, (
                    f"Unrelated pair should have ZERO similarity: "
                    f"{record['text_a']} <-> {record['text_b']} = {record['similarity']}"
                )
    
    def test_hope_fear_are_complementary_not_opposite(self):
        """Hope (الرجاء) and Fear (الخوف) are complementary in Islam, not opposites."""
        records = load_gold_data()
        
        hope_fear_found = False
        for record in records:
            if "الرجاء" in record["text_a"] and "الخوف" in record["text_b"]:
                hope_fear_found = True
                assert record["category"] == "complementary", (
                    f"Hope/Fear should be 'complementary', not '{record['category']}'"
                )
                assert record["similarity"] >= 0.6, (
                    f"Hope/Fear should have similarity >= 0.6, got {record['similarity']}"
                )
        
        assert hope_fear_found, "Hope/Fear pair not found in gold data"
    
    def test_minimum_pairs_per_category(self):
        """Ensure sufficient coverage of each category."""
        records = load_gold_data()
        
        categories = {}
        for record in records:
            cat = record["category"]
            categories[cat] = categories.get(cat, 0) + 1
        
        assert categories.get("same_concept", 0) >= 10, "Need at least 10 same_concept pairs"
        assert categories.get("opposite", 0) >= 10, "Need at least 10 opposite pairs"
        assert categories.get("unrelated", 0) >= 5, "Need at least 5 unrelated pairs"
    
    def test_no_duplicate_pairs(self):
        """No duplicate text pairs."""
        records = load_gold_data()
        
        seen = set()
        for record in records:
            pair = (record["text_a"], record["text_b"])
            assert pair not in seen, f"Duplicate pair found: {pair}"
            seen.add(pair)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
