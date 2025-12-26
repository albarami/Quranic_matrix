"""
Test: Cross-Tafsir Methodology Metadata (Phase 6.4)

Ensures tafsir_sources.json has proper structure for evidence weighting.
"""

import pytest
import json
from pathlib import Path

TAFSIR_SOURCES_FILE = Path("vocab/tafsir_sources.json")
CORE_SOURCES = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"]


@pytest.fixture(scope="module")
def tafsir_sources():
    """Load tafsir sources metadata."""
    with open(TAFSIR_SOURCES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.unit
class TestTafsirSourcesStructure:
    """Tests for tafsir sources file structure."""
    
    def test_file_exists(self):
        """tafsir_sources.json must exist."""
        assert TAFSIR_SOURCES_FILE.exists()
    
    def test_has_core_sources(self, tafsir_sources):
        """Must have core_sources array."""
        assert "core_sources" in tafsir_sources
        assert len(tafsir_sources["core_sources"]) >= 5
    
    def test_all_core_sources_present(self, tafsir_sources):
        """All 5 core sources must be defined."""
        source_ids = [s["id"] for s in tafsir_sources["core_sources"]]
        for expected in CORE_SOURCES:
            assert expected in source_ids, f"Missing core source: {expected}"
    
    def test_sources_have_required_fields(self, tafsir_sources):
        """Each source must have required metadata fields."""
        required_fields = ["id", "name_ar", "name_en", "methodology", "reliability_weight"]
        
        for source in tafsir_sources["core_sources"]:
            for field in required_fields:
                assert field in source, f"Source {source.get('id')} missing field: {field}"


@pytest.mark.unit
class TestTafsirSourcesWeighting:
    """Tests for evidence weighting metadata."""
    
    def test_reliability_weights_valid(self, tafsir_sources):
        """Reliability weights must be between 0 and 1."""
        for source in tafsir_sources["core_sources"]:
            weight = source.get("reliability_weight", 0)
            assert 0 <= weight <= 1, f"Invalid weight for {source['id']}: {weight}"
    
    def test_methodology_types_defined(self, tafsir_sources):
        """Methodology types must be defined with weights."""
        assert "methodology_types" in tafsir_sources
        
        for method_id, method in tafsir_sources["methodology_types"].items():
            assert "evidence_weight" in method, f"Missing evidence_weight for {method_id}"
            assert 0 <= method["evidence_weight"] <= 1
    
    def test_agreement_levels_defined(self, tafsir_sources):
        """Agreement levels (ijma, jumhur, khilaf) must be defined."""
        assert "agreement_levels" in tafsir_sources
        
        expected_levels = ["ijma", "jumhur", "khilaf"]
        for level in expected_levels:
            assert level in tafsir_sources["agreement_levels"], f"Missing level: {level}"


@pytest.mark.unit
class TestCrossValidationRules:
    """Tests for cross-validation rules."""
    
    def test_cross_validation_rules_exist(self, tafsir_sources):
        """Cross-validation rules must be defined."""
        assert "cross_validation_rules" in tafsir_sources
        assert len(tafsir_sources["cross_validation_rules"]) >= 1
    
    def test_multi_source_rule_exists(self, tafsir_sources):
        """Multi-source requirement rule must exist."""
        rules = tafsir_sources["cross_validation_rules"]
        rule_ids = [r["rule_id"] for r in rules]
        assert "multi_source_required" in rule_ids


@pytest.mark.unit
class TestSourceMetadata:
    """Tests for individual source metadata."""
    
    def test_tabari_is_earliest(self, tafsir_sources):
        """Tabari should be the earliest source (died 310 AH)."""
        tabari = next(s for s in tafsir_sources["core_sources"] if s["id"] == "tabari")
        assert tabari["death_year_hijri"] == 310
    
    def test_ibn_kathir_hadith_based(self, tafsir_sources):
        """Ibn Kathir should be hadith-based methodology."""
        ibn_kathir = next(s for s in tafsir_sources["core_sources"] if s["id"] == "ibn_kathir")
        assert ibn_kathir["methodology"] == "hadith_based"
    
    def test_qurtubi_fiqh_focused(self, tafsir_sources):
        """Qurtubi should be fiqh-focused methodology."""
        qurtubi = next(s for s in tafsir_sources["core_sources"] if s["id"] == "qurtubi")
        assert qurtubi["methodology"] == "fiqh_focused"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
