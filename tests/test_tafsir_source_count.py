"""
Phase 0.1b Test: Verify all modules use 7 canonical tafsir sources.

This test ensures no module silently falls back to 5 sources.
All tafsir source lists must be loaded from vocab/tafsir_sources.json
via src/ml/tafsir_constants.py.
"""

import pytest


class TestCanonicalTafsirSources:
    """Test that canonical tafsir sources are unified to 7 everywhere."""
    
    def test_tafsir_constants_has_7_sources(self):
        """Verify tafsir_constants.py exports exactly 7 sources."""
        from src.ml.tafsir_constants import CANONICAL_TAFSIR_SOURCES, TAFSIR_SOURCE_COUNT
        
        assert len(CANONICAL_TAFSIR_SOURCES) == 7, \
            f"Expected 7 tafsir sources, got {len(CANONICAL_TAFSIR_SOURCES)}"
        assert TAFSIR_SOURCE_COUNT == 7
        
        # Verify all expected sources are present
        expected = {"ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn", "baghawi", "muyassar"}
        actual = set(CANONICAL_TAFSIR_SOURCES)
        assert actual == expected, f"Missing sources: {expected - actual}, Extra: {actual - expected}"
    
    def test_all_sources_with_quran_has_8(self):
        """Verify ALL_SOURCES_WITH_QURAN includes quran + 7 tafsir."""
        from src.ml.tafsir_constants import ALL_SOURCES_WITH_QURAN
        
        assert len(ALL_SOURCES_WITH_QURAN) == 8, \
            f"Expected 8 sources (7 tafsir + quran), got {len(ALL_SOURCES_WITH_QURAN)}"
        assert "quran" in ALL_SOURCES_WITH_QURAN
    
    def test_legendary_planner_uses_7_sources(self):
        """Verify LegendaryPlanner.CORE_SOURCES has 7 sources."""
        from src.ml.legendary_planner import CORE_SOURCES
        
        assert len(CORE_SOURCES) == 7, \
            f"LegendaryPlanner.CORE_SOURCES should have 7 sources, got {len(CORE_SOURCES)}"
    
    def test_vocab_tafsir_sources_json_has_7(self):
        """Verify vocab/tafsir_sources.json defines exactly 7 sources."""
        import json
        from pathlib import Path
        
        vocab_file = Path("vocab/tafsir_sources.json")
        assert vocab_file.exists(), "vocab/tafsir_sources.json not found"
        
        with open(vocab_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        core = data.get("core_sources", [])
        supplementary = data.get("supplementary_sources", [])
        total = len(core) + len(supplementary)
        
        assert total == 7, f"vocab/tafsir_sources.json should define 7 sources, got {total}"
        assert len(core) == 5, f"Expected 5 core sources, got {len(core)}"
        assert len(supplementary) == 2, f"Expected 2 supplementary sources, got {len(supplementary)}"


class TestNoHardcodedFiveSourceLists:
    """Test that no module hardcodes 5-source lists."""
    
    def test_mandatory_proof_system_uses_shared_constant(self):
        """Verify MandatoryProofSystem uses shared tafsir constant."""
        # This test verifies the import is present in the source
        import inspect
        from src.ml import mandatory_proof_system
        
        source = inspect.getsource(mandatory_proof_system)
        
        # Should import from tafsir_constants
        assert "tafsir_constants" in source, \
            "mandatory_proof_system.py should import from tafsir_constants"
        
        # Should NOT have hardcoded 5-source list (old pattern)
        old_pattern = '["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"]'
        assert old_pattern not in source, \
            "mandatory_proof_system.py should not have hardcoded 5-source list"
    
    def test_full_power_system_uses_shared_constant(self):
        """Verify FullPowerSystem uses shared tafsir constant."""
        import inspect
        from src.ml import full_power_system
        
        source = inspect.getsource(full_power_system)
        
        # Should import from tafsir_constants
        assert "tafsir_constants" in source, \
            "full_power_system.py should import from tafsir_constants"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
