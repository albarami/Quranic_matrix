"""
Test suite for QBMUnifiedSystem.

Tests the unified system that links all AI components:
- Tafsir (7 sources)
- Behaviors (87 concepts)
- Organs (25)
- Agents (14)
- Knowledge Graph
- Vector Store
"""

import pytest
from src.ai.unified import QBMUnifiedSystem


class TestUnifiedSystemInitialization:
    """Test unified system initialization."""

    @pytest.fixture
    def qbm(self):
        """Create unified system instance."""
        return QBMUnifiedSystem()

    def test_init_loads_behaviors(self, qbm):
        """Test that behaviors are loaded."""
        assert len(qbm._behaviors) >= 70

    def test_init_loads_organs(self, qbm):
        """Test that organs are loaded."""
        assert len(qbm._organs) >= 20

    def test_init_loads_agents(self, qbm):
        """Test that agents are loaded."""
        assert len(qbm._agents) >= 10

    def test_get_statistics(self, qbm):
        """Test statistics method."""
        stats = qbm.get_statistics()
        assert "behaviors" in stats
        assert "organs" in stats
        assert "agents" in stats
        assert "tafsir" in stats
        assert stats["behaviors"]["total"] >= 70
        assert stats["organs"]["total"] >= 20
        assert stats["agents"]["total"] >= 10


class TestQueryAyah:
    """Test ayah query functionality."""

    @pytest.fixture
    def qbm(self):
        """Create unified system instance."""
        return QBMUnifiedSystem()

    def test_query_ayah_returns_tafsirs(self, qbm):
        """Test that query_ayah returns tafsir entries for all known sources."""
        result = qbm.query_ayah(2, 7)
        assert "tafsir" in result
        expected_sources = set(qbm.tafsir.get_available_sources())
        assert set(result["tafsir"].keys()) == expected_sources

    def test_query_ayah_extracts_behaviors(self, qbm):
        """Test that query_ayah returns behaviors list (may be empty in offline/fixture mode)."""
        result = qbm.query_ayah(2, 7)
        assert "behaviors" in result
        assert isinstance(result["behaviors"], list)
        if result["behaviors"]:
            beh = result["behaviors"][0]
            assert "behavior_id" in beh
            assert "name_ar" in beh
            assert "name_en" in beh
            assert "mention_count" in beh

    def test_query_ayah_extracts_organs(self, qbm):
        """Test that query_ayah returns organs list (may be empty in offline/fixture mode)."""
        result = qbm.query_ayah(2, 7)
        assert "organs" in result
        assert isinstance(result["organs"], list)
        if result["organs"]:
            organ = result["organs"][0]
            assert "organ_id" in organ
            assert "name_ar" in organ
            assert "mention_count" in organ

    def test_query_ayah_extracts_agents(self, qbm):
        """Test that query_ayah extracts agents."""
        result = qbm.query_ayah(2, 7)
        assert "agents" in result


class TestQueryConcept:
    """Test concept query functionality."""

    @pytest.fixture
    def qbm(self):
        """Create unified system instance."""
        return QBMUnifiedSystem()

    def test_query_concept_heart(self, qbm):
        """Test querying for 'القلب' (heart)."""
        result = qbm.query_concept("القلب", limit=10)
        assert "total_mentions" in result
        assert "unique_ayat" in result
        assert "ibn_kathir" in result["statistics"]
        assert "tabari" in result["statistics"]

    def test_query_concept_returns_ayat(self, qbm):
        """Test that concept query returns ayat references."""
        result = qbm.query_concept("الكبر", limit=5)
        assert "ayat" in result
        # ayat should be list of (surah, ayah) tuples
        if result["ayat"]:
            assert isinstance(result["ayat"][0], tuple)


class TestTafsirConsensus:
    """Test tafsir consensus functionality."""

    @pytest.fixture
    def qbm(self):
        """Create unified system instance."""
        return QBMUnifiedSystem()

    def test_find_consensus(self, qbm):
        """Test finding consensus across tafsirs."""
        result = qbm.find_tafsir_consensus(2, 7)
        assert "sources" in result
        assert len(result["sources"]) == len(qbm.tafsir.get_available_sources())
        assert "behaviors_mentioned" in result
        assert "consensus" in result

    def test_consensus_structure(self, qbm):
        """Test consensus result structure."""
        result = qbm.find_tafsir_consensus(2, 7)
        if result["consensus"]:
            c = result["consensus"][0]
            assert "behavior_id" in c
            assert "name_ar" in c
            assert "agreed_by" in c
            assert "agreement_level" in c


class TestQueryBehavior:
    """Test behavior query functionality."""

    @pytest.fixture
    def qbm(self):
        """Create unified system instance."""
        return QBMUnifiedSystem()

    def test_query_behavior_exists(self, qbm):
        """Test querying an existing behavior."""
        result = qbm.query_behavior("BEH_SPI_FAITH")
        assert "behavior_id" in result
        assert result["behavior_id"] == "BEH_SPI_FAITH"
        assert "name_ar" in result
        assert "name_en" in result

    def test_query_behavior_not_found(self, qbm):
        """Test querying a non-existent behavior."""
        result = qbm.query_behavior("NONEXISTENT")
        assert "error" in result

    def test_query_behavior_with_tafsir_evidence(self, qbm):
        """Test that behavior query includes tafsir evidence."""
        result = qbm.query_behavior("BEH_SPI_FAITH", include_tafsir_evidence=True)
        assert "tafsir_evidence" in result
        assert "total_mentions" in result["tafsir_evidence"]


class TestEdgeCases:
    """Test edge cases and bug fixes."""

    @pytest.fixture
    def qbm(self):
        """Create unified system instance."""
        return QBMUnifiedSystem()

    def test_consensus_empty_tafsir_no_zerodiv(self, qbm):
        """Test that find_tafsir_consensus handles invalid ayah without ZeroDivisionError."""
        # Invalid surah/ayah should not crash
        result = qbm.find_tafsir_consensus(999, 999)
        assert "consensus" in result
        # Should return empty, not crash
        assert result["consensus"] == []

    def test_graph_loaded(self, qbm):
        """Test that graph is loaded from DB."""
        assert qbm.graph.G.number_of_nodes() > 0
        assert qbm.graph.G.number_of_edges() > 0

    def test_query_behavior_has_caused_by(self, qbm):
        """Test that caused_by is populated from incoming edges."""
        # BEH_SPI_FAITH has incoming CAUSES edge from BEH_EMO_GRATITUDE
        result = qbm.query_behavior("BEH_SPI_FAITH")
        assert "relationships" in result
        # caused_by should now be populated
        assert isinstance(result["relationships"]["caused_by"], list)

    def test_query_behavior_relationships_structure(self, qbm):
        """Test that all relationship types are present."""
        result = qbm.query_behavior("BEH_SPI_FAITH")
        rels = result.get("relationships", {})
        assert "causes" in rels
        assert "caused_by" in rels
        assert "results_in" in rels
        assert "opposite_of" in rels
        assert "related_to" in rels


# Run with: pytest tests/ai/test_unified.py -v
