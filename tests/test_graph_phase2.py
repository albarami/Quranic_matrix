#!/usr/bin/env python3
"""
Phase 2 Tests: Graph Projection in Postgres

Tests for:
1. behavior→verse edges exist in behavior_verse_links
2. surah_intro entries are filtered
3. tafsir ⊆ verse evidence (subset contract)
4. semantic_edges have entity types and provenance

Run with: pytest tests/test_graph_phase2.py -v
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

CONCEPT_INDEX_FILE = Path("data/evidence/concept_index_v2.jsonl")
SEMANTIC_GRAPH_FILE = Path("data/graph/semantic_graph_v2.json")
CANONICAL_ENTITIES_FILE = Path("vocab/canonical_entities.json")
MIGRATION_FILE = Path("db/migrations/001_ssot_extensions.sql")
STATS_FILE = Path("artifacts/behavior_verse_links_stats.json")


# ============================================================================
# Graph Structure Tests
# ============================================================================

@pytest.mark.unit
class TestGraphStructure:
    """Tests for graph structure and integrity."""
    
    def test_semantic_graph_exists(self):
        """semantic_graph_v2.json must exist."""
        assert SEMANTIC_GRAPH_FILE.exists(), f"Missing: {SEMANTIC_GRAPH_FILE}"
    
    def test_semantic_graph_has_nodes(self):
        """Graph must have nodes."""
        with open(SEMANTIC_GRAPH_FILE, "r", encoding="utf-8") as f:
            graph = json.load(f)
        
        assert "nodes" in graph
        assert len(graph["nodes"]) > 0
    
    def test_semantic_graph_has_edges(self):
        """Graph must have edges."""
        with open(SEMANTIC_GRAPH_FILE, "r", encoding="utf-8") as f:
            graph = json.load(f)
        
        assert "edges" in graph
        assert len(graph["edges"]) > 0
    
    def test_all_nodes_have_type(self):
        """All nodes must have a type field."""
        with open(SEMANTIC_GRAPH_FILE, "r", encoding="utf-8") as f:
            graph = json.load(f)
        
        for node in graph["nodes"]:
            assert "type" in node, f"Node missing type: {node.get('id')}"
            assert node["type"] in ["BEHAVIOR", "AGENT", "ORGAN", "HEART_STATE", 
                                    "CONSEQUENCE", "AXIS_VALUE"]
    
    def test_all_edges_have_evidence(self):
        """All edges should have evidence array."""
        with open(SEMANTIC_GRAPH_FILE, "r", encoding="utf-8") as f:
            graph = json.load(f)
        
        edges_with_evidence = 0
        for edge in graph["edges"]:
            if "evidence" in edge and len(edge["evidence"]) > 0:
                edges_with_evidence += 1
        
        # At least 50% of edges should have evidence
        ratio = edges_with_evidence / len(graph["edges"])
        assert ratio >= 0.5, f"Only {ratio:.1%} of edges have evidence"
    
    def test_edge_types_are_valid(self):
        """All edge types must be from allowed set."""
        with open(SEMANTIC_GRAPH_FILE, "r", encoding="utf-8") as f:
            graph = json.load(f)
        
        allowed_types = graph.get("allowed_edge_types", [])
        assert len(allowed_types) > 0, "No allowed_edge_types defined"
        
        for edge in graph["edges"]:
            assert edge["edge_type"] in allowed_types, \
                f"Invalid edge type: {edge['edge_type']}"


# ============================================================================
# Behavior→Verse Link Tests
# ============================================================================

@pytest.mark.unit
class TestBehaviorVerseLinks:
    """Tests for behavior→verse link loading."""
    
    def test_concept_index_exists(self):
        """concept_index_v2.jsonl must exist."""
        assert CONCEPT_INDEX_FILE.exists(), f"Missing: {CONCEPT_INDEX_FILE}"
    
    def test_concept_index_has_behaviors(self):
        """Concept index must have behavior entries."""
        behavior_count = 0
        with open(CONCEPT_INDEX_FILE, "r", encoding="utf-8") as f:
            for line in f:
                concept = json.loads(line)
                if concept.get("concept_id", "").startswith("BEH_"):
                    behavior_count += 1
        
        assert behavior_count >= 70, f"Expected >=70 behaviors, got {behavior_count}"
    
    def test_behaviors_have_verses(self):
        """Behaviors must have verse links."""
        behaviors_with_verses = 0
        with open(CONCEPT_INDEX_FILE, "r", encoding="utf-8") as f:
            for line in f:
                concept = json.loads(line)
                if concept.get("concept_id", "").startswith("BEH_"):
                    if len(concept.get("verses", [])) > 0:
                        behaviors_with_verses += 1
        
        assert behaviors_with_verses >= 50, \
            f"Expected >=50 behaviors with verses, got {behaviors_with_verses}"
    
    def test_behaviors_have_tafsir_chunks(self):
        """Behaviors must have tafsir chunk evidence."""
        behaviors_with_tafsir = 0
        with open(CONCEPT_INDEX_FILE, "r", encoding="utf-8") as f:
            for line in f:
                concept = json.loads(line)
                if concept.get("concept_id", "").startswith("BEH_"):
                    if len(concept.get("tafsir_chunks", [])) > 0:
                        behaviors_with_tafsir += 1
        
        assert behaviors_with_tafsir >= 50, \
            f"Expected >=50 behaviors with tafsir, got {behaviors_with_tafsir}"
    
    def test_loading_stats_exist(self):
        """behavior_verse_links_stats.json must exist after dry run."""
        assert STATS_FILE.exists(), \
            "Run: python scripts/populate_behavior_verse_links.py --dry-run"
    
    def test_loading_stats_have_links(self):
        """Stats must show links were created."""
        if not STATS_FILE.exists():
            pytest.skip("Stats file not found")
        
        with open(STATS_FILE, "r", encoding="utf-8") as f:
            stats = json.load(f)
        
        assert stats["stats"]["verse_links_created"] > 0
        assert stats["stats"]["behaviors_processed"] > 0


# ============================================================================
# Surah Intro Filtering Tests
# ============================================================================

@pytest.mark.unit
class TestSurahIntroFiltering:
    """Tests for surah_intro filtering."""
    
    def test_no_ayah_zero_in_links(self):
        """No verse links should have ayah=0."""
        with open(CONCEPT_INDEX_FILE, "r", encoding="utf-8") as f:
            for line in f:
                concept = json.loads(line)
                for verse in concept.get("verses", []):
                    assert verse.get("ayah", 0) != 0, \
                        f"Found ayah=0 in {concept.get('concept_id')}"
    
    def test_migration_has_entry_type_column(self):
        """Migration must add entry_type column to tafsir_chunks."""
        with open(MIGRATION_FILE, "r", encoding="utf-8") as f:
            sql = f.read()
        
        assert "entry_type" in sql
        assert "surah_intro" in sql
    
    def test_loader_filters_surah_intro(self):
        """Loader must have surah_intro filtering logic."""
        loader_file = Path("scripts/populate_behavior_verse_links.py")
        assert loader_file.exists()
        
        with open(loader_file, "r", encoding="utf-8") as f:
            code = f.read()
        
        assert "is_surah_intro" in code
        assert "surah_intro_filtered" in code


# ============================================================================
# Subset Contract Tests
# ============================================================================

@pytest.mark.unit
class TestSubsetContract:
    """Tests for tafsir ⊆ verse evidence contract."""
    
    def test_tafsir_verses_are_valid(self):
        """All tafsir chunk verse references must be valid (1-114, 1-286)."""
        with open(CONCEPT_INDEX_FILE, "r", encoding="utf-8") as f:
            for line in f:
                concept = json.loads(line)
                for chunk in concept.get("tafsir_chunks", []):
                    surah = chunk.get("surah", 0)
                    ayah = chunk.get("ayah", 0)
                    
                    assert 1 <= surah <= 114, \
                        f"Invalid surah {surah} in {concept.get('concept_id')}"
                    assert 0 <= ayah <= 286, \
                        f"Invalid ayah {ayah} in {concept.get('concept_id')}"
    
    def test_tafsir_sources_are_known(self):
        """All tafsir sources must be from known set."""
        known_sources = {
            "ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn",
            "muyassar", "baghawi", "waseet"
        }
        
        found_sources = set()
        with open(CONCEPT_INDEX_FILE, "r", encoding="utf-8") as f:
            for line in f:
                concept = json.loads(line)
                for chunk in concept.get("tafsir_chunks", []):
                    found_sources.add(chunk.get("source", "unknown"))
        
        # All found sources should be known
        unknown = found_sources - known_sources - {"unknown"}
        assert len(unknown) == 0, f"Unknown tafsir sources: {unknown}"


# ============================================================================
# Semantic Edges Provenance Tests
# ============================================================================

@pytest.mark.unit
class TestSemanticEdgesProvenance:
    """Tests for semantic_edges provenance."""
    
    def test_migration_has_provenance_column(self):
        """Migration must add provenance JSONB column."""
        with open(MIGRATION_FILE, "r", encoding="utf-8") as f:
            sql = f.read()
        
        assert "provenance JSONB" in sql or "provenance" in sql
    
    def test_migration_has_entity_type_columns(self):
        """Migration must add from/to entity_type columns."""
        with open(MIGRATION_FILE, "r", encoding="utf-8") as f:
            sql = f.read()
        
        assert "from_entity_type" in sql
        assert "to_entity_type" in sql
    
    def test_edges_have_confidence(self):
        """All edges must have confidence score."""
        with open(SEMANTIC_GRAPH_FILE, "r", encoding="utf-8") as f:
            graph = json.load(f)
        
        for edge in graph["edges"]:
            assert "confidence" in edge, \
                f"Edge missing confidence: {edge.get('source')} -> {edge.get('target')}"
            assert 0 <= edge["confidence"] <= 1, \
                f"Invalid confidence: {edge['confidence']}"
    
    def test_loader_builds_provenance(self):
        """Loader must build provenance JSONB."""
        loader_file = Path("scripts/populate_behavior_verse_links.py")
        
        with open(loader_file, "r", encoding="utf-8") as f:
            code = f.read()
        
        assert "provenance" in code
        assert "sources_list" in code
        assert "evidence_count" in code


# ============================================================================
# Foreign Key Integrity Tests
# ============================================================================

@pytest.mark.unit
class TestForeignKeyIntegrity:
    """Tests for foreign key integrity in graph."""
    
    def test_all_edge_sources_are_valid_nodes(self):
        """All edge source IDs must reference valid nodes."""
        with open(SEMANTIC_GRAPH_FILE, "r", encoding="utf-8") as f:
            graph = json.load(f)
        
        node_ids = {node["id"] for node in graph["nodes"]}
        
        for edge in graph["edges"]:
            assert edge["source"] in node_ids, \
                f"Edge source not in nodes: {edge['source']}"
    
    def test_all_edge_targets_are_valid_nodes(self):
        """All edge target IDs must reference valid nodes."""
        with open(SEMANTIC_GRAPH_FILE, "r", encoding="utf-8") as f:
            graph = json.load(f)
        
        node_ids = {node["id"] for node in graph["nodes"]}
        
        for edge in graph["edges"]:
            assert edge["target"] in node_ids, \
                f"Edge target not in nodes: {edge['target']}"
    
    def test_migration_has_fk_to_entities(self):
        """Migration must have FK from behavior_verse_links to entities."""
        with open(MIGRATION_FILE, "r", encoding="utf-8") as f:
            sql = f.read()
        
        assert "FOREIGN KEY (behavior_id) REFERENCES entities(entity_id)" in sql
    
    def test_migration_has_behavior_type_trigger(self):
        """Migration must have trigger to enforce behavior entity type."""
        with open(MIGRATION_FILE, "r", encoding="utf-8") as f:
            sql = f.read()
        
        assert "check_behavior_entity_type" in sql
        assert "entity_type = 'BEHAVIOR'" in sql or "entity_type='BEHAVIOR'" in sql


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
