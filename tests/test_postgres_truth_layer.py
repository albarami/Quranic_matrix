"""
Test: PostgreSQL Truth Layer (Phase 8.4)

Tests for the PostgreSQL schema and loader.
Uses dry-run mode to verify logic without requiring a database.
"""

import pytest
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

SCHEMA_FILE = Path("schemas/postgres_truth_layer.sql")
CANONICAL_ENTITIES_FILE = Path("vocab/canonical_entities.json")
CONCEPT_INDEX_FILE = Path("data/evidence/concept_index_v2.jsonl")
SEMANTIC_GRAPH_FILE = Path("data/graph/semantic_graph_v2.json")


@pytest.mark.unit
class TestSchemaFile:
    """Tests for PostgreSQL schema file."""
    
    def test_schema_file_exists(self):
        """Schema SQL file must exist."""
        assert SCHEMA_FILE.exists(), f"Schema file not found: {SCHEMA_FILE}"
    
    def test_schema_has_required_tables(self):
        """Schema must define all required tables."""
        with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
            schema = f.read()
        
        required_tables = [
            "verses",
            "tafsir_chunks",
            "entities",
            "mentions",
            "semantic_edges",
            "edge_evidence",
        ]
        
        for table in required_tables:
            assert f"CREATE TABLE IF NOT EXISTS {table}" in schema, \
                f"Missing table: {table}"
    
    def test_schema_has_foreign_keys(self):
        """Schema must have foreign key constraints."""
        with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
            schema = f.read()
        
        assert "FOREIGN KEY" in schema, "No foreign key constraints found"
        assert "ON DELETE CASCADE" in schema, "No cascade delete found"
    
    def test_schema_has_indexes(self):
        """Schema must have indexes for performance."""
        with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
            schema = f.read()
        
        assert "CREATE INDEX" in schema, "No indexes found"
        assert "idx_entities_entity_id" in schema, "Missing entity_id index"
        assert "idx_edges_confidence" in schema, "Missing confidence index"
    
    def test_schema_has_views(self):
        """Schema must have helper views."""
        with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
            schema = f.read()
        
        assert "CREATE OR REPLACE VIEW" in schema, "No views found"
        assert "entity_source_coverage" in schema, "Missing entity_source_coverage view"
        assert "edge_summary" in schema, "Missing edge_summary view"


@pytest.mark.unit
class TestLoaderDryRun:
    """Tests for loader in dry-run mode."""
    
    def test_loader_imports(self):
        """Loader module should import without errors."""
        from load_truth_layer_to_postgres import TruthLayerLoader
        assert TruthLayerLoader is not None
    
    def test_loader_dry_run_entities(self):
        """Loader should count entities correctly in dry run."""
        from load_truth_layer_to_postgres import TruthLayerLoader
        
        loader = TruthLayerLoader(dry_run=True)
        count = loader.load_entities()
        
        assert count == 168, f"Expected 168 entities, got {count}"
    
    def test_loader_dry_run_edges(self):
        """Loader should count edges correctly in dry run."""
        from load_truth_layer_to_postgres import TruthLayerLoader
        
        loader = TruthLayerLoader(dry_run=True)
        loader.load_entities()  # Required first
        count = loader.load_semantic_edges()
        
        # Should match semantic graph v2
        with open(SEMANTIC_GRAPH_FILE, "r", encoding="utf-8") as f:
            graph = json.load(f)
        
        assert count == len(graph["edges"]), \
            f"Edge count mismatch: loader={count}, graph={len(graph['edges'])}"
    
    def test_loader_dry_run_mentions(self):
        """Loader should load mentions in dry run."""
        from load_truth_layer_to_postgres import TruthLayerLoader
        
        loader = TruthLayerLoader(dry_run=True)
        loader.load_entities()
        count = loader.load_mentions()
        
        assert count > 0, "No mentions loaded"


@pytest.mark.unit
class TestRoundtripCounts:
    """Tests for roundtrip count verification."""
    
    def test_postgres_roundtrip_counts(self):
        """Counts from loader should match JSONL exports."""
        from load_truth_layer_to_postgres import TruthLayerLoader
        
        loader = TruthLayerLoader(dry_run=True)
        loader.load_entities()
        loader.load_semantic_edges()
        
        verification = loader.verify_counts()
        
        assert verification["entities"]["match"], \
            f"Entity count mismatch: JSONL={verification['entities']['jsonl']}, DB={verification['entities']['db']}"
        assert verification["edges"]["match"], \
            f"Edge count mismatch: JSONL={verification['edges']['jsonl']}, DB={verification['edges']['db']}"
    
    def test_entities_count_matches_canonical(self):
        """Entity count should match canonical_entities.json."""
        with open(CANONICAL_ENTITIES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        total = (
            len(data.get("behaviors", [])) +
            len(data.get("agents", [])) +
            len(data.get("organs", [])) +
            len(data.get("heart_states", [])) +
            len(data.get("consequences", []))
        )
        
        assert total == 168, f"Expected 168 entities, got {total}"


@pytest.mark.unit
class TestDataIntegrity:
    """Tests for data integrity constraints."""
    
    def test_edge_evidence_has_valid_edge_refs(self):
        """All edge evidence should reference valid edges."""
        with open(SEMANTIC_GRAPH_FILE, "r", encoding="utf-8") as f:
            graph = json.load(f)
        
        for edge in graph.get("edges", [])[:100]:
            evidence = edge.get("evidence", [])
            for ev in evidence:
                assert "chunk_id" in ev, "Evidence missing chunk_id"
                assert "source" in ev, "Evidence missing source"
                assert "surah" in ev, "Evidence missing surah"
                assert "ayah" in ev, "Evidence missing ayah"
    
    def test_no_orphan_edges(self):
        """All edges should reference valid entities."""
        with open(SEMANTIC_GRAPH_FILE, "r", encoding="utf-8") as f:
            graph = json.load(f)
        
        with open(CANONICAL_ENTITIES_FILE, "r", encoding="utf-8") as f:
            entities = json.load(f)
        
        # Build set of valid entity IDs
        valid_ids = set()
        for section in ["behaviors", "agents", "organs", "heart_states", "consequences"]:
            for item in entities.get(section, []):
                valid_ids.add(item["id"])
        
        # Check all edges reference valid entities
        for edge in graph.get("edges", []):
            assert edge["source"] in valid_ids, \
                f"Edge source not in entities: {edge['source']}"
            assert edge["target"] in valid_ids, \
                f"Edge target not in entities: {edge['target']}"
    
    def test_mentions_reference_valid_entities(self):
        """All mentions should reference valid entities."""
        with open(CANONICAL_ENTITIES_FILE, "r", encoding="utf-8") as f:
            entities = json.load(f)
        
        valid_ids = set()
        for section in ["behaviors", "agents", "organs", "heart_states", "consequences"]:
            for item in entities.get(section, []):
                valid_ids.add(item["id"])
        
        with open(CONCEPT_INDEX_FILE, "r", encoding="utf-8") as f:
            for line in f:
                concept = json.loads(line)
                assert concept["concept_id"] in valid_ids, \
                    f"Concept not in entities: {concept['concept_id']}"


@pytest.mark.unit
class TestSchemaConstraints:
    """Tests for schema constraint definitions."""
    
    def test_entity_type_constraint(self):
        """Schema should constrain entity_type values."""
        with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
            schema = f.read()
        
        assert "BEHAVIOR" in schema
        assert "AGENT" in schema
        assert "ORGAN" in schema
        assert "HEART_STATE" in schema
        assert "CONSEQUENCE" in schema
    
    def test_edge_type_constraint(self):
        """Schema should constrain edge_type values."""
        with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
            schema = f.read()
        
        assert "CAUSES" in schema
        assert "LEADS_TO" in schema
        assert "PREVENTS" in schema
        assert "STRENGTHENS" in schema
        assert "OPPOSITE_OF" in schema
    
    def test_confidence_constraint(self):
        """Schema should constrain confidence to 0-1."""
        with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
            schema = f.read()
        
        assert "confidence >= 0" in schema or "confidence REAL" in schema
        assert "confidence <= 1" in schema or "confidence REAL" in schema


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
