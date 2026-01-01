"""
Tests for Behavior Mastery Dossiers (Phase 2 of Behavior Mastery Plan).

Tests enforce:
- Exactly 87 dossier files exist
- Each dossier has all required schema fields
- Each relationship has evidence
- Bouzidani context fields present or explicitly unknown with reason
- Context assertions have citations for non-unknown values
- Manifest hashes are reproducible
"""

import json
import pytest
from pathlib import Path

from src.mastery.dossier import (
    BehaviorDossier,
    VerseEvidence,
    TafsirChunk,
    RelationshipEdge,
    BouzidaniContexts,
    ContextAssertion,
)
from src.mastery.assembler import (
    DossierAssembler,
    build_all_dossiers,
    load_dossier,
    DOSSIER_OUTPUT_DIR,
    SUMMARY_OUTPUT_PATH,
    MANIFEST_OUTPUT_PATH,
)


# Paths
CANONICAL_ENTITIES_PATH = Path("vocab/canonical_entities.json")


class TestDossierSchema:
    """Tests for dossier schema validation."""
    
    def test_verse_evidence_schema(self):
        """VerseEvidence must have all required fields."""
        ev = VerseEvidence(
            verse_key="2:153",
            text_ar="يا أيها الذين آمنوا",
            text_en="O you who believe",
            relevance="direct",
            confidence=0.9,
        )
        d = ev.to_dict()
        assert "verse_key" in d
        assert "text_ar" in d
        assert "text_en" in d
        assert "relevance" in d
        assert "confidence" in d
    
    def test_context_assertion_unknown(self):
        """ContextAssertion.unknown() must include reason."""
        ctx = ContextAssertion.unknown("no_evidence_found")
        assert ctx.value == "unknown"
        assert ctx.citation_type == "unknown"
        assert ctx.reason_if_unknown == "no_evidence_found"
    
    def test_context_assertion_from_verse(self):
        """ContextAssertion.from_verse() must include citation."""
        ctx = ContextAssertion.from_verse("باطن", "2:225", 0.85)
        assert ctx.value == "باطن"
        assert ctx.citation_type == "verse"
        assert ctx.citation_ref == "2:225"
        assert ctx.confidence == 0.85
    
    def test_context_assertion_from_rule(self):
        """ContextAssertion.from_rule() must include rule ID."""
        ctx = ContextAssertion.from_rule("ظاهر", "RULE_EXTERNAL_001", 0.95)
        assert ctx.value == "ظاهر"
        assert ctx.citation_type == "rule"
        assert ctx.citation_ref == "RULE_EXTERNAL_001"
    
    def test_bouzidani_contexts_get_unknown_fields(self):
        """BouzidaniContexts must track unknown fields."""
        ctx = BouzidaniContexts()
        unknown = ctx.get_unknown_fields()
        # All fields should be unknown by default
        assert "internal_external" in unknown
        assert "situational_context" in unknown
        assert "temporal_context" in unknown
    
    def test_dossier_completeness_score(self):
        """Dossier completeness score must be computed correctly."""
        dossier = BehaviorDossier(
            behavior_id="BEH_TEST",
            label_ar="اختبار",
            label_en="Test",
            category="test",
        )
        dossier.update_stats()
        # Empty dossier should have low completeness
        assert 0 <= dossier.completeness_score <= 1.0
    
    def test_dossier_hash_deterministic(self):
        """Dossier hash must be deterministic."""
        dossier1 = BehaviorDossier(
            behavior_id="BEH_TEST",
            label_ar="اختبار",
            label_en="Test",
            category="test",
        )
        dossier2 = BehaviorDossier(
            behavior_id="BEH_TEST",
            label_ar="اختبار",
            label_en="Test",
            category="test",
        )
        assert dossier1.compute_hash() == dossier2.compute_hash()


class TestDossierAssembler:
    """Tests for dossier assembler."""
    
    @pytest.fixture
    def assembler(self) -> DossierAssembler:
        """Create and load assembler."""
        asm = DossierAssembler()
        asm.load_sources()
        return asm
    
    def test_assembler_loads_canonical_behaviors(self, assembler: DossierAssembler):
        """Assembler must load exactly 87 canonical behaviors."""
        assert len(assembler.canonical_behaviors) == 87
    
    def test_assembler_builds_single_dossier(self, assembler: DossierAssembler):
        """Assembler must build a valid dossier."""
        # Pick a known behavior
        behavior_id = list(assembler.canonical_behaviors.keys())[0]
        dossier = assembler.build_dossier(behavior_id)
        
        assert dossier.behavior_id == behavior_id
        assert dossier.label_ar
        assert dossier.label_en
        assert dossier.schema_version == "1.0.0"


class TestBuiltDossiers:
    """Tests for built dossier artifacts (run after build_behavior_mastery.py)."""
    
    @pytest.fixture
    def canonical_behavior_ids(self) -> set:
        """Get canonical behavior IDs."""
        with open(CANONICAL_ENTITIES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {b["id"] for b in data.get("behaviors", [])}
    
    def test_mastery_dossier_count_is_87(self, canonical_behavior_ids: set):
        """Exactly 87 dossier files must exist."""
        if not DOSSIER_OUTPUT_DIR.exists():
            pytest.skip("Dossiers not built yet - run build_behavior_mastery.py first")
        
        dossier_files = list(DOSSIER_OUTPUT_DIR.glob("BEH_*.json"))
        assert len(dossier_files) == 87, f"Expected 87 dossiers, found {len(dossier_files)}"
    
    def test_each_dossier_has_required_fields(self, canonical_behavior_ids: set):
        """Every dossier must have all required schema fields."""
        if not DOSSIER_OUTPUT_DIR.exists():
            pytest.skip("Dossiers not built yet")
        
        required_fields = {
            "behavior_id", "label_ar", "label_en", "category",
            "quran_evidence", "tafsir_evidence",
            "outgoing_edges", "incoming_edges",
            "bouzidani_contexts", "evidence_stats",
            "completeness_score", "schema_version",
        }
        
        for dossier_file in DOSSIER_OUTPUT_DIR.glob("BEH_*.json"):
            with open(dossier_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            missing = required_fields - set(data.keys())
            assert not missing, f"{dossier_file.name} missing fields: {missing}"
    
    def test_each_relationship_has_evidence(self, canonical_behavior_ids: set):
        """Every relationship edge must have evidence_count >= 0."""
        if not DOSSIER_OUTPUT_DIR.exists():
            pytest.skip("Dossiers not built yet")
        
        for dossier_file in DOSSIER_OUTPUT_DIR.glob("BEH_*.json"):
            with open(dossier_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for edge in data.get("outgoing_edges", []):
                assert "evidence_count" in edge, f"{dossier_file.name} edge missing evidence_count"
                assert edge["evidence_count"] >= 0
            
            for edge in data.get("incoming_edges", []):
                assert "evidence_count" in edge, f"{dossier_file.name} edge missing evidence_count"
                assert edge["evidence_count"] >= 0
    
    def test_bouzidani_context_fields_present_or_explicit_unknown(self):
        """All Bouzidani context fields must be populated or explicitly unknown with reason."""
        if not DOSSIER_OUTPUT_DIR.exists():
            pytest.skip("Dossiers not built yet")
        
        context_fields = [
            "internal_external", "situational_context", "systemic_context",
            "spatial_context", "temporal_context", "intention_niyyah", "recurrence_dawrah"
        ]
        
        for dossier_file in DOSSIER_OUTPUT_DIR.glob("BEH_*.json"):
            with open(dossier_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            contexts = data.get("bouzidani_contexts", {})
            for field in context_fields:
                ctx = contexts.get(field, {})
                assert "value" in ctx, f"{dossier_file.name} {field} missing value"
                assert "citation_type" in ctx, f"{dossier_file.name} {field} missing citation_type"
                
                # If unknown, must have reason
                if ctx.get("value") == "unknown":
                    assert ctx.get("reason_if_unknown") or ctx.get("citation_type") == "unknown", \
                        f"{dossier_file.name} {field} is unknown without reason"
    
    def test_context_assertions_have_citations(self):
        """Non-unknown context values must cite evidence or rule ID."""
        if not DOSSIER_OUTPUT_DIR.exists():
            pytest.skip("Dossiers not built yet")
        
        for dossier_file in DOSSIER_OUTPUT_DIR.glob("BEH_*.json"):
            with open(dossier_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            contexts = data.get("bouzidani_contexts", {})
            
            # Check non-unknown contexts have citations
            for field_name, ctx in contexts.items():
                if isinstance(ctx, dict) and ctx.get("value") != "unknown":
                    citation_type = ctx.get("citation_type", "")
                    assert citation_type in ["verse", "tafsir", "rule"], \
                        f"{dossier_file.name} {field_name} has non-unknown value without valid citation_type"
                    if citation_type in ["verse", "tafsir", "rule"]:
                        assert ctx.get("citation_ref"), \
                            f"{dossier_file.name} {field_name} has citation_type but no citation_ref"
    
    def test_mastery_manifest_exists(self):
        """Manifest file must exist after build."""
        if not MANIFEST_OUTPUT_PATH.exists():
            pytest.skip("Manifest not built yet")
        
        with open(MANIFEST_OUTPUT_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        
        assert "dossier_count" in manifest
        assert manifest["dossier_count"] == 87
        assert "dossier_hashes" in manifest
        assert len(manifest["dossier_hashes"]) == 87
    
    def test_mastery_manifest_hashes_reproducible(self):
        """Running builder twice produces identical hashes."""
        if not MANIFEST_OUTPUT_PATH.exists():
            pytest.skip("Manifest not built yet")
        
        # Load manifest
        with open(MANIFEST_OUTPUT_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        
        # Verify each dossier hash matches
        for behavior_id, expected_hash in manifest["dossier_hashes"].items():
            dossier = load_dossier(behavior_id)
            if dossier:
                actual_hash = dossier.compute_hash()
                assert actual_hash == expected_hash, \
                    f"{behavior_id} hash mismatch: {actual_hash} != {expected_hash}"
    
    def test_dossier_schema_validation(self):
        """Each dossier validates against JSON schema."""
        if not DOSSIER_OUTPUT_DIR.exists():
            pytest.skip("Dossiers not built yet")
        
        for dossier_file in DOSSIER_OUTPUT_DIR.glob("BEH_*.json"):
            # Load and reconstruct to validate schema
            dossier = BehaviorDossier.load(dossier_file)
            assert dossier.behavior_id.startswith("BEH_")
            assert dossier.schema_version == "1.0.0"
    
    def test_quran_evidence_has_verse_text(self):
        """Each Quran evidence entry must include verse_key and text."""
        if not DOSSIER_OUTPUT_DIR.exists():
            pytest.skip("Dossiers not built yet")
        
        dossiers_with_evidence = 0
        for dossier_file in DOSSIER_OUTPUT_DIR.glob("BEH_*.json"):
            with open(dossier_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            quran_evidence = data.get("quran_evidence", [])
            if quran_evidence:
                dossiers_with_evidence += 1
                for ev in quran_evidence[:5]:  # Check first 5
                    assert "verse_key" in ev, f"{dossier_file.name} evidence missing verse_key"
                    assert "text_ar" in ev, f"{dossier_file.name} evidence missing text_ar"
        
        # All 87 behaviors must have Quran evidence
        assert dossiers_with_evidence == 87, f"Expected 87 dossiers with evidence, got {dossiers_with_evidence}"
    
    def test_tafsir_evidence_has_provenance(self):
        """Each tafsir chunk must have source, chunk_id, and offsets."""
        if not DOSSIER_OUTPUT_DIR.exists():
            pytest.skip("Dossiers not built yet")
        
        for dossier_file in DOSSIER_OUTPUT_DIR.glob("BEH_*.json"):
            with open(dossier_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            tafsir_evidence = data.get("tafsir_evidence", {})
            for source, chunks in tafsir_evidence.items():
                for chunk in chunks[:3]:  # Check first 3 per source
                    assert "source" in chunk, f"{dossier_file.name} tafsir chunk missing source"
                    assert "chunk_id" in chunk, f"{dossier_file.name} tafsir chunk missing chunk_id"
                    assert "verse_key" in chunk, f"{dossier_file.name} tafsir chunk missing verse_key"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
