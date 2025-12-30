#!/usr/bin/env python3
"""
Phase 3 Tests: Precomputed Knowledge Base Build

Tests for:
1. KB build reproducibility (same inputs → same hashes)
2. Patience anchor (الصبر returns expected anchors)
3. Manifest integrity with SHA256 hashes
4. Behavior dossier completeness

Run with: pytest tests/test_kb_phase3.py -v
"""

import json
import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

KB_DIR = Path("data/kb")
MANIFEST_FILE = KB_DIR / "manifest.json"
DOSSIERS_FILE = KB_DIR / "behavior_dossiers.jsonl"
VERSES_FILE = KB_DIR / "verses.jsonl"
BEHAVIORS_FILE = KB_DIR / "behaviors.jsonl"
LINKS_FILE = KB_DIR / "behavior_verse_links.jsonl"


def file_hash(path: Path) -> str:
    """Compute SHA256 hash of file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


# ============================================================================
# KB Build Reproducibility Tests
# ============================================================================

@pytest.mark.unit
class TestKBReproducibility:
    """Tests for KB build reproducibility."""
    
    def test_manifest_exists(self):
        """manifest.json must exist."""
        assert MANIFEST_FILE.exists(), f"Missing: {MANIFEST_FILE}"
    
    def test_manifest_version_2(self):
        """Manifest must be version 2.0 with hashes."""
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        
        assert manifest.get("version") == "2.0", "Manifest must be version 2.0"
    
    def test_manifest_has_input_hashes(self):
        """Manifest must have input_hashes section."""
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        
        assert "input_hashes" in manifest, "Missing input_hashes"
        assert len(manifest["input_hashes"]) >= 3, "Expected at least 3 input hashes"
    
    def test_manifest_has_output_hashes(self):
        """Manifest must have output_hashes section."""
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        
        assert "output_hashes" in manifest, "Missing output_hashes"
        assert len(manifest["output_hashes"]) >= 5, "Expected at least 5 output hashes"
    
    def test_output_hashes_match_files(self):
        """Output hashes must match actual file hashes."""
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        
        for rel_path, expected_hash in manifest.get("output_hashes", {}).items():
            # Handle Windows path separators
            file_path = Path(rel_path.replace("\\", "/"))
            if not file_path.exists():
                file_path = Path("data/kb") / file_path.name
            
            if file_path.exists():
                actual_hash = file_hash(file_path)
                assert actual_hash == expected_hash, \
                    f"Hash mismatch for {file_path}: expected {expected_hash[:16]}..., got {actual_hash[:16]}..."
    
    def test_manifest_has_git_commit(self):
        """Manifest must have git_commit for provenance."""
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        
        assert "git_commit" in manifest, "Missing git_commit"
        assert len(manifest["git_commit"]) == 40, "git_commit must be 40-char SHA"
    
    def test_manifest_has_build_args(self):
        """Manifest must have build_args for reproducibility."""
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        
        assert "build_args" in manifest, "Missing build_args"
        assert "seed" in manifest["build_args"], "Missing seed in build_args"


# ============================================================================
# Patience Anchor Tests
# ============================================================================

@pytest.mark.unit
class TestPatienceAnchor:
    """Tests for الصبر (patience) behavior anchor."""
    
    def test_dossiers_file_exists(self):
        """behavior_dossiers.jsonl must exist."""
        assert DOSSIERS_FILE.exists(), f"Missing: {DOSSIERS_FILE}"
    
    def test_patience_dossier_exists(self):
        """Patience behavior dossier must exist."""
        patience_found = False
        with open(DOSSIERS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                dossier = json.loads(line)
                if "صبر" in dossier.get("term_ar", ""):
                    patience_found = True
                    break
        
        assert patience_found, "Patience (صبر) dossier not found"
    
    def test_patience_has_verses(self):
        """Patience dossier must have verses."""
        with open(DOSSIERS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                dossier = json.loads(line)
                if "صبر" in dossier.get("term_ar", ""):
                    assert dossier.get("verse_count", 0) > 0, \
                        "Patience dossier has no verses"
                    assert len(dossier.get("verses", [])) > 0, \
                        "Patience dossier verses list is empty"
                    return
        
        pytest.fail("Patience dossier not found")
    
    def test_patience_has_key_verses(self):
        """Patience dossier must include key patience verses."""
        key_verses = ["2:45", "2:153", "3:200", "103:3"]
        
        with open(DOSSIERS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                dossier = json.loads(line)
                if "صبر" in dossier.get("term_ar", ""):
                    verse_keys = {v.get("verse_key") for v in dossier.get("verses", [])}
                    
                    found_keys = [k for k in key_verses if k in verse_keys]
                    # At least 2 of the key verses should be present
                    assert len(found_keys) >= 2, \
                        f"Expected at least 2 key patience verses, found: {found_keys}"
                    return
        
        pytest.fail("Patience dossier not found")
    
    def test_patience_verse_count_in_range(self):
        """Patience verse count should be in expected range."""
        with open(DOSSIERS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                dossier = json.loads(line)
                if "صبر" in dossier.get("term_ar", ""):
                    count = dossier.get("verse_count", 0)
                    # Expected: 80-200 verses with patience
                    assert 50 <= count <= 300, \
                        f"Patience verse count {count} outside expected range [50, 300]"
                    return
        
        pytest.fail("Patience dossier not found")


# ============================================================================
# Behavior Dossier Completeness Tests
# ============================================================================

@pytest.mark.unit
class TestDossierCompleteness:
    """Tests for behavior dossier completeness."""
    
    def test_dossier_count_matches_manifest(self):
        """Dossier count must match manifest."""
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        
        expected_count = manifest.get("counts", {}).get("behavior_dossiers", 0)
        
        actual_count = 0
        with open(DOSSIERS_FILE, "r", encoding="utf-8") as f:
            for _ in f:
                actual_count += 1
        
        assert actual_count == expected_count, \
            f"Dossier count mismatch: manifest={expected_count}, actual={actual_count}"
    
    def test_all_dossiers_have_required_fields(self):
        """All dossiers must have required fields."""
        required_fields = ["behavior_id", "term_ar", "term_en", "category", "verse_count"]
        
        with open(DOSSIERS_FILE, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                dossier = json.loads(line)
                for field in required_fields:
                    assert field in dossier, \
                        f"Dossier {i} missing field: {field}"
    
    def test_all_dossiers_have_verses_list(self):
        """All dossiers must have verses list."""
        with open(DOSSIERS_FILE, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                dossier = json.loads(line)
                assert "verses" in dossier, f"Dossier {i} missing verses list"
                assert isinstance(dossier["verses"], list), \
                    f"Dossier {i} verses is not a list"
    
    def test_verse_count_matches_verses_list(self):
        """verse_count must match length of verses list."""
        with open(DOSSIERS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                dossier = json.loads(line)
                count = dossier.get("verse_count", 0)
                verses = dossier.get("verses", [])
                assert count == len(verses), \
                    f"Dossier {dossier.get('behavior_id')}: verse_count={count}, len(verses)={len(verses)}"


# ============================================================================
# KB File Integrity Tests
# ============================================================================

@pytest.mark.unit
class TestKBFileIntegrity:
    """Tests for KB file integrity."""
    
    def test_verses_file_exists(self):
        """verses.jsonl must exist."""
        assert VERSES_FILE.exists(), f"Missing: {VERSES_FILE}"
    
    def test_behaviors_file_exists(self):
        """behaviors.jsonl must exist."""
        assert BEHAVIORS_FILE.exists(), f"Missing: {BEHAVIORS_FILE}"
    
    def test_links_file_exists(self):
        """behavior_verse_links.jsonl must exist."""
        assert LINKS_FILE.exists(), f"Missing: {LINKS_FILE}"
    
    def test_verses_count_is_6236(self):
        """verses.jsonl must have 6236 verses."""
        count = 0
        with open(VERSES_FILE, "r", encoding="utf-8") as f:
            for _ in f:
                count += 1
        
        assert count == 6236, f"Expected 6236 verses, got {count}"
    
    def test_behaviors_count_is_73(self):
        """behaviors.jsonl must have 73 behaviors."""
        count = 0
        with open(BEHAVIORS_FILE, "r", encoding="utf-8") as f:
            for _ in f:
                count += 1
        
        assert count == 73, f"Expected 73 behaviors, got {count}"
    
    def test_links_count_matches_manifest(self):
        """behavior_verse_links count must match manifest."""
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        
        expected = manifest.get("counts", {}).get("behavior_verse_links", 0)
        
        count = 0
        with open(LINKS_FILE, "r", encoding="utf-8") as f:
            for _ in f:
                count += 1
        
        assert count == expected, f"Links count mismatch: manifest={expected}, actual={count}"
    
    def test_all_kb_files_are_valid_jsonl(self):
        """All KB files must be valid JSONL."""
        kb_files = [VERSES_FILE, BEHAVIORS_FILE, LINKS_FILE, DOSSIERS_FILE]
        
        for kb_file in kb_files:
            if not kb_file.exists():
                continue
            
            with open(kb_file, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    try:
                        json.loads(line)
                    except json.JSONDecodeError as e:
                        pytest.fail(f"{kb_file.name} line {i+1}: Invalid JSON - {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
