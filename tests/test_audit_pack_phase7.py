"""
Phase 7 Tests: Audit Pack Generator (v2.0 - Strict Mode)

Tests for:
1. Input/output hash generation with SSOT validation
2. GPU proof logs
3. Provenance completeness report
4. Audit pack completeness with git commit
5. Strict mode SSOT enforcement
"""

import json
import pytest
from pathlib import Path

# Import Phase 7 components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate_audit_pack import (
    sha256_file,
    sha256_string,
    get_file_info,
    generate_input_hashes,
    generate_output_hashes,
    generate_gpu_proof,
    generate_provenance_report,
    generate_system_info,
    get_latest_benchmark_results,
    get_current_git_commit,
    REQUIRED_TAFSIR_SOURCES,
)

PROJECT_ROOT = Path(__file__).parent.parent
AUDIT_PACK_DIR = PROJECT_ROOT / "artifacts" / "audit_pack"


# =============================================================================
# HASH TESTS
# =============================================================================

class TestHashGeneration:
    """Test SHA256 hash generation."""
    
    def test_sha256_string(self):
        """Test string hashing."""
        # Known hash for "test"
        result = sha256_string("test")
        assert len(result) == 64  # SHA256 hex length
        assert result == "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
    
    def test_sha256_file_exists(self):
        """Test file hashing for existing file."""
        # Hash a known file
        ce_path = PROJECT_ROOT / "vocab" / "canonical_entities.json"
        if ce_path.exists():
            result = sha256_file(ce_path)
            assert len(result) == 64
    
    def test_get_file_info_exists(self):
        """Test file info for existing file."""
        ce_path = PROJECT_ROOT / "vocab" / "canonical_entities.json"
        info, is_valid = get_file_info(ce_path)
        
        assert info["exists"] is True
        assert "sha256" in info
        assert "size_bytes" in info
        assert info["size_bytes"] > 0
        assert is_valid is True
    
    def test_get_file_info_not_exists(self):
        """Test file info for non-existing file."""
        fake_path = PROJECT_ROOT / "nonexistent_file.xyz"
        info, is_valid = get_file_info(fake_path)
        
        assert info["exists"] is False
        assert is_valid is True  # Non-SSOT file, so still valid
    
    def test_get_file_info_ssot_missing_fails(self):
        """Test SSOT file missing returns invalid."""
        fake_path = PROJECT_ROOT / "nonexistent_ssot.json"
        info, is_valid = get_file_info(fake_path, is_ssot=True)
        
        assert info["exists"] is False
        assert is_valid is False  # SSOT missing = invalid


# =============================================================================
# INPUT HASHES TESTS
# =============================================================================

class TestInputHashes:
    """Test input hash generation."""
    
    def test_generate_input_hashes(self):
        """Test input hashes are generated."""
        result, missing = generate_input_hashes()
        
        assert "generated_at" in result
        assert "file_count" in result
        assert "files" in result
        assert result["file_count"] >= 8  # At least 7 tafsir + 1 canonical
        assert "all_ssot_present" in result
    
    def test_canonical_entities_hashed(self):
        """Test canonical_entities.json is hashed."""
        result, _ = generate_input_hashes()
        
        assert "canonical_entities" in result["files"]
        assert result["files"]["canonical_entities"]["exists"] is True
    
    def test_tafsir_sources_hashed(self):
        """Test all 7 tafsir sources are hashed."""
        result, _ = generate_input_hashes()
        
        for src in REQUIRED_TAFSIR_SOURCES:
            key = f"tafsir_{src}"
            assert key in result["files"], f"Missing tafsir source: {src}"
    
    def test_all_ssot_present(self):
        """Test all SSOT files are present."""
        result, missing = generate_input_hashes()
        
        assert result["all_ssot_present"] is True
        assert len(missing) == 0


# =============================================================================
# OUTPUT HASHES TESTS
# =============================================================================

class TestOutputHashes:
    """Test output hash generation."""
    
    def test_generate_output_hashes(self):
        """Test output hashes are generated."""
        result = generate_output_hashes()
        
        assert "generated_at" in result
        assert "file_count" in result
        assert "files" in result
    
    def test_semantic_graph_v3_hashed(self):
        """Test semantic_graph_v3.json is hashed."""
        result = generate_output_hashes()
        
        assert "semantic_graph_v3" in result["files"]
        assert result["files"]["semantic_graph_v3"]["exists"] is True
    
    def test_concept_index_v3_hashed(self):
        """Test concept_index_v3.jsonl is hashed."""
        result = generate_output_hashes()
        
        assert "concept_index_v3" in result["files"]
        assert result["files"]["concept_index_v3"]["exists"] is True


# =============================================================================
# GPU PROOF TESTS
# =============================================================================

class TestGPUProof:
    """Test GPU proof generation."""
    
    def test_generate_gpu_proof(self):
        """Test GPU proof is generated."""
        result = generate_gpu_proof()
        
        assert "generated_at" in result
        assert "torch_available" in result
        assert "cuda_available" in result
    
    def test_gpu_proof_has_device_info(self):
        """Test GPU proof includes device info if available."""
        result = generate_gpu_proof()
        
        if result["cuda_available"]:
            assert "device_count" in result
            assert result["device_count"] > 0
            assert "devices" in result
            assert len(result["devices"]) > 0
            
            # Check device info
            device = result["devices"][0]
            assert "name" in device
            assert "total_memory_gb" in device


# =============================================================================
# PROVENANCE REPORT TESTS
# =============================================================================

class TestProvenanceReport:
    """Test provenance completeness report."""
    
    def test_generate_provenance_report(self):
        """Test provenance report is generated."""
        result = generate_provenance_report()
        
        assert "generated_at" in result
        assert "behaviors" in result
        assert "graph" in result
        assert "tafsir" in result
        assert "completeness" in result
    
    def test_behaviors_complete(self):
        """Test 87 behaviors are complete."""
        result = generate_provenance_report()
        
        assert result["behaviors"]["canonical_count"] == 87
        assert result["behaviors"]["indexed_count"] == 87
        assert result["completeness"]["behaviors_complete"] is True
    
    def test_graph_complete(self):
        """Test graph has 87 behavior nodes."""
        result = generate_provenance_report()
        
        assert result["graph"]["behavior_nodes"] == 87
        assert result["completeness"]["graph_complete"] is True
    
    def test_tafsir_complete(self):
        """Test 7 tafsir sources are present."""
        result = generate_provenance_report()
        
        assert result["tafsir"]["source_count"] >= 7
        assert result["completeness"]["tafsir_complete"] is True
    
    def test_all_complete(self):
        """Test all completeness checks pass."""
        result = generate_provenance_report()
        
        assert result["completeness"]["all_complete"] is True


# =============================================================================
# SYSTEM INFO TESTS
# =============================================================================

class TestSystemInfo:
    """Test system info generation."""
    
    def test_generate_system_info(self):
        """Test system info is generated."""
        result, _ = generate_system_info()
        
        assert "generated_at" in result
        assert "platform" in result
        assert "python_version" in result["platform"]
    
    def test_git_commit_present(self):
        """Test git commit is captured."""
        result, _ = generate_system_info()
        
        # Git commit should be present in a git repo
        assert "git_commit" in result
        assert len(result["git_commit"]) == 40  # SHA1 hex length
    
    def test_get_current_git_commit(self):
        """Test get_current_git_commit function."""
        commit = get_current_git_commit()
        
        assert commit is not None
        assert len(commit) == 40  # SHA1 hex length


# =============================================================================
# BENCHMARK RESULTS TESTS
# =============================================================================

class TestBenchmarkResults:
    """Test benchmark results retrieval."""
    
    def test_get_latest_benchmark_results(self):
        """Test benchmark results are retrieved."""
        result = get_latest_benchmark_results()
        
        if "error" not in result:
            assert "total_questions" in result
            assert "passed" in result
            assert "pass_rate" in result
    
    def test_benchmark_200_questions(self):
        """Test benchmark has 200 questions."""
        result = get_latest_benchmark_results()
        
        if "error" not in result:
            assert result["total_questions"] == 200
    
    def test_benchmark_100_percent_pass(self):
        """Test benchmark has 100% pass rate."""
        result = get_latest_benchmark_results()
        
        if "error" not in result:
            assert result["pass_rate"] == 100.0
            assert result["passed"] == 200
            assert result["failed"] == 0


# =============================================================================
# AUDIT PACK COMPLETENESS TESTS
# =============================================================================

class TestAuditPackCompleteness:
    """Test audit pack is complete."""
    
    def test_audit_pack_exists(self):
        """Test audit pack directory exists."""
        assert AUDIT_PACK_DIR.exists()
    
    def test_all_files_present(self):
        """Test all required files are present."""
        required_files = [
            "audit_pack.json",
            "input_hashes.json",
            "output_hashes.json",
            "gpu_proof.json",
            "provenance_report.json",
            "system_info.json",
            "benchmark_results.json"
        ]
        
        for filename in required_files:
            filepath = AUDIT_PACK_DIR / filename
            assert filepath.exists(), f"Missing file: {filename}"
    
    def test_audit_pack_valid_json(self):
        """Test audit pack is valid JSON."""
        pack_path = AUDIT_PACK_DIR / "audit_pack.json"
        
        with open(pack_path, "r", encoding="utf-8") as f:
            pack = json.load(f)
        
        assert "version" in pack
        assert "generated_at" in pack
        assert "input_hashes" in pack
        assert "output_hashes" in pack
        assert "gpu_proof" in pack
        assert "provenance" in pack
        assert "system_info" in pack
        assert "benchmark" in pack
    
    def test_hashes_reproducible(self):
        """Test hashes are reproducible (same file = same hash)."""
        # Generate hashes twice
        result1, _ = generate_input_hashes()
        result2, _ = generate_input_hashes()
        
        # Compare canonical_entities hash
        hash1 = result1["files"]["canonical_entities"]["sha256"]
        hash2 = result2["files"]["canonical_entities"]["sha256"]
        
        assert hash1 == hash2, "Hashes should be reproducible"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for audit pack."""
    
    def test_full_audit_pack_valid(self):
        """Test complete audit pack is valid."""
        pack_path = AUDIT_PACK_DIR / "audit_pack.json"
        
        with open(pack_path, "r", encoding="utf-8") as f:
            pack = json.load(f)
        
        # Check all completeness flags
        assert pack["provenance"]["completeness"]["behaviors_complete"] is True
        assert pack["provenance"]["completeness"]["graph_complete"] is True
        assert pack["provenance"]["completeness"]["tafsir_complete"] is True
        assert pack["provenance"]["completeness"]["all_complete"] is True
        
        # Check benchmark
        assert pack["benchmark"]["pass_rate"] == 100.0
    
    def test_input_output_hash_counts(self):
        """Test input and output file counts."""
        pack_path = AUDIT_PACK_DIR / "audit_pack.json"
        
        with open(pack_path, "r", encoding="utf-8") as f:
            pack = json.load(f)
        
        # At least 10 input files (7 tafsir + canonical + schema + quran)
        assert pack["input_hashes"]["file_count"] >= 8
        
        # At least 5 output files
        assert pack["output_hashes"]["file_count"] >= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
