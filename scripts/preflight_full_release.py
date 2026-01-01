#!/usr/bin/env python3
"""
Preflight checks for FULL SSOT release lane.

This script validates that all prerequisites are met before running
the full release pipeline. It should be run at the start of the
release-full-ssot job to fail fast if anything is missing.

Usage:
    python scripts/preflight_full_release.py

Environment Variables:
    QBM_DATASET_MODE: Must be "full"
    QBM_SSOT_DIR: Path to SSOT directory (e.g., D:\Quran_matrix\ssot_full)

Exit Codes:
    0: All checks passed
    1: One or more checks failed
"""

import json
import os
import sys
from pathlib import Path

# Canonical counts - these are non-negotiable
CANONICAL_BEHAVIORS = 87
CANONICAL_AGENTS = 14

# Required SSOT files (relative to QBM_SSOT_DIR)
REQUIRED_SSOT_FILES = [
    "data/quran/quran_uthmani.xml",
    "data/tafsir/tafsir_ibn_kathir.jsonl",
    "data/tafsir/tafsir_tabari.jsonl",
    "data/tafsir/tafsir_qurtubi.jsonl",
    "data/tafsir/tafsir_saadi.jsonl",
    "data/tafsir/tafsir_jalalayn.jsonl",
    "data/tafsir/tafsir_baghawi.jsonl",
    "data/tafsir/tafsir_muyassar.jsonl",
]

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
GRAPH_FILE = PROJECT_ROOT / "data" / "graph" / "graph_v3.json"
CONCEPT_INDEX_FILE = PROJECT_ROOT / "data" / "evidence" / "concept_index_v3.jsonl"
CANONICAL_ENTITIES_FILE = PROJECT_ROOT / "vocab" / "canonical_entities.json"


class PreflightChecker:
    """Run preflight checks for full release."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        
    def log(self, msg: str, level: str = "INFO"):
        """Log a message."""
        prefix = {"INFO": "[INFO]", "WARN": "[WARN]", "ERROR": "[ERROR]", "PASS": "[PASS]"}
        print(f"{prefix.get(level, '[INFO]')} {msg}")
        
    def check_dataset_mode(self) -> bool:
        """Check that QBM_DATASET_MODE is 'full'."""
        mode = os.environ.get("QBM_DATASET_MODE", "")
        if mode != "full":
            self.errors.append(f"QBM_DATASET_MODE must be 'full', got '{mode}'")
            self.log(f"QBM_DATASET_MODE = '{mode}' (expected 'full')", "ERROR")
            return False
        self.log(f"QBM_DATASET_MODE = 'full'", "PASS")
        return True
    
    def check_ssot_dir_exists(self) -> bool:
        """Check that QBM_SSOT_DIR exists and is accessible."""
        ssot_dir = os.environ.get("QBM_SSOT_DIR", "")
        if not ssot_dir:
            self.errors.append("QBM_SSOT_DIR environment variable not set")
            self.log("QBM_SSOT_DIR not set", "ERROR")
            return False
        
        ssot_path = Path(ssot_dir)
        if not ssot_path.exists():
            self.errors.append(f"QBM_SSOT_DIR does not exist: {ssot_dir}")
            self.log(f"QBM_SSOT_DIR does not exist: {ssot_dir}", "ERROR")
            return False
        
        if not ssot_path.is_dir():
            self.errors.append(f"QBM_SSOT_DIR is not a directory: {ssot_dir}")
            self.log(f"QBM_SSOT_DIR is not a directory: {ssot_dir}", "ERROR")
            return False
        
        self.log(f"QBM_SSOT_DIR exists: {ssot_dir}", "PASS")
        return True
    
    def check_ssot_files(self) -> bool:
        """Check that all required SSOT files exist."""
        ssot_dir = os.environ.get("QBM_SSOT_DIR", "")
        if not ssot_dir:
            return False
        
        ssot_path = Path(ssot_dir)
        missing = []
        
        for rel_path in REQUIRED_SSOT_FILES:
            file_path = ssot_path / rel_path
            if not file_path.exists():
                missing.append(rel_path)
                self.log(f"Missing SSOT file: {rel_path}", "ERROR")
            else:
                size = file_path.stat().st_size
                if size == 0:
                    missing.append(f"{rel_path} (empty)")
                    self.log(f"Empty SSOT file: {rel_path}", "ERROR")
                else:
                    self.log(f"SSOT file OK: {rel_path} ({size:,} bytes)", "PASS")
        
        if missing:
            self.errors.append(f"Missing SSOT files: {missing}")
            return False
        
        return True
    
    def check_canonical_entities(self) -> bool:
        """Check that canonical_entities.json has 87 behaviors."""
        if not CANONICAL_ENTITIES_FILE.exists():
            self.errors.append(f"Canonical entities file not found: {CANONICAL_ENTITIES_FILE}")
            self.log(f"Canonical entities file not found", "ERROR")
            return False
        
        with open(CANONICAL_ENTITIES_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        behaviors = data.get("behaviors", [])
        count = len(behaviors)
        
        if count != CANONICAL_BEHAVIORS:
            self.errors.append(f"Expected {CANONICAL_BEHAVIORS} behaviors, got {count}")
            self.log(f"Canonical behaviors: {count} (expected {CANONICAL_BEHAVIORS})", "ERROR")
            return False
        
        self.log(f"Canonical behaviors: {count}", "PASS")
        return True
    
    def check_graph_behavior_nodes(self) -> bool:
        """Check that graph_v3.json has 87 behavior nodes."""
        if not GRAPH_FILE.exists():
            self.errors.append(f"Graph file not found: {GRAPH_FILE}")
            self.log(f"Graph file not found", "ERROR")
            return False
        
        with open(GRAPH_FILE, 'r', encoding='utf-8') as f:
            graph = json.load(f)
        
        nodes = graph.get("nodes", [])
        behavior_nodes = [n for n in nodes if n.get("type") == "behavior"]
        count = len(behavior_nodes)
        
        if count != CANONICAL_BEHAVIORS:
            self.errors.append(f"Graph has {count} behavior nodes, expected {CANONICAL_BEHAVIORS}")
            self.log(f"Graph behavior nodes: {count} (expected {CANONICAL_BEHAVIORS})", "ERROR")
            return False
        
        self.log(f"Graph behavior nodes: {count}", "PASS")
        
        # Also check statistics
        stats = graph.get("statistics", {})
        stats_count = stats.get("behavior_nodes", 0)
        if stats_count != CANONICAL_BEHAVIORS:
            self.warnings.append(f"Graph statistics.behavior_nodes = {stats_count}, expected {CANONICAL_BEHAVIORS}")
            self.log(f"Graph statistics.behavior_nodes mismatch: {stats_count}", "WARN")
        
        return True
    
    def check_concept_index(self) -> bool:
        """Check that concept_index_v3.jsonl has 87 behavior entries."""
        if not CONCEPT_INDEX_FILE.exists():
            self.errors.append(f"Concept index file not found: {CONCEPT_INDEX_FILE}")
            self.log(f"Concept index file not found", "ERROR")
            return False
        
        behavior_count = 0
        with open(CONCEPT_INDEX_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("concept_type") == "BEHAVIOR" or entry.get("concept_id", "").startswith("BEH_"):
                    behavior_count += 1
        
        if behavior_count != CANONICAL_BEHAVIORS:
            self.errors.append(f"Concept index has {behavior_count} behaviors, expected {CANONICAL_BEHAVIORS}")
            self.log(f"Concept index behaviors: {behavior_count} (expected {CANONICAL_BEHAVIORS})", "ERROR")
            return False
        
        self.log(f"Concept index behaviors: {behavior_count}", "PASS")
        return True
    
    def check_python_encoding(self) -> bool:
        """Check Python encoding settings for Windows compatibility."""
        pythonutf8 = os.environ.get("PYTHONUTF8", "")
        pythonioencoding = os.environ.get("PYTHONIOENCODING", "")
        
        if pythonutf8 != "1":
            self.warnings.append("PYTHONUTF8 not set to '1' (recommended for Windows)")
            self.log("PYTHONUTF8 not set (recommended: PYTHONUTF8=1)", "WARN")
        else:
            self.log("PYTHONUTF8=1", "PASS")
        
        if not pythonioencoding:
            self.warnings.append("PYTHONIOENCODING not set (recommended: utf-8)")
            self.log("PYTHONIOENCODING not set (recommended: utf-8)", "WARN")
        else:
            self.log(f"PYTHONIOENCODING={pythonioencoding}", "PASS")
        
        return True  # Warnings only, not fatal
    
    def run_all_checks(self) -> bool:
        """Run all preflight checks."""
        print("=" * 60)
        print("QBM Full Release Preflight Checks")
        print("=" * 60)
        print()
        
        checks = [
            ("Dataset Mode", self.check_dataset_mode),
            ("Python Encoding", self.check_python_encoding),
            ("SSOT Directory", self.check_ssot_dir_exists),
            ("SSOT Files", self.check_ssot_files),
            ("Canonical Entities", self.check_canonical_entities),
            ("Graph Behavior Nodes", self.check_graph_behavior_nodes),
            ("Concept Index", self.check_concept_index),
        ]
        
        results = {}
        for name, check_func in checks:
            print(f"\n--- {name} ---")
            try:
                results[name] = check_func()
            except Exception as e:
                self.errors.append(f"{name} check failed with exception: {e}")
                self.log(f"Exception: {e}", "ERROR")
                results[name] = False
        
        # Summary
        print()
        print("=" * 60)
        print("PREFLIGHT SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        
        print(f"\nChecks: {passed}/{total} passed")
        
        if self.warnings:
            print(f"\nWarnings ({len(self.warnings)}):")
            for w in self.warnings:
                print(f"  - {w}")
        
        if self.errors:
            print(f"\nErrors ({len(self.errors)}):")
            for e in self.errors:
                print(f"  - {e}")
            print()
            print("[FAIL] Preflight checks FAILED - do not proceed with release")
            return False
        
        print()
        print("[PASS] All preflight checks passed - ready for full release")
        return True


def main():
    """Main entry point."""
    checker = PreflightChecker()
    success = checker.run_all_checks()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
