#!/usr/bin/env python
"""
QBM Release Gate Test Suite

This script runs the canonical test suite required for release approval.
All tests must pass before tagging a new release.

Usage:
    python scripts/test_release_gate.py
    
Exit codes:
    0 - All tests passed, release approved
    1 - Tests failed, release blocked
"""

import subprocess
import sys
import time
from pathlib import Path


# =============================================================================
# RELEASE GATE CONFIGURATION
# =============================================================================

# Test files that MUST pass for release
REQUIRED_TEST_FILES = [
    "tests/test_api.py",
    "tests/test_canonical_path.py", 
    "tests/test_no_fallback_invariant.py",
]

# Minimum test count to prevent accidental test deletion
MINIMUM_TEST_COUNT = 45

# Maximum allowed skipped tests
MAXIMUM_SKIPPED = 5


def run_release_gate():
    """Run the release gate test suite."""
    print("=" * 70)
    print("QBM RELEASE GATE TEST SUITE")
    print("=" * 70)
    print()
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    
    # Build pytest command
    test_files = " ".join(REQUIRED_TEST_FILES)
    cmd = f"python -m pytest {test_files} -v --tb=short"
    
    print(f"Running: {cmd}")
    print(f"Working directory: {project_root}")
    print()
    
    start_time = time.time()
    
    # Run tests
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=project_root,
        capture_output=True,
        text=True
    )
    
    elapsed = time.time() - start_time
    
    # Print output
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    # Parse results
    output = result.stdout
    
    # Extract test counts
    passed = 0
    failed = 0
    skipped = 0
    
    for line in output.split("\n"):
        if "passed" in line and ("failed" in line or "skipped" in line or "=" in line):
            # Parse summary line like "49 passed, 1 skipped"
            parts = line.split()
            for i, part in enumerate(parts):
                if part == "passed" or part == "passed,":
                    passed = int(parts[i-1])
                elif part == "failed" or part == "failed,":
                    failed = int(parts[i-1])
                elif part == "skipped" or part == "skipped,":
                    skipped = int(parts[i-1])
    
    # Determine gate status
    print()
    print("=" * 70)
    print("RELEASE GATE RESULTS")
    print("=" * 70)
    print(f"  Passed:  {passed}")
    print(f"  Failed:  {failed}")
    print(f"  Skipped: {skipped}")
    print(f"  Time:    {elapsed:.1f}s")
    print()
    
    # Check gate criteria
    gate_passed = True
    issues = []
    
    if result.returncode != 0:
        gate_passed = False
        issues.append(f"pytest returned non-zero exit code: {result.returncode}")
    
    if failed > 0:
        gate_passed = False
        issues.append(f"{failed} test(s) failed")
    
    if passed < MINIMUM_TEST_COUNT:
        gate_passed = False
        issues.append(f"Only {passed} tests passed (minimum: {MINIMUM_TEST_COUNT})")
    
    if skipped > MAXIMUM_SKIPPED:
        gate_passed = False
        issues.append(f"{skipped} tests skipped (maximum: {MAXIMUM_SKIPPED})")
    
    if gate_passed:
        print("[PASS] RELEASE GATE: PASSED")
        print()
        print("All criteria met:")
        print(f"  [OK] {passed} tests passed (minimum: {MINIMUM_TEST_COUNT})")
        print(f"  [OK] {failed} tests failed (maximum: 0)")
        print(f"  [OK] {skipped} tests skipped (maximum: {MAXIMUM_SKIPPED})")
        print()
        print("Release is APPROVED for tagging.")
        return 0
    else:
        print("[FAIL] RELEASE GATE: FAILED")
        print()
        print("Issues found:")
        for issue in issues:
            print(f"  [X] {issue}")
        print()
        print("Release is BLOCKED. Fix issues before tagging.")
        return 1


if __name__ == "__main__":
    sys.exit(run_release_gate())
