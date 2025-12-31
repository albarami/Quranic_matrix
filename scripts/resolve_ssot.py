#!/usr/bin/env python3
"""
Resolve SSOT Data for Release Lane.

This script resolves the Single Source of Truth (SSOT) data pack for the
release lane. It supports multiple sources:

1. Local directory (QBM_SSOT_DIR) - for self-hosted runners with local data
2. Local zip file (QBM_SSOT_ZIP) - for self-hosted runners with bundled data
3. Remote URL (QBM_SSOT_BUNDLE_URL) - for cloud-hosted runners
4. GitHub Release asset (QBM_SSOT_RELEASE_ASSET) - for GitHub-hosted runners

Environment Variables:
    QBM_SSOT_MODE: "fixture" or "full" (default: from QBM_DATASET_MODE)
    QBM_SSOT_DIR: Local directory containing SSOT files (e.g., D:\Quran_matrix\ssot_full)
    QBM_SSOT_ZIP: Local zip file path (e.g., D:\Quran_matrix\ssot_bundle.zip)
    QBM_SSOT_BUNDLE_URL: Remote URL to download SSOT bundle
    QBM_SSOT_RELEASE_ASSET: GitHub Release asset name
    QBM_SSOT_RELEASE_TAG: GitHub Release tag (default: latest)

Usage:
    # Option A: Local directory (self-hosted runner)
    set QBM_SSOT_DIR=D:\Quran_matrix\ssot_full
    python scripts/resolve_ssot.py

    # Option B: Local zip (self-hosted runner)
    set QBM_SSOT_ZIP=D:\Quran_matrix\ssot_bundle.zip
    python scripts/resolve_ssot.py

    # Option C: Remote URL (cloud CI)
    set QBM_SSOT_BUNDLE_URL=https://storage.example.com/ssot.zip
    python scripts/resolve_ssot.py
"""

import hashlib
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.error import URLError
from urllib.request import urlretrieve


# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# SSOT manifest path
SSOT_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "ssot_bundle" / "ssot_manifest.json"

# Required SSOT files (relative paths)
REQUIRED_SSOT_FILES = {
    "quran_text": "data/quran/quran_uthmani.xml",
    "tafsir_ibn_kathir": "data/tafsir/tafsir_ibn_kathir.jsonl",
    "tafsir_tabari": "data/tafsir/tafsir_tabari.jsonl",
    "tafsir_qurtubi": "data/tafsir/tafsir_qurtubi.jsonl",
    "tafsir_saadi": "data/tafsir/tafsir_saadi.jsonl",
    "tafsir_baghawi": "data/tafsir/tafsir_baghawi.jsonl",
    "tafsir_muyassar": "data/tafsir/tafsir_muyassar.jsonl",
    "tafsir_waseet": "data/tafsir/tafsir_waseet.jsonl",
}

# Minimum file sizes (bytes) to detect truncated files
MIN_FILE_SIZES = {
    "quran_text": 1_000_000,  # ~3MB expected
    "tafsir_": 100_000,  # Tafsir files should be substantial
}


def log(msg: str, level: str = "INFO") -> None:
    """Print log message with level prefix."""
    prefix = f"[SSOT-{level}]"
    stream = sys.stderr if level == "ERROR" else sys.stdout
    print(f"{prefix} {msg}", file=stream)


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_manifest() -> Optional[Dict]:
    """Load SSOT manifest if it exists."""
    if not SSOT_MANIFEST_PATH.exists():
        log(f"Manifest not found: {SSOT_MANIFEST_PATH}", "WARN")
        return None
    
    try:
        with open(SSOT_MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log(f"Failed to load manifest: {e}", "ERROR")
        return None


def verify_file(
    file_path: Path,
    logical_name: str,
    expected_sha256: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Verify a single SSOT file exists and optionally matches expected hash.
    
    Returns:
        (is_valid, message)
    """
    if not file_path.exists():
        return False, f"File not found: {file_path}"
    
    size = file_path.stat().st_size
    
    # Check minimum size
    for prefix, min_size in MIN_FILE_SIZES.items():
        if logical_name.startswith(prefix) and size < min_size:
            return False, f"File too small: {file_path} ({size:,} < {min_size:,} bytes)"
    
    # Verify SHA256 if provided
    if expected_sha256 and not expected_sha256.startswith("PLACEHOLDER"):
        actual_sha256 = compute_sha256(file_path)
        if actual_sha256 != expected_sha256:
            return False, f"SHA256 mismatch for {file_path}: expected {expected_sha256[:16]}..., got {actual_sha256[:16]}..."
    
    return True, f"OK: {file_path} ({size:,} bytes)"


def resolve_from_local_dir(ssot_dir: Path) -> Dict:
    """
    Resolve SSOT from a local directory.
    
    The directory should contain the SSOT files in their expected structure:
    ssot_dir/
      data/quran/quran_uthmani.xml
      data/tafsir/*.jsonl
    """
    log(f"Resolving SSOT from local directory: {ssot_dir}")
    
    result = {
        "source": "local_dir",
        "source_path": str(ssot_dir),
        "files": {},
        "missing": [],
        "errors": [],
        "valid": True,
    }
    
    manifest = load_manifest()
    manifest_files = {}
    if manifest:
        for f in manifest.get("files", []):
            manifest_files[f["path"]] = f.get("sha256")
    
    for logical_name, rel_path in REQUIRED_SSOT_FILES.items():
        # Check in local dir first
        local_path = ssot_dir / rel_path
        target_path = PROJECT_ROOT / rel_path
        
        if local_path.exists():
            # Copy to project if not already there or different
            if not target_path.exists() or compute_sha256(local_path) != compute_sha256(target_path):
                log(f"Copying {logical_name} to project...")
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(local_path, target_path)
            
            expected_sha = manifest_files.get(rel_path)
            is_valid, msg = verify_file(target_path, logical_name, expected_sha)
            
            result["files"][logical_name] = {
                "path": rel_path,
                "size": target_path.stat().st_size,
                "sha256": compute_sha256(target_path),
                "valid": is_valid,
                "message": msg,
            }
            
            if not is_valid:
                result["errors"].append(msg)
                result["valid"] = False
        else:
            result["missing"].append(rel_path)
            result["valid"] = False
    
    return result


def resolve_from_local_zip(zip_path: Path) -> Dict:
    """
    Resolve SSOT from a local zip file.
    """
    log(f"Resolving SSOT from local zip: {zip_path}")
    
    result = {
        "source": "local_zip",
        "source_path": str(zip_path),
        "files": {},
        "missing": [],
        "errors": [],
        "valid": True,
    }
    
    if not zip_path.exists():
        result["errors"].append(f"Zip file not found: {zip_path}")
        result["valid"] = False
        return result
    
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            log(f"Extracting {len(zf.namelist())} files from zip...")
            zf.extractall(PROJECT_ROOT)
    except Exception as e:
        result["errors"].append(f"Failed to extract zip: {e}")
        result["valid"] = False
        return result
    
    # Now verify extracted files
    manifest = load_manifest()
    manifest_files = {}
    if manifest:
        for f in manifest.get("files", []):
            manifest_files[f["path"]] = f.get("sha256")
    
    for logical_name, rel_path in REQUIRED_SSOT_FILES.items():
        target_path = PROJECT_ROOT / rel_path
        
        if target_path.exists():
            expected_sha = manifest_files.get(rel_path)
            is_valid, msg = verify_file(target_path, logical_name, expected_sha)
            
            result["files"][logical_name] = {
                "path": rel_path,
                "size": target_path.stat().st_size,
                "sha256": compute_sha256(target_path),
                "valid": is_valid,
                "message": msg,
            }
            
            if not is_valid:
                result["errors"].append(msg)
                result["valid"] = False
        else:
            result["missing"].append(rel_path)
            result["valid"] = False
    
    return result


def resolve_from_url(url: str) -> Dict:
    """
    Resolve SSOT from a remote URL.
    """
    log(f"Resolving SSOT from URL: {url[:60]}...")
    
    result = {
        "source": "remote_url",
        "source_path": url[:60] + "...",
        "files": {},
        "missing": [],
        "errors": [],
        "valid": True,
    }
    
    # Download to temp file
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        zip_path = Path(tmp.name)
    
    try:
        log("Downloading SSOT bundle...")
        urlretrieve(url, zip_path)
        log(f"Downloaded {zip_path.stat().st_size:,} bytes")
        
        # Extract and verify
        inner_result = resolve_from_local_zip(zip_path)
        result.update(inner_result)
        result["source"] = "remote_url"
        
    except URLError as e:
        result["errors"].append(f"Download failed: {e}")
        result["valid"] = False
    except Exception as e:
        result["errors"].append(f"Error: {e}")
        result["valid"] = False
    finally:
        if zip_path.exists():
            zip_path.unlink()
    
    return result


def check_existing_ssot() -> Dict:
    """
    Check if SSOT files already exist in project.
    """
    result = {
        "source": "existing",
        "files": {},
        "missing": [],
        "errors": [],
        "valid": True,
    }
    
    manifest = load_manifest()
    manifest_files = {}
    if manifest:
        for f in manifest.get("files", []):
            manifest_files[f["path"]] = f.get("sha256")
    
    for logical_name, rel_path in REQUIRED_SSOT_FILES.items():
        target_path = PROJECT_ROOT / rel_path
        
        if target_path.exists():
            expected_sha = manifest_files.get(rel_path)
            is_valid, msg = verify_file(target_path, logical_name, expected_sha)
            
            result["files"][logical_name] = {
                "path": rel_path,
                "size": target_path.stat().st_size,
                "sha256": compute_sha256(target_path),
                "valid": is_valid,
                "message": msg,
            }
            
            if not is_valid:
                result["errors"].append(msg)
                result["valid"] = False
        else:
            result["missing"].append(rel_path)
            result["valid"] = False
    
    return result


def print_report(result: Dict) -> None:
    """Print SSOT resolution report."""
    print("\n" + "=" * 70)
    print("SSOT RESOLUTION REPORT")
    print("=" * 70)
    
    print(f"\nSource: {result['source']}")
    if "source_path" in result:
        print(f"Path: {result['source_path']}")
    
    status = "✓ VALID" if result["valid"] else "✗ INVALID"
    print(f"Status: {status}")
    
    print(f"\nFiles ({len(result['files'])}):")
    for name, info in sorted(result["files"].items()):
        status_icon = "✓" if info.get("valid", True) else "✗"
        print(f"  {status_icon} {name}: {info['path']} ({info['size']:,} bytes)")
    
    if result["missing"]:
        print(f"\nMissing ({len(result['missing'])}):")
        for path in result["missing"]:
            print(f"  ✗ {path}")
    
    if result["errors"]:
        print(f"\nErrors ({len(result['errors'])}):")
        for err in result["errors"]:
            print(f"  ! {err}")
    
    print("\n" + "=" * 70)


def main() -> int:
    """Main entry point."""
    log("=" * 50)
    log("SSOT Resolution for Release Lane")
    log("=" * 50)
    
    # Get configuration from environment
    ssot_mode = os.getenv("QBM_SSOT_MODE", os.getenv("QBM_DATASET_MODE", "fixture"))
    ssot_dir = os.getenv("QBM_SSOT_DIR", "")
    ssot_zip = os.getenv("QBM_SSOT_ZIP", "")
    ssot_url = os.getenv("QBM_SSOT_BUNDLE_URL", os.getenv("QBM_SSOT_ZIP_URL", ""))
    
    log(f"Mode: {ssot_mode}")
    log(f"Project root: {PROJECT_ROOT}")
    
    # In fixture mode, skip SSOT resolution
    if ssot_mode.lower() == "fixture":
        log("Fixture mode - skipping full SSOT resolution")
        log("Using fixture data only")
        return 0
    
    log("Full mode - resolving SSOT data...")
    
    # Priority order for SSOT sources:
    # 1. Check if already exists in project
    # 2. Local directory (QBM_SSOT_DIR)
    # 3. Local zip (QBM_SSOT_ZIP)
    # 4. Remote URL (QBM_SSOT_BUNDLE_URL)
    
    # First check existing
    log("Checking existing SSOT files...")
    result = check_existing_ssot()
    
    if result["valid"]:
        log("All SSOT files already present and valid")
        print_report(result)
        return 0
    
    # Try local directory
    if ssot_dir:
        ssot_dir_path = Path(ssot_dir)
        if ssot_dir_path.exists():
            result = resolve_from_local_dir(ssot_dir_path)
            print_report(result)
            
            if result["valid"]:
                log("SSOT resolved from local directory")
                return 0
            else:
                log("Local directory resolution failed", "ERROR")
                return 1
        else:
            log(f"QBM_SSOT_DIR does not exist: {ssot_dir}", "ERROR")
    
    # Try local zip
    if ssot_zip:
        ssot_zip_path = Path(ssot_zip)
        if ssot_zip_path.exists():
            result = resolve_from_local_zip(ssot_zip_path)
            print_report(result)
            
            if result["valid"]:
                log("SSOT resolved from local zip")
                return 0
            else:
                log("Local zip resolution failed", "ERROR")
                return 1
        else:
            log(f"QBM_SSOT_ZIP does not exist: {ssot_zip}", "ERROR")
    
    # Try remote URL
    if ssot_url:
        result = resolve_from_url(ssot_url)
        print_report(result)
        
        if result["valid"]:
            log("SSOT resolved from remote URL")
            return 0
        else:
            log("Remote URL resolution failed", "ERROR")
            return 1
    
    # No source configured
    log("No SSOT source configured for full mode", "ERROR")
    log("Set one of:", "ERROR")
    log("  QBM_SSOT_DIR=<path to directory with SSOT files>", "ERROR")
    log("  QBM_SSOT_ZIP=<path to SSOT zip bundle>", "ERROR")
    log("  QBM_SSOT_BUNDLE_URL=<URL to download SSOT bundle>", "ERROR")
    
    print_report(result)
    return 1


if __name__ == "__main__":
    sys.exit(main())
