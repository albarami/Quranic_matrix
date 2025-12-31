#!/usr/bin/env python3
"""
Fetch Full SSOT Data Pack for Release Lane.

This script downloads the full Single Source of Truth (SSOT) data pack
required for the release lane (audit pack generation, full benchmark).

Environment Variables:
    QBM_SSOT_ZIP_URL: Direct URL to ssot_full.zip (Azure Blob SAS, S3 presigned, HTTPS)
    QBM_SSOT_RELEASE_ASSET: GitHub Release asset name (uses gh release download)
    QBM_SSOT_RELEASE_TAG: GitHub Release tag (default: latest)

Required SSOT Contents:
    data/quran/quran_uthmani.xml (or equivalent Quran text source)
    data/tafsir/*.jsonl (7 tafsir source files)

Usage:
    # Option A: Direct URL
    export QBM_SSOT_ZIP_URL="https://storage.example.com/ssot_full.zip?sas=..."
    python scripts/fetch_ssot_full.py

    # Option B: GitHub Release asset
    export QBM_SSOT_RELEASE_ASSET="ssot_full.zip"
    export QBM_SSOT_RELEASE_TAG="v1.0.0"
    python scripts/fetch_ssot_full.py
"""

import os
import sys
import json
import zipfile
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
from urllib.request import urlretrieve
from urllib.error import URLError


# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Required SSOT paths (relative to PROJECT_ROOT)
REQUIRED_SSOT_PATHS = {
    "quran": [
        "data/quran/quran_uthmani.xml",
    ],
    "tafsir": [
        "data/tafsir/tafsir_ibn_kathir.jsonl",
        "data/tafsir/tafsir_tabari.jsonl",
        "data/tafsir/tafsir_qurtubi.jsonl",
        "data/tafsir/tafsir_saadi.jsonl",
        "data/tafsir/tafsir_baghawi.jsonl",
        "data/tafsir/tafsir_muyassar.jsonl",
        "data/tafsir/tafsir_waseet.jsonl",
    ],
}

# Minimum file sizes (bytes) to detect truncated downloads
MIN_FILE_SIZES = {
    "quran_uthmani.xml": 1_000_000,  # ~3MB expected
    ".jsonl": 100_000,  # Tafsir files should be substantial
}


def log(msg: str, level: str = "INFO") -> None:
    """Print log message with level prefix."""
    print(f"[{level}] {msg}", file=sys.stderr if level == "ERROR" else sys.stdout)


def download_from_url(url: str, dest: Path) -> bool:
    """Download file from direct URL."""
    log(f"Downloading from URL: {url[:80]}...")
    try:
        urlretrieve(url, dest)
        log(f"Downloaded {dest.stat().st_size:,} bytes")
        return True
    except URLError as e:
        log(f"URL download failed: {e}", "ERROR")
        return False
    except Exception as e:
        log(f"Download error: {e}", "ERROR")
        return False


def download_from_github_release(asset_name: str, tag: str, dest: Path) -> bool:
    """Download asset from GitHub Release using gh CLI."""
    log(f"Downloading GitHub Release asset: {asset_name} from tag {tag}")
    
    # Check if gh CLI is available
    try:
        subprocess.run(["gh", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        log("GitHub CLI (gh) not found. Install it or use QBM_SSOT_ZIP_URL instead.", "ERROR")
        return False
    
    try:
        cmd = ["gh", "release", "download", tag, "--pattern", asset_name, "--output", str(dest)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            log(f"gh release download failed: {result.stderr}", "ERROR")
            return False
        
        log(f"Downloaded {dest.stat().st_size:,} bytes from GitHub Release")
        return True
    except Exception as e:
        log(f"GitHub Release download error: {e}", "ERROR")
        return False


def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    """Extract zip file to destination."""
    log(f"Extracting {zip_path} to {extract_to}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # List contents
            names = zf.namelist()
            log(f"Zip contains {len(names)} files")
            
            # Extract
            zf.extractall(extract_to)
        
        log("Extraction complete")
        return True
    except zipfile.BadZipFile:
        log("Invalid zip file", "ERROR")
        return False
    except Exception as e:
        log(f"Extraction error: {e}", "ERROR")
        return False


def verify_ssot_inventory() -> Dict[str, any]:
    """Verify all required SSOT files are present and valid."""
    inventory = {
        "all_present": True,
        "quran_files": [],
        "tafsir_files": [],
        "missing": [],
        "warnings": [],
    }
    
    # Check Quran files
    for rel_path in REQUIRED_SSOT_PATHS["quran"]:
        full_path = PROJECT_ROOT / rel_path
        if full_path.exists():
            size = full_path.stat().st_size
            inventory["quran_files"].append({
                "path": rel_path,
                "size": size,
                "exists": True,
            })
            
            # Check minimum size
            for pattern, min_size in MIN_FILE_SIZES.items():
                if pattern in rel_path and size < min_size:
                    inventory["warnings"].append(f"{rel_path} is smaller than expected ({size:,} < {min_size:,})")
        else:
            inventory["missing"].append(rel_path)
            inventory["all_present"] = False
    
    # Check Tafsir files
    for rel_path in REQUIRED_SSOT_PATHS["tafsir"]:
        full_path = PROJECT_ROOT / rel_path
        if full_path.exists():
            size = full_path.stat().st_size
            inventory["tafsir_files"].append({
                "path": rel_path,
                "size": size,
                "exists": True,
            })
            
            # Check minimum size for JSONL
            if size < MIN_FILE_SIZES[".jsonl"]:
                inventory["warnings"].append(f"{rel_path} is smaller than expected ({size:,} < {MIN_FILE_SIZES['.jsonl']:,})")
        else:
            inventory["missing"].append(rel_path)
            inventory["all_present"] = False
    
    return inventory


def print_inventory(inventory: Dict) -> None:
    """Print inventory in deterministic format."""
    print("\n" + "=" * 60)
    print("SSOT INVENTORY REPORT")
    print("=" * 60)
    
    print(f"\nStatus: {'✓ ALL PRESENT' if inventory['all_present'] else '✗ MISSING FILES'}")
    
    print(f"\nQuran Files ({len(inventory['quran_files'])}):")
    for f in inventory["quran_files"]:
        print(f"  ✓ {f['path']} ({f['size']:,} bytes)")
    
    print(f"\nTafsir Files ({len(inventory['tafsir_files'])}):")
    for f in inventory["tafsir_files"]:
        print(f"  ✓ {f['path']} ({f['size']:,} bytes)")
    
    if inventory["missing"]:
        print(f"\nMissing Files ({len(inventory['missing'])}):")
        for path in inventory["missing"]:
            print(f"  ✗ {path}")
    
    if inventory["warnings"]:
        print(f"\nWarnings ({len(inventory['warnings'])}):")
        for warn in inventory["warnings"]:
            print(f"  ⚠ {warn}")
    
    print("\n" + "=" * 60)


def main() -> int:
    """Main entry point."""
    log("Starting SSOT Full Data Pack Fetch")
    log(f"Project root: {PROJECT_ROOT}")
    
    # Check environment variables
    ssot_url = os.getenv("QBM_SSOT_ZIP_URL", "")
    ssot_asset = os.getenv("QBM_SSOT_RELEASE_ASSET", "")
    ssot_tag = os.getenv("QBM_SSOT_RELEASE_TAG", "latest")
    
    # First, check if SSOT already exists
    log("Checking existing SSOT inventory...")
    inventory = verify_ssot_inventory()
    
    if inventory["all_present"]:
        log("All SSOT files already present - skipping download")
        print_inventory(inventory)
        return 0
    
    # Need to download
    if not ssot_url and not ssot_asset:
        log("No SSOT source configured.", "ERROR")
        log("Set QBM_SSOT_ZIP_URL or QBM_SSOT_RELEASE_ASSET environment variable.", "ERROR")
        log("", "ERROR")
        log("For local development, manually place files in:", "ERROR")
        for path in inventory["missing"]:
            log(f"  {PROJECT_ROOT / path}", "ERROR")
        print_inventory(inventory)
        return 1
    
    # Create temp file for download
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        zip_path = Path(tmp.name)
    
    try:
        # Download
        success = False
        if ssot_url:
            success = download_from_url(ssot_url, zip_path)
        elif ssot_asset:
            success = download_from_github_release(ssot_asset, ssot_tag, zip_path)
        
        if not success:
            log("SSOT download failed", "ERROR")
            return 1
        
        # Extract to project root
        if not extract_zip(zip_path, PROJECT_ROOT):
            log("SSOT extraction failed", "ERROR")
            return 1
        
    finally:
        # Cleanup temp file
        if zip_path.exists():
            zip_path.unlink()
    
    # Verify after extraction
    log("Verifying SSOT inventory after extraction...")
    inventory = verify_ssot_inventory()
    print_inventory(inventory)
    
    if not inventory["all_present"]:
        log("SSOT verification failed - some files still missing", "ERROR")
        return 1
    
    log("SSOT Full Data Pack ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
