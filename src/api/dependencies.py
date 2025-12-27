"""
Shared Dependencies for API Routers

Phase 7.1: Common functions and state shared across routers.
"""

import json
from pathlib import Path
from typing import List
from fastapi import HTTPException

# Data paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"
EXPORTS_DIR = DATA_DIR / "exports"
VOCAB_DIR = DATA_DIR / "vocab"

# Cache for loaded data
_dataset_cache = {}


def load_dataset(tier: str = "silver") -> dict:
    """Load dataset from exports directory."""
    if tier in _dataset_cache:
        return _dataset_cache[tier]
    
    # Find latest export for tier
    pattern = f"qbm_{tier}_*.json"
    files = sorted(EXPORTS_DIR.glob(pattern), reverse=True)
    
    if not files:
        raise HTTPException(status_code=404, detail=f"No {tier} dataset found")
    
    with open(files[0], encoding="utf-8") as f:
        data = json.load(f)
    
    _dataset_cache[tier] = data
    return data


def get_all_spans(tier: str = "silver") -> List[dict]:
    """Get all spans from dataset."""
    data = load_dataset(tier)
    return data.get("spans", [])


def get_dataset_meta(tier: str = "silver") -> dict:
    """Get dataset metadata for a tier."""
    data = load_dataset(tier)
    return {
        "tier": data.get("tier", tier),
        "version": data.get("version", "unknown"),
        "exported_at": data.get("exported_at", "unknown"),
        "total_spans": data.get("total_spans", len(data.get("spans", []))),
    }


def build_surah_summary(spans: List[dict]) -> List[dict]:
    """Build per-surah summaries from spans."""
    surah_map = {}
    for span in spans:
        ref = span.get("reference", {})
        surah_num = ref.get("surah")
        ayah_num = ref.get("ayah")
        if not surah_num:
            continue

        entry = surah_map.setdefault(
            surah_num,
            {
                "surah": surah_num,
                "surah_name": ref.get("surah_name"),
                "spans": 0,
                "ayat": set(),
                "max_ayah": 0,
            },
        )

        entry["spans"] += 1
        if ayah_num:
            entry["ayat"].add(ayah_num)
            if ayah_num > entry["max_ayah"]:
                entry["max_ayah"] = ayah_num

        if not entry["surah_name"] and ref.get("surah_name"):
            entry["surah_name"] = ref.get("surah_name")

    summaries = []
    for surah_num in sorted(surah_map.keys()):
        entry = surah_map[surah_num]
        unique_ayat = len(entry["ayat"])
        total_ayat = entry["max_ayah"] or unique_ayat
        coverage_pct = round((unique_ayat / total_ayat) * 100, 1) if total_ayat else None
        summaries.append(
            {
                "surah": entry["surah"],
                "surah_name": entry["surah_name"],
                "spans": entry["spans"],
                "unique_ayat": unique_ayat,
                "total_ayat": total_ayat,
                "coverage_pct": coverage_pct,
            }
        )

    return summaries
