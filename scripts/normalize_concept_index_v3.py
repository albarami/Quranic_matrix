#!/usr/bin/env python3
"""
Normalize concept_index_v3.jsonl to canonical schema.

This script enforces a single canonical schema with defaults for all entries,
eliminating the need for ad-hoc field patches. Run this once and commit the result.

Schema v1.0:
- concept_id: str (required)
- term: str (Arabic term)
- term_en: str (English term)
- entity_type: str (BEHAVIOR, AGENT, ORGAN, etc.)
- status: str (active, deprecated)
- evidence_policy_mode: str (lexical, semantic, hybrid)
- verses: list[dict] (verse references with evidence)
- tafsir_chunks: list[dict] (tafsir evidence chunks)
- statistics: dict (total_sources, sources_by_type, avg_confidence)
- validation: dict (passed, errors, warnings)
- total_mentions: int
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

SCHEMA_VERSION = "1.0"

# Canonical schema with defaults
CANONICAL_SCHEMA = {
    "concept_id": "",  # Required, no default
    "term": "",
    "term_en": "",
    "entity_type": "BEHAVIOR",
    "status": "active",
    "evidence_policy_mode": "lexical",
    "verses": [],
    "tafsir_chunks": [],
    "statistics": {
        "total_sources": 0,
        "sources_by_type": {},
        "avg_confidence": 0.0
    },
    "validation": {
        "passed": True,
        "errors": [],
        "warnings": []
    },
    "total_mentions": 0
}

# Arabic terms for known behaviors (fallback mapping)
BEHAVIOR_TERMS = {
    "BEH_SOC_TRUSTWORTHINESS": ("الأمانة", "Trustworthiness"),
    "BEH_SOC_MERCY": ("الرحمة", "Mercy"),
    "BEH_PHY_WALKING": ("المشي", "Walking"),
    "BEH_FIN_ASCETICISM": ("الزهد", "Asceticism"),
    "BEH_SPI_KHUSHU": ("الخشوع", "Khushu"),
    "BEH_SOC_TRANSGRESSION": ("البغي", "Transgression"),
    "BEH_PHY_LOOKING": ("النظر", "Looking"),
    "BEH_SPI_SHIRK": ("الشرك", "Shirk"),
    "BEH_SOC_BETRAYAL": ("الخيانة", "Betrayal"),
    "BEH_PHY_DRINKING": ("الشرب", "Drinking"),
    "BEH_COG_SUPERFICIAL_KNOWING": ("المعرفة السطحية", "Superficial Knowing"),
    "BEH_PHY_EATING": ("الأكل", "Eating"),
    "BEH_PHY_SLEEPING": ("النوم", "Sleeping"),
    "BEH_PHY_MODESTY": ("الحياء", "Modesty"),
}

def normalize_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a single entry to canonical schema."""
    normalized = {}
    
    # Copy required field
    concept_id = entry.get("concept_id", "")
    if not concept_id:
        raise ValueError("Entry missing required field: concept_id")
    normalized["concept_id"] = concept_id
    
    # Apply schema with defaults
    for field, default in CANONICAL_SCHEMA.items():
        if field == "concept_id":
            continue
        
        if field in entry and entry[field] is not None:
            # Use existing value
            normalized[field] = entry[field]
        elif field == "term" and not entry.get("term"):
            # Generate Arabic term from mapping or concept_id
            if concept_id in BEHAVIOR_TERMS:
                normalized["term"] = BEHAVIOR_TERMS[concept_id][0]
            else:
                normalized["term"] = concept_id.replace("BEH_", "").replace("_", " ")
        elif field == "term_en" and not entry.get("term_en"):
            # Generate English term from mapping or concept_id
            if concept_id in BEHAVIOR_TERMS:
                normalized["term_en"] = BEHAVIOR_TERMS[concept_id][1]
            else:
                normalized["term_en"] = concept_id.replace("BEH_", "").replace("_", " ").title()
        elif isinstance(default, dict):
            # Deep copy dict defaults
            normalized[field] = entry.get(field, {}).copy() if entry.get(field) else default.copy()
        elif isinstance(default, list):
            # Deep copy list defaults
            normalized[field] = entry.get(field, []).copy() if entry.get(field) else default.copy()
        else:
            normalized[field] = default
    
    # Ensure nested structures have required fields
    if "statistics" in normalized:
        stats = normalized["statistics"]
        if "total_sources" not in stats:
            stats["total_sources"] = 0
        if "sources_by_type" not in stats:
            stats["sources_by_type"] = {}
        if "avg_confidence" not in stats:
            stats["avg_confidence"] = 0.0
    
    if "validation" in normalized:
        val = normalized["validation"]
        if "passed" not in val:
            val["passed"] = True
        if "errors" not in val:
            val["errors"] = []
        if "warnings" not in val:
            val["warnings"] = []
    
    return normalized

def main():
    project_root = Path(__file__).parent.parent
    index_path = project_root / "data" / "evidence" / "concept_index_v3.jsonl"
    
    if not index_path.exists():
        print(f"ERROR: {index_path} not found")
        sys.exit(1)
    
    print(f"Normalizing {index_path} to schema v{SCHEMA_VERSION}")
    
    # Load entries
    entries = []
    with open(index_path, 'r', encoding='utf-8') as f:
        for line in f:
            entries.append(json.loads(line))
    
    print(f"Loaded {len(entries)} entries")
    
    # Normalize
    normalized = []
    fixed_count = 0
    for entry in entries:
        original = json.dumps(entry, sort_keys=True)
        norm = normalize_entry(entry)
        normalized.append(norm)
        
        if json.dumps(norm, sort_keys=True) != original:
            fixed_count += 1
    
    # Write back
    with open(index_path, 'w', encoding='utf-8') as f:
        for entry in normalized:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Normalized {len(normalized)} entries ({fixed_count} modified)")
    
    # Generate report
    report = {
        "schema_version": SCHEMA_VERSION,
        "total_entries": len(normalized),
        "entries_modified": fixed_count,
        "fields_enforced": list(CANONICAL_SCHEMA.keys())
    }
    
    report_path = project_root / "artifacts" / "concept_index_v3_normalization_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Normalization complete. Report: {report_path}")

if __name__ == "__main__":
    main()
