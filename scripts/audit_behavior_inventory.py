#!/usr/bin/env python3
"""
Phase 1: Behavior Inventory Audit

Reconciles the 73 behaviors in vocab/canonical_entities.json with the 
87 behaviors in src/ml/behavioral_classifier.py.

Outputs:
- artifacts/behavior_inventory_audit.json

Exit codes:
- 0: Audit complete (may have gaps)
- 1: Critical error
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

CANONICAL_ENTITIES_FILE = Path("vocab/canonical_entities.json")
BEHAVIOR_CONCEPTS_FILE = Path("vocab/behavior_concepts.json")
ARTIFACTS_DIR = Path("artifacts")


def load_canonical_behaviors() -> dict:
    """Load behaviors from canonical_entities.json."""
    with open(CANONICAL_ENTITIES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    behaviors = {}
    for b in data.get("behaviors", []):
        behaviors[b["id"]] = {
            "id": b["id"],
            "ar": b["ar"],
            "en": b["en"],
            "category": b.get("category", "unknown"),
            "roots": b.get("roots", []),
            "synonyms": b.get("synonyms", []),
        }
    return behaviors


def load_behavior_concepts() -> dict:
    """Load behaviors from behavior_concepts.json."""
    with open(BEHAVIOR_CONCEPTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    behaviors = {}
    for category, items in data.get("categories", {}).items():
        for b in items:
            behaviors[b["id"]] = {
                "id": b["id"],
                "ar": b["ar"],
                "en": b["en"],
                "category": category,
                "roots": b.get("quranic_roots", []),
            }
    return behaviors


def load_classifier_behaviors() -> list:
    """Load behaviors from behavioral_classifier.py."""
    try:
        from src.ml.behavioral_classifier import BEHAVIOR_CLASSES
        return list(BEHAVIOR_CLASSES)
    except ImportError as e:
        print(f"Warning: Could not import BEHAVIOR_CLASSES: {e}")
        return []


def audit_behaviors() -> dict:
    """
    Perform full behavior inventory audit.
    
    Returns audit report with:
    - canonical_count: behaviors in canonical_entities.json
    - classifier_count: behaviors in behavioral_classifier.py
    - missing_in_canonical: behaviors in classifier but not canonical
    - missing_in_classifier: behaviors in canonical but not classifier
    - recommendations: actions to reconcile
    """
    canonical = load_canonical_behaviors()
    concepts = load_behavior_concepts()
    classifier_behaviors = load_classifier_behaviors()
    
    # Extract Arabic labels from canonical
    canonical_ar = {b["ar"] for b in canonical.values()}
    concepts_ar = {b["ar"] for b in concepts.values()}
    classifier_ar = set(classifier_behaviors)
    
    # Find gaps
    missing_in_canonical = classifier_ar - canonical_ar
    missing_in_classifier = canonical_ar - classifier_ar
    
    # Build recommendations
    recommendations = []
    
    if missing_in_canonical:
        recommendations.append({
            "action": "ADD_TO_CANONICAL",
            "description": f"Add {len(missing_in_canonical)} behaviors to vocab/canonical_entities.json",
            "behaviors": sorted(list(missing_in_canonical)),
        })
    
    if missing_in_classifier:
        recommendations.append({
            "action": "VERIFY_CLASSIFIER",
            "description": f"Verify {len(missing_in_classifier)} behaviors exist in classifier or are intentionally excluded",
            "behaviors": sorted(list(missing_in_classifier)),
        })
    
    # Check for ID consistency
    canonical_ids = set(canonical.keys())
    concepts_ids = set(concepts.keys())
    id_mismatch = canonical_ids.symmetric_difference(concepts_ids)
    
    if id_mismatch:
        recommendations.append({
            "action": "RECONCILE_IDS",
            "description": "Reconcile ID mismatches between canonical_entities.json and behavior_concepts.json",
            "ids": sorted(list(id_mismatch)),
        })
    
    audit_report = {
        "audit_timestamp": datetime.utcnow().isoformat() + "Z",
        "canonical_entities_file": str(CANONICAL_ENTITIES_FILE),
        "behavior_concepts_file": str(BEHAVIOR_CONCEPTS_FILE),
        "counts": {
            "canonical_entities": len(canonical),
            "behavior_concepts": len(concepts),
            "classifier_behaviors": len(classifier_behaviors),
            "target": 87,
        },
        "gaps": {
            "missing_in_canonical": sorted(list(missing_in_canonical)),
            "missing_in_canonical_count": len(missing_in_canonical),
            "missing_in_classifier": sorted(list(missing_in_classifier)),
            "missing_in_classifier_count": len(missing_in_classifier),
        },
        "id_consistency": {
            "canonical_ids_count": len(canonical_ids),
            "concepts_ids_count": len(concepts_ids),
            "id_mismatch_count": len(id_mismatch),
        },
        "recommendations": recommendations,
        "status": "RECONCILIATION_REQUIRED" if missing_in_canonical else "OK",
    }
    
    return audit_report


def main():
    """Run behavior inventory audit."""
    print("=" * 60)
    print("BEHAVIOR INVENTORY AUDIT")
    print("=" * 60)
    
    audit_report = audit_behaviors()
    
    # Print summary
    print(f"\nCounts:")
    print(f"  canonical_entities.json: {audit_report['counts']['canonical_entities']}")
    print(f"  behavior_concepts.json:  {audit_report['counts']['behavior_concepts']}")
    print(f"  behavioral_classifier:   {audit_report['counts']['classifier_behaviors']}")
    print(f"  Target:                  {audit_report['counts']['target']}")
    
    print(f"\nGaps:")
    print(f"  Missing in canonical: {audit_report['gaps']['missing_in_canonical_count']}")
    print(f"  Missing in classifier: {audit_report['gaps']['missing_in_classifier_count']}")
    
    if audit_report['gaps']['missing_in_canonical']:
        print(f"\n  Behaviors to add to canonical_entities.json:")
        for b in audit_report['gaps']['missing_in_canonical'][:10]:
            print(f"    - {b}")
        if len(audit_report['gaps']['missing_in_canonical']) > 10:
            print(f"    ... and {len(audit_report['gaps']['missing_in_canonical']) - 10} more")
    
    print(f"\nStatus: {audit_report['status']}")
    
    # Save report
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = ARTIFACTS_DIR / "behavior_inventory_audit.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, ensure_ascii=False, indent=2)
    
    print(f"\nReport saved to: {output_path}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
