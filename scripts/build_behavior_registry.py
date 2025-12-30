#!/usr/bin/env python3
"""
Phase 3: Build Behavior Registry with Evidence Policy

Transforms existing vocab/canonical_entities.json into the new
behavior registry format with explicit evidence policies.

Outputs:
- data/behaviors/behavior_registry.json: Complete registry
- artifacts/behavior_registry_report.json: Summary statistics

Exit codes:
- 0: Success
- 1: Failure
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.text.ar_normalize import normalize_ar


def save_json(path: Path, data: Dict) -> None:
    """Save JSON with pretty printing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_canonical_entities() -> Dict[str, Any]:
    """Load existing canonical entities."""
    with open("vocab/canonical_entities.json", 'r', encoding='utf-8') as f:
        return json.load(f)


def parse_root_to_forms(root: str) -> List[str]:
    """
    Parse Arabic root notation to get normalized forms.

    Root format: "ص-د-ق" or "صدق"
    Returns list of base forms to search for.
    """
    # Remove dashes
    clean_root = root.replace("-", "").replace("–", "")
    # Normalize
    norm_root = normalize_ar(clean_root)
    return [norm_root] if norm_root else []


def normalize_synonyms(synonyms: List[str]) -> List[str]:
    """Normalize a list of synonyms."""
    normalized = []
    for syn in synonyms:
        norm = normalize_ar(syn)
        if norm and norm not in normalized:
            normalized.append(norm)
    return normalized


def categorize_behavior(behavior_id: str, old_category: str) -> str:
    """Map old category to new standardized category."""
    category_map = {
        "speech": "communication",
        "action": "behavioral",
        "emotion": "emotional",
        "social": "social",
        "worship": "worship",
        "moral": "moral",
        "cognitive": "cognitive",
        "character": "character",
    }
    return category_map.get(old_category.lower(), old_category.lower())


def build_behavior_entry(behavior: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a behavior registry entry from canonical entity.

    Automatically assigns evidence policy based on whether
    roots/synonyms are defined.
    """
    behavior_id = behavior.get("id", "")
    label_ar = behavior.get("ar", "")
    label_en = behavior.get("en", "")
    category = behavior.get("category", "general")
    roots = behavior.get("roots", [])
    synonyms = behavior.get("synonyms", [])

    # Parse roots to get base forms
    forms = []
    for root in roots:
        forms.extend(parse_root_to_forms(root))

    # Normalize synonyms
    norm_synonyms = normalize_synonyms(synonyms)

    # Add normalized label as a form
    norm_label = normalize_ar(label_ar)
    if norm_label and norm_label not in forms:
        forms.append(norm_label)

    # Determine evidence mode
    has_lexical = bool(forms or norm_synonyms)

    if has_lexical:
        evidence_policy = {
            "mode": "lexical",
            "lexical_required": True,
            "min_sources": ["quran_tokens_norm"],
            "lexical_spec": {
                "roots": roots,
                "forms": forms,
                "synonyms": norm_synonyms,
                "exclude_patterns": []
            }
        }
    else:
        # No lexical patterns - annotation-based
        evidence_policy = {
            "mode": "annotation",
            "lexical_required": False,
            "min_sources": ["annotation_store"],
            "annotation_spec": {
                "allowed_types": ["direct", "indirect"],
                "min_confidence": 0.0,
                "required_annotators": []
            }
        }

    return {
        "behavior_id": behavior_id,
        "label_ar": label_ar,
        "label_en": label_en,
        "category": categorize_behavior(behavior_id, category),
        "evidence_policy": evidence_policy,
        "bouzidani_axes": {},
        "description_ar": "",
        "description_en": ""
    }


def build_registry() -> Dict[str, Any]:
    """Build complete behavior registry."""
    print("Loading canonical entities...")
    entities = load_canonical_entities()

    behaviors = entities.get("behaviors", [])
    print(f"Found {len(behaviors)} behaviors")

    registry_entries = []
    lexical_count = 0
    annotation_count = 0

    for behavior in behaviors:
        entry = build_behavior_entry(behavior)
        registry_entries.append(entry)

        if entry["evidence_policy"]["mode"] == "lexical":
            lexical_count += 1
        else:
            annotation_count += 1

    # Sort by behavior_id for consistency
    registry_entries.sort(key=lambda x: x["behavior_id"])

    return {
        "version": "3.0",
        "description": "Behavior registry with evidence policies",
        "total_behaviors": len(registry_entries),
        "lexical_behaviors": lexical_count,
        "annotation_behaviors": annotation_count,
        "behaviors": registry_entries
    }


def validate_registry(registry: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the built registry."""
    results = {
        "valid": True,
        "checks": {}
    }

    # Check 1: Has behaviors
    behaviors = registry.get("behaviors", [])
    has_behaviors = len(behaviors) > 0
    results["checks"]["has_behaviors"] = has_behaviors
    if not has_behaviors:
        results["valid"] = False

    # Check 2: Each behavior has required fields
    required_fields = ["behavior_id", "label_ar", "label_en", "evidence_policy"]
    all_have_fields = True
    for b in behaviors:
        for field in required_fields:
            if field not in b:
                all_have_fields = False
                break
    results["checks"]["all_have_required_fields"] = all_have_fields
    if not all_have_fields:
        results["valid"] = False

    # Check 3: Evidence policies are valid
    valid_modes = ["lexical", "annotation", "hybrid"]
    all_valid_policies = True
    for b in behaviors:
        mode = b.get("evidence_policy", {}).get("mode")
        if mode not in valid_modes:
            all_valid_policies = False
            break
    results["checks"]["all_valid_evidence_policies"] = all_valid_policies
    if not all_valid_policies:
        results["valid"] = False

    # Check 4: Patience behavior exists and has lexical spec
    patience = next(
        (b for b in behaviors if "PATIENCE" in b["behavior_id"].upper() or b["label_ar"] == "الصبر"),
        None
    )
    has_patience = patience is not None
    results["checks"]["has_patience_behavior"] = has_patience
    if not has_patience:
        results["valid"] = False
    else:
        patience_lexical = patience.get("evidence_policy", {}).get("lexical_spec") is not None
        results["checks"]["patience_has_lexical_spec"] = patience_lexical

    return results


def generate_report(registry: Dict[str, Any], validation: Dict[str, Any]) -> Dict[str, Any]:
    """Generate summary report."""
    behaviors = registry.get("behaviors", [])

    # Count by category
    categories = {}
    for b in behaviors:
        cat = b.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    # Count by evidence mode
    modes = {}
    for b in behaviors:
        mode = b.get("evidence_policy", {}).get("mode", "unknown")
        modes[mode] = modes.get(mode, 0) + 1

    # Sample behaviors
    samples = []
    for b in behaviors[:5]:
        samples.append({
            "behavior_id": b["behavior_id"],
            "label_ar": b["label_ar"],
            "label_en": b["label_en"],
            "mode": b["evidence_policy"]["mode"],
            "forms_count": len(b.get("evidence_policy", {}).get("lexical_spec", {}).get("forms", []))
        })

    return {
        "phase": 3,
        "description": "Behavior Registry with Evidence Policy",
        "statistics": {
            "total_behaviors": len(behaviors),
            "by_category": categories,
            "by_evidence_mode": modes
        },
        "validation": validation,
        "sample_behaviors": samples
    }


def main() -> int:
    """Build behavior registry."""
    print("=" * 60)
    print("Phase 3: Build Behavior Registry")
    print("=" * 60)

    try:
        # Build registry
        registry = build_registry()

        # Validate
        print("\nValidating registry...")
        validation = validate_registry(registry)

        for check, passed in validation["checks"].items():
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {check}")

        # Save registry
        registry_path = Path("data/behaviors/behavior_registry.json")
        print(f"\nSaving registry to {registry_path}...")
        save_json(registry_path, registry)

        # Generate and save report
        report = generate_report(registry, validation)
        report_path = Path("artifacts/behavior_registry_report.json")
        save_json(report_path, report)
        print(f"Saved report to {report_path}")

        # Final result
        print("\n" + "=" * 60)
        if validation["valid"]:
            print("Phase 3: BEHAVIOR REGISTRY BUILT SUCCESSFULLY")
            print(f"  Total behaviors: {registry['total_behaviors']}")
            print(f"  Lexical: {registry['lexical_behaviors']}")
            print(f"  Annotation: {registry['annotation_behaviors']}")
            print("=" * 60)
            return 0
        else:
            print("Phase 3: VALIDATION FAILED")
            print("=" * 60)
            return 1

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
