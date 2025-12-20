#!/usr/bin/env python3
"""
Calculate Inter-Annotator Agreement (IAA) for QBM annotations.

This script computes Krippendorff's alpha and Cohen's kappa for
comparing annotations from multiple annotators.

Usage:
    python calculate_iaa.py <annotator1_file> <annotator2_file>

Target: Krippendorff's α ≥ 0.7 for Phase 2 Micro-Pilot
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

try:
    import numpy as np
    from sklearn.metrics import cohen_kappa_score
except ImportError:
    print("Please install required packages: pip install numpy scikit-learn")
    sys.exit(1)

# Optional: krippendorff package for more accurate alpha
try:
    import krippendorff
    HAS_KRIPPENDORFF = True
except ImportError:
    HAS_KRIPPENDORFF = False


def load_annotations(filepath: str) -> dict:
    """Load annotations indexed by reference (surah:ayah)."""
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, list):
        records = data
    else:
        records = data.get("annotations", data.get("records", [data]))
    
    indexed = {}
    for rec in records:
        ref = rec.get("reference", {})
        key = f"{ref.get('surah')}:{ref.get('ayah')}"
        indexed[key] = rec
    
    return indexed


def extract_field_values(annotations: dict, field_path: list) -> dict:
    """Extract values for a specific field from all annotations."""
    values = {}
    for ref, ann in annotations.items():
        value = ann
        for key in field_path:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                value = None
                break
        values[ref] = value
    return values


def calculate_agreement_for_field(ann1: dict, ann2: dict, field_path: list) -> dict:
    """Calculate agreement metrics for a specific field."""
    vals1 = extract_field_values(ann1, field_path)
    vals2 = extract_field_values(ann2, field_path)
    
    # Find common references
    common_refs = set(vals1.keys()) & set(vals2.keys())
    
    if not common_refs:
        return {"error": "No common references found"}
    
    # Build parallel arrays
    y1 = []
    y2 = []
    agreements = 0
    
    for ref in sorted(common_refs):
        v1 = vals1[ref]
        v2 = vals2[ref]
        
        # Handle list values (e.g., systemic axis)
        if isinstance(v1, list):
            v1 = tuple(sorted(v1)) if v1 else None
        if isinstance(v2, list):
            v2 = tuple(sorted(v2)) if v2 else None
        
        y1.append(str(v1))
        y2.append(str(v2))
        
        if v1 == v2:
            agreements += 1
    
    # Calculate metrics
    n = len(common_refs)
    percent_agreement = agreements / n if n > 0 else 0
    
    # Cohen's Kappa
    try:
        kappa = cohen_kappa_score(y1, y2)
    except Exception:
        kappa = None
    
    # Krippendorff's Alpha (if available)
    alpha = None
    if HAS_KRIPPENDORFF:
        try:
            # Convert to numeric codes
            all_values = sorted(set(y1) | set(y2))
            value_to_code = {v: i for i, v in enumerate(all_values)}
            coded1 = [value_to_code[v] for v in y1]
            coded2 = [value_to_code[v] for v in y2]
            reliability_data = [coded1, coded2]
            alpha = krippendorff.alpha(reliability_data=reliability_data, level_of_measurement='nominal')
        except Exception:
            alpha = None
    
    return {
        "field": ".".join(field_path),
        "n_common": n,
        "agreements": agreements,
        "percent_agreement": round(percent_agreement * 100, 1),
        "cohens_kappa": round(kappa, 3) if kappa is not None else None,
        "krippendorffs_alpha": round(alpha, 3) if alpha is not None else None,
        "meets_threshold": (alpha or kappa or percent_agreement) >= 0.7 if (alpha or kappa) else percent_agreement >= 0.7
    }


def calculate_overall_iaa(ann1: dict, ann2: dict) -> dict:
    """Calculate IAA for all key fields."""
    fields_to_check = [
        ["agent", "type"],
        ["behavior_form"],
        ["normative", "speech_mode"],
        ["normative", "evaluation"],
        ["normative", "deontic_signal"],
        ["action", "class"],
        ["action", "textual_eval"],
        ["axes", "situational"],
        ["evidence", "support_type"],
    ]
    
    results = {
        "summary": {},
        "fields": []
    }
    
    total_kappa = 0
    total_alpha = 0
    valid_kappa_count = 0
    valid_alpha_count = 0
    
    for field_path in fields_to_check:
        field_result = calculate_agreement_for_field(ann1, ann2, field_path)
        results["fields"].append(field_result)
        
        if field_result.get("cohens_kappa") is not None:
            total_kappa += field_result["cohens_kappa"]
            valid_kappa_count += 1
        
        if field_result.get("krippendorffs_alpha") is not None:
            total_alpha += field_result["krippendorffs_alpha"]
            valid_alpha_count += 1
    
    # Calculate averages
    results["summary"] = {
        "total_fields": len(fields_to_check),
        "avg_cohens_kappa": round(total_kappa / valid_kappa_count, 3) if valid_kappa_count > 0 else None,
        "avg_krippendorffs_alpha": round(total_alpha / valid_alpha_count, 3) if valid_alpha_count > 0 else None,
        "fields_meeting_threshold": sum(1 for f in results["fields"] if f.get("meets_threshold", False)),
        "target_threshold": 0.7,
        "krippendorff_available": HAS_KRIPPENDORFF
    }
    
    return results


def print_results(results: dict):
    """Print IAA results in a readable format."""
    print("\n" + "="*60)
    print("INTER-ANNOTATOR AGREEMENT (IAA) RESULTS")
    print("="*60)
    
    print(f"\nTarget: Krippendorff's α ≥ {results['summary']['target_threshold']}")
    print(f"Krippendorff package available: {results['summary']['krippendorff_available']}")
    
    print("\n--- Field-Level Agreement ---")
    print(f"{'Field':<30} {'%Agree':>8} {'Kappa':>8} {'Alpha':>8} {'Pass':>6}")
    print("-" * 62)
    
    for field in results["fields"]:
        name = field.get("field", "unknown")[:28]
        pct = f"{field.get('percent_agreement', 0):.1f}%"
        kappa = f"{field.get('cohens_kappa', 0):.3f}" if field.get('cohens_kappa') is not None else "N/A"
        alpha = f"{field.get('krippendorffs_alpha', 0):.3f}" if field.get('krippendorffs_alpha') is not None else "N/A"
        passed = "✓" if field.get("meets_threshold") else "✗"
        print(f"{name:<30} {pct:>8} {kappa:>8} {alpha:>8} {passed:>6}")
    
    print("\n--- Summary ---")
    print(f"Fields meeting threshold: {results['summary']['fields_meeting_threshold']}/{results['summary']['total_fields']}")
    if results['summary']['avg_cohens_kappa']:
        print(f"Average Cohen's Kappa: {results['summary']['avg_cohens_kappa']:.3f}")
    if results['summary']['avg_krippendorffs_alpha']:
        print(f"Average Krippendorff's Alpha: {results['summary']['avg_krippendorffs_alpha']:.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python calculate_iaa.py <annotator1_file> <annotator2_file>")
        print("\nExample:")
        print("  python calculate_iaa.py annotations_ann1.json annotations_ann2.json")
        sys.exit(1)
    
    ann1 = load_annotations(sys.argv[1])
    ann2 = load_annotations(sys.argv[2])
    
    print(f"Loaded {len(ann1)} annotations from annotator 1")
    print(f"Loaded {len(ann2)} annotations from annotator 2")
    
    results = calculate_overall_iaa(ann1, ann2)
    print_results(results)
    
    # Output JSON for programmatic use
    output_path = Path("iaa_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to: {output_path}")
