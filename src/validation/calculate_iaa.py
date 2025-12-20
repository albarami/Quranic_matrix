#!/usr/bin/env python3
"""
Calculate Inter-Annotator Agreement (IAA) for QBM annotations.

This script computes Krippendorff's alpha and Cohen's kappa for
comparing annotations from multiple annotators.

Usage:
    python calculate_iaa.py <annotator1_file> <annotator2_file>

Target: Krippendorff's alpha >= 0.7 for Phase 2 Micro-Pilot
"""

import json
import sys
from pathlib import Path
from collections import Counter

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

LOW_VARIANCE_THRESHOLD = 5.0


def load_annotations(filepath: str) -> dict:
    """Load annotations indexed by span_id for exact matching."""
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, list):
        records = data
    else:
        records = data.get("annotations", data.get("records", [data]))
    
    indexed = {}
    for rec in records:
        # Key by span_id for exact span-level matching
        span_id = rec.get("span_id") or rec.get("id")
        if span_id:
            indexed[span_id] = rec
        else:
            # Fallback to reference if no span_id
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
    
    # Handle NaN values - convert to None for valid JSON
    if kappa is not None and (np.isnan(kappa) or np.isinf(kappa)):
        kappa = None
    if alpha is not None and (np.isnan(alpha) or np.isinf(alpha)):
        alpha = None
    
    # Calculate variance across both annotators (combined label distribution)
    all_values = y1 + y2
    variance_pct = 0.0
    if all_values:
        counts = Counter(all_values)
        most_common_count = counts.most_common(1)[0][1]
        variance_pct = (1 - most_common_count / len(all_values)) * 100
    variance_pct = round(variance_pct, 1)
    low_variance = variance_pct < LOW_VARIANCE_THRESHOLD
    
    # Determine if threshold is met (only if we have valid kappa or alpha)
    meets_threshold = False
    if low_variance:
        # <5% variance rule: mark as N/A, not pass/fail
        meets_threshold = None
    elif kappa is not None and kappa >= 0.7:
        meets_threshold = True
    elif alpha is not None and alpha >= 0.7:
        meets_threshold = True
    elif kappa is None and alpha is None and percent_agreement >= 70.0:
        # No variance case - use percent agreement but mark as N/A
        meets_threshold = None  # Indicates N/A, not pass/fail
    
    return {
        "field": ".".join(field_path),
        "n_common": n,
        "agreements": agreements,
        "percent_agreement": round(percent_agreement * 100, 1),
        "cohens_kappa": round(kappa, 3) if kappa is not None else None,
        "krippendorffs_alpha": round(alpha, 3) if alpha is not None else None,
        "meets_threshold": meets_threshold,
        "low_variance": low_variance,
        "variance_pct": variance_pct
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
    fields_with_variance = 0
    fields_meeting_threshold = 0
    
    for field_path in fields_to_check:
        field_result = calculate_agreement_for_field(ann1, ann2, field_path)
        results["fields"].append(field_result)
        
        # Only count fields with sufficient variance for averages
        if not field_result.get("low_variance", False):
            fields_with_variance += 1
            if field_result.get("cohens_kappa") is not None:
                total_kappa += field_result["cohens_kappa"]
                valid_kappa_count += 1
            
            if field_result.get("krippendorffs_alpha") is not None:
                total_alpha += field_result["krippendorffs_alpha"]
                valid_alpha_count += 1
            
            # Count threshold only for fields with variance
            if field_result.get("meets_threshold") is True:
                fields_meeting_threshold += 1
    
    # Calculate averages (excluding low-variance fields)
    results["summary"] = {
        "total_fields": len(fields_to_check),
        "fields_with_variance": fields_with_variance,
        "avg_cohens_kappa": round(total_kappa / valid_kappa_count, 3) if valid_kappa_count > 0 else None,
        "avg_krippendorffs_alpha": round(total_alpha / valid_alpha_count, 3) if valid_alpha_count > 0 else None,
        "fields_meeting_threshold": fields_meeting_threshold,
        "fields_evaluated": fields_with_variance,
        "target_threshold": 0.7,
        "variance_threshold": LOW_VARIANCE_THRESHOLD,
        "krippendorff_available": HAS_KRIPPENDORFF
    }
    
    return results


def print_results(results: dict):
    """Print IAA results in a readable format."""
    print("\n" + "=" * 60)
    print("INTER-ANNOTATOR AGREEMENT (IAA) RESULTS")
    print("=" * 60)

    print(f"\nTarget: Krippendorff's alpha >= {results['summary']['target_threshold']}")
    print(f"Krippendorff package available: {results['summary']['krippendorff_available']}")
    if results['summary'].get('variance_threshold') is not None:
        print(f"Low-variance cutoff: < {results['summary']['variance_threshold']}% => N/A")

    print("\n--- Field-Level Agreement ---")
    print(f"{'Field':<30} {'%Agree':>8} {'Kappa':>8} {'Alpha':>8} {'Var%':>8} {'Status':>8}")
    print("-" * 72)

    for field in results['fields']:
        name = field.get('field', 'unknown')[:28]
        pct = f"{field.get('percent_agreement', 0):.1f}%"
        kappa = f"{field.get('cohens_kappa', 0):.3f}" if field.get('cohens_kappa') is not None else "N/A"
        alpha = f"{field.get('krippendorffs_alpha', 0):.3f}" if field.get('krippendorffs_alpha') is not None else "N/A"
        variance = f"{field.get('variance_pct', 0):.1f}%"
        if field.get('meets_threshold') is None:
            status = "N/A"
        else:
            status = "PASS" if field.get('meets_threshold') else "FAIL"
        print(f"{name:<30} {pct:>8} {kappa:>8} {alpha:>8} {variance:>8} {status:>8}")

    print("\n--- Summary ---")
    evaluated = results['summary'].get('fields_evaluated', results['summary']['total_fields'])
    print(f"Fields meeting threshold: {results['summary']['fields_meeting_threshold']}/{evaluated} (excluding low-variance fields)")
    if results['summary']['avg_cohens_kappa'] is not None:
        print(f"Average Cohen's Kappa: {results['summary']['avg_cohens_kappa']:.3f}")
    if results['summary']['avg_krippendorffs_alpha'] is not None:
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
