#!/usr/bin/env python3
"""
Calculate Inter-Annotator Agreement (IAA) for QBM annotations.

Computes:
- Cohen's Kappa for categorical fields
- Jaccard similarity for multi-label fields (e.g., behavior.concepts)
- Krippendorff's alpha (optional, if krippendorff package installed)
- Percentage agreement

Aligned with PROJECT_PLAN.md requirements.

Usage:
    python src/scripts/calculate_iaa.py data/annotations/expert/ --output reports/iaa/
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import defaultdict
from datetime import datetime

try:
    import krippendorff
    import numpy as np
    HAS_KRIPPENDORFF = True
except ImportError:
    HAS_KRIPPENDORFF = False


def load_annotations(path: Path) -> Dict[str, List[Dict]]:
    """Load annotations grouped by annotator."""
    annotations_by_annotator: Dict[str, List[Dict]] = defaultdict(list)
    
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
            annotator = data.get("annotator", "unknown")
            spans = data.get("annotations", data.get("spans", []))
            annotations_by_annotator[annotator].extend(spans)
    elif path.is_dir():
        for file in path.glob("**/*.json"):
            with open(file, encoding="utf-8") as f:
                data = json.load(f)
                annotator = data.get("annotator", "unknown")
                spans = data.get("annotations", data.get("spans", []))
                annotations_by_annotator[annotator].extend(spans)
    
    return dict(annotations_by_annotator)


# Track spans missing proper identifiers for warnings
_spans_missing_id = []


def get_span_key(span: Dict) -> str:
    """Generate unique key for a span at span-level (not just ayah-level).
    
    Uses span_id if available, otherwise falls back to reference + token positions.
    Warns if span lacks both span_id and token bounds (may collapse spans).
    """
    span_id = span.get("span_id") or span.get("id")
    if span_id:
        return span_id
    
    ref = span.get("reference", {})
    surah = ref.get("surah", 0)
    ayah = ref.get("ayah", 0)
    token_start = span.get("token_start", span.get("span", {}).get("token_start"))
    token_end = span.get("token_end", span.get("span", {}).get("token_end"))
    
    # Warn if no token bounds - may collapse multiple spans in same ayah
    if token_start is None and token_end is None:
        _spans_missing_id.append(f"{surah}:{ayah}")
        return f"{surah}:{ayah}:0-0"  # Fallback, but tracked
    
    return f"{surah}:{ayah}:{token_start or 0}-{token_end or 0}"


def get_nested(d: Dict, path: str) -> Any:
    """Get nested value using dot notation (e.g., 'agent.type')."""
    keys = path.split(".")
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key)
        else:
            return None
    return d


def get_field_value(span: Dict, field_path: str) -> Any:
    """Get field value with fallback for schema variations."""
    value = get_nested(span, field_path)
    if value is not None:
        return value
    
    fallbacks = {
        "agent.type": ["agent.type"],
        "behavior.form": ["behavior_form", "behavior.form"],
        "behavior.concepts": ["behavior_concepts", "behavior.concepts"],
        "normative_textual.speech_mode": ["normative.speech_mode", "normative_textual.speech_mode"],
        "normative_textual.evaluation": ["normative.evaluation", "normative_textual.evaluation"],
        "normative_textual.quran_deontic_signal": ["normative.deontic_signal", "normative_textual.quran_deontic_signal"],
        "action.textual_eval": ["action.textual_eval"],
        "axes.systemic": ["axes.systemic"],
    }
    
    for fallback in fallbacks.get(field_path, []):
        value = get_nested(span, fallback)
        if value is not None:
            return value
    
    return None


def find_overlapping_spans(
    annotations1: List[Dict], 
    annotations2: List[Dict]
) -> List[Tuple[Dict, Dict]]:
    """Find spans that annotate the same span (span-level matching)."""
    spans1_by_key = {get_span_key(s): s for s in annotations1}
    spans2_by_key = {get_span_key(s): s for s in annotations2}
    
    overlapping = []
    for key in spans1_by_key:
        if key in spans2_by_key:
            overlapping.append((spans1_by_key[key], spans2_by_key[key]))
    
    return overlapping


def calculate_cohens_kappa(values1: List[str], values2: List[str]) -> float:
    """Calculate Cohen's Kappa coefficient for categorical agreement."""
    if len(values1) != len(values2) or len(values1) == 0:
        return 0.0
    
    n = len(values1)
    categories = list(set(values1) | set(values2))
    
    matrix = defaultdict(lambda: defaultdict(int))
    for v1, v2 in zip(values1, values2):
        matrix[v1][v2] += 1
    
    po = sum(matrix[c][c] for c in categories) / n
    
    pe = 0.0
    for c in categories:
        p1 = sum(1 for v in values1 if v == c) / n
        p2 = sum(1 for v in values2 if v == c) / n
        pe += p1 * p2
    
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    
    return (po - pe) / (1 - pe)


def calculate_jaccard(set1: Set, set2: Set) -> float:
    """Calculate Jaccard similarity for multi-label fields."""
    if not set1 and not set2:
        return 1.0
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0


def calculate_jaccard_for_pairs(
    pairs: List[Tuple[Dict, Dict]], 
    field_path: str
) -> Dict[str, float]:
    """Calculate Jaccard similarity for a multi-label field across pairs."""
    if not pairs:
        return {"jaccard": 0.0, "n": 0}
    
    scores = []
    for span1, span2 in pairs:
        val1 = get_field_value(span1, field_path)
        val2 = get_field_value(span2, field_path)
        
        set1 = set(val1) if isinstance(val1, (list, tuple)) else set()
        set2 = set(val2) if isinstance(val2, (list, tuple)) else set()
        
        if set1 or set2:
            scores.append(calculate_jaccard(set1, set2))
    
    if not scores:
        return {"jaccard": 0.0, "n": 0}
    
    return {
        "jaccard": round(sum(scores) / len(scores), 4),
        "n": len(scores)
    }


def calculate_krippendorff_alpha(
    annotations_by_annotator: Dict[str, List[Dict]],
    field_path: str
) -> Optional[float]:
    """Calculate Krippendorff's alpha for a field."""
    if not HAS_KRIPPENDORFF:
        return None
    
    annotators = list(annotations_by_annotator.keys())
    if len(annotators) < 2:
        return None
    
    all_span_ids = set()
    for ann_spans in annotations_by_annotator.values():
        for span in ann_spans:
            all_span_ids.add(get_span_key(span))
    
    if not all_span_ids:
        return None
    
    all_values = set()
    for ann_spans in annotations_by_annotator.values():
        for span in ann_spans:
            val = get_field_value(span, field_path)
            if val is not None:
                all_values.add(str(val))
    
    if not all_values:
        return None
    
    value_to_idx = {v: i for i, v in enumerate(sorted(all_values))}
    span_id_list = sorted(all_span_ids)
    reliability_data = []
    
    for annotator in annotators:
        spans_by_key = {get_span_key(s): s for s in annotations_by_annotator[annotator]}
        row = []
        for span_id in span_id_list:
            if span_id in spans_by_key:
                val = get_field_value(spans_by_key[span_id], field_path)
                if val is not None:
                    row.append(value_to_idx.get(str(val), np.nan))
                else:
                    row.append(np.nan)
            else:
                row.append(np.nan)
        reliability_data.append(row)
    
    try:
        alpha = krippendorff.alpha(reliability_data=reliability_data, level_of_measurement='nominal')
        return round(alpha, 4) if not np.isnan(alpha) else None
    except Exception:
        return None


def calculate_agreement_for_field(
    pairs: List[Tuple[Dict, Dict]], 
    field_path: str,
    method: str = "kappa"
) -> Dict[str, Any]:
    """Calculate agreement for a specific field."""
    if not pairs:
        return {"agreement": 0.0, "kappa": 0.0, "n": 0}
    
    if method == "jaccard":
        return calculate_jaccard_for_pairs(pairs, field_path)
    
    agreements = 0
    total = 0
    values1 = []
    values2 = []
    
    for span1, span2 in pairs:
        val1 = get_field_value(span1, field_path)
        val2 = get_field_value(span2, field_path)
        
        if val1 is not None and val2 is not None:
            total += 1
            values1.append(str(val1))
            values2.append(str(val2))
            if val1 == val2:
                agreements += 1
    
    if total == 0:
        return {"agreement": 0.0, "kappa": 0.0, "n": 0}
    
    agreement = agreements / total
    kappa = calculate_cohens_kappa(values1, values2)
    
    return {
        "agreement": round(agreement, 4),
        "kappa": round(kappa, 4),
        "n": total,
        "matches": agreements
    }


def calculate_iaa(annotations_by_annotator: Dict[str, List[Dict]]) -> Dict:
    """Calculate IAA across all annotator pairs.
    
    Per PROJECT_PLAN.md:
    - agent.type: kappa
    - behavior.form: kappa
    - behavior.concepts: jaccard (multi-label)
    - normative_textual.speech_mode: kappa
    - normative_textual.evaluation: kappa
    - normative_textual.quran_deontic_signal: kappa
    """
    annotators = list(annotations_by_annotator.keys())
    
    if len(annotators) < 2:
        return {
            "annotators": annotators,
            "pairs": 0,
            "note": "Single annotator - no IAA calculation possible",
            "fields": {}
        }
    
    fields_to_check = [
        ("agent.type", "kappa"),
        ("behavior.form", "kappa"),
        ("behavior.concepts", "jaccard"),
        ("normative_textual.speech_mode", "kappa"),
        ("normative_textual.evaluation", "kappa"),
        ("normative_textual.quran_deontic_signal", "kappa"),
        ("action.textual_eval", "kappa"),
        ("axes.systemic", "kappa"),
    ]
    
    all_pairs = []
    pair_results = []
    
    for i, ann1 in enumerate(annotators):
        for ann2 in annotators[i+1:]:
            overlapping = find_overlapping_spans(
                annotations_by_annotator[ann1],
                annotations_by_annotator[ann2]
            )
            all_pairs.extend(overlapping)
            
            if overlapping:
                pair_result = {
                    "annotators": [ann1, ann2],
                    "overlapping_spans": len(overlapping),
                    "fields": {}
                }
                
                for field_path, method in fields_to_check:
                    pair_result["fields"][field_path] = calculate_agreement_for_field(
                        overlapping, field_path, method
                    )
                
                pair_results.append(pair_result)
    
    aggregate_fields = {}
    for field_path, method in fields_to_check:
        aggregate_fields[field_path] = calculate_agreement_for_field(
            all_pairs, field_path, method
        )
    
    krippendorff_alpha = {}
    if HAS_KRIPPENDORFF:
        for field_path, method in fields_to_check:
            if method == "kappa":
                alpha = calculate_krippendorff_alpha(annotations_by_annotator, field_path)
                if alpha is not None:
                    krippendorff_alpha[field_path] = alpha
    
    result = {
        "annotators": annotators,
        "total_overlapping_spans": len(all_pairs),
        "pair_results": pair_results,
        "aggregate": aggregate_fields,
        "calculated_at": datetime.utcnow().isoformat(),
        "methods": {
            "categorical": "Cohen's Kappa",
            "multi_label": "Jaccard similarity",
        }
    }
    
    if krippendorff_alpha:
        result["krippendorff_alpha"] = krippendorff_alpha
    elif not HAS_KRIPPENDORFF:
        result["note_krippendorff"] = "Install krippendorff package for Krippendorff's alpha"
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Calculate IAA for QBM annotations")
    parser.add_argument("path", help="Path to annotations file or directory")
    parser.add_argument("--output", "-o", help="Output directory for report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    path = Path(args.path)
    
    print(f"Loading annotations from {path}...")
    annotations_by_annotator = load_annotations(path)
    
    print(f"Found {len(annotations_by_annotator)} annotator(s):")
    for ann, spans in annotations_by_annotator.items():
        print(f"  {ann}: {len(spans)} spans")
    
    print("\nCalculating IAA...")
    results = calculate_iaa(annotations_by_annotator)
    
    print("\n" + "=" * 60)
    print("INTER-ANNOTATOR AGREEMENT REPORT")
    print("=" * 60)
    
    if "note" in results:
        print(f"Note: {results['note']}")
    else:
        print(f"Total overlapping spans: {results['total_overlapping_spans']}")
        print("\nAggregate Agreement:")
        for field, metrics in results.get("aggregate", {}).items():
            if "kappa" in metrics:
                print(f"  {field}:")
                print(f"    Agreement: {metrics['agreement']:.2%}")
                print(f"    Kappa: {metrics['kappa']:.4f}")
                print(f"    N: {metrics['n']}")
            elif "jaccard" in metrics:
                print(f"  {field}:")
                print(f"    Jaccard: {metrics['jaccard']:.4f}")
                print(f"    N: {metrics['n']}")
        
        if results.get("krippendorff_alpha"):
            print("\nKrippendorff's Alpha:")
            for field, alpha in results["krippendorff_alpha"].items():
                print(f"  {field}: {alpha:.4f}")
    
    # Warn about spans missing proper identifiers
    if _spans_missing_id:
        unique_missing = set(_spans_missing_id)
        print(f"\nWARNING: {len(unique_missing)} spans lack span_id and token bounds.")
        print("  These may collapse multiple spans in the same ayah.")
        if args.verbose:
            for ref in sorted(unique_missing)[:10]:
                print(f"    - {ref}")
            if len(unique_missing) > 10:
                print(f"    ... and {len(unique_missing) - 10} more")
        results["warnings"] = {
            "spans_missing_id": len(unique_missing),
            "affected_ayat": sorted(unique_missing)[:100]
        }
    
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"iaa_report_{timestamp}.json"
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\nReport saved: {output_file}")


if __name__ == "__main__":
    main()
