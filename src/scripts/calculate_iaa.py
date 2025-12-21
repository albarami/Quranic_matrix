#!/usr/bin/env python3
"""
Calculate Inter-Annotator Agreement (IAA) for QBM annotations.

Computes Cohen's Kappa and percentage agreement for overlapping annotations.

Usage:
    python src/scripts/calculate_iaa.py data/annotations/expert/ --output reports/iaa/
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from datetime import datetime


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


def get_span_key(span: Dict) -> str:
    """Generate unique key for a span based on reference."""
    ref = span.get("reference", {})
    return f"{ref.get('surah', 0)}:{ref.get('ayah', 0)}"


def find_overlapping_spans(
    annotations1: List[Dict], 
    annotations2: List[Dict]
) -> List[Tuple[Dict, Dict]]:
    """Find spans that annotate the same ayah."""
    spans1_by_key = {get_span_key(s): s for s in annotations1}
    spans2_by_key = {get_span_key(s): s for s in annotations2}
    
    overlapping = []
    for key in spans1_by_key:
        if key in spans2_by_key:
            overlapping.append((spans1_by_key[key], spans2_by_key[key]))
    
    return overlapping


def calculate_agreement_for_field(
    pairs: List[Tuple[Dict, Dict]], 
    field_path: List[str]
) -> Dict[str, float]:
    """Calculate agreement for a specific field."""
    if not pairs:
        return {"agreement": 0.0, "kappa": 0.0, "n": 0}
    
    def get_nested(d: Dict, path: List[str]) -> Any:
        for key in path:
            if isinstance(d, dict):
                d = d.get(key)
            else:
                return None
        return d
    
    agreements = 0
    total = 0
    values1 = []
    values2 = []
    
    for span1, span2 in pairs:
        val1 = get_nested(span1, field_path)
        val2 = get_nested(span2, field_path)
        
        if val1 is not None and val2 is not None:
            total += 1
            values1.append(str(val1))
            values2.append(str(val2))
            if val1 == val2:
                agreements += 1
    
    if total == 0:
        return {"agreement": 0.0, "kappa": 0.0, "n": 0}
    
    # Percentage agreement
    agreement = agreements / total
    
    # Cohen's Kappa
    kappa = calculate_cohens_kappa(values1, values2)
    
    return {
        "agreement": round(agreement, 4),
        "kappa": round(kappa, 4),
        "n": total,
        "matches": agreements
    }


def calculate_cohens_kappa(values1: List[str], values2: List[str]) -> float:
    """Calculate Cohen's Kappa coefficient."""
    if len(values1) != len(values2) or len(values1) == 0:
        return 0.0
    
    n = len(values1)
    
    # Get all unique categories
    categories = list(set(values1) | set(values2))
    
    # Build confusion matrix
    matrix = defaultdict(lambda: defaultdict(int))
    for v1, v2 in zip(values1, values2):
        matrix[v1][v2] += 1
    
    # Calculate observed agreement (Po)
    po = sum(matrix[c][c] for c in categories) / n
    
    # Calculate expected agreement (Pe)
    pe = 0.0
    for c in categories:
        p1 = sum(1 for v in values1 if v == c) / n
        p2 = sum(1 for v in values2 if v == c) / n
        pe += p1 * p2
    
    # Kappa
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    
    kappa = (po - pe) / (1 - pe)
    return kappa


def calculate_iaa(annotations_by_annotator: Dict[str, List[Dict]]) -> Dict:
    """Calculate IAA across all annotator pairs."""
    annotators = list(annotations_by_annotator.keys())
    
    if len(annotators) < 2:
        # Single annotator - calculate self-consistency metrics
        return {
            "annotators": annotators,
            "pairs": 0,
            "note": "Single annotator - no IAA calculation possible",
            "fields": {}
        }
    
    # Fields to check for agreement
    fields_to_check = [
        (["agent", "type"], "agent_type"),
        (["behavior_form"], "behavior_form"),
        (["normative", "evaluation"], "evaluation"),
        (["normative", "deontic_signal"], "deontic_signal"),
        (["normative", "speech_mode"], "speech_mode"),
        (["action", "textual_eval"], "textual_eval"),
        (["axes", "systemic"], "systemic"),
    ]
    
    all_pairs = []
    pair_results = []
    
    # Calculate for each annotator pair
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
                
                for field_path, field_name in fields_to_check:
                    pair_result["fields"][field_name] = calculate_agreement_for_field(
                        overlapping, field_path
                    )
                
                pair_results.append(pair_result)
    
    # Aggregate across all pairs
    aggregate_fields = {}
    for field_path, field_name in fields_to_check:
        aggregate_fields[field_name] = calculate_agreement_for_field(
            all_pairs, field_path
        )
    
    return {
        "annotators": annotators,
        "total_overlapping_spans": len(all_pairs),
        "pair_results": pair_results,
        "aggregate": aggregate_fields,
        "calculated_at": datetime.utcnow().isoformat()
    }


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
    
    # Print summary
    print("\n" + "=" * 50)
    print("IAA SUMMARY")
    print("=" * 50)
    
    if "note" in results:
        print(f"Note: {results['note']}")
    else:
        print(f"Total overlapping spans: {results['total_overlapping_spans']}")
        print("\nAggregate Agreement:")
        for field, metrics in results.get("aggregate", {}).items():
            print(f"  {field}:")
            print(f"    Agreement: {metrics['agreement']:.2%}")
            print(f"    Kappa: {metrics['kappa']:.4f}")
            print(f"    N: {metrics['n']}")
    
    # Save report
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
