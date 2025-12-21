#!/usr/bin/env python3
"""
Generate progress report for QBM annotation project.

Produces comprehensive statistics on annotation coverage, quality, and distribution.

Usage:
    python src/scripts/progress_report.py data/annotations/expert/ --output reports/
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from collections import defaultdict

# Surah metadata
SURAH_AYAT = {
    1: 7, 2: 286, 3: 200, 4: 176, 5: 120, 6: 165, 7: 206, 8: 75, 9: 129, 10: 109,
    11: 123, 12: 111, 13: 43, 14: 52, 15: 99, 16: 128, 17: 111, 18: 110, 19: 98, 20: 135,
    21: 112, 22: 78, 23: 118, 24: 64, 25: 77, 26: 227, 27: 93, 28: 88, 29: 69, 30: 60,
    31: 34, 32: 30, 33: 73, 34: 54, 35: 45, 36: 83, 37: 182, 38: 88, 39: 75, 40: 85,
    41: 54, 42: 53, 43: 89, 44: 59, 45: 37, 46: 35, 47: 38, 48: 29, 49: 18, 50: 45,
    51: 60, 52: 49, 53: 62, 54: 55, 55: 78, 56: 96, 57: 29, 58: 22, 59: 24, 60: 13,
    61: 14, 62: 11, 63: 11, 64: 18, 65: 12, 66: 12, 67: 30, 68: 52, 69: 52, 70: 44,
    71: 28, 72: 28, 73: 20, 74: 56, 75: 40, 76: 31, 77: 50, 78: 40, 79: 46, 80: 42,
    81: 29, 82: 19, 83: 36, 84: 25, 85: 22, 86: 17, 87: 19, 88: 26, 89: 30, 90: 20,
    91: 15, 92: 21, 93: 11, 94: 8, 95: 8, 96: 19, 97: 5, 98: 8, 99: 8, 100: 11,
    101: 11, 102: 8, 103: 3, 104: 9, 105: 5, 106: 4, 107: 7, 108: 3, 109: 6, 110: 3,
    111: 5, 112: 4, 113: 5, 114: 6
}

TOTAL_AYAT = sum(SURAH_AYAT.values())  # 6236


def load_annotations(path: Path) -> List[Dict]:
    """Load all annotations from a path."""
    all_spans = []
    
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
            spans = data.get("annotations", data.get("spans", []))
            all_spans.extend(spans)
    elif path.is_dir():
        for file in sorted(path.glob("**/*.json")):
            with open(file, encoding="utf-8") as f:
                data = json.load(f)
                spans = data.get("annotations", data.get("spans", []))
                all_spans.extend(spans)
    
    return all_spans


def calculate_coverage(spans: List[Dict]) -> Dict:
    """Calculate coverage statistics."""
    covered_ayat = set()
    surah_coverage = defaultdict(set)
    
    for span in spans:
        ref = span.get("reference", {})
        surah = ref.get("surah")
        ayah = ref.get("ayah")
        if surah and ayah:
            covered_ayat.add(f"{surah}:{ayah}")
            surah_coverage[surah].add(ayah)
    
    # Calculate per-surah coverage
    surah_stats = {}
    complete_surahs = 0
    partial_surahs = 0
    empty_surahs = 0
    
    for surah_num in range(1, 115):
        total = SURAH_AYAT[surah_num]
        covered = len(surah_coverage.get(surah_num, set()))
        percentage = (covered / total) * 100 if total > 0 else 0
        
        surah_stats[surah_num] = {
            "total": total,
            "covered": covered,
            "percentage": round(percentage, 2)
        }
        
        if covered == total:
            complete_surahs += 1
        elif covered > 0:
            partial_surahs += 1
        else:
            empty_surahs += 1
    
    return {
        "total_ayat": TOTAL_AYAT,
        "covered_ayat": len(covered_ayat),
        "coverage_percentage": round((len(covered_ayat) / TOTAL_AYAT) * 100, 2),
        "complete_surahs": complete_surahs,
        "partial_surahs": partial_surahs,
        "empty_surahs": empty_surahs,
        "surah_details": surah_stats
    }


def calculate_distributions(spans: List[Dict]) -> Dict:
    """Calculate distribution statistics."""
    distributions = {
        "agent_types": defaultdict(int),
        "behavior_forms": defaultdict(int),
        "evaluations": defaultdict(int),
        "deontic_signals": defaultdict(int),
        "speech_modes": defaultdict(int),
        "systemic": defaultdict(int),
        "textual_evals": defaultdict(int),
    }
    
    for span in spans:
        # Agent type
        agent_type = span.get("agent", {}).get("type", "unknown")
        distributions["agent_types"][agent_type] += 1
        
        # Behavior form
        behavior_form = span.get("behavior_form", "unknown")
        distributions["behavior_forms"][behavior_form] += 1
        
        # Normative layer
        normative = span.get("normative", {})
        distributions["evaluations"][normative.get("evaluation", "unknown")] += 1
        distributions["deontic_signals"][normative.get("deontic_signal", "unknown")] += 1
        distributions["speech_modes"][normative.get("speech_mode", "unknown")] += 1
        
        # Axes
        axes = span.get("axes", {})
        distributions["systemic"][axes.get("systemic", "unknown")] += 1
        
        # Action
        action = span.get("action", {})
        distributions["textual_evals"][action.get("textual_eval", "unknown")] += 1
    
    # Convert to regular dicts and sort by count
    result = {}
    for key, counts in distributions.items():
        sorted_items = sorted(counts.items(), key=lambda x: -x[1])
        result[key] = {k: v for k, v in sorted_items}
    
    return result


def generate_report(spans: List[Dict]) -> Dict:
    """Generate comprehensive progress report."""
    coverage = calculate_coverage(spans)
    distributions = calculate_distributions(spans)
    
    return {
        "generated_at": datetime.utcnow().isoformat(),
        "summary": {
            "total_spans": len(spans),
            "coverage": f"{coverage['coverage_percentage']}%",
            "ayat_covered": f"{coverage['covered_ayat']}/{coverage['total_ayat']}",
            "surahs_complete": f"{coverage['complete_surahs']}/114",
        },
        "coverage": coverage,
        "distributions": distributions
    }


def print_report(report: Dict):
    """Print formatted report to console."""
    print("\n" + "=" * 60)
    print("QBM PROGRESS REPORT")
    print("=" * 60)
    print(f"Generated: {report['generated_at']}")
    
    print("\n--- SUMMARY ---")
    for key, value in report["summary"].items():
        print(f"  {key}: {value}")
    
    print("\n--- COVERAGE ---")
    cov = report["coverage"]
    print(f"  Total Ayat: {cov['total_ayat']}")
    print(f"  Covered: {cov['covered_ayat']} ({cov['coverage_percentage']}%)")
    print(f"  Complete Surahs: {cov['complete_surahs']}/114")
    print(f"  Partial Surahs: {cov['partial_surahs']}/114")
    print(f"  Empty Surahs: {cov['empty_surahs']}/114")
    
    print("\n--- DISTRIBUTIONS ---")
    for dist_name, counts in report["distributions"].items():
        print(f"\n  {dist_name}:")
        for value, count in list(counts.items())[:5]:  # Top 5
            print(f"    {value}: {count}")
        if len(counts) > 5:
            print(f"    ... and {len(counts) - 5} more")


def main():
    parser = argparse.ArgumentParser(description="Generate QBM progress report")
    parser.add_argument("path", help="Path to annotations file or directory")
    parser.add_argument("--output", "-o", help="Output directory for report")
    parser.add_argument("--json", action="store_true", help="Output JSON only")
    args = parser.parse_args()
    
    path = Path(args.path)
    
    print(f"Loading annotations from {path}...")
    spans = load_annotations(path)
    print(f"Loaded {len(spans)} spans")
    
    report = generate_report(spans)
    
    if not args.json:
        print_report(report)
    
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"progress_report_{timestamp}.json"
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\nReport saved: {output_file}")
    
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
