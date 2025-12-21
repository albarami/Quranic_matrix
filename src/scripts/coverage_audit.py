#!/usr/bin/env python3
"""
Coverage audit for QBM annotations.

Usage:
    python src/scripts/coverage_audit.py --annotations data/annotations/ --output reports/coverage/
    python src/scripts/coverage_audit.py --annotations data/exports/qbm_gold.json --create-batches
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Total ayat per surah
SURAH_AYAT = [
    7, 286, 200, 176, 120, 165, 206, 75, 129, 109,
    123, 111, 43, 52, 99, 128, 111, 110, 98, 135,
    112, 78, 118, 64, 77, 227, 93, 88, 69, 60,
    34, 30, 73, 54, 45, 83, 182, 88, 75, 85,
    54, 53, 89, 59, 37, 35, 38, 29, 18, 45,
    60, 49, 62, 55, 78, 96, 29, 22, 24, 13,
    14, 11, 11, 18, 12, 12, 30, 52, 52, 44,
    28, 28, 20, 56, 40, 31, 50, 40, 46, 42,
    29, 19, 36, 25, 22, 17, 19, 26, 30, 20,
    15, 21, 11, 8, 8, 19, 5, 8, 8, 11,
    11, 8, 3, 9, 5, 4, 7, 3, 6, 3,
    5, 4, 5, 6
]

TOTAL_AYAT = sum(SURAH_AYAT)  # 6236


def load_annotations(path: Path) -> List[Dict]:
    """Load annotations from file or directory."""
    spans = []
    
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            if path.suffix == ".jsonl":
                spans = [json.loads(line) for line in f if line.strip()]
            else:
                data = json.load(f)
                spans = data if isinstance(data, list) else data.get("spans", [])
    elif path.is_dir():
        for file in path.glob("**/*.json"):
            with open(file, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    spans.extend(data)
                else:
                    spans.extend(data.get("spans", []))
        for file in path.glob("**/*.jsonl"):
            with open(file, encoding="utf-8") as f:
                spans.extend([json.loads(line) for line in f if line.strip()])
    
    return spans


def compute_coverage(spans: List[Dict]) -> Dict:
    """Compute coverage statistics."""
    covered: Dict[int, Set[int]] = {s: set() for s in range(1, 115)}
    
    for span in spans:
        ref = span.get("reference", {})
        surah = ref.get("surah")
        ayah = ref.get("ayah")
        if surah and ayah:
            covered[surah].add(ayah)
    
    # Compute stats
    surah_stats = {}
    for surah in range(1, 115):
        total = SURAH_AYAT[surah - 1]
        annotated = len(covered[surah])
        surah_stats[surah] = {
            "total": total,
            "annotated": annotated,
            "missing": total - annotated,
            "percent": round(100 * annotated / total, 1) if total > 0 else 0,
            "missing_ayat": sorted(set(range(1, total + 1)) - covered[surah])
        }
    
    total_annotated = sum(len(v) for v in covered.values())
    
    return {
        "summary": {
            "total_ayat": TOTAL_AYAT,
            "annotated_ayat": total_annotated,
            "missing_ayat": TOTAL_AYAT - total_annotated,
            "percent_complete": round(100 * total_annotated / TOTAL_AYAT, 2),
            "surahs_complete": sum(1 for s in surah_stats.values() if s["missing"] == 0),
            "surahs_partial": sum(1 for s in surah_stats.values() if 0 < s["annotated"] < s["total"]),
            "surahs_empty": sum(1 for s in surah_stats.values() if s["annotated"] == 0)
        },
        "by_surah": surah_stats
    }


def find_gaps(coverage: Dict) -> List[Tuple[int, int]]:
    """Find all missing ayat."""
    gaps = []
    for surah, stats in coverage["by_surah"].items():
        for ayah in stats["missing_ayat"]:
            gaps.append((int(surah), ayah))
    return gaps


def create_gap_batches(gaps: List[Tuple[int, int]], batch_size: int = 50) -> List[Dict]:
    """Create batches from gaps for annotation."""
    batches = []
    current_batch = []
    
    for surah, ayah in sorted(gaps):
        current_batch.append({"surah": surah, "ayah": ayah})
        if len(current_batch) >= batch_size:
            batches.append({
                "batch_id": f"gap_batch_{len(batches) + 1}",
                "ayat": current_batch,
                "count": len(current_batch)
            })
            current_batch = []
    
    if current_batch:
        batches.append({
            "batch_id": f"gap_batch_{len(batches) + 1}",
            "ayat": current_batch,
            "count": len(current_batch)
        })
    
    return batches


def main():
    import argparse
    parser = argparse.ArgumentParser(description="QBM coverage audit")
    parser.add_argument("--annotations", required=True, help="Annotations file or directory")
    parser.add_argument("--output", default="reports/coverage", help="Output directory")
    parser.add_argument("--create-batches", action="store_true", help="Create gap batches")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for gaps")
    args = parser.parse_args()
    
    annotations_path = Path(args.annotations)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and analyze
    print(f"Loading annotations from {annotations_path}...")
    spans = load_annotations(annotations_path)
    print(f"Loaded {len(spans)} spans")
    
    coverage = compute_coverage(spans)
    summary = coverage["summary"]
    
    # Print summary
    print(f"\n{'='*50}")
    print("COVERAGE AUDIT SUMMARY")
    print(f"{'='*50}")
    print(f"Total Quran ayat: {summary['total_ayat']}")
    print(f"Annotated ayat:   {summary['annotated_ayat']} ({summary['percent_complete']}%)")
    print(f"Missing ayat:     {summary['missing_ayat']}")
    print(f"\nSurahs complete:  {summary['surahs_complete']}/114")
    print(f"Surahs partial:   {summary['surahs_partial']}/114")
    print(f"Surahs empty:     {summary['surahs_empty']}/114")
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"coverage_report_{timestamp}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(coverage, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved: {report_file}")
    
    # Create gap batches if requested
    if args.create_batches and summary["missing_ayat"] > 0:
        gaps = find_gaps(coverage)
        batches = create_gap_batches(gaps, args.batch_size)
        
        batches_dir = output_dir / "gap_batches"
        batches_dir.mkdir(exist_ok=True)
        
        for batch in batches:
            batch_file = batches_dir / f"{batch['batch_id']}.json"
            with open(batch_file, "w", encoding="utf-8") as f:
                json.dump(batch, f, indent=2)
        
        print(f"\nCreated {len(batches)} gap batches in {batches_dir}")
        print(f"Total gaps to fill: {len(gaps)} ayat")


if __name__ == "__main__":
    main()
