#!/usr/bin/env python3
"""
Export QBM annotations to tiered datasets (gold, silver, research).

Tiers:
- Gold: Fully reviewed, high confidence, IAA verified
- Silver: Quality checked, single annotator
- Research: All annotations including drafts

Usage:
    python src/scripts/export_tiers.py data/annotations/expert/ --output data/exports/
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from collections import defaultdict


def load_all_annotations(path: Path) -> List[Dict]:
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


def calculate_span_quality(span: Dict) -> Dict[str, Any]:
    """Calculate quality metrics for a span."""
    quality = {
        "has_agent": bool(span.get("agent", {}).get("type")),
        "has_behavior_form": bool(span.get("behavior_form")),
        "has_evaluation": bool(span.get("normative", {}).get("evaluation")),
        "has_deontic": bool(span.get("normative", {}).get("deontic_signal")),
        "has_text": bool(span.get("text_ar")),
        "has_reference": bool(span.get("reference", {}).get("surah")),
    }
    
    # Calculate completeness score
    quality["completeness"] = sum(quality.values()) / len(quality)
    
    # Determine tier eligibility
    quality["gold_eligible"] = quality["completeness"] == 1.0
    quality["silver_eligible"] = quality["completeness"] >= 0.8
    
    return quality


def classify_tier(span: Dict, quality: Dict) -> str:
    """Classify a span into a tier."""
    # Check for review status
    review_status = span.get("review", {}).get("status", "draft")
    
    if review_status == "approved" and quality["gold_eligible"]:
        return "gold"
    elif quality["silver_eligible"]:
        return "silver"
    else:
        return "research"


def export_tier(spans: List[Dict], tier: str, output_dir: Path, timestamp: str) -> Path:
    """Export a tier to a JSON file."""
    output_file = output_dir / f"qbm_{tier}_{timestamp}.json"
    
    export_data = {
        "tier": tier,
        "version": "1.0.0",
        "exported_at": datetime.utcnow().isoformat(),
        "total_spans": len(spans),
        "spans": spans
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)
    
    return output_file


def generate_statistics(spans: List[Dict]) -> Dict:
    """Generate statistics for a set of spans."""
    stats = {
        "total_spans": len(spans),
        "surahs": set(),
        "ayat": set(),
        "agent_types": defaultdict(int),
        "behavior_forms": defaultdict(int),
        "evaluations": defaultdict(int),
        "deontic_signals": defaultdict(int),
    }
    
    for span in spans:
        ref = span.get("reference", {})
        if ref.get("surah"):
            stats["surahs"].add(ref["surah"])
            stats["ayat"].add(f"{ref['surah']}:{ref.get('ayah', 0)}")
        
        agent_type = span.get("agent", {}).get("type", "unknown")
        stats["agent_types"][agent_type] += 1
        
        behavior_form = span.get("behavior_form", "unknown")
        stats["behavior_forms"][behavior_form] += 1
        
        evaluation = span.get("normative", {}).get("evaluation", "unknown")
        stats["evaluations"][evaluation] += 1
        
        deontic = span.get("normative", {}).get("deontic_signal", "unknown")
        stats["deontic_signals"][deontic] += 1
    
    # Convert sets to counts
    stats["unique_surahs"] = len(stats["surahs"])
    stats["unique_ayat"] = len(stats["ayat"])
    del stats["surahs"]
    del stats["ayat"]
    
    # Convert defaultdicts to regular dicts
    stats["agent_types"] = dict(stats["agent_types"])
    stats["behavior_forms"] = dict(stats["behavior_forms"])
    stats["evaluations"] = dict(stats["evaluations"])
    stats["deontic_signals"] = dict(stats["deontic_signals"])
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Export QBM annotations to tiers")
    parser.add_argument("path", help="Path to annotations file or directory")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--force-tier", choices=["gold", "silver", "research"],
                        help="Force all spans to a specific tier")
    args = parser.parse_args()
    
    path = Path(args.path)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading annotations from {path}...")
    all_spans = load_all_annotations(path)
    print(f"Loaded {len(all_spans)} spans")
    
    # Classify spans by tier
    tiers = {"gold": [], "silver": [], "research": []}
    
    for span in all_spans:
        quality = calculate_span_quality(span)
        
        if args.force_tier:
            tier = args.force_tier
        else:
            tier = classify_tier(span, quality)
        
        # Add quality metadata
        span["_quality"] = quality
        tiers[tier].append(span)
    
    # Export each tier
    timestamp = datetime.now().strftime("%Y%m%d")
    
    print("\n" + "=" * 50)
    print("EXPORT SUMMARY")
    print("=" * 50)
    
    for tier, spans in tiers.items():
        if spans:
            output_file = export_tier(spans, tier, output_dir, timestamp)
            stats = generate_statistics(spans)
            
            print(f"\n{tier.upper()} Tier:")
            print(f"  Spans: {stats['total_spans']}")
            print(f"  Surahs: {stats['unique_surahs']}")
            print(f"  Ayat: {stats['unique_ayat']}")
            print(f"  File: {output_file}")
    
    # Save combined statistics
    all_stats = {
        "exported_at": datetime.utcnow().isoformat(),
        "tiers": {
            tier: generate_statistics(spans) for tier, spans in tiers.items()
        }
    }
    
    stats_file = output_dir / f"export_stats_{timestamp}.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    
    print(f"\nStatistics saved: {stats_file}")


if __name__ == "__main__":
    main()
