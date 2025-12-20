"""
Compare pilot annotations against gold standard examples.

This script validates the quality of annotations by comparing
them to the gold standard calibration examples.
"""

import json
import os
from pathlib import Path

def load_gold_examples():
    """Load all gold standard examples."""
    base_path = Path(__file__).parent.parent.parent / "docs" / "coding_manual" / "examples"
    
    all_examples = []
    for part_file in ["gold_standard_examples_part_01_10.json", "gold_standard_examples_part_11_20.json"]:
        filepath = base_path / part_file
        if filepath.exists():
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
                all_examples.extend(data.get("examples", []))
    
    # Index by surah:ayah
    indexed = {}
    for ex in all_examples:
        ref = ex.get("reference", {})
        key = f"{ref.get('surah')}:{ref.get('ayah')}"
        indexed[key] = ex
    
    return indexed


def load_pilot_annotations():
    """Load pilot QBM records."""
    filepath = Path(__file__).parent.parent.parent / "data" / "exports" / "pilot_50_qbm_records.json"
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


def compare_annotation(pilot: dict, gold: dict) -> dict:
    """Compare a pilot annotation to gold standard."""
    results = {
        "reference": f"{pilot['reference']['surah']}:{pilot['reference']['ayah']}",
        "matches": [],
        "mismatches": [],
        "missing_in_pilot": [],
        "extra_in_pilot": [],
        "score": 0.0
    }
    
    total_checks = 0
    correct = 0
    
    # Compare agent type
    pilot_agent = pilot.get("agent", {}).get("type")
    gold_agent = gold.get("agent", {}).get("type")
    total_checks += 1
    if pilot_agent == gold_agent:
        results["matches"].append(f"agent_type: {pilot_agent}")
        correct += 1
    else:
        results["mismatches"].append(f"agent_type: pilot={pilot_agent}, gold={gold_agent}")
    
    # Compare behavior form
    pilot_form = pilot.get("behavior_form")
    gold_form = gold.get("behavior", {}).get("form")
    total_checks += 1
    if pilot_form == gold_form:
        results["matches"].append(f"behavior_form: {pilot_form}")
        correct += 1
    else:
        results["mismatches"].append(f"behavior_form: pilot={pilot_form}, gold={gold_form}")
    
    # Compare normative fields
    for field in ["speech_mode", "evaluation"]:
        pilot_val = pilot.get("normative", {}).get(field)
        gold_val = gold.get("normative_textual", {}).get(field)
        total_checks += 1
        if pilot_val == gold_val:
            results["matches"].append(f"{field}: {pilot_val}")
            correct += 1
        else:
            results["mismatches"].append(f"{field}: pilot={pilot_val}, gold={gold_val}")
    
    # Compare deontic signal
    pilot_deontic = pilot.get("normative", {}).get("deontic_signal")
    gold_deontic = gold.get("normative_textual", {}).get("quran_deontic_signal")
    total_checks += 1
    if pilot_deontic == gold_deontic:
        results["matches"].append(f"deontic_signal: {pilot_deontic}")
        correct += 1
    else:
        results["mismatches"].append(f"deontic_signal: pilot={pilot_deontic}, gold={gold_deontic}")
    
    # Calculate score
    results["score"] = correct / total_checks if total_checks > 0 else 0.0
    results["total_checks"] = total_checks
    results["correct"] = correct
    
    return results


def main():
    print("Loading gold standard examples...")
    gold_examples = load_gold_examples()
    print(f"Loaded {len(gold_examples)} gold examples")
    
    print("\nLoading pilot annotations...")
    pilot_records = load_pilot_annotations()
    print(f"Loaded {len(pilot_records)} pilot records")
    
    # Find overlapping references
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    comparisons = []
    for pilot in pilot_records:
        ref = pilot.get("reference", {})
        key = f"{ref.get('surah')}:{ref.get('ayah')}"
        
        if key in gold_examples:
            gold = gold_examples[key]
            comparison = compare_annotation(pilot, gold)
            comparisons.append(comparison)
            
            print(f"\n--- {key} ---")
            print(f"Score: {comparison['score']:.1%} ({comparison['correct']}/{comparison['total_checks']})")
            
            if comparison["matches"]:
                print("✓ Matches:")
                for m in comparison["matches"]:
                    print(f"    {m}")
            
            if comparison["mismatches"]:
                print("✗ Mismatches:")
                for m in comparison["mismatches"]:
                    print(f"    {m}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if comparisons:
        avg_score = sum(c["score"] for c in comparisons) / len(comparisons)
        total_correct = sum(c["correct"] for c in comparisons)
        total_checks = sum(c["total_checks"] for c in comparisons)
        
        print(f"Overlapping examples: {len(comparisons)}")
        print(f"Average score: {avg_score:.1%}")
        print(f"Total correct: {total_correct}/{total_checks}")
        
        # Breakdown by field
        field_stats = {}
        for c in comparisons:
            for m in c["matches"]:
                field = m.split(":")[0]
                field_stats[field] = field_stats.get(field, {"correct": 0, "total": 0})
                field_stats[field]["correct"] += 1
                field_stats[field]["total"] += 1
            for m in c["mismatches"]:
                field = m.split(":")[0]
                field_stats[field] = field_stats.get(field, {"correct": 0, "total": 0})
                field_stats[field]["total"] += 1
        
        print("\nField-level accuracy:")
        for field, stats in sorted(field_stats.items()):
            acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            print(f"  {field}: {acc:.1%} ({stats['correct']}/{stats['total']})")
    else:
        print("No overlapping examples found between pilot and gold standards.")
        print(f"Gold example references: {list(gold_examples.keys())[:10]}...")
        pilot_refs = [f"{r['reference']['surah']}:{r['reference']['ayah']}" for r in pilot_records[:10]]
        print(f"Pilot references: {pilot_refs}...")


if __name__ == "__main__":
    main()
