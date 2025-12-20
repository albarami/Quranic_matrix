"""
Compare pilot annotations against gold standard examples.

FIXES from v1:
1. Handle multiple gold spans per verse (key by gold_id, not just surah:ayah)
2. ASCII-safe output for Windows compatibility
3. Coverage reporting for gold examples not in pilot
4. Verse-level comparison (pilot is verse-level, gold is span-level)

Since pilot annotations are verse-level (full ayah) and gold examples are
span-level (sub-ayah), we compare by finding the FIRST gold span for each
verse and note when multiple gold spans exist.
"""

import json
import sys
from pathlib import Path


def load_gold_examples():
    """Load all gold standard examples, preserving all spans."""
    base_path = Path(__file__).parent.parent.parent / "docs" / "coding_manual" / "examples"
    
    all_examples = []
    for part_file in ["gold_standard_examples_part_01_10.json", "gold_standard_examples_part_11_20.json"]:
        filepath = base_path / part_file
        if filepath.exists():
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
                all_examples.extend(data.get("examples", []))
    
    # Index by surah:ayah -> list of gold examples (to handle multiple spans per verse)
    by_verse = {}
    by_gold_id = {}
    
    for ex in all_examples:
        gold_id = ex.get("id", "UNKNOWN")
        ref = ex.get("reference", {})
        key = "{}:{}".format(ref.get("surah"), ref.get("ayah"))
        
        by_gold_id[gold_id] = ex
        
        if key not in by_verse:
            by_verse[key] = []
        by_verse[key].append(ex)
    
    return by_verse, by_gold_id, all_examples


def load_pilot_annotations():
    """Load pilot QBM records."""
    filepath = Path(__file__).parent.parent.parent / "data" / "exports" / "pilot_50_qbm_records.json"
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


def safe_print(text):
    """Print text safely on Windows (ASCII fallback)."""
    try:
        print(text)
    except UnicodeEncodeError:
        # Replace non-ASCII with ?
        print(text.encode('ascii', 'replace').decode('ascii'))


def aggregate_gold_spans(gold_list):
    """Aggregate multiple gold spans into verse-level expected values."""
    if len(gold_list) == 1:
        return gold_list[0]
    
    # For multiple spans, aggregate:
    # - behavior_form: 'mixed' if different forms
    # - evaluation: 'mixed' if different, or most positive
    # - Other fields: use first span (should be consistent)
    
    forms = set(g.get("behavior", {}).get("form") for g in gold_list)
    evals = set(g.get("normative_textual", {}).get("evaluation") for g in gold_list)
    
    aggregated = dict(gold_list[0])  # Start with first
    aggregated["_aggregated"] = True
    aggregated["_source_ids"] = [g.get("id") for g in gold_list]
    
    # Aggregate behavior form
    if len(forms) > 1:
        if "behavior" not in aggregated:
            aggregated["behavior"] = {}
        aggregated["behavior"]["form"] = "mixed"
        aggregated["_form_note"] = "aggregated from: {}".format(forms)
    
    # Aggregate evaluation - if mixed, use 'mixed' or dominant positive
    if len(evals) > 1:
        if "normative_textual" not in aggregated:
            aggregated["normative_textual"] = {}
        # If contains praise, use praise; otherwise mixed
        if "praise" in evals:
            aggregated["normative_textual"]["evaluation"] = "praise"
        else:
            aggregated["normative_textual"]["evaluation"] = "mixed"
        aggregated["_eval_note"] = "aggregated from: {}".format(evals)
    
    return aggregated


def compare_annotation(pilot, gold, is_aggregated=False):
    """Compare a pilot annotation to gold standard."""
    results = {
        "reference": "{}:{}".format(pilot["reference"]["surah"], pilot["reference"]["ayah"]),
        "gold_id": gold.get("id", "UNKNOWN"),
        "aggregated": is_aggregated,
        "matches": [],
        "mismatches": [],
        "score": 0.0
    }
    
    total_checks = 0
    correct = 0
    
    # Compare agent type
    pilot_agent = pilot.get("agent", {}).get("type")
    gold_agent = gold.get("agent", {}).get("type")
    total_checks += 1
    if pilot_agent == gold_agent:
        results["matches"].append("agent_type: {}".format(pilot_agent))
        correct += 1
    else:
        results["mismatches"].append("agent_type: pilot={}, gold={}".format(pilot_agent, gold_agent))
    
    # Compare behavior form
    pilot_form = pilot.get("behavior_form")
    gold_form = gold.get("behavior", {}).get("form")
    total_checks += 1
    if pilot_form == gold_form:
        results["matches"].append("behavior_form: {}".format(pilot_form))
        correct += 1
    else:
        results["mismatches"].append("behavior_form: pilot={}, gold={}".format(pilot_form, gold_form))
    
    # Compare normative fields
    for field in ["speech_mode", "evaluation"]:
        pilot_val = pilot.get("normative", {}).get(field)
        gold_val = gold.get("normative_textual", {}).get(field)
        total_checks += 1
        if pilot_val == gold_val:
            results["matches"].append("{}: {}".format(field, pilot_val))
            correct += 1
        else:
            results["mismatches"].append("{}: pilot={}, gold={}".format(field, pilot_val, gold_val))
    
    # Compare deontic signal
    pilot_deontic = pilot.get("normative", {}).get("deontic_signal")
    gold_deontic = gold.get("normative_textual", {}).get("quran_deontic_signal")
    total_checks += 1
    if pilot_deontic == gold_deontic:
        results["matches"].append("deontic_signal: {}".format(pilot_deontic))
        correct += 1
    else:
        results["mismatches"].append("deontic_signal: pilot={}, gold={}".format(pilot_deontic, gold_deontic))
    
    results["score"] = correct / total_checks if total_checks > 0 else 0.0
    results["total_checks"] = total_checks
    results["correct"] = correct
    
    return results


def main():
    safe_print("Loading gold standard examples...")
    gold_by_verse, gold_by_id, all_gold = load_gold_examples()
    safe_print("Loaded {} gold examples across {} unique verses".format(len(all_gold), len(gold_by_verse)))
    
    # Report verses with multiple spans
    multi_span_verses = {k: v for k, v in gold_by_verse.items() if len(v) > 1}
    if multi_span_verses:
        safe_print("\nWARNING: {} verses have multiple gold spans:".format(len(multi_span_verses)))
        for verse, spans in multi_span_verses.items():
            ids = [s.get("id") for s in spans]
            safe_print("  {} -> {}".format(verse, ids))
    
    safe_print("\nLoading pilot annotations...")
    pilot_records = load_pilot_annotations()
    safe_print("Loaded {} pilot records".format(len(pilot_records)))
    
    # Build pilot index
    pilot_by_verse = {}
    for rec in pilot_records:
        ref = rec.get("reference", {})
        key = "{}:{}".format(ref.get("surah"), ref.get("ayah"))
        pilot_by_verse[key] = rec
    
    # Find overlapping and missing
    gold_verses = set(gold_by_verse.keys())
    pilot_verses = set(pilot_by_verse.keys())
    
    overlapping = gold_verses & pilot_verses
    gold_only = gold_verses - pilot_verses
    pilot_only = pilot_verses - gold_verses
    
    safe_print("\n" + "="*60)
    safe_print("COVERAGE REPORT")
    safe_print("="*60)
    safe_print("Gold verses: {}".format(len(gold_verses)))
    safe_print("Pilot verses: {}".format(len(pilot_verses)))
    safe_print("Overlapping: {}".format(len(overlapping)))
    safe_print("Gold-only (not in pilot): {}".format(len(gold_only)))
    if gold_only:
        safe_print("  Missing from pilot: {}".format(sorted(gold_only)))
    
    # Compare overlapping verses
    safe_print("\n" + "="*60)
    safe_print("COMPARISON RESULTS")
    safe_print("="*60)
    safe_print("\nNOTE: Pilot annotations are verse-level (full ayah).")
    safe_print("      Gold examples may have multiple spans per verse.")
    safe_print("      For multi-span verses, gold spans are AGGREGATED for comparison.\n")
    
    comparisons = []
    for verse in sorted(overlapping):
        pilot = pilot_by_verse[verse]
        gold_list = gold_by_verse[verse]
        
        # Aggregate gold spans for verse-level comparison
        if len(gold_list) > 1:
            gold = aggregate_gold_spans(gold_list)
            is_aggregated = True
        else:
            gold = gold_list[0]
            is_aggregated = False
        
        comparison = compare_annotation(pilot, gold, is_aggregated)
        comparison["multi_span_warning"] = len(gold_list) > 1
        comparison["all_gold_ids"] = [g.get("id") for g in gold_list]
        comparisons.append(comparison)
        
        safe_print("--- {} (vs {}) ---".format(verse, gold.get("id")))
        if len(gold_list) > 1:
            safe_print("  [!] WARNING: {} gold spans for this verse: {}".format(
                len(gold_list), comparison["all_gold_ids"]))
        safe_print("Score: {:.1%} ({}/{})".format(
            comparison["score"], comparison["correct"], comparison["total_checks"]))
        
        if comparison["matches"]:
            safe_print("[OK] Matches:")
            for m in comparison["matches"]:
                safe_print("    {}".format(m))
        
        if comparison["mismatches"]:
            safe_print("[X] Mismatches:")
            for m in comparison["mismatches"]:
                safe_print("    {}".format(m))
        safe_print("")
    
    # Summary
    safe_print("="*60)
    safe_print("SUMMARY")
    safe_print("="*60)
    
    if comparisons:
        avg_score = sum(c["score"] for c in comparisons) / len(comparisons)
        total_correct = sum(c["correct"] for c in comparisons)
        total_checks = sum(c["total_checks"] for c in comparisons)
        multi_span_count = sum(1 for c in comparisons if c.get("multi_span_warning"))
        
        safe_print("Overlapping verses compared: {}".format(len(comparisons)))
        safe_print("Average score: {:.1%}".format(avg_score))
        safe_print("Total correct: {}/{}".format(total_correct, total_checks))
        safe_print("Verses with multi-span gold (comparison may be incomplete): {}".format(multi_span_count))
        
        # Field-level breakdown
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
        
        safe_print("\nField-level accuracy:")
        for field, stats in sorted(field_stats.items()):
            acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            safe_print("  {}: {:.1%} ({}/{})".format(field, acc, stats["correct"], stats["total"]))
        
        # Reliability warning
        if multi_span_count > 0:
            safe_print("\n[!] RELIABILITY NOTE:")
            safe_print("    {} of {} compared verses have multiple gold spans.".format(
                multi_span_count, len(comparisons)))
            safe_print("    Pilot annotations are verse-level, gold examples are span-level.")
            safe_print("    For accurate comparison, either:")
            safe_print("    1. Create span-level pilot annotations, or")
            safe_print("    2. Aggregate gold spans to verse-level before comparison.")
    else:
        safe_print("No overlapping examples found.")


if __name__ == "__main__":
    main()
