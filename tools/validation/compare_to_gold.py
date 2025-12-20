"""
Compare pilot annotations against gold standard examples.

This script compares verse-level pilot annotations against span-level
gold examples. When a verse has multiple gold spans, the gold spans are
aggregated to verse-level for comparison.

Fixes in this version:
1. Preserve multiple gold spans per verse (no overwrites).
2. Aggregate multi-span gold verses for verse-level comparison.
3. ASCII-safe output for Windows compatibility.
4. Coverage reporting for missing gold examples.
"""

import json
from collections import Counter
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

    # Index by surah:ayah -> list of gold examples (multi-span friendly)
    by_verse = {}
    by_gold_id = {}
    for ex in all_examples:
        gold_id = ex.get("id", "UNKNOWN")
        ref = ex.get("reference", {})
        key = "{}:{}".format(ref.get("surah"), ref.get("ayah"))

        by_gold_id[gold_id] = ex
        by_verse.setdefault(key, []).append(ex)

    return by_verse, by_gold_id


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
        print(text.encode("ascii", "replace").decode("ascii"))

def sort_ref_key(ref):
    """Sort references by numeric surah/ayah when possible."""
    try:
        surah, ayah = ref.split(":")
        return int(surah), int(ayah)
    except ValueError:
        return 9999, 9999


def aggregate_values(values, field, mixed_value=None, ignore_values=None):
    """Aggregate a list of values into a single representative value."""
    cleaned = [v for v in values if v]
    counts = Counter(cleaned)

    note = {
        "field": field,
        "values": sorted(set(cleaned)),
        "ignored": [],
        "mode": "missing",
        "chosen": None,
    }

    if ignore_values and len(counts) > 1:
        for val in list(counts.keys()):
            if val in ignore_values:
                note["ignored"].append(val)
                counts.pop(val, None)

    if not counts:
        return None, note

    if len(counts) == 1:
        chosen = next(iter(counts))
        note["chosen"] = chosen
        note["mode"] = "single" if not note["ignored"] else "single_after_ignore"
        return chosen, note

    if mixed_value:
        note["chosen"] = mixed_value
        note["mode"] = "mixed"
        return mixed_value, note

    chosen = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    note["chosen"] = chosen
    note["mode"] = "majority"
    return chosen, note


def aggregate_gold_spans(gold_list):
    """Aggregate multiple gold spans into verse-level expected values."""
    if not gold_list:
        return {}, {"source_ids": [], "notes": [], "is_aggregated": False}

    agent_types = [g.get("agent", {}).get("type") for g in gold_list]
    behavior_forms = [g.get("behavior", {}).get("form") for g in gold_list]
    speech_modes = [g.get("normative_textual", {}).get("speech_mode") for g in gold_list]
    evaluations = [g.get("normative_textual", {}).get("evaluation") for g in gold_list]
    deontic_signals = [g.get("normative_textual", {}).get("quran_deontic_signal") for g in gold_list]

    agent_type, agent_note = aggregate_values(
        agent_types, "agent_type", ignore_values={"AGT_UNKNOWN"}
    )
    behavior_form, behavior_note = aggregate_values(
        behavior_forms, "behavior_form", mixed_value="mixed"
    )
    speech_mode, speech_note = aggregate_values(
        speech_modes, "speech_mode", ignore_values={"unknown"}
    )
    evaluation, evaluation_note = aggregate_values(
        evaluations, "evaluation", mixed_value="mixed", ignore_values={"neutral"}
    )
    deontic_signal, deontic_note = aggregate_values(
        deontic_signals, "deontic_signal"
    )

    aggregated = {
        "id": "AGGREGATED",
        "reference": gold_list[0].get("reference", {}),
        "agent": {"type": agent_type},
        "behavior": {"form": behavior_form},
        "normative_textual": {
            "speech_mode": speech_mode,
            "evaluation": evaluation,
            "quran_deontic_signal": deontic_signal,
        },
    }

    notes = []
    for note in [agent_note, behavior_note, speech_note, evaluation_note, deontic_note]:
        if note["mode"] != "single" or note["ignored"]:
            notes.append(note)

    meta = {
        "source_ids": [g.get("id") for g in gold_list],
        "notes": notes,
        "is_aggregated": len(gold_list) > 1,
    }
    return aggregated, meta


def format_aggregation_notes(notes):
    """Format aggregation notes for display."""
    lines = []
    for note in notes:
        values = ", ".join(note["values"]) if note["values"] else "none"
        line = "{}: {{{}}} -> {}".format(note["field"], values, note["chosen"])
        if note["ignored"]:
            line += " (ignored: {})".format(", ".join(note["ignored"]))
        if note["mode"] == "majority":
            line += " [majority]"
        elif note["mode"] == "single_after_ignore":
            line += " [ignored]"
        elif note["mode"] == "missing":
            line += " [missing]"
        lines.append(line)
    return lines


def compare_annotation(pilot, gold):
    """Compare a pilot annotation to gold standard."""
    results = {
        "reference": "{}:{}".format(pilot["reference"]["surah"], pilot["reference"]["ayah"]),
        "matches": [],
        "mismatches": [],
        "skipped": [],
        "score": 0.0
    }
    
    total_checks = 0
    correct = 0

    def compare_field(field, pilot_val, gold_val):
        nonlocal total_checks, correct
        if gold_val is None:
            results["skipped"].append(field)
            return
        total_checks += 1
        if pilot_val == gold_val:
            results["matches"].append("{}: {}".format(field, pilot_val))
            correct += 1
        else:
            results["mismatches"].append("{}: pilot={}, gold={}".format(field, pilot_val, gold_val))

    compare_field("agent_type", pilot.get("agent", {}).get("type"), gold.get("agent", {}).get("type"))
    compare_field("behavior_form", pilot.get("behavior_form"), gold.get("behavior", {}).get("form"))
    compare_field("speech_mode", pilot.get("normative", {}).get("speech_mode"), gold.get("normative_textual", {}).get("speech_mode"))
    compare_field("evaluation", pilot.get("normative", {}).get("evaluation"), gold.get("normative_textual", {}).get("evaluation"))
    compare_field(
        "deontic_signal",
        pilot.get("normative", {}).get("deontic_signal"),
        gold.get("normative_textual", {}).get("quran_deontic_signal"),
    )
    
    results["score"] = correct / total_checks if total_checks > 0 else 0.0
    results["total_checks"] = total_checks
    results["correct"] = correct
    
    return results


def main():
    safe_print("Loading gold standard examples...")
    gold_by_verse, gold_by_id = load_gold_examples()
    safe_print("Loaded {} gold examples across {} unique verses".format(len(gold_by_id), len(gold_by_verse)))
    
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
        missing = ", ".join(sorted(gold_only, key=sort_ref_key))
        safe_print("  Missing from pilot: {}".format(missing))
    safe_print("Pilot-only (not in gold): {}".format(len(pilot_only)))

    multi_span_count = sum(1 for verse in overlapping if len(gold_by_verse[verse]) > 1)
    
    # Compare overlapping verses
    safe_print("\n" + "="*60)
    safe_print("COMPARISON RESULTS")
    safe_print("="*60)
    if overlapping:
        safe_print("{} of {} compared verses have multiple gold spans.".format(
            multi_span_count, len(overlapping)
        ))
    safe_print("Pilot annotations are verse-level; gold examples are span-level.")
    safe_print("Gold spans are aggregated to verse-level for comparison.\n")
    
    comparisons = []
    for verse in sorted(overlapping, key=sort_ref_key):
        pilot = pilot_by_verse[verse]
        gold_list = gold_by_verse[verse]
        
        gold, meta = aggregate_gold_spans(gold_list)
        comparison = compare_annotation(pilot, gold)
        comparison["multi_span_warning"] = meta["is_aggregated"]
        comparison["all_gold_ids"] = meta["source_ids"]
        comparison["aggregation_notes"] = meta["notes"]
        comparisons.append(comparison)
        
        safe_print("--- {} (gold ids: {}) ---".format(
            verse, ", ".join(comparison["all_gold_ids"])
        ))
        if comparison["aggregation_notes"]:
            safe_print("  Aggregated fields:")
            for note_line in format_aggregation_notes(comparison["aggregation_notes"]):
                safe_print("    {}".format(note_line))
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
        if comparison["skipped"]:
            safe_print("Skipped fields: {}".format(", ".join(comparison["skipped"])))
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
        total_skipped = sum(len(c.get("skipped", [])) for c in comparisons)
        
        safe_print("Overlapping verses compared: {}".format(len(comparisons)))
        safe_print("Average score: {:.1%}".format(avg_score))
        safe_print("Total correct: {}/{}".format(total_correct, total_checks))
        safe_print("Verses with multi-span gold (comparison may be incomplete): {}".format(multi_span_count))
        safe_print("Skipped checks (missing gold fields): {}".format(total_skipped))
        
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
