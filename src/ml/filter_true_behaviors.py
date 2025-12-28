"""
Filter Annotations to TRUE Behaviors Only

Based on Usul al-Fiqh aligned taxonomy:
- Removes entity mentions (نبي, رسول, ملائكة)
- Removes state labels (مؤمن, كافر, منافق, قلب, سليم, etc.)
- Keeps only TRUE behaviors (actions, traits, dispositions, inner states)

This ensures the ML models learn actual behavioral patterns, not entity mentions.
"""

import json
from pathlib import Path
from collections import Counter
from typing import Dict, List, Set

# Import taxonomy
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.ml.qbm_usul_taxonomy import (
    BEHAVIOR_TAXONOMY,
    get_true_behaviors,
    get_non_behaviors,
    get_behavior_by_arabic,
    get_behavior_id_by_arabic,
    is_true_behavior,
)

# Paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"
ANNOTATIONS_FILE = DATA_DIR / "annotations" / "tafsir_behavioral_annotations.jsonl"
FILTERED_FILE = DATA_DIR / "annotations" / "tafsir_behavioral_annotations_filtered.jsonl"


def analyze_current_annotations():
    """Analyze current annotations and categorize them."""
    print("=" * 70)
    print("ANALYZING CURRENT ANNOTATIONS")
    print("=" * 70)
    
    annotations = []
    with open(ANNOTATIONS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                annotations.append(json.loads(line))
    
    print(f"Total annotations: {len(annotations):,}")
    
    # Count by Arabic label
    label_counts = Counter(ann.get("behavior_ar", "") for ann in annotations)
    
    # Categorize
    true_behavior_labels = set()
    entity_labels = set()
    state_labels = set()
    unknown_labels = set()
    
    true_behavior_count = 0
    entity_count = 0
    state_count = 0
    unknown_count = 0
    
    for arabic, count in label_counts.items():
        concept = get_behavior_by_arabic(arabic)
        if concept:
            if concept.is_true_behavior:
                true_behavior_labels.add(arabic)
                true_behavior_count += count
            elif concept.category.value == "ENT":
                entity_labels.add(arabic)
                entity_count += count
            elif concept.category.value == "STL":
                state_labels.add(arabic)
                state_count += count
        else:
            unknown_labels.add(arabic)
            unknown_count += count
    
    print(f"\nTRUE BEHAVIORS ({len(true_behavior_labels)} labels, {true_behavior_count:,} annotations):")
    for label in sorted(true_behavior_labels):
        beh_id = get_behavior_id_by_arabic(label)
        print(f"  {label}: {label_counts[label]:,} -> {beh_id}")
    
    print(f"\nENTITY MENTIONS - TO REMOVE ({len(entity_labels)} labels, {entity_count:,} annotations):")
    for label in sorted(entity_labels):
        print(f"  {label}: {label_counts[label]:,} (NOT a behavior)")
    
    print(f"\nSTATE LABELS - TO REMOVE ({len(state_labels)} labels, {state_count:,} annotations):")
    for label in sorted(state_labels):
        print(f"  {label}: {label_counts[label]:,} (NOT a behavior)")
    
    if unknown_labels:
        print(f"\nUNKNOWN LABELS ({len(unknown_labels)} labels, {unknown_count:,} annotations):")
        for label in sorted(unknown_labels):
            print(f"  {label}: {label_counts[label]:,} (needs mapping)")
    
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"TRUE behaviors to KEEP: {true_behavior_count:,} ({100*true_behavior_count/len(annotations):.1f}%)")
    print(f"Entities to REMOVE: {entity_count:,} ({100*entity_count/len(annotations):.1f}%)")
    print(f"State labels to REMOVE: {state_count:,} ({100*state_count/len(annotations):.1f}%)")
    print(f"Unknown to REVIEW: {unknown_count:,} ({100*unknown_count/len(annotations):.1f}%)")
    
    return {
        "total": len(annotations),
        "true_behaviors": true_behavior_count,
        "entities": entity_count,
        "states": state_count,
        "unknown": unknown_count,
        "true_behavior_labels": true_behavior_labels,
        "entity_labels": entity_labels,
        "state_labels": state_labels,
        "unknown_labels": unknown_labels,
    }


def filter_annotations():
    """Filter annotations to keep only TRUE behaviors."""
    print("\n" + "=" * 70)
    print("FILTERING ANNOTATIONS")
    print("=" * 70)
    
    kept = []
    removed_entities = 0
    removed_states = 0
    removed_unknown = 0
    
    with open(ANNOTATIONS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                ann = json.loads(line)
                arabic = ann.get("behavior_ar", "")
                
                concept = get_behavior_by_arabic(arabic)
                if concept and concept.is_true_behavior:
                    # Add controlled ID
                    ann["behavior_id"] = get_behavior_id_by_arabic(arabic)
                    ann["behavior_form"] = concept.form.value
                    ann["action_class"] = concept.action_class.value
                    ann["default_eval"] = concept.default_eval.value
                    kept.append(ann)
                elif concept:
                    if concept.category.value == "ENT":
                        removed_entities += 1
                    else:
                        removed_states += 1
                else:
                    removed_unknown += 1
    
    print(f"Kept: {len(kept):,} TRUE behavior annotations")
    print(f"Removed entities: {removed_entities:,}")
    print(f"Removed states: {removed_states:,}")
    print(f"Removed unknown: {removed_unknown:,}")
    
    # Save filtered annotations
    with open(FILTERED_FILE, 'w', encoding='utf-8') as f:
        for ann in kept:
            f.write(json.dumps(ann, ensure_ascii=False) + '\n')
    
    print(f"\nSaved filtered annotations to: {FILTERED_FILE}")
    
    # Print behavior distribution in filtered data
    print("\nFiltered behavior distribution:")
    filtered_counts = Counter(ann.get("behavior_ar", "") for ann in kept)
    for arabic, count in filtered_counts.most_common():
        beh_id = get_behavior_id_by_arabic(arabic)
        print(f"  {arabic}: {count:,} -> {beh_id}")
    
    return kept


def main():
    """Main function."""
    # Analyze
    stats = analyze_current_annotations()
    
    # Filter
    filtered = filter_annotations()
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Original: {stats['total']:,} annotations")
    print(f"Filtered: {len(filtered):,} TRUE behavior annotations")
    print(f"Removed: {stats['total'] - len(filtered):,} non-behavior annotations")
    print(f"\nFiltered file: {FILTERED_FILE}")


if __name__ == "__main__":
    main()
