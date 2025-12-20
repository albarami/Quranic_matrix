#!/usr/bin/env python3
"""
Analyze behavior_form disagreements to find patterns.
Goal: Identify if disagreements are due to unclear rules vs genuine ambiguity.
"""

import json
from pathlib import Path
from collections import defaultdict

# Load both annotator files
ann1_path = Path("data/exports/pilot_100_annotator1_qbm_records.json")
ann2_path = Path("data/exports/pilot_100_annotator2_qbm_records.json")

with open(ann1_path, encoding="utf-8") as f:
    ann1_data = {r["span_id"]: r for r in json.load(f)}

with open(ann2_path, encoding="utf-8") as f:
    ann2_data = {r["span_id"]: r for r in json.load(f)}

# Find disagreements on behavior_form
disagreements = []
agreement_matrix = defaultdict(lambda: defaultdict(int))

for span_id in ann1_data:
    if span_id in ann2_data:
        bf1 = ann1_data[span_id].get("behavior_form")
        bf2 = ann2_data[span_id].get("behavior_form")
        
        agreement_matrix[bf1][bf2] += 1
        
        if bf1 != bf2:
            ref = ann1_data[span_id].get("reference", {})
            disagreements.append({
                "span_id": span_id,
                "reference": f"{ref.get('surah')}:{ref.get('ayah')}",
                "ann1": bf1,
                "ann2": bf2
            })

print("=" * 60)
print("BEHAVIOR_FORM DISAGREEMENT ANALYSIS")
print("=" * 60)

print(f"\nTotal spans: {len(ann1_data)}")
print(f"Disagreements: {len(disagreements)} ({len(disagreements)/len(ann1_data)*100:.1f}%)")

# Print confusion matrix
print("\n--- Confusion Matrix ---")
all_forms = sorted(set(list(agreement_matrix.keys()) + [k for v in agreement_matrix.values() for k in v.keys()]))
print(f"{'Ann1 / Ann2':<15}", end="")
for f in all_forms:
    print(f"{f[:10]:<12}", end="")
print()

for f1 in all_forms:
    print(f"{f1[:14]:<15}", end="")
    for f2 in all_forms:
        count = agreement_matrix[f1][f2]
        print(f"{count:<12}", end="")
    print()

# Categorize disagreements by type
print("\n--- Disagreement Patterns ---")
patterns = defaultdict(list)
for d in disagreements:
    key = f"{d['ann1']} vs {d['ann2']}"
    # Normalize order
    if d['ann2'] < d['ann1']:
        key = f"{d['ann2']} vs {d['ann1']}"
    patterns[key].append(d)

for pattern, cases in sorted(patterns.items(), key=lambda x: -len(x[1])):
    print(f"\n{pattern}: {len(cases)} cases")
    for case in cases[:3]:  # Show first 3 examples
        print(f"  - {case['span_id']} ({case['reference']})")

# Identify fixable vs genuine ambiguity
print("\n" + "=" * 60)
print("ANALYSIS: FIXABLE vs GENUINE AMBIGUITY")
print("=" * 60)

fixable_patterns = {
    "mixed vs relational_act": "Can clarify: use 'mixed' only when 2+ distinct behavior types present",
    "mixed vs physical_act": "Can clarify: physical acts with relational targets still = physical_act",
    "physical_act vs relational_act": "Harder - depends on focus (action vs relationship)",
}

for pattern, cases in sorted(patterns.items(), key=lambda x: -len(x[1])):
    suggestion = fixable_patterns.get(pattern, "Genuine ambiguity - may need adjudication")
    print(f"\n{pattern} ({len(cases)} cases):")
    print(f"  Suggestion: {suggestion}")
