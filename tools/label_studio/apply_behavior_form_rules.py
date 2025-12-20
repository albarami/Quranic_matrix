#!/usr/bin/env python3
"""
Apply behavior_form decision rules to fix LEGITIMATE disagreements.
Only fixes cases where one annotator clearly followed the rule incorrectly.
Does NOT force agreement on genuinely ambiguous cases.
"""

import json
from pathlib import Path

# Legitimate fixes based on decision rules
# Format: span_id -> (correct_value, rule_applied, reasoning)
LEGITIMATE_FIXES = {
    # Rule 1: mixed vs relational_act - use mixed only for 2+ distinct types
    "QBM_00053": ("mixed", "Rule 1", "3:159 has relational + speech + inner (gentle, consult, trust)"),
    "QBM_00070": ("mixed", "Rule 1", "3:164 has multiple distinct behaviors (recite, purify, teach)"),
    "QBM_00079": ("relational_act", "Rule 1", "60:8 is single relational instruction (deal justly)"),
    "QBM_00086": ("relational_act", "Rule 1", "2:283 is about trust/fulfilling obligations"),
    
    # Rule 2: inner_state vs trait_disposition
    "QBM_00003": ("trait_disposition", "Rule 2", "2:5 'upon guidance' = stable established state"),
    "QBM_00019": ("trait_disposition", "Rule 2", "4:145 hypocrites in lowest depths = their nature"),
    
    # Rule 3: physical_act vs relational_act - focus on action vs relationship
    "QBM_00063": ("physical_act", "Rule 3", "2:191 qital is physical action"),
    "QBM_00080": ("physical_act", "Rule 3", "2:261 infaq is physical giving of wealth"),
    "QBM_00081": ("physical_act", "Rule 3", "9:34 taking wealth is physical action"),
    
    # Rule 4: speech_act vs relational_act
    "QBM_00017": ("speech_act", "Rule 4", "4:135 testimony is speech act for justice"),
    "QBM_00031": ("speech_act", "Rule 4", "49:1 about raising voices = speech behavior"),
    
    # Rule 5: inner_state vs speech_act
    "QBM_00046": ("speech_act", "Rule 5", "63:6 about asking forgiveness = speech act"),
    
    # Clear cases from confusion matrix
    "QBM_00048": ("speech_act", "Rule 4", "63:8 yaqulun = they say, speech act"),
    "QBM_00049": ("inner_state", "Rule 2", "63:9 about hearts being distracted = inner state"),
}

# Cases that are GENUINELY AMBIGUOUS - do not force
AMBIGUOUS_CASES = {
    "QBM_00025": "5:54 - trait vs mixed: loving believers could be trait or mixed behaviors",
    "QBM_00028": "5:93 - trait vs mixed: taqwa could be trait or ongoing state",
    "QBM_00033": "49:6 - inner vs relational: verification involves both",
    "QBM_00038": "49:13 - inner vs relational: taqwa + knowing each other",
    "QBM_00040": "49:15 - trait vs mixed: believers' characteristics",
    "QBM_00047": "63:7 - inner vs relational: withholding spending has both aspects",
    "QBM_00052": "2:225 - inner vs mixed: hearts and oaths",
    "QBM_00078": "22:5 - inner vs mixed: creation stages description",
    "QBM_00045": "63:5 - physical vs trait: turning away could be either",
}

def apply_fixes(filepath, annotator_name):
    """Apply legitimate fixes to annotation file."""
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    
    fixed_count = 0
    for record in data:
        span_id = record.get("span_id")
        if span_id in LEGITIMATE_FIXES:
            correct_value, rule, reasoning = LEGITIMATE_FIXES[span_id]
            old_value = record.get("behavior_form")
            if old_value != correct_value:
                record["behavior_form"] = correct_value
                fixed_count += 1
                print(f"  {span_id}: {old_value} -> {correct_value} ({rule})")
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return fixed_count

# Apply to both files
ann1_path = Path("data/exports/pilot_100_annotator1_qbm_records.json")
ann2_path = Path("data/exports/pilot_100_annotator2_qbm_records.json")

print("Applying LEGITIMATE fixes (not forcing ambiguous cases):\n")

print("Annotator 1:")
fixed1 = apply_fixes(ann1_path, "Annotator 1")
print(f"  Total fixed: {fixed1}\n")

print("Annotator 2:")
fixed2 = apply_fixes(ann2_path, "Annotator 2")
print(f"  Total fixed: {fixed2}\n")

print("=" * 60)
print("AMBIGUOUS CASES (left as-is, disagreement is valid):")
print("=" * 60)
for span_id, reason in AMBIGUOUS_CASES.items():
    print(f"  {span_id}: {reason}")

print(f"\nTotal legitimate fixes: {fixed1 + fixed2}")
print(f"Ambiguous cases preserved: {len(AMBIGUOUS_CASES)}")
