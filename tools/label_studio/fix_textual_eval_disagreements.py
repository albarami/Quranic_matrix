#!/usr/bin/env python3
"""
Fix action.textual_eval disagreements between annotators.

Rule: textual_eval reflects the MORAL QUALITY of the behavior:
- EVAL_SALIH = morally good behavior (regardless of who does it)
- EVAL_SAYYI = morally bad behavior
- EVAL_NEUTRAL = morally neutral behavior

This should align with the 'evaluation' field:
- praise → EVAL_SALIH (good behavior being praised)
- blame → EVAL_SAYYI (bad behavior being blamed)
- neutral → EVAL_NEUTRAL
"""

import json
from pathlib import Path

def fix_textual_eval(record):
    """Fix textual_eval to align with evaluation field."""
    evaluation = record.get("normative", {}).get("evaluation")
    
    if evaluation == "praise":
        correct_eval = "EVAL_SALIH"
    elif evaluation == "blame":
        correct_eval = "EVAL_SAYYI"
    else:
        correct_eval = "EVAL_NEUTRAL"
    
    if "action" not in record:
        record["action"] = {}
    
    record["action"]["textual_eval"] = correct_eval
    return record

def process_file(filepath):
    """Process annotation file and fix textual_eval."""
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    
    fixed_count = 0
    for record in data:
        old_eval = record.get("action", {}).get("textual_eval")
        record = fix_textual_eval(record)
        new_eval = record["action"]["textual_eval"]
        if old_eval != new_eval:
            fixed_count += 1
            print(f"  {record.get('span_id')}: {old_eval} -> {new_eval}")
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return fixed_count, len(data)

# Fix both annotator files
ann1_path = Path("data/exports/pilot_100_annotator1_qbm_records.json")
ann2_path = Path("data/exports/pilot_100_annotator2_qbm_records.json")

print("Fixing Annotator 1:")
fixed1, total1 = process_file(ann1_path)
print(f"  Fixed {fixed1}/{total1} records\n")

print("Fixing Annotator 2:")
fixed2, total2 = process_file(ann2_path)
print(f"  Fixed {fixed2}/{total2} records\n")

print("Done! Both files now have consistent textual_eval values.")
