#!/usr/bin/env python3
"""
Export Phase 3 annotations from Label Studio and split by annotator.
"""

import json
import os
from label_studio_sdk import LabelStudio

LABEL_STUDIO_URL = "http://localhost:8080"
API_KEY = "14a907022934932368e2a5cf6f697b1f772dad8a"

client = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=API_KEY)

# Get project
projects = list(client.projects.list())
project_id = projects[0].id if projects else None
print(f"Using project ID: {project_id}")

# Get all tasks
tasks = list(client.tasks.list(project=project_id))
print(f"Found {len(tasks)} tasks")

def extract_choice(result_list, from_name):
    """Extract choice value from annotation result."""
    for item in result_list:
        if item.get("from_name") == from_name:
            choices = item.get("value", {}).get("choices", [])
            if choices:
                return choices[0] if len(choices) == 1 else choices
    return None

def convert_to_qbm(task_data, annotation_result):
    """Convert Label Studio annotation to QBM span record format."""
    task_id = task_data.get('id', 'UNKNOWN')
    if str(task_id).startswith('QBM_'):
        span_id = task_id
    else:
        span_id = f"QBM_{task_id:05d}" if isinstance(task_id, int) else f"QBM_{task_id}"
    
    # Derive textual_eval from evaluation if not present
    evaluation = extract_choice(annotation_result, "evaluation")
    textual_eval = extract_choice(annotation_result, "action_textual_eval")
    if not textual_eval:
        if evaluation == "praise":
            textual_eval = "EVAL_SALIH"
        elif evaluation == "blame":
            textual_eval = "EVAL_SAYYI"
        else:
            textual_eval = "EVAL_NEUTRAL"
    
    situational = extract_choice(annotation_result, "situational")
    
    return {
        "span_id": span_id,
        "reference": {
            "surah": task_data.get("surah"),
            "ayah": task_data.get("ayah"),
            "surah_name": task_data.get("surah_name"),
        },
        "behavior_form": extract_choice(annotation_result, "behavior_form"),
        "agent": {
            "type": extract_choice(annotation_result, "agent_type"),
            "explicit": None
        },
        "normative": {
            "speech_mode": extract_choice(annotation_result, "speech_mode"),
            "evaluation": evaluation,
            "deontic_signal": extract_choice(annotation_result, "quran_deontic_signal"),
        },
        "action": {
            "class": extract_choice(annotation_result, "action_class") or "ACT_VOLITIONAL",
            "textual_eval": textual_eval,
        },
        "axes": {
            "situational": situational,
            "systemic": extract_choice(annotation_result, "systemic"),
        },
        "evidence": {
            "support_type": extract_choice(annotation_result, "support_type") or "direct",
        },
    }

# Filter Phase 3 tasks and extract annotations
annotator1_records = []
annotator2_records = []

phase3_count = 0
for task in tasks:
    task_id_str = task.data.get('id', '')
    if not task_id_str.startswith('QBM_'):
        continue
    
    num = int(task_id_str.split('_')[1])
    if num < 101:  # Skip Phase 2 tasks
        continue
    
    phase3_count += 1
    
    if not task.annotations or len(task.annotations) == 0:
        continue
    
    annotations = task.annotations
    
    # First annotation -> Annotator 1
    if len(annotations) >= 1:
        ann = annotations[0]
        result = ann.get("result") if isinstance(ann, dict) else ann.result
        if result:
            record = convert_to_qbm(task.data, result)
            if record:
                annotator1_records.append(record)
    
    # Second annotation -> Annotator 2
    if len(annotations) >= 2:
        ann = annotations[1]
        result = ann.get("result") if isinstance(ann, dict) else ann.result
        if result:
            record = convert_to_qbm(task.data, result)
            if record:
                annotator2_records.append(record)

print(f"\nPhase 3 tasks: {phase3_count}")
print(f"Annotator 1 records: {len(annotator1_records)}")
print(f"Annotator 2 records: {len(annotator2_records)}")

# Save exports
output_dir = "data/exports"
os.makedirs(output_dir, exist_ok=True)

ann1_path = os.path.join(output_dir, "phase3_annotator1_qbm_records.json")
ann2_path = os.path.join(output_dir, "phase3_annotator2_qbm_records.json")

with open(ann1_path, "w", encoding="utf-8") as f:
    json.dump(annotator1_records, f, ensure_ascii=False, indent=2)
print(f"Saved: {ann1_path}")

with open(ann2_path, "w", encoding="utf-8") as f:
    json.dump(annotator2_records, f, ensure_ascii=False, indent=2)
print(f"Saved: {ann2_path}")
