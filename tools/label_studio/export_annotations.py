"""
Export annotations from Label Studio and convert to QBM format.
"""

import json
import os
from label_studio_sdk import LabelStudio

LABEL_STUDIO_URL = "http://localhost:8080"
API_KEY = "14a907022934932368e2a5cf6f697b1f772dad8a"

client = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=API_KEY)

# Get project
projects = client.projects.list()
project_id = None
for p in projects:
    if "QBM" in p.title or "Pilot" in p.title:
        project_id = p.id
        print(f"Using project: {p.title} (ID: {p.id})")
        break

# Export annotations
print("\nExporting annotations...")
tasks = client.tasks.list(project=project_id)
task_list = list(tasks)

# Collect all annotated tasks
exported = []
for task in task_list:
    if task.annotations and len(task.annotations) > 0:
        exported.append({
            "id": task.id,
            "data": task.data,
            "annotations": [
                {
                    "id": ann.get("id") if isinstance(ann, dict) else ann.id,
                    "result": ann.get("result") if isinstance(ann, dict) else ann.result,
                    "created_at": str(ann.get("created_at")) if isinstance(ann, dict) else str(ann.created_at) if ann.created_at else None,
                }
                for ann in task.annotations
            ]
        })

print(f"Exported {len(exported)} annotated tasks")

# Save raw export
output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "exports")
os.makedirs(output_dir, exist_ok=True)

raw_export_path = os.path.join(output_dir, "pilot_50_raw_export.json")
with open(raw_export_path, "w", encoding="utf-8") as f:
    json.dump(exported, f, ensure_ascii=False, indent=2)
print(f"Saved raw export to: {raw_export_path}")

# Convert to QBM span record format
print("\nConverting to QBM format...")

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
    # Extract span
    span_info = None
    for item in annotation_result:
        if item.get("from_name") == "span_selection":
            span_info = item.get("value", {})
            break
    
    if not span_info:
        return None
    
    # Handle span_id - avoid double prefix if id already starts with QBM_
    task_id = task_data.get('id', 'UNKNOWN')
    if str(task_id).startswith('QBM_'):
        span_id = task_id
    else:
        span_id = f"QBM_{task_id:05d}" if isinstance(task_id, int) else f"QBM_{task_id}"
    
    qbm_record = {
        "span_id": span_id,
        "reference": {
            "surah": task_data.get("surah"),
            "ayah": task_data.get("ayah"),
            "surah_name": task_data.get("surah_name"),
        },
        "span": {
            "start_token": 0,  # Would need token mapping
            "end_token": task_data.get("token_count", 0) - 1,
            "start_char": span_info.get("start", 0),
            "end_char": span_info.get("end", 0),
            "text": span_info.get("text", ""),
        },
        "behavior_form": extract_choice(annotation_result, "behavior_form"),
        "agent": {
            "type": extract_choice(annotation_result, "agent_type"),
            "explicit": extract_choice(annotation_result, "agent_explicit"),
        },
        "axes": {
            "situational": extract_choice(annotation_result, "situational"),
            "systemic": extract_choice(annotation_result, "systemic"),
            "spatial": extract_choice(annotation_result, "spatial"),
            "temporal": extract_choice(annotation_result, "temporal"),
        },
        "action": {
            "class": extract_choice(annotation_result, "action_class"),
            "textual_eval": extract_choice(annotation_result, "action_textual_eval"),
        },
        "normative": {
            "speech_mode": extract_choice(annotation_result, "speech_mode"),
            "evaluation": extract_choice(annotation_result, "evaluation"),
            "deontic_signal": extract_choice(annotation_result, "quran_deontic_signal"),
        },
        "evidence": {
            "support_type": extract_choice(annotation_result, "support_type"),
            "indication_tags": extract_choice(annotation_result, "indication_tags"),
            "justification_code": extract_choice(annotation_result, "justification_code"),
        },
        "negation": {
            "negated": extract_choice(annotation_result, "negated"),
            "polarity": extract_choice(annotation_result, "polarity"),
        },
        "periodicity": {
            "value": extract_choice(annotation_result, "periodicity"),
            "grammatical_indicator": extract_choice(annotation_result, "grammatical_indicator"),
        },
    }
    
    return qbm_record

qbm_records = []
for task in exported:
    task_data = task["data"]
    for ann in task["annotations"]:
        result = ann.get("result", [])
        qbm_record = convert_to_qbm(task_data, result)
        if qbm_record:
            qbm_records.append(qbm_record)

print(f"Converted {len(qbm_records)} records to QBM format")

# Save QBM format
qbm_export_path = os.path.join(output_dir, "pilot_50_qbm_records.json")
with open(qbm_export_path, "w", encoding="utf-8") as f:
    json.dump(qbm_records, f, ensure_ascii=False, indent=2)
print(f"Saved QBM records to: {qbm_export_path}")

# Print summary
print("\n--- Summary ---")
print(f"Total tasks: {len(task_list)}")
print(f"Annotated tasks: {len(exported)}")
print(f"QBM records: {len(qbm_records)}")

# Show sample record
if qbm_records:
    print("\n--- Sample QBM Record ---")
    print(json.dumps(qbm_records[0], ensure_ascii=False, indent=2))
