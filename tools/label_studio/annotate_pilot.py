"""
Programmatically annotate pilot tasks using Label Studio SDK.

This script creates annotations for the pilot 50 tasks based on
the gold standard examples and QBM methodology.
"""

import json
from label_studio_sdk import LabelStudio

LABEL_STUDIO_URL = "http://localhost:8080"
API_KEY = "14a907022934932368e2a5cf6f697b1f772dad8a"

client = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=API_KEY)

# Verify connection
me = client.users.whoami()
print(f"Connected as: {me.email}")

# List projects
projects = client.projects.list()
for p in projects:
    print(f"Project {p.id}: {p.title}")

# Get the QBM Pilot project
project_id = None
for p in projects:
    if "QBM" in p.title or "Pilot" in p.title:
        project_id = p.id
        print(f"\nUsing project: {p.title} (ID: {p.id})")
        break

if not project_id:
    print("No QBM project found!")
    exit(1)

# Get tasks
tasks = client.tasks.list(project=project_id)
task_list = list(tasks)
print(f"\nFound {len(task_list)} tasks")

# Show first task structure
if task_list:
    first_task = task_list[0]
    print(f"\nFirst task ID: {first_task.id}")
    print(f"First task data keys: {first_task.data.keys() if first_task.data else 'None'}")
    if first_task.data:
        print(f"Reference: {first_task.data.get('reference', 'N/A')}")
        print(f"Raw text: {first_task.data.get('raw_text_ar', 'N/A')[:50]}...")


def create_annotation_for_task(task_id: int, task_data: dict, annotation_params: dict):
    """
    Create an annotation for a task.
    
    annotation_params should contain:
    - span_start: int (character offset)
    - span_end: int (character offset)
    - behavior_form: str
    - agent_type: str
    - speech_mode: str
    - evaluation: str
    - deontic_signal: str
    - support_type: str
    - action_class: str
    - action_eval: str
    - systemic: list
    - situational: str
    """
    raw_text = task_data.get('raw_text_ar', '')
    span_start = annotation_params.get('span_start', 0)
    span_end = annotation_params.get('span_end', len(raw_text))
    
    result = [
        {
            "from_name": "span_selection",
            "to_name": "quran_text",
            "type": "labels",
            "value": {
                "start": span_start,
                "end": span_end,
                "text": raw_text[span_start:span_end],
                "labels": ["BEHAVIOR_SPAN"]
            }
        },
        {
            "from_name": "behavior_form",
            "to_name": "quran_text",
            "type": "choices",
            "value": {"choices": [annotation_params.get('behavior_form', 'inner_state')]}
        },
        {
            "from_name": "agent_type",
            "to_name": "quran_text",
            "type": "choices",
            "value": {"choices": [annotation_params.get('agent_type', 'AGT_BELIEVER')]}
        },
        {
            "from_name": "speech_mode",
            "to_name": "quran_text",
            "type": "choices",
            "value": {"choices": [annotation_params.get('speech_mode', 'informative')]}
        },
        {
            "from_name": "evaluation",
            "to_name": "quran_text",
            "type": "choices",
            "value": {"choices": [annotation_params.get('evaluation', 'praise')]}
        },
        {
            "from_name": "quran_deontic_signal",
            "to_name": "quran_text",
            "type": "choices",
            "value": {"choices": [annotation_params.get('deontic_signal', 'khabar')]}
        },
        {
            "from_name": "support_type",
            "to_name": "quran_text",
            "type": "choices",
            "value": {"choices": [annotation_params.get('support_type', 'direct')]}
        },
        {
            "from_name": "action_class",
            "to_name": "quran_text",
            "type": "choices",
            "value": {"choices": [annotation_params.get('action_class', 'ACT_VOLITIONAL')]}
        },
        {
            "from_name": "action_textual_eval",
            "to_name": "quran_text",
            "type": "choices",
            "value": {"choices": [annotation_params.get('action_eval', 'EVAL_SALIH')]}
        },
        {
            "from_name": "systemic",
            "to_name": "quran_text",
            "type": "choices",
            "value": {"choices": annotation_params.get('systemic', ['SYS_GOD'])}
        },
        {
            "from_name": "situational",
            "to_name": "quran_text",
            "type": "choices",
            "value": {"choices": [annotation_params.get('situational', 'internal')]}
        },
    ]
    
    # Create annotation via API
    annotation = client.annotations.create(
        id=task_id,
        result=result
    )
    
    return annotation


# Define annotations for the first few pilot tasks based on QBM methodology
# These are based on the ayat content and Islamic scholarship

PILOT_ANNOTATIONS = {
    "2:3": {
        "span_start": 0,
        "span_end": 93,  # Full ayah
        "behavior_form": "inner_state",
        "agent_type": "AGT_BELIEVER",
        "speech_mode": "informative",
        "evaluation": "praise",
        "deontic_signal": "khabar",
        "support_type": "direct",
        "action_class": "ACT_VOLITIONAL",
        "action_eval": "EVAL_SALIH",
        "systemic": ["SYS_GOD"],
        "situational": "internal"
    },
    "2:4": {
        "span_start": 0,
        "span_end": 109,
        "behavior_form": "inner_state",
        "agent_type": "AGT_BELIEVER",
        "speech_mode": "informative",
        "evaluation": "praise",
        "deontic_signal": "khabar",
        "support_type": "direct",
        "action_class": "ACT_VOLITIONAL",
        "action_eval": "EVAL_SALIH",
        "systemic": ["SYS_GOD"],
        "situational": "internal"
    },
    "2:5": {
        "span_start": 0,
        "span_end": 80,
        "behavior_form": "trait_disposition",
        "agent_type": "AGT_BELIEVER",
        "speech_mode": "informative",
        "evaluation": "praise",
        "deontic_signal": "khabar",
        "support_type": "direct",
        "action_class": "ACT_VOLITIONAL",
        "action_eval": "EVAL_SALIH",
        "systemic": ["SYS_GOD"],
        "situational": "internal"
    },
    "2:8": {
        "span_start": 0,
        "span_end": 97,
        "behavior_form": "speech_act",
        "agent_type": "AGT_HYPOCRITE",
        "speech_mode": "informative",
        "evaluation": "blame",
        "deontic_signal": "khabar",
        "support_type": "direct",
        "action_class": "ACT_VOLITIONAL",
        "action_eval": "EVAL_SAYYI",
        "systemic": ["SYS_GOD", "SYS_SOCIETY"],
        "situational": "external"
    },
    "2:9": {
        "span_start": 0,
        "span_end": 100,
        "behavior_form": "relational_act",
        "agent_type": "AGT_HYPOCRITE",
        "speech_mode": "informative",
        "evaluation": "blame",
        "deontic_signal": "khabar",
        "support_type": "direct",
        "action_class": "ACT_VOLITIONAL",
        "action_eval": "EVAL_SAYYI",
        "systemic": ["SYS_GOD", "SYS_SELF"],
        "situational": "external"
    },
    "2:10": {
        "span_start": 0,
        "span_end": 105,
        "behavior_form": "inner_state",
        "agent_type": "AGT_HYPOCRITE",
        "speech_mode": "informative",
        "evaluation": "blame",
        "deontic_signal": "tarhib",
        "support_type": "direct",
        "action_class": "ACT_VOLITIONAL",
        "action_eval": "EVAL_SAYYI",
        "systemic": ["SYS_SELF"],
        "situational": "internal"
    },
}

# Annotate tasks
print("\n--- Creating Annotations ---")
annotated_count = 0

for task in task_list:
    reference = task.data.get('reference', '')
    if reference in PILOT_ANNOTATIONS:
        print(f"\nAnnotating task {task.id} ({reference})...")
        try:
            annotation = create_annotation_for_task(
                task_id=task.id,
                task_data=task.data,
                annotation_params=PILOT_ANNOTATIONS[reference]
            )
            print(f"  Created annotation ID: {annotation.id}")
            annotated_count += 1
        except Exception as e:
            print(f"  Error: {e}")

print(f"\n--- Done: {annotated_count} annotations created ---")
