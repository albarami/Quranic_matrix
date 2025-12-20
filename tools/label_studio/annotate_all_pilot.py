"""
Programmatically annotate ALL 50 pilot tasks using Label Studio SDK.

Annotations based on QBM methodology and Quranic scholarship.
"""

import json
from label_studio_sdk import LabelStudio

LABEL_STUDIO_URL = "http://localhost:8080"
API_KEY = "14a907022934932368e2a5cf6f697b1f772dad8a"

client = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=API_KEY)

# Verify connection
me = client.users.whoami()
print(f"Connected as: {me.email}")

# Get project
projects = client.projects.list()
project_id = None
for p in projects:
    if "QBM" in p.title or "Pilot" in p.title:
        project_id = p.id
        print(f"Using project: {p.title} (ID: {p.id})")
        break

# Get tasks
tasks = client.tasks.list(project=project_id)
task_list = list(tasks)
print(f"Found {len(task_list)} tasks")


def create_annotation(task_id: int, task_data: dict, params: dict):
    """Create annotation for a task."""
    raw_text = task_data.get('raw_text_ar', '')
    span_start = params.get('span_start', 0)
    span_end = params.get('span_end', len(raw_text))
    
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
            "value": {"choices": [params.get('behavior_form', 'inner_state')]}
        },
        {
            "from_name": "agent_type",
            "to_name": "quran_text",
            "type": "choices",
            "value": {"choices": [params.get('agent_type', 'AGT_BELIEVER')]}
        },
        {
            "from_name": "speech_mode",
            "to_name": "quran_text",
            "type": "choices",
            "value": {"choices": [params.get('speech_mode', 'informative')]}
        },
        {
            "from_name": "evaluation",
            "to_name": "quran_text",
            "type": "choices",
            "value": {"choices": [params.get('evaluation', 'praise')]}
        },
        {
            "from_name": "quran_deontic_signal",
            "to_name": "quran_text",
            "type": "choices",
            "value": {"choices": [params.get('deontic_signal', 'khabar')]}
        },
        {
            "from_name": "support_type",
            "to_name": "quran_text",
            "type": "choices",
            "value": {"choices": [params.get('support_type', 'direct')]}
        },
        {
            "from_name": "action_class",
            "to_name": "quran_text",
            "type": "choices",
            "value": {"choices": [params.get('action_class', 'ACT_VOLITIONAL')]}
        },
        {
            "from_name": "action_textual_eval",
            "to_name": "quran_text",
            "type": "choices",
            "value": {"choices": [params.get('action_eval', 'EVAL_SALIH')]}
        },
        {
            "from_name": "systemic",
            "to_name": "quran_text",
            "type": "choices",
            "value": {"choices": params.get('systemic', ['SYS_GOD'])}
        },
        {
            "from_name": "situational",
            "to_name": "quran_text",
            "type": "choices",
            "value": {"choices": [params.get('situational', 'internal')]}
        },
    ]
    
    return client.annotations.create(id=task_id, result=result)


# Complete annotations for all 50 pilot ayat
# Based on QBM methodology and Islamic scholarship

ANNOTATIONS = {
    # Surah Al-Baqarah (2) - Believers and Hypocrites
    "2:3": {"behavior_form": "inner_state", "agent_type": "AGT_BELIEVER", "speech_mode": "informative", "evaluation": "praise", "deontic_signal": "khabar", "support_type": "direct", "action_class": "ACT_VOLITIONAL", "action_eval": "EVAL_SALIH", "systemic": ["SYS_GOD"], "situational": "internal"},
    "2:4": {"behavior_form": "inner_state", "agent_type": "AGT_BELIEVER", "speech_mode": "informative", "evaluation": "praise", "deontic_signal": "khabar", "support_type": "direct", "action_class": "ACT_VOLITIONAL", "action_eval": "EVAL_SALIH", "systemic": ["SYS_GOD"], "situational": "internal"},
    "2:5": {"behavior_form": "trait_disposition", "agent_type": "AGT_BELIEVER", "speech_mode": "informative", "evaluation": "praise", "deontic_signal": "khabar", "support_type": "direct", "action_class": "ACT_VOLITIONAL", "action_eval": "EVAL_SALIH", "systemic": ["SYS_GOD"], "situational": "internal"},
    "2:8": {"behavior_form": "speech_act", "agent_type": "AGT_HYPOCRITE", "speech_mode": "informative", "evaluation": "blame", "deontic_signal": "khabar", "support_type": "direct", "action_class": "ACT_VOLITIONAL", "action_eval": "EVAL_GHAYR_SALIH", "systemic": ["SYS_GOD", "SYS_SOCIETY"], "situational": "external"},
    "2:9": {"behavior_form": "relational_act", "agent_type": "AGT_HYPOCRITE", "speech_mode": "informative", "evaluation": "blame", "deontic_signal": "khabar", "support_type": "direct", "action_class": "ACT_VOLITIONAL", "action_eval": "EVAL_GHAYR_SALIH", "systemic": ["SYS_GOD", "SYS_SELF"], "situational": "external"},
    "2:10": {"behavior_form": "inner_state", "agent_type": "AGT_HYPOCRITE", "speech_mode": "informative", "evaluation": "blame", "deontic_signal": "tarhib", "support_type": "direct", "action_class": "ACT_VOLITIONAL", "action_eval": "EVAL_GHAYR_SALIH", "systemic": ["SYS_SELF"], "situational": "internal"},
    "2:177": {"behavior_form": "mixed", "agent_type": "AGT_BELIEVER", "speech_mode": "informative", "evaluation": "praise", "deontic_signal": "khabar", "support_type": "direct", "action_class": "ACT_VOLITIONAL", "action_eval": "EVAL_SALIH", "systemic": ["SYS_GOD", "SYS_SOCIETY"], "situational": "mixed"},
    "2:195": {"behavior_form": "physical_act", "agent_type": "AGT_BELIEVER", "speech_mode": "command", "evaluation": "praise", "deontic_signal": "amr", "support_type": "direct", "action_class": "ACT_VOLITIONAL", "action_eval": "EVAL_SALIH", "systemic": ["SYS_GOD"], "situational": "external"},
    "2:267": {"behavior_form": "physical_act", "agent_type": "AGT_BELIEVER", "speech_mode": "command", "evaluation": "praise", "deontic_signal": "amr", "support_type": "direct", "action_class": "ACT_VOLITIONAL", "action_eval": "EVAL_SALIH", "systemic": ["SYS_GOD"], "situational": "external"},
    "2:275": {"behavior_form": "physical_act", "agent_type": "AGT_WRONGDOER", "speech_mode": "informative", "evaluation": "blame", "deontic_signal": "tarhib", "support_type": "direct", "action_class": "ACT_VOLITIONAL", "action_eval": "EVAL_GHAYR_SALIH", "systemic": ["SYS_SOCIETY"], "situational": "external"},
    
    # Surah An-Nisa (4) - Social/Family behaviors
    "4:1": {"behavior_form": "inner_state", "agent_type": "AGT_HUMAN_GENERAL", "speech_mode": "command", "evaluation": "praise", "deontic_signal": "amr", "support_type": "direct", "action_class": "ACT_VOLITIONAL", "action_eval": "EVAL_SALIH", "systemic": ["SYS_GOD", "SYS_FAMILY"], "situational": "internal"},
    "4:19": {"behavior_form": "relational_act", "agent_type": "AGT_BELIEVER", "speech_mode": "prohibition", "evaluation": "blame", "deontic_signal": "nahy", "support_type": "direct", "action_class": "ACT_VOLITIONAL", "action_eval": "EVAL_GHAYR_SALIH", "systemic": ["SYS_FAMILY"], "situational": "external"},
    "4:29": {"behavior_form": "physical_act", "agent_type": "AGT_BELIEVER", "speech_mode": "prohibition", "evaluation": "blame", "deontic_signal": "nahy", "support_type": "direct", "action_class": "ACT_VOLITIONAL", "action_eval": "EVAL_GHAYR_SALIH", "systemic": ["SYS_SOCIETY"], "situational": "external"},
    "4:36": {"behavior_form": "relational_act", "agent_type": "AGT_BELIEVER", "speech_mode": "command", "evaluation": "praise", "deontic_signal": "amr", "support_type": "direct", "action_class": "ACT_VOLITIONAL", "action_eval": "EVAL_SALIH", "systemic": ["SYS_GOD", "SYS_FAMILY", "SYS_SOCIETY"], "situational": "external"},
    "4:58": {"behavior_form": "relational_act", "agent_type": "AGT_BELIEVER", "speech_mode": "command", "evaluation": "praise", "deontic_signal": "amr", "support_type": "direct", "action_class": "ACT_VOLITIONAL", "action_eval": "EVAL_SALIH", "systemic": ["SYS_SOCIETY"], "situational": "external"},
    "4:86": {"behavior_form": "speech_act", "agent_type": "AGT_BELIEVER", "speech_mode": "command", "evaluation": "praise", "deontic_signal": "amr", "support_type": "direct", "action_class": "ACT_VOLITIONAL", "action_eval": "EVAL_SALIH", "systemic": ["SYS_SOCIETY"], "situational": "external"},
    "4:135": {"behavior_form": "speech_act", "agent_type": "AGT_BELIEVER", "speech_mode": "command", "evaluation": "praise", "deontic_signal": "amr", "support_type": "direct", "action_class": "ACT_VOLITIONAL", "action_eval": "EVAL_SALIH", "systemic": ["SYS_GOD", "SYS_SOCIETY"], "situational": "external"},
    "4:142": {"behavior_form": "mixed", "agent_type": "AGT_HYPOCRITE", "speech_mode": "informative", "evaluation": "blame", "deontic_signal": "khabar", "support_type": "direct", "action_class": "ACT_VOLITIONAL", "action_eval": "EVAL_GHAYR_SALIH", "systemic": ["SYS_GOD"], "situational": "mixed"},
    "4:145": {"behavior_form": "trait_disposition", "agent_type": "AGT_HYPOCRITE", "speech_mode": "informative", "evaluation": "warning", "deontic_signal": "tarhib", "support_type": "direct", "action_class": "ACT_VOLITIONAL", "action_eval": "EVAL_GHAYR_SALIH", "systemic": ["SYS_GOD"], "situational": "internal"},
    "4:148": {"behavior_form": "speech_act", "agent_type": "AGT_HUMAN_GENERAL", "speech_mode": "prohibition", "evaluation": "blame", "deontic_signal": "nahy", "support_type": "direct", "action_class": "ACT_VOLITIONAL", "action_eval": "EVAL_GHAYR_SALIH", "systemic": ["SYS_SOCIETY"], "situational": "external"},
    
    # Surah Al-Ma'idah (5) - Legal/Ethical behaviors
    "5:2": {"behavior_form": "relational_act", "agent_type": "AGT_BELIEVER", "speech_mode": "command", "evaluation": "praise", "deontic_signal": "amr", "support_type": "direct", "action_class": "ACT_VOLITIONAL", "action_eval": "EVAL_SALIH", "systemic": ["SYS_GOD", "SYS_SOCIETY"], "situational": "external"},
    "5:8": {"behavior_form": "relational_act", "agent_type": "AGT_BELIEVER", "speech_mode": "command", "evaluation": "praise", "deontic_signal": "amr", "support_type": "direct", "action_class": "ACT_VOLITIONAL", "action_eval": "EVAL_SALIH", "systemic": ["SYS_GOD", "SYS_SOCIETY"], "situational": "external"},
    "5:32": {"behavior_form": "physical_act", "agent_type": "AGT_HUMAN_GENERAL", "speech_mode": "informative", "evaluation": "mixed", "deontic_signal": "khabar", "support_type": "direct", "action_class": "ACT_VOLITIONAL", "action_eval": "EVAL_NEUTRAL", "systemic": ["SYS_SOCIETY"], "situational": "external"},
    "5:38": {"behavior_form": "physical_act", "agent_type": "AGT_WRONGDOER", "speech_mode": "command", "evaluation": "blame", "deontic_signal": "amr", "support_type": "direct", "action_class": "ACT_VOLITIONAL", "action_eval": "EVAL_GHAYR_SALIH", "systemic": ["SYS_SOCIETY"], "situational": "external"},
    "5:54": {"behavior_form": "trait_disposition", "agent_type": "AGT_BELIEVER", "speech_mode": "informative", "evaluation": "praise", "deontic_signal": "khabar", "support_type": "direct", "action_class": "ACT_VOLITIONAL", "action_eval": "EVAL_SALIH", "systemic": ["SYS_GOD"], "situational": "mixed"},
    "5:87": {"behavior_form": "physical_act", "agent_type": "AGT_BELIEVER", "speech_mode": "prohibition", "evaluation": "blame", "deontic_signal": "nahy", "support_type": "direct", "action_class": "ACT_VOLITIONAL", "action_eval": "EVAL_GHAYR_SALIH", "systemic": ["SYS_GOD"], "situational": "external"},
    "5:90": {"behavior_form": "physical_act", "agent_type": "AGT_BELIEVER", "speech_mode": "prohibition", "evaluation": "blame", "deontic_signal": "nahy", "support_type": "direct", "action_class": "ACT_VOLITIONAL", "action_eval": "EVAL_GHAYR_SALIH", "systemic": ["SYS_SELF"], "situational": "external"},
    "5:93": {"behavior_form": "trait_disposition", "agent_type": "AGT_BELIEVER", "speech_mode": "informative", "evaluation": "praise", "deontic_signal": "khabar", "support_type": "direct", "action_class": "ACT_VOLITIONAL", "action_eval": "EVAL_SALIH", "systemic": ["SYS_GOD"], "situational": "internal"},
    "5:100": {"behavior_form": "inner_state", "agent_type": "AGT_BELIEVER", "speech_mode": "command", "evaluation": "praise", "deontic_signal": "amr", "support_type": "direct", "action_class": "ACT_VOLITIONAL", "action_eval": "EVAL_SALIH", "systemic": ["SYS_GOD"], "situational": "internal"},
    "5:105": {"behavior_form": "inner_state", "agent_type": "AGT_BELIEVER", "speech_mode": "command", "evaluation": "praise", "deontic_signal": "amr", "support_type": "direct", "action_class": "ACT_VOLITIONAL", "action_eval": "EVAL_SALIH", "systemic": ["SYS_SELF"], "situational": "internal"},
    
    # Surah Al-Hujurat (49) - Social conduct
    "49:1": {"behavior_form": "speech_act", "agent_type": "AGT_BELIEVER", "speech_mode": "prohibition", "evaluation": "blame", "deontic_signal": "nahy", "support_type": "direct", "action_class": "ACT_VOLITIONAL", "action_eval": "EVAL_GHAYR_SALIH", "systemic": ["SYS_GOD"], "situational": "external"},
    "49:2": {"behavior_form": "speech_act", "agent_type": "AGT_BELIEVER", "speech_mode": "prohibition", "evaluation": "blame", "deontic_signal": "nahy", "support_type": "direct", "action_class": "ACT_VOLITIONAL", "action_eval": "EVAL_GHAYR_SALIH", "systemic": ["SYS_GOD"], "situational": "external"},
    "49:6": {"behavior_form": "inner_state", "agent_type": "AGT_BELIEVER", "speech_mode": "command", "evaluation": "praise", "deontic_signal": "amr", "support_type": "direct", "action_class": "ACT_VOLITIONAL", "action_eval": "EVAL_SALIH", "systemic": ["SYS_SOCIETY"], "situational": "internal"},
    "49:9": {"behavior_form": "relational_act", "agent_type": "AGT_BELIEVER", "speech_mode": "command", "evaluation": "praise", "deontic_signal": "amr", "support_type": "direct", "action_class": "ACT_VOLITIONAL", "action_eval": "EVAL_SALIH", "systemic": ["SYS_SOCIETY"], "situational": "external"},
    "49:10": {"behavior_form": "relational_act", "agent_type": "AGT_BELIEVER", "speech_mode": "command", "evaluation": "praise", "deontic_signal": "amr", "support_type": "direct", "action_class": "ACT_VOLITIONAL", "action_eval": "EVAL_SALIH", "systemic": ["SYS_SOCIETY"], "situational": "external"},
    "49:11": {"behavior_form": "speech_act", "agent_type": "AGT_BELIEVER", "speech_mode": "prohibition", "evaluation": "blame", "deontic_signal": "nahy", "support_type": "direct", "action_class": "ACT_VOLITIONAL", "action_eval": "EVAL_GHAYR_SALIH", "systemic": ["SYS_SOCIETY"], "situational": "external"},
    "49:12": {"behavior_form": "speech_act", "agent_type": "AGT_BELIEVER", "speech_mode": "prohibition", "evaluation": "blame", "deontic_signal": "nahy", "support_type": "direct", "action_class": "ACT_VOLITIONAL", "action_eval": "EVAL_GHAYR_SALIH", "systemic": ["SYS_SOCIETY"], "situational": "external"},
    "49:13": {"behavior_form": "inner_state", "agent_type": "AGT_HUMAN_GENERAL", "speech_mode": "informative", "evaluation": "praise", "deontic_signal": "khabar", "support_type": "direct", "action_class": "ACT_VOLITIONAL", "action_eval": "EVAL_SALIH", "systemic": ["SYS_GOD", "SYS_SOCIETY"], "situational": "internal"},
    "49:14": {"behavior_form": "speech_act", "agent_type": "AGT_HUMAN_GENERAL", "speech_mode": "informative", "evaluation": "neutral", "deontic_signal": "khabar", "support_type": "direct", "action_class": "ACT_VOLITIONAL", "action_eval": "EVAL_NEUTRAL", "systemic": ["SYS_GOD"], "situational": "external"},
    "49:15": {"behavior_form": "trait_disposition", "agent_type": "AGT_BELIEVER", "speech_mode": "informative", "evaluation": "praise", "deontic_signal": "khabar", "support_type": "direct", "action_class": "ACT_VOLITIONAL", "action_eval": "EVAL_SALIH", "systemic": ["SYS_GOD"], "situational": "mixed"},
    
    # Surah Al-Munafiqun (63) - Hypocrisy behaviors
    "63:1": {"behavior_form": "speech_act", "agent_type": "AGT_HYPOCRITE", "speech_mode": "informative", "evaluation": "blame", "deontic_signal": "khabar", "support_type": "direct", "action_class": "ACT_VOLITIONAL", "action_eval": "EVAL_GHAYR_SALIH", "systemic": ["SYS_GOD"], "situational": "external"},
    "63:2": {"behavior_form": "speech_act", "agent_type": "AGT_HYPOCRITE", "speech_mode": "informative", "evaluation": "blame", "deontic_signal": "khabar", "support_type": "direct", "action_class": "ACT_VOLITIONAL", "action_eval": "EVAL_GHAYR_SALIH", "systemic": ["SYS_GOD", "SYS_SOCIETY"], "situational": "external"},
    "63:3": {"behavior_form": "inner_state", "agent_type": "AGT_HYPOCRITE", "speech_mode": "informative", "evaluation": "blame", "deontic_signal": "khabar", "support_type": "direct", "action_class": "ACT_VOLITIONAL", "action_eval": "EVAL_GHAYR_SALIH", "systemic": ["SYS_GOD"], "situational": "internal"},
    "63:4": {"behavior_form": "physical_act", "agent_type": "AGT_HYPOCRITE", "speech_mode": "informative", "evaluation": "blame", "deontic_signal": "khabar", "support_type": "direct", "action_class": "ACT_VOLITIONAL", "action_eval": "EVAL_GHAYR_SALIH", "systemic": ["SYS_SOCIETY"], "situational": "external"},
    "63:5": {"behavior_form": "trait_disposition", "agent_type": "AGT_HYPOCRITE", "speech_mode": "informative", "evaluation": "blame", "deontic_signal": "khabar", "support_type": "direct", "action_class": "ACT_VOLITIONAL", "action_eval": "EVAL_GHAYR_SALIH", "systemic": ["SYS_GOD"], "situational": "internal"},
    "63:6": {"behavior_form": "inner_state", "agent_type": "AGT_HYPOCRITE", "speech_mode": "informative", "evaluation": "blame", "deontic_signal": "khabar", "support_type": "direct", "action_class": "ACT_VOLITIONAL", "action_eval": "EVAL_GHAYR_SALIH", "systemic": ["SYS_GOD"], "situational": "internal"},
    "63:7": {"behavior_form": "relational_act", "agent_type": "AGT_HYPOCRITE", "speech_mode": "informative", "evaluation": "blame", "deontic_signal": "khabar", "support_type": "direct", "action_class": "ACT_VOLITIONAL", "action_eval": "EVAL_GHAYR_SALIH", "systemic": ["SYS_SOCIETY"], "situational": "external"},
    "63:8": {"behavior_form": "speech_act", "agent_type": "AGT_HYPOCRITE", "speech_mode": "informative", "evaluation": "blame", "deontic_signal": "khabar", "support_type": "direct", "action_class": "ACT_VOLITIONAL", "action_eval": "EVAL_GHAYR_SALIH", "systemic": ["SYS_SOCIETY"], "situational": "external"},
    "63:9": {"behavior_form": "inner_state", "agent_type": "AGT_BELIEVER", "speech_mode": "prohibition", "evaluation": "blame", "deontic_signal": "nahy", "support_type": "direct", "action_class": "ACT_VOLITIONAL", "action_eval": "EVAL_GHAYR_SALIH", "systemic": ["SYS_GOD", "SYS_FAMILY"], "situational": "internal"},
    "63:10": {"behavior_form": "physical_act", "agent_type": "AGT_BELIEVER", "speech_mode": "command", "evaluation": "praise", "deontic_signal": "amr", "support_type": "direct", "action_class": "ACT_VOLITIONAL", "action_eval": "EVAL_SALIH", "systemic": ["SYS_GOD"], "situational": "external"},
}

# Check which tasks already have annotations
print("\n--- Checking existing annotations ---")
tasks_with_annotations = set()
for task in task_list:
    if task.annotations and len(task.annotations) > 0:
        tasks_with_annotations.add(task.data.get('reference', ''))

print(f"Tasks already annotated: {len(tasks_with_annotations)}")

# Annotate remaining tasks
print("\n--- Creating Annotations ---")
annotated_count = 0
skipped_count = 0

for task in task_list:
    reference = task.data.get('reference', '')
    
    if reference in tasks_with_annotations:
        print(f"Skipping {reference} (already annotated)")
        skipped_count += 1
        continue
    
    if reference in ANNOTATIONS:
        print(f"Annotating task {task.id} ({reference})...")
        try:
            raw_text = task.data.get('raw_text_ar', '')
            params = ANNOTATIONS[reference].copy()
            params['span_start'] = 0
            params['span_end'] = len(raw_text)
            
            annotation = create_annotation(task.id, task.data, params)
            print(f"  Created annotation ID: {annotation.id}")
            annotated_count += 1
        except Exception as e:
            print(f"  Error: {e}")
    else:
        print(f"No annotation defined for {reference}")

print(f"\n--- Done ---")
print(f"New annotations: {annotated_count}")
print(f"Skipped (already done): {skipped_count}")
print(f"Total annotated: {annotated_count + skipped_count}")
