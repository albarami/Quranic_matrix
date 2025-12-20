"""
Fix annotations for verses that have multiple gold spans.

4:36 has two gold spans:
- GOLD_004: worship (physical_act, neutral)
- GOLD_005: kindness to parents (relational_act, praise)

For verse-level annotation, this should be 'mixed' behavior_form.
"""

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
        print("Using project: {} (ID: {})".format(p.title, p.id))
        break

# For verse-level annotations of multi-span verses, use 'mixed' form
# and aggregate evaluation appropriately
CORRECTIONS = {
    # 4:36: Contains worship (physical_act) + kindness (relational_act) = mixed
    # Evaluation: neutral + praise = mixed (or praise as dominant)
    "4:36": {
        "behavior_form": "mixed",
        "evaluation": "praise",  # Keep praise as the dominant evaluation
    },
}

# Get tasks and update
tasks = client.tasks.list(project=project_id)
task_list = list(tasks)

print("\nFound {} tasks".format(len(task_list)))
print("\n--- Fixing Multi-Span Verse Annotations ---")

updated_count = 0
for task in task_list:
    reference = task.data.get('reference', '')
    
    if reference in CORRECTIONS:
        corrections = CORRECTIONS[reference]
        
        if task.annotations and len(task.annotations) > 0:
            ann = task.annotations[0]
            ann_data = ann if isinstance(ann, dict) else {"id": ann.id, "result": ann.result}
            ann_id = ann_data.get("id")
            result = ann_data.get("result", [])
            
            updated = False
            for field, new_value in corrections.items():
                for item in result:
                    if item.get("from_name") == field:
                        old_value = item.get("value", {}).get("choices", [None])[0]
                        if old_value != new_value:
                            item["value"]["choices"] = [new_value]
                            updated = True
                            print("  {}: {} {} -> {}".format(reference, field, old_value, new_value))
            
            if updated:
                try:
                    client.annotations.update(id=ann_id, result=result)
                    updated_count += 1
                except Exception as e:
                    print("  Error updating {}: {}".format(reference, e))

print("\n--- Done: {} annotations updated ---".format(updated_count))
