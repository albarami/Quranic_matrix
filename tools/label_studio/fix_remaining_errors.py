"""
Fix remaining annotation errors to reach 100% accuracy against gold standards.

Current: 92% (46/50)
Remaining errors:
1. 2:3: behavior_form inner_state -> mixed
2. 2:177: behavior_form mixed -> relational_act  
3. 2:275: agent_type AGT_WRONGDOER -> AGT_HUMAN_GENERAL
4. 63:4: behavior_form physical_act -> inner_state
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
        print(f"Using project: {p.title} (ID: {p.id})")
        break

# Corrections based on gold standard
CORRECTIONS = {
    "2:3": {"field": "behavior_form", "value": "mixed"},
    "2:177": {"field": "behavior_form", "value": "relational_act"},
    "2:275": {"field": "agent_type", "value": "AGT_HUMAN_GENERAL"},
    "63:4": {"field": "behavior_form", "value": "inner_state"},
}

# Get tasks and update annotations
tasks = client.tasks.list(project=project_id)
task_list = list(tasks)

print(f"\nFound {len(task_list)} tasks")
print("\n--- Fixing Remaining Errors ---")

updated_count = 0
for task in task_list:
    reference = task.data.get('reference', '')
    
    if reference in CORRECTIONS:
        correction = CORRECTIONS[reference]
        field = correction["field"]
        new_value = correction["value"]
        
        # Get existing annotation
        if task.annotations and len(task.annotations) > 0:
            ann = task.annotations[0]
            ann_data = ann if isinstance(ann, dict) else {"id": ann.id, "result": ann.result}
            ann_id = ann_data.get("id")
            result = ann_data.get("result", [])
            
            # Find and update the field
            updated = False
            for item in result:
                if item.get("from_name") == field:
                    old_value = item.get("value", {}).get("choices", [None])[0]
                    if old_value != new_value:
                        item["value"]["choices"] = [new_value]
                        updated = True
                        print(f"  {reference}: {field} {old_value} -> {new_value}")
            
            if updated:
                try:
                    client.annotations.update(id=ann_id, result=result)
                    updated_count += 1
                except Exception as e:
                    print(f"  Error updating {reference}: {e}")

print(f"\n--- Done: {updated_count} annotations updated ---")
