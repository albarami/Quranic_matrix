"""
Fix deontic_signal annotations based on gold standard comparison.

The comparison showed 50% accuracy on deontic_signal. Main issues:
- Using 'khabar' when should be 'targhib' (for praise/promise)
- Using 'khabar' when should be 'tarhib' (for blame/warning)

Rules from QBM methodology:
- amr = command (imperative)
- nahy = prohibition
- targhib = encouragement (praise + promise of reward)
- tarhib = warning (blame + threat of punishment)
- khabar = neutral informative statement
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

# Corrections based on gold standard and QBM methodology
# Format: reference -> correct deontic_signal
DEONTIC_CORRECTIONS = {
    # Praise of believers -> targhib (encouragement)
    "2:3": "targhib",   # Believers who believe in unseen - praise
    "2:4": "targhib",   # Believers who believe in revelation - praise
    "2:5": "targhib",   # They are on guidance - praise/promise
    "2:177": "targhib", # True righteousness - praise
    
    # Blame of hypocrites/wrongdoers -> tarhib (warning)
    "2:8": "tarhib",    # Hypocrites claim faith - blame
    "2:9": "tarhib",    # They deceive - blame
    "4:142": "tarhib",  # Hypocrites deceive Allah - blame
    "4:145": "tarhib",  # Hypocrites in lowest depths - warning
    "63:1": "tarhib",   # Hypocrites lie - blame
    "63:2": "tarhib",   # They swear falsely - blame
    "63:3": "tarhib",   # Hearts sealed - blame
    "63:4": "tarhib",   # Empty vessels - blame
    "63:5": "tarhib",   # Arrogant - blame
    "63:6": "tarhib",   # Forgiveness won't help - warning
    "63:7": "tarhib",   # They hoard - blame
    "63:8": "tarhib",   # They threaten believers - blame
    
    # Commands remain amr
    # Prohibitions remain nahy
    # Neutral informative remain khabar
}

# Get tasks and update annotations
tasks = client.tasks.list(project=project_id)
task_list = list(tasks)

print(f"\nFound {len(task_list)} tasks")
print("\n--- Updating Deontic Signals ---")

updated_count = 0
for task in task_list:
    reference = task.data.get('reference', '')
    
    if reference in DEONTIC_CORRECTIONS:
        correct_deontic = DEONTIC_CORRECTIONS[reference]
        
        # Get existing annotation
        if task.annotations and len(task.annotations) > 0:
            ann = task.annotations[0]
            ann_data = ann if isinstance(ann, dict) else {"id": ann.id, "result": ann.result}
            ann_id = ann_data.get("id")
            result = ann_data.get("result", [])
            
            # Find and update deontic signal
            updated = False
            for item in result:
                if item.get("from_name") == "quran_deontic_signal":
                    old_value = item.get("value", {}).get("choices", [None])[0]
                    if old_value != correct_deontic:
                        item["value"]["choices"] = [correct_deontic]
                        updated = True
                        print(f"  {reference}: {old_value} -> {correct_deontic}")
            
            if updated:
                # Update annotation via API
                try:
                    client.annotations.update(
                        id=ann_id,
                        result=result
                    )
                    updated_count += 1
                except Exception as e:
                    print(f"  Error updating {reference}: {e}")

print(f"\n--- Done: {updated_count} annotations updated ---")
