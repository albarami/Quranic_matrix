## Label Studio (QBM)

This folder contains Label Studio assets for annotating Qur’anic spans using the QBM controlled vocabularies.

### Files

- `labeling_interface.xml`: paste into Label Studio **Settings → Labeling Interface**
- `sample_tasks.jsonl`: example task import format (JSONL) using **Label Studio's `data` wrapper**

### Install & run (local)

```bash
pip install label-studio
label-studio start
```

### Project setup (manual)

- Create a new project named **Quranic Human-Behavior Classification Matrix**
- Go to **Settings → Labeling Interface**
- Paste `label_studio/labeling_interface.xml`
- Import tasks from `label_studio/sample_tasks.jsonl`

### Pilot import (pilot_50)

If you have generated pilot selections JSONL, you can build Label Studio import tasks:

```bash
python tools/pilot/build_label_studio_tasks_from_pilot.py --in data/pilot/generated/pilot_50_selections.jsonl --out label_studio/pilot_50_tasks.jsonl --qbm-id-prefix QBM --qbm-id-width 5 --outer-id-start 1
```

### Export conversion

Use the converter:
- `tools/label_studio/convert_export.py`

It maps Label Studio choice values to the frozen IDs in `vocab/` (e.g., `command` → `SPM_COMMAND`).


