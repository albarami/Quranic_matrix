## Label Studio (QBM)

This folder contains Label Studio assets for annotating Qur’anic spans using the QBM controlled vocabularies.

### Files

- `labeling_interface.xml`: paste into Label Studio **Settings → Labeling Interface**
- `sample_tasks.jsonl`: example task import format (JSONL)

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

### Export conversion

Use the converter:
- `tools/label_studio/convert_export.py`

It maps Label Studio choice values to the frozen IDs in `vocab/` (e.g., `command` → `SPM_COMMAND`).


