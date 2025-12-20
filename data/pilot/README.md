## Pilot dataset: `pilot_50`

This folder contains a **50-ayah pilot subset** tokenized for annotation.

### Licensing / attribution

Source metadata indicates **Tanzil Project** text, Uthmani, licensed under **Creative Commons Attribution 3.0**.

Please ensure downstream use complies with the license and provides attribution:
- `https://tanzil.net`

### Files

- `pilot_50_metadata.json`: dataset metadata (tokenization + selection criteria)
- `pilot_50_references.txt`: the 50 ayah references (one per line, `surah:ayah`)

### Generating the tokenized selections (recommended)

We do **not** commit full Qur’an corpora to git by default.

To generate `pilot_50` selections locally (JSONL) from your local Tanzil XML:

1) Ensure you have `data/quran/quran-uthmani.xml` (Tanzil Uthmani v1.1).
2) Run:

```bash
python tools/pilot/build_pilot_50_from_xml.py --xml data/quran/quran-uthmani.xml --refs data/pilot/pilot_50_references.txt --out data/pilot/generated/pilot_50_selections.jsonl
```

Then use that JSONL to import tasks into Label Studio (via the task builder in `tools/pilot/`).


