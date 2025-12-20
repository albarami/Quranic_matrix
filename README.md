## Quranic Behavior Matrix (QBM)

This repository hosts a **Qur’an‑grounded, audit‑grade classification matrix for human behavior** aligned with **Usul al‑Fiqh evidence discipline** (fail‑closed: **no evidence → no label**).

### Core documents

- **Methodology + schema spec**: `docs/qbm_spec/` (split into parts to respect the repo **500-line/file** rule)
- **Startup requirements**: `docs/startup/startup_requirements.md`
- **Coding manual (outline; expand before pilot)**: `docs/coding_manual/coding_manual_v1_outline.md`

### Repository structure (v1)

- `docs/`
  - `qbm_spec/` – full specification (Markdown + Word-openable `.doc` text parts)
  - `startup/` – project bootstrap requirements and checklists
  - `coding_manual/` – coding manual v1 outline + examples folder (to be expanded)
- `vocab/` – frozen controlled vocabularies (JSON)
- `schemas/` – JSON schemas for Quran tokenized text + tafsir records (v1)
- `data/` – ingestion stubs + policy notes (do not commit copyrighted corpora)
- `PLANNING.md` – repo constraints + layout
- `TASK.md` – task tracking log

---

## Quick start (what to do next)

### 1) Freeze Quran text + tokenization

- Choose Qur’an source (e.g., Tanzil Uthmani Hafs).
- Define token boundaries and freeze:
  - `quran_text_version` (e.g., `uthmani_hafs_v1`)
  - `tokenization_id` (e.g., `tok_v1`)
- Validate against `schemas/quran_tokenized_v1.schema.json`.

### 2) Acquire tafsir baseline (minimum 4 sources)

- Ibn Kathir, al-Tabari, al-Sa‘di, al-Qurtubi
- Validate records against `schemas/tafsir_record_v1.schema.json`.

### 3) Freeze controlled vocabularies

Edit/extend JSON files under `vocab/` and keep IDs stable:
- `organs.json`
- `agents.json`
- `systemic.json`
- `spatial.json`
- `temporal.json`
- `behavior_concepts.json`
- `thematic_constructs.json`
- `justification_codes.json`
- `grammatical_indicators.json`

### 4) Write the coding manual v1 (required before pilot)

Expand:
- `docs/coding_manual/coding_manual_v1_outline.md`
Add:
- 20+ worked JSON examples under `docs/coding_manual/examples/`

### 5) Run micro‑pilot → pilot

See: `docs/qbm_spec/part_03_measurement_appendices.md` for IAA targets and export tiers.


