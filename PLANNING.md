# PLANNING

## Project: Quranic Behavior Matrix (QBM)

This repo hosts the **Qur’anic Human‑Behavior Classification Matrix** specification, controlled vocabularies, and implementation assets needed to build an annotation pipeline and exportable datasets.

Authoritative methodology/spec: see `docs/qbm_spec/`.

## Repository constraints (mandatory)

- **Max 500 lines per file**: split immediately; no exceptions.
- **Fail‑closed** data discipline: no evidence → no label; all categorical fields are controlled IDs.
- **Documentation-first**: update `TASK.md` for all work items (dated).

## Proposed repository layout (v1)

- `docs/`
  - `qbm_spec/` – methodology + schema specification (already split).
  - `startup/` – startup requirements + operational checklists.
  - `coding_manual/` – coding manual v1 (outline + examples; expanded before pilot).
- `vocab/` – **frozen controlled vocabularies** (JSON) with version + frozen_date.
- `schemas/` – JSON Schemas for Quran text, tafsir records, and span records.
- `data/` – ingested raw text datasets (Qur’an tokens, tafsir) and/or pointers + checksums.
- `app/` – future FastAPI/SQLModel implementation (created only when work starts).
- `tests/` – future pytest suite mirroring `app/` structure (created with code).

## Versioning policy (v1)

- Tokenization is versioned via `tokenization_id` (e.g., `tok_v1`).
- Text version is versioned via `quran_text_version` (e.g., `uthmani_hafs_v1`).
- Each vocab file includes `vocabulary_id`, `version`, and `frozen_date`.



