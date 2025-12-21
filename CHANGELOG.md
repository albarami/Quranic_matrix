# Changelog

All notable changes to the Quranic Human-Behavior Classification Matrix (QBM) project.

## [1.0.0] - 2025-12-21

### Added
- **REST API** (Phase 7)
  - FastAPI backend with full OpenAPI documentation
  - Endpoints: `/datasets/{tier}`, `/spans`, `/surahs/{num}`, `/stats`, `/vocabularies`
  - Filtering by surah, agent type, behavior form, evaluation, deontic signal
  - Full-text Arabic search
  - Pagination support
  - Health check endpoints

- **Complete Quran Coverage** (Phase 6)
  - 6,236 ayat annotated (100% coverage)
  - 114 surahs complete
  - Silver tier export with all annotations

- **Quality Metrics** (Phase 6)
  - Inter-Annotator Agreement: κ = 0.925
  - 7/7 evaluated fields pass threshold (κ ≥ 0.7)
  - Export tiers: Gold, Silver, Research

- **Infrastructure** (Phase 5)
  - PostgreSQL database with 6 tables
  - Quality check and coverage audit scripts
  - Batch annotation system (127 batch files)

- **Tafsir Integration** (Phase 4)
  - SQLite database with Ibn Kathir tafsir
  - 6,236 ayat with tafsir lookup
  - Command-line lookup tool

- **Training Materials** (Phase 2-3)
  - Annotator Quick Start Guide
  - Decision Flowcharts
  - Behavior Form Decision Rules
  - Coding Manual v2.0

### Dataset Statistics
| Metric | Value |
|--------|-------|
| Total Ayat | 6,236 |
| Total Surahs | 114 |
| Average IAA (κ) | 0.925 |

### Distribution Highlights
| Field | Top Value | Count |
|-------|-----------|-------|
| Agent Type | AGT_ALLAH | 2,855 |
| Evaluation | neutral | 4,808 |
| Deontic Signal | khabar | 4,741 |

### Contributors
- Dr. Ibrahim Bouzidani (Foundational Research)
- Expert Annotators
- Reviewers

---

## [0.6.0] - 2025-12-21

### Added
- Progress report script (`progress_report.py`)
- IAA calculation script (`calculate_iaa.py`)
- Export tiers script (`export_tiers.py`)
- Silver tier dataset export

### Changed
- Completed Phase 6 infrastructure

---

## [0.5.0] - 2025-12-21

### Added
- Full Quran annotation (6,236 ayat)
- 127 batch annotation files
- Coverage tracking and reporting
- Quality check automation

### Fixed
- Pattern normalization for consistent matching
- Alif-wasla handling in agent/evidence detection
- Deontic signal mapping alignment

---

## [0.4.0] - 2025-12-XX

### Added
- Tafsir database (SQLite)
- Ibn Kathir tafsir integration
- Tafsir lookup tool

---

## [0.3.0] - 2025-12-XX

### Added
- Full pilot annotations (550 spans)
- Second annotator data
- IAA measurement infrastructure

### Achieved
- Inter-Annotator Agreement: κ = 0.925

---

## [0.2.0] - 2025-12-XX

### Added
- Micro-pilot expansion (50 → 100 spans)
- Annotator training materials
- Decision flowcharts
- Behavior form disambiguation rules

### Improved
- behavior_form IAA: 0.703 → 0.884

---

## [0.1.0] - 2025-12-XX

### Added
- Initial pilot annotations (50 spans)
- Gold standard examples
- Controlled vocabularies v1
- Label Studio configuration
- Project structure and documentation

---

## Version History

| Version | Phase | Milestone |
|---------|-------|-----------|
| v0.1.0 | Phase 1 | Self-calibration complete |
| v0.2.0 | Phase 2 | Micro-pilot complete |
| v0.3.0 | Phase 3 | Full pilot complete |
| v0.4.0 | Phase 4 | Tafsir integration |
| v0.5.0 | Phase 5 | Scale-up complete |
| v0.6.0 | Phase 6 | Full coverage |
| v1.0.0 | Phase 7 | Production release |
