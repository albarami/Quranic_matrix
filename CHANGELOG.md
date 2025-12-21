# Changelog

All notable changes to the Quranic Behavioral Matrix (QBM) project.

## [0.7.0] - 2025-12-21

### Added
- **FastAPI REST API** with OpenAPI documentation
  - `GET /` - Health check
  - `GET /datasets/{tier}` - Full dataset (gold/silver/research)
  - `GET /spans` - Search with filters (surah, agent, behavior, evaluation, deontic)
  - `GET /spans/{id}` - Get specific span
  - `GET /surahs/{num}` - Get spans by surah
  - `GET /stats` - Dataset statistics
  - `GET /vocabularies` - Controlled vocabularies
  - `GET /docs` - Swagger UI
  - `GET /redoc` - ReDoc documentation
- Pydantic models for API responses
- Test suite with pytest
- FastAPI, uvicorn, httpx dependencies

## [0.6.0] - 2025-12-21

### Added
- **Full Quran Coverage**: 6,236 ayat (100%), 114 surahs
- IAA calculation script with span-level keys
- Jaccard similarity for multi-label fields
- Krippendorff's alpha support (optional)
- Export tiers script (gold/silver/research)
- Progress report script
- Silver tier export: `qbm_silver_20251221.json`

### Fixed
- IAA script aligned with PROJECT_PLAN.md requirements
- Field paths support both plan schema and current schema

## [0.5.0] - 2025-12-21

### Added
- Scale-up annotation infrastructure
- Expert scholar annotation persona
- Arabic text normalization (alif-wasla, diacritics)
- Pattern normalization at load time

### Fixed
- Quran text loading from tokenized source
- EVAL_FASID → EVAL_SAYYI vocabulary alignment
- Targhib/tarhib deontic mapping

## [0.4.0] - 2025-12-20

### Added
- Tafsir integration (Ibn Kathir 100%)
- Tafsir database and lookup tools

## [0.3.0] - 2025-12-19

### Added
- Micro-pilot complete (550 spans)
- IAA measurement: κ = 0.925

## [0.2.0] - 2025-12-18

### Added
- Self-calibration phase
- Annotation guidelines
- Quality check infrastructure

## [0.1.0] - 2025-12-17

### Added
- Initial repository setup
- Project plan and documentation
- Schema definitions
- Vocabulary files
