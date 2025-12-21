# Quranic Behavioral Matrix (QBM)

> مصفوفة التصنيف القرآني للسلوك البشري

A **structured dataset of Quranic behavioral classifications** grounded in Islamic scholarship, covering all **6,236 ayat** of the Holy Quran.

[![Version](https://img.shields.io/badge/version-0.8.0-blue.svg)](CHANGELOG.md)
[![Coverage](https://img.shields.io/badge/coverage-100%25-green.svg)](reports/coverage/)
[![IAA](https://img.shields.io/badge/IAA-κ%200.925-brightgreen.svg)](reports/iaa/)

---

## Overview

This project implements Dr. Ibrahim Bouzidani's five-context behavioral classification framework from "السلوك البشري في سياقه القرآني" (Human Behavior in the Quranic Context).

### Key Statistics

| Metric | Value |
|--------|-------|
| **Ayat Covered** | 6,236 (100%) |
| **Surahs** | 114 (100%) |
| **Behavioral Spans** | 6,236+ |
| **Tafsir Sources** | Ibn Kathir |
| **IAA (Cohen's κ)** | 0.925 |

### Classification Framework

Behaviors are classified across five contexts:

1. **العضوي (Organic)** — Body organs involved (heart, tongue, eyes, hands)
2. **الموضعي (Situational)** — Internal (باطني) vs External (ظاهري)
3. **النسقي (Systemic)** — Social contexts (SYS_GOD, SYS_SOCIAL)
4. **المكاني (Spatial)** — Location context
5. **الزماني (Temporal)** — Time context

---

## Quick Start

### 1. Backend API

```bash
# Install dependencies
pip install -r requirements.txt

# Run FastAPI server
uvicorn src.api.main:app --reload

# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### 2. Frontend (Optional)

```bash
cd qbm-frontend
npm install
npm run dev

# Frontend at http://localhost:3000
```

### 3. Run Tests

```bash
pytest tests/ -v
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/datasets/{tier}` | GET | Full dataset (gold/silver/research) |
| `/spans` | GET | Search with filters |
| `/spans/{id}` | GET | Get specific span |
| `/surahs/{num}` | GET | Get spans by surah |
| `/stats` | GET | Dataset statistics |
| `/vocabularies` | GET | Controlled vocabularies |
| `/tafsir/{surah}/{ayah}` | GET | Get tafsir for ayah |
| `/tafsir/compare/{surah}/{ayah}` | GET | Compare tafsir sources |
| `/ayah/{surah}/{ayah}` | GET | Get ayah with annotations |
| `/api/spans/search` | POST | Search spans (C1 frontend) |

---

## Project Structure

```
quranic-behavior-matrix/
├── src/
│   ├── api/                 # FastAPI backend
│   │   ├── main.py          # API endpoints
│   │   └── models.py        # Pydantic models
│   ├── scripts/             # Utility scripts
│   │   ├── annotate_batch_expert.py
│   │   ├── calculate_iaa.py
│   │   ├── export_tiers.py
│   │   ├── progress_report.py
│   │   └── quality_check.py
│   └── validation/          # Schema validation
├── data/
│   ├── annotations/         # Annotation files
│   ├── exports/             # Exported datasets
│   ├── tafsir/              # Tafsir sources
│   └── vocab/               # Controlled vocabularies
├── qbm-frontend/            # Next.js + C1 frontend
├── tests/                   # Test suite
├── reports/                 # Coverage & IAA reports
├── docs/                    # Documentation
├── CHANGELOG.md             # Version history
├── PROJECT_PLAN.md          # Full project plan
└── requirements.txt         # Python dependencies
```

---

## Version History

| Version | Date | Milestone |
|---------|------|-----------|
| v0.7.0 | 2025-12-21 | Production Release (API + Frontend) |
| v0.6.0 | 2025-12-21 | Full Quran Coverage (6,236 ayat) |
| v0.5.0 | 2025-12-21 | Scale-Up Complete |

See [CHANGELOG.md](CHANGELOG.md) for full history.

---

## Citation

```bibtex
@misc{qbm2025,
  title={Quranic Behavioral Matrix: A Structured Dataset of Quranic Behavioral Classifications},
  author={Al-Barami, Salim and Bouzidani, Ibrahim},
  year={2025},
  publisher={GitHub},
  url={https://github.com/albarami/Quranic_matrix}
}
```

---

## Contributors

- **Salim Al-Barami** — Project Lead
- **Dr. Ibrahim Bouzidani** — Framework Design

---

## License

This project is part of the Quranic Behavioral Matrix research initiative.
Contact for permissions.

---

*Built with ❤️ for Islamic scholarship*
