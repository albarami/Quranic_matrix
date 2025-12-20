## Complete Startup Requirements for Quranic Behavior Matrix Project

This document captures the **startup requirements** and **immediate next steps** for bootstrapping the Quranic Behavior Matrix (QBM) pipeline.

---

## 1) TEXT DATA (The Quran)

### What you need

| Item | Format | Source |
|---|---|---|
| Uthmani Quran Text | JSON with token indices | Tanzil.net, QuranEnc, or King Fahd Complex |
| Tokenization Scheme | Documented token boundaries | Must define and freeze |

### Required structure

```json
{
  "quran_text_version": "uthmani_hafs_v1",
  "tokenization_id": "tok_v1",
  "surahs": [
    {
      "surah": 1,
      "ayat": [
        {
          "ayah": 1,
          "text": "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
          "tokens": [
            {"index": 0, "text": "بِسْمِ", "start_char": 0, "end_char": 4},
            {"index": 1, "text": "اللَّهِ", "start_char": 5, "end_char": 9}
          ]
        }
      ]
    }
  ]
}
```

### Recommended sources

- Tanzil.net — free, well-maintained, multiple formats
- QuranEnc.com — API available
- King Fahd Glorious Quran Printing Complex — authoritative

---

## 2) TAFSIR DATA

### Minimum required (4 sources)

| Tafsir | Priority | Availability |
|---|---|---|
| Ibn Kathir | Required | Widely available |
| al-Tabari | Required | Available digitally |
| al-Sa'di | Required | Available digitally |
| al-Qurtubi | Required | Available digitally |

### Format needed

```json
{
  "tafsir_id": "ibn_kathir",
  "reference": {"surah": 24, "ayah": 24},
  "text_ar": "...",
  "text_en": "..."
}
```

### Sources (examples)

- Altafsir.com — multiple tafsirs, Arabic
- QuranX.com — API access
- Tanzil.net — some tafsirs included
- IslamWeb.net — comprehensive Arabic tafsir library

---

## 3) CONTROLLED VOCABULARIES (Freeze Before Pilot)

### Files you must create

| File | Content | Format |
|---|---|---|
| `vocab/organs.json` | ORG_HEART, ORG_TONGUE, etc. + Arabic equivalents | JSON |
| `vocab/agents.json` | AGT_BELIEVER, AGT_HYPOCRITE, etc. | JSON |
| `vocab/systemic.json` | SYS_SELF, SYS_CREATION, etc. | JSON |
| `vocab/spatial.json` | LOC_HOME, LOC_WORK, etc. | JSON |
| `vocab/temporal.json` | TMP_MORNING, TMP_NIGHT, etc. | JSON |
| `vocab/behavior_concepts.json` | BEH_SPEECH_TRUTHFULNESS, etc. | JSON |
| `vocab/thematic_constructs.json` | THM_ACCOUNTABILITY, etc. | JSON |
| `vocab/justification_codes.json` | JST_EXPLICIT_MENTION, etc. | JSON |
| `vocab/grammatical_indicators.json` | GRM_KANA_IMPERFECT, etc. | JSON |

### Example format

```json
{
  "vocabulary_id": "organs_v1",
  "version": "1.0",
  "frozen_date": "2025-01-15",
  "items": [
    {"id": "ORG_HEART", "ar": "قلب", "semantic_domains": ["physiological", "cognitive", "spiritual", "emotional"]},
    {"id": "ORG_TONGUE", "ar": "لسان", "semantic_domains": []},
    {"id": "ORG_EYE", "ar": "عين/بصر", "semantic_domains": []}
  ]
}
```

---

## 4) CODING MANUAL (Must Write Before Pilot)

### Sections required

- Span Segmentation Rules (token_start/token_end)
- Organ Annotation Playbook (incl. heart semantic domains)
- Deontic Decision Tree (amr/nahy/targhib/tarhib/khabar)
- Negation Pattern Guide (لا…ولكن)
- Periodicity Rules (PER_HABIT vs PER_UNKNOWN)
- Confidence Calibration (ESS)
- Tafsir Consultation Protocol (agreement recording)
- Edge Case Examples (20+ fully-worked JSON examples)

### Format

- Word/PDF document (30–50 pages)
- Supplementary JSON examples file

---

## 5) ANNOTATION TOOL

### Options

| Tool | Pros | Cons | Cost |
|---|---|---|---|
| Label Studio | Open source, customizable, RTL support | Needs configuration | Free |
| Prodigy | Fast, good UX | Less customizable for complex schema | Paid |
| INCEpTION | Academic, multi-layer annotation | Learning curve | Free |
| Custom Build | Exactly what you need | Development time | Dev cost |
| Google Sheets + Scripts | Simple, collaborative | Limited for complex schema | Free |

### Minimum requirements

- RTL Arabic text display
- Multi-assertion per span
- JSON export
- Multi-annotator support
- Review/approval workflow

### Recommended for pilot

- Label Studio (free, configurable) or custom lightweight web app

---

## 6) DATABASE

### Schema requirements (starter)

```sql
-- Core tables
spans (id, surah, ayah, token_start, token_end, raw_text_ar, ...)
assertions (id, span_id, axis, value, support_type, confidence, ...)
tafsir_consultations (id, span_id, sources_used, agreement_level, ...)
reviews (id, span_id, status, annotator_id, reviewer_id, ...)

-- Lookup tables
vocabularies (id, vocabulary_type, item_id, item_ar, ...)
```

### Recommended

- PostgreSQL — robust, JSON support
- MongoDB — if you prefer document-based (matches JSON schema)

---

## 7) HUMAN RESOURCES

### Minimum team for pilot

| Role | Count | Requirements |
|---|---:|---|
| Project Lead | 1 | You (Salim) |
| Annotators | 2–3 | Arabic fluency + Quranic knowledge + training |
| Reviewer | 1 | Senior Islamic studies background |
| Technical Support | 1 | Database + export scripts |

### Annotator qualifications

- Native/fluent Arabic
- Familiarity with Quranic Arabic (not just MSA)
- Basic tafsir literacy
- Ability to follow systematic coding rules
- 8–16 hours training on coding manual

---

## 8) CALIBRATION DATASET

### Before pilot, create

| Item | Count | Purpose |
|---|---:|---|
| Gold-standard examples | 20–30 spans | Training + calibration |
| Edge case examples | 10–15 spans | Difficult decisions |
| IAA test set | 50 spans | Measure inter-annotator agreement |

These must be:
- Fully annotated by you + reviewer
- Documented with justifications
- Used to train annotators before they start

---

## 9) IMPLEMENTATION CHECKLIST

### Phase 0: Setup (2–4 weeks)

- Acquire tokenized Quran text (Uthmani Hafs)
- Define and document tokenization scheme
- Acquire tafsir data (4 sources minimum)
- Create all controlled vocabulary JSON files
- Freeze vocabularies with version number
- Set up database
- Set up annotation tool
- Write coding manual v1

### Phase 1: Calibration (1–2 weeks)

- Create 30 gold-standard examples yourself
- Have reviewer validate gold examples
- Recruit 2–3 annotators
- Train annotators (8–16 hours)
- Run calibration test (each annotates same 20 spans)
- Calculate IAA, identify disagreements
- Revise coding manual based on disagreements

### Phase 2: Micro-Pilot (2–3 weeks)

- Select 50–100 spans for micro-pilot
- Double-annotate all spans
- Calculate IAA per axis
- Adjudicate disagreements
- Revise manual again if needed
- Export first test dataset

### Phase 3: Pilot (4–6 weeks)

- Scale to 200–500 spans
- Measure IAA against targets
- Review and approve spans
- Export Gold/Silver/Research tiers
- Document lessons learned

---

## 11) IMMEDIATE NEXT STEPS

### This week

- Download Quran text from Tanzil.net (JSON format)
- Define tokenization — decide token boundaries, document
- Create vocabulary files — start with organs, agents, systemic

### Next 2 weeks

- Write coding manual — at least span segmentation + organ rules + 10 examples
- Set up Label Studio or choose annotation tool
- Create 20 gold examples yourself

### Week 3–4

- Recruit annotators — universities, Islamic studies programs
- Train annotators on coding manual
- Run calibration — measure IAA, refine


