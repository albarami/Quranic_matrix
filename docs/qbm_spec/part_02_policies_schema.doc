## Quranic Human‑Behavior Classification Matrix (Usul al‑Fiqh–Aligned)
### Standalone methodology and implementation specification (enhanced using Bouzidani’s new paper version)

---

## 6) Tafsir and qira’at policies

### 6.1 Tafsir role
Tafsir is metadata to:
- disambiguate meaning,
- stabilize annotation,
- document interpretive variance,
- justify indirect inferences.

It does not convert indirect into direct.

### 6.2 Baseline tafsir set (example)
- Ibn Kathir
- al‑Tabari
- al‑Sa‘di
- al‑Qurtubi

### 6.3 Tafsir agreement levels (operational criteria)
| Level | Criteria |
|---|---|
| `high` | All consulted sources agree on the meaning affecting the current label. |
| `mixed` | Agree on primary meaning; differ on secondary implications not changing label. |
| `disputed` | Differences would lead to different labels or force `unknown`. |

### 6.4 Qira’at policy (v1 default)
- Default text: Hafs ‘an ‘Asim (standard Uthmani).
- Qira’at variants are **out of scope for v1** unless explicitly flagged.
- If a variant plausibly changes a behavioral label, set `qiraat_note.affects_label=true` and record variants.

---

## 7) Normative / evaluative layer

### 7.1 Qur’an‑only textual layer
Store **how the Qur’an speaks**, not juristic rulings:
- `speech_mode`: command | prohibition | informative | narrative | parable | unknown
- `evaluation`: praise | blame | warning | promise | neutral | mixed | unknown
- `quran_deontic_signal`: amr | nahy | targhib | tarhib | khabar

### 7.2 Deontic signal classification rules (coding‑manual minimum)
| Grammatical form | Default signal | Notes |
|---|---|---|
| Explicit imperative (افعل) | `amr` | If quoted within narrative, keep `amr`, set `speech_mode=narrative`, explain in note. |
| Explicit prohibition (لا تفعل) | `nahy` | If exception (لا…إلا), keep `nahy`, document exception. |
| Descriptive praise + reward | `targhib` | Require explicit reward/praise cues. |
| Descriptive blame + threat | `tarhib` | Require explicit threat/punishment cues. |
| Pure report/narrative | `khabar` | Use `evaluation` to capture praise/blame without forcing targhib/tarhib. |

### 7.3 Juristic layer (optional)
If enabled, store separately with provenance:
- `juristic_deontic_status`: wajib|mustahabb|mubah|makruh|haram
- `madhhab`, `usul_rules_invoked[]`, `sources[]`, `disputed`

### 7.4 Abrogation (optional)
Default: `abrogation.status="unknown"` unless assessed and sourced.

---

## 8) Agent modeling

### 8.1 Agent types (controlled)
- AGT_BELIEVER
- AGT_HYPOCRITE
- AGT_DISBELIEVER
- AGT_HUMAN_GENERAL
- AGT_PROPHET_MESSENGER
- AGT_ANGEL
- AGT_JINN
- AGT_ANIMAL
- AGT_COLLECTIVE
- AGT_UNKNOWN

### 8.2 Multi-agent interaction (recommended)
For collective/multi-agent spans:
- `agent.composition`: AGT_*
- `agent.multi_agent_interaction`: true|false
- `agent.interaction_type`: conflict | cooperation | dialogue | unknown

---

## 9) Data model (span record)

### 9.1 Span record (canonical JSON)

```json
{
  "id": "QBM_000001",
  "quran_text_version": "uthmani_hafs_v1",
  "tokenization_id": "tok_v1",

  "reference": {"surah": 24, "ayah": 24},

  "span": {
    "token_start": 0,
    "token_end": 35,
    "raw_text_ar": "...",
    "boundary_confidence": "certain",
    "alternative_boundaries": []
  },

  "linked_spans": [
    {"span_id": "QBM_000002", "relation": "continuation"}
  ],

  "intra_textual_references": [
    {"target_span_id": "QBM_XXXXX", "relation_type": "contrast", "note": "Short rationale"}
  ],

  "behavior": {"concepts": [], "form": "speech_act"},
  "thematic_constructs": ["THM_ACCOUNTABILITY"],
  "meta_discourse_constructs": [],

  "agent": {
    "type": "AGT_HUMAN_GENERAL",
    "group": "unknown",
    "explicit": false,
    "support_type": "indirect",
    "evidence_anchor": null,
    "note": "context inferred"
  },

  "provenance": {
    "extraction_method": "keyword_search",
    "search_query": "",
    "candidate_set_size": 0,
    "selection_rationale": "",
    "excluded_alternatives": [],
    "timestamp": ""
  },

  "tafsir": {
    "sources_used": ["IbnKathir"],
    "agreement_level": "high",
    "consultation_trigger": "baseline",
    "notes": "short summary"
  },

  "assertions": [
    {
      "assertion_id": "QBM_000001_A001",
      "axis": "AX_ORGAN",
      "value": "ORG_TONGUE",
      "organ_role": "tool",
      "organ_semantic_domains": [],
      "primary_organ_semantic_domain": "unknown",
      "support_type": "direct",
      "indication_tags": ["dalalah_mantuq"],
      "evidence_anchor": {"token_start": 5, "token_end": 6},
      "justification_code": "JST_EXPLICIT_MENTION",
      "justification": "Explicit organ mention.",
      "confidence": 0.95,
      "polarity": "neutral",
      "negated": false,
      "negation_type": "unknown",
      "ess": {"directness": 1.0, "clarity": 0.9, "specificity": 0.9, "tafsir_agreement": 1.0, "final": 0.81}
    },
    {
      "assertion_id": "QBM_000001_A002",
      "axis": "AX_THEMATIC_CONSTRUCT",
      "value": "THM_ACCOUNTABILITY",
      "support_type": "direct",
      "indication_tags": ["dalalah_mantuq"],
      "evidence_anchor": null,
      "justification_code": "JST_NARRATIVE_CONTEXT",
      "justification": "Span-level accountability framing.",
      "confidence": 0.80,
      "polarity": "neutral",
      "negated": false,
      "negation_type": "unknown",
      "ess": {"directness": 0.7, "clarity": 0.8, "specificity": 0.6, "tafsir_agreement": 1.0, "final": 0.34}
    }
  ],

  "periodicity": {
    "category": "PER_UNKNOWN",
    "grammatical_indicator": "GRM_NONE",
    "support_type": "unknown",
    "indication_tags": [],
    "evidence_anchor": null,
    "justification_code": null,
    "justification": "",
    "confidence": null,
    "ess": null
  },

  "frequency_prescription": {"source": "unknown", "allow": false, "note": ""},

  "normative_textual": {
    "speech_mode": "informative",
    "evaluation": "neutral",
    "quran_deontic_signal": "khabar",
    "support_type": "direct",
    "note": ""
  },

  "normative_juristic": null,

  "agent_intention_interaction": null,

  "abrogation": {"status": "unknown", "related_spans": [], "sources": [], "note": ""},

  "qiraat_note": {"variant_exists": false, "affects_label": false, "variants": [], "policy": "out_of_scope_v1"},

  "review": {
    "status": "draft",
    "annotator_id": "A1",
    "reviewer_id": "",
    "created_at": "",
    "manual_version": "v1.0"
  }
}
```

### 9.2 Periodicity + frequency (paper alignment)
The updated paper distinguishes routine/instinctive patterns and emphasizes intention.

Rules:
- Periodicity is only labeled when supported by explicit grammar/lexemes; otherwise `PER_UNKNOWN`.
- If a frequency is known from Sunnah/ijma (e.g., prayer counts), store in `frequency_prescription` and keep Qur’anic periodicity unknown unless the Qur’an encodes it.

Paper‑aligned periodicity interpretation notes:
- When periodicity is applicable, map to: `PER_AUTOMATIC` (تلقائي/غريزي/لا إرادي), `PER_HABIT` (عادة متكررة), and optionally `PER_ROUTINE_DAILY` as a **project operationalization** for highly regular daily practices (not a Qur’anic grammar claim unless explicitly encoded).
- Application guidance (not textual labeling): the paper references research indicating habit formation ranges widely (e.g., **~3 weeks** up to **~9 months**) and also suggests applying the matrix over an application window (e.g., **~3–36 weeks** with a minimum not less than 3 weeks). Do **not** backfill Qur’anic periodicity labels from these timelines.


