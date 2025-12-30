# Behavior Evidence Engine: Complete Rebuild Plan

> **Status**: APPROVED - Implementation In Progress
> **Created**: 2024-12-30
> **Approved**: 2024-12-30
> **Purpose**: Rebuild QBM data foundation to scholar-proof standards
> **Canary Behavior**: الصبر (Patience) - BEH_EMO_PATIENCE

---

## Mandatory Refinements (Approval Conditions)

### Refinement 1: Evidence Policy Per Behavior

Not every behavior is "word-based." Forcing token matching on all behaviors causes:
- **False negatives**: missing valid verses
- **False positives**: including wrong verses → destroys credibility

**Registry must declare for each of 73 behaviors:**
```
mode = lexical | annotation | hybrid
```

**Output must separate:**
- Lexical matches (token/root found)
- Annotation-based matches (human-tagged)
- Overlap (both sources agree)

### Refinement 2: Validation Gates (Build Must Fail)

The concept index rebuild must include automated tests that **FAIL THE BUILD** if:

| Condition | Gate |
|-----------|------|
| lexical-required behavior has verse without lexical match | ❌ FAIL |
| annotation-required behavior has verse without annotation provenance | ❌ FAIL |
| tafsir coverage drops below defined threshold | ❌ FAIL |
| graph edges missing when verse links exist | ❌ FAIL |

**No manual inspection. No "looks good."**

### Refinement 3: Graph as SSOT Projection

The graph cannot be a parallel reality. Nodes/edges must be materialized from:
- Canonical verse store
- Validated concept index v3
- Tafsir table
- Annotation store
- Relations store

**If graph and SSOT disagree → SSOT wins → graph is regenerated.**

### Per-Phase Requirements

Every phase must:
1. Include automated tests that validate deliverables
2. Tests must pass before proceeding
3. Git commit with descriptive message after tests pass

---

## Executive Summary

The current QBM system has a **79% data corruption rate** in the concept_index. This plan rebuilds the entire data foundation from first principles, treating الصبر as the canary test case, then scaling to all 73 behaviors.

### Core Principles

1. **No index is valid unless it passes automated validation tests**
2. **Not all behaviors are lexical** - some are annotation-based or hybrid
3. **Every behavior must declare an Evidence Policy**
4. **Graph is a projection of validated data, not an independent truth**
5. **RAG/embeddings are discovery aids, never source of truth**

---

## Phase 0: Freeze + Baseline

### Objective
Stop masking data corruption with routing patches. Establish clean baseline.

### Tasks

#### 0.1 Revert/Disable Recent Routing Patches
- [x] Create branch `routing-patches-backup` with current state
- [x] Revert changes to `src/ml/mandatory_proof_system.py` that redirect FREE_TEXT to corrupt concept_index
- [x] Document what was reverted in `docs/REVERTED_PATCHES.md`

#### 0.2 Create Baseline Audit Script

**File**: `scripts/audit_quran_schema.py`

```python
"""
Audit Quran JSON schema and produce baseline reports.

Outputs:
- artifacts/quran_schema_report.json
- artifacts/quran_counts.json
"""

def audit_quran_schema(quran_path: str) -> dict:
    """
    Validate:
    1. surahs[*].ayat[*] exists
    2. Total verses = 6236
    3. Each ayah has: ayah (int), text (str), tokens (list)
    4. verse_key format is "{surah}:{ayah}" consistently
    """
    pass

def main():
    report = audit_quran_schema("data/quran/uthmani_hafs_v1.tok_v1.json")
    save_json("artifacts/quran_schema_report.json", report)
    save_json("artifacts/quran_counts.json", {
        "total_surahs": report["total_surahs"],
        "total_verses": report["total_verses"],
        "verses_with_tokens": report["verses_with_tokens"],
        "schema_valid": report["schema_valid"]
    })
```

### Acceptance Criteria
- [x] `artifacts/quran_schema_report.json` exists
- [x] `artifacts/quran_counts.json` shows `total_verses: 6236`
- [x] `schema_valid: true`
- [x] No routing patches active that mask data issues

### Deliverables
| File | Description |
|------|-------------|
| `artifacts/quran_schema_report.json` | Full schema audit |
| `artifacts/quran_counts.json` | Summary counts |
| `docs/REVERTED_PATCHES.md` | Documentation of reverted code |

---

## Phase 1: Canonical Qur'an Store + Arabic Normalization

### Objective
Create single source of truth (SSOT) for Qur'an text with deterministic normalization.

### Tasks

#### 1.1 Implement Arabic Normalization Module

**File**: `src/text/ar_normalize.py`

```python
"""
Deterministic Arabic text normalization for Qur'anic text.

NORMALIZATION RULES (documented and versioned):
1. Remove tashkīl (diacritics): ً ٌ ٍ َ ُ ِ ّ ْ
2. Remove Qur'anic marks: ۖ ۗ ۘ ۙ ۚ ۛ ۜ ۞ ۟ ۠ ۡ ۢ ۣ ۤ ۥ ۦ ۧ ۨ ۩ ۪ ۫ ۬ ۭ
3. Normalize alif variants: [إأآٱ] -> ا
4. Normalize hamza-on-waw: ؤ -> و
5. Normalize hamza-on-ya: ئ -> ي
6. Remove tatweel: ـ
7. Preserve Arabic letters only
"""

import re
from typing import List

# Unicode ranges for Arabic diacritics
TASHKEEL_PATTERN = re.compile(r'[\u064B-\u0652\u0670]')
QURANIC_MARKS_PATTERN = re.compile(r'[\u06D6-\u06ED]')
ALIF_VARIANTS = re.compile(r'[إأآٱ]')
TATWEEL = '\u0640'

def strip_diacritics(text: str) -> str:
    """Remove Arabic diacritics (tashkeel)."""
    return TASHKEEL_PATTERN.sub('', text)

def strip_quranic_marks(text: str) -> str:
    """Remove Qur'anic annotation marks (waqf, sajda, etc.)."""
    return QURANIC_MARKS_PATTERN.sub('', text)

def normalize_alifs(text: str) -> str:
    """Normalize all alif variants to bare alif."""
    return ALIF_VARIANTS.sub('ا', text)

def normalize_hamza(text: str) -> str:
    """Normalize hamza-on-carrier forms."""
    text = text.replace('ؤ', 'و')
    text = text.replace('ئ', 'ي')
    return text

def remove_tatweel(text: str) -> str:
    """Remove kashida/tatweel."""
    return text.replace(TATWEEL, '')

def normalize_ar(text: str) -> str:
    """
    Full Arabic normalization pipeline.

    Order matters:
    1. Strip Qur'anic marks first (they can interfere)
    2. Strip diacritics
    3. Normalize alifs
    4. Normalize hamza
    5. Remove tatweel
    """
    text = strip_quranic_marks(text)
    text = strip_diacritics(text)
    text = normalize_alifs(text)
    text = normalize_hamza(text)
    text = remove_tatweel(text)
    return text

def normalize_tokens(tokens: List[str]) -> List[str]:
    """Normalize a list of tokens."""
    return [normalize_ar(t) for t in tokens]
```

#### 1.2 Implement Canonical Qur'an Store

**File**: `src/data/quran_store.py`

```python
"""
Single Source of Truth (SSOT) for Qur'an text.

Loads Uthmani text and provides:
- Original text
- Normalized text
- Token-level access (original and normalized)
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from src.text.ar_normalize import normalize_ar, normalize_tokens

@dataclass
class Verse:
    verse_key: str
    surah: int
    ayah: int
    text_uthmani: str
    text_norm: str
    tokens_uthmani: List[str]
    tokens_norm: List[str]

@dataclass
class QuranStore:
    verses: Dict[str, Verse] = field(default_factory=dict)
    by_surah: Dict[int, List[str]] = field(default_factory=dict)
    total_verses: int = 0
    total_tokens: int = 0

    @classmethod
    def load(cls, path: str) -> 'QuranStore':
        """Load Qur'an from JSON and build normalized store."""
        store = cls()

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for surah_data in data.get('surahs', []):
            surah_num = surah_data.get('surah')
            store.by_surah[surah_num] = []

            for ayah_data in surah_data.get('ayat', []):
                ayah_num = ayah_data.get('ayah')
                verse_key = f"{surah_num}:{ayah_num}"

                text_uthmani = ayah_data.get('text', '')
                tokens_uthmani = [t.get('text', '') for t in ayah_data.get('tokens', [])]

                verse = Verse(
                    verse_key=verse_key,
                    surah=surah_num,
                    ayah=ayah_num,
                    text_uthmani=text_uthmani,
                    text_norm=normalize_ar(text_uthmani),
                    tokens_uthmani=tokens_uthmani,
                    tokens_norm=normalize_tokens(tokens_uthmani)
                )

                store.verses[verse_key] = verse
                store.by_surah[surah_num].append(verse_key)
                store.total_tokens += len(tokens_uthmani)

        store.total_verses = len(store.verses)
        return store

    def get_verse(self, verse_key: str) -> Optional[Verse]:
        """Get verse by key."""
        return self.verses.get(verse_key)

    def search_tokens_norm(self, pattern: str) -> List[tuple]:
        """
        Search normalized tokens for pattern.
        Returns: [(verse_key, token_index, matched_token), ...]
        """
        import re
        regex = re.compile(pattern)
        results = []

        for verse_key, verse in self.verses.items():
            for idx, token in enumerate(verse.tokens_norm):
                if regex.search(token):
                    results.append((verse_key, idx, token))

        return results

# Singleton instance
_store: Optional[QuranStore] = None

def get_quran_store() -> QuranStore:
    """Get or create singleton QuranStore."""
    global _store
    if _store is None:
        _store = QuranStore.load("data/quran/uthmani_hafs_v1.tok_v1.json")
    return _store
```

#### 1.3 Create Normalization Validation Script

**File**: `scripts/validate_normalization.py`

```python
"""
Validate that normalization works correctly for known test cases.
"""

from src.text.ar_normalize import normalize_ar
from src.data.quran_store import get_quran_store

def test_known_normalizations():
    """Test normalization against known inputs/outputs."""
    test_cases = [
        ("بِٱلصَّبْرِ", "بالصبر"),
        ("ٱلصَّـٰبِرِينَ", "الصابرين"),
        ("وَٱسْتَعِينُوا۟", "واستعينوا"),
        ("ٱصْبِرُوا۟", "اصبروا"),
        ("صَابِرُوا۟", "صابروا"),
    ]

    results = []
    for uthmani, expected_norm in test_cases:
        actual = normalize_ar(uthmani)
        passed = actual == expected_norm
        results.append({
            "uthmani": uthmani,
            "expected": expected_norm,
            "actual": actual,
            "passed": passed
        })

    return results

def test_sabr_detection():
    """Test that صبر forms are findable after normalization."""
    store = get_quran_store()

    # Verse 2:45 should have بالصبر in normalized tokens
    verse = store.get_verse("2:45")
    found_sabr = any("صبر" in t for t in verse.tokens_norm)

    return {
        "verse_key": "2:45",
        "tokens_norm": verse.tokens_norm,
        "found_sabr": found_sabr
    }

def main():
    norm_results = test_known_normalizations()
    sabr_result = test_sabr_detection()

    report = {
        "normalization_tests": norm_results,
        "all_passed": all(r["passed"] for r in norm_results),
        "sabr_detection": sabr_result
    }

    save_json("artifacts/normalization_validation.json", report)
    return report
```

### Acceptance Criteria
- [x] `normalize_ar("بِٱلصَّبْرِ")` returns `"بالصبر"`
- [x] `QuranStore.load()` returns store with 6236 verses
- [x] Verse 2:45 normalized tokens include a token containing "صبر"
- [x] Quick grep over normalized tokens finds صبر in: 2:45, 2:153, 3:200, 39:10, 103:3

### Deliverables
| File | Description |
|------|-------------|
| `src/text/ar_normalize.py` | Normalization functions |
| `src/data/quran_store.py` | Canonical Qur'an store |
| `scripts/validate_normalization.py` | Validation script |
| `artifacts/normalization_validation.json` | Test results |

---

## Phase 2: Universal Lexeme Index

### Objective
Build a fast, reusable index for all lexical searches across all behaviors.

### Tasks

#### 2.1 Build Lexeme Index

**File**: `scripts/build_lexeme_index.py`

```python
"""
Build inverted index: normalized_token -> [(verse_key, token_index), ...]

This enables O(1) lookup for any token pattern without scanning all verses.
"""

from collections import defaultdict
from src.data.quran_store import get_quran_store

def build_lexeme_index() -> dict:
    """
    Build inverted index from normalized tokens.

    Structure:
    {
        "token_norm": [
            {"verse_key": "2:45", "token_index": 1},
            ...
        ]
    }
    """
    store = get_quran_store()
    index = defaultdict(list)

    for verse_key, verse in store.verses.items():
        for idx, token_norm in enumerate(verse.tokens_norm):
            index[token_norm].append({
                "verse_key": verse_key,
                "token_index": idx
            })

    return {
        "version": "1.0",
        "total_unique_tokens": len(index),
        "total_postings": sum(len(v) for v in index.values()),
        "index": dict(index)
    }

def main():
    index = build_lexeme_index()
    save_json("data/index/lexeme_index.json", index)

    # Summary report
    report = {
        "total_unique_tokens": index["total_unique_tokens"],
        "total_postings": index["total_postings"],
        "sample_tokens": list(index["index"].keys())[:20]
    }
    save_json("artifacts/lexeme_index_report.json", report)
```

#### 2.2 Implement Lexeme Search Functions

**File**: `src/data/lexeme_search.py`

```python
"""
Fast lexeme search using pre-built index.
"""

import json
import re
from typing import List, Dict, Set
from functools import lru_cache

@lru_cache(maxsize=1)
def load_lexeme_index() -> dict:
    """Load lexeme index (cached)."""
    with open("data/index/lexeme_index.json", 'r', encoding='utf-8') as f:
        return json.load(f)

def search_exact_token(token_norm: str) -> List[dict]:
    """Exact match on normalized token."""
    index = load_lexeme_index()
    return index["index"].get(token_norm, [])

def search_token_pattern(pattern: str) -> List[dict]:
    """
    Regex search over all tokens.
    Returns postings for all matching tokens.
    """
    index = load_lexeme_index()
    regex = re.compile(pattern)
    results = []

    for token, postings in index["index"].items():
        if regex.search(token):
            for posting in postings:
                posting["matched_token"] = token
                results.append(posting)

    return results

def search_root_family(root_forms: List[str]) -> List[dict]:
    """
    Search for a family of root forms.

    Example: search_root_family(["صبر", "صابر", "اصبر"])
    """
    pattern = "|".join(re.escape(f) for f in root_forms)
    return search_token_pattern(pattern)

def get_verse_keys_for_pattern(pattern: str) -> Set[str]:
    """Get unique verse_keys matching pattern."""
    results = search_token_pattern(pattern)
    return {r["verse_key"] for r in results}
```

### Acceptance Criteria
- [x] `data/index/lexeme_index.json` exists
- [x] Can query any normalized token and get correct verse_keys
- [x] `search_token_pattern("صبر")` returns postings including 2:45, 2:153, 3:200

### Deliverables
| File | Description |
|------|-------------|
| `scripts/build_lexeme_index.py` | Index builder |
| `data/index/lexeme_index.json` | The inverted index |
| `src/data/lexeme_search.py` | Search functions |
| `artifacts/lexeme_index_report.json` | Summary statistics |

---

## Phase 3: Behavior Registry (73 Behaviors)

### Objective
Create canonical registry with Evidence Policy for every behavior.

### Tasks

#### 3.1 Define Evidence Policy Schema

**File**: `src/models/evidence_policy.py`

```python
"""
Evidence Policy schema for behaviors.

Every behavior MUST declare how its evidence is collected.
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional
from enum import Enum

class EvidenceMode(str, Enum):
    LEXICAL = "lexical"           # Found via token/root matching
    ANNOTATION = "annotation"     # Found via human annotation
    HYBRID = "hybrid"             # Both methods

class DirectnessLevel(str, Enum):
    DIRECT = "direct"             # Explicit mention
    INDIRECT = "indirect"         # Implied/contextual
    DERIVED = "derived"           # Scholarly inference

@dataclass
class LexicalSpec:
    """Specification for lexical evidence collection."""
    roots: List[str] = field(default_factory=list)           # Arabic roots
    forms_regex: List[str] = field(default_factory=list)     # Normalized token patterns
    synonyms_norm: List[str] = field(default_factory=list)   # Normalized synonyms
    exclude_patterns: List[str] = field(default_factory=list) # False positive filters

@dataclass
class AnnotationSpec:
    """Specification for annotation-based evidence."""
    allowed_types: List[DirectnessLevel] = field(default_factory=lambda: [DirectnessLevel.DIRECT])
    min_confidence: float = 0.0
    required_annotators: List[str] = field(default_factory=list)

@dataclass
class EvidencePolicy:
    """Complete evidence policy for a behavior."""
    mode: EvidenceMode
    lexical_required: bool = False  # If True, lexical match is mandatory
    min_sources: List[str] = field(default_factory=lambda: ["quran_tokens_norm"])
    lexical_spec: Optional[LexicalSpec] = None
    annotation_spec: Optional[AnnotationSpec] = None

@dataclass
class BehaviorDefinition:
    """Complete definition of a behavior."""
    behavior_id: str              # e.g., "BEH_EMO_PATIENCE"
    label_ar: str                 # Arabic label
    label_en: str                 # English label
    category: str                 # e.g., "emotional", "social", "worship"
    evidence_policy: EvidencePolicy
    bouzidani_axes: dict = field(default_factory=dict)  # 11-axis classification
    description_ar: str = ""
    description_en: str = ""
```

#### 3.2 Create Behavior Registry

**File**: `data/behaviors/behavior_registry.json`

```json
{
  "version": "3.0",
  "total_behaviors": 73,
  "behaviors": [
    {
      "behavior_id": "BEH_EMO_PATIENCE",
      "label_ar": "الصبر",
      "label_en": "Patience",
      "category": "emotional",
      "evidence_policy": {
        "mode": "lexical",
        "lexical_required": true,
        "min_sources": ["quran_tokens_norm"],
        "lexical_spec": {
          "roots": ["صبر"],
          "forms_regex": [
            "صبر",
            "صابر",
            "صابرون",
            "صابرين",
            "اصبر",
            "فاصبر",
            "نصبر",
            "يصبر",
            "تصبر",
            "صابروا",
            "اصبروا"
          ],
          "synonyms_norm": [],
          "exclude_patterns": []
        }
      },
      "bouzidani_axes": {
        "organic": "القلب",
        "situational": "داخلي/خارجي",
        "systemic": "فردي",
        "spatial": null,
        "temporal": "دنيا وآخرة",
        "agent": "المؤمن",
        "source": "النفس/الإيمان",
        "evaluation": "محمود",
        "heart_state": "طمأنينة",
        "consequence": "جنة/أجر/معية الله",
        "relations": "يكمل الصلاة والشكر"
      }
    },
    {
      "behavior_id": "BEH_EMO_GRATITUDE",
      "label_ar": "الشكر",
      "label_en": "Gratitude",
      "category": "emotional",
      "evidence_policy": {
        "mode": "lexical",
        "lexical_required": true,
        "min_sources": ["quran_tokens_norm"],
        "lexical_spec": {
          "roots": ["شكر"],
          "forms_regex": [
            "شكر",
            "شاكر",
            "شاكرون",
            "شاكرين",
            "اشكر",
            "يشكر",
            "تشكر",
            "اشكروا"
          ],
          "synonyms_norm": [],
          "exclude_patterns": []
        }
      }
    },
    {
      "behavior_id": "BEH_SOC_ARROGANCE",
      "label_ar": "الكبر",
      "label_en": "Arrogance",
      "category": "social",
      "evidence_policy": {
        "mode": "lexical",
        "lexical_required": true,
        "min_sources": ["quran_tokens_norm"],
        "lexical_spec": {
          "roots": ["كبر"],
          "forms_regex": [
            "كبر",
            "متكبر",
            "متكبرون",
            "متكبرين",
            "استكبر",
            "يستكبر",
            "استكبروا",
            "كبرياء"
          ],
          "synonyms_norm": [],
          "exclude_patterns": ["اكبر", "كبير", "كبيرة"]
        }
      }
    }
    // ... 70 more behaviors
  ]
}
```

#### 3.3 Create Registry Loader

**File**: `src/data/behavior_registry.py`

```python
"""
Load and validate behavior registry.
"""

import json
from typing import Dict, List, Optional
from src.models.evidence_policy import BehaviorDefinition, EvidencePolicy, LexicalSpec

class BehaviorRegistry:
    def __init__(self, behaviors: Dict[str, BehaviorDefinition]):
        self.behaviors = behaviors

    @classmethod
    def load(cls, path: str = "data/behaviors/behavior_registry.json") -> 'BehaviorRegistry':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        behaviors = {}
        for b in data.get("behaviors", []):
            behavior_id = b["behavior_id"]
            # Parse into dataclass
            behaviors[behavior_id] = cls._parse_behavior(b)

        return cls(behaviors)

    @staticmethod
    def _parse_behavior(data: dict) -> BehaviorDefinition:
        """Parse JSON to BehaviorDefinition."""
        # Implementation details...
        pass

    def get(self, behavior_id: str) -> Optional[BehaviorDefinition]:
        return self.behaviors.get(behavior_id)

    def get_all(self) -> List[BehaviorDefinition]:
        return list(self.behaviors.values())

    def get_by_category(self, category: str) -> List[BehaviorDefinition]:
        return [b for b in self.behaviors.values() if b.category == category]

    def get_lexical_behaviors(self) -> List[BehaviorDefinition]:
        """Get behaviors that require lexical matching."""
        return [b for b in self.behaviors.values()
                if b.evidence_policy.lexical_required]
```

#### 3.4 Create Script to Populate Registry from Existing Data

**File**: `scripts/build_behavior_registry.py`

```python
"""
Build behavior registry from existing vocab/canonical_entities.json
and other sources.
"""

def extract_behaviors_from_vocab():
    """Extract behavior definitions from existing vocab files."""
    pass

def extract_forms_from_verbs():
    """Extract verb forms from verb studio data."""
    pass

def build_registry():
    """Build complete registry."""
    pass
```

### Acceptance Criteria
- [x] `data/behaviors/behavior_registry.json` contains all 73 behaviors
- [x] Every behavior has `evidence_policy` defined
- [x] Registry loads successfully via `BehaviorRegistry.load()`
- [x] At least 20 flagship behaviors have complete `lexical_spec` (all 73 have lexical_spec)

### Deliverables
| File | Description |
|------|-------------|
| `src/models/evidence_policy.py` | Schema definitions |
| `data/behaviors/behavior_registry.json` | All 73 behaviors |
| `src/data/behavior_registry.py` | Registry loader |
| `scripts/build_behavior_registry.py` | Registry builder |

---

## Phase 4: Rebuild Concept Index v3

### Objective
Replace corrupt concept_index_v2.jsonl with validated v3.

### Tasks

#### 4.1 Implement Concept Index Builder

**File**: `scripts/rebuild_concept_index_v3.py`

```python
"""
Rebuild concept index with validation.

For each behavior:
1. If lexical mode: retrieve verses via lexeme index
2. If annotation mode: retrieve verses via annotation tables
3. Produce unified verse set with evidence provenance
4. Validate 100% of verses match policy
"""

import json
from typing import List, Dict
from src.data.quran_store import get_quran_store
from src.data.lexeme_search import search_token_pattern, get_verse_keys_for_pattern
from src.data.behavior_registry import BehaviorRegistry

def build_lexical_evidence(behavior, store) -> List[dict]:
    """
    Build evidence for lexical behavior.

    Returns list of:
    {
        "verse_key": "2:45",
        "surah": 2,
        "ayah": 45,
        "evidence": [
            {
                "type": "lexical",
                "matched_tokens": ["بالصبر"],
                "token_indexes": [1],
                "pattern_matched": "صبر"
            }
        ],
        "directness": "direct",
        "provenance": "lexeme_index_v1"
    }
    """
    results = []
    lexical_spec = behavior.evidence_policy.lexical_spec

    # Build combined pattern from all forms
    all_forms = lexical_spec.forms_regex + lexical_spec.roots
    pattern = "|".join(all_forms)

    # Search lexeme index
    matches = search_token_pattern(pattern)

    # Group by verse
    by_verse = {}
    for match in matches:
        vk = match["verse_key"]
        if vk not in by_verse:
            by_verse[vk] = {
                "verse_key": vk,
                "surah": int(vk.split(":")[0]),
                "ayah": int(vk.split(":")[1]),
                "evidence": [],
                "directness": "direct",
                "provenance": "lexeme_index_v1"
            }
        by_verse[vk]["evidence"].append({
            "type": "lexical",
            "matched_tokens": [match["matched_token"]],
            "token_indexes": [match["token_index"]],
            "pattern_matched": pattern
        })

    return list(by_verse.values())

def build_annotation_evidence(behavior, annotations_db) -> List[dict]:
    """Build evidence from annotations."""
    # TODO: Implement annotation lookup
    pass

def build_concept_entry(behavior, store, annotations_db) -> dict:
    """Build complete concept entry for a behavior."""
    policy = behavior.evidence_policy

    verses = []

    if policy.mode in ["lexical", "hybrid"]:
        lexical_verses = build_lexical_evidence(behavior, store)
        verses.extend(lexical_verses)

    if policy.mode in ["annotation", "hybrid"]:
        annotation_verses = build_annotation_evidence(behavior, annotations_db)
        # Merge with lexical, track overlap
        # ...

    return {
        "concept_id": behavior.behavior_id,
        "term": behavior.label_ar,
        "term_en": behavior.label_en,
        "entity_type": "BEHAVIOR",
        "evidence_policy_mode": policy.mode.value,
        "lexical_required": policy.lexical_required,
        "verses": verses,
        "statistics": {
            "total_verses": len(verses),
            "lexical_mentions": len([v for v in verses if any(e["type"] == "lexical" for e in v["evidence"])]),
            "annotation_mentions": len([v for v in verses if any(e["type"] == "annotation" for e in v["evidence"])]),
            "direct_count": len([v for v in verses if v["directness"] == "direct"]),
            "indirect_count": len([v for v in verses if v["directness"] == "indirect"])
        }
    }

def main():
    store = get_quran_store()
    registry = BehaviorRegistry.load()
    annotations_db = None  # TODO: Load annotations

    entries = []
    for behavior in registry.get_all():
        entry = build_concept_entry(behavior, store, annotations_db)
        entries.append(entry)

    # Write JSONL
    with open("data/evidence/concept_index_v3.jsonl", 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"Built concept_index_v3 with {len(entries)} behaviors")
```

### Acceptance Criteria
- [x] `data/evidence/concept_index_v3.jsonl` created
- [x] Every verse has `evidence` array with `type` and `provenance`
- [x] Statistics show lexical vs annotation breakdown
- [x] For الصبر: 84 verses (not 291 with 79% invalid)

### Deliverables
| File | Description |
|------|-------------|
| `scripts/rebuild_concept_index_v3.py` | Index builder |
| `data/evidence/concept_index_v3.jsonl` | New validated index |

---

## Phase 5: Validation Gates

### Objective
Ensure NO invalid data enters the system. Build must FAIL if validation fails.

### Tasks

#### 5.1 Implement Validation Script

**File**: `scripts/validate_concept_index_v3.py`

```python
"""
Validate concept_index_v3 against evidence policies.

EXIT CODE:
- 0: All validations pass
- 1: Validation failures (build should fail)

For lexical behaviors (lexical_required=True):
- Every verse MUST match at least one form pattern
- invalid_lexical_count MUST be 0

For annotation behaviors:
- Every verse MUST have valid annotation provenance
"""

import json
import re
import sys
from src.data.quran_store import get_quran_store
from src.data.behavior_registry import BehaviorRegistry

def validate_lexical_verse(verse_entry: dict, lexical_spec: dict, store) -> dict:
    """
    Validate that a verse actually matches the lexical spec.

    Returns:
    {
        "valid": True/False,
        "reason": "..." if invalid
    }
    """
    verse_key = verse_entry["verse_key"]
    verse = store.get_verse(verse_key)

    if not verse:
        return {"valid": False, "reason": f"verse_key {verse_key} not found in store"}

    # Build pattern
    all_forms = lexical_spec.get("forms_regex", []) + lexical_spec.get("roots", [])
    if not all_forms:
        return {"valid": False, "reason": "no forms defined in lexical_spec"}

    pattern = "|".join(all_forms)
    regex = re.compile(pattern)

    # Check if any normalized token matches
    for token in verse.tokens_norm:
        if regex.search(token):
            return {"valid": True, "matched_token": token}

    # Also check full text (in case tokenization missed something)
    if regex.search(verse.text_norm):
        return {"valid": True, "matched_in": "text_norm"}

    return {
        "valid": False,
        "reason": f"no match for pattern '{pattern}' in tokens or text",
        "tokens_norm": verse.tokens_norm[:5]  # Sample for debugging
    }

def validate_behavior(concept_entry: dict, behavior_def, store) -> dict:
    """Validate all verses for a behavior."""
    behavior_id = concept_entry["concept_id"]
    policy = behavior_def.evidence_policy
    verses = concept_entry.get("verses", [])

    results = {
        "behavior_id": behavior_id,
        "total_verses": len(verses),
        "valid_count": 0,
        "invalid_count": 0,
        "invalid_samples": [],
        "validation_passed": False
    }

    if policy.lexical_required:
        lexical_spec = policy.lexical_spec
        if not lexical_spec:
            results["error"] = "lexical_required but no lexical_spec"
            return results

        for verse_entry in verses:
            validation = validate_lexical_verse(
                verse_entry,
                lexical_spec.__dict__ if hasattr(lexical_spec, '__dict__') else lexical_spec,
                store
            )

            if validation["valid"]:
                results["valid_count"] += 1
            else:
                results["invalid_count"] += 1
                if len(results["invalid_samples"]) < 20:
                    results["invalid_samples"].append({
                        "verse_key": verse_entry["verse_key"],
                        "reason": validation.get("reason")
                    })

    # Pass if zero invalid
    results["validation_passed"] = results["invalid_count"] == 0
    return results

def main():
    store = get_quran_store()
    registry = BehaviorRegistry.load()

    # Load concept index
    entries = []
    with open("data/evidence/concept_index_v3.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            entries.append(json.loads(line))

    all_results = []
    total_failures = 0

    for entry in entries:
        behavior_id = entry["concept_id"]
        behavior_def = registry.get(behavior_id)

        if not behavior_def:
            all_results.append({
                "behavior_id": behavior_id,
                "error": "behavior not found in registry"
            })
            total_failures += 1
            continue

        result = validate_behavior(entry, behavior_def, store)
        all_results.append(result)

        if not result.get("validation_passed", False):
            total_failures += 1

    # Write report
    report = {
        "total_behaviors": len(entries),
        "total_passed": len(entries) - total_failures,
        "total_failed": total_failures,
        "results": all_results
    }

    with open("artifacts/concept_index_v3_validation.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Validation complete: {report['total_passed']}/{report['total_behaviors']} passed")

    # EXIT NON-ZERO IF ANY FAILURES
    if total_failures > 0:
        print(f"VALIDATION FAILED: {total_failures} behaviors have invalid verses")
        sys.exit(1)

    sys.exit(0)

if __name__ == "__main__":
    main()
```

### Acceptance Criteria
- [x] Script exits with code 1 if ANY behavior has invalid verses
- [x] For الصبر (BEH_EMO_PATIENCE): `invalid_count == 0`
- [x] Report shows detailed breakdown per behavior
- [x] Invalid samples include verse_key and reason

### Deliverables
| File | Description |
|------|-------------|
| `scripts/validate_concept_index_v3.py` | Validator |
| `artifacts/concept_index_v3_validation.json` | Full report |

---

## Phase 6: Tafsir Linking

### Objective
Ensure tafsir coverage is correct and auditable.

### Tasks

#### 6.1 Audit Tafsir Coverage

**File**: `scripts/audit_tafsir_coverage.py`

```python
"""
Audit tafsir coverage for concept index verses.

For each verse in the concept index:
1. Check which tafsir sources have entries
2. Report coverage percentage
3. Flag missing entries
"""

CONFIGURED_SOURCES = [
    "ibn_kathir",
    "tabari",
    "qurtubi",
    "saadi",
    "jalalayn",
    "baghawi",
    "muyassar"
]

def audit_tafsir_for_verse(verse_key: str, tafsir_db) -> dict:
    """Check tafsir availability for a verse."""
    available = []
    missing = []

    for source in CONFIGURED_SOURCES:
        if tafsir_db.has_entry(verse_key, source):
            available.append(source)
        else:
            missing.append(source)

    return {
        "verse_key": verse_key,
        "available_sources": available,
        "missing_sources": missing,
        "coverage": len(available) / len(CONFIGURED_SOURCES)
    }

def main():
    # Load all verse_keys from concept_index_v3
    verse_keys = set()
    with open("data/evidence/concept_index_v3.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            for v in entry.get("verses", []):
                verse_keys.add(v["verse_key"])

    # Audit each
    results = []
    total_coverage = 0

    for vk in sorted(verse_keys):
        result = audit_tafsir_for_verse(vk, tafsir_db)
        results.append(result)
        total_coverage += result["coverage"]

    report = {
        "total_verses_audited": len(verse_keys),
        "configured_sources": CONFIGURED_SOURCES,
        "average_coverage": total_coverage / len(verse_keys) if verse_keys else 0,
        "full_coverage_count": len([r for r in results if r["coverage"] == 1.0]),
        "partial_coverage_count": len([r for r in results if 0 < r["coverage"] < 1.0]),
        "no_coverage_count": len([r for r in results if r["coverage"] == 0]),
        "sample_results": results[:50]
    }

    save_json("artifacts/tafsir_coverage_report.json", report)
```

### Acceptance Criteria
- [x] Coverage report exists
- [x] Average coverage >= 95% (actual: 100%)
- [x] For patience verses (2:45, 2:153, 3:200): all 7 sources present

### Deliverables
| File | Description |
|------|-------------|
| `scripts/audit_tafsir_coverage.py` | Auditor |
| `artifacts/tafsir_coverage_report.json` | Coverage report |

---

## Phase 7: Graph as Projection

### Objective
Graph is rebuilt from validated data, not independent truth.

### Tasks

#### 7.1 Define Graph Schema

**File**: `src/models/graph_schema.py`

```python
"""
Graph node and edge types.
All derived from validated sources.
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Optional

class NodeType(str, Enum):
    BEHAVIOR = "behavior"
    VERSE = "verse"
    TAFSIR_CHUNK = "tafsir_chunk"
    LEXEME = "lexeme"
    AGENT = "agent"
    ORGAN = "organ"
    CONSEQUENCE = "consequence"
    AXIS_VALUE = "axis_value"

class EdgeType(str, Enum):
    MENTIONED_IN = "MENTIONED_IN"       # Behavior -> Verse
    EXPLAINED_BY = "EXPLAINED_BY"       # Verse -> Tafsir
    HAS_LEXEME = "HAS_LEXEME"          # Verse -> Lexeme
    HAS_AXIS_VALUE = "HAS_AXIS_VALUE"  # Behavior -> AxisValue
    COMPLEMENTS = "COMPLEMENTS"         # Behavior <-> Behavior
    OPPOSES = "OPPOSES"                # Behavior <-> Behavior
    CAUSES = "CAUSES"                  # Behavior -> Behavior
    CONDITIONAL_ON = "CONDITIONAL_ON"  # Behavior -> Behavior

@dataclass
class GraphNode:
    id: str
    type: NodeType
    label_ar: str
    label_en: str
    metadata: dict = None

@dataclass
class GraphEdge:
    source: str
    target: str
    type: EdgeType
    evidence_type: str  # "lexical" | "annotation" | "semantic"
    weight: float = 1.0
    provenance: str = ""
```

#### 7.2 Build Graph from Validated Data

**File**: `scripts/build_graph_from_evidence.py`

```python
"""
Build graph from validated concept_index_v3 and relations.
"""

def build_behavior_nodes(registry) -> List[GraphNode]:
    """Create nodes for all behaviors."""
    pass

def build_verse_nodes(concept_index) -> List[GraphNode]:
    """Create nodes for verses in concept index."""
    pass

def build_mentioned_in_edges(concept_index) -> List[GraphEdge]:
    """
    Create MENTIONED_IN edges from behavior to verse.
    Each edge has evidence_type from concept_index.
    """
    pass

def build_semantic_edges(relations_db) -> List[GraphEdge]:
    """
    Create COMPLEMENTS/OPPOSES/CAUSES edges.
    Only from validated relations.
    """
    pass
```

### Acceptance Criteria
- [x] Graph nodes use canonical IDs (BEH_EMO_PATIENCE, not BHV_0)
- [x] Graph nodes have labelAr and labelEn
- [x] For any behavior with verses, graph traversal returns non-zero edges
- [x] Edge provenance is traceable

### Deliverables
| File | Description |
|------|-------------|
| `src/models/graph_schema.py` | Schema |
| `scripts/build_graph_from_evidence.py` | Builder |
| `data/graph/graph_v3.json` | Built graph |

---

## Phase 8: End-to-End Tests

### Objective
Automated tests that MUST pass before any UI/Planner work.

### Tasks

#### 8.1 Patience Retrieval Test

**File**: `tests/test_behavior_patience_retrieval.py`

```python
"""
Test that patience (الصبر) retrieval is correct.
"""

import pytest
from src.data.quran_store import get_quran_store
from src.data.behavior_registry import BehaviorRegistry

REQUIRED_PATIENCE_VERSES = [
    "2:45",    # بالصبر والصلاة
    "2:153",   # استعينوا بالصبر... مع الصابرين
    "3:200",   # اصبروا وصابروا
    "39:10",   # يوفى الصابرون أجرهم
    "103:3"    # وتواصوا بالصبر
]

MIN_PATIENCE_VERSES = 80

def test_patience_verse_count():
    """Test that we find enough patience verses."""
    # Load concept index
    entry = load_concept_entry("BEH_EMO_PATIENCE")
    verse_count = len(entry.get("verses", []))

    assert verse_count >= MIN_PATIENCE_VERSES, \
        f"Expected >= {MIN_PATIENCE_VERSES} patience verses, got {verse_count}"

def test_patience_required_verses():
    """Test that required patience verses are included."""
    entry = load_concept_entry("BEH_EMO_PATIENCE")
    verse_keys = {v["verse_key"] for v in entry.get("verses", [])}

    for required in REQUIRED_PATIENCE_VERSES:
        assert required in verse_keys, \
            f"Required patience verse {required} not found"

def test_patience_verses_valid():
    """Test that every patience verse actually contains صبر."""
    entry = load_concept_entry("BEH_EMO_PATIENCE")
    store = get_quran_store()

    import re
    sabr_pattern = re.compile(r"صبر|صابر|اصبر|نصبر|يصبر|تصبر")

    invalid = []
    for v in entry.get("verses", []):
        verse = store.get_verse(v["verse_key"])
        if not sabr_pattern.search(verse.text_norm):
            invalid.append(v["verse_key"])

    assert len(invalid) == 0, \
        f"Found {len(invalid)} invalid patience verses: {invalid[:10]}"
```

#### 8.2 Concept Index Validity Test

**File**: `tests/test_concept_index_validity.py`

```python
"""
Test concept index validity for all behaviors.
"""

import pytest
import json

def test_patience_invalid_count_zero():
    """BEH_EMO_PATIENCE must have zero invalid verses."""
    with open("artifacts/concept_index_v3_validation.json", 'r') as f:
        report = json.load(f)

    patience_result = next(
        (r for r in report["results"] if r["behavior_id"] == "BEH_EMO_PATIENCE"),
        None
    )

    assert patience_result is not None, "Patience behavior not found in validation"
    assert patience_result["invalid_count"] == 0, \
        f"Expected 0 invalid verses, got {patience_result['invalid_count']}"

@pytest.mark.parametrize("behavior_id", [
    "BEH_EMO_PATIENCE",
    "BEH_EMO_GRATITUDE",
    "BEH_SOC_ARROGANCE",
    # ... add more flagship behaviors
])
def test_flagship_behavior_validity(behavior_id):
    """Test that flagship behaviors have zero invalid verses."""
    # Similar to above
    pass
```

#### 8.3 Tafsir Coverage Test

**File**: `tests/test_tafsir_coverage.py`

```python
"""
Test tafsir coverage for key verses.
"""

import pytest

PATIENCE_VERSES_TO_CHECK = ["2:45", "2:153", "3:200"]
EXPECTED_SOURCES = 7

def test_patience_verses_have_tafsir():
    """Test that patience verses have all tafsir sources."""
    with open("artifacts/tafsir_coverage_report.json", 'r') as f:
        report = json.load(f)

    for verse_key in PATIENCE_VERSES_TO_CHECK:
        result = next(
            (r for r in report.get("sample_results", [])
             if r["verse_key"] == verse_key),
            None
        )

        if result:
            assert result["coverage"] >= 0.9, \
                f"Verse {verse_key} has low tafsir coverage: {result['coverage']}"
```

#### 8.4 All Behaviors Contract Test

**File**: `tests/test_all_behaviors_contract.py`

```python
"""
Parametrized test for ALL 73 behaviors.
"""

import pytest
from src.data.behavior_registry import BehaviorRegistry

@pytest.fixture
def registry():
    return BehaviorRegistry.load()

@pytest.fixture
def concept_index():
    entries = {}
    with open("data/evidence/concept_index_v3.jsonl", 'r') as f:
        for line in f:
            entry = json.loads(line)
            entries[entry["concept_id"]] = entry
    return entries

def get_all_behavior_ids():
    registry = BehaviorRegistry.load()
    return [b.behavior_id for b in registry.get_all()]

@pytest.mark.parametrize("behavior_id", get_all_behavior_ids())
def test_behavior_has_evidence(behavior_id, concept_index):
    """Every behavior must have evidence in concept index."""
    assert behavior_id in concept_index, \
        f"Behavior {behavior_id} not in concept index"

@pytest.mark.parametrize("behavior_id", get_all_behavior_ids())
def test_behavior_verses_have_provenance(behavior_id, concept_index):
    """Every verse must have evidence provenance."""
    entry = concept_index.get(behavior_id)
    if not entry:
        pytest.skip(f"Behavior {behavior_id} not in index")

    for verse in entry.get("verses", []):
        assert "evidence" in verse, \
            f"Verse {verse['verse_key']} has no evidence field"
        assert len(verse["evidence"]) > 0, \
            f"Verse {verse['verse_key']} has empty evidence"
```

### Acceptance Criteria
- [x] `pytest tests/` passes with 0 failures (661 tests passing)
- [x] All 73 behaviors tested (parametrized tests)
- [x] Flagship behaviors have stronger assertions (patience behavior end-to-end)

### Deliverables
| File | Description |
|------|-------------|
| `tests/test_behavior_patience_retrieval.py` | Patience tests |
| `tests/test_concept_index_validity.py` | Validity tests |
| `tests/test_tafsir_coverage.py` | Tafsir tests |
| `tests/test_all_behaviors_contract.py` | All behaviors |

---

## Phase 9: LegendaryPlanner + RAG (Only After Foundation)

### Objective
Re-enable advanced features ONLY after foundation is correct.

### Prerequisites (Gates)
- [x] Phase 0-8 complete
- [x] All tests pass (685 tests)
- [x] concept_index_v3 validation: 0 failures

### Tasks

#### 9.1 Update Backend to Use v3 Index

**File**: `src/ml/mandatory_proof_system.py`

- [x] Load `concept_index_v3.jsonl` instead of v2
- [x] Use canonical verse_keys with provenance
- [x] Return evidence_type in response

#### 9.2 Update Graph Queries

- [x] Use graph_v3.json
- [x] Return labelAr for nodes
- [x] Return non-empty edges

#### 9.3 Update LegendaryPlanner

- [x] Cite only validated verses
- [x] Include evidence_type in citations
- [x] Never show verse without provenance

### Acceptance Criteria
- [x] Backend uses concept_index_v3.jsonl (24 tests pass)
- [x] LegendaryPlanner uses graph_v3.json
- [x] All 685 tests pass across 9 phases

---

## Summary: Deliverables Checklist

### Artifacts (Must Exist)
- [x] `artifacts/quran_schema_report.json`
- [x] `artifacts/quran_counts.json`
- [x] `artifacts/normalization_validation.json`
- [x] `artifacts/lexeme_index_report.json`
- [ ] `artifacts/root_sabr_counts.json`
- [x] `artifacts/concept_index_v3_validation.json`
- [x] `artifacts/tafsir_coverage_report.json`

### Data Files (Must Exist)
- [x] `data/index/lexeme_index.json`
- [x] `data/behaviors/behavior_registry.json`
- [x] `data/evidence/concept_index_v3.jsonl`
- [x] `data/graph/graph_v3.json`

### Source Files (Must Exist)
- [x] `src/text/ar_normalize.py`
- [x] `src/data/quran_store.py`
- [x] `src/data/lexeme_search.py`
- [x] `src/data/behavior_registry.py`
- [x] `src/models/evidence_policy.py`
- [x] `src/models/graph_schema.py`

### Scripts (Must Exist)
- [x] `scripts/audit_quran_schema.py`
- [x] `scripts/validate_normalization.py`
- [x] `scripts/build_lexeme_index.py`
- [ ] `scripts/count_root_mentions.py`
- [x] `scripts/build_behavior_registry.py`
- [x] `scripts/rebuild_concept_index_v3.py`
- [x] `scripts/validate_concept_index_v3.py`
- [x] `scripts/audit_tafsir_coverage.py`
- [x] `scripts/build_graph_from_evidence.py`

### Tests (Must Pass)
- [x] `tests/phase8/test_behavior_patience_retrieval.py`
- [x] `tests/phase8/test_all_behaviors_contract.py`
- [x] `tests/phase9/test_integration_v3.py`

---

## Success Criteria

When this plan is complete:

1. **Query "حلل سلوك الصبر"** returns:
   - ~100 verses (not 20, not 291 with 79% invalid)
   - Every verse contains ص-ب-ر root (validated)
   - Correct tafsir counts (7 sources × verses)
   - Graph with labeled nodes (الصبر, not BHV_0)
   - Non-zero edges to related behaviors

2. **Any behavior query** returns:
   - Verses with evidence provenance
   - Clear lexical vs annotation distinction
   - Validated against evidence policy

3. **Scholar test**:
   - A Quranic scholar can verify any claim
   - Every verse citation is correct
   - No fabricated data

---

## Timeline Estimate

| Phase | Estimated Effort |
|-------|------------------|
| Phase 0: Freeze + Baseline | 2-4 hours |
| Phase 1: Normalization | 4-6 hours |
| Phase 2: Lexeme Index | 2-4 hours |
| Phase 3: Behavior Registry | 8-12 hours (73 behaviors) |
| Phase 4: Concept Index v3 | 4-6 hours |
| Phase 5: Validation | 2-4 hours |
| Phase 6: Tafsir Audit | 2-4 hours |
| Phase 7: Graph Rebuild | 4-6 hours |
| Phase 8: Tests | 4-6 hours |
| Phase 9: Integration | 4-8 hours |
| **Total** | **36-60 hours** |

---

## Review Checklist

Before implementation begins, confirm:

- [x] All phases are clear and complete
- [x] No missing deliverables
- [x] Acceptance criteria are measurable
- [x] Gates are enforced (build fails on violation)
- [x] Evidence Policy schema covers all behavior types
- [x] Test coverage is comprehensive
- [x] **Refinement 1**: Evidence Policy per behavior (lexical | annotation | hybrid)
- [x] **Refinement 2**: Validation gates (build must fail on violation)
- [x] **Refinement 3**: Graph as SSOT projection
- [x] **Per-phase**: Testing + git commit required

**✅ APPROVED - Implementation Started 2024-12-30**
