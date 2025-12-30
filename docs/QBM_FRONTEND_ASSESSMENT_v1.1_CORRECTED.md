# QBM FRONTEND COMPREHENSIVE ASSESSMENT
## Version 1.1 (CORRECTED) | December 30, 2025

---

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    QBM FRONTEND COMPREHENSIVE ASSESSMENT                      ║
║                       Version 1.1 (CORRECTED) | 2025-12-30                    ║
║                                                                               ║
║  Assessment conducted by: Senior UX/UI Architect                              ║
║  Framework: Next.js 14 + React 18 + TypeScript                                ║
║  Current Version: qbm-frontendv3                                              ║
║                                                                               ║
║  CORRECTIONS APPLIED from peer review - statistics and capabilities updated  ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## CORRECTIONS LOG (v1.0 → v1.1)

| Original Claim | Corrected Value | Source |
|----------------|-----------------|--------|
| 736,302 semantic relations | **4,460 edges** | semantic_graph_v2.json |
| 15,847 behavioral spans | **6,236 spans** | /stats endpoint |
| 13-component validation | **6-7 major components** | Backend validation structure |
| 5-axis Bouzidani | **11-axis Bouzidani** | canonical_entities.json |
| No planner architecture | **25 question classes, 10+ specialized planners** | legendary_planner.py |
| Missing entity resolution | **720+ Arabic synonym mappings** | canonical_entities.json |
| Benchmark "20 per section" | **Variable: 25/25/25/25/20/20/15/15/15/15** | Actual benchmark results |
| No root morphology | **Arabic triliteral roots for morphological matching** | canonical_entities.json |
| "ML intent classifier" | **Deterministic pattern-based classifier** | intent_classifier.py |

---

## EXECUTIVE SUMMARY

### Overall Frontend Maturity Score: 6.5/10

The QBM frontend is a **functional research platform** with solid foundations but significant gaps between its powerful backend capabilities and what users can actually experience. The interface works well as a prototype but falls short of the "world-class research platform" standard.

### Top 5 Critical Issues (Updated)

| Priority | Issue | Impact |
|----------|-------|--------|
| **CRITICAL** | Legendary Planner System not exposed | 25 question classes invisible |
| **CRITICAL** | 720+ Arabic synonyms not surfaced | Entity resolution hidden |
| **CRITICAL** | No dedicated Annotate page exists | Annotation workflow missing |
| **HIGH** | 11-axis framework shown as 5-axis | Taxonomy incomplete |
| **HIGH** | 200/200 benchmark achievement not shown | Trust indicator hidden |

### Current State vs Potential (CORRECTED)

```
┌─────────────────────────────────────────────────────────────────┐
│ BACKEND POWER                      FRONTEND EXPOSURE             │
├─────────────────────────────────────────────────────────────────┤
│ 25 Legendary Question Classes  →   Not visible                  │
│ 10+ Specialized Planners       →   Not visible                  │
│ 720+ Arabic Synonym Mappings   →   Not visible                  │
│ 4,460 semantic relations       →   ~20 nodes shown (simulated)  │
│ 107,646 vector embeddings      →   Not visible                  │
│ 7 tafsir sources               →   5 shown (2 missing in some)  │
│ 11-axis Bouzidani Framework    →   Only 5 axes shown            │
│ 6,236 behavioral spans         →   Basic stats only             │
│ 73 behavior concepts           →   ~5-6 shown                   │
│ 200/200 benchmark pass         →   Not shown                    │
│ Fail-closed validation         →   Not shown                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## SECTION 1: MISSED BACKEND CAPABILITIES (NEW SECTION)

### 1.1 Legendary Planner Architecture

**What Exists (Not Surfaced):**

The backend has a sophisticated 25-question-class routing system with 10+ specialized planners:

| Category | Question Classes | Planner |
|----------|-----------------|---------|
| **Causal Chain Analysis** | Q1-Q3: Causal Chain, Shortest Path, Reinforcement Loop | CausalChainPlanner |
| **Cross-Tafsir Comparative** | Q4-Q6: Cross-Tafsir, Makki/Madani, Consensus/Dispute | CrossTafsirPlanner |
| **11-Axis + Taxonomy** | Q7-Q9: Behavior Profile, Organ Mapping, State Transition | Profile11DPlanner |
| **Agent-Based Analysis** | Q10-Q12: Agent Attribution, Contrast Matrix, Prophetic Archetype | AgentPlanner |
| **Network + Graph Analytics** | Q13-Q15: Network Centrality, Community Detection, Bridge Behaviors | GraphMetricsPlanner |
| **Temporal + Spatial Context** | Q16-Q17: Temporal Mapping, Spatial Mapping | TemporalSpatialPlanner |
| **Consequence Analysis** | NEW: Consequence Mapping | ConsequencePlanner |
| **Statistics + Patterns** | Q18-Q20: Surah Fingerprints, Frequency, Makki/Madani Shift | StatsPlanner |
| **Embeddings + Semantics** | Q21-Q22: Semantic Landscape, Meaning Drift | EmbeddingsPlanner |
| **Complex Multi-System** | Q23-Q25: Complete Analysis, Prescription Generator, Genome | IntegrationPlanner |

**Frontend Need:**
- **Planner Selection Indicator** - Show which planner handled each query
- **Question Class Display** - Show detected question type
- **Plan Steps Visualization** - Show the execution pipeline

---

### 1.2 Arabic Synonym Resolution System (720+ Mappings)

**What Exists (Not Surfaced):**

The `canonical_entities.json` contains comprehensive Arabic term mappings:

| Entity Type | Count | Arabic Synonyms |
|-------------|-------|-----------------|
| Behaviors | 73 | ~350 verb forms, plurals, participles |
| Organs | 40 | ~160 plurals, possessed forms |
| Agents | 14 | ~70 Quranic forms, phrases |
| Heart States | 12 | ~50 verb forms, phrases |
| Consequences | 16 | ~90 names of Jannah/Jahannam |
| **TOTAL** | **155** | **~720+ synonyms** |

**Example Resolution:**
```
User searches: "المؤمنين" or "الذين آمنوا" or "أهل الإيمان"
All resolve to: AGT_BELIEVER (canonical agent)
```

**Frontend Need:**
- **Entity Resolution Display** - "Your search 'المؤمنين' resolved to: Believer (AGT_BELIEVER)"
- **Synonym Explorer** - Browse all 720+ Arabic terms and their canonical mappings
- **Canonical Entity Badges** - Show resolved entity type on results

---

### 1.2b Arabic Root-Based Morphological Matching

**What Exists (Not Surfaced):**

Each behavior in `canonical_entities.json` includes Arabic triliteral roots:

```json
{"id": "BEH_SPEECH_TRUTHFULNESS", "ar": "الصدق", "roots": ["ص-د-ق"], "synonyms": [...]}
{"id": "BEH_SPI_PRAYER", "ar": "الصلاة", "roots": ["ص-ل-و"], "synonyms": [...]}
{"id": "BEH_SPI_PROSTRATION", "ar": "السجود", "roots": ["س-ج-د"], "synonyms": [...]}
```

This enables **morphological matching** where all derived forms match:
- Root س-ج-د → سجد، يسجد، ساجد، سجود، ساجدين، المسجد all match
- Root ص-ب-ر → صبر، يصبر، صابر، صابرين، الصابرون all match

**Frontend Need:**
- **Root Display** - Show the Arabic root alongside canonical term
- **Morphological Variants** - Display all matched forms
- **Root-based Search** - Allow searching by root pattern

---

### 1.2c Deterministic Intent Classifier

**What Exists (Not Surfaced):**

The system uses a **deterministic pattern-based intent classifier** (`src/ml/intent_classifier.py`), NOT probabilistic ML:

| Intent Type | Benchmark Section | Pattern Examples |
|-------------|-------------------|------------------|
| GRAPH_CAUSAL | A | "causal chain", "trace all chains", "CAUSES" |
| CROSS_TAFSIR_ANALYSIS | B | "compare tafsir", "multi-source" |
| PROFILE_11D | C | "11 axis", "profile", "dimensions" |
| GRAPH_METRICS | D | "centrality", "network", "graph statistics" |
| HEART_STATE | E | "قلب", "heart state", "قسوة" |
| AGENT_ANALYSIS | F | "agent", "فاعل", "مؤمن مقابل كافر" |
| TEMPORAL_SPATIAL | G | "Makki", "Madani", "temporal" |
| CONSEQUENCE_ANALYSIS | H | "عاقبة", "جنة", "نار", "consequence" |
| EMBEDDINGS_ANALYSIS | I | "semantic", "embedding", "similarity" |
| INTEGRATION_E2E | J | "complete analysis", "full profile" |

**Key Differentiator:** Deterministic routing means:
- Same query always routes to same planner
- No randomness or stochastic behavior
- Fully reproducible results

**Frontend Need:**
- **Intent Label Display** - Show detected intent type (e.g., "GRAPH_CAUSAL")
- **Pattern Match Highlight** - Show which words triggered the classification
- **Routing Explanation** - "This query was routed to CausalChainPlanner because..."

---

### 1.3 Complete 11-Axis Bouzidani Framework

**What Exists (Only 5 Shown):**

The full 11-axis framework in `canonical_entities.json`:

| # | Axis (AR) | Axis (EN) | Values |
|---|-----------|-----------|--------|
| 1 | التصنيف العضوي | Organic Classification | External, Internal, Heart, Tongue, Limbs |
| 2 | التصنيف الموقفي | Situational Classification | Self, Horizons, Creator, Universe, Life |
| 3 | التصنيف النسقي | Systemic Classification | Home, Work, Public, Mosque |
| 4 | التصنيف المكاني | Spatial Classification | Sacred, Ordinary, Private, Public |
| 5 | التصنيف الزماني | Temporal Classification | This World, Hereafter, Both |
| 6 | التصنيف الفاعلي | Agent Classification | Individual, Collective, Divine |
| 7 | التصنيف المصدري | Source Classification | Self/Nafs, Satan, Divine Inspiration |
| 8 | التصنيف التقييمي | Evaluation Classification | Obligatory, Recommended, Permissible, Disliked, Forbidden |
| 9 | تأثير القلب | Heart Impact | Purifies, Hardens, Softens, Seals |
| 10 | العاقبة | Consequence | Reward, Punishment, Neutral |
| 11 | العلاقات | Relationships | Causal, Opposite, Strengthening, Conditional |

**Frontend Currently Shows:** Only axes 1-5 (missing 6-11)

**Frontend Need:**
- Full 11-axis radar chart
- Axis-by-axis breakdown in proof results
- Axis filtering in explorer

---

### 1.4 Fail-Closed Validation Architecture

**What Exists (Not Surfaced):**

The backend implements strict fail-closed behavior:
- If entities not found → returns empty, NOT hallucinated data
- If graph has no paths → says so explicitly
- Zero fabrication policy enforced at all levels

**Frontend Need:**
- **"No Fabrication" Badge** - Trust indicator
- **Empty State Messaging** - "No causal paths found" instead of generated fake paths
- **Data Source Transparency** - Show exactly where each claim comes from

---

### 1.5 200/200 Benchmark Achievement (Validation, Not Limitation)

**What Exists (Not Surfaced):**

The system is designed to answer **ANY scholarly question** in the Quranic behavioral domain. The 200-question benchmark was created to **validate** the system's capabilities across 10 categories - it does NOT limit what scholars can ask.

**Benchmark Purpose:** Quality assurance validation
**System Purpose:** Open-ended scholarly research on ANY Quranic behavioral question

The benchmark covers 10 representative categories:

| Section | Category | Test Questions | Pass Rate |
|---------|----------|----------------|-----------|
| A | Causal Chain (GRAPH_CAUSAL) | 25 | 100% |
| B | Cross-Tafsir (CROSS_TAFSIR_ANALYSIS) | 25 | 100% |
| C | 11-Axis Profile (PROFILE_11D) | 25 | 100% |
| D | Graph Metrics (GRAPH_METRICS) | 25 | 100% |
| E | Heart State (HEART_STATE) | 20 | 100% |
| F | Agent Analysis (AGENT_ANALYSIS) | 20 | 100% |
| G | Temporal/Spatial (TEMPORAL_SPATIAL) | 15 | 100% |
| H | Consequence (CONSEQUENCE_ANALYSIS) | 15 | 100% |
| I | Embeddings (EMBEDDINGS_ANALYSIS) | 15 | 100% |
| J | Integration E2E (INTEGRATION_E2E) | 15 | 100% |
| **TOTAL** | | **200** | **100%** |

**What This Means for Scholars:**
- The system handles questions **beyond** the 200 test cases
- Any question about Quranic behaviors, agents, consequences, tafsir comparisons, etc.
- The benchmark proves the system works reliably across diverse query types
- Scholars are NOT limited to asking only benchmark-style questions

**Validation Guarantees:**
- Zero "generic opening verses" defaults
- Zero fabricated statistics
- Zero synthetic graph edges

**Frontend Need:**
- **"Validated Research Platform" Badge** - Trust indicator showing system quality
- **Open Query Interface** - Make clear scholars can ask ANY question
- **Example Query Categories** - Show the 10 types as inspiration, not limitation
- **Validation Methodology Link** - Explain benchmark process for academic credibility

---

## SECTION 2: CORRECTED STATISTICS

### Actual Backend Numbers

| Metric | Previous Claim | Actual Value | Source |
|--------|---------------|--------------|--------|
| Semantic Relations | 736,302 | **4,460 edges** | semantic_graph_v2.json |
| Behavioral Spans | 15,847 | **6,236 spans** | /stats endpoint |
| Tafsir Sources | 5 | **7** | tafsir_sources.json |
| Behavior Concepts | 87 | **73** | canonical_entities.json |
| Agents | ~5 | **14** | canonical_entities.json |
| Organs | ~10 | **40** | canonical_entities.json |
| Heart States | ~5 | **12** | canonical_entities.json |
| Consequences | ~5 | **16** | canonical_entities.json |
| Arabic Synonyms | Not counted | **720+** | canonical_entities.json |
| Question Classes | Not mentioned | **25** | legendary_planner.py |
| Specialized Planners | Not mentioned | **10+** | src/ml/planners/ |

### Edge Type Breakdown (4,460 total)

| Edge Type | Count | Description |
|-----------|-------|-------------|
| COMPLEMENTS | 2,943 | Mutually reinforcing behaviors |
| CAUSES | 584 | Direct causal relationship |
| OPPOSITE_OF | 367 | Antithetical behaviors |
| STRENGTHENS | 318 | One enhances another |
| CONDITIONAL_ON | 119 | Conditional dependency |
| PREVENTS | 106 | One blocks another |
| LEADS_TO | 23 | Sequential progression |

---

## SECTION 3: CORRECTED VALIDATION STRUCTURE

### Previous Claim: "13 Components"
### Actual Structure: 6-7 Major Components

| Component | Sub-components | Description |
|-----------|----------------|-------------|
| **quran_refs** | verses with text | Quranic verse citations |
| **tafsir_refs** | All 7 tafsirs combined | Ibn Kathir, Tabari, Qurtubi, Saadi, Jalalayn, Baghawi, Muyassar |
| **graph_metrics** | nodes, edges, paths | Semantic graph evidence |
| **embeddings** | similarity pairs, clusters | Vector-based similarity |
| **taxonomy** | 11-axis classification | Bouzidani framework values |
| **statistics** | counts, distributions | Quantitative analysis |
| **provenance** | source tracking | Full citation trail |

The frontend was displaying tafsirs as 7 separate "components" - they should be shown as ONE component with 7 sources.

---

## SECTION 4: UPDATED FRONTEND RECOMMENDATIONS

### NEW Recommendations (From Peer Review)

| # | Recommendation | Priority | Effort |
|---|----------------|----------|--------|
| 1 | **Planner Selection Indicator** - Show which of 25 question classes was detected | CRITICAL | LOW |
| 2 | **Entity Resolution Display** - "Your search resolved to: X" | CRITICAL | LOW |
| 3 | **Synonym Explorer Page** - Browse 720+ Arabic terms → canonical mappings | HIGH | MEDIUM |
| 4 | **200/200 Benchmark Badge** - Trust indicator prominently displayed | HIGH | LOW |
| 5 | **Fail-Closed Indicator** - "No fabricated data" trust badge | HIGH | LOW |
| 6 | **11-Axis Full Radar Chart** - Replace 5-axis with complete 11-axis | HIGH | MEDIUM |
| 7 | **Plan Steps Visualization** - Show execution pipeline for each query | MEDIUM | MEDIUM |
| 8 | **Question Class Label** - Display detected question type | MEDIUM | LOW |

### Updated Priority Matrix

```
┌────────────────────────────────────────────────────────────────┐
│                    PRIORITY MATRIX (UPDATED)                   │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  QUICK WINS (Low Effort, High Impact)                          │
│  ────────────────────────────────────                          │
│  • Add Planner Selection Indicator                             │
│  • Add Entity Resolution Display                               │
│  • Add 200/200 Benchmark Badge                                 │
│  • Add Fail-Closed Indicator                                   │
│  • Fix 5-axis → 11-axis in Taxonomy page                       │
│  • Add 2 missing tafsirs to Research page                      │
│                                                                │
│  MEDIUM EFFORT                                                 │
│  ────────────                                                  │
│  • Create Synonym Explorer page                                │
│  • Create Annotate page                                        │
│  • Plan Steps Visualization                                    │
│  • Full 11-axis radar chart                                    │
│                                                                │
│  HIGH EFFORT                                                   │
│  ───────────                                                   │
│  • Connect real graph data (4,460 edges)                       │
│  • Embedding cluster visualization                             │
│  • PDF/BibTeX export                                           │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## SECTION 5: PAGE-BY-PAGE AUDIT (Retained with Corrections)

### 5.1 Home Page (`/`)

**Critical Updates Needed:**
- Change stats bar to show **4,460** relations (not 736K)
- Add "**200/200 Validated**" trust badge
- Add "**720+ Arabic Synonyms**" metric
- Add "**25 Question Classes**" metric

### 5.2 Research Page (`/research`)

**Critical Updates Needed:**
- Add all **7 tafsir sources** (currently 5)
- Add **Planner Selection Indicator**
- Add **Entity Resolution Display**
- Add link to full proof with **11-axis** breakdown

### 5.3 Proof Page (`/proof`)

**Already Correct:**
- Shows 7 tafsir sources
- Shows validation percentage

**Needs Update:**
- Replace current validation display with **6-7 component structure**
- Add **11-axis taxonomy** (currently 11 dimensions mentioned but not all shown)
- Add **Question Class Label**
- Add **Plan Steps**

### 5.4 Taxonomy Page (`/taxonomy`)

**Critical Error:**
- Currently shows **5 axes** (Organic, Situational, Systemic, Temporal, Evaluation)
- Missing axes 4, 6, 7, 9, 10, 11

**Required Fix:**
- Expand to full **11-axis framework**
- Add real data from annotations (currently uses `Math.random()`)

### 5.5 Explorer/Discovery Pages

**Critical Issue:**
- Graph data is **simulated** using `buildGraphFromStats()`
- Should connect to actual **4,460 edges** from semantic_graph_v2.json

---

## SECTION 6: REVISED IMPLEMENTATION ROADMAP

### Phase 1: Trust Indicators & Quick Fixes (Week 1)

| Task | Effort | Files to Change |
|------|--------|-----------------|
| Add 200/200 Benchmark Badge | LOW | Home, Research, Proof pages |
| Add Fail-Closed Indicator | LOW | Proof page |
| Add Planner Selection Display | LOW | Research, Proof pages |
| Add Entity Resolution Display | LOW | Research page |
| Fix tafsir count (5→7) in Research | LOW | Research page |
| Update stats (736K→4,460) | LOW | All pages with stats |

### Phase 2: 11-Axis & Synonym Explorer (Week 2)

| Task | Effort | Files to Change |
|------|--------|-----------------|
| Expand Taxonomy to 11 axes | MEDIUM | Taxonomy page |
| Create Synonym Explorer page | MEDIUM | New page: /synonyms |
| Add canonical entity badges | LOW | Search results components |
| Connect radar chart to real data | MEDIUM | Taxonomy page |

### Phase 3: Annotate Page (Week 3)

| Task | Effort | Files to Change |
|------|--------|-----------------|
| Create /annotate page | HIGH | New page |
| Integrate 7 tafsirs | MEDIUM | Tafsir sidebar component |
| Add 11-axis selector | MEDIUM | Annotation form |
| Add behavior/agent pickers | MEDIUM | Entity selection components |

### Phase 4: Real Graph & Exports (Week 4)

| Task | Effort | Files to Change |
|------|--------|-----------------|
| Connect ForceGraph to 4,460 edges | HIGH | Explorer, Discovery pages |
| Add edge type coloring | MEDIUM | Graph component |
| Add PDF export | HIGH | Proof page |
| Add BibTeX export | MEDIUM | Proof page |

---

## APPENDIX A: Technical Corrections

| Original Claim | Correction | Evidence |
|----------------|------------|----------|
| Graph uses 736K relations | Graph has 4,460 edges in 7 types | semantic_graph_v2.json lines 789-797 |
| Frontend shows 13 components | Should show 6-7 grouped components | Backend validation structure |
| Taxonomy shows 5 axes | Should show 11 axes | canonical_entities.json axes_11 |
| No planner info | 25 question classes with 10+ planners | legendary_planner.py |
| No entity resolution | 720+ Arabic synonyms with resolution | canonical_entities.json behaviors[].synonyms |

## APPENDIX B: Files Containing Correct Data

| Data | File | Key |
|------|------|-----|
| Edge counts | data/graph/semantic_graph_v2.json | edges_by_type |
| 11 axes | vocab/canonical_entities.json | axes_11 |
| Behaviors + synonyms | vocab/canonical_entities.json | behaviors[].synonyms |
| Question classes | src/ml/legendary_planner.py | QuestionClass enum |
| Specialized planners | src/ml/planners/*.py | 10 planner files |
| Benchmark results | reports/benchmarks/*.json | 200/200 PASS |

---

## CONCLUSION (Updated)

The original assessment correctly identified that the frontend under-represents backend power, but significantly overstated some numbers while missing critical capabilities:

**What Was Over-Stated:**
- 736K relations → Actually 4,460 edges
- 15,847 spans → Actually 6,236 spans
- 13 components → Actually 6-7 major components

**What Was Under-Stated/Missed:**
- 25-class Legendary Planner system (not mentioned)
- 720+ Arabic synonym mappings (not mentioned)
- 11-axis Bouzidani framework (showed only 5)
- 200/200 benchmark achievement (not mentioned)
- Fail-closed validation architecture (not mentioned)
- 10+ specialized planners (not mentioned)

**The Most Critical Action Items Now:**

1. **Add trust indicators** - 200/200 badge, fail-closed badge
2. **Expose the planner system** - Show which of 25 classes was detected
3. **Add entity resolution display** - Show Arabic → canonical mapping
4. **Expand to 11 axes** - Complete Bouzidani framework
5. **Create Synonym Explorer** - Let users browse 720+ Arabic terms
6. **Create Annotate page** - Still missing entirely
7. **Connect real graph** - Use actual 4,460 edges

---

**Assessment v1.1 completed: December 30, 2025**
**Corrections applied from peer review**
**Ready for implementation**
