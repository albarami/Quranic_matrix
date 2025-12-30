# QBM FRONTEND COMPREHENSIVE ASSESSMENT
## Version 1.0 | December 30, 2025

---

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    QBM FRONTEND COMPREHENSIVE ASSESSMENT                      ║
║                           Version 1.0 | 2025-12-30                            ║
║                                                                               ║
║  Assessment conducted by: Senior UX/UI Architect                              ║
║  Framework: Next.js 14 + React 18 + TypeScript                                ║
║  Current Version: qbm-frontendv3                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## EXECUTIVE SUMMARY

### Overall Frontend Maturity Score: 6.5/10

The QBM frontend is a **functional research platform** with solid foundations but significant gaps between its powerful backend capabilities and what users can actually experience. The interface works well as a prototype but falls short of the "world-class research platform" standard required for Islamic scholars and academic researchers.

### Top 5 Critical Issues

| Priority | Issue | Impact |
|----------|-------|--------|
| **CRITICAL** | No dedicated Annotate page exists | Annotation workflow entirely missing |
| **CRITICAL** | 736K graph relations not visualized | Backend power invisible to users |
| **HIGH** | Research vs Proof overlap | Confusing user journey |
| **HIGH** | Output format too casual | Not publication-ready |
| **HIGH** | 107K embeddings not surfaced | Semantic clustering hidden |

### Current State vs Potential

```
┌─────────────────────────────────────────────────────────────────┐
│ BACKEND POWER                      FRONTEND EXPOSURE             │
├─────────────────────────────────────────────────────────────────┤
│ 736,302 semantic relations    →    ~20 nodes shown (simulated)  │
│ 107,646 vector embeddings     →    Not visible                  │
│ 7 tafsir sources              →    5 shown (2 missing)          │
│ 13-component validation       →    Partially exposed            │
│ 15,847 behavioral spans       →    Basic stats only             │
│ 87 behavior concepts          →    ~5-6 shown                   │
│ 100% validation rate          →    Shown but not celebrated     │
└─────────────────────────────────────────────────────────────────┘
```

---

## SECTION 1: PAGE-BY-PAGE AUDIT

### 1.1 Home Page (`/`)

**Current State:**
- Feature-rich landing with animated counters
- Featured verse display (An-Nahl 16:97) with decorative brackets
- Statistics bar with 4 KPIs (Ayat, Spans, Tafsir Sources, Behavior Forms)
- AI-discovered insights section with 4 behavior categories
- Live research demo with 3 tabs (chart, table, comparison)
- Features grid linking to main pages
- Full footer with navigation

**UX/UI Issues:**
| Issue | Severity | Description |
|-------|----------|-------------|
| Tafsir count shows 5, not 7 | HIGH | Misleading information |
| Example queries hardcoded | MEDIUM | Should show real top queries |
| Hero verse static | LOW | Could rotate featured verses |
| Statistics bar not showing 736K relations | HIGH | Missing key metric |
| No mention of 100% validation | HIGH | Key differentiator hidden |

**Enhancement Opportunities:**
1. Add "100% Validated Responses" badge prominently
2. Display "736K+ Semantic Relations" in stats bar
3. Add "107K Vector Embeddings" metric
4. Rotate featured verses from high-impact annotations
5. Show real-time annotation activity feed

**Priority:** HIGH

---

### 1.2 Research Page (`/research`)

**Current State:**
- Chat interface with streaming support
- Example queries organized by category (4 categories)
- Proof result display with KPI cards (Quran verses, Tafsir citations, Behavior nodes, Validation %)
- Quran verses section with expandable list
- Tafsir tabs interface (5 sources shown)
- Graph nodes display (tags only, no visualization)
- Metrics truth layer integration for deterministic stats

**UX/UI Issues:**
| Issue | Severity | Description |
|-------|----------|-------------|
| Only 5 tafsir tabs | HIGH | Missing Baghawi and Muyassar |
| Graph section shows tags, not graph | HIGH | No actual graph visualization |
| No embedding/similarity display | HIGH | Vector features invisible |
| Output looks like chat, not scholarly | MEDIUM | Too casual for academics |
| No citation export | HIGH | Cannot use in papers |
| Markdown rendering basic | MEDIUM | Not publication-quality |

**Enhancement Opportunities:**
1. Add all 7 tafsir sources
2. Integrate actual graph visualization (force-directed)
3. Add embedding similarity display with semantic clusters
4. Add "Export Citation" button (BibTeX, APA)
5. Add "View Full Proof" link to dedicated proof page
6. Professional response formatting with proper Arabic typography

**Priority:** CRITICAL

---

### 1.3 Explorer Page (`/explorer`)

**Current State:**
- Surah grid view (114 surahs as colored cells)
- List view with coverage bars
- Graph view with ForceGraph2D (behavior-agent relationships)
- Behavior filter chips
- Surah detail panel with charts
- Stats bar (Total Surahs, Annotated Ayat, Behavioral Spans, Avg Coverage)

**UX/UI Issues:**
| Issue | Severity | Description |
|-------|----------|-------------|
| Graph data simulated | HIGH | Not actual 736K relations |
| Surah grid small cells | MEDIUM | Hard to read on mobile |
| Coverage colors need legend | LOW | Already present but small |
| No ayah-level navigation | MEDIUM | Can only see surah level |
| Tafsir not integrated | MEDIUM | Cannot compare tafsir per surah |

**Enhancement Opportunities:**
1. Connect to actual graph database for real relations
2. Add ayah-level drill-down within surahs
3. Integrate tafsir comparison in surah detail
4. Add "View in Proof System" action
5. Show annotation density heatmap

**Priority:** MEDIUM

---

### 1.4 Dashboard Page (`/dashboard`)

**Current State:**
- Header with refresh and export buttons
- Quick stats cards (Coverage, Spans, Surahs, Ayat)
- Coverage progress circle
- Behavior distribution bar chart
- Agent analysis pie chart + legend
- Top surahs bar chart
- Recent activity table

**UX/UI Issues:**
| Issue | Severity | Description |
|-------|----------|-------------|
| Export button non-functional | MEDIUM | Just UI, no action |
| No graph/embedding metrics | HIGH | Backend power hidden |
| Recent activity limited | LOW | Only 5 items shown |
| No filtering options | MEDIUM | Cannot filter by date/type |
| Charts basic | LOW | Recharts defaults, not custom |

**Enhancement Opportunities:**
1. Add "736K Relations" card with graph icon
2. Add "107K Vectors" card with embedding icon
3. Implement actual export (CSV, PDF)
4. Add date range filtering
5. Add annotation timeline visualization
6. Add validation score trend chart

**Priority:** MEDIUM

---

### 1.5 Proof Page (`/proof`)

**Current State:**
- RTL-first Arabic interface
- Header showing system capabilities (107,646 vectors, 736,302 relations, 7 tafsir)
- Query input with Arabic placeholder
- Validation badge with percentage and component status
- Collapsible sections:
  - Answer (Main response)
  - Quran Evidence (with relevance scores)
  - Tafsir (7 tabs including Baghawi and Muyassar)
  - Graph Evidence (nodes, edges, paths counts + path visualization)
  - Taxonomy (11 dimensions + behaviors)
  - Statistics (counts grid)
  - RAG Retrieval (sources breakdown)
  - Embeddings (similarity pairs)
- Example queries grid

**UX/UI Issues:**
| Issue | Severity | Description |
|-------|----------|-------------|
| No visual graph | HIGH | Just counts, no interactive graph |
| Embedding display minimal | HIGH | Just similarity pairs, no clusters |
| 13 components not all explicit | MEDIUM | Some merged/hidden |
| Citation format informal | HIGH | Not APA/Chicago compliant |
| No export functionality | CRITICAL | Cannot export proof for papers |
| PDF generation missing | CRITICAL | Academic necessity |

**Enhancement Opportunities:**
1. Add interactive graph visualization
2. Add embedding cluster visualization (t-SNE/UMAP)
3. Explicit 13-component checklist with details
4. Add "Export as PDF" with academic formatting
5. Add "Copy Citation" for each evidence piece
6. Add page numbers for tafsir references
7. Formal document structure (sections, numbering)

**Priority:** CRITICAL

---

### 1.6 Annotate Page (`/annotate`)

**Current State:**
⚠️ **THIS PAGE DOES NOT EXIST**

The navigation does not include an Annotate link, and no `/annotate/page.tsx` file exists in the codebase.

**Critical Gap:**
- No annotation workflow interface
- No tafsir integration for annotation
- No span creation/editing
- No quality tier assignment
- No inter-annotator agreement display

**Required Features:**
1. Annotation workspace with verse display
2. Tafsir sidebar (all 7 sources)
3. Behavior selection interface
4. Agent type assignment
5. Evaluation (praise/blame/neutral) selector
6. Quality tier indicator
7. Annotation history/audit trail
8. Batch annotation support

**Priority:** CRITICAL (Must Create)

---

### 1.7 Insights Page (`/insights`)

**Current State:**
- Header with AI-Powered Discovery badge
- Quick stats (Total Spans, Behavior Forms, Insights Generated, Data Source)
- Insight cards (5 featured discoveries)
- Detail panel with chart visualization
- Methodology note

**UX/UI Issues:**
| Issue | Severity | Description |
|-------|----------|-------------|
| Insights hardcoded from stats | MEDIUM | Not truly "AI-discovered" |
| Charts basic | LOW | Standard Recharts |
| No drill-down to proof | MEDIUM | Link exists but shallow |
| Limited to 5 insights | LOW | Could show more patterns |

**Enhancement Opportunities:**
1. Generate insights dynamically from backend analysis
2. Add "pattern discovery" algorithm results
3. Show co-occurrence matrices
4. Add temporal patterns (Meccan vs Medinan)
5. Link each insight to full proof query

**Priority:** LOW

---

### 1.8 Taxonomy Page (`/taxonomy`)

**Current State:**
- Bouzidani's 5-Axis Framework display
- Hero section with framework description
- Axis quick stats (5 buttons)
- Radar chart overview
- Axis cards with expandable categories
- Examples for each category
- Methodology section
- Stats footer

**UX/UI Issues:**
| Issue | Severity | Description |
|-------|----------|-------------|
| Radar chart uses simulated data | HIGH | Not from actual annotations |
| Categories static | MEDIUM | Should reflect real data |
| Examples hardcoded | MEDIUM | Should be from dataset |
| No annotation counts per category | HIGH | Missing key metric |

**Enhancement Opportunities:**
1. Connect radar chart to actual annotation distribution
2. Show annotation counts per axis/category
3. Add "Explore in Proof" for each category
4. Add cross-axis correlation visualization
5. Show top behaviors per category

**Priority:** MEDIUM

---

### 1.9 Discovery Page (`/discovery`)

**Current State:**
- Semantic search interface
- Quick search tags (Arabic/English)
- Stats cards (Ayat, Spans, Behavior Types, Tafsir Sources)
- View toggle (Overview/Graph)
- Behavior distribution bar chart
- Agent types pie chart
- ForceGraph2D visualization
- Search results display with source highlighting

**UX/UI Issues:**
| Issue | Severity | Description |
|-------|----------|-------------|
| Graph data simulated | HIGH | Not actual relations |
| Link types labeled as CO_OCCURRENCE | MEDIUM | Real types (causal, temporal) not shown |
| Search limited to proof endpoint | MEDIUM | No dedicated semantic search |
| Results not ranked | LOW | No relevance sorting |

**Enhancement Opportunities:**
1. Connect to actual graph database
2. Show real edge types (causal, temporal, thematic)
3. Add semantic similarity clustering
4. Add faceted search (by surah, behavior, agent)
5. Add "Save Search" functionality

**Priority:** MEDIUM

---

### 1.10 Genome Page (`/genome`)

**Current State:**
- Q25 canonical entity registry export
- Status display (version, statistics)
- Entity counts (behaviors, agents, organs, heart states, consequences)
- Semantic edges count
- Export functionality

**Assessment:** Functional but utilitarian. Serves its purpose for data export.

**Priority:** LOW

---

### 1.11 Reviews Page (`/reviews`)

**Current State:**
- Annotation review and quality control interface
- Review status filtering
- Review list with metadata
- Status update functionality

**Assessment:** Basic functionality present. Needs workflow improvements.

**Priority:** MEDIUM

---

## SECTION 2: SYSTEM POWER VISUALIZATION

### 2.1 Graph & Connection Visualization

**Current Exposure:**

| Capability | Backend | Frontend Display |
|------------|---------|------------------|
| 736,302 semantic relations | Available | **NOT SHOWN** - Only simulated ~20-30 node graphs |
| Relationship types | Stored (causal, temporal, thematic) | **NOT SHOWN** - Links unlabeled |
| Multi-hop paths | Computed | Shown as text list only |
| Node types | 5+ types | Only behavior/agent shown |

**Recommendation:**

The graph visualization is the **single biggest gap** in the frontend. The current ForceGraph2D implementation uses **simulated data** based on behavior/agent counts, NOT actual semantic relations from the graph database.

**Required Visualization Approach:**
1. **Primary View:** Force-directed graph with 100-500 nodes (paginated)
2. **Node coloring:** By type (behavior=green, agent=blue, surah=purple, etc.)
3. **Edge coloring:** By relationship type (causal=red, temporal=orange, thematic=blue)
4. **Interactive features:**
   - Click node → Show connected nodes
   - Hover edge → Show relationship details
   - Filter by relationship type
   - Search within graph
5. **Path visualization:** Animated traversal showing multi-hop connections

---

### 2.2 Embedding & Similarity Display

**Current Exposure:**

| Capability | Backend | Frontend Display |
|------------|---------|------------------|
| 107,646 vector embeddings | Available | **NOT SHOWN** |
| Semantic clustering | Computed | **NOT SHOWN** |
| Similarity scores | Available | Shown minimally in Proof page |
| Nearest neighbors | Available | **NOT SHOWN** |

**Recommendation:**

Vector embeddings should be visualized as:
1. **Cluster map:** 2D projection (t-SNE/UMAP) showing semantic groupings
2. **Similarity heatmap:** For selected concepts
3. **Nearest neighbors panel:** "Related concepts" for any behavior/ayah
4. **Semantic distance indicator:** Visual proximity on queries

---

### 2.3 13-Component Proof System

**Current Exposure:**

The Proof page shows validation percentage and some component checks, but the full 13 components are not explicitly enumerated or individually accessible.

**The 13 Components (Should Be Displayed):**

| # | Component | Current Status |
|---|-----------|----------------|
| 1 | Quran verses | ✅ Shown |
| 2 | Ibn Kathir tafsir | ✅ Shown |
| 3 | Tabari tafsir | ✅ Shown |
| 4 | Qurtubi tafsir | ✅ Shown |
| 5 | Saadi tafsir | ✅ Shown |
| 6 | Jalalayn tafsir | ✅ Shown |
| 7 | Baghawi tafsir | ⚠️ In Proof, not Research |
| 8 | Muyassar tafsir | ⚠️ In Proof, not Research |
| 9 | Graph evidence | ⚠️ Counts only, no visual |
| 10 | Embeddings/similarity | ⚠️ Minimal display |
| 11 | Taxonomy classification | ✅ Shown |
| 12 | RAG retrieval | ✅ Shown |
| 13 | Statistics | ✅ Shown |

**Recommendation:**

Create an explicit **13-Component Verification Panel** showing:
- Each component with check/cross status
- Individual component drill-down
- Component contribution to final score
- Methodology explanation per component

---

## SECTION 3: PROFESSIONAL STYLING GAPS

### 3.1 Typography Assessment

| Aspect | Current State | Professional Standard | Gap |
|--------|---------------|----------------------|-----|
| Arabic body font | Amiri/Scheherazade | ✅ Appropriate | None |
| Quranic text | Scheherazade New | ✅ Good | None |
| Latin text | Inter | ✅ Clean | None |
| Tashkeel support | Basic | Full diacritics | MEDIUM |
| Calligraphic hierarchy | None | Surah names in thuluth style | HIGH |
| Verse brackets | CSS pseudo-elements | Proper ﴿ ﴾ with decorative styling | MEDIUM |

### 3.2 Citation Format

| Aspect | Current State | Academic Standard | Gap |
|--------|---------------|-------------------|-----|
| Quran citation | "Surah X:Y" | "Quran X:Y (Translation)" | MEDIUM |
| Tafsir citation | Source name only | "Ibn Kathir, Tafsir al-Quran al-Azim, Vol. X, p. Y" | CRITICAL |
| Page numbers | Not shown | Required for verification | CRITICAL |
| Volume numbers | Not shown | Required for tafsir | HIGH |
| Edition/publisher | Not shown | Required for academic use | HIGH |

### 3.3 Export Capabilities

| Format | Current | Required | Priority |
|--------|---------|----------|----------|
| PDF | ❌ None | Academic-formatted document | CRITICAL |
| Word/DOCX | ❌ None | Editable document | HIGH |
| BibTeX | ❌ None | Citation management | CRITICAL |
| Plain text | ❌ None | Copy-paste | MEDIUM |
| JSON | ✅ API response | Structured data | LOW |

### 3.4 Table Design

| Aspect | Current State | Professional Standard |
|--------|---------------|----------------------|
| Headers | Basic gray background | Distinct hierarchy with borders |
| Row striping | Inconsistent | Alternating colors |
| Cell padding | Adequate | Slightly tight for Arabic |
| Sorting | Not available | Should be sortable |
| Pagination | Some pages | Should be consistent |

---

## SECTION 4: RESEARCH vs PROOF STRATEGY

### 4.1 Current State

Both pages serve similar purposes with significant overlap:

```
┌─────────────────────────────────────────────────────────────────┐
│                        CURRENT OVERLAP                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  RESEARCH PAGE                    PROOF PAGE                    │
│  ─────────────                    ──────────                    │
│  • Chat interface                 • Form-based query            │
│  • Proof results display          • Proof results display       │
│  • KPI cards                      • Validation badge            │
│  • Quran verses                   • Quran verses                │
│  • Tafsir tabs (5)                • Tafsir tabs (7)            │
│  • Graph nodes (tags)             • Graph evidence (counts)     │
│  • Link to proof page             • Self-contained              │
│                                                                 │
│  UNIQUE: Metrics truth layer      UNIQUE: 13-component focus    │
│  UNIQUE: Example queries          UNIQUE: Collapsible sections  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Recommended Differentiation

**RESEARCH PAGE - Discovery & Exploration**

Purpose: Interactive discovery, visual insight generation, conversational exploration

**Should Include:**
- Chat-style interface with C1 generative UI
- Auto-generated visualizations (charts, graphs, clusters)
- Interactive concept maps
- Quick insights and pattern discovery
- Conversational follow-up queries
- Example queries with visual previews
- "Quick answers" optimized for discovery

**Should NOT Include:**
- Full 13-component proof breakdown
- Academic citation details
- Export functionality
- Methodology explanations

**Tone:** Accessible, engaging, exploration-focused

---

**PROOF PAGE - Academic Verification**

Purpose: Rigorous scholarly verification, complete evidence chain, publication-ready output

**Should Include:**
- Formal query interface (not chat)
- Explicit 13-component verification display
- All 7 tafsir sources with full citations
- Page numbers and volume references
- Interactive graph visualization
- Embedding similarity display
- Validation methodology explanation
- Export options (PDF, BibTeX, Word)
- Academic formatting throughout

**Should NOT Include:**
- Chat conversation history
- Casual language
- Auto-generated visualizations without citations

**Tone:** Rigorous, comprehensive, publication-ready

---

### 4.3 User Journey

```
┌─────────────────────────────────────────────────────────────────┐
│                    RECOMMENDED USER JOURNEY                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  HOME → RESEARCH (Discovery)                                    │
│           │                                                     │
│           ├─→ Quick insights, patterns                          │
│           │                                                     │
│           ├─→ "Want full proof?" → PROOF (Verification)         │
│           │                           │                         │
│           │                           ├─→ 13-component evidence │
│           │                           ├─→ Export for paper      │
│           │                           └─→ Citation details      │
│           │                                                     │
│           ├─→ "Explore more?" → EXPLORER                        │
│           │                                                     │
│           └─→ "See patterns?" → INSIGHTS                        │
│                                                                 │
│  ANNOTATE → RESEARCH (Validate annotation)                      │
│          → PROOF (Get evidence for annotation)                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.4 Decision: Do We Need Both Pages?

**YES** - Both pages serve distinct purposes when properly differentiated:

| Aspect | Research | Proof |
|--------|----------|-------|
| Primary user | Explorer, student | Scholar, researcher |
| Output format | Visual, interactive | Textual, citable |
| Depth | Breadth-first | Depth-first |
| Session length | Quick (2-5 min) | Extended (10-30 min) |
| Export need | Low | High |
| Citation need | Low | High |

---

## SECTION 5: COMPONENT-LEVEL REVIEW

### 5.1 Navigation Component

| Aspect | Status | Notes |
|--------|--------|-------|
| Sticky header | ✅ Good | Stays visible on scroll |
| Mobile menu | ✅ Good | Hamburger with full nav |
| Language toggle | ✅ Good | AR/EN switching works |
| Active state | ✅ Good | Clear visual indicator |
| Missing link | ⚠️ | No Annotate, Taxonomy, Metrics links |
| Page count | ⚠️ | 8 items shown, some pages hidden |

### 5.2 Chat/C1Chat Interface

| Aspect | Status | Notes |
|--------|--------|-------|
| Streaming | ❓ Not tested | Code supports streaming |
| Response format | ⚠️ Mixed | Structured cards + some markdown |
| Scholarly tone | ⚠️ Casual | Could be more formal |
| Error handling | ✅ Good | Shows error messages |

### 5.3 Tafsir Tabs

| Aspect | Status | Notes |
|--------|--------|-------|
| Sources present | ⚠️ 5 in Research, 7 in Proof | Inconsistent |
| Comparison ease | ⚠️ One at a time | Side-by-side would be better |
| Text display | ✅ Good | Arabic with proper font |
| Citation | ❌ Poor | No page/volume numbers |

### 5.4 Surah Grid

| Aspect | Status | Notes |
|--------|--------|-------|
| Coverage visualization | ✅ Good | Color gradient clear |
| Tooltip | ✅ Good | Shows name and coverage |
| Click action | ✅ Good | Opens detail panel |
| Accessibility | ⚠️ | Cells very small |

### 5.5 Statistics Cards

| Aspect | Status | Notes |
|--------|--------|-------|
| Metrics meaningful | ⚠️ | Missing graph/embedding counts |
| Hierarchy clear | ✅ Good | Primary/secondary clear |
| Animation | ✅ Good | Animated counters |
| Responsiveness | ✅ Good | Grid adjusts on mobile |

### 5.6 Graph Visualization

| Aspect | Status | Notes |
|--------|--------|-------|
| Exists | ⚠️ | Only in Explorer/Discovery |
| Data source | ❌ | Simulated, not real |
| Interactive | ✅ | Click, drag, zoom work |
| Edge labels | ❌ | No relationship types shown |

### 5.7 Arabic Rendering

| Aspect | Status | Notes |
|--------|--------|-------|
| Tashkeel | ⚠️ | Displayed but not emphasized |
| Font choice | ✅ | Appropriate scholarly fonts |
| RTL support | ✅ | Works throughout |
| Line height | ✅ | Good for readability |

### 5.8 Proof Components

| Aspect | Status | Notes |
|--------|--------|-------|
| 13 components accessible | ⚠️ | Not all explicit |
| Validation score | ✅ | Prominently displayed |
| Drill-down | ⚠️ | Limited expansion |
| Methodology | ⚠️ | Mentioned, not explained |

---

## SECTION 6: PRIORITIZED RECOMMENDATIONS

### Tier 1: CRITICAL (Must Fix)

| # | Recommendation | Effort | Impact |
|---|----------------|--------|--------|
| 1 | **Create Annotate Page** | HIGH | Enables annotation workflow |
| 2 | **Connect real graph data** | HIGH | Shows 736K relations |
| 3 | **Add 7 tafsir sources everywhere** | LOW | Consistency |
| 4 | **Add PDF export** | MEDIUM | Academic necessity |
| 5 | **Add BibTeX export** | LOW | Citation management |

### Tier 2: HIGH (Should Fix)

| # | Recommendation | Effort | Impact |
|---|----------------|--------|--------|
| 6 | **Differentiate Research/Proof** | MEDIUM | Clear user journey |
| 7 | **Add embedding visualization** | HIGH | Shows 107K vectors |
| 8 | **Explicit 13-component panel** | MEDIUM | Validation visibility |
| 9 | **Academic citation format** | LOW | Professional output |
| 10 | **Graph edge type coloring** | LOW | Relationship clarity |

### Tier 3: MEDIUM (Nice to Have)

| # | Recommendation | Effort | Impact |
|---|----------------|--------|--------|
| 11 | **Calligraphic Arabic headers** | MEDIUM | Islamic aesthetic |
| 12 | **Tafsir side-by-side comparison** | MEDIUM | Better analysis |
| 13 | **Semantic cluster visualization** | HIGH | Pattern discovery |
| 14 | **Timeline view** | MEDIUM | Temporal patterns |
| 15 | **Advanced search filters** | MEDIUM | Better navigation |

### Tier 4: LOW (Future Enhancement)

| # | Recommendation | Effort | Impact |
|---|----------------|--------|--------|
| 16 | **Word/DOCX export** | MEDIUM | Additional format |
| 17 | **Annotation leaderboard** | LOW | Gamification |
| 18 | **Dark mode** | LOW | User preference |
| 19 | **Print stylesheet** | LOW | Physical printing |
| 20 | **Offline support** | HIGH | PWA capability |

---

## SECTION 7: IMPLEMENTATION ROADMAP

### Phase 1: Foundation Fixes (Quick Wins)

**Objectives:**
- Fix inconsistencies
- Add missing exports
- Improve professional appearance

**Tasks:**
1. Add missing 2 tafsir sources to Research page
2. Add BibTeX export button
3. Add academic citation formatting
4. Update stats to show graph/embedding counts
5. Add page/volume numbers to tafsir citations

**Deliverable:** Consistent, citable output

---

### Phase 2: Annotate Page Creation

**Objectives:**
- Create full annotation workflow
- Integrate tafsir sources
- Enable quality control

**Tasks:**
1. Create `/annotate/page.tsx`
2. Design annotation workspace
3. Integrate 7 tafsir sources as sidebar
4. Add behavior/agent selection UI
5. Add evaluation (praise/blame/neutral)
6. Add quality tier assignment
7. Add annotation history

**Deliverable:** Complete annotation interface

---

### Phase 3: Graph Visualization Overhaul

**Objectives:**
- Connect to real graph database
- Show actual 736K relations
- Enable exploration

**Tasks:**
1. Create graph API endpoint
2. Implement paginated graph loading
3. Color nodes by type
4. Color edges by relationship type
5. Add path animation
6. Add node search
7. Add filter by relationship type

**Deliverable:** Interactive graph exploration

---

### Phase 4: Embedding Visualization

**Objectives:**
- Surface 107K vectors
- Show semantic clustering
- Enable similarity exploration

**Tasks:**
1. Add t-SNE/UMAP projection endpoint
2. Create cluster visualization component
3. Add similarity heatmap
4. Add "Related concepts" panel
5. Integrate into Proof page

**Deliverable:** Semantic similarity visualization

---

### Phase 5: Research/Proof Differentiation

**Objectives:**
- Clear user journey
- Distinct purposes
- Appropriate depth

**Tasks:**
1. Redesign Research page for discovery
2. Redesign Proof page for verification
3. Add clear navigation between them
4. Add "See Full Proof" CTA in Research
5. Add "Quick Explore" CTA in Proof

**Deliverable:** Differentiated user experience

---

### Phase 6: Export & Academic Features

**Objectives:**
- Publication-ready output
- Academic compliance
- Citation management

**Tasks:**
1. Add PDF generation (react-pdf or server-side)
2. Add Word export
3. Add styled print view
4. Add methodology explanation page
5. Add citation guide

**Deliverable:** Academic-ready platform

---

## APPENDIX A: Technical Debt

| Item | Location | Description |
|------|----------|-------------|
| Simulated graph data | Explorer, Discovery | `buildGraphFromStats()` uses random weights |
| Unused Zustand | package.json | Installed but not used |
| Stats endpoint 501 | `/api/stats/route.ts` | Returns "Not implemented" |
| Spans endpoint 501 | `/api/spans/recent/route.ts` | Returns "Not implemented" |
| Hardcoded radar data | Taxonomy | Uses `Math.random()` |

## APPENDIX B: Missing Pages

| Page | Route | Status |
|------|-------|--------|
| Annotate | `/annotate` | **DOES NOT EXIST** |
| Metrics | `/metrics` | File exists but not in nav |
| Behavior Profile | `/behavior-profile` | File exists but not in nav |

## APPENDIX C: Backend Dependencies

The frontend requires these backend endpoints:

| Endpoint | Purpose | Status |
|----------|---------|--------|
| `/stats` | Overview statistics | ✅ Used |
| `/surahs` | Surah list | ✅ Used |
| `/spans/recent` | Recent annotations | ✅ Used |
| `/tafsir/compare/{surah}/{ayah}` | Tafsir comparison | ✅ Used |
| `/api/proof/query` | Proof system | ✅ Used |
| `/api/metrics/overview` | Truth metrics | ✅ Used |
| `/api/genome/status` | Genome registry | ✅ Used |
| `/api/reviews` | Review management | ✅ Used |
| `/api/graph/query` | Graph exploration | ❌ **NEEDED** |
| `/api/embeddings/cluster` | Vector clustering | ❌ **NEEDED** |

---

## CONCLUSION

The QBM frontend is a **solid foundation** with excellent bilingual support and modern architecture, but it significantly **under-represents** the power of the backend system. The 736K semantic relations and 107K vector embeddings - the platform's key differentiators - are essentially invisible to users.

**The most critical action items are:**

1. **Create the Annotate page** - Currently missing entirely
2. **Connect real graph data** - Stop using simulated data
3. **Surface embedding visualizations** - Show semantic clustering
4. **Add academic exports** - PDF and BibTeX are non-negotiable for scholars
5. **Differentiate Research vs Proof** - Clear user journey

With these improvements, the QBM frontend can evolve from a functional prototype into the **world-class Islamic research platform** it aspires to be.

---

**Assessment completed: December 30, 2025**
**Next step: Await user approval before implementing recommendations**
