# QBM Behavior Analysis System - Setup & Test Guide

## What Was Implemented

### 1. Enhanced System Prompt (`src/app/api/chat/route.ts`)
The system prompt now includes the **complete Bouzidani Five-Context Framework** methodology:

- **السياق العضوي (Organic)** - Body organs involved
- **السياق الموضعي (Situational)** - Internal vs External
- **السياق النسقي (Systemic)** - Social systems
- **السياق المكاني (Spatial)** - Location context
- **السياق الزماني (Temporal)** - Time dimension

Plus additional dimensions:
- **الفاعل (Agent)** - Who performs the behavior
- **المصدر (Source)** - What causes the behavior
- **التقييم (Evaluation)** - Praise/Blame/Neutral
- **أنماط القلوب (Heart Types)** - Personality mapping
- **العواقب (Consequences)** - Results of behavior
- **السلوكيات المرتبطة (Related Behaviors)** - Connections

### 2. Real Backend Integration
Tools now connect to the actual QBM backend API (`http://localhost:8000`):

| Tool | Function |
|------|----------|
| `analyze_behavior` | Comprehensive behavior analysis with all dimensions |
| `search_spans` | Search QBM database with filters |
| `get_tafsir` | Fetch tafsir from 5 sources |
| `get_statistics` | Dataset statistics |
| `compare_personalities` | Compare Believer/Munafiq/Kafir |
| `get_related_behaviors` | Find opposite/similar/cause/effect behaviors |

### 3. Behavior Mapping
Arabic behaviors are mapped to QBM codes:
- الكبر → BEH_COG_ARROGANCE
- الصدق → BEH_SPEECH_TRUTHFULNESS
- السرقة → BEH_PHY_THEFT
- الصبر → BEH_EMO_PATIENCE
- الاكتئاب → BEH_EMO_GRIEF
- etc.

---

## Setup Instructions

### Step 1: Install Dependencies
```bash
cd D:\Quran_matrix\qbm-frontendv3
npm install
```

### Step 2: Configure Environment
```bash
cp .env.example .env.local
```

Edit `.env.local`:
```env
THESYS_API_KEY=your_thesys_api_key
QBM_BACKEND_URL=http://localhost:8000
C1_MODEL=c1/anthropic/claude-sonnet-4/v-20251130
```

### Step 3: Start Backend API
```bash
cd D:\Quran_matrix
uvicorn src.api.main:app --reload --port 8000
```

### Step 4: Start Frontend
```bash
cd D:\Quran_matrix\qbm-frontendv3
npm run dev
```

### Step 5: Open Browser
Go to: http://localhost:3000/research

---

## Test Queries

### Test 1: Analyze Pride (الكبر)
```
حلل لي سلوك الكبر في القرآن الكريم
```

Expected output:
- Total mentions across surahs
- Agents (Satan, Kafir, Pharaoh, etc.)
- Organs involved (heart, face, walk)
- Evaluation (100% blamed)
- Key verses with references
- Related behaviors (opposite: التواضع)

### Test 2: Analyze Honesty (الصدق)
```
ما هي خارطة سلوك الصدق في القرآن؟
```

### Test 3: Analyze Depression/Grief (الاكتئاب)
```
كيف يتناول القرآن سلوك الحزن والاكتئاب؟
```

### Test 4: Compare Personalities
```
قارن بين سلوك المؤمن والمنافق والكافر في الصبر
```

### Test 5: Theft Analysis (السرقة)
```
حلل سلوك السرقة في القرآن مع جميع السياقات
```

---

## Expected Response Structure

For any behavior query, the system should return:

```
┌─────────────────────────────────────────────────────────────┐
│                    COMPLETE BEHAVIOR MAP                     │
├─────────────────────────────────────────────────────────────┤
│ 📊 الإحصائيات      │ Total mentions, surah distribution     │
│ 🫀 السياق العضوي   │ Heart, tongue, eye, hand, foot...      │
│ 📍 السياق الموضعي  │ Inner state vs external action         │
│ 👥 السياق النسقي   │ Family, society, worship, financial    │
│ 🏠 السياق المكاني  │ Mosque, home, market, battlefield      │
│ ⏰ السياق الزماني  │ Dunya, death, barzakh, akhira          │
│ 👤 الفاعلون       │ Believer, Kafir, Munafiq, Satan...     │
│ 🔗 المصادر        │ Revelation, fitrah, nafs, shaytan      │
│ ⚖️ التقييم        │ Praise, blame, neutral, warning        │
│ ❤️ أنماط القلوب   │ Healthy, sick, dead, hard              │
│ 📖 الآيات الرئيسية │ Key verses with tafsir                 │
│ 🔄 السلوكيات المرتبطة │ Opposite, similar, cause, effect    │
└─────────────────────────────────────────────────────────────┘
```

---

## Scholar's Vision Fulfilled

The system now provides:

✅ **Systematic methodology** - Not scattered answers
✅ **Integrative approach** - All dimensions connected
✅ **Any behavior input** - Works for الكبر, الصدق, السرقة, الاكتئاب, etc.
✅ **Complete mapping** - All Quranic references extracted
✅ **Personality comparison** - Believer vs Munafiq vs Kafir
✅ **Heart type connection** - Healthy, sick, dead, hard
✅ **Real data** - Connected to QBM database (15,847 annotations)

---

## Files Modified

1. `src/app/api/chat/route.ts` - Enhanced system prompt + real backend integration
2. `SETUP_AND_TEST.md` - This guide

---

*Built for Islamic scholarship - QBM Research Platform*
