# QBM Local Development Runbook

This document provides exact commands to run the QBM backend and frontend locally.

## Prerequisites

- Python 3.11+
- Node.js 18+
- npm

## Step 1: Build Truth Metrics (Required First)

Before starting the backend, you must build the truth metrics artifact:

```bash
cd D:\Quran_matrix
python scripts/build_truth_metrics_v1.py
```

Expected output:
```
============================================================
Building Truth Metrics v1
============================================================
✓ Found canonical dataset: data\exports\qbm_silver_20251221.json
✓ Loaded 6236 spans
Computing metrics...
✓ Agent distribution: 7 types
✓ Behavior forms: 5 types
✓ Evaluations: 3 types
✓ Written to data\metrics\truth_metrics_v1.json
✓ Checksum: 13a62fa655ed5e2a...
✓ Build complete. Status: ready
```

This creates `data/metrics/truth_metrics_v1.json` which is the **source of truth** for all statistics.

## Step 2: Start Backend

```bash
cd D:\Quran_matrix
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Verify backend is running:
```bash
curl http://localhost:8000/
# Should return: {"status":"healthy",...}

curl http://localhost:8000/api/metrics/overview
# Should return the truth metrics JSON
```

## Step 3: Start Frontend

In a new terminal:

```bash
cd D:\Quran_matrix\qbm-frontendv3
npm install
npm run dev
```

The frontend will start on http://localhost:3000 (or 3001 if 3000 is busy).

## Environment Variables

### Backend (.env in D:\Quran_matrix)
```
QBM_DATA_DIR=data
```

### Frontend (.env.local in qbm-frontendv3)
```
NEXT_PUBLIC_QBM_BACKEND_URL=http://localhost:8000
QBM_BACKEND_URL=http://localhost:8000
THESYS_API_KEY=your-thesys-key  # Optional, for AI chat
```

## Verification Checklist

### 1. Metrics Endpoint Returns Real Data
```bash
curl http://localhost:8000/api/metrics/overview | jq '.metrics.agent_distribution.items[0]'
```
Expected:
```json
{
  "key": "AGT_ALLAH",
  "label_ar": "الله",
  "count": 2855,
  "percentage": 45.78
}
```

### 2. Frontend Shows Professional UI
- Visit http://localhost:3000/metrics
- Should show:
  - KPI cards (Total Spans: 6,236, Unique Verses: 6,236, Tafsir Sources: 7)
  - Donut chart for agent distribution
  - Styled tables (not ASCII)
  - No raw `###` markdown headers

### 3. Numbers Match Backend Exactly
- AGT_ALLAH should show 2,855 (45.78%) in both backend JSON and frontend UI
- AGT_DISBELIEVER should show 1,142 (18.31%)
- No "90%/10%" anywhere

### 4. Chat Does Not Invent Statistics
- If asked "أنشئ مخطط دائري لتوزيع أنواع الفاعلين"
- Response should use numbers from metrics JSON
- If metrics unavailable, should say "الإحصائيات غير متوفرة"

## Running Tests

```bash
# Truth metrics tests
python -m pytest tests/test_truth_metrics_v1.py -v

# API metrics tests
python -m pytest tests/test_api_metrics_overview.py -v

# All tests
python -m pytest tests/ -v --ignore=tests/tools
```

## Troubleshooting

### Backend returns 503 for /api/metrics/overview
Run `python scripts/build_truth_metrics_v1.py` to generate the truth metrics file.

### Frontend shows "Data Unavailable"
1. Check backend is running on port 8000
2. Check NEXT_PUBLIC_QBM_BACKEND_URL is set correctly
3. Check browser console for CORS errors

### Charts not rendering
Ensure `recharts` is installed: `npm install recharts`

### Markdown showing as raw text
Ensure `react-markdown` and `remark-gfm` are installed:
```bash
npm install react-markdown remark-gfm
```

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    TRUTH LAYER                              │
│  data/metrics/truth_metrics_v1.json (build-time artifact)   │
│  - Computed from canonical qbm_silver dataset               │
│  - Checksummed for integrity                                │
│  - NEVER hardcoded, NEVER guessed                           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    BACKEND API                              │
│  GET /api/metrics/overview                                  │
│  - Serves truth_metrics_v1.json VERBATIM                    │
│  - Returns 503 if file missing                              │
│  - No computation at request time                           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    FRONTEND UI                              │
│  /metrics page                                              │
│  - Fetches from /api/metrics/overview                       │
│  - Renders MetricCard, DistributionTable, DistributionChart │
│  - Numbers come from JSON, NOT from LLM                     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    LLM CHAT                                 │
│  - Receives metrics JSON in system prompt                   │
│  - May ONLY reference numbers from that JSON                │
│  - If no metrics, says "unavailable"                        │
│  - NEVER invents statistics                                 │
└─────────────────────────────────────────────────────────────┘
```

## Key Files

| File | Purpose |
|------|---------|
| `scripts/build_truth_metrics_v1.py` | Computes metrics from canonical data |
| `data/metrics/truth_metrics_v1.json` | Truth artifact (build output) |
| `src/api/main.py` | Backend API with `/api/metrics/overview` |
| `qbm-frontendv3/src/lib/metricsStore.ts` | Frontend metrics fetching |
| `qbm-frontendv3/src/app/components/MetricsDashboard.tsx` | Main dashboard component |
| `qbm-frontendv3/src/app/metrics/page.tsx` | Metrics page |
