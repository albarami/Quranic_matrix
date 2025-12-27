# QBM Deployment Guide

## Prerequisites

- Python 3.11+
- Node.js 18+
- Git

## Environment Variables

### Backend (.env)

```bash
# Server
HOST=0.0.0.0
PORT=8000
DEBUG=false

# Data paths
DATA_DIR=./data
VOCAB_DIR=./vocab

# Database
REVIEWS_DB_URL=sqlite:///data/reviews.db
# For production Postgres:
# REVIEWS_DB_URL=postgresql://user:pass@host:5432/qbm_reviews

# Optional: OpenAI for embeddings
OPENAI_API_KEY=sk-...

# Optional: FAISS index
FAISS_INDEX_PATH=./data/faiss_index
```

### Frontend (.env.local)

```bash
# Backend URL
NEXT_PUBLIC_QBM_BACKEND_URL=http://localhost:8000

# For production:
# NEXT_PUBLIC_QBM_BACKEND_URL=https://api.qbm.example.com
```

## Local Development

### Backend

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: .\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd qbm-frontendv3

# Install dependencies
npm install

# Run development server
npm run dev
```

### Run Tests

```bash
# Backend tests
pytest tests/ -v

# Frontend E2E tests (requires backend running)
cd qbm-frontendv3
npm run test:e2e
```

## Production Deployment

### Docker

#### Backend Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/ data/
COPY vocab/ vocab/

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Frontend Dockerfile

```dockerfile
FROM node:18-alpine AS builder

WORKDIR /app
COPY qbm-frontendv3/package*.json ./
RUN npm ci
COPY qbm-frontendv3/ .
RUN npm run build

FROM node:18-alpine AS runner
WORKDIR /app
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public

EXPOSE 3000
CMD ["node", "server.js"]
```

#### Docker Compose

```yaml
version: '3.8'

services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    ports:
      - "8000:8000"
    environment:
      - DEBUG=false
      - REVIEWS_DB_URL=postgresql://qbm:password@db:5432/qbm
    volumes:
      - ./data:/app/data
    depends_on:
      - db

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_QBM_BACKEND_URL=http://backend:8000
    depends_on:
      - backend

  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=qbm
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=qbm
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

### Cloud Deployment

#### Vercel (Frontend)

1. Connect GitHub repository
2. Set root directory to `qbm-frontendv3`
3. Add environment variable:
   - `NEXT_PUBLIC_QBM_BACKEND_URL=https://api.qbm.example.com`
4. Deploy

#### Railway/Render (Backend)

1. Connect GitHub repository
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`
4. Add environment variables
5. Deploy

## Production Checklist

### Security

- [ ] API keys not hardcoded
- [ ] CORS configured for production domain
- [ ] HTTPS enabled
- [ ] Rate limiting configured
- [ ] Input validation enabled

### Performance

- [ ] FAISS index pre-built
- [ ] Database connection pooling
- [ ] Static assets cached
- [ ] Gzip compression enabled

### Monitoring

- [ ] Health check endpoint accessible
- [ ] Error logging configured
- [ ] Performance metrics collected
- [ ] Alerting configured

### Data

- [ ] All data files present:
  - `data/quran/quran_uthmani.json`
  - `data/tafsir/*.json` (5 sources)
  - `data/graph/semantic_graph_v2.json`
  - `vocab/canonical_entities.json`
- [ ] Database migrations applied
- [ ] Backup strategy in place

### Testing

- [ ] All backend tests pass: `pytest tests/ -v`
- [ ] Frontend builds: `npm run build`
- [ ] E2E tests pass: `npm run test:e2e`
- [ ] Genome export checksum reproducible

## Tier Configuration

QBM supports tiered deployment:

### Tier A (Minimal)

- Quran text only
- No tafsir
- No embeddings
- Fastest startup

```bash
TIER=A
```

### Tier B (Standard)

- Quran + Tafsir
- BM25 search
- No embeddings

```bash
TIER=B
```

### Tier C (Full)

- All data sources
- FAISS embeddings
- Full graph

```bash
TIER=C
```

## Fail-Fast Behavior

QBM fails fast on critical errors:

| Error | Behavior |
|-------|----------|
| Missing Quran data | Startup fails |
| Missing canonical entities | Startup fails |
| Missing tafsir | Warning, continues |
| Missing FAISS index | Falls back to BM25 |
| Database connection failed | Reviews disabled |

## Scaling

### Horizontal Scaling

- Backend is stateless (except SQLite reviews)
- Use Postgres for reviews in multi-instance deployment
- Load balance with nginx/Traefik

### Vertical Scaling

- Increase memory for larger FAISS index
- Use GPU for embedding generation
- SSD for faster data loading

## Backup & Recovery

### Data Backup

```bash
# Backup reviews database
pg_dump qbm > backup_$(date +%Y%m%d).sql

# Backup data files
tar -czf data_backup_$(date +%Y%m%d).tar.gz data/ vocab/
```

### Recovery

```bash
# Restore database
psql qbm < backup_20251227.sql

# Restore data files
tar -xzf data_backup_20251227.tar.gz
```

## Troubleshooting

### Backend won't start

1. Check Python version: `python --version` (need 3.11+)
2. Check dependencies: `pip install -r requirements.txt`
3. Check data files exist
4. Check port not in use: `lsof -i :8000`

### Frontend won't build

1. Check Node version: `node --version` (need 18+)
2. Clear cache: `rm -rf .next node_modules && npm install`
3. Check environment variables

### Tests failing

1. Ensure backend is running for E2E tests
2. Check database is clean: delete `data/reviews.db`
3. Run specific test: `pytest tests/test_api.py -v`

### Slow queries

1. Check FAISS index is loaded
2. Enable summary mode for large results
3. Check database indexes
