# QBM Project: END-TO-END Completion Guide
## Complete Roadmap from Pilot to Full Quran Coverage

---

# TABLE OF CONTENTS

1. [Executive Summary](#executive-summary)
2. [Part 1: New Scripts to Create](#part-1-new-files-to-create)
3. [Part 2: PROJECT_PLAN.md Modifications](#part-2-modifications-to-existing-project_planmd)
4. [Part 3: Dependencies](#part-3-new-requirementstxt-additions)
5. [Part 4: Folder Structure](#part-4-folder-structure-additions)
6. [Part 5: Implementation Checklist](#part-5-summary-checklist-for-ai-coder)
7. [Execution Order](#execution-order)

---

# EXECUTIVE SUMMARY

## Vision
Extend the QBM pilot project to achieve **100% Quran coverage** with **structured tafsir integration**, creating the world's most comprehensive Quranic behavioral classification dataset.

## Current vs Target State

The current `PROJECT_PLAN.md` covers a **pilot project** (20 weeks, ~10,000 spans, ~3,000 ayat).

To complete the **entire Quran** (6,236 ayat, 15,000-20,000 spans) with **tafsir integration**, the following additions are required:

| Gap | Current State | Required State | Priority |
|-----|---------------|----------------|----------|
| Ayat Coverage | ~3,000 | **6,236 (100%)** | Critical |
| Spans | 10,000 | **15,000-20,000** | Critical |
| Tafsir | Manual consultation | **Structured DB + tools** | High |
| Timeline | 20 weeks | **52-78 weeks** | - |
| Budget | $23,500 | **$80,000-$259,000** | - |
| Team | 4-6 people | **10-15 people** | Medium |
| IAA Target | ≥ 0.70 | **≥ 0.75** | High |
| Publications | Draft | **2-3 papers** | Medium |

## Key Milestones

```
Week 16  → Pilot Complete (500 spans, IAA ≥ 0.70)
Week 24  → Tafsir Integration Complete (5+ sources, DB live)
Week 40  → Scale-Up Complete (3,000 ayat, 10,000 spans)
Week 60  → Full Coverage (6,236 ayat, 15,000+ spans)
Week 70  → Production Release (API live, docs complete)
Week 78  → Publication (2+ papers submitted)
```

---

# PART 1: NEW FILES TO CREATE

> **Note:** All scripts follow PEP8 style, include type hints, and have Google-style docstrings.
> Copy these scripts verbatim to the specified locations.

## 1.1 Tafsir Download Script

**Location:** `src/scripts/download_tafsir.py`
**Purpose:** Download tafsir data from Quran.com API and GitHub repositories
**Dependencies:** `requests`, `tqdm`, `aiohttp` (optional for async)

```python
#!/usr/bin/env python3
"""
Download tafsir data from multiple sources.
Sources: Quran.com API, GitHub repositories, Shamela exports

Usage:
    python download_tafsir.py --sources quran_api,github --output data/tafsir/
"""

import argparse
import json
import os
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional

# Quran.com API tafsir IDs
QURAN_API_TAFSIR = {
    "ibn_kathir_ar": 169,      # تفسير ابن كثير
    "ibn_kathir_en": 169,      # Ibn Kathir (English)
    "tabari": 167,             # جامع البيان للطبري
    "saadi": 170,              # تيسير الكريم الرحمن
    "jalalayn_ar": 74,         # تفسير الجلالين
    "muyassar": 16,            # التفسير الميسر
    "qurtubi": 168,            # الجامع لأحكام القرآن
    "baghawi": 166,            # معالم التنزيل
    "waseet": 171,             # الوسيط لطنطاوي
}

# GitHub sources
GITHUB_SOURCES = {
    "tafsir_api": "https://raw.githubusercontent.com/spa5k/tafsir_api/main/tafsir",
    "quran_json": "https://raw.githubusercontent.com/risan/quran-json/main/dist",
}

# Quran structure
SURAH_AYAH_COUNT = {
    1: 7, 2: 286, 3: 200, 4: 176, 5: 120, 6: 165, 7: 206, 8: 75, 9: 129, 10: 109,
    11: 123, 12: 111, 13: 43, 14: 52, 15: 99, 16: 128, 17: 111, 18: 110, 19: 98, 20: 135,
    21: 112, 22: 78, 23: 118, 24: 64, 25: 77, 26: 227, 27: 93, 28: 88, 29: 69, 30: 60,
    31: 34, 32: 30, 33: 73, 34: 54, 35: 45, 36: 83, 37: 182, 38: 88, 39: 75, 40: 85,
    41: 54, 42: 53, 43: 89, 44: 59, 45: 37, 46: 35, 47: 38, 48: 29, 49: 18, 50: 45,
    51: 60, 52: 49, 53: 62, 54: 55, 55: 78, 56: 96, 57: 29, 58: 22, 59: 24, 60: 13,
    61: 14, 62: 11, 63: 11, 64: 18, 65: 12, 66: 12, 67: 30, 68: 52, 69: 52, 70: 44,
    71: 28, 72: 28, 73: 20, 74: 56, 75: 40, 76: 31, 77: 50, 78: 40, 79: 46, 80: 42,
    81: 29, 82: 19, 83: 36, 84: 25, 85: 22, 86: 17, 87: 19, 88: 26, 89: 30, 90: 20,
    91: 15, 92: 21, 93: 11, 94: 8, 95: 8, 96: 19, 97: 5, 98: 8, 99: 8, 100: 11,
    101: 11, 102: 8, 103: 3, 104: 9, 105: 5, 106: 4, 107: 7, 108: 3, 109: 6, 110: 3,
    111: 5, 112: 4, 113: 5, 114: 6
}

TOTAL_AYAT = sum(SURAH_AYAH_COUNT.values())  # 6236


class TafsirDownloader:
    """Download and structure tafsir data from multiple sources."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "QBM-Tafsir-Downloader/1.0",
            "Accept": "application/json"
        })
    
    def download_from_quran_api(self, tafsir_key: str, tafsir_id: int) -> Dict:
        """Download tafsir from Quran.com API."""
        print(f"Downloading {tafsir_key} from Quran.com API...")
        
        tafsir_data = {
            "id": f"TAFSIR_{tafsir_key.upper()}",
            "name_ar": "",
            "name_en": tafsir_key.replace("_", " ").title(),
            "source": "quran.com-api",
            "version": "1.0",
            "ayat": {}
        }
        
        for surah in range(1, 115):
            ayah_count = SURAH_AYAH_COUNT[surah]
            print(f"  Surah {surah}/{114}...", end="\r")
            
            for ayah in range(1, ayah_count + 1):
                try:
                    url = f"https://api.quran.com/api/v4/tafsirs/{tafsir_id}/by_ayah/{surah}:{ayah}"
                    response = self.session.get(url, timeout=30)
                    
                    if response.status_code == 200:
                        data = response.json()
                        tafsir_text = data.get("tafsir", {}).get("text", "")
                        
                        key = f"{surah}:{ayah}"
                        tafsir_data["ayat"][key] = {
                            "surah": surah,
                            "ayah": ayah,
                            "text": tafsir_text
                        }
                    
                    # Rate limiting
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"\n  Error at {surah}:{ayah}: {e}")
                    continue
        
        print(f"\n  Downloaded {len(tafsir_data['ayat'])} ayat for {tafsir_key}")
        return tafsir_data
    
    def download_from_github(self, source_key: str) -> List[Dict]:
        """Download tafsir from GitHub repositories."""
        print(f"Downloading from GitHub: {source_key}...")
        
        base_url = GITHUB_SOURCES.get(source_key)
        if not base_url:
            print(f"  Unknown source: {source_key}")
            return []
        
        # Implementation depends on specific repository structure
        # This is a template - adjust based on actual repo structure
        tafsir_list = []
        
        try:
            # Try to get index/manifest first
            index_url = f"{base_url}/index.json"
            response = self.session.get(index_url, timeout=30)
            
            if response.status_code == 200:
                index = response.json()
                # Process based on repository structure
                pass
                
        except Exception as e:
            print(f"  Error downloading from {source_key}: {e}")
        
        return tafsir_list
    
    def save_tafsir(self, tafsir_data: Dict, filename: str):
        """Save tafsir data to JSON file."""
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(tafsir_data, f, ensure_ascii=False, indent=2)
        print(f"  Saved to {filepath}")
    
    def download_all(self, sources: List[str] = None):
        """Download all tafsir from specified sources."""
        if sources is None:
            sources = ["quran_api"]
        
        all_tafsir = []
        
        if "quran_api" in sources:
            for tafsir_key, tafsir_id in QURAN_API_TAFSIR.items():
                tafsir_data = self.download_from_quran_api(tafsir_key, tafsir_id)
                self.save_tafsir(tafsir_data, f"{tafsir_key}.json")
                all_tafsir.append(tafsir_data)
        
        if "github" in sources:
            for source_key in GITHUB_SOURCES:
                github_tafsir = self.download_from_github(source_key)
                all_tafsir.extend(github_tafsir)
        
        # Create combined index
        index = {
            "total_ayat": TOTAL_AYAT,
            "tafsir_sources": [t["id"] for t in all_tafsir],
            "download_date": time.strftime("%Y-%m-%d"),
            "version": "1.0"
        }
        
        self.save_tafsir(index, "tafsir_index.json")
        print(f"\nDownload complete. {len(all_tafsir)} tafsir sources saved.")
        
        return all_tafsir


def main():
    parser = argparse.ArgumentParser(description="Download tafsir data")
    parser.add_argument(
        "--sources", 
        default="quran_api",
        help="Comma-separated list of sources: quran_api,github"
    )
    parser.add_argument(
        "--output", 
        default="data/tafsir",
        help="Output directory"
    )
    parser.add_argument(
        "--tafsir",
        default=None,
        help="Specific tafsir to download (e.g., ibn_kathir_ar)"
    )
    
    args = parser.parse_args()
    sources = args.sources.split(",")
    
    downloader = TafsirDownloader(args.output)
    
    if args.tafsir:
        # Download specific tafsir
        if args.tafsir in QURAN_API_TAFSIR:
            tafsir_data = downloader.download_from_quran_api(
                args.tafsir, 
                QURAN_API_TAFSIR[args.tafsir]
            )
            downloader.save_tafsir(tafsir_data, f"{args.tafsir}.json")
        else:
            print(f"Unknown tafsir: {args.tafsir}")
            print(f"Available: {list(QURAN_API_TAFSIR.keys())}")
    else:
        downloader.download_all(sources)


if __name__ == "__main__":
    main()
```

### Usage Examples
```bash
# Download all tafsir from Quran.com API
python src/scripts/download_tafsir.py --sources quran_api --output data/tafsir/

# Download specific tafsir
python src/scripts/download_tafsir.py --tafsir ibn_kathir_ar --output data/tafsir/

# Download from multiple sources
python src/scripts/download_tafsir.py --sources quran_api,github --output data/tafsir/
```

---

## 1.2 Tafsir Database Schema

**Location:** `src/scripts/setup_tafsir_db.py`

```python
#!/usr/bin/env python3
"""
Set up PostgreSQL database schema for tafsir storage.

Usage:
    python setup_tafsir_db.py --host localhost --db qbm
"""

import argparse
from sqlalchemy import create_engine, text

TAFSIR_SCHEMA = """
-- Tafsir Sources Table
CREATE TABLE IF NOT EXISTS tafsir_sources (
    id SERIAL PRIMARY KEY,
    source_id VARCHAR(50) UNIQUE NOT NULL,      -- e.g., "TAFSIR_IBN_KATHIR"
    name_ar VARCHAR(200),                        -- تفسير ابن كثير
    name_en VARCHAR(200),                        -- Ibn Kathir
    author_ar VARCHAR(200),
    author_en VARCHAR(200),
    death_year_hijri INTEGER,                    -- 774
    methodology VARCHAR(50),                      -- bil_mathur, bil_ray, mixed
    language VARCHAR(10) DEFAULT 'ar',
    source_url VARCHAR(500),
    version VARCHAR(20),
    download_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tafsir Content Table (one row per ayah per tafsir)
CREATE TABLE IF NOT EXISTS tafsir_content (
    id SERIAL PRIMARY KEY,
    source_id VARCHAR(50) NOT NULL REFERENCES tafsir_sources(source_id),
    surah INTEGER NOT NULL CHECK (surah >= 1 AND surah <= 114),
    ayah INTEGER NOT NULL CHECK (ayah >= 1),
    text_ar TEXT,
    text_en TEXT,
    word_count INTEGER,
    has_hadith BOOLEAN DEFAULT FALSE,
    has_asbab_nuzul BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_id, surah, ayah)
);

-- Index for fast ayah lookup
CREATE INDEX IF NOT EXISTS idx_tafsir_ayah 
ON tafsir_content(surah, ayah);

CREATE INDEX IF NOT EXISTS idx_tafsir_source 
ON tafsir_content(source_id);

-- Full-text search index (Arabic)
CREATE INDEX IF NOT EXISTS idx_tafsir_text_ar 
ON tafsir_content USING gin(to_tsvector('arabic', text_ar));

-- Tafsir Consultation Log (tracks which tafsir was consulted for each annotation)
CREATE TABLE IF NOT EXISTS tafsir_consultations (
    id SERIAL PRIMARY KEY,
    span_id VARCHAR(20) NOT NULL,               -- QBM_00001
    tafsir_source_id VARCHAR(50) NOT NULL,
    surah INTEGER NOT NULL,
    ayah INTEGER NOT NULL,
    consulted_by VARCHAR(100),                  -- annotator username
    consultation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    influenced_decision BOOLEAN DEFAULT FALSE,
    notes TEXT
);

-- Tafsir Agreement Tracking
CREATE TABLE IF NOT EXISTS tafsir_agreement (
    id SERIAL PRIMARY KEY,
    surah INTEGER NOT NULL,
    ayah INTEGER NOT NULL,
    topic VARCHAR(100),                         -- e.g., "agent_identification"
    sources_agree TEXT[],                       -- array of agreeing source_ids
    sources_disagree TEXT[],                    -- array of disagreeing source_ids
    agreement_level VARCHAR(20),                -- unanimous, majority, split
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- View: Tafsir Coverage Summary
CREATE OR REPLACE VIEW tafsir_coverage_summary AS
SELECT 
    source_id,
    COUNT(DISTINCT CONCAT(surah, ':', ayah)) as ayat_covered,
    COUNT(DISTINCT surah) as surahs_covered,
    SUM(word_count) as total_words,
    MIN(surah) as first_surah,
    MAX(surah) as last_surah
FROM tafsir_content
GROUP BY source_id;

-- View: Annotation-Tafsir Link Summary
CREATE OR REPLACE VIEW annotation_tafsir_summary AS
SELECT 
    tc.span_id,
    COUNT(DISTINCT tc.tafsir_source_id) as tafsir_consulted,
    array_agg(DISTINCT tc.tafsir_source_id) as sources_used,
    bool_or(tc.influenced_decision) as any_influence
FROM tafsir_consultations tc
GROUP BY tc.span_id;
"""


def setup_database(host: str, port: int, db: str, user: str, password: str):
    """Create database schema for tafsir storage."""
    
    connection_string = f"postgresql://{user}:{password}@{host}:{port}/{db}"
    engine = create_engine(connection_string)
    
    print(f"Connecting to {host}:{port}/{db}...")
    
    with engine.connect() as conn:
        print("Creating tafsir schema...")
        
        # Execute schema
        for statement in TAFSIR_SCHEMA.split(';'):
            statement = statement.strip()
            if statement:
                conn.execute(text(statement))
        
        conn.commit()
        print("Schema created successfully.")
    
    return engine


def load_tafsir_json(engine, json_file: str, source_id: str):
    """Load tafsir from JSON file into database."""
    import json
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loading {source_id} from {json_file}...")
    
    with engine.connect() as conn:
        # Insert source
        conn.execute(text("""
            INSERT INTO tafsir_sources (source_id, name_en, version, download_date)
            VALUES (:source_id, :name_en, :version, CURRENT_DATE)
            ON CONFLICT (source_id) DO UPDATE SET version = :version
        """), {
            "source_id": source_id,
            "name_en": data.get("name_en", source_id),
            "version": data.get("version", "1.0")
        })
        
        # Insert content
        ayat = data.get("ayat", {})
        count = 0
        
        for key, content in ayat.items():
            surah = content.get("surah")
            ayah = content.get("ayah")
            text_ar = content.get("text", "")
            
            conn.execute(text("""
                INSERT INTO tafsir_content (source_id, surah, ayah, text_ar, word_count)
                VALUES (:source_id, :surah, :ayah, :text_ar, :word_count)
                ON CONFLICT (source_id, surah, ayah) DO UPDATE SET text_ar = :text_ar
            """), {
                "source_id": source_id,
                "surah": surah,
                "ayah": ayah,
                "text_ar": text_ar,
                "word_count": len(text_ar.split()) if text_ar else 0
            })
            count += 1
        
        conn.commit()
        print(f"  Loaded {count} ayat.")


def main():
    parser = argparse.ArgumentParser(description="Setup tafsir database")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--db", default="qbm")
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password", default="")
    parser.add_argument("--load", help="JSON file to load")
    parser.add_argument("--source-id", help="Source ID for JSON load")
    
    args = parser.parse_args()
    
    engine = setup_database(
        args.host, args.port, args.db, args.user, args.password
    )
    
    if args.load and args.source_id:
        load_tafsir_json(engine, args.load, args.source_id)


if __name__ == "__main__":
    main()
```

### Usage Examples
```bash
# Set up database schema
python src/scripts/setup_tafsir_db.py --host localhost --db qbm --user postgres --password YOUR_PASSWORD

# Load tafsir JSON into database
python src/scripts/setup_tafsir_db.py --load data/tafsir/ibn_kathir_ar.json --source-id TAFSIR_IBN_KATHIR_AR
```

---

## 1.3 Tafsir Lookup Tool (for Annotators)

**Location:** `src/scripts/tafsir_lookup.py`

```python
#!/usr/bin/env python3
"""
Tafsir lookup tool for annotators.

Usage:
    python tafsir_lookup.py --surah 2 --ayah 255
    python tafsir_lookup.py --surah 2 --ayah 255 --tafsir ibn_kathir
    python tafsir_lookup.py --search "الكرسي"
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


class TafsirLookup:
    """Look up tafsir for ayat during annotation."""
    
    def __init__(self, tafsir_dir: str = "data/tafsir"):
        self.tafsir_dir = Path(tafsir_dir)
        self.tafsir_cache: Dict[str, Dict] = {}
        self._load_available_tafsir()
    
    def _load_available_tafsir(self):
        """Load index of available tafsir."""
        self.available = []
        
        for json_file in self.tafsir_dir.glob("*.json"):
            if json_file.name != "tafsir_index.json":
                self.available.append(json_file.stem)
        
        print(f"Available tafsir: {self.available}")
    
    def _load_tafsir(self, tafsir_key: str) -> Dict:
        """Load tafsir data into cache."""
        if tafsir_key not in self.tafsir_cache:
            filepath = self.tafsir_dir / f"{tafsir_key}.json"
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    self.tafsir_cache[tafsir_key] = json.load(f)
        
        return self.tafsir_cache.get(tafsir_key, {})
    
    def lookup(self, surah: int, ayah: int, tafsir_key: str = None) -> Dict:
        """Look up tafsir for a specific ayah."""
        result = {
            "surah": surah,
            "ayah": ayah,
            "reference": f"{surah}:{ayah}",
            "tafsir": {}
        }
        
        sources = [tafsir_key] if tafsir_key else self.available
        
        for source in sources:
            data = self._load_tafsir(source)
            ayat = data.get("ayat", {})
            key = f"{surah}:{ayah}"
            
            if key in ayat:
                result["tafsir"][source] = {
                    "text": ayat[key].get("text", ""),
                    "source_id": data.get("id", source)
                }
        
        return result
    
    def lookup_range(self, surah: int, start_ayah: int, end_ayah: int) -> List[Dict]:
        """Look up tafsir for a range of ayat."""
        results = []
        for ayah in range(start_ayah, end_ayah + 1):
            results.append(self.lookup(surah, ayah))
        return results
    
    def search(self, query: str, tafsir_key: str = None) -> List[Dict]:
        """Search tafsir text for a query."""
        results = []
        sources = [tafsir_key] if tafsir_key else self.available
        
        for source in sources:
            data = self._load_tafsir(source)
            ayat = data.get("ayat", {})
            
            for key, content in ayat.items():
                text = content.get("text", "")
                if query in text:
                    surah, ayah = key.split(":")
                    results.append({
                        "surah": int(surah),
                        "ayah": int(ayah),
                        "source": source,
                        "snippet": text[:500] + "..." if len(text) > 500 else text
                    })
        
        return results
    
    def compare(self, surah: int, ayah: int) -> Dict:
        """Compare tafsir from multiple sources for an ayah."""
        result = self.lookup(surah, ayah)
        
        # Add comparison metadata
        sources = list(result["tafsir"].keys())
        result["comparison"] = {
            "sources_count": len(sources),
            "sources": sources,
            "word_counts": {
                src: len(data.get("text", "").split())
                for src, data in result["tafsir"].items()
            }
        }
        
        return result
    
    def format_for_annotator(self, surah: int, ayah: int) -> str:
        """Format tafsir output for annotator console."""
        result = self.lookup(surah, ayah)
        
        output = []
        output.append(f"\n{'='*60}")
        output.append(f"TAFSIR LOOKUP: {surah}:{ayah}")
        output.append(f"{'='*60}\n")
        
        for source, data in result["tafsir"].items():
            output.append(f"📖 {source.upper()}")
            output.append("-" * 40)
            text = data.get("text", "No text available")
            # Truncate for display
            if len(text) > 1000:
                text = text[:1000] + "\n... [truncated]"
            output.append(text)
            output.append("")
        
        output.append(f"{'='*60}\n")
        return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(description="Tafsir lookup tool")
    parser.add_argument("--surah", type=int, help="Surah number")
    parser.add_argument("--ayah", type=int, help="Ayah number")
    parser.add_argument("--end-ayah", type=int, help="End ayah for range lookup")
    parser.add_argument("--tafsir", help="Specific tafsir source")
    parser.add_argument("--search", help="Search query")
    parser.add_argument("--compare", action="store_true", help="Compare sources")
    parser.add_argument("--dir", default="data/tafsir", help="Tafsir directory")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    lookup = TafsirLookup(args.dir)
    
    if args.search:
        results = lookup.search(args.search, args.tafsir)
        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            print(f"Found {len(results)} matches:")
            for r in results[:10]:
                print(f"  {r['surah']}:{r['ayah']} ({r['source']})")
    
    elif args.surah and args.ayah:
        if args.end_ayah:
            results = lookup.lookup_range(args.surah, args.ayah, args.end_ayah)
            if args.json:
                print(json.dumps(results, ensure_ascii=False, indent=2))
            else:
                for r in results:
                    print(lookup.format_for_annotator(r["surah"], r["ayah"]))
        elif args.compare:
            result = lookup.compare(args.surah, args.ayah)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            if args.json:
                result = lookup.lookup(args.surah, args.ayah, args.tafsir)
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print(lookup.format_for_annotator(args.surah, args.ayah))
    
    else:
        print("Available tafsir sources:")
        for source in lookup.available:
            print(f"  - {source}")


if __name__ == "__main__":
    main()
```

### Usage Examples
```bash
# Look up tafsir for a specific ayah
python src/scripts/tafsir_lookup.py --surah 2 --ayah 255

# Look up with specific tafsir source
python src/scripts/tafsir_lookup.py --surah 2 --ayah 255 --tafsir ibn_kathir_ar

# Look up a range of ayat
python src/scripts/tafsir_lookup.py --surah 2 --ayah 255 --end-ayah 257

# Compare multiple tafsir sources
python src/scripts/tafsir_lookup.py --surah 2 --ayah 255 --compare --json

# Search tafsir text
python src/scripts/tafsir_lookup.py --search "الكرسي"
```

---

## 1.4 Full Quran Span Selector

**Location:** `src/scripts/select_full_quran.py`

```python
#!/usr/bin/env python3
"""
Select and prepare all 6,236 ayat for annotation.
Creates batches with priority ordering.

Usage:
    python select_full_quran.py --output data/processed/full_quran_batches/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

# Quran structure
SURAH_AYAH_COUNT = {
    1: 7, 2: 286, 3: 200, 4: 176, 5: 120, 6: 165, 7: 206, 8: 75, 9: 129, 10: 109,
    11: 123, 12: 111, 13: 43, 14: 52, 15: 99, 16: 128, 17: 111, 18: 110, 19: 98, 20: 135,
    21: 112, 22: 78, 23: 118, 24: 64, 25: 77, 26: 227, 27: 93, 28: 88, 29: 69, 30: 60,
    31: 34, 32: 30, 33: 73, 34: 54, 35: 45, 36: 83, 37: 182, 38: 88, 39: 75, 40: 85,
    41: 54, 42: 53, 43: 89, 44: 59, 45: 37, 46: 35, 47: 38, 48: 29, 49: 18, 50: 45,
    51: 60, 52: 49, 53: 62, 54: 55, 55: 78, 56: 96, 57: 29, 58: 22, 59: 24, 60: 13,
    61: 14, 62: 11, 63: 11, 64: 18, 65: 12, 66: 12, 67: 30, 68: 52, 69: 52, 70: 44,
    71: 28, 72: 28, 73: 20, 74: 56, 75: 40, 76: 31, 77: 50, 78: 40, 79: 46, 80: 42,
    81: 29, 82: 19, 83: 36, 84: 25, 85: 22, 86: 17, 87: 19, 88: 26, 89: 30, 90: 20,
    91: 15, 92: 21, 93: 11, 94: 8, 95: 8, 96: 19, 97: 5, 98: 8, 99: 8, 100: 11,
    101: 11, 102: 8, 103: 3, 104: 9, 105: 5, 106: 4, 107: 7, 108: 3, 109: 6, 110: 3,
    111: 5, 112: 4, 113: 5, 114: 6
}

TOTAL_AYAT = 6236

# Priority surahs (high behavioral content based on Bouzidani paper)
PRIORITY_SURAHS = {
    "tier_1": [2, 3, 4, 5, 6, 7, 16, 17, 24, 49],  # Core behavioral guidance
    "tier_2": [8, 9, 10, 11, 12, 18, 23, 25, 33, 47],  # Stories + rulings
    "tier_3": [13, 14, 15, 19, 20, 21, 22, 26, 27, 28],  # Prophetic narratives
}


class FullQuranSelector:
    """Prepare full Quran for annotation in prioritized batches."""
    
    def __init__(self, quran_file: str):
        with open(quran_file, 'r', encoding='utf-8') as f:
            self.quran_data = json.load(f)
    
    def get_ayah(self, surah: int, ayah: int) -> Dict:
        """Get ayah data from Quran file."""
        # Adjust based on actual Quran data structure
        key = f"{surah}:{ayah}"
        return self.quran_data.get(key, {})
    
    def create_batches(self, batch_size: int = 100) -> List[Dict]:
        """Create annotation batches with priority ordering."""
        batches = []
        batch_id = 1
        
        # Priority ordering
        all_surahs_ordered = (
            PRIORITY_SURAHS["tier_1"] + 
            PRIORITY_SURAHS["tier_2"] + 
            PRIORITY_SURAHS["tier_3"] +
            [s for s in range(1, 115) if s not in 
             PRIORITY_SURAHS["tier_1"] + 
             PRIORITY_SURAHS["tier_2"] + 
             PRIORITY_SURAHS["tier_3"]]
        )
        
        current_batch = {
            "batch_id": f"BATCH_{batch_id:04d}",
            "priority": "high" if batch_id <= 10 else "medium" if batch_id <= 30 else "low",
            "ayat": [],
            "surah_coverage": set()
        }
        
        for surah in all_surahs_ordered:
            ayah_count = SURAH_AYAH_COUNT[surah]
            
            for ayah in range(1, ayah_count + 1):
                ayah_record = {
                    "surah": surah,
                    "ayah": ayah,
                    "reference": f"{surah}:{ayah}",
                    "text_ar": self.get_ayah(surah, ayah).get("text", ""),
                    "annotation_status": "pending"
                }
                
                current_batch["ayat"].append(ayah_record)
                current_batch["surah_coverage"].add(surah)
                
                if len(current_batch["ayat"]) >= batch_size:
                    current_batch["surah_coverage"] = list(current_batch["surah_coverage"])
                    current_batch["ayat_count"] = len(current_batch["ayat"])
                    batches.append(current_batch)
                    
                    batch_id += 1
                    current_batch = {
                        "batch_id": f"BATCH_{batch_id:04d}",
                        "priority": "high" if batch_id <= 10 else "medium" if batch_id <= 30 else "low",
                        "ayat": [],
                        "surah_coverage": set()
                    }
        
        # Final batch
        if current_batch["ayat"]:
            current_batch["surah_coverage"] = list(current_batch["surah_coverage"])
            current_batch["ayat_count"] = len(current_batch["ayat"])
            batches.append(current_batch)
        
        return batches
    
    def create_coverage_tracker(self) -> Dict:
        """Create tracker for annotation progress across full Quran."""
        tracker = {
            "total_ayat": TOTAL_AYAT,
            "total_surahs": 114,
            "surahs": {}
        }
        
        for surah, ayah_count in SURAH_AYAH_COUNT.items():
            tracker["surahs"][surah] = {
                "total_ayat": ayah_count,
                "annotated": 0,
                "gold": 0,
                "silver": 0,
                "pending": ayah_count,
                "percent_complete": 0.0
            }
        
        return tracker


def main():
    parser = argparse.ArgumentParser(description="Prepare full Quran for annotation")
    parser.add_argument("--quran", default="data/raw/quran_tokenized_full.json")
    parser.add_argument("--output", default="data/processed/full_quran_batches")
    parser.add_argument("--batch-size", type=int, default=100)
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    selector = FullQuranSelector(args.quran)
    
    # Create batches
    batches = selector.create_batches(args.batch_size)
    
    # Save batches
    for batch in batches:
        filepath = output_dir / f"{batch['batch_id']}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(batch, f, ensure_ascii=False, indent=2)
    
    # Save batch index
    index = {
        "total_batches": len(batches),
        "total_ayat": TOTAL_AYAT,
        "batch_size": args.batch_size,
        "batches": [
            {
                "batch_id": b["batch_id"],
                "priority": b["priority"],
                "ayat_count": b["ayat_count"],
                "surahs": b["surah_coverage"]
            }
            for b in batches
        ]
    }
    
    with open(output_dir / "batch_index.json", 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    
    # Save coverage tracker
    tracker = selector.create_coverage_tracker()
    with open(output_dir / "coverage_tracker.json", 'w', encoding='utf-8') as f:
        json.dump(tracker, f, ensure_ascii=False, indent=2)
    
    print(f"Created {len(batches)} batches in {output_dir}")
    print(f"Total ayat: {TOTAL_AYAT}")


if __name__ == "__main__":
    main()
```

### Usage Examples
```bash
# Create batches for full Quran
python src/scripts/select_full_quran.py --quran data/raw/quran_tokenized_full.json --output data/processed/full_quran_batches/ --batch-size 100
```

---

## 1.5 Coverage Audit Script

**Location:** `src/scripts/coverage_audit.py`
**Purpose:** Audit annotation coverage and identify gaps

```python
#!/usr/bin/env python3
"""
Audit annotation coverage across the full Quran.
Identifies gaps, generates reports, and creates gap-filling batches.

Usage:
    python coverage_audit.py --annotations data/annotations/ --output reports/coverage/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from datetime import datetime

# Quran structure
SURAH_AYAH_COUNT = {
    1: 7, 2: 286, 3: 200, 4: 176, 5: 120, 6: 165, 7: 206, 8: 75, 9: 129, 10: 109,
    11: 123, 12: 111, 13: 43, 14: 52, 15: 99, 16: 128, 17: 111, 18: 110, 19: 98, 20: 135,
    21: 112, 22: 78, 23: 118, 24: 64, 25: 77, 26: 227, 27: 93, 28: 88, 29: 69, 30: 60,
    31: 34, 32: 30, 33: 73, 34: 54, 35: 45, 36: 83, 37: 182, 38: 88, 39: 75, 40: 85,
    41: 54, 42: 53, 43: 89, 44: 59, 45: 37, 46: 35, 47: 38, 48: 29, 49: 18, 50: 45,
    51: 60, 52: 49, 53: 62, 54: 55, 55: 78, 56: 96, 57: 29, 58: 22, 59: 24, 60: 13,
    61: 14, 62: 11, 63: 11, 64: 18, 65: 12, 66: 12, 67: 30, 68: 52, 69: 52, 70: 44,
    71: 28, 72: 28, 73: 20, 74: 56, 75: 40, 76: 31, 77: 50, 78: 40, 79: 46, 80: 42,
    81: 29, 82: 19, 83: 36, 84: 25, 85: 22, 86: 17, 87: 19, 88: 26, 89: 30, 90: 20,
    91: 15, 92: 21, 93: 11, 94: 8, 95: 8, 96: 19, 97: 5, 98: 8, 99: 8, 100: 11,
    101: 11, 102: 8, 103: 3, 104: 9, 105: 5, 106: 4, 107: 7, 108: 3, 109: 6, 110: 3,
    111: 5, 112: 4, 113: 5, 114: 6
}

TOTAL_AYAT = 6236


class CoverageAuditor:
    """Audit annotation coverage and identify gaps."""
    
    def __init__(self, annotations_dir: str):
        self.annotations_dir = Path(annotations_dir)
        self.annotated_ayat: Set[Tuple[int, int]] = set()
        self.span_count: Dict[Tuple[int, int], int] = {}
        self.quality_status: Dict[str, List[Tuple[int, int]]] = {
            "gold": [],
            "silver": [],
            "draft": [],
            "disputed": []
        }
    
    def load_annotations(self) -> int:
        """Load all annotation files and extract coverage data."""
        total_spans = 0
        
        for json_file in self.annotations_dir.glob("**/*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Handle different formats
                if isinstance(data, list):
                    spans = data
                elif "spans" in data:
                    spans = data["spans"]
                elif "annotations" in data:
                    spans = data["annotations"]
                else:
                    continue
                
                for span in spans:
                    ref = span.get("reference", {})
                    surah = ref.get("surah")
                    ayah = ref.get("ayah")
                    
                    if surah and ayah:
                        key = (surah, ayah)
                        self.annotated_ayat.add(key)
                        self.span_count[key] = self.span_count.get(key, 0) + 1
                        
                        # Track quality status
                        status = span.get("review", {}).get("status", "draft")
                        if status == "approved":
                            self.quality_status["gold"].append(key)
                        elif status == "disputed":
                            self.quality_status["disputed"].append(key)
                        else:
                            self.quality_status["draft"].append(key)
                        
                        total_spans += 1
                        
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        
        return total_spans
    
    def find_gaps(self) -> Dict[int, List[int]]:
        """Find all ayat that have not been annotated."""
        gaps = {}
        
        for surah, ayah_count in SURAH_AYAH_COUNT.items():
            missing = []
            for ayah in range(1, ayah_count + 1):
                if (surah, ayah) not in self.annotated_ayat:
                    missing.append(ayah)
            
            if missing:
                gaps[surah] = missing
        
        return gaps
    
    def generate_report(self) -> Dict:
        """Generate comprehensive coverage report."""
        gaps = self.find_gaps()
        
        total_annotated = len(self.annotated_ayat)
        total_gaps = sum(len(ayat) for ayat in gaps.values())
        
        # Calculate per-surah coverage
        surah_coverage = {}
        for surah, ayah_count in SURAH_AYAH_COUNT.items():
            annotated = sum(1 for ayah in range(1, ayah_count + 1) 
                          if (surah, ayah) in self.annotated_ayat)
            surah_coverage[surah] = {
                "total": ayah_count,
                "annotated": annotated,
                "gaps": len(gaps.get(surah, [])),
                "percent": round(100 * annotated / ayah_count, 1)
            }
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_ayat": TOTAL_AYAT,
                "annotated_ayat": total_annotated,
                "gap_ayat": total_gaps,
                "coverage_percent": round(100 * total_annotated / TOTAL_AYAT, 2),
                "total_spans": sum(self.span_count.values()),
                "avg_spans_per_ayah": round(sum(self.span_count.values()) / max(total_annotated, 1), 2)
            },
            "quality_breakdown": {
                "gold": len(set(self.quality_status["gold"])),
                "draft": len(set(self.quality_status["draft"])),
                "disputed": len(set(self.quality_status["disputed"]))
            },
            "surah_coverage": surah_coverage,
            "gaps_by_surah": gaps,
            "surahs_complete": [s for s, c in surah_coverage.items() if c["percent"] == 100],
            "surahs_not_started": [s for s, c in surah_coverage.items() if c["percent"] == 0]
        }
        
        return report
    
    def create_gap_batches(self, batch_size: int = 50) -> List[Dict]:
        """Create batches for gap-filling annotation."""
        gaps = self.find_gaps()
        batches = []
        current_batch = []
        batch_id = 1
        
        for surah in sorted(gaps.keys()):
            for ayah in gaps[surah]:
                current_batch.append({
                    "surah": surah,
                    "ayah": ayah,
                    "reference": f"{surah}:{ayah}"
                })
                
                if len(current_batch) >= batch_size:
                    batches.append({
                        "batch_id": f"GAP_BATCH_{batch_id:04d}",
                        "type": "gap_fill",
                        "ayat_count": len(current_batch),
                        "ayat": current_batch
                    })
                    current_batch = []
                    batch_id += 1
        
        # Final batch
        if current_batch:
            batches.append({
                "batch_id": f"GAP_BATCH_{batch_id:04d}",
                "type": "gap_fill",
                "ayat_count": len(current_batch),
                "ayat": current_batch
            })
        
        return batches


def main():
    parser = argparse.ArgumentParser(description="Audit annotation coverage")
    parser.add_argument("--annotations", required=True, help="Annotations directory")
    parser.add_argument("--output", default="reports/coverage", help="Output directory")
    parser.add_argument("--create-batches", action="store_true", help="Create gap-filling batches")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for gap filling")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    auditor = CoverageAuditor(args.annotations)
    total_spans = auditor.load_annotations()
    
    print(f"Loaded {total_spans} spans from {args.annotations}")
    
    # Generate report
    report = auditor.generate_report()
    
    report_file = output_dir / f"coverage_report_{datetime.now().strftime('%Y%m%d')}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\nCoverage Report:")
    print(f"  Total ayat: {report['summary']['total_ayat']}")
    print(f"  Annotated: {report['summary']['annotated_ayat']} ({report['summary']['coverage_percent']}%)")
    print(f"  Gaps: {report['summary']['gap_ayat']}")
    print(f"  Total spans: {report['summary']['total_spans']}")
    print(f"  Surahs complete: {len(report['surahs_complete'])}")
    print(f"  Surahs not started: {len(report['surahs_not_started'])}")
    print(f"\nReport saved to: {report_file}")
    
    # Create gap-filling batches if requested
    if args.create_batches:
        batches = auditor.create_gap_batches(args.batch_size)
        
        batches_dir = output_dir / "gap_batches"
        batches_dir.mkdir(exist_ok=True)
        
        for batch in batches:
            batch_file = batches_dir / f"{batch['batch_id']}.json"
            with open(batch_file, 'w', encoding='utf-8') as f:
                json.dump(batch, f, ensure_ascii=False, indent=2)
        
        print(f"\nCreated {len(batches)} gap-filling batches in {batches_dir}")


if __name__ == "__main__":
    main()
```

### Usage Examples
```bash
# Generate coverage report
python src/scripts/coverage_audit.py --annotations data/annotations/ --output reports/coverage/

# Generate report and create gap-filling batches
python src/scripts/coverage_audit.py --annotations data/annotations/ --output reports/coverage/ --create-batches --batch-size 50
```

---

## 1.6 Batch Distribution Script

**Location:** `src/scripts/distribute_batches.py`
**Purpose:** Distribute annotation batches among annotators

```python
#!/usr/bin/env python3
"""
Distribute annotation batches among annotators.
Ensures balanced workload and overlap for IAA calculation.

Usage:
    python distribute_batches.py --input data/processed/batches/ --annotators 6 --overlap 0.2
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import random


def distribute_batches(
    batches_dir: str,
    num_annotators: int,
    overlap_ratio: float = 0.2,
    output_dir: str = None
) -> Dict[str, List[str]]:
    """
    Distribute batches among annotators with overlap for IAA.
    
    Args:
        batches_dir: Directory containing batch JSON files
        num_annotators: Number of annotators
        overlap_ratio: Fraction of batches to double-annotate
        output_dir: Output directory for assignments
    
    Returns:
        Dictionary mapping annotator IDs to batch IDs
    """
    batches_path = Path(batches_dir)
    batch_files = sorted(batches_path.glob("BATCH_*.json"))
    
    if not batch_files:
        print(f"No batch files found in {batches_dir}")
        return {}
    
    batch_ids = [f.stem for f in batch_files]
    total_batches = len(batch_ids)
    
    # Calculate overlap batches
    overlap_count = int(total_batches * overlap_ratio)
    overlap_batches = random.sample(batch_ids, min(overlap_count, total_batches))
    
    # Initialize assignments
    assignments = {f"annotator_{i+1}": [] for i in range(num_annotators)}
    
    # Distribute primary batches (round-robin)
    for i, batch_id in enumerate(batch_ids):
        annotator = f"annotator_{(i % num_annotators) + 1}"
        assignments[annotator].append({
            "batch_id": batch_id,
            "type": "primary"
        })
    
    # Assign overlap batches to second annotator
    for batch_id in overlap_batches:
        # Find who has this as primary
        primary_annotator = None
        for ann, batches in assignments.items():
            if any(b["batch_id"] == batch_id for b in batches):
                primary_annotator = ann
                break
        
        # Assign to different annotator for overlap
        other_annotators = [a for a in assignments.keys() if a != primary_annotator]
        if other_annotators:
            secondary = random.choice(other_annotators)
            assignments[secondary].append({
                "batch_id": batch_id,
                "type": "overlap"
            })
    
    # Save assignments
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        summary = {
            "total_batches": total_batches,
            "num_annotators": num_annotators,
            "overlap_ratio": overlap_ratio,
            "overlap_batches": len(overlap_batches),
            "assignments": {}
        }
        
        for annotator, batches in assignments.items():
            summary["assignments"][annotator] = {
                "primary": len([b for b in batches if b["type"] == "primary"]),
                "overlap": len([b for b in batches if b["type"] == "overlap"]),
                "total": len(batches),
                "batches": batches
            }
            
            # Save individual assignment file
            ann_file = output_path / f"{annotator}_assignment.json"
            with open(ann_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "annotator": annotator,
                    "batches": batches
                }, f, ensure_ascii=False, indent=2)
        
        # Save summary
        summary_file = output_path / "distribution_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"Saved assignments to {output_path}")
    
    return assignments


def main():
    parser = argparse.ArgumentParser(description="Distribute batches among annotators")
    parser.add_argument("--input", required=True, help="Batches directory")
    parser.add_argument("--annotators", type=int, required=True, help="Number of annotators")
    parser.add_argument("--overlap", type=float, default=0.2, help="Overlap ratio for IAA")
    parser.add_argument("--output", help="Output directory for assignments")
    
    args = parser.parse_args()
    
    output = args.output or f"{args.input}/assignments"
    
    assignments = distribute_batches(
        args.input,
        args.annotators,
        args.overlap,
        output
    )
    
    print(f"\nDistribution Summary:")
    for annotator, batches in assignments.items():
        primary = len([b for b in batches if b["type"] == "primary"])
        overlap = len([b for b in batches if b["type"] == "overlap"])
        print(f"  {annotator}: {primary} primary + {overlap} overlap = {len(batches)} total")


if __name__ == "__main__":
    main()
```

---

# PART 2: MODIFICATIONS TO EXISTING PROJECT_PLAN.md

## 2.1 Update Section 1.2 (Key Deliverables)

**Location:** Line 29-37

**REPLACE WITH:**

```markdown
## 1.2 Key Deliverables
| Deliverable | Description | Pilot Target | Production Target |
|-------------|-------------|--------------|-------------------|
| Gold Dataset | Fully validated, reviewer-approved annotations | 500 spans | **4,000+ spans** |
| Silver Dataset | High-confidence annotations meeting ESS threshold | 1,000 spans | **10,000+ spans** |
| Research Dataset | All annotations including disputed | 2,500 spans | **20,000+ spans** |
| **Quran Coverage** | Ayat annotated | 500 ayat | **6,236 ayat (100%)** |
| **Tafsir Integration** | Structured tafsir database + lookup tools | Manual | **5+ sources, DB + API** |
| Coding Manual | Comprehensive annotator training guide | v1.0 | **v2.0** |
| API/Tools | Export tools, validation scripts, graph builder | Basic | **Production-ready** |
| Publications | Academic papers documenting methodology | Draft | **2-3 papers** |
```

---

## 2.2 Update Section 1.3 (Timeline Summary)

**Location:** Line 39-48

**REPLACE WITH:**

```markdown
## 1.3 Timeline Summary
| Phase | Duration | Milestone |
|-------|----------|-----------|
| Phase 0: Setup | Weeks 1-2 | Repository ready, tools configured |
| Phase 1: Self-Calibration | Weeks 3-4 | Personal baseline established |
| Phase 2: Micro-Pilot | Weeks 5-8 | 100 spans, IAA measured |
| Phase 3: Full Pilot | Weeks 9-16 | 500 spans, Gold v0.1 released |
| **Phase 4: Tafsir Integration** | **Weeks 17-24** | **Tafsir DB + lookup tools** |
| **Phase 5: Scale-Up** | **Weeks 25-40** | **3,000 ayat, 10,000 spans** |
| **Phase 6: Full Coverage** | **Weeks 41-60** | **6,236 ayat (100%), 15,000+ spans** |
| **Phase 7: Production** | **Weeks 61-70** | **Full release, API live** |
| **Phase 8: Publication** | **Weeks 71-78** | **2-3 papers, v1.0.0 release** |
| Phase 9: Maintenance | Ongoing | Continuous improvement |

**Total Duration: 78 weeks (~18 months)**
```

---

## 2.3 ADD New Section: Phase 4 - Tafsir Integration

**Location:** After current Phase 3 (around line 750)

**INSERT:**

```markdown
---

# 6. PHASE 4: TAFSIR INTEGRATION (Weeks 17-24)

## 6.1 Week 17-18: Tafsir Data Acquisition

### Task 4.1.1: Download Tafsir Sources
```bash
# Create tafsir directory
mkdir -p data/tafsir

# Download from Quran.com API
python src/scripts/download_tafsir.py \
    --sources quran_api \
    --output data/tafsir/

# Download priority tafsir individually
python src/scripts/download_tafsir.py --tafsir ibn_kathir_ar --output data/tafsir/
python src/scripts/download_tafsir.py --tafsir tabari --output data/tafsir/
python src/scripts/download_tafsir.py --tafsir qurtubi --output data/tafsir/
python src/scripts/download_tafsir.py --tafsir saadi --output data/tafsir/
python src/scripts/download_tafsir.py --tafsir jalalayn_ar --output data/tafsir/

git add data/tafsir/
git commit -m "data: add tafsir sources (ibn kathir, tabari, qurtubi, saadi, jalalayn)"
git push origin main
```

### Task 4.1.2: Verify Tafsir Coverage
```bash
# Check download completeness
python -c "
import json
from pathlib import Path

tafsir_dir = Path('data/tafsir')
for f in tafsir_dir.glob('*.json'):
    if f.name != 'tafsir_index.json':
        data = json.load(open(f))
        ayat_count = len(data.get('ayat', {}))
        print(f'{f.name}: {ayat_count} ayat')
"
```

## 6.2 Week 19-20: Database Setup

### Task 4.2.1: Initialize Tafsir Database
```bash
# Set up PostgreSQL (if not already done)
createdb qbm

# Run tafsir schema setup
python src/scripts/setup_tafsir_db.py \
    --host localhost \
    --db qbm \
    --user postgres \
    --password YOUR_PASSWORD

# Load tafsir data into database
for tafsir in ibn_kathir_ar tabari qurtubi saadi jalalayn_ar; do
    python src/scripts/setup_tafsir_db.py \
        --load data/tafsir/${tafsir}.json \
        --source-id TAFSIR_${tafsir^^}
done
```

### Task 4.2.2: Verify Database
```sql
-- Check tafsir coverage
SELECT * FROM tafsir_coverage_summary;

-- Verify all 6,236 ayat have tafsir
SELECT source_id, COUNT(*) as ayat_count 
FROM tafsir_content 
GROUP BY source_id;
```

## 6.3 Week 21-22: Tafsir Lookup Tool

### Task 4.3.1: Test Lookup Tool
```bash
# Test single ayah lookup
python src/scripts/tafsir_lookup.py --surah 2 --ayah 255

# Test range lookup
python src/scripts/tafsir_lookup.py --surah 2 --ayah 255 --end-ayah 257

# Test comparison mode
python src/scripts/tafsir_lookup.py --surah 2 --ayah 255 --compare --json

# Test search
python src/scripts/tafsir_lookup.py --search "الكرسي"
```

### Task 4.3.2: Integrate with Label Studio
```python
# Add to Label Studio annotation interface
# config/label_studio_config.json - add tafsir panel

{
    "panels": {
        "tafsir": {
            "enabled": true,
            "position": "right",
            "sources": ["ibn_kathir_ar", "tabari", "qurtubi"],
            "auto_load": true
        }
    }
}
```

## 6.4 Week 23-24: Annotator Training on Tafsir Usage

### Task 4.4.1: Create Tafsir Consultation Protocol
```markdown
## Tafsir Consultation Protocol

### When to Consult Tafsir
1. **Agent identification** - unclear who is addressed
2. **Behavior classification** - ambiguous action type
3. **Context determination** - need سبب نزول
4. **Disputed interpretations** - multiple valid readings

### Consultation Hierarchy
1. Ibn Kathir (primary - most comprehensive)
2. Al-Tabari (linguistic depth)
3. Al-Qurtubi (fiqh implications)
4. Al-Sa'di (modern clarity)

### Documentation Requirements
- Record which tafsir was consulted
- Note if tafsir influenced decision
- Flag disagreements between sources
```

### Task 4.4.2: Milestone Checkpoint
```bash
# Verify Phase 4 completion
- [ ] 5+ tafsir sources downloaded
- [ ] Database populated with all 6,236 ayat
- [ ] Lookup tool functional
- [ ] Annotators trained on tafsir protocol
- [ ] Label Studio integrated

git tag -a v0.4.0 -m "Phase 4 Complete - Tafsir Integration"
git push origin main --tags
```
```

---

## 2.4 ADD New Section: Phase 6 - Full Quran Coverage

**Location:** After Phase 5 Scale-Up

**INSERT:**

```markdown
---

# 8. PHASE 6: FULL QURAN COVERAGE (Weeks 41-60)

## 8.1 Coverage Strategy

### Target: 6,236 ayat (100%)

| Week Block | Surahs | Ayat Target | Cumulative |
|------------|--------|-------------|------------|
| 41-45 | 29-50 | 1,200 | 4,200 |
| 46-50 | 51-80 | 1,100 | 5,300 |
| 51-55 | 81-100 | 500 | 5,800 |
| 56-60 | 101-114 + gaps | 436 | 6,236 |

## 8.2 Week 41-45: Surahs 29-50

### Task 6.1.1: Prepare Batches
```bash
# Generate batches for surahs 29-50
python src/scripts/select_full_quran.py \
    --surahs 29-50 \
    --output data/processed/batches_week41-45/

# Distribute to annotators
python src/scripts/distribute_batches.py \
    --input data/processed/batches_week41-45/ \
    --annotators 6
```

### Task 6.1.2: Quality Gates
```markdown
## Weekly Quality Check
- [ ] IAA ≥ 0.72 maintained
- [ ] Tafsir consultation logged
- [ ] All spans validated against schema
- [ ] Coverage tracker updated
```

## 8.3 Week 46-50: Surahs 51-80

### Task 6.2.1: Continue Annotation
```bash
# Similar process for surahs 51-80
python src/scripts/select_full_quran.py \
    --surahs 51-80 \
    --output data/processed/batches_week46-50/
```

## 8.4 Week 51-55: Surahs 81-100

### Task 6.3.1: Short Surahs Strategy
```markdown
## Short Surah Annotation Notes
- Many short surahs have dense behavioral content
- May require multiple spans per ayah
- Focus on eschatological themes (akhira behaviors)
```

## 8.5 Week 56-60: Final Coverage + Gap Filling

### Task 6.4.1: Coverage Audit
```bash
# Check for gaps
python src/scripts/coverage_audit.py \
    --annotations data/annotations/ \
    --output reports/coverage_gaps.json

# Fill gaps
python src/scripts/fill_gaps.py \
    --gaps reports/coverage_gaps.json \
    --output data/processed/gap_batches/
```

### Task 6.4.2: Final Coverage Verification
```bash
# Verify 100% coverage
python -c "
from src.scripts.select_full_quran import TOTAL_AYAT
import json

with open('data/processed/full_quran_batches/coverage_tracker.json') as f:
    tracker = json.load(f)

annotated = sum(s['annotated'] for s in tracker['surahs'].values())
print(f'Annotated: {annotated}/{TOTAL_AYAT} ({100*annotated/TOTAL_AYAT:.1f}%)')
"
```

### Task 6.4.3: Phase 6 Milestone
```bash
# Commit and tag
git add data/annotations/
git commit -m "milestone: 100% Quran coverage achieved"
git tag -a v0.6.0 -m "Phase 6 Complete - Full Quran Coverage"
git push origin main --tags
```
```

---

## 2.5 Update Section 12 (Resource Requirements)

**Location:** Line 2137-2166

**REPLACE WITH:**

```markdown
# 12. RESOURCE REQUIREMENTS

## 12.1 Human Resources

| Role | Count | Hours/Week | Duration | Notes |
|------|-------|------------|----------|-------|
| Project Lead | 1 | 20 | Full project | You (Salim) |
| Senior Annotator | 3 | 15 | Weeks 3-70 | Islamic studies background |
| Junior Annotator | 6 | 10 | Weeks 17-70 | Trained on coding manual |
| Tafsir Consultant | 1 | 5 | Weeks 17-70 | Scholar for edge cases |
| Reviewer | 2 | 10 | Weeks 9-70 | Final quality gate |
| Developer | 1 | 15 | Weeks 1-70 | API, pipeline, tools |

## 12.2 Infrastructure

| Item | Cost/Month | Duration | Total |
|------|------------|----------|-------|
| Cloud hosting (API) | $100 | 18 months | $1,800 |
| Database (PostgreSQL) | $50 | 18 months | $900 |
| Label Studio hosting | $50 | 18 months | $900 |
| Git LFS storage | $20 | 18 months | $360 |
| Backup storage | $30 | 18 months | $540 |
| **Infrastructure Total** | | | **$4,500** |

## 12.3 Personnel Costs

| Role | Rate/Hour | Hours | Total |
|------|-----------|-------|-------|
| Senior Annotators (3) | $25 | 3 × 15 × 68 weeks | $76,500 |
| Junior Annotators (6) | $15 | 6 × 10 × 54 weeks | $48,600 |
| Tafsir Consultant | $50 | 5 × 54 weeks | $13,500 |
| Reviewers (2) | $30 | 2 × 10 × 62 weeks | $37,200 |
| Developer | $40 | 15 × 70 weeks | $42,000 |
| **Personnel Total** | | | **$217,800** |

## 12.4 Total Budget Estimate (Full Project)

| Category | Amount (USD) |
|----------|--------------|
| Personnel | $217,800 |
| Infrastructure | $4,500 |
| Software/Tools | $3,000 |
| Contingency (15%) | $33,795 |
| **TOTAL** | **$259,095** |

### Budget Notes
- Can be reduced significantly with volunteer annotators
- Academic partnerships can offset costs
- Phased funding approach recommended
- Minimum viable: $80,000 (reduced team, longer timeline)
```

---

## 2.6 Update Section 13 (Success Metrics)

**Location:** Line 2170-2189

**REPLACE WITH:**

```markdown
# 13. SUCCESS METRICS

## 13.1 Quantitative

| Metric | Pilot Target | Scale Target | Production Target |
|--------|--------------|--------------|-------------------|
| Gold spans | 500 | 2,000 | **4,000+** |
| Silver spans | 1,000 | 5,000 | **10,000+** |
| Research spans | 2,500 | 10,000 | **20,000+** |
| **Ayat coverage** | 500 | 3,000 | **6,236 (100%)** |
| IAA (Cohen's κ) | ≥ 0.70 | ≥ 0.72 | **≥ 0.75** |
| Surahs | 20+ | 60+ | **114 (100%)** |
| Behavior concepts | 30+ | 50+ | **80+** |
| **Tafsir sources** | Manual | 3 | **5+** |
| **Tafsir consultations** | N/A | 50% | **80%** |

## 13.2 Qualitative

- [ ] Coding manual stable (no major revisions in 8 weeks)
- [ ] Annotator satisfaction (survey score ≥ 4/5)
- [ ] Academic endorsement (3+ scholars review)
- [ ] **Tafsir integration validated by Islamic studies faculty**
- [ ] Publication acceptance (2+ papers submitted)
- [ ] User adoption (API users, downloads)
- [ ] **Complete Quran behavioral map published**

## 13.3 Milestone Checkpoints

| Phase | Gate | Criteria |
|-------|------|----------|
| 0-1 | Setup Complete | Tools configured |
| 2-3 | Pilot Pass | IAA ≥ 0.70, 500 spans |
| **4** | **Tafsir Ready** | **5 sources, DB live** |
| **5** | **Scale Pass** | **3,000 ayat, 10,000 spans** |
| **6** | **Full Coverage** | **6,236 ayat (100%)** |
| **7** | **Production** | **API live, docs complete** |
| **8** | **Publication** | **2+ papers submitted** |
```

---

# PART 3: NEW REQUIREMENTS.TXT ADDITIONS

**Location:** Line 196-232 (requirements.txt section)

**ADD these dependencies:**

```text
# Tafsir processing
aiohttp>=3.8.0          # Async HTTP for bulk download
tqdm>=4.65.0            # Progress bars
arabic-reshaper>=3.0.0  # Arabic text processing

# Database (expanded)
asyncpg>=0.28.0         # Async PostgreSQL
alembic>=1.11.0         # Database migrations

# Full-text search
elasticsearch>=8.0.0    # Optional: advanced tafsir search

# Export formats
markdown>=3.4.0         # Markdown export
jinja2>=3.1.0           # Template rendering
```

---

# PART 4: FOLDER STRUCTURE ADDITIONS

**Location:** Line 69-108 (folder structure section)

**ADD these directories:**

```bash
mkdir -p {data/tafsir,data/batches,reports/coverage,src/scripts/tafsir}

# Updated structure:
quranic-behavior-matrix/
├── data/
│   ├── raw/                    # Original Quran text
│   ├── processed/              # Pilot selections
│   ├── annotations/            # Annotator outputs
│   ├── exports/                # Final datasets
│   ├── tafsir/                 # ← NEW: Tafsir JSON files
│   │   ├── ibn_kathir_ar.json
│   │   ├── tabari.json
│   │   ├── qurtubi.json
│   │   ├── saadi.json
│   │   ├── jalalayn_ar.json
│   │   └── tafsir_index.json
│   └── batches/                # ← NEW: Full Quran batches
│       ├── BATCH_0001.json
│       ├── ...
│       ├── batch_index.json
│       └── coverage_tracker.json
├── src/
│   ├── scripts/
│   │   ├── download_tafsir.py      # ← NEW
│   │   ├── setup_tafsir_db.py      # ← NEW
│   │   ├── tafsir_lookup.py        # ← NEW
│   │   ├── select_full_quran.py    # ← NEW
│   │   ├── coverage_audit.py       # ← NEW
│   │   └── ... (existing scripts)
│   ├── validation/
│   └── api/
├── reports/
│   ├── coverage/               # ← NEW: Coverage reports
│   │   ├── weekly_coverage.json
│   │   └── gap_analysis.json
│   └── iaa/
└── config/
    ├── tafsir_sources.json     # ← NEW: Tafsir metadata
    └── ... (existing configs)
```

---

# PART 5: SUMMARY CHECKLIST FOR AI CODER

## Files to CREATE:

| File | Location | Purpose |
|------|----------|---------|
| `download_tafsir.py` | `src/scripts/` | Download tafsir from APIs |
| `setup_tafsir_db.py` | `src/scripts/` | PostgreSQL schema for tafsir |
| `tafsir_lookup.py` | `src/scripts/` | Query tool for annotators |
| `select_full_quran.py` | `src/scripts/` | Batch all 6,236 ayat |
| `coverage_audit.py` | `src/scripts/` | Check annotation gaps |
| `tafsir_sources.json` | `config/` | Tafsir metadata config |

## Sections to ADD to PROJECT_PLAN.md:

| Section | After | Content |
|---------|-------|---------|
| Phase 4: Tafsir Integration | Phase 3 | Weeks 17-24, tafsir setup |
| Phase 6: Full Coverage | Phase 5 | Weeks 41-60, 6,236 ayat |
| Phase 7: Production | Phase 6 | Weeks 61-70, API + docs |
| Phase 8: Publication | Phase 7 | Weeks 71-78, papers |

## Sections to UPDATE in PROJECT_PLAN.md:

| Section | Change |
|---------|--------|
| 1.2 Key Deliverables | Add 100% coverage, tafsir targets |
| 1.3 Timeline | Extend to 78 weeks |
| 12. Resources | Expand team, budget to $259K |
| 13. Success Metrics | Add full coverage metrics |

## Data to DOWNLOAD:

| Source | Files | Method |
|--------|-------|--------|
| Quran.com API | 5 tafsir × 6,236 ayat | `download_tafsir.py` |
| GitHub | tafsir_api repo | Clone/download |
| Shamela (optional) | Full tafsir DB | Manual export |

---

# EXECUTION ORDER

```
1. Create new scripts (Part 1)
2. Update PROJECT_PLAN.md (Part 2)
3. Update requirements.txt (Part 3)
4. Create folder structure (Part 4)
5. Download tafsir data
6. Set up database
7. Test lookup tool
8. Begin Phase 4
```

---

# APPENDIX A: TAFSIR SOURCE PRIORITY

| Priority | Source | Arabic Name | Methodology | Best For |
|----------|--------|-------------|-------------|----------|
| 1 | Ibn Kathir | تفسير ابن كثير | bil-ma'thur | Comprehensive, hadith-based |
| 2 | Al-Tabari | جامع البيان | bil-ma'thur | Linguistic analysis |
| 3 | Al-Qurtubi | الجامع لأحكام القرآن | mixed | Fiqh implications |
| 4 | Al-Sa'di | تيسير الكريم الرحمن | bil-ra'y | Modern clarity |
| 5 | Al-Jalalayn | تفسير الجلالين | concise | Quick reference |

---

# APPENDIX B: PHASE DEPENDENCY DIAGRAM

```
Phase 0-3 (Pilot)          Phase 4 (Tafsir)         Phase 5-6 (Scale)
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│ Setup           │       │ Download Tafsir │       │ Expand Team     │
│ Calibration     │──────▶│ Setup Database  │──────▶│ Bulk Annotation │
│ Micro-Pilot     │       │ Lookup Tools    │       │ Full Coverage   │
│ Full Pilot      │       │ Train Annotators│       │ Gap Filling     │
└─────────────────┘       └─────────────────┘       └─────────────────┘
        │                         │                         │
        │                         │                         │
        ▼                         ▼                         ▼
   IAA ≥ 0.70              5+ Tafsir Sources         6,236 Ayat (100%)
   500 Spans               DB + Lookup Live          15,000+ Spans


Phase 7 (Production)       Phase 8 (Publication)
┌─────────────────┐       ┌─────────────────┐
│ API Development │       │ Methodology Paper│
│ Documentation   │──────▶│ Dataset Paper    │
│ Testing         │       │ Conference       │
│ Launch          │       │ v1.0.0 Release   │
└─────────────────┘       └─────────────────┘
        │                         │
        ▼                         ▼
   API Live                  2+ Papers
   Docs Complete             Academic Recognition
```

---

# APPENDIX C: RISK MITIGATION FOR SCALE-UP

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Annotator burnout | High | Medium | Rotate batches, limit daily quota |
| IAA degradation at scale | High | Medium | Weekly calibration sessions |
| Tafsir API rate limits | Medium | High | Implement caching, async downloads |
| Database performance | Medium | Low | Index optimization, partitioning |
| Budget overrun | High | Medium | Phased funding, volunteer recruitment |
| Scope creep | Medium | High | Strict phase gates, change control |

---

# APPENDIX D: QUALITY ASSURANCE CHECKLIST

## Per-Batch Quality Gate
- [ ] All spans pass schema validation
- [ ] All vocabulary IDs are valid
- [ ] Agent type present for all spans
- [ ] At least one assertion per span
- [ ] Normative layer complete
- [ ] Heart domains specified when ORG_HEART used
- [ ] Tafsir consultation logged (Phase 4+)

## Weekly Quality Gate
- [ ] IAA calculated for overlap batches
- [ ] IAA ≥ threshold (0.70 pilot, 0.72 scale, 0.75 production)
- [ ] Coverage tracker updated
- [ ] Disputed spans adjudicated
- [ ] Progress report generated

## Phase Quality Gate
- [ ] All weekly gates passed
- [ ] Milestone targets met
- [ ] Documentation updated
- [ ] Git tag created
- [ ] Stakeholder sign-off

---

*Document Version: 2.0*
*Created: December 2025*
*Last Updated: December 2025*
*Purpose: End-to-End QBM Project Completion Guide*
*Maintainer: QBM Project Team*
