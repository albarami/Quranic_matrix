#!/usr/bin/env python3
"""
Download tafsir data from spa5k/tafsir_api GitHub CDN.
This is more reliable than the Quran.com API.
"""

import json
import time
from pathlib import Path
import requests

CDN_BASE = "https://cdn.jsdelivr.net/gh/spa5k/tafsir_api@main/tafsir"

# Available tafsir sources on CDN
TAFSIR_SOURCES = {
    "ibn_kathir": "ar-tafsir-ibn-kathir",
    "tabari": "ar-tafsir-al-tabari", 
    "saadi": "ar-tafseer-al-saddi",
    "muyassar": "ar-tafsir-muyassar",
    "baghawi": "ar-tafsir-al-baghawi",
    "qurtubi": "ar-tafsir-al-qurtubi",
}

# Surah ayat counts
SURAH_AYAT = [
    7, 286, 200, 176, 120, 165, 206, 75, 129, 109,
    123, 111, 43, 52, 99, 128, 111, 110, 98, 135,
    112, 78, 118, 64, 77, 227, 93, 88, 69, 60,
    34, 30, 73, 54, 45, 83, 182, 88, 75, 85,
    54, 53, 89, 59, 37, 35, 38, 29, 18, 45,
    60, 49, 62, 55, 78, 96, 29, 22, 24, 13,
    14, 11, 11, 18, 12, 12, 30, 52, 52, 44,
    28, 28, 20, 56, 40, 31, 50, 40, 46, 42,
    29, 19, 36, 25, 22, 17, 19, 26, 30, 20,
    15, 21, 11, 8, 8, 19, 5, 8, 8, 11,
    11, 8, 3, 9, 5, 4, 7, 3, 6, 3,
    5, 4, 5, 6
]


def download_tafsir(source_key: str, output_dir: Path, delay: float = 0.1):
    """Download complete tafsir from CDN."""
    if source_key not in TAFSIR_SOURCES:
        print(f"Unknown source: {source_key}")
        print(f"Available: {list(TAFSIR_SOURCES.keys())}")
        return
    
    cdn_name = TAFSIR_SOURCES[source_key]
    output_file = output_dir / f"{source_key}.ar.jsonl"
    
    print(f"Downloading {source_key} from CDN...", flush=True)
    print(f"Output: {output_file}", flush=True)
    
    # Load existing records
    existing = set()
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    key = f"{rec['reference']['surah']}:{rec['reference']['ayah']}"
                    existing.add(key)
        print(f"Resuming from {len(existing)} existing records", flush=True)
    
    total = sum(SURAH_AYAT)
    downloaded = len(existing)
    errors = 0
    
    with open(output_file, "a", encoding="utf-8") as f:
        for surah in range(1, 115):
            num_ayat = SURAH_AYAT[surah - 1]
            
            for ayah in range(1, num_ayat + 1):
                key = f"{surah}:{ayah}"
                if key in existing:
                    continue
                
                url = f"{CDN_BASE}/{cdn_name}/{surah}/{ayah}.json"
                
                try:
                    r = requests.get(url, timeout=30)
                    if r.ok:
                        data = r.json()
                        record = {
                            "tafsir_id": source_key,
                            "reference": {"surah": surah, "ayah": ayah},
                            "text_ar": data.get("text", ""),
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        f.flush()
                        downloaded += 1
                    else:
                        errors += 1
                        if errors < 10:
                            print(f"  Error {surah}:{ayah}: {r.status_code}", flush=True)
                except Exception as e:
                    errors += 1
                    if errors < 10:
                        print(f"  Error {surah}:{ayah}: {e}", flush=True)
                
                if downloaded % 100 == 0:
                    print(f"  Progress: {downloaded}/{total} ({100*downloaded/total:.1f}%)", flush=True)
                
                time.sleep(delay)
    
    print(f"Complete: {downloaded} ayat, {errors} errors", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=list(TAFSIR_SOURCES.keys()), default="ibn_kathir")
    parser.add_argument("--output", default="data/tafsir")
    parser.add_argument("--delay", type=float, default=0.05)
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    download_tafsir(args.source, output_dir, args.delay)
