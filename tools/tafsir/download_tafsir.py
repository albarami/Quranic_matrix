#!/usr/bin/env python3
"""
Download tafsir data from Quran.com API.

Supported tafsir sources:
- ibn-kathir (Arabic)
- al-tabari (Arabic)
- al-qurtubi (Arabic)
- al-saddi (Arabic) - Sa'di
- al-jalalayn (Arabic)
"""

import argparse
import json
import os
import time
from pathlib import Path
import requests

# Quran.com API base URL
API_BASE = "https://api.quran.com/api/v4"

# Tafsir resource IDs from Quran.com API (verified)
TAFSIR_RESOURCES = {
    "ibn_kathir": {"id": 14, "name": "Tafsir Ibn Kathir", "language": "ar"},
    "tabari": {"id": 15, "name": "Tafsir al-Tabari", "language": "ar"},
    "qurtubi": {"id": 90, "name": "Tafsir al-Qurtubi", "language": "ar"},
    "saadi": {"id": 91, "name": "Tafsir al-Sa'di", "language": "ar"},
    "muyassar": {"id": 16, "name": "Tafsir Muyassar", "language": "ar"},
    "baghawi": {"id": 94, "name": "Tafsir al-Baghawi", "language": "ar"},
    "tantawi": {"id": 93, "name": "Al-Tafsir al-Wasit (Tantawi)", "language": "ar"},
    # English
    "ibn_kathir_en": {"id": 169, "name": "Ibn Kathir (Abridged)", "language": "en"},
}

# Surah info (number of ayat per surah)
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


def get_tafsir_for_ayah(tafsir_id: int, surah: int, ayah: int, retries: int = 3) -> dict:
    """Fetch tafsir for a single ayah with retry logic."""
    url = f"{API_BASE}/tafsirs/{tafsir_id}/by_ayah/{surah}:{ayah}"
    
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            tafsir_data = data.get("tafsir", {})
            return {
                "text": tafsir_data.get("text", ""),
                "resource_name": tafsir_data.get("resource_name", ""),
            }
        except requests.RequestException as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            print(f"  Error fetching {surah}:{ayah}: {e}", flush=True)
            return None
    return None


def download_tafsir(tafsir_key: str, output_dir: Path, delay: float = 0.5):
    """Download complete tafsir for all 6236 ayat."""
    import sys
    
    if tafsir_key not in TAFSIR_RESOURCES:
        print(f"Unknown tafsir: {tafsir_key}")
        print(f"Available: {list(TAFSIR_RESOURCES.keys())}")
        return
    
    resource = TAFSIR_RESOURCES[tafsir_key]
    tafsir_id = resource["id"]
    
    output_file = output_dir / f"{tafsir_key}.ar.jsonl"
    
    print(f"Downloading {resource['name']} (ID: {tafsir_id})", flush=True)
    print(f"Output: {output_file}", flush=True)
    
    # Check for existing progress
    existing_refs = set()
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    ref = f"{rec['reference']['surah']}:{rec['reference']['ayah']}"
                    existing_refs.add(ref)
        print(f"Resuming from {len(existing_refs)} existing records")
    
    total_ayat = sum(SURAH_AYAT)
    downloaded = len(existing_refs)
    
    with open(output_file, "a", encoding="utf-8") as f:
        for surah_num in range(1, 115):
            num_ayat = SURAH_AYAT[surah_num - 1]
            
            for ayah_num in range(1, num_ayat + 1):
                ref = f"{surah_num}:{ayah_num}"
                
                if ref in existing_refs:
                    continue
                
                tafsir_data = get_tafsir_for_ayah(tafsir_id, surah_num, ayah_num)
                
                if tafsir_data:
                    record = {
                        "tafsir_id": tafsir_key,
                        "reference": {
                            "surah": surah_num,
                            "ayah": ayah_num
                        },
                        "text_ar": tafsir_data["text"],
                        "resource_name": tafsir_data["resource_name"]
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    downloaded += 1
                
                if downloaded % 50 == 0:
                    print(f"  Progress: {downloaded}/{total_ayat} ({100*downloaded/total_ayat:.1f}%)", flush=True)
                
                time.sleep(delay)
    
    print(f"Complete: {downloaded} ayat downloaded to {output_file}", flush=True)


def list_available_tafsirs():
    """List available tafsir resources from API."""
    url = f"{API_BASE}/resources/tafsirs"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        print("Available Tafsir Resources:")
        print("-" * 60)
        for tafsir in data.get("tafsirs", []):
            print(f"  ID: {tafsir['id']:4d} | {tafsir['name']} ({tafsir.get('language_name', 'unknown')})")
    except requests.RequestException as e:
        print(f"Error fetching tafsir list: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download tafsir data from Quran.com API")
    parser.add_argument("--tafsir", choices=list(TAFSIR_RESOURCES.keys()) + ["all"],
                        help="Tafsir to download (or 'all')")
    parser.add_argument("--output", default="data/tafsir", help="Output directory")
    parser.add_argument("--delay", type=float, default=0.3, help="Delay between API calls (seconds)")
    parser.add_argument("--list", action="store_true", help="List available tafsir resources")
    args = parser.parse_args()
    
    if args.list:
        list_available_tafsirs()
        return
    
    if not args.tafsir:
        parser.print_help()
        return
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.tafsir == "all":
        for tafsir_key in ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"]:
            download_tafsir(tafsir_key, output_dir, args.delay)
    else:
        download_tafsir(args.tafsir, output_dir, args.delay)


if __name__ == "__main__":
    main()
