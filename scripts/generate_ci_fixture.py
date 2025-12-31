#!/usr/bin/env python3
"""
Generate minimal CI test fixture from existing JSON SSOT.

This creates a small, deterministic dataset that CI can always run with,
eliminating dependency on large corpora files not committed to git.

Fixture includes:
- ~100 key verses covering test behaviors (patience, gratitude, prayer, etc.)
- Minimal tafsir chunks for those verses from available sources
- All data derived from committed JSON files, not XML
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Set

# Key verses for testing - covers main behaviors used in tests
KEY_VERSES = [
    # Patience (الصبر) - heavily tested
    (2, 45), (2, 153), (2, 155), (2, 156), (2, 157), (3, 200), (103, 3),
    # Gratitude (الشكر)
    (2, 152), (14, 7), (31, 12),
    # Prayer (الصلاة)
    (2, 3), (2, 43), (2, 45), (4, 103), (29, 45),
    # Truthfulness (الصدق)
    (9, 119), (33, 35),
    # Envy (الحسد)
    (113, 5), (4, 54),
    # Remembrance (الذكر)
    (2, 152), (13, 28), (33, 41),
    # Disbelief (الكفر)
    (2, 6), (2, 7), (4, 136),
    # Faith (الإيمان)
    (2, 3), (2, 4), (2, 285), (49, 15),
    # Heart states
    (2, 10), (2, 74), (3, 159), (22, 46), (26, 89),
    # Surah Al-Fatiha (complete - for surah ref tests)
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
    # Surah Al-Ikhlas (complete - for surah ref tests)
    (112, 1), (112, 2), (112, 3), (112, 4),
    # Surah An-Nas (complete - for surah ref tests)
    (114, 1), (114, 2), (114, 3), (114, 4), (114, 5), (114, 6),
    # Additional test verses
    (59, 23), (59, 24),  # Names of Allah
    (2, 255),  # Ayat al-Kursi
    (36, 1), (36, 2), (36, 3),  # Ya-Sin opening
]

def load_quran_json(path: Path) -> Dict:
    """Load Quran from JSON SSOT."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_fixture_verses(quran_data: Dict, key_verses: List[tuple]) -> List[Dict]:
    """Extract only the key verses needed for fixture."""
    verses = []
    key_set = set(key_verses)
    
    # Handle different JSON structures
    if "surahs" in quran_data:
        for surah in quran_data["surahs"]:
            # Handle both 'surah' (number) and 'number' keys
            surah_num = surah.get("surah") or surah.get("number") or surah.get("surah_number")
            # Handle both 'ayat' and 'ayahs' keys
            ayat_list = surah.get("ayat", []) or surah.get("ayahs", []) or surah.get("verses", [])
            for ayah in ayat_list:
                ayah_num = ayah.get("ayah") or ayah.get("number") or ayah.get("verse_number")
                if (surah_num, ayah_num) in key_set:
                    verses.append({
                        "surah": surah_num,
                        "ayah": ayah_num,
                        "text": ayah.get("text") or ayah.get("text_uthmani", ""),
                        "key": f"{surah_num}:{ayah_num}"
                    })
    elif "verses" in quran_data:
        for v in quran_data["verses"]:
            surah_num = v.get("surah") or v.get("surah_number")
            ayah_num = v.get("ayah") or v.get("verse_number")
            if (surah_num, ayah_num) in key_set:
                verses.append({
                    "surah": surah_num,
                    "ayah": ayah_num,
                    "text": v.get("text") or v.get("text_uthmani", ""),
                    "key": f"{surah_num}:{ayah_num}"
                })
    else:
        # Flat structure with verse keys
        for key, text in quran_data.items():
            if ":" in str(key):
                parts = str(key).split(":")
                if len(parts) == 2:
                    try:
                        surah_num, ayah_num = int(parts[0]), int(parts[1])
                        if (surah_num, ayah_num) in key_set:
                            verses.append({
                                "surah": surah_num,
                                "ayah": ayah_num,
                                "text": text if isinstance(text, str) else text.get("text", ""),
                                "key": f"{surah_num}:{ayah_num}"
                            })
                    except ValueError:
                        continue
    
    return sorted(verses, key=lambda x: (x["surah"], x["ayah"]))

def generate_minimal_tafsir(verses: List[Dict]) -> List[Dict]:
    """
    Generate minimal tafsir chunks for fixture verses.
    
    CRITICAL: Must include ALL 7 sources required by StratifiedTafsirRetriever.
    """
    tafsir_chunks = []
    
    # All 7 tafsir sources required by StratifiedTafsirRetriever
    sources = [
        "ibn_kathir", "tabari", "qurtubi", "saadi", 
        "jalalayn", "baghawi", "muyassar"
    ]
    
    for verse in verses:
        key = verse["key"]
        for source in sources:  # ALL 7 sources for each verse
            tafsir_chunks.append({
                "verse_key": key,
                "surah": verse["surah"],
                "ayah": verse["ayah"],
                "source": source,
                "text": f"[Fixture tafsir for {key} from {source}] This is a minimal fixture entry for CI testing.",
                "chunk_id": f"{source}_{key.replace(':', '_')}_0",
                "char_start": 0,
                "char_end": 100
            })
    
    return tafsir_chunks

def main():
    project_root = Path(__file__).parent.parent
    fixture_dir = project_root / "data" / "test_fixtures" / "fixture_v1"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to load from JSON SSOT
    quran_json_path = project_root / "data" / "quran" / "uthmani_hafs_v1.tok_v1.json"
    
    if quran_json_path.exists():
        print(f"Loading Quran from {quran_json_path}")
        quran_data = load_quran_json(quran_json_path)
        verses = extract_fixture_verses(quran_data, KEY_VERSES)
        print(f"Extracted {len(verses)} verses from JSON SSOT")
    else:
        print(f"WARNING: {quran_json_path} not found, generating stub verses")
        verses = []
        for surah, ayah in KEY_VERSES:
            verses.append({
                "surah": surah,
                "ayah": ayah,
                "text": f"[Fixture verse {surah}:{ayah}]",
                "key": f"{surah}:{ayah}"
            })
    
    # Deduplicate
    seen = set()
    unique_verses = []
    for v in verses:
        if v["key"] not in seen:
            seen.add(v["key"])
            unique_verses.append(v)
    verses = unique_verses
    
    # Write quran fixture
    quran_fixture_path = fixture_dir / "quran_verses.jsonl"
    with open(quran_fixture_path, 'w', encoding='utf-8') as f:
        for verse in verses:
            f.write(json.dumps(verse, ensure_ascii=False) + '\n')
    print(f"Wrote {len(verses)} verses to {quran_fixture_path}")
    
    # Generate and write tafsir fixture
    tafsir_chunks = generate_minimal_tafsir(verses)
    tafsir_fixture_path = fixture_dir / "tafsir_chunks.jsonl"
    with open(tafsir_fixture_path, 'w', encoding='utf-8') as f:
        for chunk in tafsir_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    print(f"Wrote {len(tafsir_chunks)} tafsir chunks to {tafsir_fixture_path}")
    
    # Write fixture manifest
    manifest = {
        "version": "1.0",
        "description": "Minimal CI test fixture - deterministic, committed to repo",
        "verses_count": len(verses),
        "tafsir_chunks_count": len(tafsir_chunks),
        "tafsir_sources": ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn", "baghawi", "muyassar"],
        "key_behaviors_covered": [
            "patience", "gratitude", "prayer", "truthfulness",
            "envy", "remembrance", "disbelief", "faith"
        ]
    }
    manifest_path = fixture_dir / "manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Wrote manifest to {manifest_path}")
    
    print("\n✓ CI fixture generated successfully")
    print(f"  Total size: {sum(f.stat().st_size for f in fixture_dir.glob('*')) / 1024:.1f} KB")

if __name__ == "__main__":
    main()
