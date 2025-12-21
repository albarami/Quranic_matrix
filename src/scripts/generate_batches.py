#!/usr/bin/env python3
"""
Generate annotation batches for QBM scale-up phase.

Usage:
    python src/scripts/generate_batches.py --surahs 1-14 --output data/batches/week29-32/
    python src/scripts/generate_batches.py --surahs 15-28 --output data/batches/week33-36/
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

# Surah metadata
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

SURAH_NAMES = [
    "الفاتحة", "البقرة", "آل عمران", "النساء", "المائدة", "الأنعام", "الأعراف",
    "الأنفال", "التوبة", "يونس", "هود", "يوسف", "الرعد", "إبراهيم", "الحجر",
    "النحل", "الإسراء", "الكهف", "مريم", "طه", "الأنبياء", "الحج", "المؤمنون",
    "النور", "الفرقان", "الشعراء", "النمل", "القصص", "العنكبوت", "الروم",
    "لقمان", "السجدة", "الأحزاب", "سبأ", "فاطر", "يس", "الصافات", "ص",
    "الزمر", "غافر", "فصلت", "الشورى", "الزخرف", "الدخان", "الجاثية",
    "الأحقاف", "محمد", "الفتح", "الحجرات", "ق", "الذاريات", "الطور",
    "النجم", "القمر", "الرحمن", "الواقعة", "الحديد", "المجادلة", "الحشر",
    "الممتحنة", "الصف", "الجمعة", "المنافقون", "التغابن", "الطلاق", "التحريم",
    "الملك", "القلم", "الحاقة", "المعارج", "نوح", "الجن", "المزمل",
    "المدثر", "القيامة", "الإنسان", "المرسلات", "النبأ", "النازعات", "عبس",
    "التكوير", "الانفطار", "المطففين", "الانشقاق", "البروج", "الطارق", "الأعلى",
    "الغاشية", "الفجر", "البلد", "الشمس", "الليل", "الضحى", "الشرح",
    "التين", "العلق", "القدر", "البينة", "الزلزلة", "العاديات", "القارعة",
    "التكاثر", "العصر", "الهمزة", "الفيل", "قريش", "الماعون", "الكوثر",
    "الكافرون", "النصر", "المسد", "الإخلاص", "الفلق", "الناس"
]


def parse_surah_range(range_str: str) -> List[int]:
    """Parse surah range like '1-14' or '1,2,5-10'."""
    surahs = []
    for part in range_str.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            surahs.extend(range(start, end + 1))
        else:
            surahs.append(int(part))
    return sorted(set(surahs))


def generate_ayat_list(surahs: List[int]) -> List[Dict]:
    """Generate list of all ayat for given surahs."""
    ayat = []
    for surah in surahs:
        if surah < 1 or surah > 114:
            continue
        num_ayat = SURAH_AYAT[surah - 1]
        surah_name = SURAH_NAMES[surah - 1]
        for ayah in range(1, num_ayat + 1):
            ayat.append({
                "surah": surah,
                "ayah": ayah,
                "surah_name": surah_name,
                "reference": f"{surah}:{ayah}"
            })
    return ayat


def create_batches(ayat: List[Dict], batch_size: int = 50) -> List[Dict]:
    """Split ayat into batches."""
    batches = []
    for i in range(0, len(ayat), batch_size):
        batch_ayat = ayat[i:i + batch_size]
        batch_id = f"batch_{len(batches) + 1:03d}"
        
        # Get surah range for this batch
        surahs_in_batch = sorted(set(a["surah"] for a in batch_ayat))
        surah_range = f"{min(surahs_in_batch)}-{max(surahs_in_batch)}" if len(surahs_in_batch) > 1 else str(surahs_in_batch[0])
        
        batches.append({
            "batch_id": batch_id,
            "surah_range": surah_range,
            "ayat_count": len(batch_ayat),
            "ayat": batch_ayat
        })
    
    return batches


def save_batches(batches: List[Dict], output_dir: Path):
    """Save batches to individual files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    manifest = {
        "total_batches": len(batches),
        "total_ayat": sum(b["ayat_count"] for b in batches),
        "batches": []
    }
    
    for batch in batches:
        batch_file = output_dir / f"{batch['batch_id']}.json"
        with open(batch_file, "w", encoding="utf-8") as f:
            json.dump(batch, f, ensure_ascii=False, indent=2)
        
        manifest["batches"].append({
            "batch_id": batch["batch_id"],
            "file": batch_file.name,
            "surah_range": batch["surah_range"],
            "ayat_count": batch["ayat_count"]
        })
    
    # Save manifest
    manifest_file = output_dir / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Generate annotation batches")
    parser.add_argument("--surahs", required=True, help="Surah range (e.g., 1-14, 15-28)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--batch-size", type=int, default=50, help="Ayat per batch")
    args = parser.parse_args()
    
    surahs = parse_surah_range(args.surahs)
    output_dir = Path(args.output)
    
    print(f"Generating batches for surahs: {surahs[0]}-{surahs[-1]}")
    
    # Generate ayat list
    ayat = generate_ayat_list(surahs)
    print(f"Total ayat: {len(ayat)}")
    
    # Create batches
    batches = create_batches(ayat, args.batch_size)
    print(f"Created {len(batches)} batches")
    
    # Save
    manifest = save_batches(batches, output_dir)
    print(f"\nSaved to: {output_dir}")
    print(f"Manifest: {output_dir / 'manifest.json'}")
    
    # Summary
    print(f"\n{'='*50}")
    print("BATCH SUMMARY")
    print(f"{'='*50}")
    for b in manifest["batches"]:
        print(f"  {b['batch_id']}: Surahs {b['surah_range']} ({b['ayat_count']} ayat)")


if __name__ == "__main__":
    main()
