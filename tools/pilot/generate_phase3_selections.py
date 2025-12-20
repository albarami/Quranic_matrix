#!/usr/bin/env python3
"""
Generate 500+ verse selections for Phase 3 full pilot.
Selects verses with behavioral content from across the Quran.
"""

import json
from pathlib import Path
import random

# Load tokenized Quran
quran_path = Path("data/quran/_incoming/quran_tokenized_full.source.json")
with open(quran_path, encoding="utf-8") as f:
    quran_data = json.load(f)

# Behavioral keywords to prioritize (Arabic roots/patterns indicating behavior)
BEHAVIOR_INDICATORS = [
    "يؤمن", "يكفر", "يصل", "ينفق", "يتق", "يصبر", "يشكر", "يذكر",
    "يعمل", "يفعل", "يقول", "يأمر", "ينهى", "يطيع", "يعصي",
    "آمن", "كفر", "صل", "أنفق", "اتق", "اصبر", "اشكر", "اذكر",
    "المؤمن", "الكافر", "المنافق", "الصالح", "الفاسق", "المتق",
    "الصابر", "الشاكر", "المحسن", "الظالم", "المفسد",
    "تعاون", "تناج", "تواص", "تحاب", "تباغض",
    "بر", "إحسان", "صدق", "أمانة", "عدل", "ظلم", "فساد",
    "قلوب", "نفس", "صدور", "أعين", "آذان", "ألسن",
    "خشية", "خوف", "رجاء", "حب", "بغض", "حسد", "كبر",
    "توكل", "إخلاص", "رياء", "نفاق", "شرك", "توحيد"
]

# Surahs with high behavioral content (prioritize these)
PRIORITY_SURAHS = [
    2,   # Al-Baqarah - comprehensive behavioral guidance
    3,   # Al-Imran - faith and conduct
    4,   # An-Nisa - social/family behaviors
    5,   # Al-Ma'idah - legal/ethical
    6,   # Al-An'am - belief and disbelief
    7,   # Al-A'raf - prophetic narratives with behavioral lessons
    9,   # At-Tawbah - hypocrites and believers
    16,  # An-Nahl - gratitude and ingratitude
    17,  # Al-Isra - ethical commands
    18,  # Al-Kahf - narratives with behavioral themes
    23,  # Al-Mu'minun - believers' characteristics
    24,  # An-Nur - social conduct
    25,  # Al-Furqan - servants of Rahman
    31,  # Luqman - wisdom and conduct
    33,  # Al-Ahzab - Prophet's household conduct
    41,  # Fussilat - good and evil responses
    49,  # Al-Hujurat - social etiquette
    57,  # Al-Hadid - faith and spending
    58,  # Al-Mujadila - social behaviors
    59,  # Al-Hashr - believers vs hypocrites
    60,  # Al-Mumtahina - relations with non-Muslims
    61,  # As-Saff - striving in Allah's cause
    63,  # Al-Munafiqun - hypocrite behaviors
    64,  # At-Taghabun - faith and family
    65,  # At-Talaq - divorce conduct
    66,  # At-Tahrim - household matters
    68,  # Al-Qalam - character
    70,  # Al-Ma'arij - human nature
    76,  # Al-Insan - righteous deeds
    90,  # Al-Balad - moral choices
    103, # Al-Asr - faith and righteous deeds
]

def has_behavioral_content(text_ar):
    """Check if verse likely contains behavioral content."""
    for indicator in BEHAVIOR_INDICATORS:
        if indicator in text_ar:
            return True
    return False

def select_verses(target_count=550):
    """Select verses with behavioral content."""
    selections = []
    seen_refs = set()
    
    # Load existing pilot selections to avoid duplicates
    existing_path = Path("data/pilot/pilot_50_selections.jsonl")
    if existing_path.exists():
        with open(existing_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    seen_refs.add(rec.get("reference"))
    
    # Also check pilot_100_tasks
    pilot100_path = Path("label_studio/pilot_100_tasks.jsonl")
    if pilot100_path.exists():
        with open(pilot100_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    data = rec.get("data", {})
                    ref = f"{data.get('surah')}:{data.get('ayah')}"
                    seen_refs.add(ref)
    
    print(f"Excluding {len(seen_refs)} already-used verses")
    
    # First pass: priority surahs with behavioral indicators
    for surah in quran_data["surahs"]:
        if surah["surah_number"] not in PRIORITY_SURAHS:
            continue
        for ayah in surah["ayat"]:
            ref = f"{surah['surah_number']}:{ayah['ayah_number']}"
            if ref in seen_refs:
                continue
            if has_behavioral_content(ayah["text_ar"]):
                selections.append({
                    "reference": ref,
                    "surah_number": surah["surah_number"],
                    "surah_name": surah["name_ar"],
                    "ayah_number": ayah["ayah_number"],
                    "text_ar": ayah["text_ar"],
                    "token_count": ayah["token_count"],
                    "tokens": ayah["tokens"],
                    "priority": 1
                })
                seen_refs.add(ref)
    
    print(f"After priority surahs with indicators: {len(selections)}")
    
    # Second pass: all surahs with behavioral indicators
    for surah in quran_data["surahs"]:
        if surah["surah_number"] in PRIORITY_SURAHS:
            continue  # Already processed
        for ayah in surah["ayat"]:
            ref = f"{surah['surah_number']}:{ayah['ayah_number']}"
            if ref in seen_refs:
                continue
            if has_behavioral_content(ayah["text_ar"]):
                selections.append({
                    "reference": ref,
                    "surah_number": surah["surah_number"],
                    "surah_name": surah["name_ar"],
                    "ayah_number": ayah["ayah_number"],
                    "text_ar": ayah["text_ar"],
                    "token_count": ayah["token_count"],
                    "tokens": ayah["tokens"],
                    "priority": 2
                })
                seen_refs.add(ref)
    
    print(f"After all surahs with indicators: {len(selections)}")
    
    # Third pass: priority surahs without indicators (still likely behavioral)
    if len(selections) < target_count:
        for surah in quran_data["surahs"]:
            if surah["surah_number"] not in PRIORITY_SURAHS:
                continue
            for ayah in surah["ayat"]:
                ref = f"{surah['surah_number']}:{ayah['ayah_number']}"
                if ref in seen_refs:
                    continue
                # Skip very short verses (likely not behavioral)
                if ayah["token_count"] < 4:
                    continue
                selections.append({
                    "reference": ref,
                    "surah_number": surah["surah_number"],
                    "surah_name": surah["name_ar"],
                    "ayah_number": ayah["ayah_number"],
                    "text_ar": ayah["text_ar"],
                    "token_count": ayah["token_count"],
                    "tokens": ayah["tokens"],
                    "priority": 3
                })
                seen_refs.add(ref)
                if len(selections) >= target_count:
                    break
            if len(selections) >= target_count:
                break
    
    print(f"After priority surahs fill: {len(selections)}")
    
    # Sort by surah:ayah for consistency
    selections.sort(key=lambda x: (x["surah_number"], x["ayah_number"]))
    
    # Trim to target if exceeded
    if len(selections) > target_count:
        # Keep priority 1 and 2, sample from priority 3
        p1_p2 = [s for s in selections if s["priority"] <= 2]
        p3 = [s for s in selections if s["priority"] == 3]
        if len(p1_p2) >= target_count:
            selections = p1_p2[:target_count]
        else:
            needed = target_count - len(p1_p2)
            random.seed(42)  # Reproducible
            p3_sample = random.sample(p3, min(needed, len(p3)))
            selections = p1_p2 + p3_sample
            selections.sort(key=lambda x: (x["surah_number"], x["ayah_number"]))
    
    return selections

# Generate selections
selections = select_verses(550)

# Remove priority field before saving
for s in selections:
    del s["priority"]

# Save to file
output_path = Path("data/pilot/phase3_550_selections.jsonl")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    for s in selections:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"\nGenerated {len(selections)} selections -> {output_path}")

# Show distribution by surah
surah_counts = {}
for s in selections:
    sn = s["surah_number"]
    surah_counts[sn] = surah_counts.get(sn, 0) + 1

print("\nTop 10 surahs by selection count:")
for sn, count in sorted(surah_counts.items(), key=lambda x: -x[1])[:10]:
    print(f"  Surah {sn}: {count} verses")
