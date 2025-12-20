#!/usr/bin/env python3
"""
Expand pilot from 50 to 100 spans for Phase 2 micro-pilot.

Strategic selection criteria:
1. Keep all 50 existing spans (continuity)
2. Add 50 new spans covering:
   - Heart domain examples (spiritual, emotional, cognitive)
   - Negation patterns (nahy, absolute, conditional)
   - Instinctive/automatic actions
   - Multiple agents (hypocrites, disbelievers, prophets)
   - Various speech modes (command, prohibition, informative)
   - Edge cases from gold standards

Priority surahs for behavioral diversity:
- Surah 2 (Al-Baqarah): Comprehensive behavioral content
- Surah 3 (Aal-Imran): Faith, patience, battle ethics
- Surah 4 (An-Nisa): Family, social obligations
- Surah 9 (At-Tawbah): Hypocrisy, repentance
- Surah 17 (Al-Isra): Parent respect, ethics
- Surah 24 (An-Nur): Social conduct, modesty
- Surah 49 (Al-Hujurat): Community ethics
- Surah 63 (Al-Munafiqun): Hypocrisy patterns
"""

import json
from pathlib import Path

# Strategic ayat selection for Phase 2
# Format: (surah, ayah, reason)
STRATEGIC_ADDITIONS = [
    # Heart domain - spiritual
    (2, 74, "Hearts harder than stone - spiritual metaphor"),
    (2, 225, "Hearts and oaths - intention"),
    (3, 159, "Soft heart - leadership quality"),
    (5, 13, "Hard hearts - consequence of breaking covenant"),
    (22, 46, "Hearts in chests that understand"),
    (39, 22, "Expanded heart for Islam"),

    # Heart domain - emotional
    (3, 151, "Terror cast into hearts"),
    (8, 12, "Strike terror into hearts"),
    (59, 2, "Allah cast terror into hearts"),

    # Heart domain - cognitive
    (7, 179, "Hearts that do not understand"),
    (9, 87, "Hearts sealed"),
    (47, 24, "Do they not ponder - hearts locked"),

    # Negation patterns
    (2, 191, "Kill them - conditional permission"),
    (4, 29, "Do not kill yourselves - prohibition"),
    (5, 2, "Do not transgress - prohibition"),
    (17, 31, "Do not kill children - prohibition"),
    (17, 32, "Do not approach zina - prohibition"),
    (17, 33, "Do not kill soul - prohibition"),

    # Hypocrite behaviors
    (2, 14, "When they meet believers vs alone"),
    (4, 142, "Hypocrites deceiving Allah"),
    (9, 67, "Hypocrites enjoin evil"),
    (9, 68, "Promise of hellfire for hypocrites"),
    (63, 1, "Hypocrites testify falsely"),
    (63, 2, "Shields with oaths"),

    # Prophet/messenger behaviors
    (3, 164, "Prophet recites and purifies"),
    (33, 21, "Prophet as excellent example"),
    (68, 4, "Great moral character"),

    # Disbeliever behaviors
    (2, 6, "Sealed hearts - disbelief"),
    (2, 171, "Like cattle - do not understand"),
    (7, 176, "Like dog - panting"),

    # Instinctive/automatic states
    (16, 78, "Born knowing nothing"),
    (19, 4, "Old age weakness"),
    (22, 5, "Stages of human creation"),

    # Social behaviors
    (4, 1, "Fear Allah regarding relatives"),
    (4, 135, "Stand for justice"),
    (49, 11, "Do not mock others"),
    (49, 12, "Avoid suspicion, spying, backbiting"),
    (60, 8, "Deal justly with non-hostile non-Muslims"),

    # Financial behaviors
    (2, 261, "Spending multiplied"),
    (2, 267, "Spend from good things"),
    (9, 34, "Hoarding gold and silver - warning"),

    # Speech acts
    (2, 83, "Speak kindly to people"),
    (4, 148, "Allah dislikes public evil speech"),
    (33, 32, "Do not be soft in speech (to wives of Prophet)"),
    (33, 70, "Speak straight/correct words"),

    # Additional heart domain
    (2, 97, "Enemy to Jibreel - hearts"),
    (2, 283, "Do not conceal testimony - sinful heart"),
    (3, 8, "Hearts that deviate"),
    (6, 43, "Hearts hardened"),
    (8, 24, "Allah between person and heart"),
    (16, 22, "Hearts in denial"),
    (26, 89, "Sound heart on Day of Judgment"),

    # Additional prohibition/negation
    (2, 188, "Do not consume wealth unjustly"),
    (4, 2, "Do not consume orphan wealth"),
    (6, 151, "Do not kill children for poverty"),
    (24, 27, "Do not enter houses without permission"),
    (24, 30, "Lower gaze - command"),
    (24, 31, "Lower gaze - women"),

    # Additional agent types
    (2, 34, "Angels prostrated - except Iblis"),
    (7, 11, "Iblis refused to prostrate"),
    (15, 39, "Iblis swears to mislead"),
    (18, 50, "Iblis from jinn - refused"),

    # Mixed behaviors
    (2, 263, "Kind word better than charity with harm"),
    (4, 86, "Return greeting better"),
    (41, 34, "Repel evil with good"),
]


def main():
    base_dir = Path(__file__).parent.parent.parent

    # Load existing pilot
    existing_refs = set()
    with open(base_dir / "label_studio" / "pilot_50_tasks.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            task = json.loads(line)
            existing_refs.add((task["data"]["surah"], task["data"]["ayah"]))

    print(f"Existing spans: {len(existing_refs)}")

    # Load full Quran data
    with open(base_dir / "data" / "quran" / "uthmani_hafs_v1.tok_v1.json", "r", encoding="utf-8") as f:
        quran = json.load(f)

    # Build lookup
    ayah_lookup = {}
    for surah in quran["surahs"]:
        for ayah in surah["ayat"]:
            ayah_lookup[(surah["surah"], ayah["ayah"])] = {
                "surah": surah["surah"],
                "surah_name": surah["name_ar"],
                "ayah": ayah["ayah"],
                "text": ayah["text"],
                "tokens": ayah["tokens"]
            }

    # Filter strategic additions to exclude existing
    new_additions = []
    for surah, ayah, reason in STRATEGIC_ADDITIONS:
        if (surah, ayah) not in existing_refs:
            if (surah, ayah) in ayah_lookup:
                new_additions.append((surah, ayah, reason))
            else:
                print(f"Warning: {surah}:{ayah} not found in Quran data")

    print(f"New additions available: {len(new_additions)}")

    # Take first 50 new additions
    selected = new_additions[:50]

    # Generate new tasks
    new_tasks = []
    task_id = 51  # Continue from existing

    for surah, ayah, reason in selected:
        data = ayah_lookup[(surah, ayah)]
        task = {
            "id": task_id,
            "data": {
                "id": f"QBM_{task_id:05d}",
                "surah": data["surah"],
                "surah_name": data["surah_name"],
                "ayah": data["ayah"],
                "reference": f"{data['surah']}:{data['ayah']}",
                "raw_text_ar": data["text"],
                "token_count": len(data["tokens"]),
                "tokens": data["tokens"],
                "selection_reason": reason
            }
        }
        new_tasks.append(task)
        task_id += 1

    print(f"Generated {len(new_tasks)} new tasks")

    # Write new tasks file
    output_path = base_dir / "label_studio" / "pilot_100_new_50_tasks.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for task in new_tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")

    print(f"Written to: {output_path}")

    # Also create combined file
    combined_path = base_dir / "label_studio" / "pilot_100_tasks.jsonl"
    with open(base_dir / "label_studio" / "pilot_50_tasks.jsonl", "r", encoding="utf-8") as f:
        existing_lines = f.readlines()

    with open(combined_path, "w", encoding="utf-8") as f:
        for line in existing_lines:
            f.write(line)
        for task in new_tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")

    print(f"Combined file: {combined_path}")

    # Print summary by category
    print("\n=== Selection Summary ===")
    categories = {
        "Heart domain": 0,
        "Negation": 0,
        "Hypocrite": 0,
        "Prophet": 0,
        "Disbeliever": 0,
        "Instinctive": 0,
        "Social": 0,
        "Financial": 0,
        "Speech": 0,
    }

    for surah, ayah, reason in selected:
        reason_lower = reason.lower()
        if "heart" in reason_lower:
            categories["Heart domain"] += 1
        elif "prohibition" in reason_lower or "negation" in reason_lower:
            categories["Negation"] += 1
        elif "hypocrit" in reason_lower:
            categories["Hypocrite"] += 1
        elif "prophet" in reason_lower:
            categories["Prophet"] += 1
        elif "disbeliev" in reason_lower:
            categories["Disbeliever"] += 1
        elif "instinct" in reason_lower or "automatic" in reason_lower or "weakness" in reason_lower:
            categories["Instinctive"] += 1
        elif "social" in reason_lower or "justice" in reason_lower or "mock" in reason_lower:
            categories["Social"] += 1
        elif "spending" in reason_lower or "financial" in reason_lower or "hoard" in reason_lower:
            categories["Financial"] += 1
        elif "speech" in reason_lower or "speak" in reason_lower:
            categories["Speech"] += 1

    for cat, count in categories.items():
        if count > 0:
            print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
