"""
Day 1: Foundation Data Validation

Validates all required data before training:
- 5 Tafsir sources (31,180 total entries)
- Behavioral spans (15,847+)
- Behavior taxonomy (87 classes)
- Creates train/val/test splits (80/10/10)
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple
import random

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
TAFSIR_DIR = DATA_DIR / "tafsir"
VOCAB_DIR = PROJECT_ROOT / "vocab"
OUTPUT_DIR = DATA_DIR / "training_splits"

TAFSIR_SOURCES = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"]
EXPECTED_VERSE_COUNT = 6236


def load_jsonl(filepath: Path) -> List[Dict]:
    """Load JSONL file."""
    data = []
    if filepath.exists():
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return data


def load_json(filepath: Path) -> Any:
    """Load JSON file."""
    if filepath.exists():
        with open(filepath, encoding="utf-8") as f:
            return json.load(f)
    return {}


def validate_tafsir_sources() -> Dict[str, Any]:
    """Validate all 5 tafsir sources."""
    print("\n" + "=" * 60)
    print("VALIDATING TAFSIR SOURCES")
    print("=" * 60)
    
    results = {
        "valid": True,
        "sources": {},
        "total_entries": 0,
        "issues": [],
    }
    
    for source in TAFSIR_SOURCES:
        filepath = TAFSIR_DIR / f"{source}.ar.jsonl"
        
        if not filepath.exists():
            results["valid"] = False
            results["issues"].append(f"Missing: {filepath}")
            results["sources"][source] = {"exists": False, "count": 0}
            print(f"  ❌ {source}: MISSING")
            continue
        
        entries = load_jsonl(filepath)
        count = len(entries)
        results["sources"][source] = {"exists": True, "count": count}
        results["total_entries"] += count
        
        # Validate structure
        valid_entries = 0
        for entry in entries:
            ref = entry.get("reference", {})
            if ref.get("surah") and ref.get("ayah") and entry.get("text_ar"):
                valid_entries += 1
        
        if count >= EXPECTED_VERSE_COUNT:
            print(f"  ✅ {source}: {count} entries ({valid_entries} valid)")
        else:
            results["issues"].append(f"{source}: Only {count} entries (expected {EXPECTED_VERSE_COUNT})")
            print(f"  ⚠️  {source}: {count} entries (expected {EXPECTED_VERSE_COUNT})")
    
    print(f"\n  Total tafsir entries: {results['total_entries']}")
    print(f"  Expected: {EXPECTED_VERSE_COUNT * 5} ({EXPECTED_VERSE_COUNT} × 5 sources)")
    
    return results


def validate_behavioral_spans() -> Dict[str, Any]:
    """Validate behavioral span annotations."""
    print("\n" + "=" * 60)
    print("VALIDATING BEHAVIORAL SPANS")
    print("=" * 60)
    
    results = {
        "valid": True,
        "total_spans": 0,
        "by_behavior": defaultdict(int),
        "by_source": defaultdict(int),
        "spans": [],
        "issues": [],
    }
    
    # Try multiple possible locations
    span_patterns = [
        DATA_DIR / "spans" / "*.jsonl",
        DATA_DIR / "annotations" / "*.jsonl",
        DATA_DIR / "*.jsonl",
        DATA_DIR / "gold_spans.jsonl",
    ]
    
    found_files = []
    for pattern in span_patterns:
        if "*" in str(pattern):
            found_files.extend(pattern.parent.glob(pattern.name))
        elif pattern.exists():
            found_files.append(pattern)
    
    # Remove duplicates
    found_files = list(set(found_files))
    
    for filepath in found_files:
        # Include tafsir_behavioral_annotations but skip raw tafsir
        if "tafsir" in filepath.name.lower() and "behavioral" not in filepath.name.lower():
            continue  # Skip raw tafsir files, keep behavioral annotations
        
        spans = load_jsonl(filepath)
        if spans:
            # Check if these are behavioral spans
            has_behavior = any(
                s.get("behavior_label") or s.get("behavior") or s.get("behavior_concept") or s.get("behavior_ar")
                for s in spans[:10]
            )
            
            if has_behavior:
                print(f"  Found: {filepath.name} ({len(spans)} spans)")
                results["spans"].extend(spans)
                results["by_source"][filepath.name] = len(spans)
    
    results["total_spans"] = len(results["spans"])
    
    # Count by behavior
    for span in results["spans"]:
        behavior = (
            span.get("behavior_label") or 
            span.get("behavior") or 
            span.get("behavior_concept") or
            span.get("behavior_ar") or
            "unknown"
        )
        results["by_behavior"][behavior] += 1
    
    # Validate minimum counts
    if results["total_spans"] < 1000:
        results["valid"] = False
        results["issues"].append(f"Only {results['total_spans']} spans (need at least 1000)")
        print(f"  ❌ Total spans: {results['total_spans']} (INSUFFICIENT)")
    else:
        print(f"  ✅ Total spans: {results['total_spans']}")
    
    print(f"  Unique behaviors: {len(results['by_behavior'])}")
    
    # Show top behaviors
    sorted_behaviors = sorted(results["by_behavior"].items(), key=lambda x: x[1], reverse=True)
    print(f"  Top 10 behaviors:")
    for behavior, count in sorted_behaviors[:10]:
        print(f"    - {behavior}: {count}")
    
    # Find behaviors with few examples
    low_count = [b for b, c in results["by_behavior"].items() if c < 10 and b != "unknown"]
    if low_count:
        print(f"  ⚠️  Behaviors with <10 examples: {len(low_count)}")
    
    return results


def validate_behavior_taxonomy() -> Dict[str, Any]:
    """Validate behavior taxonomy."""
    print("\n" + "=" * 60)
    print("VALIDATING BEHAVIOR TAXONOMY")
    print("=" * 60)
    
    results = {
        "valid": True,
        "total_behaviors": 0,
        "behaviors": [],
        "issues": [],
    }
    
    # Try multiple locations
    taxonomy_files = [
        VOCAB_DIR / "behavior_concepts.json",
        DATA_DIR / "behaviors.json",
        PROJECT_ROOT / "behavior_taxonomy.json",
    ]
    
    for filepath in taxonomy_files:
        if filepath.exists():
            data = load_json(filepath)
            
            # Handle different formats
            if isinstance(data, list):
                results["behaviors"] = [b.get("name", b) if isinstance(b, dict) else b for b in data]
            elif isinstance(data, dict):
                # Check if it has categories structure
                if "categories" in data:
                    # Extract behaviors from all categories
                    all_behaviors = []
                    for category, behaviors in data["categories"].items():
                        for b in behaviors:
                            if isinstance(b, dict):
                                all_behaviors.append(b.get("ar", b.get("id", "")))
                            else:
                                all_behaviors.append(b)
                    results["behaviors"] = all_behaviors
                else:
                    results["behaviors"] = list(data.keys())
            
            results["total_behaviors"] = len(results["behaviors"])
            print(f"  Found: {filepath.name} ({results['total_behaviors']} behaviors)")
            break
    
    if results["total_behaviors"] == 0:
        # Create default taxonomy
        results["behaviors"] = [
            "الإيمان", "الصبر", "الشكر", "التوبة", "التقوى", "الإحسان", "الصدق", "الأمانة",
            "العدل", "الرحمة", "التواضع", "الخشوع", "الذكر", "الدعاء", "التوكل", "الرضا",
            "الحياء", "الزهد", "الورع", "الإخلاص", "اليقين", "الخوف", "الرجاء", "المحبة",
            "الكفر", "النفاق", "الكبر", "الحسد", "الغيبة", "الكذب", "الظلم", "الفسق",
            "الرياء", "الغضب", "البخل", "الغفلة", "الشرك", "الفجور", "الخيانة", "الجهل",
            "العجب", "الحقد", "السخرية", "اللعن", "السرقة", "الزنا", "القتل", "الربا",
            "قسوة_القلب", "مرض_القلب", "ختم_القلب", "طبع_القلب", "إنابة_القلب",
            "الإعراض", "التكذيب", "الاستهزاء", "المكر", "الخداع", "الغش", "النميمة",
            "الإنفاق", "الجهاد", "الأمر_بالمعروف", "النهي_عن_المنكر", "صلة_الرحم",
            "بر_الوالدين", "الوفاء", "الحلم", "العفو", "الكرم", "الشجاعة", "التفكر",
            "الإسراف", "التبذير", "الجبن", "الكسل", "اليأس", "القنوط", "الجزع",
        ]
        results["total_behaviors"] = len(results["behaviors"])
        print(f"  ⚠️  Using default taxonomy ({results['total_behaviors']} behaviors)")
    
    if results["total_behaviors"] == 87:
        print(f"  ? Behavior taxonomy: {results['total_behaviors']} behaviors")
    else:
        results["valid"] = False
        results["issues"].append(f"Expected 87 behaviors, got {results['total_behaviors']}")
        print(f"  ??  Behavior taxonomy: {results['total_behaviors']} (expected 87)")
    
    return results


def create_train_val_test_splits(spans: List[Dict], output_dir: Path) -> Dict[str, int]:
    """Create stratified train/val/test splits (80/10/10)."""
    print("\n" + "=" * 60)
    print("CREATING TRAIN/VAL/TEST SPLITS")
    print("=" * 60)
    
    if not spans:
        print("  ❌ No spans to split")
        return {"train": 0, "val": 0, "test": 0}
    
    # Group spans by behavior for stratified split
    by_behavior = defaultdict(list)
    for span in spans:
        behavior = (
            span.get("behavior_label") or 
            span.get("behavior") or 
            span.get("behavior_concept") or
            "unknown"
        )
        by_behavior[behavior].append(span)
    
    train, val, test = [], [], []
    
    for behavior, behavior_spans in by_behavior.items():
        random.shuffle(behavior_spans)
        n = len(behavior_spans)
        
        # 80/10/10 split
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)
        
        train.extend(behavior_spans[:train_end])
        val.extend(behavior_spans[train_end:val_end])
        test.extend(behavior_spans[val_end:])
    
    # Shuffle final splits
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    
    print(f"  Train: {len(train)} spans (80%)")
    print(f"  Val:   {len(val)} spans (10%)")
    print(f"  Test:  {len(test)} spans (10%)")
    
    # Save splits
    output_dir.mkdir(parents=True, exist_ok=True)
    
    splits = {"train": train, "val": val, "test": test}
    counts = {}
    
    for split_name, data in splits.items():
        filepath = output_dir / f"{split_name}_spans.jsonl"
        with open(filepath, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        counts[split_name] = len(data)
        print(f"  Saved: {filepath.name}")
    
    return counts


def run_validation() -> Dict[str, Any]:
    """Run complete validation."""
    print("\n" + "=" * 60)
    print("QBM FOUNDATION DATA VALIDATION")
    print("Day 1 of 25-Day Implementation Plan")
    print("=" * 60)
    
    results = {
        "tafsir": validate_tafsir_sources(),
        "spans": validate_behavioral_spans(),
        "taxonomy": validate_behavior_taxonomy(),
        "splits": {},
        "overall_valid": True,
        "issues": [],
    }
    
    # Collect all issues
    for key in ["tafsir", "spans", "taxonomy"]:
        if not results[key].get("valid", True):
            results["overall_valid"] = False
        results["issues"].extend(results[key].get("issues", []))
    
    # Create splits if we have spans
    if results["spans"]["spans"]:
        results["splits"] = create_train_val_test_splits(
            results["spans"]["spans"],
            OUTPUT_DIR
        )
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    print(f"\n  Tafsir Sources: {results['tafsir']['total_entries']} entries")
    print(f"  Behavioral Spans: {results['spans']['total_spans']} spans")
    print(f"  Behavior Taxonomy: {results['taxonomy']['total_behaviors']} behaviors")
    
    if results["splits"]:
        print(f"\n  Training Splits Created:")
        print(f"    Train: {results['splits'].get('train', 0)}")
        print(f"    Val:   {results['splits'].get('val', 0)}")
        print(f"    Test:  {results['splits'].get('test', 0)}")
    
    if results["issues"]:
        print(f"\n  ⚠️  Issues Found ({len(results['issues'])}):")
        for issue in results["issues"]:
            print(f"    - {issue}")
    
    if results["overall_valid"]:
        print("\n  ✅ VALIDATION PASSED - Ready for training")
    else:
        print("\n  ❌ VALIDATION FAILED - Fix issues before training")
    
    return results


if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    results = run_validation()
    
    # Save validation report
    report_path = DATA_DIR / "validation_report.json"
    
    # Remove non-serializable data
    report = {
        "tafsir": {k: v for k, v in results["tafsir"].items() if k != "spans"},
        "spans": {
            "total_spans": results["spans"]["total_spans"],
            "by_behavior": dict(results["spans"]["by_behavior"]),
            "by_source": dict(results["spans"]["by_source"]),
            "valid": results["spans"]["valid"],
        },
        "taxonomy": results["taxonomy"],
        "splits": results["splits"],
        "overall_valid": results["overall_valid"],
        "issues": results["issues"],
    }
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n  Report saved: {report_path}")
