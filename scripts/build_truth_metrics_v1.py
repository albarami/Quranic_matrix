"""
Build Truth Metrics v1 - Deterministic metrics from canonical data.

This script computes metrics from the canonical QBM silver dataset and writes
a versioned artifact that serves as the SOURCE OF TRUTH for all UI metrics.

Usage:
    python scripts/build_truth_metrics_v1.py

Output:
    data/metrics/truth_metrics_v1.json

Non-negotiables:
    - All numbers computed from canonical data only
    - No hardcoded stats, no guessed values
    - If source file missing or schema mismatch -> exit nonzero
    - Checksum included for integrity verification
"""

import json
import hashlib
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Canonical source files
CANONICAL_DATASET_DIR = Path("data/exports")
CANONICAL_DATASET_PATTERN = "qbm_silver_*.json"
TAFSIR_DIR = Path("data/tafsir")
EVIDENCE_INDEX = Path("data/evidence/evidence_index_v2_chunked.jsonl")

# Output
OUTPUT_FILE = Path("data/metrics/truth_metrics_v1.json")


def get_git_sha() -> str:
    """Get current git SHA for versioning."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def find_canonical_dataset() -> Optional[Path]:
    """Find the latest canonical silver dataset."""
    if not CANONICAL_DATASET_DIR.exists():
        return None
    
    files = sorted(CANONICAL_DATASET_DIR.glob(CANONICAL_DATASET_PATTERN), reverse=True)
    return files[0] if files else None


def load_canonical_dataset(path: Path) -> Dict[str, Any]:
    """Load and validate canonical dataset."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Validate required fields
    if "spans" not in data:
        raise ValueError(f"Schema mismatch: 'spans' key missing in {path}")
    
    spans = data["spans"]
    if not spans:
        raise ValueError(f"Schema mismatch: 'spans' is empty in {path}")
    
    # Validate span schema
    required_span_keys = ["agent", "behavior_form", "normative", "reference"]
    sample = spans[0]
    missing = [k for k in required_span_keys if k not in sample]
    if missing:
        raise ValueError(f"Schema mismatch: spans missing keys {missing}")
    
    return data


def count_tafsir_sources() -> int:
    """Count available tafsir sources."""
    if not TAFSIR_DIR.exists():
        return 0
    
    # Canonical 7 tafsir sources
    CANONICAL_SOURCES = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn", "baghawi", "muyassar"]
    count = 0
    
    for source in CANONICAL_SOURCES:
        for ext in [".ar.jsonl", ".ar.json", ".jsonl", ".json"]:
            if (TAFSIR_DIR / f"{source}{ext}").exists():
                count += 1
                break
    
    return count


def compute_distribution(spans: List[Dict], field_path: List[str]) -> Dict[str, Any]:
    """
    Compute distribution for a nested field.
    
    Args:
        spans: List of span dictionaries
        field_path: Path to field, e.g., ["agent", "type"] or ["behavior_form"]
    
    Returns:
        Dict with counts, percentages, and items list
    """
    counts = {}
    total = len(spans)
    
    for span in spans:
        value = span
        for key in field_path:
            value = value.get(key, {}) if isinstance(value, dict) else None
            if value is None:
                break
        
        if value is None or value == {}:
            value = "unknown"
        
        counts[value] = counts.get(value, 0) + 1
    
    # Build items list sorted by count descending
    items = []
    for key, count in sorted(counts.items(), key=lambda x: -x[1]):
        items.append({
            "key": key,
            "count": count,
            "percentage": round(count / total * 100, 2) if total > 0 else 0
        })
    
    # Verify percentages sum to ~100
    pct_sum = sum(item["percentage"] for item in items)
    
    return {
        "total": total,
        "unique_values": len(counts),
        "items": items,
        "percentage_sum": round(pct_sum, 2)
    }


def compute_metrics(dataset: Dict[str, Any], dataset_path: Path) -> Dict[str, Any]:
    """Compute all metrics from canonical dataset."""
    spans = dataset["spans"]
    total_spans = len(spans)
    
    # Compute unique verse keys
    verse_keys = set()
    for span in spans:
        ref = span.get("reference", {})
        surah = ref.get("surah")
        ayah = ref.get("ayah")
        if surah and ayah:
            verse_keys.add(f"{surah}:{ayah}")
    
    # Agent distribution with Arabic labels
    agent_labels = {
        "AGT_ALLAH": "الله",
        "AGT_DISBELIEVER": "الكافر",
        "AGT_BELIEVER": "المؤمن",
        "AGT_HUMAN_GENERAL": "الإنسان",
        "AGT_WRONGDOER": "الظالم",
        "AGT_PROPHET": "النبي",
        "AGT_HYPOCRITE": "المنافق",
        "AGT_ANGEL": "الملك",
        "AGT_JINN": "الجن",
        "AGT_HISTORICAL_FIGURE": "شخصية تاريخية",
        "AGT_POLYTHEIST": "المشرك",
        "AGT_PEOPLE_BOOK": "أهل الكتاب",
        "AGT_OTHER": "أخرى",
    }
    
    agent_dist = compute_distribution(spans, ["agent", "type"])
    # Add Arabic labels
    for item in agent_dist["items"]:
        item["label_ar"] = agent_labels.get(item["key"], item["key"])
    
    # Behavior form distribution with Arabic labels
    form_labels = {
        "inner_state": "حالة داخلية",
        "speech_act": "فعل كلامي",
        "relational_act": "فعل علائقي",
        "physical_act": "فعل جسدي",
        "trait_disposition": "سمة",
        "omission": "ترك",
        "mixed": "مختلط",
        "unknown": "غير محدد",
    }
    
    form_dist = compute_distribution(spans, ["behavior_form"])
    for item in form_dist["items"]:
        item["label_ar"] = form_labels.get(item["key"], item["key"])
    
    # Evaluation distribution with Arabic labels
    eval_labels = {
        "neutral": "محايد",
        "blame": "ذم",
        "praise": "مدح",
        "warning": "تحذير",
        "unknown": "غير محدد",
    }
    
    eval_dist = compute_distribution(spans, ["normative", "evaluation"])
    for item in eval_dist["items"]:
        item["label_ar"] = eval_labels.get(item["key"], item["key"])
    
    # Systemic distribution
    systemic_dist = compute_distribution(spans, ["axes", "systemic"])
    
    # Deontic signal distribution
    deontic_dist = compute_distribution(spans, ["normative", "deontic_signal"])
    
    # Tafsir sources count
    tafsir_count = count_tafsir_sources()
    
    return {
        "totals": {
            "spans": total_spans,
            "unique_verse_keys": len(verse_keys),
            "tafsir_sources_count": tafsir_count,
        },
        "agent_distribution": agent_dist,
        "behavior_forms": form_dist,
        "evaluations": eval_dist,
        "systemic_distribution": systemic_dist,
        "deontic_signals": deontic_dist,
    }


def compute_checksum(payload: Dict) -> str:
    """Compute SHA256 checksum of metrics payload."""
    # Serialize deterministically
    payload_str = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload_str.encode("utf-8")).hexdigest()


def main():
    """Main entry point."""
    print("=" * 60)
    print("Building Truth Metrics v1")
    print("=" * 60)
    
    # Step 1: Find canonical dataset
    dataset_path = find_canonical_dataset()
    if not dataset_path:
        print(f"ERROR: No canonical dataset found matching {CANONICAL_DATASET_PATTERN}")
        print(f"       in directory {CANONICAL_DATASET_DIR}")
        sys.exit(1)
    
    print(f"✓ Found canonical dataset: {dataset_path}")
    
    # Step 2: Load and validate dataset
    try:
        dataset = load_canonical_dataset(dataset_path)
        print(f"✓ Loaded {len(dataset['spans'])} spans")
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in {dataset_path}: {e}")
        sys.exit(1)
    
    # Step 3: Compute metrics
    print("Computing metrics...")
    metrics = compute_metrics(dataset, dataset_path)
    
    # Validate: no "unknown" = 100%
    agent_items = metrics["agent_distribution"]["items"]
    if len(agent_items) == 1 and agent_items[0]["key"] == "unknown":
        print("ERROR: All agents are 'unknown' - schema mismatch or wrong source file")
        sys.exit(1)
    
    print(f"✓ Agent distribution: {len(agent_items)} types")
    print(f"✓ Behavior forms: {len(metrics['behavior_forms']['items'])} types")
    print(f"✓ Evaluations: {len(metrics['evaluations']['items'])} types")
    
    # Step 4: Build final artifact
    git_sha = get_git_sha()
    
    artifact = {
        "schema_version": "metrics_v1",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "build_version": git_sha,
        "source_files": [
            str(dataset_path.absolute()),
        ],
        "status": "ready",
        "metrics": metrics,
    }
    
    # Add checksum
    artifact["checksum"] = compute_checksum(artifact["metrics"])
    
    # Step 5: Write output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Written to {OUTPUT_FILE}")
    print(f"✓ Checksum: {artifact['checksum'][:16]}...")
    
    # Summary
    print("\n" + "=" * 60)
    print("METRICS SUMMARY")
    print("=" * 60)
    print(f"Total spans: {metrics['totals']['spans']}")
    print(f"Unique verses: {metrics['totals']['unique_verse_keys']}")
    print(f"Tafsir sources: {metrics['totals']['tafsir_sources_count']}")
    print("\nAgent Distribution:")
    for item in metrics["agent_distribution"]["items"][:5]:
        print(f"  {item['key']}: {item['count']} ({item['percentage']}%)")
    print("\nBehavior Forms:")
    for item in metrics["behavior_forms"]["items"]:
        print(f"  {item['key']}: {item['count']} ({item['percentage']}%)")
    print("\nEvaluations:")
    for item in metrics["evaluations"]["items"]:
        print(f"  {item['key']}: {item['count']} ({item['percentage']}%)")
    
    print("\n✓ Build complete. Status: ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
