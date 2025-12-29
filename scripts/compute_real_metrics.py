"""
Compute REAL metrics from the actual data files.
This is the deterministic truth layer for metrics.
"""

import json
from pathlib import Path
from datetime import datetime

DATA_FILE = Path("data/annotations/tafsir_behavioral_annotations.jsonl")

def compute_metrics():
    """Compute metrics from real data files."""
    if not DATA_FILE.exists():
        print(f"ERROR: Data file not found: {DATA_FILE}")
        return None
    
    agents = {}
    forms = {}
    evals = {}
    surahs = set()
    ayat = set()
    
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    for line in lines:
        d = json.loads(line.strip())
        
        # Agent distribution
        a = d.get("agent", {}).get("type", "unknown")
        agents[a] = agents.get(a, 0) + 1
        
        # Behavior forms
        bf = d.get("behavior_form", "unknown")
        forms[bf] = forms.get(bf, 0) + 1
        
        # Evaluations
        e = d.get("normative", {}).get("evaluation", "unknown")
        evals[e] = evals.get(e, 0) + 1
        
        # Surah/Ayah tracking
        ref = d.get("reference", {})
        if ref.get("surah"):
            surahs.add(ref["surah"])
            ayat.add(f"{ref['surah']}:{ref.get('ayah', 0)}")
    
    total = len(lines)
    
    # Calculate percentages
    agent_pcts = {k: round(v / total * 100, 2) for k, v in agents.items()}
    form_pcts = {k: round(v / total * 100, 2) for k, v in forms.items()}
    eval_pcts = {k: round(v / total * 100, 2) for k, v in evals.items()}
    
    metrics = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source_file": str(DATA_FILE),
        "total_spans": total,
        "unique_surahs": len(surahs),
        "unique_ayat": len(ayat),
        "agent_distribution": {
            "counts": dict(sorted(agents.items(), key=lambda x: -x[1])),
            "percentages": dict(sorted(agent_pcts.items(), key=lambda x: -x[1])),
        },
        "behavior_forms": {
            "counts": dict(sorted(forms.items(), key=lambda x: -x[1])),
            "percentages": dict(sorted(form_pcts.items(), key=lambda x: -x[1])),
        },
        "evaluations": {
            "counts": dict(sorted(evals.items(), key=lambda x: -x[1])),
            "percentages": dict(sorted(eval_pcts.items(), key=lambda x: -x[1])),
        },
    }
    
    return metrics


if __name__ == "__main__":
    metrics = compute_metrics()
    if metrics:
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        
        # Save to file
        output_file = Path("data/metrics_truth.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {output_file}")
