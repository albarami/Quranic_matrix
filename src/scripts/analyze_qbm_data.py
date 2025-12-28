"""
QBM Data Analysis Script

Analyzes the real QBM silver dataset to understand what data is available.
NO MOCK DATA - Only real annotations from the database.
"""

import json
from pathlib import Path
from collections import Counter

DATA_FILE = Path(__file__).parent.parent.parent / "data" / "exports" / "qbm_silver_20251221.json"

def load_data():
    """Load the real QBM dataset."""
    with open(DATA_FILE, encoding="utf-8") as f:
        return json.load(f)

def analyze_dataset():
    """Analyze the complete dataset."""
    data = load_data()
    spans = data["spans"]
    
    print(f"=" * 60)
    print(f"QBM SILVER DATASET ANALYSIS")
    print(f"=" * 60)
    print(f"Total spans: {len(spans)}")
    print(f"Exported at: {data.get('exported_at')}")
    print()
    
    # Agent distribution
    agents = Counter(s.get("agent", {}).get("type", "unknown") for s in spans)
    print("AGENT DISTRIBUTION:")
    for agent, count in agents.most_common():
        pct = count / len(spans) * 100
        print(f"  {agent}: {count} ({pct:.1f}%)")
    print()
    
    # Behavior form distribution
    forms = Counter(s.get("behavior_form", "unknown") for s in spans)
    print("BEHAVIOR FORM DISTRIBUTION:")
    for form, count in forms.most_common():
        pct = count / len(spans) * 100
        print(f"  {form}: {count} ({pct:.1f}%)")
    print()
    
    # Evaluation distribution
    evals = Counter(s.get("normative", {}).get("evaluation", "unknown") for s in spans)
    print("EVALUATION DISTRIBUTION:")
    for ev, count in evals.most_common():
        pct = count / len(spans) * 100
        print(f"  {ev}: {count} ({pct:.1f}%)")
    print()
    
    # Situational axis
    situational = Counter(s.get("axes", {}).get("situational", "unknown") for s in spans)
    print("SITUATIONAL AXIS (Internal/External):")
    for sit, count in situational.most_common():
        pct = count / len(spans) * 100
        print(f"  {sit}: {count} ({pct:.1f}%)")
    print()
    
    # Systemic axis
    systemic = Counter(s.get("axes", {}).get("systemic", "unknown") for s in spans)
    print("SYSTEMIC AXIS:")
    for sys, count in systemic.most_common():
        pct = count / len(spans) * 100
        print(f"  {sys}: {count} ({pct:.1f}%)")
    print()
    
    # Surah distribution (top 10)
    surahs = Counter(s.get("reference", {}).get("surah", 0) for s in spans)
    print("TOP 10 SURAHS BY ANNOTATION COUNT:")
    for surah, count in surahs.most_common(10):
        surah_name = next((s.get("reference", {}).get("surah_name", "") 
                          for s in spans if s.get("reference", {}).get("surah") == surah), "")
        print(f"  Surah {surah} ({surah_name}): {count} spans")
    print()
    
    # Deontic signals
    deontic = Counter(s.get("normative", {}).get("deontic_signal", "unknown") for s in spans)
    print("DEONTIC SIGNALS:")
    for sig, count in deontic.most_common():
        pct = count / len(spans) * 100
        print(f"  {sig}: {count} ({pct:.1f}%)")
    print()
    
    # Speech modes
    speech = Counter(s.get("normative", {}).get("speech_mode", "unknown") for s in spans)
    print("SPEECH MODES:")
    for mode, count in speech.most_common():
        pct = count / len(spans) * 100
        print(f"  {mode}: {count} ({pct:.1f}%)")
    print()
    
    # Sample spans for each agent type
    print("=" * 60)
    print("SAMPLE SPANS BY AGENT TYPE")
    print("=" * 60)
    
    for agent_type in ["AGT_BELIEVER", "AGT_DISBELIEVER", "AGT_HYPOCRITE", "AGT_ALLAH"]:
        agent_spans = [s for s in spans if s.get("agent", {}).get("type") == agent_type]
        print(f"\n{agent_type} ({len(agent_spans)} spans):")
        for span in agent_spans[:3]:
            ref = span.get("reference", {})
            print(f"  [{ref.get('surah')}:{ref.get('ayah')}] {span.get('text_ar', '')[:50]}...")
            print(f"    Form: {span.get('behavior_form')}, Eval: {span.get('normative', {}).get('evaluation')}")

if __name__ == "__main__":
    analyze_dataset()
