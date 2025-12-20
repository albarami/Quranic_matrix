#!/usr/bin/env python3
"""
Validate that all IDs in annotations match controlled vocabularies.

Usage:
    python validate_vocabularies.py <filepath>
"""

import json
import sys
from pathlib import Path


def load_vocabularies() -> dict:
    """Load all controlled vocabulary files."""
    vocab_dir = Path(__file__).parent.parent.parent / "vocab"
    
    vocabularies = {}
    
    # Map vocabulary files to their ID prefixes
    vocab_files = {
        "agents": "AGT_",
        "organs": "ORG_",
        "behavior_concepts": "BEH_",
        "thematic_constructs": "THM_",
        "systemic": "SYS_",
        "spatial": "LOC_",
        "temporal": "TMP_",
        "action_class": "ACT_",
        "action_textual_eval": "EVAL_",
        "periodicity": "PER_",
        "justification_codes": "JST_",
        "grammatical_indicators": "GRM_",
    }
    
    for vocab_name, prefix in vocab_files.items():
        vocab_path = vocab_dir / f"{vocab_name}.json"
        if vocab_path.exists():
            with open(vocab_path, encoding="utf-8") as f:
                data = json.load(f)
                # Extract IDs from vocabulary
                ids = set()
                if "items" in data:
                    for item in data["items"]:
                        if "id" in item:
                            ids.add(item["id"])
                vocabularies[vocab_name] = ids
    
    return vocabularies


def validate_span_vocabularies(span: dict, vocabularies: dict) -> list:
    """Validate all vocabulary references in a span."""
    errors = []
    
    # Check agent type
    agent_type = span.get("agent", {}).get("type")
    if agent_type and "agents" in vocabularies:
        if agent_type not in vocabularies["agents"]:
            errors.append(f"Invalid agent type: {agent_type}")
    
    # Check behavior form (not from vocab, but fixed set)
    valid_forms = {"speech_act", "physical_act", "inner_state", "trait_disposition", 
                   "relational_act", "omission", "mixed", "unknown"}
    behavior_form = span.get("behavior_form")
    if behavior_form and behavior_form not in valid_forms:
        errors.append(f"Invalid behavior_form: {behavior_form}")
    
    # Check axes
    axes = span.get("axes", {})
    
    if axes.get("systemic") and "systemic" in vocabularies:
        systemic = axes["systemic"]
        if isinstance(systemic, str):
            systemic = [systemic]
        for s in systemic:
            if s not in vocabularies["systemic"]:
                errors.append(f"Invalid systemic value: {s}")
    
    if axes.get("spatial") and "spatial" in vocabularies:
        if axes["spatial"] not in vocabularies["spatial"]:
            errors.append(f"Invalid spatial value: {axes['spatial']}")
    
    if axes.get("temporal") and "temporal" in vocabularies:
        if axes["temporal"] not in vocabularies["temporal"]:
            errors.append(f"Invalid temporal value: {axes['temporal']}")
    
    # Check action
    action = span.get("action", {})
    if action.get("class") and "action_class" in vocabularies:
        if action["class"] not in vocabularies["action_class"]:
            errors.append(f"Invalid action class: {action['class']}")
    
    if action.get("textual_eval") and "action_textual_eval" in vocabularies:
        if action["textual_eval"] not in vocabularies["action_textual_eval"]:
            errors.append(f"Invalid action textual_eval: {action['textual_eval']}")
    
    # Check normative (fixed sets)
    normative = span.get("normative", {})
    valid_speech_modes = {"command", "prohibition", "informative", "narrative", "parable", "unknown"}
    valid_evaluations = {"praise", "blame", "warning", "promise", "neutral", "mixed", "unknown"}
    valid_deontic = {"amr", "nahy", "targhib", "tarhib", "khabar"}
    
    if normative.get("speech_mode") and normative["speech_mode"] not in valid_speech_modes:
        errors.append(f"Invalid speech_mode: {normative['speech_mode']}")
    
    if normative.get("evaluation") and normative["evaluation"] not in valid_evaluations:
        errors.append(f"Invalid evaluation: {normative['evaluation']}")
    
    if normative.get("deontic_signal") and normative["deontic_signal"] not in valid_deontic:
        errors.append(f"Invalid deontic_signal: {normative['deontic_signal']}")
    
    return errors


def validate_file(filepath: str) -> dict:
    """Validate all spans in a file against vocabularies."""
    vocabularies = load_vocabularies()
    
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, list):
        spans = data
    elif "annotations" in data:
        spans = data["annotations"]
    else:
        spans = [data]
    
    results = {
        "total": len(spans),
        "valid": 0,
        "invalid": 0,
        "errors": [],
        "vocabularies_loaded": list(vocabularies.keys())
    }
    
    for i, span in enumerate(spans):
        errors = validate_span_vocabularies(span, vocabularies)
        if errors:
            results["invalid"] += 1
            results["errors"].append({
                "index": i,
                "id": span.get("span_id", span.get("id", f"span_{i}")),
                "errors": errors
            })
        else:
            results["valid"] += 1
    
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_vocabularies.py <filepath>")
        sys.exit(1)
    
    results = validate_file(sys.argv[1])
    print(json.dumps(results, indent=2, ensure_ascii=False))
    
    print(f"\n--- Summary ---")
    print(f"Vocabularies loaded: {', '.join(results['vocabularies_loaded'])}")
    print(f"Total: {results['total']}")
    print(f"Valid: {results['valid']}")
    print(f"Invalid: {results['invalid']}")
    
    if results["invalid"] > 0:
        sys.exit(1)
