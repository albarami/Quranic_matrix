#!/usr/bin/env python3
"""
Run automated quality checks on QBM annotations.

Validates a subset of fields:
- agent.type (required)
- behavior_form (valid vocabulary)
- assertions (confidence range, presence)
- normative.speech_mode (warning if missing)
- reference (surah/ayah completeness)

Usage:
    python src/scripts/quality_check.py data/exports/qbm_gold_*.json
    python src/scripts/quality_check.py data/pilot/phase3_550_selections.jsonl --format jsonl
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any


def load_spans(filepath: str, format: str = "json") -> List[Dict]:
    """Load spans from file."""
    with open(filepath, encoding="utf-8") as f:
        if format == "jsonl":
            return [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)
            if isinstance(data, list):
                return data
            # Support multiple key names for spans/annotations
            return data.get("spans", data.get("annotations", []))


def check_quality(spans: List[Dict]) -> Dict[str, Any]:
    """Run quality checks on spans."""
    issues = []
    warnings = []
    
    for span in spans:
        span_id = span.get("id", span.get("span_id", "unknown"))
        
        # CRITICAL: Agent must be present
        agent = span.get("agent", {})
        if not agent.get("type"):
            issues.append(f"{span_id}: Missing agent type")
        
        # CRITICAL: At least one assertion (if not a selection-only record)
        assertions = span.get("assertions", [])
        if "behavior" in span and not assertions:
            issues.append(f"{span_id}: Has behavior but no assertions")
        
        # Check: Confidence scores in valid range [0, 1]
        for i, a in enumerate(assertions):
            conf = a.get("confidence")
            if conf is not None:
                if not isinstance(conf, (int, float)):
                    issues.append(f"{span_id}: Assertion {i} confidence is not numeric")
                elif conf < 0 or conf > 1:
                    issues.append(f"{span_id}: Assertion {i} has invalid confidence {conf}")
        
        # Check: Normative layer present (if annotated)
        # Support both "normative_textual" and "normative" keys
        normative = span.get("normative_textual", span.get("normative", {}))
        behavior_form = span.get("behavior_form") or span.get("behavior", {}).get("form")
        if behavior_form and not normative.get("speech_mode"):
            warnings.append(f"{span_id}: Missing speech_mode in normative layer")
        
        # Check: Heart domains if ORG_HEART assertion
        for a in assertions:
            if a.get("axis") == "AX_ORGAN" and a.get("value") == "ORG_HEART":
                if not a.get("organ_semantic_domains"):
                    warnings.append(f"{span_id}: ORG_HEART without semantic domains")
        
        # Check: Behavior form is valid (from vocab/behavior_form.json)
        valid_forms = [
            "physical_act", "speech_act", "inner_state", "trait_disposition",
            "relational_act", "omission", "mixed", "unknown"
        ]
        if behavior_form and behavior_form not in valid_forms:
            issues.append(f"{span_id}: Invalid behavior_form '{behavior_form}'")
        
        # Check: Agent type is valid (support both AGT_ prefixed and plain)
        valid_agents = [
            "ALLAH", "PROPHET", "BELIEVER", "DISBELIEVER", "HYPOCRITE",
            "HUMAN_GENERAL", "ANGEL", "JINN", "HISTORICAL_FIGURE", "OTHER",
            "WRONGDOER", "POLYTHEIST", "PEOPLE_BOOK"
        ]
        valid_agents_prefixed = [f"AGT_{a}" for a in valid_agents]
        agent_type = agent.get("type", "")
        if agent_type and agent_type not in valid_agents and agent_type not in valid_agents_prefixed:
            warnings.append(f"{span_id}: Non-standard agent_type '{agent_type}'")
        
        # Check: Reference is complete
        ref = span.get("reference", {})
        if not ref.get("surah") or not ref.get("ayah"):
            issues.append(f"{span_id}: Incomplete reference (missing surah/ayah)")
        
        # Check: Arabic text present (optional for annotation records without source text)
        # Only warn if this looks like a full span record, not an annotation-only record
        if span.get("token_start") is not None or span.get("tokens"):
            if not span.get("raw_text_ar") and not span.get("text_ar"):
                warnings.append(f"{span_id}: Missing Arabic text")
    
    return {
        "total_spans": len(spans),
        "issues_found": len(issues),
        "warnings_found": len(warnings),
        "issues": issues,
        "warnings": warnings,
        "pass": len(issues) == 0
    }


def check_coverage(spans: List[Dict]) -> Dict[str, Any]:
    """Check surah/ayah coverage."""
    coverage = {}
    
    for span in spans:
        ref = span.get("reference", {})
        surah = ref.get("surah")
        ayah = ref.get("ayah")
        
        if surah:
            if surah not in coverage:
                coverage[surah] = set()
            if ayah:
                coverage[surah].add(ayah)
    
    return {
        "surahs_covered": len(coverage),
        "ayat_covered": sum(len(ayat) for ayat in coverage.values()),
        "by_surah": {s: len(a) for s, a in sorted(coverage.items())}
    }


def check_iaa_readiness(spans: List[Dict]) -> Dict[str, Any]:
    """Check if spans are ready for IAA calculation."""
    annotator_counts = {}
    
    for span in spans:
        annotator = span.get("annotator_id", span.get("annotator", "unknown"))
        annotator_counts[annotator] = annotator_counts.get(annotator, 0) + 1
    
    return {
        "annotators": len(annotator_counts),
        "by_annotator": annotator_counts,
        "ready_for_iaa": len(annotator_counts) >= 2
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="QBM quality check")
    parser.add_argument("file", help="Annotation file to check")
    parser.add_argument("--format", choices=["json", "jsonl"], default="json")
    parser.add_argument("--coverage", action="store_true", help="Show coverage stats")
    parser.add_argument("--iaa", action="store_true", help="Check IAA readiness")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all issues")
    args = parser.parse_args()
    
    spans = load_spans(args.file, args.format)
    results = check_quality(spans)
    
    print(f"\n{'='*50}")
    print(f"QBM QUALITY CHECK: {args.file}")
    print(f"{'='*50}")
    print(f"Total spans: {results['total_spans']}")
    print(f"Issues: {results['issues_found']}")
    print(f"Warnings: {results['warnings_found']}")
    print(f"Status: {'[PASS]' if results['pass'] else '[FAIL]'}")
    
    if args.verbose or results['issues_found'] > 0:
        if results['issues']:
            print(f"\nIssues ({len(results['issues'])}):")
            for issue in results['issues'][:20]:
                print(f"  [X] {issue}")
            if len(results['issues']) > 20:
                print(f"  ... and {len(results['issues']) - 20} more")
        
        if results['warnings'] and args.verbose:
            print(f"\nWarnings ({len(results['warnings'])}):")
            for warning in results['warnings'][:10]:
                print(f"  [!] {warning}")
    
    if args.coverage:
        cov = check_coverage(spans)
        print(f"\nCoverage:")
        print(f"  Surahs: {cov['surahs_covered']}/114")
        print(f"  Ayat: {cov['ayat_covered']}")
    
    if args.iaa:
        iaa = check_iaa_readiness(spans)
        print(f"\nIAA Readiness:")
        print(f"  Annotators: {iaa['annotators']}")
        print(f"  Ready: {'[YES]' if iaa['ready_for_iaa'] else '[NO]'}")
    
    # Exit with error code if issues found
    sys.exit(0 if results['pass'] else 1)


if __name__ == "__main__":
    main()
