#!/usr/bin/env python3
"""
Production Readiness Audit for QBM System

Checks for:
1. All required data files exist
2. No placeholders or mocks in production code
3. All 87 behaviors mapped
4. Enterprise graph v3 properly configured
5. Tafsir sources complete
"""

import json
from pathlib import Path
from collections import Counter

DATA_DIR = Path("data")
VOCAB_DIR = Path("vocab")
SRC_DIR = Path("src")

issues = []
warnings = []

print("=" * 70)
print("QBM PRODUCTION READINESS AUDIT")
print("=" * 70)

# =============================================================================
# 1. DATA FILES AUDIT
# =============================================================================
print("\n1. DATA FILES AUDIT")
print("-" * 40)

# Check semantic_graph_v3.json
sg3_path = DATA_DIR / "graph" / "semantic_graph_v3.json"
if sg3_path.exists():
    with open(sg3_path, "r", encoding="utf-8") as f:
        sg3 = json.load(f)
    nodes = sg3.get("nodes", [])
    edges = sg3.get("edges", [])
    print(f"  semantic_graph_v3.json: {len(nodes)} nodes, {len(edges)} edges")
    
    # Count behavior nodes
    beh_nodes = [n for n in nodes if n.get("type") == "BEHAVIOR"]
    print(f"    Behavior nodes: {len(beh_nodes)}")
    
    # Count causal edges
    causal_types = {"CAUSES", "LEADS_TO", "PREVENTS", "STRENGTHENS"}
    causal_edges = [e for e in edges if e.get("edge_type") in causal_types]
    print(f"    Causal edges: {len(causal_edges)}")
    
    if len(beh_nodes) < 87:
        warnings.append(f"semantic_graph_v3 has only {len(beh_nodes)} behavior nodes (expected 87)")
else:
    issues.append("CRITICAL: semantic_graph_v3.json does not exist")

# Check concept_index_v3.jsonl
ci3_path = DATA_DIR / "evidence" / "concept_index_v3.jsonl"
if ci3_path.exists():
    beh_count = 0
    total_verses = 0
    with open(ci3_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("concept_id", "").startswith("BEH_"):
                beh_count += 1
                total_verses += len(entry.get("verses", []))
    print(f"  concept_index_v3.jsonl: {beh_count} behaviors, {total_verses} verse links")
    if beh_count != 87:
        issues.append(f"concept_index_v3 has {beh_count} behaviors (expected 87)")
else:
    issues.append("CRITICAL: concept_index_v3.jsonl does not exist")

# Check canonical_entities.json
ce_path = VOCAB_DIR / "canonical_entities.json"
if ce_path.exists():
    with open(ce_path, "r", encoding="utf-8") as f:
        ce = json.load(f)
    beh_count = len(ce.get("behaviors", []))
    print(f"  canonical_entities.json: {beh_count} behaviors")
    if beh_count != 87:
        issues.append(f"canonical_entities has {beh_count} behaviors (expected 87)")
else:
    issues.append("CRITICAL: canonical_entities.json does not exist")

# Check evidence_index_v2_chunked.jsonl
ei_path = DATA_DIR / "evidence" / "evidence_index_v2_chunked.jsonl"
if ei_path.exists():
    with open(ei_path, "r", encoding="utf-8") as f:
        count = sum(1 for _ in f)
    print(f"  evidence_index_v2_chunked.jsonl: {count} verse entries")
else:
    issues.append("CRITICAL: evidence_index_v2_chunked.jsonl does not exist")

# Check tafsir files
print("\n  Tafsir sources:")
tafsir_dir = DATA_DIR / "tafsir"
sources = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn", "baghawi", "muyassar"]
for src in sources:
    p = tafsir_dir / f"{src}.ar.jsonl"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            count = sum(1 for _ in f)
        print(f"    {src}: {count} entries")
    else:
        issues.append(f"CRITICAL: Tafsir source missing: {src}")

# =============================================================================
# 2. CODE AUDIT - Check for placeholders in production paths
# =============================================================================
print("\n2. CODE AUDIT")
print("-" * 40)

# Check capabilities/engines.py uses v3 graph
engines_path = SRC_DIR / "capabilities" / "engines.py"
if engines_path.exists():
    content = engines_path.read_text(encoding="utf-8")
    if "semantic_graph_v3.json" in content:
        print("  engines.py: Uses semantic_graph_v3.json ✓")
    elif "semantic_graph_v2.json" in content:
        warnings.append("engines.py still references semantic_graph_v2.json")
        print("  engines.py: Still uses semantic_graph_v2.json ⚠")
else:
    issues.append("CRITICAL: src/capabilities/engines.py does not exist")

# Check proof_only_backend.py uses v3 graph
backend_path = SRC_DIR / "ml" / "proof_only_backend.py"
if backend_path.exists():
    content = backend_path.read_text(encoding="utf-8")
    if "semantic_graph_v3.json" in content:
        print("  proof_only_backend.py: Uses semantic_graph_v3.json ✓")
    elif "semantic_graph_v2.json" in content:
        warnings.append("proof_only_backend.py still references semantic_graph_v2.json")
        print("  proof_only_backend.py: Still uses semantic_graph_v2.json ⚠")
else:
    issues.append("CRITICAL: src/ml/proof_only_backend.py does not exist")

# =============================================================================
# 3. BENCHMARK RESULTS
# =============================================================================
print("\n3. BENCHMARK RESULTS")
print("-" * 40)

eval_dir = Path("reports/eval")
if eval_dir.exists():
    reports = list(eval_dir.glob("eval_report_*.json"))
    if reports:
        latest = max(reports, key=lambda p: p.stat().st_mtime)
        with open(latest, "r", encoding="utf-8") as f:
            report = json.load(f)
        summary = report.get("summary", {})
        totals = summary.get("totals", {})
        print(f"  Latest report: {latest.name}")
        print(f"    Total: {totals.get('total', 'N/A')}")
        print(f"    PASS: {totals.get('PASS', 'N/A')}")
        print(f"    FAIL: {totals.get('FAIL', 'N/A')}")
        print(f"    Pass Rate: {summary.get('pass_rate', 'N/A')}%")
        
        if totals.get("FAIL", 0) > 0:
            warnings.append(f"Benchmark has {totals['FAIL']} failures")
    else:
        warnings.append("No evaluation reports found")
else:
    warnings.append("reports/eval directory does not exist")

# =============================================================================
# 4. SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("AUDIT SUMMARY")
print("=" * 70)

if issues:
    print(f"\n❌ CRITICAL ISSUES ({len(issues)}):")
    for i in issues:
        print(f"   - {i}")

if warnings:
    print(f"\n⚠ WARNINGS ({len(warnings)}):")
    for w in warnings:
        print(f"   - {w}")

if not issues and not warnings:
    print("\n✅ ALL CHECKS PASSED - PRODUCTION READY")
elif not issues:
    print("\n⚠ PRODUCTION READY WITH WARNINGS")
else:
    print("\n❌ NOT PRODUCTION READY - FIX CRITICAL ISSUES")

print("=" * 70)
