#!/usr/bin/env python3
"""
Phase 7: Audit Pack Generator

Generates a complete audit pack for QBM system verification:
1. Input file hashes (SHA256)
2. Output file hashes (SHA256)
3. GPU proof logs
4. Provenance completeness report
5. System configuration snapshot

Usage:
    python scripts/generate_audit_pack.py [--output-dir artifacts/audit_pack]
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VOCAB_DIR = PROJECT_ROOT / "vocab"
SCHEMAS_DIR = PROJECT_ROOT / "schemas"


# =============================================================================
# HASH GENERATION
# =============================================================================

def sha256_file(filepath: Path) -> str:
    """Calculate SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def sha256_string(content: str) -> str:
    """Calculate SHA256 hash of a string."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def get_file_info(filepath: Path) -> Dict[str, Any]:
    """Get file info including hash and metadata."""
    if not filepath.exists():
        return {"exists": False, "path": str(filepath)}
    
    stat = filepath.stat()
    return {
        "exists": True,
        "path": str(filepath.relative_to(PROJECT_ROOT)),
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "sha256": sha256_file(filepath)
    }


# =============================================================================
# INPUT HASHES
# =============================================================================

def generate_input_hashes() -> Dict[str, Any]:
    """Generate hashes for all input files (SSOT sources)."""
    input_files = {
        "quran_text": DATA_DIR / "quran" / "quran_index.source.json",
        "canonical_entities": VOCAB_DIR / "canonical_entities.json",
        "postgres_schema": SCHEMAS_DIR / "postgres_truth_layer.sql",
    }
    
    # Tafsir sources
    tafsir_sources = [
        "ibn_kathir", "tabari", "qurtubi", "saadi", 
        "jalalayn", "baghawi", "muyassar"
    ]
    for src in tafsir_sources:
        input_files[f"tafsir_{src}"] = DATA_DIR / "tafsir" / f"{src}.ar.jsonl"
    
    hashes = {}
    for name, path in input_files.items():
        hashes[name] = get_file_info(path)
    
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "file_count": len(input_files),
        "files": hashes
    }


# =============================================================================
# OUTPUT HASHES
# =============================================================================

def generate_output_hashes() -> Dict[str, Any]:
    """Generate hashes for all output/derived files."""
    output_files = {
        # Graph files
        "semantic_graph_v3": DATA_DIR / "graph" / "semantic_graph_v3.json",
        "semantic_graph_v2": DATA_DIR / "graph" / "semantic_graph_v2.json",
        "graph_v3": DATA_DIR / "graph" / "graph_v3.json",
        
        # Evidence indices
        "concept_index_v3": DATA_DIR / "evidence" / "concept_index_v3.jsonl",
        "evidence_index_chunked": DATA_DIR / "evidence" / "evidence_index_v2_chunked.jsonl",
        
        # Embeddings
        "gpu_index": DATA_DIR / "embeddings" / "gpu_index.npy",
        "annotations_embeddings": DATA_DIR / "embeddings" / "annotations_embeddings.npy",
        
        # KB artifacts
        "behavior_dossiers": DATA_DIR / "kb" / "behavior_dossiers.jsonl",
        "kb_manifest": DATA_DIR / "kb" / "manifest.json",
    }
    
    hashes = {}
    for name, path in output_files.items():
        hashes[name] = get_file_info(path)
    
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "file_count": len(output_files),
        "files": hashes
    }


# =============================================================================
# GPU PROOF LOGS
# =============================================================================

def generate_gpu_proof() -> Dict[str, Any]:
    """Generate GPU availability and configuration proof."""
    gpu_info = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "torch_available": False,
        "cuda_available": False,
        "device_count": 0,
        "devices": []
    }
    
    try:
        import torch
        gpu_info["torch_available"] = True
        gpu_info["torch_version"] = torch.__version__
        gpu_info["cuda_available"] = torch.cuda.is_available()
        
        if torch.cuda.is_available():
            gpu_info["cuda_version"] = torch.version.cuda
            gpu_info["cudnn_version"] = torch.backends.cudnn.version()
            gpu_info["device_count"] = torch.cuda.device_count()
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu_info["devices"].append({
                    "id": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multi_processor_count": props.multi_processor_count
                })
    except ImportError:
        gpu_info["error"] = "PyTorch not installed"
    except Exception as e:
        gpu_info["error"] = str(e)
    
    # Try nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            gpu_info["nvidia_smi_output"] = result.stdout.strip()
    except Exception:
        pass
    
    return gpu_info


# =============================================================================
# PROVENANCE COMPLETENESS REPORT
# =============================================================================

def generate_provenance_report() -> Dict[str, Any]:
    """Generate provenance completeness report."""
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "behaviors": {},
        "graph": {},
        "tafsir": {},
        "completeness": {}
    }
    
    # Check behaviors
    ce_path = VOCAB_DIR / "canonical_entities.json"
    if ce_path.exists():
        with open(ce_path, "r", encoding="utf-8") as f:
            entities = json.load(f)
        report["behaviors"]["canonical_count"] = len(entities.get("behaviors", []))
        report["behaviors"]["agents_count"] = len(entities.get("agents", []))
        report["behaviors"]["organs_count"] = len(entities.get("organs", []))
        report["behaviors"]["heart_states_count"] = len(entities.get("heart_states", []))
        report["behaviors"]["consequences_count"] = len(entities.get("consequences", []))
    
    # Check concept index
    ci_path = DATA_DIR / "evidence" / "concept_index_v3.jsonl"
    if ci_path.exists():
        behavior_count = 0
        total_verses = 0
        with open(ci_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("concept_id", "").startswith("BEH_"):
                    behavior_count += 1
                    total_verses += len(entry.get("verses", []))
        report["behaviors"]["indexed_count"] = behavior_count
        report["behaviors"]["total_verse_links"] = total_verses
    
    # Check graph
    sg_path = DATA_DIR / "graph" / "semantic_graph_v3.json"
    if sg_path.exists():
        with open(sg_path, "r", encoding="utf-8") as f:
            graph = json.load(f)
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        
        report["graph"]["version"] = graph.get("version")
        report["graph"]["total_nodes"] = len(nodes)
        report["graph"]["total_edges"] = len(edges)
        
        # Count by type
        beh_nodes = [n for n in nodes if n.get("type") == "BEHAVIOR"]
        report["graph"]["behavior_nodes"] = len(beh_nodes)
        
        # Count causal edges
        causal_types = {"CAUSES", "LEADS_TO", "PREVENTS", "STRENGTHENS"}
        causal_edges = [e for e in edges if e.get("edge_type") in causal_types]
        report["graph"]["causal_edges"] = len(causal_edges)
        
        # Count edges with provenance
        edges_with_evidence = [e for e in edges if e.get("evidence")]
        report["graph"]["edges_with_evidence"] = len(edges_with_evidence)
    
    # Check tafsir coverage
    evidence_path = DATA_DIR / "evidence" / "evidence_index_v2_chunked.jsonl"
    if evidence_path.exists():
        source_counts = {}
        total_chunks = 0
        with open(evidence_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                for chunk in entry.get("chunks", [entry]):
                    source = chunk.get("source", "unknown")
                    source_counts[source] = source_counts.get(source, 0) + 1
                    total_chunks += 1
        
        report["tafsir"]["total_chunks"] = total_chunks
        report["tafsir"]["sources"] = source_counts
        report["tafsir"]["source_count"] = len(source_counts)
    
    # Completeness checks
    report["completeness"]["behaviors_complete"] = (
        report["behaviors"].get("canonical_count", 0) == 87 and
        report["behaviors"].get("indexed_count", 0) == 87
    )
    report["completeness"]["graph_complete"] = (
        report["graph"].get("behavior_nodes", 0) == 87
    )
    report["completeness"]["tafsir_complete"] = (
        report["tafsir"].get("source_count", 0) >= 7
    )
    report["completeness"]["all_complete"] = all([
        report["completeness"]["behaviors_complete"],
        report["completeness"]["graph_complete"],
        report["completeness"]["tafsir_complete"]
    ])
    
    return report


# =============================================================================
# SYSTEM INFO
# =============================================================================

def generate_system_info() -> Dict[str, Any]:
    """Generate system configuration snapshot."""
    info = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        }
    }
    
    # Git info
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=10
        )
        if result.returncode == 0:
            info["git_commit"] = result.stdout.strip()
        
        result = subprocess.run(
            ["git", "describe", "--tags", "--always"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=10
        )
        if result.returncode == 0:
            info["git_describe"] = result.stdout.strip()
    except Exception:
        pass
    
    # Environment variables (filtered)
    env_vars = {}
    for key in ["AZURE_OPENAI_DEPLOYMENT", "QBM_DATA_DIR", "CUDA_VISIBLE_DEVICES"]:
        if key in os.environ:
            env_vars[key] = os.environ[key]
    info["environment"] = env_vars
    
    return info


# =============================================================================
# BENCHMARK RESULTS
# =============================================================================

def get_latest_benchmark_results() -> Dict[str, Any]:
    """Get latest benchmark evaluation results."""
    eval_dir = PROJECT_ROOT / "reports" / "eval"
    if not eval_dir.exists():
        return {"error": "No evaluation reports found"}
    
    reports = list(eval_dir.glob("eval_report_*.json"))
    if not reports:
        return {"error": "No evaluation reports found"}
    
    latest = max(reports, key=lambda p: p.stat().st_mtime)
    
    with open(latest, "r", encoding="utf-8") as f:
        report = json.load(f)
    
    summary = report.get("summary", {})
    totals = summary.get("totals", {})
    
    return {
        "report_file": str(latest.relative_to(PROJECT_ROOT)),
        "timestamp": report.get("meta", {}).get("timestamp_utc"),
        "total_questions": totals.get("total", 0),
        "passed": totals.get("PASS", 0),
        "failed": totals.get("FAIL", 0),
        "partial": totals.get("PARTIAL", 0),
        "pass_rate": summary.get("pass_rate", 0),
        "per_section": summary.get("per_section", {})
    }


# =============================================================================
# MAIN GENERATOR
# =============================================================================

def generate_audit_pack(output_dir: Path) -> Dict[str, Any]:
    """Generate complete audit pack."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("QBM AUDIT PACK GENERATOR")
    print("=" * 70)
    
    audit_pack = {
        "version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts/generate_audit_pack.py"
    }
    
    # 1. Input hashes
    print("\n1. Generating input file hashes...")
    audit_pack["input_hashes"] = generate_input_hashes()
    input_path = output_dir / "input_hashes.json"
    with open(input_path, "w", encoding="utf-8") as f:
        json.dump(audit_pack["input_hashes"], f, indent=2, ensure_ascii=False)
    print(f"   Saved: {input_path}")
    
    # 2. Output hashes
    print("\n2. Generating output file hashes...")
    audit_pack["output_hashes"] = generate_output_hashes()
    output_path = output_dir / "output_hashes.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(audit_pack["output_hashes"], f, indent=2, ensure_ascii=False)
    print(f"   Saved: {output_path}")
    
    # 3. GPU proof
    print("\n3. Generating GPU proof logs...")
    audit_pack["gpu_proof"] = generate_gpu_proof()
    gpu_path = output_dir / "gpu_proof.json"
    with open(gpu_path, "w", encoding="utf-8") as f:
        json.dump(audit_pack["gpu_proof"], f, indent=2, ensure_ascii=False)
    print(f"   Saved: {gpu_path}")
    
    # 4. Provenance report
    print("\n4. Generating provenance completeness report...")
    audit_pack["provenance"] = generate_provenance_report()
    prov_path = output_dir / "provenance_report.json"
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(audit_pack["provenance"], f, indent=2, ensure_ascii=False)
    print(f"   Saved: {prov_path}")
    
    # 5. System info
    print("\n5. Generating system info...")
    audit_pack["system_info"] = generate_system_info()
    sys_path = output_dir / "system_info.json"
    with open(sys_path, "w", encoding="utf-8") as f:
        json.dump(audit_pack["system_info"], f, indent=2, ensure_ascii=False)
    print(f"   Saved: {sys_path}")
    
    # 6. Benchmark results
    print("\n6. Getting benchmark results...")
    audit_pack["benchmark"] = get_latest_benchmark_results()
    bench_path = output_dir / "benchmark_results.json"
    with open(bench_path, "w", encoding="utf-8") as f:
        json.dump(audit_pack["benchmark"], f, indent=2, ensure_ascii=False)
    print(f"   Saved: {bench_path}")
    
    # 7. Complete audit pack
    print("\n7. Generating complete audit pack...")
    pack_path = output_dir / "audit_pack.json"
    with open(pack_path, "w", encoding="utf-8") as f:
        json.dump(audit_pack, f, indent=2, ensure_ascii=False)
    print(f"   Saved: {pack_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("AUDIT PACK SUMMARY")
    print("=" * 70)
    print(f"\nInput files: {audit_pack['input_hashes']['file_count']}")
    print(f"Output files: {audit_pack['output_hashes']['file_count']}")
    print(f"GPU available: {audit_pack['gpu_proof'].get('cuda_available', False)}")
    print(f"Behaviors complete: {audit_pack['provenance']['completeness']['behaviors_complete']}")
    print(f"Graph complete: {audit_pack['provenance']['completeness']['graph_complete']}")
    print(f"Tafsir complete: {audit_pack['provenance']['completeness']['tafsir_complete']}")
    print(f"Benchmark pass rate: {audit_pack['benchmark'].get('pass_rate', 'N/A')}%")
    print(f"\nAudit pack saved to: {output_dir}")
    print("=" * 70)
    
    return audit_pack


def main():
    parser = argparse.ArgumentParser(description="Generate QBM audit pack")
    parser.add_argument(
        "--output-dir",
        default="artifacts/audit_pack",
        help="Output directory for audit pack"
    )
    
    args = parser.parse_args()
    output_dir = PROJECT_ROOT / args.output_dir
    
    audit_pack = generate_audit_pack(output_dir)
    
    # Return success if all completeness checks pass
    if audit_pack["provenance"]["completeness"]["all_complete"]:
        return 0
    else:
        print("\nWARNING: Some completeness checks failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
