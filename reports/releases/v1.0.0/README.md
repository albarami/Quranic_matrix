# QBM Enterprise Brain v1.0.0 - Academic Release

> **Release Date**: 2025-12-31  
> **Commit**: a1513b91db26e74e6bde32b954be607a02d62f30  
> **Status**: Release Governance Complete

---

## Overview

This release represents the first enterprise-grade version of the Quranic Behavioral Matrix (QBM) system. It implements a zero-hallucination, Arabic-first, academically defensible architecture for analyzing human behavior in the Quran.

### Key Capabilities

| Capability | Description |
|------------|-------------|
| **GRAPH_CAUSAL** | Multi-hop causal chain analysis with edge provenance |
| **TAFSIR_MULTI_SOURCE** | 7 canonical tafsir sources with verse-keyed citations |
| **CONSENSUS** | Cross-source agreement/divergence analysis |
| **PROVENANCE** | Every claim backed by verse_key + tafsir source |
| **11D_AXES** | Bouzidani's 5-axis behavioral classification |

---

## How to Reproduce

### Prerequisites

```bash
# Python 3.11+
pip install -r requirements.txt

# Environment variables (see .env.example)
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_API_KEY=...
```

### Generate Answers (Strict Mode)

```bash
# Run benchmark with strict verifier
python -m pytest tests/test_benchmark_legendary.py -v

# Generate single question output
python scripts/run_query.py --question-id A01 --strict --output reports/releases/v1.0.0/showcase/A01.json
```

### Verify Audit Pack

```bash
# Generate audit pack (requires clean git tree)
python scripts/generate_audit_pack.py --strict

# Validate audit pack
QBM_REQUIRE_AUDIT_PACK=1 python -m pytest tests/test_audit_pack_phase7.py -v
```

---

## What Each Section Proves

### Section A: Graph Causality (A01-A25)

Demonstrates multi-hop causal chain analysis:
- **A01**: Complete destruction pathway (الغفلة → الكفر)
- **A03**: Circular reinforcement cycles (A→B→C→A)
- **A11**: Causal strength quantification

Each answer includes:
- `paths[]` with intermediate behaviors
- `edge_provenance` for each link
- `verse_keys_per_link` from SSOT

### Section B: Tafsir Divergence (B01-B25)

Demonstrates multi-source scholarly analysis:
- **B02**: Riba interpretation divergence matrix
- **B11**: Disputed behaviors with contradictory interpretations

Each answer includes:
- `per_source_provenance` (Ibn Kathir, Tabari, Qurtubi, Sa'di, Jalalayn)
- `agreement_metrics` (consensus percentage)

### Section D: 11-Dimensional Axes (D01-D25)

Demonstrates Bouzidani's behavioral classification:
- **D07**: Organ-behavior mapping (اليد، العين، القلب)
- **D24**: Temporal classification (صباحاً، ليلاً)

Each answer includes:
- `axis_classification` per behavior
- `verse_evidence` for classification

### Section J: Integration (J01-J25)

Demonstrates cross-capability synthesis:
- **J11**: Complete behavioral profile with all axes
- **J15**: Causal + tafsir + axes integration

---

## Interpretation Rules

### Direct Lexical vs Annotation-Derived

| Type | Description | Example |
|------|-------------|---------|
| **Direct Lexical** | Behavior term appears in verse text | "الصبر" in 2:45 |
| **Annotation-Derived** | Behavior inferred from tafsir | Patience implied in 2:153 context |

### Claim Structure

Every claim must have:
```json
{
  "claim_id": "C1",
  "text": "Patience is commanded in the Quran",
  "supporting_verse_keys": ["2:45", "2:153"],
  "supporting_tafsir_refs": [
    {"source": "ibn_kathir", "verse_key": "2:45"}
  ]
}
```

### Strict Mode Behavior

In strict mode, the verifier **fails closed** on:
- Missing `claims[]` for substantive responses
- Empty `supporting_verse_keys`
- Invalid verse keys (not in SSOT)
- Narrative references to non-existent claim IDs
- Surah introductions in tafsir (must be verse-specific)

---

## Audit Pack Contents

| File | Description |
|------|-------------|
| `audit_pack.json` | Main manifest with git commit, validation status |
| `input_hashes.json` | SHA-256 hashes of SSOT inputs |
| `output_hashes.json` | SHA-256 hashes of KB outputs |
| `gpu_proof.json` | GPU computation proof (if claimed) |
| `benchmark_results.json` | 200/200 pass rate |
| `provenance_report.json` | Completeness metrics |

### GPU Proof Fields

```json
{
  "gpu_compute_claimed": true,
  "gpu_proof_valid": true,
  "gpus_available": 8,
  "gpus_utilized": [0],
  "multi_gpu_used": false
}
```

---

## Citation

```bibtex
@software{qbm_enterprise_brain,
  title = {Quranic Behavioral Matrix (QBM) Enterprise Brain},
  version = {1.0.0},
  year = {2025},
  author = {Bouzidani et al.},
  url = {https://github.com/albarami/Quranic_matrix}
}
```

---

## Contact

For academic inquiries, contact the repository maintainers.
