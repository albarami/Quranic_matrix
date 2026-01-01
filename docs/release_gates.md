# QBM Release Gates

## Overview

This document defines the **pass/fail criteria** for QBM releases. All gates must pass before a release can be tagged and deployed.

**Canonical Truth: 87 behaviors** - This is non-negotiable across all lanes.

---

## Two-Lane CI Architecture

| Lane | Runner | Dataset Mode | Purpose |
|------|--------|--------------|---------|
| **Fixture** | GitHub-hosted (ubuntu) | `fixture` | Fast CI on every push |
| **Full SSOT** | Self-hosted Windows | `full` | Release validation with complete data |

---

## Fixture Lane Gates (GitHub-hosted)

These gates run on every push to `main` or PR.

### Gate F1: Canonical Entity Count
- **Requirement**: `expected_behavior_count() == 87`
- **Test**: `tests/phase8/test_all_behaviors_contract.py`
- **Failure Action**: Fix fixture bootstrap to include all 87 behaviors

### Gate F2: Graph Behavior Nodes
- **Requirement**: `graph_v3.json` contains exactly 87 behavior nodes
- **Test**: `tests/phase7/test_graph_projection.py::test_behavior_nodes_match_registry`
- **Failure Action**: Rebuild graph with `scripts/ci_bootstrap_all.py`

### Gate F3: Concept Index Completeness
- **Requirement**: `concept_index_v3.jsonl` has 87 entries
- **Test**: `tests/phase8/test_all_behaviors_contract.py::TestCrossModuleConsistency`
- **Failure Action**: Run `ensure_all_canonical_entities()` in bootstrap

### Gate F4: Validation Report Consistency
- **Requirement**: Validation report shows 87 behaviors passed
- **Test**: `tests/phase5/test_validation_gates.py`
- **Failure Action**: Rebuild validation report

### Gate F5: Unit/Integration Tests
- **Requirement**: All Tier A tests pass
- **Test**: `pytest tests/ -m "not gpu and not slow and not tier_b"`
- **Failure Action**: Fix failing tests

### Gate F6: Lint
- **Requirement**: No critical lint errors
- **Test**: `ruff check src/ tests/`
- **Failure Action**: Fix lint issues

---

## Full SSOT Lane Gates (Self-hosted Windows)

These gates run on tags or `workflow_dispatch` only.

### Gate S1: Dataset Mode
- **Requirement**: `QBM_DATASET_MODE == "full"`
- **Check**: Environment variable verification
- **Failure Action**: Set correct environment variables

### Gate S2: SSOT Directory Exists
- **Requirement**: `QBM_SSOT_DIR` exists and is accessible
- **Check**: `scripts/preflight_full_release.py`
- **Failure Action**: Create/populate SSOT directory

### Gate S3: SSOT Manifest Validation
- **Requirement**: All files in manifest exist with correct hashes
- **Check**: `scripts/resolve_ssot.py`
- **Failure Action**: Populate missing SSOT files

### Gate S4: Canonical Behavior Count (Full)
- **Requirement**: 87 behaviors in registry, index, graph, and API
- **Check**: `scripts/preflight_full_release.py`
- **Failure Action**: Rebuild artifacts from full SSOT

### Gate S5: Full Benchmark (200 Questions)
- **Requirement**: `total_questions == 200` in benchmark report
- **Check**: `scripts/run_qbm_benchmark.py --dataset data/benchmarks/qbm_legendary_200.v1.jsonl`
- **Failure Action**: Ensure full benchmark dataset is used

### Gate S6: Benchmark Pass Rate
- **Requirement**: Pass rate meets minimum threshold (TBD based on baseline)
- **Check**: Benchmark report analysis
- **Failure Action**: Improve retrieval/proof quality

### Gate S7: Strict Audit Pack
- **Requirement**: `python scripts/generate_audit_pack.py --strict` passes with 0 missing SSOT
- **Check**: Audit pack generation script
- **Failure Action**: Populate all required SSOT files

### Gate S8: Audit Pack Validation
- **Requirement**: `pytest tests/test_audit_pack_phase7.py -v` passes
- **Check**: Audit pack test suite
- **Failure Action**: Fix audit pack issues

### Gate S9: API Health Check
- **Requirement**: Backend API starts and returns 200 on health endpoint
- **Check**: API smoke test
- **Failure Action**: Fix API startup issues

---

## Pre-Release Checklist

Before tagging a release:

- [ ] Fixture lane CI is green
- [ ] Full SSOT lane passes all gates (S1-S9)
- [ ] No `if: false` hacks in CI workflow
- [ ] All 87 behaviors present everywhere
- [ ] Benchmark results documented
- [ ] Audit pack archived
- [ ] CHANGELOG.md updated

---

## Tag Discipline

- **Tags are immutable** - never move a tag after it's pushed
- **Versioning**: Use semantic versioning (v1.1.0, v1.1.1, etc.)
- **Hotfixes**: Create new patch version (v1.1.1) instead of moving tag
- **Pre-releases**: Use `-rc.N` suffix (v1.2.0-rc.1)

---

## Environment Variables

### Fixture Lane (GitHub-hosted)
```bash
QBM_DATASET_MODE=fixture
QBM_USE_FIXTURE=1
QBM_DATA_MODE=fixture
```

### Full SSOT Lane (Self-hosted Windows)
```bash
QBM_DATASET_MODE=full
QBM_SSOT_MODE=full
QBM_USE_FIXTURE=0
QBM_SSOT_DIR=D:\Quran_matrix\ssot_full
PYTHONUTF8=1
PYTHONIOENCODING=utf-8
```

---

## SSOT Directory Structure

Required layout for `D:\Quran_matrix\ssot_full\`:

```
ssot_full/
├── data/
│   ├── quran/
│   │   └── quran_uthmani.xml
│   └── tafsir/
│       ├── tafsir_ibn_kathir.jsonl
│       ├── tafsir_tabari.jsonl
│       ├── tafsir_qurtubi.jsonl
│       ├── tafsir_saadi.jsonl
│       ├── tafsir_jalalayn.jsonl
│       ├── tafsir_baghawi.jsonl
│       └── tafsir_muyassar.jsonl
├── indexes/
│   └── (generated indexes)
└── manifest.json
```

---

## Failure Escalation

1. **Fixture lane fails**: Block merge, fix immediately
2. **Full SSOT lane fails**: Do not tag release, investigate
3. **Post-release issue**: Create hotfix branch, new patch version

---

## Metrics Targets (Future)

Once fully operational, enforce these in release lane:

| Metric | Target | Gate |
|--------|--------|------|
| Benchmark pass rate | ≥ 80% | S6 |
| Evidence coverage | ≥ 90% | New |
| API latency (p95) | < 500ms | New |
| Retrieval recall | ≥ 85% | New |

---

*Last updated: 2026-01-01*
