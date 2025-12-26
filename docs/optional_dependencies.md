# Optional Dependencies

This document describes optional dependencies in the QBM system and how to enable/disable features.

## torch_geometric (Graph Reasoning)

**Status**: Optional  
**Feature**: Layer 5 Graph Neural Network reasoning for multi-hop behavioral discovery

### Installation

```bash
# Full installation (requires compatible CUDA/PyTorch versions)
pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv

# Or install from PyG wheels for your specific PyTorch/CUDA version
# See: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
```

### Windows DLL Issues

On Windows, torch_geometric may fail to load with errors like:
```
UserWarning: An issue occurred while importing 'pyg-lib'. Disabling its usage.
Stacktrace: [WinError 127] The specified procedure could not be found
```

This is a known issue with PyG on Windows. The system handles this gracefully:
- Graph reasoning features return empty results with `status="unavailable"`
- All other features (retrieval, tafsir, proof system) work normally

### Behavior When Unavailable

When torch_geometric is not installed or fails to load:

1. **Graph Reasoner**: Returns empty graph evidence
2. **Proof System**: Graph component shows `fallback_used: true` with reason `"graph: torch_geometric unavailable"`
3. **Tests**: Graph-specific tests are skipped automatically

### Enabling/Disabling

The graph reasoner is lazily imported only when needed:

```python
# This does NOT trigger torch_geometric import
from src.ml import get_hybrid_system

# This DOES trigger torch_geometric import (if available)
from src.ml import get_reasoning_engine
engine = get_reasoning_engine()
```

## Running Tests Without Optional Dependencies

To run the test suite without graph-related tests:

```bash
# Run all tests except graph-marked tests
pytest tests/ -v -m "not graph"

# Run only unit and integration tests (safe on Windows without PyG)
pytest tests/ -v -m "unit or integration"
```

## CI Configuration

For CI environments without torch_geometric:

```yaml
# .github/workflows/test.yml
- name: Run tests (no graph)
  run: pytest tests/ -v -m "not graph" --ignore=tests/test_graph_reasoner.py
```

## UTF-8 Encoding

All file reads in the system must use explicit UTF-8 encoding:

```python
# Correct
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Incorrect (will fail on Windows with Arabic text)
with open(file_path, 'r') as f:
    data = json.load(f)
```

This is enforced by the test `test_evidence_index_loads_utf8()`.
