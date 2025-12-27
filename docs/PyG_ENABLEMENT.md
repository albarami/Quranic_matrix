# PyG (torch_geometric) Enablement Guide

> Phase 10 Enterprise Hardening: Safe PyG Enablement

## Overview

The QBM system uses PyTorch Geometric (PyG) for Graph Neural Network reasoning. However, on Windows, ABI-mismatched DLLs can **hard-crash the Python process** in a way that `try/except` **cannot catch**.

To prevent this, PyG is **opt-in only** via environment variable. The system will use a JSON-based fallback graph reasoner by default.

## Quick Reference

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `QBM_ENABLE_PYG` | `0` | Set to `1` to enable PyG after validation |
| `QBM_FULLPOWER_READY` | `0` | Set to `1` to allow FullPower index building |

## Step-by-Step: Enabling PyG Safely

### Step 1: Run the Readiness Probe

Before enabling PyG, run the health check script in a **separate process**:

```bash
python scripts/check_pyg_health.py
```

This script will:
1. Check Python version
2. Verify PyTorch installation and CUDA availability
3. Verify torch_geometric installation
4. Check optional dependencies (torch_sparse, torch_scatter, torch_cluster)
5. Run a kernel load test (GATConv forward pass)

**Expected output on success:**
```
============================================================
SUCCESS: PyG environment is healthy!
============================================================

You can now safely enable PyG by setting:
  export QBM_ENABLE_PYG=1  (Linux/Mac)
  set QBM_ENABLE_PYG=1     (Windows CMD)
  $env:QBM_ENABLE_PYG='1'  (Windows PowerShell)
```

### Step 2: Enable PyG

Only if Step 1 succeeds, set the environment variable:

**Linux/Mac:**
```bash
export QBM_ENABLE_PYG=1
```

**Windows CMD:**
```cmd
set QBM_ENABLE_PYG=1
```

**Windows PowerShell:**
```powershell
$env:QBM_ENABLE_PYG='1'
```

**Docker/docker-compose:**
```yaml
environment:
  - QBM_ENABLE_PYG=1
```

### Step 3: Verify Backend Mode

After starting the API, check the `/api/proof/query` response:

```json
{
  "debug": {
    "graph_backend": "pyg",
    "graph_backend_reason": "torch_geometric available and loaded (QBM_ENABLE_PYG=1)"
  }
}
```

## Fallback Mode (Default)

When `QBM_ENABLE_PYG` is not set or set to `0`:

- **No PyG import is attempted** (prevents Windows DLL crashes)
- System uses JSON-based semantic graph (`data/graph/semantic_graph_v2.json`)
- All graph reasoning features work via BFS/DFS on the JSON graph
- API response shows:

```json
{
  "debug": {
    "graph_backend": "json_fallback",
    "graph_backend_reason": "PyG not enabled (set QBM_ENABLE_PYG=1 after running check_pyg_health.py)"
  }
}
```

## Troubleshooting

### Health Check Fails

If `check_pyg_health.py` fails:

1. **torch not found**: Install PyTorch first
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   ```

2. **torch_geometric not found**: Install PyG
   ```bash
   pip install torch-geometric
   ```

3. **DLL load error (Windows)**: Version mismatch
   - Ensure PyTorch, CUDA, and PyG versions are compatible
   - See: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

4. **Kernel load test fails**: GPU driver issue
   - Update NVIDIA drivers
   - Try CPU-only mode first

### API Returns 503

If the API returns `503 Service Unavailable` with `"error": "full_power_index_missing"`:

1. Build the FullPower index:
   ```bash
   python -m src.ml.full_power_system
   ```

2. Or set the flag to build on first request:
   ```bash
   export QBM_FULLPOWER_READY=1
   ```

## Test Tiers

| Tier | Marker | Requirements | Description |
|------|--------|--------------|-------------|
| **A** | `tier_a` | None (CPU only) | Schema, routing, JSON fallback, no-fabrication |
| **B** | `tier_b` | `QBM_FULLPOWER_READY=1` + index | FullPower GPU tests |

Run Tier A tests (CI default):
```bash
pytest tests/ -m "tier_a or (not tier_b)" -v
```

Run Tier B tests (requires GPU + index):
```bash
QBM_FULLPOWER_READY=1 pytest tests/ -m "tier_b" -v
```

## CI Configuration

CI runs Tier A tests by default on Ubuntu (no GPU). PyG is **not enabled** in CI to ensure clean, reproducible builds.

To run GPU tests in CI, use a self-hosted runner with:
- NVIDIA GPU
- CUDA toolkit
- PyG installed and validated via `check_pyg_health.py`
- `QBM_ENABLE_PYG=1` and `QBM_FULLPOWER_READY=1` set

## Security Notes

- **Never auto-detect PyG** — always require explicit opt-in
- **Run health check in separate process** — crashes won't affect main app
- **Log backend mode** — always visible in API debug for transparency
- **No silent fallback** — backend reason always explains why
