#!/usr/bin/env python3
"""
Phase 10.1e: PyG Readiness Probe

This script must be run BEFORE enabling QBM_ENABLE_PYG=1.
It verifies that torch_geometric and all its dependencies are
properly installed and can load without crashing.

On Windows, ABI-mismatched DLLs can hard-crash the Python process
in a way that try/except CANNOT catch. This script runs as a
separate process to safely probe the environment.

Usage:
    python scripts/check_pyg_health.py

If this script completes successfully, you can safely set:
    QBM_ENABLE_PYG=1

If it crashes or fails, do NOT enable PyG - use json_fallback instead.
"""

import sys
import os

def main():
    print("=" * 60)
    print("QBM PyG Readiness Probe")
    print("=" * 60)
    
    errors = []
    warnings = []
    
    # 1. Check Python version
    print(f"\n[1/7] Python: {sys.version}")
    if sys.version_info < (3, 9):
        warnings.append("Python 3.9+ recommended for PyG")
    
    # 2. Check torch
    print("\n[2/7] Checking PyTorch...")
    try:
        import torch
        print(f"  torch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            warnings.append("CUDA not available - PyG will run on CPU only")
    except ImportError as e:
        errors.append(f"torch import failed: {e}")
        print(f"  ERROR: {e}")
    except Exception as e:
        errors.append(f"torch check failed: {e}")
        print(f"  ERROR: {e}")
    
    # 3. Check torch_geometric
    print("\n[3/7] Checking torch_geometric...")
    try:
        import torch_geometric
        print(f"  torch_geometric version: {torch_geometric.__version__}")
    except ImportError as e:
        errors.append(f"torch_geometric import failed: {e}")
        print(f"  ERROR: {e}")
    except OSError as e:
        errors.append(f"torch_geometric DLL load failed: {e}")
        print(f"  ERROR (DLL): {e}")
    except Exception as e:
        errors.append(f"torch_geometric check failed: {e}")
        print(f"  ERROR: {e}")
    
    # 4. Check torch_sparse
    print("\n[4/7] Checking torch_sparse...")
    try:
        import torch_sparse
        print(f"  torch_sparse version: {torch_sparse.__version__ if hasattr(torch_sparse, '__version__') else 'unknown'}")
    except ImportError as e:
        warnings.append(f"torch_sparse not installed: {e}")
        print(f"  WARNING: {e}")
    except Exception as e:
        warnings.append(f"torch_sparse check failed: {e}")
        print(f"  WARNING: {e}")
    
    # 5. Check torch_scatter
    print("\n[5/7] Checking torch_scatter...")
    try:
        import torch_scatter
        print(f"  torch_scatter version: {torch_scatter.__version__ if hasattr(torch_scatter, '__version__') else 'unknown'}")
    except ImportError as e:
        warnings.append(f"torch_scatter not installed: {e}")
        print(f"  WARNING: {e}")
    except Exception as e:
        warnings.append(f"torch_scatter check failed: {e}")
        print(f"  WARNING: {e}")
    
    # 6. Check torch_cluster
    print("\n[6/7] Checking torch_cluster...")
    try:
        import torch_cluster
        print(f"  torch_cluster version: {torch_cluster.__version__ if hasattr(torch_cluster, '__version__') else 'unknown'}")
    except ImportError as e:
        warnings.append(f"torch_cluster not installed: {e}")
        print(f"  WARNING: {e}")
    except Exception as e:
        warnings.append(f"torch_cluster check failed: {e}")
        print(f"  WARNING: {e}")
    
    # 7. Run a tiny no-op to force-load kernels
    print("\n[7/7] Running kernel load test...")
    try:
        import torch
        from torch_geometric.nn import GATConv
        from torch_geometric.data import Data
        
        # Create a tiny graph and run a forward pass
        x = torch.randn(4, 16)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        
        conv = GATConv(16, 8, heads=2)
        out = conv(x, edge_index)
        
        print(f"  GATConv forward pass: OK (output shape: {out.shape})")
        
        # Test Data object
        data = Data(x=x, edge_index=edge_index)
        print(f"  Data object creation: OK (nodes: {data.num_nodes}, edges: {data.num_edges})")
        
    except Exception as e:
        errors.append(f"Kernel load test failed: {e}")
        print(f"  ERROR: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
    
    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")
    
    if not errors:
        print("\n" + "=" * 60)
        print("SUCCESS: PyG environment is healthy!")
        print("=" * 60)
        print("\nYou can now safely enable PyG by setting:")
        print("  export QBM_ENABLE_PYG=1  (Linux/Mac)")
        print("  set QBM_ENABLE_PYG=1     (Windows CMD)")
        print("  $env:QBM_ENABLE_PYG='1'  (Windows PowerShell)")
        return 0
    else:
        print("\n" + "=" * 60)
        print("FAILED: PyG environment has errors!")
        print("=" * 60)
        print("\nDo NOT enable QBM_ENABLE_PYG=1")
        print("The system will use json_fallback mode instead.")
        print("\nTo fix, try:")
        print("  pip install torch-geometric torch-sparse torch-scatter torch-cluster")
        print("  (ensure versions match your PyTorch and CUDA versions)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
