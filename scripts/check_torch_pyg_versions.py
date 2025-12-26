#!/usr/bin/env python3
"""
Version Check Script for PyTorch + torch_geometric Stack

Verifies that all components are ABI-compatible.
Fails fast if versions are mismatched.

Usage:
    python scripts/check_torch_pyg_versions.py
"""

import sys


def check_versions():
    """Check and print all relevant versions."""
    print("=" * 60)
    print("PyTorch + torch_geometric Version Check")
    print("=" * 60)
    
    errors = []
    warnings = []
    
    # 1. Check torch
    try:
        import torch
        torch_version = torch.__version__
        cuda_version = torch.version.cuda
        cuda_available = torch.cuda.is_available()
        
        print(f"\n[torch]")
        print(f"  Version: {torch_version}")
        print(f"  CUDA version: {cuda_version}")
        print(f"  CUDA available: {cuda_available}")
        
        if cuda_version is None:
            warnings.append("torch is CPU-only build (no CUDA)")
        elif not cuda_available:
            warnings.append("CUDA version set but not available (driver issue?)")
            
    except ImportError as e:
        errors.append(f"torch not installed: {e}")
        print(f"\n[torch] NOT INSTALLED")
        return errors, warnings
    
    # 2. Check torch_geometric
    try:
        import torch_geometric
        pyg_version = torch_geometric.__version__
        print(f"\n[torch_geometric]")
        print(f"  Version: {pyg_version}")
    except ImportError:
        warnings.append("torch_geometric not installed (optional)")
        print(f"\n[torch_geometric] NOT INSTALLED (optional)")
    except Exception as e:
        errors.append(f"torch_geometric import failed: {e}")
        print(f"\n[torch_geometric] IMPORT FAILED: {e}")
    
    # 3. Check PyG compiled extensions
    pyg_extensions = [
        ("pyg_lib", "pyg_lib"),
        ("torch_scatter", "torch_scatter"),
        ("torch_sparse", "torch_sparse"),
        ("torch_cluster", "torch_cluster"),
        ("torch_spline_conv", "torch_spline_conv"),
    ]
    
    print(f"\n[PyG Extensions]")
    for name, module_name in pyg_extensions:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "unknown")
            print(f"  {name}: {version}")
        except ImportError:
            print(f"  {name}: not installed")
        except Exception as e:
            error_str = str(e)
            if "0xc0000139" in error_str or "WinError 127" in error_str:
                errors.append(f"{name}: DLL load failed (ABI mismatch)")
                print(f"  {name}: DLL LOAD FAILED (ABI mismatch)")
            else:
                warnings.append(f"{name}: import error: {e}")
                print(f"  {name}: error - {e}")
    
    # 4. Platform info
    import platform
    print(f"\n[Platform]")
    print(f"  Python: {sys.version}")
    print(f"  OS: {platform.platform()}")
    
    # 5. Recommended install command
    if cuda_version:
        cuda_tag = f"cu{cuda_version.replace('.', '')[:3]}"
        torch_tag = torch_version.split('+')[0]
        print(f"\n[Recommended PyG Install]")
        print(f"  For torch {torch_version}:")
        print(f"  pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \\")
        print(f"    -f https://data.pyg.org/whl/torch-{torch_tag}+{cuda_tag}.html")
        print(f"  pip install torch-geometric")
    
    # Summary
    print(f"\n{'=' * 60}")
    if errors:
        print(f"ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  ❌ {e}")
    if warnings:
        print(f"WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  ⚠️  {w}")
    if not errors and not warnings:
        print("✅ All components compatible")
    print("=" * 60)
    
    return errors, warnings


def main():
    errors, warnings = check_versions()
    
    if errors:
        print("\n❌ FAIL: Critical errors detected. Fix before proceeding.")
        sys.exit(1)
    elif warnings:
        print("\n⚠️  WARN: Some warnings. System may work with fallbacks.")
        sys.exit(0)
    else:
        print("\n✅ PASS: All versions compatible.")
        sys.exit(0)


if __name__ == "__main__":
    main()
