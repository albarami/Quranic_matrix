# Phase 5: Environment Snapshot

**Date**: 2025-12-25  
**Purpose**: Freeze environment state before benchmark restructuring

---

## Python Environment

| Package | Version |
|---------|---------|
| Python | 3.11.8 |
| PyTorch | 2.5.1+cu121 |
| Transformers | 4.57.1 |
| Sentence-Transformers | 5.1.2 |
| NumPy | 2.2.6 |
| SciPy | 1.16.2 |
| Scikit-learn | 1.7.2 |

---

## GPU Configuration

| Property | Value |
|----------|-------|
| CUDA Available | True |
| CUDA Version | 12.1 |
| GPU Count | 8 |

### GPU Details

| GPU | Model |
|-----|-------|
| GPU 0 | NVIDIA A100-SXM4-80GB |
| GPU 1 | NVIDIA A100-SXM4-80GB |
| GPU 2 | NVIDIA A100-SXM4-80GB |
| GPU 3 | NVIDIA A100-SXM4-80GB |
| GPU 4 | NVIDIA A100-SXM4-80GB |
| GPU 5 | NVIDIA A100-SXM4-80GB |
| GPU 6 | NVIDIA A100-SXM4-80GB |
| GPU 7 | NVIDIA A100-SXM4-80GB |

**Total VRAM**: 640 GB

---

## Status

- ✅ Environment frozen - no further torch/transformers upgrades
- ✅ CUDA functional
- ✅ Multi-GPU available for batch processing

---

## Known Issues

1. **CAMeL model loading**: Requires TransformersBackend due to torch.load security check (torch < 2.6)
2. **torchvision/torchaudio**: May have version mismatch warnings (non-blocking)
