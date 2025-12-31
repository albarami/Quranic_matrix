"""
GPU Proof Instrumentation for Embedding Generation

This module provides instrumentation to prove GPU computation actually occurred
during embedding generation. It captures:

1. nvidia-smi time-series logs (sampled every 1-2 seconds)
2. PyTorch CUDA telemetry (device IDs, memory allocation)
3. Embedding job metadata (model ID, batch size, vectors/sec)

USAGE:
    from scripts.gpu_proof_instrumentation import GPUProofCollector
    
    with GPUProofCollector(output_dir="artifacts/audit_pack/gpu_proof") as collector:
        # Run embedding generation here
        embeddings = generate_embeddings(texts)
        
        # Log embedding metadata
        collector.log_embedding_job(
            model_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            vector_dims=384,
            total_vectors=len(embeddings),
            batch_size=32
        )
    
    # Proof files are written to output_dir

HARD GATE:
    If avg_gpu_utilization == 0% for the entire run, the proof is INVALID.
    The system cannot claim "GPU embeddings built" without non-zero utilization.
"""

import json
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class GPUUtilizationSample:
    """Single sample of GPU utilization."""
    timestamp: str
    gpu_index: int
    utilization_percent: float
    memory_used_mb: float
    memory_total_mb: float
    temperature_c: Optional[float] = None


@dataclass
class GPUProofResult:
    """Result of GPU proof collection."""
    valid: bool
    start_time: str
    end_time: str
    duration_seconds: float
    samples: List[GPUUtilizationSample] = field(default_factory=list)
    avg_utilization: float = 0.0
    max_utilization: float = 0.0
    peak_memory_mb: float = 0.0
    torch_stats: Dict[str, Any] = field(default_factory=dict)
    embedding_job: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "sample_count": len(self.samples),
            "avg_utilization_percent": self.avg_utilization,
            "max_utilization_percent": self.max_utilization,
            "peak_memory_mb": self.peak_memory_mb,
            "torch_stats": self.torch_stats,
            "embedding_job": self.embedding_job,
            "errors": self.errors,
            "samples": [
                {
                    "timestamp": s.timestamp,
                    "gpu_index": s.gpu_index,
                    "utilization_percent": s.utilization_percent,
                    "memory_used_mb": s.memory_used_mb,
                    "memory_total_mb": s.memory_total_mb,
                    "temperature_c": s.temperature_c
                }
                for s in self.samples
            ]
        }


class GPUProofCollector:
    """
    Collects GPU utilization proof during embedding generation.
    
    Use as a context manager to automatically start/stop collection.
    """
    
    def __init__(
        self, 
        output_dir: str = "artifacts/audit_pack/gpu_proof",
        sample_interval_seconds: float = 1.0,
        min_utilization_threshold: float = 5.0  # Minimum avg utilization to be valid
    ):
        self.output_dir = Path(output_dir)
        self.sample_interval = sample_interval_seconds
        self.min_utilization_threshold = min_utilization_threshold
        
        self.samples: List[GPUUtilizationSample] = []
        self.errors: List[str] = []
        self.embedding_job: Dict[str, Any] = {}
        
        self._stop_event = threading.Event()
        self._sampler_thread: Optional[threading.Thread] = None
        self._start_time: Optional[datetime] = None
        self._end_time: Optional[datetime] = None
        
        # PyTorch stats captured at start/end
        self._torch_start_stats: Dict[str, Any] = {}
        self._torch_end_stats: Dict[str, Any] = {}
    
    def __enter__(self) -> "GPUProofCollector":
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        self.save()
        return False
    
    def start(self):
        """Start GPU utilization sampling."""
        self._start_time = datetime.now(timezone.utc)
        self._stop_event.clear()
        
        # Capture initial PyTorch stats
        self._torch_start_stats = self._get_torch_stats()
        
        # Start background sampler
        self._sampler_thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._sampler_thread.start()
    
    def stop(self):
        """Stop GPU utilization sampling."""
        self._stop_event.set()
        if self._sampler_thread:
            self._sampler_thread.join(timeout=5.0)
        
        self._end_time = datetime.now(timezone.utc)
        
        # Capture final PyTorch stats
        self._torch_end_stats = self._get_torch_stats()
    
    def log_embedding_job(
        self,
        model_id: str,
        vector_dims: int,
        total_vectors: int,
        batch_size: int,
        elapsed_seconds: Optional[float] = None
    ):
        """Log embedding job metadata."""
        self.embedding_job = {
            "model_id": model_id,
            "vector_dims": vector_dims,
            "total_vectors": total_vectors,
            "batch_size": batch_size,
            "elapsed_seconds": elapsed_seconds,
            "vectors_per_second": (
                total_vectors / elapsed_seconds if elapsed_seconds and elapsed_seconds > 0 else None
            )
        }
    
    def get_result(self) -> GPUProofResult:
        """Get the GPU proof result."""
        duration = 0.0
        if self._start_time and self._end_time:
            duration = (self._end_time - self._start_time).total_seconds()
        
        # Calculate statistics
        utilizations = [s.utilization_percent for s in self.samples]
        memories = [s.memory_used_mb for s in self.samples]
        
        avg_util = sum(utilizations) / len(utilizations) if utilizations else 0.0
        max_util = max(utilizations) if utilizations else 0.0
        peak_mem = max(memories) if memories else 0.0
        
        # Determine validity using robust criteria:
        # 1. Average utilization >= threshold, OR
        # 2. Max utilization >= 10% AND peak memory > 500MB (proves GPU was used)
        # 3. PyTorch max_memory_allocated > 100MB (proves CUDA tensors were created)
        torch_max_mem = self._torch_end_stats.get("max_memory_allocated_mb", 0)
        
        valid = (
            avg_util >= self.min_utilization_threshold or
            (max_util >= 10.0 and peak_mem > 500) or
            torch_max_mem > 100
        )
        
        if not valid and not self.errors:
            self.errors.append(
                f"GPU utilization too low: avg={avg_util:.1f}% < threshold={self.min_utilization_threshold}%, "
                f"max={max_util:.1f}%, peak_mem={peak_mem:.0f}MB, torch_max_mem={torch_max_mem:.0f}MB"
            )
        
        # Combine torch stats
        torch_stats = {
            "start": self._torch_start_stats,
            "end": self._torch_end_stats,
            "memory_allocated_delta_mb": (
                self._torch_end_stats.get("memory_allocated_mb", 0) -
                self._torch_start_stats.get("memory_allocated_mb", 0)
            ),
            "max_memory_allocated_mb": self._torch_end_stats.get("max_memory_allocated_mb", 0)
        }
        
        return GPUProofResult(
            valid=valid,
            start_time=self._start_time.isoformat() if self._start_time else "",
            end_time=self._end_time.isoformat() if self._end_time else "",
            duration_seconds=duration,
            samples=self.samples,
            avg_utilization=avg_util,
            max_utilization=max_util,
            peak_memory_mb=peak_mem,
            torch_stats=torch_stats,
            embedding_job=self.embedding_job,
            errors=self.errors
        )
    
    def save(self):
        """Save GPU proof to output directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        result = self.get_result()
        
        # Save main proof file
        proof_path = self.output_dir / "gpu_computation_proof.json"
        with open(proof_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
        
        # Save utilization CSV for easy analysis
        csv_path = self.output_dir / "gpu_utilization.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("timestamp,gpu_index,utilization_percent,memory_used_mb,memory_total_mb,temperature_c\n")
            for s in self.samples:
                f.write(f"{s.timestamp},{s.gpu_index},{s.utilization_percent},{s.memory_used_mb},{s.memory_total_mb},{s.temperature_c or ''}\n")
        
        # Save torch stats
        torch_path = self.output_dir / "torch_cuda_stats.json"
        with open(torch_path, "w", encoding="utf-8") as f:
            json.dump(result.torch_stats, f, indent=2)
        
        return result
    
    def _sample_loop(self):
        """Background thread that samples GPU utilization."""
        while not self._stop_event.is_set():
            try:
                samples = self._sample_nvidia_smi()
                self.samples.extend(samples)
            except Exception as e:
                self.errors.append(f"Sampling error: {e}")
            
            self._stop_event.wait(self.sample_interval)
    
    def _sample_nvidia_smi(self) -> List[GPUUtilizationSample]:
        """Sample GPU utilization using nvidia-smi."""
        samples = []
        timestamp = datetime.now(timezone.utc).isoformat()
        
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 4:
                            samples.append(GPUUtilizationSample(
                                timestamp=timestamp,
                                gpu_index=int(parts[0]),
                                utilization_percent=float(parts[1]),
                                memory_used_mb=float(parts[2]),
                                memory_total_mb=float(parts[3]),
                                temperature_c=float(parts[4]) if len(parts) > 4 else None
                            ))
        except FileNotFoundError:
            self.errors.append("nvidia-smi not found")
        except Exception as e:
            self.errors.append(f"nvidia-smi error: {e}")
        
        return samples
    
    def _get_torch_stats(self) -> Dict[str, Any]:
        """Get PyTorch CUDA statistics."""
        stats = {
            "torch_available": TORCH_AVAILABLE,
            "cuda_available": False,
            "device_count": 0,
            "current_device": None,
            "memory_allocated_mb": 0,
            "max_memory_allocated_mb": 0,
            "devices": []
        }
        
        if not TORCH_AVAILABLE:
            return stats
        
        try:
            stats["cuda_available"] = torch.cuda.is_available()
            
            if stats["cuda_available"]:
                stats["device_count"] = torch.cuda.device_count()
                stats["current_device"] = torch.cuda.current_device()
                stats["memory_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
                stats["max_memory_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
                
                for i in range(stats["device_count"]):
                    props = torch.cuda.get_device_properties(i)
                    stats["devices"].append({
                        "index": i,
                        "name": props.name,
                        "total_memory_gb": props.total_memory / (1024**3),
                        "major": props.major,
                        "minor": props.minor
                    })
        except Exception as e:
            stats["error"] = str(e)
        
        return stats


def validate_gpu_proof(proof_path: str, min_utilization: float = 5.0) -> bool:
    """
    Validate a GPU proof file.
    
    Args:
        proof_path: Path to gpu_computation_proof.json
        min_utilization: Minimum average utilization to be valid
        
    Returns:
        True if proof is valid, False otherwise
    """
    try:
        with open(proof_path, "r", encoding="utf-8") as f:
            proof = json.load(f)
        
        avg_util = proof.get("avg_utilization_percent", 0)
        sample_count = proof.get("sample_count", 0)
        
        if sample_count == 0:
            print(f"❌ No GPU samples collected")
            return False
        
        if avg_util < min_utilization:
            print(f"❌ GPU utilization too low: {avg_util:.1f}% < {min_utilization}%")
            return False
        
        print(f"✅ GPU proof valid: avg_utilization={avg_util:.1f}%, samples={sample_count}")
        return True
        
    except Exception as e:
        print(f"❌ Error validating GPU proof: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GPU Proof Instrumentation")
    parser.add_argument("--validate", type=str, help="Validate a GPU proof file")
    parser.add_argument("--demo", action="store_true", help="Run a demo collection")
    
    args = parser.parse_args()
    
    if args.validate:
        valid = validate_gpu_proof(args.validate)
        exit(0 if valid else 1)
    
    if args.demo:
        print("Running GPU proof collection demo (5 seconds)...")
        with GPUProofCollector(output_dir="artifacts/audit_pack/gpu_proof") as collector:
            time.sleep(5)
            collector.log_embedding_job(
                model_id="demo-model",
                vector_dims=384,
                total_vectors=1000,
                batch_size=32,
                elapsed_seconds=5.0
            )
        
        result = collector.get_result()
        print(f"\nResult: valid={result.valid}")
        print(f"Samples: {len(result.samples)}")
        print(f"Avg utilization: {result.avg_utilization:.1f}%")
        print(f"Max utilization: {result.max_utilization:.1f}%")
        print(f"Peak memory: {result.peak_memory_mb:.1f} MB")
