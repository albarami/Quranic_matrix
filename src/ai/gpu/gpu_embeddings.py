"""
GPU-Accelerated Embedding Pipeline.

Uses multiple GPUs for fast embedding generation with AraBERT.
Supports batch processing of large datasets (322,939+ annotations).
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


class TextDataset(Dataset):
    """Simple dataset for text batching."""

    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


class GPUEmbeddingPipeline:
    """
    GPU-accelerated embedding generation using AraBERT.
    
    Features:
    - Multi-GPU support (DataParallel)
    - Batch processing for efficiency
    - Memory-efficient processing of large datasets
    - Automatic GPU detection and allocation
    """

    def __init__(
        self,
        model_name: str = "aubmindlab/bert-base-arabertv2",
        batch_size: int = 64,
        max_length: int = 512,
        device: Optional[str] = None,
        use_multi_gpu: bool = True,
    ):
        """
        Initialize GPU embedding pipeline.

        Args:
            model_name: HuggingFace model name for embeddings.
            batch_size: Batch size for processing (adjust based on GPU memory).
            max_length: Maximum token length.
            device: Device to use ('cuda', 'cuda:0', 'cpu', or None for auto).
            use_multi_gpu: Use all available GPUs with DataParallel.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_multi_gpu = use_multi_gpu

        # Detect device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Move to GPU and wrap with DataParallel if multiple GPUs
        self.model = self.model.to(self.device)
        
        if use_multi_gpu and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for embedding generation")
            self.model = torch.nn.DataParallel(self.model)
        
        self.model.eval()

        # Get embedding dimension
        self.embedding_dim = self.model.module.config.hidden_size if hasattr(self.model, 'module') else self.model.config.hidden_size

        print(f"GPU Embedding Pipeline initialized:")
        print(f"  Model: {model_name}")
        print(f"  Device: {self.device}")
        print(f"  GPUs: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
        print(f"  Batch size: {batch_size}")
        print(f"  Embedding dim: {self.embedding_dim}")

    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling over token embeddings."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def embed_texts(
        self,
        texts: List[str],
        show_progress: bool = True,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of texts to embed.
            show_progress: Show progress during processing.
            normalize: L2 normalize embeddings (recommended for similarity search).

        Returns:
            NumPy array of shape (len(texts), embedding_dim).
        """
        if not texts:
            return np.array([])

        all_embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                batch_num = i // self.batch_size + 1

                if show_progress and batch_num % 100 == 0:
                    print(f"Processing batch {batch_num}/{total_batches}")

                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )

                # Move to device
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

                # Forward pass
                outputs = self.model(**encoded)

                # Mean pooling
                embeddings = self._mean_pooling(outputs, encoded["attention_mask"])

                # Normalize if requested
                if normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                all_embeddings.append(embeddings.cpu().numpy())

        # Concatenate all batches
        result = np.vstack(all_embeddings)
        
        if show_progress:
            print(f"Generated {len(result)} embeddings of dimension {self.embedding_dim}")

        return result

    def embed_annotations(
        self,
        annotations_path: str,
        output_path: str,
        text_field: str = "context",
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate embeddings for all annotations in a JSONL file.

        Args:
            annotations_path: Path to annotations JSONL file.
            output_path: Path to save embeddings (.npy file).
            text_field: Field name containing text to embed.
            show_progress: Show progress during processing.

        Returns:
            Statistics about the embedding process.
        """
        # Load annotations
        texts = []
        metadata = []
        
        with open(annotations_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                text = record.get(text_field, "")
                if text:
                    texts.append(text)
                    metadata.append({
                        "surah": record.get("surah"),
                        "ayah": record.get("ayah"),
                        "source": record.get("source"),
                        "type": record.get("type"),
                        "value": record.get("value"),
                    })

        print(f"Loaded {len(texts)} annotations from {annotations_path}")

        # Generate embeddings
        embeddings = self.embed_texts(texts, show_progress=show_progress)

        # Save embeddings
        np.save(output_path, embeddings)
        print(f"Saved embeddings to {output_path}")

        # Save metadata
        metadata_path = output_path.replace(".npy", "_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False)
        print(f"Saved metadata to {metadata_path}")

        return {
            "total_annotations": len(texts),
            "embedding_dim": self.embedding_dim,
            "embeddings_path": output_path,
            "metadata_path": metadata_path,
        }

    def embed_tafsir(
        self,
        tafsir_dir: str,
        output_dir: str,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate embeddings for all tafsir JSONL files.

        Args:
            tafsir_dir: Directory containing tafsir JSONL files.
            output_dir: Directory to save embeddings.
            show_progress: Show progress during processing.

        Returns:
            Statistics about the embedding process.
        """
        tafsir_dir = Path(tafsir_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stats = {"sources": {}, "total_embeddings": 0}

        for jsonl_file in tafsir_dir.glob("*.jsonl"):
            source_name = jsonl_file.stem
            print(f"\nProcessing {source_name}...")

            texts = []
            metadata = []

            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    text = record.get("text", "")
                    if text:
                        texts.append(text)
                        metadata.append({
                            "surah": record.get("surah"),
                            "ayah": record.get("ayah"),
                            "source": source_name,
                        })

            if texts:
                embeddings = self.embed_texts(texts, show_progress=show_progress)
                
                # Save
                emb_path = output_dir / f"{source_name}_embeddings.npy"
                meta_path = output_dir / f"{source_name}_metadata.json"
                
                np.save(emb_path, embeddings)
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False)

                stats["sources"][source_name] = {
                    "count": len(texts),
                    "embeddings_path": str(emb_path),
                }
                stats["total_embeddings"] += len(texts)

        return stats

    def get_gpu_info(self) -> Dict[str, Any]:
        """Get information about available GPUs."""
        if not torch.cuda.is_available():
            return {"available": False, "count": 0}

        info = {
            "available": True,
            "count": torch.cuda.device_count(),
            "devices": [],
        }

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["devices"].append({
                "index": i,
                "name": props.name,
                "total_memory_gb": props.total_memory / (1024**3),
                "compute_capability": f"{props.major}.{props.minor}",
            })

        return info
